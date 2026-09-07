"""Offline track stitching + interpolation over one scene's I-4 records (spec §3)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, median

import numpy as np
from scipy.optimize import linear_sum_assignment

from pipeline.release.config import StitchConfig
from pipeline.release.frames import BASIS_RAW, BASIS_UNAVAILABLE, CloudSource, SceneFrames
from pipeline.release.geometry import box_ego_to_global, box_global_to_ego, points_in_box, slerp

TIER_ORDER = {"auto_accept": 0, "flagged": 1, "rejected": 2}


@dataclass
class StitchStats:
    n_records_in: int = 0
    n_fragments: int = 0
    n_chains: int = 0
    joins_by_gap: dict = field(default_factory=dict)
    n_interpolated: int = 0
    n_interpolated_raw_basis: int = 0
    n_interpolated_no_cloud: int = 0
    median_len_before: float = 0.0
    median_len_after: float = 0.0
    mean_len_before: float = 0.0
    mean_len_after: float = 0.0
    frac_rows_on_chains_ge3_before: float = 0.0
    frac_rows_on_chains_ge3_after: float = 0.0


@dataclass
class Fragment:
    fid: str                      # Stage 7 track_id, or the record token for an untracked singleton
    category: str
    rows: list                    # ordered by keyframe index
    kidx: list                    # keyframe indices
    centers: np.ndarray           # (n, 3) global
    quats: list                   # global [w,x,y,z] per row
    sizes: np.ndarray             # (n, 3) w,l,h

    @property
    def first(self) -> int:
        return self.kidx[0]

    @property
    def last(self) -> int:
        return self.kidx[-1]

    def bev_area(self) -> float:
        return float(self.sizes[:, 0].mean() * self.sizes[:, 1].mean())

    def end_velocity(self, frames: SceneFrames):
        if len(self.rows) < 2:
            return None
        dt = frames.dt_s(self.kidx[-2], self.kidx[-1])
        return (self.centers[-1] - self.centers[-2]) / dt if dt > 0 else None

    def start_velocity(self, frames: SceneFrames):
        if len(self.rows) < 2:
            return None
        dt = frames.dt_s(self.kidx[0], self.kidx[1])
        return (self.centers[1] - self.centers[0]) / dt if dt > 0 else None


def _fragment_id(r: dict) -> str:
    return str(r["track_id"]) if r.get("track_id") is not None else f"det:{r['token']}"


def _build_fragments(records: list, frames: SceneFrames) -> list:
    groups: dict = defaultdict(list)
    for r in records:
        groups[_fragment_id(r)].append(r)
    out = []
    for fid in sorted(groups, key=lambda k: (len(k), k)):
        rows = sorted(groups[fid], key=lambda r: frames.index[r["sample_token"]])
        cats = {r["category"] for r in rows}
        if len(cats) != 1:
            raise ValueError(f"fragment {fid!r} spans categories {sorted(cats)}; Stage 7 gates per class")
        centers, quats = [], []
        for r in rows:
            t, q = box_ego_to_global(r["translation_m"], r["rotation_wxyz"], frames.poses[r["sample_token"]])
            centers.append(t)
            quats.append(q)
        out.append(Fragment(fid=fid, category=rows[0]["category"], rows=rows,
                            kidx=[frames.index[r["sample_token"]] for r in rows],
                            centers=np.asarray(centers, dtype=np.float64), quats=quats,
                            sizes=np.asarray([r["size_wlh_m"] for r in rows], dtype=np.float64)))
    return out


def _distance(p: Fragment, s: Fragment, frames: SceneFrames):
    dt = frames.dt_s(p.last, s.first)
    cands = []
    vp, vs = p.end_velocity(frames), s.start_velocity(frames)
    if vp is not None:
        cands.append(np.linalg.norm(p.centers[-1] + vp * dt - s.centers[0]))
    if vs is not None:
        cands.append(np.linalg.norm(s.centers[0] - vs * dt - p.centers[-1]))
    if not cands:
        cands.append(np.linalg.norm(s.centers[0] - p.centers[-1]))
    return float(min(cands))


def _assign(frags: list, frames: SceneFrames, cfg: StitchConfig):
    """Returns next_of: fragment index -> successor index, plus joins_by_gap."""
    next_of: dict = {}
    prev_of: dict = {}
    joins: dict = {}
    for gap in range(1, cfg.max_gap_keyframes + 1):
        gate = cfg.base_gate_m + cfg.gap_slack_m * (gap - 1)
        preds = [i for i, f in enumerate(frags) if i not in next_of]
        succs = [j for j, f in enumerate(frags) if j not in prev_of]
        pairs = []
        for i in preds:
            p = frags[i]
            for j in succs:
                s = frags[j]
                if i == j or s.first - p.last != gap:
                    continue
                if not cfg.class_agnostic and s.category != p.category:
                    continue
                ratio = p.bev_area() / max(s.bev_area(), 1e-9)
                if ratio > cfg.size_ratio_max or ratio < 1.0 / cfg.size_ratio_max:
                    continue
                d = _distance(p, s, frames)
                if d <= gate:
                    pairs.append((i, j, d / gate))
        if not pairs:
            continue
        rows_i = sorted({i for i, _, _ in pairs})
        cols_j = sorted({j for _, j, _ in pairs})
        cost = np.full((len(rows_i), len(cols_j)), 10.0)
        ri, cj = {i: a for a, i in enumerate(rows_i)}, {j: b for b, j in enumerate(cols_j)}
        for i, j, c in pairs:
            cost[ri[i], cj[j]] = c
        for a, b in zip(*linear_sum_assignment(cost)):
            if cost[a, b] <= 1.0:
                i, j = rows_i[a], cols_j[b]
                next_of[i] = j
                prev_of[j] = i
                joins[gap] = joins.get(gap, 0) + 1
    return next_of, joins


def _worse_tier(a: str, b: str) -> str:
    return a if TIER_ORDER.get(a, 9) >= TIER_ORDER.get(b, 9) else b


def _interpolate(p: Fragment, s: Fragment, chain_id: str, frames: SceneFrames,
                 clouds: CloudSource | None, stats: StitchStats) -> list:
    rows = []
    k0, k1 = p.last, s.first
    c0, c1 = p.centers[-1], s.centers[0]
    q0, q1 = np.asarray(p.quats[-1]), np.asarray(s.quats[0])
    z0, z1 = p.sizes[-1], s.sizes[0]
    tier = _worse_tier(p.rows[-1]["provenance"]["tier"], s.rows[0]["provenance"]["tier"])
    for k in range(k0 + 1, k1):
        f = (k - k0) / (k1 - k0)
        tok = frames.tokens[k]
        cg = (1 - f) * c0 + f * c1
        qg = slerp(q0, q1, f)
        size = ((1 - f) * z0 + f * z1).tolist()
        te, qe = box_global_to_ego(cg.tolist(), qg.tolist(), frames.poses[tok])
        n_pts, basis = 0, BASIS_UNAVAILABLE
        if clouds is not None:
            pts, basis = clouds.points(tok)
            if pts is not None:
                n_pts = points_in_box(pts, te, size, qe)
                if basis == BASIS_RAW:
                    stats.n_interpolated_raw_basis += 1
            else:
                stats.n_interpolated_no_cloud += 1
        else:
            stats.n_interpolated_no_cloud += 1
        src = p.rows[-1]
        rows.append({
            "token": f"{tok}:INTERP:{chain_id}", "sample_token": tok,
            "instance_token": f"chain:{frames.scene_token}:{chain_id}", "category": p.category,
            "frame": "ego", "t_ns": int(frames.timestamps_ns[k]), "time_base": "unix_ns",
            "translation_m": te, "size_wlh_m": size, "rotation_wxyz": qe,
            "num_lidar_pts": int(n_pts), "num_lidar_pts_basis": basis,
            "provenance": {"source": "pipeline", "tier": tier, "gates": None,
                           "verified_by": None, "verification_pass": 0},
            "coverage_config": src.get("coverage_config"), "velocity_mps": None, "track_id": None,
            "split": None, "attribute": None, "visibility": None,
            "stitch_chain_id": chain_id, "stitch_track_id_pre": None,
            "stitch_interpolated": True, "stitch_tier_basis": "inherited_from_endpoints",
        })
        stats.n_interpolated += 1
    return rows


def _length_stats(groups: dict, n_rows: int):
    lens = [len(v) for v in groups.values()]
    ge3 = sum(len(v) for v in groups.values() if len(v) >= 3)
    return (float(median(lens)) if lens else 0.0, float(mean(lens)) if lens else 0.0,
            ge3 / n_rows if n_rows else 0.0)


def stitch_scene(records: list, frames: SceneFrames, clouds: CloudSource | None,
                 cfg: StitchConfig) -> tuple:
    stats = StitchStats(n_records_in=len(records))
    frags = _build_fragments(records, frames)
    stats.n_fragments = len(frags)
    before = {f.fid: f.rows for f in frags}
    stats.median_len_before, stats.mean_len_before, stats.frac_rows_on_chains_ge3_before = \
        _length_stats(before, len(records))

    next_of, joins = _assign(frags, frames, cfg)
    stats.joins_by_gap = dict(sorted(joins.items()))
    has_prev = set(next_of.values())
    out: list = []
    chain_rows: dict = {}
    for i, f in enumerate(frags):
        if i in has_prev:
            continue
        chain_id = f.fid
        j = i
        members = []
        while True:
            members.append(j)
            if j not in next_of:
                break
            j = next_of[j]
        rows_here = []
        for a, b in zip(members, members[1:]):
            rows_here.extend(frags[a].rows)
            if frags[b].first - frags[a].last >= 2:
                rows_here.extend(_interpolate(frags[a], frags[b], chain_id, frames, clouds, stats))
        rows_here.extend(frags[members[-1]].rows)
        for r in rows_here:
            if not r.get("stitch_interpolated"):
                r["stitch_track_id_pre"] = str(r["track_id"]) if r.get("track_id") is not None else None
                r["stitch_interpolated"] = False
                r["stitch_tier_basis"] = "gate"
            r["stitch_chain_id"] = chain_id
            r["instance_token"] = f"chain:{frames.scene_token}:{chain_id}"
        chain_rows[chain_id] = rows_here
        out.extend(rows_here)
    stats.n_chains = len(chain_rows)
    stats.median_len_after, stats.mean_len_after, stats.frac_rows_on_chains_ge3_after = \
        _length_stats(chain_rows, len(out))
    out.sort(key=lambda r: (frames.index[r["sample_token"]], r["stitch_chain_id"], r["token"]))
    return out, stats
