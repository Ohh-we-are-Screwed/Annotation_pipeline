#!/usr/bin/env python3
"""ZED stereo vs LiDAR agreement by range, on Stage 1's fused single sweep.

For every stereo point (ring 100/101) the nearest LiDAR point (rings 0-3) within
0.6 m is its reference. Per 1 m LiDAR-range bin and per ring: count, median and
MAD of d_range = |p_zed| - |p_lidar| and of dz = z_zed - z_lidar (ego frame).
Decision rule (spec §3.4): cap = largest bin with median |d_range| <= 0.5 m and
MAD <= 1.0 m for BOTH rings, bins with >= 200 pairs only. A per-ring dz that is
constant across 3-15 m (|slope| < 0.01 m/m) and > 0.2 m is reported as a
constant offset to correct; otherwise reported and not corrected.

Two statistics ride alongside the paired ones because the pairing CENSORS them:
d_range and dz are both bounded by the 0.6 m pairing radius, so neither can see
a disagreement bigger than that, and `frac_paired` -- the share of a ring's
points in the bin that found any reference at all -- is the uncensored view of
where agreement breaks. The second is `dz_plane`, z minus the height of the
keyframe's own per-sector LiDAR ground plane (the SAME planes Stage 1 filtered
with): its p10 per ring per bin is the height of the ring's floor above the
LiDAR road, unbounded, and it is the only estimator here that can measure an
offset of the size the reviewer saw (~1.4 m) -- so it is the one the z decision
rule reads. Note that Stage 1 has already deleted |dz_plane| < ground_band_m,
so a floor within that band of zero is indistinguishable from aligned.

Ring 100 (rear ZED) and ring 101 (front ZED) share one global->ego transform,
so a difference between their floors is a per-camera export calibration defect.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.stage1_ingestion.ingest import read_pcd_bin, sector_index  # noqa: E402

STEREO = (100, 101)
PAIR_RADIUS_M = 0.6             # a stereo point's LiDAR reference must be this close, in 3D
PLANE_FLOOR_PCT = 10.0          # the p10 of dz_plane is the ring's floor: its road returns
DZ_FIT_RANGE_M = (3.0, 15.0)    # the window the z decision rule fits over
MIN_BIN_PAIRS = 200             # bins thinner than this are not trusted by either rule


def ground_plane_z(cloud: np.ndarray, sector_planes: list[dict], n_sectors: int) -> np.ndarray:
    """Ground height under each point, from its OWN sector's plane (ingest.ground_distance)."""
    sectors = sector_index(cloud[:, :2], n_sectors)
    out = np.full(cloud.shape[0], np.nan)
    for plane in sector_planes:
        m = sectors == plane["sector"]
        if m.any():
            out[m] = plane["a"] * cloud[m, 0] + plane["b"] * cloud[m, 1] + plane["d"]
    return out


def measure_cloud(cloud: np.ndarray, sector_planes: list[dict], n_sectors: int,
                  max_pair_m: float = PAIR_RADIUS_M):
    """(pairs, all_points) per stereo ring, or (None, None) with too little LiDAR.

    pairs[ring]: (N, 3) [lidar range, d_range, dz] over PAIRED stereo points,
    keyed on the reference's range exactly as the decision rule specifies.
    all_points[ring]: (M, 3) [horizontal range, dz_plane, paired] over EVERY
    stereo point of the ring, which is what the two uncensored statistics read.
    """
    lidar = cloud[cloud[:, 4] < 10]
    if len(lidar) < 100:
        return None, None
    tree = cKDTree(lidar[:, :3])
    plane_z = ground_plane_z(cloud, sector_planes, n_sectors) if sector_planes else None
    pairs, allpts = {}, {}
    for ring in STEREO:
        m_ring = cloud[:, 4] == ring
        z = cloud[m_ring]
        if not len(z):
            continue
        d, j = tree.query(z[:, :3], distance_upper_bound=max_pair_m)
        ok = np.isfinite(d)
        if plane_z is not None:
            allpts[ring] = np.column_stack([
                np.linalg.norm(z[:, :2], axis=1),      # horizontal range
                z[:, 2] - plane_z[m_ring],             # dz_plane (uncensored)
                ok.astype(np.float64),                 # found a reference within max_pair_m
            ])
        if not ok.any():
            continue
        ref = lidar[j[ok]]
        pairs[ring] = np.column_stack([
            np.linalg.norm(ref[:, :3], axis=1),                                     # lidar range
            np.linalg.norm(z[ok, :3], axis=1) - np.linalg.norm(ref[:, :3], axis=1),  # d_range
            z[ok, 2] - ref[:, 2],                                                    # dz
        ])
    return pairs, allpts


def mad(x):
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n-keyframes", type=int, default=120)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    paths = load_paths(a.paths)
    scene_dir = os.path.join(paths.work_root, "stage1_ingestion", "scenes", a.scene)
    rows = [json.loads(line) for line in open(os.path.join(scene_dir, "keyframes.jsonl")) if line.strip()]
    diag = json.load(open(os.path.join(scene_dir, "filter_diagnostics.json")))
    n_sectors = int(diag["config"]["n_sectors"])
    ground_band_m = float(diag["config"]["ground_band_m"])
    diag_by_token = {k["keyframe_token"]: k for k in diag["keyframes"]}

    step = max(1, len(rows) // a.n_keyframes)
    acc = {r: [] for r in STEREO}
    pacc = {r: [] for r in STEREO}
    used = n_planes = 0
    for row in rows[::step][: a.n_keyframes]:
        kdiag = diag_by_token.get(row["keyframe_token"], {})
        # Only keyframes whose ground fit succeeded carry usable sector planes.
        sector_planes = kdiag["sector_planes"] if kdiag.get("ground_reference_plane") else []
        pairs, allpts = measure_cloud(read_pcd_bin(row["single_sweep_cloud"]["path"]),
                                      sector_planes, n_sectors)
        if pairs is None:
            continue
        used += 1
        n_planes += bool(sector_planes)
        for r, arr in pairs.items():
            acc[r].append(arr)
        for r, arr in allpts.items():
            pacc[r].append(arr)

    bins = {}
    for r in STEREO:
        arr = np.vstack(acc[r]) if acc[r] else np.zeros((0, 3))
        parr = np.vstack(pacc[r]) if pacc[r] else np.zeros((0, 3))
        per = []
        for lo in range(0, 40):
            sel = arr[(arr[:, 0] >= lo) & (arr[:, 0] < lo + 1)]
            psel = parr[(parr[:, 0] >= lo) & (parr[:, 0] < lo + 1)]
            per.append({"range_m": [lo, lo + 1], "n": int(len(sel)),
                        "d_range_median": float(np.median(sel[:, 1])) if len(sel) else None,
                        "d_range_mad": mad(sel[:, 1]) if len(sel) else None,
                        "dz_median": float(np.median(sel[:, 2])) if len(sel) else None,
                        "dz_mad": mad(sel[:, 2]) if len(sel) else None,
                        "n_stereo": int(len(psel)),
                        "frac_paired": float(np.mean(psel[:, 2])) if len(psel) else None,
                        "dz_plane_p10": float(np.percentile(psel[:, 1], PLANE_FLOOR_PCT)) if len(psel) else None,
                        "dz_plane_median": float(np.median(psel[:, 1])) if len(psel) else None,
                        "frac_below_plane": float(np.mean(psel[:, 1] < 0.0)) if len(psel) else None})
        bins[r] = per

    # --- cap decision rule (spec §3.4), applied verbatim -------------------
    cap = 0
    for lo in range(0, 40):
        ok = all(bins[r][lo]["n"] >= MIN_BIN_PAIRS and abs(bins[r][lo]["d_range_median"]) <= 0.5
                 and bins[r][lo]["d_range_mad"] <= 1.0 for r in STEREO)
        if ok:
            cap = lo + 1
        elif cap:
            break
    # Which of the rule's three conditions failed in the FIRST rejected bin (index
    # `cap`, since cap is the upper edge of the last accepted one): d_range is
    # censored by PAIR_RADIUS_M, so if only `n` fails, the cap the rule returns is a
    # pair-count floor and not a measured disagreement.
    cap_binding = ["saturated_at_bin_loop_limit"] if cap >= 40 else sorted({
        ("n_pairs" if bins[r][cap]["n"] < MIN_BIN_PAIRS else
         "d_range_median" if abs(bins[r][cap]["d_range_median"] or 0.0) > 0.5 else
         "d_range_mad" if (bins[r][cap]["d_range_mad"] or 0.0) > 1.0 else "none")
        for r in STEREO})

    # --- z decision rule --------------------------------------------------
    zcorr, zreport = {}, {}
    lo_m, hi_m = DZ_FIT_RANGE_M
    for r in STEREO:
        arr = np.vstack(acc[r]) if acc[r] else np.zeros((0, 3))
        sel = arr[(arr[:, 0] >= lo_m) & (arr[:, 0] < hi_m)]
        # The paired estimator, censored by the 0.6 m pairing radius: reference only.
        nn = ("insufficient pairs" if len(sel) < 500 else
              {"dz_median_3_15m": float(np.median(sel[:, 2])),
               "dz_slope_m_per_m": float(np.polyfit(sel[:, 0], sel[:, 2], 1)[0]),
               "n": int(len(sel)), "censored_at_m": PAIR_RADIUS_M})
        # The plane-relative estimator: per-bin floors, slope on the bin centres.
        def floor_fit(bin_lo, bin_hi):
            f = [(b["range_m"][0] + 0.5, b["dz_plane_p10"], b["n_stereo"]) for b in bins[r]
                 if bin_lo <= b["range_m"][0] < bin_hi and b["n_stereo"] >= MIN_BIN_PAIRS]
            if len(f) < 3:
                return None
            x = np.array([p[0] for p in f])
            y = np.array([p[1] for p in f])
            s = float(np.polyfit(x, y, 1)[0])
            return {"floor_median": float(np.median(y)), "floor_slope_m_per_m": s,
                    "implied_pitch_deg": float(np.degrees(np.arctan(s))),
                    "floor_first_bin": float(y[0]), "floor_last_bin": float(y[-1]),
                    "floor_min": float(y.min()), "floor_max": float(y.max()),
                    # How far the deepest bin's floor falls past the ground band edge.
                    # ~0 means every bin's floor IS the band edge, i.e. the ring's road
                    # returns coincide with the LiDAR road and it is aligned.
                    "max_depth_past_band_m": float(max(0.0, -y.min() - ground_band_m)),
                    "range_m": [float(x[0] - 0.5), float(x[-1] + 0.5)], "n_bins": len(f),
                    "n_points": int(sum(p[2] for p in f))}
        rule = floor_fit(lo_m, hi_m)          # the window the decision rule reads
        if rule is None:
            zreport[r] = {"paired_censored": nn, "plane": "insufficient points"}
            continue
        med, slope = rule["floor_median"], rule["floor_slope_m_per_m"]
        constant = abs(med) > 0.2 and abs(slope) < 0.01
        span = floor_fit(0, 40)               # the whole measured span, for the pitch magnitude
        # A ring whose floor never falls more than one band width past the band edge
        # has its road returns ON the LiDAR road: its apparent -band offset is the
        # ground filter, not a calibration error, and there is nothing to correct.
        aligned = span is not None and span["max_depth_past_band_m"] <= ground_band_m
        zreport[r] = {"paired_censored": nn, "plane": {
            **rule, "all_bins": span,
            "within_ground_band": bool(abs(med) <= ground_band_m),
            "diagnosis": ("aligned with the LiDAR road" if aligned else
                          "MISALIGNED with the LiDAR road"),
            "verdict": ("constant offset" if constant else
                        "range-dependent (not a constant offset)" if abs(slope) >= 0.01 else
                        "no offset worth correcting")}}
        if constant:
            zcorr[r] = round(-med, 3)   # correction = minus the offset

    result = {"scene": a.scene, "keyframes_used": used, "keyframes_with_plane": n_planes,
              "keyframes_total": len(rows), "keyframe_stride": step,
              "ground_band_m": ground_band_m, "pair_radius_m": PAIR_RADIUS_M,
              "min_bin_pairs": MIN_BIN_PAIRS,
              "bins": bins, "stereo_range_cap_m": cap or None, "cap_binding_condition": cap_binding,
              "stereo_z_correction_m": zcorr, "dz_report": zreport}
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(result, open(a.out_json, "w"), indent=1)
    with open(a.out_md, "w") as f:
        f.write(f"# ZED stereo vs LiDAR — {a.scene}\n\nKeyframes used: {used} "
                f"({n_planes} with a ground fit) of {len(rows)}, every {step}th. "
                f"Pairs within {PAIR_RADIUS_M} m; "
                f"Stage 1 ground band {ground_band_m} m.\n\n")
        f.write(f"**stereo_range_cap_m = {cap or 'UNDETERMINED (default 25.0)'}** "
                f"(bound at {cap} m by: {', '.join(cap_binding) or 'n/a'})  \n")
        f.write(f"**stereo_z_correction_m = {zcorr or 'none'}**\n\n")

        f.write("## Reading\n\n### Range cap\n\n")
        f.write(f"The cap rule's agreement statistic is CENSORED by the pairing: `d_range` is\n"
                f"`|p_zed| - |p_lidar|` for a reference chosen within {result['pair_radius_m']} m in 3D, so\n"
                f"`|d_range| <= {result['pair_radius_m']}` by construction and its MAD cannot reach the rule's 1.0 m\n"
                f"threshold whatever the data does. The only condition left able to fail is the\n"
                f">= {MIN_BIN_PAIRS}-pair floor.\n\n")
        if set(cap_binding) <= {"saturated_at_bin_loop_limit", "n_pairs", "none"}:
            f.write(f"That is what happened here: the rule returned {cap} m, bound by "
                    f"`{', '.join(cap_binding)}`, not by any\nmeasured disagreement. The number is an "
                    f"artefact of the method and is NOT written to the\nconfig; "
                    f"`stereo_range_cap_m` keeps its prior value.\n\n")
        f.write("The uncensored view is `frac_paired`, the share of a ring's points in the bin that\n"
                "found any LiDAR reference at all:\n\n")
        for r in STEREO:
            got = [b for b in bins[r] if b["n_stereo"] >= MIN_BIN_PAIRS]
            best = max(got, key=lambda b: b["frac_paired"]) if got else None
            half = next((b for b in got if b["frac_paired"] < 0.5), None)
            if best:
                f.write(f"- ring {r}: peaks at {best['frac_paired']:.2f} in the "
                        f"{best['range_m'][0]}-{best['range_m'][1]} m bin"
                        + (f"; first falls below 0.50 in the {half['range_m'][0]}-{half['range_m'][1]} m bin"
                           if half else "; never falls below 0.50")
                        + f", {got[-1]['frac_paired']:.2f} in the "
                          f"{got[-1]['range_m'][0]}-{got[-1]['range_m'][1]} m bin.\n")
        f.write("\n### Per-ring z offset\n\n")
        for r in STEREO:
            pl = zreport[r]["plane"]
            if not isinstance(pl, dict):
                f.write(f"- ring {r}: {pl}.\n")
                continue
            ab = pl["all_bins"]
            f.write(f"- ring {r}: **{pl['diagnosis']}** — {pl['verdict']}. Floor over "
                    f"{pl['range_m'][0]}-{pl['range_m'][1]} m: median {pl['floor_median']:.3f} m, "
                    f"slope {pl['floor_slope_m_per_m']:+.4f} m/m "
                    f"({pl['implied_pitch_deg']:+.2f} deg), {pl['n_bins']} bins / "
                    f"{pl['n_points']} points. Over the whole measured span "
                    f"{ab['range_m'][0]}-{ab['range_m'][1]} m: floor {ab['floor_first_bin']:+.2f} m in the "
                    f"first bin to {ab['floor_last_bin']:+.2f} m in the last, slope "
                    f"{ab['floor_slope_m_per_m']:+.4f} m/m ({ab['implied_pitch_deg']:+.2f} deg), "
                    f"deepest bin {ab['floor_min']:+.2f} m = "
                    f"{ab['max_depth_past_band_m']:.3f} m past the band edge. "
                    f"The paired estimator says {zreport[r]['paired_censored']['dz_median_3_15m']:+.3f} m "
                    f"but is censored at {result['pair_radius_m']} m and cannot see an offset this big.\n")
        f.write(f"\nA floor within Stage 1's ground band ({ground_band_m} m) of zero is "
                f"indistinguishable from\naligned: Stage 1 has already deleted "
                f"`|z - ground| < {ground_band_m}`, so an aligned ring's road returns\nare gone and "
                f"the floor of what survives sits at the band edge.\n\n")
        f.write("```json\n" + json.dumps(zreport, indent=1) + "\n```\n\n")
        for r in STEREO:
            f.write(f"## ring {r}\n\n"
                    "| range | pairs | d_range med | d_range MAD | dz med | dz MAD "
                    "| stereo pts | frac paired | plane p10 | plane med | frac below plane |\n"
                    "|---|---|---|---|---|---|---|---|---|---|---|\n")
            for b in bins[r]:
                if not b["n_stereo"]:
                    continue
                pair_cols = (f"{b['n']} | {b['d_range_median']:.2f} | {b['d_range_mad']:.2f} "
                             f"| {b['dz_median']:.2f} | {b['dz_mad']:.2f}" if b["n"] else "0 | - | - | - | -")
                f.write(f"| {b['range_m'][0]}-{b['range_m'][1]} | {pair_cols} "
                        f"| {b['n_stereo']} | {b['frac_paired']:.2f} | {b['dz_plane_p10']:.2f} "
                        f"| {b['dz_plane_median']:.2f} | {b['frac_below_plane']:.2f} |\n")
            f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "bins"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
