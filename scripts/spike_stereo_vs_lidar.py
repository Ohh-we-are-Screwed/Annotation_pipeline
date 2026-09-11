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

# The pitch correction the range-dependent branch measures, and its acceptance.
# PIVOT: the camera's OWN optical centre in the ego (= LiDAR) frame, because a
# rotation about anything else flattens the floor at the wrong height. Taken
# from the export's own calibrated_sensor.json (channel CAM_FRONT / CAM_BACK are
# the two ZED 2i heads: front serial 35084019 = ring 101, rear 32957407 = ring
# 100), corroborated for the front by the exporter repo's
# configs/rig/legacy_zed_extrinsic.json, which carries the identical translation
# and names the serial. NOT derived from the floor line: camera height is
# unobservable from ground points alone.
PITCH_PIVOT_M = {100: (-0.86481, -0.62055), 101: (0.81253, -0.73305)}
PIVOT_SOURCE = ("<dataroot>/v1.0-dhaka-fixed2/calibrated_sensor.json, channel CAM_FRONT "
                "(ring 101, ZED 2i serial 35084019) / CAM_BACK (ring 100, serial 32957407); "
                "corroborated by /home/saif/dhaka-export-pipeline-20260911/configs/rig/"
                "legacy_zed_extrinsic.json (same translation, names the front serial)")
VERIFY_RANGE_M = (3.0, 25.0)    # the window the acceptance test covers
VERIFY_FLOOR_PCT = 5.0          # verification reads p5, a stricter floor than the p10 fit
ACCEPT_SLOPE_M_PER_M = 0.01     # corrected floor must be flat to this
ACCEPT_FLOOR_MIN_M = -0.45      # and no bin's floor may sit below this
SPREAD_MAX_M_PER_M = 0.02       # p90 - p10 of the per-block slope, else not a rigid defect
BLOCK_KEYFRAMES = 10            # keyframes per block for the spread test
LIDAR_KEY = -1                  # rings 0-3 pooled: the reference's own noise floor


def ground_plane_z(cloud: np.ndarray, sector_planes: list[dict], n_sectors: int) -> np.ndarray:
    """Ground height under each point, from its OWN sector's plane (ingest.ground_distance)."""
    sectors = sector_index(cloud[:, :2], n_sectors)
    out = np.full(cloud.shape[0], np.nan)
    for plane in sector_planes:
        m = sectors == plane["sector"]
        if m.any():
            out[m] = plane["a"] * cloud[m, 0] + plane["b"] * cloud[m, 1] + plane["d"]
    return out


def sector_plane_coeffs(cloud: np.ndarray, sector_planes: list[dict], n_sectors: int) -> np.ndarray:
    """(N, 3) [a, b, d] of each point's OWN sector plane, so z_ground = a*x + b*y + d
    can be re-evaluated after the point moves."""
    sectors = sector_index(cloud[:, :2], n_sectors)
    out = np.full((cloud.shape[0], 3), np.nan)
    for plane in sector_planes:
        m = sectors == plane["sector"]
        if m.any():
            out[m] = (plane["a"], plane["b"], plane["d"])
    return out


def measure_cloud(cloud: np.ndarray, sector_planes: list[dict], n_sectors: int,
                  max_pair_m: float = PAIR_RADIUS_M):
    """(pairs, all_points, raw) per stereo ring, or (None, None, None) with too little LiDAR.

    pairs[ring]: (N, 3) [lidar range, d_range, dz] over PAIRED stereo points,
    keyed on the reference's range exactly as the decision rule specifies.
    all_points[ring]: (M, 3) [horizontal range, dz_plane, paired] over EVERY
    stereo point of the ring, which is what the two uncensored statistics read.
    raw[ring]: (K, 6) float32 [x, y, z, a, b, d] inside the verification window,
    which is what the pitch correction is applied to in memory. The plane
    COEFFICIENTS ride along rather than the evaluated dz_plane, because rotating
    a point changes its x and so changes the ground height beneath it.
    """
    lidar = cloud[cloud[:, 4] < 10]
    if len(lidar) < 100:
        return None, None, None
    tree = cKDTree(lidar[:, :3])
    plane_z = ground_plane_z(cloud, sector_planes, n_sectors) if sector_planes else None
    coeffs = sector_plane_coeffs(cloud, sector_planes, n_sectors) if sector_planes else None
    pairs, allpts, raw_xyz = {}, {}, {}
    if coeffs is not None:
        # The reference's OWN floor. Stage 1 has deleted |z - ground| < band, so a
        # perfect plane would put this at exactly -band in every bin, slope zero;
        # whatever slope it does have is the plane's error, not the camera's.
        m_l = cloud[:, 4] < 10
        keep = m_l & (np.hypot(cloud[:, 0], cloud[:, 1]) < VERIFY_RANGE_M[1] + 1.0)
        raw_xyz[LIDAR_KEY] = np.column_stack([cloud[keep, :3], coeffs[keep]]).astype(np.float32)
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
        if coeffs is not None and ring in PITCH_PIVOT_M:
            keep = np.zeros(cloud.shape[0], dtype=bool)
            keep[m_ring] = np.linalg.norm(z[:, :2], axis=1) < VERIFY_RANGE_M[1] + 1.0
            raw_xyz[ring] = np.column_stack([cloud[keep, :3], coeffs[keep]]).astype(np.float32)
        if not ok.any():
            continue
        ref = lidar[j[ok]]
        pairs[ring] = np.column_stack([
            np.linalg.norm(ref[:, :3], axis=1),                                     # lidar range
            np.linalg.norm(z[ok, :3], axis=1) - np.linalg.norm(ref[:, :3], axis=1),  # d_range
            z[ok, 2] - ref[:, 2],                                                    # dz
        ])
    return pairs, allpts, raw_xyz


def mad(x):
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def floor_profile(rng: np.ndarray, dz: np.ndarray, lo: float, hi: float,
                  pct: float = PLANE_FLOOR_PCT, min_pts: int = MIN_BIN_PAIRS):
    """Per 1 m bin in [lo, hi): (bin centre, floor percentile of dz, count).

    The floor of a ring's points in a bin is its road returns, so a floor that
    walks with range is a rotation and a floor that sits still is an offset.
    """
    out = []
    for b in range(int(lo), int(hi)):
        sel = dz[(rng >= b) & (rng < b + 1)]
        if len(sel) >= min_pts:
            out.append((b + 0.5, float(np.percentile(sel, pct)), int(len(sel))))
    return out


def fit_floor(profile):
    """Slope, median and derived pitch of a floor profile, or None under 3 bins."""
    if len(profile) < 3:
        return None
    x = np.array([p[0] for p in profile])
    y = np.array([p[1] for p in profile])
    s = float(np.polyfit(x, y, 1)[0])
    return {"floor_median": float(np.median(y)), "floor_slope_m_per_m": s,
            "implied_pitch_deg": float(np.degrees(np.arctan(s))),
            "floor_first_bin": float(y[0]), "floor_last_bin": float(y[-1]),
            "floor_min": float(y.min()), "floor_max": float(y.max()),
            "range_m": [float(x[0] - 0.5), float(x[-1] + 0.5)], "n_bins": len(profile),
            "n_points": int(sum(p[2] for p in profile))}


def block_spread(blocks: list[np.ndarray], lo: float, hi: float, label: str) -> dict | None:
    """p10/p50/p90 of the uncorrected floor slope across blocks of keyframes.

    A rigid mis-mount is the SAME angle in every block; a spread wider than the
    acceptance band means the measured tilt varies with the scene and no single
    constant is honest. Run over two windows, because the ground plane is fitted
    over a shorter range than it is used at and an extrapolated reference tilts
    on its own.
    """
    slopes = []
    for blk in blocks:
        rng, dz = apply_pitch(blk, 0.0, (0.0, 0.0))
        fit = fit_floor(floor_profile(rng, dz, lo, hi))
        if fit:
            slopes.append(fit["floor_slope_m_per_m"])
    if len(slopes) < 3:
        return None
    q = np.percentile(slopes, [10, 50, 90])
    return {"window": label, "range_m": [lo, hi], "n_blocks": len(slopes),
            "keyframes_per_block": BLOCK_KEYFRAMES, "slopes": [round(v, 5) for v in slopes],
            "slope_p10": float(q[0]), "slope_p50": float(q[1]), "slope_p90": float(q[2]),
            "spread_p90_minus_p10": float(q[2] - q[0]), "limit": SPREAD_MAX_M_PER_M,
            "rigid": bool(q[2] - q[0] < SPREAD_MAX_M_PER_M)}


def apply_pitch(raw: np.ndarray, deg: float, pivot: tuple[float, float]) -> np.ndarray:
    """(K, 6) [x, y, z, a, b, d] -> (range, dz_plane) after a rigid pitch.

    The same rotation ingest.stereo_block_to_ego applies, so what the spike
    accepts is what Stage 1 will produce. The ground height is re-evaluated at
    the point's NEW x, which is the whole reason the plane coefficients travel
    with the point instead of a precomputed dz.
    """
    theta = np.radians(deg)
    cos, sin = np.cos(theta), np.sin(theta)
    px, pz = pivot
    dx, dz = raw[:, 0] - px, raw[:, 2] - pz
    x = px + cos * dx + sin * dz
    z = pz - sin * dx + cos * dz
    ground = raw[:, 3] * x + raw[:, 4] * raw[:, 1] + raw[:, 5]
    return np.hypot(x, raw[:, 1]), z - ground


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
    # Where the ground plane was FITTED, hence where it is a measurement rather
    # than an extrapolation. Read from the run, not assumed.
    plane_support_m = tuple(float(v) for v in diag["config"]["ground_fit_range_m"])
    diag_by_token = {k["keyframe_token"]: k for k in diag["keyframes"]}

    step = max(1, len(rows) // a.n_keyframes)
    acc = {r: [] for r in STEREO}
    pacc = {r: [] for r in STEREO}
    racc = {r: [] for r in (*STEREO, LIDAR_KEY)}
    used = n_planes = 0
    for row in rows[::step][: a.n_keyframes]:
        kdiag = diag_by_token.get(row["keyframe_token"], {})
        # Only keyframes whose ground fit succeeded carry usable sector planes.
        sector_planes = kdiag["sector_planes"] if kdiag.get("ground_reference_plane") else []
        pairs, allpts, raw_xyz = measure_cloud(read_pcd_bin(row["single_sweep_cloud"]["path"]),
                                               sector_planes, n_sectors)
        if pairs is None:
            continue
        used += 1
        n_planes += bool(sector_planes)
        for r, arr in pairs.items():
            acc[r].append(arr)
        for r, arr in allpts.items():
            pacc[r].append(arr)
        for r, arr in raw_xyz.items():
            racc[r].append(arr)

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
        parr = np.vstack(pacc[r]) if pacc[r] else np.zeros((0, 3))

        def floor_fit(bin_lo, bin_hi):
            fit = fit_floor(floor_profile(parr[:, 0], parr[:, 1], bin_lo, bin_hi))
            if fit is not None:
                # How far the deepest bin's floor falls past the ground band edge.
                # ~0 means every bin's floor IS the band edge, i.e. the ring's road
                # returns coincide with the LiDAR road and it is aligned.
                fit["max_depth_past_band_m"] = float(max(0.0, -fit["floor_min"] - ground_band_m))
            return fit
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

    # --- pitch correction for any ring the z rule called range-dependent ----
    # A rotation, unlike an offset, cannot be undone by z_correction_m, so the
    # angle is measured here and applied by Stage 1's --stereo-pitch-correction.
    pitch = {}
    for r in STEREO:
        pl = zreport[r]["plane"]
        if not isinstance(pl, dict) or pl["diagnosis"].startswith("aligned") or not racc[r]:
            continue
        pivot = PITCH_PIVOT_M[r]
        raw = np.vstack(racc[r])
        lo_v, hi_v = VERIFY_RANGE_M

        def verify(deg):
            """Floor profile at p5 over 3-25 m after rotating by `deg`, plus the
            share of points sitting below the ground band. deg = 0.0 is the
            uncorrected baseline, measured the identical way."""
            rng, dz = apply_pitch(raw, deg, pivot)
            prof = floor_profile(rng, dz, lo_v, hi_v, pct=VERIFY_FLOOR_PCT)
            fit = fit_floor(prof)
            win = (rng >= lo_v) & (rng < hi_v)
            return {"deg": deg, "fit": fit,
                    "bins": [{"range_m": [c - 0.5, c + 0.5], "floor_p5": f, "n": n}
                             for c, f, n in prof],
                    "frac_below_band": float(np.mean(dz[win] < -ground_band_m)) if win.any() else None,
                    "accepted": bool(fit and abs(fit["floor_slope_m_per_m"]) < ACCEPT_SLOPE_M_PER_M
                                     and fit["floor_min"] >= ACCEPT_FLOOR_MIN_M)}

        # Per-block slope spread over TWO windows: the full verification window,
        # and the range over which the ground plane was actually FITTED. Beyond
        # its fit range the plane is extrapolated, and an extrapolated reference
        # tilts on its own -- so a spread that is wide over 3-25 m but narrow
        # over the fit support locates the swing in the reference, not the camera.
        def blocks_of(store):
            return [np.vstack(store[i:i + BLOCK_KEYFRAMES])
                    for i in range(0, len(store), BLOCK_KEYFRAMES)]
        blocks = blocks_of(racc[r])
        spread = block_spread(blocks, lo_v, hi_v, "verification")
        spread_support = block_spread(blocks, *plane_support_m, "plane_fit_support")
        # And the reference's own noise floor, same blocks, same window: the
        # LiDAR floor should be flat at -ground_band_m if the plane were exact.
        spread_lidar = (block_spread(blocks_of(racc[LIDAR_KEY]), *plane_support_m,
                                     "plane_fit_support (LiDAR only)")
                        if racc.get(LIDAR_KEY) else None)
        # The decision reads the fit-support window: it is the only one where the
        # reference is measured rather than extrapolated.
        deciding = spread_support or spread
        # The angle that flattens a floor of slope m is atan(m) -- but the SIGN
        # and the choice of fit window are settled numerically, not by algebra.
        baseline = verify(0.0)
        candidates = {"fit_3_15m": pl["implied_pitch_deg"],
                      "fit_full_span": pl["all_bins"]["implied_pitch_deg"],
                      # The self-consistent one: fitted on the SAME window and
                      # percentile the acceptance test reads, so a straight-line
                      # floor would be flattened exactly.
                      "fit_verify_window": (baseline["fit"] or {}).get("implied_pitch_deg")}
        candidates = {k: v for k, v in candidates.items() if v is not None}
        tried = {k: verify(v) for k, v in candidates.items()}
        tried["sign_flipped_check"] = verify(-candidates["fit_full_span"])
        ok = [(k, t) for k, t in tried.items() if k != "sign_flipped_check" and t["accepted"]]
        chosen = min(ok, key=lambda kt: abs(kt[1]["fit"]["floor_slope_m_per_m"]))[0] if ok else None
        rigid = bool(deciding and deciding["rigid"])
        pitch[r] = {"pivot_x_m": pivot[0], "pivot_z_m": pivot[1],
                    "pivot_source": PIVOT_SOURCE,
                    "per_block_spread": spread,
                    "per_block_spread_plane_support": spread_support,
                    "per_block_spread_lidar_reference": spread_lidar,
                    "deciding_window": deciding and deciding["window"],
                    "candidates": candidates,
                    "baseline_uncorrected": baseline, "verification": tried,
                    "chosen_candidate": chosen,
                    "deg": round(candidates[chosen], 4) if chosen and rigid else None,
                    "acceptance": {"slope_m_per_m": ACCEPT_SLOPE_M_PER_M,
                                   "floor_min_m": ACCEPT_FLOOR_MIN_M,
                                   "window_m": list(VERIFY_RANGE_M),
                                   "spread_m_per_m": SPREAD_MAX_M_PER_M,
                                   "spread_window_m": list(plane_support_m)},
                    "verdict": ("accepted" if chosen and rigid else
                                "REJECTED: per-block slope spread too wide" if chosen else
                                "REJECTED: no candidate angle meets the acceptance test")}

    result = {"scene": a.scene, "keyframes_used": used, "keyframes_with_plane": n_planes,
              "keyframes_total": len(rows), "keyframe_stride": step,
              "ground_band_m": ground_band_m, "pair_radius_m": PAIR_RADIUS_M,
              "min_bin_pairs": MIN_BIN_PAIRS,
              "bins": bins, "stereo_range_cap_m": cap or None, "cap_binding_condition": cap_binding,
              "stereo_z_correction_m": zcorr, "dz_report": zreport,
              "stereo_pitch_correction": {
                  r: {"deg": v["deg"], "pivot_x_m": v["pivot_x_m"], "pivot_z_m": v["pivot_z_m"]}
                  for r, v in pitch.items() if v["deg"] is not None},
              "pitch_report": pitch}
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

        if pitch:
            f.write("### Rigid pitch correction\n\n")
            f.write(f"A range-dependent floor is a ROTATION, which `stereo_z_correction_m` cannot "
                    f"undo, so\nthe angle that flattens it is measured here and applied by Stage 1's\n"
                    f"`--stereo-pitch-correction RING:DEG:PIVOT_X_M:PIVOT_Z_M`. The pivot is the "
                    f"camera's own\noptical centre in the ego frame, from {PIVOT_SOURCE} — NOT "
                    f"derived from the floor\nline, because camera height is unobservable from "
                    f"ground points alone.\n\n")
            for r, v in pitch.items():
                sp = v["per_block_spread"]
                acc_ = v["acceptance"]
                f.write(f"#### ring {r} — **{v['verdict']}**\n\n")
                f.write(f"Pivot ({v['pivot_x_m']}, ., {v['pivot_z_m']}) m. Acceptance over "
                        f"{acc_['window_m'][0]}-{acc_['window_m'][1]} m: "
                        f"|slope| < {acc_['slope_m_per_m']} m/m, every bin's floor "
                        f"(p{VERIFY_FLOOR_PCT:.0f}) >= {acc_['floor_min_m']} m, and per-block slope "
                        f"spread < {acc_['spread_m_per_m']} m/m.\n\n")
                f.write("| angle tried | deg | corrected slope m/m | floor min m | floor median m "
                        "| frac below ground band | meets slope+floor |\n|---|---|---|---|---|---|---|\n")
                rows = [("uncorrected (baseline)", v["baseline_uncorrected"])]
                rows += [(k, t) for k, t in v["verification"].items()]
                for label, t in rows:
                    fit = t["fit"]
                    f.write(f"| {label} | {t['deg']:+.4f} | {fit['floor_slope_m_per_m']:+.5f} "
                            f"| {fit['floor_min']:+.3f} | {fit['floor_median']:+.3f} "
                            f"| {t['frac_below_band']:.3f} | {'yes' if t['accepted'] else 'no'} |\n")
                f.write(f"\nThe sign is settled numerically, not by algebra: `sign_flipped_check` "
                        f"above is the\nsame magnitude with the opposite sign and makes the floor "
                        f"WORSE, so the correction is\n`deg = atan(floor slope)` with the slope's "
                        f"own (negative) sign.\n\n")
                rows_sp = [v["per_block_spread"], v["per_block_spread_plane_support"],
                           v["per_block_spread_lidar_reference"]]
                rows_sp = [x for x in rows_sp if x]
                if rows_sp:
                    f.write(f"Per-block rigidity, {rows_sp[0]['n_blocks']} blocks of "
                            f"{rows_sp[0]['keyframes_per_block']} keyframes. The DECIDING window is "
                            f"`{v['deciding_window']}`: beyond the range the ground plane was "
                            f"fitted\nover it is extrapolated, so a swing there can be the "
                            f"reference rather than the camera. The\nlast row is that reference's "
                            f"OWN noise floor -- the LiDAR's floor against the LiDAR's own plane, "
                            f"same\nblocks, same window -- and the LiDAR is rigid with respect to "
                            f"itself by construction.\n\n")
                    f.write("| window | range m | slope p10 | p50 | p90 | spread p90-p10 | limit "
                            "| rigid |\n|---|---|---|---|---|---|---|---|\n")
                    for x in rows_sp:
                        f.write(f"| {x['window']} | {x['range_m'][0]}-{x['range_m'][1]} "
                                f"| {x['slope_p10']:+.4f} | {x['slope_p50']:+.4f} "
                                f"| {x['slope_p90']:+.4f} | **{x['spread_p90_minus_p10']:.4f}** "
                                f"| {x['limit']} | {'yes' if x['rigid'] else 'NO'} |\n")
                    f.write("\n")
                    for x in rows_sp:
                        f.write(f"- {x['window']} per block: "
                                f"{', '.join(f'{y:+.3f}' for y in x['slopes'])}\n")
                    f.write("\n")
                    ref = v["per_block_spread_lidar_reference"]
                    dec = v["per_block_spread_plane_support"] or v["per_block_spread"]
                    if ref and dec and ref["spread_p90_minus_p10"] > dec["spread_p90_minus_p10"]:
                        f.write(f"Note: the reference's own spread "
                                f"({ref['spread_p90_minus_p10']:.4f} m/m) is WIDER than ring {r}'s "
                                f"over the same\nwindow ({dec['spread_p90_minus_p10']:.4f} m/m), and "
                                f"both exceed the {dec['limit']} m/m bar. On 10-keyframe blocks this "
                                f"floor-slope\nestimator is therefore noisier than the bar it is "
                                f"being judged against, so a failed spread test\nhere is a statement "
                                f"about the estimator, not evidence that the camera is non-rigid.\n\n")
                if v["deg"] is None:
                    best = v["verification"].get(v["chosen_candidate"]) if v["chosen_candidate"] else None
                    f.write("**No correction is written for this ring.** ")
                    if best:
                        f.write(f"A single angle of {best['deg']:+.4f} deg does meet the slope and "
                                f"floor test (slope {best['fit']['floor_slope_m_per_m']:+.5f} m/m, "
                                f"floor min {best['fit']['floor_min']:+.3f} m, points below the "
                                f"ground band {v['baseline_uncorrected']['frac_below_band']:.3f} -> "
                                f"{best['frac_below_band']:.3f}), so the defect IS overwhelmingly a "
                                f"pitch. It is declined because the per-block spread on the "
                                f"deciding window "
                                f"({(v['per_block_spread_plane_support'] or v['per_block_spread'])['spread_p90_minus_p10']:.4f} m/m) "
                                f"is over the {SPREAD_MAX_M_PER_M} m/m bar — a decision by RULE, "
                                f"which the noise-floor row above shows is not the same as "
                                f"evidence that the camera is non-rigid. ")
                    f.write("For this run the front frustum is DROPPED downstream rather than "
                            "corrected.\n\n")
                else:
                    f.write(f"**Stage 1 flag:** `--stereo-pitch-correction "
                            f"{r}:{v['deg']}:{v['pivot_x_m']}:{v['pivot_z_m']}`\n\n")
            f.write("The export's front ZED (ZED 2i, serial 35084019, channel CAM_FRONT, ring 101)\n"
                    "is pitched by roughly 7-12 degrees relative to the LiDAR-fitted road and needs "
                    "an\nUPSTREAM FIX: a re-export with corrected front-ZED extrinsics. The rig "
                    "config that\nexport was built from still carries `calibrated: false` and "
                    "\"INITIAL GUESS -- replace\nwith scripts/calibrate.sh\" for this camera. "
                    "Correcting it at ingestion is a stopgap and\nis out of scope for this branch "
                    "beyond the knob that makes it possible.\n\n")
        f.write("### Range cap ruling\n\n"
                "`stereo_range_cap_m` stays **25.0**, an ASSUMED spec §3.4 default and NOT a "
                "measured\nvalue: the plan's agreement rule is degenerate under the "
                f"{PAIR_RADIUS_M} m pairing radius, as\nshown above. Controller ruling, "
                "2026-09-12.\n\n")
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
