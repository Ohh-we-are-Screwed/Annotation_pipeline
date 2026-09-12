#!/usr/bin/env python3
"""Stage 6s — per-mask stereo boxing on the ZED frusta (approach A, 2026-09-12).

One Stage 4 mask -> one box. Geometry from the ZED's stereo points in the ZED's
own optical frame; no clustering. Spec: docs/superpowers/specs/2026-09-12-stereo-box-a-design.md §4.
Emits Stage 6's exact boxes.jsonl row (+ additive `stereo` block) so 7/8/9 run unchanged.

Why there is no DBSCAN here. Stage 6's clustering exists to separate an object
from the reprojection ghost behind it in a LiDAR return set that has no depth
prior. A ZED instance already carries depth for every painted pixel, so the same
separation is one robust statistic on the depth histogram — a MAD trim about the
median — and the ghost is the tail that trim removes. What the trim cannot give
is the far face: stereo sees a SURFACE, so the object's length is unobservable
and comes from the class prior, anchored on the near face rather than centred on
the returns.

**The near face is the 20th percentile, not the median.** On an oblique view the
visible surface runs away from the camera and its median depth sits behind the
nearest point by half the object; anchoring there puts the box half a length too
far. p20 of the MAD-trimmed depths is the near face with a noise margin.

**The centre is pushed by the box's half-extent along the ray, not by l/2.**
With `theta` the angle between the length axis and the BEV ray,
`push = (l/2)|cos theta| + (w/2)|sin theta|`. Head-on that is l/2 as before; a
rickshaw crossing the frame side-on gets w/2, which is 0.6 m nearer — the flat
l/2 rule put the same fixture at x = 12.578 against a truth of 12.0. The ray
itself passes through the MIDPOINT of the same p1/p99 window that measured w and
h, not the lateral median: on an L-shaped visible surface the median bearing sits
on whichever leg carries more points, which side-on is 1.20 m off the centre line.

**Yaw is an L-shape fit, not the footprint's principal axis.** Measured on this
stage's own synthetic acceptance case (a rickshaw's end face plus one side face,
15 cm range noise): PCA answers 52.6 deg where the truth is 20 deg, because an
L-shaped footprint has a centroid off BOTH legs and the resulting cross-moment
rotates the principal axis toward the diagonal. Zhang et al.'s closeness
criterion — already implemented and audited in `stage6_cluster.fit_rectangle`,
reused here rather than copied — answers 20.0 deg on the same points. The PCA
eigenvalue ratio is still computed: it is the isotropy test (a footprint with no
axis falls back to the ray bearing) and it is recorded per box.

**Front ZED is off for this run.** `active_channels` names the ZED channels this
run trusts. The export's CAM_FRONT (ring 101) is pitched ~9 deg with a
range-dependent error that no constant correction can remove
(docs/evidence/2026-09-12-stereo-vs-lidar-chunk_0010.md), so its instances are
recorded with status `channel_disabled` rather than boxed. `stereo_z_correction_m`
and `stereo_pitch_correction` are Stage 1 ingestion knobs: this stage records
them in its manifest and NEVER re-applies them.

    python3 -m pipeline.stage6_stereo_box.stereo_box [--paths configs/paths.yaml]

Exit codes:
    0  every scene produced boxes under contract
    1  ran, but at least one scene is quality-flagged (no box, or more starved
       instances than fitted ones)
    2  upstream contract broken; nothing was written
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Sequence

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.common.conventions import (  # noqa: E402
    CAMERA,
    EGO,
    Transform,
    apply_transform,
    quaternion_from_yaw_rad,
)
from pipeline.common.manifest import (  # noqa: E402
    UpstreamRefusal,
    clear_markers,
    require_upstream,
    write_json_atomic,
    write_jsonl_atomic,
    write_marker,
)
from pipeline.common.paths import (  # noqa: E402
    PathValidationError,
    Paths,
    assert_dataroot_read_only,
    load_paths,
    metadata_fingerprint,
)
from pipeline.stage0_data_probe.probe import Substrate  # noqa: E402
from pipeline.stage1_ingestion.ingest import read_pcd_bin  # noqa: E402
from pipeline.stage6_cluster.cluster import (  # noqa: E402
    PRIORS_SCENE_SUBSET,
    STATUS_FIT,
    ClusterConfig,
    fit_rectangle,
)
from pipeline.stage6_cluster.priors import PRIORS_NAME, Priors, load_priors  # noqa: E402

STAGE = "stage6_stereo_box"
STAGE_SPEC = "dhakascenes-pilot/stage6_stereo_box/v1"

# The channels whose instances are CANDIDATES at all. Anything else is a mask on
# a camera with no stereo cloud behind it and is recorded out_of_r3.
ZED_CHANNELS = ("CAM_FRONT", "CAM_BACK")
STEREO_RINGS = (100.0, 101.0)

EXIT_OK, EXIT_DEGRADED, EXIT_REFUSED = 0, 1, 2

STATUS_OUT_OF_R3 = "out_of_r3"
STATUS_CHANNEL_DISABLED = "channel_disabled"
STATUS_NO_PRIOR = "no_prior"
STATUS_NO_POINTS = "no_points"
STATUS_NO_GROUND = "no_ground_plane"
STATUS_TOO_FEW = "too_few_stereo"
STATUS_BEYOND_CAP = "beyond_stereo_cap"

DEFAULT_CFG = {
    "stereo_range_cap_m": 25.0, "stereo_z_correction_m": {}, "k_mad": 3.0, "mad_floor_m": 0.10,
    "min_stereo_pts": 20, "lidar_refine_min_pts": 5, "eig_ratio_isotropic": 1.5,
    "percentile_lo": 1, "percentile_hi": 99, "prior_clamp_sigma": 2.0, "min_samples": 5,
    "near_face_percentile": 20, "single_face_minor_frac": 0.50,
    # Stage 1 knob, recorded here and never applied here (see the module docstring).
    "stereo_pitch_correction": {},
    # Which ZED channels this run trusts. CAM_FRONT is out for the 2026-09-11
    # export: its extrinsics are pitched and cannot be corrected constantly.
    "active_channels": ["CAM_FRONT", "CAM_BACK"],
}

# The yaw fitter. Reused from Stage 6 rather than reimplemented: the closeness
# criterion, its plateau tie-break and its determinism guarantees are already
# written down and tested there. Only the criterion is pinned; the angle grid is
# Stage 6's, and the manifest says so.
_YAW_FIT_CFG = ClusterConfig(fit_criterion="closeness")

# Below this w:l prior ratio a class's two footprint axes are the same length to
# within the measurement, so the width of ONE visible face cannot say which face
# it is (pedestrian: w 0.77 / l 0.76). Such a box is near-square either way.
_PRIOR_WL_RATIO_MIN = 1.15


def _plane_z(abd, x, y):
    a, b, d = abd
    return a * x + b * y + d


def _clamp_extent(meas: float, mu: float, sigma: float, k: float) -> tuple[float, str | None]:
    """Asymmetric extent clamp (spec Sec 4.2 step 5, controller ruling R20, 2026-09-12).

    A measurement BELOW mu - k*sigma is treated as unreliable, not as a small
    object: occlusion and stereo holes at a depth edge can only SHRINK the
    points a mask owns, never grow them, so a low outlier -> the prior MEAN.
    A measurement ABOVE mu + k*sigma is mask bleed at a depth edge, which CAN
    only grow the footprint and is bounded -> clamps to mu + k*sigma as before.

    Evidence (chunk_0010, 4,247 boxes, rear ZED): median w_meas/mu was
    0.49-0.87 and h_meas/mu 0.35-0.70 across classes, independent of range —
    loosening the MAD trim barely moved it (pedestrian w/mu 0.49 -> 0.52), so
    the bias is not the trim but points that never reach the silhouette. Under
    the old symmetric +/-2 sigma clamp, 83% of boxes were pinned at the prior
    FLOOR (e.g. pedestrians 0.62 x 1.38 m) -- too small.
    """
    lo, hi = mu - k * sigma, mu + k * sigma
    if meas < lo:
        return mu, "low_to_mu"
    if meas > hi:
        return hi, "high"
    return meas, None


def box_from_stereo(pts_ego, rings, *, K, T_ego_cam, prior, ground_abd, cfg):
    """(box | None, status, stereo_block). Spec §4.2 steps 1-9, in that order."""
    T_cam_ego = np.linalg.inv(T_ego_cam)
    cam = apply_transform(T_cam_ego, np.asarray(pts_ego, dtype=np.float64))   # optical: x right, y down, z fwd
    rings = np.asarray(rings)
    is_st = np.isin(rings, STEREO_RINGS)
    is_li = ~is_st
    st = cam[is_st]
    li = cam[is_li]
    stereo = {"n_stereo_pts": int(len(st)), "n_stereo_kept": 0, "d_med_m": None, "d_near_m": None, "mad_m": None,
              "depth_source": None, "w_meas_m": None, "h_meas_m": None, "ray_yaw_rad": None,
              "footprint_eig_ratio": None, "push_m": None, "theta_deg": None, "zed_ring": int(rings[is_st][0]) if is_st.any() else None,
              "n_lidar_in_box": 0, "n_stereo_in_box": 0, "clamp": {"w": None, "h": None},
              "single_face": None, "range_gate_m": None}
    front = st[st[:, 2] > 0.1]
    if len(front) < cfg["min_stereo_pts"]:
        return None, STATUS_TOO_FEW, stereo

    # 1. robust depth: MAD trim about the median, then the NEAR FACE is the 20th
    #    percentile of what survives (spec §4.2 step 7: stereo sees a surface, and
    #    the median of an oblique surface sits behind the nearest point).
    d = front[:, 2]
    d_med = float(np.median(d))
    mad = max(float(np.median(np.abs(d - d_med))), cfg["mad_floor_m"])
    kept = front[np.abs(d - d_med) <= cfg["k_mad"] * mad]
    stereo.update(n_stereo_kept=int(len(kept)), d_med_m=round(d_med, 4), mad_m=round(mad, 4))
    if len(kept) < cfg["min_stereo_pts"]:
        return None, STATUS_TOO_FEW, stereo
    d_near = float(np.percentile(kept[:, 2], cfg["near_face_percentile"]))

    # 2. LiDAR refinement, about the near face
    depth_source = "stereo"
    if len(li):
        li_front = li[li[:, 2] > 0.1]
        band = li_front[np.abs(li_front[:, 2] - d_near) <= 2.0 * mad]
        if len(band) >= cfg["lidar_refine_min_pts"]:
            d_near = float(np.median(band[:, 2]))
            depth_source = "lidar_refined"
    stereo["depth_source"] = depth_source
    stereo["d_near_m"] = round(d_near, 4)

    # 3. measured lateral / vertical extent (pixels at the near face -> metres)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * kept[:, 0] / kept[:, 2] + cx
    v = fy * kept[:, 1] / kept[:, 2] + cy
    lo, hi = cfg["percentile_lo"], cfg["percentile_hi"]
    u_lo, u_hi = np.percentile(u, lo), np.percentile(u, hi)
    v_lo, v_hi = np.percentile(v, lo), np.percentile(v, hi)
    w_meas = float((u_hi - u_lo) * d_near / fx)
    h_meas = float((v_hi - v_lo) * d_near / fy)
    stereo.update(w_meas_m=round(w_meas, 4), h_meas_m=round(h_meas, 4))

    # 4./5. length from the prior; w, h measured and clamped ASYMMETRICALLY
    #       (see `_clamp_extent`): low measurements go to the prior mean, high
    #       measurements are capped at mu + k sigma. `clamped_axes` keeps its
    #       existing meaning (the axis differs from the measurement); the
    #       direction of each clamp is additionally recorded in `stereo["clamp"]`.
    (mu_w, s_w), (mu_l, s_l), (mu_h, s_h) = prior["w"], prior["l"], prior["h"]
    k = cfg["prior_clamp_sigma"]
    w, rule_w = _clamp_extent(w_meas, mu_w, s_w, k)
    h, rule_h = _clamp_extent(h_meas, mu_h, s_h, k)
    clamped = [axis for axis, rule in (("w", rule_w), ("h", rule_h)) if rule is not None]
    stereo["clamp"] = {"w": rule_w, "h": rule_h}
    l = float(mu_l)

    # 7a. the near-face anchor: the MIDPOINT of the same p1/p99 window that
    #     measured w and h, at depth d_near — so the centre and the width are one
    #     measurement. NOT the median bearing: on an L-shaped visible surface the
    #     median sits on whichever leg carries more points, which side-on puts the
    #     anchor 1.18 m off the object's centre line (measured 2026-09-12).
    p_near = np.array([(0.5 * (u_hi + u_lo) - cx) * d_near / fx,
                       (0.5 * (v_hi + v_lo) - cy) * d_near / fy,
                       d_near])
    p_near_ego = apply_transform(T_ego_cam, p_near[None, :])[0]
    stereo["range_gate_m"] = round(float(math.hypot(p_near_ego[0], p_near_ego[1])), 4)

    # 6. yaw. The eigenvalue ratio of the ground-projected footprint is the
    #    ISOTROPY test (and a recorded diagnostic); the angle itself comes from
    #    the L-shape fit, because the principal axis of an L is biased toward the
    #    diagonal by tens of degrees (module docstring).
    kept_ego = apply_transform(T_ego_cam, kept)
    xy = kept_ego[:, :2] - kept_ego[:, :2].mean(axis=0)
    cov = xy.T @ xy / max(1, len(xy) - 1)
    evals, _ = np.linalg.eigh(cov)
    ratio = float(evals[1] / max(evals[0], 1e-9))
    stereo["footprint_eig_ratio"] = round(ratio, 3)
    ray_yaw = math.atan2(p_near_ego[1], p_near_ego[0])
    stereo["ray_yaw_rad"] = round(ray_yaw, 6)
    reasons = ["axis_only"]
    if ratio < cfg["eig_ratio_isotropic"]:
        yaw = ray_yaw
        yaw_source = "ray_bearing"
        reasons.insert(0, "footprint_isotropic")
    else:
        rect = fit_rectangle(xy, _YAW_FIT_CFG)
        u_major = rect.extent_u_m >= rect.extent_v_m
        e_major, e_minor = ((rect.extent_u_m, rect.extent_v_m) if u_major
                            else (rect.extent_v_m, rect.extent_u_m))
        # The bearing of the longer visible extent. WHICH object axis that is
        # depends on how many faces are visible — the next test decides that.
        theta_major = rect.theta_rad if u_major else rect.theta_rad + math.pi / 2.0
        # Two faces (an L): the long BEV side is the heading axis (Stage 6's
        # near-square policy), so `w <= l` survives without a second swap.
        yaw, yaw_source = theta_major, "l_shape_closeness"
        if e_minor < cfg["single_face_minor_frac"] * min(mu_w, mu_l):
            # ONE face visible — the bus of keyframe 575, chunk_0010, seen from
            # directly behind, is a 2.48 m x 1.23 m strip, and "the longer extent
            # is the length" then lays the 11.19 m prior ACROSS the road. A strip
            # IS a face, so its width says WHICH face it is and the length axis
            # follows. The isotropy gate above never catches this: a flat face is
            # strongly ANISOTROPIC (that box's eigenvalue ratio is 6.0).
            if max(mu_w, mu_l) / min(mu_w, mu_l) < _PRIOR_WL_RATIO_MIN:
                matched = "ambiguous"               # near-square prior: fall through
            elif abs(math.log(e_major / mu_w)) <= abs(math.log(e_major / mu_l)):
                matched = "w"                       # front/rear face: length is PERPENDICULAR to it
                yaw, yaw_source = theta_major + math.pi / 2.0, "single_face_prior_match"
            else:
                matched = "l"                       # side face: the strip is the length axis itself
                yaw_source = "single_face_prior_match"
            stereo["single_face"] = {"e_major_m": round(e_major, 4),
                                     "e_minor_m": round(e_minor, 4), "matched": matched}
    yaw = yaw % math.pi                                  # axis only: [0, pi)
    axis_swapped = False
    if w > l:                                            # keep the [w, l, h] invariant
        w, l = l, w
        yaw = (yaw + math.pi / 2) % math.pi
        axis_swapped = True

    # Quantise once, here: every derived field below (the push, the centre, the z
    # extent, the point-in-box test) is then consistent with the [w, l, h] that is
    # actually written, to the last decimal place a reader sees.
    w, l, h = round(w, 4), round(l, 4), round(h, 4)

    # 7b. centre: the near face pushed along the ray by the box's OWN half-extent
    #     in that direction. theta is the angle between the length axis and the
    #     BEV ray. Head-on (theta 0) this is l/2, exactly the old rule; side-on
    #     (theta 90 deg) it is w/2, which is 0.6 m nearer for a rickshaw crossing
    #     the frame — the old l/2 put that box 0.58 m beyond the truth.
    ray_bev = p_near_ego[:2] - np.asarray(T_ego_cam, dtype=np.float64)[:2, 3]
    theta = yaw - math.atan2(ray_bev[1], ray_bev[0])
    push = (l / 2.0) * abs(math.cos(theta)) + (w / 2.0) * abs(math.sin(theta))
    stereo["theta_deg"] = round(math.degrees(theta) % 180.0, 3)
    stereo["push_m"] = round(push, 4)
    c_cam = p_near + (p_near / np.linalg.norm(p_near)) * push
    c_ego = apply_transform(T_ego_cam, c_cam[None, :])[0]

    # 9. range gate (BEV) — on the NEAR FACE, not the centre (controller ruling
    #    R25, 2026-09-12). The cap is a statement about where this stage's stereo
    #    EVIDENCE is trustworthy, and the evidence is the visible face; the centre
    #    is that face extrapolated by a class prior. Gating the extrapolation
    #    penalises exactly the boxes the single-face rule got RIGHT: a bus turned
    #    to face the camera is pushed l/2 = 5.6 m instead of w/2 = 1.5 m, so on
    #    chunk_0010 fitted bus boxes fell 114 -> 44 purely because the correct
    #    orientation moved a correctly measured face past the cap.
    if stereo["range_gate_m"] > cfg["stereo_range_cap_m"]:
        return None, STATUS_BEYOND_CAP, stereo

    # 8. ground snap
    z_min = float(_plane_z(ground_abd, c_ego[0], c_ego[1]))
    z_max = z_min + h
    center = [float(c_ego[0]), float(c_ego[1]), z_min + h / 2.0]

    # points inside the final box (all rings), for num_lidar_pts and the split
    rel = np.asarray(pts_ego, dtype=np.float64) - np.array(center)
    cy_, sy_ = math.cos(-yaw), math.sin(-yaw)
    bx = rel[:, 0] * cy_ - rel[:, 1] * sy_
    by = rel[:, 0] * sy_ + rel[:, 1] * cy_
    inside = (np.abs(bx) <= l / 2) & (np.abs(by) <= w / 2) & (rel[:, 2] >= -h / 2) & (rel[:, 2] <= h / 2)
    stereo["n_lidar_in_box"] = int(np.count_nonzero(inside & is_li))
    stereo["n_stereo_in_box"] = int(np.count_nonzero(inside & is_st))
    box = {
        "translation_m": [round(c, 4) for c in center], "size_wlh_m": [w, l, h],
        "size_order": "w,l,h", "yaw_rad": round(yaw, 6), "rotation_wxyz": [round(q, 9) for q in quaternion_from_yaw_rad(yaw)],
        "yaw_axis_only": True, "yaw_ambiguous": True, "yaw_ambiguous_reasons": reasons, "axis_swapped": axis_swapped,
        "clamped_axes": clamped, "z_min_m": round(z_min, 4), "z_max_m": round(z_max, 4),
        "footprint_diagonal_m": round(math.hypot(w, l), 4), "aspect_ratio_w_over_l": round(w / l, 4),
        "fit": {"method": "per_mask_stereo", "k_mad": cfg["k_mad"], "length_source": "prior_mu",
                "depth_source": depth_source, "anchor": "near_face_at_robust_depth", "bottom": "ground_plane",
                "centre_push": "half_extent_along_ray",
                "yaw_source": yaw_source},
    }
    return box, STATUS_FIT, stereo


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_cfg(path: str | None) -> dict:
    """`DEFAULT_CFG` overlaid with the yaml. Unknown keys are kept, not rejected.

    The yaml is the provenance document for these numbers (every value there
    carries the measurement behind it), so a key this code has not heard of is
    something a later task added, not an error — it rides through into the
    manifest either way.
    """
    cfg = dict(DEFAULT_CFG)
    if path:
        if not os.path.isfile(path):
            raise PathValidationError(f"{path} not found; --config names this stage's tunables")
        with open(path, "r", encoding="utf-8") as fh:
            cfg.update(yaml.safe_load(fh) or {})
    # Ring ids are integers everywhere else in the pipeline; yaml keys are not.
    cfg["stereo_z_correction_m"] = {int(k): float(v) for k, v in (cfg.get("stereo_z_correction_m") or {}).items()}
    cfg["active_channels"] = list(cfg.get("active_channels") or [])
    return cfg


# ---------------------------------------------------------------------------
# Per-scene inputs: the ground plane and the ZED calibrations
# ---------------------------------------------------------------------------


def load_ground_planes(stage1_dir: str, scene: str) -> dict:
    """keyframe_token -> (a, b, d), from Stage 1's own ground reference plane.

    A keyframe whose plane is null is absent from the mapping rather than given
    a default: the box bottom is SNAPPED to this plane, so a guessed one is a
    silent vertical error on every box of that keyframe.
    """
    path = os.path.join(stage1_dir, "scenes", scene, "filter_diagnostics.json")
    if not os.path.isfile(path):
        raise UpstreamRefusal(f"{path} not found; run `python3 -m pipeline.stage1_ingestion.ingest` first")
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    planes = {}
    for entry in payload.get("keyframes", []):
        plane = entry.get("ground_reference_plane")
        if plane is not None:
            planes[entry["keyframe_token"]] = (float(plane["a"]), float(plane["b"]), float(plane["d"]))
    return planes


def load_calibs(stage1_dir: str, scene: str, substrate: Substrate) -> dict:
    """channel -> (K, T_ego_cam) for the ZED channels this scene carries.

    The substrate is passed in rather than re-loaded per scene: its metadata
    tables are the same file on every scene of a run.
    """
    path = os.path.join(stage1_dir, "scenes", scene, "keyframes.jsonl")
    if not os.path.isfile(path):
        raise UpstreamRefusal(f"{path} not found; run `python3 -m pipeline.stage1_ingestion.ingest` first")
    with open(path, "r", encoding="utf-8") as fh:
        first = next((json.loads(line) for line in fh if line.strip()), None)
    if first is None:
        raise UpstreamRefusal(f"{path} is empty; Stage 1 wrote no keyframe for scene {scene!r}")
    calibrated = substrate.by_token("calibrated_sensor.json")
    out = {}
    for channel, camera in (first.get("cameras") or {}).items():
        if channel not in ZED_CHANNELS:
            continue
        record = calibrated[camera["calibrated_sensor_token"]]
        out[channel] = (
            np.array(record["camera_intrinsic"], dtype=np.float64),
            Transform.from_nuscenes(record, source_frame=CAMERA, parent_frame=EGO).matrix(),
        )
    return out


def prior_for(priors: Priors, class_name: str) -> dict | None:
    """`{axis: (mu, sigma)}`, or None when the class has no measured dims.

    Length comes from `mu('l')` and is the ONLY source of length in this stage,
    so a class without dims cannot be boxed at all — it is recorded, not guessed.
    """
    prior = priors.get(class_name)
    if prior is None or prior.dims is None:
        return None
    return {axis: (prior.mu(axis), prior.sigma(axis)) for axis in ("w", "l", "h")}


# ---------------------------------------------------------------------------
# One keyframe
# ---------------------------------------------------------------------------


def _empty_totals() -> dict:
    return {
        "n_keyframes": 0, "n_instances": 0, "n_fit": 0, "n_out_of_r3": 0, "n_channel_disabled": 0,
        "n_too_few_stereo": 0, "n_beyond_stereo_cap": 0, "n_no_points": 0, "n_no_prior": 0,
        "n_no_ground_plane": 0, "n_lidar_refined": 0, "n_clamped_w": 0, "n_clamped_h": 0,
        "n_isotropic_yaw": 0, "n_single_face_yaw": 0, "n_boxes_lidar_lt5": 0,
    }


_STATUS_TOTAL = {
    STATUS_FIT: "n_fit", STATUS_OUT_OF_R3: "n_out_of_r3", STATUS_CHANNEL_DISABLED: "n_channel_disabled",
    STATUS_TOO_FEW: "n_too_few_stereo", STATUS_BEYOND_CAP: "n_beyond_stereo_cap",
    STATUS_NO_POINTS: "n_no_points", STATUS_NO_PRIOR: "n_no_prior", STATUS_NO_GROUND: "n_no_ground_plane",
}


def box_keyframe(lift_row: dict, stage5_dir: str, calibs: dict, ground: dict,
                 priors: Priors, cfg: dict) -> tuple[list[dict], dict]:
    """Every instance of one keyframe: one mask -> one box."""
    cloud = read_pcd_bin(lift_row["cloud_path"])
    points_path = os.path.join(stage5_dir, lift_row["points_path"])
    if not os.path.isfile(points_path):
        raise UpstreamRefusal(f"{points_path} not found; run `python3 -m pipeline.stage5_lift.lift` first")
    with np.load(points_path) as npz:
        point_index = npz["point_index"].astype(np.int64)
        instance_id = npz["instance_id"].astype(np.int64)
    token = lift_row["keyframe_token"]
    abd = ground.get(token)

    envelope = {
        "spec": STAGE_SPEC,
        "keyframe_token": token,
        "scene_token": lift_row["scene_token"],
        "t_ns": lift_row["t_ns"],
        "time_base": lift_row["time_base"],
        "coverage_config": lift_row["coverage_config"],
        "cloud_path": lift_row["cloud_path"],
        "points_path": lift_row["points_path"],
    }

    rows: list[dict] = []
    totals = _empty_totals()
    for inst in lift_row["instances"]:
        rows_of_cloud = point_index[instance_id == inst["instance_id"]]
        eps, eps_src = priors.eps_bev(inst["class_name"], fallback_m=_YAW_FIT_CFG.eps_fallback_m)
        base = {
            **envelope,
            "instance_id": inst["instance_id"], "channel": inst["channel"], "proposal_index": inst["proposal_index"],
            "class_name": inst["class_name"], "score": inst["score"], "n_mask_px": inst["n_mask_px"],
            "n_points_instance": int(len(rows_of_cloud)), "eps_m": round(float(eps), 5), "eps_source": eps_src,
            "min_samples": cfg["min_samples"], "cluster_space": "none:per_mask_stereo", "canonical_sort": "n/a",
            "cluster_tie_break": "n/a", "cloud_kind": "single_sweep", "frame": EGO,
            "num_lidar_pts_basis": "single_sweep_ground_filtered_pre_inflation", "num_lidar_pts": 0,
            "n_points_below_gate": len(rows_of_cloud) < cfg["min_samples"],
            "cluster": None, "near_cut": None, "box": None, "stereo": None,
        }

        prior = prior_for(priors, inst["class_name"])
        if inst["channel"] not in ZED_CHANNELS:
            status, box, stereo = STATUS_OUT_OF_R3, None, None
        elif inst["channel"] not in cfg["active_channels"]:
            status, box, stereo = STATUS_CHANNEL_DISABLED, None, None
        elif prior is None:
            status, box, stereo = STATUS_NO_PRIOR, None, None
        elif len(rows_of_cloud) == 0:
            status, box, stereo = STATUS_NO_POINTS, None, None
        elif abd is None:
            status, box, stereo = STATUS_NO_GROUND, None, None
        elif inst["channel"] not in calibs:
            # An active ZED channel with no calibration is a contract break,
            # not a per-instance quality flag: every box on it would be wrong.
            raise UpstreamRefusal(
                f"keyframe {token}: instance {inst['instance_id']} is on {inst['channel']}, which "
                "carries no calibrated_sensor record in Stage 1's keyframes.jsonl for this scene")
        else:
            K, T = calibs[inst["channel"]]
            box, status, stereo = box_from_stereo(
                cloud[rows_of_cloud, :3].astype(np.float64), cloud[rows_of_cloud, 4],
                K=K, T_ego_cam=T, prior=prior, ground_abd=abd, cfg=cfg,
            )
        row = {**base, "status": status, "box": box, "stereo": stereo}
        if status == STATUS_FIT:
            row["num_lidar_pts"] = stereo["n_lidar_in_box"] + stereo["n_stereo_in_box"]
            totals["n_lidar_refined"] += int(stereo["depth_source"] == "lidar_refined")
            totals["n_clamped_w"] += int("w" in box["clamped_axes"])
            totals["n_clamped_h"] += int("h" in box["clamped_axes"])
            totals["n_isotropic_yaw"] += int("footprint_isotropic" in box["yaw_ambiguous_reasons"])
            totals["n_single_face_yaw"] += int(box["fit"]["yaw_source"] == "single_face_prior_match")
            totals["n_boxes_lidar_lt5"] += int(stereo["n_lidar_in_box"] < 5)
        rows.append(row)
        totals["n_instances"] += 1
        totals[_STATUS_TOTAL[status]] += 1
    return rows, totals


# ---------------------------------------------------------------------------
# Upstream
# ---------------------------------------------------------------------------


def load_upstream(paths: Paths, stage5_dir: str, priors_path: str, *, accept_degraded: bool = False):
    """Refuse to start unless Stage 5 COMPLETED on THIS substrate and the priors match it."""
    current = metadata_fingerprint(paths)
    stage5, marker5 = require_upstream(
        stage5_dir,
        stage_name="Stage 5",
        module_hint="pipeline.stage5_lift.lift",
        current_fingerprint=current,
        accept_degraded=accept_degraded,
    )
    if stage5.get("frame") != EGO:
        raise UpstreamRefusal(f"Stage 5 output claims frame={stage5.get('frame')!r}, not {EGO!r}")

    priors = load_priors(priors_path)
    if not priors.metadata_fingerprint:
        raise UpstreamRefusal(
            f"{priors.path} records no derived_from.metadata_fingerprint; a priors file that does "
            "not bind itself to a substrate cannot be checked against this one"
        )
    if priors.metadata_fingerprint != current:
        raise UpstreamRefusal(
            f"priors fingerprint mismatch: {priors.path} was derived against "
            f"{priors.metadata_fingerprint}, this dataroot is {current}"
        )
    subset = priors.derived_from.get("scene_subset")
    if subset != PRIORS_SCENE_SUBSET:
        raise UpstreamRefusal(
            f"priors scene_subset={subset!r}: {priors.path} was not derived on the "
            f"{PRIORS_SCENE_SUBSET!r} partition, so its class means were tuned on scenes this "
            "pipeline scores (P1-5, §11 decision 3)"
        )
    return stage5, marker5, priors


def read_lift_index(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise UpstreamRefusal(f"{path} not found; run `python3 -m pipeline.stage5_lift.lift` first")
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(paths: Paths, stage5_manifest: dict, stage5_marker, priors: Priors, cfg: dict,
        stage1_dir: str, stage5_dir: str, out_dir: str, scene_names: Sequence[str] | None,
        accept_degraded: bool) -> tuple[dict, int, list[str]]:
    started = time.time()
    # Any marker still standing describes the PREVIOUS run of this stage; it
    # comes down before the first write (C16).
    clear_markers(out_dir)

    root = os.path.join(stage5_dir, "scenes")
    if not os.path.isdir(root):
        raise UpstreamRefusal(f"{root} not found; Stage 5 wrote no scenes")
    names = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
    if scene_names:
        missing = sorted(set(scene_names) - set(names))
        if missing:
            raise UpstreamRefusal(f"requested scene(s) not present in Stage 5 output: {missing}")
        names = [n for n in names if n in scene_names]

    substrate = Substrate.load(paths)
    per_scene: list[dict] = []
    totals = _empty_totals()
    causes: list[str] = []

    for scene_name in names:
        lift_rows = read_lift_index(os.path.join(root, scene_name, "lift.jsonl"))
        ground = load_ground_planes(stage1_dir, scene_name)
        calibs = load_calibs(stage1_dir, scene_name, substrate)
        scene_rows: list[dict] = []
        scene_totals = _empty_totals()

        for lift_row in lift_rows:
            rows, kf_totals = box_keyframe(lift_row, stage5_dir, calibs, ground, priors, cfg)
            scene_rows.extend(rows)
            scene_totals["n_keyframes"] += 1
            for key, value in kf_totals.items():
                scene_totals[key] += value

        write_jsonl_atomic(os.path.join(out_dir, "scenes", scene_name, "boxes.jsonl"), scene_rows)
        # An instance that produced no box is reportable; a scene where nothing
        # fitted, or where starvation beats the fit, means the stage did not do
        # its job on that scene.
        no_box = scene_totals["n_instances"] > 0 and scene_totals["n_fit"] == 0
        starved = scene_totals["n_too_few_stereo"] > scene_totals["n_fit"]
        summary = {"scene": scene_name, **scene_totals, "degraded": bool(no_box or starved)}
        per_scene.append(summary)
        if no_box:
            causes.append(f"{scene_name}: {scene_totals['n_instances']} instance(s), 0 boxes")
        elif starved:
            causes.append(
                f"{scene_name}: {scene_totals['n_too_few_stereo']} instance(s) too_few_stereo vs "
                f"{scene_totals['n_fit']} fitted"
            )
        for key in totals:
            totals[key] += scene_totals[key]
        print(
            f"  {scene_name}  {scene_totals['n_keyframes']:>3} kf  "
            f"{scene_totals['n_instances']:>5} inst  {scene_totals['n_fit']:>5} boxes  "
            f"{scene_totals['n_channel_disabled']:>5} channel-disabled  "
            f"{scene_totals['n_too_few_stereo']:>5} too-few"
            + ("  DEGRADED" if summary["degraded"] else "")
        )

    manifest = {
        "spec": STAGE_SPEC,
        "stage": STAGE,
        # This stage draws no random numbers; the repo-wide seed is recorded so
        # every stage manifest carries the same provenance field (§1.9).
        "seed": _YAW_FIT_CFG.global_seed,
        "config": cfg,
        "upstream": {
            "metadata_fingerprint": stage5_manifest["upstream"]["metadata_fingerprint"],
            "fingerprint_spec": stage5_manifest["upstream"]["fingerprint_spec"],
            "stage5_spec": stage5_manifest["spec"],
            # C16: a run built on accepted degradation says so in its provenance.
            "stage5_degraded": stage5_marker.degraded,
            "stage5_degraded_causes": list(stage5_marker.causes),
            "accepted_degraded_upstream": accept_degraded,
            "priors": {
                **priors.as_reference(),
                "scene_subset": priors.derived_from.get("scene_subset"),
                "scenes": list(priors.derived_from.get("scenes", [])),
            },
        },
        "paths": paths.as_dict(),
        "frame": EGO,
        "cloud_kind": "single_sweep",
        "box_fit": {
            "method": "per_mask_stereo",
            "scope": "per_mask_instance, no clustering",
            "size_order": "w,l,h",
            "length_source": "prior_mu",
            "num_lidar_pts_basis_detail": "painted_points_inside_box_lidar_plus_stereo",
            "width_height_source": "measured from the mask's stereo points ("
                                   f"p{cfg['percentile_lo']}-p{cfg['percentile_hi']} pixel window), "
                                   "clamped asymmetrically to the class prior: below mu -> mu, "
                                   f"above mu + {cfg['prior_clamp_sigma']} sigma -> mu + "
                                   f"{cfg['prior_clamp_sigma']} sigma (controller ruling R20, 2026-09-12)",
            "extent_clamp": "asymmetric_low_to_mu_high_to_plus_k_sigma",
            "extent_percentiles_pct": [cfg["percentile_lo"], cfg["percentile_hi"]],
            "anchor": f"near face at the p{cfg['near_face_percentile']} of the MAD-trimmed depths, "
                      f"on the ray through the midpoint of the p{cfg['percentile_lo']}-"
                      f"p{cfg['percentile_hi']} window that measured w and h",
            "centre_push": "(l/2)|cos theta| + (w/2)|sin theta|, theta = angle(length axis, BEV ray)",
            "bottom": "Stage 1 ground_reference_plane, per keyframe",
            "range_gate": f"near-face BEV range > {cfg['stereo_range_cap_m']} m -> "
                          "beyond_stereo_cap. The gate is on the MEASURED face, not on the "
                          "prior-extrapolated centre (controller ruling R25, 2026-09-12); the range "
                          "it tested is recorded per row as stereo.range_gate_m",
            "yaw_source": "stage6_cluster.fit_rectangle (Zhang closeness, 1 deg grid + 3 refine "
                          "passes); the footprint eigenvalue ratio is the isotropy test only. "
                          "When only ONE face is visible (fitted minor extent < "
                          f"{cfg['single_face_minor_frac']} x min(mu_w, mu_l)) the single-face rule "
                          "decides instead: the visible strip's width is matched in log-ratio to "
                          "mu_w vs mu_l, and a front/rear face puts the LENGTH axis perpendicular "
                          "to it (yaw_source single_face_prior_match, counted in n_single_face_yaw)",
            "yaw_convention": "conventions.py: about +z, from +x, ISO 8855",
            "yaw_axis_only": True,
            "active_channels": list(cfg["active_channels"]),
            "zed_channels": list(ZED_CHANNELS),
            "corrections_applied_here": [],
        },
        "known_gaps": [
            "CAM_FRONT (ring 101) is NOT boxed on this run: active_channels excludes it because the "
            "export's front-ZED extrinsics are pitched ~9 deg with a range-dependent error that no "
            "constant correction removes (docs/evidence/2026-09-12-stereo-vs-lidar-chunk_0010.md). "
            "Its instances carry status channel_disabled and are counted in n_channel_disabled",
            "stereo_z_correction_m and stereo_pitch_correction are Stage 1 ingestion knobs, recorded "
            "in config above and NEVER re-applied here; double-applying them would move every box",
            "box LENGTH is the class prior's mean, not a measurement: stereo sees one surface, so "
            "the far face is unobservable. A class whose prior is wrong produces a box whose length "
            "is wrong by exactly that amount, on every instance of it",
            "the 180 deg heading direction is not decided here: yaw_rad names an AXIS and every row "
            "says so. Stage 7's yaw-consistency enforcement along tracks is the producer of that bit",
            "no clustering means no reprojection-ghost filter beyond the MAD depth trim: a "
            "background surface within k_mad of the object's own depth stays in the fit",
            "num_lidar_pts (basis num_lidar_pts_basis_detail=painted_points_inside_box_lidar_plus_stereo) "
            "counts only the points Stage 5 already painted to this instance (rows_of_cloud), not every "
            "point of the fused single sweep inside the final box: an unpainted LiDAR or stereo return "
            "that happens to fall inside the box's geometry is never counted",
        ],
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "elapsed_s": round(time.time() - started, 2),
        "scenes": per_scene,
        "totals": totals,
    }
    return manifest, (EXIT_DEGRADED if causes else EXIT_OK), causes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--stage5-dir", default=None, help="default <work_root>/stage5_lift")
    parser.add_argument("--stage1-dir", default=None, help="default <work_root>/stage1_ingestion")
    parser.add_argument("--priors", default=None, help=f"default <out_root>/priors/{PRIORS_NAME}.json")
    parser.add_argument("--out-dir", default=None, help=f"default <work_root>/{STAGE}")
    parser.add_argument("--config", default="configs/stereo_box.yaml", help="this stage's tunables")
    parser.add_argument("--scenes", nargs="*", default=None, help="subset of Stage 5 scene names")
    parser.add_argument(
        "--accept-degraded-upstream",
        action="store_true",
        help="consume a DEGRADED (complete, quality-flagged) Stage 5 output; recorded (C16)",
    )
    args = parser.parse_args(argv)

    try:
        paths = load_paths(args.paths)
        cfg = load_cfg(args.config)
    except PathValidationError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    stage1_dir = args.stage1_dir or os.path.join(paths.work_root, "stage1_ingestion")
    stage5_dir = args.stage5_dir or os.path.join(paths.work_root, "stage5_lift")
    priors_path = args.priors or os.path.join(paths.out_root, "priors", f"{PRIORS_NAME}.json")
    out_dir = args.out_dir or os.path.join(paths.work_root, STAGE)
    assert_dataroot_read_only(paths, out_dir)

    try:
        stage5_manifest, stage5_marker, priors = load_upstream(
            paths, stage5_dir, priors_path, accept_degraded=args.accept_degraded_upstream
        )
        manifest, code, causes = run(
            paths, stage5_manifest, stage5_marker, priors, cfg, stage1_dir, stage5_dir, out_dir,
            args.scenes, args.accept_degraded_upstream,
        )
    except UpstreamRefusal as exc:
        print(f"REFUSING TO START: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except PathValidationError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)
    # Three-state marker (§1.9, C16): clean / degraded-with-causes / absent.
    write_marker(out_dir, manifest["upstream"]["metadata_fingerprint"],
                 degraded=code == EXIT_DEGRADED, causes=causes)

    t = manifest["totals"]
    print(f"keyframes            : {t['n_keyframes']}")
    print(f"instances            : {t['n_instances']}")
    print(
        f"boxes                : {t['n_fit']}  ({t['n_too_few_stereo']} too few stereo, "
        f"{t['n_beyond_stereo_cap']} beyond cap, {t['n_no_points']} no points, "
        f"{t['n_no_prior']} no prior, {t['n_no_ground_plane']} no ground plane)"
    )
    print(
        f"not a candidate      : {t['n_out_of_r3']} out of R3, {t['n_channel_disabled']} on a "
        f"disabled channel (active: {', '.join(cfg['active_channels']) or 'none'})"
    )
    print(
        f"depth / extents      : {t['n_lidar_refined']} lidar-refined, {t['n_clamped_w']} w clamped, "
        f"{t['n_clamped_h']} h clamped, {t['n_isotropic_yaw']} isotropic yaw, "
        f"{t['n_single_face_yaw']} single-face yaw"
    )
    print(f"boxes with < 5 lidar : {t['n_boxes_lidar_lt5']} / {t['n_fit']}")
    print(f"wrote {out_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
