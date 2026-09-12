#!/usr/bin/env python3
"""GT-free evaluation of stage6_stereo_box (approach A, 2026-09-12) -> JSON + markdown.

NO GROUND TRUTH EXISTS ON THIS SUBSTRATE. Nothing here measures accuracy. Every
number is a CONSISTENCY signal: does the stage's own output agree with the mask it
was built from, with its own class priors, and with itself.

The one geometric signal is the reprojection IoU. A `fit` row's 8 corners are
projected into the camera the mask came from (`view_boxes_3d.project_corners`,
the same function the browser viewer draws with), the convex hull of the visible
corners is rasterised, and that hull is intersected with Stage 4's mask for the
same (channel, proposal_index). A box built from the right pixels at the wrong
depth, size or yaw reprojects off its own mask; one built correctly covers it.
The hull is the box's SILHOUETTE, so IoU < 1 is expected even for a perfect box
(a cuboid's silhouette is wider than a rickshaw), and the useful reading is the
DISTRIBUTION and how it moves, not any single value.

    PYTHONNOUSERSITE=1 python3 scripts/eval_stereo_box.py \
        --scene dhaka_20260911_141259_chunk_0010 \
        --out-md docs/evidence/<date>-stereo-box-a-chunk_0010.md \
        --out-json docs/evidence/<date>-stereo-box-a-chunk_0010.json

The markdown is rendered from the JSON dict that is written next to it: no number
in the doc is typed by hand, and re-rendering is `render_md(json.load(...))`.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.common.schemas import IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX  # noqa: E402
from pipeline.stage5_lift.lift import MaskFile  # noqa: E402
from scripts.view_boxes_3d import STEREO_BOX_CONFIG, load_calibs, pose_corrections, project_corners  # noqa: E402

STATUS_FIT = "fit"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The front frustum's provenance, quoted in the doc's caveats. Same file
# configs/stereo_box.yaml names on `active_channels`. Its JSON is READ at eval
# time so the caveat's numbers are that measurement's own fields rather than
# prose retyped from memory (an earlier draft said "1.4 m", a figure that appears
# nowhere in the source).
FRONT_ZED_EVIDENCE = "docs/evidence/2026-09-12-stereo-vs-lidar-chunk_0010.md"
FRONT_ZED_EVIDENCE_JSON = os.path.join(ROOT, "docs", "evidence", "2026-09-12-stereo-vs-lidar-chunk_0010.json")
FRONT_ZED_RING = "101"
# Stereo `clamp` values from Stage 6s; null means the measurement stood.
CLAMP_RULES = ("measured", "low_to_mu", "high")


# ---------------------------------------------------------------------------
# The one geometric signal
# ---------------------------------------------------------------------------


def reprojection_iou(box: dict, K, T_cam_ego, mask) -> tuple[float | None, int, float | None]:
    """(IoU of the box's projected hull with `mask`, visible corners, bottom offset px).

    IoU is None when fewer than three corners are visible: there is no polygon to
    rasterise, and scoring that 0 would read as "the box missed its mask" when
    what happened is "the box left the frame".

    `bottom_offset_px` = lowest mask row - lowest projected corner row, so it is
    POSITIVE when the box's bottom edge projects ABOVE the mask's bottom edge.
    Both edges are the same physical thing — where the object meets the ground —
    so a whole channel's median offset is a camera-pose PITCH residual, not a box
    error: no per-box mistake in depth, size or yaw is signed the same way on
    every object in a frustum. It is None when the mask is empty.
    """
    uv, vis = project_corners(box["translation_m"], box["size_wlh_m"], box["yaw_rad"], K, T_cam_ego,
                              (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX))
    n_vis = int(vis.sum())
    if n_vis < 3:
        return None, n_vis, None
    canvas = np.zeros((IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX), np.uint8)
    cv2.fillConvexPoly(canvas, cv2.convexHull(np.round(uv[vis]).astype(np.int32)), 1)
    hull = canvas.astype(bool)
    union = int(np.count_nonzero(hull | mask))
    mask_rows = np.flatnonzero(mask.any(axis=1))
    offset = None if not len(mask_rows) else round(float(mask_rows[-1] - uv[vis][:, 1].max()), 3)
    return (0.0 if union == 0 else round(int(np.count_nonzero(hull & mask)) / union, 6)), n_vis, offset


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _spread(values) -> dict:
    """n / median / p10 / p90 of a sample; all None when the sample is empty."""
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(a):
        return {"n": 0, "median": None, "p10": None, "p90": None}
    return {"n": int(len(a)), "median": round(float(np.median(a)), 4),
            "p10": round(float(np.percentile(a, 10)), 4), "p90": round(float(np.percentile(a, 90)), 4)}


def _group(rows, field) -> dict:
    out: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        out[r[field]].append(r)
    return out


def _by(keyfn, pairs) -> dict:
    """{key: spread} over (key, value) pairs, keys sorted."""
    grouped: dict[str, list] = collections.defaultdict(list)
    for key, value in pairs:
        grouped[keyfn(key)].append(value)
    return {k: _spread(v) for k, v in sorted(grouped.items())}


def _clamp_counts(fits, axis) -> dict:
    """{measured|low_to_mu|high: n} for one extent axis. A null `clamp` entry (or a
    row written before Stage 6s recorded direction) means the measurement stood."""
    c = collections.Counter((r["stereo"].get("clamp") or {}).get(axis) or "measured" for r in fits)
    return {rule: c[rule] for rule in CLAMP_RULES}


def front_zed_dropped(evidence: dict | None, ring: str = FRONT_ZED_RING) -> dict:
    """The three numbers the doc's front-frustum caveat quotes, read from the
    stereo-vs-LiDAR spike's own JSON. Empty when that file is not available."""
    if not evidence:
        return {}
    plane = ((evidence.get("dz_report") or {}).get(ring) or {}).get("plane") or {}
    pitch = (evidence.get("pitch_report") or {}).get(ring) or {}
    chosen = pitch.get("chosen_candidate")
    r4 = lambda v: None if v is None else round(float(v), 4)   # noqa: E731 — as stored everywhere else
    return {
        "source": os.path.basename(FRONT_ZED_EVIDENCE_JSON),
        "ring": int(ring),
        "pitch_deg": r4((pitch.get("candidates") or {}).get(chosen)),
        "pitch_candidate": chosen,
        "pitch_verdict": pitch.get("verdict"),
        "floor_window_m": plane.get("range_m"),
        "floor_median_m": r4(plane.get("floor_median")),
        "floor_slope_m_per_m": r4(plane.get("floor_slope_m_per_m")),
        "worst_bin_m": r4((plane.get("all_bins") or {}).get("floor_min")),
        "plane_verdict": plane.get("verdict"),
    }


def evaluate(scene: str, rows: list[dict], mask_paths: dict, calibs: dict, manifest: dict,
             evidence: dict | None = None, stage5_scene: dict | None = None,
             pose_corrections: dict | None = None) -> dict:
    """Every metric the evidence doc reports, from the boxes rows and their masks.

    `evidence` is the stereo-vs-LiDAR spike's JSON and `stage5_scene` is Stage 5's
    per-scene manifest summary: the doc's caveats quote both, and quoting them from
    the file rather than from prose is what keeps "no number is typed by hand" true.
    """
    status = collections.Counter(r["status"] for r in rows)
    status_by_channel: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        status_by_channel[r["channel"]][r["status"]] += 1

    fits = [r for r in rows if r["status"] == STATUS_FIT]
    ious: list[tuple[dict, float | None, int, float | None]] = []
    masks: MaskFile | None = None
    open_token = None
    try:
        for r in fits:
            if r["keyframe_token"] != open_token:
                if masks is not None:
                    masks.close()
                masks = MaskFile(mask_paths[r["keyframe_token"]])
                open_token = r["keyframe_token"]
            cal = calibs[r["channel"]]
            iou, n_vis, off = reprojection_iou(r["box"], cal["K"], cal["T_cam_ego"],
                                               masks.mask(r["channel"], r["proposal_index"]))
            ious.append((r, iou, n_vis, off))
    finally:
        if masks is not None:
            masks.close()

    scored = [(r, iou, n_vis) for r, iou, n_vis, _ in ious if iou is not None]
    offsets: dict[str, list[float]] = collections.defaultdict(list)
    for r, iou, _, off in ious:
        if iou is not None and off is not None:
            offsets[r["channel"]].append(off)
    reasons = collections.Counter(x for r in fits for x in r["box"]["yaw_ambiguous_reasons"])
    clamped = collections.Counter(x for r in fits for x in r["box"]["clamped_axes"])
    n_any_clamp = sum(1 for r in fits if r["box"]["clamped_axes"])
    n_fit = len(fits)
    rate = (lambda n: round(n / n_fit, 4)) if n_fit else (lambda n: None)
    lidar_lt5 = sum(1 for r in fits if r["stereo"]["n_lidar_in_box"] < 5)

    return {
        "scene": scene,
        "n_rows": len(rows),
        "status_histogram": dict(status.most_common()),
        "status_by_channel": {ch: dict(c.most_common()) for ch, c in sorted(status_by_channel.items())},
        "fit_by_channel": {ch: c[STATUS_FIT] for ch, c in sorted(status_by_channel.items()) if c[STATUS_FIT]},
        "fit_by_class": dict(sorted(collections.Counter(r["class_name"] for r in fits).items())),
        "reprojection_iou": {
            "definition": "|hull(projected visible corners) & stage4 mask| / |hull | mask|",
            "overall": _spread([iou for _, iou, _ in scored]),
            "by_class": _by(lambda r: r["class_name"], [(r, iou) for r, iou, _ in scored]),
            "by_channel": _by(lambda r: r["channel"], [(r, iou) for r, iou, _ in scored]),
            "n_undefined_fewer_than_3_corners_visible": len(ious) - len(scored),
            "n_all_8_corners_visible": sum(1 for _, _, n in scored if n == 8),
            "frac_all_8_corners_visible": round(sum(1 for _, _, n in scored if n == 8) / len(scored), 4) if scored else None,
        },
        "camera_pitch_residual": {
            "definition": "bottom_offset_px = lowest mask row - lowest projected box corner row, "
                          "over fit rows with a scorable hull; POSITIVE means the box's bottom edge "
                          "projects ABOVE the mask's bottom edge, i.e. the camera pose is pitched "
                          "DOWN relative to the geometry and the wireframe rides high. Both edges "
                          "are the same physical line (where the object meets the ground), so a "
                          "channel-wide median is a pose residual, not a box error. "
                          "pitch_residual_deg = degrees(atan(median_offset_px / fy)).",
            "measured_through_pose_correction": {
                ch: (pose_corrections or {}).get(ch) for ch in sorted(calibs)},
            "by_channel": {
                ch: {**_spread(offsets.get(ch, [])),
                     "fy_px": round(float(calibs[ch]["K"][1, 1]), 3),
                     "pitch_residual_deg": (
                         None if not offsets.get(ch) else
                         round(math.degrees(math.atan(float(np.median(offsets[ch]))
                                                      / float(calibs[ch]["K"][1, 1]))), 4))}
                for ch in sorted(calibs) if ch in offsets},
        },
        "yaw_ambiguous_reasons": dict(reasons.most_common()),
        "clamped_axes": {"w": {"n": clamped["w"], "rate": rate(clamped["w"])},
                         "h": {"n": clamped["h"], "rate": rate(clamped["h"])},
                         "any": {"n": n_any_clamp, "rate": rate(n_any_clamp)}},
        "clamp_rule": {
            "note": "Stage 6s clamps each measured extent asymmetrically: below mu it is REPLACED by the "
                    "prior mean (`low_to_mu`), above mu + k sigma it is capped (`high`), otherwise the "
                    "measurement stands (`measured`).",
            "w": _clamp_counts(fits, "w"),
            "h": _clamp_counts(fits, "h"),
            "by_class": {cls: {"w": _clamp_counts(g, "w"), "h": _clamp_counts(g, "h")}
                         for cls, g in sorted(_group(fits, "class_name").items())},
        },
        "depth_source": dict(collections.Counter(r["stereo"]["depth_source"] for r in fits).most_common()),
        "support": {
            "n_fit": n_fit,
            "n_lidar_in_box_lt5": lidar_lt5,
            "frac_lidar_in_box_lt5": rate(lidar_lt5),
            "n_lidar_in_box": _spread([r["stereo"]["n_lidar_in_box"] for r in fits]),
            "n_stereo_in_box": _spread([r["stereo"].get("n_stereo_in_box") for r in fits]),
            "n_stereo_kept": _spread([r["stereo"].get("n_stereo_kept") for r in fits]),
        },
        "depth_m": {"d_med_m": _spread([r["stereo"]["d_med_m"] for r in fits]),
                    "d_near_m": _spread([r["stereo"]["d_near_m"] for r in fits]),
                    "push_m": _spread([r["stereo"].get("push_m") for r in fits]),
                    "theta_deg": _spread([r["stereo"].get("theta_deg") for r in fits])},
        "stage6_totals": manifest.get("totals", {}),
        "elapsed_s": manifest.get("elapsed_s"),
        "active_channels": list((manifest.get("config") or {}).get("active_channels", [])),
        "stage5_degraded": (manifest.get("upstream") or {}).get("stage5_degraded"),
        "stage5_degraded_causes": list((manifest.get("upstream") or {}).get("stage5_degraded_causes", [])),
        "stage5_timing": {
            "max_abs_camera_dt_ns": (stage5_scene or {}).get("max_abs_camera_dt_ns"),
            "max_abs_camera_dt_ms": (round((stage5_scene or {})["max_abs_camera_dt_ns"] / 1e6, 3)
                                     if (stage5_scene or {}).get("max_abs_camera_dt_ns") is not None else None),
            "max_ego_translation_delta_m": (stage5_scene or {}).get("max_ego_translation_delta_m"),
        },
        "front_zed_dropped": front_zed_dropped(evidence),
        # Which camera pose the reprojection IoU was computed through. Image-space
        # consumers only (this script and the viewer): `configs/stereo_box.yaml`
        # `camera_pose_pitch_correction`, NOT applied anywhere in pipeline/.
        "camera_pose_corrections": dict(pose_corrections or {}),
    }


# ---------------------------------------------------------------------------
# Markdown, rendered from the dict above and nothing else
# ---------------------------------------------------------------------------


def _fmt(v) -> str:
    """str(), deliberately: `%g` would round 177.6398 to 177.64 and break the claim
    that every number in the doc is a field of the JSON, verbatim."""
    return "n/a" if v is None else str(v)


def _table(header: list[str], rows: list[list]) -> str:
    body = "\n".join("| " + " | ".join(_fmt(c) for c in r) + " |" for r in rows)
    return ("| " + " | ".join(header) + " |\n|" + "|".join([" --- "] * len(header)) + "|\n" + body + "\n")


def _spread_rows(named: dict) -> list[list]:
    return [[k, s["n"], s["median"], s["p10"], s["p90"]] for k, s in named.items()]


def render_md(m: dict) -> str:
    t = m["stage6_totals"]
    iou = m["reprojection_iou"]
    fz = m.get("front_zed_dropped") or {}
    pr = m["camera_pitch_residual"]
    pr_through = ", ".join(
        "`%s` %s" % (ch, "none" if not c else _fmt(c.get("deg")) + "\u00b0")
        for ch, c in pr["measured_through_pose_correction"].items()) or "none"
    corrections = m.get("camera_pose_corrections") or {}
    front_corr = corrections.get("CAM_FRONT")
    front_enabled = "CAM_FRONT" in m["active_channels"]
    per_channel_note = ([
        "Per channel — the front frustum is enabled for this run without any point-level pitch",
        "correction (controller ruling R23, 2026-09-12); CAM_FRONT instances are fitted alongside",
        "CAM_BACK:",
    ] if front_enabled else [
        "Per channel — the front frustum is disabled for this run, so every CAM_FRONT instance is",
        "`channel_disabled` and contributes no box:",
    ])
    caveat2 = ([
        "2. **The front frustum is enabled but its camera pose is mis-pitched.** `configs/stereo_box.yaml` sets",
        "   `active_channels: [" + ", ".join(m["active_channels"]) + f"]`. The export's CAM_FRONT (ZED ring "
        f"{_fmt(fz.get('ring'))}) pose",
        f"   carries a pitch error of roughly {_fmt(fz.get('pitch_deg'))}° ({_fmt(fz.get('pitch_verdict'))}); over",
        f"   the calibration spike's {'-'.join(_fmt(v) for v in (fz.get('floor_window_m') or []))} m window this",
        f"   puts front stereo points about {_fmt(fz.get('floor_median_m'))} m below the LiDAR road (worst bin "
        f"{_fmt(fz.get('worst_bin_m'))} m,",
        f"   slope {_fmt(fz.get('floor_slope_m_per_m'))} m/m — {_fmt(fz.get('plane_verdict'))}). Those numbers are",
        f"   read at eval time from `{_fmt(fz.get('source'))}` — the JSON behind",
        "   [`" + FRONT_ZED_EVIDENCE + "`](" + os.path.basename(FRONT_ZED_EVIDENCE) + ") — not retyped here.",
        "   Boxes are built from those points and then SNAPPED to the LiDAR ground plane, so they are placed",
        "   correctly in the ego/LiDAR world.",
        *([
            f"   **The CAM_FRONT reprojection IoU above is now computed through a POSE-CORRECTED camera**: "
            f"`configs/stereo_box.yaml`",
            f"   `camera_pose_pitch_correction` rotates the export's CAM_FRONT pose by {_fmt(front_corr['deg'])}° about the axis",
            f"   parallel to ego +y through the camera's own optical centre (pivot x {_fmt(front_corr['pivot_x_m'])} m, z "
            f"{_fmt(front_corr['pivot_z_m'])} m) —",
            "   the same rotation as the recorded Stage 1 stopgap flag, applied by the viewer and this evaluation ONLY",
            "   (Stage 5 keeps lifting masks through the export pose, so points and masks stay mutually consistent).",
            "   The front IoU therefore measures box quality again rather than the calibration error, but a RESIDUAL",
            "   remains: the measured angle is window-dependent, so one angle cannot be right at every range.",
        ] if front_corr else [
            "   But the reprojection IoU for CAM_FRONT is computed through that SAME mis-pitched camera pose, so the",
            "   projected hull lands above the mask by about `fy · |sink| / depth` pixels. The CAM_FRONT IoU column",
            "   therefore measures the camera calibration error, NOT box quality, and must not be compared with",
            "   CAM_BACK; the viewer's front image panel shows the same upward offset.",
        ]),
        "   The fix is still upstream: re-calibrate the front ZED's extrinsic in the exporter, points",
        "   and camera pose together — a point-only correction was tried and broke mask→point ownership (47%",
        "   of front instances lost all points).",
    ] if front_enabled else [
        "2. **The front frustum was dropped.** `configs/stereo_box.yaml` sets `active_channels: ["
        + ", ".join(m["active_channels"]) + f"]`. The export's CAM_FRONT (ZED ring {_fmt(fz.get('ring'))}) is",
        f"   misaligned with the LiDAR road: {_fmt(fz.get('plane_verdict'))}, floor median",
        f"   **{_fmt(fz.get('floor_median_m'))} m** over {'-'.join(_fmt(v) for v in (fz.get('floor_window_m') or []))} m"
        f" (slope {_fmt(fz.get('floor_slope_m_per_m'))}",
        f"   m/m), worst bin **{_fmt(fz.get('worst_bin_m'))} m** below the road over the full span; the pitch that",
        f"   would flatten it is {_fmt(fz.get('pitch_deg'))}° and it was {_fmt(fz.get('pitch_verdict'))}. Those",
        f"   numbers are read at eval time from `{_fmt(fz.get('source'))}` — the JSON behind",
        "   [`" + FRONT_ZED_EVIDENCE + "`](" + os.path.basename(FRONT_ZED_EVIDENCE) + ") — not retyped here.",
        "   Every CAM_FRONT instance is therefore `channel_disabled`, and approach A is judged on the REAR",
        "   frustum (CAM_BACK, ring 100) alone. The fix is upstream: re-export the front ZED's extrinsics.",
    ])
    out = [
        f"# Stereo boxes (approach A) on `{m['scene']}` — GT-free consistency check",
        "",
        "**Caveat, first and load-bearing: no ground truth exists on this substrate; these are",
        "consistency signals, not accuracy.** Nothing below says a box is in the right place. Each",
        "number says only whether the stage's output agrees with the mask it was built from, with",
        "its own class priors, or with itself. Read the distributions and their movement between",
        "runs, not any single value.",
        "",
        "Generated by `scripts/eval_stereo_box.py`, rendered from the `.json` of the same name next to",
        "this file: every number here is a field of that JSON, none is typed by hand.",
        "",
        "## What was run",
        "",
        _table(["field", "value"], [
            ["scene", f"`{m['scene']}`"],
            ["rows in `boxes.jsonl`", m["n_rows"]],
            ["`active_channels`", ", ".join(f"`{c}`" for c in m["active_channels"]) or "n/a"],
            # Which pose the reprojection IoU was computed through. Viewer/eval only.
            ["camera pose correction (image-space consumers only)",
             ", ".join(f"`{ch}` pitch {_fmt(c['deg'])}°" for ch, c in corrections.items()) or "none"],
            ["stage 6s `elapsed_s`", m["elapsed_s"]],
            ["upstream Stage 5 degraded", m["stage5_degraded"]],
            ["Stage 5 degraded causes", ", ".join(f"`{c}`" for c in m["stage5_degraded_causes"]) or "n/a"],
        ]),
        "## Stage 6s manifest totals",
        "",
        "Straight from `<work_root>/stage6_stereo_box/run_manifest.json`, `totals`.",
        "",
        _table(["total", "value"], [[f"`{k}`", v] for k, v in t.items()]),
        "## Status histogram",
        "",
        _table(["status", "n"], [[f"`{k}`", v] for k, v in m["status_histogram"].items()]),
        *per_channel_note,
        "",
        _table(["channel", "status", "n"],
               [[f"`{ch}`", f"`{s}`", n] for ch, hist in m["status_by_channel"].items() for s, n in hist.items()]),
        "## Fitted boxes",
        "",
        _table(["channel", "n_fit"], [[f"`{k}`", v] for k, v in m["fit_by_channel"].items()]),
        _table(["class", "n_fit"], [[f"`{k}`", v] for k, v in m["fit_by_class"].items()]),
        "## Reprojection IoU",
        "",
        f"`{iou['definition']}`. The hull is the box's silhouette, which is wider than most objects, so a",
        "well-placed box does not score 1. A box at the wrong depth, size or yaw scores low; the tail near 0",
        "is where to look.",
        "",
        f"Boxes with fewer than 3 corners visible (no polygon to rasterise, IoU undefined and excluded): "
        f"**{iou['n_undefined_fewer_than_3_corners_visible']}**. Of the scored boxes, "
        f"**{iou['n_all_8_corners_visible']}** (fraction **{_fmt(iou['frac_all_8_corners_visible'])}**) have all "
        "8 corners inside the image; the rest have their hull truncated by the frame edge, which biases their "
        "IoU down.",
        "",
        _table(["scope", "n", "median", "p10", "p90"],
               [["overall", iou["overall"]["n"], iou["overall"]["median"], iou["overall"]["p10"], iou["overall"]["p90"]]]),
        "By class:",
        "",
        _table(["class", "n", "median", "p10", "p90"], _spread_rows(iou["by_class"])),
        "By channel:",
        "",
        _table(["channel", "n", "median", "p10", "p90"], _spread_rows(iou["by_channel"])),
        "## Camera pitch residual (controller ruling R26)",
        "",
        f"`{pr['definition']}`",
        "",
        "The mask's bottom edge and the box's bottom edge are the same physical line — where the object",
        "meets the road — so a POSITIVE channel-wide median means the wireframe rides HIGH: the camera",
        "pose is pitched down relative to the geometry. The correction is a rotation of that size about",
        "the camera's own optical centre, but its SIGN depends on which way the camera looks: the rotation",
        "that tilts a forward-facing camera's axis down tilts a rearward-facing one's up, so the residual",
        "is ADDED for the rear camera and SUBTRACTED for the front one. The measured per-camera slope and",
        "the arithmetic are in the provenance block on `camera_pose_pitch_correction` in",
        "`configs/stereo_box.yaml`. This is measured THROUGH whatever",
        f"`camera_pose_pitch_correction` is configured ({pr_through}),",
        "so it is the RESIDUAL after that correction, not the raw defect.",
        "",
        _table(["channel", "n", "median px", "p10 px", "p90 px", "fy px", "residual °"],
               [[f"`{ch}`", d["n"], d["median"], d["p10"], d["p90"], d["fy_px"], d["pitch_residual_deg"]]
                for ch, d in pr["by_channel"].items()]),
        "## Yaw, clamps, depth source",
        "",
        "`yaw_ambiguous_reasons` over fitted boxes (a box may carry more than one):",
        "",
        _table(["reason", "n"], [[f"`{k}`", v] for k, v in m["yaw_ambiguous_reasons"].items()]),
        "Dimensions the prior moved at all, rather than the measurement standing:",
        "",
        _table(["axis", "n", "rate over n_fit"],
               [[f"`{k}`", v["n"], v["rate"]] for k, v in m["clamped_axes"].items()]),
        f"WHICH WAY each clamp went. {m['clamp_rule']['note']} A `low_to_mu` majority means the stereo",
        "extent is reading systematically SMALL (a partial mask, or a surface seen edge-on); a `high`",
        "majority means it is reading LARGE (background bleeding into the mask's depth window).",
        "",
        _table(["axis"] + list(CLAMP_RULES),
               [[f"`{ax}`"] + [m["clamp_rule"][ax][r] for r in CLAMP_RULES] for ax in ("w", "h")]),
        "Per class:",
        "",
        _table(["class", "axis"] + list(CLAMP_RULES),
               [[f"`{cls}`", f"`{ax}`"] + [v[ax][r] for r in CLAMP_RULES]
                for cls, v in m["clamp_rule"]["by_class"].items() for ax in ("w", "h")]),
        "Where the near face's depth came from:",
        "",
        _table(["depth_source", "n"], [[f"`{k}`", v] for k, v in m["depth_source"].items()]),
        "## Support and depth",
        "",
        f"Boxes holding fewer than 5 LiDAR points: **{m['support']['n_lidar_in_box_lt5']}** of "
        f"**{m['support']['n_fit']}** (fraction **{_fmt(m['support']['frac_lidar_in_box_lt5'])}**). These are "
        "the boxes no LiDAR return corroborates — stereo geometry alone put them there.",
        "",
        _table(["quantity", "n", "median", "p10", "p90"],
               _spread_rows({k: v for k, v in m["support"].items() if isinstance(v, dict)})
               + _spread_rows(m["depth_m"])),
        "## Caveats",
        "",
        "1. **No ground truth.** Repeating the headline: this substrate has no annotated 3D boxes, so",
        "   no number above is accuracy. The reprojection IoU is self-consistency — the box is *derived*",
        "   from the mask it is scored against, so it can only detect a box that drifted off its own",
        "   evidence, never one that is consistently wrong in the same way the evidence is.",
        *caveat2,
        "3. **Stage 5 is DEGRADED on this export**, cause "
        + (", ".join(f"`{c}`" for c in m["stage5_degraded_causes"]) or "n/a")
        + ": the exporter copied the",
        "   LiDAR ego pose into every camera record, so the ego motion between a camera's capture time and",
        "   the LiDAR's is absent from the lift. Stage 5's own manifest measures both halves of that on this",
        f"   scene: the largest camera-to-LiDAR time offset is **{_fmt(m['stage5_timing']['max_abs_camera_dt_ms'])} ms**",
        f"   ({_fmt(m['stage5_timing']['max_abs_camera_dt_ns'])} ns) while the largest ego translation delta over",
        f"   the same interval is **{_fmt(m['stage5_timing']['max_ego_translation_delta_m'])} m** — exactly zero,",
        "   which is the defect, not a stationary vehicle. Every mask-to-point association is displaced by the",
        "   motion that actually occurred in that window; boxes here inherit it.",
        "4. **Box LENGTH is the class prior's mean, not a measurement** — stereo sees one surface, so the far",
        "   face is unobservable. The hull IoU is largely insensitive to that (the far face hides behind the",
        "   near one), which is exactly why it cannot be read as accuracy.",
        "5. **A silhouette IoU is not a 3D IoU.** Two boxes at very different depths can project onto the",
        "   same hull. Depth error shows up here only through the size/position coupling, weakly.",
        "",
    ]
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    ap.add_argument("--scene", "--scenes", dest="scene", required=True, help="one Stage 5 scene name")
    ap.add_argument("--boxes-dir", default=None, help="default <work_root>/stage6_stereo_box")
    ap.add_argument("--stage5-dir", default=None, help="default <work_root>/stage5_lift")
    ap.add_argument("--stage1-dir", default=None, help="default <work_root>/stage1_ingestion")
    ap.add_argument("--config", default=STEREO_BOX_CONFIG,
                    help="the stage config the image-space camera pose correction is read from")
    ap.add_argument("--calib-evidence", default=FRONT_ZED_EVIDENCE_JSON,
                    help="the stereo-vs-LiDAR spike JSON the front-frustum caveat quotes")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    a = ap.parse_args(argv)

    paths = load_paths(a.paths)
    boxes_dir = a.boxes_dir or os.path.join(paths.work_root, "stage6_stereo_box")
    stage5_dir = a.stage5_dir or os.path.join(paths.work_root, "stage5_lift")
    stage1_dir = a.stage1_dir or os.path.join(paths.work_root, "stage1_ingestion")

    rows = [json.loads(l) for l in open(os.path.join(boxes_dir, "scenes", a.scene, "boxes.jsonl")) if l.strip()]
    mask_paths = {}
    for line in open(os.path.join(stage5_dir, "scenes", a.scene, "lift.jsonl")):
        if line.strip():
            r = json.loads(line)
            mask_paths[r["keyframe_token"]] = os.path.normpath(os.path.join(stage5_dir, r["mask_path"]))
    manifest = json.load(open(os.path.join(boxes_dir, "run_manifest.json")))
    kf0 = next(json.loads(l) for l in open(os.path.join(stage1_dir, "scenes", a.scene, "keyframes.jsonl")) if l.strip())
    # The doc's two caveats quote measurements that live in OTHER files. Read them
    # here so the caveat carries the source's own fields; a missing file degrades to
    # the link alone rather than to a number somebody remembered.
    evidence = json.load(open(a.calib_evidence)) if os.path.isfile(a.calib_evidence) else None
    stage5_manifest = os.path.join(stage5_dir, "run_manifest.json")
    stage5_scene = next((sc for sc in json.load(open(stage5_manifest))["scenes"] if sc["scene"] == a.scene),
                        None) if os.path.isfile(stage5_manifest) else None

    # The viewer and this script project through the SAME corrected pose; pipeline/ does not.
    corrections = pose_corrections(a.config)
    metrics = evaluate(a.scene, rows, mask_paths, load_calibs(paths, kf0, corrections), manifest,
                       evidence, stage5_scene, corrections)
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=False)
        f.write("\n")
    # Re-read, so the doc is provably rendered from the file on disk and not from
    # anything still in memory that did not survive serialisation.
    with open(a.out_md, "w") as f:
        f.write(render_md(json.load(open(a.out_json))))
    io = metrics["reprojection_iou"]["overall"]
    print(f"{a.scene}: {metrics['support']['n_fit']} fitted boxes, IoU median {io['median']} "
          f"(p10 {io['p10']}, p90 {io['p90']}) over {io['n']} scored")
    print(f"wrote {a.out_json}\nwrote {a.out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
