"""eval_stereo_box.py: the reprojection IoU on known answers, and the summary (2026-09-12).

There is no ground truth on this substrate, so the evaluation's own arithmetic is
the only thing a test can pin. Two known answers do that: a mask that IS the box's
own projected hull must score 1, and a mask disjoint from it must score 0. The
third test walks a four-row synthetic scene end to end so the histograms, the
percentiles and the markdown rendering are exercised against hand-countable input.
"""
from __future__ import annotations
import json, os, sys
import cv2
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.common.schemas import IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX  # noqa: E402
from scripts.eval_stereo_box import evaluate, render_md, reprojection_iou  # noqa: E402
from scripts.view_boxes_3d import project_corners  # noqa: E402

# The viewer test's rig: 953 px focal, camera 0.8 m ahead of the ego origin,
# optical axis along +x. A box at 12 m lands well inside the frame.
K = np.array([[953.16, 0.0, 656.28], [0.0, 953.16, 375.74], [0.0, 0.0, 1.0]])
T_CAM_EGO = np.array([[0, -1.0, 0, 0], [0, 0, -1.0, -0.7], [1.0, 0, 0, -0.8], [0, 0, 0, 1.0]])
BOX = {"translation_m": [12.0, 0.0, -1.5], "size_wlh_m": [1.15, 2.4, 1.75], "yaw_rad": 0.3,
       "clamped_axes": ["w"], "yaw_ambiguous_reasons": ["axis_only"]}


def own_hull_mask(box) -> np.ndarray:
    """The box's own projected hull, rasterised independently of the script."""
    uv, vis = project_corners(box["translation_m"], box["size_wlh_m"], box["yaw_rad"], K, T_CAM_EGO,
                             (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX))
    canvas = np.zeros((IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX), np.uint8)
    cv2.fillConvexPoly(canvas, cv2.convexHull(np.round(uv[vis]).astype(np.int32)), 1)
    return canvas.astype(bool)


def test_iou_of_a_box_against_its_own_projected_hull_is_one():
    iou, n_vis = reprojection_iou(BOX, K, T_CAM_EGO, own_hull_mask(BOX))
    assert n_vis == 8
    assert iou is not None and iou > 0.99


def test_iou_of_a_disjoint_mask_is_zero():
    mask = np.zeros((IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX), bool)
    mask[:40, :40] = True                                   # top-left corner; the box is mid-frame
    iou, _ = reprojection_iou(BOX, K, T_CAM_EGO, mask)
    assert iou == 0.0


def test_iou_is_undefined_when_the_box_is_behind_the_camera():
    behind = {**BOX, "translation_m": [-12.0, 0.0, -1.5]}
    iou, n_vis = reprojection_iou(behind, K, T_CAM_EGO, own_hull_mask(BOX))
    assert iou is None and n_vis == 0


def _write_scene(tmp_path) -> tuple[list[dict], dict]:
    """Four instances: two fitted on CAM_BACK, one channel_disabled, one too_few_stereo."""
    far = {**BOX, "translation_m": [18.0, 2.0, -1.5], "size_wlh_m": [0.9, 1.8, 1.6],
           "clamped_axes": [], "yaw_ambiguous_reasons": ["footprint_isotropic", "axis_only"]}
    masks = np.stack([own_hull_mask(BOX), np.zeros((IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX), bool)]).astype(np.uint8)
    masks[1, :40, :40] = 1                                  # proposal 1's mask misses its box entirely
    mask_path = os.path.join(tmp_path, "masks_kf1.npz")
    np.savez(mask_path, CAM_BACK=masks, __width_px__=np.array([IMAGE_WIDTH_PX]),
             __height_px__=np.array([IMAGE_HEIGHT_PX]), __bit_packed__=np.array([0]))

    def row(**kw):
        base = {"keyframe_token": "kf1", "channel": "CAM_BACK", "class_name": "car",
                "proposal_index": 0, "instance_id": 0, "status": "fit", "box": None, "stereo": None}
        return {**base, **kw}

    rows = [
        row(instance_id=0, proposal_index=0, box=BOX,
            stereo={"depth_source": "stereo", "d_med_m": 11.0, "d_near_m": 10.0, "n_lidar_in_box": 9}),
        row(instance_id=1, proposal_index=1, class_name="rickshaw", box=far,
            stereo={"depth_source": "lidar_refined", "d_med_m": 17.0, "d_near_m": 16.0, "n_lidar_in_box": 2}),
        row(instance_id=2, proposal_index=0, channel="CAM_FRONT", status="channel_disabled"),
        row(instance_id=3, proposal_index=2, class_name="rickshaw", status="too_few_stereo",
            stereo={"depth_source": None, "d_med_m": 30.0, "d_near_m": None, "n_lidar_in_box": 0}),
    ]
    return rows, {"kf1": mask_path}


def test_evaluate_and_render_a_tiny_scene(tmp_path):
    rows, mask_paths = _write_scene(str(tmp_path))
    manifest = {"elapsed_s": 12.5, "config": {"active_channels": ["CAM_BACK"]},
                "totals": {"n_instances": 4, "n_fit": 2, "n_channel_disabled": 1, "n_too_few_stereo": 1,
                           "n_out_of_r3": 0, "n_no_prior": 0, "n_no_points": 0, "n_no_ground_plane": 0,
                           "n_beyond_stereo_cap": 0, "n_lidar_refined": 1, "n_clamped_w": 1, "n_clamped_h": 0,
                           "n_isotropic_yaw": 1, "n_boxes_lidar_lt5": 1}}
    calibs = {"CAM_BACK": {"K": K, "T_cam_ego": T_CAM_EGO}, "CAM_FRONT": {"K": K, "T_cam_ego": T_CAM_EGO}}

    m = evaluate("scene_x", rows, mask_paths, calibs, manifest)

    assert m["scene"] == "scene_x"
    assert m["status_histogram"] == {"fit": 2, "channel_disabled": 1, "too_few_stereo": 1}
    assert m["status_by_channel"]["CAM_BACK"] == {"fit": 2, "too_few_stereo": 1}
    assert m["status_by_channel"]["CAM_FRONT"] == {"channel_disabled": 1}
    assert m["fit_by_channel"] == {"CAM_BACK": 2}
    assert m["fit_by_class"] == {"car": 1, "rickshaw": 1}
    # one mask is the box's own hull (IoU 1), the other is disjoint from it (IoU 0)
    assert m["reprojection_iou"]["overall"]["n"] == 2
    assert m["reprojection_iou"]["overall"]["median"] == 0.5
    assert m["reprojection_iou"]["by_class"]["car"]["median"] > 0.99
    assert m["reprojection_iou"]["by_class"]["rickshaw"]["median"] == 0.0
    assert m["reprojection_iou"]["by_channel"]["CAM_BACK"]["n"] == 2
    assert m["yaw_ambiguous_reasons"] == {"axis_only": 2, "footprint_isotropic": 1}
    assert m["clamped_axes"] == {"w": {"n": 1, "rate": 0.5}, "h": {"n": 0, "rate": 0.0},
                                 "any": {"n": 1, "rate": 0.5}}
    assert m["depth_source"] == {"stereo": 1, "lidar_refined": 1}
    assert m["support"]["n_lidar_in_box_lt5"] == 1 and m["support"]["frac_lidar_in_box_lt5"] == 0.5
    assert m["depth_m"]["d_med_m"]["median"] == 14.0        # fitted rows only: 11 and 17
    assert m["depth_m"]["d_near_m"]["median"] == 13.0
    assert m["stage6_totals"] == manifest["totals"]
    assert m["elapsed_s"] == 12.5

    md = render_md(m)
    assert "no ground truth exists on this substrate" in md
    assert "2026-09-12-stereo-vs-lidar-chunk_0010.md" in md          # why the front frustum is out
    assert "ego_motion_between_capture_times_absent" in md           # the Stage 5 caveat
    # every number in the doc comes from the JSON
    assert "12.5" in md and "0.5" in md
    json.loads(json.dumps(m))                                        # the metrics dict is JSON-serialisable
