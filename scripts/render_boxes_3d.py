#!/usr/bin/env python3
"""OpenCV: OUR 3D boxes and the nuScenes HUMAN 3D annotations, on the images.

Both box sets are drawn as projected cuboids on the six ring cameras, in one
frame of reference, through ONE projection chain — so this is a check of the
geometry, not only of the detector. A box that floats off its object here is a
frame or a timing error upstream; a box that sits on the right object with the
wrong extent is Stage 6/8 doing its job badly. Those two failures look nothing
alike on screen and identical in a metric.

    OURS   per-class colour, solid; the heading (+x) face is drawn thicker
    HUMAN  green (#2ecc71), the same green the CVAT answer-key project uses

Per selected keyframe, under <work_root>/viz_3d/<scene>/<token>/:
    <CHANNEL>.png   one camera
    grid.png        all six as a 3x2 contact sheet

**The projection is the pipeline's own** (`conventions.project_lidar_to_image`,
the four-hop chain of §1.3): ego(t_lidar) -> global -> ego(t_cam) -> camera ->
pixel. It is not re-derived here. Dropping the two middle hops still draws
boxes on objects, just several pixels along the direction of travel — the exact
silent error that chain exists to prevent — so a viewer that re-implemented it
could disagree with the pipeline and look right.

**Near-plane clipping without a second chain.** A cuboid straddling the camera
plane has corners behind it, and those cannot be projected. Rather than clip in
camera space — which would mean composing the hops here, i.e. a second
implementation of the thing being checked — each edge is SAMPLED along its
length and the samples go through the same chain. The samples in front of the
camera survive the chain's own cull and are drawn as a polyline; the rest
disappear. The clip is exact to the sample spacing and the chain stays single.

Diagnostic only: no markers, nothing downstream reads these files.

    python -m scripts.render_boxes_3d [--scenes scene-0061] [--every 8]
    python -m scripts.render_boxes_3d --boxes-dir <work_root>/stage6_cluster
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.conventions import (  # noqa: E402
    CAMERA,
    EGO,
    NUSCENES_GLOBAL,
    Transform,
    apply_transform,
    project_lidar_to_image,
    quaternion_to_rotation_matrix,
)
from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.common.schemas import IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX, RING_CAMERAS  # noqa: E402
from pipeline.stage0_data_probe.probe import Substrate  # noqa: E402

# The answer key's green, kept identical to CVAT_GT_LABEL_COLOR in
# scripts/run_stages.sh: the same colour must mean "human" everywhere.
GT_COLOR_BGR = (113, 204, 46)  # #2ecc71
EDGE_SAMPLES = 24  # per cuboid edge; the near-plane clip is exact to this spacing

# 8 corners in box frame, x = length (heading), y = width, z = height.
_CORNER_SIGNS = np.array(
    [
        [+1, +1, -1], [+1, -1, -1], [-1, -1, -1], [-1, +1, -1],  # 0-3 bottom
        [+1, +1, +1], [+1, -1, +1], [-1, -1, +1], [-1, +1, +1],  # 4-7 top
    ],
    dtype=np.float64,
)
_EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),  # top
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
)
# The +x face: drawn thicker so heading is readable without a separate arrow.
_HEADING_EDGES = frozenset({(0, 1), (4, 5), (0, 4), (1, 5)})


def class_color_bgr(name: str) -> tuple[int, int, int]:
    """Stable per-class hue, same derivation as render_annotations.py."""
    h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, 0.85, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def box_corners_ego(center_m, size_wlh_m, rotation) -> np.ndarray:
    """(8, 3) cuboid corners. `size_wlh_m` is nuScenes order (width, length, height).

    `rotation` is a (3, 3) matrix or a wxyz quaternion. Both appear: our boxes
    carry `rotation_wxyz` straight from Stage 6, while a GT box arrives as a
    global-frame quaternion that has already been composed with the inverse ego
    pose, i.e. as a matrix.
    """
    w, l, h = (float(v) for v in size_wlh_m)
    half = np.array([l / 2.0, w / 2.0, h / 2.0], dtype=np.float64)
    R = np.asarray(rotation, dtype=np.float64)
    if R.shape != (3, 3):
        R = quaternion_to_rotation_matrix(rotation)
    return (_CORNER_SIGNS * half) @ R.T + np.asarray(center_m, dtype=np.float64)


def edge_sample_points(corners: np.ndarray) -> np.ndarray:
    """((12 * EDGE_SAMPLES), 3) points along the cuboid's edges, edge-major."""
    t = np.linspace(0.0, 1.0, EDGE_SAMPLES)[:, None]
    return np.concatenate([corners[a] * (1 - t) + corners[b] * t for a, b in _EDGES])


def draw_cuboid(canvas, uv_by_sample, visible, color, *, label="", thin=1, thick=3):
    """Draw the runs of consecutive VISIBLE samples of each edge as polylines.

    An edge with no visible sample is simply absent — the part of the cuboid
    behind the camera. Returns the anchor pixel for a label, or None.
    """
    anchor = None
    for edge_index, (a, b) in enumerate(_EDGES):
        lo = edge_index * EDGE_SAMPLES
        seg_uv = uv_by_sample[lo : lo + EDGE_SAMPLES]
        seg_ok = visible[lo : lo + EDGE_SAMPLES]
        width = thick if (a, b) in _HEADING_EDGES else thin
        run: list = []
        for point, ok in zip(seg_uv, seg_ok):
            if ok:
                run.append(point)
                continue
            if len(run) > 1:
                cv2.polylines(canvas, [np.asarray(run, np.int32)], False, color, width, cv2.LINE_AA)
            run = []
        if len(run) > 1:
            cv2.polylines(canvas, [np.asarray(run, np.int32)], False, color, width, cv2.LINE_AA)
        for point, ok in zip(seg_uv, seg_ok):
            if ok and (anchor is None or point[1] < anchor[1]):
                anchor = point
    if label and anchor is not None:
        x, y = int(anchor[0]), max(14, int(anchor[1]) - 6)
        cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return anchor


def legend(canvas, lines):
    pad, line_h = 8, 22
    box_w = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for t, _ in lines) + 2 * pad
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (box_w, pad + line_h * len(lines)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for i, (text, color) in enumerate(lines):
        cv2.putText(canvas, text, (pad, pad + line_h * (i + 1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def read_boxes(boxes_dir: str, scene: str) -> dict[str, list[dict]]:
    """keyframe token -> rows that actually carry a box (status 'fit')."""
    for name in ("inflated.jsonl", "boxes.jsonl"):
        path = os.path.join(boxes_dir, "scenes", scene, name)
        if os.path.isfile(path):
            break
    else:
        raise SystemExit(f"no inflated.jsonl or boxes.jsonl under {boxes_dir}/scenes/{scene}")
    out: dict[str, list[dict]] = {}
    for line in open(path):
        row = json.loads(line)
        # A row without a box is a real Stage 6 outcome (no_points,
        # below_min_samples, all_noise), not a missing field. Counted by the
        # caller so the legend can say how many of the proposals never became
        # a box — otherwise the picture flatters the run.
        out.setdefault(row["keyframe_token"], []).append(row)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--boxes-dir", default=None, help="default <work_root>/stage8_inflate")
    parser.add_argument("--out-dir", default=None, help="default <work_root>/viz_3d")
    parser.add_argument("--every", type=int, default=8, help="render every Nth keyframe")
    parser.add_argument("--cameras", nargs="*", default=list(RING_CAMERAS))
    parser.add_argument("--no-gt", action="store_true", help="ours only")
    parser.add_argument("--no-ours", action="store_true", help="the answer key only")
    parser.add_argument("--min-gt-lidar-pts", type=int, default=1,
                        help="skip GT boxes with fewer raw LiDAR returns; 0 draws every annotation")
    args = parser.parse_args(argv)

    paths = load_paths(args.paths)
    boxes_dir = args.boxes_dir or os.path.join(paths.work_root, "stage8_inflate")
    out_root = args.out_dir or os.path.join(paths.work_root, "viz_3d")
    stage1 = os.path.join(paths.work_root, "stage1_ingestion")

    substrate = Substrate.load(paths)
    ego_poses = substrate.by_token("ego_pose.json")
    calibrated = substrate.by_token("calibrated_sensor.json")
    instances = substrate.by_token("instance.json")
    categories = substrate.by_token("category.json")
    annotations: dict[str, list[dict]] = {}
    for ann in substrate.tables["sample_annotation.json"]:
        annotations.setdefault(ann["sample_token"], []).append(ann)

    scene_root = os.path.join(stage1, "scenes")
    names = sorted(n for n in os.listdir(scene_root) if os.path.isdir(os.path.join(scene_root, n)))
    if args.scenes:
        names = [n for n in names if n in args.scenes]

    n_written = 0
    for scene in names:
        by_keyframe = read_boxes(boxes_dir, scene)
        keyframes = [json.loads(l) for l in open(os.path.join(scene_root, scene, "keyframes.jsonl"))]
        for keyframe in keyframes[:: max(1, args.every)]:
            token = keyframe["keyframe_token"]
            rows = by_keyframe.get(token, [])
            ours = [] if args.no_ours else [r for r in rows if r.get("box")]
            n_no_box = sum(1 for r in rows if not r.get("box"))

            ego_pose_lidar = Transform.from_nuscenes(
                ego_poses[keyframe["lidar_ego_pose_token"]], source_frame=EGO, parent_frame=NUSCENES_GLOBAL
            )
            # GT is stored in the global frame; it comes back through the SAME
            # lidar ego_pose the pipeline used, so a pose mix-up shows up as the
            # answer key sliding off its own objects.
            t_global_ego = ego_pose_lidar.inverse_matrix()
            gt = []
            if not args.no_gt:
                for ann in annotations.get(token, []):
                    if int(ann.get("num_lidar_pts", 0)) < args.min_gt_lidar_pts:
                        continue
                    center = apply_transform(t_global_ego, np.asarray(ann["translation"], np.float64)[None, :])[0]
                    rotation_ego = quaternion_to_rotation_matrix(ego_pose_lidar.rotation_wxyz).T @ \
                        quaternion_to_rotation_matrix(ann["rotation"])
                    gt.append((center, ann["size"], rotation_ego,
                               categories[instances[ann["instance_token"]]["category_token"]]["name"]))

            panels: dict[str, np.ndarray] = {}
            for channel in args.cameras:
                camera = keyframe["cameras"].get(channel)
                if camera is None:
                    continue
                image = cv2.imread(os.path.join(paths.dataroot, camera["path"]))
                if image is None:
                    continue
                cs = calibrated[camera["calibrated_sensor_token"]]
                extrinsic = Transform.from_nuscenes(cs, source_frame=CAMERA, parent_frame=EGO)
                ego_pose_camera = Transform.from_nuscenes(
                    ego_poses[camera["ego_pose_token"]], source_frame=EGO, parent_frame=NUSCENES_GLOBAL
                )
                intrinsic = np.asarray(cs["camera_intrinsic"], dtype=np.float64)

                # Every cuboid of both sets goes through the chain in ONE call:
                # the ego-motion delta between t_lidar and t_cam is a property of
                # the frame pair, and projecting the two sets separately would
                # invite them to be projected differently.
                specs = [(box_corners_ego(np.asarray(r["box"]["translation_m"], np.float64),
                                          r["box"]["size_wlh_m"], r["box"]["rotation_wxyz"]),
                          class_color_bgr(r["class_name"]),
                          f'{r["class_name"]} {r.get("score", 0):.2f}') for r in ours]
                specs += [(box_corners_ego(c, s, R), GT_COLOR_BGR, n.split(".")[-1]) for c, s, R, n in gt]
                if specs:
                    samples = np.concatenate([edge_sample_points(c) for c, _, _ in specs])
                    projection = project_lidar_to_image(
                        samples, ego_pose_lidar, ego_pose_camera, extrinsic, intrinsic,
                        (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX),
                    )
                    per_box = len(_EDGES) * EDGE_SAMPLES
                    uv = np.full((len(samples), 2), np.nan)
                    ok = np.zeros(len(samples), dtype=bool)
                    uv[projection.source_index] = projection.uv_px
                    # In front of the camera AND on the sensor: an edge running
                    # off the side of the image is clipped the same way it is
                    # clipped by the near plane, by dropping its samples.
                    ok[projection.source_index] = projection.in_image
                    for i, (_, color, label) in enumerate(specs):
                        lo = i * per_box
                        draw_cuboid(image, uv[lo : lo + per_box], ok[lo : lo + per_box], color, label=label)

                legend(image, [
                    (f"{channel}  dt={camera['dt_ns'] / 1e6:+.1f} ms", (255, 255, 255)),
                    (f"OURS  {len(ours)} boxes ({n_no_box} proposals made none)", (255, 255, 255)),
                    (f"HUMAN {len(gt)} annotations", GT_COLOR_BGR),
                    (f"ego motion between lidar and camera: {projection.ego_translation_delta_m * 100:.1f} cm"
                     if specs else "no boxes this frame", (200, 200, 200)),
                ])
                out_path = os.path.join(out_root, scene, token, f"{channel}.png")
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                cv2.imwrite(out_path, image)
                panels[channel] = image
                n_written += 1

            if len(panels) >= 2:
                order = [c for c in ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
                                     "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT") if c in panels]
                scale = 0.5
                tiles = [cv2.resize(panels[c], None, fx=scale, fy=scale) for c in order]
                rows_img = [np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3) if len(tiles[i : i + 3]) == 3]
                if rows_img:
                    cv2.imwrite(os.path.join(out_root, scene, token, "grid.png"), np.vstack(rows_img))
            print(f"  {scene}/{token}  ours={len(ours):>3}  human={len(gt):>3}  cameras={len(panels)}")

    print(f"\nwrote {n_written} camera PNG(s) under {out_root}")
    print(f"boxes from {boxes_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
