#!/usr/bin/env python3
"""Project nuScenes GT 3D boxes into the six cameras as COCO for CVAT GT twins.

These annotations did NOT pass through the pipeline: they are the dataset's own
human labels (sample_annotation.json), projected 3D->2D per camera so each
scene gets a "GT twin" CVAT task over the SAME images in the SAME frame order
as the pre-annotated task. Frame N in both tasks is the same picture.

Projection per camera uses the camera's OWN ego pose (§1.2/§1.3): box corners
global -> ego(t_cam) -> camera -> pixel, corners behind the camera culled, bbox
clipped to 1600x900. A box survives if >= 2 corners project in front and the
clipped bbox is non-degenerate. Categories map to the taxonomy phrases the
pipeline labels with, so the GT twin lives in the same CVAT project.

Two things about this answer key that its numbers depend on:

  * The projected box is **amodal** — the AABB of the whole cuboid, including
    the occluded part — while a 2D detector emits a **modal** box around visible
    pixels. IoU between the two is depressed systematically, not randomly, which
    is why scripts/eval_2d.py reports an IoU sweep rather than one 0.5 gate.
  * By default NOTHING is filtered: an object 90% behind a bus, or with zero
    lidar returns, is still in the denominator. That is a deliberate, harsh
    baseline. `--min-visibility` and `--min-lidar-pts` narrow it to what a camera
    could plausibly see, mirroring eval_3d.py's eligibility rule; both default to
    off so previously recorded numbers stay reproducible, and whatever was used
    is written into each `instances.json` under `info`.

Output: <work_root>/cvat_export_gt/<scene>/instances.json

    python -m scripts.export_gt_coco [--scenes scene-0061]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.conventions import (  # noqa: E402
    CAMERA,
    EGO,
    NUSCENES_GLOBAL,
    Transform,
    quaternion_to_rotation_matrix,
)
from pipeline.common.manifest import write_json_atomic  # noqa: E402
from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.common.schemas import IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX  # noqa: E402

MIN_DEPTH_M = 0.1
MIN_SIDE_PX = 4.0
MIN_CORNERS_IN_FRONT = 2

# nuScenes visibility tiers, worst to best (visibility.json `level`). Ordered so
# "at least this visible" is an index comparison rather than a string one.
VISIBILITY_ORDER = ("v0-40", "v40-60", "v60-80", "v80-100")


def box_corners_global(center, wlh, rotation_wxyz) -> np.ndarray:
    """(8, 3) corners; nuScenes size [w,l,h], box frame x = length."""
    w, l, h = wlh
    signs = np.array(
        [[sx, sy, sz] for sx in (1, -1) for sy in (1, -1) for sz in (1, -1)], dtype=np.float64
    )
    local = signs * np.array([l / 2.0, w / 2.0, h / 2.0])
    R = quaternion_to_rotation_matrix(rotation_wxyz)
    return local @ R.T + np.asarray(center, dtype=np.float64)


def project_bbox(corners_global, ego_pose, extrinsic, K):
    """Clipped 2D bbox of the corners in one camera, or None."""
    p_ego = corners_global @ ego_pose.inverse_matrix()[:3, :3].T + ego_pose.inverse_matrix()[:3, 3]
    p_cam = p_ego @ extrinsic.inverse_matrix()[:3, :3].T + extrinsic.inverse_matrix()[:3, 3]
    in_front = p_cam[:, 2] > MIN_DEPTH_M
    if int(in_front.sum()) < MIN_CORNERS_IN_FRONT:
        return None
    uv_h = p_cam[in_front] @ K.T
    uv = uv_h[:, :2] / uv_h[:, 2:3]
    x1, y1 = np.clip(uv.min(axis=0), [0, 0], [IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX])
    x2, y2 = np.clip(uv.max(axis=0), [0, 0], [IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX])
    if x2 - x1 < MIN_SIDE_PX or y2 - y1 < MIN_SIDE_PX:
        return None
    return float(x1), float(y1), float(x2), float(y2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--scenes", nargs="*", default=None)
    # Both gates default to OFF, i.e. to the historical behaviour: turning them
    # on changes the recall denominator, and a metric that silently moves is
    # worse than a harsh one. eval_3d.py's own eligibility rule is
    # `num_lidar_pts >= 5` within 40 m, so these mirror that convention rather
    # than inventing a second one.
    parser.add_argument(
        "--min-visibility",
        default="none",
        choices=("none", *VISIBILITY_ORDER),
        help="keep only GT at or above this nuScenes visibility tier (default none: keep all, "
        "including objects that are almost entirely occluded)",
    )
    parser.add_argument(
        "--min-lidar-pts",
        type=int,
        default=0,
        help="keep only GT with at least this many lidar returns (default 0: keep all, including "
        "boxes with no returns at all)",
    )
    args = parser.parse_args(argv)

    paths = load_paths(args.paths)
    stage1 = os.path.join(paths.work_root, "stage1_ingestion")
    out_root = os.path.join(paths.work_root, "cvat_export_gt")

    def table(name):
        with open(paths.table(name), "r", encoding="utf-8") as fh:
            return json.load(fh)

    categories = {c["token"]: c["name"] for c in table("category.json")}
    instance_category = {i["token"]: categories[i["category_token"]] for i in table("instance.json")}
    visibility_level = {v["token"]: v["level"] for v in table("visibility.json")}

    min_visibility_rank = (
        -1 if args.min_visibility == "none" else VISIBILITY_ORDER.index(args.min_visibility)
    )

    def eligible(ann: dict) -> bool:
        """The GT filters, applied ONCE so every camera sees the same answer key."""
        if int(ann.get("num_lidar_pts", 0)) < args.min_lidar_pts:
            return False
        if min_visibility_rank >= 0:
            level = visibility_level.get(ann.get("visibility_token"))
            if level is None or VISIBILITY_ORDER.index(level) < min_visibility_rank:
                return False
        return True

    annotations_by_sample = defaultdict(list)
    n_gt_total = n_gt_filtered = 0
    for ann in table("sample_annotation.json"):
        n_gt_total += 1
        if not eligible(ann):
            n_gt_filtered += 1
            continue
        annotations_by_sample[ann["sample_token"]].append(ann)
    ego_poses = {e["token"]: e for e in table("ego_pose.json")}
    calibrated = {c["token"]: c for c in table("calibrated_sensor.json")}

    import yaml
    with open("configs/taxonomy_pilot_nuscenes.yaml") as fh:
        phrase_of = yaml.safe_load(fh)["prompt_phrase"]
    phrases = sorted(set(phrase_of.values()))
    category_id = {p: i + 1 for i, p in enumerate(phrases)}
    coco_categories = [{"id": i, "name": p, "supercategory": ""} for p, i in category_id.items()]

    scene_root = os.path.join(stage1, "scenes")
    names = sorted(n for n in os.listdir(scene_root) if os.path.isdir(os.path.join(scene_root, n)))
    if args.scenes:
        names = [n for n in names if n in args.scenes]

    for scene in names:
        with open(os.path.join(scene_root, scene, "keyframes.jsonl")) as fh:
            keyframes = [json.loads(line) for line in fh if line.strip()]

        images, coco_annotations = [], []
        image_id, ann_id, n_unmapped = 0, 0, 0
        for keyframe in keyframes:
            gt_rows = annotations_by_sample.get(keyframe["keyframe_token"], [])
            # A category with no prompt phrase is OUT OF THE CLASS SPACE, not an
            # error: a collapsed taxonomy deliberately drops categories the
            # detector is never asked to find (nuScenes' own benchmark ignores
            # debris, pushable_pullable and bicycle_rack the same way). Scoring
            # the pipeline against objects it was never prompted for would make
            # recall a measure of the taxonomy, not of the detector. Indexing
            # `phrase_of` directly would raise KeyError the moment that happens.
            corners = [
                (box_corners_global(a["translation"], a["size"], a["rotation"]), phrase)
                for a in gt_rows
                if (phrase := phrase_of.get(instance_category[a["instance_token"]])) is not None
            ]
            n_unmapped += sum(
                1 for a in gt_rows
                if instance_category[a["instance_token"]] not in phrase_of
            )
            # Same image order as export_cvat_coco.py: sorted by channel.
            for channel, cam in sorted(keyframe["cameras"].items()):
                image_id += 1
                images.append({
                    "id": image_id,
                    "file_name": cam["path"],
                    "width": IMAGE_WIDTH_PX,
                    "height": IMAGE_HEIGHT_PX,
                })
                ego_pose = Transform.from_nuscenes(
                    ego_poses[cam["ego_pose_token"]], source_frame=EGO, parent_frame=NUSCENES_GLOBAL
                )
                cs = calibrated[cam["calibrated_sensor_token"]]
                extrinsic = Transform.from_nuscenes(cs, source_frame=CAMERA, parent_frame=EGO)
                K = np.asarray(cs["camera_intrinsic"], dtype=np.float64)
                for corner_set, phrase in corners:
                    bbox = project_bbox(corner_set, ego_pose, extrinsic, K)
                    if bbox is None:
                        continue
                    x1, y1, x2, y2 = bbox
                    ann_id += 1
                    coco_annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id[phrase],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": float((x2 - x1) * (y2 - y1)),
                        "iscrowd": 0,
                        "segmentation": [],
                    })

        doc = {
            "info": {
                "description": f"nuScenes GT projected 3D->2D, {scene} (NOT pipeline output)",
                # The gates ride in the artifact: a recall number is meaningless
                # without the denominator it was computed against (§1.9).
                "min_visibility": args.min_visibility,
                "min_lidar_pts": args.min_lidar_pts,
                "n_3d_annotations_out_of_class_space": n_unmapped,
            },
            "licenses": [],
            "categories": coco_categories,
            "images": images,
            "annotations": coco_annotations,
        }
        out_path = os.path.join(out_root, scene, "instances.json")
        write_json_atomic(out_path, doc)
        print(f"  {scene}: {len(images)} images, {len(coco_annotations)} GT boxes"
              + (f", {n_unmapped} 3D annotations outside the class space" if n_unmapped else "")
              + f" -> {out_path}")

    if n_gt_filtered:
        print(f"GT filter: kept {n_gt_total - n_gt_filtered}/{n_gt_total} 3D annotations "
              f"(min_visibility={args.min_visibility}, min_lidar_pts={args.min_lidar_pts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
