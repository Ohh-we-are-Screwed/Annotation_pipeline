#!/usr/bin/env python3
"""Package Stage 1 clouds as a CVAT 3D task, with OUR cuboids and the human ones.

Per scene, under <work_root>/cvat_export_3d/<scene>/:

    task.zip                  pointcloud/000001.pcd + related_images/000001_pcd/CAM_*.jpg
    annotations_ours.json     Datumaro 3D 1.0 — Stage 8 cuboids
    annotations_gt.json       Datumaro 3D 1.0 — the nuScenes annotations

The zip is CVAT's 3D upload layout; the two JSONs are the 3D twins of the 2D
`instances.json` pair, imported as "Datumaro 3D 1.0". Frame N of a 3D task is
the same keyframe as frames 6N..6N+5 of the 2D task for that scene.

Clouds are the EGO-FRAME ground-filtered single sweeps Stage 1 wrote (§1.4) —
the same points Stage 5 painted and Stage 6 clustered — converted from the
20-byte .pcd.bin layout to binary PCD v0.7 (x y z intensity). Because the cloud
is in the ego frame and our boxes are too, a cuboid needs no transform to reach
CVAT: what the reviewer sees is the Stage 6/8 box against the points it was fit
to. The human boxes DO get transformed (global -> ego at the LiDAR anchor,
through the same ego_pose the pipeline used), so a pose error shows up as the
answer key floating off its own objects.

THE TWO CONVENTIONS THIS FILE DEPENDS ON, both established by measurement
rather than by reading a format description:

  1. **Datumaro `scale` is (x, y, z) extent in the cuboid's own frame.**
     CVAT stores a 3D cuboid as points[0:3]=position, [3:6]=rotation,
     [6:9]=scale (`dataset_manager/bindings.py`), and the 3D canvas builds a
     unit `BoxGeometry(1,1,1)` and calls `scale.set(points[6], points[7],
     points[8])` — so slot 6 is the local-x extent, i.e. the LENGTH along
     heading. Datumaro's own KITTI-raw exporter labels those same slots
     (w, h, l), which disagrees; the canvas is what draws, so the canvas wins.
     Get this wrong and every cuboid is drawn with its length and width
     swapped: still a box, still on the object, wrong shape for every vehicle.
  2. **Our `size_wlh_m` really is (width, length, height).** Checked against
     the points each box was fitted to: reading it as (l, w, h) encloses 28,345
     cloud points over 400 boxes, reading it as (w, l, h) encloses 13,623.

Yaw only. Stage 6 fits yaw-axis boxes (`yaw_axis_only: true`) and the human
boxes are re-expressed the same way, so both sets are comparable and neither
depends on a Euler-order convention.

    python -m scripts.export_cvat_3d [--scenes scene-0061]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import zipfile

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.conventions import (  # noqa: E402
    EGO,
    NUSCENES_GLOBAL,
    Transform,
    apply_transform,
    quaternion_to_rotation_matrix,
)
from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.stage0_data_probe.probe import Substrate  # noqa: E402
from pipeline.stage1_ingestion.ingest import read_pcd_bin  # noqa: E402

TAXONOMY = "configs/taxonomy_pilot_nuscenes.yaml"


def write_pcd(path: str, points: np.ndarray) -> None:
    """Binary PCD v0.7, fields x y z intensity (float32)."""
    n = points.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(points[:, :4].astype("<f4").tobytes())


def cuboid(index: int, label_id: int, center_xyz, yaw_rad: float, extent_lwh) -> dict:
    """One Datumaro 3D cuboid. `extent_lwh` is (along heading, across, up)."""
    return {
        "id": index,
        "type": "cuboid_3d",
        "attributes": {"occluded": False},
        "group": 0,
        "label_id": label_id,
        "position": [round(float(v), 4) for v in center_xyz],
        # Yaw about ego +z. Roll and pitch are zero by construction on both
        # sides, which is why no Euler-order question arises here.
        "rotation": [0.0, 0.0, round(float(yaw_rad), 6)],
        # (x, y, z) extent in the cuboid's own frame — see the module docstring.
        "scale": [round(float(v), 4) for v in extent_lwh],
    }


def datumaro_document(labels: list[str], items: list[dict]) -> dict:
    return {
        "info": {},
        "categories": {"label": {"labels": [{"name": n, "parent": "", "attributes": []} for n in labels]}},
        "items": items,
    }


def item_skeleton(index: int, channels: list[str]) -> dict:
    name = f"{index + 1:06d}"
    return {
        "id": name,
        "annotations": [],
        "attr": {"frame": index},
        "point_cloud": {"path": f"pointcloud/{name}.pcd"},
        "related_images": [{"path": f"related_images/{name}_pcd/{c}.jpg"} for c in sorted(channels)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--boxes-dir", default=None, help="default <work_root>/stage8_inflate")
    parser.add_argument("--taxonomy", default=TAXONOMY)
    parser.add_argument("--skip-archive", action="store_true",
                        help="rebuild the annotation JSONs only; the task.zip on disk is kept")
    args = parser.parse_args(argv)

    paths = load_paths(args.paths)
    stage1 = os.path.join(paths.work_root, "stage1_ingestion")
    boxes_dir = args.boxes_dir or os.path.join(paths.work_root, "stage8_inflate")
    out_root = os.path.join(paths.work_root, "cvat_export_3d")

    with open(args.taxonomy) as fh:
        phrase_of = yaml.safe_load(fh)["prompt_phrase"]
    # The label set is the taxonomy's phrases, identical to the 2D exports', so
    # a class means the same thing in the 2D task, the 3D task and the metrics.
    labels = sorted(set(phrase_of.values()))
    label_id = {name: i for i, name in enumerate(labels)}

    substrate = Substrate.load(paths)
    ego_poses = substrate.by_token("ego_pose.json")
    instances = substrate.by_token("instance.json")
    categories = substrate.by_token("category.json")
    annotations: dict[str, list[dict]] = {}
    for ann in substrate.tables["sample_annotation.json"]:
        annotations.setdefault(ann["sample_token"], []).append(ann)

    scene_root = os.path.join(stage1, "scenes")
    names = sorted(n for n in os.listdir(scene_root) if os.path.isdir(os.path.join(scene_root, n)))
    if args.scenes:
        names = [n for n in names if n in args.scenes]

    for scene in names:
        keyframes = [json.loads(l) for l in open(os.path.join(scene_root, scene, "keyframes.jsonl"))]
        boxes_path = os.path.join(boxes_dir, "scenes", scene, "inflated.jsonl")
        if not os.path.isfile(boxes_path):
            boxes_path = os.path.join(boxes_dir, "scenes", scene, "boxes.jsonl")
        by_keyframe: dict[str, list[dict]] = {}
        for line in open(boxes_path):
            row = json.loads(line)
            if row.get("box"):
                by_keyframe.setdefault(row["keyframe_token"], []).append(row)

        scene_dir = os.path.join(out_root, scene)
        staging = os.path.join(scene_dir, "_staging")
        if not args.skip_archive:
            shutil.rmtree(staging, ignore_errors=True)
            os.makedirs(os.path.join(staging, "pointcloud"))
        os.makedirs(scene_dir, exist_ok=True)

        items_ours, items_gt = [], []
        n_ours = n_gt = n_gt_unmapped = 0

        for index, keyframe in enumerate(keyframes):
            name = f"{index + 1:06d}"
            channels = sorted(keyframe["cameras"])
            if not args.skip_archive:
                write_pcd(
                    os.path.join(staging, "pointcloud", f"{name}.pcd"),
                    read_pcd_bin(keyframe["single_sweep_cloud"]["path"]),
                )
                image_dir = os.path.join(staging, "related_images", f"{name}_pcd")
                os.makedirs(image_dir)
                for channel in channels:
                    shutil.copy(
                        os.path.join(paths.dataroot, keyframe["cameras"][channel]["path"]),
                        os.path.join(image_dir, f"{channel}.jpg"),
                    )

            # --- ours: already in the ego frame, no transform -----------------
            item = item_skeleton(index, channels)
            for row in by_keyframe.get(keyframe["keyframe_token"], []):
                box = row["box"]
                width, length, height = (float(v) for v in box["size_wlh_m"])
                n_ours += 1
                item["annotations"].append(
                    cuboid(n_ours, label_id[row["class_name"]], box["translation_m"],
                           box["yaw_rad"], (length, width, height))
                )
            items_ours.append(item)

            # --- the answer key: global -> ego at the LiDAR anchor -------------
            pose = Transform.from_nuscenes(
                ego_poses[keyframe["lidar_ego_pose_token"]], source_frame=EGO, parent_frame=NUSCENES_GLOBAL
            )
            r_ego_global = quaternion_to_rotation_matrix(pose.rotation_wxyz).T
            item = item_skeleton(index, channels)
            for ann in annotations.get(keyframe["keyframe_token"], []):
                category = categories[instances[ann["instance_token"]]["category_token"]]["name"]
                phrase = phrase_of.get(category)
                if phrase is None:
                    # Out of the class space by decision (C21), not an error —
                    # the same rule export_gt_coco applies, counted so the two
                    # exports can be reconciled.
                    n_gt_unmapped += 1
                    continue
                center = apply_transform(
                    pose.inverse_matrix(), np.asarray(ann["translation"], np.float64)[None, :]
                )[0]
                r = r_ego_global @ quaternion_to_rotation_matrix(ann["rotation"])
                width, length, height = (float(v) for v in ann["size"])
                n_gt += 1
                item["annotations"].append(
                    cuboid(n_gt, label_id[phrase], center, math.atan2(r[1, 0], r[0, 0]),
                           (length, width, height))
                )
            items_gt.append(item)

        for suffix, items in (("ours", items_ours), ("gt", items_gt)):
            with open(os.path.join(scene_dir, f"annotations_{suffix}.json"), "w") as fh:
                json.dump(datumaro_document(labels, items), fh)

        zip_path = os.path.join(scene_dir, "task.zip")
        if not args.skip_archive:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
                for root, _, files in os.walk(staging):
                    for f in sorted(files):
                        full = os.path.join(root, f)
                        zf.write(full, os.path.relpath(full, staging))
            shutil.rmtree(staging)
        size_mb = os.path.getsize(zip_path) / 1e6 if os.path.isfile(zip_path) else 0.0
        print(f"  {scene}: {len(keyframes)} frames, ours {n_ours:>5}, human {n_gt:>5} "
              f"({n_gt_unmapped} out of class space) -> {scene_dir} ({size_mb:.0f} MB)")

    print(f"\nwrote {len(names)} scene(s) under {out_root}")
    print("publish with: python -m scripts.cvat_setup_3d")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
