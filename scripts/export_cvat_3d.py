#!/usr/bin/env python3
"""Package Stage 1 single-sweep clouds as a CVAT 3D task archive (one scene).

CVAT's "3D pointcloud" upload format:

    <scene>.zip
      pointcloud/
        000001.pcd            # keyframe clouds, chronological order
      related_images/
        000001_pcd/           # the six ring cameras for that keyframe
          CAM_FRONT.jpg ...

Clouds are the EGO-FRAME ground-filtered single sweeps Stage 1 wrote (§1.4) —
the same points Stage 5 painted — converted from the 20-byte .pcd.bin layout to
binary PCD v0.7 (x y z intensity). Frame N here is the same keyframe as frame
6N..6N+5 in the 2D tasks.

No cuboids are imported: Stage 6 does not exist yet, so this task is a viewer
(and a hand-annotation surface) — the 3D twin of the 2D pre-annotation tasks
comes with Stage 6's boxes.

    python -m scripts.export_cvat_3d --scene scene-0061
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.stage1_ingestion.ingest import read_pcd_bin  # noqa: E402


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--scene", default="scene-0061")
    args = parser.parse_args(argv)

    paths = load_paths(args.paths)
    keyframes_path = os.path.join(
        paths.work_root, "stage1_ingestion", "scenes", args.scene, "keyframes.jsonl"
    )
    if not os.path.isfile(keyframes_path):
        print(f"{keyframes_path} not found — run Stage 1 first", file=sys.stderr)
        return 2
    with open(keyframes_path) as fh:
        keyframes = [json.loads(line) for line in fh if line.strip()]

    staging = os.path.join(paths.work_root, "cvat_export_3d", args.scene)
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(os.path.join(staging, "pointcloud"))
    os.makedirs(os.path.join(staging, "related_images"))

    for index, keyframe in enumerate(keyframes, start=1):
        name = f"{index:06d}"
        cloud = read_pcd_bin(keyframe["single_sweep_cloud"]["path"])  # (N, 5) x y z i ring
        write_pcd(os.path.join(staging, "pointcloud", f"{name}.pcd"), cloud)
        image_dir = os.path.join(staging, "related_images", f"{name}_pcd")
        os.makedirs(image_dir)
        for channel, cam in sorted(keyframe["cameras"].items()):
            shutil.copy(
                os.path.join(paths.dataroot, cam["path"]),
                os.path.join(image_dir, f"{channel}.jpg"),
            )

    zip_path = os.path.join(paths.work_root, "cvat_export_3d", f"{args.scene}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(staging):
            for f in sorted(files):
                full = os.path.join(root, f)
                zf.write(full, os.path.relpath(full, staging))
    shutil.rmtree(staging)
    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"{args.scene}: {len(keyframes)} clouds + related images -> {zip_path} ({size_mb:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
