from __future__ import annotations

import io
import json
import os
import sys
import zipfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.conventions import EGO, NUSCENES_GLOBAL, Transform, quaternion_from_yaw_rad  # noqa: E402
from pipeline.release.frames import CloudSource, SceneFrames, read_pcd_v07_binary, scene_frames_from_root  # noqa: E402


class FakeRoot:
    """The slice of export_release.SourceRoot the module reads."""

    def __init__(self):
        self.scene = {"sc": {"token": "sc", "name": "chunk_test"}}
        self.tables = {"sample": [
            {"token": "s2", "scene_token": "sc", "timestamp": 2_400_000},
            {"token": "s1", "scene_token": "sc", "timestamp": 2_000_000},
            {"token": "s3", "scene_token": "sc", "timestamp": 2_800_000},
            {"token": "other", "scene_token": "sc2", "timestamp": 1},
        ]}
        self.sample = {r["token"]: r for r in self.tables["sample"]}

    def lidar_ego_pose(self, tok):
        x = {"s1": 0.0, "s2": 4.0, "s3": 8.0}[tok]
        return Transform.from_nuscenes({"translation": [x, 0, 0], "rotation": list(quaternion_from_yaw_rad(0))},
                                       source_frame=EGO, parent_frame=NUSCENES_GLOBAL)


def test_scene_frames_are_time_ordered_and_scene_scoped():
    fr = scene_frames_from_root(FakeRoot(), "sc")
    assert fr.tokens == ["s1", "s2", "s3"]
    assert fr.timestamps_ns == [2_000_000_000, 2_400_000_000, 2_800_000_000]
    assert fr.index["s3"] == 2
    assert fr.dt_s(0, 2) == pytest.approx(0.8)
    assert fr.poses["s2"].translation_m[0] == 4.0


def _pcd_bytes(points):
    n = len(points)
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\n"
              "SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
              f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    return header.encode() + np.asarray(points, np.float32).tobytes()


def test_read_pcd_v07_binary():
    arr = read_pcd_v07_binary(_pcd_bytes([[1, 2, 3, 0.5], [4, 5, 6, 0.1]]))
    assert arr.shape == (2, 4) and arr[1, 2] == 6.0


def test_cloud_source_prefers_task_zip_then_raw(tmp_path):
    fr = SceneFrames(scene_token="sc", scene_name="chunk_test", tokens=["s1", "s2"],
                     timestamps_ns=[0, 400_000_000], poses={}, index={"s1": 0, "s2": 1})
    export_dir = tmp_path / "cvat_export_3d" / "chunk_test"
    export_dir.mkdir(parents=True)
    with zipfile.ZipFile(export_dir / "task.zip", "w") as zf:
        zf.writestr("pointcloud/000001.pcd", _pcd_bytes([[1, 1, 1, 0], [2, 2, 2, 0], [3, 3, 3, 0]]))
    (export_dir / "frames.json").write_text(json.dumps(
        [{"frame": 0, "name": "000001", "sample_token": "s1", "channels": []}]))
    dataroot = tmp_path / "root"
    (dataroot / "samples" / "LIDAR_TOP").mkdir(parents=True)
    np.zeros((7, 5), np.float32).tofile(dataroot / "samples" / "LIDAR_TOP" / "s2.pcd.bin")

    class Root:
        def lidar_sd(self, tok):
            return {"filename": f"samples/LIDAR_TOP/{tok}.pcd.bin"}

    src = CloudSource(str(dataroot), str(tmp_path / "cvat_export_3d"), fr, Root())
    pts, basis = src.points("s1")
    assert pts.shape == (3, 3) and basis == "single_sweep_ground_filtered_pre_inflation"
    pts, basis = src.points("s2")
    assert pts.shape == (7, 3) and basis == "single_sweep_raw"
    src.close()
