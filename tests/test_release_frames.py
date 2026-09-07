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


# --- C1: Stage 1's own ground-filtered clouds are the authoritative source ----
# The chain runs export_release BEFORE export_cvat_3d (the review task has to
# show post-stitch identities), so on a fresh work root <work>/cvat_export_3d
# does not exist yet and every interpolated row used to fall back to the RAW,
# ground-INCLUDED sweep. Stage 1 wrote the ground-filtered cloud the task.zip is
# merely a copy of; CloudSource reads that first.


def _stage1_tree(tmp_path, scene, clouds):
    """<stage1_dir>/scenes/<scene>/keyframes.jsonl + the .pcd.bin files it names."""
    stage1 = tmp_path / "stage1_ingestion"
    scene_dir = stage1 / "scenes" / scene
    scene_dir.mkdir(parents=True)
    cloud_dir = stage1 / "clouds" / scene / "single_sweep"
    cloud_dir.mkdir(parents=True)
    lines = []
    for token, points in clouds.items():
        path = cloud_dir / f"{token}.pcd.bin"
        if points is not None:
            np.asarray(points, np.float32).reshape(-1, 5).tofile(path)
        lines.append(json.dumps({
            "keyframe_token": token, "t_ns": 0,
            "single_sweep_cloud": {"cloud_kind": "single_sweep", "frame": "ego",
                                   "path": str(path), "point_record_bytes": 20},
        }))
    (scene_dir / "keyframes.jsonl").write_text("\n".join(lines) + "\n")
    return str(stage1)


def _fixture_root(tmp_path):
    """A scene with a task.zip cloud for s1, a raw dataroot sweep for s1 and s2."""
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
    for tok, n in (("s1", 11), ("s2", 7)):
        np.zeros((n, 5), np.float32).tofile(dataroot / "samples" / "LIDAR_TOP" / f"{tok}.pcd.bin")

    class Root:
        def lidar_sd(self, tok):
            return {"filename": f"samples/LIDAR_TOP/{tok}.pcd.bin"}

    return fr, str(dataroot), str(tmp_path / "cvat_export_3d"), Root()


def test_cloud_source_prefers_stage1_over_task_zip_and_raw(tmp_path):
    fr, dataroot, cvat_dir, root = _fixture_root(tmp_path)
    stage1 = _stage1_tree(tmp_path, "chunk_test",
                          {"s1": np.zeros((5, 5)), "s2": np.zeros((9, 5))})
    src = CloudSource(dataroot, cvat_dir, fr, root, stage1_dir=stage1)
    # s1 has all three sources; Stage 1's 5-point cloud wins over the zip's 3.
    pts, basis = src.points("s1")
    assert pts.shape == (5, 3) and basis == "single_sweep_ground_filtered_pre_inflation"
    # s2 has no zip entry at all, and used to degrade to the raw sweep.
    pts, basis = src.points("s2")
    assert pts.shape == (9, 3) and basis == "single_sweep_ground_filtered_pre_inflation"
    src.close()


def test_cloud_source_falls_back_when_the_stage1_cloud_was_pruned(tmp_path):
    # The chain prunes <work>/stage1_ingestion/clouds AFTER the export; a tree
    # pruned by an earlier run still has keyframes.jsonl naming absent files.
    fr, dataroot, cvat_dir, root = _fixture_root(tmp_path)
    stage1 = _stage1_tree(tmp_path, "chunk_test", {"s1": None, "s2": None})
    src = CloudSource(dataroot, cvat_dir, fr, root, stage1_dir=stage1)
    pts, basis = src.points("s1")     # task.zip is still ground-filtered
    assert pts.shape == (3, 3) and basis == "single_sweep_ground_filtered_pre_inflation"
    pts, basis = src.points("s2")     # nothing but the raw sweep left
    assert pts.shape == (7, 3) and basis == "single_sweep_raw"
    src.close()


def test_cloud_source_reports_unavailable_when_nothing_resolves(tmp_path):
    fr = SceneFrames(scene_token="sc", scene_name="chunk_test", tokens=["s1"],
                     timestamps_ns=[0], poses={}, index={"s1": 0})

    class Root:
        def lidar_sd(self, tok):
            raise KeyError(tok)

    src = CloudSource(str(tmp_path / "nope"), None, fr, Root(), stage1_dir=str(tmp_path / "gone"))
    assert src.points("s1") == (None, "unavailable")
    src.close()
