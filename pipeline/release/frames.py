"""Per-scene keyframe order, poses, and the cloud each keyframe's boxes were fit to."""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass, field

import numpy as np

from pipeline.common.conventions import Transform
from pipeline.stage1_ingestion.ingest import read_pcd_bin

BASIS_GROUND_FILTERED = "single_sweep_ground_filtered_pre_inflation"
BASIS_RAW = "single_sweep_raw"
BASIS_UNAVAILABLE = "unavailable"
US_TO_NS = 1_000


@dataclass
class SceneFrames:
    scene_token: str
    scene_name: str
    tokens: list
    timestamps_ns: list
    poses: dict
    index: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.index:
            self.index = {t: i for i, t in enumerate(self.tokens)}

    def dt_s(self, i: int, j: int) -> float:
        return (self.timestamps_ns[j] - self.timestamps_ns[i]) / 1e9


def scene_frames_from_root(root, scene_token: str) -> SceneFrames:
    rows = sorted((r for r in root.tables["sample"] if r["scene_token"] == scene_token),
                  key=lambda r: (r["timestamp"], r["token"]))
    tokens = [r["token"] for r in rows]
    return SceneFrames(
        scene_token=scene_token,
        scene_name=root.scene[scene_token]["name"],
        tokens=tokens,
        timestamps_ns=[int(r["timestamp"]) * US_TO_NS for r in rows],
        poses={t: root.lidar_ego_pose(t) for t in tokens},
    )


def read_pcd_v07_binary(data: bytes) -> np.ndarray:
    head_end = data.index(b"DATA binary\n") + len(b"DATA binary\n")
    header = data[:head_end].decode("ascii", "replace")
    fields = next(l for l in header.splitlines() if l.startswith("FIELDS")).split()[1:]
    n = int(next(l for l in header.splitlines() if l.startswith("POINTS")).split()[1])
    return np.frombuffer(data[head_end:], dtype=np.float32, count=n * len(fields)).reshape(n, len(fields))


class CloudSource:
    def __init__(self, dataroot: str, cvat_export_3d_dir: str | None, frames: SceneFrames, root):
        self.dataroot = dataroot
        self.frames = frames
        self.root = root
        self._zip = None
        self._name_of: dict = {}
        if cvat_export_3d_dir:
            scene_dir = os.path.join(cvat_export_3d_dir, frames.scene_name)
            zpath, fpath = os.path.join(scene_dir, "task.zip"), os.path.join(scene_dir, "frames.json")
            if os.path.isfile(zpath) and os.path.isfile(fpath):
                with open(fpath, "r", encoding="utf-8") as fh:
                    self._name_of = {r["sample_token"]: r["name"] for r in json.load(fh)}
                self._zip = zipfile.ZipFile(zpath)

    def points(self, sample_token: str):
        name = self._name_of.get(sample_token)
        if self._zip is not None and name is not None:
            try:
                arr = read_pcd_v07_binary(self._zip.read(f"pointcloud/{name}.pcd"))
                return arr[:, :3].astype(np.float64), BASIS_GROUND_FILTERED
            except KeyError:
                pass
        try:
            sd = self.root.lidar_sd(sample_token)
            path = os.path.join(self.dataroot, sd["filename"])
            if os.path.isfile(path):
                return read_pcd_bin(path)[:, :3].astype(np.float64), BASIS_RAW
        except Exception:  # noqa: BLE001 — a missing sweep is reported as unavailable, not raised
            pass
        return None, BASIS_UNAVAILABLE

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None
