"""Stage 5's exact-zero ego-delta cause: DEGRADED, not REFUSED (R19).

`lift.py`'s scene loop used to raise `LiftContractError` when
`|ego(t_cam) - ego(t_lidar)|` was exactly 0.0 for every camera of every
keyframe in a scene. On the current export this is a verified PROPERTY OF THE
DATA, not a broken join: every camera `sample_data` row carries its own
`ego_pose` token and its own capture timestamp, but the exporter copied the
LiDAR keyframe's translation/rotation into every camera's `ego_pose` row
instead of interpolating ego motion to each camera's own capture time. The two
middle hops of the projection chain (§1.3) legitimately cancel, and refusing
made the whole pipeline unusable on this export. This module pins the
replacement behaviour: the scene lifts, the scene summary is DEGRADED with
cause `ego_motion_between_capture_times_absent`, and a scene with genuinely
distinct per-camera ego poses is unaffected.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pipeline.common.conventions import EGO  # noqa: E402
from pipeline.common.manifest import Marker, write_jsonl_atomic, write_marker  # noqa: E402
from pipeline.common.paths import METADATA_TABLES, Paths  # noqa: E402
from pipeline.common.schemas import (  # noqa: E402
    IMAGE_HEIGHT_PX,
    IMAGE_WIDTH_PX,
    CameraObservation,
    CloudArtifact,
    KeyframeRecord,
    write_records,
)
from pipeline.stage1_ingestion.ingest import write_pcd_bin  # noqa: E402
from pipeline.stage5_lift.lift import EXIT_DEGRADED, EXIT_OK, LiftConfig, run  # noqa: E402

CAMERA_CHANNEL = "CAM_FRONT"  # in RING_CAMERAS under every SUBSTRATE_PROFILES entry
SCENE_NAME = "sc1"


def _write_metadata(meta_root: str, version: str, ego_poses: list, calibrated_sensors: list) -> None:
    version_dir = os.path.join(meta_root, version)
    os.makedirs(version_dir, exist_ok=True)
    tables = {name: [] for name in METADATA_TABLES}
    tables["ego_pose.json"] = ego_poses
    tables["calibrated_sensor.json"] = calibrated_sensors
    for name, rows in tables.items():
        with open(os.path.join(version_dir, name), "w", encoding="utf-8") as fh:
            json.dump(rows, fh)


def _run_one_keyframe_one_camera(tmp_path, *, camera_ego_pose_token: str, cam_dt_ns: int):
    """One scene, one keyframe, one camera, zero mask instances.

    Zero instances keeps `n_instances == 0`, so the pre-existing "0 points
    painted" degraded condition never fires here — the only thing this
    fixture can trip is the ego-motion-absent cause. The LiDAR anchor's
    ego_pose is always token "ep_lidar" (translation [1, 2, 3]);
    `camera_ego_pose_token` selects which row the one camera is handed: the
    SAME token reproduces the exporter's bug (exact-zero delta), a different
    token with a different translation reproduces a normal, moving ego.
    """
    meta_root = str(tmp_path / "meta")
    version = "v1.0-test"
    ego_poses = [
        {"token": "ep_lidar", "timestamp": 1_000_000, "translation": [1.0, 2.0, 3.0],
         "rotation": [1.0, 0.0, 0.0, 0.0]},
        {"token": "ep_cam", "timestamp": 1_000_045_000, "translation": [1.4, 2.0, 3.0],
         "rotation": [1.0, 0.0, 0.0, 0.0]},
    ]
    calibrated_sensors = [
        {"token": "cs_lidar", "sensor_token": "sen_lidar", "translation": [0.0, 0.0, 0.0],
         "rotation": [1.0, 0.0, 0.0, 0.0], "camera_intrinsic": []},
        {"token": "cs_cam", "sensor_token": "sen_cam", "translation": [0.0, 0.0, 0.0],
         "rotation": [1.0, 0.0, 0.0, 0.0],
         "camera_intrinsic": [[1000.0, 0.0, IMAGE_WIDTH_PX / 2.0],
                               [0.0, 1000.0, IMAGE_HEIGHT_PX / 2.0],
                               [0.0, 0.0, 1.0]]},
    ]
    _write_metadata(meta_root, version, ego_poses, calibrated_sensors)

    paths = Paths(
        dataroot=str(tmp_path / "dataroot"),
        meta_root=meta_root,
        version=version,
        work_root=str(tmp_path / "work"),
        out_root=str(tmp_path / "out"),
        probe_out_root=str(tmp_path / "probe"),
    )
    stage1_dir = str(tmp_path / "stage1")
    stage4_dir = str(tmp_path / "stage4")
    out_dir = str(tmp_path / "stage5")

    cloud_path = str(tmp_path / "clouds" / "kf1.pcd.bin")
    write_pcd_bin(cloud_path, np.array([[5.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))

    keyframe = KeyframeRecord(
        keyframe_token="kf1",
        scene_token=SCENE_NAME,
        t_ns=1_700_000_000_000_000_000,
        time_base="unix_ns",
        lidar_sample_data_token="sd_lidar",
        lidar_path="samples/LIDAR_TOP/kf1.pcd.bin",
        lidar_ego_pose_token="ep_lidar",
        lidar_calibrated_sensor_token="cs_lidar",
        cameras={
            CAMERA_CHANNEL: CameraObservation(
                channel=CAMERA_CHANNEL,
                sample_data_token="sd_cam",
                path=f"samples/{CAMERA_CHANNEL}/kf1.jpg",
                dt_ns=cam_dt_ns,
                ego_pose_token=camera_ego_pose_token,
                calibrated_sensor_token="cs_cam",
            ),
        },
        single_sweep_cloud=CloudArtifact(
            path=cloud_path, cloud_kind="single_sweep", frame=EGO,
            n_points=1, n_sweeps_actual=1, window_ns=0,
        ),
        accumulated_cloud=CloudArtifact(
            path=cloud_path, cloud_kind="accumulated", frame=EGO,
            n_points=1, n_sweeps_actual=1, window_ns=0,
        ),
        coverage_config="R1",
        is_first_in_scene=True,
    )
    write_records(
        os.path.join(stage1_dir, "scenes", SCENE_NAME, "keyframes.jsonl"),
        [keyframe],
        expect_type=KeyframeRecord,
    )

    scene_stage4_dir = os.path.join(stage4_dir, "scenes", SCENE_NAME)
    os.makedirs(scene_stage4_dir, exist_ok=True)
    mask_path = f"scenes/{SCENE_NAME}/mask_kf1.npz"
    np.savez(
        os.path.join(stage4_dir, mask_path),
        __width_px__=np.array([IMAGE_WIDTH_PX]),
        __height_px__=np.array([IMAGE_HEIGHT_PX]),
        __bit_packed__=np.array([0]),
    )
    write_jsonl_atomic(
        os.path.join(scene_stage4_dir, "masks.jsonl"),
        [{"keyframe_token": "kf1", "mask_path": mask_path, "candidates": []}],
    )

    stage1_manifest = {
        "spec": "test/stage1/v1",
        "upstream": {"metadata_fingerprint": "fp0", "fingerprint_spec": "test-spec/v1"},
    }
    stage4_manifest = {"spec": "test/stage4/v1", "upstream": {"prompt_caption_sha256": "deadbeef"}}
    stage1_marker = Marker(state="clean", fingerprint="fp0")
    stage4_marker = Marker(state="clean", fingerprint="fp0")

    manifest, code = run(
        paths, stage1_manifest, stage1_marker, stage4_manifest, stage4_marker,
        LiftConfig(), stage1_dir, stage4_dir, out_dir, None,
    )
    return manifest, code, out_dir


class TestEgoMotionBetweenCaptureTimesAbsent:
    def test_identical_camera_ego_pose_degrades_the_scene_instead_of_refusing(self, tmp_path):
        """The exporter's bug, reproduced: the scene LIFTS (no LiftContractError)."""
        manifest, code, out_dir = _run_one_keyframe_one_camera(
            tmp_path, camera_ego_pose_token="ep_lidar", cam_dt_ns=-20_000_000
        )
        assert code == EXIT_DEGRADED

        scene = manifest["scenes"][0]
        assert scene["degraded"] is True
        assert scene["causes"] == ["ego_motion_between_capture_times_absent"]
        assert scene["max_ego_translation_delta_m"] == 0.0
        assert scene["max_abs_camera_dt_ns"] == 20_000_000

        assert any(
            "ego_motion_between_capture_times_absent" in gap for gap in manifest["known_gaps"]
        )

        # Mirrors main()'s marker-writing call exactly (manifest["scenes"] ->
        # flattened per-scene causes) so the on-disk artifact is checked too.
        marker_causes = [cause for s in manifest["scenes"] for cause in s["causes"]]
        marker_path = write_marker(
            out_dir, manifest["upstream"]["metadata_fingerprint"],
            degraded=(code == EXIT_DEGRADED), causes=marker_causes,
        )
        assert os.path.basename(marker_path) == "_SUCCESS.degraded"
        with open(marker_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        assert "ego_motion_between_capture_times_absent" in payload["causes"]

    def test_distinct_camera_ego_pose_is_not_degraded(self, tmp_path):
        """Control: a camera with its OWN, different ego_pose lifts clean."""
        manifest, code, _ = _run_one_keyframe_one_camera(
            tmp_path, camera_ego_pose_token="ep_cam", cam_dt_ns=-20_000_000
        )
        assert code == EXIT_OK

        scene = manifest["scenes"][0]
        assert scene["degraded"] is False
        assert scene["causes"] == []
        assert scene["max_ego_translation_delta_m"] > 0.0
        assert not any(
            "ego_motion_between_capture_times_absent" in gap for gap in manifest["known_gaps"]
        )
