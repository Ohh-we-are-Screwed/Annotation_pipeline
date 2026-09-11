"""Stage 1 ZED_WORLD ingestion (2026-09-12): a GLOBAL-frame stereo channel with
identity ego_pose/calibrated_sensor is brought into ego with the inverse of the
LiDAR ego_pose of the SAME sample; ring tags 100/101 ride through untouched."""
from __future__ import annotations
import json, math, os, sys
import numpy as np
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.common.conventions import EGO, NUSCENES_GLOBAL, Transform  # noqa: E402
from pipeline.stage1_ingestion import ingest  # noqa: E402


def _ego_pose(yaw_deg, t):
    h = math.radians(yaw_deg) / 2
    return {"translation": list(t), "rotation": [math.cos(h), 0.0, 0.0, math.sin(h)]}


def test_stereo_channels_declare_zed_world_as_global_identity():
    assert ingest.STEREO_CHANNELS["ZED_WORLD"] == {"frame": "global_identity", "ring": None}
    assert ingest.STEREO_CHANNELS["ZED_FRONT"] == {"frame": "sensor", "ring": 10}


def test_global_identity_block_lands_in_ego():
    pose = _ego_pose(90.0, (100.0, 50.0, 2.0))           # ego at (100,50,2), facing +y
    T = Transform.from_nuscenes(pose, source_frame=EGO, parent_frame=NUSCENES_GLOBAL)
    # a point 10 m AHEAD of the ego in ego coords is (10,0,0) -> global (100, 60, 2)
    raw = np.array([[100.0, 60.0, 2.0, 200.0, 101.0]], dtype=np.float32)
    out = ingest.stereo_block_to_ego(raw, frame="global_identity", ring=None, t_sensor_to_ego=None,
                                     t_global_to_ego=T.inverse_matrix(), z_correction_m={})
    assert out.shape == (1, 5)
    assert np.allclose(out[0, :3], [10.0, 0.0, 0.0], atol=1e-6)
    assert out[0, 3] == 200.0 and out[0, 4] == 101.0      # intensity + file ring untouched


def test_sensor_frame_block_uses_fixed_ring_and_sensor_transform():
    Ts = np.eye(4); Ts[0, 3] = 1.0                          # sensor 1 m ahead of ego origin
    raw = np.array([[0.0, 0.0, 0.0, 7.0, 0.0]], dtype=np.float32)
    out = ingest.stereo_block_to_ego(raw, frame="sensor", ring=10, t_sensor_to_ego=Ts,
                                     t_global_to_ego=None, z_correction_m={})
    assert np.allclose(out[0, :3], [1.0, 0.0, 0.0]) and out[0, 4] == 10.0


def test_z_correction_applies_per_ring_only():
    T = np.eye(4)
    raw = np.array([[0, 0, 0, 1, 100.0], [0, 0, 0, 1, 101.0]], dtype=np.float32)
    out = ingest.stereo_block_to_ego(raw, frame="global_identity", ring=None, t_sensor_to_ego=None,
                                     t_global_to_ego=T, z_correction_m={100: 0.69})
    assert abs(out[0, 2] - 0.69) < 1e-9 and out[1, 2] == 0.0


def test_cli_parses_new_knobs():
    p = ingest.build_parser()
    a = p.parse_args(["--stereo-stride", "1", "--coverage-config", "R3",
                      "--stereo-z-correction", "100:0.69", "--stereo-z-correction", "101:-0.1"])
    assert a.stereo_stride == 1 and a.coverage_config == "R3"
    assert ingest.parse_z_corrections(a.stereo_z_correction) == {100: 0.69, 101: -0.1}


# --- edge cases ------------------------------------------------------------


def test_empty_blob_and_ring_absent_from_the_correction_dict():
    """A sample whose ZED_WORLD blob is empty (24 of 15,547 have none) and a
    correction naming a ring that is not in the block must both be no-ops, not
    exceptions: the run walks 668 keyframes and one bad shape ends it."""
    empty = np.zeros((0, 5), dtype=np.float32)
    out = ingest.stereo_block_to_ego(empty, frame="global_identity", ring=None, t_sensor_to_ego=None,
                                     t_global_to_ego=np.eye(4), z_correction_m={100: 0.69})
    assert out.shape == (0, 5) and out.dtype == np.float64

    raw = np.array([[0, 0, 0, 1, 101.0]], dtype=np.float32)
    out = ingest.stereo_block_to_ego(raw, frame="global_identity", ring=None, t_sensor_to_ego=None,
                                     t_global_to_ego=np.eye(4), z_correction_m={7: 5.0})
    assert out[0, 2] == 0.0


def test_unknown_frame_handling_is_refused():
    """A typo'd frame convention must raise, never fall through to "leave the
    points where they are" — a cloud in the wrong frame looks plausible."""
    with pytest.raises(ValueError, match="glorbal"):
        ingest.stereo_block_to_ego(np.zeros((1, 5), dtype=np.float32), frame="glorbal", ring=None,
                                   t_sensor_to_ego=None, t_global_to_ego=np.eye(4), z_correction_m={})


def test_stereo_thinning_is_audited_per_channel_not_as_one_pair_of_totals():
    """Every cloud a keyframe opens is thinned, so the record has to name which
    channel each count came from: reporting one pair credited the LIDAR_TOP
    anchor's numbers and left the ZED_WORLD blob — the cloud the stride is FOR —
    out of the audit entirely (found in review, 2026-09-12)."""
    lidar_kept, lidar_removed = ingest.thin_stereo(
        np.array([[i, 0, 0, 1, i % 4] for i in range(40)], dtype=np.float32), (100, 101), 8)
    zed_raw = np.array([[i, 0, 0, 1, 100 + i % 2] for i in range(160)], dtype=np.float32)
    zed_kept, zed_removed = ingest.thin_stereo(zed_raw, (100, 101), 8)

    audit = ingest.stereo_thinning_audit(
        (100, 101), 8,
        {"LIDAR_TOP": 40, "ZED_WORLD": len(zed_raw)},
        {"LIDAR_TOP": lidar_removed, "ZED_WORLD": zed_removed},
    )
    assert audit["rings"] == [100, 101] and audit["stride"] == 8
    # The LiDAR blob holds no stereo rings, so nothing is removed from it; the
    # ZED blob loses 7 of every 8 and that has to be visible, not hidden.
    assert audit["n_removed"] == {"LIDAR_TOP": 0, "ZED_WORLD": 140}
    assert audit["n_raw_in_file"] == {"LIDAR_TOP": 40, "ZED_WORLD": 160}
    assert len(lidar_kept) == 40 and len(zed_kept) == 20
    assert json.loads(json.dumps(audit)) == audit      # it is written to disk


def test_config_serialises_the_z_correction_with_string_keys():
    cfg = ingest.IngestConfig(stereo_z_correction_m={100: 0.69})
    assert cfg.as_dict()["stereo_z_correction_m"] == {"100": 0.69}
