"""view_boxes_3d.py: the two pieces of geometry the browser cannot check (2026-09-12).

The viewer is read-only and the HTML is exercised by eye; what a test can pin is
the wire format the page decodes (decimate + encode_cloud) and the projection the
image panels re-implement in JavaScript, which must agree with this one.
"""
from __future__ import annotations
import base64, json, math, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.stage1_ingestion.ingest import pitch_rotate_xz  # noqa: E402
from scripts import view_boxes_3d  # noqa: E402
from scripts.view_boxes_3d import (  # noqa: E402
    decimate, encode_cloud, export_keyframe, load_calibs, pitch_correction_matrix, plane_abd,
    pose_corrections, project_corners,
)


def test_decimate_and_encode_roundtrip():
    cloud = np.column_stack([np.arange(1000.0), np.zeros(1000), np.zeros(1000), np.ones(1000), np.full(1000, 101.0)]).astype(np.float32)
    d = decimate(cloud, 100)
    assert d.shape[0] <= 100 and d.shape[1] == 5
    enc = encode_cloud(d)
    xyz = np.frombuffer(base64.b64decode(enc["xyz_b64"]), dtype=np.float32).reshape(-1, 3)
    ring = np.frombuffer(base64.b64decode(enc["ring_b64"]), dtype=np.uint8)
    assert xyz.shape[0] == ring.shape[0] == enc["n"] and set(ring.tolist()) == {101}


def test_project_corners_in_front_of_camera_land_in_image():
    K = np.array([[953.16, 0, 656.28], [0, 953.16, 375.74], [0, 0, 1.0]])
    T_cam_ego = np.array([[0, -1.0, 0, 0], [0, 0, -1.0, -0.7], [1.0, 0, 0, -0.8], [0, 0, 0, 1.0]])  # ego -> optical, cam 0.8 ahead
    uv, vis = project_corners([12.0, 0.0, -1.5], [1.15, 2.4, 1.75], 0.3, K, T_cam_ego, (1280, 720))
    assert uv.shape == (8, 2) and vis.all()
    assert (uv[:, 0] > 0).all() and (uv[:, 0] < 1280).all() and (uv[:, 1] > 0).all() and (uv[:, 1] < 720).all()


CHANNELS = ("CAM_FRONT", "CAM_BACK")


def _keyframe(tmp_path, token, image):
    """A minimal keyframe row + calibs export_keyframe can be run on."""
    cloud = tmp_path / f"{token}.pcd.bin"
    np.zeros((4, 5), dtype=np.float32).tofile(cloud)
    kf_row = {"keyframe_token": token, "t_ns": 7, "single_sweep_cloud": {"path": str(cloud)},
              "cameras": {ch: {"path": str(image)} for ch in CHANNELS}}
    return kf_row, {ch: {"K": np.eye(3), "T_cam_ego": np.eye(4)} for ch in CHANNELS}


def test_keyframe_without_a_ground_plane_still_exports(tmp_path):
    """Stage 1 writes ground_reference_plane: null when RANSAC found no plane
    (ingest.py:1451). Rare, but one such keyframe must not abort the scene."""
    assert plane_abd(None) is None
    assert plane_abd({"a": 1.0, "b": 2.0, "d": -1.6, "inliers": 9}) == {"a": 1.0, "b": 2.0, "d": -1.6}

    img = tmp_path / "x.jpg"
    img.write_bytes(b"\xff\xd8\xff")
    kf_row, calibs = _keyframe(tmp_path, "k0", img)
    out = tmp_path / "view"
    entry = export_keyframe(0, kf_row, [], calibs, plane_abd(None), str(tmp_path), str(out), 100)

    assert entry == {"index": 0, "token": "k0", "n_boxes": 0}
    payload = json.loads((out / "kf" / "00000.json").read_text())
    assert payload["ground"] is None and payload["cloud"]["n"] == 4


def test_images_are_keyed_by_token_and_refreshed(tmp_path):
    """One --out re-used for a second scene must not draw new boxes over old photos:
    the JPEGs are named by keyframe token, and an existing one is replaced."""
    out = tmp_path / "view"
    first = tmp_path / "first.jpg"
    first.write_bytes(b"\xff\xd8\xff-one")
    kf_row, calibs = _keyframe(tmp_path, "tok9", first)
    export_keyframe(0, kf_row, [], calibs, None, str(tmp_path), str(out), 100)

    payload = json.loads((out / "kf" / "00000.json").read_text())
    assert [payload["cameras"][ch]["image"] for ch in CHANNELS] == [f"img/tok9_{ch}.jpg" for ch in CHANNELS]
    for ch in CHANNELS:
        assert (out / "img" / f"tok9_{ch}.jpg").read_bytes() == b"\xff\xd8\xff-one"

    # a different source file for the same keyframe token: the copy is replaced, not kept
    second = tmp_path / "second.jpg"
    second.write_bytes(b"\xff\xd8\xff-two")
    kf_row["cameras"] = {ch: {"path": str(second)} for ch in CHANNELS}
    export_keyframe(0, kf_row, [], calibs, None, str(tmp_path), str(out), 100)
    for ch in CHANNELS:
        assert (out / "img" / f"tok9_{ch}.jpg").read_bytes() == b"\xff\xd8\xff-two"


# ---------------------------------------------------------------------------
# The CAM_FRONT pose correction (configs/stereo_box.yaml camera_pose_pitch_correction)
# ---------------------------------------------------------------------------

# The recorded stopgap: the same rotation as Stage 1's
# `--stereo-pitch-correction 101:-9.0974:0.81253:-0.73305`, pivot = the front ZED's
# own optical centre in the ego frame.
PITCH = {"deg": -9.0974, "pivot_x_m": 0.81253, "pivot_z_m": -0.73305}
K_ZED = np.array([[953.16, 0.0, 656.28], [0.0, 953.16, 375.74], [0.0, 0.0, 1.0]])
# cam -> ego for a camera at the pivot looking along ego +x: x_cam = -y_ego (right),
# y_cam = -z_ego (down), z_cam = +x_ego. Standard nuScenes front-camera quaternion.
CAM_QUAT_WXYZ = [0.5, -0.5, 0.5, -0.5]
GROUND_PT = [15.0, 0.0, -1.6]                     # ego, on the road 15 m ahead


def test_pitch_correction_matrix_agrees_with_stage1_pitch_rotate_xz():
    """One convention, two readers: the viewer's 4x4 must move a point exactly where
    Stage 1's `pitch_rotate_xz` moves it, or the image-space correction and the
    recorded flag are not the same rotation."""
    M = pitch_correction_matrix(PITCH["deg"], PITCH["pivot_x_m"], PITCH["pivot_z_m"])
    pts = np.array([[0.0, 0.0, 0.0], [15.0, 2.0, -1.6],
                    [PITCH["pivot_x_m"], -3.0, PITCH["pivot_z_m"]], [-4.0, 0.5, 2.0]])
    got = (np.column_stack([pts, np.ones(len(pts))]) @ M.T)[:, :3]
    wx, wz = pitch_rotate_xz(pts[:, 0], pts[:, 2], PITCH["deg"],
                             PITCH["pivot_x_m"], PITCH["pivot_z_m"])
    assert np.allclose(got[:, 0], wx, atol=1e-9)
    assert np.allclose(got[:, 2], wz, atol=1e-9)
    assert np.allclose(got[:, 1], pts[:, 1], atol=1e-9)        # y is invariant


def _stub_substrate(monkeypatch):
    """load_calibs' only substrate use: one calibrated_sensor record per ZED channel."""
    rec = {"translation": [PITCH["pivot_x_m"], 0.0, PITCH["pivot_z_m"]],
           "rotation": CAM_QUAT_WXYZ, "camera_intrinsic": K_ZED.tolist()}

    class _Sub:
        def by_token(self, name):
            assert name == "calibrated_sensor.json"
            return {"cs": rec}

    monkeypatch.setattr(view_boxes_3d.Substrate, "load", lambda paths: _Sub())
    return {"cameras": {ch: {"calibrated_sensor_token": "cs"} for ch in CHANNELS}}


def _v_of_ground_point(calib) -> float:
    """Image row of a point-sized box at GROUND_PT, through this channel's pose."""
    uv, vis = project_corners(GROUND_PT, [0.01, 0.01, 0.01], 0.0,
                              calib["K"], calib["T_cam_ego"], (1280, 720))
    assert vis.all()
    return float(uv[:, 1].mean())


def test_pose_correction_moves_the_projection_down_onto_the_object(monkeypatch):
    """SIGN, the whole point: the export's CAM_FRONT pose is pitched, so wireframes
    land ~150 px ABOVE their objects. The correction must move them DOWN (larger v),
    and land where the angle sum says they should."""
    kf_row = _stub_substrate(monkeypatch)
    plain = load_calibs(None, kf_row)
    corrected = load_calibs(None, kf_row, {"CAM_FRONT": PITCH})

    v_plain = _v_of_ground_point(plain["CAM_FRONT"])
    v_corr = _v_of_ground_point(corrected["CAM_FRONT"])
    assert 100.0 < v_corr - v_plain < 200.0                    # ~9.1 deg over 953 px focal

    # Hand-computed: the point sits atan(drop/depth) below the optical axis; correcting
    # the pose pitches the camera up by |deg|, so it lands at the SUM of the two angles.
    fy, cy = K_ZED[1, 1], K_ZED[1, 2]
    below = math.atan2(PITCH["pivot_z_m"] - GROUND_PT[2], GROUND_PT[0] - PITCH["pivot_x_m"])
    assert abs(v_corr - (cy + fy * math.tan(below + math.radians(-PITCH["deg"])))) < 1.0

    # the other channel is untouched, and so is the uncorrected call
    assert np.allclose(corrected["CAM_BACK"]["T_cam_ego"], plain["CAM_BACK"]["T_cam_ego"])


def test_config_carries_the_cam_front_pose_correction():
    """The yaml key the two image-space consumers read, and the value they read."""
    got = pose_corrections()
    assert got["CAM_FRONT"] == PITCH
