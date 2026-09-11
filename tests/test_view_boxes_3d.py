"""view_boxes_3d.py: the two pieces of geometry the browser cannot check (2026-09-12).

The viewer is read-only and the HTML is exercised by eye; what a test can pin is
the wire format the page decodes (decimate + encode_cloud) and the projection the
image panels re-implement in JavaScript, which must agree with this one.
"""
from __future__ import annotations
import base64, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.view_boxes_3d import decimate, encode_cloud, export_keyframe, plane_abd, project_corners  # noqa: E402


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
