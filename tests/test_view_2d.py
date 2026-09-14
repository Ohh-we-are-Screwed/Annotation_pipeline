"""view_2d.py: the wire format the 2D page decodes (2026-09-12).

The HTML is read-only and checked by eye; what a test can pin is the export — that a
proposal keeps its arm and class, that the Stage 7 join by (keyframe, channel,
proposal_index) puts the right `status` on the right box, that a Stage 4 mask becomes a
polygon in image pixels, and that an absent Stage 7 tree degrades to no badges instead
of an exception.
"""
from __future__ import annotations
import json, os, sys, types
import numpy as np
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts import view_2d, view_2d_compare  # noqa: E402
from scripts.view_2d import export_keyframe, mask_polygon, write_image  # noqa: E402

CHANNELS = ("CAM_FRONT", "CAM_BACK")
TOKENS = ("kf0", "kf1")
W, H = 1280, 720   # MaskFile asserts the real image size; the fixture honours it


def _image(path):
    import cv2
    img = np.full((H, W, 3), 40, np.uint8)
    img[20:50, 30:80] = 200
    cv2.imwrite(str(path), img)
    return str(path)


def _mask_npz(path, per_channel):
    """A Stage 4 npz in the real format: bit-packed along the last axis, size in-file."""
    import cv2  # noqa: F401
    arrays = {"__width_px__": np.array([W], np.int32), "__height_px__": np.array([H], np.int32),
              "__bit_packed__": np.array([1], np.int8)}
    for ch, masks in per_channel.items():
        stack = np.zeros((len(masks), H, W), bool)
        for i, (x0, y0, x1, y1) in enumerate(masks):
            stack[i, y0:y1, x0:x1] = True
        arrays[ch] = np.packbits(stack, axis=-1)
    np.savez_compressed(path, **arrays)
    return str(path)


@pytest.fixture()
def scene(tmp_path, monkeypatch):
    """A two-keyframe, two-camera scene laid out exactly like the real work_root."""
    work, dataroot = tmp_path / "work", tmp_path / "data"
    (dataroot / "samples").mkdir(parents=True)
    for i, tok in enumerate(TOKENS):
        for ch in CHANNELS:
            _image(dataroot / "samples" / f"{tok}_{ch}.jpg")

    def _write(rel, rows):
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))

    _write("stage1_ingestion/scenes/s/keyframes.jsonl", [
        {"keyframe_token": tok, "t_ns": 100 + i,
         "cameras": {ch: {"path": f"samples/{tok}_{ch}.jpg", "channel": ch,
                          "calibrated_sensor_token": "cs", "width_px": W, "height_px": H}
                     for ch in CHANNELS}}
        for i, tok in enumerate(TOKENS)])
    _write("stage3_merged/scenes/s/proposals.jsonl", [
        {"keyframe_token": tok, "channel": ch, "image_size_px": [W, H],
         "boxes_xyxy_px": [[30.0, 20.0, 80.0, 50.0], [5.0, 5.0, 20.0, 20.0]],
         "class_names": ["a car", "a pedestrian"], "scores": [0.9012345, 0.42],
         "proposal_arm": ["arm_a", "arm_b"],
         "vlm_check": {"verdicts": [
             {"action": "relabeled", "vlm_phrase": "a car", "original_class_name": "a truck"},
             {"action": "skipped_small", "vlm_phrase": None}]}}
        for tok in TOKENS for ch in CHANNELS])
    _write("stage7_track/scenes/s/boxes.jsonl", [
        {"keyframe_token": "kf0", "channel": "CAM_FRONT", "proposal_index": 0, "status": "fit",
         "track_id": 7, "stereo": {"d_med_m": 12.5},
         "box": {"translation_m": [12.0, 0.0, -1.5], "size_wlh_m": [1.8, 4.2, 1.6], "yaw_rad": 0.1}},
        {"keyframe_token": "kf0", "channel": "CAM_FRONT", "proposal_index": 1,
         "status": "out_of_r3", "track_id": None, "box": None},
    ])
    masks = work / "stage4_masks/scenes/s/masks"
    masks.mkdir(parents=True)
    _write("stage4_masks/scenes/s/masks.jsonl", [
        {"keyframe_token": tok, "mask_path": f"scenes/s/masks/{tok}.npz"} for tok in TOKENS])
    for tok in TOKENS:
        _mask_npz(masks / f"{tok}.npz", {ch: [(30, 20, 80, 50), (5, 5, 20, 20)] for ch in CHANNELS})

    monkeypatch.setattr(view_2d, "load_paths",
                        lambda p: types.SimpleNamespace(work_root=str(work), dataroot=str(dataroot)))
    # the calibration path is view_boxes_3d's and tested there; here it only has to
    # produce a K and a pose so the `fit` row gets a wireframe.
    K = np.array([[60.0, 0, W / 2], [0, 60.0, H / 2], [0, 0, 1.0]])
    T = np.array([[0, -1.0, 0, 0], [0, 0, -1.0, 0.0], [1.0, 0, 0, 0.0], [0, 0, 0, 1.0]])
    monkeypatch.setattr(view_2d, "load_calibs",
                        lambda paths, kf, corr=None, channels=(): {ch: {"K": K, "T_cam_ego": T} for ch in CHANNELS})
    monkeypatch.setattr(view_2d, "CHANNELS", CHANNELS)
    return types.SimpleNamespace(work=work, dataroot=dataroot, out=tmp_path / "out")


def _run(scene, *extra):
    assert view_2d.main(["--paths", "unused", "--scene", "s", "--out", str(scene.out),
                         "--max-width", "64", "--workers", "2", *extra]) == 0
    index = json.loads((scene.out / "index.json").read_text())
    kfs = [json.loads((scene.out / "kf" / f"{i:05d}.json").read_text()) for i in range(2)]
    return index, kfs


def test_mask_polygon_traces_the_blob_in_image_pixels():
    m = np.zeros((H, W), bool)
    m[20:50, 30:80] = True
    poly = mask_polygon(m)
    xs, ys = poly[0::2], poly[1::2]
    assert len(poly) % 2 == 0 and len(xs) == 4                 # a rectangle survives as 4 corners
    assert (min(xs), max(xs), min(ys), max(ys)) == (30, 79, 20, 49)
    assert mask_polygon(np.zeros((H, W), bool)) is None


def test_write_image_downscales_and_reports_its_own_size(tmp_path):
    src = _image(tmp_path / "src.jpg")
    assert write_image(src, str(tmp_path / "a.jpg"), 64) == (64, 36)
    assert write_image(src, str(tmp_path / "b.jpg"), 4096) == (W, H)   # never upscales


def test_export_carries_proposals_masks_and_the_stage7_status(scene):
    index, (kf0, kf1) = _run(scene)

    assert index["scene"] == "s" and index["has_outcomes"] and index["has_masks"]
    assert [e["index"] for e in index["keyframes"]] == [0, 1]
    assert [e["token"] for e in index["keyframes"]] == list(TOKENS)
    assert index["keyframes"][0]["n_proposals"] == 4 and index["keyframes"][0]["n_fit"] == 1

    front = kf0["cameras"]["CAM_FRONT"]
    assert front["image"] == "img/kf0_CAM_FRONT.jpg"
    assert (scene.out / front["image"]).is_file()
    assert (front["w"], front["h"]) == (64, 36) and front["native"] == [W, H]

    car, ped = front["boxes"]
    assert (car["cls"], car["arm"], car["score"], car["i"]) == ("a car", "arm_a", 0.9012, 0)
    assert car["xyxy"] == [30.0, 20.0, 80.0, 50.0]
    assert (car["status"], car["track_id"], car["depth_m"]) == ("fit", 7, 12.5)
    assert len(car["wire"]) == 16                              # the fitted 3D box, projected
    assert (ped["cls"], ped["arm"], ped["status"]) == ("a pedestrian", "arm_b", "out_of_r3")
    assert "wire" not in ped

    # coordinates stay in the ORIGINAL image frame; the page scales by w/native
    assert min(car["poly"][0::2]) == 30 and max(car["poly"][1::2]) == 49
    assert min(ped["poly"][0::2]) == 5

    # a channel Stage 7 said nothing about gets no badge, and kf1 has no Stage 7 rows at all
    assert "status" not in kf0["cameras"]["CAM_BACK"]["boxes"][0]
    assert all("status" not in b for c in kf1["cameras"].values() for b in c["boxes"])
    assert kf1["cameras"]["CAM_FRONT"]["boxes"][0]["poly"]     # masks still there


def test_missing_stage7_and_stage4_trees_still_export(scene):
    index, (kf0, _) = _run(scene, "--boxes-dir", str(scene.work / "nope"), "--masks-dir", "")
    assert index["has_outcomes"] is False and index["has_masks"] is False
    box = kf0["cameras"]["CAM_FRONT"]["boxes"][0]
    assert box["cls"] == "a car" and "status" not in box and "poly" not in box


def test_polygon_budget_caps_what_one_keyframe_carries(scene):
    _, (kf0, _) = _run(scene, "--poly-budget", "4")
    polys = [b for c in kf0["cameras"].values() for b in c["boxes"] if "poly" in b]
    assert 0 < len(polys) < 4                                  # first few fit the budget, rest drop


def test_stage3c_verdict_rides_along_with_its_box(scene):
    """A checked proposals dir carries one verdict per proposal index into the JSON."""
    _, (kf0, _) = _run(scene)
    car, ped = kf0["cameras"]["CAM_FRONT"]["boxes"]
    assert car["vlm"] == {"action": "relabeled", "vlm_phrase": "a car",
                          "original_class_name": "a truck"}
    assert ped["vlm"]["action"] == "skipped_small"


def test_unchecked_proposals_carry_no_vlm_key(scene):
    """stage3_merged never saw Stage 3c; the box must not grow an empty verdict."""
    rows = [json.loads(l) for l in
            (scene.work / "stage3_merged/scenes/s/proposals.jsonl").read_text().splitlines()]
    for r in rows:
        r.pop("vlm_check")
    (scene.work / "stage3_merged/scenes/s/proposals.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    _, (kf0, _) = _run(scene)
    assert all("vlm" not in b for c in kf0["cameras"].values() for b in c["boxes"])


def test_compare_server_routes_two_exports_and_the_page(tmp_path):
    """/left/ and /right/ are the two --out dirs, / is compare.html, anything else is 404."""
    import threading, urllib.error, urllib.request
    from http.server import ThreadingHTTPServer

    left, right = tmp_path / "l", tmp_path / "r"
    for d, text in ((left, "LEFT"), (right, "RIGHT")):
        (d / "kf").mkdir(parents=True)
        (d / "index.json").write_text(text)
        (d / "kf" / "00000.json").write_text(text + " kf")
    handler = view_2d_compare.make_handler(str(left), str(right),
                                           {"left_label": "no VLM", "right_label": "VLM"})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        get = lambda p: urllib.request.urlopen(base + p, timeout=5).read().decode()
        assert get("/left/index.json") == "LEFT" and get("/right/index.json") == "RIGHT"
        assert get("/left/kf/00000.json") == "LEFT kf"
        assert json.loads(get("/meta.json"))["left_label"] == "no VLM"
        assert "<title>2D results" in get("/")            # viewer2d/compare.html, from the repo
        assert "paintBoxes" in get("/draw.js")            # shared with index.html
        for missing in ("/left/nope.json", "/nope.html"):
            with pytest.raises(urllib.error.HTTPError) as e:
                get(missing)
            assert e.value.code == 404
    finally:
        srv.shutdown()
        srv.server_close()
