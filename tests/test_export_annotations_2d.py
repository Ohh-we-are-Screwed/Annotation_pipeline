"""export_annotations_2d.py: the 2D sidecar's wire format and its three joins (2026-09-13).

A synthetic work_root (2 keyframes x 2 cameras, a Stage 3m proposals.jsonl, a Stage 4
npz in the REAL bit-packed format, a Stage 7 boxes.jsonl, a Stage 9 prelabels.jsonl)
plus a minimal release (sample_data.json, sample_annotation.json carrying the matching
`dhakascenes_record_token`, stitch_map.json, DELIVERY_NOTE.md). What is pinned is what
a consumer reads: the COCO skeleton, one polygon per mask BLOB, and that the three
joins — Stage 7 status/track, stitch_map, sample_annotation — land on the right box.
"""

from __future__ import annotations

import json
import os
import sys
import types

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts import export_annotations_2d as ex  # noqa: E402

CHANNELS = ("CAM_BACK", "CAM_FRONT")
TOKENS = ("kf0", "kf1")
W, H = 1280, 720  # MaskFile asserts the substrate image size; the fixture honours it
VERSION = "v1.0-dhaka-fixed2"
SCENE = "s"
# kf0/CAM_FRONT/0 is the one detection that became a cuboid: Stage 7 fits it and gives
# it track 7, the stitcher renames that chain "c7", and the release shipped it.
REC = "kf0:CAM_FRONT:0"
SA_TOKEN = "sa-token-0"
INSTANCE = "inst-0"
# Two blobs in one mask (an occluded object) — the case view_2d's single-contour
# outline drops on purpose and a COCO segmentation must not.
BLOBS = {0: [(30, 20, 80, 50), (300, 200, 340, 260)], 1: [(5, 5, 20, 20)]}


def _mask_npz(path, per_channel):
    arrays = {"__width_px__": np.array([W], np.int32), "__height_px__": np.array([H], np.int32),
              "__bit_packed__": np.array([1], np.int8)}
    for ch, masks in per_channel.items():
        stack = np.zeros((len(masks), H, W), bool)
        for i, boxes in enumerate(masks):
            for (x0, y0, x1, y1) in boxes:
                stack[i, y0:y1, x0:x1] = True
        arrays[ch] = np.packbits(stack, axis=-1)
    np.savez_compressed(path, **arrays)


def _sample_data_token(tok, ch):
    return f"sd-{tok}-{ch}"


@pytest.fixture()
def scene(tmp_path, monkeypatch):
    work, export = tmp_path / "work", tmp_path / "chunk_99" / "boxes"

    def _write(rel, rows):
        p = work / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))

    _write(f"stage3_merged/scenes/{SCENE}/proposals.jsonl", [
        {"keyframe_token": tok, "channel": ch, "scene_token": "sc",
         "sample_data_token": _sample_data_token(tok, ch),
         "image_path": f"samples/{ch}/{tok}.jpg", "image_size_px": [W, H],
         "t_ns": 100 + i, "n_proposals": 2,
         "boxes_xyxy_px": [[30.0, 20.0, 80.0, 50.0], [5.0, 5.0, 20.0, 20.0]],
         "class_names": ["a car", "a pedestrian"], "scores": [0.9012345, 0.42],
         "nuscenes_categories": [["vehicle.car"], ["human.pedestrian.adult"]],
         "proposal_arm": ["arm_a", "arm_b"]}
        for i, tok in enumerate(TOKENS) for ch in CHANNELS])

    _write(f"stage7_track/scenes/{SCENE}/boxes.jsonl", [
        {"keyframe_token": "kf0", "channel": "CAM_FRONT", "proposal_index": 0,
         "status": "fit", "track_id": 7,
         "stereo": {"d_near_m": 12.5, "d_med_m": 12.9, "mad_m": 0.1, "n_stereo_kept": 166,
                    "frame_truncated": True, "zed_ring": 101},
         "box": {"translation_m": [12.0, 0.0, -1.5]}},
        {"keyframe_token": "kf0", "channel": "CAM_FRONT", "proposal_index": 1,
         "status": "out_of_r3", "track_id": None, "box": None},
        {"keyframe_token": "kf1", "channel": "CAM_FRONT", "proposal_index": 0,
         "status": "fit", "track_id": 7, "stereo": {"d_near_m": 13.5},
         "box": {"translation_m": [13.0, 0.0, -1.5]}},
    ])
    _write(f"stage9_qa/scenes/{SCENE}/prelabels.jsonl", [
        {"token": REC, "sample_token": "kf0", "category": "a car",
         "provenance": {"source": "pipeline", "tier": "auto_accept"}},
        {"token": "kf1:CAM_FRONT:0", "sample_token": "kf1", "category": "a car",
         "provenance": {"source": "pipeline", "tier": "flagged"}},
    ])

    masks = work / f"stage4_masks/scenes/{SCENE}/masks"
    masks.mkdir(parents=True)
    _write(f"stage4_masks/scenes/{SCENE}/masks.jsonl",
           [{"keyframe_token": tok, "mask_path": f"scenes/{SCENE}/masks/{tok}.npz"} for tok in TOKENS])
    for tok in TOKENS:
        _mask_npz(masks / f"{tok}.npz", {ch: [BLOBS[0], BLOBS[1]] for ch in CHANNELS})

    # --- the minimal release ------------------------------------------------
    (export / VERSION).mkdir(parents=True)

    def _json(rel, payload):
        (export / rel).write_text(json.dumps(payload))

    _json(f"{VERSION}/sample_data.json", [
        {"token": _sample_data_token(tok, ch), "sample_token": tok,
         "filename": f"samples/{ch}/{tok}.jpg", "fileformat": "jpg",
         "width": W, "height": H, "timestamp": 1_700_000_000_000_000 + i, "is_key_frame": True}
        for i, tok in enumerate(TOKENS) for ch in CHANNELS
    ] + [{"token": f"lidar-{tok}", "sample_token": tok, "filename": f"samples/LIDAR_TOP/{tok}.pcd.bin",
          "fileformat": "pcd", "width": 0, "height": 0, "timestamp": 1, "is_key_frame": True}
         for tok in TOKENS])
    _json(f"{VERSION}/sample_annotation.json", [
        {"token": SA_TOKEN, "sample_token": "kf0", "instance_token": INSTANCE,
         "dhakascenes_record_token": REC, "dhakascenes_tier": "auto_accept",
         "dhakascenes_chain_id": "c7", "dhakascenes_track_id_pre_stitch": "7",
         "dhakascenes_velocity_chain_mps": [3.0, 4.0]},
    ])
    _json(f"{VERSION}/category.json", [{"token": "cat-car", "name": "car", "description": ""}])
    _json("stitch_map.json", {REC: "c7"})
    _json("release_meta.json", {"version": VERSION, "git_sha": "deadbeef",
                                "mapper": {"used": {"a car": "car", "a pedestrian": "pedestrian"}}})
    (export / "DELIVERY_NOTE.md").write_text(
        "# chunk_99 — delivery note\n\n## Annotation rule\nA measured box ships iff X.\n\n"
        "## Range\n- pipeline range cap: 50 m\n")

    monkeypatch.setattr(ex, "load_paths", lambda p: types.SimpleNamespace(work_root=str(work)))
    return types.SimpleNamespace(work=work, export=export)


def _run(scene, *extra):
    assert ex.main(["--paths", "unused", "--scene", SCENE,
                    "--export-dir", str(scene.export), *extra]) == 0
    out = scene.export / "annotations_2d"
    return (json.loads((out / "instances_2d.json").read_text()),
            json.loads((out / "tracks.json").read_text()))


# ---------------------------------------------------------------------------
# COCO skeleton
# ---------------------------------------------------------------------------

def test_coco_skeleton_and_counts(scene):
    coco, _ = _run(scene)
    assert set(coco) >= {"info", "licenses", "images", "categories", "annotations"}
    assert len(coco["images"]) == 4                       # 2 keyframes x 2 cameras, no LIDAR row
    assert len(coco["annotations"]) == 8                  # x 2 proposals
    assert {im["channel"] for im in coco["images"]} == set(CHANNELS)
    one = coco["images"][0]
    assert one["file_name"].startswith("samples/") and one["file_name"].endswith(".jpg")
    assert (one["width"], one["height"]) == (W, H)
    assert one["sample_data_token"] and one["sample_token"] and one["timestamp"]
    # ids are dense, unique, 1-based — what every COCO loader assumes
    assert sorted(im["id"] for im in coco["images"]) == list(range(1, 5))
    assert sorted(a["id"] for a in coco["annotations"]) == list(range(1, 9))
    assert {a["image_id"] for a in coco["annotations"]} == {im["id"] for im in coco["images"]}


def test_categories_carry_the_phrase_and_the_release_class(scene):
    coco, _ = _run(scene)
    by_name = {c["name"]: c for c in coco["categories"]}
    assert "a car" in by_name and "a pedestrian" in by_name
    assert by_name["a car"]["nuscenes_category"] == "car"
    assert by_name["a car"]["nuscenes_category_token"] == "cat-car"
    assert by_name["a pedestrian"]["nuscenes_category_token"] is None   # absent from category.json
    assert len({c["id"] for c in coco["categories"]}) == len(coco["categories"])
    cat_of = {c["id"]: c["name"] for c in coco["categories"]}
    for a in coco["annotations"]:
        assert cat_of[a["category_id"]] in ("a car", "a pedestrian")


def test_bbox_area_score_and_arm(scene):
    coco, _ = _run(scene)
    a = _one(coco, REC)
    assert a["bbox"] == [30.0, 20.0, 50.0, 30.0]          # xyxy -> xywh
    assert a["area"] == 50.0 * 30.0
    assert a["iscrowd"] == 0
    assert a["score"] == pytest.approx(0.9012345)
    assert a["dhakascenes"]["detector_arm"] == "arm_a"
    assert a["dhakascenes"]["proposal_index"] == 0
    assert a["dhakascenes"]["keyframe_token"] == "kf0"


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def test_a_two_blob_mask_yields_two_polygons(scene):
    coco, _ = _run(scene)
    a = _one(coco, REC)
    assert len(a["segmentation"]) == 2                    # both blobs, not just the largest
    for poly in a["segmentation"]:
        assert len(poly) % 2 == 0 and len(poly) >= 6
    boxes = sorted(_poly_bbox(p) for p in a["segmentation"])
    assert boxes == [(30, 20, 79, 49), (300, 200, 339, 259)]
    assert a["dhakascenes"]["mask_area_px"] == 30 * 50 + 40 * 60


def test_empty_mask_is_an_empty_segmentation(scene):
    for tok in TOKENS:      # blank every mask of one channel
        _mask_npz(scene.work / f"stage4_masks/scenes/{SCENE}/masks/{tok}.npz",
                  {"CAM_FRONT": [[], []], "CAM_BACK": [BLOBS[0], BLOBS[1]]})
    coco, _ = _run(scene)
    a = _one(coco, REC)
    assert a["segmentation"] == [] and a["dhakascenes"]["mask_area_px"] == 0
    assert a["bbox"] == [30.0, 20.0, 50.0, 30.0]          # the 2D box is unaffected


def test_absent_mask_tree_still_exports(scene):
    coco, _ = _run(scene, "--masks-dir", "")
    assert len(coco["annotations"]) == 8
    assert all(a["segmentation"] == [] for a in coco["annotations"])
    assert coco["info"]["counts"]["annotations_with_polygons"] == 0


# ---------------------------------------------------------------------------
# The three joins
# ---------------------------------------------------------------------------

def test_status_3d_track_and_release_links_resolve(scene):
    coco, _ = _run(scene)
    a = _one(coco, REC)["dhakascenes"]
    assert a["status_3d"] == "fit"
    assert a["track_id"] == "7"
    assert a["track_id_stitched"] == "c7"
    assert a["sample_annotation_token"] == SA_TOKEN
    assert a["instance_token"] == INSTANCE
    assert a["tier"] == "auto_accept"
    assert a["depth_m"] == pytest.approx(12.5)            # stereo.d_near_m
    assert a["vlm_label"] is None

    lifted = _one(coco, "kf0:CAM_FRONT:1")["dhakascenes"]
    assert lifted["status_3d"] == "out_of_r3"
    assert lifted["track_id"] is None and lifted["track_id_stitched"] is None
    assert lifted["sample_annotation_token"] is None and lifted["instance_token"] is None
    assert lifted["depth_m"] is None

    # A Stage 7 row that fitted but never shipped keeps its track and its tier.
    kept = _one(coco, "kf1:CAM_FRONT:0")["dhakascenes"]
    assert kept["status_3d"] == "fit" and kept["track_id"] == "7"
    assert kept["tier"] == "flagged" and kept["sample_annotation_token"] is None

    # No Stage 7 row at all -> absent, never a KeyError.
    assert _one(coco, "kf0:CAM_BACK:0")["dhakascenes"]["status_3d"] == "absent"


def test_vlm_label_comes_from_a_checked_tree(scene):
    checked = scene.work / f"stage3_checked/scenes/{SCENE}"
    checked.mkdir(parents=True)
    (checked / "proposals.jsonl").write_text("".join(
        json.dumps({"keyframe_token": tok, "channel": ch,
                    "vlm_check": {"verdicts": [{"action": "relabeled", "vlm_phrase": "a truck"},
                                               {"action": "confirmed", "vlm_phrase": "a pedestrian"}]}}) + "\n"
        for tok in TOKENS for ch in CHANNELS))
    coco, _ = _run(scene)
    assert _one(coco, REC)["dhakascenes"]["vlm_label"] == "a truck"
    assert _one(coco, "kf0:CAM_FRONT:1")["dhakascenes"]["vlm_label"] == "a pedestrian"


# ---------------------------------------------------------------------------
# tracks.json
# ---------------------------------------------------------------------------

def test_tracks_index_links_2d_to_3d(scene):
    coco, tracks = _run(scene)
    assert set(tracks) >= {"info", "tracks"}
    assert len(tracks["tracks"]) == 1
    t = tracks["tracks"][0]
    assert t["track_id"] == "7" and t["track_id_stitched"] == "c7"
    assert t["instance_token"] == INSTANCE
    assert t["class_name"] == "a car"
    assert t["n_keyframes"] == 2
    assert t["first_sample_token"] == "kf0" and t["last_sample_token"] == "kf1"
    assert [k["sample_token"] for k in t["keyframes"]] == ["kf0", "kf1"]
    assert t["keyframes"][0]["sample_annotation_token"] == SA_TOKEN
    assert t["keyframes"][1]["sample_annotation_token"] is None
    assert t["mean_velocity_mps"] == pytest.approx([3.0, 4.0])
    assert t["mean_speed_mps"] == pytest.approx(5.0)
    # the annotation ids it names really are the 2D rows of that keyframe/channel
    ann_id = t["keyframes"][0]["annotation_ids"]["CAM_FRONT"]
    assert _by_id(coco, ann_id)["dhakascenes"]["keyframe_token"] == "kf0"
    assert _by_id(coco, ann_id)["dhakascenes"]["track_id"] == "7"


# ---------------------------------------------------------------------------
# Provenance, idempotence, and the note
# ---------------------------------------------------------------------------

def test_info_carries_provenance_and_the_release_caveats(scene):
    coco, _ = _run(scene)
    info = coco["info"]
    assert info["scenes"] == [SCENE] and info["chunk"] == "chunk_99"
    assert info["release"]["git_sha"] == "deadbeef" and info["release"]["version"] == VERSION
    assert info["sources"]["stage3_dir"].endswith("stage3_merged")
    assert info["counts"] == {
        "images": 4, "annotations": 8, "annotations_with_polygons": 8,
        "annotations_with_empty_segmentation": 0, "annotations_with_untraceable_mask": 0,
        "annotations_linked_to_3d": 1, "annotations_with_a_3d_status": 3, "tracks": 1}
    assert "A measured box ships iff X." in info["caveats"]["annotation_rule"]
    assert "pipeline range cap: 50 m" in info["caveats"]["range"]
    assert info["caveats"]["delivery_note"] == "../DELIVERY_NOTE.md"
    assert info["counts_by_status_3d"]["fit"] == 2 and info["counts_by_status_3d"]["absent"] == 5


def test_rerun_is_idempotent_and_the_note_keeps_every_sentence(scene):
    before = (scene.export / "DELIVERY_NOTE.md").read_text()
    first, _ = _run(scene)
    once = (scene.export / "DELIVERY_NOTE.md").read_text()
    second, _ = _run(scene)
    twice = (scene.export / "DELIVERY_NOTE.md").read_text()
    assert once == twice                                  # the section is replaced, not appended twice
    assert once.count(ex.NOTE_HEADING) == 1
    for line in before.splitlines():
        assert line in once                               # every existing sentence survives
    assert "annotations_2d/instances_2d.json" in once
    for doc in (first, second):
        for volatile in ("created_utc", "elapsed_s"):
            doc["info"].pop(volatile, None)
    assert first == second


def test_output_loads_as_coco(scene):
    out = scene.export / "annotations_2d" / "instances_2d.json"
    _run(scene)
    pycocotools = pytest.importorskip("pycocotools.coco", reason="optional COCO loader")
    coco = pycocotools.COCO(str(out))
    assert len(coco.getImgIds()) == 4 and len(coco.getAnnIds()) == 8
    ann = coco.loadAnns(coco.getAnnIds())[0]
    assert ann["bbox"] and "segmentation" in ann


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _one(coco, record_token):
    hits = [a for a in coco["annotations"]
            if a["dhakascenes"]["record_token"] == record_token]
    assert len(hits) == 1, f"{record_token}: {len(hits)} annotations"
    return hits[0]


def _by_id(coco, ann_id):
    return next(a for a in coco["annotations"] if a["id"] == ann_id)


def _poly_bbox(poly):
    xs, ys = poly[0::2], poly[1::2]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def test_every_scene_of_the_release_is_exported_when_no_scene_is_named(scene, tmp_path):
    """A second scene under stage3_merged joins the same COCO doc — the wrapper
    passes run_stages.sh's scene list, which is empty on a whole-release run."""
    src = scene.work / f"stage3_merged/scenes/{SCENE}"
    dst = scene.work / "stage3_merged/scenes/s2"
    dst.mkdir(parents=True)
    rows = [json.loads(line) for line in (src / "proposals.jsonl").read_text().splitlines()]
    rows = rows[:len(CHANNELS)]          # one keyframe, one row per camera
    for r in rows:                       # a distinct keyframe, so the two scenes cannot collide
        r["keyframe_token"] = "kf2"
        r["sample_data_token"] = f"sd-kf2-{r['channel']}"
    (dst / "proposals.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    tables = scene.export / VERSION / "sample_data.json"
    sd = json.loads(tables.read_text())
    sd += [{"token": f"sd-kf2-{ch}", "sample_token": "kf2", "filename": f"samples/{ch}/kf2.jpg",
            "fileformat": "jpg", "width": W, "height": H, "timestamp": 1_700_000_000_000_009,
            "is_key_frame": True} for ch in CHANNELS]
    tables.write_text(json.dumps(sd))

    assert ex.main(["--paths", "unused", "--export-dir", str(scene.export)]) == 0
    coco = json.loads((scene.export / "annotations_2d" / "instances_2d.json").read_text())
    assert coco["info"]["scenes"] == [SCENE, "s2"]
    assert len(coco["images"]) == 6 and len(coco["annotations"]) == 12
    assert sorted(im["id"] for im in coco["images"]) == list(range(1, 7))   # ids never restart


# ---------------------------------------------------------------------------
# review round: empty-mask caveat, category fallback, stereo diagnostics, area
# ---------------------------------------------------------------------------

def _blank_cam_front(scene):
    """Every CAM_FRONT mask empty -> 4 of the 8 annotations get `segmentation: []`."""
    for tok in TOKENS:
        _mask_npz(scene.work / f"stage4_masks/scenes/{SCENE}/masks/{tok}.npz",
                  {"CAM_FRONT": [[], []], "CAM_BACK": [BLOBS[0], BLOBS[1]]})


def test_empty_segmentation_is_counted_and_named_in_the_caveats(scene):
    _blank_cam_front(scene)
    coco, _ = _run(scene)
    counts = coco["info"]["counts"]
    assert counts["annotations_with_empty_segmentation"] == 4
    assert counts["annotations_with_empty_segmentation"] == (
        counts["annotations"] - counts["annotations_with_polygons"])
    caveat = _caveat(coco, "segmentation: []")
    assert "4 of 8 annotations" in caveat
    # the actionable half: the exact failure and the exact guard
    assert "annToMask" in caveat and "frPyObjects" in caveat and "IndexError" in caveat
    assert 'if ann["segmentation"]' in caveat
    # the selector must be the empty list, NOT the mask area
    assert "ONLY RELIABLE SELECTOR IS THE EMPTY LIST" in caveat
    empty = [a for a in coco["annotations"] if not a["segmentation"]]
    assert len(empty) == 4
    assert all(a["dhakascenes"]["mask_area_px"] == 0 for a in empty)
    assert counts["annotations_with_untraceable_mask"] == 0    # these really are blank masks
    assert "4 of them have `dhakascenes.mask_area_px == 0`" in caveat


def test_the_delivery_note_carries_the_empty_mask_count_for_this_chunk(scene):
    _blank_cam_front(scene)
    _run(scene)
    note = (scene.export / "DELIVERY_NOTE.md").read_text()
    assert "4 of 8 annotations" in note and "annToMask" in note
    assert "4 have `segmentation: []`" in note


def test_polygon_area_caveat_blames_the_contour_convention_not_epsilon(scene):
    coco, _ = _run(scene)
    caveat = _caveat(coco, "POLYGON ENCLOSES LESS AREA")
    assert "CENTRES of the boundary pixels" in caveat
    assert "NOT" in caveat and "simplification" in caveat
    assert "0.9843" in caveat and "0.9865" in caveat     # both measured sweeps
    assert "mask_area_px" in caveat


def test_category_falls_back_to_the_release_mapper_file(scene, tmp_path):
    """`mapper.used` lists only what shipped; the recorded mapper file fills the rest."""
    mapper = tmp_path / "release_category_map.yaml"
    mapper.write_text("map:\n  \"a car\": car\n  \"a traffic cone\": traffic_cone\n")
    import hashlib
    meta = json.loads((scene.export / "release_meta.json").read_text())
    meta["mapper"] = {"used": {"a car": "car"}, "path": str(mapper),
                      "sha256": hashlib.sha256(mapper.read_bytes()).hexdigest()}
    (scene.export / "release_meta.json").write_text(json.dumps(meta))

    coco, _ = _run(scene)
    by_name = {c["name"]: c for c in coco["categories"]}
    assert by_name["a car"]["nuscenes_category"] == "car"          # from `used`
    assert by_name["a traffic cone"]["nuscenes_category"] == "traffic_cone"   # from the file
    assert by_name["a pedestrian"]["nuscenes_category"] is None    # in neither: still null
    assert "sha256 verified" in coco["info"]["category_mapping_source"]


def test_a_changed_mapper_file_is_not_trusted(scene, tmp_path):
    mapper = tmp_path / "release_category_map.yaml"
    mapper.write_text("map:\n  \"a traffic cone\": traffic_cone\n")
    meta = json.loads((scene.export / "release_meta.json").read_text())
    meta["mapper"] = {"used": {"a car": "car"}, "path": str(mapper), "sha256": "0" * 64}
    (scene.export / "release_meta.json").write_text(json.dumps(meta))

    coco, _ = _run(scene)
    by_name = {c["name"]: c for c in coco["categories"]}
    assert by_name["a traffic cone"]["nuscenes_category"] is None
    assert "has changed on disk" in coco["info"]["category_mapping_source"]


def test_categories_carry_the_detectors_own_nuscenes_names(scene):
    """A DIFFERENT namespace from `nuscenes_category`, so it gets its own field."""
    coco, _ = _run(scene)
    by_name = {c["name"]: c for c in coco["categories"]}
    assert by_name["a car"]["detector_nuscenes_categories"] == ["vehicle.car"]
    assert by_name["a pedestrian"]["detector_nuscenes_categories"] == ["human.pedestrian.adult"]
    assert by_name["a truck"]["detector_nuscenes_categories"] is None   # no proposal used it
    # and it is NOT repeated on every annotation
    assert "nuscenes_categories" not in coco["annotations"][0]["dhakascenes"]


def test_stereo_diagnostics_ride_with_the_row(scene):
    coco, _ = _run(scene)
    fit = _one(coco, REC)["dhakascenes"]
    assert fit["depth_m"] == pytest.approx(12.5)       # d_near_m, unchanged
    assert fit["d_med_m"] == pytest.approx(12.9)
    assert fit["mad_m"] == pytest.approx(0.1)
    assert fit["n_stereo_kept"] == 166
    assert fit["frame_truncated"] is True
    assert fit["zed_ring"] == 101
    # a row Stage 7 wrote without a stereo block: present, null, no KeyError
    lifted = _one(coco, "kf0:CAM_FRONT:1")["dhakascenes"]
    assert lifted["d_med_m"] is None and lifted["zed_ring"] is None
    # a row Stage 7 never wrote at all
    absent = _one(coco, "kf0:CAM_BACK:0")["dhakascenes"]
    assert absent["d_med_m"] is None and absent["n_stereo_kept"] is None


def test_area_is_the_product_of_the_rounded_bbox(scene):
    """What a strict validator checks: area == w * h, bit-exact, on every row."""
    coco, _ = _run(scene)
    for a in coco["annotations"]:
        assert a["area"] == a["bbox"][2] * a["bbox"][3]


def test_area_holds_for_an_awkward_float_box(scene):
    """A box whose extents do not round to whole pixels is the case that breaks
    `round(raw_w * raw_h, 2)`."""
    path = scene.work / f"stage3_merged/scenes/{SCENE}/proposals.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for r in rows:
        r["boxes_xyxy_px"] = [[655.128, 392.7614, 675.7891, 409.6502], [5.0, 5.0, 20.0, 20.0]]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    coco, _ = _run(scene)
    a = _one(coco, REC)
    assert a["bbox"] == [655.13, 392.76, 20.66, 16.89]
    assert a["area"] == 20.66 * 16.89


def test_a_named_scene_without_proposals_is_a_refusal_not_a_traceback(scene):
    with pytest.raises(SystemExit) as exc:
        ex.main(["--paths", "unused", "--scene", "no_such_scene",
                 "--export-dir", str(scene.export)])
    message = str(exc.value)
    assert "no_such_scene" in message and "no Stage 3m proposals" in message
    assert "proposals.jsonl" in message


def test_a_stale_file_in_the_layer_is_removed(scene):
    out = scene.export / "annotations_2d"
    out.mkdir(parents=True)
    (out / "instances_2d.json.tmp").write_text("half a write")
    (out / "from_an_older_exporter.json").write_text("{}")
    (out / "masks" / "stale_scene").mkdir(parents=True)
    _run(scene)
    assert sorted(p.name for p in out.iterdir()) == ["instances_2d.json", "masks", "tracks.json"]
    assert sorted(p.name for p in (out / "masks").iterdir()) == [SCENE]


def test_stage4_pixel_masks_are_carbon_copied_into_the_layer(scene):
    coco, _ = _run(scene)
    src = scene.work / f"stage4_masks/scenes/{SCENE}"
    dst = scene.export / "annotations_2d" / "masks" / SCENE
    assert (dst / "masks.jsonl").read_bytes() == (src / "masks.jsonl").read_bytes()
    for tok in TOKENS:
        f = dst / "masks" / f"{tok}.npz"
        assert f.is_file() and not f.is_symlink()
        assert f.read_bytes() == (src / "masks" / f"{tok}.npz").read_bytes()
    m = coco["info"]["masks"]
    assert m["pixel_masks_copied"] and m["scenes"][0]["files"] == 1 + len(TOKENS)
    assert "proposal_index" in m["format"] and "unpackbits" in m["format"]
    note = (scene.export / "DELIVERY_NOTE.md").read_text()
    assert "annotations_2d/masks/" in note and f"{1 + len(TOKENS)} Stage 4 npz" in note


def test_no_stage4_tree_means_no_masks_dir_and_says_so(scene):
    coco, _ = _run(scene, "--masks-dir", "")
    assert not (scene.export / "annotations_2d" / "masks").exists()
    assert coco["info"]["masks"]["pixel_masks_copied"] is False
    assert "none: the run had no Stage 4 tree" in (scene.export / "DELIVERY_NOTE.md").read_text()


def _caveat(coco, needle):
    hits = [c for c in coco["info"]["caveats"]["two_d_layer"] if needle in c]
    assert len(hits) == 1, f"{needle!r}: {len(hits)} caveats"
    return hits[0]


def test_a_speckle_mask_has_pixels_but_no_polygon(scene):
    """The case that makes `mask_area_px == 0` the WRONG selector for empty
    segmentations: single-pixel speckle has mask area but nothing traceable.
    1 567 of the 2 932 empty segmentations across the eight shipped chunks are this."""
    import numpy as np
    W_, H_ = W, H
    arrays = {"__width_px__": np.array([W_], np.int32), "__height_px__": np.array([H_], np.int32),
              "__bit_packed__": np.array([1], np.int8)}
    for ch in CHANNELS:
        stack = np.zeros((2, H_, W_), bool)
        stack[0][::40, ::40] = True        # scattered single pixels: many blobs, no area
        for (x0, y0, x1, y1) in BLOBS[1]:  # index 1 keeps a normal traceable blob
            stack[1, y0:y1, x0:x1] = True
        arrays[ch] = np.packbits(stack, axis=-1)
    for tok in TOKENS:
        np.savez_compressed(scene.work / f"stage4_masks/scenes/{SCENE}/masks/{tok}.npz", **arrays)

    coco, _ = _run(scene)
    speckle = _one(coco, REC)
    assert speckle["segmentation"] == []
    assert speckle["dhakascenes"]["mask_area_px"] > 0       # pixels, but untraceable
    counts = coco["info"]["counts"]
    assert counts["annotations_with_untraceable_mask"] == 4
    assert counts["annotations_with_empty_segmentation"] == 4
    caveat = _caveat(coco, "segmentation: []")
    assert "0 of them have `dhakascenes.mask_area_px == 0`" in caveat
    assert "the other 4 DO have mask pixels" in caveat
