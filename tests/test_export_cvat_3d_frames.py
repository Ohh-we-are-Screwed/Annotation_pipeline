from __future__ import annotations

import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml  # noqa: E402

from pipeline.common.paths import METADATA_TABLES  # noqa: E402
import numpy as np  # noqa: E402

from scripts.export_cvat_3d import (  # noqa: E402
    CUBOID_ATTRIBUTES, archive_matches_selection, cuboid, datumaro_document, frames_manifest,
    load_stitch_map, main, stable_track_int,
)

SCENE = "scene-0001"
KEYFRAMES = ("s1", "s2", "s3")
TAXONOMY = os.path.join(ROOT, "configs/taxonomy_pilot_dhaka.yaml")


def _substrate(tmp_path, clouds=False):
    """A synthetic tree main() can run over with --skip-archive: no clouds, no CVAT.

    `clouds=True` also writes the per-keyframe .pcd.bin and the camera jpgs the
    archive is packed from, so a real task.zip can be built (and reused, or not).
    """
    dataroot = tmp_path / "data"
    work = tmp_path / "work"
    for sub in ("v1.0-mini", "samples", "sweeps"):
        (dataroot / sub).mkdir(parents=True)
    for d in (work, tmp_path / "out", tmp_path / "probe"):
        d.mkdir()

    tables = {name: [] for name in METADATA_TABLES}
    tables["sensor.json"] = [{"token": "sen", "channel": "LIDAR_TOP"}]
    tables["calibrated_sensor.json"] = [{"token": "cs0", "sensor_token": "sen"}]
    tables["scene.json"] = [{"token": "sc0", "name": SCENE}]
    tables["sample.json"] = [{"token": t, "scene_token": "sc0"} for t in KEYFRAMES]
    tables["ego_pose.json"] = [{"token": f"pose_{t}", "translation": [1.0, 2.0, 0.0],
                                "rotation": [1.0, 0.0, 0.0, 0.0]} for t in KEYFRAMES]
    tables["category.json"] = [{"token": "cat0", "name": "human.pedestrian.adult"}]
    tables["instance.json"] = [{"token": "inst0", "category_token": "cat0"}]
    tables["sample_annotation.json"] = [
        {"token": "ann0", "sample_token": "s1", "instance_token": "inst0",
         "translation": [3.0, 4.0, 1.0], "rotation": [1.0, 0.0, 0.0, 0.0], "size": [0.7, 0.8, 1.8]}]
    for name, rows in tables.items():
        (dataroot / "v1.0-mini" / name).write_text(json.dumps(rows))

    scene_dir = work / "stage1_ingestion" / "scenes" / SCENE
    scene_dir.mkdir(parents=True)
    scene_dir.joinpath("keyframes.jsonl").write_text("".join(json.dumps({
        "keyframe_token": t, "lidar_ego_pose_token": f"pose_{t}",
        "cameras": {"CAM_FRONT": {"path": "samples/CAM_FRONT/x.jpg"},
                    "CAM_BACK": {"path": "samples/CAM_BACK/x.jpg"}},
        "single_sweep_cloud": {"path": str(work / "clouds" / f"{t}.pcd.bin")
                               if clouds else "unused-with-skip-archive"}}) + "\n" for t in KEYFRAMES))
    if clouds:
        (work / "clouds").mkdir()
        for t in KEYFRAMES:
            np.arange(15, dtype=np.float32).reshape(3, 5).tofile(work / "clouds" / f"{t}.pcd.bin")
        for channel in ("CAM_FRONT", "CAM_BACK"):
            (dataroot / "samples" / channel).mkdir(parents=True)
            (dataroot / "samples" / channel / "x.jpg").write_bytes(b"\xff\xd8\xff")

    boxes = work / "stage8_inflate" / "scenes" / SCENE
    boxes.mkdir(parents=True)
    # a stitched chain over s1/s2, a lone detection, and a row the stitch map misses
    rows = (("s1", 0, "7"), ("s2", 1, "7"), ("s2", 2, None), ("s3", 0, "9"))
    boxes.joinpath("inflated.jsonl").write_text("".join(json.dumps({
        "keyframe_token": kf, "channel": "CAM_FRONT", "proposal_index": pi, "track_id": tid,
        "class_name": "a pedestrian",
        "box": {"translation_m": [1.0, 2.0, 0.5], "size_wlh_m": [0.7, 0.8, 1.8],
                "yaw_rad": 0.25}}) + "\n" for kf, pi, tid in rows))

    cfg = tmp_path / "paths.yaml"
    cfg.write_text(yaml.safe_dump({
        "dataroot": str(dataroot), "meta_root": str(dataroot), "version": "v1.0-mini",
        "work_root": str(work), "out_root": str(tmp_path / "out"),
        "probe_out_root": str(tmp_path / "probe")}))
    return str(cfg), work


def test_cuboid_carries_declared_attributes():
    c = cuboid(3, 1, [1, 2, 3], 0.5, (4.5, 1.8, 1.6), record_token="k0:CAM_FRONT:0", track_id=7)
    assert c["attributes"]["record_token"] == "k0:CAM_FRONT:0" and c["attributes"]["track_id"] == 7
    assert c["attributes"]["uncertain"] is False and c["attributes"]["attribute"] == ""
    assert c["scale"] == [4.5, 1.8, 1.6] and c["rotation"] == [0.0, 0.0, 0.5]
    c2 = cuboid(4, 1, [0, 0, 0], 0.0, (1, 1, 1))
    assert "track_id" not in c2["attributes"] and c2["attributes"]["record_token"] == ""


def test_document_declares_attributes_on_every_label():
    doc = datumaro_document(["a car", "a pedestrian"], [])
    for lab in doc["categories"]["label"]["labels"]:
        assert lab["attributes"] == list(CUBOID_ATTRIBUTES)


def test_frames_manifest_and_stitch_map(tmp_path):
    kfs = [{"keyframe_token": "s1", "cameras": {"CAM_FRONT": {}, "CAM_BACK": {}}}, {"keyframe_token": "s2", "cameras": {"CAM_FRONT": {}}}]
    fm = frames_manifest(kfs)
    assert fm == [{"frame": 0, "name": "000001", "sample_token": "s1", "channels": ["CAM_BACK", "CAM_FRONT"]},
                  {"frame": 1, "name": "000002", "sample_token": "s2", "channels": ["CAM_FRONT"]}]
    p = tmp_path / "stitch_map.json"
    p.write_text(json.dumps({"k0:CAM_FRONT:0": "7", "k1:CAM_FRONT:2": "det:k1:CAM_FRONT:2"}))
    m = load_stitch_map(str(p))
    assert stable_track_int(m["k0:CAM_FRONT:0"]) == 7
    assert stable_track_int(m["k1:CAM_FRONT:2"]) > 0 and stable_track_int(m["k1:CAM_FRONT:2"]) == stable_track_int("det:k1:CAM_FRONT:2")


def test_start_index_offsets_frames_and_track_id_is_coerced_to_int():
    kfs = [{"keyframe_token": "s9", "cameras": {"CAM_FRONT": {}}}]
    assert frames_manifest(kfs, start_index=4) == [
        {"frame": 4, "name": "000005", "sample_token": "s9", "channels": ["CAM_FRONT"]}
    ]
    # Stage 7 writes track ids as strings; CVAT needs a number attribute.
    c = cuboid(0, 0, [0, 0, 0], 0.0, (1, 1, 1), track_id="7")
    assert c["attributes"]["track_id"] == 7 and isinstance(c["attributes"]["track_id"], int)


def test_track_id_travels_with_keyframe_so_cvat_builds_a_track():
    # CVAT's importer files an annotation under a track only when track_id is set
    # AND "keyframe" is in the attributes (bindings.py import_dm_annotations);
    # without the pair the cuboid lands as a plain shape and the chain is lost.
    tracked = cuboid(0, 0, [0, 0, 0], 0.0, (1, 1, 1), record_token="k0:CAM_FRONT:0", track_id=7)
    assert tracked["attributes"]["keyframe"] is True
    plain = cuboid(1, 0, [0, 0, 0], 0.0, (1, 1, 1), record_token="k0:CAM_FRONT:1")
    assert "keyframe" not in plain["attributes"]


def test_review_export_writes_frames_track_ids_and_record_tokens(tmp_path):
    cfg, work = _substrate(tmp_path)
    stitch = tmp_path / "stitch_map.json"
    stitch.write_text(json.dumps({"s1:CAM_FRONT:0": "7", "s2:CAM_FRONT:1": "7",
                                  "s2:CAM_FRONT:2": "det:s2:CAM_FRONT:2"}))
    assert main(["--paths", cfg, "--taxonomy", TAXONOMY, "--skip-archive",
                 "--stitch-map", str(stitch)]) == 0

    out = work / "cvat_export_3d" / SCENE
    assert json.loads(out.joinpath("frames.json").read_text()) == [
        {"frame": i, "name": f"{i + 1:06d}", "sample_token": t, "channels": ["CAM_BACK", "CAM_FRONT"]}
        for i, t in enumerate(KEYFRAMES)]

    doc = json.loads(out.joinpath("annotations_ours.json").read_text())
    attrs = {a["attributes"]["record_token"]: a["attributes"]
             for item in doc["items"] for a in item["annotations"]}
    # Stage 9's token format, so the importer can key on it.
    assert set(attrs) == {"s1:CAM_FRONT:0", "s2:CAM_FRONT:1", "s2:CAM_FRONT:2", "s3:CAM_FRONT:0"}
    # one chain over two frames; the det: chain hashed; the unmapped row keeps Stage 7's id
    assert attrs["s1:CAM_FRONT:0"]["track_id"] == attrs["s2:CAM_FRONT:1"]["track_id"] == 7
    assert attrs["s2:CAM_FRONT:2"]["track_id"] == stable_track_int("det:s2:CAM_FRONT:2")
    assert attrs["s3:CAM_FRONT:0"]["track_id"] == 9
    track_ids = json.loads(out.joinpath("track_ids.json").read_text())
    assert track_ids["7"] == "7" and track_ids[str(stable_track_int("det:s2:CAM_FRONT:2"))] == "det:s2:CAM_FRONT:2"


def test_blank_double_export_packs_selected_frames_with_no_cuboids(tmp_path):
    cfg, work = _substrate(tmp_path)
    double = tmp_path / "double_annotation.json"
    double.write_text(json.dumps({"spec": "dhakascenes/double_annotation/v1",
                                  "selected": [{"sample_token": "s3"}, {"sample_token": "s1"}]}))
    assert main(["--paths", cfg, "--taxonomy", TAXONOMY, "--skip-archive", "--frames", str(double),
                 "--blank", "--out-subdir", "cvat_export_3d_double"]) == 0

    out = work / "cvat_export_3d_double" / SCENE
    # scene order kept, frames renumbered — which is why frames.json has to exist
    assert json.loads(out.joinpath("frames.json").read_text()) == [
        {"frame": 0, "name": "000001", "sample_token": "s1", "channels": ["CAM_BACK", "CAM_FRONT"]},
        {"frame": 1, "name": "000002", "sample_token": "s3", "channels": ["CAM_BACK", "CAM_FRONT"]}]
    blank = json.loads(out.joinpath("annotations_blank.json").read_text())
    assert [it["id"] for it in blank["items"]] == ["000001", "000002"]
    assert all(it["annotations"] == [] for it in blank["items"])
    assert blank["categories"]["label"]["labels"][0]["attributes"] == list(CUBOID_ATTRIBUTES)
    # neither our cuboids nor the answer key may reach a double-pass annotator
    assert sorted(os.listdir(out)) == ["annotations_blank.json", "frames.json"]


def test_scene_with_no_selected_keyframe_is_skipped(tmp_path):
    cfg, work = _substrate(tmp_path)
    double = tmp_path / "double_annotation.json"
    double.write_text(json.dumps({"selected": [{"sample_token": "not-in-this-scene"}]}))
    assert main(["--paths", cfg, "--taxonomy", TAXONOMY, "--skip-archive", "--frames", str(double),
                 "--blank", "--out-subdir", "cvat_export_3d_double"]) == 0
    assert not os.path.isdir(work / "cvat_export_3d_double" / SCENE)


# --- I7: a packed archive is only reusable while the selection is unchanged ---

SENTINEL = "STALE-CLOUDS-MARKER"


def _selection(path, tokens):
    path.write_text(json.dumps({"spec": "dhakascenes/double_annotation/v1",
                                "selected": [{"sample_token": t} for t in tokens]}))
    return str(path)


def _mark(zip_path):
    """Stamp the packed archive so a rebuild is visible as the mark disappearing."""
    with open(zip_path, "ab") as fh:
        fh.write(SENTINEL.encode())


def _marked(zip_path) -> bool:
    with open(zip_path, "rb") as fh:
        return SENTINEL.encode() in fh.read()


def test_archive_matches_selection_is_exact(tmp_path):
    scene_dir = tmp_path / "scene"
    scene_dir.mkdir()
    manifest = [{"frame": 0, "name": "000001", "sample_token": "s1", "channels": []},
                {"frame": 1, "name": "000002", "sample_token": "s3", "channels": []}]
    assert archive_matches_selection(str(scene_dir), manifest, {"s1", "s3"}) is False   # no task.zip
    (scene_dir / "task.zip").write_bytes(b"PK")
    assert archive_matches_selection(str(scene_dir), manifest, {"s1", "s3"}) is False   # no frames.json
    (scene_dir / "frames.json").write_text(json.dumps(manifest))
    assert archive_matches_selection(str(scene_dir), manifest, {"s1", "s3"}) is True
    # a different selection, the same count
    other = [dict(manifest[0]), {**manifest[1], "sample_token": "s2"}]
    assert archive_matches_selection(str(scene_dir), other, {"s1", "s2"}) is False
    # the same tokens in a different order are a different frame mapping
    swapped = [{**manifest[0], "sample_token": "s3"}, {**manifest[1], "sample_token": "s1"}]
    assert archive_matches_selection(str(scene_dir), swapped, {"s1", "s3"}) is False
    # frames.json that is not a subset of the selection this run was given
    assert archive_matches_selection(str(scene_dir), manifest, {"s1"}) is False
    (scene_dir / "frames.json").write_text("{ not json")
    assert archive_matches_selection(str(scene_dir), manifest, {"s1", "s3"}) is False


def test_an_unchanged_selection_reuses_the_packed_clouds(tmp_path):
    cfg, work = _substrate(tmp_path, clouds=True)
    sel = _selection(tmp_path / "double_annotation.json", ["s1", "s3"])
    args = ["--paths", cfg, "--taxonomy", TAXONOMY, "--frames", sel, "--blank",
            "--out-subdir", "cvat_export_3d_double", "--skip-archive-if-frames-match", sel]
    assert main(args) == 0
    zip_path = work / "cvat_export_3d_double" / SCENE / "task.zip"
    assert zip_path.is_file()
    _mark(zip_path)
    assert main(args) == 0
    assert _marked(zip_path)          # 45-80 MB of point cloud not re-packed


def test_a_changed_selection_rebuilds_rather_than_pairing_stale_clouds(tmp_path):
    # I7's failure: `out` is new so export_release reselects, `work` is not so the
    # old task.zip is there. frames.json is rewritten unconditionally, so keeping
    # the archive would name frame N sample Y while its cloud is keyframe X, and
    # every A/B box would import against the wrong sample_token.
    cfg, work = _substrate(tmp_path, clouds=True)
    first = _selection(tmp_path / "double_a.json", ["s1", "s3"])
    assert main(["--paths", cfg, "--taxonomy", TAXONOMY, "--frames", first, "--blank",
                 "--out-subdir", "cvat_export_3d_double", "--skip-archive-if-frames-match", first]) == 0
    scene_dir = work / "cvat_export_3d_double" / SCENE
    _mark(scene_dir / "task.zip")
    second = _selection(tmp_path / "double_b.json", ["s1", "s2"])
    assert main(["--paths", cfg, "--taxonomy", TAXONOMY, "--frames", second, "--blank",
                 "--out-subdir", "cvat_export_3d_double", "--skip-archive-if-frames-match", second]) == 0
    assert not _marked(scene_dir / "task.zip")
    assert [r["sample_token"] for r in json.loads((scene_dir / "frames.json").read_text())] == ["s1", "s2"]
    with zipfile.ZipFile(scene_dir / "task.zip") as zf:
        assert sorted(n for n in zf.namelist() if n.endswith(".pcd")) == [
            "pointcloud/000001.pcd", "pointcloud/000002.pcd"]
