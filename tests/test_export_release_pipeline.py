"""Tests for the post-processing pipeline scripts/export_release.py orchestrates.

Stitch -> human merge -> tier filter -> instance/chain rebuild -> attributes ->
strata/double selection -> tables + sidecars + meta (spec §2, §5, §6). The
synthetic root comes from tests/test_export_release.py, grown to 5 keyframes so
a gap can be interpolated and a chain velocity is well defined.

Run (cwd anywhere; the test uses tmp_path):
    /home/mt/miniconda3/envs/ano_pipe/bin/python -m pytest tests/test_export_release_pipeline.py
"""

from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)   # `tests` is not an importable package here (site-packages has one)

import numpy as np  # noqa: E402

from pipeline.common.eval_region import _R_MAX_M  # noqa: E402
from scripts import export_release as er  # noqa: E402
from test_export_release import VERSION, _record, build_dataroot  # noqa: E402

MAPPER = os.path.join(ROOT, "configs", "release_category_map.yaml")
SCENE_NAME = "synthetic-0001"
# The car track sits at this ego position in every keyframe (see _prelabels), so
# the interpolated row at sample 3 lands here too.
CAR_EGO = (10.0, 0.5, 0.8)


def _stage1_tree(root: str, scene: str, sample_tokens, n_points: int = 8) -> str:
    """A Stage 1 output tree: keyframes.jsonl + the ground-filtered single sweeps.

    This is what `CloudSource` reads first (spec §3.5 / C1): on a fresh work root
    the CVAT 3D archive does not exist yet, because the release export runs
    before the 3D publish.
    """
    stage1 = os.path.join(root, "stage1_ingestion")
    scene_dir = os.path.join(stage1, "scenes", scene)
    cloud_dir = os.path.join(stage1, "clouds", scene, "single_sweep")
    os.makedirs(scene_dir)
    os.makedirs(cloud_dir)
    lines = []
    for tok in sample_tokens:
        path = os.path.join(cloud_dir, f"{tok}.pcd.bin")
        pts = np.zeros((n_points, 5), np.float32)
        pts[:, 0] = CAR_EGO[0] + np.linspace(-0.2, 0.2, n_points)   # inside the 4.5 m box
        pts[:, 1] = CAR_EGO[1]
        pts[:, 2] = CAR_EGO[2]
        pts.tofile(path)
        lines.append(json.dumps({"keyframe_token": tok, "t_ns": 0,
                                 "single_sweep_cloud": {"cloud_kind": "single_sweep", "frame": "ego",
                                                        "path": path, "point_record_bytes": 20}}))
    with open(os.path.join(scene_dir, "keyframes.jsonl"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return stage1


def _write(path, recs):
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


def _prelabels(s):
    # car track 7 at samples 1,2 then track 8 at sample 4 (gap of 2 -> one interpolated row at sample 3);
    # the ego translates 5 m per keyframe from sample 1 on, so a box fixed in the ego frame moves at a
    # constant 12.5 m/s in global and the forward prediction lands exactly on track 8.
    # A rejected pedestrian and a flagged bus at sample 0 exercise the sidecar.
    recs = [
        _record("k1:CAM_FRONT:0", s[1], "a car", [10.0, 0.5, 0.8], [1.9, 4.5, 1.6], 0.0, "7", 120),
        _record("k2:CAM_FRONT:0", s[2], "a car", [10.0, 0.5, 0.8], [1.9, 4.5, 1.6], 0.0, "7", 110),
        _record("k4:CAM_FRONT:0", s[4], "a car", [10.0, 0.5, 0.8], [1.9, 4.5, 1.6], 0.0, "8", 100),
        _record("k0:CAM_FRONT:3", s[0], "a pedestrian", [-8.0, 1.0, 0.9], [0.6, 0.7, 1.7], 2.0, None, 9),
        _record("k0:CAM_FRONT:4", s[0], "a bus", [20.0, 3.0, 1.5], [2.5, 11.0, 3.2], 0.0, "9", 40),
    ]
    recs[3]["provenance"]["tier"] = "rejected"
    recs[4]["provenance"]["tier"] = "flagged"
    return recs


@pytest.fixture
def exported(tmp_path):
    src = str(tmp_path / "src")
    info = build_dataroot(src, n_samples=5)
    pre = str(tmp_path / "prelabels.jsonl")
    _write(pre, _prelabels(info["sample_tokens"]))
    out = str(tmp_path / "release")
    stage1 = _stage1_tree(str(tmp_path / "work"), SCENE_NAME, info["sample_tokens"])
    res = er.export_release(pre, src, VERSION, out, MAPPER, double_fraction=0.5, stage1_dir=stage1)
    return {"src": src, "out": out, "res": res, "stage1": stage1, **info}


def _load(out, name):
    with open(os.path.join(out, VERSION, f"{name}.json")) as fh:
        return json.load(fh)


def test_accepted_only_with_sidecar(exported):
    anns = _load(exported["out"], "sample_annotation")
    assert all(a["dhakascenes_tier"] == "auto_accept" for a in anns)
    exc = json.load(open(os.path.join(exported["out"], er.EXCLUDED_TABLE)))
    assert sorted(e["dhakascenes_excluded_reason"] for e in exc) == ["tier_flagged", "tier_rejected"]
    assert all(len(e["instance_token"]) == 32 for e in exc)


def test_stitched_chain_and_interpolated_row(exported):
    anns = _load(exported["out"], "sample_annotation")
    inst = _load(exported["out"], "instance")
    car = [a for a in anns if not a["dhakascenes_record_token"].endswith(":3")]
    assert len(car) == 4 and len({a["instance_token"] for a in car}) == 1
    assert len(inst) == 1 and inst[0]["nbr_annotations"] == 4
    interp = [a for a in car if a["dhakascenes_interpolated"]]
    assert len(interp) == 1 and interp[0]["sample_token"] == exported["sample_tokens"][3]
    assert interp[0]["dhakascenes_record_token"].split(":")[1] == "INTERP"
    assert {a["dhakascenes_track_id_pre_stitch"] for a in car} == {"7", "8", None}
    ordered = sorted(car, key=lambda a: a["sample_token"] and exported["sample_tokens"].index(a["sample_token"]))
    for a, b in zip(ordered, ordered[1:]):
        assert a["next"] == b["token"] and b["prev"] == a["token"]
    stitch_map = json.load(open(os.path.join(exported["out"], er.STITCH_MAP)))
    assert stitch_map["k4:CAM_FRONT:0"] == "7" and "k0:CAM_FRONT:3" in stitch_map


def test_attributes_from_chain_velocity(exported):
    anns = _load(exported["out"], "sample_annotation")
    attrs = {a["token"]: a["name"] for a in _load(exported["out"], "attribute")}
    for a in anns:
        assert [attrs[t] for t in a["attribute_tokens"]] == ["vehicle.moving"]   # 12.5 m/s in global (see _prelabels)
        assert a["dhakascenes_attribute_basis"] == "chain_velocity"
        assert a["is_uncertain"] is False and a["is_uncertain_reason"] == ""


def test_double_selection_flags_and_meta(exported):
    samples = _load(exported["out"], "sample")
    flagged = [s["token"] for s in samples if s["dbench_double_annotated"]]
    doc = json.load(open(os.path.join(exported["out"], er.DOUBLE_FILE)))
    assert sorted(flagged) == sorted(s["sample_token"] for s in doc["selected"])
    assert doc["n_selected"] >= 2
    meta = json.load(open(os.path.join(exported["out"], "release_meta.json")))
    assert meta["tiers_admitted"] == "auto_accept"
    assert meta["stitch"]["totals"]["n_interpolated"] == 1
    assert meta["excluded"]["by_reason"] == {"tier_flagged": 1, "tier_rejected": 1}
    assert meta["classes"]["present"]["car"] == 1
    assert "battery_rickshaw" in meta["classes"]["not_producible"]
    # `cap_m` is the pipeline's cap; the furthest box that shipped is its own key
    # (they shared `cap_m` until 2026-09-07 and read as one number).
    assert meta["range"]["cap_m"] == _R_MAX_M
    assert 0 < meta["range"]["max_exported_range_m"] <= _R_MAX_M
    assert meta["release_config"]["sha256"]


def test_overwrite_tables_reuses_selection_and_keeps_blobs(exported, tmp_path):
    out = exported["out"]
    before = json.load(open(os.path.join(out, er.DOUBLE_FILE)))["selected"]
    pre = str(tmp_path / "prelabels.jsonl")   # same records, re-export in place
    er.export_release(pre, exported["src"], VERSION, out, MAPPER, overwrite_tables=True, double_fraction=0.5)
    after = json.load(open(os.path.join(out, er.DOUBLE_FILE)))["selected"]
    assert after == before
    assert os.path.exists(os.path.join(out, "samples"))


def test_tiers_all_reproduces_legacy(tmp_path):
    src = str(tmp_path / "src")
    info = build_dataroot(src, n_samples=5)
    pre = str(tmp_path / "prelabels.jsonl")
    _write(pre, _prelabels(info["sample_tokens"]))
    res = er.export_release(pre, src, VERSION, str(tmp_path / "rel"), MAPPER, tiers="all", stitch=False,
                            attributes=False, double_fraction=0.0)
    anns = res.tables["sample_annotation"]
    assert len(anns) == 5 and res.excluded == []
    assert all(a["attribute_tokens"] == [] for a in anns)
    assert len(res.tables["instance"]) == 4          # 7, 8, 9, and the untracked pedestrian


# --- C1: the interpolated row's points come from Stage 1, not the raw sweep ---


def test_interpolated_point_count_reads_the_stage1_ground_filtered_cloud(exported):
    anns = _load(exported["out"], "sample_annotation")
    interp = [a for a in anns if a["dhakascenes_interpolated"]]
    assert len(interp) == 1
    # 8 points were written inside the box; the raw dataroot sweep holds one
    # point at the origin, so a raw-basis count would be 0.
    assert interp[0]["num_lidar_pts"] == 8
    assert interp[0]["num_lidar_pts_basis"] == "single_sweep_ground_filtered_pre_inflation"
    meta = json.load(open(os.path.join(exported["out"], "release_meta.json")))
    assert meta["stitch"]["totals"]["n_interpolated_raw_basis"] == 0
    assert meta["stitch"]["totals"]["n_interpolated_no_cloud"] == 0
    assert meta["num_lidar_pts_basis"] == ["single_sweep_ground_filtered_pre_inflation"]
    assert meta["source"]["stage1_dir"] == os.path.abspath(exported["stage1"])


def test_without_stage1_the_interpolated_row_degrades_to_the_raw_sweep(tmp_path):
    # What the chain did before C1: no Stage 1 clouds and no CVAT archive, so
    # the count comes from the ground-INCLUDED dataroot sweep and says so.
    src = str(tmp_path / "src")
    info = build_dataroot(src, n_samples=5)
    pre = str(tmp_path / "prelabels.jsonl")
    _write(pre, _prelabels(info["sample_tokens"]))
    res = er.export_release(pre, src, VERSION, str(tmp_path / "rel"), MAPPER, double_fraction=0.0)
    # 0 points against the raw sweep, so C2's floor sends it to the sidecar; the
    # basis it was counted on is still recorded on the row.
    interp = [a for a in res.excluded if a["dhakascenes_interpolated"]]
    assert len(interp) == 1 and interp[0]["num_lidar_pts_basis"] == "single_sweep_raw"
    assert res.meta["stitch"]["totals"]["n_interpolated_raw_basis"] == 1
    assert res.meta["source"]["stage1_dir"] is None


def test_cli_derives_stage1_dir_from_the_stage_tree(tmp_path, capsys):
    src = str(tmp_path / "src")
    info = build_dataroot(src, n_samples=5)
    pre = str(tmp_path / "prelabels.jsonl")
    _write(pre, _prelabels(info["sample_tokens"]))
    work = str(tmp_path / "work")
    _stage1_tree(work, SCENE_NAME, info["sample_tokens"])
    out = str(tmp_path / "rel")
    assert er.main(["--prelabels", pre, "--dataroot", src, "--version", VERSION, "--out", out,
                    "--stage-tree", work, "--no-note"]) == 0
    meta = json.load(open(os.path.join(out, "release_meta.json")))
    assert meta["source"]["stage1_dir"] == os.path.join(work, "stage1_ingestion")
    assert meta["stitch"]["totals"]["n_interpolated_raw_basis"] == 0


# --- C2: an interpolated row ships only with the LiDAR evidence the note claims


def _manifest(tmp_path, min_lidar_returns):
    p = tmp_path / "run_manifest.json"
    p.write_text(json.dumps({"spec": "dhakascenes-pilot/stage9_qa/v1",
                             "config": {"min_lidar_returns": min_lidar_returns,
                                        "conf_gate": 0.5, "spatial_multiplier": 2.0}}))
    return str(p)


def test_an_interpolated_row_with_no_evidence_goes_to_the_sidecar(tmp_path):
    src = str(tmp_path / "src")
    info = build_dataroot(src, n_samples=5)
    pre = str(tmp_path / "prelabels.jsonl")
    _write(pre, _prelabels(info["sample_tokens"]))
    # no Stage 1 tree: the only cloud is the dataroot's 1-point raw sweep
    res = er.export_release(pre, src, VERSION, str(tmp_path / "rel"), MAPPER, double_fraction=0.0)
    assert not [a for a in res.tables["sample_annotation"] if a["dhakascenes_interpolated"]]
    exc = [a for a in res.excluded if a["dhakascenes_interpolated"]]
    assert len(exc) == 1 and exc[0]["dhakascenes_excluded_reason"] == "interpolated_below_point_floor"
    assert exc[0]["num_lidar_pts"] == 0
    meta = res.meta
    assert meta["excluded"]["by_reason"]["interpolated_below_point_floor"] == 1
    assert meta["stitch"]["interpolated_point_floor"]["value"] == 5
    assert meta["stitch"]["interpolated_point_floor"]["source"] == "release_config"
    # the chain itself survives the loss: identity is the join, not the fill
    assert len(res.tables["instance"]) == len(
        {a["instance_token"] for a in res.tables["sample_annotation"]})
    car = [a for a in res.tables["sample_annotation"]
           if not a["dhakascenes_record_token"].endswith(":3")]
    assert len({a["instance_token"] for a in car}) == 1 and len(car) == 3


def test_the_floor_comes_from_the_stage9_run_manifest_when_there_is_one(tmp_path):
    src = str(tmp_path / "src")
    info = build_dataroot(src, n_samples=5)
    pre = str(tmp_path / "prelabels.jsonl")
    _write(pre, _prelabels(info["sample_tokens"]))
    stage1 = _stage1_tree(str(tmp_path / "work"), SCENE_NAME, info["sample_tokens"])   # 8 points
    res = er.export_release(pre, src, VERSION, str(tmp_path / "rel"), MAPPER, double_fraction=0.0,
                            stage1_dir=stage1, run_manifest_path=_manifest(tmp_path, 9))
    assert res.meta["stitch"]["interpolated_point_floor"] == {
        "value": 9, "source": "stage9_run_manifest", "field": "config.min_lidar_returns"}
    assert not [a for a in res.tables["sample_annotation"] if a["dhakascenes_interpolated"]]
    res2 = er.export_release(pre, src, VERSION, str(tmp_path / "rel2"), MAPPER, double_fraction=0.0,
                             stage1_dir=stage1, run_manifest_path=_manifest(tmp_path, 8))
    assert res2.meta["stitch"]["interpolated_point_floor"]["value"] == 8
    assert len([a for a in res2.tables["sample_annotation"] if a["dhakascenes_interpolated"]]) == 1


def test_the_checker_agrees_with_the_note_on_the_shipped_table(tmp_path):
    from scripts.check_release import check_release

    src = str(tmp_path / "src")
    info = build_dataroot(src, n_samples=5)
    pre = str(tmp_path / "prelabels.jsonl")
    _write(pre, _prelabels(info["sample_tokens"]))
    stage1 = _stage1_tree(str(tmp_path / "work"), SCENE_NAME, info["sample_tokens"])
    out = str(tmp_path / "rel")
    er.export_release(pre, src, VERSION, out, MAPPER, double_fraction=0.0, stage1_dir=stage1)
    assert check_release(out, VERSION).errors == []
