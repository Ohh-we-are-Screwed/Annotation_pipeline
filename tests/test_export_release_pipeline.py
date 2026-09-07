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

from scripts import export_release as er  # noqa: E402
from test_export_release import VERSION, _record, build_dataroot  # noqa: E402

MAPPER = os.path.join(ROOT, "configs", "release_category_map.yaml")


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
    res = er.export_release(pre, src, VERSION, out, MAPPER, double_fraction=0.5)
    return {"src": src, "out": out, "res": res, **info}


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
