"""Tests for scripts/check_release.py — the benchmark checklist as a validator (spec §8).

Every test builds a minimal, valid export in tmp_path with `_export()` and then
breaks exactly one thing, so a failure names the rule that broke. Errors are the
checklist's hard failures (exit 2); warnings are the things a reader of
DELIVERY_NOTE.md has to be told about (exit 1).

Run (cwd anywhere; the test uses tmp_path):
    /home/mt/miniconda3/envs/ano_pipe/bin/python -m pytest tests/test_check_release.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)   # `tests` is not an importable package here (site-packages has one)

from scripts.check_release import check_release  # noqa: E402

V = "v1.0-x"


def _export(tmp_path, anns=None, samples=None, meta=None, instances=None, note=True, double=None):
    d = tmp_path / V
    d.mkdir(exist_ok=True)
    cat = [{"token": "c1", "name": "car"}, {"token": "c2", "name": "pedestrian"}]
    inst = instances if instances is not None else [
        {"token": "i1", "category_token": "c1", "nbr_annotations": 2, "first_annotation_token": "a1", "last_annotation_token": "a2"}]
    base_anns = [
        {"token": "a1", "sample_token": "s1", "instance_token": "i1", "visibility_token": "4", "attribute_tokens": ["t1"],
         "translation": [1, 2, 0], "size": [1.8, 4.5, 1.6], "rotation": [1, 0, 0, 0], "prev": "", "next": "a2",
         "num_lidar_pts": 5, "num_radar_pts": 0, "dhakascenes_source": "pipeline", "dhakascenes_tier": "auto_accept",
         "dhakascenes_interpolated": False},
        {"token": "a2", "sample_token": "s2", "instance_token": "i1", "visibility_token": "4", "attribute_tokens": ["t1"],
         "translation": [2, 2, 0], "size": [1.8, 4.5, 1.6], "rotation": [1, 0, 0, 0], "prev": "a1", "next": "",
         "num_lidar_pts": 5, "num_radar_pts": 0, "dhakascenes_source": "pipeline", "dhakascenes_tier": "auto_accept",
         "dhakascenes_interpolated": False},
    ]
    tables = {
        "sample_annotation": anns if anns is not None else base_anns, "instance": inst, "category": cat,
        "attribute": [{"token": "t1", "name": "vehicle.moving"}], "visibility": [{"token": "4", "level": "v80-100"}],
        "sample": samples if samples is not None else [{"token": "s1", "dbench_double_annotated": True}, {"token": "s2", "dbench_double_annotated": False}],
    }
    for k, v in tables.items():
        (d / f"{k}.json").write_text(json.dumps(v))
    (tmp_path / "release_meta.json").write_text(json.dumps(meta or {"tiers_admitted": "auto_accept", "human": {"enabled": False}}))
    if double is not None:
        (tmp_path / "double_annotation.json").write_text(json.dumps(double))
    if note:
        (tmp_path / "DELIVERY_NOTE.md").write_text("# x")
    return str(tmp_path)


def _anns(root):
    with open(os.path.join(root, V, "sample_annotation.json")) as fh:
        return json.load(fh)


def test_clean_export_has_no_errors(tmp_path):
    rep = check_release(_export(tmp_path))
    assert rep.errors == []


def test_bad_size_quaternion_and_dangling(tmp_path):
    a = _anns(_export(tmp_path))
    a[0]["size"] = [1.8, -1.0, 1.6]
    a[0]["rotation"] = [1, 1, 0, 0]
    a[1]["instance_token"] = "nope"
    rep = check_release(_export(tmp_path, anns=a))
    assert any("size" in e for e in rep.errors) and any("quaternion" in e for e in rep.errors)
    assert any("instance_token" in e for e in rep.errors)


def test_chain_and_count_errors(tmp_path):
    a = _anns(_export(tmp_path))
    a[1]["prev"] = ""
    rep = check_release(_export(tmp_path, anns=a))
    assert any("prev/next" in e for e in rep.errors)
    inst = [{"token": "i1", "category_token": "c1", "nbr_annotations": 3, "first_annotation_token": "a1", "last_annotation_token": "a2"}]
    rep = check_release(_export(tmp_path, instances=inst))
    assert any("nbr_annotations" in e for e in rep.errors)


def test_tier_and_human_rules(tmp_path):
    a = _anns(_export(tmp_path))
    a[0]["dhakascenes_tier"] = "flagged"
    a[1].update(dhakascenes_source="human_created", dhakascenes_tier=None)
    rep = check_release(_export(tmp_path, anns=a))
    assert any("tier" in e for e in rep.errors) and any("verified_by" in e for e in rep.errors)
    # `--tiers all` is a legitimate export mode: the same rows are then legal.
    meta = {"tiers_admitted": "all", "human": {"enabled": False}}
    rep = check_release(_export(tmp_path, anns=a, meta=meta))
    assert not any("tier" in e for e in rep.errors)


def test_double_fraction_and_half_import(tmp_path):
    root = _export(tmp_path, double={"fraction": 0.5, "n_keyframes": 2, "n_selected": 1, "selected": [{"sample_token": "s1"}]})
    assert check_release(root).errors == []
    root = _export(tmp_path, samples=[{"token": "s1", "dbench_double_annotated": False}, {"token": "s2", "dbench_double_annotated": False}],
                   double={"fraction": 0.5, "n_keyframes": 2, "n_selected": 1, "selected": [{"sample_token": "s1"}]})
    assert any("double" in e.lower() for e in check_release(root).errors)


def test_double_flagged_sample_with_one_pass_only(tmp_path):
    # The note claims A/B rows were imported, but s1 carries pass A rows alone:
    # a half-imported double frame (the B task was never imported).
    a = _anns(_export(tmp_path))
    a[0]["annotator_pass"] = "A"
    meta = {"tiers_admitted": "auto_accept", "human": {"enabled": True, "stats": {"n_rows_double_A": 3, "n_rows_double_B": 0}}}
    rep = check_release(_export(tmp_path, anns=a, meta=meta,
                               double={"fraction": 0.5, "n_keyframes": 2, "n_selected": 1, "selected": [{"sample_token": "s1"}]}))
    assert any("s1" in e and "pass" in e for e in rep.errors)


def test_warnings(tmp_path):
    a = _anns(_export(tmp_path, note=False))
    a[0]["size"] = [4.5, 1.8, 1.6]
    a[1]["attribute_tokens"] = []
    a[1]["annotator_pass"] = "B"
    rep = check_release(_export(tmp_path, anns=a, note=False))
    joined = "\n".join(rep.warnings)
    assert "w > l" in joined and "without attribute" in joined and "pass B" in joined and "DELIVERY_NOTE" in joined
    assert "pedestrian" in joined     # zero-instance class


def test_interpolated_row_on_a_human_sample_warns(tmp_path):
    # A stitched interpolation that outlived a human-superseded endpoint: the
    # human owns this keyframe, but a machine-interpolated box still sits on it.
    a = _anns(_export(tmp_path))
    a[0].update(dhakascenes_source="human_verified", dhakascenes_tier=None, dhakascenes_verified_by="ann_a")
    a.append({**a[0], "token": "a3", "instance_token": "i2", "prev": "", "next": "",
              "dhakascenes_source": "pipeline", "dhakascenes_tier": "auto_accept",
              "dhakascenes_verified_by": None, "dhakascenes_interpolated": True})
    inst = [{"token": "i1", "category_token": "c1", "nbr_annotations": 2, "first_annotation_token": "a1", "last_annotation_token": "a2"},
            {"token": "i2", "category_token": "c1", "nbr_annotations": 1, "first_annotation_token": "a3", "last_annotation_token": "a3"}]
    rep = check_release(_export(tmp_path, anns=a, instances=inst))
    assert rep.errors == []
    hits = [w for w in rep.warnings if "interpolated" in w and "human" in w]
    assert hits and "a3" in hits[0]
    # ... and not on an all-pipeline sample (the clean export interpolates nothing).
    assert not [w for w in check_release(_export(tmp_path)).warnings if "interpolated" in w and "human" in w]


def test_benchmark_source_digest_mismatch_warns(tmp_path):
    bench = tmp_path / "benchmark_v1.0.yaml"
    bench.write_text("classes: [car]\n")
    digest = hashlib.sha256(bench.read_bytes()).hexdigest()
    meta = {"tiers_admitted": "auto_accept", "human": {"enabled": False},
            "release_config": {"sha256": "cfg", "benchmark_source": {"path": str(bench), "sha256": "0" * 64}}}
    assert any("benchmark" in w for w in check_release(_export(tmp_path, meta=meta)).warnings)
    meta["release_config"]["benchmark_source"]["sha256"] = digest
    assert not any("benchmark" in w for w in check_release(_export(tmp_path, meta=meta)).warnings)
    # Validated on another machine: the benchmark file is simply not there.
    meta["release_config"]["benchmark_source"]["path"] = str(tmp_path / "gone.yaml")
    meta["release_config"]["benchmark_source"]["sha256"] = "0" * 64
    assert not any("benchmark" in w for w in check_release(_export(tmp_path, meta=meta)).warnings)


def test_main_exit_codes_and_output(tmp_path, capsys):
    from scripts.check_release import main

    assert main([_export(tmp_path)]) == 1                      # zero-instance class -> warning
    assert "WARNING" in capsys.readouterr().out
    a = _anns(_export(tmp_path))
    a[0]["size"] = [0.0, 4.5, 1.6]
    assert main([_export(tmp_path, anns=a), "--version", V]) == 2
    assert "ERROR" in capsys.readouterr().out


def test_export_release_cli_runs_the_checker(tmp_path, capsys):
    from scripts import export_release as er
    from test_export_release import VERSION, build_dataroot, write_prelabels

    src = str(tmp_path / "src")
    info = build_dataroot(src)
    pre = str(tmp_path / "prelabels.jsonl")
    write_prelabels(pre, info["sample_tokens"])
    out = str(tmp_path / "rel")
    rc = er.main(["--prelabels", pre, "--dataroot", src, "--version", VERSION, "--out", out])
    text = capsys.readouterr().out
    assert "check_release:" in text and "0 error(s)" in text
    assert rc == 0
