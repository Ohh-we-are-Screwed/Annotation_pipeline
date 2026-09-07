"""Tests for pipeline/release/note.py — the generated DELIVERY_NOTE.md (spec §8).

The note is the per-chunk text block the benchmark's checklist asks for:
annotation rule, range, tiers, classes, identity, attributes, uncertainty,
double annotation, anonymisation, extra layers. Every number comes from
`release_meta.json`; only the fixed sentences quoted in spec §8 are typed.

Run (cwd anywhere; the test uses tmp_path):
    /home/mt/miniconda3/envs/ano_pipe/bin/python -m pytest tests/test_release_note.py
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)   # `tests` is not an importable package here (site-packages has one)

from pipeline.release.note import render_note, write_note  # noqa: E402


def _meta():
    return {
        "version": "v1.0-dhaka-fixed", "created_utc": "2026-09-08T00:00:00Z", "git_sha": "abc123",
        "pipeline_version": "dhakascenes-pilot/stage9_qa/v1", "tiers_admitted": "auto_accept",
        "counts": {"n_annotations": 10, "n_instances": 4, "n_scenes": 1, "per_class": {"car": 3, "pedestrian": 1}},
        "excluded": {"table": "sample_annotation_excluded.json", "n": 6, "by_reason": {"tier_rejected": 4, "tier_flagged": 2}},
        "stitch": {"enabled": True, "totals": {"n_fragments": 9, "n_chains": 4, "n_interpolated": 1, "joins_by_gap": {"1": 4, "3": 1}},
                   "per_scene": {"chunk_0000": {"median_len_before": 1.0, "median_len_after": 2.5}}, "config": {"max_gap_keyframes": 3}},
        "attributes": {"enabled": True, "threshold_mps": 0.5, "n_with_attribute": 8, "by_name": {"vehicle.moving": 8}, "by_basis": {"chain_velocity": 8}},
        "range": {"cap_m": 50.0, "max_exported_range_m": 47.3,
                  "effective_p99_m_by_class": {"car": 41.2, "pedestrian": 22.0}},
        "classes": {"present": {"car": 3, "pedestrian": 1}, "absent_on_route": ["barrier", "traffic_cone", "construction_element"],
                    "not_producible": ["animal", "battery_rickshaw"], "producible_by_vocabulary": ["car", "pedestrian", "barrier"]},
        "human": {"enabled": False},
        "double_annotation": {"file": "double_annotation.json", "reused": False, "n_selected": 35,
                              "cells": [{"density": "High", "illumination": "day", "n": 300, "selected": 20}]},
        "strata": {"density_bin_edges": [0.001, 0.002, 0.003], "illumination_bin_edges": [45, 75, 95], "n_illumination_unknown": 0},
        "release_config": {"sha256": "cfgsha", "benchmark_source": {"path": "/x/benchmark_v1.0.yaml", "sha256": "bsha"}},
        "visibility": {"basis": "camera_fov_corner_fraction"},
        "num_lidar_pts_basis": ["single_sweep_ground_filtered_pre_inflation"],
        "mapper": {"used": {"a car": "car", "a pedestrian": "pedestrian"}},
    }


S9 = {"spec": "dhakascenes-pilot/stage9_qa/v1", "config": {"min_lidar_returns": 5, "conf_gate": 0.5, "spatial_multiplier": 2.0}}


def test_note_states_the_rule_range_classes_and_anonymisation():
    text = render_note(_meta(), S9, None, {"fraction": 0.05, "n_keyframes": 684, "seed": 20260812}, "/work/chunk_0000", "day1_chunk_0000")
    assert ">= 5 LiDAR returns" in text and "confidence >= 0.5" in text and "2.0x class prior" in text
    assert "V does not apply" in text
    assert "50" in text and "41.2" in text
    assert "not producible" in text.lower() and "battery_rickshaw" in text
    assert "absent on this route" in text.lower() and "traffic_cone" in text
    assert "after annotation" in text and "un-blurred" in text
    assert "35" in text and "20260812" in text
    assert "no human pass" in text.lower()
    assert "abc123" in text and "bsha" in text


def test_note_separates_density_radius_from_the_annotation_cap():
    # The rho radius (30 m) and the annotation range cap (50 m) used to coincide;
    # the cap moved to 50 m and the radius did not, so the note has to say so.
    text = render_note(_meta(), S9, None, None, None, "day1_chunk_0000")
    rng = text.split("## Range", 1)[1].split("## Tiers", 1)[0]
    assert "_RHO_RADIUS_M" in rng and "stratification.density.radius_m" in rng
    assert "30 m disc inside a 50 m" in rng


def test_note_does_not_call_the_observed_max_range_a_cap():
    # release_meta's `range.max_exported_range_m` is the furthest exported box,
    # not the cap (`range.cap_m`) — the note must not relabel it.
    meta = _meta()
    meta["range"]["max_exported_range_m"] = 20.2
    rng = render_note(meta, S9, None, None, None, "x").split("## Range", 1)[1].split("## Tiers", 1)[0]
    assert "pipeline range cap: 50 m" in rng
    assert "20.2 m" in rng and "observed" in rng


def test_note_reports_human_pass_when_present():
    meta = _meta()
    meta["human"] = {"enabled": True, "stats": {"n_rows_review": 12, "n_rows_double_A": 3, "n_rows_double_B": 4,
                                                 "n_samples_review": 5, "n_samples_double_A": 1, "n_samples_double_B": 1},
                     "half_imported_samples": ["s9"]}
    imp = {"tasks": [{"task_id": 41, "kind": "review", "scene": "chunk_0000", "assignee": "ann_a"}]}
    text = render_note(meta, S9, imp, None, None, "day1_chunk_0000")
    assert "ann_a" in text and "task 41" in text and "s9" in text


def test_write_note_writes_file(tmp_path):
    (tmp_path / "release_meta.json").write_text(json.dumps(_meta()))
    p = write_note(str(tmp_path), stage9_manifest_path=None, import_manifest_path=None, stage_tree=None, chunk_name="x")
    assert os.path.isfile(p) and open(p).read().startswith("# ")


# --- CLI wiring -----------------------------------------------------------


def test_export_release_cli_writes_the_note(tmp_path, capsys):
    from scripts import export_release as er
    from test_export_release import VERSION, build_dataroot, write_prelabels

    src = str(tmp_path / "src")
    info = build_dataroot(src)
    pre = str(tmp_path / "prelabels.jsonl")
    write_prelabels(pre, info["sample_tokens"])
    man = tmp_path / "run_manifest.json"
    man.write_text(json.dumps(S9))
    out = str(tmp_path / "day1_chunk_0003" / "release")

    rc = er.main(["--prelabels", pre, "--dataroot", src, "--version", VERSION, "--out", out,
                  "--run-manifest", str(man), "--stage-tree", "/work/day1_chunk_0003"])
    assert rc == 0
    note = os.path.join(out, "DELIVERY_NOTE.md")
    assert os.path.isfile(note)
    text = open(note).read()
    assert text.startswith("# day1_chunk_0003 ")          # chunk name defaults to --out's parent
    assert ">= 5 LiDAR returns" in text                    # Stage 9 manifest was resolved
    assert "/work/day1_chunk_0003" in text
    for section in ("## Provenance", "## Human pass", "## Annotation rule", "## Range", "## Tiers",
                    "## Classes", "## Identity (stitching)", "## Attributes", "## Uncertainty",
                    "## Double annotation", "## Anonymisation", "## Extra layers", "## Files"):
        assert section in text, section
    assert note in capsys.readouterr().out


def test_export_release_cli_no_note(tmp_path):
    from scripts import export_release as er
    from test_export_release import VERSION, build_dataroot, write_prelabels

    src = str(tmp_path / "src")
    info = build_dataroot(src)
    pre = str(tmp_path / "prelabels.jsonl")
    write_prelabels(pre, info["sample_tokens"])
    out = str(tmp_path / "rel")
    assert er.main(["--prelabels", pre, "--dataroot", src, "--version", VERSION, "--out", out,
                    "--no-note"]) == 0
    assert not os.path.exists(os.path.join(out, "DELIVERY_NOTE.md"))


# --- C2(c) / I4: what the note must say about interpolated rows and CVAT ------


def test_the_rule_sentence_is_true_of_interpolated_rows(tmp_path):
    meta = _meta()
    meta["stitch"]["interpolated_point_floor"] = {"value": 5, "source": "stage9_run_manifest"}
    rule = render_note(meta, S9, None, None, None, "x").split("## Annotation rule", 1)[1].split("## Range", 1)[0]
    # the measured-box rule keeps all three clauses ...
    assert ">= 5 LiDAR returns" in rule and "confidence >= 0.5" in rule and "2.0x class prior" in rule
    # ... and the note says which of them an interpolated row does NOT satisfy,
    # how to find one, and that its point count still clears the floor.
    assert "dhakascenes_interpolated" in rule
    assert "interpolated_below_point_floor" in rule
    assert "inherited_from_endpoints" in rule
    low = rule.lower()
    assert "confidence and" in low or "confidence nor" in low or "do not apply" in low


def test_the_floor_falls_back_to_the_stage9_manifest(tmp_path):
    # An export whose release_meta predates the floor still prints a number.
    rule = render_note(_meta(), S9, None, None, None, "x").split("## Annotation rule", 1)[1]
    assert "?" not in rule.split("## Range", 1)[0]
