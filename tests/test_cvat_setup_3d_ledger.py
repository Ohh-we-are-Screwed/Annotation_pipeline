from __future__ import annotations

import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import scripts.cvat_setup_3d as setup_3d  # noqa: E402
from scripts.cvat_setup_3d import (  # noqa: E402
    ATTRIBUTE_VALUES, cuboid_attribute_specs, double_project_names, find_user_id, label_spec, ledger_append,
    ledger_drop_tasks, ledger_load, main,
)


def test_label_spec_declares_the_five_attributes():
    spec = label_spec(["a car"])
    names = [a["name"] for a in spec[0]["attributes"]]
    assert names == ["record_token", "track_id", "attribute", "uncertain", "uncertain_reason"]
    assert spec[0]["type"] == "cuboid"
    sel = next(a for a in spec[0]["attributes"] if a["name"] == "attribute")
    assert sel["input_type"] == "select" and sel["values"] == ATTRIBUTE_VALUES and sel["values"][0] == ""


def test_every_attribute_is_mutable_so_it_is_stored_per_shape():
    """CVAT stores an IMMUTABLE attribute once per TRACK and a mutable one per
    SHAPE (docs/evidence/2026-09-08-cvat-3d-roundtrip.md §3). The exporter
    writes a different `record_token` on every frame of a track — the Stage 9
    token of the pre-label that frame's box came from — so an immutable
    declaration would collapse them all to the track's first value and the
    importer could no longer tell which pre-label a reviewer corrected. A label
    schema is written ONCE, at project creation, so this cannot be fixed later.
    """
    assert [a["mutable"] for a in cuboid_attribute_specs()] == [True] * 5


def test_double_project_names():
    assert double_project_names("day1_chunk_0000 (3D)") == ("day1_chunk_0000 (3D) — double pass A", "day1_chunk_0000 (3D) — double pass B")


def test_ledger_appends_and_loads(tmp_path):
    p = tmp_path / "stage10_human" / "cvat_tasks.json"
    ledger_append(str(p), {"project": "x", "project_id": 1, "task_id": 10, "kind": "double_A", "scene": "chunk_0", "frames_json": "/f.json", "assignee": None, "created_utc": "t", "run_tag": ""})
    ledger_append(str(p), {"project": "y", "project_id": 2, "task_id": 11, "kind": "double_B", "scene": "chunk_0", "frames_json": "/f.json", "assignee": "b", "created_utc": "t", "run_tag": ""})
    rows = ledger_load(str(p))
    assert [r["task_id"] for r in rows] == [10, 11] and rows[1]["assignee"] == "b"


def test_ledger_row_is_keyed_by_task_id(tmp_path):
    """Re-recording a task updates its row: the importer must not import one
    task twice because the publish was run again."""
    p = str(tmp_path / "cvat_tasks.json")
    ledger_append(p, {"task_id": 10, "kind": "double_A", "assignee": None})
    ledger_append(p, {"task_id": 10, "kind": "double_A", "assignee": "a"})
    rows = ledger_load(p)
    assert len(rows) == 1 and rows[0]["assignee"] == "a"


def test_ledger_drop_tasks_forgets_deleted_tasks(tmp_path):
    """--replace deletes a task server-side; its row would 404 the importer."""
    p = str(tmp_path / "cvat_tasks.json")
    for task_id in (10, 11, 12):
        ledger_append(p, {"task_id": task_id, "kind": "review"})
    assert ledger_drop_tasks(p, [10, 12]) == 2
    assert [r["task_id"] for r in ledger_load(p)] == [11]
    assert ledger_drop_tasks(p, [99]) == 0


def test_ledger_load_of_a_missing_file_is_empty(tmp_path):
    assert ledger_load(str(tmp_path / "nope.json")) == []


def test_find_user_id_is_exact_or_none():
    class U:
        def __init__(self, username, id):
            self.username, self.id = username, id

    users = [U("ann_a", 5), U("ann_b", 6)]
    assert find_user_id(users, "ann_b") == 6
    assert find_user_id(users, "ann_c") is None      # unassigned, never someone else
    assert find_user_id(users, None) is None


def test_double_refuses_replace_and_reimport(capsys):
    """The A/B tasks hold the annotators' own work: neither flag may reach them,
    and the refusal comes before any client is built (no server needed here)."""
    for flag in ("--replace", "--reimport"):
        assert main(["--password", "x", "--which", "double", flag]) == 2
        err = capsys.readouterr().err
        assert flag in err and "destroy" in err


def test_double_reads_the_blank_export_root(tmp_path, capsys, monkeypatch):
    """--which double publishes cvat_export_3d_double, and says which exporter
    call fills it when it is empty."""
    monkeypatch.setattr(setup_3d, "load_paths",
                        lambda *a, **k: types.SimpleNamespace(work_root=str(tmp_path)))
    os.makedirs(tmp_path / "cvat_export_3d" / "scene-0001")
    open(tmp_path / "cvat_export_3d" / "scene-0001" / "task.zip", "w").close()
    assert main(["--password", "x", "--which", "double"]) == 2
    err = capsys.readouterr().err
    assert "cvat_export_3d_double" in err and "--blank" in err


def test_attribute_specs_are_independent_copies():
    """label_spec must not hand every label the SAME attribute dicts: CVAT's
    project creation is fed one JSON per label and a shared mutable dict would
    let an edit to one label's schema reach the others."""
    a, b = label_spec(["a car", "a pedestrian"])
    assert a["attributes"][0] is not b["attributes"][0]
    assert cuboid_attribute_specs()[0] is not cuboid_attribute_specs()[0]
