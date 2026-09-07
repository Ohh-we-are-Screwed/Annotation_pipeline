from __future__ import annotations

import json
import math
import os
import sys
import types
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.schemas import AnnotationRecord, ProvenancePolicy, read_records  # noqa: E402
from pipeline.release.human import COVERAGE_SPEC, load_human  # noqa: E402
from scripts.export_cvat_3d import cuboid, datumaro_document, item_skeleton  # noqa: E402
from scripts.import_cvat_3d import (  # noqa: E402
    dataset_json_from_zip, incomplete_reason, parse_datumaro_3d, records_from_boxes, write_scene,
)

HUMAN = ProvenancePolicy(allow_human_provenance=True)
FRAMES = [{"frame": 0, "name": "000001", "sample_token": "s1", "channels": []},
          {"frame": 1, "name": "000002", "sample_token": "s2", "channels": []}]


def _doc():
    labels = ["a car", "a pedestrian"]
    i0, i1 = item_skeleton(0, []), item_skeleton(1, [])
    c = cuboid(1, 0, [10.0, 2.0, 0.5], 0.3, (4.5, 1.8, 1.6), record_token="k0:CAM_FRONT:0", track_id=7)
    c["attributes"].update(attribute="vehicle.parked", uncertain=True, uncertain_reason="could be a microbus")
    i0["annotations"].append(c)
    i1["annotations"].append(cuboid(2, 0, [11.0, 2.0, 0.5], 0.3, (4.5, 1.8, 1.6), track_id=7))
    i1["annotations"].append(cuboid(3, 1, [-3.0, 1.0, 0.9], 1.0, (0.7, 0.6, 1.7)))
    return datumaro_document(labels, [i0, i1])


def test_parse_inverts_the_exporter():
    boxes = parse_datumaro_3d(_doc(), FRAMES)
    assert [b.sample_token for b in boxes] == ["s1", "s2", "s2"]
    car = boxes[0]
    assert car.size_wlh_m == [1.8, 4.5, 1.6] and car.translation_m == [10.0, 2.0, 0.5]
    assert abs(2 * math.atan2(car.rotation_wxyz[3], car.rotation_wxyz[0]) - 0.3) < 1e-6
    assert car.record_token == "k0:CAM_FRONT:0" and car.track_id == 7 and car.attribute == "vehicle.parked"
    assert car.uncertain is True and car.uncertain_reason == "could be a microbus"
    ped = boxes[2]
    assert ped.record_token is None and ped.track_id is None and ped.attribute is None and ped.uncertain is False


def test_records_sources_identity_and_policy(tmp_path):
    boxes = parse_datumaro_3d(_doc(), FRAMES)
    recs = records_from_boxes(boxes, scene_token="sc", kind="review", verified_by="ann_a",
                              timestamps_ns={"s1": 1_700_000_000_000_000_000, "s2": 1_700_000_000_400_000_000},
                              point_counter=lambda tok, t, s, q: 12, mapper_classes={"a car", "a pedestrian"})
    assert [r.provenance.source for r in recs] == ["human_verified", "human_created", "human_created"]
    assert recs[0].instance_token == recs[1].instance_token == "human:review:7"
    assert recs[0].attribute == "vehicle.parked" and recs[0].is_uncertain is True
    assert recs[0].num_lidar_pts == 12 and recs[0].provenance.verification_pass == 1
    assert all(r.provenance.annotator_pass is None for r in recs)
    dbl = records_from_boxes(boxes, scene_token="sc", kind="double_B", verified_by="ann_b",
                             timestamps_ns={"s1": 1, "s2": 2}, point_counter=lambda *a: 0, mapper_classes={"a car", "a pedestrian"})
    assert all(r.provenance.annotator_pass == "B" for r in dbl)
    v, c = write_scene(str(tmp_path), "chunk_x", recs, {"review": ["s1", "s2"], "double_A": [], "double_B": []})
    back = read_records(v, policy=HUMAN, expect_type=AnnotationRecord)
    assert len(back) == 3 and json.load(open(c))["review"] == ["s1", "s2"]


def test_unmapped_label_aborts():
    boxes = parse_datumaro_3d(_doc(), FRAMES)
    try:
        records_from_boxes(boxes, scene_token="sc", kind="review", verified_by="x", timestamps_ns={"s1": 1, "s2": 2},
                           point_counter=lambda *a: 0, mapper_classes={"a car"})
    except ValueError as exc:
        assert "a pedestrian" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unmapped label")


# --- the traps the live spike found (docs/evidence/2026-09-08-cvat-3d-roundtrip.md) ---


def test_untracked_shape_does_not_join_track_zero():
    """§2's trap: `track_id` is one of OUR declared attributes, so an untracked
    shape comes back carrying its default `0.0` — which collides with the first
    real track's exported index `0`. `keyframe` present, not the value of
    `track_id`, is the discriminator; getting it wrong merges an unrelated box
    into track 0's identity, and (both being on one frame here) makes the
    release exporter abort with "two annotations in one sample".
    """
    i0 = item_skeleton(0, [])
    i0["annotations"].append(cuboid(1, 0, [1.0, 0.0, 0.0], 0.0, (4.5, 1.8, 1.6),
                                    record_token="k0:CAM_FRONT:0", track_id=0))
    plain = cuboid(2, 0, [9.0, 0.0, 0.0], 0.0, (4.5, 1.8, 1.6), record_token="k0:CAM_FRONT:1")
    plain["attributes"]["track_id"] = 0.0   # what CVAT sends for a shape with no track
    i0["annotations"].append(plain)
    boxes = parse_datumaro_3d(datumaro_document(["a car"], [i0]), FRAMES[:1])
    assert boxes[0].track_id == 0 and boxes[1].track_id is None
    recs = records_from_boxes(boxes, scene_token="sc", kind="review", verified_by="ann_a",
                              timestamps_ns={"s1": 1_700_000_000_000_000_000},
                              point_counter=lambda *a: 3, mapper_classes={"a car"})
    assert recs[0].instance_token != recs[1].instance_token
    assert recs[1].instance_token == "human:review:k0:CAM_FRONT:1"


def test_checkbox_may_arrive_as_a_string():
    """§3: `uncertain` comes back a real bool from CVAT 2.72, but a hand-edited
    dataset carries "true"; and a reason typed without the box ticked is doubt
    the schema only lets us keep alongside the flag."""
    i0 = item_skeleton(0, [])
    c = cuboid(1, 0, [1.0, 0.0, 0.0], 0.0, (4.5, 1.8, 1.6), record_token="k0:CAM_FRONT:0")
    c["attributes"].update(uncertain="true", uncertain_reason="half behind a bus")
    i0["annotations"].append(c)
    c2 = cuboid(2, 0, [9.0, 0.0, 0.0], 0.0, (4.5, 1.8, 1.6), record_token="k0:CAM_FRONT:1")
    c2["attributes"].update(uncertain=False, uncertain_reason="not sure it is a car")
    i0["annotations"].append(c2)
    boxes = parse_datumaro_3d(datumaro_document(["a car"], [i0]), FRAMES[:1])
    assert boxes[0].uncertain is True and boxes[1].uncertain is False
    recs = records_from_boxes(boxes, scene_token="sc", kind="review", verified_by="ann_a",
                              timestamps_ns={"s1": 1_700_000_000_000_000_000},
                              point_counter=lambda *a: 3, mapper_classes={"a car"})
    assert recs[0].is_uncertain is True and recs[0].is_uncertain_reason == "half behind a bus"
    assert recs[1].is_uncertain is True and recs[1].is_uncertain_reason == "not sure it is a car"


def test_dataset_json_is_read_from_annotations_default_json(tmp_path):
    """§6: the export zip's whole member list is ['annotations/default.json'];
    a task split into subsets would yield one file per subset, which is refused
    rather than half-read."""
    one = tmp_path / "one.zip"
    with zipfile.ZipFile(one, "w") as zf:
        zf.writestr("annotations/default.json", json.dumps(_doc()))
    assert len(dataset_json_from_zip(str(one))["items"]) == 2

    two = tmp_path / "two.zip"
    with zipfile.ZipFile(two, "w") as zf:
        zf.writestr("annotations/default.json", json.dumps(_doc()))
        zf.writestr("annotations/train.json", json.dumps(_doc()))
    try:
        dataset_json_from_zip(str(two))
    except ValueError as exc:
        assert "2 annotation subsets" in str(exc)
    else:
        raise AssertionError("expected ValueError for a multi-subset export")


def test_coverage_is_written_even_with_no_boxes_and_human_py_reads_the_pair(tmp_path):
    """A reviewer who deleted every pre-label on a frame has still covered it.
    human.py cannot infer that from an empty verified.jsonl, so the pass is
    recorded in coverage.json — without it the deleted rows come back."""
    v, c = write_scene(str(tmp_path), "chunk_y", [], {"review": ["s9", "s9"], "double_A": [],
                                                      "double_B": []})
    assert os.path.getsize(v) == 0
    cov = json.load(open(c))
    assert cov["spec"] == COVERAGE_SPEC and cov["review"] == ["s9"] and cov["double_B"] == []
    rows, coverage = load_human(str(tmp_path))
    assert rows == [] and coverage["review"] == {"s9"} and coverage["double_A"] == set()


def test_double_a_rows_round_trip_into_human_py_as_double_A(tmp_path):
    boxes = parse_datumaro_3d(_doc(), FRAMES)
    recs = records_from_boxes(boxes, scene_token="sc", kind="double_A", verified_by="ann_a",
                              timestamps_ns={"s1": 1_700_000_000_000_000_000,
                                             "s2": 1_700_000_000_400_000_000},
                              point_counter=lambda *a: 4, mapper_classes={"a car", "a pedestrian"},
                              task_id=42)
    # The identity carries the kind AND the task: pass A and pass B annotate the
    # same frames, and CVAT's exported track index is dense and per task, so
    # neither may collide with the other's on one sample.
    assert recs[0].provenance.annotator_pass == "A"
    assert recs[0].instance_token == "human:double_A:42:7"
    write_scene(str(tmp_path), "chunk_z", recs, {"review": [], "double_A": ["s1", "s2"],
                                                 "double_B": []})
    rows, coverage = load_human(str(tmp_path))
    assert {r["human_kind"] for r in rows} == {"double_A"} and coverage["double_A"] == {"s1", "s2"}
    assert all(r["annotator_pass"] == "A" for r in rows)


def test_incomplete_jobs_are_named_not_guessed():
    job = lambda state: types.SimpleNamespace(state=state)  # noqa: E731
    assert incomplete_reason([job("completed"), job("completed")]) is None
    assert "1/2 job(s) not completed" in incomplete_reason([job("completed"), job("in progress")])
    assert incomplete_reason([]) == "the task has no jobs"
