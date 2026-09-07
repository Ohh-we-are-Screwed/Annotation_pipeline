from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.schemas import (  # noqa: E402
    AnnotationRecord, GateVector, Provenance, ProvenancePolicy, Record, SchemaValidationError,
    read_records, write_records,
)

HUMAN = ProvenancePolicy(allow_human_provenance=True)


def _rec(**over):
    base = dict(
        token="s1:CAM_FRONT:0", sample_token="s1", instance_token="i", category="a car", frame="ego",
        t_ns=1_700_000_000_000_000_000, time_base="unix_ns", translation_m=[1.0, 2.0, 0.0],
        size_wlh_m=[1.8, 4.5, 1.6], rotation_wxyz=[1.0, 0.0, 0.0, 0.0], num_lidar_pts=10,
        provenance=Provenance(source="human_created", tier="auto_accept",
                              gates=GateVector(conf=1.0, lidar_pts_ok=True, spatial_ok=True, spatial_ok_source="human"),
                              verified_by="ann_a", verification_pass=1, annotator_pass="A"),
        coverage_config="R2",
    )
    base.update(over)
    return AnnotationRecord(**base)


def test_human_record_with_new_fields_round_trips(tmp_path):
    rec = _rec(attribute="vehicle.parked", is_uncertain=True, is_uncertain_reason="could be a microbus")
    assert rec.validate(policy=HUMAN) == []
    p = tmp_path / "v.jsonl"
    write_records(p, [rec], policy=HUMAN)
    back = read_records(p, policy=HUMAN)[0]
    assert back.provenance.annotator_pass == "A" and back.is_uncertain is True
    assert back.attribute == "vehicle.parked"


def test_default_policy_still_refuses_human(tmp_path):
    with pytest.raises(SchemaValidationError):
        write_records(tmp_path / "v.jsonl", [_rec()])


def test_annotator_pass_values_and_source_gate():
    bad = _rec(provenance=Provenance(source="human_created", tier="auto_accept",
                                     gates=GateVector(conf=1.0, lidar_pts_ok=True, spatial_ok=True, spatial_ok_source="h"),
                                     verified_by="x", verification_pass=1, annotator_pass="C"))
    assert any("annotator_pass" in e for e in bad.validate(policy=HUMAN))
    pipe = _rec(provenance=Provenance(source="pipeline", tier="auto_accept",
                                      gates=GateVector(conf=1.0, lidar_pts_ok=True, spatial_ok=True, spatial_ok_source="g"),
                                      annotator_pass="A"))
    assert any("annotator_pass" in e for e in pipe.validate())


def test_pipeline_records_keep_the_no_producer_rules():
    pipe = _rec(provenance=Provenance(source="pipeline", tier="auto_accept",
                                      gates=GateVector(conf=1.0, lidar_pts_ok=True, spatial_ok=True, spatial_ok_source="g")),
                attribute="vehicle.moving", is_uncertain=False)
    errs = pipe.validate()
    assert any("attribute" in e for e in errs) and any("is_uncertain" in e for e in errs)


def test_human_attribute_must_be_a_nuscenes_name_and_reason_needs_flag():
    assert any("attribute" in e for e in _rec(attribute="vehicle.flying").validate(policy=HUMAN))
    assert any("is_uncertain_reason" in e for e in _rec(is_uncertain_reason="why").validate(policy=HUMAN))


def test_schema_attribute_names_still_agree_with_the_release_builder():
    """schemas.py restates the eight names so it stays stdlib-only; keep the copies equal."""
    from pipeline.release.attributes import ATTRIBUTE_NAMES  # noqa: E402

    from pipeline.common.schemas import ATTRIBUTE_NAMES_NUSCENES  # noqa: E402

    assert tuple(ATTRIBUTE_NAMES_NUSCENES) == tuple(ATTRIBUTE_NAMES)
