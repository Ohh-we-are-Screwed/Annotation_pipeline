from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.schemas import AnnotationRecord, GateVector, Provenance, ProvenancePolicy, write_records  # noqa: E402
from pipeline.release.human import KIND_A, KIND_B, KIND_REVIEW, load_human, merge_human  # noqa: E402
from pipeline.release.tiers import REASON_DOUBLE, REASON_HUMAN  # noqa: E402

HUMAN = ProvenancePolicy(allow_human_provenance=True)


def _human(tok, sample, annotator_pass=None, source="human_created"):
    return AnnotationRecord(
        token=tok, sample_token=sample, instance_token=f"h:{tok}", category="a car", frame="ego",
        t_ns=1_700_000_000_000_000_000, time_base="unix_ns", translation_m=[1.0, 0.0, 0.0],
        size_wlh_m=[1.8, 4.5, 1.6], rotation_wxyz=[1.0, 0.0, 0.0, 0.0], num_lidar_pts=3,
        provenance=Provenance(source=source, tier="auto_accept",
                              gates=GateVector(conf=1.0, lidar_pts_ok=True, spatial_ok=True, spatial_ok_source="h"),
                              verified_by="ann", verification_pass=1, annotator_pass=annotator_pass),
        coverage_config="R2")


def _pipe(tok, sample):
    return {"token": tok, "sample_token": sample, "provenance": {"source": "pipeline", "tier": "auto_accept"}}


def test_load_reads_every_scene(tmp_path):
    d = tmp_path / "scenes" / "chunk_x"
    d.mkdir(parents=True)
    write_records(d / "verified.jsonl", [_human("r1", "s1"), _human("a1", "s2", "A")], policy=HUMAN)
    (d / "coverage.json").write_text(json.dumps({"spec": "dhakascenes/human_coverage/v1", "scene": "chunk_x",
                                                 "review": ["s1", "s9"], "double_A": ["s2"], "double_B": []}))
    rows, cov = load_human(str(tmp_path))
    assert {r["token"] for r in rows} == {"r1", "a1"}
    assert cov[KIND_REVIEW] == {"s1", "s9"} and cov[KIND_A] == {"s2"} and cov[KIND_B] == set()
    kinds = {r["token"]: r["human_kind"] for r in rows}
    assert kinds == {"r1": KIND_REVIEW, "a1": KIND_A}
    assert next(r for r in rows if r["token"] == "a1")["annotator_pass"] == "A"


def test_precedence_rules():
    pipeline = [_pipe("p1", "s1"), _pipe("p2", "s2"), _pipe("p3", "s3"), _pipe("p4", "s4")]
    human = [
        {**_human("r1", "s1").to_dict(), "human_kind": KIND_REVIEW, "annotator_pass": None},
        {**_human("r2", "s2").to_dict(), "human_kind": KIND_REVIEW, "annotator_pass": None},
        {**_human("a2", "s2", "A").to_dict(), "human_kind": KIND_A, "annotator_pass": "A"},
        {**_human("b2", "s2", "B").to_dict(), "human_kind": KIND_B, "annotator_pass": "B"},
        {**_human("b3", "s3", "B").to_dict(), "human_kind": KIND_B, "annotator_pass": "B"},
    ]
    cov = {KIND_REVIEW: {"s1", "s2", "s9"}, KIND_A: {"s2"}, KIND_B: {"s2", "s3"}}
    m = merge_human(pipeline, human, cov)
    assert m.superseded == {"p1": REASON_HUMAN, "p2": REASON_DOUBLE, "r2": REASON_DOUBLE}
    assert {r["token"] for r in m.rows} == {"r1", "r2", "a2", "b2", "b3"}   # r2 is returned; tiers.partition drops it
    assert m.half_imported == ["s3"]
    assert m.stats["n_samples_review"] == 3 and m.stats["n_samples_double_A"] == 1


def test_review_deletion_is_honoured():
    # s1 is review-covered but the reviewer removed every box: pipeline rows go, nothing comes back
    m = merge_human([_pipe("p1", "s1")], [], {KIND_REVIEW: {"s1"}, KIND_A: set(), KIND_B: set()})
    assert m.superseded == {"p1": REASON_HUMAN} and m.rows == []
