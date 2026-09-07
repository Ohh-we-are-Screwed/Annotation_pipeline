"""I-5 human rows: load, classify, and decide what they supersede (spec §7).

The human loop produces three kinds of pass over a sample, and which kinds a
sample got is a property of the CVAT tasks, not of the rows that came back: an
annotator who deletes every box on a reviewed frame has still covered it. That
is why coverage comes from `coverage.json` (written when the jobs complete)
rather than being inferred from `verified.jsonl` — inferring it would silently
resurrect the pipeline rows a reviewer deliberately removed.

Precedence per sample (§7):
  - `double_A` covered -> the two independent passes are the label of record:
    every pipeline row AND every review row on that sample is superseded with
    `REASON_DOUBLE`; the A and B rows are included.
  - else `review` covered -> pipeline rows are superseded with `REASON_HUMAN`
    and the review rows are included.
  - else the pipeline rows stand. Any B rows present are still included, and
    the sample is reported in `half_imported`: a B pass with no A coverage is a
    half-imported double, which is an operational fact the release metadata has
    to carry rather than a state to silently drop.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

from pipeline.common.schemas import AnnotationRecord, ProvenancePolicy, read_records
from pipeline.release.tiers import REASON_DOUBLE, REASON_HUMAN

KIND_REVIEW = "review"
KIND_A = "double_A"
KIND_B = "double_B"
KINDS = (KIND_REVIEW, KIND_A, KIND_B)
COVERAGE_SPEC = "dhakascenes/human_coverage/v1"
HUMAN_POLICY = ProvenancePolicy(allow_human_provenance=True)


@dataclass
class HumanMerge:
    """What a human pass adds and what it displaces."""

    rows: list
    superseded: dict
    coverage: dict
    half_imported: list
    stats: dict = field(default_factory=dict)


def _kind_of(row: dict) -> str:
    ap = (row.get("provenance") or {}).get("annotator_pass")
    return {None: KIND_REVIEW, "A": KIND_A, "B": KIND_B}[ap]


def load_human(human_dir: str):
    """Read every `scenes/*/verified.jsonl` under `human_dir` with its coverage.

    Returns `(rows, coverage)`. Each row is a plain dict with `human_kind` and a
    top-level `annotator_pass` copied out of the provenance block, so the
    exporter never has to reach into provenance to know which pass a row is.
    """
    rows: list = []
    coverage = {k: set() for k in KINDS}
    for vpath in sorted(glob.glob(os.path.join(human_dir, "scenes", "*", "verified.jsonl"))):
        for rec in read_records(vpath, policy=HUMAN_POLICY, expect_type=AnnotationRecord):
            d = rec.to_dict()
            d["human_kind"] = _kind_of(d)
            d["annotator_pass"] = d["provenance"].get("annotator_pass")
            d["__source_file__"] = vpath
            rows.append(d)
        cpath = os.path.join(os.path.dirname(vpath), "coverage.json")
        if os.path.isfile(cpath):
            with open(cpath, "r", encoding="utf-8") as fh:
                cov = json.load(fh)
            if cov.get("spec") != COVERAGE_SPEC:
                raise ValueError(f"{cpath}: spec={cov.get('spec')!r}, expected {COVERAGE_SPEC!r}")
            for k in KINDS:
                coverage[k].update(cov.get(k) or [])
    return rows, coverage


def merge_human(pipeline_rows: list, human_rows: list, coverage: dict) -> HumanMerge:
    """Apply the §7 precedence rules. Nothing is dropped here, only marked.

    `superseded` maps a token to its exclusion reason and covers pipeline rows
    AND human review rows; `tiers.partition` is the single place that acts on
    it, so a superseded row still appears once in the excluded sidecar.
    """
    cov = {k: set(coverage.get(k) or ()) for k in KINDS}
    superseded: dict = {}
    for r in pipeline_rows:
        s = r["sample_token"]
        if s in cov[KIND_A]:
            superseded[r["token"]] = REASON_DOUBLE
        elif s in cov[KIND_REVIEW]:
            superseded[r["token"]] = REASON_HUMAN
    for r in human_rows:
        if r["human_kind"] == KIND_REVIEW and r["sample_token"] in cov[KIND_A]:
            superseded[r["token"]] = REASON_DOUBLE
    b_samples = {r["sample_token"] for r in human_rows if r["human_kind"] == KIND_B}
    half = sorted(b_samples - cov[KIND_A])
    stats = {f"n_samples_{k}": len(cov[k]) for k in KINDS}
    stats.update({f"n_rows_{k}": sum(1 for r in human_rows if r["human_kind"] == k) for k in KINDS})
    stats["n_superseded_by_human"] = sum(1 for v in superseded.values() if v == REASON_HUMAN)
    stats["n_superseded_by_double_pass"] = sum(1 for v in superseded.values() if v == REASON_DOUBLE)
    stats["n_half_imported_samples"] = len(half)
    return HumanMerge(rows=list(human_rows), superseded=superseded, coverage=cov, half_imported=half, stats=stats)
