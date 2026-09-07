"""Operator decision 2026-09-07: the pipeline annotates to the benchmark's
evaluation range (class_range 50/40/30 m); the Stage 9 point floor decides
what survives. Both caps must agree or Stage 1 prunes to one radius while
Stage 5/6 score against another."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common import eval_region  # noqa: E402
from pipeline.stage1_ingestion.ingest import IngestConfig  # noqa: E402


def test_stage1_cap_is_50m():
    assert IngestConfig().range_cap_m == 50.0


def test_eval_region_cap_is_50m():
    assert eval_region._R_MAX_M == 50.0
    assert eval_region.region_spec_from_config({"coverage_config": "R2"}).r_max_m == 50.0


def test_the_two_caps_agree():
    assert IngestConfig().range_cap_m == eval_region._R_MAX_M


def test_provenance_names_the_decision():
    assert "2026-09-07" in IngestConfig().provenance["range_cap_m"]
