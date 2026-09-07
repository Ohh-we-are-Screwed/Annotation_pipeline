from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.release.tiers import (  # noqa: E402
    ADMIT_ALL, ADMIT_AUTO, REASON_DOUBLE, REASON_FLAGGED, REASON_HUMAN, REASON_REJECTED, partition,
)


def _r(tok, tier=None, source="pipeline"):
    return {"token": tok, "provenance": {"source": source, "tier": tier}}


def test_auto_accept_only_by_default_with_reasons():
    rows = [_r("a", "auto_accept"), _r("b", "flagged"), _r("c", "rejected"),
            _r("h", None, "human_verified")]
    inc, exc = partition(rows, ADMIT_AUTO, {})
    assert [r["token"] for r in inc] == ["a", "h"]
    assert {r["token"]: r["excluded_reason"] for r in exc} == {"b": REASON_FLAGGED, "c": REASON_REJECTED}


def test_superseded_wins_over_tier_and_applies_to_humans():
    rows = [_r("a", "auto_accept"), _r("h", None, "human_verified")]
    inc, exc = partition(rows, ADMIT_AUTO, {"a": REASON_HUMAN, "h": REASON_DOUBLE})
    assert inc == [] and {r["token"]: r["excluded_reason"] for r in exc} == {"a": REASON_HUMAN, "h": REASON_DOUBLE}


def test_admit_all_reproduces_legacy():
    rows = [_r("a", "auto_accept"), _r("b", "flagged"), _r("c", "rejected")]
    inc, exc = partition(rows, ADMIT_ALL, {})
    assert len(inc) == 3 and exc == []


def test_unknown_admit_raises():
    with pytest.raises(ValueError):
        partition([], "some", {})
