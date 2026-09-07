from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.release.tiers import (  # noqa: E402
    ADMIT_ALL, ADMIT_AUTO, REASON_DOUBLE, REASON_FLAGGED, REASON_HUMAN, REASON_INTERPOLATED_FLOOR,
    REASON_REJECTED, partition,
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


# --- C2: an interpolated row must carry the LiDAR evidence the note claims ----


def _interp(tok, n_pts, tier="auto_accept"):
    r = _r(tok, tier)
    r.update(stitch_interpolated=True, num_lidar_pts=n_pts)
    return r


def test_interpolated_rows_below_the_point_floor_are_excluded_with_a_reason():
    rows = [_interp("i5", 5), _interp("i4", 4), _interp("i0", 0),
            _r("real", "auto_accept")]
    rows[3]["num_lidar_pts"] = 0          # a measured row is never floored here
    inc, exc = partition(rows, ADMIT_AUTO, {}, interpolated_min_lidar_points=5)
    assert [r["token"] for r in inc] == ["i5", "real"]
    assert {r["token"]: r["excluded_reason"] for r in exc} == {
        "i4": REASON_INTERPOLATED_FLOOR, "i0": REASON_INTERPOLATED_FLOOR}


def test_the_worse_tier_still_names_the_exclusion():
    # An interpolated row that inherited `flagged` is excluded as tier_flagged,
    # so the reason counts still reconcile with the tier table.
    inc, exc = partition([_interp("f", 0, tier="flagged")], ADMIT_AUTO, {},
                         interpolated_min_lidar_points=5)
    assert inc == [] and exc[0]["excluded_reason"] == REASON_FLAGGED


def test_no_floor_means_no_rule():
    inc, exc = partition([_interp("i0", 0)], ADMIT_AUTO, {})
    assert [r["token"] for r in inc] == ["i0"] and exc == []


def test_admit_all_keeps_every_interpolated_row():
    inc, exc = partition([_interp("i0", 0)], ADMIT_ALL, {}, interpolated_min_lidar_points=5)
    assert [r["token"] for r in inc] == ["i0"] and exc == []
