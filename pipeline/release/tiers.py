"""Which rows ship in sample_annotation and why the rest do not (spec §5)."""

from __future__ import annotations

from pipeline.common.schemas import HUMAN_SOURCES

REASON_REJECTED = "tier_rejected"
REASON_FLAGGED = "tier_flagged"
REASON_HUMAN = "superseded_by_human"
REASON_DOUBLE = "superseded_by_double_pass"
# An interpolated row is a geometric fill, so it never faced Stage 9's confidence
# or footprint gates — but it does carry a real point count, measured against its
# own keyframe's ground-filtered cloud, and DELIVERY_NOTE.md states a return
# floor. A fill under it is ground truth no LiDAR detector can hit; it goes to
# the sidecar rather than shipping or being dropped (final review C2).
REASON_INTERPOLATED_FLOOR = "interpolated_below_point_floor"
ADMIT_AUTO = "auto_accept"
ADMIT_ALL = "all"
ADMIT_MODES = (ADMIT_AUTO, ADMIT_ALL)


def partition(rows: list, admit: str, superseded: dict,
              interpolated_min_lidar_points: int | None = None) -> tuple:
    """included, excluded — the one place a row is dropped, and it says why.

    `interpolated_min_lidar_points` is the LiDAR-return floor an interpolated row
    has to clear (None disables it, as does `--tiers all`, which admits
    everything by definition). The tier reasons win over it, so an interpolated
    row that inherited `flagged` is still excluded as `tier_flagged` and the
    counts reconcile with the tier table.
    """
    if admit not in ADMIT_MODES:
        raise ValueError(f"admit={admit!r} is not one of {ADMIT_MODES}")
    included, excluded = [], []
    for r in rows:
        prov = r.get("provenance") or {}
        reason = superseded.get(r["token"])
        if reason is None and prov.get("source") not in HUMAN_SOURCES and admit == ADMIT_AUTO:
            tier = prov.get("tier")
            if tier == "rejected":
                reason = REASON_REJECTED
            elif tier == "flagged":
                reason = REASON_FLAGGED
            elif tier != ADMIT_AUTO:
                reason = f"tier_{tier}"
            elif (interpolated_min_lidar_points and r.get("stitch_interpolated")
                  and int(r.get("num_lidar_pts") or 0) < interpolated_min_lidar_points):
                reason = REASON_INTERPOLATED_FLOOR
        if reason is None:
            included.append(r)
        else:
            r["excluded_reason"] = reason
            excluded.append(r)
    return included, excluded
