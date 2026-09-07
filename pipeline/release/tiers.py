"""Which rows ship in sample_annotation and why the rest do not (spec §5)."""

from __future__ import annotations

from pipeline.common.schemas import HUMAN_SOURCES

REASON_REJECTED = "tier_rejected"
REASON_FLAGGED = "tier_flagged"
REASON_HUMAN = "superseded_by_human"
REASON_DOUBLE = "superseded_by_double_pass"
ADMIT_AUTO = "auto_accept"
ADMIT_ALL = "all"
ADMIT_MODES = (ADMIT_AUTO, ADMIT_ALL)


def partition(rows: list, admit: str, superseded: dict) -> tuple:
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
        if reason is None:
            included.append(r)
        else:
            r["excluded_reason"] = reason
            excluded.append(r)
    return included, excluded
