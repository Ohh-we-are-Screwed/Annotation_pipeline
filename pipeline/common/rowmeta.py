"""Readers for Stage 3b's optional per-box provenance arrays (C3, C27).

`optional_c27_array` was written inside `pipeline/stage4_masks/masks.py`, where
it was the only reader. Stage 3c's track-aware mode needs the identical
absent-is-legal / short-is-a-refusal semantics for `track_ids`, and importing a
helper out of a sibling stage package is exactly the coupling C3 records (Stage
5 came to depend on three sibling stages it never calls). So the function moves
here — beside the atomic writers and the upstream gate, which were promoted for
the same reason — and both stages import it from one place. The semantics are
untouched: absent -> `[]`, present-but-wrong-length -> `UpstreamRefusal`.

Phase 2 constraints: no GPU, no models, stdlib only.
"""

from __future__ import annotations

from pipeline.common.manifest import UpstreamRefusal

__all__ = ["optional_c27_array"]


def optional_c27_array(row: dict, key: str, n_boxes: int) -> list:
    """One of Stage 3b's optional per-box arrays, or [] — but never a SHORT one.

    ABSENT is legal: a plain Stage 3 row (a pre-C27 tree) carries no recovery
    provenance and the caller's defaults are the honest reading of that. PRESENT
    but shorter than `boxes_xyxy_px` is not: every box past the array's end then
    took the default `"yolo"` / None / 0, so a RECOVERED box — one no detector
    proposed on this frame — was recorded as detector output. That is exactly the
    provenance laundering C27 exists to prevent, and a truncated or partially
    written upstream row is how it happens. Refused as the upstream-contract
    failure it is, the way load_upstream and read_proposal_rows refuse.
    """
    values = row.get(key)
    if values is None:
        return []
    values = list(values)
    if len(values) != n_boxes:
        raise UpstreamRefusal(
            f"sample_data_token={row.get('sample_data_token')!r}: {key} has {len(values)} "
            f"entries for {n_boxes} boxes in boxes_xyxy_px. These are parallel arrays (C27): a "
            "mismatch does not shift the provenance, it silently DROPS it — every box past the "
            "end reads as a plain detection. Re-run "
            "`python3 -m pipeline.stage3b_track2d.track2d` over this scene"
        )
    return values
