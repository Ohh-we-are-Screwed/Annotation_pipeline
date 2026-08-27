"""Stage 3 merge — arm A (COCO YOLO11x) + arm B (RSUD20K fine-tune) -> stage3_merged/.

Design: docs/RUNNING.md "Stage 3 arm A / arm B" (2026-08-26) and DECISIONS C28.

Emits Stage 3's exact schema in Stage 3's row order plus additive keys (the
Stage 3b trick, C27, used a second time), so Stage 4 consumes the output
unchanged via --stage3-dir. Arbitration is a CLASS-PAIR TABLE, never a score
contest: arm A is confident on its wrong answers (`car` 0.85 beats `cng`
0.55), and a score contest would rebuild C21's failure in a new mechanism.

A suppressed arm A box LEAVES the parallel arrays and moves, whole, into the
row's `merge.suppressed_arm_a` ledger. In the arrays it would ride through
Stage 4 (one mask per box, in order) and Stage 5 (one lift per mask) under a
label the merge just ruled impossible; in the ledger it is retained, auditable
and inert — Stage 3's own `dedup` ledger precedent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.common.manifest import (  # noqa: E402
    UpstreamRefusal,
    clear_markers,
    require_upstream,
    write_json_atomic,
    write_jsonl_atomic,
    write_marker,
)
from pipeline.stage3_proposals.proposals import (  # noqa: E402
    build_caption,
    load_taxonomy,
    pairwise_iou,
)

STAGE = "stage3_merge"
STAGE_SPEC = "dhakascenes-pilot/stage3_merge/v1"

KEEP_BOTH = "keep_both"
SUPPRESS_ARM_A = "suppress_arm_a"

# The vocabulary-authority table (docs/RUNNING.md, verbatim). Keyed by arm A
# PHRASE. Absent phrase = keep both, counted as out-of-table so an unforeseen
# contest is visible rather than silently resolved.
ARBITRATION: dict[str, str] = {
    "a car": SUPPRESS_ARM_A,          # COCO has no word for the object
    "a truck": SUPPRESS_ARM_A,
    "a bus": SUPPRESS_ARM_A,
    "a motorcycle": SUPPRESS_ARM_A,   # three-wheeler forced onto a two-wheeler label
    "a bicycle": SUPPRESS_ARM_A,
    "a pedestrian": KEEP_BOTH,        # the puller/rider is a SEPARATE object
}

# What arm B is allowed to put in a row. Anything else is a leaked scaffolding
# class: the ship filter failed, and merging would poison arm A's turf.
ARM_B_PHRASES = ("a rickshaw", "an auto rickshaw")

# Stage 3b's per-box parallel arrays (track2d.py:rewrite_row), extended — never
# rebuilt — when the arm A tree is a stage3b_track2d tree. Values are the fill
# for an appended arm B box: not tracked, provenance arm_b, zero hops.
STAGE3B_EXTENSIONS: dict[str, object] = {
    "track_ids": None,
    "box_sources": "arm_b",
    "n_propagated_hops": 0,
    "refined": False,
    "boxes_xyxy_px_original": None,
}

# Row keys that must agree between the two arms for the rows to describe the
# same camera of the same keyframe of the same substrate.
FRAME_IDENTITY_KEYS = (
    "keyframe_token", "scene_token", "channel", "sample_data_token",
    "image_path", "image_size_px",
)


class MergeContractError(RuntimeError):
    """A pair of rows (or trees) that cannot honestly be merged."""


def merge_rows(row_a: dict, row_b: dict, *, caption, taxonomy, iou_threshold: float) -> dict:
    """One arm A row + its arm B counterpart -> one merged row.

    Copies, never recomputes: every surviving arm A value rides through
    byte-identical (the C27 rule). Arm B boxes are appended after the surviving
    arm A boxes; `suppressed_by` indices in the ledger point into the MERGED
    arrays.
    """
    for key in FRAME_IDENTITY_KEYS:
        if row_a.get(key) != row_b.get(key):
            raise MergeContractError(
                f"row identity mismatch on {key!r}: arm A {row_a.get(key)!r} vs "
                f"arm B {row_b.get(key)!r} — these rows do not describe the same frame"
            )
    leaked = sorted({n for n in row_b["class_names"] if n not in ARM_B_PHRASES})
    if leaked:
        raise MergeContractError(
            f"arm B row {row_b['sample_data_token']} emits {leaked}: the ship filter "
            f"leaked a non-shipped class; refusing to arbitrate on arm A's own turf"
        )

    boxes_a = [list(b) for b in row_a["boxes_xyxy_px"]]
    boxes_b = [list(b) for b in row_b["boxes_xyxy_px"]]
    n_a, n_b = len(boxes_a), len(boxes_b)

    n_kept_both = 0
    n_out_of_table = 0
    suppressed: dict[int, tuple[int, float]] = {}  # arm A index -> (arm B index, IoU)
    if n_a and n_b:
        iou = pairwise_iou(
            np.asarray(boxes_a, dtype=np.float32), np.asarray(boxes_b, dtype=np.float32)
        )
        for i in range(n_a):
            j = int(np.argmax(iou[i]))
            best = float(iou[i, j])
            if best <= iou_threshold:
                continue
            action = ARBITRATION.get(row_a["class_names"][i])
            if action == SUPPRESS_ARM_A:
                suppressed[i] = (j, best)
            elif action == KEEP_BOTH:
                n_kept_both += 1
            else:
                n_out_of_table += 1

    keep_a = [i for i in range(n_a) if i not in suppressed]

    def take(seq, idxs):
        return [seq[i] for i in idxs]

    span_of = {p: list(s) for p, s in zip(caption.phrases, caption.phrase_char_spans)}
    p2c = taxonomy.phrase_to_categories

    merged = dict(row_a)  # shallow: every list we touch is rebuilt below
    merged["boxes_xyxy_px"] = take(boxes_a, keep_a) + boxes_b
    merged["scores"] = take(list(row_a["scores"]), keep_a) + list(row_b["scores"])
    merged["class_names"] = take(list(row_a["class_names"]), keep_a) + list(row_b["class_names"])
    merged["nuscenes_categories"] = (
        take([list(c) for c in row_a["nuscenes_categories"]], keep_a)
        + [list(p2c[n]) for n in row_b["class_names"]]
    )
    merged["phrase_char_spans"] = (
        take([list(s) for s in row_a["phrase_char_spans"]], keep_a)
        + [span_of[n] for n in row_b["class_names"]]
    )
    merged["n_proposals"] = len(merged["boxes_xyxy_px"])
    merged["proposal_arm"] = ["arm_a"] * len(keep_a) + ["arm_b"] * n_b
    for key, fill in STAGE3B_EXTENSIONS.items():
        if key in row_a:
            merged[key] = take(list(row_a[key]), keep_a) + [fill] * n_b

    # The class space widened: the prompt block must name the caption these
    # rows are actually scored against. Arm A spans stay valid because the v2
    # caption is a byte prefix of the v3 caption (asserted by the driver).
    merged["prompt"] = {
        **row_a["prompt"],
        "caption_sha256": caption.sha256,
        "taxonomy_sha256": taxonomy.sha256,
    }
    merged["merge"] = {
        "spec": STAGE_SPEC,
        "iou_threshold": float(iou_threshold),
        "n_arm_a_in": n_a,
        "n_arm_b_in": n_b,
        "n_suppressed_arm_a": len(suppressed),
        "n_kept_both": n_kept_both,
        "n_overlap_out_of_table": n_out_of_table,
        "suppressed_arm_a": [
            {
                "index_in_arm_a": i,
                "box_xyxy_px": boxes_a[i],
                "score": row_a["scores"][i],
                "class_name": row_a["class_names"][i],
                "suppressed_by": len(keep_a) + j,   # index in MERGED arrays
                "iou": round(best, 4),
            }
            for i, (j, best) in sorted(suppressed.items())
        ],
    }
    return merged
