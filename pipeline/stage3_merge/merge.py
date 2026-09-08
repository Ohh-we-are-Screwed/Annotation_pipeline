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

# C34: arm A phrases that keep authority over their OWN box when arm A is
# confident. The RSUD20K arm over-calls `a rickshaw` on plain bicycles: on
# pilot_1632 (2026-09-02) every one of the 22 suppressed bicycles sat at or
# above 0.40 and every suppressor was `a rickshaw`. An arm A box whose phrase
# is listed here, scored at or above the value, is never suppressed; every arm
# B box contesting it leaves the arrays for `merge.suppressed_arm_b` and takes
# no further part in the arbitration. The C28 table is untouched below the
# floor. CLI: --protect-arm-a "a bicycle:0.40" (repeatable) or
# --no-protect-arm-a for C28 verbatim.
PROTECTED_ARM_A: dict[str, float] = {"a bicycle": 0.40}

# C36 (2026-09-06): rickshaw PARTS are not bicycles. A rickshaw's front wheel
# comes back from arm A as `a bicycle` / `a motorcycle` INSIDE the arm B
# rickshaw box and survives C28/C34, which arbitrate by IoU — a wheel-sized
# box inside a rickshaw-sized box has IoU ~0.15 however completely it is
# contained. Operator's rule: "if any bicycle or motorcycle is over 60 % of a
# rickshaw, discard it", read as the fraction of the PART box's own area
# covered by a surviving rickshaw / auto-rickshaw box. Runs after the C28/C34
# passes, against arm B boxes that survived them. CLI: --suppress-part
# "a bicycle:0.6" (repeatable) or --no-suppress-parts.
PART_SUPPRESSION: dict[str, float] = {"a bicycle": 0.6, "a motorcycle": 0.6}


def parse_part_args(values) -> dict[str, float]:
    """`--suppress-part PHRASE:MIN_OVERLAP` -> {phrase: floor}. Only arm A
    phrases the table knows can be parts; the floor is a fraction in [0, 1]."""
    out: dict[str, float] = {}
    for item in values or ():
        if ":" not in item:
            raise ValueError(f"--suppress-part {item!r}: expected PHRASE:MIN_OVERLAP")
        phrase, _, floor = item.rpartition(":")
        if phrase not in ARBITRATION:
            raise ValueError(f"--suppress-part {item!r}: {phrase!r} is not an arm A phrase; "
                             f"choose from {sorted(ARBITRATION)}")
        try:
            value = float(floor)
        except ValueError:
            raise ValueError(f"--suppress-part {item!r}: {floor!r} is not a number") from None
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--suppress-part {item!r}: the floor must lie in [0, 1]")
        out[phrase] = value
    return out


def _containment(part, whole) -> float:
    """Fraction of `part`'s area covered by `whole` (xyxy boxes). 0 for an
    empty part box."""
    ax0, ay0, ax1, ay1 = part
    bx0, by0, bx1, by1 = whole
    area = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    if area <= 0.0:
        return 0.0
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    return (iw * ih) / area


def parse_protect_args(values) -> dict[str, float]:
    """`PHRASE:MIN_SCORE` items -> {phrase: inclusive score floor}.

    Refuses a phrase the table does not suppress (protection is meaningless
    where no contest can remove arm A) and a floor outside [0, 1].
    """
    suppressible = sorted(k for k, v in ARBITRATION.items() if v == SUPPRESS_ARM_A)
    out: dict[str, float] = {}
    for item in values or ():
        phrase, sep, score = str(item).rpartition(":")
        if not sep or not phrase:
            raise ValueError(f"--protect-arm-a {item!r}: expected PHRASE:MIN_SCORE")
        if ARBITRATION.get(phrase) != SUPPRESS_ARM_A:
            raise ValueError(f"--protect-arm-a {item!r}: {phrase!r} is not an arm A phrase "
                             f"the table suppresses; choose from {suppressible}")
        try:
            floor = float(score)
        except ValueError:
            raise ValueError(f"--protect-arm-a {item!r}: {score!r} is not a number") from None
        if not 0.0 <= floor <= 1.0:
            raise ValueError(f"--protect-arm-a {item!r}: the floor must lie in [0, 1]")
        out[phrase] = floor
    return out

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


def merge_rows(row_a: dict, row_b: dict, *, caption, taxonomy, iou_threshold: float,
               protected: dict | None = None, part_floor: dict | None = None) -> dict:
    """One arm A row + its arm B counterpart -> one merged row.

    Copies, never recomputes: every surviving arm A value rides through
    byte-identical (the C27 rule). Surviving arm B boxes are appended after the
    surviving arm A boxes; `suppressed_by` / `absorbed_by` / `protected_by`
    indices in the ledgers point into the MERGED arrays, except that
    `protected_by` is None when C36 later absorbed the protecting arm A box —
    that entry carries `protected_survived: false` and `protected_removed_by`
    instead. `protected` (C34) maps an arm A phrase to the score at or above
    which arm A keeps its box; None means the module default, {} means C28
    verbatim.
    """
    protected = PROTECTED_ARM_A if protected is None else dict(protected)
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
    vetoed: dict[int, tuple[int, float]] = {}      # arm B index -> (arm A index, IoU)
    protectors: set[int] = set()

    def _is_protected(i: int) -> bool:
        floor = protected.get(row_a["class_names"][i])
        return floor is not None and float(row_a["scores"][i]) >= floor

    if n_a and n_b:
        iou = pairwise_iou(
            np.asarray(boxes_a, dtype=np.float32), np.asarray(boxes_b, dtype=np.float32)
        )
        # C34 pass first: a confident protected arm A box removes EVERY arm B
        # box contesting it from the arbitration. A vetoed box then contests
        # nothing else — otherwise a neighbour it also overlapped would be
        # suppressed by a box that is no longer in the row, and the object
        # would vanish from both arms.
        for i in range(n_a):
            if not _is_protected(i):
                continue
            for j in np.flatnonzero(iou[i] > iou_threshold):
                j = int(j)
                protectors.add(i)
                if j not in vetoed or float(iou[i, j]) > vetoed[j][1]:
                    vetoed[j] = (i, float(iou[i, j]))
        live = np.ones(n_b, dtype=bool)
        if vetoed:
            live[list(vetoed)] = False
        for i in range(n_a):
            if _is_protected(i):
                continue                      # never suppressed, whatever survives
            contest = np.where(live, iou[i], -1.0)
            j = int(np.argmax(contest))
            best = float(contest[j])
            if best <= iou_threshold:
                continue
            action = ARBITRATION.get(row_a["class_names"][i])
            if action == SUPPRESS_ARM_A:
                suppressed[i] = (j, best)
            elif action == KEEP_BOTH:
                n_kept_both += 1
            else:
                n_out_of_table += 1

    keep_b = [j for j in range(n_b) if j not in vetoed]

    # C36 pass, after C28/C34: an arm A part-class box mostly covered by a
    # SURVIVING arm B rickshaw box is that rickshaw's wheel, not a bicycle.
    # Containment, not IoU (see PART_SUPPRESSION). A protected bicycle that
    # vetoed its rickshaw has nothing left to absorb it.
    part_floor = PART_SUPPRESSION if part_floor is None else dict(part_floor)
    absorbed: dict[int, tuple[int, float]] = {}   # arm A index -> (arm B index, overlap)
    authorities = [j for j in keep_b if row_b["class_names"][j] in ARM_B_PHRASES]
    if part_floor and authorities:
        for i in range(n_a):
            if i in suppressed:
                continue
            floor = part_floor.get(row_a["class_names"][i])
            if floor is None:
                continue
            best_j, best_ov = None, 0.0
            for j in authorities:
                ov = _containment(boxes_a[i], boxes_b[j])
                if ov > best_ov:
                    best_j, best_ov = j, ov
            if best_j is not None and best_ov >= floor:
                absorbed[i] = (best_j, best_ov)

    keep_a = [i for i in range(n_a) if i not in suppressed and i not in absorbed]
    pos_a = {i: k for k, i in enumerate(keep_a)}   # arm A index -> MERGED index
    pos_b = {j: k for k, j in enumerate(keep_b)}   # arm B index -> offset after arm A

    # C34 x C36 (2026-09-08): a PROTECTING arm A box can itself leave the
    # arrays. The C28 pass never removes one — it skips every protected box —
    # so C36 absorption is the only route: a confident bicycle vetoes the
    # rickshaw it overlaps (IoU > threshold) and is then absorbed as a part of
    # a DIFFERENT, larger rickshaw that never contested it (containment 0.977
    # at IoU 0.076 on the frame that found this: chunk_0000 CAM_BACK
    # a2cc8778cb09e6152d6ac302e1c88d8e, arm A box 15; 2 rows in 8,184). The
    # veto still happened and the arm B box is still gone, so the entry stays
    # in the ledger — but `protected_by` cannot name a merged box, and must not
    # point at whatever box now sits at that index. The entry therefore always
    # carries the protector's ORIGINAL arm A index, says whether it survived,
    # and names what removed it: "protected by a box in the output" and
    # "protected by a box that was itself removed" are different facts and a
    # consumer must be able to read them apart.
    def _protector_of(i: int) -> tuple:
        """(index in MERGED arrays or None, survived, removal record or None)."""
        if i in pos_a:
            return pos_a[i], True, None
        if i in absorbed:
            j, ov = absorbed[i]
            return None, False, {
                "reason": "absorbed_as_part",
                "absorbed_by": len(keep_a) + pos_b[j],   # index in MERGED arrays
                "absorbed_class_name": row_b["class_names"][j],
                "overlap": round(ov, 4),
            }
        raise MergeContractError(                        # unreachable by construction
            f"arm A box {i} vetoed an arm B box but left the merged arrays by no "
            "recorded route: the merge cannot say what removed it"
        )

    vetoed_ledger = []
    for j, (i, best) in sorted(vetoed.items()):
        merged_i, survived, removed_by = _protector_of(i)
        vetoed_ledger.append({
            "index_in_arm_b": j,
            "box_xyxy_px": boxes_b[j],
            "score": row_b["scores"][j],
            "class_name": row_b["class_names"][j],
            "protected_index_in_arm_a": i,           # index in ARM A, always present
            "protected_by": merged_i,                # index in MERGED arrays, or None
            "protected_survived": survived,          # False -> the protector was
            "protected_removed_by": removed_by,      # itself removed (C36); see above
            "protected_class_name": row_a["class_names"][i],
            "protected_score": row_a["scores"][i],
            "iou": round(best, 4),
        })
    # Of the protectors counted in n_protected_arm_a, those not in the output.
    removed_protectors = sorted(protectors - set(pos_a))

    def take(seq, idxs):
        return [seq[i] for i in idxs]

    names_b = take(list(row_b["class_names"]), keep_b)

    span_of = {p: list(s) for p, s in zip(caption.phrases, caption.phrase_char_spans)}
    p2c = taxonomy.phrase_to_categories

    merged = dict(row_a)  # shallow: every list we touch is rebuilt below
    merged["boxes_xyxy_px"] = take(boxes_a, keep_a) + take(boxes_b, keep_b)
    merged["scores"] = take(list(row_a["scores"]), keep_a) + take(list(row_b["scores"]), keep_b)
    merged["class_names"] = take(list(row_a["class_names"]), keep_a) + names_b
    merged["nuscenes_categories"] = (
        take([list(c) for c in row_a["nuscenes_categories"]], keep_a)
        + [list(p2c[n]) for n in names_b]
    )
    merged["phrase_char_spans"] = (
        take([list(s) for s in row_a["phrase_char_spans"]], keep_a)
        + [span_of[n] for n in names_b]
    )
    merged["n_proposals"] = len(merged["boxes_xyxy_px"])
    merged["proposal_arm"] = ["arm_a"] * len(keep_a) + ["arm_b"] * len(keep_b)
    for key, fill in STAGE3B_EXTENSIONS.items():
        if key in row_a:
            merged[key] = take(list(row_a[key]), keep_a) + [fill] * len(keep_b)

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
        "n_protected_arm_a": len(protectors),
        # ... of which this many were themselves removed afterwards (C36) and
        # so are NOT in the merged arrays, however many arm B boxes they vetoed.
        "n_protected_arm_a_removed": len(removed_protectors),
        "n_suppressed_arm_b": len(vetoed),
        # C36: arm A part-class boxes absorbed by a surviving arm B rickshaw.
        "part_floor": {k: float(v) for k, v in part_floor.items()},
        "n_suppressed_parts": len(absorbed),
        "suppressed_parts": [
            {
                "index_in_arm_a": i,
                "box_xyxy_px": boxes_a[i],
                "score": row_a["scores"][i],
                "class_name": row_a["class_names"][i],
                # index in MERGED arrays: j is drawn from keep_b, and no pass
                # after the C34 veto removes an arm B box, so it is always there.
                "absorbed_by": len(keep_a) + pos_b[j],
                "absorbed_class_name": row_b["class_names"][j],
                "overlap": round(ov, 4),
            }
            for i, (j, ov) in sorted(absorbed.items())
        ],
        "suppressed_arm_a": [
            {
                "index_in_arm_a": i,
                "box_xyxy_px": boxes_a[i],
                "score": row_a["scores"][i],
                "class_name": row_a["class_names"][i],
                # index in MERGED arrays: j won an argmax over `live` (== keep_b,
                # vetoed boxes scored -1.0), so it is a surviving arm B box.
                "suppressed_by": len(keep_a) + pos_b[j],
                "iou": round(best, 4),
            }
            for i, (j, best) in sorted(suppressed.items())
        ],
        # C34: arm B boxes a confident protected arm A box removed from the row.
        # Built above, because a protector can itself have been absorbed (C36).
        "suppressed_arm_b": vetoed_ledger,
    }
    return merged


def _read_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(
    arm_a_dir: str,
    arm_b_dir: str,
    out_dir: str,
    taxonomy_path: str,
    *,
    iou_threshold: float = 0.5,
    accept_degraded: bool = False,
    protected_arm_a: dict | None = None,
    part_floor: dict | None = None,
) -> int:
    protected_arm_a = PROTECTED_ARM_A if protected_arm_a is None else dict(protected_arm_a)
    part_floor = PART_SUPPRESSION if part_floor is None else dict(part_floor)
    taxonomy = load_taxonomy(taxonomy_path)
    caption = build_caption(taxonomy.phrases)

    man_a, marker_a = require_upstream(
        arm_a_dir, stage_name="Stage 3 (arm A)",
        module_hint="pipeline.stage3_proposals.proposals",
        accept_degraded=accept_degraded,
    )
    # Cross-bound to arm A's substrate fingerprint: the two trees must descend
    # from the same Stage 1 output, or the row pairing compares different worlds.
    man_b, marker_b = require_upstream(
        arm_b_dir, stage_name="Stage 3 (arm B)",
        module_hint="pipeline.stage3_proposals.proposals",
        current_fingerprint=marker_a.fingerprint,
        accept_degraded=accept_degraded,
    )

    cap_a = str(man_a["prompt"]["caption"])
    if not caption.text.startswith(cap_a):
        raise UpstreamRefusal(
            f"arm A's caption is not a prefix of {taxonomy_path}'s caption: arm A spans "
            "would be invalid under the merged class space. The v3 taxonomy must extend "
            "the arm A taxonomy by APPENDING phrases only"
        )
    cap_b = str(man_b["prompt"]["caption"])
    if cap_b != caption.text:
        raise UpstreamRefusal(
            f"arm B ran under a different caption than {taxonomy_path}: re-run arm B "
            "with --taxonomy pointing at the same file the merge uses"
        )
    # Stage 4 cross-checks the upstream manifest's resolution against its own
    # (masks.py:1811) and refuses a mismatch, so the merged manifest must carry
    # it. Both arms must agree first: boxes measured at two resolutions are not
    # comparable, and the pixel IoU that drives arbitration would be meaningless.
    size_a = man_a.get("image_size_px")
    size_b = man_b.get("image_size_px")
    if not size_a:
        raise UpstreamRefusal(
            f"arm A's manifest carries no image_size_px; Stage 4 refuses an upstream "
            "that does not state the resolution its boxes were measured at"
        )
    if size_a != size_b:
        raise UpstreamRefusal(
            f"arms ran at different resolutions (arm A {size_a}, arm B {size_b}): "
            "their boxes are not comparable and the arbitration IoU would be meaningless"
        )

    b_in_use = tuple((man_b.get("class_map") or {}).get("phrases_in_use") or ())
    stray = sorted(set(b_in_use) - set(ARM_B_PHRASES))
    if stray:
        raise UpstreamRefusal(
            f"arm B's class map puts {stray} in use; arm B may only ship {ARM_B_PHRASES}"
        )

    root_a = os.path.join(arm_a_dir, "scenes")
    root_b = os.path.join(arm_b_dir, "scenes")
    scenes_a = sorted(os.listdir(root_a)) if os.path.isdir(root_a) else []
    scenes_b = sorted(os.listdir(root_b)) if os.path.isdir(root_b) else []
    if scenes_a != scenes_b or not scenes_a:
        raise UpstreamRefusal(
            f"scene sets differ (arm A {scenes_a} vs arm B {scenes_b}): the merge "
            "pairs rows frame-by-frame and cannot invent an absent arm"
        )

    clear_markers(out_dir)
    totals = {"n_rows": 0, "n_arm_a_in": 0, "n_arm_b_in": 0, "n_suppressed_arm_a": 0,
              "n_kept_both": 0, "n_overlap_out_of_table": 0,
              "n_protected_arm_a": 0, "n_protected_arm_a_removed": 0,
              "n_suppressed_arm_b": 0, "n_suppressed_parts": 0,
              "n_out": 0}
    per_scene: dict[str, dict] = {}
    for scene in scenes_a:
        rows_a = _read_rows(os.path.join(root_a, scene, "proposals.jsonl"))
        rows_b = _read_rows(os.path.join(root_b, scene, "proposals.jsonl"))
        index_b = {(r["keyframe_token"], r["channel"]): r for r in rows_b}
        if len(index_b) != len(rows_b):
            raise MergeContractError(f"{scene}: duplicate (keyframe, channel) rows in arm B")
        merged_rows = []
        seen = set()
        for row_a in rows_a:  # arm A's row order IS the output row order (C27)
            key = (row_a["keyframe_token"], row_a["channel"])
            if key not in index_b:
                raise MergeContractError(f"{scene}: arm B has no row for {key}")
            seen.add(key)
            merged = merge_rows(row_a, index_b[key], caption=caption,
                                taxonomy=taxonomy, iou_threshold=iou_threshold,
                                protected=protected_arm_a, part_floor=part_floor)
            led = merged["merge"]
            totals["n_rows"] += 1
            for k in ("n_arm_a_in", "n_arm_b_in", "n_suppressed_arm_a",
                      "n_kept_both", "n_overlap_out_of_table",
                      "n_protected_arm_a", "n_protected_arm_a_removed",
                      "n_suppressed_arm_b", "n_suppressed_parts"):
                totals[k] += led[k]
            totals["n_out"] += merged["n_proposals"]
            merged_rows.append(merged)
        extra = set(index_b) - seen
        if extra:
            raise MergeContractError(f"{scene}: arm B rows with no arm A counterpart: {sorted(extra)}")
        write_jsonl_atomic(os.path.join(out_dir, "scenes", scene, "proposals.jsonl"), merged_rows)
        per_scene[scene] = {"n_rows": len(merged_rows)}

    in_use_a = tuple((man_a.get("class_map") or {}).get("phrases_in_use") or ())
    union = [p for p in caption.phrases if p in set(in_use_a) | set(b_in_use)]
    manifest = {
        "spec": STAGE_SPEC,
        "stage": STAGE,
        "provider": "yolo11_two_arm_merge",
        "score_semantics": "yolo_class_confidence (both arms; scores are never compared across arms)",
        "upstream": {
            "metadata_fingerprint": marker_a.fingerprint,
            "fingerprint_spec": man_a["upstream"]["fingerprint_spec"],
            "arm_a": {"dir": os.path.realpath(arm_a_dir), "spec": man_a.get("spec"),
                      "checkpoint": man_a.get("checkpoint"), "degraded": marker_a.degraded,
                      "degraded_causes": list(marker_a.causes)},
            "arm_b": {"dir": os.path.realpath(arm_b_dir), "spec": man_b.get("spec"),
                      "checkpoint": man_b.get("checkpoint"), "degraded": marker_b.degraded,
                      "degraded_causes": list(marker_b.causes)},
            "accepted_degraded_upstream": accept_degraded,
        },
        "taxonomy": taxonomy.as_dict(),
        # Carried from arm A (== arm B, asserted above): Stage 4 reads this key
        # and refuses the tree without it.
        "image_size_px": list(size_a),
        "prompt": {
            "caption": caption.text,
            "caption_sha256": caption.sha256,
            "caption_is_input": False,
            "phrases": list(caption.phrases),
        },
        "class_map": {
            # C25 reads this block: the merged run's reachable set is the UNION.
            "path": f"merge({(man_a.get('class_map') or {}).get('path')}, "
                    f"{(man_b.get('class_map') or {}).get('path')})",
            "sha256": "",
            "n_mapped": len(union),
            "phrases_in_use": union,
            "unreachable_phrases": [p for p in caption.phrases if p not in set(union)],
            "arm_a_class_map": man_a.get("class_map"),
            "arm_b_class_map": man_b.get("class_map"),
        },
        "arbitration": {
            "table": dict(ARBITRATION),
            "iou_threshold": float(iou_threshold),
            # C34: arm A phrase -> inclusive score floor above which arm A keeps
            # its own box and the contesting arm B box is dropped. {} = C28 verbatim.
            "protected_arm_a": {k: float(v) for k, v in protected_arm_a.items()},
            # C36: arm A part phrase -> min fraction of ITS area a surviving arm B
            # rickshaw must cover for it to be absorbed as that rickshaw's part.
            "part_floor": {k: float(v) for k, v in part_floor.items()},
            "provenance": "docs/RUNNING.md two-arm design 2026-08-26; DECISIONS C28 (table), "
                          "C34 (protected_arm_a), C36 (part_floor, operator 2026-09-06: a "
                          "bicycle/motorcycle >= 60 % covered by a rickshaw is its wheel). "
                          "Vocabulary authority, never score (C21), except that a protected "
                          "arm A phrase at or above its floor is never suppressed. "
                          "iou_threshold and the floors UNVALIDATED.",
        },
        "totals": totals,
        "scenes": per_scene,
    }
    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)

    degraded = marker_a.degraded or marker_b.degraded
    causes = tuple(f"arm_a: {c}" for c in marker_a.causes) + tuple(
        f"arm_b: {c}" for c in marker_b.causes)
    write_marker(out_dir, marker_a.fingerprint, degraded=degraded, causes=causes)
    print(f"stage3_merge: {totals['n_rows']} rows, {totals['n_out']} boxes out, "
          f"{totals['n_suppressed_arm_a']} arm A suppressed, "
          f"{totals['n_kept_both']} kept-both, "
          f"{totals['n_overlap_out_of_table']} out-of-table overlaps, "
          f"{totals['n_suppressed_arm_b']} arm B dropped under "
          f"{totals['n_protected_arm_a']} protected arm A boxes "
          f"(protect: {protected_arm_a or 'none'})")
    if totals["n_protected_arm_a_removed"]:
        print(f"stage3_merge: {totals['n_protected_arm_a_removed']} of those protected arm A "
              f"boxes were themselves absorbed as rickshaw parts (C36) and are not in the "
              f"output; their suppressed_arm_b entries carry protected_survived false")
    return 1 if degraded else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a-dir", required=True,
                        help="stage3_proposals or stage3b_track2d tree (arm A)")
    parser.add_argument("--arm-b-dir", required=True,
                        help="arm B stage3 tree (stage3_finetuned)")
    parser.add_argument("--out-dir", required=True, help="stage3_merged tree to write")
    parser.add_argument("--taxonomy", default="configs/taxonomy_pilot_dhaka.yaml")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="overlap that makes a pair a contest (recorded; unvalidated)")
    parser.add_argument("--accept-degraded-upstream", action="store_true")
    parser.add_argument("--protect-arm-a", action="append", default=None,
                        metavar="PHRASE:MIN_SCORE",
                        help="C34: arm A phrase that keeps its box at or above this score; "
                             "repeatable. Default: "
                             + ", ".join(f"'{k}:{v}'" for k, v in PROTECTED_ARM_A.items()))
    parser.add_argument("--no-protect-arm-a", action="store_true",
                        help="C28 verbatim: no arm A phrase is protected")
    parser.add_argument("--suppress-part", action="append", default=None, metavar="PHRASE:MIN_OVERLAP",
                        help="C36: an arm A PHRASE box whose own area is covered >= MIN_OVERLAP by a "
                             "surviving arm B rickshaw box is that rickshaw's part and is dropped "
                             "(repeatable; default 'a bicycle:0.6' 'a motorcycle:0.6')")
    parser.add_argument("--no-suppress-parts", action="store_true",
                        help="disable C36: keep every part-class box")
    args = parser.parse_args(argv)
    if args.no_suppress_parts and args.suppress_part:
        print("REFUSED: --no-suppress-parts and --suppress-part contradict each other", file=sys.stderr)
        return 2
    try:
        part_floor = ({} if args.no_suppress_parts
                      else PART_SUPPRESSION if args.suppress_part is None
                      else parse_part_args(args.suppress_part))
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    if args.no_protect_arm_a and args.protect_arm_a:
        print("REFUSED: --no-protect-arm-a and --protect-arm-a contradict each other",
              file=sys.stderr)
        return 2
    try:
        protected = ({} if args.no_protect_arm_a
                     else PROTECTED_ARM_A if args.protect_arm_a is None
                     else parse_protect_args(args.protect_arm_a))
    except ValueError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    try:
        return run(
            args.arm_a_dir, args.arm_b_dir, args.out_dir, args.taxonomy,
            iou_threshold=args.iou_threshold,
            accept_degraded=args.accept_degraded_upstream,
            protected_arm_a=protected,
            part_floor=part_floor,
        )
    except (UpstreamRefusal, MergeContractError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
