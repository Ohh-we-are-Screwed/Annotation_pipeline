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
) -> int:
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
              "n_kept_both": 0, "n_overlap_out_of_table": 0, "n_out": 0}
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
                                taxonomy=taxonomy, iou_threshold=iou_threshold)
            led = merged["merge"]
            totals["n_rows"] += 1
            for k in ("n_arm_a_in", "n_arm_b_in", "n_suppressed_arm_a",
                      "n_kept_both", "n_overlap_out_of_table"):
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
            "provenance": "docs/RUNNING.md two-arm design 2026-08-26; DECISIONS C28. "
                          "Vocabulary authority, never score (C21). iou_threshold UNVALIDATED.",
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
          f"{totals['n_overlap_out_of_table']} out-of-table overlaps")
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
    args = parser.parse_args(argv)
    try:
        return run(
            args.arm_a_dir, args.arm_b_dir, args.out_dir, args.taxonomy,
            iou_threshold=args.iou_threshold,
            accept_degraded=args.accept_degraded_upstream,
        )
    except (UpstreamRefusal, MergeContractError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
