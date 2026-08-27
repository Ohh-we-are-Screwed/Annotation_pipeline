#!/usr/bin/env python3
"""2D detection test: OUR pipeline boxes vs the nuScenes human answer key.

Compares the two COCO exports already on disk — `cvat_export/` (Stage 3+4
output; kept shapes only) against `cvat_export_gt/` (human GT projected 3D->2D)
— image by image, matched by `file_name`. Greedy IoU matching at 0.5, predictions
sorted by score, in two modes:

`cvat_export/` is a CVAT REVIEW export and is NOT a detection result set: every
kept proposal appears in it twice, as the Stage 3 rectangle and the Stage 4 mask
polygon, sharing one bbox (that pairing is the point of the review task). v1 of
this script counted both, so 10 433 kept boxes arrived as 20 866 "predictions",
the twins competed for the same GT under greedy matching, and precision could
not exceed 50%. `predictions_of()` now keeps one row per proposal and records
the drop in `dedup`.

  localization   a prediction matching ANY GT box counts (was there an object?)
  class-aware    the GT box's category must map to the predicted phrase too

Both are reported because the pilot's known failure split is exactly there: the
under-tier detector places boxes well and names them badly (§13.2; see also
paint_metrics.json, where "a police car" is 88% inside-GT and 0% class-correct).

GT recall is additionally split by GT box size — the projected answer key
includes every human-labelled object at any distance and occlusion, which no
2D detector at 1600x900 could fully recall; area >= 32x32 px approximates
"visibly present".

Results are reported at an IoU SWEEP (0.3 / 0.5 / 0.75) as well as the single
`--iou` gate, because the answer key is amodal — each GT box is the clipped AABB
of a projected 3D cuboid, occluded extent included — while the detector emits
modal boxes around visible pixels. That mismatch depresses IoU systematically, so
a single 0.5 gate cannot separate "wrong object" from "right object, loose box".

Descriptive on 10 scenes, never a measurement (§0.2). Writes
<work_root>/metrics/detect2d_metrics.json.

    python -m scripts.eval_2d [--scenes scene-0061] [--iou 0.5]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.manifest import write_json_atomic  # noqa: E402
from pipeline.common.paths import load_paths  # noqa: E402

MIN_VISIBLE_AREA_PX = 32 * 32

# Reported alongside the primary --iou gate. One threshold cannot distinguish
# "the box is on the wrong object" from "the box is on the right object but
# loose", and that distinction is the whole question here: the GT twins are
# AMODAL (the clipped AABB of the projected 3D cuboid, occluded extent included)
# while the detector emits MODAL boxes around visible pixels, so IoU is depressed
# systematically rather than randomly. A number that climbs steeply from 0.5 to
# 0.3 is a looseness/amodal story; one that stays flat is a placement story.
IOU_SWEEP = (0.3, 0.5, 0.75)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, M) IoU of xywh boxes."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    ax1, ay1 = a[:, 0], a[:, 1]
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    ix = np.maximum(
        0.0, np.minimum(ax2[:, None], bx2[None]) - np.maximum(ax1[:, None], bx1[None])
    )
    iy = np.maximum(
        0.0, np.minimum(ay2[:, None], by2[None]) - np.maximum(ay1[:, None], by1[None])
    )
    inter = ix * iy
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    return inter / np.maximum(union, 1e-9)


def by_image(doc: dict) -> tuple[dict, dict]:
    """file_name -> annotation rows; category_id -> name."""
    names = {c["id"]: c["name"] for c in doc["categories"]}
    file_of = {img["id"]: img["file_name"] for img in doc["images"]}
    rows = defaultdict(list)
    for ann in doc["annotations"]:
        rows[file_of[ann["image_id"]]].append(ann)
    return rows, names


def predictions_of(rows, ledger: dict) -> list[dict]:
    """The detections on one image — ONE row per proposal, score descending.

    `cvat_export/` is a CVAT REVIEW export, not a detection result set, and the
    difference is a factor of two. scripts/export_cvat_coco.py deliberately emits
    each kept proposal TWICE — once as a rectangle (the Stage 3 box, "the
    detector's claim") and once as a polygon (the Stage 4 mask) — so a human
    reviewer sees box-vs-mask disagreement. Both rows carry the SAME bbox.

    Counting both as predictions is what produced 20 866 "detections" from 10 433
    kept boxes. Under greedy matching the twins compete for one GT box: one wins,
    the other is a false positive by construction, and precision is capped at 50%
    before any detector quality is measured. So the polygon twin is dropped here.

    Nothing geometric is lost — the twin's bbox is identical; only its
    `segmentation` differs, and this file never looks at segmentation.
    """
    kept = []
    seen: set = set()
    for row in rows:
        ledger["annotations_read"] += 1
        if row.get("attributes", {}).get("suppressed"):
            # Stage 4's cross-camera contest already removed these; they are not
            # part of the pipeline's answer.
            ledger["suppressed_dropped"] += 1
            continue
        if row.get("segmentation"):
            ledger["mask_twins_dropped"] += 1
            continue
        # Emptiness of `segmentation` is NOT a reliable twin test on its own: a
        # mask whose contours all fall under mask_to_polygons()'s 20 px floor
        # yields a polygon row with an empty segmentation, indistinguishable from
        # the rectangle. That let 45 twins through on the first run (10 478
        # counted against 10 433 kept). The identity test is the tuple the twins
        # share by construction — same box, same class, same score.
        key = (tuple(row["bbox"]), row["category_id"],
               row.get("attributes", {}).get("score"))
        if key in seen:
            ledger["mask_twins_dropped"] += 1
            continue
        seen.add(key)
        kept.append(row)
    kept.sort(key=lambda p: -p.get("attributes", {}).get("score", 0.0))
    ledger["predictions_counted"] += len(kept)
    return kept


def greedy_match(ious: np.ndarray, pred_classes, gt_classes, threshold: float):
    """Greedy IoU matching at one threshold; predictions already score-sorted.

    Returns (tp_localization, tp_class_aware, gt_hit_mask). Two independent
    matchings, because the pilot's known failure split is exactly here: a box on
    a real object with the wrong name is a localization hit and a class miss.
    Each GT box may be claimed once per mode.
    """
    n_pred, n_gt = ious.shape
    taken_loc = np.zeros(n_gt, dtype=bool)
    taken_cls = np.zeros(n_gt, dtype=bool)
    tp_loc = np.zeros(n_pred, dtype=bool)
    tp_cls = np.zeros(n_pred, dtype=bool)
    if not n_pred or not n_gt:
        return tp_loc, tp_cls, taken_loc

    for i in range(n_pred):
        order = np.argsort(-ious[i])
        for j in order:
            if ious[i, j] < threshold:
                break
            if not taken_loc[j]:
                taken_loc[j] = True
                tp_loc[i] = True
                break
        for j in order:
            if ious[i, j] < threshold:
                break
            if not taken_cls[j] and gt_classes[j] == pred_classes[i]:
                taken_cls[j] = True
                tp_cls[i] = True
                break
    return tp_loc, tp_cls, taken_loc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--pred-export", default="cvat_export",
                        help="prediction export dir under work_root (default cvat_export; "
                             "e.g. cvat_export_fixed from scripts/review_fix_sam31.py). A "
                             "non-default dir writes metrics/detect2d_metrics.<dir>.json so the "
                             "baseline file is never overwritten")
    args = parser.parse_args(argv)

    paths = load_paths(args.paths)
    pred_root = os.path.join(paths.work_root, args.pred_export)
    gt_root = os.path.join(paths.work_root, "cvat_export_gt")
    names = sorted(
        n for n in os.listdir(pred_root)
        if os.path.isfile(os.path.join(gt_root, n, "instances.json"))
    )
    if args.scenes:
        names = [n for n in names if n in args.scenes]

    n_pred = 0
    per_class = defaultdict(lambda: {"pred": 0, "tp_loc": 0, "tp_cls": 0})
    gt_total = gt_total_visible = 0
    dedup = {"annotations_read": 0, "suppressed_dropped": 0,
             "mask_twins_dropped": 0, "predictions_counted": 0}
    # One accumulator per threshold in the sweep; the primary gate is included
    # so the headline numbers and the sweep can never disagree.
    thresholds = sorted({*IOU_SWEEP, args.iou})
    acc = {t: {"tp_loc": 0, "tp_cls": 0, "gt_hit": 0, "gt_hit_visible": 0} for t in thresholds}

    for scene in names:
        with open(os.path.join(pred_root, scene, "instances.json")) as fh:
            pred_rows, pred_names = by_image(json.load(fh))
        with open(os.path.join(gt_root, scene, "instances.json")) as fh:
            gt_rows, gt_names = by_image(json.load(fh))

        for file_name in set(pred_rows) | set(gt_rows):
            preds = predictions_of(pred_rows.get(file_name, ()), dedup)
            gts = gt_rows.get(file_name, ())
            pred_boxes = np.array([p["bbox"] for p in preds], dtype=np.float64).reshape(-1, 4)
            gt_boxes = np.array([g["bbox"] for g in gts], dtype=np.float64).reshape(-1, 4)
            gt_areas = gt_boxes[:, 2] * gt_boxes[:, 3] if len(gts) else np.zeros(0)
            # The IoU matrix does not depend on the threshold, so it is computed
            # once and every gate in the sweep is matched against it.
            ious = iou_matrix(pred_boxes, gt_boxes)
            pred_classes = [pred_names[p["category_id"]] for p in preds]
            gt_classes = [gt_names[g["category_id"]] for g in gts]
            visible = gt_areas >= MIN_VISIBLE_AREA_PX

            n_pred += len(preds)
            for cls in pred_classes:
                per_class[cls]["pred"] += 1
            gt_total += len(gts)
            gt_total_visible += int(visible.sum())

            for t in thresholds:
                tp_loc, tp_cls, taken_loc = greedy_match(ious, pred_classes, gt_classes, t)
                acc[t]["tp_loc"] += int(tp_loc.sum())
                acc[t]["tp_cls"] += int(tp_cls.sum())
                acc[t]["gt_hit"] += int(taken_loc.sum())
                acc[t]["gt_hit_visible"] += int((taken_loc & visible).sum())
                # Per-class detail is reported at the primary gate only.
                if t == args.iou:
                    for i, cls in enumerate(pred_classes):
                        if tp_loc[i]:
                            per_class[cls]["tp_loc"] += 1
                        if tp_cls[i]:
                            per_class[cls]["tp_cls"] += 1

    primary = acc[args.iou]
    tp_loc, tp_cls = primary["tp_loc"], primary["tp_cls"]
    gt_hit, gt_hit_visible = primary["gt_hit"], primary["gt_hit_visible"]

    report = {
        # v2: predictions are deduplicated against the review export's box+mask
        # pairing, and an IoU sweep accompanies the single gate. Both change the
        # numbers, so the spec string changes with them (§1.9).
        "spec": "dhakascenes-pilot/detect2d-metrics/v2",
        "pred_export": args.pred_export,
        "scenes": names,
        "iou_threshold": args.iou,
        "n_predictions": n_pred,
        # The audit trail for the factor of two. `predictions_counted` must equal
        # Stage 4's n_kept; if it is 2x that, the dedup stopped working.
        "dedup": {
            **dedup,
            "note": (
                "cvat_export is a CVAT review export: each kept proposal appears as a "
                "rectangle (Stage 3 box) AND a polygon (Stage 4 mask) with the same bbox. "
                "Only the rectangle is counted as a detection; counting both capped "
                "precision at 50% (detect2d-metrics/v1 did exactly that)"
            ),
        },
        "precision_localization": round(tp_loc / max(1, n_pred), 4),
        "precision_class_aware": round(tp_cls / max(1, n_pred), 4),
        "gt_recall_all": round(gt_hit / max(1, gt_total), 4),
        "gt_recall_visible": round(gt_hit_visible / max(1, gt_total_visible), 4),
        "gt_boxes": {"all": gt_total, "visible_32px": gt_total_visible},
        "iou_sweep": {
            f"{t:g}": {
                "precision_localization": round(acc[t]["tp_loc"] / max(1, n_pred), 4),
                "precision_class_aware": round(acc[t]["tp_cls"] / max(1, n_pred), 4),
                "gt_recall_all": round(acc[t]["gt_hit"] / max(1, gt_total), 4),
                "gt_recall_visible": round(acc[t]["gt_hit_visible"] / max(1, gt_total_visible), 4),
            }
            for t in thresholds
        },
        "per_class": {
            cls: {
                "predictions": row["pred"],
                "precision_localization": round(row["tp_loc"] / max(1, row["pred"]), 4),
                "precision_class_aware": round(row["tp_cls"] / max(1, row["pred"]), 4),
            }
            for cls, row in sorted(per_class.items(), key=lambda kv: -kv[1]["pred"])
        },
        "caveat": (
            "descriptive on 10 scenes (§0.2); GT includes all distances/occlusions, so "
            "gt_recall_all is a harsh denominator by construction"
        ),
    }
    out_name = ("detect2d_metrics.json" if args.pred_export == "cvat_export"
                else f"detect2d_metrics.{args.pred_export}.json")
    out_path = os.path.join(paths.work_root, "metrics", out_name)
    write_json_atomic(out_path, report)

    print(f"predictions          : {n_pred}  (kept boxes, {len(names)} scenes, IoU >= {args.iou})")
    print(f"  dedup              : read {dedup['annotations_read']} annotations, dropped "
          f"{dedup['suppressed_dropped']} suppressed + {dedup['mask_twins_dropped']} mask twins")
    print(f"precision (location) : {report['precision_localization']:.1%}   a real object was there")
    print(f"precision (class)    : {report['precision_class_aware']:.1%}   ...AND the name matched")
    print(f"GT recall (all)      : {report['gt_recall_all']:.1%}   of {gt_total} human boxes")
    print(f"GT recall (>=32px)   : {report['gt_recall_visible']:.1%}   of {gt_total_visible} visibly-sized boxes")
    print("IoU sweep (GT is amodal — a steep climb toward 0.3 means loose boxes, not wrong ones):")
    for t in thresholds:
        row = report["iou_sweep"][f"{t:g}"]
        print(f"  IoU >= {t:<5g} loc {row['precision_localization']:>6.1%}   "
              f"class {row['precision_class_aware']:>6.1%}   recall {row['gt_recall_visible']:>6.1%}")
    print("per class (top 8 by prediction count):")
    for cls, row in list(report["per_class"].items())[:8]:
        print(f"  {cls:<26} {row['predictions']:>5} boxes   loc {row['precision_localization']:.1%}"
              f"   class {row['precision_class_aware']:.1%}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
