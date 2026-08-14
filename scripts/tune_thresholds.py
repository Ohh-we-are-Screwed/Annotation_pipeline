#!/usr/bin/env python3
"""Tune Stage 3 per-class thresholds on the TUNING split (§11 decision 3, C22).

Reads a Stage 3 output (run with a LOW default_threshold so the full score
range is visible — a sweep can only raise a floor it can see) and the GT 2D
export, and picks one threshold per class:

  - Class has GT here and true positives: sweep thresholds, maximise F0.5
    (precision-weighted — the QA gate wants clean auto-accepts more than
    coverage) at IoU >= `--iou` (default 0.3: the GT is amodal, §C21, so 0.5
    punishes box tightness the detector cannot express).
  - Class has GT here but ZERO true positives: nothing to optimise; the
    threshold is set to the 95th percentile of the class's (all-false-positive)
    score distribution, which suppresses ~95% of its junk. Flagged.
  - Class ABSENT from the tuning split's GT: same p95-of-FP rule, flagged
    louder — the choice is informed only by where the class fires falsely.

Matching is per (image, class): predictions greedy by score against GT boxes
of the same class. Amodal-GT caveat recorded in the output.

This script only PRINTS the result (a YAML-ready `thresholds:` block plus the
evidence table). Writing it into `configs/taxonomy_pilot_nuscenes.yaml` is a
human-reviewed edit, with this script's output as the provenance.

    python -m scripts.tune_thresholds --stage3-dir <dir> [--scenes scene-0655 scene-1094]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.paths import load_paths  # noqa: E402

TUNING_SCENES = ("scene-0655", "scene-1094")  # §11 decision 3
BETA = 0.5
MIN_TP_FOR_SWEEP = 5
FP_PERCENTILE = 95.0


def box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def load_gt(gt_root: str, scenes) -> tuple[dict, dict]:
    """(basename, class) -> [xyxy boxes]; class -> total GT count."""
    gt = defaultdict(list)
    totals: dict[str, int] = defaultdict(int)
    for scene in scenes:
        path = os.path.join(gt_root, scene, "instances.json")
        if not os.path.isfile(path):
            raise SystemExit(f"{path} not found; run scripts/export_gt_coco.py first")
        doc = json.load(open(path))
        cats = {c["id"]: c["name"] for c in doc["categories"]}
        imgs = {i["id"]: os.path.basename(i["file_name"]) for i in doc["images"]}
        for a in doc["annotations"]:
            x, y, w, h = a["bbox"]
            gt[(imgs[a["image_id"]], cats[a["category_id"]])].append([x, y, x + w, y + h])
            totals[cats[a["category_id"]]] += 1
    return gt, dict(totals)


def score_tp_pairs(stage3_dir: str, scenes, gt, iou_gate: float) -> dict:
    """class -> array of (score, is_tp), greedy per (image, class) by score."""
    per_class: dict[str, list] = defaultdict(list)
    floor_seen = 1.0
    for scene in scenes:
        path = os.path.join(stage3_dir, "scenes", scene, "proposals.jsonl")
        if not os.path.isfile(path):
            raise SystemExit(f"{path} not found")
        for line in open(path):
            r = json.loads(line)
            base = os.path.basename(r["image_path"])
            by_cls = defaultdict(list)
            for box, s, c in zip(r["boxes_xyxy_px"], r["scores"], r["class_names"]):
                by_cls[c].append((float(s), box))
                floor_seen = min(floor_seen, float(s))
            for c, items in by_cls.items():
                gts = list(gt.get((base, c), []))
                taken = [False] * len(gts)
                for s, box in sorted(items, key=lambda t: -t[0]):
                    best, bi = 0.0, -1
                    for j, g in enumerate(gts):
                        if not taken[j]:
                            v = box_iou(box, g)
                            if v > best:
                                best, bi = v, j
                    tp = best >= iou_gate
                    if tp:
                        taken[bi] = True
                    per_class[c].append((s, tp))
    return {c: np.array(v) for c, v in per_class.items()}, floor_seen


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    ap.add_argument("--stage3-dir", required=True,
                    help="Stage 3 output run with a LOW floor (e.g. default_threshold 0.20)")
    ap.add_argument("--gt-dir", default=None, help="default <work_root>/cvat_export_gt")
    ap.add_argument("--scenes", nargs="*", default=list(TUNING_SCENES))
    ap.add_argument("--iou", type=float, default=0.3)
    args = ap.parse_args(argv)

    illegal = sorted(set(args.scenes) - set(TUNING_SCENES))
    if illegal:
        raise SystemExit(
            f"scene(s) {illegal} are not in the tuning split {list(TUNING_SCENES)} (§11 d3); "
            "tuning on run/priors scenes is overfitting the report"
        )

    paths = load_paths(args.paths)
    gt_dir = args.gt_dir or os.path.join(paths.work_root, "cvat_export_gt")
    gt, gt_totals = load_gt(gt_dir, args.scenes)
    pairs, floor_seen = score_tp_pairs(args.stage3_dir, args.scenes, gt, args.iou)

    manifest_path = os.path.join(args.stage3_dir, "run_manifest.json")
    aggregation = "unknown"
    provider = "unknown"
    if os.path.isfile(manifest_path):
        manifest = json.load(open(manifest_path))
        provider = manifest.get("provider", "unknown")
        # `score_semantics` is what the scores in these records actually MEAN;
        # `config.score_aggregation` is the caption-provider knob, which is
        # still populated (and inert) on a closed-vocabulary run. Reading the
        # knob would label YOLO confidences as "mean_over_content_tokens" and
        # a threshold table tuned here would carry the wrong provenance.
        aggregation = manifest.get(
            "score_semantics", manifest.get("config", {}).get("score_aggregation", "unknown")
        )

    print(f"# tuning split {args.scenes}, IoU >= {args.iou} vs AMODAL GT (C21 caveat)")
    print(f"# stage3 provider: {provider}   score semantics: {aggregation}")
    print(f"# stage3 score floor seen: {floor_seen:.3f}")
    print(f"{'class':<26}{'n_pred':>7}{'n_gt':>6}{'n_tp':>6}  {'rule':<14}{'thr':>5}{'P@thr':>7}{'R@thr':>7}{'kept':>6}")

    chosen: dict[str, tuple[float, str]] = {}
    for c in sorted(set(pairs) | set(gt_totals), key=lambda c: -(len(pairs.get(c, [])))):
        arr = pairs.get(c)
        n_gt = gt_totals.get(c, 0)
        if arr is None or len(arr) == 0:
            print(f"{c:<26}{0:>7}{n_gt:>6}{0:>6}  {'no-preds':<14}{'--':>5}")
            continue
        s, t = arr[:, 0], arr[:, 1].astype(bool)
        n_tp = int(t.sum())
        if n_tp >= MIN_TP_FOR_SWEEP:
            best = (0.0, -1.0, 0.0, 0.0, 0)
            for thr in np.round(np.arange(0.20, 0.951, 0.01), 2):
                keep = s >= thr
                if not keep.any():
                    continue
                p = float(t[keep].mean())
                r = float(t[keep].sum()) / max(1, n_gt)
                b2 = BETA * BETA
                f = (1 + b2) * p * r / (b2 * p + r) if (p + r) > 0 else 0.0
                if f > best[1]:
                    best = (float(thr), f, p, r, int(keep.sum()))
            thr, _, p, r, kept = best
            rule = f"F{BETA}-sweep"
            chosen[c] = (thr, f"F{BETA} max on tuning split, {n_tp} TPs, P {p:.0%} R {r:.0%}")
            print(f"{c:<26}{len(s):>7}{n_gt:>6}{n_tp:>6}  {rule:<14}{thr:>5.2f}{p:>7.1%}{r:>7.1%}{kept:>6}")
        else:
            thr = float(np.round(np.percentile(s[~t], FP_PERCENTILE), 2)) if (~t).any() else 0.95
            if n_gt == 0:
                rule, note = "p95-FP;noGT", (
                    f"class absent from tuning GT; p{FP_PERCENTILE:.0f} of its FP scores — "
                    "informed only by where it fires falsely"
                )
            else:
                rule, note = "p95-FP;noTP", (
                    f"{n_gt} GT but {n_tp} TPs on tuning split; p{FP_PERCENTILE:.0f} of FP scores "
                    "(suppressing — the detector cannot ground this phrase here)"
                )
            kept = int((s >= thr).sum())
            chosen[c] = (thr, note)
            print(f"{c:<26}{len(s):>7}{n_gt:>6}{n_tp:>6}  {rule:<14}{thr:>5.2f}{'--':>7}{'--':>7}{kept:>6}")

    print("\n# YAML-ready block (paste into configs/taxonomy_pilot_nuscenes.yaml with this")
    print(f"# script's full output as provenance; aggregation={aggregation}):")
    print("thresholds:")
    for c, (thr, note) in sorted(chosen.items()):
        print(f'  "{c}": {thr:.2f}   # {note}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
