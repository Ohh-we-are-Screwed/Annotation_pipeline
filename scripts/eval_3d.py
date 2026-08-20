#!/usr/bin/env python3
"""3D box test: OUR Stage 6 boxes vs the nuScenes human 3D boxes.

nuScenes-style center-distance matching (greedy, BEV center distance <= 2 m,
per keyframe), because IoU matching would double-punish the known failure
modes: an under-tier detector's boxes are on the right object but loosely
sized, and a 90-degree-ambiguous yaw makes IoU collapse while the object is
still correctly found.

Reported, matched pairs only:
  ATE   mean BEV center error (m)
  ASE   mean (1 - IoU of the two boxes after aligning center+yaw) — pure size
  AOE   mean |yaw error| mod pi, on boxes whose yaw the fitter TRUSTS
        (yaw_ambiguous excluded; mod pi because Stage 6 declares axis-only yaw)

Precision is split localization vs class-aware like eval_2d.py, and recall is
against GT boxes within 40 m carrying >= 5 LiDAR points (the same eligibility
paint_metrics.py uses). Descriptive on 10 scenes (§0.2). Writes
<work_root>/metrics/detect3d_metrics.json.

    python -m scripts.eval_3d [--scenes scene-0061] [--dist 2.0]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.class_space import all_phrases_reachable, load_detectable_classes  # noqa: E402
from pipeline.common.conventions import quaternion_to_rotation_matrix, wrap_to_pi_rad  # noqa: E402
from pipeline.common.manifest import write_json_atomic  # noqa: E402
from pipeline.common.paths import load_paths  # noqa: E402

R_MAX_M = 40.0
MIN_GT_LIDAR_PTS = 5


def aligned_size_iou(size_a_wlh, size_b_wlh) -> float:
    """3D IoU after aligning centers and yaw: a pure size comparison."""
    w = min(size_a_wlh[0], size_b_wlh[0])
    l = min(size_a_wlh[1], size_b_wlh[1])
    h = min(size_a_wlh[2], size_b_wlh[2])
    inter = w * l * h
    va = size_a_wlh[0] * size_a_wlh[1] * size_a_wlh[2]
    vb = size_b_wlh[0] * size_b_wlh[1] * size_b_wlh[2]
    return inter / max(va + vb - inter, 1e-9)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--dist", type=float, default=2.0, help="BEV center-distance gate (m)")
    parser.add_argument(
        "--include-unreachable",
        action="store_true",
        help="score GT classes the detector cannot emit (recall 0 by construction); "
        "off by default, and the choice is recorded in the report",
    )
    args = parser.parse_args(argv)

    paths = load_paths(args.paths)
    stage6 = os.path.join(paths.work_root, "stage6_cluster")
    detectable = (
        all_phrases_reachable() if args.include_unreachable
        else load_detectable_classes(paths.work_root)
    )

    def table(name):
        with open(paths.table(name), "r", encoding="utf-8") as fh:
            return json.load(fh)

    categories = {c["token"]: c["name"] for c in table("category.json")}
    instance_category = {i["token"]: categories[i["category_token"]] for i in table("instance.json")}
    annotations = defaultdict(list)
    for ann in table("sample_annotation.json"):
        annotations[ann["sample_token"]].append(ann)
    ego_poses = {e["token"]: e for e in table("ego_pose.json")}

    import yaml
    with open("configs/taxonomy_pilot_nuscenes.yaml") as fh:
        phrase_of = yaml.safe_load(fh)["prompt_phrase"]

    scene_root = os.path.join(stage6, "scenes")
    names = sorted(n for n in os.listdir(scene_root) if os.path.isdir(os.path.join(scene_root, n)))
    if args.scenes:
        names = [n for n in names if n in args.scenes]

    n_pred = tp_loc = tp_cls = 0
    gt_total = gt_hit = 0
    n_gt_out_of_class_space = 0
    n_gt_unreachable = 0
    per_class_unreachable = defaultdict(int)
    ate, ase, aoe = [], [], []
    per_class = defaultdict(lambda: {"pred": 0, "tp_loc": 0, "tp_cls": 0})

    # Stage 1's keyframes give the lidar ego pose per keyframe token.
    for scene in names:
        with open(os.path.join(paths.work_root, "stage1_ingestion", "scenes", scene, "keyframes.jsonl")) as fh:
            pose_of = {
                r["keyframe_token"]: ego_poses[r["lidar_ego_pose_token"]]
                for r in map(json.loads, fh) if r
            }
        with open(os.path.join(scene_root, scene, "boxes.jsonl")) as fh:
            rows = [r for r in map(json.loads, fh) if r and r.get("box")]
        by_keyframe = defaultdict(list)
        for r in rows:
            by_keyframe[r["keyframe_token"]].append(r)

        for token, pose in pose_of.items():
            R_e = quaternion_to_rotation_matrix(pose["rotation"])
            t_e = np.asarray(pose["translation"], dtype=np.float64)

            gts = []
            for ann in annotations.get(token, ()):
                center_ego = R_e.T @ (np.asarray(ann["translation"]) - t_e)
                if math.hypot(center_ego[0], center_ego[1]) > R_MAX_M:
                    continue
                if ann.get("num_lidar_pts", 0) < MIN_GT_LIDAR_PTS:
                    continue
                # A category with no prompt phrase is OUT OF THE CLASS SPACE, not
                # an error: the collapsed taxonomy (C21) deliberately drops
                # categories the detector is never asked to find, exactly as the
                # nuScenes benchmark does. Counting them would make recall a
                # measure of the taxonomy rather than of the detector. Indexing
                # phrase_of directly raised KeyError the moment that happened.
                phrase = phrase_of.get(instance_category[ann["instance_token"]])
                if phrase is None:
                    n_gt_out_of_class_space += 1
                    continue
                # In the class space, but OUTSIDE what this run's detector could
                # emit. A closed-vocabulary provider (C23) has no source class
                # for four of the ten phrases, so their recall is 0 by
                # construction and counting them measures the vocabulary gap
                # rather than the pipeline. Excluded by default, counted per
                # class, and the exclusion is named in the report.
                if not detectable.is_reachable(phrase):
                    n_gt_unreachable += 1
                    per_class_unreachable[phrase] += 1
                    continue
                R_rel = R_e.T @ quaternion_to_rotation_matrix(ann["rotation"])
                gts.append({
                    "center": center_ego,
                    "size_wlh": [float(v) for v in ann["size"]],
                    "yaw": float(np.arctan2(R_rel[1, 0], R_rel[0, 0])),
                    "phrase": phrase,
                })
            gt_total += len(gts)

            preds = sorted(by_keyframe.get(token, ()), key=lambda r: -r["score"])
            taken = np.zeros(len(gts), dtype=bool)
            for row in preds:
                n_pred += 1
                cls = row["class_name"]
                per_class[cls]["pred"] += 1
                box = row["box"]
                center = np.asarray(box["translation_m"], dtype=np.float64)
                distances = [
                    math.hypot(center[0] - g["center"][0], center[1] - g["center"][1])
                    for g in gts
                ]
                best = None
                for j in np.argsort(distances) if gts else []:
                    if distances[j] > args.dist:
                        break
                    if not taken[j]:
                        best = int(j)
                        break
                if best is None:
                    continue
                taken[best] = True
                g = gts[best]
                tp_loc += 1
                per_class[cls]["tp_loc"] += 1
                if g["phrase"] == cls:
                    tp_cls += 1
                    per_class[cls]["tp_cls"] += 1
                ate.append(distances[best])
                ase.append(1.0 - aligned_size_iou(box["size_wlh_m"], g["size_wlh"]))
                if not box["yaw_ambiguous"]:
                    err = abs(wrap_to_pi_rad(box["yaw_rad"] - g["yaw"]))
                    aoe.append(min(err, math.pi - err))  # axis-only yaw (mod pi)
            gt_hit += int(taken.sum())

    report = {
        "spec": "dhakascenes-pilot/detect3d-metrics/v1",
        "scenes": names,
        "match": f"greedy BEV center distance <= {args.dist} m, score order",
        "n_pred_boxes": n_pred,
        "precision_localization": round(tp_loc / max(1, n_pred), 4),
        "precision_class_aware": round(tp_cls / max(1, n_pred), 4),
        "gt_recall": round(gt_hit / max(1, gt_total), 4),
        "gt_boxes_eligible": gt_total,
        # Excluded from the denominator because the taxonomy never prompts for
        # them (C21), NOT because they were missed. Reported so the exclusion is
        # auditable rather than implicit in a recall number.
        "gt_boxes_out_of_class_space": n_gt_out_of_class_space,
        # Also excluded, for a DIFFERENT reason: in the class space, but no
        # source class of this run's detector maps to them (C23). Recall on
        # these is 0 by construction. Counted per class so the size of the
        # capability gap is readable next to the recall it was removed from.
        "gt_boxes_unreachable_class": n_gt_unreachable,
        "gt_boxes_unreachable_per_class": dict(sorted(per_class_unreachable.items())),
        "class_space": detectable.as_dict(),
        "matched": {
            "n": len(ate),
            "ate_m_mean": round(float(np.mean(ate)), 3) if ate else None,
            "ase_mean": round(float(np.mean(ase)), 3) if ase else None,
            "aoe_rad_mean_trusted_yaw": round(float(np.mean(aoe)), 3) if aoe else None,
            "n_trusted_yaw": len(aoe),
        },
        "per_class": {
            cls: {
                "boxes": row["pred"],
                "precision_localization": round(row["tp_loc"] / max(1, row["pred"]), 4),
                "precision_class_aware": round(row["tp_cls"] / max(1, row["pred"]), 4),
            }
            for cls, row in sorted(per_class.items(), key=lambda kv: -kv[1]["pred"])
        },
        "caveat": (
            "descriptive on 10 scenes (§0.2); eligibility mirrors paint_metrics "
            f"(<= {R_MAX_M} m, num_lidar_pts >= {MIN_GT_LIDAR_PTS}); recall is over the "
            f"{len(detectable.reachable)} phrase(s) this run's detector could emit"
        ),
    }
    out_path = os.path.join(paths.work_root, "metrics", "detect3d_metrics.json")
    write_json_atomic(out_path, report)

    print(f"our 3D boxes         : {n_pred}  matched at <= {args.dist} m BEV center distance")
    print(f"precision (location) : {report['precision_localization']:.1%}   a human box was there")
    print(f"precision (class)    : {report['precision_class_aware']:.1%}   ...AND the name matched")
    print(f"GT recall            : {report['gt_recall']:.1%}   of {gt_total} eligible human boxes")
    print(f"class space          : {detectable.describe()}")
    if n_gt_unreachable:
        share = n_gt_unreachable / max(1, n_gt_unreachable + gt_total)
        print(f"  suppressed from the recall denominator: {n_gt_unreachable} eligible GT boxes "
              f"({share:.1%}) whose class this detector cannot emit")
        for phrase, count in sorted(per_class_unreachable.items(), key=lambda kv: -kv[1]):
            print(f"    {phrase:<26} {count:>5}")
    m = report["matched"]
    print(f"matched pairs        : {m['n']}   ATE {m['ate_m_mean']} m   ASE {m['ase_mean']}   "
          f"AOE {m['aoe_rad_mean_trusted_yaw']} rad (over {m['n_trusted_yaw']} trusted-yaw boxes)")
    print("per class (top 8):")
    for cls, row in list(report["per_class"].items())[:8]:
        print(f"  {cls:<26} {row['boxes']:>5} boxes   loc {row['precision_localization']:.1%}"
              f"   class {row['precision_class_aware']:.1%}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
