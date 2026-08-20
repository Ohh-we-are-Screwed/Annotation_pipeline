#!/usr/bin/env python3
"""Paint-inside-GT rate and runtime metrics for the Stage 3-5 chain.

The Phase 7 exit gate (pilot_plan.md §12): "paint-inside-GT rate is high and
stated as a number. This is the single strongest correctness signal available
in the whole pilot, and it uses data already on disk."

The number: of the LiDAR points Stage 5 painted, what fraction lies inside ANY
nuScenes ground-truth 3D box of the same keyframe? Painted points are ego-frame
at the LiDAR anchor time; GT boxes are global-frame; the comparison transforms
the points through the SAME lidar ego_pose Stage 5's chain used, so a pose or
frame error upstream shows up here as a collapsed rate, not as a plausible one.

Three views, because the headline alone can lie:
  paint_inside_gt      painted points inside any GT box / painted points
  base_rate            ALL single-sweep points inside any GT box / all points
                       (the enrichment denominator: painting at the base rate
                       means the lift did nothing)
  class_correct        painted points inside a GT box whose category maps back
                       to the predicted phrase (taxonomy §0.3), per class

Plus GT coverage — GT boxes within 40 m carrying >= 5 LiDAR returns that
received >= 5 painted points — and the runtime table from the three manifests.

Diagnostic only: writes <work_root>/metrics/paint_metrics.json, no markers.

    python -m scripts.paint_metrics [--scenes scene-0061]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.class_space import all_phrases_reachable, load_detectable_classes  # noqa: E402
from pipeline.common.conventions import quaternion_to_rotation_matrix  # noqa: E402
from pipeline.common.manifest import write_json_atomic  # noqa: E402
from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.stage1_ingestion.ingest import read_pcd_bin  # noqa: E402

R_MAX_M = 40.0  # spec §3.6; the same cap Stage 1 pruned at
MIN_GT_LIDAR_PTS = 5  # comprehensive.md §7.3.9's gate, applied to the GT side


def load_tables(paths):
    def table(name):
        with open(paths.table(name), "r", encoding="utf-8") as fh:
            return json.load(fh)

    categories = {c["token"]: c["name"] for c in table("category.json")}
    instances = {i["token"]: categories[i["category_token"]] for i in table("instance.json")}
    annotations = defaultdict(list)
    for ann in table("sample_annotation.json"):
        annotations[ann["sample_token"]].append(ann)
    ego_poses = {e["token"]: e for e in table("ego_pose.json")}
    return instances, annotations, ego_poses


def gt_boxes_for(sample_token, annotations, instances):
    """(M,3) centers, (M,3) wlh, (M,3,3) rotations, categories, num_lidar_pts."""
    rows = annotations.get(sample_token, [])
    if not rows:
        z = np.zeros((0, 3))
        return z, z, np.zeros((0, 3, 3)), [], np.zeros(0, dtype=int)
    centers = np.array([r["translation"] for r in rows], dtype=np.float64)
    wlh = np.array([r["size"] for r in rows], dtype=np.float64)
    rotations = np.stack([quaternion_to_rotation_matrix(r["rotation"]) for r in rows])
    names = [instances[r["instance_token"]] for r in rows]
    n_pts = np.array([r.get("num_lidar_pts", 0) for r in rows], dtype=int)
    return centers, wlh, rotations, names, n_pts


def points_in_box(points_global, center, wlh, rotation):
    """nuScenes box: size [w,l,h]; box-frame x = length, y = width, z = height."""
    local = (points_global - center) @ rotation  # R^T applied from the right
    w, l, h = wlh
    return (
        (np.abs(local[:, 0]) <= l / 2.0)
        & (np.abs(local[:, 1]) <= w / 2.0)
        & (np.abs(local[:, 2]) <= h / 2.0)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument(
        "--include-unreachable",
        action="store_true",
        help="count GT classes the detector cannot emit in the coverage denominator "
        "(never painted by construction); off by default, recorded in the report",
    )
    args = parser.parse_args(argv)

    paths = load_paths(args.paths)
    stage1 = os.path.join(paths.work_root, "stage1_ingestion")
    stage5 = os.path.join(paths.work_root, "stage5_lift")
    instances, annotations, ego_poses = load_tables(paths)

    with open("configs/taxonomy_pilot_nuscenes.yaml") as fh:
        import yaml
        phrase_of = yaml.safe_load(fh)["prompt_phrase"]  # category -> phrase
    detectable = (
        all_phrases_reachable() if args.include_unreachable
        else load_detectable_classes(paths.work_root)
    )

    scene_root = os.path.join(stage5, "scenes")
    names = sorted(n for n in os.listdir(scene_root) if os.path.isdir(os.path.join(scene_root, n)))
    if args.scenes:
        names = [n for n in names if n in args.scenes]

    totals = {
        "n_points_painted": 0,
        "n_painted_inside_gt": 0,
        "n_points_cloud": 0,
        "n_cloud_inside_gt": 0,
        "n_gt_boxes_eligible": 0,
        "n_gt_boxes_covered": 0,
        # Eligible on range and returns, but of a class this run's detector
        # cannot emit (C23): never proposed, so never painted. Held out of the
        # coverage denominator and counted, so the gap stays visible.
        "n_gt_boxes_unreachable_class": 0,
    }
    per_class = defaultdict(lambda: {"painted": 0, "inside_any_gt": 0, "class_correct": 0})

    for scene in names:
        with open(os.path.join(stage5, "scenes", scene, "lift.jsonl")) as fh:
            lift_rows = [json.loads(line) for line in fh if line.strip()]
        with open(os.path.join(stage1, "scenes", scene, "keyframes.jsonl")) as fh:
            keyframes = {r["keyframe_token"]: r for r in map(json.loads, fh) if r}

        for lift_row in lift_rows:
            keyframe = keyframes[lift_row["keyframe_token"]]
            pose = ego_poses[keyframe["lidar_ego_pose_token"]]
            R = quaternion_to_rotation_matrix(pose["rotation"])
            t = np.asarray(pose["translation"], dtype=np.float64)

            cloud_ego = read_pcd_bin(keyframe["single_sweep_cloud"]["path"])[:, :3].astype(np.float64)
            cloud_global = cloud_ego @ R.T + t

            with np.load(os.path.join(stage5, lift_row["points_path"])) as npz:
                painted_index = npz["point_index"].astype(np.int64)
                instance_id = npz["instance_id"].astype(np.int64)
            painted_global = cloud_global[painted_index]
            class_of_instance = {r["instance_id"]: r["class_name"] for r in lift_row["instances"]}
            painted_class = np.array(
                [class_of_instance[i] for i in instance_id], dtype=object
            ) if len(instance_id) else np.zeros(0, dtype=object)

            centers, wlh, rotations, cats, gt_npts = gt_boxes_for(
                lift_row["keyframe_token"], annotations, instances
            )
            painted_hit = np.zeros(len(painted_global), dtype=bool)
            cloud_hit = np.zeros(len(cloud_global), dtype=bool)
            painted_class_hit = np.zeros(len(painted_global), dtype=bool)

            range_ego = np.linalg.norm((centers - t) if len(centers) else centers, axis=1) \
                if len(centers) else np.zeros(0)
            for b in range(len(centers)):
                inside_painted = points_in_box(painted_global, centers[b], wlh[b], rotations[b])
                painted_hit |= inside_painted
                cloud_hit |= points_in_box(cloud_global, centers[b], wlh[b], rotations[b])
                gt_phrase = phrase_of.get(cats[b])
                if gt_phrase is not None and len(painted_class):
                    painted_class_hit |= inside_painted & (painted_class == gt_phrase)
                eligible = range_ego[b] <= R_MAX_M and gt_npts[b] >= MIN_GT_LIDAR_PTS
                if eligible and not detectable.is_reachable(gt_phrase):
                    totals["n_gt_boxes_unreachable_class"] += 1
                elif eligible:
                    totals["n_gt_boxes_eligible"] += 1
                    if int(inside_painted.sum()) >= MIN_GT_LIDAR_PTS:
                        totals["n_gt_boxes_covered"] += 1

            totals["n_points_painted"] += len(painted_global)
            totals["n_painted_inside_gt"] += int(painted_hit.sum())
            totals["n_points_cloud"] += len(cloud_global)
            totals["n_cloud_inside_gt"] += int(cloud_hit.sum())
            for i, cls in enumerate(painted_class):
                row = per_class[cls]
                row["painted"] += 1
                row["inside_any_gt"] += bool(painted_hit[i])
                row["class_correct"] += bool(painted_class_hit[i])

    paint_rate = totals["n_painted_inside_gt"] / max(1, totals["n_points_painted"])
    base_rate = totals["n_cloud_inside_gt"] / max(1, totals["n_points_cloud"])
    coverage = totals["n_gt_boxes_covered"] / max(1, totals["n_gt_boxes_eligible"])

    runtime = {}
    for stage in ("stage3_proposals", "stage4_masks", "stage5_lift"):
        p = os.path.join(paths.work_root, stage, "run_manifest.json")
        if os.path.isfile(p):
            with open(p) as fh:
                m = json.load(fh)
            runtime[stage] = {"elapsed_s": m["elapsed_s"], "vram_cap": m.get("vram_cap")}

    report = {
        "spec": "dhakascenes-pilot/paint-metrics/v1",
        "class_space": detectable.as_dict(),
        "scenes": names,
        "paint_inside_gt_rate": round(paint_rate, 4),
        "base_rate_all_points": round(base_rate, 4),
        "enrichment": round(paint_rate / max(base_rate, 1e-9), 2),
        "gt_coverage": {
            "eligible_boxes": totals["n_gt_boxes_eligible"],
            "covered_boxes": totals["n_gt_boxes_covered"],
            "unreachable_class_boxes": totals["n_gt_boxes_unreachable_class"],
            "rate": round(coverage, 4),
            "eligibility": f"center <= {R_MAX_M} m from ego, num_lidar_pts >= {MIN_GT_LIDAR_PTS}",
        },
        "totals": totals,
        "per_class": {
            cls: {
                "painted": row["painted"],
                "inside_any_gt_rate": round(row["inside_any_gt"] / max(1, row["painted"]), 4),
                "class_correct_rate": round(row["class_correct"] / max(1, row["painted"]), 4),
            }
            for cls, row in sorted(per_class.items(), key=lambda kv: -kv[1]["painted"])
        },
        "runtime": runtime,
        "caveat": (
            "descriptive on 10 scenes, never a measurement (§0.2); GT boxes are the nuScenes "
            "annotations, so this validates the LIFT geometry, not the detector's taxonomy"
        ),
    }
    out_path = os.path.join(paths.work_root, "metrics", "paint_metrics.json")
    write_json_atomic(out_path, report)

    print(f"paint-inside-GT rate : {paint_rate:.1%}   (of {totals['n_points_painted']} painted points)")
    print(f"base rate, all points: {base_rate:.1%}   -> enrichment x{report['enrichment']}")
    print(f"GT boxes covered     : {totals['n_gt_boxes_covered']}/{totals['n_gt_boxes_eligible']}"
          f" ({coverage:.1%})  [{report['gt_coverage']['eligibility']}]")
    print(f"class space          : {detectable.describe()}")
    if totals["n_gt_boxes_unreachable_class"]:
        print(f"  held out of the denominator: {totals['n_gt_boxes_unreachable_class']} otherwise-"
              "eligible GT boxes whose class this detector cannot emit")
    print("per class (top 8, by painted points):")
    for cls, row in list(report["per_class"].items())[:8]:
        print(f"  {cls:<24} {row['painted']:>7} painted   inside-GT {row['inside_any_gt_rate']:.1%}"
              f"   class-correct {row['class_correct_rate']:.1%}")
    for stage, r in runtime.items():
        print(f"{stage:<16}: {r['elapsed_s']}s")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
