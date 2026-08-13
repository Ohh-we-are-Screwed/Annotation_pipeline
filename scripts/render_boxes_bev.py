#!/usr/bin/env python3
"""BEV comparison: OUR Stage 6 3D boxes vs the nuScenes human 3D boxes.

One PNG per sampled keyframe under <work_root>/viz_boxes/<scene>/: the
single-sweep cloud in grey, OUR oriented boxes solid (colored by class, heading
tick on the +x face), the HUMAN answer key dashed black. Both are drawn in the
EGO frame at the LiDAR anchor time — GT comes global and is transformed through
the same lidar ego_pose the pipeline used, so a frame error upstream shows here
as boxes sliding off the points.

    python -m scripts.render_boxes_bev [--scenes scene-0061] [--every 8]
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.conventions import quaternion_to_rotation_matrix  # noqa: E402
from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.stage1_ingestion.ingest import read_pcd_bin  # noqa: E402

R_MAX_M = 40.0


def class_color(name: str):
    h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    r, g, b = colorsys.hsv_to_rgb((h % 360) / 360.0, 0.85, 0.9)
    return (r, g, b)


def footprint(center_xy, wl, yaw):
    """(5, 2) closed BEV rectangle; nuScenes/[w,l] order, x-axis = length."""
    w, l = wl
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners = np.array([[l, w], [l, -w], [-l, -w], [-l, w], [l, w]]) / 2.0
    return corners @ R.T + np.asarray(center_xy)


def draw_box(ax, center_xy, wl, yaw, *, color, ls, lw, heading=False):
    poly = footprint(center_xy, wl, yaw)
    ax.plot(-poly[:, 1], poly[:, 0], color=color, ls=ls, lw=lw)
    if heading:
        tip = footprint(center_xy, wl, yaw)[0:2].mean(axis=0)
        ax.plot([-center_xy[1], -tip[1]], [center_xy[0], tip[0]], color=color, lw=lw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--every", type=int, default=8)
    args = parser.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = load_paths(args.paths)
    stage1 = os.path.join(paths.work_root, "stage1_ingestion")
    stage6 = os.path.join(paths.work_root, "stage6_cluster")
    out_root = os.path.join(paths.work_root, "viz_boxes")

    def table(name):
        with open(paths.table(name), "r", encoding="utf-8") as fh:
            return json.load(fh)

    ego_poses = {e["token"]: e for e in table("ego_pose.json")}
    annotations = defaultdict(list)
    for ann in table("sample_annotation.json"):
        annotations[ann["sample_token"]].append(ann)

    scene_root = os.path.join(stage6, "scenes")
    names = sorted(n for n in os.listdir(scene_root) if os.path.isdir(os.path.join(scene_root, n)))
    if args.scenes:
        names = [n for n in names if n in args.scenes]

    n_rendered = 0
    for scene in names:
        with open(os.path.join(scene_root, scene, "boxes.jsonl")) as fh:
            box_rows = [json.loads(line) for line in fh if line.strip()]
        by_keyframe = defaultdict(list)
        for row in box_rows:
            by_keyframe[row["keyframe_token"]].append(row)
        with open(os.path.join(stage1, "scenes", scene, "keyframes.jsonl")) as fh:
            keyframes = [json.loads(line) for line in fh if line.strip()]

        for keyframe in keyframes[:: max(1, args.every)]:
            token = keyframe["keyframe_token"]
            # Rows without a box are recorded outcomes (no points, below
            # min_samples, all noise) — present in boxes.jsonl, nothing to draw.
            ours = [r for r in by_keyframe.get(token, []) if r.get("box")]
            cloud = read_pcd_bin(keyframe["single_sweep_cloud"]["path"])[:, :3]
            pose = ego_poses[keyframe["lidar_ego_pose_token"]]
            R_e = quaternion_to_rotation_matrix(pose["rotation"])
            t_e = np.asarray(pose["translation"])

            fig, ax = plt.subplots(figsize=(10, 10), dpi=110)
            ax.scatter(-cloud[::3, 1], cloud[::3, 0], s=0.5, c="#c9c9c9", linewidths=0)

            n_gt = 0
            for ann in annotations.get(token, ()):
                center_ego = R_e.T @ (np.asarray(ann["translation"]) - t_e)
                if np.hypot(center_ego[0], center_ego[1]) > R_MAX_M:
                    continue
                R_box = quaternion_to_rotation_matrix(ann["rotation"])
                R_rel = R_e.T @ R_box
                yaw = float(np.arctan2(R_rel[1, 0], R_rel[0, 0]))
                w, l, _ = ann["size"]
                draw_box(ax, center_ego[:2], (w, l), yaw, color="black", ls="--", lw=1.0)
                n_gt += 1

            classes_seen = {}
            for row in ours:
                box = row["box"]
                color = class_color(row["class_name"])
                w, l, _ = box["size_wlh_m"]
                draw_box(
                    ax, box["translation_m"][:2], (w, l), box["yaw_rad"],
                    color=color, ls="-", lw=1.6, heading=not box["yaw_ambiguous"],
                )
                classes_seen[row["class_name"]] = color

            theta = np.linspace(0, 2 * np.pi, 256)
            ax.plot(R_MAX_M * np.cos(theta), R_MAX_M * np.sin(theta), color="#999999", lw=0.7, ls=":")
            ax.annotate("", xy=(0, 4), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
            handles = [plt.Line2D([0], [0], color="black", ls="--", label=f"human GT ({n_gt})")]
            handles += [plt.Line2D([0], [0], color=c, label=k) for k, c in sorted(classes_seen.items())]
            ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)
            ax.set_xlim(-42, 42), ax.set_ylim(-42, 42)
            ax.set_aspect("equal")
            ax.set_xlabel("left  <-  y (m)  ->  right"), ax.set_ylabel("x forward (m)")
            ax.set_title(
                f"{scene}  {token[:12]}…   OUR boxes: {len(ours)} solid   "
                f"human GT: {n_gt} dashed   (ego frame, <= {R_MAX_M:.0f} m)"
            )
            os.makedirs(os.path.join(out_root, scene), exist_ok=True)
            out_path = os.path.join(out_root, scene, f"{token[:12]}.png")
            fig.tight_layout()
            fig.savefig(out_path)
            plt.close(fig)
            n_rendered += 1

    print(f"rendered {n_rendered} BEV comparisons under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
