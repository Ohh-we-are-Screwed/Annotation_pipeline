#!/usr/bin/env python3
"""Render Stage 3/4/5 outputs onto the camera images and a BEV plot.

Diagnostic eyeballs, not a pipeline stage: reads whatever stages 3-5 wrote,
draws it, and writes PNGs under <work_root>/viz/. No markers are written and
nothing downstream may consume these files.

Per selected keyframe:
  <scene>/<token>/<CHANNEL>.png   one camera: mask overlays + proposal boxes
  <scene>/<token>/grid.png        all six cameras as a 3x2 contact sheet
  <scene>/<token>/bev.png         painted points in ego-frame BEV, 40 m ring

Box drawing encodes the Stage 4 contest: solid = kept mask, dashed = suppressed
cross-camera duplicate (its IoA and winner are in masks.jsonl). Colors are
stable per class name, so the same class is the same hue in every image.

    python -m scripts.render_annotations [--scenes scene-0061] [--every 8]
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.stage1_ingestion.ingest import read_pcd_bin  # noqa: E402
from pipeline.stage5_lift.lift import MaskFile  # noqa: E402

GRID_COLS = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
             "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")


def class_color(name: str) -> tuple[int, int, int]:
    """Stable, distinct-ish RGB per class name."""
    h = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    hue = (h % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def load_font(size: int = 16):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if os.path.isfile(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def dashed_rectangle(draw: ImageDraw.ImageDraw, box, color, width=3, dash=10):
    x1, y1, x2, y2 = box
    for (ax, ay, bx, by) in ((x1, y1, x2, y1), (x2, y1, x2, y2), (x2, y2, x1, y2), (x1, y2, x1, y1)):
        length = max(abs(bx - ax), abs(by - ay))
        steps = max(1, int(length // dash))
        for i in range(0, steps, 2):
            t0, t1 = i / steps, min((i + 1) / steps, 1.0)
            draw.line(
                [(ax + (bx - ax) * t0, ay + (by - ay) * t0),
                 (ax + (bx - ax) * t1, ay + (by - ay) * t1)],
                fill=color, width=width,
            )


def render_camera(
    image_path: str,
    channel: str,
    candidates: list[dict],
    masks: MaskFile,
    font,
    mask_alpha: int = 96,
) -> Image.Image:
    with Image.open(image_path) as im:
        base = im.convert("RGB")
    overlay = np.zeros((base.height, base.width, 4), dtype=np.uint8)

    rows = [c for c in candidates if c["channel"] == channel]
    for c in rows:
        color = class_color(c["class_name"])
        mask = masks.mask(channel, c["proposal_index"])
        alpha = mask_alpha if c["kept"] else mask_alpha // 3
        overlay[mask] = (*color, alpha)

    out = Image.alpha_composite(base.convert("RGBA"), Image.fromarray(overlay, "RGBA"))
    draw = ImageDraw.Draw(out)
    for c in rows:
        color = class_color(c["class_name"])
        box = c["proposal_box_xyxy_px"]
        label = f"{c['class_name']} {c['score']:.2f}"
        if c["kept"]:
            draw.rectangle(box, outline=color, width=3)
        else:
            dashed_rectangle(draw, box, color)
            label += f"  suppressed by {c['suppressed_by'][0]} (IoA {c['suppression_ioa']:.2f})"
        tx, ty = box[0] + 2, max(0.0, box[1] - 20)
        bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0, 180))
        draw.text((tx, ty), label, fill=color, font=font)
    return out.convert("RGB")


def render_bev(points_ego: np.ndarray, lift_row: dict, points_npz, out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    idx = points_npz["point_index"]
    inst = points_npz["instance_id"]
    fig, ax = plt.subplots(figsize=(9, 9), dpi=110)
    background = points_ego[::4]  # thin the unpainted cloud for legibility
    ax.scatter(-background[:, 1], background[:, 0], s=0.5, c="#c8c8c8", linewidths=0)

    by_class: dict[str, int] = {}
    for row in lift_row["instances"]:
        if row["n_points"] == 0:
            continue
        selected = inst == row["instance_id"]
        xyz = points_ego[idx[selected]]
        color = np.array(class_color(row["class_name"])) / 255.0
        ax.scatter(-xyz[:, 1], xyz[:, 0], s=2.5, color=color, linewidths=0)
        by_class[row["class_name"]] = by_class.get(row["class_name"], 0) + 1

    theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(40 * np.cos(theta), 40 * np.sin(theta), color="#888888", lw=0.8, ls="--")
    ax.annotate("", xy=(0, 4), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.set_xlim(-42, 42), ax.set_ylim(-42, 42)
    ax.set_aspect("equal")
    ax.set_xlabel("left  <-  y (m)  ->  right"), ax.set_ylabel("x forward (m)")
    ax.set_title(
        f"{lift_row['keyframe_token'][:12]}…  painted {lift_row['contest']['n_points_painted']} pts, "
        f"{sum(1 for r in lift_row['instances'] if r['n_points'] > 0)} instances "
        f"({', '.join(f'{v} {k}' for k, v in sorted(by_class.items(), key=lambda kv: -kv[1])[:4])})"
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--every", type=int, default=8, help="render every Nth keyframe of a scene")
    parser.add_argument("--out-dir", default=None, help="default <work_root>/viz")
    args = parser.parse_args(argv)

    paths = load_paths(args.paths)
    stage1 = os.path.join(paths.work_root, "stage1_ingestion")
    stage4 = os.path.join(paths.work_root, "stage4_masks")
    stage5 = os.path.join(paths.work_root, "stage5_lift")
    out_root = args.out_dir or os.path.join(paths.work_root, "viz")
    font = load_font()

    scene_root = os.path.join(stage5, "scenes")
    if not os.path.isdir(scene_root):
        print(f"{scene_root} not found — run stages 3-5 first", file=sys.stderr)
        return 2
    names = sorted(n for n in os.listdir(scene_root) if os.path.isdir(os.path.join(scene_root, n)))
    if args.scenes:
        names = [n for n in names if n in args.scenes]

    n_rendered = 0
    for scene in names:
        with open(os.path.join(stage4, "scenes", scene, "masks.jsonl")) as fh:
            mask_rows = {r["keyframe_token"]: r for r in map(json.loads, fh) if r}
        with open(os.path.join(stage5, "scenes", scene, "lift.jsonl")) as fh:
            lift_rows = [json.loads(line) for line in fh if line.strip()]
        with open(os.path.join(stage1, "scenes", scene, "keyframes.jsonl")) as fh:
            keyframes = {r["keyframe_token"]: r for r in map(json.loads, fh) if r}

        for lift_row in lift_rows[:: max(1, args.every)]:
            token = lift_row["keyframe_token"]
            mask_row = mask_rows[token]
            keyframe = keyframes[token]
            out_dir = os.path.join(out_root, scene, token[:12])
            os.makedirs(out_dir, exist_ok=True)

            masks = MaskFile(os.path.join(stage4, mask_row["mask_path"]))
            panels = {}
            try:
                for channel, cam in sorted(keyframe["cameras"].items()):
                    panel = render_camera(
                        os.path.join(paths.dataroot, cam["path"]),
                        channel, mask_row["candidates"], masks, font,
                    )
                    panel.save(os.path.join(out_dir, f"{channel}.png"))
                    panels[channel] = panel.resize((800, 450), Image.BILINEAR)
            finally:
                masks.close()

            grid = Image.new("RGB", (2400, 900), "black")
            for i, channel in enumerate(GRID_COLS):
                if channel in panels:
                    grid.paste(panels[channel], ((i % 3) * 800, (i // 3) * 450))
            grid.save(os.path.join(out_dir, "grid.png"))

            points_ego = read_pcd_bin(keyframe["single_sweep_cloud"]["path"])[:, :3]
            with np.load(os.path.join(stage5, lift_row["points_path"])) as npz:
                render_bev(points_ego, lift_row, npz, os.path.join(out_dir, "bev.png"))
            n_rendered += 1
            print(f"  {scene} {token[:12]}…  grid + bev + 6 cameras")

    print(f"rendered {n_rendered} keyframes under {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
