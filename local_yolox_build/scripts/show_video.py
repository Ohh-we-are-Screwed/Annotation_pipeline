"""Run the arm B checkpoint over a video and draw its boxes, live.

Displays with cv2.imshow -- keys: q/ESC quit, SPACE pause/resume, s save the
current annotated frame as a PNG next to the video.

THE ARM B SHIP LIST IS NOT APPLIED HERE. The checkpoint predicts all 13
RSUD20K classes (configs/rsud20k_yolo11x.yaml); Stage 3 keeps only `rickshaw`
and `cng`. This viewer shows everything by default so you can see WHY a box was
or was not kept -- pass `--ship-only` to see the arm B view instead.

ROTATION METADATA IS APPLIED. Phone captures carry a rotate-90 side-data box
that this OpenCV build does NOT auto-apply (CAP_PROP_ORIENTATION_AUTO reads 0),
so frames arrive on their side and a detector trained on upright road scenes
finds almost nothing. `--rotate auto` reads CAP_PROP_ORIENTATION_META and turns
the frame upright before inference; `--rotate 0` reproduces the raw decode.

Headless is the normal case on this box (XDG_SESSION_TYPE=tty, no DISPLAY): if
imshow cannot open a window, the script says so and falls back to writing an
annotated mp4, which `--save` also forces. It never silently skips frames.

Usage:
    python local_yolox_build/scripts/show_video.py VID_20260827_205445448.mp4
    python local_yolox_build/scripts/show_video.py video.mp4 --conf 0.5 --ship-only
    python local_yolox_build/scripts/show_video.py video.mp4 --save out.mp4
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
from pathlib import Path

import cv2

BUILD = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = BUILD / "artifacts" / "yolo11x-rsud20k-armb.pt"

# The two classes arm B ships (scripts/export_armb.py:SHIP_NAMES).
SHIP_NAMES = ("rickshaw", "cng")

# Distinct BGR per class index, ordered as configs/rsud20k_yolo11x.yaml names.
PALETTE = [
    (180, 180, 180),  # 0 person
    (60, 220, 60),    # 1 rickshaw   <- shipped
    (60, 200, 200),   # 2 rickshaw van
    (60, 120, 255),   # 3 cng        <- shipped
    (200, 90, 90),    # 4 truck
    (220, 140, 60),   # 5 pickup truck
    (255, 120, 200),  # 6 car
    (120, 90, 220),   # 7 motorcycle
    (200, 200, 60),   # 8 bicycle
    (90, 60, 200),    # 9 bus
    (150, 150, 250),  # 10 micro bus
    (100, 180, 140),  # 11 covered van
    (230, 230, 120),  # 12 human hauler
]


def draw(frame, result, names, tally, thickness: int = 2) -> int:
    """Paint one ultralytics Result onto `frame` in place; return box count.

    `tally` is a Counter accumulated across frames for the end-of-run summary.
    """
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return 0
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        tally[names[k]] += 1
        colour = PALETTE[k % len(PALETTE)]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, colour, thickness)
        label = f"{names[k]} {c:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        # Label above the box, or inside it when the box touches the top edge.
        ty = p1[1] - 4 if p1[1] - th - base - 4 >= 0 else p1[1] + th + 4
        cv2.rectangle(frame, (p1[0], ty - th - base), (p1[0] + tw + 4, ty + base),
                      colour, cv2.FILLED)
        cv2.putText(frame, label, (p1[0] + 2, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 1, cv2.LINE_AA)
    return len(cls)


def can_display() -> bool:
    """Whether a real imshow window is reachable, asked by trying it."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return False
    try:
        probe = "__probe__"
        cv2.namedWindow(probe, cv2.WINDOW_NORMAL)
        cv2.destroyWindow(probe)
        cv2.waitKey(1)
        return True
    except cv2.error:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", type=Path)
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.7,
                    help="ultralytics NMS IoU; matches ProposalConfig.yolo_iou")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="1280 is the training resolution (runs/r1280-4/args.yaml)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--rotate", default="auto",
                    choices=("auto", "0", "90", "180", "270"),
                    help="clockwise degrees; auto = CAP_PROP_ORIENTATION_META")
    ap.add_argument("--ship-only", action="store_true",
                    help=f"restrict to the arm B ship list {SHIP_NAMES}")
    ap.add_argument("--save", type=Path, default=None, metavar="OUT.mp4",
                    help="also write an annotated mp4 (forced when headless)")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    args = ap.parse_args()

    if not args.video.exists():
        print(f"no such video: {args.video}", file=sys.stderr)
        return 2
    if not args.weights.exists():
        print(f"no such checkpoint: {args.weights}", file=sys.stderr)
        return 2

    from ultralytics import YOLO

    model = YOLO(str(args.weights))
    names = model.names
    print(f"checkpoint : {args.weights}")
    print(f"classes    : {len(names)} -> {list(names.values())}")

    keep = None
    if args.ship_only:
        keep = [i for i, n in names.items() if n in SHIP_NAMES]
        if len(keep) != len(SHIP_NAMES):
            print(f"ship list {SHIP_NAMES} not all present in model.names",
                  file=sys.stderr)
            return 2
        print(f"filtering  : ship-only, class ids {keep}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        print(f"could not open {args.video}", file=sys.stderr)
        return 2
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video      : {w}x{h} @ {fps:.2f} fps, {total} frames")

    # This build decodes raw frames and leaves ORIENTATION_META unapplied
    # (ORIENTATION_AUTO == 0), so do the rotation here rather than trust it.
    meta = int(cap.get(cv2.CAP_PROP_ORIENTATION_META) or 0) % 360
    deg = meta if args.rotate == "auto" else int(args.rotate)
    rotation = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE}.get(deg)
    if deg in (90, 270):
        w, h = h, w
    src = "metadata" if args.rotate == "auto" else "flag"
    print(f"rotation   : {deg} deg cw ({src}, meta={meta}) -> {w}x{h}")

    live = can_display()
    out_path = args.save
    if not live and out_path is None:
        out_path = args.video.with_name(args.video.stem + "_pred.mp4")
        print("display    : none reachable (no DISPLAY/WAYLAND_DISPLAY) -- "
              f"cv2.imshow skipped, writing {out_path} instead")
    elif live:
        print("display    : cv2.imshow  [q/ESC quit, SPACE pause, s snapshot]")

    writer = None
    if out_path is not None:
        writer = cv2.VideoWriter(str(out_path),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not writer.isOpened():
            print(f"could not open writer for {out_path}", file=sys.stderr)
            return 2

    win = args.video.name
    if live:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, min(1600, w), int(min(1600, w) * h / w))

    n = 0
    n_boxes = 0
    tally = collections.Counter()
    paused = False
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok:
                    break
                if rotation is not None:
                    frame = cv2.rotate(frame, rotation)
                result = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                       iou=args.iou, device=args.device,
                                       classes=keep, verbose=False)[0]
                n_boxes += draw(frame, result, names, tally)
                n += 1
                hud = f"{n}/{total}  conf>={args.conf}  {len(result.boxes)} det"
                cv2.putText(frame, hud, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(frame, hud, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 1, cv2.LINE_AA)
                if writer is not None:
                    writer.write(frame)
                if not live and n % 50 == 0:
                    print(f"  {n}/{total} frames, {n_boxes} boxes so far", flush=True)
                if args.max_frames and n >= args.max_frames:
                    break

            if live:
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                if key == ord("s"):
                    snap = args.video.with_name(f"{args.video.stem}_frame{n:05d}.png")
                    cv2.imwrite(str(snap), frame)
                    print(f"saved {snap}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if live:
            cv2.destroyAllWindows()

    per = n_boxes / n if n else 0.0
    print(f"done       : {n} frames, {n_boxes} boxes ({per:.1f}/frame)")
    for name, count in tally.most_common():
        mark = "  <- shipped" if name in SHIP_NAMES else ""
        print(f"    {name:14s} {count}{mark}")
    if writer is not None:
        print(f"written    : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
