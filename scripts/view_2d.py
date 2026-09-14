#!/usr/bin/env python3
"""Read-only 2D results viewer: six ring cameras per keyframe (2026-09-12).

Sibling of scripts/view_boxes_3d.py and built the same way — one JSON per keyframe
under <out>/kf/, an <out>/index.json, downscaled JPEGs under <out>/img/, and
viewer2d/index.html copied in as <out>/index.html — but the subject is the 2D chain:

  * Stage 3m proposals (`stage3_merged/scenes/<scene>/proposals.jsonl`): the box,
    its class, score and arm (arm_a = COCO YOLO, arm_b = RSUD20K fine-tune) — or, with
    `--proposals-dir .../stage3_checked`, the same rows after the Stage 3c VLM check,
    whose per-box verdict rides into the keyframe JSON as `vlm`;
  * Stage 4 SAM masks (`stage4_masks/scenes/<scene>/masks.jsonl` + masks/*.npz),
    as one polygon outline per box rather than a PNG per mask;
  * the 3D outcome (`stage7_track/scenes/<scene>/boxes.jsonl`), joined by
    (keyframe_token, channel, proposal_index): the `status` every 2D box ended in,
    and for `fit` rows the 3D box projected back into the image it came from.

    python -m scripts.view_2d --paths configs/batch_20260912/chunk_33.yaml \
        --scene dhaka_20260911_170051_chunk_0005 --out /tmp/view2d --serve 8768
"""
from __future__ import annotations
import argparse, collections, json, os, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.stage5_lift.lift import MaskFile  # noqa: E402
from scripts.view_boxes_3d import load_calibs, pose_corrections, project_corners  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEREO_BOX_CONFIG = os.path.join(ROOT, "configs", "stereo_box.yaml")
# Grid order, row-major: the page lays these out 3-up over 3-up, so the first row is
# what the driver sees ahead and the second the two sides with the rear between them.
CHANNELS = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_LEFT", "CAM_BACK", "CAM_RIGHT")


def read_jsonl(path: str):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def mask_polygon(mask: np.ndarray, epsilon_px: float = 1.5, max_points: int = 200) -> list[int] | None:
    """Largest external contour of a bool mask as a flat [x0,y0,x1,y1,...] pixel list.

    One polygon, not one PNG: a 1280x720 mask is 1.4 MB unpacked and ~40 vertices as an
    outline, and the page only ever draws it as a translucent fill. A mask that decomposes
    into several blobs (an occluded car) loses its smaller parts — deliberate, the outline
    is an overlay, never the annotation.
    """
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    contour = max(cnts, key=cv2.contourArea)
    poly = cv2.approxPolyDP(contour, epsilon_px, True).reshape(-1, 2)
    while len(poly) > max_points and epsilon_px < 64.0:        # bounded: doubles at most 6x
        epsilon_px *= 2.0
        poly = cv2.approxPolyDP(contour, epsilon_px, True).reshape(-1, 2)
    return poly.reshape(-1).astype(int).tolist() if len(poly) >= 3 else None


def write_image(src: str, dst: str, max_width: int) -> tuple[int, int]:
    """Downscaled JPEG copy; returns the DESTINATION (w, h). Hard-links when it can.

    view_boxes_3d hard-links two images per keyframe; six cameras over a 1494-keyframe
    scene is 1.8 GB of originals, and the export root is a different filesystem from the
    dataroot anyway (link() would fall back to a full copy). Downscaling is the cheaper
    copy, and 960 px is plenty for a 2x3 grid.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"{src}: not a readable image")
    h, w = img.shape[:2]
    if w > max_width:
        h, w = int(round(h * max_width / w)), max_width
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if os.path.lexists(dst):
        os.unlink(dst)
    cv2.imwrite(dst, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return w, h


def wireframe(box: dict, calib: dict, image_size) -> list[float] | None:
    """The fitted 3D box projected back into its own camera, as [u0,v0,...,u7,v7].

    `project_corners` zeroes a corner that is behind the camera; the page skips the
    edges that touch one. Reuses view_boxes_3d so the 2D and 3D viewers cannot disagree
    about where a box lands in an image.
    """
    uv, vis = project_corners(box["translation_m"], box["size_wlh_m"], box["yaw_rad"],
                              np.asarray(calib["K"]), np.asarray(calib["T_cam_ego"]), image_size)
    if not vis.any():
        return None
    return [round(float(v), 1) for v in uv.reshape(-1)]


def export_keyframe(i, kf_row, props_by_ch, boxes_by_key, mask_file, calibs, dataroot,
                    out_dir, max_width=960, poly_budget=6000) -> dict:
    """One <out>/kf/<i>.json + its six JPEGs. Returns the index.json entry."""
    token = kf_row["keyframe_token"]
    cams, totals = {}, collections.Counter()
    for ch in CHANNELS:
        cam = kf_row["cameras"].get(ch)
        if cam is None:
            continue
        src = cam["path"] if os.path.isabs(cam["path"]) else os.path.join(dataroot, cam["path"])
        rel = f"img/{token}_{ch}.jpg"                          # by token: two scenes, one --out
        w, h = write_image(src, os.path.join(out_dir, rel), max_width)
        row = props_by_ch.get(ch) or {}
        # Stage 3c relabels IN PLACE, one verdict per proposal index, so the verdict for
        # box j rides along with it; a proposals dir that never saw 3c has no vlm_check.
        verdicts = (row.get("vlm_check") or {}).get("verdicts") or ()
        native = tuple(row.get("image_size_px") or (cam.get("width_px", w), cam.get("height_px", h)))
        boxes = []
        for j, xyxy in enumerate(row.get("boxes_xyxy_px", ())):
            tracked = boxes_by_key.get((ch, j))
            b = {"i": j, "xyxy": [round(float(v), 1) for v in xyxy],
                 "cls": row["class_names"][j], "score": round(float(row["scores"][j]), 4),
                 "arm": row["proposal_arm"][j]}
            if j < len(verdicts):
                b["vlm"] = verdicts[j]
            if tracked is not None:
                b["status"] = tracked["status"]
                b["track_id"] = tracked.get("track_id")
                if tracked["status"] == "fit" and tracked.get("box") and calibs and ch in calibs:
                    b["wire"] = wireframe(tracked["box"], calibs[ch], native)
                    b["depth_m"] = (tracked.get("stereo") or {}).get("d_med_m")
                totals[tracked["status"]] += 1
            if mask_file is not None and poly_budget > 0:
                try:
                    poly = mask_polygon(mask_file.mask(ch, j))
                except Exception:                              # a channel Stage 4 never wrote
                    poly = None
                if poly:
                    b["poly"] = poly
                    poly_budget -= len(poly) // 2
            boxes.append(b)
        cams[ch] = {"image": rel, "w": w, "h": h, "native": list(native), "boxes": boxes}
        totals["n_proposals"] += len(boxes)
    payload = {"index": i, "token": token, "t_ns": kf_row["t_ns"], "cameras": cams}
    os.makedirs(os.path.join(out_dir, "kf"), exist_ok=True)
    with open(os.path.join(out_dir, "kf", f"{i:05d}.json"), "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    return {"index": i, "token": token, "t_ns": kf_row["t_ns"],
            "n_proposals": totals["n_proposals"], "n_fit": totals["fit"]}


def group_by_keyframe(path: str, key) -> dict:
    """{keyframe_token: {key(row): row}} for a whole-scene jsonl; {} when it is absent."""
    out: dict[str, dict] = {}
    if not path or not os.path.isfile(path):
        return out
    for row in read_jsonl(path):
        out.setdefault(row["keyframe_token"], {})[key(row)] = row
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    ap.add_argument("--config", default=STEREO_BOX_CONFIG,
                    help="stage config the image-space camera pose correction is read from")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage1-dir", default=None, help="default <work_root>/stage1_ingestion")
    ap.add_argument("--proposals-dir", default=None, help="default <work_root>/stage3_merged")
    ap.add_argument("--masks-dir", default=None, help="default <work_root>/stage4_masks; '' disables")
    ap.add_argument("--boxes-dir", default=None,
                    help="default <work_root>/stage7_track; absent tree = no 3D outcome badges")
    ap.add_argument("--max-width", type=int, default=960, help="downscale images to this width")
    ap.add_argument("--poly-budget", type=int, default=6000, help="max mask-polygon points per keyframe")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--serve", type=int, default=0, help="port; 0 = export only")
    a = ap.parse_args(argv)
    paths = load_paths(a.paths)
    work = paths.work_root
    stage1 = a.stage1_dir or os.path.join(work, "stage1_ingestion")
    props_dir = a.proposals_dir or os.path.join(work, "stage3_merged")
    masks_dir = os.path.join(work, "stage4_masks") if a.masks_dir is None else a.masks_dir
    boxes_dir = a.boxes_dir or os.path.join(work, "stage7_track")

    kfs = list(read_jsonl(os.path.join(stage1, "scenes", a.scene, "keyframes.jsonl")))
    props = group_by_keyframe(os.path.join(props_dir, "scenes", a.scene, "proposals.jsonl"),
                              lambda r: r["channel"])
    boxes = group_by_keyframe(os.path.join(boxes_dir, "scenes", a.scene, "boxes.jsonl"),
                              lambda r: (r["channel"], r["proposal_index"]))
    mask_paths = {r["keyframe_token"]: os.path.join(masks_dir, r["mask_path"])
                  for r in read_jsonl(os.path.join(masks_dir, "scenes", a.scene, "masks.jsonl"))
                  } if masks_dir and os.path.isfile(os.path.join(masks_dir, "scenes", a.scene, "masks.jsonl")) else {}
    corrections = pose_corrections(a.config)
    calibs = load_calibs(paths, kfs[0], corrections, CHANNELS)
    os.makedirs(a.out, exist_ok=True)

    def one(item):
        i, kf = item
        mf = None
        path = mask_paths.get(kf["keyframe_token"])
        if path and os.path.isfile(path):
            mf = MaskFile(path)
        try:
            return export_keyframe(i, kf, props.get(kf["keyframe_token"], {}),
                                   boxes.get(kf["keyframe_token"], {}), mf, calibs,
                                   paths.dataroot, a.out, a.max_width, a.poly_budget)
        finally:
            if mf is not None:
                mf.close()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as pool:
        index = list(pool.map(one, enumerate(kfs)))
    json.dump({"scene": a.scene, "chunk": os.path.basename(os.path.dirname(work)),
               "channels": list(CHANNELS), "keyframes": index,
               "has_outcomes": bool(boxes), "has_masks": bool(mask_paths),
               "camera_pose_corrections": corrections},
              open(os.path.join(a.out, "index.json"), "w"), separators=(",", ":"))
    for name in ("index.html", "draw.js"):
        shutil.copy2(os.path.join(ROOT, "viewer2d", name), os.path.join(a.out, name))
    print(f"exported {len(index)} keyframes to {a.out} in {time.time() - t0:.1f}s")
    if a.serve:
        os.chdir(a.out)
        print(f"serving http://127.0.0.1:{a.serve}/  (Ctrl-C to stop)")
        ThreadingHTTPServer(("127.0.0.1", a.serve), SimpleHTTPRequestHandler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
