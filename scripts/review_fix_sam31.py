#!/usr/bin/env python3
"""Click-to-fix review of low-IoU pipeline boxes with SAM 3.1 (local, CVAT-like).

The CVAT review loop (docs/CVAT_GUIDE.md) shows the reviewer every machine box
next to the human answer key but offers only vertex dragging to repair one.
This tool is the same review, narrowed to the boxes that NEED it and armed with
SAM 3.1 point prompts:

  1. QUEUE   — every kept Stage 3/4 box whose best IoU against the projected
               nuScenes GT twin lies in [--iou-min, --iou-max) (default 0.3–0.5:
               "on the right object, loosely"; widen with --iou-min 0 to include
               the unmatched ones). Matching reuses scripts/eval_2d.py's own
               dedup + IoU code, so the queue and the metric agree by construction.
  2. FIX     — a browser page (stdlib http.server, no CVAT needed): the image,
               the machine box (red), the GT box (green), and the SAM 3.1 mask
               preview (cyan) re-segmented live from the reviewer's clicks
               (left = object, right/shift = background) optionally seeded with
               the existing box. Accept / Keep / Delete / Skip per box.
  3. STORE   — every decision is appended to <work_root>/review_sam31/fixes.jsonl
               (last record per box wins; the mask PNG is kept alongside) and
               `export` folds them into <work_root>/cvat_export_fixed/, a COCO
               export with the SAME shape as cvat_export/ — publishable with
               scripts/cvat_setup.py, re-scorable with
               `scripts/eval_2d.py --pred-export cvat_export_fixed`.

Provenance (decision C13): anything that leaves this tool is HUMAN-edited. Each
record says so (`provenance.human = true`, the click list, the checkpoint that
produced the mask); the fixed export is a separate directory and never
overwrites cvat_export/ or any stage tree. Feeding it back into Stages 5–9 is a
decision to record first, not something this script does.

    python -m scripts.review_fix_sam31 queue  [--scenes scene-0061] [--iou-min 0.3 --iou-max 0.5]
    python -m scripts.review_fix_sam31 serve  [--port 8765] [same filters]
    python -m scripts.review_fix_sam31 export [--scenes ...]

Run `serve` with the ano_pipe interpreter (it imports `sam3`), then open
http://localhost:8765 (or the machine's tailnet address; bind with --host 0.0.0.0).
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
import threading
import time
import uuid
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
# The click client lives in the sam31_example/ checkout (the directory name has
# moved once already; both spellings are searched).
for _cand in ("sam31_example", "sam31"):
    if os.path.isdir(os.path.join(ROOT, _cand)):
        sys.path.insert(0, os.path.join(ROOT, _cand))
        break

from pipeline.common.manifest import write_json_atomic  # noqa: E402
from pipeline.common.paths import load_paths  # noqa: E402
from scripts.eval_2d import by_image, iou_matrix, predictions_of  # noqa: E402
from scripts.export_cvat_coco import mask_to_polygons  # noqa: E402

SPEC = "dhakascenes-pilot/sam31-click-fix/v1"
REVIEW_DIR = "review_sam31"
FIXED_EXPORT_DIR = "cvat_export_fixed"
ACTIONS = ("fix", "keep", "delete", "skip")
COUNT_KEY = {"fix": "fixed", "keep": "kept", "delete": "deleted", "skip": "skipped"}

# Extra COCO attributes the fixed export carries on EVERY annotation. They are
# declared in the export's `info.cvat_label_attributes` so cvat_setup.py can add
# them to a NEW project's label schema without touching the existing projects.
REVIEW_ATTRIBUTES = [
    {"name": "review", "input_type": "select", "mutable": False,
     "default_value": "none", "values": ["none", "sam31_fixed", "kept_by_reviewer", "human_added", "human_tracked"]},
    {"name": "iou_before", "input_type": "number", "mutable": False,
     "default_value": "0", "values": ["0", "1", "0.01"]},
    {"name": "iou_after", "input_type": "number", "mutable": False,
     "default_value": "0", "values": ["0", "1", "0.01"]},
]


# ----------------------------------------------------------------------------
# queue
# ----------------------------------------------------------------------------

def _channel_of(file_name: str) -> str:
    m = re.search(r"(CAM_[A-Z_]+)", file_name)
    if not m:
        raise ValueError(f"no camera channel in {file_name!r}")
    return m.group(1)


def _xyxy(bbox_xywh) -> list[float]:
    x, y, w, h = bbox_xywh
    return [float(x), float(y), float(x + w), float(y + h)]


def _iou_xyxy(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_chains(paths, scene: str) -> dict[str, list[dict]]:
    """channel -> [{file_name, keyframe_token}] in keyframe order (Stage 1)."""
    kf_path = os.path.join(paths.work_root, "stage1_ingestion", "scenes", scene, "keyframes.jsonl")
    chains: dict[str, list[dict]] = defaultdict(list)
    if not os.path.isfile(kf_path):
        return chains
    with open(kf_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            for channel, cam in r["cameras"].items():
                chains[channel].append({"file_name": cam["path"], "keyframe_token": r["keyframe_token"]})
    return chains


def load_stage4_index(paths, scene: str) -> dict[tuple, dict]:
    """(channel, rounded proposal xyxy, class) -> candidate provenance."""
    path = os.path.join(paths.work_root, "stage4_masks", "scenes", scene, "masks.jsonl")
    index: dict[tuple, dict] = {}
    if not os.path.isfile(path):
        return index
    with open(path) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            for c in row["candidates"]:
                key = (c["channel"], tuple(round(v, 2) for v in c["proposal_box_xyxy_px"]), c["class_name"])
                index[key] = {
                    "keyframe_token": c.get("keyframe_token", row.get("keyframe_token")),
                    "proposal_index": c["proposal_index"],
                    "mask_box_xyxy_px": c.get("mask_box_xyxy_px"),
                    "n_mask_px": c.get("n_mask_px"),
                }
    return index


def build_queue(paths, scenes, iou_min: float, iou_max: float) -> dict:
    """Every kept prediction with best-GT IoU in [iou_min, iou_max), plus per-image context."""
    pred_root = os.path.join(paths.work_root, "cvat_export")
    gt_root = os.path.join(paths.work_root, "cvat_export_gt")
    names = sorted(
        n for n in os.listdir(pred_root)
        if os.path.isfile(os.path.join(gt_root, n, "instances.json"))
    )
    if scenes:
        names = [n for n in names if n in scenes]

    items: list[dict] = []
    categories: set[str] = set()        # full taxonomy from the export docs
    context: dict[str, dict] = {}       # f"{scene}|{file_name}" -> {preds, gts}
    chains_out: dict[str, dict[str, list[str]]] = {}
    counts = {"predictions": 0, "in_band": 0, "unmatched": 0}
    ledger = {"annotations_read": 0, "suppressed_dropped": 0,
              "mask_twins_dropped": 0, "predictions_counted": 0}

    for scene in names:
        with open(os.path.join(pred_root, scene, "instances.json")) as fh:
            pred_doc = json.load(fh)
        with open(os.path.join(gt_root, scene, "instances.json")) as fh:
            gt_doc = json.load(fh)
        categories.update(c["name"] for c in pred_doc.get("categories", []))
        pred_rows, pred_names = by_image(pred_doc)
        gt_rows, gt_names = by_image(gt_doc)
        stage1_chains = load_chains(paths, scene)
        stage4 = load_stage4_index(paths, scene)

        # Chain = the keyframe images of one camera, chronological. Stage 1's
        # order when available (it is the time base), the export's order otherwise.
        chains: dict[str, list[str]] = {}
        token_of: dict[str, str | None] = {}
        if stage1_chains:
            for channel, rows in stage1_chains.items():
                chains[channel] = [r["file_name"] for r in rows]
                for r in rows:
                    token_of[r["file_name"]] = r["keyframe_token"]
        else:
            for img in pred_doc["images"]:
                chains.setdefault(_channel_of(img["file_name"]), []).append(img["file_name"])
        chains_out[scene] = chains
        frame_index = {fn: i for ch in chains.values() for i, fn in enumerate(ch)}

        for file_name in sorted(set(pred_rows) | set(gt_rows)):
            preds = predictions_of(pred_rows.get(file_name, ()), ledger)
            gts = list(gt_rows.get(file_name, ()))
            pb = np.array([p["bbox"] for p in preds], dtype=np.float64).reshape(-1, 4)
            gb = np.array([g["bbox"] for g in gts], dtype=np.float64).reshape(-1, 4)
            ious = iou_matrix(pb, gb)
            channel = _channel_of(file_name)
            # Stage 4 polygon of each kept box, from its mask-twin row (same
            # bbox/category/score key as export_fixed) — the frame checker draws
            # them so mask quality is judged at a glance, not by opening CVAT.
            seg_of = {}
            for a in pred_rows.get(file_name, ()):
                if a.get("segmentation"):
                    seg_of[(tuple(a["bbox"]), a["category_id"],
                            (a.get("attributes") or {}).get("score"))] = a["segmentation"]
            context[f"{scene}|{file_name}"] = {
                "preds": [{"ann_id": p["id"], "bbox_xyxy": _xyxy(p["bbox"]),
                           "category": pred_names[p["category_id"]],
                           "score": p.get("attributes", {}).get("score"),
                           "segmentation": seg_of.get((tuple(p["bbox"]), p["category_id"],
                                                       (p.get("attributes") or {}).get("score")))} for p in preds],
                "gts": [{"bbox_xyxy": _xyxy(g["bbox"]), "category": gt_names[g["category_id"]]}
                        for g in gts],
            }
            counts["predictions"] += len(preds)
            for i, p in enumerate(preds):
                if len(gts):
                    j = int(np.argmax(ious[i]))
                    best = float(ious[i, j])
                    gt_box, gt_cat = _xyxy(gts[j]["bbox"]), gt_names[gts[j]["category_id"]]
                else:
                    best, gt_box, gt_cat = 0.0, None, None
                if best == 0.0:
                    counts["unmatched"] += 1
                    gt_box, gt_cat = None, None     # argmax of all-zeros is not a match
                if not (iou_min <= best < iou_max):
                    continue
                counts["in_band"] += 1
                bbox = _xyxy(p["bbox"])
                cat = pred_names[p["category_id"]]
                prov = stage4.get((channel, tuple(round(v, 2) for v in bbox), cat), {})
                items.append({
                    "id": f"{scene}/{p['id']}",
                    "scene": scene,
                    "file_name": file_name,
                    "channel": channel,
                    "frame_index": frame_index.get(file_name),
                    "keyframe_token": token_of.get(file_name, prov.get("keyframe_token")),
                    "proposal_index": prov.get("proposal_index"),
                    "ann_id": p["id"],
                    "category": cat,
                    "score": p.get("attributes", {}).get("score"),
                    "source": p.get("attributes", {}).get("source"),
                    "bbox_xyxy": bbox,
                    "stage4_mask_box_xyxy": prov.get("mask_box_xyxy_px"),
                    "iou": round(best, 4),
                    "gt_bbox_xyxy": gt_box,
                    "gt_category": gt_cat,
                })
    items.sort(key=lambda it: (it["scene"], it["channel"], it["frame_index"] or 0, it["iou"]))
    return {
        "spec": SPEC,
        "band": {"iou_min": iou_min, "iou_max": iou_max},
        "scenes": names,
        "categories": sorted(categories),
        "counts": counts,
        "dedup": ledger,
        "items": items,
        "context": context,
        "chains": chains_out,
    }


# ----------------------------------------------------------------------------
# ledger
# ----------------------------------------------------------------------------

class Ledger:
    """Append-only fixes.jsonl; the latest record per item id is the decision."""

    def __init__(self, review_root: str) -> None:
        self.root = review_root
        self.path = os.path.join(review_root, "fixes.jsonl")
        self.masks_dir = os.path.join(review_root, "masks")
        os.makedirs(self.masks_dir, exist_ok=True)
        self._lock = threading.Lock()
        self.latest: dict[str, dict] = {}
        if os.path.isfile(self.path):
            with open(self.path) as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        self.latest[rec["id"]] = rec

    def append(self, record: dict) -> None:
        with self._lock:
            with open(self.path, "a") as fh:
                fh.write(json.dumps(record, separators=(",", ":")) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self.latest[record["id"]] = record

    def mask_path(self, item_id: str) -> str:
        scene, ann = item_id.split("/", 1)
        d = os.path.join(self.masks_dir, scene)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"{ann}.png")

    def snapshot(self) -> dict[str, dict]:
        """Point-in-time copy of latest. Readers MUST use this, not .latest:
        the server is threaded and append() inserts new keys mid-iteration."""
        with self._lock:
            return dict(self.latest)

    def status(self) -> dict[str, str]:
        return {k: v["action"] for k, v in self.snapshot().items()}


# ----------------------------------------------------------------------------
# SAM 3.1 session over keyframe chains
# ----------------------------------------------------------------------------

class ChainSegmenter:
    """One warm SAM 3.1 process; one frame folder per (scene, channel) chain."""

    def __init__(self, paths, review_root: str, chains: dict[str, dict[str, list[str]]]) -> None:
        from sam31_click_tracker import Sam31ClickSession  # sam31_example/ on sys.path
        self._Session = Sam31ClickSession
        self.paths = paths
        self.chains = chains
        self.chain_root = os.path.join(review_root, "chains")
        self.workdir = os.path.join(review_root, "session")
        self.lock = threading.Lock()
        self.session = None
        self.current_chain: tuple[str, str] | None = None
        self.current_item: str | None = None
        self.checkpoint_path: str | None = None
        self.checkpoint_sha256: str | None = None

    def chain_dir(self, scene: str, channel: str) -> str:
        d = os.path.join(self.chain_root, scene, channel)
        files = self.chains[scene][channel]
        if not os.path.isdir(d) or len(os.listdir(d)) != len(files):
            os.makedirs(d, exist_ok=True)
            for old in os.listdir(d):
                os.unlink(os.path.join(d, old))
            for i, file_name in enumerate(files):
                os.symlink(os.path.join(self.paths.dataroot, file_name), os.path.join(d, f"{i:05d}.jpg"))
        return d

    def _ensure(self, scene: str, channel: str) -> None:
        d = self.chain_dir(scene, channel)
        if self.session is None:
            t0 = time.time()
            self.session = self._Session(None, None, self.workdir, frames_dir=d)
            self.checkpoint_path = os.path.realpath(str(self.session.checkpoint_path))
            blob = os.path.basename(self.checkpoint_path)
            self.checkpoint_sha256 = blob if re.fullmatch(r"[0-9a-f]{64}", blob) else None
            print(f"SAM 3.1 ready in {time.time() - t0:.1f}s ({self.checkpoint_path})", flush=True)
        elif self.current_chain != (scene, channel):
            self.session.load(d, len(self.chains[scene][channel]))
        self.current_chain = (scene, channel)

    def refine(self, item: dict, clicks: list, box: list | None) -> dict:
        with self.lock:
            self._ensure(item["scene"], item["channel"])
            if self.current_item != item["id"]:
                # A new object: forget the previous one's prompts and memories so
                # nothing from another frame conditions this mask.
                self.session.reset()
                self.current_item = item["id"]
            mask = self.session.refine(item["frame_index"], [tuple(c) for c in clicks], box=box)
            confidence = float(self.session.last_confidence)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return {"empty": True, "area": 0, "confidence": confidence}
        tight = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        ok, png = cv2.imencode(".png", mask.astype(np.uint8) * 255)
        if not ok:
            raise RuntimeError("mask PNG encode failed")
        return {
            "empty": False,
            "area": int(len(xs)),
            "confidence": confidence,
            "bbox_xyxy": tight,
            "segmentation": mask_to_polygons(mask),
            "mask_png_b64": base64.b64encode(png.tobytes()).decode("ascii"),
            "_mask": mask,
        }

    def track(self, item: dict, clicks: list, box: list | None) -> dict:
        """Replay one object's stored prompts on its seed frame, then propagate
        it across the (scene, channel) chain. Returns per-frame masks keyed by
        frame index (seed frame excluded) with SAM's per-frame confidence."""
        with self.lock:
            self._ensure(item["scene"], item["channel"])
            self.session.reset()
            self.current_item = None       # the replay owns the session now
            self.session.refine(item["frame_index"], [tuple(c) for c in clicks], box=box)
            result = self.session.track(seed_frame_index=item["frame_index"])
            confidences = result.get("confidences") or []
            frames: dict[int, dict] = {}
            for fi in sorted(result.get("covered") or []):
                if fi == item["frame_index"]:
                    continue
                m = cv2.imread(os.path.join(str(self.session.masks_dir), f"{fi:06d}.png"),
                               cv2.IMREAD_GRAYSCALE)
                if m is None:
                    continue
                mask = m > 0
                if not mask.any():
                    continue
                frames[fi] = {"mask": mask,
                              "confidence": float(confidences[fi]) if fi < len(confidences) else 0.0}
            stats = {k: result.get(k) for k in
                     ("tracked", "seed_confidence", "confidence_mean", "confidence_min")}
            return {"frames": frames, "stats": stats}

    def reset(self) -> None:
        with self.lock:
            if self.session is not None:
                self.session.reset()
            self.current_item = None

    def close(self) -> None:
        with self.lock:
            if self.session is not None:
                self.session.close()
                self.session = None


def _bbox_of_polygons(seg, fallback):
    """Tight box of the kept polygon fragments. mask_to_polygons drops tiny
    blobs, so a raw-mask bbox can be far wider than what is actually drawn
    (stray shadow pixels); the stored box must match the stored polygons."""
    xs = [v for ring in (seg or []) for v in ring[0::2]]
    ys = [v for ring in (seg or []) for v in ring[1::2]]
    if not xs:
        return fallback
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


class Suggester:
    """Zero-shot class suggestion for NEW objects (frames mode).

    SigLIP (google/siglip-base-patch16-224, already in the local HF cache)
    scores the drawn box's crop against the 10 taxonomy phrases: measured on
    this substrate at 73% top-1 / 87% top-3, 9 ms per crop. It is a SUGGESTION:
    the reviewer confirms or overrides, and the add record's category_source
    says which happened — the class label's provenance stays explicit.
    """

    MODEL = "google/siglip-base-patch16-224"

    def __init__(self, paths, categories) -> None:
        self.paths = paths
        self.categories = list(categories)
        self.lock = threading.Lock()
        self._ready = False

    def _ensure(self) -> None:
        if self._ready:
            return
        import torch
        from transformers import AutoModel, AutoProcessor
        self._torch = torch
        t0 = time.time()
        self.proc = AutoProcessor.from_pretrained(self.MODEL)
        self.model = AutoModel.from_pretrained(self.MODEL).to("cuda").eval()
        feats = lambda o: o if isinstance(o, torch.Tensor) else o.pooler_output  # noqa: E731 — transformers v5 returns ModelOutput
        self._feats = feats
        with torch.no_grad():
            t = self.proc(text=[f"a photo of {c}" for c in self.categories],
                          padding="max_length", return_tensors="pt").to("cuda")
            self.temb = torch.nn.functional.normalize(feats(self.model.get_text_features(**t)), dim=-1)
        print(f"SigLIP suggester ready in {time.time() - t0:.1f}s", flush=True)
        self._ready = True

    def suggest(self, image_bgr, bbox_xyxy) -> dict | None:
        if not self.categories:
            return None
        with self.lock:
            self._ensure()
            torch = self._torch
            x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
            h, w = image_bgr.shape[:2]
            m = 8
            crop = np.ascontiguousarray(image_bgr[max(0, y1 - m):min(h, y2 + m),
                                                  max(0, x1 - m):min(w, x2 + m), ::-1])
            if crop.size == 0:
                return None
            with torch.no_grad():
                i = self.proc(images=crop, return_tensors="pt").to("cuda")
                iemb = torch.nn.functional.normalize(self._feats(self.model.get_image_features(**i)), dim=-1)
                p = (iemb @ self.temb.T * self.model.logit_scale.exp() + self.model.logit_bias).softmax(-1)[0]
            top = p.topk(min(3, len(self.categories)))
            return {"category": self.categories[int(top.indices[0])],
                    "top": [[self.categories[int(j)], round(float(v), 3)]
                            for v, j in zip(top.values, top.indices)],
                    "model": self.MODEL}


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------

class ReviewApp:
    def __init__(self, paths, queue: dict, ledger: Ledger, segmenter: ChainSegmenter,
                 mode: str = "band") -> None:
        self.paths = paths
        self.queue = queue
        self.items = {it["id"]: it for it in queue["items"]}
        self.ledger = ledger
        self.seg = segmenter
        self.mode = mode                           # "band" (IoU queue) | "frames" (checker gate)
        self.last_refine: dict[str, dict] = {}     # item id -> last refine result (server-side truth)
        self.suggester = Suggester(paths, queue.get("categories") or [])

    # -- GET ---------------------------------------------------------------
    def api_queue(self) -> dict:
        status = self.ledger.status()
        out = {
            "band": self.queue["band"], "scenes": self.queue["scenes"], "counts": self.queue["counts"],
            "items": [{**it, "status": status.get(it["id"], "pending")} for it in self.queue["items"]],
        }
        if self.mode == "frames":
            out["chains"] = self.queue["chains"]
            out["categories"] = self.queue.get("categories") or []
            out["frames_ok"] = sorted(k[len("frame|"):] for k, v in status.items()
                                      if k.startswith("frame|") and v == "frame_ok")
            out["adds"] = [{k: r.get(k) for k in ("id", "scene", "file_name", "category",
                                                  "bbox_xyxy", "segmentation", "track_group",
                                                  "review_kind", "mask_confidence", "hops",
                                                  "mask_area_px", "seed_area_px")}
                           for r in self.ledger.snapshot().values() if r.get("action") == "add"]
        return out

    def api_context(self, scene: str, file_name: str) -> dict:
        ctx = self.queue["context"]
        # nuScenes file names contain '+' (timezone); a client that does not
        # percent-encode it arrives here with a space instead.
        for candidate in (file_name, file_name.replace(" ", "+")):
            if f"{scene}|{candidate}" in ctx:
                return ctx[f"{scene}|{candidate}"]
        return {"preds": [], "gts": []}

    def image_bytes(self, file_name: str) -> bytes:
        if ".." in file_name.split("/") or os.path.isabs(file_name):
            raise FileNotFoundError(file_name)
        path = os.path.join(self.paths.dataroot, file_name)
        with open(path, "rb") as fh:
            return fh.read()

    # -- POST --------------------------------------------------------------
    def api_refine(self, body: dict) -> dict:
        clicks = [[float(c[0]), float(c[1]), int(c[2])] for c in body.get("clicks", [])]
        if body.get("new"):
            # Frames mode, new object: the reviewer drew a box on empty canvas.
            # There is no queue item — synthesize one so ChainSegmenter can key
            # the (scene, channel) chain and the reset-per-object semantics.
            scene, file_name = body["scene"], body["file_name"]
            channel = _channel_of(file_name)
            try:
                fi = self.queue["chains"][scene][channel].index(file_name)
            except (KeyError, ValueError):
                raise ValueError(f"unknown frame: {scene} {file_name}")
            item = {"id": body["temp_id"], "scene": scene, "channel": channel,
                    "frame_index": fi, "gt_bbox_xyxy": None}
            box = [float(v) for v in body["box"]] if body.get("box") else None
            gts = self.api_context(scene, file_name)["gts"]
        else:
            item = self.items[body["id"]]
            box = item["bbox_xyxy"] if body.get("use_box") else None
            gts = None
        if not clicks and box is None:
            raise ValueError("need at least one click or the seed box")
        result = self.seg.refine(item, clicks, box)
        if body.get("new"):
            result["iou_after"] = (
                round(max((_iou_xyxy(result["bbox_xyxy"], g["bbox_xyxy"]) for g in gts), default=0.0), 4)
                if not result["empty"] and gts else None
            )
        else:
            result["iou_after"] = (
                round(_iou_xyxy(result["bbox_xyxy"], item["gt_bbox_xyxy"]), 4)
                if not result["empty"] and item["gt_bbox_xyxy"] else None
            )
        result["clicks"], result["used_box"] = clicks, box is not None
        result["seed_box_xyxy"] = box
        if body.get("new") and body.get("suggest") and not result["empty"]:
            try:
                img = cv2.imread(os.path.join(self.paths.dataroot, file_name))
                result["suggest"] = self.suggester.suggest(img, result["bbox_xyxy"]) if img is not None else None
            except Exception as exc:  # noqa: BLE001 — a suggestion must never break the refine
                print(f"suggest failed: {type(exc).__name__}: {exc}", flush=True)
                result["suggest"] = None
        self.last_refine[item["id"]] = result
        return {k: v for k, v in result.items() if k != "_mask"}

    def api_decide(self, body: dict) -> dict:
        item = self.items[body["id"]]
        action = body["action"]
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {ACTIONS}")
        record = {
            "spec": SPEC, "ts_unix": time.time(), "id": item["id"], "action": action,
            "scene": item["scene"], "file_name": item["file_name"], "channel": item["channel"],
            "keyframe_token": item["keyframe_token"], "proposal_index": item["proposal_index"],
            "ann_id": item["ann_id"], "category": item["category"], "score": item["score"],
            "bbox_before_xyxy": item["bbox_xyxy"], "iou_before": item["iou"],
            "gt_bbox_xyxy": item["gt_bbox_xyxy"], "gt_category": item["gt_category"],
            "note": body.get("note") or None,
            "provenance": {"tool": "sam31_click", "human": True,
                           "checkpoint_path": self.seg.checkpoint_path,
                           "checkpoint_sha256": self.seg.checkpoint_sha256},
        }
        if action == "fix":
            last = self.last_refine.get(item["id"])
            if not last or last.get("empty"):
                raise ValueError("nothing to accept: refine a non-empty mask first")
            expect = body.get("expect_bbox")
            if expect is not None and [round(v, 1) for v in expect] != [round(v, 1) for v in last["bbox_xyxy"]]:
                raise ValueError("the mask changed since you looked (another tab refined this box) — re-segment and accept again")
            mask_path = self.ledger.mask_path(item["id"])
            cv2.imwrite(mask_path, last["_mask"].astype(np.uint8) * 255)
            record.update({
                "clicks": last["clicks"], "used_box": last["used_box"],
                "bbox_after_xyxy": last["bbox_xyxy"], "segmentation": last["segmentation"],
                "mask_area_px": last["area"], "mask_confidence": last["confidence"],
                "iou_after": last["iou_after"],
                "mask_png": os.path.relpath(mask_path, self.ledger.root),
            })
            # Reflect the accepted geometry in the served context and the item,
            # so the canvas (and any reload) shows the FIX, not the stale box.
            ctx_entry = self.queue["context"].get(f"{item['scene']}|{item['file_name']}")
            if ctx_entry:
                for p in ctx_entry["preds"]:
                    if p["ann_id"] == item["ann_id"]:
                        p["bbox_xyxy"] = last["bbox_xyxy"]
                        p["segmentation"] = last["segmentation"]
            item["bbox_xyxy"] = last["bbox_xyxy"]
        self.ledger.append(record)
        return {"ok": True, "id": item["id"], "action": action, "status": self.ledger.status()}

    def api_reset(self, body: dict) -> dict:
        self.seg.reset()
        self.last_refine.pop(body.get("id", ""), None)
        return {"ok": True}

    def api_frame_ok(self, body: dict) -> dict:
        """Frames mode: the human attests one whole image (Enter). Untouched boxes
        on an OK'd frame are APPROVED, not merely unreviewed — that distinction is
        the point of the checker gate, so it is recorded per frame."""
        scene, file_name = body["scene"], body["file_name"]
        ok = body.get("ok", True)
        record = {
            "spec": SPEC, "ts_unix": time.time(),
            "id": f"frame|{scene}|{file_name}",
            "action": "frame_ok" if ok else "frame_pending",
            "scene": scene, "file_name": file_name,
            "channel": body.get("channel") or _channel_of(file_name),
            "n_preds": body.get("n_preds"),
            "provenance": {"tool": "frame_review", "human": True},
        }
        self.ledger.append(record)
        return {"ok": True, "id": record["id"], "action": record["action"]}

    def api_add(self, body: dict) -> dict:
        """Frames mode: create (or retract) an object the pipeline missed.

        Create: the reviewer drew a box, SAM segmented it (api_refine with
        new=true), a class was picked — store mask + polygon + tight box as a
        NEW annotation. Retract: append add_removed for a given id (last wins).
        """
        if body.get("remove_group"):
            group = body["remove_group"]
            doomed = [r for r in self.ledger.snapshot().values() if r.get("action") == "add"
                      and (r["id"] == group or r.get("track_group") == group)]
            if not doomed:
                raise ValueError(f"no added objects in track group: {group}")
            for r in doomed:
                self.ledger.append({"spec": SPEC, "ts_unix": time.time(), "id": r["id"],
                                    "action": "add_removed", "scene": r["scene"],
                                    "file_name": r["file_name"],
                                    "provenance": {"tool": "sam31_click", "human": True}})
            return {"ok": True, "group": group, "removed": len(doomed)}

        if body.get("amend_id"):
            # A human re-segmented an existing added/tracked object: replace its
            # geometry, and the record becomes human-verified (C13) — the mask
            # on screen is now one a person shaped and approved.
            rec = self.ledger.latest.get(body["amend_id"])
            if not rec or rec.get("action") != "add":
                raise ValueError(f"not an added object: {body['amend_id']}")
            last = self.last_refine.get(body["temp_id"])
            if not last or last.get("empty"):
                raise ValueError("nothing to save: click the object until the mask is non-empty")
            mask_path = os.path.join(self.ledger.root, rec["mask_png"]) if rec.get("mask_png") else None
            if mask_path:
                cv2.imwrite(mask_path, last["_mask"].astype(np.uint8) * 255)
            amended = {
                **rec, "ts_unix": time.time(),
                "bbox_xyxy": _bbox_of_polygons(last["segmentation"], last["bbox_xyxy"]),
                "segmentation": last["segmentation"],
                "mask_area_px": last["area"], "mask_confidence": last["confidence"],
                "clicks": last["clicks"], "used_box": last["used_box"],
                "provenance": {"tool": "sam31_click", "human": True, "amended": True,
                               "amended_from": rec.get("provenance", {}).get("tool"),
                               "checkpoint_path": self.seg.checkpoint_path,
                               "checkpoint_sha256": self.seg.checkpoint_sha256},
            }
            self.ledger.append(amended)
            self.last_refine.pop(body["temp_id"], None)
            return {k: v for k, v in amended.items() if k != "clicks"}

        if body.get("remove_id"):
            prev = self.ledger.latest.get(body["remove_id"])
            if not prev or prev.get("action") != "add":
                raise ValueError(f"not an added object: {body['remove_id']}")
            self.ledger.append({"spec": SPEC, "ts_unix": time.time(), "id": prev["id"],
                                "action": "add_removed", "scene": prev["scene"],
                                "file_name": prev["file_name"],
                                "provenance": {"tool": "sam31_click", "human": True}})
            return {"ok": True, "id": prev["id"], "action": "add_removed"}

        scene, file_name, category = body["scene"], body["file_name"], body["category"]
        last = self.last_refine.get(body["temp_id"])
        if not last or last.get("empty"):
            raise ValueError("nothing to add: draw a box / click until the mask is non-empty")
        rec_id = f"new|{scene}|{uuid.uuid4().hex[:8]}"
        mask_dir = os.path.join(self.ledger.masks_dir, scene)
        os.makedirs(mask_dir, exist_ok=True)
        mask_path = os.path.join(mask_dir, f"{rec_id.rsplit('|', 1)[1]}-new.png")
        cv2.imwrite(mask_path, last["_mask"].astype(np.uint8) * 255)
        record = {
            "spec": SPEC, "ts_unix": time.time(), "id": rec_id, "action": "add",
            "scene": scene, "file_name": file_name, "channel": _channel_of(file_name),
            "category": category, "score": 0.0,
            "clicks": last["clicks"], "used_box": last["used_box"],
            "seed_box_xyxy": last.get("seed_box_xyxy"),
            "bbox_xyxy": _bbox_of_polygons(last["segmentation"], last["bbox_xyxy"]),
            "segmentation": last["segmentation"],
            "mask_area_px": last["area"], "mask_confidence": last["confidence"],
            "iou_vs_gt": last.get("iou_after"),
            "category_source": body.get("category_source") or "human_picked",
            "category_suggest": body.get("category_suggest"),
            "mask_png": os.path.relpath(mask_path, self.ledger.root),
            "note": body.get("note") or None,
            "provenance": {"tool": "sam31_click", "human": True,
                           "checkpoint_path": self.seg.checkpoint_path,
                           "checkpoint_sha256": self.seg.checkpoint_sha256},
        }
        self.ledger.append(record)
        self.last_refine.pop(body["temp_id"], None)
        return {k: v for k, v in record.items() if k != "clicks"}

    def api_track(self, body: dict) -> dict:
        """Propagate a saved added object across its (scene, channel) chain.

        SAM 3.1 is a video model: the seed object's stored prompts are replayed
        on its frame, then propagate_in_video runs both directions over the
        keyframe chain. Every frame whose confidence clears min_conf becomes its
        own `add` record — review_kind "tracked", hops = keyframe distance from
        the seed — so each propagated copy is individually reviewable and
        removable. Re-tracking a seed replaces its previous propagation.
        """
        rec = self.ledger.latest.get(body["id"])
        if not rec or rec.get("action") not in ("add", "fix"):
            raise ValueError("track needs a saved added object, or a pipeline box you FIXED (A) first")
        if rec.get("review_kind") == "tracked":
            raise ValueError("this is a propagated copy — track from its seed object")
        min_conf = float(body.get("min_conf", 0.5))
        scene, channel = rec["scene"], rec["channel"]
        chain = self.queue["chains"][scene][channel]
        seed_fi = chain.index(rec["file_name"])
        clicks = [tuple(c) for c in (rec.get("clicks") or [])]
        # add records store the drawn box; fix records store whether the
        # ORIGINAL machine box seeded the accepted refinement.
        box = rec.get("seed_box_xyxy") if rec["action"] == "add" else (
            rec.get("bbox_before_xyxy") if rec.get("used_box") else None)
        if not clicks and box is None:
            raise ValueError("the seed record stored no prompts to replay")
        result = self.seg.track(
            {"id": rec["id"], "scene": scene, "channel": channel, "frame_index": seed_fi},
            clicks, box)

        group = rec["id"]
        seed_area = rec.get("mask_area_px")
        replaced = 0
        for old in [r for r in self.ledger.snapshot().values() if r.get("action") == "add"
                    and r.get("track_group") == group and r["id"] != group]:
            self.ledger.append({"spec": SPEC, "ts_unix": time.time(), "id": old["id"],
                                "action": "add_removed", "scene": old["scene"],
                                "file_name": old["file_name"],
                                "provenance": {"tool": "sam31_track", "human": True}})
            replaced += 1

        fresh = self.ledger.latest.get(group)
        if not fresh or fresh.get("action") not in ("add", "fix"):
            raise ValueError("the seed was removed or re-decided while tracking — nothing added")
        rec = fresh
        made, skipped = [], []
        for fi in sorted(result["frames"]):
            data = result["frames"][fi]
            entry = {"frame_index": fi, "file_name": chain[fi],
                     "confidence": round(data["confidence"], 4),
                     "area": int(data["mask"].sum())}
            if data["confidence"] < min_conf:
                skipped.append(entry)
                continue
            mask = data["mask"]
            ys, xs = np.nonzero(mask)
            seg = mask_to_polygons(mask)
            tight = _bbox_of_polygons(seg, [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
            rec_id = f"new|{scene}|{uuid.uuid4().hex[:8]}"
            mask_dir = os.path.join(self.ledger.masks_dir, scene)
            os.makedirs(mask_dir, exist_ok=True)
            mask_path = os.path.join(mask_dir, f"{rec_id.rsplit('|', 1)[1]}-new.png")
            cv2.imwrite(mask_path, mask.astype(np.uint8) * 255)
            self.ledger.append({
                "spec": SPEC, "ts_unix": time.time(), "id": rec_id, "action": "add",
                "scene": scene, "file_name": chain[fi], "channel": channel,
                "category": rec["category"], "score": 0.0,
                "review_kind": "tracked", "track_group": group, "seed_id": group,
                "seed_area_px": seed_area, "hops": abs(fi - seed_fi),
                "bbox_xyxy": tight, "segmentation": seg,
                "mask_area_px": int(len(xs)), "mask_confidence": round(data["confidence"], 4),
                "mask_png": os.path.relpath(mask_path, self.ledger.root),
                # C13: SAM produced this mask; no human has seen this frame's
                # copy yet. human is FALSE until the frame is (re-)attested.
                "provenance": {"tool": "sam31_track", "human": False,
                               "human_initiated": True, "seed_id": group,
                               "checkpoint_path": self.seg.checkpoint_path,
                               "checkpoint_sha256": self.seg.checkpoint_sha256},
            })
            made.append({**entry, "id": rec_id, "bbox_xyxy": tight})
        if not rec.get("track_group"):
            self.ledger.append({**rec, "track_group": group, "ts_unix": time.time()})
        # A frame that gains a tracked box is no longer the frame the reviewer
        # attested: reopen it, so "frame OK" keeps meaning "a human saw
        # everything on it" (the checker gate's whole point).
        reopened = 0
        latest = self.ledger.snapshot()
        for m in made:
            fid = f"frame|{scene}|{m['file_name']}"
            if latest.get(fid, {}).get("action") == "frame_ok":
                self.ledger.append({"spec": SPEC, "ts_unix": time.time(), "id": fid,
                                    "action": "frame_pending", "scene": scene,
                                    "file_name": m["file_name"], "channel": channel,
                                    "note": f"reopened: tracked object {group} landed here",
                                    "provenance": {"tool": "sam31_track", "human": False}})
                reopened += 1
        return {"ok": True, "group": group, "seed_frame": seed_fi, "min_conf": min_conf,
                "made": made, "skipped": skipped, "replaced": replaced,
                "frames_reopened": reopened, "stats": result["stats"]}


def make_handler(app: ReviewApp):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quieter console
            if "/api/image/" not in (args[0] if args else ""):
                sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code: int = 200) -> None:
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            url = urlparse(self.path)
            try:
                if url.path == "/":
                    page = PAGE_FRAMES if app.mode == "frames" else PAGE
                    self._send(200, page.encode(), "text/html; charset=utf-8")
                elif url.path == "/api/queue":
                    self._json(app.api_queue())
                elif url.path == "/api/context":
                    q = parse_qs(url.query)
                    self._json(app.api_context(q["scene"][0], q["file"][0]))
                elif url.path.startswith("/api/image/"):
                    file_name = url.path[len("/api/image/"):]
                    self._send(200, app.image_bytes(file_name),
                               mimetypes.guess_type(file_name)[0] or "image/jpeg")
                else:
                    self._json({"error": "not found"}, 404)
            except FileNotFoundError as exc:
                self._json({"error": f"not found: {exc}"}, 404)
            except Exception as exc:  # noqa: BLE001 — surfaced to the page
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

        def do_POST(self):
            url = urlparse(self.path)
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                if url.path == "/api/refine":
                    self._json(app.api_refine(body))
                elif url.path == "/api/decide":
                    self._json(app.api_decide(body))
                elif url.path == "/api/reset":
                    self._json(app.api_reset(body))
                elif url.path == "/api/frame_ok":
                    self._json(app.api_frame_ok(body))
                elif url.path == "/api/add":
                    self._json(app.api_add(body))
                elif url.path == "/api/track":
                    self._json(app.api_track(body))
                else:
                    self._json({"error": "not found"}, 404)
            except Exception as exc:  # noqa: BLE001 — surfaced to the page
                self._json({"error": f"{type(exc).__name__}: {exc}"}, 500)

    return Handler


# ----------------------------------------------------------------------------
# export
# ----------------------------------------------------------------------------

def export_fixed(paths, scenes, ledger: Ledger) -> dict:
    """cvat_export + ledger -> cvat_export_fixed (same COCO shape, review attributes added)."""
    pred_root = os.path.join(paths.work_root, "cvat_export")
    out_root = os.path.join(paths.work_root, FIXED_EXPORT_DIR)
    names = sorted(n for n in os.listdir(pred_root) if os.path.isfile(os.path.join(pred_root, n, "instances.json")))
    if scenes:
        names = [n for n in names if n in scenes]
    by_scene: dict[str, dict[int, dict]] = defaultdict(dict)
    for rec in ledger.snapshot().values():
        if "ann_id" not in rec:
            continue        # frame_ok attestations (frames mode) decide no box
        by_scene[rec["scene"]][rec["ann_id"]] = rec

    summary = {"scenes": {}, "totals": {"fixed": 0, "kept": 0, "deleted": 0, "skipped": 0, "added": 0}}
    for scene in names:
        with open(os.path.join(pred_root, scene, "instances.json")) as fh:
            doc = json.load(fh)
        decisions = by_scene.get(scene, {})
        # The rectangle row carries the decision; its polygon twin is the next
        # row with the same (image, bbox, category, score) — export_cvat_coco.py
        # writes them adjacently, but we match by key rather than by position.
        twin_key = lambda a: (a["image_id"], tuple(a["bbox"]), a["category_id"],  # noqa: E731
                              (a.get("attributes") or {}).get("score"))
        decided_by_key: dict[tuple, dict] = {}
        for ann in doc["annotations"]:
            if ann["id"] in decisions:
                decided_by_key[twin_key(ann)] = decisions[ann["id"]]

        annotations = []
        counts = {"fixed": 0, "kept": 0, "deleted": 0, "skipped": 0, "added": 0}
        seen_fixed: set[tuple] = set()
        for ann in doc["annotations"]:
            attrs = dict(ann.get("attributes") or {})
            attrs.setdefault("review", "none")
            attrs.setdefault("iou_before", 0)
            attrs.setdefault("iou_after", 0)
            rec = decided_by_key.get(twin_key(ann))
            if rec is None:
                annotations.append({**ann, "attributes": attrs})
                continue
            key = twin_key(ann)
            first_of_pair = key not in seen_fixed
            seen_fixed.add(key)
            action = rec["action"]
            if first_of_pair:
                counts[COUNT_KEY[action]] += 1
            if action == "delete":
                continue
            attrs["iou_before"] = rec.get("iou_before") or 0
            if action == "keep":
                attrs["review"] = "kept_by_reviewer"
                attrs["iou_after"] = rec.get("iou_before") or 0
                annotations.append({**ann, "attributes": attrs})
                continue
            if action == "fix":
                x1, y1, x2, y2 = rec["bbox_after_xyxy"]
                attrs["review"] = "sam31_fixed"
                attrs["iou_after"] = rec.get("iou_after") or 0
                new = {**ann, "attributes": attrs, "bbox": [x1, y1, x2 - x1, y2 - y1]}
                if ann.get("segmentation") or not first_of_pair:
                    new["segmentation"] = rec["segmentation"]
                    new["area"] = float(rec["mask_area_px"])
                else:
                    new["area"] = float((x2 - x1) * (y2 - y1))
                annotations.append(new)
                continue
            annotations.append({**ann, "attributes": attrs})   # skip: unchanged

        # Objects the checker ADDED (frames mode): rect row + polygon twin, the
        # same two-row shape every pipeline box has, appended with fresh ids.
        # source = "human": no detector ever fired on these (C27 third value).
        img_id = {img["file_name"]: img["id"] for img in doc["images"]}
        cat_id = {c["name"]: c["id"] for c in doc["categories"]}
        next_id = max((a["id"] for a in doc["annotations"]), default=0) + 1
        for rec in sorted((r for r in ledger.snapshot().values()
                           if r.get("action") == "add" and r["scene"] == scene),
                          key=lambda r: r["ts_unix"]):
            if rec["file_name"] not in img_id or rec["category"] not in cat_id:
                print(f"  {scene}: SKIPPING added object {rec['id']} — "
                      f"unknown image or category {rec['category']!r}")
                continue
            x1, y1, x2, y2 = rec["bbox_xyxy"]
            attrs = {"score": 0.0, "suppressed": False, "source": "human",
                     "track_id": rec.get("track_group") or "",
                     "hops": rec.get("hops") or 0,
                     "review": "human_tracked" if rec.get("review_kind") == "tracked" else "human_added",
                     "iou_before": 0, "iou_after": rec.get("iou_vs_gt") or 0}
            base = {"image_id": img_id[rec["file_name"]], "category_id": cat_id[rec["category"]],
                    "bbox": [x1, y1, x2 - x1, y2 - y1], "iscrowd": 0, "attributes": dict(attrs)}
            annotations.append({**base, "id": next_id, "area": float((x2 - x1) * (y2 - y1))})
            annotations.append({**base, "id": next_id + 1, "segmentation": rec["segmentation"],
                                "area": float(rec["mask_area_px"]), "attributes": dict(attrs)})
            next_id += 2
            counts["added"] += 1

        out = {
            **doc,
            "info": {
                **doc.get("info", {}),
                "description": f"{doc.get('info', {}).get('description', scene)} + SAM 3.1 click fixes (HUMAN-edited, C13)",
                "review": {"spec": SPEC, "source_export": "cvat_export", "counts": counts,
                           "ledger": os.path.relpath(ledger.path, paths.work_root)},
                "cvat_label_attributes": REVIEW_ATTRIBUTES,
            },
            "annotations": annotations,
        }
        write_json_atomic(os.path.join(out_root, scene, "instances.json"), out)
        summary["scenes"][scene] = counts
        for k in counts:
            summary["totals"][k] += counts[k]
        print(f"  {scene}: {counts['fixed']} fixed, {counts['kept']} kept, {counts['deleted']} deleted, "
              f"{counts['skipped']} skipped, {counts['added']} added -> {out_root}/{scene}/instances.json")
    write_json_atomic(os.path.join(out_root, "review_summary.json"), summary)
    return summary


# ----------------------------------------------------------------------------
# page
# ----------------------------------------------------------------------------

PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>SAM 3.1 click-fix review</title>
<style>
  :root { --bg:#15171b; --panel:#1e2127; --fg:#e6e6e6; --mut:#9aa0a6; --acc:#4cc9f0; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.4 system-ui, sans-serif; display:flex; height:100vh; }
  #side { width:320px; background:var(--panel); display:flex; flex-direction:column; border-right:1px solid #2a2e36; }
  #side header { padding:10px 12px; border-bottom:1px solid #2a2e36; }
  #side header h1 { font-size:14px; margin:0 0 6px; }
  #filters select, #filters input { background:#111; color:var(--fg); border:1px solid #333; padding:3px 5px; margin:2px 0; width:100%; }
  #list { flex:1; overflow:auto; }
  .it { padding:6px 12px; border-bottom:1px solid #262a31; cursor:pointer; display:flex; gap:8px; align-items:center; }
  .it:hover { background:#262a31; } .it.sel { background:#2d3340; }
  .it .iou { font-variant-numeric:tabular-nums; width:44px; color:var(--acc); }
  .it .cat { flex:1; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .it .st { font-size:11px; padding:1px 6px; border-radius:8px; background:#333; color:var(--mut); }
  .st.fix { background:#1f6f43; color:#fff; } .st.keep { background:#3b5b8a; color:#fff; }
  .st.delete { background:#7a2d2d; color:#fff; } .st.skip { background:#5a5a2d; color:#fff; }
  #main { flex:1; display:flex; flex-direction:column; min-width:0; }
  #bar { padding:8px 12px; background:var(--panel); border-bottom:1px solid #2a2e36; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  #bar button { background:#2a2f3a; color:var(--fg); border:1px solid #3a404d; padding:5px 10px; cursor:pointer; border-radius:4px; }
  #bar button:hover { background:#343a48; } #bar button.pri { background:#1f6f43; border-color:#2a8f58; }
  #bar button.danger { background:#7a2d2d; border-color:#9a3d3d; }
  #bar label { color:var(--mut); display:flex; align-items:center; gap:4px; }
  #wrap { flex:1; position:relative; overflow:hidden; background:#000; }
  canvas { position:absolute; left:0; top:0; }
  #info { padding:6px 12px; background:var(--panel); border-top:1px solid #2a2e36; color:var(--mut); display:flex; gap:18px; flex-wrap:wrap; font-variant-numeric:tabular-nums; }
  #info b { color:var(--fg); }
  .legend span { display:inline-block; width:10px; height:10px; margin-right:4px; vertical-align:middle; }
  #msg { color:#ffb347; }
</style></head><body>
<div id="side">
  <header>
    <h1>SAM 3.1 click-fix review</h1>
    <div id="filters">
      <select id="fScene"><option value="">all scenes</option></select>
      <select id="fStatus"><option value="">all statuses</option><option value="pending">pending</option><option value="fix">fixed</option><option value="keep">kept</option><option value="delete">deleted</option><option value="skip">skipped</option></select>
      <input id="fCat" placeholder="class filter (substring)">
    </div>
    <div id="counts" style="color:var(--mut);margin-top:6px"></div>
  </header>
  <div id="list"></div>
</div>
<div id="main">
  <div id="bar">
    <button id="bPrev" title="←">◀</button><button id="bNext" title="→">▶</button>
    <button id="bNextPending" title="N">next pending</button>
    <label><input type="checkbox" id="useBox" checked> seed from box (B)</label>
    <label><input type="checkbox" id="autoZoom" checked> auto-zoom (F)</label>
    <button id="bSeed" title="G">re-segment (G)</button>
    <button id="bClear" title="R">clear clicks (R)</button>
    <span style="flex:1"></span>
    <button id="bFix" class="pri" title="A">accept fix (A)</button>
    <button id="bKeep" title="K">keep as is (K)</button>
    <button id="bDelete" class="danger" title="D">delete box (D)</button>
    <button id="bSkip" title="S">skip (S)</button>
  </div>
  <div id="wrap"><canvas id="cv"></canvas><div id="palette"></div></div>
  <div id="info">
    <span class="legend"><span style="background:#ff3b3b"></span>machine box <span style="background:#2ecc71"></span>GT box <span style="background:#4cc9f0"></span>SAM mask <span style="background:#ffd60a"></span>new tight box <span style="background:#ffffff55"></span>other preds · wheel = zoom, ctrl/middle-drag = pan, Z = undo click</span>
    <span id="iinfo"></span><span id="minfo"></span><span id="msg"></span>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
let items = [], view = [], cur = -1, ctx = {preds:[],gts:[]}, img = null, clicks = [], mask = null, last = null, busy = false;
const cv = $('#cv'), g = cv.getContext('2d');
let scale = 1, offx = 0, offy = 0;

async function api(path, body) {
  const r = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)} : {});
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
function msg(t, err) { $('#msg').textContent = t || ''; $('#msg').style.color = err ? '#ff6b6b' : '#ffb347'; }

async function loadQueue(keepSel) {
  const q = await api('/api/queue');
  items = q.items;
  const sel = $('#fScene');
  if (sel.options.length === 1) q.scenes.forEach(s => { const o = document.createElement('option'); o.value = o.textContent = s; sel.appendChild(o); });
  const done = items.filter(i => i.status !== 'pending').length;
  $('#counts').textContent = `${items.length} boxes in IoU [${q.band.iou_min}, ${q.band.iou_max}) · ${done} decided · ${q.counts.predictions} predictions total`;
  const id = keepSel && cur >= 0 ? view[cur].id : null;
  applyFilter(id);
}
function applyFilter(keepId) {
  const sc = $('#fScene').value, st = $('#fStatus').value, ct = $('#fCat').value.toLowerCase();
  view = items.filter(i => (!sc || i.scene === sc) && (!st || i.status === st) && (!ct || i.category.includes(ct)));
  const list = $('#list'); list.innerHTML = '';
  view.forEach((it, k) => {
    const d = document.createElement('div'); d.className = 'it' + (k === cur ? ' sel' : ''); d.dataset.k = k;
    d.innerHTML = `<span class="iou">${it.iou.toFixed(2)}</span><span class="cat">${it.category}<br><small style="color:var(--mut)">${it.scene} · ${it.channel.replace('CAM_','')} #${it.frame_index}</small></span><span class="st ${it.status}">${it.status}</span>`;
    d.onclick = () => select(k);
    list.appendChild(d);
  });
  let k = keepId ? view.findIndex(i => i.id === keepId) : -1;
  if (k < 0) k = view.findIndex(i => i.status === 'pending');
  if (k < 0 && view.length) k = 0;
  if (k >= 0) select(k); else { cur = -1; img = null; draw(); }
}

async function select(k) {
  if (k < 0 || k >= view.length) return;
  cur = k; clicks = []; mask = null; last = null;
  document.querySelectorAll('.it').forEach(e => e.classList.toggle('sel', +e.dataset.k === k));
  const el = document.querySelector(`.it[data-k="${k}"]`); if (el) el.scrollIntoView({block:'nearest'});
  const it = view[k];
  ctx = await api(`/api/context?scene=${encodeURIComponent(it.scene)}&file=${encodeURIComponent(it.file_name)}`);
  const im = new Image(); im.onload = () => { img = im; fit(); if ($('#autoZoom').checked) zoomToBox(); draw(); if ($('#useBox').checked) refine(); }; im.src = '/api/image/' + it.file_name;
  $('#iinfo').innerHTML = `<b>${it.category}</b> score ${(it.score ?? 0).toFixed(2)} · IoU vs GT <b>${it.iou.toFixed(3)}</b>${it.gt_category ? ' (GT: ' + it.gt_category + ')' : ' (no GT overlap)'} · ${it.file_name.split('/').pop()}`;
  $('#minfo').textContent = ''; msg('');
}

let fitScale = 1, zoomed = false;
function fit() {
  const w = $('#wrap').clientWidth, h = $('#wrap').clientHeight;
  cv.width = w; cv.height = h;
  if (!img) return;
  fitScale = Math.min(w / img.width, h / img.height);
  if (zoomed && cur >= 0) zoomToBox(); else { scale = fitScale; offx = (w - img.width * scale) / 2; offy = (h - img.height * scale) / 2; }
}
function zoomToBox() {
  // Frame the machine box (and its GT twin) with margin: small objects are the
  // bulk of the queue and need pixels under the cursor to click on.
  const it = view[cur]; if (!it) return;
  let b = it.bbox_xyxy.slice();
  if (it.gt_bbox_xyxy) b = [Math.min(b[0], it.gt_bbox_xyxy[0]), Math.min(b[1], it.gt_bbox_xyxy[1]), Math.max(b[2], it.gt_bbox_xyxy[2]), Math.max(b[3], it.gt_bbox_xyxy[3])];
  const bw = Math.max(40, b[2] - b[0]), bh = Math.max(40, b[3] - b[1]);
  scale = Math.min(6, cv.width / (bw * 3), cv.height / (bh * 3));
  scale = Math.max(scale, fitScale);
  const cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2;
  offx = cv.width / 2 - cx * scale; offy = cv.height / 2 - cy * scale;
  zoomed = true;
}
function zoomAt(px, py, factor) {
  const [ix, iy] = toImage(px, py);
  scale = Math.min(12, Math.max(fitScale * 0.5, scale * factor));
  offx = px - ix * scale; offy = py - iy * scale; zoomed = true; draw();
}
function toCanvas(x, y) { return [offx + x * scale, offy + y * scale]; }
function toImage(x, y) { return [(x - offx) / scale, (y - offy) / scale]; }
function rect(b, color, lw, dash) {
  const [x1, y1] = toCanvas(b[0], b[1]), [x2, y2] = toCanvas(b[2], b[3]);
  g.save(); g.strokeStyle = color; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.strokeRect(x1, y1, x2 - x1, y2 - y1); g.restore();
}
function draw() {
  g.clearRect(0, 0, cv.width, cv.height);
  if (!img) return;
  g.drawImage(img, offx, offy, img.width * scale, img.height * scale);
  const it = view[cur];
  ctx.preds.forEach(p => { if (p.ann_id !== it.ann_id) rect(p.bbox_xyxy, 'rgba(255,255,255,0.35)', 1); });
  ctx.gts.forEach(gt => rect(gt.bbox_xyxy, 'rgba(46,204,113,0.35)', 1));
  if (mask) { g.save(); g.globalAlpha = 0.45; g.drawImage(mask, offx, offy, img.width * scale, img.height * scale); g.restore(); }
  if (it.gt_bbox_xyxy) rect(it.gt_bbox_xyxy, '#2ecc71', 2);
  rect(it.bbox_xyxy, '#ff3b3b', 2);
  if (last && !last.empty) rect(last.bbox_xyxy, '#ffd60a', 2, [6, 4]);
  clicks.forEach(c => { const [x, y] = toCanvas(c[0], c[1]); g.beginPath(); g.arc(x, y, 6, 0, 7); g.fillStyle = c[2] ? '#4cc9f0' : '#ff3b3b'; g.fill(); g.strokeStyle = '#000'; g.lineWidth = 1.5; g.stroke(); });
}

async function refine() {
  if (cur < 0 || busy) return;
  const it = view[cur];
  if (!clicks.length && !$('#useBox').checked) { mask = null; last = null; draw(); return; }
  busy = true; msg('segmenting…');
  try {
    const r = await api('/api/refine', {id: it.id, clicks, use_box: $('#useBox').checked});
    last = r;
    if (r.empty) { mask = null; $('#minfo').textContent = 'empty mask'; }
    else {
      const m = new Image(); m.onload = () => { mask = tint(m); draw(); }; m.src = 'data:image/png;base64,' + r.mask_png_b64;
      $('#minfo').innerHTML = `mask <b>${r.area}</b> px · new box [${r.bbox_xyxy.join(', ')}] · IoU after <b>${r.iou_after == null ? '–' : r.iou_after.toFixed(3)}</b> (was ${it.iou.toFixed(3)}) · conf ${r.confidence.toFixed(2)}`;
    }
    msg('');
  } catch (e) { msg(e.message, true); }
  busy = false; draw();
}
function tint(m) {
  const c = document.createElement('canvas'); c.width = m.width; c.height = m.height;
  const x = c.getContext('2d'); x.drawImage(m, 0, 0);
  const d = x.getImageData(0, 0, c.width, c.height), p = d.data;
  for (let i = 0; i < p.length; i += 4) { const on = p[i] > 127; p[i] = 76; p[i+1] = 201; p[i+2] = 240; p[i+3] = on ? 255 : 0; }
  x.putImageData(d, 0, 0); return c;
}
async function decide(action) {
  if (cur < 0 || busy) return;
  const it = view[cur];
  if (action === 'fix' && (!last || last.empty)) { msg('refine a mask first (click the object)', true); return; }
  try {
    await api('/api/decide', {id: it.id, action});
    it.status = action; items.find(i => i.id === it.id).status = action;
    const next = view.findIndex((i, k) => k > cur && i.status === 'pending');
    await loadQueue(true);
    if (next >= 0) select(next);
  } catch (e) { msg(e.message, true); }
}
async function clearClicks() { clicks = []; mask = null; last = null; try { await api('/api/reset', {id: view[cur]?.id}); } catch (e) {} draw(); if ($('#useBox').checked) refine(); }

cv.addEventListener('contextmenu', e => e.preventDefault());
cv.addEventListener('wheel', e => { if (!img) return; e.preventDefault(); zoomAt(e.offsetX, e.offsetY, e.deltaY < 0 ? 1.25 : 0.8); }, {passive: false});
let pan = null;
cv.addEventListener('mousedown', e => {
  if (!img || busy) return;
  if (e.button === 1 || e.ctrlKey || e.altKey) { pan = {x: e.offsetX, y: e.offsetY, ox: offx, oy: offy}; e.preventDefault(); return; }
  const [x, y] = toImage(e.offsetX, e.offsetY);
  if (x < 0 || y < 0 || x >= img.width || y >= img.height) return;
  const label = (e.button === 2 || e.shiftKey) ? 0 : 1;
  clicks.push([Math.round(x), Math.round(y), label]); draw(); refine();
});
cv.addEventListener('mousemove', e => { if (pan) { offx = pan.ox + e.offsetX - pan.x; offy = pan.oy + e.offsetY - pan.y; draw(); } });
window.addEventListener('mouseup', () => { pan = null; });
window.addEventListener('resize', () => { fit(); draw(); });
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  const k = e.key.toLowerCase();
  if (k === 'a') decide('fix'); else if (k === 'k') decide('keep'); else if (k === 'd') decide('delete'); else if (k === 's') decide('skip');
  else if (k === 'r') clearClicks();
  else if (k === 'b') { $('#useBox').checked = !$('#useBox').checked; refine(); }
  else if (k === 'arrowleft') select(cur - 1); else if (k === 'arrowright') select(cur + 1);
  else if (k === 'z' && !clicks.length) { /* nothing to undo */ }
  else if (k === 'n') { const n = view.findIndex((i, j) => j > cur && i.status === 'pending'); if (n >= 0) select(n); }
  else if (k === 'z' && clicks.length) { clicks.pop(); draw(); refine(); }
  else if (k === 'f') { if (zoomed) { zoomed = false; fit(); } else zoomToBox(); draw(); }
});
$('#bPrev').onclick = () => select(cur - 1); $('#bNext').onclick = () => select(cur + 1);
$('#bNextPending').onclick = () => { const n = view.findIndex((i, j) => j > cur && i.status === 'pending'); if (n >= 0) select(n); };
$('#bSeed').onclick = refine; $('#bClear').onclick = clearClicks; $('#useBox').onchange = refine;
$('#bFix').onclick = () => decide('fix'); $('#bKeep').onclick = () => decide('keep'); $('#bDelete').onclick = () => decide('delete'); $('#bSkip').onclick = () => decide('skip');
['#fScene', '#fStatus'].forEach(s => $(s).onchange = () => applyFilter()); $('#fCat').oninput = () => applyFilter();
loadQueue().catch(e => msg(e.message, true));
</script></body></html>
"""


# The frame-by-frame checker page (serve mode `frames`): the human gate between
# Stage 4 (2D masks) and Stage 5 (the 3D lift). One image at a time, EVERY kept
# box drawn with its Stage 4 polygon, decision state as colour. Enter attests
# the whole frame; boxes are only touched when something is wrong. GT overlay is
# OFF by default — in production (DhakaScenes) there is no GT, the checker is
# the truth source. Box-seed is OFF by default (measured: it hurts; see
# docs/SAM31_REVIEW.md).
PAGE_FRAMES = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Frame checker — SAM 3.1</title>
<style>
  :root { --bg:#15171b; --panel:#1e2127; --fg:#e6e6e6; --mut:#9aa0a6; --acc:#4cc9f0; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg); font:13px/1.4 system-ui, sans-serif; display:flex; height:100vh; }
  #side { width:290px; background:var(--panel); display:flex; flex-direction:column; border-right:1px solid #2a2e36; }
  #side header { padding:10px 12px; border-bottom:1px solid #2a2e36; }
  #side header h1 { font-size:14px; margin:0 0 6px; }
  #filters select { background:#111; color:var(--fg); border:1px solid #333; padding:3px 5px; margin:2px 0; width:100%; }
  #list { flex:1; overflow:auto; }
  .fr { padding:5px 12px; border-bottom:1px solid #262a31; cursor:pointer; display:flex; gap:8px; align-items:center; }
  .fr:hover { background:#262a31; } .fr.sel { background:#2d3340; }
  .fr .idx { font-variant-numeric:tabular-nums; width:56px; color:var(--mut); }
  .fr .cam { flex:1; }
  .fr .nb { color:var(--mut); }
  .fr .ok { width:18px; text-align:center; color:#2ecc71; }
  #main { flex:1; display:flex; flex-direction:column; min-width:0; }
  #bar { padding:8px 12px; background:var(--panel); border-bottom:1px solid #2a2e36; display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  #bar button { background:#2a2f3a; color:var(--fg); border:1px solid #3a404d; padding:5px 10px; cursor:pointer; border-radius:4px; }
  #bar button:hover { background:#343a48; } #bar button.pri { background:#1f6f43; border-color:#2a8f58; }
  #bar button.danger { background:#7a2d2d; border-color:#9a3d3d; }
  #bar button:disabled { opacity:.4; cursor:default; }
  #bar label { color:var(--mut); display:flex; align-items:center; gap:4px; }
  #wrap { flex:1; position:relative; overflow:hidden; background:#000; }
  canvas { position:absolute; left:0; top:0; }
  #info { padding:6px 12px; background:var(--panel); border-top:1px solid #2a2e36; color:var(--mut); display:flex; gap:18px; flex-wrap:wrap; font-variant-numeric:tabular-nums; }
  #info b { color:var(--fg); }
  #msg { color:#ffb347; }
  #prog { color:var(--mut); margin-top:6px; }
  #palette { position:absolute; display:none; background:rgba(30,33,39,.96); border:1px solid #3a404d;
             border-radius:6px; padding:6px 0; min-width:190px; z-index:5; box-shadow:0 4px 16px rgba(0,0,0,.5); }
  #palette .hd { padding:2px 10px 6px; color:var(--mut); font-size:11px; border-bottom:1px solid #2a2e36; margin-bottom:4px; }
  #palette .hd b { color:#b884f0; }
  #palette .row { padding:3px 10px; cursor:pointer; display:flex; gap:8px; align-items:baseline; }
  #palette .row:hover { background:#2d3340; }
  #palette .row.on { background:#3b5b8a; }
  #palette .row .key { font-weight:700; color:var(--acc); width:14px; }
  #palette .row .ai { margin-left:auto; color:#b884f0; font-size:11px; }
</style></head><body>
<div id="side">
  <header>
    <h1>Frame checker</h1>
    <div id="filters">
      <select id="fScene"></select>
      <select id="fChan"><option value="">all cameras</option></select>
    </div>
    <div id="prog"></div>
  </header>
  <div id="list"></div>
</div>
<div id="main">
  <div id="bar">
    <button id="bPrev" title="D / ←">◀ frame (D)</button>
    <button id="bNext" title="F / →">frame (F) ▶</button>
    <button id="bOk" class="pri" title="Enter">frame OK ✓ + next (Enter)</button>
    <label><input type="checkbox" id="showMasks" checked> masks (M)</label>
    <label><input type="checkbox" id="showGT"> GT (T)</label>
    <label><input type="checkbox" id="useBox"> seed from box (B)</label>
    <label title="drag on empty canvas to draw a NEW object; this is its class">new: <select id="newCat"></select></label>
    <span style="flex:1"></span>
    <span id="selbar" style="display:flex;gap:8px">
      <button id="bFix" class="pri" title="A">accept fix (A)</button>
      <button id="bKeep" title="K">keep (K)</button>
      <button id="bTrack" title="G — propagate the selected object across the clip (P also works)">track clip (G)</button>
      <button id="bDelete" class="danger" title="X">delete (X)</button>
      <button id="bSkip" title="S">skip (S)</button>
      <button id="bClear" title="R">clear clicks (R)</button>
      <button id="bEsc" title="Esc">deselect (Esc)</button>
    </span>
  </div>
  <div id="wrap"><canvas id="cv"></canvas><div id="palette"></div></div>
  <div id="info">
    <span>click a box to select · then left click = object, right = background · <b>Enter</b> = frame is fine · Tab cycles boxes · wheel zoom, ctrl-drag pan</span>
    <span id="iinfo"></span><span id="minfo"></span><span id="msg"></span>
  </div>
</div>
<script>
const $ = s => document.querySelector(s);
let frames = [], cur = -1, items = [], byFile = {}, framesOk = new Set();
let ctx = {preds:[],gts:[]}, img = null, sel = null, clicks = [], mask = null, last = null, busy = false;
let adds = [], adding = false, draftBox = null, tempId = null, drag = null, selAdd = null;
let showGen = 0, ctxKey = null;
let cats = [], userPicked = false, lastSuggest = null;
const ctxCache = {};
const cv = $('#cv'), g = cv.getContext('2d');
let scale = 1, offx = 0, offy = 0, fitScale = 1, zoomed = false, pan = null;

const STATUS_COLOR = {pending:'rgba(255,255,255,0.55)', keep:'#5b8dd6', fix:'#ffd60a', skip:'#b8a24a', 'delete':'#803030'};

async function api(path, body) {
  const r = await fetch(path, body ? {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)} : {});
  const j = await r.json();
  if (!r.ok || j.error) throw new Error(j.error || r.statusText);
  return j;
}
function msg(t, err) { $('#msg').textContent = t || ''; $('#msg').style.color = err ? '#ff6b6b' : '#ffb347'; }

async function load() {
  const q = await api('/api/queue');
  items = q.items;
  byFile = {};
  items.forEach(it => { (byFile[it.scene + '|' + it.file_name] ??= []).push(it); });
  framesOk = new Set(q.frames_ok || []);
  frames = [];
  Object.keys(q.chains).sort().forEach(scene => {
    Object.keys(q.chains[scene]).sort().forEach(channel => {
      q.chains[scene][channel].forEach((file_name, idx) => frames.push({scene, channel, file_name, idx}));
    });
  });
  const sSel = $('#fScene');
  if (!sSel.options.length) Object.keys(q.chains).sort().forEach(s => { const o = document.createElement('option'); o.value = o.textContent = s; sSel.appendChild(o); });
  const cSel = $('#fChan');
  if (cSel.options.length === 1) [...new Set(frames.map(f => f.channel))].sort().forEach(c => { const o = document.createElement('option'); o.value = c; o.textContent = c.replace('CAM_',''); cSel.appendChild(o); });
  adds = q.adds || [];
  const nc = $('#newCat');
  cats = (q.categories && q.categories.length) ? q.categories : [...new Set(items.map(i => i.category))].sort();
  if (!nc.options.length) cats.forEach(c => { const o = document.createElement('option'); o.value = o.textContent = c; nc.appendChild(o); });
  renderList();
  if (cur < 0) { const k = frames.findIndex(f => !framesOk.has(f.scene + '|' + f.file_name)); show(k >= 0 ? k : 0); }
  progress();
}
function progress() {
  const tally = {fix:0, keep:0, 'delete':0, skip:0};
  items.forEach(i => { if (tally[i.status] != null) tally[i.status]++; });
  $('#prog').innerHTML = `<b>${framesOk.size}</b>/${frames.length} frames OK · boxes: ${tally.fix} fixed, ${tally.keep} kept, ${tally['delete']} deleted, ${tally.skip} skipped, ${adds.length} added`;
}
function renderList() {
  const sc = $('#fScene').value, ch = $('#fChan').value;
  const list = $('#list'); list.innerHTML = '';
  frames.forEach((f, k) => {
    if (f.scene !== sc || (ch && f.channel !== ch)) return;
    const key = f.scene + '|' + f.file_name;
    const n = (byFile[key] || []).length;
    const d = document.createElement('div');
    d.className = 'fr' + (k === cur ? ' sel' : ''); d.dataset.k = k;
    d.innerHTML = `<span class="idx">#${String(f.idx).padStart(2,'0')}</span><span class="cam">${f.channel.replace('CAM_','')}</span><span class="nb">${n} box${n===1?'':'es'}</span><span class="ok">${framesOk.has(key) ? '✓' : ''}</span>`;
    d.onclick = () => show(k);
    list.appendChild(d);
  });
}

async function show(k) {
  if (k < 0 || k >= frames.length) return;
  if (adding) { msg('finish the new object first (A saves, Esc cancels)', true); return; }
  const gen = ++showGen;
  cur = k; sel = null; clicks = []; mask = null; last = null; zoomed = false;
  selAdd = null; drag = null; ctxKey = null;
  const f = frames[k];
  $('#fScene').value = f.scene;
  renderList();
  const el = document.querySelector(`.fr[data-k="${k}"]`); if (el) el.scrollIntoView({block:'nearest'});
  const key = f.scene + '|' + f.file_name;
  try {
    ctx = ctxCache[key] ??= await api(`/api/context?scene=${encodeURIComponent(f.scene)}&file=${encodeURIComponent(f.file_name)}`);
    if (gen !== showGen) return;                       // user already moved on
    ctxKey = key;                                       // overlays are trustworthy now
  } catch (e) { if (gen !== showGen) return; ctx = {preds:[],gts:[]}; msg('context failed: ' + e.message + ' — overlays missing, do NOT attest', true); }
  const im = new Image();
  im.onload = () => { if (gen !== showGen) return; img = im; fit(); draw(); };
  im.src = '/api/image/' + f.file_name;
  const nxt = frames[k+1]; if (nxt) { const p = new Image(); p.src = '/api/image/' + nxt.file_name; }
  info();
}
function info() {
  const f = frames[cur]; if (!f) return;
  const boxes = byFile[f.scene + '|' + f.file_name] || [];
  const okMark = framesOk.has(f.scene + '|' + f.file_name) ? ' · <b style="color:#2ecc71">frame OK ✓</b>' : '';
  const nAdds = adds.filter(a => a.scene === f.scene && a.file_name === f.file_name).length;
  $('#iinfo').innerHTML = `<b>${f.scene}</b> ${f.channel.replace('CAM_','')} #${f.idx} · frame ${cur+1}/${frames.length} · ${boxes.length} boxes${nAdds ? ' +' + nAdds + ' added' : ''}${okMark}` +
    (adding ? ` · <b style="color:#b884f0">NEW ${$('#newCat').value}</b> — click to refine, A/Enter save, Esc cancel` :
     selAdd ? (selAdd.review_kind === 'tracked' ? ` · tracked <b>${selAdd.category}</b> (conf ${(selAdd.mask_confidence ?? 0).toFixed(2)}, ${selAdd.hops} keyframes from seed${driftRatio(selAdd) ? `, <b style="color:#ff9f1c">${driftRatio(selAdd).toFixed(1)}× the seed's size — likely drift</b>` : ''}) — click the object to re-segment, A saves the repair · X removes it, Shift+X the whole track` : ` · added <b>${selAdd.category}</b> — P tracks it across the clip, X removes it`) :
     sel ? ` · selected: <b>${sel.category}</b> score ${(sel.score ?? 0).toFixed(2)} [${sel.status}]` : '');
  $('#minfo').textContent = '';
}

function fit() {
  const w = $('#wrap').clientWidth, h = $('#wrap').clientHeight;
  cv.width = w; cv.height = h;
  if (!img) return;
  fitScale = Math.min(w / img.width, h / img.height);
  if (!zoomed) { scale = fitScale; offx = (w - img.width * scale) / 2; offy = (h - img.height * scale) / 2; }
}
function zoomToSel() {
  if (!sel) return;
  const b = sel.bbox_xyxy;
  const bw = Math.max(40, b[2] - b[0]), bh = Math.max(40, b[3] - b[1]);
  scale = Math.max(fitScale, Math.min(6, cv.width / (bw * 3), cv.height / (bh * 3)));
  offx = cv.width / 2 - (b[0] + b[2]) / 2 * scale; offy = cv.height / 2 - (b[1] + b[3]) / 2 * scale;
  zoomed = true;
}
function zoomAt(px, py, factor) {
  const [ix, iy] = toImage(px, py);
  scale = Math.min(12, Math.max(fitScale * 0.5, scale * factor));
  offx = px - ix * scale; offy = py - iy * scale; zoomed = true; draw();
}
function toCanvas(x, y) { return [offx + x * scale, offy + y * scale]; }
function toImage(x, y) { return [(x - offx) / scale, (y - offy) / scale]; }
function rect(b, color, lw, dash) {
  const [x1, y1] = toCanvas(b[0], b[1]), [x2, y2] = toCanvas(b[2], b[3]);
  g.save(); g.strokeStyle = color; g.lineWidth = lw; if (dash) g.setLineDash(dash); g.strokeRect(x1, y1, x2 - x1, y2 - y1); g.restore();
}
function poly(seg, stroke, fill) {
  g.save();
  seg.forEach(ring => {
    g.beginPath();
    for (let i = 0; i + 1 < ring.length; i += 2) {
      const [x, y] = toCanvas(ring[i], ring[i+1]);
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    }
    g.closePath();
    if (fill) { g.fillStyle = fill; g.fill(); }
    if (stroke) { g.strokeStyle = stroke; g.lineWidth = 1; g.stroke(); }
  });
  g.restore();
}
function statusOf(p) {
  const it = (byFile[frames[cur].scene + '|' + frames[cur].file_name] || []).find(i => i.ann_id === p.ann_id);
  return it ? it.status : 'pending';
}
function draw() {
  g.clearRect(0, 0, cv.width, cv.height);
  if (!img) return;
  g.drawImage(img, offx, offy, img.width * scale, img.height * scale);
  if ($('#showGT').checked) ctx.gts.forEach(gt => rect(gt.bbox_xyxy, 'rgba(46,204,113,0.8)', 1));
  ctx.preds.forEach(p => {
    const st = statusOf(p);
    const isSel = sel && p.ann_id === sel.ann_id;
    if (st === 'delete' && !isSel) {           // deleted: faded, crossed out
      rect(p.bbox_xyxy, STATUS_COLOR['delete'], 1);
      const [x1,y1] = toCanvas(p.bbox_xyxy[0], p.bbox_xyxy[1]), [x2,y2] = toCanvas(p.bbox_xyxy[2], p.bbox_xyxy[3]);
      g.save(); g.strokeStyle = STATUS_COLOR['delete']; g.lineWidth = 1;
      g.beginPath(); g.moveTo(x1,y1); g.lineTo(x2,y2); g.moveTo(x2,y1); g.lineTo(x1,y2); g.stroke(); g.restore();
      return;
    }
    const col = isSel ? '#ff3b3b' : STATUS_COLOR[st] || STATUS_COLOR.pending;
    if ($('#showMasks').checked && p.segmentation) {
      poly(p.segmentation, col, isSel ? 'rgba(255,59,59,0.10)' : 'rgba(255,255,255,0.06)');
    }
    rect(p.bbox_xyxy, col, isSel ? 2.5 : 1.2);
  });
  const fr = frames[cur];
  adds.forEach(a => {
    if (!fr || a.scene !== fr.scene || a.file_name !== fr.file_name) return;
    const isSel = selAdd && selAdd.id === a.id;
    const col = isSel ? '#ff3b3b' : (driftRatio(a) != null ? '#ff9f1c' : '#b884f0');
    if ($('#showMasks').checked && a.segmentation) poly(a.segmentation, col, isSel ? 'rgba(255,59,59,0.10)' : 'rgba(184,132,240,0.10)');
    rect(a.bbox_xyxy, col, isSel ? 2.5 : 1.6, a.review_kind === 'tracked' ? [6, 4] : null);
  });
  if (mask) { g.save(); g.globalAlpha = 0.45; g.drawImage(mask, offx, offy, img.width * scale, img.height * scale); g.restore(); }
  if (draftBox) { rect(draftBox, '#ffffff', 1.5, [8, 5]); positionPalette(); }
  if (drag && drag.moved) rect([Math.min(drag.x0, drag.x1), Math.min(drag.y0, drag.y1), Math.max(drag.x0, drag.x1), Math.max(drag.y0, drag.y1)], '#ffffff', 1, [4, 4]);
  if (last && !last.empty) rect(last.bbox_xyxy, '#ffd60a', 2, [6, 4]);
  clicks.forEach(c => { const [x, y] = toCanvas(c[0], c[1]); g.beginPath(); g.arc(x, y, 6, 0, 7); g.fillStyle = c[2] ? '#4cc9f0' : '#ff3b3b'; g.fill(); g.strokeStyle = '#000'; g.lineWidth = 1.5; g.stroke(); });
}

function hit(x, y) {
  const f = frames[cur];
  const cand = (byFile[f.scene + '|' + f.file_name] || []).filter(it => {
    const b = it.bbox_xyxy; return x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3];
  });
  cand.sort((a, b) => (a.bbox_xyxy[2]-a.bbox_xyxy[0])*(a.bbox_xyxy[3]-a.bbox_xyxy[1]) - (b.bbox_xyxy[2]-b.bbox_xyxy[0])*(b.bbox_xyxy[3]-b.bbox_xyxy[1]));
  return cand[0] || null;
}
async function pick(it) {
  sel = it; selAdd = null; clicks = []; mask = null; last = null;
  try { await api('/api/reset', {id: it.id}); } catch (e) {}
  zoomToSel(); info(); draw();
}
function deselect() { sel = null; selAdd = null; clicks = []; mask = null; last = null; zoomed = false; fit(); info(); draw(); }
function cycle(dir) {
  const f = frames[cur];
  const boxes = byFile[f.scene + '|' + f.file_name] || [];
  if (!boxes.length) return;
  const k = sel ? (boxes.findIndex(i => i.ann_id === sel.ann_id) + dir + boxes.length) % boxes.length : 0;
  pick(boxes[k]);
}

function hitAdd(x, y) {
  const f = frames[cur];
  const cand = adds.filter(a => a.scene === f.scene && a.file_name === f.file_name &&
    x >= a.bbox_xyxy[0] && x <= a.bbox_xyxy[2] && y >= a.bbox_xyxy[1] && y <= a.bbox_xyxy[3]);
  cand.sort((a, b) => (a.bbox_xyxy[2]-a.bbox_xyxy[0])*(a.bbox_xyxy[3]-a.bbox_xyxy[1]) - (b.bbox_xyxy[2]-b.bbox_xyxy[0])*(b.bbox_xyxy[3]-b.bbox_xyxy[1]));
  return cand[0] || null;
}
function paletteKeys() { return cats.map((c, i) => [i < 9 ? String(i + 1) : (i === 9 ? '0' : null), c]); }
function renderPalette() {
  const pal = $('#palette');
  if (!adding || !cats.length) { pal.style.display = 'none'; return; }
  const cur = $('#newCat').value;
  const aiCat = lastSuggest ? lastSuggest.category : null;
  const aiP = lastSuggest && lastSuggest.top ? lastSuggest.top[0][1] : null;
  pal.innerHTML = `<div class="hd">${lastSuggest ? `AI: <b>${aiCat}</b> ${(aiP * 100).toFixed(0)}%` : 'class — press a number'}</div>` +
    paletteKeys().map(([k, c]) => `<div class="row${c === cur ? ' on' : ''}" data-c="${c}">` +
      `<span class="key">${k ?? ''}</span><span>${c}</span>${c === aiCat ? '<span class="ai">AI</span>' : ''}</div>`).join('');
  pal.querySelectorAll('.row').forEach(r => r.onclick = () => { setCat(r.dataset.c, true); });
  pal.style.display = 'block';
  positionPalette();
}
function positionPalette() {
  const pal = $('#palette');
  if (!adding || !draftBox || pal.style.display === 'none') return;
  const [bx, by] = toCanvas(draftBox[2], draftBox[1]);
  pal.style.left = Math.max(4, Math.min(cv.width - pal.offsetWidth - 8, bx + 12)) + 'px';
  pal.style.top = Math.max(4, Math.min(cv.height - pal.offsetHeight - 8, by)) + 'px';
}
function setCat(c, byHuman) {
  if (!cats.includes(c)) return;
  $('#newCat').value = c;
  if (byHuman) userPicked = true;
  renderPalette(); info();
}
function driftRatio(a) {
  // suspiciously sized tracked copy: mask far larger/smaller than its seed's
  if (a.review_kind !== 'tracked' || !a.mask_area_px) return null;
  const sa = a.seed_area_px || (adds.find(x => x.id === a.track_group) || {}).mask_area_px;
  if (!sa) return null;
  const r = a.mask_area_px / sa;
  return (r > 3 || r < 0.1) ? r : null;
}
function pickAdd(a) { sel = null; clicks = []; mask = null; last = null; selAdd = a; info(); draw(); }
async function startAdd(box) {
  sel = null; selAdd = null; adding = true; draftBox = box; tempId = 'tmp-' + Date.now();
  clicks = []; mask = null; last = null; userPicked = false; lastSuggest = null;
  try { await api('/api/reset', {id: tempId}); } catch (e) {}
  info(); draw(); renderPalette(); refineNew();
}
async function refineNew() {
  if (!adding || busy) return;
  busy = true; msg('segmenting…');
  const f = frames[cur], myTemp = tempId;
  try {
    const r = await api('/api/refine', {new: true, temp_id: myTemp, scene: f.scene, file_name: f.file_name, clicks, box: draftBox, suggest: true});
    if (!adding || tempId !== myTemp) { busy = false; return; }   // cancelled meanwhile
    last = r;
    if (r.suggest) { lastSuggest = r.suggest; if (!userPicked) setCat(r.suggest.category, false); else renderPalette(); }
    if (r.empty) { mask = null; $('#minfo').textContent = 'empty mask — click the object'; }
    else {
      const m = new Image(); m.onload = () => { mask = tint(m); draw(); }; m.src = 'data:image/png;base64,' + r.mask_png_b64;
      $('#minfo').innerHTML = `mask <b>${r.area}</b> px · conf ${r.confidence.toFixed(2)}` + (r.iou_after != null ? ` · best GT IoU ${r.iou_after.toFixed(3)}` : '');
    }
    msg('');
  } catch (e) { msg(e.message, true); }
  busy = false; draw();
}
async function saveAdd() {
  if (!adding || busy) return;
  if (!last || last.empty) { msg('nothing to save yet — the mask is empty', true); return; }
  busy = true;
  const f = frames[cur];
  try {
    const rec = await api('/api/add', {scene: f.scene, file_name: f.file_name, category: $('#newCat').value, temp_id: tempId,
      category_source: userPicked ? 'human_picked' : (lastSuggest ? 'ai_suggested' : 'human_default'),
      category_suggest: lastSuggest ? lastSuggest.top : null});
    adds.push(rec);
    adding = false; draftBox = null; tempId = null; clicks = []; mask = null; last = null;
    selAdd = rec;                                  // stay selected: P tracks it across the clip
    $('#palette').style.display = 'none';
    msg('saved — press P to track it across the clip'); info(); progress(); draw();
  } catch (e) { msg(e.message, true); }
  busy = false;
}
async function trackSel() {
  if (busy) { msg('still segmenting — wait a moment, then press P again', true); return; }
  let id = null, keepAdd = false;
  if (selAdd) {
    if (selAdd.review_kind === 'tracked') { msg('this is a propagated copy — select the original (solid outline) to re-track', true); return; }
    id = selAdd.id; keepAdd = true;
  } else if (sel) {
    if (sel.status !== 'fix') { msg('fix this box first (click the object, then A) — P tracks the FIXED geometry', true); return; }
    id = sel.id;
  } else return;
  busy = true; msg('propagating across the clip… (~5–20 s)'); draw();
  try {
    const myCur = cur;
    const r = await api('/api/track', {id});
    const q = await api('/api/queue'); adds = q.adds || []; framesOk = new Set(q.frames_ok || []);
    if (keepAdd) selAdd = (cur === myCur) ? (adds.find(a => a.id === r.group) || null) : null;
    msg(`tracked into ${r.made.length} frames (conf ≥ ${r.min_conf}); ${r.skipped.length} low-confidence skipped` + (r.replaced ? `; replaced ${r.replaced} previous` : '') + (r.frames_reopened ? `; ${r.frames_reopened} attested frame(s) reopened for review` : ''));
    renderList();
    info(); progress(); draw();
  } catch (e) { msg(e.message, true); }
  busy = false;
}
async function removeGroup() {
  if (!selAdd || busy) return;
  const group = selAdd.track_group || selAdd.id;
  try {
    const r = await api('/api/add', {remove_group: group});
    adds = adds.filter(a => a.id !== group && a.track_group !== group);
    selAdd = null; msg(`removed ${r.removed} object(s) of the track`); info(); progress(); draw();
  } catch (e) { msg(e.message, true); }
}
async function refineAdd() {
  if (!selAdd || busy || !clicks.length) return;
  busy = true; msg('segmenting…');
  const f = frames[cur], myId = selAdd.id;
  try {
    const r = await api('/api/refine', {new: true, temp_id: 'amend-' + myId, scene: f.scene, file_name: f.file_name, clicks, box: null});
    if (!selAdd || selAdd.id !== myId) { busy = false; return; }
    last = r;
    if (r.empty) { mask = null; $('#minfo').textContent = 'empty mask — keep clicking'; }
    else {
      const m = new Image(); m.onload = () => { mask = tint(m); draw(); }; m.src = 'data:image/png;base64,' + r.mask_png_b64;
      $('#minfo').innerHTML = `repair mask <b>${r.area}</b> px — A saves it`;
    }
    msg('');
  } catch (e) { msg(e.message, true); }
  busy = false; draw();
}
async function amendAccept() {
  if (!selAdd || busy) return;
  if (!last || last.empty) { msg('click the object first — then A saves the repair', true); return; }
  busy = true;
  try {
    const rec = await api('/api/add', {amend_id: selAdd.id, temp_id: 'amend-' + selAdd.id});
    const i = adds.findIndex(a => a.id === rec.id);
    if (i >= 0) { adds[i] = {...adds[i], bbox_xyxy: rec.bbox_xyxy, segmentation: rec.segmentation, mask_area_px: rec.mask_area_px}; selAdd = adds[i]; }
    clicks = []; mask = null; last = null;
    msg('repaired'); info(); progress(); draw();
  } catch (e) { msg(e.message, true); }
  busy = false;
}
function cancelAdd() { $('#palette').style.display = 'none'; const t = tempId; adding = false; draftBox = null; tempId = null; clicks = []; mask = null; last = null; if (t) api('/api/reset', {id: t}).catch(() => {}); info(); draw(); }
async function removeAdd() {
  if (!selAdd || busy) return;
  try {
    await api('/api/add', {remove_id: selAdd.id});
    adds = adds.filter(a => a.id !== selAdd.id);
    selAdd = null; info(); progress(); draw();
  } catch (e) { msg(e.message, true); }
}

async function refine() {
  if (!sel || busy) return;
  if (!clicks.length && !$('#useBox').checked) { mask = null; last = null; draw(); return; }
  busy = true; msg('segmenting…');
  const myId = sel.id, myCur = cur;
  try {
    const r = await api('/api/refine', {id: myId, clicks, use_box: $('#useBox').checked});
    if (!sel || sel.id !== myId || cur !== myCur) { busy = false; return; }   // moved on meanwhile
    last = r;
    if (r.empty) { mask = null; $('#minfo').textContent = 'empty mask'; }
    else {
      const m = new Image(); m.onload = () => { mask = tint(m); draw(); }; m.src = 'data:image/png;base64,' + r.mask_png_b64;
      $('#minfo').innerHTML = `mask <b>${r.area}</b> px · IoU after <b>${r.iou_after == null ? '–' : r.iou_after.toFixed(3)}</b> · conf ${r.confidence.toFixed(2)}`;
    }
    msg('');
  } catch (e) { msg(e.message, true); }
  busy = false; draw();
}
function tint(m) {
  const c = document.createElement('canvas'); c.width = m.width; c.height = m.height;
  const x = c.getContext('2d'); x.drawImage(m, 0, 0);
  const d = x.getImageData(0, 0, c.width, c.height), p = d.data;
  for (let i = 0; i < p.length; i += 4) { const on = p[i] > 127; p[i] = 76; p[i+1] = 201; p[i+2] = 240; p[i+3] = on ? 255 : 0; }
  x.putImageData(d, 0, 0); return c;
}
async function decide(action) {
  if (!sel || busy) return;
  if (action === 'fix' && (!last || last.empty)) { msg('refine a mask first (click the object)', true); return; }
  try {
    await api('/api/decide', {id: sel.id, action, ...(action === 'fix' && last && !last.empty ? {expect_bbox: last.bbox_xyxy} : {})});
    const it = items.find(i => i.id === sel.id); if (it) it.status = action;
    if (action === 'fix') {
      // show the ACCEPTED geometry and stay selected so P can track it
      const f = frames[cur], key = f.scene + '|' + f.file_name;
      if (it && last && !last.empty) it.bbox_xyxy = last.bbox_xyxy;
      sel.status = action;
      delete ctxCache[key];
      try { ctx = ctxCache[key] = await api(`/api/context?scene=${encodeURIComponent(f.scene)}&file=${encodeURIComponent(f.file_name)}`); } catch (e) {}
      clicks = []; mask = null; last = null;
      msg('fixed — press P to track the repaired box across the clip');
      info(); progress(); draw();
    } else { deselect(); progress(); }
  } catch (e) { msg(e.message, true); }
}
async function frameOk(ok) {
  const f = frames[cur]; if (!f) return;
  if (adding || busy) { msg('finish the new object first', true); return; }
  if (ctxKey !== f.scene + '|' + f.file_name) { msg('frame still loading — try again in a moment', true); return; }
  try {
    await api('/api/frame_ok', {scene: f.scene, channel: f.channel, file_name: f.file_name, n_preds: ctx.preds.length, ok});
    const key = f.scene + '|' + f.file_name;
    if (ok) framesOk.add(key); else framesOk.delete(key);
    renderList(); progress();
    if (ok) show(cur + 1); else info();
  } catch (e) { msg(e.message, true); }
}
async function clearClicks() { clicks = []; mask = null; last = null; try { await api('/api/reset', {id: sel?.id}); } catch (e) {} draw(); }

cv.addEventListener('contextmenu', e => e.preventDefault());
cv.addEventListener('wheel', e => { if (!img) return; e.preventDefault(); zoomAt(e.offsetX, e.offsetY, e.deltaY < 0 ? 1.25 : 0.8); }, {passive: false});
cv.addEventListener('mousedown', e => {
  if (!img || busy) return;
  if (e.button === 1 || e.ctrlKey || e.altKey) { pan = {x: e.offsetX, y: e.offsetY, ox: offx, oy: offy}; e.preventDefault(); return; }
  const [x, y] = toImage(e.offsetX, e.offsetY);
  if (x < 0 || y < 0 || x >= img.width || y >= img.height) return;
  if (e.button === 2 || e.shiftKey) {          // negative prompt, immediate
    if (sel || adding || selAdd) { clicks.push([Math.round(x), Math.round(y), 0]); draw(); adding ? refineNew() : (selAdd ? refineAdd() : refine()); }
    return;
  }
  drag = {x0: x, y0: y, x1: x, y1: y, moved: false};   // click OR rubber-band — decided on mouseup
});
cv.addEventListener('mousemove', e => {
  if (pan) { offx = pan.ox + e.offsetX - pan.x; offy = pan.oy + e.offsetY - pan.y; draw(); return; }
  if (drag) {
    [drag.x1, drag.y1] = toImage(e.offsetX, e.offsetY);
    if (Math.abs(drag.x1 - drag.x0) * scale > 6 || Math.abs(drag.y1 - drag.y0) * scale > 6) drag.moved = true;
    if (drag.moved) draw();
  }
});
window.addEventListener('mouseup', e => {
  pan = null;
  if (!drag || !img || busy) { drag = null; return; }
  const d = drag; drag = null;
  const x = Math.round(d.x1), y = Math.round(d.y1);
  if (d.moved) {                                // a drag DRAWS — never prompts, never selects
    if (!sel && !adding) {
      const b = [Math.max(0, Math.min(d.x0, d.x1)), Math.max(0, Math.min(d.y0, d.y1)),
                 Math.min(img.width, Math.max(d.x0, d.x1)), Math.min(img.height, Math.max(d.y0, d.y1))];
      if (b[2] - b[0] > 8 && b[3] - b[1] > 8) startAdd(b);
    }
    draw(); return;
  }
  if (x < 0 || y < 0 || x >= img.width || y >= img.height) return;
  if (adding) { clicks.push([x, y, 1]); draw(); refineNew(); return; }
  if (sel) {
    // A fresh click far outside the selected box, before any prompting, reads
    // as "select that one instead" — after prompting starts, clicks are prompts.
    if (!clicks.length) {
      const b = sel.bbox_xyxy, m = 24;
      if (x < b[0]-m || x > b[2]+m || y < b[1]-m || y > b[3]+m) { const it = hit(x, y); if (it) { pick(it); return; } }
    }
    clicks.push([x, y, 1]); draw(); refine(); return;
  }
  if (selAdd) {
    if (!clicks.length) {                        // first click far outside = select something else
      const b = selAdd.bbox_xyxy, m2 = 24;
      if (x < b[0]-m2 || x > b[2]+m2 || y < b[1]-m2 || y > b[3]+m2) {
        const ad2 = hitAdd(x, y); if (ad2) { pickAdd(ad2); return; }
        const it2 = hit(x, y); if (it2) { pick(it2); return; }
      }
    }
    clicks.push([x, y, 1]); draw(); refineAdd(); return;
  }
  const ad = hitAdd(x, y); if (ad) { pickAdd(ad); return; }
  const it = hit(x, y); if (it) pick(it);
});
window.addEventListener('resize', () => { fit(); draw(); });
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.target.tagName === 'BUTTON' && (e.key === 'Enter' || e.key === ' ')) return;  // the button handles it
  const k = e.key.toLowerCase();
  if (adding) {                                  // composing a new object
    if (k === 'a' || e.key === 'Enter') saveAdd();
    else if (e.key === 'Escape') { if (!busy) cancelAdd(); }
    else if (busy) { /* wait for the running segment */ }
    else if (k === 'z' && clicks.length) { clicks.pop(); draw(); refineNew(); }
    else if (k === 'r') { clicks = []; mask = null; last = null; draw(); refineNew(); }
    else if (k === 'g') refineNew();
    else if (/^[0-9]$/.test(k)) { const i = k === '0' ? 9 : +k - 1; if (cats[i]) setCat(cats[i], true); }
    return;                                      // frame nav locked until save/cancel
  }
  if (selAdd) {
    if (k === 'p' || k === 'g') { trackSel(); return; }
    if (k === 'a') { amendAccept(); return; }
    if (k === 'r' && !busy) { clicks = []; mask = null; last = null; draw(); return; }
    if (k === 'z' && !busy && clicks.length) { clicks.pop(); draw(); refineAdd(); return; }
    if (k === 'x') {
      // a copy: X removes just it; a seed that HAS a track: X means the whole
      // object, i.e. the whole track. Shift+X always takes the whole group.
      const wholeGroup = e.shiftKey || (selAdd.review_kind !== 'tracked' && selAdd.track_group);
      wholeGroup ? removeGroup() : removeAdd(); return;
    }
    if (e.key === 'Escape') { selAdd = null; info(); draw(); return; }
  }
  if (busy && 'zrgb'.includes(k)) return;        // no prompt edits mid-segment
  if (k === 'p' || k === 'g') {
    if (sel) { trackSel(); return; }             // fixed boxes track; others get told why
    const f = frames[cur];
    const here = f ? adds.filter(a => a.scene === f.scene && a.file_name === f.file_name) : [];
    msg(here.length ? 'click the purple object first (it turns red), then P tracks it'
        : 'select a purple object or a FIXED box — or drag a box to add one', true);
    return;
  }
  if (e.key === 'Enter') { frameOk(!e.shiftKey); }
  else if (k === 'f' || e.key === 'ArrowRight') show(cur + 1);
  else if (k === 'd' || e.key === 'ArrowLeft') show(cur - 1);
  else if (e.key === 'Escape') deselect();
  else if (e.key === 'Tab') { e.preventDefault(); cycle(e.shiftKey ? -1 : 1); }
  else if (k === 'a') decide('fix'); else if (k === 'k') decide('keep');
  else if (k === 'x') decide('delete'); else if (k === 's') decide('skip');
  else if (k === 'r') clearClicks();
  else if (k === 'b') { $('#useBox').checked = !$('#useBox').checked; refine(); }
  else if (k === 'z' && clicks.length) { clicks.pop(); draw(); refine(); }
  else if (k === 't') { $('#showGT').checked = !$('#showGT').checked; draw(); }
  else if (k === 'm') { $('#showMasks').checked = !$('#showMasks').checked; draw(); }
  else if (k === 'v') { if (zoomed) { zoomed = false; fit(); } else zoomToSel(); draw(); }
});
$('#bPrev').onclick = () => show(cur - 1); $('#bNext').onclick = () => show(cur + 1);
$('#bOk').onclick = () => frameOk(true);
$('#bFix').onclick = () => adding ? saveAdd() : decide('fix'); $('#bKeep').onclick = () => decide('keep');
$('#bTrack').onclick = trackSel;
$('#bDelete').onclick = () => selAdd ? ((selAdd.review_kind !== 'tracked' && selAdd.track_group) ? removeGroup() : removeAdd()) : decide('delete'); $('#bSkip').onclick = () => decide('skip');
$('#bClear').onclick = clearClicks; $('#bEsc').onclick = () => adding ? cancelAdd() : (selAdd ? (selAdd = null, info(), draw()) : deselect());
$('#showMasks').onchange = draw; $('#showGT').onchange = draw; $('#useBox').onchange = refine;
$('#fScene').onchange = () => { const k = frames.findIndex(f => f.scene === $('#fScene').value && (!$('#fChan').value || f.channel === $('#fChan').value)); renderList(); if (k >= 0) show(k); };
$('#fChan').onchange = () => $('#fScene').onchange();
load().catch(e => msg(e.message, true));
</script></body></html>
"""


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    sub = parser.add_subparsers(dest="cmd", required=True)

    def band_args(p):
        p.add_argument("--scenes", nargs="*", default=None)
        p.add_argument("--iou-min", type=float, default=0.3, help="lower IoU bound, inclusive (0 = include unmatched boxes)")
        p.add_argument("--iou-max", type=float, default=0.5, help="upper IoU bound, exclusive")

    pq = sub.add_parser("queue", help="list the low-IoU boxes and write queue.json")
    band_args(pq)
    ps = sub.add_parser("serve", help="run the click-fix page")
    band_args(ps)
    ps.add_argument("--host", default="127.0.0.1")
    ps.add_argument("--port", type=int, default=8765)
    pf = sub.add_parser("frames", help="frame-by-frame checker page: EVERY kept box, "
                                      "flip through images, fix only what needs it "
                                      "(the human gate between Stage 4 and Stage 5)")
    pf.add_argument("--scenes", nargs="*", default=None)
    pf.add_argument("--host", default="127.0.0.1")
    pf.add_argument("--port", type=int, default=8765)
    pe = sub.add_parser("export", help="fold fixes.jsonl into cvat_export_fixed/")
    pe.add_argument("--scenes", nargs="*", default=None)
    args = parser.parse_args(argv)

    paths = load_paths(args.paths)
    review_root = os.path.join(paths.work_root, REVIEW_DIR)
    os.makedirs(review_root, exist_ok=True)
    ledger = Ledger(review_root)

    if args.cmd == "export":
        summary = export_fixed(paths, args.scenes, ledger)
        t = summary["totals"]
        print(f"\n{t['fixed']} fixed, {t['kept']} kept, {t['deleted']} deleted, {t['skipped']} skipped, "
              f"{t['added']} added -> {os.path.join(paths.work_root, FIXED_EXPORT_DIR)}")
        print(f"score it:   python -m scripts.eval_2d --pred-export {FIXED_EXPORT_DIR}")
        print(f"publish it: scripts/cvat_setup.py --export-dir {FIXED_EXPORT_DIR} "
              f"--project 'OUR PIPELINE — SAM 3.1 click-fixed (2D)' --task-suffix '— SAM 3.1 click-fixed'")
        return 0

    if args.cmd == "frames":
        # The checker gate reviews EVERYTHING, not an IoU-selected slice: with no
        # GT (the DhakaScenes case) there is nothing to select by, and the human
        # attests whole frames. Unmatched boxes score 0.0 and land in [0, 1.01).
        queue = build_queue(paths, args.scenes, 0.0, 1.01)
        # Overlay accepted fixes onto the freshly built queue, so a restart
        # keeps showing the repaired geometry (cvat_export only learns of the
        # fixes when `export` runs).
        by_id = {it["id"]: it for it in queue["items"]}
        n_fixes = 0
        for rec in ledger.snapshot().values():
            if rec.get("action") == "fix" and rec.get("bbox_after_xyxy"):
                ctx = queue["context"].get(f"{rec['scene']}|{rec['file_name']}")
                if ctx:
                    for p in ctx["preds"]:
                        if p["ann_id"] == rec["ann_id"]:
                            p["bbox_xyxy"] = rec["bbox_after_xyxy"]
                            if rec.get("segmentation"):
                                p["segmentation"] = rec["segmentation"]
                it = by_id.get(f"{rec['scene']}/{rec['ann_id']}")
                if it:
                    it["bbox_xyxy"] = rec["bbox_after_xyxy"]
                    n_fixes += 1
        status = ledger.status()
        n_frames = sum(len(fs) for chans in queue["chains"].values() for fs in chans.values())
        n_ok = sum(1 for k, v in status.items() if k.startswith("frame|") and v == "frame_ok")
        print(f"{len(queue['items'])} kept boxes over {n_frames} frames, "
              f"{len(queue['scenes'])} scenes; {n_ok} frames already marked OK"
              + (f"; {n_fixes} accepted fixes overlaid" if n_fixes else ""))
    else:
        queue = build_queue(paths, args.scenes, args.iou_min, args.iou_max)
        c = queue["counts"]
        status = ledger.status()
        pending = sum(1 for it in queue["items"] if status.get(it["id"], "pending") == "pending")
        print(f"{c['in_band']} of {c['predictions']} kept boxes have best IoU in "
              f"[{args.iou_min}, {args.iou_max}) ({c['unmatched']} overlap no GT at all); {pending} pending")
    if args.cmd == "queue":
        out = os.path.join(review_root, "queue.json")
        write_json_atomic(out, {k: v for k, v in queue.items() if k != "context"})
        per_scene = defaultdict(int)
        for it in queue["items"]:
            per_scene[it["scene"]] += 1
        for scene, n in sorted(per_scene.items()):
            print(f"  {scene}: {n}")
        print(f"wrote {out}")
        return 0

    segmenter = ChainSegmenter(paths, review_root, queue["chains"])
    app = ReviewApp(paths, queue, ledger, segmenter,
                    mode=("frames" if args.cmd == "frames" else "band"))
    server = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"review page: http://{args.host}:{args.port}/   (ledger: {ledger.path})")
    print("SAM 3.1 loads on the first click (~10 s).")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        segmenter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
