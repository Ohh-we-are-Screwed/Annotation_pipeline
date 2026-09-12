#!/usr/bin/env python3
"""The release's 2D layer: `<export>/boxes/annotations_2d/` (2026-09-13).

The nuScenes tables beside this folder are 3D only — a cuboid per shipped
detection and nothing about the image it was detected in. This writes the other
half — one COCO document plus one index, covering every scene in the release
(one, for a chunk):

  * `instances_2d.json` — every Stage 3m proposal as a COCO annotation: the
    detector's box, its class phrase, its score, and the Stage 4
    SAM mask traced to polygons (ALL external blobs, not just the largest — an
    occluded object is two blobs and a segmentation that drops one is wrong).
    Each annotation carries a `dhakascenes` block saying what became of it in
    3D: the Stage 7 `status`, the track, the stitched chain, and — when it
    actually shipped — the `sample_annotation` token and `instance_token` of the
    cuboid it turned into.
  * `tracks.json` — one entry per Stage 7 track: the 2D <-> 3D <-> identity
    index, so "show me every image crop of instance X" is one lookup.

THE JOIN, and why it needs no guessing. Stage 9 mints one pre-label token per
detection, `f"{keyframe_token}:{channel}:{proposal_index}"` (`gate.record_of`),
and `scripts/export_release.py` copies it onto the shipped row verbatim as
`dhakascenes_record_token` (and keys `stitch_map.json` with the same string).
That one key therefore addresses the same detection in Stage 3m, Stage 4,
Stage 7, Stage 9 and the release. Every lookup here is that key; nothing is
matched by geometry, class or proximity.

Idempotent: the folder is rewritten and the DELIVERY_NOTE section replaced.
Reads the release; writes ONLY `annotations_2d/` and `DELIVERY_NOTE.md`.

    python scripts/export_annotations_2d.py --paths configs/batch_20260912/chunk_17.yaml \
        --scene dhaka_20260911_154512_chunk_0003 \
        --export-dir /mnt/exoshdd/.../exports/chunk_17/boxes
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.stage5_lift.lift import LiftContractError, MaskFile  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYER = "annotations_2d"
NOTE_HEADING = "## 2D layer (annotations_2d/)"
DEFAULT_TAXONOMY = os.path.join(REPO_ROOT, "configs", "taxonomy_pilot_dhaka.yaml")
# A contour this small is a mask-edge speck, not a part of the object. Same
# order as export_cvat_coco.py's 20 px, lower because this layer is the
# annotation rather than a review overlay.
MIN_POLYGON_AREA_PX = 4.0
# Image formats a `sample_data` row can name. The release writes jpg; the other
# two are here so a future substrate does not silently lose its camera rows.
IMAGE_FORMATS = ("jpg", "jpeg", "png")
# The delivery note sections copied verbatim into `info.caveats`. They are the
# release's own words about what a box means; repeating them here keeps a
# consumer who only ever opens instances_2d.json from missing them.
QUOTED_NOTE_SECTIONS = {"annotation_rule": "## Annotation rule", "range": "## Range"}
TWO_D_CAVEATS = (
    "Every row here is the DETECTOR's 2D claim. `bbox` is the Stage 3m proposal and "
    "`segmentation` the Stage 4 SAM mask; neither passed the release's annotation rule, "
    "which gates only the 3D cuboid a row may have become. Filter on `dhakascenes.tier` "
    "or `dhakascenes.sample_annotation_token` to get the shipped subset.",
    "`dhakascenes.status_3d` says why a proposal did or did not become a cuboid "
    "(`absent` = the 2D->3D chain wrote no row for it at all). Only `fit` rows carry "
    "`depth_m`, and only rows with a non-null `sample_annotation_token` are in "
    "`sample_annotation.json`. `info.counts_by_status_3d_and_channel` shows which "
    "channels the box producer was active on — the others are not failures, they were "
    "never in its scope.",
    "`area` is the BBOX area, so a box-only consumer reads it correctly; the mask's own "
    "pixel count is `dhakascenes.mask_area_px`.",
    "The images referenced by `file_name` are the RELEASED blobs (face and plate "
    "blurred); the detector saw the originals. See the delivery note's Anonymisation "
    "section.",
)


# ---------------------------------------------------------------------------
# small io
# ---------------------------------------------------------------------------


def read_jsonl(path: str):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def read_json(path: str, default=None):
    if not os.path.isfile(path):
        return default
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def write_json_compact(path: str, payload) -> str:
    """Atomic, and COMPACT — `manifest.write_json_atomic` indents, and this file
    holds ~80 000 annotations whose polygons are millions of numbers.

    Plain `open`, not `mkstemp`: the delivery folders carry a default ACL, and a
    0600 mkstemp file lands in a 0660 release unreadable by anyone but the user
    who ran the export. A normal create takes the umask and the ACL like every
    other file beside it.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, separators=(",", ":"))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


def git_sha() -> str | None:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=REPO_ROOT)
    except OSError:
        return None
    return proc.stdout.strip() or None


# ---------------------------------------------------------------------------
# masks -> COCO polygons
# ---------------------------------------------------------------------------


def mask_to_polygons(mask: np.ndarray, epsilon_px: float = 1.0,
                     min_area_px: float = MIN_POLYGON_AREA_PX) -> list[list[float]]:
    """COCO `[[x1,y1,x2,y2,...], ...]` from a bool mask: every external blob.

    `view_2d.mask_polygon` keeps only the largest contour because it draws an
    overlay; here the polygon IS the annotation, so an occluded object's second
    fragment must survive. Empty mask -> [].
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area_px:
            continue
        approx = cv2.approxPolyDP(contour, epsilon_px, True)
        if len(approx) < 3:
            continue
        polygons.append([float(v) for v in approx.reshape(-1)])
    return polygons


# ---------------------------------------------------------------------------
# the release side
# ---------------------------------------------------------------------------


def version_dir(export_dir: str) -> str:
    """The `v1.0-*` directory holding the tables. `release_meta.json` names it;
    a release without one (the test fixture) is found by looking."""
    meta = read_json(os.path.join(export_dir, "release_meta.json")) or {}
    named = meta.get("version")
    if named and os.path.isfile(os.path.join(export_dir, named, "sample_data.json")):
        return os.path.join(export_dir, named)
    found = sorted(d for d in os.listdir(export_dir)
                   if os.path.isfile(os.path.join(export_dir, d, "sample_data.json")))
    if not found:
        raise SystemExit(f"{export_dir}: no version directory with a sample_data.json")
    return os.path.join(export_dir, found[0])


def camera_images(sample_data: list[dict], keyframe_tokens: set[str],
                  first_id: int = 1) -> list[dict]:
    """One COCO image per camera blob of this scene, ids running from `first_id`
    in (timestamp, filename) order.

    The channel is the blob's own directory (`samples/<CHANNEL>/*.jpg`) — the
    layout the release writes and the one Stage 3 records as `image_path`, so
    the two cannot disagree. `_assert_covered` below checks that every proposal
    row found its image, which is what would catch a layout change.
    """
    rows = [r for r in sample_data
            if str(r.get("fileformat", "")).lower() in IMAGE_FORMATS
            and r["sample_token"] in keyframe_tokens]
    images = []
    for i, r in enumerate(sorted(rows, key=lambda r: (r["timestamp"], r["filename"])),
                      start=first_id):
        images.append({
            "id": i,
            "file_name": r["filename"],
            "width": int(r["width"]),
            "height": int(r["height"]),
            "sample_data_token": r["token"],
            "sample_token": r["sample_token"],
            "channel": os.path.basename(os.path.dirname(r["filename"])),
            "timestamp": int(r["timestamp"]),
        })
    return images


def note_section(text: str, heading: str) -> str:
    """One `## ...` section of DELIVERY_NOTE.md, heading excluded, or ''."""
    lines = text.splitlines()
    try:
        start = lines.index(heading) + 1
    except ValueError:
        return ""
    end = next((i for i in range(start, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end]).strip()


def update_note(path: str, section: str) -> None:
    """Append (or replace) this layer's section. Every other line is untouched.

    The section is written HERE rather than in `pipeline/release/note.py`
    because that note is rendered by the release export, which runs BEFORE this
    layer exists — a note.py branch could only ever describe a folder that was
    not there yet. Re-running the release export drops the section; re-running
    this script puts it back, which is exactly the order run_stages.sh uses.
    """
    text = open(path, encoding="utf-8").read() if os.path.isfile(path) else ""
    lines = text.splitlines()
    if NOTE_HEADING in lines:
        start = lines.index(NOTE_HEADING)
        end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
        lines = lines[:start] + lines[end:]
    body = "\n".join(lines).rstrip("\n")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(body + "\n\n" + section.rstrip("\n") + "\n")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# one keyframe
# ---------------------------------------------------------------------------


def keyframe_annotations(rows: list[dict], mask_path: str | None, ctx: dict) -> list[dict]:
    """Every proposal of one keyframe, channels in name order, as COCO rows
    without ids (assigned once the whole scene is in, so they are stable)."""
    out: list[dict] = []
    masks = MaskFile(mask_path) if mask_path else None
    try:
        for row in sorted(rows, key=lambda r: r["channel"]):
            channel = row["channel"]
            keyframe = row["keyframe_token"]
            image_id = ctx["image_id"].get(row.get("sample_data_token"))
            if image_id is None:
                raise SystemExit(
                    f"{keyframe}/{channel}: sample_data_token "
                    f"{row.get('sample_data_token')!r} is not a camera row of this release; "
                    "the 2D layer would reference an image the release does not ship")
            vlm = ctx["vlm"].get((keyframe, channel)) or []
            for index, xyxy in enumerate(row["boxes_xyxy_px"]):
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                record = f"{keyframe}:{channel}:{index}"
                box3d = ctx["boxes3d"].get(record)
                shipped = ctx["shipped"].get(record) or {}
                stereo = (box3d or {}).get("stereo") or {}
                track = (box3d or {}).get("track_id")
                polygons, mask_area = [], 0
                if masks is not None:
                    try:
                        mask = masks.mask(channel, index)
                    except LiftContractError:
                        # Stage 4 wrote no stack for this channel, or fewer masks
                        # than Stage 3m has proposals. The 2D box is still real.
                        mask = None
                    if mask is not None:
                        polygons = mask_to_polygons(mask, ctx["epsilon_px"])
                        mask_area = int(mask.sum())
                out.append({
                    "id": None,
                    "image_id": image_id,
                    "category_id": None,   # the caller resolves it once every scene is in
                    "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    "area": round((x2 - x1) * (y2 - y1), 2),
                    "score": round(float(row["scores"][index]), 6),
                    "iscrowd": 0,
                    "segmentation": polygons,
                    "dhakascenes": {
                        "record_token": record,
                        "keyframe_token": keyframe,
                        "channel": channel,
                        "proposal_index": index,
                        "detector_arm": row["proposal_arm"][index],
                        "class_name": row["class_names"][index],
                        "mask_area_px": mask_area,
                        "status_3d": (box3d or {}).get("status", "absent"),
                        "track_id": None if track is None else str(track),
                        "track_id_stitched": ctx["stitch"].get(record),
                        "instance_token": shipped.get("instance_token"),
                        "sample_annotation_token": shipped.get("token"),
                        "tier": ctx["tier"].get(record, shipped.get("dhakascenes_tier")),
                        "depth_m": stereo.get("d_near_m") if (box3d or {}).get("status") == "fit" else None,
                        "vlm_label": vlm[index] if index < len(vlm) else None,
                    },
                })
    finally:
        if masks is not None:
            masks.close()
    return out


# ---------------------------------------------------------------------------
# tracks.json
# ---------------------------------------------------------------------------


def build_tracks(annotations: list[dict], images: list[dict]) -> list[dict]:
    """One entry per Stage 7 track: class, keyframes, and the 2D/3D/identity links."""
    order = {im["id"]: (im["timestamp"], im["channel"]) for im in images}
    by_track: dict[str, list[dict]] = collections.defaultdict(list)
    for ann in annotations:
        track = ann["dhakascenes"]["track_id"]
        if track is not None:
            by_track[track].append(ann)
    tracks = []
    for track, anns in sorted(by_track.items(), key=lambda kv: _numeric(kv[0])):
        anns.sort(key=lambda a: order[a["image_id"]])
        classes = collections.Counter(a["dhakascenes"]["class_name"] for a in anns)
        per_kf: dict[str, dict] = {}
        velocities = []
        for a in anns:
            d = a["dhakascenes"]
            kf = per_kf.setdefault(d["keyframe_token"], {
                "sample_token": d["keyframe_token"], "sample_annotation_token": None,
                "annotation_ids": {}})
            kf["annotation_ids"][d["channel"]] = a["id"]
            if d["sample_annotation_token"]:
                kf["sample_annotation_token"] = d["sample_annotation_token"]
            if d.get("velocity_chain_mps"):
                velocities.append(d["velocity_chain_mps"])
        keyframes = list(per_kf.values())
        stitched = next((a["dhakascenes"]["track_id_stitched"] for a in anns
                         if a["dhakascenes"]["track_id_stitched"]), None)
        instance = next((a["dhakascenes"]["instance_token"] for a in anns
                         if a["dhakascenes"]["instance_token"]), None)
        mean = ([sum(v[i] for v in velocities) / len(velocities) for i in range(2)]
                if velocities else None)
        tracks.append({
            "track_id": track,
            "track_id_stitched": stitched,
            "instance_token": instance,
            "class_name": classes.most_common(1)[0][0],
            "class_names": dict(classes),
            "n_keyframes": len(keyframes),
            "n_annotations": len(anns),
            "n_shipped_3d": sum(1 for k in keyframes if k["sample_annotation_token"]),
            "first_sample_token": keyframes[0]["sample_token"],
            "last_sample_token": keyframes[-1]["sample_token"],
            "mean_velocity_mps": mean,
            "mean_speed_mps": math.hypot(*mean) if mean else None,
            "keyframes": keyframes,
        })
    return tracks


def _numeric(track_id: str):
    """Stage 7 ids are integers carried as text; sort them as integers so track
    10 follows track 9 rather than track 1."""
    return (0, int(track_id), "") if track_id.lstrip("-").isdigit() else (1, 0, track_id)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def scene_layer(scene: str, *, dirs: dict, release: dict, first_image_id: int,
                epsilon_px: float, workers: int) -> tuple[list[dict], list[dict]]:
    """One scene's images and annotations. Categories are resolved by the caller,
    once every scene's phrases are known, so `category_id` is filled in there."""
    proposals = list(read_jsonl(os.path.join(dirs["stage3"], "scenes", scene, "proposals.jsonl")))
    if not proposals:
        raise SystemExit(f"{dirs['stage3']}/scenes/{scene}/proposals.jsonl: no rows")
    by_keyframe: dict[str, list[dict]] = collections.OrderedDict()
    for row in proposals:
        by_keyframe.setdefault(row["keyframe_token"], []).append(row)

    images = camera_images(release["sample_data"], set(by_keyframe), first_image_id)
    ctx = {
        "image_id": {im["sample_data_token"]: im["id"] for im in images},
        "shipped": release["shipped"],
        "stitch": release["stitch"],
        "epsilon_px": epsilon_px,
        "boxes3d": {f"{r['keyframe_token']}:{r['channel']}:{r['proposal_index']}": r
                    for r in _rows(dirs["boxes"], scene, "boxes.jsonl")},
        "tier": {r["token"]: (r.get("provenance") or {}).get("tier")
                 for r in _rows(dirs["stage9"], scene, "prelabels.jsonl")},
        "vlm": {(r["keyframe_token"], r["channel"]):
                [v.get("vlm_phrase") for v in (r.get("vlm_check") or {}).get("verdicts", [])]
                for r in _rows(dirs["checked"], scene, "proposals.jsonl")},
    }
    masks_dir = dirs["masks"]
    mask_path_of = {r["keyframe_token"]: os.path.join(masks_dir, r["mask_path"])
                    for r in _rows(masks_dir, scene, "masks.jsonl")}

    def one(item):
        keyframe, rows = item
        path = mask_path_of.get(keyframe)
        return keyframe_annotations(rows, path if path and os.path.isfile(path) else None, ctx)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        batches = list(pool.map(one, by_keyframe.items()))
    return images, [ann for batch in batches for ann in batch]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    ap.add_argument("--scene", default=None, help="the one scene to export")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="the plural run_stages.sh passes; default is every scene under "
                         "--stage3-dir, which for a one-chunk release is the one scene")
    ap.add_argument("--export-dir", required=True,
                    help="the release's boxes/ directory; annotations_2d/ is written inside it")
    ap.add_argument("--stage3-dir", default=None, help="default <work_root>/stage3_merged")
    ap.add_argument("--checked-dir", default=None,
                    help="default <work_root>/stage3_checked; absent = no vlm_label")
    ap.add_argument("--masks-dir", default=None, help="default <work_root>/stage4_masks; '' disables")
    ap.add_argument("--boxes-dir", default=None, help="default <work_root>/stage7_track")
    ap.add_argument("--stage9-dir", default=None, help="default <work_root>/stage9_qa")
    ap.add_argument("--taxonomy", default=DEFAULT_TAXONOMY,
                    help="phrases that become the COCO categories; phrases the proposals "
                         "use but this file lacks are appended rather than KeyErroring")
    ap.add_argument("--polygon-epsilon-px", type=float, default=1.0,
                    help="cv2.approxPolyDP tolerance when tracing a mask")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)

    t0 = time.time()
    work = load_paths(a.paths).work_root
    dirs = {
        "stage3": a.stage3_dir or os.path.join(work, "stage3_merged"),
        "checked": a.checked_dir if a.checked_dir is not None else os.path.join(work, "stage3_checked"),
        "masks": os.path.join(work, "stage4_masks") if a.masks_dir is None else a.masks_dir,
        "boxes": a.boxes_dir or os.path.join(work, "stage7_track"),
        "stage9": a.stage9_dir or os.path.join(work, "stage9_qa"),
    }
    export = os.path.abspath(a.export_dir)
    tables = version_dir(export)
    names = a.scenes or ([a.scene] if a.scene else sorted(
        n for n in os.listdir(os.path.join(dirs["stage3"], "scenes"))
        if os.path.isfile(os.path.join(dirs["stage3"], "scenes", n, "proposals.jsonl"))))
    if not names:
        raise SystemExit(f"{dirs['stage3']}: no scene has a proposals.jsonl")

    # --- the release side, read once: it is not per-scene ---------------------
    meta = read_json(os.path.join(export, "release_meta.json"), {}) or {}
    release = {
        "sample_data": read_json(os.path.join(tables, "sample_data.json"), []),
        # `dhakascenes_record_token` embeds the keyframe token, so one map covers
        # every scene of the release without collision.
        "shipped": {r["dhakascenes_record_token"]: r
                    for r in read_json(os.path.join(tables, "sample_annotation.json"), [])
                    if r.get("dhakascenes_record_token")},
        "stitch": {k: str(v) for k, v in
                   (read_json(os.path.join(export, "stitch_map.json"), {}) or {}).items()},
    }

    images: list[dict] = []
    annotations: list[dict] = []
    for scene in names:
        scene_images, scene_annotations = scene_layer(
            scene, dirs=dirs, release=release, first_image_id=len(images) + 1,
            epsilon_px=a.polygon_epsilon_px, workers=a.workers)
        images += scene_images
        annotations += scene_annotations

    # --- categories: the taxonomy's phrases plus anything the proposals used ---
    with open(a.taxonomy, encoding="utf-8") as fh:
        phrases = set(yaml.safe_load(fh)["prompt_phrase"].values())
    phrases |= {ann["dhakascenes"]["class_name"] for ann in annotations}
    category_id = {phrase: i for i, phrase in enumerate(sorted(phrases), start=1)}
    phrase_to_class = ((meta.get("mapper") or {}).get("used")) or {}
    category_token = {r["name"]: r["token"]
                      for r in read_json(os.path.join(tables, "category.json"), []) or []}
    categories = [{"id": i, "name": phrase, "supercategory": "",
                   "nuscenes_category": phrase_to_class.get(phrase),
                   "nuscenes_category_token": category_token.get(phrase_to_class.get(phrase))}
                  for phrase, i in sorted(category_id.items(), key=lambda kv: kv[1])]
    for i, ann in enumerate(annotations, start=1):
        ann["id"] = i
        ann["category_id"] = category_id[ann["dhakascenes"]["class_name"]]

    # The chain velocity lives on the shipped cuboid, not on the 2D row; lend it
    # to the track builder rather than publish it twice in every annotation.
    for ann in annotations:
        if ann["dhakascenes"]["sample_annotation_token"]:
            velocity = release["shipped"][ann["dhakascenes"]["record_token"]].get(
                "dhakascenes_velocity_chain_mps")
            if velocity:
                ann["dhakascenes"]["velocity_chain_mps"] = [float(v) for v in velocity]
    tracks = build_tracks(annotations, images)
    for ann in annotations:
        ann["dhakascenes"].pop("velocity_chain_mps", None)

    by_status = collections.Counter(x["dhakascenes"]["status_3d"] for x in annotations)
    by_status_channel: dict[str, dict[str, int]] = {}
    for (channel, status), n in sorted(collections.Counter(
            (x["dhakascenes"]["channel"], x["dhakascenes"]["status_3d"]) for x in annotations).items()):
        by_status_channel.setdefault(channel, {})[status] = n
    counts = {
        "images": len(images), "annotations": len(annotations),
        "annotations_with_polygons": sum(1 for x in annotations if x["segmentation"]),
        "annotations_linked_to_3d": sum(1 for x in annotations
                                        if x["dhakascenes"]["sample_annotation_token"]),
        "annotations_with_a_3d_status": sum(1 for x in annotations
                                            if x["dhakascenes"]["status_3d"] != "absent"),
        "tracks": len(tracks),
    }

    note_path = os.path.join(export, "DELIVERY_NOTE.md")
    note_text = open(note_path, encoding="utf-8").read() if os.path.isfile(note_path) else ""
    caveats = {"delivery_note": "../DELIVERY_NOTE.md", "two_d_layer": list(TWO_D_CAVEATS)}
    caveats.update({key: note_section(note_text, heading)
                    for key, heading in QUOTED_NOTE_SECTIONS.items()})
    info = {
        "description": "DhakaScenes 2D layer — Stage 3m boxes + Stage 4 masks, joined to the "
                       "3D release by dhakascenes_record_token",
        "scenes": names,
        "chunk": os.path.basename(os.path.dirname(export)),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "exporter": "scripts/export_annotations_2d.py",
        "exporter_git_sha": git_sha(),
        "join_key": "dhakascenes_record_token = '<keyframe_token>:<channel>:<proposal_index>'",
        "release": {"version": os.path.basename(tables), "git_sha": meta.get("git_sha"),
                    "created_utc": meta.get("created_utc"),
                    "tiers_admitted": meta.get("tiers_admitted")},
        "sources": {"work_root": work, "stage3_dir": dirs["stage3"],
                    "masks_dir": dirs["masks"] or None, "boxes_dir": dirs["boxes"],
                    "stage9_dir": dirs["stage9"], "checked_dir": dirs["checked"] or None,
                    "taxonomy": a.taxonomy},
        "polygon": {"epsilon_px": a.polygon_epsilon_px, "min_area_px": MIN_POLYGON_AREA_PX,
                    "contours": "all external (cv2.RETR_EXTERNAL)"},
        "counts": counts,
        "counts_by_status_3d": dict(sorted(by_status.items())),
        "counts_by_status_3d_and_channel": by_status_channel,
        "caveats": caveats,
        "elapsed_s": round(time.time() - t0, 1),
    }

    out_dir = os.path.join(export, LAYER)
    write_json_compact(os.path.join(out_dir, "instances_2d.json"),
                       {"info": info, "licenses": [], "categories": categories,
                        "images": images, "annotations": annotations})
    write_json_compact(os.path.join(out_dir, "tracks.json"),
                       {"info": {k: info[k] for k in ("scenes", "chunk", "created_utc", "release",
                                                      "join_key", "counts")},
                        "tracks": tracks})
    if note_text:
        update_note(note_path, _note_block(info))
    print(f"{', '.join(names)}: {counts['images']} images, {counts['annotations']} annotations "
          f"({counts['annotations_with_polygons']} with polygons, "
          f"{counts['annotations_linked_to_3d']} linked to a 3D cuboid), "
          f"{counts['tracks']} tracks -> {out_dir}  ({info['elapsed_s']}s)")
    return 0


def _rows(root: str | None, scene: str, name: str) -> list[dict]:
    """A whole-scene jsonl under `<root>/scenes/<scene>/`, or [] when absent.

    Absent is normal, not an error: a run without Stage 9, without Stage 4 or
    without the VLM check still has a 2D layer — it just has fewer columns.
    """
    path = os.path.join(root, "scenes", scene, name) if root else None
    return list(read_jsonl(path)) if path and os.path.isfile(path) else []


def _note_block(info: dict) -> str:
    counts, sources = info["counts"], info["sources"]
    rows = [("rows", f"{counts['annotations']} annotations over {counts['images']} camera images "
                     f"({counts['annotations_with_polygons']} carry mask polygons)"),
            ("linked to 3D", f"{counts['annotations_linked_to_3d']} became a row in "
                             f"sample_annotation.json; {counts['annotations_with_a_3d_status']} "
                             f"reached the 2D->3D chain at all"),
            ("tracks", f"{counts['tracks']} in tracks.json"),
            ("status_3d", ", ".join(f"{k} {v}" for k, v in info["counts_by_status_3d"].items())),
            ("sources", f"{sources['stage3_dir']} (boxes), {sources['masks_dir']} (masks), "
                        f"{sources['boxes_dir']} (3D outcome), {sources['stage9_dir']} (tier)"),
            ("written by", f"{info['exporter']} @ {info['exporter_git_sha']}")]
    return "\n".join([
        NOTE_HEADING, "",
        "- annotations_2d/instances_2d.json: COCO 1.0. Every Stage 3m proposal as a 2D box, "
        "with the detector's phrase as its category, the Stage 4 SAM mask as polygon "
        "`segmentation` (all external blobs), and a `dhakascenes` block naming what became of it "
        "in 3D — `status_3d`, `track_id`, `track_id_stitched`, `tier`, `depth_m`, and the "
        "`sample_annotation_token` / `instance_token` of the cuboid it shipped as, when it "
        "shipped. The join key is `dhakascenes_record_token`, the same string "
        "`sample_annotation.json` and `stitch_map.json` carry.",
        "- annotations_2d/tracks.json: one entry per Stage 7 track — class, keyframes, mean "
        "velocity, and per keyframe the `sample_annotation_token` and the 2D annotation ids by "
        "channel. This is the 2D <-> 3D <-> identity index.",
        "- NOT gated: these are the detector's claims, not the release's ground truth. The "
        "Annotation rule above applies to the 3D cuboid, not to the 2D box; filter on "
        "`dhakascenes.sample_annotation_token` (or `dhakascenes.tier`) for the shipped subset.",
        "",
        "\n".join(f"- {k}: {v}" for k, v in rows),
    ])


if __name__ == "__main__":
    raise SystemExit(main())
