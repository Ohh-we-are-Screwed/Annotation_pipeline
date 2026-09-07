#!/usr/bin/env python3
"""Package Stage 1 clouds as a CVAT 3D task, with OUR cuboids and the human ones.

Per scene, under <work_root>/<out-subdir>/<scene>/ (default `cvat_export_3d`):

    task.zip                  pointcloud/000001.pcd + related_images/000001_pcd/CAM_*.jpg
    annotations_ours.json     Datumaro 3D 1.0 — Stage 8 cuboids
    annotations_gt.json       Datumaro 3D 1.0 — the nuScenes annotations
    frames.json               frame index -> sample token (+ frame name, channels)
    track_ids.json            cuboid track id (int) -> stitch chain id, with --stitch-map

The zip is CVAT's 3D upload layout; the two JSONs are the 3D twins of the 2D
`instances.json` pair, imported as "Datumaro 3D 1.0". Frame N of a 3D task is
the same keyframe as frames 6N..6N+5 of the 2D task for that scene.

`frames.json` is the key the importer needs: a CVAT 3D task numbers its frames
0..N-1 and neither the task nor the exported dataset says which keyframe a frame
was, so the frame -> sample-token map has to travel beside the task.

IDENTITY DOES NOT RIDE ON `track_id` ON THE WAY BACK. A cuboid is written with
`track_id` + `keyframe: true` so CVAT builds a real 3D track — measured, see
docs/evidence/2026-09-08-cvat-3d-roundtrip.md, and required by CVAT's importer,
which routes an annotation to a track only when both are present — and the
reviewer sees that id in the UI. But CVAT's Datumaro export OVERWRITES it with
its own dense per-task track index (7 -> 0, 9 -> 1), and an untracked shape
comes back carrying the number attribute's default 0.0, which collides with the
first track's index. The authoritative key is the `record_token` attribute
(`<keyframe_token>:<channel>:<proposal_index>`, the Stage 9 token), which round
trips byte-exact: an importer must read identity from that, use the exported
`track_id` only as a within-one-export grouping key, and gate that on `keyframe`
being present.

`--frames <double_annotation.json> --blank` packs only the selected
double-annotation keyframes with zero cuboids (`annotations_blank.json`) — the
empty task the two independent A/B annotators start from.
`--skip-archive-if-frames-match <double_annotation.json>` reuses a scene's
packed clouds only while that selection is unchanged: `frames.json` is
rewritten on every run, so a plain `--skip-archive` after a re-selection would
name frame N sample Y while frame N's cloud is keyframe X.

Clouds are the EGO-FRAME ground-filtered single sweeps Stage 1 wrote (§1.4) —
the same points Stage 5 painted and Stage 6 clustered — converted from the
20-byte .pcd.bin layout to binary PCD v0.7 (x y z intensity). Because the cloud
is in the ego frame and our boxes are too, a cuboid needs no transform to reach
CVAT: what the reviewer sees is the Stage 6/8 box against the points it was fit
to. The human boxes DO get transformed (global -> ego at the LiDAR anchor,
through the same ego_pose the pipeline used), so a pose error shows up as the
answer key floating off its own objects.

THE TWO CONVENTIONS THIS FILE DEPENDS ON, both established by measurement
rather than by reading a format description:

  1. **Datumaro `scale` is (x, y, z) extent in the cuboid's own frame.**
     CVAT stores a 3D cuboid as points[0:3]=position, [3:6]=rotation,
     [6:9]=scale (`dataset_manager/bindings.py`), and the 3D canvas builds a
     unit `BoxGeometry(1,1,1)` and calls `scale.set(points[6], points[7],
     points[8])` — so slot 6 is the local-x extent, i.e. the LENGTH along
     heading. Datumaro's own KITTI-raw exporter labels those same slots
     (w, h, l), which disagrees; the canvas is what draws, so the canvas wins.
     Get this wrong and every cuboid is drawn with its length and width
     swapped: still a box, still on the object, wrong shape for every vehicle.
  2. **Our `size_wlh_m` really is (width, length, height).** Checked against
     the points each box was fitted to: reading it as (l, w, h) encloses 28,345
     cloud points over 400 boxes, reading it as (w, l, h) encloses 13,623.

Yaw only. Stage 6 fits yaw-axis boxes (`yaw_axis_only: true`) and the human
boxes are re-expressed the same way, so both sets are comparable and neither
depends on a Euler-order convention.

    python -m scripts.export_cvat_3d [--scenes scene-0061] [--stitch-map .../stitch_map.json]
    python -m scripts.export_cvat_3d --frames .../double_annotation.json --blank \
        --out-subdir cvat_export_3d_double
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import zipfile

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.conventions import (  # noqa: E402
    EGO,
    NUSCENES_GLOBAL,
    Transform,
    apply_transform,
    quaternion_to_rotation_matrix,
)
from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.stage0_data_probe.probe import Substrate  # noqa: E402
from pipeline.stage1_ingestion.ingest import read_pcd_bin  # noqa: E402

TAXONOMY = "configs/taxonomy_pilot_nuscenes.yaml"

# Declared on every cuboid label, so every one of them comes back on every
# exported cuboid (a declared-but-unset attribute exports as its default; an
# undeclared one is dropped silently). `track_id` is a CVAT-internal name and is
# clobbered on export — see the module docstring.
CUBOID_ATTRIBUTES = ("record_token", "track_id", "attribute", "uncertain", "uncertain_reason")


def stable_track_int(chain_id: str) -> int:
    """A chain id as the positive integer CVAT's `track_id` number attribute wants.

    Stage 7 track ids are decimal strings and survive as themselves; a stitched
    chain that is only a single detection is named `det:<record token>`, which is
    hashed. The `| 1` keeps a hash away from 0 — the value an untracked shape
    exports as — and out of collision with it.
    """
    s = str(chain_id)
    if s.isdigit():
        return int(s)
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) | 1


def load_stitch_map(path: str) -> dict:
    """`export_release`'s stitch_map.json: record token -> stitch chain id."""
    with open(path, "r", encoding="utf-8") as fh:
        return {str(k): str(v) for k, v in json.load(fh).items()}


def load_selected_tokens(path: str) -> set:
    """The sample tokens of a double-annotation selection (`double_annotation.json`)."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    rows = doc.get("selected", []) if isinstance(doc, dict) else doc
    return {r["sample_token"] if isinstance(r, dict) else str(r) for r in rows}


def archive_matches_selection(scene_dir: str, manifest: list, selected) -> bool:
    """May the `task.zip` already in `scene_dir` be reused for this run?

    Only when the `frames.json` beside it lists exactly the frames this run
    would write — same sample tokens, same order — and every one of them is in
    the selection this run was given. `frames.json` is rewritten
    unconditionally, so any other answer would pair the OLD clouds with a FRESH
    frame mapping: frame N's point cloud would be keyframe X while frames.json
    said sample Y, and every box an annotator drew would import against the
    wrong `sample_token` with nothing to detect it (final review I7).
    """
    if not os.path.isfile(os.path.join(scene_dir, "task.zip")):
        return False
    fpath = os.path.join(scene_dir, "frames.json")
    if not os.path.isfile(fpath):
        return False
    try:
        with open(fpath, "r", encoding="utf-8") as fh:
            existing = json.load(fh)
        have = [r["sample_token"] for r in existing]
    except (OSError, ValueError, TypeError, KeyError):
        return False
    want = [r["sample_token"] for r in manifest]
    return have == want and set(want) <= set(selected)


def frames_manifest(keyframes: list, start_index: int = 0) -> list:
    """frame index -> sample token, the map the packed task itself does not carry."""
    return [{"frame": start_index + i, "name": f"{start_index + i + 1:06d}",
             "sample_token": kf["keyframe_token"], "channels": sorted(kf["cameras"])} for i, kf in enumerate(keyframes)]


def write_pcd(path: str, points: np.ndarray) -> None:
    """Binary PCD v0.7, fields x y z intensity (float32)."""
    n = points.shape[0]
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
        f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n"
    )
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(points[:, :4].astype("<f4").tobytes())


def cuboid(index: int, label_id: int, center_xyz, yaw_rad: float, extent_lwh,
           *, record_token=None, track_id=None) -> dict:
    """One Datumaro 3D cuboid. `extent_lwh` is (along heading, across, up).

    Every declared attribute is written, answered or not: the human-answered ones
    (`attribute`, `uncertain`, `uncertain_reason`) go out at their unanswered
    values, `record_token` carries identity, and `track_id` + `keyframe` are what
    make CVAT build a track out of the cuboids of one chain — its importer files
    an annotation under a track only when `track_id` is set AND `keyframe` is
    present, so the pair is written together or not at all.
    """
    attributes = {
        "occluded": False,
        "record_token": record_token or "",
        "attribute": "",
        "uncertain": False,
        "uncertain_reason": "",
    }
    if track_id is not None:
        attributes["track_id"] = int(track_id)
        attributes["keyframe"] = True
    return {
        "id": index,
        "type": "cuboid_3d",
        "attributes": attributes,
        "group": 0,
        "label_id": label_id,
        "position": [round(float(v), 4) for v in center_xyz],
        # Yaw about ego +z. Roll and pitch are zero by construction on both
        # sides, which is why no Euler-order question arises here.
        "rotation": [0.0, 0.0, round(float(yaw_rad), 6)],
        # (x, y, z) extent in the cuboid's own frame — see the module docstring.
        "scale": [round(float(v), 4) for v in extent_lwh],
    }


def datumaro_document(labels: list[str], items: list[dict]) -> dict:
    return {
        "info": {},
        "categories": {"label": {"labels": [{"name": n, "parent": "", "attributes": list(CUBOID_ATTRIBUTES)}
                                            for n in labels]}},
        "items": items,
    }


def item_skeleton(index: int, channels: list[str]) -> dict:
    name = f"{index + 1:06d}"
    return {
        "id": name,
        "annotations": [],
        "attr": {"frame": index},
        "point_cloud": {"path": f"pointcloud/{name}.pcd"},
        "related_images": [{"path": f"related_images/{name}_pcd/{c}.jpg"} for c in sorted(channels)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--boxes-dir", default=None, help="default <work_root>/stage8_inflate")
    parser.add_argument("--taxonomy", default=TAXONOMY)
    parser.add_argument("--skip-archive", action="store_true",
                        help="rebuild the annotation JSONs only; the task.zip on disk is kept "
                             "UNCONDITIONALLY (see --skip-archive-if-frames-match for the safe form)")
    parser.add_argument("--skip-archive-if-frames-match", default=None, metavar="SELECTION_JSON",
                        help="keep an existing task.zip only when the frames.json beside it lists "
                             "exactly the keyframes SELECTION_JSON (e.g. double_annotation.json) "
                             "selects for that scene, in the same order; otherwise repack it. A "
                             "changed selection must never keep old clouds against a fresh "
                             "frames.json")
    parser.add_argument("--stitch-map", default=None,
                        help="export_release's stitch_map.json (record token -> chain id); the chain "
                             "id becomes the cuboid's track id, so one object is one CVAT track")
    parser.add_argument("--frames", default=None,
                        help="double_annotation.json — pack only its selected keyframes")
    parser.add_argument("--blank", action="store_true",
                        help="write annotations_blank.json with zero cuboids (what the A/B "
                             "double-annotation pass starts from)")
    parser.add_argument("--out-subdir", default="cvat_export_3d",
                        help="output directory under <work_root> (use cvat_export_3d_double "
                             "for the blank double set so the review export is not overwritten)")
    args = parser.parse_args(argv)

    paths = load_paths(args.paths)
    stage1 = os.path.join(paths.work_root, "stage1_ingestion")
    boxes_dir = args.boxes_dir or os.path.join(paths.work_root, "stage8_inflate")
    out_root = os.path.join(paths.work_root, args.out_subdir)
    stitch_map = load_stitch_map(args.stitch_map) if args.stitch_map else {}
    selected = load_selected_tokens(args.frames) if args.frames else None
    match_selected = (load_selected_tokens(args.skip_archive_if_frames_match)
                      if args.skip_archive_if_frames_match else None)

    with open(args.taxonomy) as fh:
        phrase_of = yaml.safe_load(fh)["prompt_phrase"]
    # The label set is the taxonomy's phrases, identical to the 2D exports', so
    # a class means the same thing in the 2D task, the 3D task and the metrics.
    labels = sorted(set(phrase_of.values()))
    label_id = {name: i for i, name in enumerate(labels)}

    substrate = Substrate.load(paths)
    ego_poses = substrate.by_token("ego_pose.json")
    instances = substrate.by_token("instance.json")
    categories = substrate.by_token("category.json")
    annotations: dict[str, list[dict]] = {}
    for ann in substrate.tables["sample_annotation.json"]:
        annotations.setdefault(ann["sample_token"], []).append(ann)

    scene_root = os.path.join(stage1, "scenes")
    names = sorted(n for n in os.listdir(scene_root) if os.path.isdir(os.path.join(scene_root, n)))
    if args.scenes:
        names = [n for n in names if n in args.scenes]

    written = []
    for scene in names:
        keyframes = [json.loads(l) for l in open(os.path.join(scene_root, scene, "keyframes.jsonl"))]
        if selected is not None:
            # Scene order is kept and the frames are renumbered 000001.. — which
            # is why frames.json, not arithmetic, is what maps a frame back.
            keyframes = [k for k in keyframes if k["keyframe_token"] in selected]
            if not keyframes:
                continue
        by_keyframe: dict[str, list[dict]] = {}
        if not args.blank:
            boxes_path = os.path.join(boxes_dir, "scenes", scene, "inflated.jsonl")
            if not os.path.isfile(boxes_path):
                boxes_path = os.path.join(boxes_dir, "scenes", scene, "boxes.jsonl")
            for line in open(boxes_path):
                row = json.loads(line)
                if row.get("box"):
                    by_keyframe.setdefault(row["keyframe_token"], []).append(row)

        scene_dir = os.path.join(out_root, scene)
        manifest = frames_manifest(keyframes)
        # The reuse decision is made BEFORE anything is written, because
        # frames.json below is rewritten either way.
        skip_archive = args.skip_archive
        reused = False
        if not skip_archive and match_selected is not None:
            skip_archive = reused = archive_matches_selection(scene_dir, manifest, match_selected)
        staging = os.path.join(scene_dir, "_staging")
        if not skip_archive:
            shutil.rmtree(staging, ignore_errors=True)
            os.makedirs(os.path.join(staging, "pointcloud"))
        os.makedirs(scene_dir, exist_ok=True)

        items_ours, items_gt = [], []
        track_ids: dict[int, str] = {}
        n_ours = n_gt = n_gt_unmapped = 0

        for index, keyframe in enumerate(keyframes):
            name = f"{index + 1:06d}"
            channels = sorted(keyframe["cameras"])
            if not skip_archive:
                write_pcd(
                    os.path.join(staging, "pointcloud", f"{name}.pcd"),
                    read_pcd_bin(keyframe["single_sweep_cloud"]["path"]),
                )
                image_dir = os.path.join(staging, "related_images", f"{name}_pcd")
                os.makedirs(image_dir)
                for channel in channels:
                    shutil.copy(
                        os.path.join(paths.dataroot, keyframe["cameras"][channel]["path"]),
                        os.path.join(image_dir, f"{channel}.jpg"),
                    )

            # --- ours: already in the ego frame, no transform -----------------
            item = item_skeleton(index, channels)
            for row in by_keyframe.get(keyframe["keyframe_token"], []):
                box = row["box"]
                width, length, height = (float(v) for v in box["size_wlh_m"])
                n_ours += 1
                # The Stage 9 record token — the only identity that survives the
                # CVAT round trip (module docstring).
                record_token = f"{row['keyframe_token']}:{row['channel']}:{row['proposal_index']}"
                # The stitched chain if there is one, else Stage 7's own track id.
                chain_id = stitch_map.get(record_token, row.get("track_id"))
                track_id = None
                if chain_id is not None:
                    track_id = stable_track_int(chain_id)
                    seen = track_ids.setdefault(track_id, str(chain_id))
                    if seen != str(chain_id):
                        print(f"  ! {scene}: track id {track_id} is both {seen!r} and "
                              f"{chain_id!r}; the two chains will merge in CVAT")
                item["annotations"].append(
                    cuboid(n_ours, label_id[row["class_name"]], box["translation_m"],
                           box["yaw_rad"], (length, width, height),
                           record_token=record_token, track_id=track_id)
                )
            items_ours.append(item)

            if args.blank:
                continue

            # --- the answer key: global -> ego at the LiDAR anchor -------------
            pose = Transform.from_nuscenes(
                ego_poses[keyframe["lidar_ego_pose_token"]], source_frame=EGO, parent_frame=NUSCENES_GLOBAL
            )
            r_ego_global = quaternion_to_rotation_matrix(pose.rotation_wxyz).T
            item = item_skeleton(index, channels)
            for ann in annotations.get(keyframe["keyframe_token"], []):
                category = categories[instances[ann["instance_token"]]["category_token"]]["name"]
                phrase = phrase_of.get(category)
                if phrase is None:
                    # Out of the class space by decision (C21), not an error —
                    # the same rule export_gt_coco applies, counted so the two
                    # exports can be reconciled.
                    n_gt_unmapped += 1
                    continue
                center = apply_transform(
                    pose.inverse_matrix(), np.asarray(ann["translation"], np.float64)[None, :]
                )[0]
                r = r_ego_global @ quaternion_to_rotation_matrix(ann["rotation"])
                width, length, height = (float(v) for v in ann["size"])
                n_gt += 1
                item["annotations"].append(
                    cuboid(n_gt, label_id[phrase], center, math.atan2(r[1, 0], r[0, 0]),
                           (length, width, height))
                )
            items_gt.append(item)

        # A blank set ships the empty items only: the A/B annotators must not see
        # our cuboids, and they certainly must not see the answer key.
        outputs = (("blank", items_ours),) if args.blank else (("ours", items_ours), ("gt", items_gt))
        for suffix, items in outputs:
            with open(os.path.join(scene_dir, f"annotations_{suffix}.json"), "w") as fh:
                json.dump(datumaro_document(labels, items), fh)

        with open(os.path.join(scene_dir, "frames.json"), "w") as fh:
            json.dump(manifest, fh, indent=1)
        if args.stitch_map:
            with open(os.path.join(scene_dir, "track_ids.json"), "w") as fh:
                json.dump({str(k): v for k, v in sorted(track_ids.items())}, fh, indent=1)

        zip_path = os.path.join(scene_dir, "task.zip")
        if not skip_archive:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
                for root, _, files in os.walk(staging):
                    for f in sorted(files):
                        full = os.path.join(root, f)
                        zf.write(full, os.path.relpath(full, staging))
            shutil.rmtree(staging)
        size_mb = os.path.getsize(zip_path) / 1e6 if os.path.isfile(zip_path) else 0.0
        how = " [archive reused: frames.json unchanged]" if reused else ""
        written.append(scene)
        if args.blank:
            print(f"  {scene}: {len(keyframes)} frames, BLANK (0 cuboids, no answer key) "
                  f"-> {scene_dir} ({size_mb:.0f} MB){how}")
        else:
            print(f"  {scene}: {len(keyframes)} frames, ours {n_ours:>5} in {len(track_ids)} track(s), "
                  f"human {n_gt:>5} ({n_gt_unmapped} out of class space) "
                  f"-> {scene_dir} ({size_mb:.0f} MB){how}")

    print(f"\nwrote {len(written)} scene(s) under {out_root}")
    print(f"publish with: python -m scripts.cvat_setup_3d --which {'double' if args.blank else 'ours'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
