#!/usr/bin/env python3
"""Per-point OBJECT labels for the release: `<export>/lidarseg/` (2026-09-13).

`scripts/export_road_lidarseg.py` ships the ROAD surface as a separate
nuScenes-lidarseg root. This is the other half the operator asked for — car,
bus, cycle_rickshaw, pedestrian … per point — and it is written INTO the
release's `boxes/` directory, beside the tables, so one root carries cuboids and
segmentation.

DEVKIT-INVISIBLE BY DEFAULT, one rename away from devkit use. The index table is
`<version>/dhakascenes_lidarseg.json`, NOT `lidarseg.json`: `NuScenes.__init__`
auto-detects the latter, and the moment it does it merges its hard-coded 32-name
colormap against `category.json` BY NAME and raises `KeyError: 'car'` — this
release's taxonomy is its own, and that colormap is closed with no override
argument (docs/EXPORT_STATUS_AND_DECISIONS.md §3.1). `boxes/` is the root the
operator's BEVFusion conversion loads, so a bare `NuScenes(version, dataroot)`
must keep working, and it does. A consumer who wants the devkit's lidarseg APIs
copies the table to `lidarseg.json` IN A COPY OF THE ROOT and installs a
two-line `get_colormap` shim; both, or it crashes. The recipe is in the delivery
note and in `lidarseg_meta.json` (`devkit_recipe`), and reading a bin without
the devkit is one `numpy.fromfile` (`read_without_the_devkit`).

WHERE THE LABELS COME FROM. Stage 5 painted Stage 4's masks onto the fused
cloud: per keyframe a `point_index` (a row of Stage 1's FUSED single-sweep
cloud) and an `instance_id` (a row of that keyframe's `instances` list, whose
`class_name` is the detector's phrase). `release_phrase_map` turns the phrase
into the release's class, and `category.json` turns that into the label index.

WHY IT IS NOT A ONE-LINE COPY. A lidarseg `.bin` is POSITIONAL over the RAW
sensor blob, and Stage 5's indices are positional over the FUSED, GROUND-
FILTERED ego-frame cloud — a different array, built from two blobs
(`samples/LIDAR_TOP/*.pcd.bin` then `samples/ZED_WORLD/*.pcd.bin`) and then cut
down by a boolean mask. So this exporter REBUILDS the unfiltered fused cloud
from the released blobs using ingest.py's OWN functions (`read_pcd_bin`,
`thin_stereo`, `Transform`, `apply_transform`, `stereo_block_to_ego`), in
ingest.py's order, and carries a parallel (channel, raw row) array through it.
Both sides are the same float64 arithmetic cast to float32 exactly once, so the
kept cloud is a BYTE-IDENTICAL SUBSEQUENCE of the reconstruction: the match is
exact equality on the whole 20-byte record, not a nearest-neighbour with a
tolerance. Measured on chunk 14: 0 unmatched rows in 1291 keyframes.

TWO BINS PER KEYFRAME, which is our extension. The devkit's own layout is one
bin per LIDAR_TOP blob; here MOST object points are stereo (~92 % of the fused
cloud is ZED_WORLD), so a LIDAR_TOP-only layer would drop most of the
segmentation. A ZED_WORLD bin is still a legal lidarseg row — the devkit only
requires one row per bin, bound to a sample_data token.

Refusals (exit 2, one line to stderr, nothing written): stage5_lift missing its
manifest or marker; a `lidarseg.json` already in the table root, which this
exporter never writes and which breaks the bare load for every other consumer;
a keyframe whose kept cloud does not match `n_points_cloud`
or whose painted points do not all match the reconstruction is refused and
COUNTED, and if more than 1 % of a scene's keyframes are refused the whole
export is. Everything is computed before the first byte lands.

    python scripts/export_lidarseg.py --paths configs/batch_20260912/chunk_14.yaml \
        --scene dhaka_20260911_154512_chunk_0000 \
        --export-dir /mnt/exoshdd/.../exports/chunk_14/boxes
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import shutil
import stat
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.conventions import (  # noqa: E402
    EGO,
    LIDAR,
    NUSCENES_GLOBAL,
    Transform,
    apply_transform,
)
from pipeline.common.manifest import UpstreamRefusal, require_upstream, write_json_atomic  # noqa: E402
from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.stage1_ingestion.ingest import (  # noqa: E402
    STEREO_CHANNELS,
    read_pcd_bin,
    stereo_block_to_ego,
    thin_stereo,
)
# The note handling, the release's phrase -> class map and the version-directory
# lookup are the 2D layer's; one implementation of each, not two.
from scripts.export_annotations_2d import (  # noqa: E402
    git_sha,
    read_json,
    read_jsonl,
    release_phrase_map,
    update_note,
    version_dir,
)

LIDAR_CHANNEL = "LIDAR_TOP"
NOTE_HEADING = "## LiDAR segmentation (lidarseg/)"
CATEGORY_BACKUP = "category.json.pre_lidarseg.bak"
# The index table, and the name the devkit LOOKS FOR. They differ on purpose:
# `NuScenes.__init__` auto-detects `lidarseg.json` / `panoptic.json` in the
# table root, and the moment it finds one it merges its closed 32-name colormap
# against category.json BY NAME and dies on ours (`KeyError: 'car'`). This root
# is what the operator's BEVFusion conversion loads, so the layer is
# devkit-INVISIBLE by default: same schema, same rows, one rename away from
# devkit use. See `devkit_recipe` in lidarseg_meta.json.
TABLE_NAME = "dhakascenes_lidarseg.json"
DEVKIT_TABLE_NAME = "lidarseg.json"
NOISE_NAME = "noise"
NOISE_DESCRIPTION = (
    "index 0 = NOT PAINTED BY ANY OBJECT MASK (road, buildings, vegetation, sky returns, "
    "unlabelled) — NOT literally noise. The devkit convention reserves index 0 for noise "
    "and every consumer's colour map assumes it, so the row carries that name; this layer "
    "segments objects only, and the road surface is a separate export (road/)."
)
# Refuse the scene, not just the keyframe, past this. A handful of keyframes
# whose reconstruction does not match is a data accident; 1 % is a broken
# assumption about how the fused cloud was built, and a positional .bin written
# on a broken assumption shears silently.
MAX_REFUSED_FRACTION = 0.01
CAVEATS = (
    "LABEL 0 IS 'UNLABELLED', NOT 'NOISE'. A point is 0 because no Stage 4 object mask "
    "painted it: road, buildings, vegetation, anything outside the detector's vocabulary, "
    "and anything the detector missed. The devkit reserves index 0 for `noise`, so that is "
    "the name the category row carries, but do not read it as a return quality judgement.",
    "THE ROAD IS NOT LABELLED HERE. stage_road did not run for this release, so the "
    "driveable surface is 0 like everything else unpainted. The road layer, where a run has "
    "one, is the separate `road/` nuScenes-lidarseg root.",
    "ZED_WORLD BINS ARE AN EXTENSION. nuScenes-lidarseg ships one bin per LIDAR_TOP blob; "
    "this layer also ships one per ZED_WORLD blob, positional over that file's own point "
    "order. Most object points in this capture are stereo, so a LIDAR_TOP-only layer would "
    "carry a small minority of the segmentation. The devkit's own APIs — once the "
    "`devkit_recipe` below is applied — read the LIDAR_TOP bin and ignore the ZED_WORLD one; "
    "read the stereo bin directly with `numpy.fromfile(path, dtype=numpy.uint8)`.",
    "THESE ARE THE DETECTOR'S CLAIMS. A label is Stage 3m's phrase, through the Stage 4 "
    "SAM mask, through Stage 5's painting. It is NOT gated by the release's annotation "
    "rule, which admits 3D cuboids; a point can carry a class here whose cuboid never "
    "shipped. Nothing here was reviewed by a human.",
    "STEREO POINTS CARRY STAGE 1'S CORRECTIONS. The label for a ZED_WORLD row is computed "
    "on that row after the same ring thinning, z correction and pitch correction Stage 1 "
    "applied when it built the fused cloud, with the same ego hop (the inverse of the "
    "sample's LIDAR_TOP ego_pose). The label is positional, so the raw blob you read it "
    "against must be the released one, untouched.",
    "ONE LABEL PER POINT, LAST PAINTER WINS. Stage 5 resolved multi-camera contests before "
    "this exporter saw them (`point_index` is unique per keyframe), so no point here is "
    "shared between two classes.",
)


class ExportRefusal(RuntimeError):
    """A precondition of the export is violated; nothing has been written."""


def write_table(path: str, payload, reference: str) -> str:
    """`write_json_atomic`, then the mode the release gave its OWN tables.

    `write_json_atomic` creates through `mkstemp`, which is 0600 by design. The
    delivery folders carry a group ACL and every table the release wrote is
    0660, so a 0600 table beside them is one nobody but the user who ran the
    export can read — and this exporter rewrites `category.json`, a table the
    release already shipped. Same trap `export_annotations_2d.write_json_compact`
    documents; here the round-trip check is worth keeping, so the mode is fixed
    afterwards instead of the writer being swapped. `reference` is a table the
    release wrote, so nothing has to hardcode 0660 or guess at the umask.
    """
    write_json_atomic(path, payload)
    os.chmod(path, stat.S_IMODE(os.stat(reference).st_mode))
    return path


# ---------------------------------------------------------------------------
# Index mapping: Stage 5's fused-cloud rows -> raw blob rows
# ---------------------------------------------------------------------------


def row_keys(cloud: np.ndarray) -> np.ndarray:
    """One opaque 20-byte key per point, for exact whole-record comparison."""
    return np.ascontiguousarray(cloud, dtype=np.float32).view(np.dtype((np.void, 20))).ravel()


def rebuild_fused_cloud(export_dir: str, sd_rows: dict, cs: dict, ep: dict,
                        cfg: dict) -> tuple[np.ndarray, list[tuple[str, np.ndarray]]]:
    """Stage 1's fused single sweep BEFORE its ground/range/height filter, from
    the released blobs, plus the raw file row each point came from.

    Every step is ingest.py's own (`ingest_keyframe`, ~line 1415): LIDAR_TOP
    through its calibrated_sensor into ego, then each STEREO_CHANNELS blob the
    sample ships, in that dict's order, `thin_stereo` first, a sensor-frame
    channel through its own calibrated_sensor and a `global_identity` one
    (ZED_WORLD) through the inverse of THIS sample's LIDAR_TOP ego_pose. The
    float64 result is cast to float32 exactly once, as `write_pcd_bin` does, so
    the bytes are the bytes Stage 1 wrote.

    Returns (cloud, [(channel, raw_row_index, n_rows_in_raw_file), ...]) with
    the blocks in cloud order, so a fused row resolves to (channel, row of that
    channel's raw file) and the bin can be sized against the FILE rather than
    against `sample_data.num_points`, which is a table's claim about it.
    """
    rings, stride = cfg["stereo_rings"], int(cfg["stereo_stride"])

    def thin(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
        """`thin_stereo`'s own keep-decision, and the raw rows it kept.

        The stride is a property of the ring column alone, so running the SAME
        function over a probe array that carries the row number in column 0
        cannot disagree with running it over the cloud (stride 1: both no-ops).
        """
        probe = np.zeros_like(raw, dtype=np.float64)
        probe[:, 0] = np.arange(raw.shape[0])
        probe[:, 4] = raw[:, 4]
        kept_probe, _ = thin_stereo(probe, rings, stride)
        thinned, _ = thin_stereo(raw, rings, stride)
        return thinned, kept_probe[:, 0].astype(np.int64), int(raw.shape[0])

    lidar = sd_rows[LIDAR_CHANNEL]
    raw, raw_index, n_raw = thin(read_pcd_bin(os.path.join(export_dir, lidar["filename"])))
    t_ego_lidar = Transform.from_nuscenes(cs[lidar["calibrated_sensor_token"]],
                                          source_frame=LIDAR, parent_frame=EGO)
    blocks = [np.column_stack([apply_transform(t_ego_lidar.matrix(), raw[:, :3].astype(np.float64)),
                               raw[:, 3:5].astype(np.float64)])]
    provenance = [(LIDAR_CHANNEL, raw_index, n_raw)]

    if cfg.get("fuse_stereo", True):
        t_global_to_ego = Transform.from_nuscenes(
            ep[lidar["ego_pose_token"]], source_frame=EGO, parent_frame=NUSCENES_GLOBAL,
        ).inverse_matrix()
        for channel, how in STEREO_CHANNELS.items():
            record = sd_rows.get(channel)
            if record is None:
                continue
            raw, raw_index, n_raw = thin(read_pcd_bin(os.path.join(export_dir,
                                                                  record["filename"])))
            t_sensor_to_ego = None
            if how["frame"] == "sensor":
                t_sensor_to_ego = Transform.from_nuscenes(
                    cs[record["calibrated_sensor_token"]], source_frame=LIDAR, parent_frame=EGO,
                ).matrix()
            blocks.append(stereo_block_to_ego(
                raw, frame=how["frame"], ring=how["ring"], t_sensor_to_ego=t_sensor_to_ego,
                t_global_to_ego=t_global_to_ego,
                z_correction_m=cfg.get("stereo_z_correction_m") or {},
                pitch_correction=cfg.get("stereo_pitch_correction") or {},
            ))
            provenance.append((channel, raw_index, n_raw))
    return np.vstack(blocks).astype(np.float32), provenance


def match_kept_rows(kept: np.ndarray, fused: np.ndarray) -> tuple[np.ndarray, int]:
    """Row of `fused` for every row of `kept`, -1 where none, + ambiguous count.

    Stage 1's filter is a boolean mask that PRESERVES ORDER, and both arrays are
    the same float64 arithmetic cast to float32 once, so a kept row is byte-
    identical to its fused row. Exact equality on the whole 20-byte record is
    therefore available and is what is used — no tolerance to tune, no ring
    tie-break, and a mismatch is a loud -1 rather than a plausible neighbour.
    A row whose bytes occur more than once in `fused` resolves to the FIRST
    occurrence; those are coincident points with the same ring, so the label
    lands on the same place in space either way, and the count is reported
    (`points_ambiguous`) rather than hidden.
    """
    fused_keys, kept_keys = row_keys(fused), row_keys(kept)
    order = np.argsort(fused_keys, kind="stable")
    sorted_keys = fused_keys[order]
    first = np.searchsorted(sorted_keys, kept_keys, side="left")
    last = np.searchsorted(sorted_keys, kept_keys, side="right")
    found = first < last
    position = np.where(found, order[np.minimum(first, order.size - 1)], -1)
    return position, int((last - first)[found].sum() - int(found.sum()))


# ---------------------------------------------------------------------------
# One keyframe
# ---------------------------------------------------------------------------


def label_keyframe(lift: dict, kf: dict, ctx: dict) -> dict:
    """Both bins for one keyframe, or a refusal reason and no labels.

    The refusal is per keyframe on purpose: one keyframe whose reconstruction
    does not line up must not take the other 1290 with it, and must not be
    labelled on a guess either. The caller decides whether there are too many.
    """
    sd_rows = ctx["sample_data"][lift["keyframe_token"]]
    kept = read_pcd_bin(kf["single_sweep_cloud"]["path"])
    if kept.shape[0] != int(lift["n_points_cloud"]):
        return {"refused": f"kept cloud has {kept.shape[0]} rows, lift.jsonl says "
                           f"n_points_cloud={lift['n_points_cloud']}"}
    fused, provenance = rebuild_fused_cloud(ctx["export_dir"], sd_rows, ctx["calibrated_sensor"],
                                            ctx["ego_pose"], ctx["stage1_config"])
    mismatch = [f"{channel}: blob has {n_raw} points, sample_data says "
                f"{sd_rows[channel]['num_points']}"
                for channel, _, n_raw in provenance
                if int(sd_rows[channel].get("num_points", n_raw)) != n_raw]
    if mismatch:
        return {"refused": "; ".join(mismatch)}
    position, ambiguous = match_kept_rows(kept, fused)

    with np.load(os.path.join(ctx["stage5_dir"], lift["points_path"])) as z:
        point_index = np.asarray(z["point_index"], dtype=np.int64)
        instance_id = np.asarray(z["instance_id"], dtype=np.int64)
    if point_index.size and (point_index.min() < 0 or point_index.max() >= kept.shape[0]):
        return {"refused": f"point_index out of range [0, {kept.shape[0]})"}
    painted = position[point_index]
    if int((painted < 0).sum()):
        return {"refused": f"{int((painted < 0).sum())} of {point_index.size} painted points did "
                           "not match the rebuilt fused cloud"}

    labels_of_instance = ctx["labels_of_instance"](lift)
    fused_labels = np.zeros(fused.shape[0], dtype=np.uint8)
    fused_labels[painted] = labels_of_instance[instance_id]

    bins, counts, start = {}, collections.Counter(), 0
    for channel, raw_index, n_raw in provenance:
        block = fused_labels[start:start + raw_index.size]
        start += raw_index.size
        # Sized against the FILE. `sample_data.num_points` is the table's claim
        # about the blob; a bin that trusts it and is wrong is a silent shear.
        labels = np.zeros(n_raw, dtype=np.uint8)
        labels[raw_index] = block
        bins[channel] = labels
        for label, n in zip(*np.unique(labels[labels > 0], return_counts=True)):
            counts[(channel, int(label))] += int(n)
    return {"refused": None, "bins": bins, "counts": counts, "ambiguous": ambiguous,
            "n_points_fused": int(fused.shape[0]),
            "n_points_painted": int(point_index.size),
            "n_points_unmapped": int((labels_of_instance[instance_id] == 0).sum())}


# ---------------------------------------------------------------------------
# category.json
# ---------------------------------------------------------------------------


def indexed_categories(rows: list[dict]) -> list[dict]:
    """The release's own rows, order and content untouched, each gaining
    `index` 1..N, with a `noise` row prepended at 0.

    The devkit asserts an `index` on every category row the moment a
    lidarseg.json sits beside it, and its stats APIs assume index == list
    position, so the table must be contiguous from 0. Prepending is the only
    edit: the tokens stay exactly as `sample_annotation.json` and
    `instance.json` reference them.

    Idempotent — a re-run drops the noise row it wrote last time and re-indexes
    the rest to the same numbers, so the file converges after one pass.
    """
    kept = [r for r in rows if r.get("name") != NOISE_NAME]
    noise = {
        "token": hashlib.sha256(b"dhakascenes/lidarseg/category/noise").hexdigest()[:32],
        "name": NOISE_NAME,
        "description": NOISE_DESCRIPTION,
        "index": 0,
    }
    return [noise] + [{**r, "index": i} for i, r in enumerate(kept, start=1)]


def devkit_colormap_gap(categories: list[dict]) -> list[str]:
    """Category names the INSTALLED devkit's colormap does not have.

    `nuscenes.py:110` runs the moment a `lidarseg.json` sits beside
    `category.json`:

        self.colormap = dict({c['name']: self.colormap[c['name']]
                              for c in sorted(self.category, key=...)})

    so a bare `NuScenes(version, dataroot)` raises `KeyError: 'car'` on a
    release with its own taxonomy. docs/EXPORT_STATUS_AND_DECISIONS.md §3.1
    measured this and is why `road/` is a separate root with the canonical 32
    names. This layer cannot take that way out — it must sit beside the box
    tables, whose category tokens `sample_annotation.json` references — so the
    gap is MEASURED and shouted instead of discovered by the consumer. The
    layer itself is unaffected: the bins, `lidarseg.json` and the `index`
    column are exactly what the format specifies, and a two-line colormap shim
    (see the delivery note) loads it.
    """
    try:
        from nuscenes.utils.color_map import get_colormap
    except ImportError:      # no devkit here; nothing to compare against
        return []
    known = get_colormap()
    return [r["name"] for r in categories if r["name"] not in known]


# ---------------------------------------------------------------------------
# The export
# ---------------------------------------------------------------------------


def export_lidarseg(export_dir: str, stage1_dir: str, stage5_dir: str, scenes: list[str],
                    *, workers: int = 4, upstream_marker: str | None = None) -> dict:
    """The whole export. Raises ExportRefusal with nothing written."""
    t0 = time.time()
    tables = version_dir(export_dir)
    version = os.path.basename(tables)
    # Before anything is read, let alone written: this exporter never writes
    # that name, so one being there means somebody applied the devkit recipe (or
    # an older version of this script ran). Either way a bare NuScenes() on this
    # root is currently broken, and silently writing beside it would leave the
    # break in place and unexplained.
    if os.path.exists(os.path.join(tables, DEVKIT_TABLE_NAME)):
        raise ExportRefusal(
            f"{os.path.join(tables, DEVKIT_TABLE_NAME)} exists. This exporter writes "
            f"{TABLE_NAME} precisely so the devkit does NOT auto-detect the layer — with a "
            f"{DEVKIT_TABLE_NAME} present, a bare NuScenes(version, dataroot) raises "
            "KeyError on this release's class names. If you made it by hand to use the devkit's "
            f"lidarseg APIs, delete the copy and re-run; {TABLE_NAME} is the source of truth")
    sample_data = read_json(os.path.join(tables, "sample_data.json"), []) or []
    by_sample: dict[str, dict] = {}
    for row in sample_data:
        by_sample.setdefault(row["sample_token"], {})[
            os.path.basename(os.path.dirname(row["filename"]))] = row
    categories = indexed_categories(read_json(os.path.join(tables, "category.json"), []) or [])
    if len(categories) < 2:
        raise ExportRefusal(f"{tables}/category.json has no classes; there is nothing to label")
    label_of_class = {r["name"]: r["index"] for r in categories}
    phrase_map, mapper_source = release_phrase_map(
        read_json(os.path.join(export_dir, "release_meta.json"), {}) or {})
    unmapped: collections.Counter = collections.Counter()

    def labels_of_instance(lift: dict) -> np.ndarray:
        """Label per `instances` row of this keyframe; 0 where the detector's
        phrase has no class in this release (counted, never guessed)."""
        out = np.zeros(len(lift["instances"]), dtype=np.uint8)
        for i, instance in enumerate(lift["instances"]):
            phrase = instance["class_name"]
            label = label_of_class.get(phrase_map.get(phrase))
            if label is None:
                unmapped[phrase] += 1
                continue
            out[i] = label
        return out

    ctx = {
        "export_dir": export_dir,
        "stage5_dir": stage5_dir,
        "sample_data": by_sample,
        "calibrated_sensor": {r["token"]: r for r in
                              read_json(os.path.join(tables, "calibrated_sensor.json"), []) or []},
        "ego_pose": {r["token"]: r for r in
                     read_json(os.path.join(tables, "ego_pose.json"), []) or []},
        "labels_of_instance": labels_of_instance,
    }

    # --- every keyframe, before the first byte lands -------------------------
    written: dict[str, np.ndarray] = {}
    channels: set[str] = set()
    counts: collections.Counter = collections.Counter()
    totals = collections.Counter()
    refused: list[str] = []
    per_scene = {}
    for scene in scenes:
        diagnostics = read_json(os.path.join(stage1_dir, "scenes", scene,
                                             "filter_diagnostics.json"))
        if not diagnostics or "config" not in diagnostics:
            raise ExportRefusal(
                f"{stage1_dir}/scenes/{scene}/filter_diagnostics.json missing its `config`; the "
                "fused cloud cannot be rebuilt without the stereo thinning and corrections the "
                "run actually used")
        ctx["stage1_config"] = diagnostics["config"]
        keyframes = {r["keyframe_token"]: r for r in
                     read_jsonl(os.path.join(stage1_dir, "scenes", scene, "keyframes.jsonl"))}
        lift_path = os.path.join(stage5_dir, "scenes", scene, "lift.jsonl")
        if not os.path.isfile(lift_path):
            raise ExportRefusal(f"{lift_path} not found: scene {scene!r} has no Stage 5 painting")
        # A lift row whose keyframe the release does not ship is scoped out, not
        # broken: --scenes narrows the release, and a bin for a sample it has no
        # sample_data row for could not be addressed anyway.
        all_rows = list(read_jsonl(lift_path))
        rows = [r for r in all_rows if r["keyframe_token"] in by_sample]
        skipped = len(all_rows) - len(rows)
        missing = [r["keyframe_token"] for r in rows if r["keyframe_token"] not in keyframes]
        if missing:
            raise ExportRefusal(
                f"{len(missing)} Stage 5 keyframes of {scene!r} have no Stage 1 keyframes.jsonl "
                f"row (first: {missing[0]}); the fused cloud they were painted on is unknown")

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            results = list(pool.map(
                lambda r: label_keyframe(r, keyframes[r["keyframe_token"]], ctx), rows))
        scene_refused = 0
        for lift, result in zip(rows, results):
            if result["refused"]:
                scene_refused += 1
                refused.append(f"{lift['keyframe_token']}: {result['refused']}")
                continue
            for channel, labels in result["bins"].items():
                written[ctx["sample_data"][lift["keyframe_token"]][channel]["token"]] = labels
                channels.add(channel)
            counts.update(result["counts"])
            for key in ("n_points_fused", "n_points_painted", "n_points_unmapped", "ambiguous"):
                totals[key] += result[key]
        if rows and scene_refused > MAX_REFUSED_FRACTION * len(rows):
            raise ExportRefusal(
                f"{scene}: {scene_refused} of {len(rows)} keyframes refused "
                f"(> {MAX_REFUSED_FRACTION:.0%}), first: {refused[0]}. A positional lidarseg bin "
                "written on a reconstruction this unreliable would shear silently")
        per_scene[scene] = {"keyframes": len(rows), "refused": scene_refused,
                            "keyframes_not_in_release": skipped,
                            "stage1_config": {k: diagnostics["config"].get(k) for k in
                                              ("stereo_rings", "stereo_stride", "fuse_stereo",
                                               "stereo_z_correction_m", "stereo_pitch_correction")}}

    # --- write: bins, then the tables, then the note -------------------------
    lidarseg_dir = os.path.join(export_dir, "lidarseg", version)
    os.makedirs(lidarseg_dir, exist_ok=True)
    for token, labels in written.items():
        path = os.path.join(lidarseg_dir, f"{token}_lidarseg.bin")
        labels.tofile(path + ".tmp")
        os.replace(path + ".tmp", path)

    # One record per .bin FILE in the directory — the equality the devkit
    # asserts the moment this table is renamed to lidarseg.json. It is built
    # from what is on disk, not from what this run wrote: a second run over one
    # scene of a two-scene release must not orphan the other scene's bins.
    on_disk = sorted(n for n in os.listdir(lidarseg_dir) if n.endswith("_lidarseg.bin"))
    known = {r["token"] for r in sample_data}
    index_rows = []
    for name in on_disk:
        token = name[: -len("_lidarseg.bin")]
        if token not in known:
            raise ExportRefusal(
                f"{os.path.join(lidarseg_dir, name)} names sample_data token {token!r}, which "
                "this release does not have; move the stale bin aside and re-run")
        index_rows.append({"token": token, "sample_data_token": token,
                           "filename": f"lidarseg/{version}/{name}"})

    # The mode every table this release wrote carries; see write_table.
    reference = os.path.join(tables, "sample_data.json")
    backup = os.path.join(tables, CATEGORY_BACKUP)
    if not os.path.exists(backup):
        shutil.copyfile(os.path.join(tables, "category.json"), backup)
    write_table(os.path.join(tables, "category.json"), categories, reference)
    write_table(os.path.join(tables, TABLE_NAME), index_rows, reference)

    gap = devkit_colormap_gap(categories)
    if gap:
        print(f"note: the index table is {version}/{TABLE_NAME}, NOT {DEVKIT_TABLE_NAME}, so a "
              f"bare NuScenes('{version}', dataroot) keeps working — {len(gap)} of this "
              f"release's class names ({', '.join(gap[:3])}...) are absent from the devkit's "
              f"closed colormap and it would KeyError on them. To use the devkit's lidarseg "
              f"APIs: copy {TABLE_NAME} to {DEVKIT_TABLE_NAME} AND install the get_colormap "
              "shim — both, or it crashes. Recipe in lidarseg_meta.json 'devkit_recipe' and in "
              "the delivery note.", file=sys.stderr, flush=True)
    by_class: dict[str, dict[str, int]] = {}
    name_of = {r["index"]: r["name"] for r in categories}
    for (channel, label), n in sorted(counts.items()):
        by_class.setdefault(name_of[label], {})[channel] = n
    meta = {
        "description": "DhakaScenes lidarseg — per-point OBJECT class labels from Stage 5's "
                       "painted points, positional over the RELEASED raw blobs",
        "exporter": "scripts/export_lidarseg.py",
        "exporter_git_sha": git_sha(),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "version": version,
        "scenes": per_scene,
        "sources": {"stage1_dir": stage1_dir, "stage5_dir": stage5_dir,
                    "category_mapping_source": mapper_source,
                    "stage5_marker": upstream_marker},
        # What a bin is positional over. The whole correctness of a positional
        # .bin is this string being true of the file beside it.
        "basis": {channel: f"raw_{channel.lower()}_file_order"
                  for channel in sorted(channels or {LIDAR_CHANNEL})},
        "counts": {
            "keyframes": sum(s["keyframes"] for s in per_scene.values()),
            "keyframes_refused": len(refused),
            "keyframes_not_in_release": sum(s["keyframes_not_in_release"]
                                            for s in per_scene.values()),
            "bins_written": len(written),
            "bins_in_table": len(index_rows),
            "points_fused_total": totals["n_points_fused"],
            "points_painted": totals["n_points_painted"],
            "points_labelled": sum(counts.values()),
            "points_unmapped_phrase": totals["n_points_unmapped"],
            "points_ambiguous": totals["ambiguous"],
        },
        "table": f"{version}/{TABLE_NAME}",
        "read_without_the_devkit": (
            "labels = numpy.fromfile('lidarseg/<version>/<sample_data_token>_lidarseg.bin', "
            "dtype=numpy.uint8) — row i is row i of the blob that sample_data token names "
            f"(samples/<CHANNEL>/NNNNNN.pcd.bin, float32 (N,5)). {version}/{TABLE_NAME} maps a "
            f"sample_data token to its bin; {version}/category.json maps a label to a class "
            "name through its `index` field (0 = unlabelled). No devkit needed."
        ),
        "devkit_colormap_gap": gap,
        "devkit_recipe": (
            f"The index table is deliberately NOT named {DEVKIT_TABLE_NAME}: NuScenes.__init__ "
            f"auto-detects that name, and the moment it does it merges its closed 32-name "
            f"colormap against category.json BY NAME and raises KeyError on {len(gap)} of this "
            f"release's classes ({', '.join(gap[:4])}…). So a bare NuScenes(version, dataroot) "
            "on this root works exactly as it did before this layer existed. To use the "
            "devkit's lidarseg APIs (get_sample_lidarseg_stats, "
            "render_sample_data(show_lidarseg=True)) do BOTH of these, in a COPY of the root "
            "or a throwaway table dir — one without the other crashes:\n"
            f"  1. cp {version}/{TABLE_NAME} {version}/{DEVKIT_TABLE_NAME}\n"
            "  2. import json, nuscenes.nuscenes as nu\n"
            "     _orig = nu.get_colormap\n"
            "     nu.get_colormap = lambda: {**_orig(), **{c['name']: (150, 150, 150)\n"
            "         for c in json.load(open(f'{dataroot}/{version}/category.json'))}}\n"
            "     nusc = nu.NuScenes(version, dataroot=dataroot)\n"
            f"Leaving a {DEVKIT_TABLE_NAME} in the delivered root breaks the bare load for "
            "everyone else, so this exporter REFUSES to run while one is present."
        ) if gap else "NuScenes(version, dataroot) loads this root as-is",
        "points_by_class_and_channel": by_class,
        "unmapped_phrases": dict(sorted(unmapped.items())),
        "refused_keyframes": refused,
        "caveats": list(CAVEATS),
        "elapsed_s": round(time.time() - t0, 1),
    }
    write_table(os.path.join(export_dir, "lidarseg", "lidarseg_meta.json"), meta, reference)

    note_path = os.path.join(export_dir, "DELIVERY_NOTE.md")
    if os.path.isfile(note_path):
        update_note(note_path, note_block(meta, version), NOTE_HEADING)
    return meta


def note_block(meta: dict, version: str) -> str:
    counts, by_class = meta["counts"], meta["points_by_class_and_channel"]
    top = ", ".join(f"{name} {sum(v.values()):,}" for name, v in
                    sorted(by_class.items(), key=lambda kv: -sum(kv[1].values()))[:6])
    return "\n".join([
        NOTE_HEADING, "",
        f"- `lidarseg/{version}/<sample_data_token>_lidarseg.bin`: one `uint8` label PER POINT, "
        f"in the same order as the points in the blob that `sample_data` token names. **No "
        f"devkit needed**: `labels = numpy.fromfile(path, dtype=numpy.uint8)`, and row *i* of "
        f"the bin is row *i* of `samples/<CHANNEL>/NNNNNN.pcd.bin` (float32 `(N, 5)`). "
        f"`{version}/{TABLE_NAME}` binds each bin to its sample_data token and "
        f"`{version}/category.json` turns a label into a class name through its `index` field "
        f"(1..18, the release's own classes in their own order).",
        "- **0 means UNLABELLED, not noise.** No object mask painted that point: road, "
        "buildings, vegetation, anything the detector has no word for, anything it missed. "
        "The category row is named `noise` because the devkit reserves index 0 for that name "
        "and every colour map assumes it. The ROAD is not labelled in this release.",
        "- **There are TWO bins per keyframe**, one for `LIDAR_TOP` and one for `ZED_WORLD`. "
        "nuScenes-lidarseg ships only the first; most object points in this capture are "
        "stereo, so shipping only the lidar bin would carry a small minority of the "
        "segmentation. The devkit's own lidarseg APIs read the LIDAR_TOP bin and ignore the "
        "other; read the stereo bin directly.",
        "- These are the DETECTOR's claims (Stage 3m phrase -> Stage 4 SAM mask -> Stage 5 "
        "painting), not reviewed labels, and they are NOT gated by the Annotation rule above "
        "— that rule admits 3D cuboids. A point may carry a class whose cuboid never shipped.",
        "- Full wording, per-class counts and provenance: `lidarseg/lidarseg_meta.json`.",
        *([
            f"- **The table is `{version}/{TABLE_NAME}`, not `{DEVKIT_TABLE_NAME}` — on "
            "purpose, and a bare `NuScenes(version, dataroot)` on this root works exactly as it "
            "did before this layer existed.** `NuScenes.__init__` auto-detects the name "
            f"`{DEVKIT_TABLE_NAME}`, and the moment it does it merges its hard-coded 32-name "
            "colormap against `category.json` BY NAME and raises `KeyError: 'car'` — this "
            "release's classes are its own. The schema and rows are the standard ones; only the "
            "file name differs.",
            "- **To use the devkit's lidarseg APIs** (`get_sample_lidarseg_stats`, "
            "`render_sample_data(show_lidarseg=True)`), do BOTH of the following, in a COPY of "
            "the table directory — one without the other crashes, and leaving a "
            f"`{DEVKIT_TABLE_NAME}` in the delivered root breaks the bare load for everyone "
            "else (this exporter refuses to run while one is present):",
            "  ```python",
            f"  # 1.  cp {version}/{TABLE_NAME}  {version}/{DEVKIT_TABLE_NAME}",
            "  # 2.  give the devkit colours for our class names, BEFORE constructing it:",
            "  import json, nuscenes.nuscenes as nu",
            "  _orig = nu.get_colormap",
            "  nu.get_colormap = lambda: {**_orig(), **{c['name']: (150, 150, 150)",
            "      for c in json.load(open(f'{dataroot}/{version}/category.json'))}}",
            "  nusc = nu.NuScenes(version, dataroot=dataroot)",
            "  ```",
            f"  `{CATEGORY_BACKUP}` beside `category.json` is the pre-lidarseg table, if this "
            "layer needs to be undone entirely.",
        ] if meta["devkit_colormap_gap"] else []),
        "",
        f"- keyframes: {counts['keyframes']:,} ({counts['keyframes_refused']} refused)",
        f"- bins: {counts['bins_in_table']:,} over {counts['points_fused_total']:,} points",
        f"- labelled: {counts['points_labelled']:,} points "
        f"({counts['points_unmapped_phrase']:,} painted by a phrase this release has no class "
        f"for, left 0)",
        f"- classes: {top}",
        f"- written by: {meta['exporter']} @ {meta['exporter_git_sha']}",
    ])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG",
                                                      "configs/paths.yaml"))
    ap.add_argument("--export-dir", required=True,
                    help="the release's boxes/ directory; lidarseg/ is written inside it")
    ap.add_argument("--scene", default=None, help="the one scene to export")
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="the plural run_stages.sh passes; default is every scene with a "
                         "lift.jsonl, which for a one-chunk release is the one scene")
    ap.add_argument("--stage1-dir", default=None, help="default <work_root>/stage1_ingestion")
    ap.add_argument("--stage5-dir", default=None, help="default <work_root>/stage5_lift")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)

    try:
        work = load_paths(a.paths).work_root
        stage1_dir = a.stage1_dir or os.path.join(work, "stage1_ingestion")
        stage5_dir = a.stage5_dir or os.path.join(work, "stage5_lift")
        # accept_degraded is unconditional: Stage 5 is degraded on every chunk of
        # this batch for `ego_motion_between_capture_times_absent`, which says
        # nothing about the painting this layer reads. The marker state is
        # recorded in lidarseg_meta.json rather than being waved through in
        # silence — a flag nobody ever omits is a flag, not a decision.
        _manifest, marker = require_upstream(
            stage5_dir, stage_name="stage5_lift", module_hint="pipeline.stage5_lift.lift",
            accept_degraded=True)
        export_dir = os.path.abspath(a.export_dir)
        scenes = a.scenes or ([a.scene] if a.scene else sorted(
            n for n in os.listdir(os.path.join(stage5_dir, "scenes"))
            if os.path.isfile(os.path.join(stage5_dir, "scenes", n, "lift.jsonl"))))
        if not scenes:
            raise ExportRefusal(f"{stage5_dir}: no scene has a lift.jsonl")
        meta = export_lidarseg(export_dir, stage1_dir, stage5_dir, scenes,
                               workers=a.workers, upstream_marker=marker.state)
    except (ExportRefusal, UpstreamRefusal) as exc:
        print(f"export_lidarseg: {exc}", file=sys.stderr)
        return 2

    counts = meta["counts"]
    print(f"{', '.join(scenes)}: {counts['bins_in_table']} bins, "
          f"{counts['points_labelled']:,} of {counts['points_fused_total']:,} points labelled "
          f"over {counts['keyframes']} keyframes ({counts['keyframes_refused']} refused) "
          f"-> {os.path.join(export_dir, 'lidarseg')}  ({meta['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
