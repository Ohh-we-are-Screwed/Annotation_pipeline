#!/usr/bin/env python3
"""Release writer: I-4/I-5 `prelabels.jsonl` -> nuScenes v1.0 annotation tables.

Zami's terminal output is one `AnnotationRecord` per line (ego frame, phrase
categories, `track_id`, `num_lidar_pts`; `pipeline/common/schemas.py`, README
"Output contract"). dbench (`/home/mt/dataset_benchmark`) reads nuScenes tables:
global-frame boxes, an `instance` row per identity, `category` resolved through
`instance`, a `visibility` token, both point-count fields, and bidirectional
`prev`/`next` chains (`dbench/ingest/contract.md` §1, §6). This script is the
bridge (audit Z-11 / dbench B2, B3).

Input: an existing nuScenes-format dataroot whose `sample`, `sample_data`,
`ego_pose`, `calibrated_sensor`, `sensor`, `scene`, `log` tables are populated
(the sensor_suit exporter writes exactly that, with the five annotation tables
empty). Output: a NEW root (`--out`) with every table copied through, blobs
symlinked (or copied), and `sample_annotation`, `instance`, `category`,
`attribute`, `visibility` written from the records. The source root is never
modified.

Geometry
  box_global = T_ego->global(t_lidar) . box_ego, using the `ego_pose` of the
  keyframe's LIDAR_TOP `sample_data` — the pose dbench's `ego_pose_of_frame`
  reads back, so the round trip is exact. Quaternions are [w, x, y, z]; size is
  [w, l, h] (nuScenes order, same as `size_wlh_m`). Verified at runtime against
  nuscenes-devkit's `Box.rotate/translate` when the devkit is importable.

Identity
  Stage 7 tracks are re-stitched offline across short gaps
  (`pipeline/release/stitch.py`, gates from `configs/release.yaml`), and a
  keyframe missing inside a joined chain is interpolated: an extra row flagged
  `dhakascenes_interpolated` whose point count is recounted against that
  keyframe's cloud. `instance_token` = hash(scene_token, chain_id); the
  pre-stitch Stage 7 id survives as `dhakascenes_track_id_pre_stitch` and the
  record -> chain map is written to `<out>/stitch_map.json`. `--no-stitch`
  restores the old identity (Stage 7 `track_id`, untracked records singletons).
  Annotations of one instance are chained by sample timestamp; `instance`
  carries `first/last_annotation_token` and `nbr_annotations`.

Tiers
  `--tiers auto_accept` (the default) ships auto-accepted pipeline rows and
  every human row; flagged/rejected rows, and rows a human pass superseded, go
  to `<out>/sample_annotation_excluded.json` with a
  `dhakascenes_excluded_reason` — nothing is silently dropped. `--tiers all`
  admits every row and reproduces the pre-2026-09-07 output.

Categories
  Phrase -> dbench 18-class name through `configs/release_category_map.yaml`.
  An unmapped string aborts the export and lists every offender.

Visibility
  Fraction of the 8 box corners that land inside at least one camera image of
  the same sample, projected through the four-hop chain in
  `pipeline/common/conventions.py`. That is a field-of-view proxy, not an
  occlusion estimate; `release_meta.json` says so. If any camera lacks
  intrinsics or image size the whole export falls back to token "4" (80-100 %)
  and `visibility_basis` records the fallback.

Attributes
  Derived from the chain's own global velocity (`pipeline/release/attributes.py`),
  not from a record field: speed above `attributes.moving_speed_threshold_mps`
  -> `<group>.moving`, below -> `vehicle.stopped` / `pedestrian.standing`
  (cycles are always `cycle.with_rider`). parked / sitting_lying_down /
  without_rider are never emitted — the pipeline cannot tell them apart. An
  `attribute` a human set on an I-5 row wins. An undefined velocity (singleton
  chain, or neighbours further apart than `attributes.max_time_diff_s`) -> [].
  `dhakascenes_attribute_basis` records which of the three it was.

Double annotation
  `sample.json` gains `dbench_double_annotated`. The stratified selection
  (density x illumination cell, `pipeline/release/{strata,double}.py`) is
  written once to `<out>/double_annotation.json` and reused by every later
  export unless `--reselect-double`.

Delivery note
  `<out>/DELIVERY_NOTE.md` (`pipeline/release/note.py`, spec §8): the per-chunk
  text block the benchmark's checklist asks for — annotation rule, range,
  tiers, classes, identity, attributes, uncertainty, double annotation,
  anonymisation, extra layers — rendered from `release_meta.json` and the Stage
  9 manifest. `--no-note` skips it.

Point counts
  `num_lidar_pts` copied verbatim; its basis is the record's
  `num_lidar_pts_basis` (single-sweep, ground-filtered, PRE-inflation — NOT
  nuScenes' single-sweep-with-ground count) and is written both per annotation
  and in `release_meta.json`. `num_radar_pts` = 0: the rig has no radar.

    python scripts/export_release.py --prelabels <jsonl|stage9 dir> \
        --dataroot <nusc root> --version v1.0-dhaka --out <new root> \
        [--mapper configs/release_category_map.yaml] \
        [--release-config configs/release.yaml] [--tiers auto_accept|all] \
        [--no-stitch] [--no-attributes] [--double-fraction F] [--reselect-double] \
        [--human <work_root>/stage10_human] [--cvat-export-3d-dir <work>/cvat_export_3d] \
        [--overwrite-tables] [--human-verified-scenes scenes.txt] [--blobs symlink|copy] \
        [--no-note] [--stage-tree <work_root>] [--chunk-name NAME] [--import-manifest F]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from typing import Iterable

import numpy as np
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from pipeline.common.conventions import (  # noqa: E402
    CAMERA,
    EGO,
    NUSCENES_GLOBAL,
    Transform,
    project_lidar_to_image,
    transform_matrix,
)
from pipeline.release.attributes import assign_attributes  # noqa: E402
from pipeline.release.config import load_release_config  # noqa: E402
from pipeline.release.double import load_or_select  # noqa: E402
from pipeline.release.frames import CloudSource, SceneFrames, scene_frames_from_root  # noqa: E402
from pipeline.release.geometry import (  # noqa: E402
    box_corners_ego, box_ego_to_global, box_global_to_ego, make_token, normalise_quat, quat_multiply,
)
from pipeline.release.human import load_human, merge_human  # noqa: E402
from pipeline.release.note import write_note  # noqa: E402
from pipeline.release.stitch import stitch_scene  # noqa: E402
from pipeline.release.strata import compute_strata  # noqa: E402
from pipeline.release.tiers import ADMIT_AUTO, ADMIT_MODES, partition  # noqa: E402

EXPORTER_SPEC = "dhakascenes/export_release/v1"
SCHEMA_VERSION_EXPECTED = "dhakascenes-pilot/schemas/v1"
ANNOTATION_TABLES = ("sample_annotation", "instance", "category", "attribute", "visibility")
PASSTHROUGH_TABLES = (
    "log", "scene", "sample", "sample_data", "ego_pose", "calibrated_sensor", "sensor", "map",
)
HUMAN_SOURCES = ("human_verified", "human_created")
LIDAR_CHANNEL = "LIDAR_TOP"

DEFAULT_RELEASE_CONFIG = os.path.join(REPO_ROOT, "configs", "release.yaml")
TAXONOMY_DHAKA = os.path.join(REPO_ROOT, "configs", "taxonomy_pilot_dhaka.yaml")
# The sidecars live beside release_meta.json at <out>/, NOT inside
# <out>/<version>/: a nuScenes table directory holds nuScenes tables and nothing
# else, so a devkit or dbench consumer never has to know about them.
EXCLUDED_TABLE = "sample_annotation_excluded.json"
STITCH_MAP = "stitch_map.json"
DOUBLE_FILE = "double_annotation.json"
# A record's t_ns is expected to agree with its sample's timestamp to 1 ms; the
# sample table is the timeline nuScenes' box_velocity() reads, so it wins.
T_TOLERANCE_NS = 1_000_000

# nuScenes' own visibility table, tokens included, so a dbench consumer
# calibrated on nuScenes reads the same strings.
VISIBILITY_TABLE = [
    {"token": "1", "level": "v0-40", "description": "visibility of whole object is between 0 and 40%"},
    {"token": "2", "level": "v40-60", "description": "visibility of whole object is between 40 and 60%"},
    {"token": "3", "level": "v60-80", "description": "visibility of whole object is between 60 and 80%"},
    {"token": "4", "level": "v80-100", "description": "visibility of whole object is between 80 and 100%"},
]
VISIBILITY_FALLBACK_TOKEN = "4"

class ExportError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# hashes  (`make_token` lives in pipeline/release/geometry.py)
# ---------------------------------------------------------------------------


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# mapper
# ---------------------------------------------------------------------------


@dataclass
class CategoryMapper:
    path: str
    classes: list[str]
    mapping: dict[str, str]
    unresolved: dict[str, str]
    sha256: str

    @staticmethod
    def normalise(name: str) -> str:
        return re.sub(r"\s+", " ", str(name).strip().lower())

    @classmethod
    def load(cls, path: str) -> "CategoryMapper":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        classes = list(raw.get("classes") or [])
        if len(classes) != len(set(classes)):
            raise ExportError(f"{path}: duplicate entries in `classes`")
        mapping = {cls.normalise(k): str(v) for k, v in (raw.get("map") or {}).items()}
        unresolved = {cls.normalise(k): str(v) for k, v in (raw.get("unresolved") or {}).items()}
        bad = sorted(set(mapping.values()) - set(classes))
        if bad:
            raise ExportError(f"{path}: `map` targets not in `classes`: {bad}")
        both = sorted(set(mapping) & set(unresolved))
        if both:
            raise ExportError(f"{path}: keys in both `map` and `unresolved`: {both}")
        return cls(path=path, classes=classes, mapping=mapping, unresolved=unresolved,
                   sha256=sha256_of(path))

    def resolve(self, categories: Iterable[str]) -> dict[str, str]:
        """Map every distinct source string or raise listing all failures."""
        out: dict[str, str] = {}
        unmapped: list[str] = []
        blocked: list[str] = []
        for cat in sorted(set(categories)):
            key = self.normalise(cat)
            if key in self.mapping:
                out[cat] = self.mapping[key]
            elif key in self.unresolved:
                blocked.append(f"{cat!r}: {self.unresolved[key]}")
            else:
                unmapped.append(cat)
        if unmapped or blocked:
            lines = [f"category mapping failed ({self.path}):"]
            if unmapped:
                lines.append(f"  unmapped strings: {unmapped}")
            if blocked:
                lines.append("  unresolved (decision owed):")
                lines.extend(f"    {b}" for b in blocked)
            raise ExportError("\n".join(lines))
        return out


# ---------------------------------------------------------------------------
# source dataroot
# ---------------------------------------------------------------------------


class SourceRoot:
    def __init__(self, dataroot: str, version: str):
        self.dataroot = os.path.abspath(dataroot)
        self.version = version
        self.table_dir = os.path.join(self.dataroot, version)
        if not os.path.isdir(self.table_dir):
            raise ExportError(f"no table directory at {self.table_dir}")
        self.tables: dict[str, list[dict]] = {}
        for name in PASSTHROUGH_TABLES:
            p = os.path.join(self.table_dir, f"{name}.json")
            if not os.path.isfile(p):
                raise ExportError(f"required table missing: {p}")
            with open(p, "r", encoding="utf-8") as fh:
                self.tables[name] = json.load(fh)
        self.sample = {r["token"]: r for r in self.tables["sample"]}
        self.scene = {r["token"]: r for r in self.tables["scene"]}
        self.ego_pose = {r["token"]: r for r in self.tables["ego_pose"]}
        self.calib = {r["token"]: r for r in self.tables["calibrated_sensor"]}
        self.sensor = {r["token"]: r for r in self.tables["sensor"]}
        self.sd_by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
        for sd in self.tables["sample_data"]:
            if not sd.get("is_key_frame", False):
                continue
            channel = self.sensor[self.calib[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]
            self.sd_by_sample[sd["sample_token"]][channel] = sd

    def lidar_sd(self, sample_token: str) -> dict:
        chans = self.sd_by_sample.get(sample_token, {})
        if LIDAR_CHANNEL not in chans:
            raise ExportError(f"sample {sample_token} has no keyframe {LIDAR_CHANNEL} sample_data")
        return chans[LIDAR_CHANNEL]

    def lidar_ego_pose(self, sample_token: str) -> Transform:
        sd = self.lidar_sd(sample_token)
        return Transform.from_nuscenes(self.ego_pose[sd["ego_pose_token"]],
                                       source_frame=EGO, parent_frame=NUSCENES_GLOBAL)

    def cameras(self, sample_token: str) -> list[tuple[str, dict, dict, dict]]:
        """(channel, sample_data, calibrated_sensor, ego_pose) for each camera keyframe."""
        out = []
        for channel, sd in sorted(self.sd_by_sample.get(sample_token, {}).items()):
            calib = self.calib[sd["calibrated_sensor_token"]]
            if self.sensor[calib["sensor_token"]].get("modality") != "camera":
                continue
            out.append((channel, sd, calib, self.ego_pose[sd["ego_pose_token"]]))
        return out


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = ("token", "sample_token", "instance_token", "category", "frame",
                    "translation_m", "size_wlh_m", "rotation_wxyz", "num_lidar_pts")


def load_prelabels(spec: str) -> tuple[list[dict], list[str]]:
    """One jsonl file, or a Stage 9 out dir (`scenes/*/prelabels.jsonl`)."""
    if os.path.isdir(spec):
        files = sorted(glob.glob(os.path.join(spec, "scenes", "*", "prelabels.jsonl")))
        if not files:
            files = sorted(glob.glob(os.path.join(spec, "*.jsonl")))
        if not files:
            raise ExportError(f"{spec}: no prelabels.jsonl found under scenes/*/ or *.jsonl")
    else:
        files = [spec]
    records: list[dict] = []
    seen: set[str] = set()
    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                where = f"{path}:{lineno}"
                tag = rec.get("__contract__")
                if tag not in ("I-4", "I-5"):
                    raise ExportError(f"{where}: __contract__={tag!r}, expected I-4/I-5")
                if rec.get("__schema_version__") != SCHEMA_VERSION_EXPECTED:
                    raise ExportError(f"{where}: __schema_version__={rec.get('__schema_version__')!r}")
                missing = [k for k in _REQUIRED_FIELDS if k not in rec]
                if missing:
                    raise ExportError(f"{where}: missing fields {missing}")
                if rec["frame"] != EGO:
                    raise ExportError(f"{where}: frame={rec['frame']!r}, expected {EGO!r}")
                if rec["token"] in seen:
                    raise ExportError(f"{where}: duplicate record token {rec['token']!r}")
                seen.add(rec["token"])
                rec["__source_file__"] = path
                records.append(rec)
    return records, files


# ---------------------------------------------------------------------------
# geometry  (the helpers themselves live in pipeline/release/geometry.py)
# ---------------------------------------------------------------------------


def verify_with_devkit(translation_ego, size_wlh, rotation_ego, ego_pose: Transform,
                       t_global, q_global, tol: float = 1e-6) -> bool | None:
    """Cross-check one box against nuscenes-devkit Box.rotate/translate. None = devkit absent."""
    try:
        from nuscenes.utils.data_classes import Box
        from pyquaternion import Quaternion
    except Exception:  # noqa: BLE001
        return None
    box = Box(list(translation_ego), list(size_wlh), Quaternion(list(rotation_ego)))
    box.rotate(Quaternion(list(ego_pose.rotation_wxyz)))
    box.translate(np.asarray(ego_pose.translation_m))
    dq = min(np.linalg.norm(box.orientation.elements - np.asarray(q_global)),
             np.linalg.norm(box.orientation.elements + np.asarray(q_global)))
    return bool(np.linalg.norm(box.center - np.asarray(t_global)) < tol and dq < tol)


# ---------------------------------------------------------------------------
# visibility
# ---------------------------------------------------------------------------


def visibility_token_of_fraction(frac: float) -> str:
    if frac < 0.4:
        return "1"
    if frac < 0.6:
        return "2"
    if frac < 0.8:
        return "3"
    return "4"


class VisibilityEstimator:
    """Corner-in-any-image fraction through the four-hop chain. FOV proxy only."""

    def __init__(self, root: SourceRoot):
        self.root = root
        self.disabled_reason: str | None = None
        self._cache: dict[str, list] = {}

    def _cameras(self, sample_token: str) -> list | None:
        if sample_token in self._cache:
            return self._cache[sample_token]
        cams = []
        for channel, sd, calib, pose in self.root.cameras(sample_token):
            K = calib.get("camera_intrinsic") or []
            w, h = int(sd.get("width") or 0), int(sd.get("height") or 0)
            if len(K) != 3 or any(len(r) != 3 for r in K) or w <= 0 or h <= 0:
                self.disabled_reason = (f"{channel} of sample {sample_token} has no usable "
                                        f"camera_intrinsic / image size; visibility falls back "
                                        f"to token {VISIBILITY_FALLBACK_TOKEN}")
                self._cache[sample_token] = None
                return None
            cams.append((
                Transform.from_nuscenes(pose, source_frame=EGO, parent_frame=NUSCENES_GLOBAL),
                Transform.from_nuscenes(calib, source_frame=CAMERA, parent_frame=EGO),
                np.asarray(K, dtype=np.float64), (w, h),
            ))
        if not cams:
            self.disabled_reason = (f"sample {sample_token} has no camera keyframes; visibility "
                                    f"falls back to token {VISIBILITY_FALLBACK_TOKEN}")
            self._cache[sample_token] = None
            return None
        self._cache[sample_token] = cams
        return cams

    def fraction(self, sample_token: str, corners_ego: np.ndarray,
                 lidar_pose: Transform) -> float | None:
        cams = self._cameras(sample_token)
        if cams is None:
            return None
        visible = np.zeros(len(corners_ego), dtype=bool)
        for cam_pose, extrinsic, K, size in cams:
            proj = project_lidar_to_image(corners_ego, lidar_pose, cam_pose, extrinsic, K, size)
            visible[proj.source_index[proj.in_image]] = True
        return float(visible.mean())


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@dataclass
class ExportResult:
    out_root: str
    meta: dict
    tables: dict[str, list] = field(default_factory=dict)
    double: dict | None = None
    excluded: list[dict] = field(default_factory=list)


def read_scene_list(path: str | None) -> set[str]:
    if not path:
        return set()
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("scenes") or payload.get("human_verified_scenes") or []
        return {str(s) for s in payload}
    except json.JSONDecodeError:
        return {ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")}


def _legacy_identity(rows: list[dict], frames: SceneFrames) -> None:
    """--no-stitch: today's identity (Stage 7 track_id, else singleton), same keys stitch.py sets."""
    for r in rows:
        chain = str(r["track_id"]) if r.get("track_id") is not None else f"det:{r['token']}"
        r.update(instance_token=f"chain:{frames.scene_token}:{chain}", stitch_chain_id=chain,
                 stitch_track_id_pre=str(r["track_id"]) if r.get("track_id") is not None else None,
                 stitch_interpolated=False, stitch_tier_basis="gate")


def _producible_classes(mapper: CategoryMapper) -> set[str]:
    """The dbench classes the detection vocabulary can even propose (spec §2)."""
    with open(TAXONOMY_DHAKA, "r", encoding="utf-8") as fh:
        phrases = set((yaml.safe_load(fh) or {}).get("prompt_phrase", {}).values())
    return {mapper.mapping[CategoryMapper.normalise(p)] for p in phrases
            if CategoryMapper.normalise(p) in mapper.mapping}


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=REPO_ROOT)
    except OSError:
        return None
    return proc.stdout.strip() or None


def _link_or_copy(src: str, dst: str, mode: str) -> None:
    if os.path.lexists(dst):
        return
    if mode == "symlink":
        os.symlink(src, dst)
    elif mode == "copy":
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        raise ExportError(f"unknown blob mode {mode!r}")


def export_release(
    prelabels: str,
    dataroot: str,
    version: str,
    out: str,
    mapper_path: str,
    human_verified_scenes_path: str | None = None,
    blobs: str = "symlink",
    run_manifest_path: str | None = None,
    pipeline_version: str | None = None,
    *,
    release_config_path: str = DEFAULT_RELEASE_CONFIG,
    tiers: str = ADMIT_AUTO,
    stitch: bool = True,
    attributes: bool = True,
    double_fraction: float | None = None,
    human_dir: str | None = None,
    overwrite_tables: bool = False,
    cvat_export_3d_dir: str | None = None,
    reselect_double: bool = False,
) -> ExportResult:
    """stitch -> human merge -> tier filter -> chains -> attributes -> strata/double -> tables."""
    started = time.time()
    out = os.path.abspath(out)
    src = SourceRoot(dataroot, version)
    if out == src.dataroot:
        raise ExportError("--out must differ from --dataroot; the source root is never modified")
    if tiers not in ADMIT_MODES:
        raise ExportError(f"--tiers {tiers!r} is not one of {ADMIT_MODES}")
    out_tables = os.path.join(out, version)
    if os.path.isdir(out_tables) and os.listdir(out_tables) and not overwrite_tables:
        raise ExportError(f"{out}/{version} already exists and is not empty; refusing to overwrite "
                          "(--overwrite-tables rewrites the annotation tables in place)")
    if overwrite_tables and not os.path.isfile(os.path.join(out_tables, "sample.json")):
        raise ExportError(f"--overwrite-tables needs an existing export at {out_tables}")
    cfg = load_release_config(release_config_path)
    if double_fraction is not None:
        cfg = dc_replace(cfg, double=dc_replace(cfg.double, fraction=float(double_fraction)))

    mapper = CategoryMapper.load(mapper_path)
    records, source_files = load_prelabels(prelabels)
    if not records:
        raise ExportError("no records to export")
    unknown = sorted({r["sample_token"] for r in records if r["sample_token"] not in src.sample})
    if unknown:
        raise ExportError(f"{len(unknown)} sample_token(s) not in {src.table_dir}/sample.json: "
                          f"{unknown[:5]}{' ...' if len(unknown) > 5 else ''}")
    human_scenes = read_scene_list(human_verified_scenes_path)

    # --- scenes, keyframe order, record timestamps ---------------------------
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_scene[src.sample[r["sample_token"]]["scene_token"]].append(r)
    frames_of: dict[str, SceneFrames] = {st: scene_frames_from_root(src, st) for st in by_scene}
    n_t_ns_corrected = 0
    for scene_token, rows in by_scene.items():
        fr = frames_of[scene_token]
        for r in rows:
            t_sample = fr.timestamps_ns[fr.index[r["sample_token"]]]
            t_rec = r.get("t_ns")
            if t_rec is None or abs(int(t_rec) - t_sample) > T_TOLERANCE_NS:
                n_t_ns_corrected += 1   # the sample table is the timeline nuScenes' box_velocity()
            r["t_ns"] = t_sample        # reads; the disagreement is recorded in release_meta.json
    if n_t_ns_corrected:
        print(f"export_release: {n_t_ns_corrected} record(s) carried a t_ns that disagrees with "
              f"sample.timestamp by > 1 ms; the sample timestamp was used", file=sys.stderr)

    # --- 1. stitch (all tiers; the tier filter runs after the chains exist) ---
    stitch_meta: dict = {"enabled": bool(stitch), "per_scene": {}, "totals": {}}
    rows_all: list[dict] = []
    for scene_token in sorted(by_scene):
        fr = frames_of[scene_token]
        if stitch:
            clouds = CloudSource(src.dataroot, cvat_export_3d_dir, fr, src)
            try:
                rows, stats = stitch_scene(by_scene[scene_token], fr, clouds, cfg.stitch)
            except ValueError as exc:   # stitch.py speaks ValueError; the exporter speaks ExportError
                raise ExportError(str(exc)) from exc
            finally:
                clouds.close()
            stitch_meta["per_scene"][fr.scene_name] = vars(stats)
        else:
            rows = list(by_scene[scene_token])
            _legacy_identity(rows, fr)
        rows_all.extend(rows)
    if stitch:
        keys = ("n_records_in", "n_fragments", "n_chains", "n_interpolated", "n_interpolated_raw_basis",
                "n_interpolated_no_cloud")
        stitch_meta["totals"] = {k: sum(v[k] for v in stitch_meta["per_scene"].values()) for k in keys}
        stitch_meta["totals"]["joins_by_gap"] = {}
        for v in stitch_meta["per_scene"].values():
            for g, n in v["joins_by_gap"].items():
                stitch_meta["totals"]["joins_by_gap"][str(g)] = \
                    stitch_meta["totals"]["joins_by_gap"].get(str(g), 0) + n
        stitch_meta["config"] = vars(cfg.stitch)

    # --- 2. human merge ------------------------------------------------------
    superseded: dict[str, str] = {}
    human_meta: dict = {"enabled": bool(human_dir),
                        "dir": os.path.abspath(human_dir) if human_dir else None}
    if human_dir:
        human_rows, coverage = load_human(human_dir)
        for r in human_rows:
            if r["sample_token"] not in src.sample:
                raise ExportError(f"human record {r['token']}: sample {r['sample_token']} "
                                  f"not in this dataroot")
            scene_token = src.sample[r["sample_token"]]["scene_token"]
            if scene_token not in frames_of:
                frames_of[scene_token] = scene_frames_from_root(src, scene_token)
            fr = frames_of[scene_token]
            r["t_ns"] = fr.timestamps_ns[fr.index[r["sample_token"]]]
            chain = r["instance_token"]   # the human loop's own identity, per scene
            r.update(instance_token=f"chain:{scene_token}:{chain}", stitch_chain_id=chain,
                     stitch_track_id_pre=None, stitch_interpolated=False, stitch_tier_basis="human")
        merge = merge_human(rows_all, human_rows, coverage)
        rows_all.extend(merge.rows)
        superseded = merge.superseded
        human_meta.update(stats=merge.stats, half_imported_samples=merge.half_imported,
                          coverage={k: len(v) for k, v in merge.coverage.items()})
    category_of = mapper.resolve(r["category"] for r in rows_all)

    # --- 3. tier filter (the only place a row is dropped, and it is recorded) -
    included, excluded = partition(rows_all, tiers, superseded)
    if not included:
        raise ExportError("no rows admitted to sample_annotation")

    # --- 4. instances (by chain), global geometry ----------------------------
    def instance_token_of(r: dict) -> str:
        return make_token("instance", src.sample[r["sample_token"]]["scene_token"], r["stitch_chain_id"])

    pose_of: dict[str, Transform] = {}
    global_center_of: dict[str, np.ndarray] = {}
    global_quat_of: dict[str, list[float]] = {}
    for r in included + excluded:
        sample_token = r["sample_token"]
        if sample_token not in pose_of:
            pose_of[sample_token] = src.lidar_ego_pose(sample_token)
        try:
            t_g, q_g = box_ego_to_global(r["translation_m"], r["rotation_wxyz"], pose_of[sample_token])
        except ValueError as exc:  # geometry.py raises ValueError; the exporter speaks ExportError
            raise ExportError(f"record {r['token']}: {exc}") from exc
        global_center_of[r["token"]] = np.asarray(t_g, dtype=np.float64)
        global_quat_of[r["token"]] = q_g
    by_instance: dict[str, list[dict]] = defaultdict(list)
    instance_scene: dict[str, str] = {}
    instance_category: dict[str, str] = {}
    for r in included:
        inst = instance_token_of(r)
        cat = category_of[r["category"]]
        if inst in instance_category and instance_category[inst] != cat:
            raise ExportError(
                f"instance {inst} (scene {src.sample[r['sample_token']]['scene_token']}, chain "
                f"{r['stitch_chain_id']!r}) changes class {instance_category[inst]} -> {cat}; "
                f"nuScenes stores the category on the instance, so a class change mid-chain must "
                f"be resolved upstream")
        instance_category[inst] = cat
        instance_scene[inst] = src.sample[r["sample_token"]]["scene_token"]
        r["__instance__"] = inst
        by_instance[inst].append(r)

    # --- 5. attributes (final chains only) -----------------------------------
    for r in included:
        r["instance_token"] = r["__instance__"]     # attributes.py groups by instance_token
    if attributes:
        assign_attributes(included, instance_category, global_center_of, cfg.attributes)
    else:
        for r in included:
            r.update(velocity_chain_mps=None, attr_state=None, attr_name=None, attribute_basis=None)

    # --- category / attribute tables -----------------------------------------
    category_token = {name: make_token("category", name) for name in mapper.classes}
    category_table = [{"token": category_token[n], "name": n, "description": ""}
                      for n in mapper.classes]
    attribute_tokens: dict[str, str] = {}

    vis = VisibilityEstimator(src)
    vis_fractions: list[float] = []

    def visibility_of(r: dict, pose: Transform) -> tuple[str, float | None]:
        frac = vis.fraction(r["sample_token"],
                            box_corners_ego(r["translation_m"], r["size_wlh_m"], r["rotation_wxyz"]),
                            pose)
        if frac is None:
            return VISIBILITY_FALLBACK_TOKEN, None
        vis_fractions.append(frac)
        return visibility_token_of_fraction(frac), frac

    def annotation_row(r, inst, tok, prev, nxt, vis_token, frac, t_g, q_g) -> dict:
        prov = r.get("provenance") or {}
        attr_list: list[str] = []
        name = r.get("attr_name")
        if name:
            attribute_tokens.setdefault(name, make_token("attribute", name))
            attr_list = [attribute_tokens[name]]
        row = {
            "token": tok, "sample_token": r["sample_token"], "instance_token": inst,
            "visibility_token": vis_token, "attribute_tokens": attr_list,
            "translation": t_g, "size": [float(v) for v in r["size_wlh_m"]], "rotation": q_g,
            "prev": prev, "next": nxt, "num_lidar_pts": int(r["num_lidar_pts"]), "num_radar_pts": 0,
            # --- provenance extras (ignored by dbench/devkit, kept for audit) ---
            "dhakascenes_record_token": r["token"], "dhakascenes_source": prov.get("source", "pipeline"),
            "dhakascenes_tier": prov.get("tier") if prov.get("source") not in HUMAN_SOURCES else None,
            "dhakascenes_tier_basis": r.get("stitch_tier_basis"),
            "dhakascenes_interpolated": bool(r.get("stitch_interpolated")),
            "dhakascenes_chain_id": r.get("stitch_chain_id"),
            "dhakascenes_track_id_pre_stitch": r.get("stitch_track_id_pre"),
            "num_lidar_pts_basis": r.get("num_lidar_pts_basis"),
            "visibility_basis": "camera_fov_corner_fraction" if frac is not None else "assumed_full",
            "dhakascenes_velocity_chain_mps": r.get("velocity_chain_mps"),
            "dhakascenes_attribute_basis": r.get("attribute_basis"),
            "dhakascenes_verified_by": prov.get("verified_by"),
            "is_uncertain": bool(r.get("is_uncertain")) if r.get("is_uncertain") is not None else False,
            "is_uncertain_reason": r.get("is_uncertain_reason") or "",
        }
        if r.get("velocity_mps") is not None:
            row["dhakascenes_velocity_ego_mps"] = [float(v) for v in r["velocity_mps"]]
        if prov.get("annotator_pass"):
            row["annotator_pass"] = prov["annotator_pass"]
        return row

    devkit_checks: list[bool] = []
    annotations: list[dict] = []
    instances: list[dict] = []
    per_scene: dict[str, dict] = defaultdict(lambda: {
        "n_annotations": 0, "n_instances": 0, "tiers": defaultdict(int), "sources": defaultdict(int)})

    for inst in sorted(by_instance):
        rows = by_instance[inst]
        rows.sort(key=lambda r: (src.sample[r["sample_token"]]["timestamp"], r["token"]))
        stamps = [src.sample[r["sample_token"]]["timestamp"] for r in rows]
        if len(set(stamps)) != len(stamps):
            raise ExportError(f"instance {inst} has two annotations in one sample; chain "
                              f"{rows[0].get('stitch_chain_id')!r} is not unique per keyframe")
        ann_tokens = [make_token("annotation", inst, r["token"]) for r in rows]
        scene_name = src.scene[instance_scene[inst]]["name"]
        for i, r in enumerate(rows):
            pose = pose_of[r["sample_token"]]
            t_g = [float(v) for v in global_center_of[r["token"]]]
            q_g = global_quat_of[r["token"]]
            if len(devkit_checks) < 64:
                ok = verify_with_devkit(r["translation_m"], r["size_wlh_m"], r["rotation_wxyz"],
                                        pose, t_g, q_g)
                if ok is False:
                    raise ExportError(f"devkit cross-check failed for record {r['token']}")
                if ok is not None:
                    devkit_checks.append(ok)
            vis_token, frac = visibility_of(r, pose)
            annotations.append(annotation_row(
                r, inst, ann_tokens[i], ann_tokens[i - 1] if i > 0 else "",
                ann_tokens[i + 1] if i + 1 < len(rows) else "", vis_token, frac, t_g, q_g))
            prov = r.get("provenance") or {}
            ps = per_scene[scene_name]
            ps["n_annotations"] += 1
            ps["tiers"][prov.get("tier") or "unknown"] += 1
            ps["sources"][prov.get("source", "pipeline")] += 1
        per_scene[scene_name]["n_instances"] += 1
        instances.append({
            "token": inst,
            "category_token": category_token[instance_category[inst]],
            "nbr_annotations": len(rows),
            "first_annotation_token": ann_tokens[0],
            "last_annotation_token": ann_tokens[-1],
        })

    # --- the excluded sidecar: same shape, plus why it is not in the release --
    excluded_rows: list[dict] = []
    for r in sorted(excluded, key=lambda r: (src.sample[r["sample_token"]]["timestamp"], r["token"])):
        inst = instance_token_of(r)
        vis_token, frac = visibility_of(r, pose_of[r["sample_token"]])
        row = annotation_row(r, inst, make_token("annotation", inst, r["token"]), "", "",
                             vis_token, frac, [float(v) for v in global_center_of[r["token"]]],
                             global_quat_of[r["token"]])
        row["dhakascenes_excluded_reason"] = r["excluded_reason"]
        excluded_rows.append(row)

    attribute_table = [{"token": tok, "name": name, "description": ""}
                       for name, tok in sorted(attribute_tokens.items())]

    # --- 6. strata + double annotation ---------------------------------------
    centers_by_sample: dict[str, list] = defaultdict(list)
    for r in included:
        centers_by_sample[r["sample_token"]].append(global_center_of[r["token"]])
    all_tokens: list[str] = []
    all_ts: list[int] = []
    all_poses: dict = {}
    scene_names: list[str] = []
    for scene_token in sorted(frames_of, key=lambda s: frames_of[s].scene_name):
        fr = frames_of[scene_token]
        all_tokens.extend(fr.tokens)
        all_ts.extend(fr.timestamps_ns)
        all_poses.update(fr.poses)
        scene_names.append(fr.scene_name)
    all_frames = SceneFrames("*", "+".join(scene_names), all_tokens, all_ts, all_poses)
    image_path_of: dict[str, str | None] = {}
    for tok in all_tokens:
        sd = src.sd_by_sample.get(tok, {}).get(cfg.strata.illumination_channel)
        image_path_of[tok] = os.path.join(src.dataroot, sd["filename"]) if sd else None
    strata = compute_strata(all_frames, centers_by_sample, image_path_of, cfg.strata)
    double_doc: dict | None = None
    double_reused = False
    if cfg.double.fraction > 0:
        os.makedirs(out, exist_ok=True)
        double_doc, double_reused = load_or_select(os.path.join(out, DOUBLE_FILE), strata, all_frames,
                                                   cfg.double, cfg.strata, reselect_double)
    double_tokens = {s["sample_token"] for s in (double_doc or {}).get("selected", [])}
    sample_rows = [dict(r, dbench_double_annotated=(r["token"] in double_tokens))
                   for r in src.tables["sample"]]
    stitch_map = {r["token"]: r["stitch_chain_id"] for r in rows_all
                  if not r.get("stitch_interpolated")
                  and (r.get("provenance") or {}).get("source") not in HUMAN_SOURCES}

    # --- write the new root --------------------------------------------------
    os.makedirs(out_tables, exist_ok=True)
    if not overwrite_tables:
        for entry in sorted(os.listdir(src.dataroot)):
            if entry == version:
                continue
            _link_or_copy(os.path.join(src.dataroot, entry), os.path.join(out, entry), blobs)
        for entry in sorted(os.listdir(src.table_dir)):
            stem = entry[:-5] if entry.endswith(".json") else entry
            if stem in ANNOTATION_TABLES:
                continue
            _link_or_copy(os.path.join(src.table_dir, entry), os.path.join(out_tables, entry), "copy")

    tables = {
        "sample_annotation": annotations,
        "instance": instances,
        "category": category_table,
        "attribute": attribute_table,
        "visibility": VISIBILITY_TABLE,
    }
    for name, rows in tables.items():
        with open(os.path.join(out_tables, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1)
    # sample.json carries dbench_double_annotated, so it is rewritten (not just
    # copied through) in both modes.
    with open(os.path.join(out_tables, "sample.json"), "w", encoding="utf-8") as fh:
        json.dump(sample_rows, fh, indent=1)
    for fname, payload in ((EXCLUDED_TABLE, excluded_rows), (STITCH_MAP, stitch_map)):
        with open(os.path.join(out, fname), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)

    run_manifest = None
    if run_manifest_path is None and os.path.isdir(prelabels):
        cand = os.path.join(prelabels, "run_manifest.json")
        run_manifest_path = cand if os.path.isfile(cand) else None
    if run_manifest_path:
        with open(run_manifest_path, "r", encoding="utf-8") as fh:
            run_manifest = json.load(fh)

    scenes_in_export = sorted(per_scene)
    for name in sorted(human_scenes - set(scenes_in_export)):
        per_scene[name]  # materialise so the flag is visible even with 0 records

    # --- what the vocabulary can produce, and what the route actually held ----
    producible = _producible_classes(mapper)
    present = Counter(instance_category.values())
    ego_range = [math.hypot(float(r["translation_m"][0]), float(r["translation_m"][1]))
                 for r in included + excluded]
    ranges_by_class: dict[str, list[float]] = defaultdict(list)
    for r in included:
        ranges_by_class[instance_category[r["__instance__"]]].append(
            math.hypot(float(r["translation_m"][0]), float(r["translation_m"][1])))
    attr_names = [r["attr_name"] for r in included if r.get("attr_name")]
    attr_bases = [r["attribute_basis"] for r in included if r.get("attribute_basis")]

    meta = {
        "spec": EXPORTER_SPEC,
        "version": version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "pipeline_version": pipeline_version or (run_manifest or {}).get("spec") or "unknown",
        "schema_version": SCHEMA_VERSION_EXPECTED,
        "git_sha": _git_sha(),
        "release_config": cfg.as_dict(),
        "tiers_admitted": tiers,
        "source": {
            "dataroot": src.dataroot,
            "prelabels": os.path.abspath(prelabels),
            "prelabel_files": [{"path": os.path.abspath(p), "sha256": sha256_of(p)} for p in source_files],
            "run_manifest": ({"path": os.path.abspath(run_manifest_path),
                              "sha256": sha256_of(run_manifest_path),
                              "spec": run_manifest.get("spec"),
                              "upstream": run_manifest.get("upstream")}
                             if run_manifest_path else None),
            "blobs": blobs,
            "overwrite_tables": bool(overwrite_tables),
            "cvat_export_3d_dir": os.path.abspath(cvat_export_3d_dir) if cvat_export_3d_dir else None,
        },
        "mapper": {"path": os.path.abspath(mapper_path), "sha256": mapper.sha256,
                   "n_classes": len(mapper.classes),
                   "used": {src_name: dst for src_name, dst in sorted(category_of.items())}},
        "counts": {
            "n_records": len(records),
            "n_annotations": len(annotations),
            "n_instances": len(instances),
            "n_scenes": len(scenes_in_export),
            "n_untracked_singletons": sum(1 for r in records if r.get("track_id") is None),
            "per_class": dict(sorted(
                (c, sum(1 for i in instances if i["category_token"] == category_token[c]))
                for c in mapper.classes)),
        },
        "records": {"n_loaded": len(records), "n_t_ns_corrected": n_t_ns_corrected},
        "stitch": stitch_meta,
        "human": human_meta,
        "excluded": {"table": EXCLUDED_TABLE, "n": len(excluded_rows),
                     "by_reason": dict(sorted(Counter(r["excluded_reason"] for r in excluded).items()))},
        "num_lidar_pts_basis": sorted({str(r.get("num_lidar_pts_basis")) for r in rows_all}),
        "num_radar_pts": "0 for every annotation: the rig carries no radar",
        "visibility": {
            "basis": "assumed_full" if vis.disabled_reason else "camera_fov_corner_fraction",
            "note": ("fraction of the 8 box corners inside at least one camera image of the same "
                     "sample via conventions.project_lidar_to_image; a field-of-view proxy, NOT an "
                     "occlusion estimate"),
            "fallback_reason": vis.disabled_reason,
            "n_estimated": len(vis_fractions),
            "n_assumed_full": len(annotations) + len(excluded_rows) - len(vis_fractions),
        },
        "attributes": {"enabled": bool(attributes),
                       "threshold_mps": cfg.attributes.moving_speed_threshold_mps,
                       "n_with_attribute": len(attr_names),
                       "by_name": dict(sorted(Counter(attr_names).items())),
                       "by_basis": dict(sorted(Counter(attr_bases).items()))},
        "strata": {"density_bin_edges": strata.density_edges,
                   "density_bin_names": list(cfg.strata.density_bin_names),
                   "illumination_bin_edges": list(cfg.strata.illumination_bin_edges),
                   "illumination_bin_names": list(cfg.strata.illumination_bin_names),
                   "n_illumination_unknown": sum(1 for v in strata.illumination.values() if v is None),
                   "per_keyframe": {tok: {"density": strata.density[tok],
                                          "density_bin": strata.density_bin[tok],
                                          "luma": strata.illumination[tok],
                                          "illumination_bin": strata.illumination_bin[tok]}
                                    for tok in all_tokens}},
        "double_annotation": None if double_doc is None else {
            "file": DOUBLE_FILE, "reused": double_reused,
            "n_selected": double_doc["n_selected"], "cells": double_doc["cells"]},
        "range": {"cap_m": round(max(ego_range), 1) if ego_range else None,
                  "note": "BEV range of the exported boxes in the ego frame, not the pipeline's cap",
                  "effective_p99_m_by_class": {c: round(float(np.percentile(v, 99)), 2)
                                               for c, v in sorted(ranges_by_class.items())}},
        "classes": {"present": dict(sorted(present.items())),
                    "absent_on_route": sorted(producible - set(present)),
                    "not_producible": sorted(set(mapper.classes) - producible),
                    "producible_by_vocabulary": sorted(producible)},
        "devkit_cross_check": {"n_checked": len(devkit_checks), "all_passed": all(devkit_checks)
                               if devkit_checks else None},
        "scenes": {
            name: {
                "human_verified": name in human_scenes,
                "n_annotations": per_scene[name]["n_annotations"],
                "n_instances": per_scene[name]["n_instances"],
                "tiers": dict(per_scene[name]["tiers"]),
                "sources": dict(per_scene[name]["sources"]),
            }
            for name in sorted(per_scene)
        },
        "human_verified_scenes_file": os.path.abspath(human_verified_scenes_path)
        if human_verified_scenes_path else None,
        "elapsed_s": round(time.time() - started, 3),
    }
    with open(os.path.join(out, "release_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    return ExportResult(out_root=out, meta=meta, tables=tables, double=double_doc,
                        excluded=excluded_rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    here = REPO_ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prelabels", required=True, help="prelabels.jsonl, or a Stage 9 out dir")
    ap.add_argument("--dataroot", required=True, help="existing nuScenes-format root (read-only)")
    ap.add_argument("--version", default="v1.0-dhaka")
    ap.add_argument("--out", required=True, help="new root to write (must not be --dataroot)")
    ap.add_argument("--mapper", default=os.path.join(here, "configs", "release_category_map.yaml"))
    ap.add_argument("--human-verified-scenes", default=None,
                    help="text (one scene name per line) or JSON list of human-verified scenes")
    ap.add_argument("--blobs", choices=("symlink", "copy"), default="symlink")
    ap.add_argument("--run-manifest", default=None, help="Stage 9 run_manifest.json (auto-found for a dir)")
    ap.add_argument("--pipeline-version", default=None)
    ap.add_argument("--release-config", default=DEFAULT_RELEASE_CONFIG)
    ap.add_argument("--tiers", choices=ADMIT_MODES, default=ADMIT_AUTO,
                    help="pipeline tiers admitted to sample_annotation; 'all' reproduces the pre-2026-09-07 output")
    ap.add_argument("--stitch", dest="stitch", action="store_true", default=True)
    ap.add_argument("--no-stitch", dest="stitch", action="store_false")
    ap.add_argument("--attributes", dest="attributes", action="store_true", default=True)
    ap.add_argument("--no-attributes", dest="attributes", action="store_false")
    ap.add_argument("--double-fraction", type=float, default=None,
                    help="override configs/release.yaml double.fraction; 0 disables")
    ap.add_argument("--reselect-double", action="store_true",
                    help="discard an existing double_annotation.json (kept as .superseded-*)")
    ap.add_argument("--human", default=None, help="<work_root>/stage10_human from scripts/import_cvat_3d.py")
    ap.add_argument("--cvat-export-3d-dir", default=None,
                    help="<work_root>/cvat_export_3d: task.zip clouds for interpolated point counts")
    ap.add_argument("--overwrite-tables", action="store_true",
                    help="rewrite annotation tables/sidecars/meta in an existing export; blobs untouched")
    ap.add_argument("--note", dest="note", action="store_true", default=True,
                    help="write <out>/DELIVERY_NOTE.md from release_meta.json (default)")
    ap.add_argument("--no-note", dest="note", action="store_false")
    ap.add_argument("--stage-tree", default=None,
                    help="work root the stages ran in; recorded in the delivery note's provenance")
    ap.add_argument("--chunk-name", default=None,
                    help="delivery-note title (default: the basename of --out's parent directory)")
    ap.add_argument("--import-manifest", default=None,
                    help="CVAT import_manifest.json (default: <--human>/import_manifest.json)")
    args = ap.parse_args(argv)
    try:
        res = export_release(args.prelabels, args.dataroot, args.version, args.out, args.mapper,
                             args.human_verified_scenes, args.blobs, args.run_manifest,
                             args.pipeline_version,
                             release_config_path=args.release_config, tiers=args.tiers,
                             stitch=args.stitch, attributes=args.attributes,
                             double_fraction=args.double_fraction, human_dir=args.human,
                             overwrite_tables=args.overwrite_tables,
                             cvat_export_3d_dir=args.cvat_export_3d_dir,
                             reselect_double=args.reselect_double)
    except ExportError as exc:
        print(f"export_release: {exc}", file=sys.stderr)
        return 2
    c = res.meta["counts"]
    print(f"wrote {res.out_root}/{args.version}: {c['n_annotations']} annotations, "
          f"{c['n_instances']} instances, {c['n_scenes']} scenes; "
          f"visibility basis={res.meta['visibility']['basis']}; "
          f"devkit check={res.meta['devkit_cross_check']}")
    print(f"tiers={args.tiers}; stitch={res.meta['stitch']['totals'] or 'off'}; "
          f"excluded={res.meta['excluded']['n']} {res.meta['excluded']['by_reason']} "
          f"-> {os.path.join(res.out_root, EXCLUDED_TABLE)}")
    dbl = res.meta["double_annotation"]
    how = "off" if dbl is None else (f"{dbl['n_selected']} keyframes "
                                     f"({'reused' if dbl['reused'] else 'selected'})")
    print(f"double_annotation={how}")
    print(f"release_meta.json -> {os.path.join(res.out_root, 'release_meta.json')}")
    if args.note:
        # The Stage 9 manifest the exporter actually resolved (it auto-finds one
        # when --prelabels is a directory), not the flag as typed.
        src_manifest = (res.meta.get("source") or {}).get("run_manifest") or {}
        imp = args.import_manifest
        if imp is None and args.human:
            imp = os.path.join(args.human, "import_manifest.json")
        chunk = args.chunk_name or os.path.basename(os.path.dirname(os.path.abspath(res.out_root)))
        note_path = write_note(res.out_root, stage9_manifest_path=src_manifest.get("path"),
                               import_manifest_path=imp, stage_tree=args.stage_tree,
                               chunk_name=chunk)
        print(f"delivery note -> {note_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
