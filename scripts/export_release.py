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
  `instance_token` = hash(scene_token, track_id). Untracked records
  (`track_id` None) become singleton instances from their own `instance_token`.
  Annotations of one instance are chained by sample timestamp; `instance`
  carries `first/last_annotation_token` and `nbr_annotations`.

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
  `record.attribute` in {moving, stopped, parked} -> `<group>.moving` etc.
  (vehicle.* / pedestrian.* / cycle.* as in nuScenes). Absent -> []. No
  inference from velocity.

Point counts
  `num_lidar_pts` copied verbatim; its basis is the record's
  `num_lidar_pts_basis` (single-sweep, ground-filtered, PRE-inflation — NOT
  nuScenes' single-sweep-with-ground count) and is written both per annotation
  and in `release_meta.json`. `num_radar_pts` = 0: the rig has no radar.

    python scripts/export_release.py --prelabels <jsonl|stage9 dir> \
        --dataroot <nusc root> --version v1.0-dhaka --out <new root> \
        [--mapper configs/release_category_map.yaml] \
        [--human-verified-scenes scenes.txt] [--blobs symlink|copy]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.conventions import (  # noqa: E402
    CAMERA,
    EGO,
    NUSCENES_GLOBAL,
    Transform,
    project_lidar_to_image,
    transform_matrix,
)
from pipeline.release.geometry import (  # noqa: E402
    box_corners_ego, box_ego_to_global, box_global_to_ego, make_token, normalise_quat, quat_multiply,
)

EXPORTER_SPEC = "dhakascenes/export_release/v1"
SCHEMA_VERSION_EXPECTED = "dhakascenes-pilot/schemas/v1"
ANNOTATION_TABLES = ("sample_annotation", "instance", "category", "attribute", "visibility")
PASSTHROUGH_TABLES = (
    "log", "scene", "sample", "sample_data", "ego_pose", "calibrated_sensor", "sensor", "map",
)
HUMAN_SOURCES = ("human_verified", "human_created")
LIDAR_CHANNEL = "LIDAR_TOP"

# nuScenes' own visibility table, tokens included, so a dbench consumer
# calibrated on nuScenes reads the same strings.
VISIBILITY_TABLE = [
    {"token": "1", "level": "v0-40", "description": "visibility of whole object is between 0 and 40%"},
    {"token": "2", "level": "v40-60", "description": "visibility of whole object is between 40 and 60%"},
    {"token": "3", "level": "v60-80", "description": "visibility of whole object is between 60 and 80%"},
    {"token": "4", "level": "v80-100", "description": "visibility of whole object is between 80 and 100%"},
]
VISIBILITY_FALLBACK_TOKEN = "4"

# nuScenes attribute vocabulary. Which group a class belongs to decides the prefix.
ATTRIBUTE_STATES = ("moving", "stopped", "parked")
ATTRIBUTE_GROUP_OF_CLASS = {
    "pedestrian": "pedestrian",
    "animal": "pedestrian",
    "bicycle": "cycle",
    "motorcycle": "cycle",
    "cycle_rickshaw": "cycle",
    "pushcart": "cycle",
}
PEDESTRIAN_ATTRIBUTES = {"moving": "pedestrian.moving", "stopped": "pedestrian.standing",
                         "parked": "pedestrian.sitting_lying_down"}
CYCLE_ATTRIBUTES = {"moving": "cycle.with_rider", "stopped": "cycle.with_rider",
                    "parked": "cycle.without_rider"}


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
# attributes
# ---------------------------------------------------------------------------


def attribute_name(state: str | None, dbench_class: str) -> str | None:
    if state is None:
        return None
    s = str(state).strip().lower()
    if s not in ATTRIBUTE_STATES:
        raise ExportError(f"attribute {state!r} not in {ATTRIBUTE_STATES}")
    group = ATTRIBUTE_GROUP_OF_CLASS.get(dbench_class, "vehicle")
    if group == "pedestrian":
        return PEDESTRIAN_ATTRIBUTES[s]
    if group == "cycle":
        return CYCLE_ATTRIBUTES[s]
    return f"vehicle.{s}"


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------


@dataclass
class ExportResult:
    out_root: str
    meta: dict
    tables: dict[str, list] = field(default_factory=dict)


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
) -> ExportResult:
    started = time.time()
    out = os.path.abspath(out)
    src = SourceRoot(dataroot, version)
    if os.path.abspath(out) == src.dataroot:
        raise ExportError("--out must differ from --dataroot; the source root is never modified")
    if os.path.isdir(os.path.join(out, version)) and os.listdir(os.path.join(out, version)):
        raise ExportError(f"{out}/{version} already exists and is not empty; refusing to overwrite")

    mapper = CategoryMapper.load(mapper_path)
    records, source_files = load_prelabels(prelabels)
    if not records:
        raise ExportError("no records to export")
    category_of = mapper.resolve(r["category"] for r in records)
    human_scenes = read_scene_list(human_verified_scenes_path)

    # --- resolve each record to its sample / scene ---------------------------
    unknown = sorted({r["sample_token"] for r in records if r["sample_token"] not in src.sample})
    if unknown:
        raise ExportError(f"{len(unknown)} sample_token(s) not in {src.table_dir}/sample.json: "
                          f"{unknown[:5]}{' ...' if len(unknown) > 5 else ''}")

    # --- category / attribute tables ---------------------------------------
    category_token = {name: make_token("category", name) for name in mapper.classes}
    category_table = [{"token": category_token[n], "name": n, "description": ""}
                      for n in mapper.classes]
    attribute_tokens: dict[str, str] = {}

    # --- group by instance -------------------------------------------------
    by_instance: dict[str, list[dict]] = defaultdict(list)
    instance_scene: dict[str, str] = {}
    instance_category: dict[str, str] = {}
    for r in records:
        scene_token = src.sample[r["sample_token"]]["scene_token"]
        if r.get("track_id") is not None:
            inst = make_token("instance", scene_token, r["track_id"])
        else:
            inst = make_token("instance", scene_token, "untracked", r["instance_token"])
        cat = category_of[r["category"]]
        if inst in instance_category and instance_category[inst] != cat:
            raise ExportError(
                f"instance {inst} (scene {scene_token}, track_id {r.get('track_id')!r}) changes "
                f"class {instance_category[inst]} -> {cat}; nuScenes stores the category on the "
                f"instance, so a class change mid-track must be resolved upstream")
        instance_category[inst] = cat
        instance_scene[inst] = scene_token
        by_instance[inst].append(r)

    vis = VisibilityEstimator(src)
    devkit_checks: list[bool] = []
    annotations: list[dict] = []
    instances: list[dict] = []
    per_scene: dict[str, dict] = defaultdict(lambda: {
        "n_annotations": 0, "n_instances": 0, "tiers": defaultdict(int), "sources": defaultdict(int)})
    vis_fractions: list[float] = []

    for inst in sorted(by_instance):
        rows = by_instance[inst]
        rows.sort(key=lambda r: (src.sample[r["sample_token"]]["timestamp"], r["token"]))
        stamps = [src.sample[r["sample_token"]]["timestamp"] for r in rows]
        if len(set(stamps)) != len(stamps):
            raise ExportError(f"instance {inst} has two annotations in one sample; track_id "
                              f"{rows[0].get('track_id')!r} is not unique per keyframe upstream")
        ann_tokens = [make_token("annotation", inst, r["token"]) for r in rows]
        cat = instance_category[inst]
        scene_token = instance_scene[inst]
        scene_name = src.scene[scene_token]["name"]
        for i, r in enumerate(rows):
            pose = src.lidar_ego_pose(r["sample_token"])
            try:
                t_g, q_g = box_ego_to_global(r["translation_m"], r["rotation_wxyz"], pose)
            except ValueError as exc:  # geometry.py raises ValueError; the exporter speaks ExportError
                raise ExportError(f"record {r['token']}: {exc}") from exc
            if len(devkit_checks) < 64:
                ok = verify_with_devkit(r["translation_m"], r["size_wlh_m"], r["rotation_wxyz"],
                                        pose, t_g, q_g)
                if ok is False:
                    raise ExportError(f"devkit cross-check failed for record {r['token']}")
                if ok is not None:
                    devkit_checks.append(ok)

            frac = vis.fraction(r["sample_token"],
                                box_corners_ego(r["translation_m"], r["size_wlh_m"], r["rotation_wxyz"]),
                                pose)
            if frac is None:
                vis_token = VISIBILITY_FALLBACK_TOKEN
            else:
                vis_fractions.append(frac)
                vis_token = visibility_token_of_fraction(frac)

            attr = attribute_name(r.get("attribute"), cat)
            attr_list: list[str] = []
            if attr:
                attribute_tokens.setdefault(attr, make_token("attribute", attr))
                attr_list = [attribute_tokens[attr]]

            prov = r.get("provenance") or {}
            source = prov.get("source", "pipeline")
            ann = {
                "token": ann_tokens[i],
                "sample_token": r["sample_token"],
                "instance_token": inst,
                "visibility_token": vis_token,
                "attribute_tokens": attr_list,
                "translation": t_g,
                "size": [float(v) for v in r["size_wlh_m"]],
                "rotation": q_g,
                "prev": ann_tokens[i - 1] if i > 0 else "",
                "next": ann_tokens[i + 1] if i + 1 < len(rows) else "",
                "num_lidar_pts": int(r["num_lidar_pts"]),
                "num_radar_pts": 0,
                # --- provenance extras (ignored by dbench/devkit, kept for audit) ---
                "dhakascenes_record_token": r["token"],
                "dhakascenes_source": source,
                "dhakascenes_tier": prov.get("tier"),
                "num_lidar_pts_basis": r.get("num_lidar_pts_basis"),
                "visibility_basis": "camera_fov_corner_fraction" if frac is not None else "assumed_full",
            }
            if r.get("velocity_mps") is not None:
                ann["dhakascenes_velocity_ego_mps"] = [float(v) for v in r["velocity_mps"]]
            if source in HUMAN_SOURCES and prov.get("verification_pass"):
                ann["annotator_pass"] = "A"  # single pass; double-annotation pairs come from CVAT, not here
            annotations.append(ann)
            ps = per_scene[scene_name]
            ps["n_annotations"] += 1
            ps["tiers"][prov.get("tier") or "unknown"] += 1
            ps["sources"][source] += 1
        per_scene[scene_name]["n_instances"] += 1
        instances.append({
            "token": inst,
            "category_token": category_token[cat],
            "nbr_annotations": len(rows),
            "first_annotation_token": ann_tokens[0],
            "last_annotation_token": ann_tokens[-1],
        })

    attribute_table = [{"token": tok, "name": name, "description": ""}
                       for name, tok in sorted(attribute_tokens.items())]

    # --- write the new root --------------------------------------------------
    out_tables = os.path.join(out, version)
    os.makedirs(out_tables, exist_ok=True)
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
    meta = {
        "spec": EXPORTER_SPEC,
        "version": version,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "pipeline_version": pipeline_version or (run_manifest or {}).get("spec") or "unknown",
        "schema_version": SCHEMA_VERSION_EXPECTED,
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
        "num_lidar_pts_basis": sorted({str(r.get("num_lidar_pts_basis")) for r in records}),
        "num_radar_pts": "0 for every annotation: the rig carries no radar",
        "visibility": {
            "basis": "assumed_full" if vis.disabled_reason else "camera_fov_corner_fraction",
            "note": ("fraction of the 8 box corners inside at least one camera image of the same "
                     "sample via conventions.project_lidar_to_image; a field-of-view proxy, NOT an "
                     "occlusion estimate"),
            "fallback_reason": vis.disabled_reason,
            "n_estimated": len(vis_fractions),
            "n_assumed_full": len(annotations) - len(vis_fractions),
        },
        "attributes": {"n_with_attribute": sum(1 for a in annotations if a["attribute_tokens"]),
                       "names": sorted(attribute_tokens)},
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
    return ExportResult(out_root=out, meta=meta, tables=tables)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    args = ap.parse_args(argv)
    try:
        res = export_release(args.prelabels, args.dataroot, args.version, args.out, args.mapper,
                             args.human_verified_scenes, args.blobs, args.run_manifest,
                             args.pipeline_version)
    except ExportError as exc:
        print(f"export_release: {exc}", file=sys.stderr)
        return 2
    c = res.meta["counts"]
    print(f"wrote {res.out_root}/{args.version}: {c['n_annotations']} annotations, "
          f"{c['n_instances']} instances, {c['n_scenes']} scenes; "
          f"visibility basis={res.meta['visibility']['basis']}; "
          f"devkit check={res.meta['devkit_cross_check']}")
    print(f"release_meta.json -> {os.path.join(res.out_root, 'release_meta.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
