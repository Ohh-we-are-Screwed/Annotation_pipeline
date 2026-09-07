#!/usr/bin/env python3
"""Pull the finished CVAT 3D annotations back as I-5 human rows (spec §7).

The return leg of scripts/export_cvat_3d.py + scripts/cvat_setup_3d.py. For
every task in the ledger `<work_root>/stage10_human/cvat_tasks.json` this
exports "Datumaro 3D 1.0", inverts the exporter's cuboid encoding, and writes

    <work_root>/stage10_human/scenes/<scene>/verified.jsonl   I-5 AnnotationRecords
    <work_root>/stage10_human/scenes/<scene>/coverage.json    which samples got which pass
    <work_root>/stage10_human/import_manifest.json            what was read, and what was skipped

which is exactly what `pipeline/release/human.py` reads and what
`scripts/export_release.py --human` merges (§7 precedence: a double-annotated
sample supersedes review rows, a reviewed sample supersedes pipeline rows).

COVERAGE.JSON IS WRITTEN EVEN WHEN A TASK CAME BACK WITH ZERO BOXES. Coverage
is a property of the CVAT task, not of the rows: an annotator who deleted every
pre-label on a frame has still covered it, and the release must supersede that
frame's pipeline rows rather than resurrect the boxes the reviewer removed.
`human.py` cannot infer that from an empty verified.jsonl, so the pass over a
frame is recorded separately, here, whenever the task's jobs are complete.

WHAT COMES BACK, AND WHAT DOES NOT — measured, docs/evidence/2026-09-08-cvat-3d-roundtrip.md:

  * IDENTITY IS ON `record_token`, NEVER ON THE EXPORTED `track_id`. CVAT's
    Datumaro exporter overwrites our track id with its own dense per-task track
    index (7 -> 0, 9 -> 1), and an untracked shape comes back carrying the
    number attribute's default `0.0`, which collides with the first track's
    index 0. So `record_token` (text, byte-exact) is what links a row to the
    pre-label it corrects, and the exported `track_id` is used ONLY as a
    within-this-export grouping key — gated on `keyframe` being present, which
    is the sole reliable "this shape belongs to a track" discriminator.
  * Geometry is QUANTIZED to 2 decimals by datumaro on import (position: 1 cm,
    yaw: 0.01 rad ~ 0.573 deg, extents: 1 cm). A reviewed box that was never
    touched comes back different from the box we sent, by up to 0.005 per
    component. Nothing here compares the two; anything that does must use a
    1e-2 grid, not 1e-4.
  * `scale` is NOT reordered: it comes back in the (length, width, height)
    slots export_cvat_3d.cuboid() wrote, so the inverse is just
    size_wlh_m = (scale[1], scale[0], scale[2]).
  * ATTRIBUTION COMES FROM THE LIVE TASK, not from the ledger row the publish
    wrote: `cvat_setup_3d` registers a review task with `assignee: null` and the
    operator assigns it in the CVAT UI afterwards, so the ledger would credit
    every reviewed box to whoever published the project. `verified_by` is
    resolved as task assignee -> job assignee -> task owner -> ledger, and
    `import_manifest.json` records which of those answered, per task.
  * Every DECLARED attribute is always present, at its default when unanswered
    — absence means "the label never declared it", never "the annotator left it
    blank". `uncertain_reason: ""` therefore cannot be told apart from an
    unanswered one, which is why an empty reason is dropped rather than stored.

    python -m scripts.import_cvat_3d                       # every ledger task
    python -m scripts.import_cvat_3d --tasks 292 293       # a subset
    python -m scripts.import_cvat_3d --zip export.zip --kind review --scene chunk_0000 \
        --frames-json .../cvat_export_3d/chunk_0000/frames.json --verified-by ann_a

The offline form reads a dataset zip that was downloaded by hand and never
touches the server. Both forms still need `--paths`: sample timestamps come
from the substrate's `sample.json` and the output root is `<work_root>`.

ONE SCENE'S verified.jsonl IS REWRITTEN WHOLE from the tasks read in this run.
Import all of a scene's tasks together (the default); a `--tasks` subset that
names only some of them replaces the file with just those rows, and says so.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.conventions import (  # noqa: E402
    EGO,
    INTERNAL_TIME_BASE,
    quaternion_from_yaw_rad,
)
from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.common.schemas import (  # noqa: E402
    AnnotationRecord,
    GateVector,
    Provenance,
    ProvenancePolicy,
    write_records,
)
from pipeline.release.frames import BASIS_GROUND_FILTERED, read_pcd_v07_binary  # noqa: E402
from pipeline.release.geometry import points_in_box  # noqa: E402
from pipeline.release.human import COVERAGE_SPEC, KIND_A, KIND_B, KIND_REVIEW, KINDS  # noqa: E402
from scripts.cvat_setup_3d import LEDGER_NAME, ledger_load  # noqa: E402
from scripts.export_release import CategoryMapper  # noqa: E402

MANIFEST_SPEC = "dhakascenes/import_cvat_3d/v1"
MANIFEST_NAME = "import_manifest.json"
FORMAT = "Datumaro 3D 1.0"
OUT_SUBDIR = os.path.dirname(LEDGER_NAME) or "stage10_human"
MAPPER = "configs/release_category_map.yaml"
US_TO_NS = 1_000

# Which of the two independent passes a task's rows belong to (§7). `review` is
# not a pass: `annotator_pass` is None there, and human.py reads exactly that to
# classify a row back into its kind.
ANNOTATOR_PASS_OF = {KIND_REVIEW: None, KIND_A: "A", KIND_B: "B"}

# The pilot has no human annotators, so schemas.py forbids a human `source`
# unless the caller says otherwise. This is the one place that says otherwise.
HUMAN_POLICY = ProvenancePolicy(allow_human_provenance=True)

# A human box is not gated: a person looked at the cloud and drew it, which is
# the highest-confidence evidence the pipeline has. `tier` is required by the
# schema and nulled for human rows on the way into the release
# (export_release.py: `dhakascenes_tier` is None when the source is human), so
# it records "this row ships", nothing more.
HUMAN_TIER = "auto_accept"
DEFAULT_COVERAGE_CONFIG = "R2"


@dataclass
class ImportedBox:
    """One cuboid as CVAT gave it back, in the pipeline's own conventions.

    Faithful to the export: `uncertain` is the checkbox as ticked and
    `uncertain_reason` the text as typed, with no coercion between them (that
    belongs to `records_from_boxes`, where the schema's rule lives).
    """

    sample_token: str
    frame: int
    label: str
    translation_m: list
    size_wlh_m: list
    rotation_wxyz: list
    record_token: str | None
    track_id: int | None
    attribute: str | None
    uncertain: bool
    uncertain_reason: str


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------


def _text(value) -> str:
    """An attribute value as text. Everything CVAT sends is str-able."""
    return "" if value is None else str(value).strip()


def _as_bool(value) -> bool:
    """A checkbox attribute. It comes back as a real bool, but a hand-edited
    dataset (or a future CVAT) may carry "true"/"false"/"0"/"1" instead."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return _text(value).lower() in ("true", "1", "yes")


def _track_id(attributes: dict) -> int | None:
    """The track this shape belongs to, or None if it is a plain shape.

    Gated on `keyframe`, not on the value: an untracked shape carries the
    `track_id` attribute's own default `0.0`, which collides with the first
    track's index. The integer is a WITHIN-THIS-EXPORT grouping key, never an
    identity — CVAT overwrote our value with its own dense index.
    """
    if "keyframe" not in attributes:
        return None
    try:
        return int(float(attributes.get("track_id", 0)))
    except (TypeError, ValueError):
        return None


def _frame_lookup(frames: list[dict]) -> tuple[dict, dict]:
    by_name, by_frame = {}, {}
    for entry in frames:
        by_name[str(entry["name"])] = entry
        by_frame[int(entry["frame"])] = entry
    return by_name, by_frame


def dataset_json_from_zip(path: str) -> dict:
    """The Datumaro document inside an exported dataset zip.

    `annotations/default.json` for a single-subset task, which is what every
    task this pipeline creates is (evidence §6). Globbed rather than hardcoded,
    and more than one subset is refused rather than silently half-read.
    """
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist()
                   if fnmatch.fnmatch(n, "annotations/*.json") and not n.endswith("/")]
        if not members:
            raise ValueError(f"{path}: no annotations/*.json inside "
                             f"(members: {archive.namelist()[:10]})")
        if len(members) > 1:
            raise ValueError(f"{path}: {len(members)} annotation subsets ({sorted(members)}); "
                             "this importer reads one subset per task")
        with archive.open(members[0]) as fh:
            return json.load(fh)


def parse_datumaro_3d(doc: dict, frames: list[dict]) -> list[ImportedBox]:
    """Invert `export_cvat_3d.cuboid()` over a whole Datumaro 3D document.

    position -> translation_m, rotation[2] (yaw about ego +z) -> rotation_wxyz,
    scale (length, width, height) -> size_wlh_m (width, length, height), and the
    item -> `frames.json` -> sample token. Frames are matched by the point-cloud
    file NAME first (that name travels inside task.zip and is what the item id
    is), then by `attr.frame`; an item that matches neither is an error, because
    guessing which keyframe a box belongs to would misplace it silently.
    """
    labels = [str(l.get("name")) for l in
              (((doc.get("categories") or {}).get("label") or {}).get("labels") or [])]
    by_name, by_frame = _frame_lookup(frames)
    boxes: list[ImportedBox] = []
    for item in doc.get("items") or []:
        key = os.path.splitext(os.path.basename(_text(item.get("id"))))[0]
        entry = by_name.get(key)
        if entry is None:
            raw = (item.get("attr") or {}).get("frame")
            if raw is not None:
                try:
                    entry = by_frame.get(int(raw))
                except (TypeError, ValueError):
                    entry = None
        if entry is None:
            raise ValueError(
                f"item {item.get('id')!r} (attr={item.get('attr')}) is in neither the name nor "
                f"the frame index of frames.json ({len(frames)} frames, "
                f"names {sorted(by_name)[:3]}...): the wrong frames.json for this task?")
        for ann in item.get("annotations") or []:
            if ann.get("type") != "cuboid_3d":
                continue
            attributes = dict(ann.get("attributes") or {})
            label_id = ann.get("label_id")
            if not isinstance(label_id, int) or not 0 <= label_id < len(labels):
                raise ValueError(f"item {item.get('id')!r}: label_id={label_id!r} is outside "
                                 f"the export's {len(labels)} categories")
            position = [float(v) for v in ann.get("position") or []]
            rotation = [float(v) for v in ann.get("rotation") or []]
            scale = [float(v) for v in ann.get("scale") or []]
            if len(position) != 3 or len(rotation) != 3 or len(scale) != 3:
                raise ValueError(f"item {item.get('id')!r}: cuboid {ann.get('id')} is not "
                                 "3+3+3 (position, rotation, scale)")
            length, width, height = scale
            record_token = _text(attributes.get("record_token")) or None
            attribute = _text(attributes.get("attribute")) or None
            boxes.append(ImportedBox(
                sample_token=str(entry["sample_token"]),
                frame=int(entry["frame"]),
                label=labels[label_id],
                translation_m=position,
                # (w, l, h) — nuScenes order, `l` along heading. `scale` is not
                # permuted by the round trip, so this is the only swap needed.
                size_wlh_m=[width, length, height],
                rotation_wxyz=[float(v) for v in quaternion_from_yaw_rad(rotation[2])],
                record_token=record_token,
                track_id=_track_id(attributes),
                attribute=attribute,
                uncertain=_as_bool(attributes.get("uncertain")),
                uncertain_reason=_text(attributes.get("uncertain_reason")),
            ))
    return boxes


# ---------------------------------------------------------------------------
# records
# ---------------------------------------------------------------------------


def _normalise(name: str) -> str:
    """The mapper's own key normalisation (export_release.CategoryMapper)."""
    return re.sub(r"\s+", " ", str(name).strip().lower())


def records_from_boxes(boxes, *, scene_token: str, kind: str, verified_by: str,
                       timestamps_ns: dict, point_counter, mapper_classes,
                       task_id: int | None = None,
                       coverage_config: str = DEFAULT_COVERAGE_CONFIG) -> list:
    """I-5 `AnnotationRecord`s for one task's boxes. Nothing is written here.

    `source` is `human_verified` when the box carries a `record_token` — the
    annotator was correcting one of our pre-labels — and `human_created` when it
    does not: a box the annotator drew themselves, which is every box of a blank
    A/B task.

    IDENTITY. A box on a CVAT track gets `human:<kind>:<track id>` so the frames
    of one object become one instance; a lone box gets its record token, or its
    frame position, so it stays a singleton. The kind is inside the token
    deliberately: pass A and pass B annotate the SAME frame, and two rows of one
    sample sharing an instance token is what the release exporter refuses. When
    a scene has more than one task of a kind, `task_id` namespaces it further —
    CVAT's exported track index is dense and PER TASK, so task 41's track 0 and
    task 42's track 0 are different objects.

    `mapper_classes` is the source vocabulary of `configs/release_category_map.yaml`
    (its `map` keys). A label outside it aborts listing every offender: it would
    otherwise fail much later, inside the release export, with the rows already
    on disk.
    """
    if kind not in KINDS:
        raise ValueError(f"kind={kind!r} is not one of {KINDS}")
    if not verified_by:
        raise ValueError(f"{scene_token} {kind}: no annotator to credit (the task has no "
                         "assignee and no owner); a human row must carry verified_by")
    known = {_normalise(c) for c in mapper_classes}
    unmapped = sorted({b.label for b in boxes if _normalise(b.label) not in known})
    if unmapped:
        raise ValueError(f"{scene_token} {kind}: label(s) the release category map does not "
                         f"know: {unmapped}. Add them to the map (or fix the CVAT project's "
                         "labels) — nothing was written.")
    missing = sorted({b.sample_token for b in boxes if b.sample_token not in timestamps_ns})
    if missing:
        raise ValueError(f"{scene_token} {kind}: no timestamp for sample(s) {missing[:5]} "
                         f"({len(missing)} total) — frames.json points at a substrate this "
                         "dataroot does not hold")

    prefix = f"human:{kind}" if task_id is None else f"human:{kind}:{task_id}"
    records = []
    for i, b in enumerate(boxes):
        if b.track_id is not None:
            identity = f"{prefix}:{b.track_id}"
        else:
            identity = f"{prefix}:{b.record_token or f'{b.sample_token}:{b.frame}:{i}'}"
        token = (f"{b.sample_token}:HUMAN_{kind}:{i}" if task_id is None
                 else f"{b.sample_token}:HUMAN_{kind}:{task_id}:{i}")
        # A reason typed without the checkbox ticked is still doubt expressed by
        # the annotator, and the schema only allows the reason alongside the
        # flag; raising the flag keeps the answer, dropping it would lose it.
        uncertain = bool(b.uncertain or b.uncertain_reason)
        records.append(AnnotationRecord(
            token=token,
            sample_token=b.sample_token,
            instance_token=identity,
            category=b.label,
            frame=EGO,
            t_ns=int(timestamps_ns[b.sample_token]),
            time_base=INTERNAL_TIME_BASE,
            translation_m=[float(v) for v in b.translation_m],
            size_wlh_m=[float(v) for v in b.size_wlh_m],
            rotation_wxyz=[float(v) for v in b.rotation_wxyz],
            num_lidar_pts=int(point_counter(b.sample_token, b.translation_m,
                                            b.size_wlh_m, b.rotation_wxyz)),
            provenance=Provenance(
                source="human_verified" if b.record_token else "human_created",
                tier=HUMAN_TIER,
                # A human box passes every gate by construction: the conf term
                # is the annotator, the point floor was theirs to judge on the
                # cloud they saw, and the spatial term is sourced to the human
                # rather than to the (disabled) drivable-area map.
                gates=GateVector(conf=1.0, lidar_pts_ok=True, spatial_ok=True,
                                 spatial_ok_source="human"),
                verified_by=verified_by,
                verification_pass=1,
                annotator_pass=ANNOTATOR_PASS_OF[kind],
            ),
            coverage_config=coverage_config,
            # The same identity as the instance token, so a human row on a CVAT
            # track is not counted among the release's untracked singletons.
            # NOT the bare CVAT index: that would collide with a Stage 7 id.
            track_id=identity if b.track_id is not None else None,
            num_lidar_pts_basis=BASIS_GROUND_FILTERED,
            attribute=b.attribute,
            is_uncertain=uncertain,
            is_uncertain_reason=(b.uncertain_reason or None) if uncertain else None,
        ))
    return records


def write_scene(human_dir: str, scene_name: str, records: list, coverage: dict) -> tuple[str, str]:
    """One scene's I-5 rows and its coverage, where `pipeline/release/human.py` looks.

    An EMPTY verified.jsonl is written when a completed task came back with no
    boxes, and coverage.json is written either way: the pass happened, the
    frames are covered, and the release must supersede their pipeline rows.
    """
    scene_dir = os.path.join(human_dir, "scenes", scene_name)
    os.makedirs(scene_dir, exist_ok=True)
    verified = write_records(os.path.join(scene_dir, "verified.jsonl"), records,
                             policy=HUMAN_POLICY, expect_type=AnnotationRecord, allow_empty=True)
    doc = {"spec": COVERAGE_SPEC, "scene": scene_name,
           "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    doc.update({k: sorted(set(coverage.get(k) or ())) for k in KINDS})
    path = os.path.join(scene_dir, "coverage.json")
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, path)
    return verified, path


# ---------------------------------------------------------------------------
# the cloud the annotator saw
# ---------------------------------------------------------------------------


class TaskCloudPoints:
    """`num_lidar_pts` from the point cloud that was IN the task.

    Same basis as the pipeline's own count — single sweep, ground filtered, pre
    inflation (that is what export_cvat_3d packed) — so a human row's count is
    comparable with a pipeline row's. With no task.zip beside frames.json every
    count is 0 and the caller says so out loud; a silently zeroed count would
    read downstream as "this box has no returns".
    """

    def __init__(self, task_zip: str, frames: list[dict]):
        self.path = task_zip
        self.available = os.path.isfile(task_zip)
        self._zip = zipfile.ZipFile(task_zip) if self.available else None
        self._name_of = {str(r["sample_token"]): str(r["name"]) for r in frames}
        self._cached_name: str | None = None
        self._cached: np.ndarray | None = None

    def points(self, sample_token: str):
        name = self._name_of.get(sample_token)
        if self._zip is None or name is None:
            return None
        if name != self._cached_name:
            try:
                arr = read_pcd_v07_binary(self._zip.read(f"pointcloud/{name}.pcd"))
                self._cached = arr[:, :3].astype(np.float64)
            except (KeyError, ValueError):
                self._cached = None
            self._cached_name = name
        return self._cached

    def __call__(self, sample_token: str, translation, size_wlh, rotation_wxyz) -> int:
        pts = self.points(sample_token)
        if pts is None:
            return 0
        return points_in_box(pts, translation, size_wlh, rotation_wxyz)

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None


def load_timestamps_ns(paths) -> dict:
    """sample token -> Unix nanoseconds, from the substrate's own sample.json.

    nuScenes stores microseconds; records are `unix_ns` (schemas._check_time
    refuses a microsecond value outright), so the conversion happens once, here.
    """
    for root in (paths.version_dir, os.path.join(paths.dataroot, paths.version)):
        table = os.path.join(root, "sample.json")
        if os.path.isfile(table):
            with open(table, "r", encoding="utf-8") as fh:
                return {r["token"]: int(r["timestamp"]) * US_TO_NS for r in json.load(fh)}
    raise FileNotFoundError(f"no sample.json under {paths.version_dir} or "
                            f"{os.path.join(paths.dataroot, paths.version)}")


def job_state(job) -> str:
    """A job's state as a plain lowercase string ('new', 'in progress', ...)."""
    state = getattr(job, "state", None)
    return _text(getattr(state, "value", state)).lower()


def _username(value) -> str | None:
    """A CVAT user reference -> its username, whether it arrives as an object or a dict."""
    if value is None:
        return None
    name = value.get("username") if isinstance(value, dict) else getattr(value, "username", None)
    name = _text(name).strip()
    return name or None


def resolve_verified_by(task, jobs, ledger_assignee) -> tuple:
    """Who to credit a task's boxes to, and where that answer came from.

    Spec §7.5: "`verified_by` = CVAT assignee username (task owner if
    unassigned)". The LIVE task is the authority, not the ledger:
    `cvat_setup_3d` registers a review task with `assignee: null` and the
    operator assigns it in the UI afterwards, so reading the ledger credited
    every reviewed box to whoever published the project (final review I5). The
    same applies to an A/B task reassigned after publish.

    Order: task assignee -> the first job with an assignee -> task owner ->
    the ledger's publish-time value -> nobody.
    """
    name = _username(getattr(task, "assignee", None))
    if name:
        return name, "task_assignee"
    for job in jobs or ():
        name = _username(getattr(job, "assignee", None))
        if name:
            return name, "job_assignee"
    name = _username(getattr(task, "owner", None))
    if name:
        return name, "task_owner"
    name = _text(ledger_assignee).strip() or None
    return (name, "ledger") if name else (None, None)


def incomplete_reason(jobs) -> str | None:
    """None when every job of the task is completed, else why it is not."""
    states = [job_state(j) for j in jobs]
    if not states:
        return "the task has no jobs"
    unfinished = [s for s in states if s != "completed"]
    if unfinished:
        return (f"{len(unfinished)}/{len(states)} job(s) not completed "
                f"({', '.join(sorted(set(unfinished)))})")
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--host", default=os.environ.get("CVAT_HOST", "http://localhost:8081"))
    parser.add_argument("--user", default=os.environ.get("CVAT_USER", "mt"))
    parser.add_argument("--password", default=os.environ.get("CVAT_PASSWORD", ""))
    parser.add_argument("--ledger", default=None,
                        help="task ledger cvat_setup_3d wrote; default <work_root>/" + LEDGER_NAME)
    parser.add_argument("--tasks", nargs="*", type=int, default=None,
                        help="import only these CVAT task ids (default: every ledger task)")
    parser.add_argument("--out", default=None, help="default <work_root>/" + OUT_SUBDIR)
    parser.add_argument("--mapper", default=os.path.join(here, MAPPER))
    parser.add_argument("--include-incomplete", action="store_true",
                        help="import a task whose jobs are not all in state 'completed' "
                             "(default: skip it and say so)")
    parser.add_argument("--zip", default=None,
                        help="offline: a dataset zip exported by hand, instead of the server")
    parser.add_argument("--kind", choices=KINDS, default=None, help="offline: the pass --zip holds")
    parser.add_argument("--scene", default=None, help="offline: the scene --zip belongs to")
    parser.add_argument("--frames-json", default=None,
                        help="offline: the frames.json of that task (task.zip is read beside it)")
    parser.add_argument("--verified-by", default=None,
                        help="offline: the annotator to credit; on the server path the task's "
                             "assignee (else its owner) is used")
    args = parser.parse_args(argv)

    offline = args.zip is not None
    if offline:
        missing = [f for f, v in (("--kind", args.kind), ("--scene", args.scene),
                                  ("--frames-json", args.frames_json),
                                  ("--verified-by", args.verified_by)) if not v]
        if missing:
            print(f"--zip needs {', '.join(missing)}", file=sys.stderr)
            return 2
    elif not args.password:
        print("set CVAT_PASSWORD or pass --password (or use --zip for the offline form)",
              file=sys.stderr)
        return 2

    paths = load_paths(args.paths)
    out_dir = args.out or os.path.join(paths.work_root, OUT_SUBDIR)
    mapper = CategoryMapper.load(args.mapper)
    mapper_classes = set(mapper.mapping)
    timestamps_ns = load_timestamps_ns(paths)

    if offline:
        entries = [{"task_id": None, "kind": args.kind, "scene": args.scene,
                    "frames_json": args.frames_json, "assignee": args.verified_by,
                    "zip": args.zip}]
    else:
        ledger_path = args.ledger or os.path.join(paths.work_root, LEDGER_NAME)
        entries = [e for e in ledger_load(ledger_path) if e.get("kind") in KINDS]
        if args.tasks:
            wanted = set(args.tasks)
            entries = [e for e in entries if e.get("task_id") in wanted]
            unknown = sorted(wanted - {e.get("task_id") for e in entries})
            if unknown:
                print(f"!!! {ledger_path} has no row for task(s) {unknown}", file=sys.stderr)
                return 2
        if not entries:
            print(f"nothing to import: {ledger_path} lists no review/double task"
                  + (" matching --tasks" if args.tasks else ""), file=sys.stderr)
            return 2
        print(f"{len(entries)} task(s) from {ledger_path}")

    client = None
    if not offline:
        from cvat_sdk import make_client  # noqa: PLC0415 — the offline form needs no SDK

        client = make_client(host=args.host, credentials=(args.user, args.password))

    manifest_tasks: list[dict] = []
    by_scene: dict[str, list] = {}
    coverage_of: dict[str, dict] = {}
    covered_by: dict[tuple, dict] = {}
    tmpdir = tempfile.mkdtemp(prefix="import_cvat_3d-")
    try:
        for entry in entries:
            task_id, kind, scene = entry.get("task_id"), entry["kind"], entry["scene"]
            row = {"task_id": task_id, "kind": kind, "scene": scene,
                   "assignee": entry.get("assignee"), "verified_by": None,
                   "verified_by_source": None, "n_boxes": 0, "n_frames_covered": 0,
                   "skipped_reason": None}
            manifest_tasks.append(row)
            label = f"task #{task_id}" if task_id is not None else os.path.basename(str(entry.get("zip")))

            verified_by = entry.get("assignee")
            verified_by_source = "cli" if offline else "ledger"
            if offline:
                zip_path = entry["zip"]
            else:
                try:
                    task = client.tasks.retrieve(task_id)
                except Exception as exc:  # noqa: BLE001 — a deleted task is skipped, not fatal
                    row["skipped_reason"] = f"not retrievable from {args.host}: {exc}"
                    print(f"  {label} {scene} [{kind}]: SKIPPED — {row['skipped_reason']}",
                          file=sys.stderr)
                    continue
                jobs = task.get_jobs()
                reason = incomplete_reason(jobs)
                if reason and not args.include_incomplete:
                    row["skipped_reason"] = reason
                    print(f"  {label} {scene} [{kind}]: SKIPPED — {reason} "
                          "(--include-incomplete to read it anyway)")
                    continue
                row["incomplete"] = bool(reason)
                # The live task, not the ledger: a review task is published
                # unassigned and assigned in the UI afterwards (spec §7.5, I5).
                verified_by, verified_by_source = resolve_verified_by(task, jobs, verified_by)
                zip_path = os.path.join(tmpdir, f"task_{task_id}.zip")
                task.export_dataset(FORMAT, zip_path, include_images=False)

            frames_json = entry["frames_json"]
            with open(frames_json, "r", encoding="utf-8") as fh:
                frames = json.load(fh)
            doc = dataset_json_from_zip(zip_path)
            boxes = parse_datumaro_3d(doc, frames)

            counter = TaskCloudPoints(os.path.join(os.path.dirname(frames_json), "task.zip"), frames)
            if not counter.available:
                print(f"  ! {label}: no task.zip beside {frames_json} — num_lidar_pts is 0 on "
                      f"every row of this task", file=sys.stderr)
            try:
                records = records_from_boxes(
                    boxes, scene_token=scene, kind=kind, verified_by=verified_by,
                    timestamps_ns=timestamps_ns, point_counter=counter,
                    mapper_classes=mapper_classes, task_id=task_id)
            finally:
                counter.close()

            covered = [str(f["sample_token"]) for f in frames]
            for sample in covered:
                seen = covered_by.setdefault((scene, kind, sample), {"task": task_id})
                if seen["task"] != task_id:
                    print(f"  ! {scene}: sample {sample} is covered by BOTH task #{seen['task']} "
                          f"and {label} of kind {kind}; both sets of boxes will ship",
                          file=sys.stderr)
            by_scene.setdefault(scene, []).extend(records)
            cov = coverage_of.setdefault(scene, {k: set() for k in KINDS})
            cov[kind].update(covered)
            row["n_boxes"] = len(records)
            row["n_frames_covered"] = len(covered)
            row["points_from"] = "task.zip" if counter.available else "unavailable"
            row["verified_by"] = verified_by
            row["verified_by_source"] = verified_by_source
            print(f"  {label} {scene} [{kind}]: {len(records)} box(es) over "
                  f"{len(covered)} frame(s), credited to {verified_by} ({verified_by_source})")
    finally:
        if client is not None:
            client.close()
        shutil.rmtree(tmpdir, ignore_errors=True)

    rows_by_source: dict[str, int] = {}
    rows_by_scene: dict[str, int] = {}
    for scene in sorted(coverage_of):
        records = by_scene.get(scene, [])
        # A scene's file is REPLACED by the tasks read in this run, so a --tasks
        # subset that names only some of a scene's tasks drops the rest. Said out
        # loud rather than discovered as rows missing from the release.
        previous = os.path.join(out_dir, "scenes", scene, "verified.jsonl")
        if os.path.isfile(previous):
            with open(previous, "r", encoding="utf-8") as fh:
                had = sum(1 for line in fh if line.strip())
            print(f"  ! {scene}: replacing an existing verified.jsonl ({had} row(s)) with the "
                  f"{len(records)} row(s) of this run's task(s)", file=sys.stderr)
        verified, coverage_path = write_scene(out_dir, scene, records, coverage_of[scene])
        rows_by_scene[scene] = len(records)
        for r in records:
            rows_by_source[r.provenance.source] = rows_by_source.get(r.provenance.source, 0) + 1
        kinds = ", ".join(f"{k}={len(v)}" for k, v in coverage_of[scene].items() if v)
        print(f"  wrote {verified} ({len(records)} row(s)) and coverage [{kinds}]")

    manifest = {
        "spec": MANIFEST_SPEC,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": None if offline else args.host,
        "mapper": {"path": os.path.abspath(args.mapper), "sha256": mapper.sha256},
        "tasks": manifest_tasks,
        "rows_by_source": dict(sorted(rows_by_source.items())),
        "rows_by_scene": dict(sorted(rows_by_scene.items())),
    }
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, MANIFEST_NAME)
    tmp = f"{manifest_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    os.replace(tmp, manifest_path)

    imported = sum(1 for t in manifest_tasks if t["skipped_reason"] is None)
    skipped = len(manifest_tasks) - imported
    print(f"\n{imported} task(s) imported, {skipped} skipped -> {manifest_path}")
    if rows_by_scene:
        print(f"merge into a release with: python -m scripts.export_release --human {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
