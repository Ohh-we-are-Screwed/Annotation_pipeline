"""Stage 0 — substrate probe: prove completeness, do not repair it (§5.1).

Stage 0 is a VERIFICATION stage. The "partial blob" motivation it was
originally written for is gone; what stayed is the reason it must exist at all:
a probe that *assumes* completeness is exactly the class of assumption this
project is trying to eliminate.

Rev 1's probe checked file existence, which on a complete v1.0-mini passes
trivially and catches nothing. `fully_present` is replaced here by six named,
executable predicates, each emitted BY NAME when it fails (P0-6):

  version_matches      the metadata directory name equals the configured version
  channels_complete    every keyframe has a sample_data record for every channel
                       in the declared required-channel set
  files_resolve        every referenced file exists
  files_parse          size and parse validity: `size % 20 == 0` and a sane size
                       band for `.pcd.bin`, SOI/EOI markers and a sane size band
                       for JPEG
  sweeps_cover_window  every LiDAR sweep inside W_acc of every keyframe exists
  token_graph_closed   sample_data -> ego_pose, sample_data -> calibrated_sensor,
                       sample -> sample_annotation -> instance -> category, and
                       intact prev/next chains

Nothing here repairs, substitutes, or skips. A scene that fails any predicate is
excluded LOUDLY, with the failing predicate named, rather than having its bad
frames skipped quietly at Stage 5.

Outputs, both written atomically and read back before they land:

  usable_scenes.json   the only scene list any later stage reads. Carries the
                       dataroot realpath, the metadata fingerprint, the
                       required-channel set, and the W_acc actually used —
                       each of which changes the answer — plus the scene
                       partition.
  probe_report.json    per-scene failing predicates and the measurements behind
                       them.

Both go under `work_root`. They do NOT go to `probe_out_root`: that root is
hard-separated for the §8 indigenous-prompt probe, and this is a different
thing that happens to share the word.

Phase 1 constraints hold: no GPU, no models, no nuscenes-devkit. stdlib only —
verifying the substrate with the library that assumes the substrate would be
circular.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field

from pipeline.common.manifest import (
    clear_markers,
    write_json_atomic as _write_json_atomic,
    write_marker,
)
from pipeline.common.paths import (
    METADATA_TABLES,
    FINGERPRINT_SPEC,
    Paths,
    PathValidationError,
    assert_dataroot_read_only,
    load_paths,
    metadata_fingerprint,
)
from pipeline.common.schemas import (
    POINT_RECORD_BYTES,
    REQUIRED_CHANNELS,
    RING_CAMERAS,
    SchemaValidationError,
    SubstrateManifest,
    validate_records,
)

PROBE_SPEC = "dhakascenes-pilot/stage0_data_probe/v1"

# --- accumulation window (§11, decision 2) ----------------------------------
# Duration is preserved, not count: at this substrate's measured ~20 Hz,
# preserving the spec's 5-sweep count would halve the time window and change
# what "accumulated" means. Both are recorded because both are load-bearing.
W_ACC_DURATION_NS = 500_000_000  # 0.5 s   — spec §7.3.2, duration preserved
W_ACC_COUNT = 5  # sweeps  — derived: 0.5 s at the measured 10.00 Hz (v1.0-dhaka-fixed)

# --- measured on v1.0-mini, 2026-08-12 --------------------------------------
# LIDAR_TOP median inter-sweep period 49.788 ms (20.09 Hz) over 3935 records.
NOMINAL_SWEEP_PERIOD_NS = 49_788_000
# The largest observed intra-scene gap is 100.079 ms — exactly one dropped
# sweep, in scene-0061. That is a property of the substrate, declared here so
# the gate does not fail the dataset for being what it measurably is. The gate
# fires at three nominal periods, i.e. two or more consecutive dropped sweeps.
MAX_SWEEP_GAP_NS = 3 * NOMINAL_SWEEP_PERIOD_NS
# A full (untruncated) window must be at least 80 % covered. At scene starts the
# window IS truncated; that is expected, recorded, and exempted rather than
# counted as a failure.
MIN_SWEEP_COVERAGE = 0.8

# --- parse bands (§5.1 predicate 3) -----------------------------------------
# A truncated .pcd.bin whose length is still a multiple of 20 reshapes to (-1,5)
# without error and yields silently fewer points, so the modulo check alone is
# not enough: the size band is what catches a half-written cloud.
# Measured: 34368-34816 points per cloud, i.e. 687360-696320 bytes.
PCD_MIN_POINTS = 10_000
PCD_MAX_POINTS = 300_000
JPEG_MIN_BYTES = 20_000
JPEG_MAX_BYTES = 4_000_000
JPEG_SOI = b"\xff\xd8\xff"
JPEG_EOI = b"\xff\xd9"

# --- scene partition (§11, decision 3) --------------------------------------
# Disjoint, and stratified so each subset spans BOTH locations and holds at
# least one night scene. Recorded by scene NAME, and verified against the
# measured stratification below — an assignment that is merely asserted is how
# "tuning prompts on S0 and evaluating on S0" (GAP_ANALYSIS §3) gets reproduced.
PARTITION: dict[str, tuple[str, ...]] = {
    "priors": ("scene-0061", "scene-0103", "scene-0553", "scene-1077"),
    "tuning": ("scene-0655", "scene-1094"),
    "run": ("scene-0757", "scene-0796", "scene-0916", "scene-1100"),
}
LOCATIONS: tuple[str, ...] = ("singapore", "boston")
# Illumination is DERIVED from the scene description, not measured luminance.
# Recorded as a method string so nobody later reads it as §8.3.2's photometric
# binning, which this is not.
NIGHT_METHOD = "case-insensitive substring 'night' in scene.description"

PREDICATES: tuple[str, ...] = (
    "version_matches",
    "channels_complete",
    "files_resolve",
    "files_parse",
    "sweeps_cover_window",
    "token_graph_closed",
)

EXIT_OK = 0
EXIT_INCOMPLETE = 1  # probe ran; at least one scene excluded or the partition broke
EXIT_HARD_STOP = 2  # the substrate is not addressable at all


class HardStop(RuntimeError):
    """A condition that makes the probe's question unanswerable (§5.1).

    Version mismatch, zero usable scenes, dataroot missing samples/ or the
    version directory. These do not produce a smaller allowlist; they produce
    no allowlist.
    """


# ---------------------------------------------------------------------------
# Predicate accounting
# ---------------------------------------------------------------------------


@dataclass
class PredicateResult:
    """One named predicate's verdict for one scene.

    `detail` holds the evidence, bounded: the first few offending items plus a
    total. A predicate that fails with no evidence is only marginally better
    than a bare bucket label.
    """

    name: str
    ok: bool
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"predicate": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class SceneVerdict:
    name: str
    token: str
    location: str
    is_night: bool
    nbr_samples: int
    predicates: list[PredicateResult] = field(default_factory=list)
    measurements: dict = field(default_factory=dict)

    @property
    def failing(self) -> list[str]:
        return [p.name for p in self.predicates if not p.ok]

    @property
    def usable(self) -> bool:
        return not self.failing

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "token": self.token,
            "location": self.location,
            "is_night": self.is_night,
            "nbr_samples": self.nbr_samples,
            "usable": self.usable,
            "failing_predicates": self.failing,
            "predicates": [p.as_dict() for p in self.predicates],
            "measurements": self.measurements,
        }


def _sample(items, limit: int = 5) -> list:
    """Bounded evidence: the first `limit` offenders, in a stable order."""
    out = sorted(items)[:limit] if not isinstance(items, list) else items[:limit]
    return list(out)


# ---------------------------------------------------------------------------
# Substrate: the tables, plus the indices every predicate needs
# ---------------------------------------------------------------------------


@dataclass
class Substrate:
    paths: Paths
    tables: dict
    channel_of_calibrated_sensor: dict
    sample_data_by_scene: dict
    samples_by_token: dict
    sample_data_by_token: dict
    ego_pose_tokens: set
    calibrated_sensor_tokens: set
    instance_by_token: dict
    category_tokens: set
    annotations_by_sample: dict

    @classmethod
    def load(cls, paths: Paths) -> "Substrate":
        tables = {}
        for name in METADATA_TABLES:
            with open(paths.table(name), "r", encoding="utf-8") as fh:
                tables[name] = json.load(fh)

        sensor_channel = {s["token"]: s["channel"] for s in tables["sensor.json"]}
        channel_of_cs = {
            c["token"]: sensor_channel[c["sensor_token"]]
            for c in tables["calibrated_sensor.json"]
            if c["sensor_token"] in sensor_channel
        }

        samples_by_token = {s["token"]: s for s in tables["sample.json"]}
        scene_of_sample = {s["token"]: s["scene_token"] for s in tables["sample.json"]}

        sd_by_scene: dict = defaultdict(list)
        for record in tables["sample_data.json"]:
            scene_token = scene_of_sample.get(record["sample_token"])
            # A sample_data whose sample is unknown cannot be attributed to a
            # scene; token_graph_closed reports it globally rather than letting
            # it vanish from every per-scene check.
            sd_by_scene[scene_token].append(record)

        annotations_by_sample: dict = defaultdict(list)
        for ann in tables["sample_annotation.json"]:
            annotations_by_sample[ann["sample_token"]].append(ann)

        return cls(
            paths=paths,
            tables=tables,
            channel_of_calibrated_sensor=channel_of_cs,
            sample_data_by_scene=dict(sd_by_scene),
            samples_by_token=samples_by_token,
            sample_data_by_token={r["token"]: r for r in tables["sample_data.json"]},
            ego_pose_tokens={e["token"] for e in tables["ego_pose.json"]},
            calibrated_sensor_tokens={c["token"] for c in tables["calibrated_sensor.json"]},
            instance_by_token={i["token"]: i for i in tables["instance.json"]},
            category_tokens={c["token"] for c in tables["category.json"]},
            annotations_by_sample=dict(annotations_by_sample),
        )

    def channel(self, record: dict) -> str | None:
        return self.channel_of_calibrated_sensor.get(record["calibrated_sensor_token"])

    def by_token(self, table: str) -> dict:
        """token -> record for any metadata table, memoised on first use."""
        cache = self.__dict__.setdefault("_by_token", {})
        if table not in cache:
            cache[table] = {r["token"]: r for r in self.tables[table]}
        return cache[table]

    def blob(self, record: dict) -> str:
        return os.path.join(self.paths.dataroot, record["filename"])

    def scene_samples(self, scene: dict) -> list[dict]:
        """Samples in `next`-chain order. Order is the chain's, not the table's."""
        out: list[dict] = []
        token = scene["first_sample_token"]
        seen: set = set()
        while token and token in self.samples_by_token and token not in seen:
            seen.add(token)
            record = self.samples_by_token[token]
            out.append(record)
            token = record.get("next") or ""
        return out


# ---------------------------------------------------------------------------
# Predicate 1 — version_matches (global)
# ---------------------------------------------------------------------------


def check_version_matches(paths: Paths) -> PredicateResult:
    """The on-disk metadata directory name equals the configured version.

    Hard stop, not a scene exclusion: mini metadata pointed at trainval blobs
    yields zero token matches, which reads downstream as "no usable scenes"
    rather than as the configuration error it is.
    """
    present = sorted(
        name
        for name in (os.listdir(paths.meta_root) if os.path.isdir(paths.meta_root) else [])
        if name.startswith("v") and os.path.isdir(os.path.join(paths.meta_root, name))
    )
    # The comparison is against what is ON DISK, not against the path this
    # config composed: `basename(meta_root/version)` is the configured version
    # by construction and would agree with itself no matter what is there.
    ok = paths.version in present and os.path.isdir(paths.version_dir)
    return PredicateResult(
        "version_matches",
        ok,
        {
            "configured": paths.version,
            "version_dirs_on_disk": present,
            "version_dir": paths.version_dir,
        },
    )


# ---------------------------------------------------------------------------
# Predicate 2 — channels_complete
# ---------------------------------------------------------------------------


def check_channels_complete(sub: Substrate, scene: dict, samples: list[dict]) -> PredicateResult:
    """Every keyframe carries a sample_data for every REQUIRED channel.

    This is I-1's analogue of "a bag missing a mandatory topic fails ingestion".
    RADAR is excluded from the required set, declared: gating on it would drop
    scenes for reasons the pipeline does not care about, and dropping it
    silently would make `usable_scenes.json` not mean what its name says. RADAR
    presence is measured below and reported, never gated.
    """
    present: dict = defaultdict(set)
    for record in sub.sample_data_by_scene.get(scene["token"], []):
        if record.get("is_key_frame"):
            channel = sub.channel(record)
            if channel is not None:
                present[record["sample_token"]].add(channel)

    required = set(REQUIRED_CHANNELS)
    missing: dict = {}
    for sample in samples:
        gap = required - present.get(sample["token"], set())
        if gap:
            missing[sample["token"]] = sorted(gap)

    return PredicateResult(
        "channels_complete",
        not missing,
        {
            "required_channels": list(REQUIRED_CHANNELS),
            "n_keyframes": len(samples),
            "n_keyframes_incomplete": len(missing),
            "first_incomplete": {k: missing[k] for k in _sample(list(missing))},
        },
    )


# ---------------------------------------------------------------------------
# Predicates 3 and 4 — files_resolve, files_parse
# ---------------------------------------------------------------------------


def _parse_pcd_bin(path: str, size: int) -> str | None:
    """Return a reason string if the cloud is not parseable, else None."""
    if size % POINT_RECORD_BYTES != 0:
        return f"size {size} is not a multiple of {POINT_RECORD_BYTES}"
    n_points = size // POINT_RECORD_BYTES
    if not PCD_MIN_POINTS <= n_points <= PCD_MAX_POINTS:
        return f"{n_points} points is outside [{PCD_MIN_POINTS}, {PCD_MAX_POINTS}]"
    return None


def _parse_jpeg(path: str, size: int) -> str | None:
    """Return a reason string if the JPEG is not parseable, else None.

    Header AND tail: a truncated JPEG keeps a valid SOI, often decodes to a
    partial image, and produces plausible downstream output. The EOI marker is
    what distinguishes it from a complete file.
    """
    if not JPEG_MIN_BYTES <= size <= JPEG_MAX_BYTES:
        return f"size {size} is outside [{JPEG_MIN_BYTES}, {JPEG_MAX_BYTES}]"
    with open(path, "rb") as fh:
        if fh.read(3) != JPEG_SOI:
            return "missing JPEG SOI marker"
        fh.seek(-2, os.SEEK_END)
        if fh.read(2) != JPEG_EOI:
            return "missing JPEG EOI marker (truncated file)"
    return None


def check_files(sub: Substrate, scene: dict) -> tuple[PredicateResult, PredicateResult, dict]:
    """files_resolve and files_parse over the required channels, in one pass.

    Both predicates are evaluated over the REQUIRED channels only, matching
    channels_complete's declared scope. Files for non-required channels (RADAR)
    are still stat-ed, and their misses are reported as a measurement so that
    "we did not look" is never confused with "we looked and it was fine".
    """
    required = set(REQUIRED_CHANNELS)
    missing: list[str] = []
    unparseable: list[dict] = []
    n_checked = 0
    non_required_missing: list[str] = []
    n_non_required = 0

    for record in sub.sample_data_by_scene.get(scene["token"], []):
        channel = sub.channel(record)
        path = sub.blob(record)
        if channel not in required:
            n_non_required += 1
            if not os.path.isfile(path):
                non_required_missing.append(record["filename"])
            continue

        n_checked += 1
        try:
            size = os.path.getsize(path)
        except OSError:
            missing.append(record["filename"])
            continue

        fileformat = (record.get("fileformat") or "").lower()
        if fileformat == "pcd" or record["filename"].endswith(".pcd.bin"):
            reason = _parse_pcd_bin(path, size)
        elif fileformat in ("jpg", "jpeg") or record["filename"].endswith((".jpg", ".jpeg")):
            reason = _parse_jpeg(path, size)
        else:
            reason = f"unhandled fileformat {fileformat!r}"
        if reason is not None:
            unparseable.append({"filename": record["filename"], "reason": reason})

    resolve = PredicateResult(
        "files_resolve",
        not missing,
        {
            "scope": "required channels only (RADAR reported, not gated)",
            "n_checked": n_checked,
            "n_missing": len(missing),
            "first_missing": _sample(missing),
        },
    )
    parse = PredicateResult(
        "files_parse",
        not unparseable,
        {
            "n_checked": n_checked - len(missing),
            "n_unparseable": len(unparseable),
            "first_unparseable": unparseable[:5],
            "point_record_bytes": POINT_RECORD_BYTES,
        },
    )
    measurement = {
        "n_non_required_files": n_non_required,
        "n_non_required_missing": len(non_required_missing),
        "first_non_required_missing": _sample(non_required_missing),
    }
    return resolve, parse, measurement


# ---------------------------------------------------------------------------
# Predicate 5 — sweeps_cover_window
# ---------------------------------------------------------------------------


def check_sweeps_cover_window(sub: Substrate, scene: dict, samples: list[dict]) -> PredicateResult:
    """Every LiDAR sweep inside W_acc of every keyframe exists and resolves.

    Rev 1 never mentioned sweeps. Accumulation reads `sweeps/LIDAR_TOP/*.pcd.bin`
    and would hit a missing file mid-run — the precise failure Stage 0 exists to
    prevent, discovered at Stage 1 instead of here.

    Three things are checked per keyframe, and one is deliberately not:
      - every LIDAR_TOP record whose timestamp falls in
        [t_anchor - W_ACC_DURATION_NS, t_anchor] resolves on disk;
      - the window holds at least MIN_SWEEP_COVERAGE of W_ACC_COUNT sweeps,
        UNLESS the window predates the scene's first sweep, in which case it is
        truncated by construction and exempted — recorded, not failed, because
        point density (hence cluster size, hence box dimensions) differs
        systematically for the first keyframes of every scene (§1.4);
      - no gap inside the window exceeds MAX_SWEEP_GAP_NS.
    The largest gap actually present in v1.0-mini is one dropped sweep
    (100.079 ms, scene-0061); the gate sits above it on purpose and the measured
    maximum is reported so a tightening is a decision, not an accident.
    """
    lidar = sorted(
        (
            r
            for r in sub.sample_data_by_scene.get(scene["token"], [])
            if sub.channel(r) == "LIDAR_TOP"
        ),
        key=lambda r: r["timestamp"],
    )
    if not lidar:
        return PredicateResult("sweeps_cover_window", False, {"reason": "no LIDAR_TOP records"})

    timestamps = [r["timestamp"] * 1000 for r in lidar]  # us -> ns, comparison only
    scene_start_ns = timestamps[0]
    anchors = {
        r["sample_token"]: r["timestamp"] * 1000 for r in lidar if r.get("is_key_frame")
    }

    missing: list[str] = []
    short_windows: list[dict] = []
    big_gaps: list[dict] = []
    truncated = 0
    coverage: list[int] = []
    max_gap_ns = 0

    for sample in samples:
        anchor_ns = anchors.get(sample["token"])
        if anchor_ns is None:
            short_windows.append({"sample_token": sample["token"], "reason": "no LIDAR_TOP anchor"})
            continue
        window_start = anchor_ns - W_ACC_DURATION_NS
        in_window = [
            (t, r) for t, r in zip(timestamps, lidar) if window_start <= t <= anchor_ns
        ]
        coverage.append(len(in_window))

        for _, record in in_window:
            if not os.path.isfile(sub.blob(record)):
                missing.append(record["filename"])

        for (t_prev, _), (t_next, record) in zip(in_window, in_window[1:]):
            gap = t_next - t_prev
            max_gap_ns = max(max_gap_ns, gap)
            if gap > MAX_SWEEP_GAP_NS:
                big_gaps.append(
                    {"filename": record["filename"], "gap_ns": gap, "limit_ns": MAX_SWEEP_GAP_NS}
                )

        if window_start < scene_start_ns:
            truncated += 1  # expected at scene starts; exempt from the count gate
        elif len(in_window) < MIN_SWEEP_COVERAGE * W_ACC_COUNT:
            short_windows.append(
                {
                    "sample_token": sample["token"],
                    "n_sweeps_actual": len(in_window),
                    "expected": W_ACC_COUNT,
                }
            )

    ok = not missing and not short_windows and not big_gaps
    return PredicateResult(
        "sweeps_cover_window",
        ok,
        {
            "w_acc_count": W_ACC_COUNT,
            "w_acc_duration_ns": W_ACC_DURATION_NS,
            "n_keyframes_truncated_window": truncated,
            "n_sweeps_in_window_min": min(coverage) if coverage else 0,
            "n_sweeps_in_window_median": statistics.median(coverage) if coverage else 0,
            "max_gap_ns": max_gap_ns,
            "n_missing_in_window": len(missing),
            "first_missing_in_window": _sample(missing),
            "short_windows": short_windows[:5],
            "oversized_gaps": big_gaps[:5],
        },
    )


# ---------------------------------------------------------------------------
# Predicate 6 — token_graph_closed
# ---------------------------------------------------------------------------


def check_token_graph_closed(sub: Substrate, scene: dict, samples: list[dict]) -> PredicateResult:
    """Referential integrity of the token graph, per scene.

    A partially extracted metadata tarball otherwise surfaces as a `KeyError`
    inside Stage 5, nine stages after the point where it was detectable. Stage
    8's priors depend on `sample_annotation`, which rev 1's probe never opened.
    """
    problems: dict = defaultdict(list)

    # sample chain: exactly nbr_samples, terminated at both ends.
    if len(samples) != scene["nbr_samples"]:
        problems["sample_chain_length"].append(
            {"walked": len(samples), "nbr_samples": scene["nbr_samples"]}
        )
    if samples:
        if samples[0].get("prev"):
            problems["first_sample_has_prev"].append(samples[0]["token"])
        if samples[-1].get("next"):
            problems["last_sample_has_next"].append(samples[-1]["token"])
        if samples[-1]["token"] != scene["last_sample_token"]:
            problems["chain_end_mismatch"].append(samples[-1]["token"])
    else:
        problems["empty_sample_chain"].append(scene["first_sample_token"])

    sample_tokens = {s["token"] for s in samples}
    for record in samples:
        for link in ("prev", "next"):
            token = record.get(link) or ""
            if token and token not in sub.samples_by_token:
                problems[f"sample_{link}_dangling"].append(token)

    # sample_data -> ego_pose, calibrated_sensor, and its own prev/next links.
    for record in sub.sample_data_by_scene.get(scene["token"], []):
        if record["ego_pose_token"] not in sub.ego_pose_tokens:
            problems["sample_data_ego_pose_dangling"].append(record["token"])
        if record["calibrated_sensor_token"] not in sub.calibrated_sensor_tokens:
            problems["sample_data_calibrated_sensor_dangling"].append(record["token"])
        if sub.channel(record) is None:
            problems["sample_data_channel_unresolvable"].append(record["token"])
        if record["sample_token"] not in sub.samples_by_token:
            problems["sample_data_sample_dangling"].append(record["token"])
        for link in ("prev", "next"):
            token = record.get(link) or ""
            if token and token not in sub.sample_data_by_token:
                problems[f"sample_data_{link}_dangling"].append(token)

    # sample -> sample_annotation -> instance -> category.
    n_annotations = 0
    for token in sample_tokens:
        for ann in sub.annotations_by_sample.get(token, []):
            n_annotations += 1
            instance = sub.instance_by_token.get(ann["instance_token"])
            if instance is None:
                problems["annotation_instance_dangling"].append(ann["token"])
            elif instance["category_token"] not in sub.category_tokens:
                problems["instance_category_dangling"].append(instance["token"])

    return PredicateResult(
        "token_graph_closed",
        not problems,
        {
            "n_samples_walked": len(samples),
            "n_sample_data": len(sub.sample_data_by_scene.get(scene["token"], [])),
            "n_annotations": n_annotations,
            "violations": {k: {"n": len(v), "first": _sample(v)} for k, v in problems.items()},
        },
    )


# ---------------------------------------------------------------------------
# Stratification and partition
# ---------------------------------------------------------------------------


def classify_location(log_location: str) -> str:
    """Map a nuScenes log location onto the stratification axis.

    v1.0-mini's four location strings (singapore-onenorth, singapore-queenstown,
    singapore-hollandvillage, boston-seaport) collapse to two cities. An
    unrecognised string is returned verbatim so it fails the partition check
    loudly instead of being bucketed into whichever branch came first.
    """
    lowered = (log_location or "").lower()
    for city in LOCATIONS:
        if city in lowered:
            return city
    return f"unknown:{log_location}"


def classify_night(description: str) -> bool:
    return "night" in (description or "").lower()


def verify_partition(verdicts: list[SceneVerdict]) -> dict:
    """Check the §11 decision-3 partition against what the substrate actually is.

    The partition is a fixed assignment, not a computed one — it is recorded in
    the manifest and must reproduce byte-for-byte. What is computed here is
    whether that assignment still holds: disjoint, complete, every subset
    spanning both locations with at least one night scene, and every named
    scene actually usable. A scene excluded by a predicate does NOT get silently
    swapped out; the partition is reported unsatisfiable and the run stops
    short of pretending otherwise.
    """
    by_name = {v.name: v for v in verdicts}
    usable = {v.name for v in verdicts if v.usable}
    errors: list[str] = []

    assigned: list[str] = [name for names in PARTITION.values() for name in names]
    duplicates = sorted({n for n in assigned if assigned.count(n) > 1})
    if duplicates:
        errors.append(f"partition subsets are not disjoint: {duplicates}")

    unknown = sorted(set(assigned) - set(by_name))
    if unknown:
        errors.append(f"partition names scenes that are not in scene.json: {unknown}")
    unassigned = sorted(set(by_name) - set(assigned))
    if unassigned:
        errors.append(f"scenes present but unassigned: {unassigned}")

    subsets: dict = {}
    for subset, names in PARTITION.items():
        members = [by_name[n] for n in names if n in by_name]
        locations = sorted({m.location for m in members})
        nights = [m.name for m in members if m.is_night]
        excluded = sorted(n for n in names if n in by_name and n not in usable)
        subsets[subset] = {
            "scenes": list(names),
            "scene_tokens": [by_name[n].token for n in names if n in by_name],
            "locations": locations,
            "night_scenes": nights,
            "excluded_by_probe": excluded,
        }
        missing_locations = sorted(set(LOCATIONS) - set(locations))
        if missing_locations:
            errors.append(f"subset {subset!r} does not span location(s): {missing_locations}")
        if not nights:
            errors.append(f"subset {subset!r} contains no night scene")
        if excluded:
            errors.append(
                f"subset {subset!r} names scene(s) the probe excluded: {excluded}; "
                "the partition is fixed and is not re-drawn to hide an exclusion"
            )

    return {
        "satisfiable": not errors,
        "errors": errors,
        "subsets": subsets,
        "stratification": {
            "locations": list(LOCATIONS),
            "night_method": NIGHT_METHOD,
            "provenance": "pilot_plan.md §11, decision 3",
        },
    }


# ---------------------------------------------------------------------------
# Atomic output — canonical implementation now in pipeline.common.manifest (C3).
# The name stays importable from here for the stages that predate the move.
# ---------------------------------------------------------------------------

write_json_atomic = _write_json_atomic


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


def run_probe(paths: Paths, out_dir: str) -> tuple[dict, dict, int]:
    """Evaluate every predicate over every scene. Returns (allowlist, report, exit code)."""
    version = check_version_matches(paths)
    if not version.ok:
        raise HardStop(
            f"version_matches failed: configured {paths.version!r}, present under meta_root: "
            f"{version.detail['version_dirs_on_disk']}. Mini metadata against trainval blobs "
            "yields zero "
            "token matches, which would read downstream as 'no usable scenes'."
        )
    for name, directory in (("samples", paths.samples_dir), ("sweeps", paths.sweeps_dir)):
        if not os.path.isdir(directory):
            raise HardStop(f"dataroot is missing {name}/: {directory}")

    fingerprint, digests = metadata_fingerprint(paths, with_digests=True)
    sub = Substrate.load(paths)

    logs = {log["token"]: log for log in sub.tables["log.json"]}
    verdicts: list[SceneVerdict] = []

    for scene in sorted(sub.tables["scene.json"], key=lambda s: s["name"]):
        log = logs.get(scene["log_token"], {})
        samples = sub.scene_samples(scene)
        verdict = SceneVerdict(
            name=scene["name"],
            token=scene["token"],
            location=classify_location(log.get("location", "")),
            is_night=classify_night(scene.get("description", "")),
            nbr_samples=scene["nbr_samples"],
        )
        resolve, parse, blob_measurement = check_files(sub, scene)
        verdict.predicates = [
            check_channels_complete(sub, scene, samples),
            resolve,
            parse,
            check_sweeps_cover_window(sub, scene, samples),
            check_token_graph_closed(sub, scene, samples),
        ]
        verdict.measurements = {
            "description": scene.get("description", ""),
            "log_location": log.get("location", ""),
            "date_captured": log.get("date_captured", ""),
            **blob_measurement,
        }
        verdicts.append(verdict)

    usable = [v for v in verdicts if v.usable]
    if not usable:
        raise HardStop("zero usable scenes: every scene failed at least one predicate")

    partition = verify_partition(verdicts)

    manifest = SubstrateManifest(
        dataroot_realpath=paths.dataroot,
        version=paths.version,
        metadata_fingerprint=fingerprint,
        fingerprint_spec=FINGERPRINT_SPEC,
        required_channels=list(REQUIRED_CHANNELS),
        camera_subset=list(RING_CAMERAS),
        coverage_config="R2",
        usable_scene_tokens=[v.token for v in usable],
        w_acc_count=W_ACC_COUNT,
        w_acc_duration_ns=W_ACC_DURATION_NS,
    )
    errors = validate_records([manifest])
    if errors:
        # The allowlist is the one artifact every later stage reads; an invalid
        # I-1 record must not reach disk in any form.
        raise SchemaValidationError(errors, context="usable_scenes.json manifest")

    allowlist = {
        "spec": PROBE_SPEC,
        "manifest": manifest.to_dict(),
        "scenes": [
            {
                "name": v.name,
                "token": v.token,
                "location": v.location,
                "is_night": v.is_night,
                "nbr_samples": v.nbr_samples,
            }
            for v in usable
        ],
        "partition": partition,
    }

    report = {
        "spec": PROBE_SPEC,
        "predicates": list(PREDICATES),
        "paths": paths.as_dict(),
        "metadata_fingerprint": fingerprint,
        "fingerprint_spec": FINGERPRINT_SPEC,
        "metadata_file_digests": digests,
        "config": {
            "required_channels": list(REQUIRED_CHANNELS),
            "excluded_channels_note": "RADAR is measured and reported, never gated (§5.1)",
            "w_acc_count": W_ACC_COUNT,
            "w_acc_duration_ns": W_ACC_DURATION_NS,
            "nominal_sweep_period_ns": NOMINAL_SWEEP_PERIOD_NS,
            "max_sweep_gap_ns": MAX_SWEEP_GAP_NS,
            "min_sweep_coverage": MIN_SWEEP_COVERAGE,
            "point_record_bytes": POINT_RECORD_BYTES,
            "pcd_point_band": [PCD_MIN_POINTS, PCD_MAX_POINTS],
            "jpeg_byte_band": [JPEG_MIN_BYTES, JPEG_MAX_BYTES],
        },
        "global_predicates": [version.as_dict()],
        "totals": {
            "n_scenes": len(verdicts),
            "n_usable": len(usable),
            "n_excluded": len(verdicts) - len(usable),
            "n_keyframes_usable": sum(v.nbr_samples for v in usable),
            "n_night_scenes": sum(1 for v in verdicts if v.is_night),
            "scenes_by_location": {
                city: sum(1 for v in verdicts if v.location == city) for city in LOCATIONS
            },
            "failing_predicate_counts": {
                name: sum(1 for v in verdicts if name in v.failing) for name in PREDICATES
            },
        },
        "partition": partition,
        "scenes": [v.as_dict() for v in verdicts],
    }

    exit_code = EXIT_OK if len(usable) == len(verdicts) and partition["satisfiable"] else EXIT_INCOMPLETE
    return allowlist, report, exit_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"), help="path contract (§1.8; default $DHAKASCENES_PATHS_CONFIG)")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output directory; default <work_root>/stage0_data_probe (never probe_out_root)",
    )
    args = parser.parse_args(argv)

    try:
        paths = load_paths(args.paths)
    except PathValidationError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_HARD_STOP

    out_dir = args.out_dir or os.path.join(paths.work_root, "stage0_data_probe")
    allowlist_path = os.path.join(out_dir, "usable_scenes.json")
    report_path = os.path.join(out_dir, "probe_report.json")
    for target in (allowlist_path, report_path):
        assert_dataroot_read_only(paths, target)

    try:
        allowlist, report, code = run_probe(paths, out_dir)
    except HardStop as exc:
        print(f"HARD STOP: {exc}", file=sys.stderr)
        return EXIT_HARD_STOP
    except (SchemaValidationError, PathValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_HARD_STOP

    # The marker state cannot straddle two runs: clear before the first write,
    # write after the last, so a crash in between leaves NO marker (§1.9, C16).
    clear_markers(out_dir)
    write_json_atomic(report_path, report)
    write_json_atomic(allowlist_path, allowlist)
    causes = [
        f"{v['name']}: failing {','.join(v['failing_predicates'])}"
        for v in report["scenes"]
        if not v["usable"]
    ] + [f"partition: {e}" for e in report["partition"]["errors"]]
    if code != EXIT_OK and not causes:
        causes = [f"probe exited {code} without a per-scene cause; see {report_path}"]
    write_marker(
        out_dir,
        report["metadata_fingerprint"],
        degraded=code != EXIT_OK,
        causes=causes,
    )

    totals = report["totals"]
    print(f"metadata fingerprint : {report['metadata_fingerprint']}")
    print(f"scenes usable        : {totals['n_usable']}/{totals['n_scenes']}"
          f"  keyframes {totals['n_keyframes_usable']}")
    for verdict in report["scenes"]:
        flag = "ok  " if verdict["usable"] else "FAIL"
        night = "night" if verdict["is_night"] else "day  "
        print(
            f"  {flag} {verdict['name']}  {verdict['location']:<9} {night} "
            f"{verdict['nbr_samples']:>3} kf"
            + (f"  failing: {','.join(verdict['failing_predicates'])}" if not verdict["usable"] else "")
        )
    partition = report["partition"]
    state = "satisfiable" if partition["satisfiable"] else "UNSATISFIABLE"
    print(f"partition            : {state}")
    for subset, block in partition["subsets"].items():
        print(
            f"  {subset:<7} {len(block['scenes'])} scenes  "
            f"locations={'+'.join(block['locations'])}  night={len(block['night_scenes'])}"
        )
    for error in partition["errors"]:
        print(f"  ! {error}", file=sys.stderr)
    print(f"wrote {allowlist_path}")
    print(f"wrote {report_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
