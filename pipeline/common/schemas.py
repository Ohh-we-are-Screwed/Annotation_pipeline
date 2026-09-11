"""Contract records I-1 … I-5, their validators, and the raising write boundary (§1.7, §3).

Every record that crosses a stage boundary is defined here, once. Three rules
shape the whole module:

  1. **`validate() -> [errors]` is kept**, because stages legitimately need
     check-then-decide (one scene's failure lands in `failures.json`; it does
     not abort the run). **`write_records()` raises**, because rev 1's
     non-raising design made "just don't inspect the returned list" the easiest
     bypass of the invariant it existed to protect. All persistence goes through
     `write_records()`, and it validates on write AND on read-back —
     serialization was the open path: records round-tripping through plain
     dicts get reconstructed downstream with the validator never consulted.
  2. **Every geometric and temporal field carries a unit suffix** (`_m`, `_rad`,
     `_ns`, `_mps`) and every geometric record carries an explicit `frame`
     (§1.1 rule 2, §1.2 rule 5). This also settles the inherited
     radians/degrees inconsistency: the pilot stores `sigma_yaw_rad` and
     converts at the release boundary only.
  3. **Absent quality is `None`, never zero.** nuScenes supplies none of I-2's
     quality fields. Writing `sigma_pos_m = 0.0` to satisfy a dataclass makes
     every consumer read the pose as *better than PPK* (§3.1). Consumers fail
     closed on `None` via `require_*()` accessors; they do not default.

Deliberately shape-only, with no producer and no consumer, declared rather than
silently dropped (§4): `GroupAnnotation`, and the `visibility` field of
`AnnotationRecord`. `attribute` left that list on 2026-09-07 (§7) — it has a
producer in the human-annotation loop and nowhere else, so it is still refused
on a pipeline-sourced record.

No GPU, no models: numpy is not even needed here — stdlib only, so a record can
be validated anywhere, including inside the substrate probe.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import MISSING, dataclass, field, fields, is_dataclass
from typing import Any, Iterable, Sequence

from pipeline.common.conventions import (
    EGO,
    FRAMES,
    INTERNAL_TIME_BASE,
    NUSCENES_GLOBAL,
    TIME_BASES,
    yaw_rad_from_quaternion,
)
from pipeline.common.eval_region import COVERAGE_CONFIGS

__all__ = [
    "SCHEMA_VERSION",
    "SOURCES",
    "TIERS",
    "SPLITS",
    "CLOUD_KINDS",
    "ANNOTATOR_PASSES",
    "ATTRIBUTE_NAMES_NUSCENES",
    "REQUIRED_CHANNELS",
    "RING_CAMERAS",
    "IMAGE_WIDTH_PX",
    "IMAGE_HEIGHT_PX",
    "SUBSTRATE",
    "SUBSTRATE_PROFILES",
    "W_ACC_COUNT",
    "W_ACC_DURATION_NS",
    "PCD_POINT_BAND",
    "JPEG_BYTE_BAND",
    "STEREO_RINGS",
    "STEREO_STRIDE",
    "GROUND_Z_BAND_M",
    "GROUND_FIT_RINGS",
    "GROUND_FIT_RANGE_M",
    "ProvenancePolicy",
    "POLICY",
    "SchemaValidationError",
    "Record",
    "SubstrateManifest",
    "EgoPoseRecord",
    "CameraObservation",
    "CloudArtifact",
    "KeyframeRecord",
    "GateVector",
    "Provenance",
    "AnnotationRecord",
    "GroupAnnotation",
    "RECORD_TYPES",
    "validate_records",
    "write_records",
    "read_records",
]

SCHEMA_VERSION = "dhakascenes-pilot/schemas/v1"

# --- enumerations (§A.1, §1.7) ---------------------------------------------

# `source` and `tier` are DIFFERENT fields with DIFFERENT rules. Rev 1's prose
# conflated `tier: auto_accept` with `source: pipeline_accepted`; neither
# implies the other and nothing here couples them (§1.7 rule 3).
SOURCES: tuple[str, ...] = ("pipeline", "pipeline_accepted", "human_verified", "human_created")
HUMAN_SOURCES: tuple[str, ...] = ("human_verified", "human_created")
TIERS: tuple[str, ...] = ("auto_accept", "flagged", "rejected")
SPLITS: tuple[str, ...] = ("train", "val", "test")
CLOUD_KINDS: tuple[str, ...] = ("single_sweep", "accumulated")

# The human-annotation loop's two vocabularies (spec 2026-09-07 §7).
#
# `ATTRIBUTE_NAMES_NUSCENES` restates `pipeline.release.attributes.ATTRIBUTE_NAMES`
# rather than importing it, deliberately: schemas.py is the persistence boundary
# and is stdlib-only on purpose (module docstring, rule "no GPU, no models"), so
# that a record can be validated anywhere — including inside the substrate probe,
# where the release package's numpy dependency is not available. The duplication
# is guarded by a test that asserts the two tuples still agree.
ANNOTATOR_PASSES: tuple[str, ...] = ("A", "B")
ATTRIBUTE_NAMES_NUSCENES: tuple[str, ...] = (
    "vehicle.moving",
    "vehicle.stopped",
    "vehicle.parked",
    "pedestrian.moving",
    "pedestrian.standing",
    "pedestrian.sitting_lying_down",
    "cycle.with_rider",
    "cycle.without_rider",
)

# The required-channel set: I-1's analogue of "a bag missing a mandatory topic
# fails ingestion" (§5.1). RADAR is EXCLUDED, declared: including it would drop
# scenes for reasons the pipeline does not care about, and excluding it silently
# would make the allowlist not mean what its name says.
#
# THE SUBSTRATE PROFILE (added 2026-09-03)
# ---------------------------------------
# The ring and the image pin below are properties of the RIG, not of the
# pipeline, and this project now has two rigs. They were hard-coded to the
# GA-01 Dhaka rig, and a v1.0-mini run walks straight into that: Stage 0
# excludes all ten scenes on `channels_complete` (nuScenes has no CAM_LEFT or
# CAM_RIGHT), and had it not, Stage 5's `!= (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX)`
# assertion would have refused the 1600x900 images one stage later.
#
# So the pair is selected by DHAKASCENES_SUBSTRATE, resolved ONCE at import.
# The default is "dhaka" — every existing run, script, archived number and
# committed test keeps meaning exactly what it meant, and a reader who sets
# nothing gets no change at all. An unknown value RAISES rather than falling
# back: a typo that silently reinstates the Dhaka ring over nuScenes imagery
# would produce a full run whose allowlist is empty for a reason nobody sees.
#
# This does NOT make a mixed tree safe. It makes a mixed tree DETECTABLE: the
# probe already writes `required_channels` into its manifest and Stages 3 and 5
# already write the image size into theirs, so the profile a tree was built
# under is on disk in every case. Two profiles' outputs are not comparable and
# must not share a work_root.
SUBSTRATE_PROFILES: dict[str, dict[str, Any]] = {
    # The GA-01 rig carries EIGHT cameras. The nuScenes six leave two blind
    # wedges at the sides -- measured on v1.0-dhaka-fixed: 75.5..104.4 deg
    # (29.0 deg) and -100.1..-80.0 deg (20.2 deg), i.e. 86.3% of azimuth, while
    # coverage_config R2 claims a full annulus. CAM_LEFT (yaw 90) and CAM_RIGHT
    # (yaw -90) close both exactly, taking the ring to 100.0%. Adding them made
    # R2 honest rather than redefining it, but it DID change what "the full
    # ring" counts as: runs before and after that line are not directly
    # comparable.
    # Image pin measured on v1.0-dhaka-fixed, 2026-08-30: all ring cameras, all
    # 2303 keyframes, 1280x720.
    "dhaka": {
        "ring_cameras": (
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_LEFT",
            "CAM_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ),
        "image_width_px": 1280,
        "image_height_px": 720,
        # Accumulation window (§11 decision 2: duration preserved, not count):
        # 0.5 s = 5 sweeps at the measured 10.00 Hz (v1.0-dhaka-fixed).
        "w_acc_count": 5,
        "w_acc_duration_ns": 500_000_000,
        # Stage 0 parse bands (§5.1 predicate 3). Measured on the pilot:
        # 34,368-34,816 points per cloud; JPEGs 20 kB-4 MB.
        "pcd_point_band": (10_000, 300_000),
        "jpeg_byte_band": (20_000, 4_000_000),
        # Stereo arrives as separate ZED channels that Stage 1 fuses itself
        # (rings 10/11); nothing in LIDAR_TOP is thinned.
        "stereo_rings": (),
        "stereo_stride": 1,
        # Stage 1 RANSAC ground candidates: ISO 8855 ego frame, z=0 at ground.
        "ground_z_band_m": (-1.5, 1.5),
        # Every ring, every range is a candidate (the pilot's behaviour).
        "ground_fit_rings": (),
        "ground_fit_range_m": None,
    },
    # Stock nuScenes: the six-camera ring, 1600x900. Verified on v1.0-mini,
    # 2026-09-03 — all 6 channels present on all 404 keyframes, every
    # sample_data row 1600x900. The two side wedges the Dhaka note measures are
    # simply UNCOVERED here; that is the substrate, not a defect to gate on.
    "nuscenes": {
        "ring_cameras": (
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ),
        "image_width_px": 1600,
        "image_height_px": 900,
        # Same 0.5 s window the pilot ran with on v1.0-mini (20 Hz: ~10 records
        # in the window, gate is 0.8 x 5). Unchanged so the archived run
        # keeps meaning what it meant.
        "w_acc_count": 5,
        "w_acc_duration_ns": 500_000_000,
        # The pilot's bands, under which v1.0-mini passed on 2026-09-03.
        "pcd_point_band": (10_000, 300_000),
        "jpeg_byte_band": (20_000, 4_000_000),
        "stereo_rings": (),
        "stereo_stride": 1,
        "ground_z_band_m": (-1.5, 1.5),
        "ground_fit_rings": (),
        "ground_fit_range_m": None,
    },
    # The 2026-09-04 Dhaka capture (Dataset/A_nusc, "day 1"): the GA-01 ring
    # WITHOUT its two rear corners — six cameras, and a different six from
    # nuScenes (CAM_LEFT/CAM_RIGHT in, CAM_BACK_LEFT/CAM_BACK_RIGHT out).
    # Verified 2026-09-06 on chunk_0000: every channel present on every sample
    # (CAM_FRONT_RIGHT short by one frame), all 1280x720. LIDAR_TOP arrives
    # already fused (Mid-360 rings 0-3 + ZED as ring 100+k); there is no
    # separate ZED channel, so Stage 1's ring-10/11 merge and the road arm's
    # ZED refinement are simply inactive under this profile, not wrong.
    "dhaka6": {
        "ring_cameras": (
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_LEFT",
            "CAM_RIGHT",
            "CAM_BACK",
        ),
        "image_width_px": 1280,
        "image_height_px": 720,
        # NO SWEEPS: the exporter kept every 2nd LiDAR frame as a keyframe and
        # wrote nothing between them, so the accumulation window is the anchor
        # alone. Declared rather than tolerated: Stage 0's sweeps_cover_window
        # and Stage 1's accumulation both read this, and with 5 / 0.5 s they
        # refused every chunk (1-2 records per window, 2026-09-06). Density is
        # not the concern it was for the pilot — the fused cloud already runs
        # to ~380k points per sample.
        "w_acc_count": 1,
        "w_acc_duration_ns": 0,
        # Parse bands measured 2026-09-06 over ALL 4,536 clouds and 26,799
        # JPEGs of Dataset/A_nusc: 37,920-490,859 points (fused LiDAR + two
        # ZED depth clouds), 15,027-428,741 bytes (the yuyv-re-encoded side
        # cameras are small: 925 JPEGs sit under the pilot's 20 kB floor).
        # Under the pilot bands 93 of chunk_0006's 94 clouds failed. Headroom
        # above the extremes, because the band exists to catch a truncated
        # file, not to pin a density.
        "pcd_point_band": (10_000, 1_000_000),
        "jpeg_byte_band": (10_000, 4_000_000),
        # LIDAR_TOP arrives FUSED: Mid-360 rings 0-3 (39,936 points/frame) plus
        # the two ZED depth clouds as rings 100/101 (350,595 points — 8.8x the
        # LiDAR, measured 2026-09-06). One parked car in front of a ZED painted
        # ~38k points per keyframe and Stage 6's DBSCAN neighbour graph on that
        # single instance reached 53 GB RSS + 14 GB swap and never finished.
        # Stage 1 keeps every 8th point of each stereo ring — deterministic,
        # file order, recorded in its manifest — so stereo density lands near
        # the LiDAR's (~44k) and the largest instance near ~5k points.
        "stereo_rings": (100, 101),
        "stereo_stride": 8,
        # The ego origin IS the LiDAR (calibrated_sensor translation 0,0,0),
        # mounted ~2.3 m up. Measured on chunk_0000 keyframe 100: road at
        # z = -2.0..-2.75 (LiDAR rings), -2.5..-3.0 (ZED, biased low). With the
        # ISO 8855 band the ground was never a candidate; RANSAC fit planes at
        # -0.3..-0.6 m through the scene, the removal slab cut objects at
        # mid-height (heights 55-65 % of true) and the surviving road points
        # elongated 84-93 % of boxes along the viewing ray (2026-09-06).
        "ground_z_band_m": (-3.5, -1.0),
        # Who may vote for the ground (operator decision, 2026-09-06 15:40):
        # the Mid-360 rings and the FRONT ZED (ring 101), which agree on the
        # road to ~0.2 m, at 3-12 m where the stereo is dense and reliable.
        # The REAR ZED (ring 100) puts the road 0.69 m lower at its own camera
        # and sinks further with range (stereo noise) — a miscalibration the
        # exporter baked into LIDAR_TOP — so it never votes; its points are
        # still filtered by the plane like everything else.
        "ground_fit_rings": (0, 1, 2, 3, 101),
        "ground_fit_range_m": (3.0, 12.0),
    },
}

SUBSTRATE: str = os.environ.get("DHAKASCENES_SUBSTRATE", "dhaka").strip().lower() or "dhaka"
if SUBSTRATE not in SUBSTRATE_PROFILES:
    raise ValueError(
        f"DHAKASCENES_SUBSTRATE={SUBSTRATE!r} is not a known substrate profile; "
        f"expected one of {sorted(SUBSTRATE_PROFILES)}. Refusing to guess: the wrong "
        "profile silently changes the required-channel ring and the image pin."
    )
_PROFILE = SUBSTRATE_PROFILES[SUBSTRATE]

RING_CAMERAS: tuple[str, ...] = _PROFILE["ring_cameras"]
REQUIRED_CHANNELS: tuple[str, ...] = ("LIDAR_TOP",) + RING_CAMERAS

# Every 2D quantity crossing a stage boundary is absolute pixels at THIS
# resolution (§1.5 rule 1) — not normalised, not at a model's input scale.
IMAGE_WIDTH_PX: int = _PROFILE["image_width_px"]
IMAGE_HEIGHT_PX: int = _PROFILE["image_height_px"]
# Accumulation window — a substrate property since 2026-09-06 (it was a
# literal in probe.py and IngestConfig). Stage 0 gates on it, Stage 1 builds
# the accumulated cloud with it; the two must agree, and they agree here.
W_ACC_COUNT: int = _PROFILE["w_acc_count"]
W_ACC_DURATION_NS: int = _PROFILE["w_acc_duration_ns"]
# Stage 0 parse bands — (min, max) inclusive. Profile-owned since 2026-09-06:
# they catch a half-written blob, and what "half" means depends on the rig.
PCD_POINT_BAND: tuple[int, int] = tuple(_PROFILE["pcd_point_band"])
JPEG_BYTE_BAND: tuple[int, int] = tuple(_PROFILE["jpeg_byte_band"])
# Stereo thinning at ingestion (2026-09-06): rings of LIDAR_TOP that carry
# fused stereo depth, and the stride Stage 1 keeps them at. () / 1 = none.
STEREO_RINGS: tuple[int, ...] = tuple(_PROFILE["stereo_rings"])
STEREO_STRIDE: int = int(_PROFILE["stereo_stride"])
# Stage 1 ground-plane candidate band in the ego frame (2026-09-06): where the
# road can be, given where the ego origin sits on this rig.
GROUND_Z_BAND_M: tuple[float, float] = tuple(float(v) for v in _PROFILE["ground_z_band_m"])
# Which rings may be RANSAC ground candidates (() = all) and within what
# radial window (None = any). Profile-owned since 2026-09-06.
GROUND_FIT_RINGS: tuple[int, ...] = tuple(int(v) for v in _PROFILE["ground_fit_rings"])
GROUND_FIT_RANGE_M = None if _PROFILE["ground_fit_range_m"] is None else tuple(float(v) for v in _PROFILE["ground_fit_range_m"])
# LiDAR point record: 5 x float32 (x, y, z, intensity, ring). Measured.
POINT_RECORD_BYTES = 20

# Camera-to-LiDAR-anchor offsets measured over all 404 keyframes span
# [-48.35, +1.20] ms (§1.2). The per-camera gate lives in config with that
# provenance; this is only the outer sanity bound, wide enough that it fires on
# a unit error rather than on a distribution the pilot has already measured.
MAX_ABS_CAMERA_DT_NS = 100_000_000


@dataclass(frozen=True)
class ProvenancePolicy:
    """Pilot-wide provenance policy (§1.7 rule 2).

    The pilot has no human annotators. `allow_human_provenance=False` makes any
    human-provenance record a hard error rather than an unremarkable value that
    a later release build would treat as verified ground truth.
    """

    allow_human_provenance: bool = False


POLICY = ProvenancePolicy()


class SchemaValidationError(ValueError):
    """Raised by the write/read boundary. Carries every error, not the first."""

    def __init__(self, errors: Sequence[str], *, context: str | None = None) -> None:
        self.errors = list(errors)
        head = f"{len(self.errors)} schema violation(s)"
        if context:
            head += f" in {context}"
        super().__init__(head + ":\n" + "\n".join(f"  - {e}" for e in self.errors))


# ---------------------------------------------------------------------------
# Field-level checks
# ---------------------------------------------------------------------------


def _err(out: list[str], prefix: str, message: str) -> None:
    out.append(f"{prefix}{message}")


def _check_str(out, p, name, value, *, allowed=None, allow_none=False) -> None:
    if value is None:
        if not allow_none:
            _err(out, p, f"{name} is required")
        return
    if not isinstance(value, str) or not value:
        _err(out, p, f"{name} must be a non-empty string, got {value!r}")
    elif allowed is not None and value not in allowed:
        _err(out, p, f"{name}={value!r} is not one of {tuple(allowed)}")


def _check_frame(out, p, value, *, expected: str | None = None) -> None:
    if value is None:
        _err(out, p, "frame is required; a geometric record with no frame is invalid")
    elif value not in FRAMES:
        _err(out, p, f"frame={value!r} is not one of {FRAMES}")
    elif expected is not None and value != expected:
        _err(out, p, f"frame must be {expected!r} for this record, got {value!r}")


def _check_time(out, p, t_ns, time_base) -> None:
    if time_base not in TIME_BASES:
        _err(out, p, f"time_base={time_base!r} is not one of {TIME_BASES}")
    elif time_base != INTERNAL_TIME_BASE:
        _err(
            out,
            p,
            f"time_base={time_base!r}: persisted records are {INTERNAL_TIME_BASE} "
            "(convert once, in conventions.to_unix_ns)",
        )
    if isinstance(t_ns, bool) or not isinstance(t_ns, int):
        _err(out, p, f"t_ns must be an int (nanoseconds), got {type(t_ns).__name__}")
    elif t_ns <= 0:
        _err(out, p, f"t_ns must be positive, got {t_ns}")
    elif t_ns < 10**18:
        # 1e18 ns = 2001-09-09. A microsecond value stored unconverted lands
        # here; this is the check that catches the 1000x error at the boundary.
        _err(out, p, f"t_ns={t_ns} is too small for a Unix nanosecond timestamp (unit error?)")


def _check_vec(out, p, name, value, n, *, positive=False, allow_none=False) -> None:
    if value is None:
        if not allow_none:
            _err(out, p, f"{name} is required")
        return
    if not isinstance(value, (list, tuple)) or len(value) != n:
        _err(out, p, f"{name} must have {n} components, got {value!r}")
        return
    for i, v in enumerate(value):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            _err(out, p, f"{name}[{i}] must be a finite number, got {v!r}")
        elif positive and float(v) <= 0.0:
            _err(out, p, f"{name}[{i}] must be positive, got {v!r}")


def _check_quaternion(out, p, name, value) -> None:
    _check_vec(out, p, name, value, 4)
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            norm = math.sqrt(sum(float(v) * float(v) for v in value))
        except (TypeError, ValueError):
            return
        if abs(norm - 1.0) > 1e-3:
            _err(out, p, f"{name} must be unit-norm [w, x, y, z], |q|={norm:.6g}")


def _check_number(out, p, name, value, *, allow_none=False, minimum=None, maximum=None) -> None:
    if value is None:
        if not allow_none:
            _err(out, p, f"{name} is required")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _err(out, p, f"{name} must be a finite number, got {value!r}")
        return
    if minimum is not None and float(value) < minimum:
        _err(out, p, f"{name} must be >= {minimum}, got {value!r}")
    if maximum is not None and float(value) > maximum:
        _err(out, p, f"{name} must be <= {maximum}, got {value!r}")


def _check_int(out, p, name, value, *, minimum=None, allow_none=False) -> None:
    if value is None:
        if not allow_none:
            _err(out, p, f"{name} is required")
        return
    if isinstance(value, bool) or not isinstance(value, int):
        _err(out, p, f"{name} must be an int, got {type(value).__name__}")
    elif minimum is not None and value < minimum:
        _err(out, p, f"{name} must be >= {minimum}, got {value}")


def _check_bool(out, p, name, value, *, allow_none=False) -> None:
    if value is None:
        if not allow_none:
            _err(out, p, f"{name} is required")
    elif not isinstance(value, bool):
        _err(out, p, f"{name} must be a bool, got {type(value).__name__}")


def _check_coverage(out, p, value) -> None:
    _check_str(out, p, "coverage_config", value, allowed=COVERAGE_CONFIGS)


# ---------------------------------------------------------------------------
# Base record
# ---------------------------------------------------------------------------


@dataclass
class Record:
    """Base for every contract record.

    `contract` is the I-n tag; it is written into every serialised record so
    `read_records()` reconstructs the right type instead of trusting the caller
    to remember what a file holds.
    """

    contract: str = field(init=False, default="", repr=False)

    def validate(self, *, policy: ProvenancePolicy = POLICY, prefix: str = "") -> list[str]:
        raise NotImplementedError

    def to_dict(self) -> dict:
        payload = _as_plain(self)
        payload["__contract__"] = self.contract
        payload["__schema_version__"] = SCHEMA_VERSION
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "Record":
        if not isinstance(payload, dict):
            raise SchemaValidationError([f"record must be a mapping, got {type(payload).__name__}"])
        data = {k: v for k, v in payload.items() if not k.startswith("__")}
        target = cls
        tag = payload.get("__contract__")
        if cls is Record:
            if tag not in RECORD_TYPES:
                raise SchemaValidationError([f"unknown or missing __contract__: {tag!r}"])
            target = RECORD_TYPES[tag]
        elif tag is not None and tag != cls.contract_tag():
            raise SchemaValidationError(
                [f"record is {tag!r} but {cls.__name__} was expected ({cls.contract_tag()!r})"]
            )
        return _build(target, data)

    @classmethod
    def contract_tag(cls) -> str:
        return getattr(cls, "_contract_tag", "")


def _as_plain(obj: Any) -> Any:
    """dataclass -> JSON-ready dict, without dataclasses.asdict's tuple mangling."""
    if is_dataclass(obj) and not isinstance(obj, type):
        out = {}
        for f in fields(obj):
            if f.name == "contract":
                continue
            out[f.name] = _as_plain(getattr(obj, f.name))
        return out
    if isinstance(obj, dict):
        return {str(k): _as_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_as_plain(v) for v in obj]
    return obj


_NESTED_TYPES: dict[str, dict[str, Any]] = {}


def _build(target: type, data: dict) -> Any:
    """Reconstruct a dataclass, recursing into declared nested record types."""
    nested = _NESTED_TYPES.get(target.__name__, {})
    known = {f.name for f in fields(target) if f.init}
    unknown = sorted(set(data) - known)
    if unknown:
        raise SchemaValidationError([f"{target.__name__}: unknown field(s) {unknown}"])
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        spec = nested.get(name)
        if spec is None or value is None:
            kwargs[name] = value
        elif isinstance(spec, tuple):  # a mapping of channel -> nested record
            sub = spec[0]
            kwargs[name] = {k: _build(sub, v) for k, v in value.items()}
        else:
            kwargs[name] = _build(spec, value)
    missing = sorted(_required(target) - set(kwargs))
    if missing:
        raise SchemaValidationError([f"{target.__name__}: missing field(s) {missing}"])
    try:
        return target(**kwargs)
    except TypeError as exc:
        raise SchemaValidationError([f"{target.__name__}: {exc}"]) from exc


def _required(target: type) -> set[str]:
    """Fields a caller must supply: no default, no default factory."""
    return {
        f.name
        for f in fields(target)
        if f.init and f.default is MISSING and f.default_factory is MISSING
    }


# ---------------------------------------------------------------------------
# I-1 — capture rig -> everything
# ---------------------------------------------------------------------------


@dataclass
class SubstrateManifest(Record):
    """I-1: the pilot's `session_meta.json` analogue (§3, §5.1).

    nuScenes has no MCAP bundle and no session metadata, so I-1's identity role
    is carried by the dataroot fingerprint: the exact bytes of the 13 metadata
    tables, the declared required-channel set, and the accumulation window —
    each of which changes the answer of every stage downstream.
    """

    _contract_tag = "I-1"

    dataroot_realpath: str
    version: str
    metadata_fingerprint: str
    fingerprint_spec: str
    required_channels: list
    camera_subset: list
    coverage_config: str
    usable_scene_tokens: list
    w_acc_count: int
    w_acc_duration_ns: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract", self._contract_tag)

    def validate(self, *, policy: ProvenancePolicy = POLICY, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        _check_str(e, p, "dataroot_realpath", self.dataroot_realpath)
        if isinstance(self.dataroot_realpath, str) and not os.path.isabs(self.dataroot_realpath):
            _err(e, p, "dataroot_realpath must be absolute (§1.8 records the realpath)")
        _check_str(e, p, "version", self.version)
        _check_str(e, p, "metadata_fingerprint", self.metadata_fingerprint)
        if isinstance(self.metadata_fingerprint, str) and (
            len(self.metadata_fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in self.metadata_fingerprint)
        ):
            _err(e, p, "metadata_fingerprint must be 64 lowercase hex characters (SHA-256)")
        _check_str(e, p, "fingerprint_spec", self.fingerprint_spec)
        _check_coverage(e, p, self.coverage_config)

        if not isinstance(self.required_channels, (list, tuple)) or not self.required_channels:
            _err(e, p, "required_channels must be a non-empty list")
        if not isinstance(self.camera_subset, (list, tuple)) or not self.camera_subset:
            _err(e, p, "camera_subset must be a non-empty list")
        else:
            unknown = sorted(set(self.camera_subset) - set(RING_CAMERAS))
            if unknown:
                _err(e, p, f"camera_subset contains unknown channel(s): {unknown}")
            # §1.10 / §11 decision 1: a run on fewer than the full ring is a
            # DIFFERENT experiment and must say so. R2 with one camera would
            # publish full-ring claims off a 70-degree frustum.
            full_ring = set(self.camera_subset) == set(RING_CAMERAS)
            if self.coverage_config == "R2" and not full_ring:
                _err(
                    e,
                    p,
                    "coverage_config=R2 requires the full six-camera ring; a reduced "
                    "camera_subset must record coverage_config=R1",
                )
            if self.coverage_config == "R1" and full_ring:
                _err(
                    e,
                    p,
                    "coverage_config=R1 with the full ring: R1 scopes every claim to the "
                    "frontal region and would discard five cameras' coverage silently",
                )

        if not isinstance(self.usable_scene_tokens, (list, tuple)):
            _err(e, p, "usable_scene_tokens must be a list")
        elif not self.usable_scene_tokens:
            _err(e, p, "usable_scene_tokens is empty: zero usable scenes is a hard stop (§5.1)")

        # Both the count and the duration, because preserving one changes the
        # other at a different sweep rate and "accumulated" then means something
        # else (§11, decision 2).
        _check_int(e, p, "w_acc_count", self.w_acc_count, minimum=1)
        # 0 is a declared value since 2026-09-06 (dhaka6: the anchor alone, no
        # sweeps exported); the accumulated CloudArtifact rule below already
        # carries the (1 sweep, window 0) case as the truncated-window shape.
        _check_int(e, p, "w_acc_duration_ns", self.w_acc_duration_ns, minimum=0)
        return e


# ---------------------------------------------------------------------------
# I-2 — trajectory -> ingestion, release
# ---------------------------------------------------------------------------


@dataclass
class EgoPoseRecord(Record):
    """I-2, honestly downgraded (§3.1).

    The spec's I-2 is PPK+RTS output with per-frame uncertainty. nuScenes gives
    translation, rotation and a timestamp and NOTHING else. The three quality
    fields are therefore `None`, `pose_source` and `quality_known` are
    mandatory, and consumers fail closed through `require_sigma_pos_m()` rather
    than reading a fabricated zero as sub-centimetre truth.

    `stationary_flag` is the one quality field that IS honestly derivable — from
    ego speed, with `can_bus/` on disk for a better source. `fix_type` is not
    derivable and stays `None`.
    """

    _contract_tag = "I-2"

    token: str
    t_ns: int
    time_base: str
    frame: str  # the frame the translation is EXPRESSED IN: nuscenes_global
    translation_m: list
    rotation_wxyz: list
    pose_source: str = "nuscenes_ego_pose"
    quality_known: bool = False
    sigma_pos_m: float | None = None
    sigma_yaw_rad: float | None = None
    fix_type: str | None = None
    stationary_flag: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract", self._contract_tag)

    @property
    def yaw_rad(self) -> float:
        return yaw_rad_from_quaternion(self.rotation_wxyz)

    def require_sigma_pos_m(self) -> float:
        """Fail closed. A stage that needs a sigma and finds None errors here."""
        if self.sigma_pos_m is None:
            raise SchemaValidationError(
                [
                    f"ego_pose {self.token}: sigma_pos_m is null (pose_source="
                    f"{self.pose_source!r}, quality_known={self.quality_known}); this substrate "
                    "has no pose uncertainty and a consumer must not substitute a default"
                ]
            )
        return float(self.sigma_pos_m)

    def require_sigma_yaw_rad(self) -> float:
        if self.sigma_yaw_rad is None:
            raise SchemaValidationError(
                [f"ego_pose {self.token}: sigma_yaw_rad is null; consumers fail closed (§3.1)"]
            )
        return float(self.sigma_yaw_rad)

    def validate(self, *, policy: ProvenancePolicy = POLICY, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        _check_str(e, p, "token", self.token)
        _check_time(e, p, self.t_ns, self.time_base)
        _check_frame(e, p, self.frame, expected=NUSCENES_GLOBAL)
        _check_vec(e, p, "translation_m", self.translation_m, 3)
        _check_quaternion(e, p, "rotation_wxyz", self.rotation_wxyz)
        _check_str(e, p, "pose_source", self.pose_source)
        _check_bool(e, p, "quality_known", self.quality_known)
        _check_bool(e, p, "stationary_flag", self.stationary_flag, allow_none=True)
        _check_str(e, p, "fix_type", self.fix_type, allow_none=True)
        _check_number(e, p, "sigma_pos_m", self.sigma_pos_m, allow_none=True, minimum=0.0)
        _check_number(e, p, "sigma_yaw_rad", self.sigma_yaw_rad, allow_none=True, minimum=0.0)

        # The mechanised downgrade, both directions.
        if self.quality_known is False:
            for name in ("sigma_pos_m", "sigma_yaw_rad", "fix_type"):
                if getattr(self, name) is not None:
                    _err(
                        e,
                        p,
                        f"quality_known=False but {name} is populated: this substrate supplies "
                        "no pose quality, and a value here would be fabricated",
                    )
        elif self.quality_known is True:
            for name in ("sigma_pos_m", "sigma_yaw_rad", "fix_type"):
                if getattr(self, name) is None:
                    _err(e, p, f"quality_known=True requires {name} to be populated")
        if self.sigma_pos_m == 0.0 or self.sigma_yaw_rad == 0.0:
            _err(
                e,
                p,
                "sigma of exactly 0.0 is rejected: absent uncertainty is null, and zero reads "
                "downstream as better than PPK (§3.1)",
            )
        return e


# ---------------------------------------------------------------------------
# I-3 — ingestion -> annotation
# ---------------------------------------------------------------------------


@dataclass
class CameraObservation:
    """One camera's contribution to a keyframe.

    `dt_ns` is the MEASURED camera-minus-LiDAR-anchor offset, not a nominal
    value: every camera fires before the anchor, and by how much is a
    deterministic function of where it sits in the LiDAR's rotation (§1.2).
    """

    channel: str
    sample_data_token: str
    path: str
    dt_ns: int
    ego_pose_token: str
    calibrated_sensor_token: str
    width_px: int = IMAGE_WIDTH_PX
    height_px: int = IMAGE_HEIGHT_PX
    undistorted: bool = True
    undistortion_method: str = "nuscenes_native"

    def validate(self, *, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        _check_str(e, p, "channel", self.channel, allowed=RING_CAMERAS)
        for name in ("sample_data_token", "path", "ego_pose_token", "calibrated_sensor_token"):
            _check_str(e, p, name, getattr(self, name))
        _check_int(e, p, "dt_ns", self.dt_ns)
        if isinstance(self.dt_ns, int) and not isinstance(self.dt_ns, bool):
            if abs(self.dt_ns) > MAX_ABS_CAMERA_DT_NS:
                _err(
                    e,
                    p,
                    f"dt_ns={self.dt_ns} exceeds {MAX_ABS_CAMERA_DT_NS} ns; the measured "
                    "camera-to-anchor span on this substrate is [-48.35, +1.20] ms",
                )
            # dt_ns == 0 WAS rejected here, as "no camera is synchronous with the
            # LiDAR anchor (§1.2)". Removed 2026-09-11: the claim is about how the
            # rig is BUILT — no camera is hardware-triggered off the LiDAR — and an
            # exact integer tie is not evidence against it. Timestamps are stored in
            # MICROSECONDS, so dt_ns is quantised to 1000 ns, and an asynchronous
            # camera still lands on the anchor's exact microsecond by chance.
            #
            # Measured on the 2026-09-11 Dhaka export (15,346 keyframes x 6 cameras):
            #   camera keyframes            92,076
            #   dt span                     -36.19 ms .. +45.70 ms  (81,895 us wide)
            #   exact ties expected, uniform  1.12
            #   exact ties observed           1      (chunk_0007, CAM_FRONT_RIGHT)
            # One tie in 92,076 is what chance predicts. The old check turned that
            # coincidence into rc=2 REFUSED after twelve minutes of ingestion, and it
            # gets MORE likely on every larger substrate: the 2,424-image pilot this
            # rule was written against needed ~38x more keyframes to expect one.
            #
            # What the rule was really guarding — a camera silently TREATED as the
            # anchor — is systematic, not singular: it would zero every camera on
            # every keyframe, tens of thousands of rows, not one. A per-record
            # equality test cannot tell those two apart and fires only on the
            # harmless one. The magnitude ceiling above still bounds dt_ns, which is
            # the check that catches a genuinely wrong association.
        # §1.5 rule 1 / rule 4: original resolution, asserted rather than assumed.
        if self.width_px != IMAGE_WIDTH_PX or self.height_px != IMAGE_HEIGHT_PX:
            _err(
                e,
                p,
                f"image is {self.width_px}x{self.height_px}; every 2D quantity crossing a stage "
                f"boundary is absolute pixels at {IMAGE_WIDTH_PX}x{IMAGE_HEIGHT_PX}",
            )
        _check_bool(e, p, "undistorted", self.undistorted)
        _check_str(e, p, "undistortion_method", self.undistortion_method)
        if self.undistortion_method != "nuscenes_native":
            _err(
                e,
                p,
                "undistortion_method must be 'nuscenes_native': nuScenes ships rectified "
                "imagery with no distortion coefficients, and the distortion path is untested",
            )
        return e


@dataclass
class CloudArtifact:
    """One point cloud on disk, in EGO frame (§1.1 rule 3, §1.4).

    Both clouds are produced. The ground plane is FIT on the accumulation and
    APPLIED to the single sweep; the lift and `num_lidar_pts` use the single
    sweep. An accumulated cloud is ego-motion compensated only, so dynamic
    objects smear across the window: painting a smeared cloud with a
    single-frame mask inflates box length along the direction of travel, and the
    result looks like "fast objects are harder" rather than like a bug.

    `n_sweeps_actual` and `window_ns` are recorded because at scene starts the
    window is truncated, so point density — hence cluster size, hence box
    dimensions — differs systematically for the first keyframes of every scene.
    """

    path: str
    cloud_kind: str
    frame: str
    n_points: int
    n_sweeps_actual: int
    window_ns: int
    point_record_bytes: int = POINT_RECORD_BYTES

    def validate(self, *, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        _check_str(e, p, "path", self.path)
        _check_str(e, p, "cloud_kind", self.cloud_kind, allowed=CLOUD_KINDS)
        _check_frame(e, p, self.frame, expected=EGO)
        _check_int(e, p, "n_points", self.n_points, minimum=1)
        _check_int(e, p, "n_sweeps_actual", self.n_sweeps_actual, minimum=1)
        _check_int(e, p, "window_ns", self.window_ns, minimum=0)
        _check_int(e, p, "point_record_bytes", self.point_record_bytes, minimum=1)
        if self.point_record_bytes != POINT_RECORD_BYTES:
            _err(e, p, f"point_record_bytes must be {POINT_RECORD_BYTES} (5 x float32)")
        if self.cloud_kind == "single_sweep":
            if self.n_sweeps_actual != 1:
                _err(e, p, f"single_sweep cloud has n_sweeps_actual={self.n_sweeps_actual}")
            if self.window_ns != 0:
                _err(e, p, f"single_sweep cloud has window_ns={self.window_ns}, expected 0")
        elif self.cloud_kind == "accumulated":
            # NOT "at least 2 sweeps": the FIRST keyframe of every scene has no
            # predecessor inside W_acc, so its accumulation is one sweep over a
            # zero-length window and is byte-identical to the single sweep. That
            # is the truncated-window case §1.4 requires to be carried, not
            # hidden — so what is enforced is CONSISTENCY between the two fields,
            # and the degeneracy stays visible to any consumer that checks
            # n_sweeps_actual before assuming constant density.
            if isinstance(self.n_sweeps_actual, int) and isinstance(self.window_ns, int):
                if self.n_sweeps_actual > 1 and self.window_ns <= 0:
                    _err(e, p, f"accumulated cloud spans {self.n_sweeps_actual} sweeps but window_ns=0")
                if self.n_sweeps_actual == 1 and self.window_ns != 0:
                    _err(e, p, f"accumulated cloud holds 1 sweep but window_ns={self.window_ns}")
        return e


@dataclass
class KeyframeRecord(Record):
    """I-3: the keyframe pack (§1.4, §5.2).

    `t_ns` is the LIDAR_TOP `sample_data.timestamp` converted once — never
    `sample.timestamp`, which is a different quantity (§1.2 rule 4).
    """

    _contract_tag = "I-3"

    keyframe_token: str  # nuScenes sample token
    scene_token: str
    t_ns: int
    time_base: str
    lidar_sample_data_token: str
    lidar_path: str
    lidar_ego_pose_token: str
    lidar_calibrated_sensor_token: str
    cameras: dict  # channel -> CameraObservation
    single_sweep_cloud: CloudArtifact
    accumulated_cloud: CloudArtifact
    coverage_config: str
    is_first_in_scene: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract", self._contract_tag)

    def validate(self, *, policy: ProvenancePolicy = POLICY, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        for name in (
            "keyframe_token",
            "scene_token",
            "lidar_sample_data_token",
            "lidar_path",
            "lidar_ego_pose_token",
            "lidar_calibrated_sensor_token",
        ):
            _check_str(e, p, name, getattr(self, name))
        _check_time(e, p, self.t_ns, self.time_base)
        _check_coverage(e, p, self.coverage_config)
        _check_bool(e, p, "is_first_in_scene", self.is_first_in_scene)

        if not isinstance(self.cameras, dict) or not self.cameras:
            _err(e, p, "cameras must be a non-empty mapping of channel -> CameraObservation")
        else:
            for channel, obs in self.cameras.items():
                if not isinstance(obs, CameraObservation):
                    _err(e, p, f"cameras[{channel!r}] must be a CameraObservation")
                    continue
                if obs.channel != channel:
                    _err(e, p, f"cameras[{channel!r}] carries channel={obs.channel!r}")
                e.extend(obs.validate(prefix=f"{p}cameras[{channel!r}]."))
            if self.coverage_config == "R2" and set(self.cameras) != set(RING_CAMERAS):
                missing = sorted(set(RING_CAMERAS) - set(self.cameras))
                _err(e, p, f"coverage_config=R2 keyframe is missing camera(s): {missing}")

        for name, cloud, expected_kind in (
            ("single_sweep_cloud", self.single_sweep_cloud, "single_sweep"),
            ("accumulated_cloud", self.accumulated_cloud, "accumulated"),
        ):
            if not isinstance(cloud, CloudArtifact):
                # I-3 mandates BOTH clouds; producing one is a contract breach,
                # not a quality difference (§1.4).
                _err(e, p, f"{name} is required and must be a CloudArtifact")
                continue
            if cloud.cloud_kind != expected_kind:
                _err(e, p, f"{name} carries cloud_kind={cloud.cloud_kind!r}")
            e.extend(cloud.validate(prefix=f"{p}{name}."))
        return e


# ---------------------------------------------------------------------------
# I-4 / I-5 — pre-labels and verified labels
# ---------------------------------------------------------------------------


@dataclass
class GateVector:
    """The QA gate vector (spec §7.3.9).

    `spatial_ok` is `None` by default: the drivable-area term is OFF (§11,
    decision 7). The map expansion is on disk so it is implementable, but a
    pilot component with no production counterpart risks reading as a capability
    the real pipeline has. `None` says "not evaluated"; `False` would say
    "evaluated and failed". Silence is the only unacceptable option.
    """

    conf: float
    lidar_pts_ok: bool
    spatial_ok: bool | None = None
    spatial_ok_source: str | None = None

    def validate(self, *, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        _check_number(e, p, "conf", self.conf, minimum=0.0, maximum=1.0)
        _check_bool(e, p, "lidar_pts_ok", self.lidar_pts_ok)
        _check_bool(e, p, "spatial_ok", self.spatial_ok, allow_none=True)
        _check_str(e, p, "spatial_ok_source", self.spatial_ok_source, allow_none=True)
        if self.spatial_ok is not None and not self.spatial_ok_source:
            _err(
                e,
                p,
                "spatial_ok is populated but spatial_ok_source is absent; an enabled drivable "
                "term must be marked (no_production_counterpart)",
            )
        return e


@dataclass
class Provenance:
    """§A.1 provenance block. `source` and `tier` are independent (§1.7 rule 3)."""

    source: str
    tier: str
    gates: GateVector
    verified_by: str | None = None
    verification_pass: int = 0
    # Which of the two independent human passes produced this row (§7). Human
    # sources only: the pipeline does not annotate twice.
    annotator_pass: str | None = None

    def validate(self, *, policy: ProvenancePolicy = POLICY, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        _check_str(e, p, "source", self.source, allowed=SOURCES)
        _check_str(e, p, "tier", self.tier, allowed=TIERS)
        _check_str(e, p, "verified_by", self.verified_by, allow_none=True)
        _check_int(e, p, "verification_pass", self.verification_pass, minimum=0)
        _check_str(e, p, "annotator_pass", self.annotator_pass, allowed=ANNOTATOR_PASSES,
                   allow_none=True)
        if not isinstance(self.gates, GateVector):
            _err(e, p, "gates must be a GateVector")
        else:
            e.extend(self.gates.validate(prefix=f"{p}gates."))

        if self.source in HUMAN_SOURCES:
            # The converse direction, which rev 1 never stated (§1.7 rule 2).
            if not self.verified_by:
                _err(e, p, f"source={self.source!r} requires verified_by")
            if not isinstance(self.verification_pass, int) or self.verification_pass < 1:
                _err(e, p, f"source={self.source!r} requires verification_pass >= 1")
            if not policy.allow_human_provenance:
                _err(
                    e,
                    p,
                    f"source={self.source!r} is forbidden: the pilot has no human annotators "
                    "(allow_human_provenance=False)",
                )
        else:
            if self.verified_by is not None:
                _err(e, p, f"source={self.source!r} must not carry verified_by")
            if self.verification_pass != 0:
                _err(e, p, f"source={self.source!r} must have verification_pass=0")
            if self.annotator_pass is not None:
                _err(
                    e,
                    p,
                    f"source={self.source!r} must not carry annotator_pass (double annotation "
                    "is a human pass)",
                )
        return e


@dataclass
class AnnotationRecord(Record):
    """I-4 pre-label / I-5 verified label — one §A.1 record.

    Both contracts share this shape; `provenance.source` is what distinguishes
    them, which is exactly why the split/source invariant is enforced here and
    not at a release-builder boundary the pilot never reaches.

    Field-order traps this schema exists to close (§3.2):
      - `size_wlh_m` is [w, l, h] in nuScenes order. KITTI is [h, w, l] and many
        L-shape fitters return (length, width). A swap rotates every box 90 deg
        while every value stays plausible.
      - `rotation_wxyz` is the only stored orientation. Stage 6 produces a
        scalar yaw; the conversion goes through conventions.quaternion_from_yaw_rad
        so the round trip is one tested function rather than an inline
        expression repeated per call site.
      - `num_lidar_pts` is counted on the SINGLE-SWEEP, ground-filtered,
        PRE-inflation cloud. nuScenes' own field counts single-sweep INCLUDING
        ground; the two are not comparable and `num_lidar_pts_basis` records
        which one this is.
    """

    _contract_tag = "I-4"

    token: str
    sample_token: str
    instance_token: str
    category: str
    frame: str
    t_ns: int
    time_base: str
    translation_m: list
    size_wlh_m: list
    rotation_wxyz: list
    num_lidar_pts: int
    provenance: Provenance
    coverage_config: str
    velocity_mps: list | None = None
    track_id: str | None = None
    split: str | None = None
    num_lidar_pts_basis: str = "single_sweep_ground_filtered_pre_inflation"
    # No producer in the pilot; declared, not silently dropped (§4, §3.2).
    attribute: str | None = None
    visibility: str | None = None
    # Human-pass only (spec 2026-09-07 §7): the contract's per-row uncertainty flag.
    is_uncertain: bool | None = None
    is_uncertain_reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract", self._contract_tag)

    @property
    def yaw_rad(self) -> float:
        return yaw_rad_from_quaternion(self.rotation_wxyz)

    def validate(self, *, policy: ProvenancePolicy = POLICY, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        for name in ("token", "sample_token", "instance_token", "category"):
            _check_str(e, p, name, getattr(self, name))
        _check_str(e, p, "track_id", self.track_id, allow_none=True)
        _check_time(e, p, self.t_ns, self.time_base)
        # Everything crossing a stage boundary is in ego frame (§1.1 rule 1).
        _check_frame(e, p, self.frame, expected=EGO)
        _check_vec(e, p, "translation_m", self.translation_m, 3)
        _check_vec(e, p, "size_wlh_m", self.size_wlh_m, 3, positive=True)
        _check_quaternion(e, p, "rotation_wxyz", self.rotation_wxyz)
        _check_vec(e, p, "velocity_mps", self.velocity_mps, 2, allow_none=True)
        _check_int(e, p, "num_lidar_pts", self.num_lidar_pts, minimum=0)
        _check_str(e, p, "num_lidar_pts_basis", self.num_lidar_pts_basis)
        if self.num_lidar_pts_basis != "single_sweep_ground_filtered_pre_inflation":
            _err(
                e,
                p,
                "num_lidar_pts_basis must be 'single_sweep_ground_filtered_pre_inflation'; "
                "a count from the accumulation loosens the >=5-return gate by ~10x on this "
                "substrate while the gate still appears to fire",
            )
        _check_coverage(e, p, self.coverage_config)
        _check_str(e, p, "split", self.split, allowed=SPLITS, allow_none=True)
        _check_str(e, p, "attribute", self.attribute, allow_none=True)
        _check_str(e, p, "visibility", self.visibility, allow_none=True)
        # `attribute` and `is_uncertain*` acquired a producer in the human loop
        # (§7) and only there; on a pipeline row the "no producer" refusal of §4
        # is unchanged, so an I-4 record that gained either field is still a bug.
        human = isinstance(self.provenance, Provenance) and self.provenance.source in HUMAN_SOURCES
        _check_bool(e, p, "is_uncertain", self.is_uncertain, allow_none=True)
        _check_str(e, p, "is_uncertain_reason", self.is_uncertain_reason, allow_none=True)
        if human:
            if self.attribute is not None and self.attribute not in ATTRIBUTE_NAMES_NUSCENES:
                _err(e, p, f"attribute={self.attribute!r} is not a nuScenes attribute name")
            if self.is_uncertain_reason is not None and self.is_uncertain is not True:
                _err(e, p, "is_uncertain_reason requires is_uncertain=true")
        else:
            if self.attribute is not None:
                _err(
                    e,
                    p,
                    "attribute has no producer in the pipeline (Stage 10 waived, §4); "
                    "human passes only",
                )
            if self.is_uncertain is not None or self.is_uncertain_reason is not None:
                _err(e, p, "is_uncertain is set by a human pass only")
        if self.visibility is not None:
            _err(e, p, "visibility has no producer in the pilot (§4)")

        if not isinstance(self.provenance, Provenance):
            _err(e, p, "provenance is required and must be a Provenance")
        else:
            e.extend(self.provenance.validate(policy=policy, prefix=f"{p}provenance."))
            # The direction rev 1 did state — kept, and now non-vacuous because
            # `split` exists (§1.7 rule 2).
            if self.split in ("val", "test") and self.provenance.source == "pipeline_accepted":
                _err(
                    e,
                    p,
                    f"split={self.split!r} with source='pipeline_accepted' is a build error: "
                    "pseudo-labels must never enter val/test",
                )
        return e


@dataclass
class GroupAnnotation(Record):
    """§A.2 group/ignore box — SHAPE ONLY. No producer, no consumer (§4).

    Kept in the schema so that rho's `n_min` term is declared rather than
    silently misapplied: `rho()` accepts the weights, nothing in the pilot
    supplies them, and that gap is visible here instead of being discovered as a
    density number that quietly counts crowds as one object.
    """

    _contract_tag = "I-4G"

    token: str
    sample_token: str
    class_group: str
    frame: str
    polygon_bev_m: list
    height_range_m: list
    n_min: int
    coverage_config: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract", self._contract_tag)

    def validate(self, *, policy: ProvenancePolicy = POLICY, prefix: str = "") -> list[str]:
        e: list[str] = []
        p = prefix
        for name in ("token", "sample_token", "class_group"):
            _check_str(e, p, name, getattr(self, name))
        _check_frame(e, p, self.frame, expected=EGO)
        _check_coverage(e, p, self.coverage_config)
        _check_int(e, p, "n_min", self.n_min, minimum=1)
        if not isinstance(self.polygon_bev_m, (list, tuple)) or len(self.polygon_bev_m) < 3:
            _err(e, p, "polygon_bev_m must have at least 3 vertices")
        else:
            for i, vertex in enumerate(self.polygon_bev_m):
                _check_vec(e, p, f"polygon_bev_m[{i}]", vertex, 2)
        _check_vec(e, p, "height_range_m", self.height_range_m, 2)
        if (
            isinstance(self.height_range_m, (list, tuple))
            and len(self.height_range_m) == 2
            and all(isinstance(v, (int, float)) for v in self.height_range_m)
            and self.height_range_m[0] >= self.height_range_m[1]
        ):
            _err(e, p, "height_range_m must be [z0, z1] with z0 < z1")
        return e


RECORD_TYPES: dict[str, type] = {
    SubstrateManifest._contract_tag: SubstrateManifest,
    EgoPoseRecord._contract_tag: EgoPoseRecord,
    KeyframeRecord._contract_tag: KeyframeRecord,
    AnnotationRecord._contract_tag: AnnotationRecord,
    GroupAnnotation._contract_tag: GroupAnnotation,
}

# Nested dataclasses, for reconstruction on read-back. A tuple marks a mapping
# of key -> nested record.
_NESTED_TYPES.update(
    {
        "KeyframeRecord": {
            "cameras": (CameraObservation,),
            "single_sweep_cloud": CloudArtifact,
            "accumulated_cloud": CloudArtifact,
        },
        "AnnotationRecord": {"provenance": Provenance},
        "Provenance": {"gates": GateVector},
    }
)


# ---------------------------------------------------------------------------
# The write boundary (§1.7 rule 1)
# ---------------------------------------------------------------------------


def validate_records(
    records: Iterable[Record],
    *,
    policy: ProvenancePolicy = POLICY,
    expect_type: type | None = None,
) -> list[str]:
    """Non-raising check-then-decide path. Returns every error found."""
    errors: list[str] = []
    for i, record in enumerate(records):
        prefix = f"[{i}] "
        if not isinstance(record, Record):
            errors.append(f"{prefix}not a Record: {type(record).__name__}")
            continue
        if expect_type is not None and not isinstance(record, expect_type):
            errors.append(f"{prefix}expected {expect_type.__name__}, got {type(record).__name__}")
            continue
        errors.extend(record.validate(policy=policy, prefix=prefix))
    return errors


def write_records(
    path: str | os.PathLike,
    records: Sequence[Record],
    *,
    policy: ProvenancePolicy = POLICY,
    expect_type: type | None = None,
    allow_empty: bool = False,
) -> str:
    """The single persistence boundary. Raises on any invalid record.

    Sequence, deliberately:
      1. validate every record in memory;
      2. serialise to a temp file in the destination directory, fsync;
      3. READ THE TEMP FILE BACK, reconstruct every record, and validate again;
      4. only then os.replace() into place.

    Step 3 is the one rev 1 lacked. A record can satisfy its dataclass and still
    lose an invariant in serialisation — a tuple that becomes a list of the
    wrong length, a `None` that becomes the string "None", a nested block that
    round-trips into a plain dict and is reconstructed downstream with the
    validator never consulted. Validating the bytes that will actually be read
    is the only check that covers that path.

    Step 4 after step 3 means a failed read-back never lands on disk: there is
    no half-written artifact for a later stage to mistake for a complete one.
    """
    path = os.fspath(path)
    records = list(records)
    if not records and not allow_empty:
        raise SchemaValidationError(
            ["refusing to write an empty record file; pass allow_empty=True to declare it"],
            context=path,
        )

    contracts = {type(r).__name__ for r in records if isinstance(r, Record)}
    if len(contracts) > 1:
        raise SchemaValidationError(
            [f"mixed record types in one file: {sorted(contracts)}"], context=path
        )

    errors = validate_records(records, policy=policy, expect_type=expect_type)
    if errors:
        raise SchemaValidationError(errors, context=path)

    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record.to_dict(), sort_keys=True, separators=(",", ":")))
                fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())

        # Read-back over the bytes that will actually be persisted.
        replayed = read_records(tmp_path, policy=policy, expect_type=expect_type)
        if len(replayed) != len(records):
            raise SchemaValidationError(
                [f"read-back returned {len(replayed)} record(s), wrote {len(records)}"],
                context=path,
            )
        for i, (original, restored) in enumerate(zip(records, replayed)):
            if original.to_dict() != restored.to_dict():
                raise SchemaValidationError(
                    [f"[{i}] record does not survive the round trip unchanged"], context=path
                )

        os.replace(tmp_path, path)
        tmp_path = ""
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return path


def read_records(
    path: str | os.PathLike,
    *,
    policy: ProvenancePolicy = POLICY,
    expect_type: type | None = None,
) -> list[Record]:
    """Reconstruct and re-validate. The read side of the same boundary."""
    path = os.fspath(path)
    records: list[Record] = []
    errors: list[str] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"line {lineno}: not valid JSON ({exc.msg})")
                continue
            if payload.get("__schema_version__") != SCHEMA_VERSION:
                errors.append(
                    f"line {lineno}: __schema_version__="
                    f"{payload.get('__schema_version__')!r}, expected {SCHEMA_VERSION!r}"
                )
                continue
            try:
                records.append(Record.from_dict(payload))
            except SchemaValidationError as exc:
                errors.extend(f"line {lineno}: {e}" for e in exc.errors)
    errors.extend(validate_records(records, policy=policy, expect_type=expect_type))
    if errors:
        raise SchemaValidationError(errors, context=path)
    return records
