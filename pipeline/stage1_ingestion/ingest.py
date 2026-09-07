"""Stage 1 — ingestion: both clouds, ground removal, and the filter ledger (§5.2, §1.4).

Builds I-3. Four things happen here and each has a specific failure mode that
this module is written to make impossible or visible:

  1. **LiDAR points enter the ego frame exactly once** (§1.1 rule 3). The
     sensor->ego hop is applied here and nowhere else; Stage 5's projection
     chain starts from ego-frame points and goes forward. LIDAR_TOP's extrinsic
     is a -89.9 deg yaw, so a second application rotates the world by another
     90 deg and leaves every value plausible.

  2. **Both clouds are produced** (§1.4). The accumulation is ego-motion
     compensated using each sweep's OWN `ego_pose` — the devkit's approach and
     the correct one for accumulation. An accumulated cloud is ego-motion
     compensated ONLY: dynamic objects smear across the window, so it is not
     the cloud that gets lifted.

  3. **The ground plane is FIT on the accumulation and APPLIED to the single
     sweep.** Fitting on the accumulation is spec-faithful (more points, better
     fit); lifting from the single sweep is what keeps `num_lidar_pts` and the
     ">= 5 returns" gate meaning what §6.3 defines. Fitting and lifting on the
     same accumulated cloud loosens that gate by roughly the accumulation
     factor (~10x here) while it still appears to fire.

  4. **Per-filter, per-sector point counts are a primary output**, not a log
     line (P1-13). input -> post-ground -> post-range -> post-height, per
     keyframe AND per sector, for both clouds. Without them there is no way to
     tell "few objects because detection is bad" from "few objects because
     filtering deleted them" — and with a 0.3 m ground band applied to a
     32-beam LiDAR, the second is a live possibility.

**Parameter provenance warning, carried in the config and in every output.**
0.3 m / 40 m / 4 m were chosen for a Livox Mid-360 on Dhaka roads and are
applied here to a 32-beam spinning LiDAR on Boston and Singapore roads. A 0.3 m
band removes all wheel returns from every vehicle and most of a traffic cone's
body, pushing small classes below the >= 5-point gate. Sector RANSAC mis-fits
on ramps, speed bumps, and cambered roads, and a single plane per sector cannot
represent a curb. Nothing here corrects for that; the diagnostics are what make
it measurable.

**Determinism.** RANSAC is stochastic and is the FIRST geometric operation in
the pipeline, so unseeded it produces a different ground plane, different
retained points, different clusters, and different boxes on every run (§1.9).
The RNG is seeded per (global seed, keyframe token, sector) so the stream does
not depend on processing order, and the derivation is recorded.

No models, no GPU: numpy and stdlib only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from pipeline.common.conventions import (
    EGO,
    LIDAR,
    NUSCENES_GLOBAL,
    Transform,
    apply_transform,
    quaternion_to_rotation_matrix,
    to_unix_ns,
)
from pipeline.common.eval_region import RegionSpec, in_region, region_spec_from_config
from pipeline.common.paths import Paths, PathValidationError, assert_dataroot_read_only, load_paths, metadata_fingerprint
from pipeline.common.schemas import (
    POINT_RECORD_BYTES,
    GROUND_FIT_RANGE_M,
    GROUND_FIT_RINGS,
    GROUND_Z_BAND_M,
    RING_CAMERAS,
    STEREO_RINGS,
    STEREO_STRIDE,
    W_ACC_COUNT,
    W_ACC_DURATION_NS,
    CameraObservation,
    CloudArtifact,
    KeyframeRecord,
    Record,
    SchemaValidationError,
    write_records,
)
from pipeline.common.manifest import (
    UpstreamRefusal as _UpstreamRefusal,
    clear_markers,
    read_marker,
    write_json_atomic,
    write_marker,
)
from pipeline.stage0_data_probe.probe import Substrate

STAGE = "stage1_ingestion"
STAGE_SPEC = "dhakascenes-pilot/stage1_ingestion/v1"

EXIT_OK = 0
EXIT_DEGRADED = 1  # ran, but at least one keyframe or sector fell back
EXIT_REFUSED = 2  # upstream contract broken; nothing was written

# The ego-compensation check has an exact expected value, so this is a
# floating-point tolerance, not a tuned threshold.
COMPENSATION_TOLERANCE_M = 1e-6


# Canonical class lives in pipeline.common.manifest (C3); the local name stays
# so every `except UpstreamRefusal` written against this stage keeps catching.
UpstreamRefusal = _UpstreamRefusal


# ---------------------------------------------------------------------------
# Configuration — every value carries a provenance note (§10)
# ---------------------------------------------------------------------------


# ZED depth-pass channels merged into the single sweep, and the value each one
# writes into the ring column. The Livox Mid-360 emits rings 0-3, so 10/11 are
# free and a reader can split the fused cloud back apart without a side file.
STEREO_CHANNELS: dict[str, int] = {"ZED_FRONT": 10, "ZED_BACK": 11}


@dataclass(frozen=True)
class IngestConfig:
    """Stage 1 tunables. None of these may appear as a literal in the code below."""

    # --- accumulation (§11, decision 2: duration preserved, not count) ---
    # Defaults come from the substrate profile (schemas.W_ACC_*, 2026-09-06) so
    # Stage 0's gate and this stage's window can never disagree; dhaka and
    # nuscenes still resolve to 0.5 s / 5.
    w_acc_duration_ns: int = W_ACC_DURATION_NS
    w_acc_count: int = W_ACC_COUNT

    # --- stereo thinning (profile property, 2026-09-06) ---
    # Rings of LIDAR_TOP that carry fused stereo depth, kept at every
    # stereo_stride-th point (file order, deterministic). () / 1 = untouched.
    # A list, not a tuple: the config is serialised into every manifest and
    # diagnostics payload, and write_json_atomic refuses a payload that does
    # not survive the JSON round trip — a tuple comes back as a list.
    stereo_rings: list = field(default_factory=lambda: list(STEREO_RINGS))
    stereo_stride: int = STEREO_STRIDE

    # --- ground removal ---
    n_sectors: int = 8
    ransac_iterations: int = 200
    ransac_distance_threshold_m: float = 0.10
    ransac_min_candidates: int = 50
    # Profile-owned since 2026-09-06 (schemas.GROUND_Z_BAND_M): the day-1 rig's
    # ego origin is the LiDAR ~2.3 m up, and the ISO 8855 band never held its
    # road — see the dhaka6 profile for what that did to the boxes.
    ransac_candidate_z_band_m: tuple[float, float] = GROUND_Z_BAND_M
    # Which rings may vote for the ground ([] = all) and within what radial
    # window (None = any). Profile-owned (schemas.GROUND_FIT_*): dhaka6 votes
    # with the Mid-360 rings + the front ZED at 3-12 m, never the rear ZED.
    ground_fit_rings: list = field(default_factory=lambda: list(GROUND_FIT_RINGS))
    ground_fit_range_m: list | None = field(
        default_factory=lambda: None if GROUND_FIT_RANGE_M is None else list(GROUND_FIT_RANGE_M))
    ground_band_m: float = 0.30
    reject_tilt_deg: float = 15.0
    # A wedge fit is rejected on what actually harms the ground filter: a tilt
    # no road has, or a HEIGHT that disagrees with the robust whole-cloud
    # reference plane (evaluated at the wedge's own candidate centroid). The
    # inlier ratio is NOT a fit-quality measure on a crowded substrate — it
    # measures how much clutter shares the band — so its gate is off (0.0)
    # unless a profile has a reason to turn it back on.
    reject_height_disagreement_m: float = 0.25
    reject_min_inlier_ratio: float = 0.0

    # --- stereo fusion (ZED depth pass) ---
    # The Mid-360 puts a MEDIAN OF 10 returns on an object at this range, and a
    # 10-point cluster cannot determine a box: measured on the 2026-08-30 run,
    # 64% of the median box's volume came from the priors rather than the data
    # and 56% of yaws were ambiguous. The ZED pass carries ~88k points per
    # keyframe inside 20 m against the lidar's ~20k over the full annulus, so
    # fusing it is what makes a near-field box a measurement.
    fuse_stereo: bool = True

    # --- pruning ---
    range_cap_m: float = 50.0
    height_cap_m: float = 4.0
    prune_accumulated: bool = False

    # --- determinism (§1.9) ---
    global_seed: int = 20260812

    # --- validation (§5.2, "validation worth having") ---
    degraded_rejection_rate: float = 0.10

    # --- upstream gate (C16) ---
    accept_degraded_upstream: bool = False

    coverage_config: str = "R2"

    provenance: dict = field(
        default_factory=lambda: {
            "w_acc_duration_ns": "pilot_plan.md §11 decision 2; 0.5 s duration preserved",
            "w_acc_count": "derived: 0.5 s at the measured 10.00 Hz (v1.0-dhaka-fixed, 2026-08-30)",
            "stereo_rings": "substrate profile (schemas.STEREO_RINGS): LIDAR_TOP rings that are fused "
                            "stereo depth, not lidar returns",
            "ground_fit_rings": "substrate profile (schemas.GROUND_FIT_RINGS): rings allowed as RANSAC "
                                "ground candidates; dhaka6 = Mid-360 + front ZED, which agree on the road "
                                "to ~0.2 m (the rear ZED sits 0.69 m low at its camera; measured 2026-09-06)",
            "ground_fit_range_m": "substrate profile (schemas.GROUND_FIT_RANGE_M): radial window for "
                                  "ground candidates; dhaka6 = 3-12 m where stereo is dense and reliable",
            "stereo_stride": "substrate profile (schemas.STEREO_STRIDE): every k-th stereo point kept, "
                             "file order. dhaka6: 8, measured 2026-09-06 — ZED 8.8x the Mid-360's "
                             "density, one instance of ~38k points exhausted RAM in Stage 6's DBSCAN",
            "fuse_stereo": "ZED_FRONT/ZED_BACK merged into the SINGLE SWEEP only. Stereo, "
            "not lidar: error grows with the square of range and the pass caps at 20 m, so it "
            "densifies the near field and adds nothing beyond it. The accumulation stays "
            "lidar-only so the ground fit is unchanged. Provenance rides in the ring column "
            "(Mid-360 uses 0-3; ZED_FRONT=10, ZED_BACK=11), which keeps the 20-byte record.",
            "n_sectors": "arbitrary, needs tuning — absent from both governing documents",
            "ransac_iterations": "arbitrary, needs tuning — absent from both governing documents",
            "ransac_distance_threshold_m": "arbitrary, needs tuning — NOT the same quantity as "
            "ground_band_m; this is the RANSAC inlier tolerance",
            "ransac_min_candidates": "arbitrary, needs tuning",
            "ransac_candidate_z_band_m": "substrate profile (schemas.GROUND_Z_BAND_M): where the road "
            "can be in THIS rig's ego frame — ISO 8855 (z=0 at ground) for dhaka/nuscenes, "
            "[-3.5, -1.0] for dhaka6 whose ego origin is the LiDAR ~2.3 m up (measured "
            "2026-09-06); restricts the fit so a building facade cannot win the sector",
            "ground_band_m": "comprehensive.md §7.3.1, unvalidated on this substrate — chosen "
            "for a Livox Mid-360 on Dhaka roads, applied here to a 32-beam spinning LiDAR",
            "range_cap_m": "50 m, operator decision 2026-09-07: annotate to the benchmark's evaluation range (class_range 50/40/30 m). The Stage 9 point floor (>= 5 returns) decides what survives; the delivery note reports the effective per-class range. Must match eval_region._R_MAX_M or Stage 1 prunes to one radius while Stage 5/6 score against another. Was 30 m (human-directed 2026-08-30); runs before and after are not comparable.",
            "height_cap_m": "comprehensive.md §7.3.1, unvalidated on this substrate",
            "prune_accumulated": "pilot decision: the 40 m+ stratified bin exists only for the "
            "accumulated-cloud secondary track (spec §3.6), so the accumulation is NOT "
            "range-pruned by default; the counts it would have lost are recorded anyway",
            "global_seed": "pilot_plan.md §1.9 — one global seed, recorded",
            "reject_tilt_deg": "arbitrary, needs tuning — a sector fit steeper than this is not "
            "a road; measured on v1.0-mini, scene-0553/0757 sector 5 fits ~60 deg at inlier "
            "ratio 0.29, which is a facade or a stopped bus flank, not ground",
            "reject_min_inlier_ratio": "OFF (0.0) since 2026-09-06 — on Dataset/A_nusc the ratio "
            "measured clutter, not fit quality: 707 of 1167 substituted wedge fits on chunk_0000 "
            "were rejected at ratio ~0.30 while sitting within 0.1 m of the road, and the "
            "substitute was worse (see reject_height_disagreement_m). On v1.0-mini good fits "
            "scored 0.47-0.79, mis-fits 0.26-0.30 — a profile may re-enable it",
            "reject_height_disagreement_m": "2026-09-06 — a wedge whose own plane sits more than "
            "this above/below the robust reference plane at the wedge centroid is not the road "
            "(a flat truck bed, a platform, a plane through a crowd's knees). Chosen from the "
            "accepted-fit disagreement distribution on chunk_0000 (see the Stage 1 handover "
            "note of 2026-09-06); road camber/ramps within 12 m stay well inside it",
            "degraded_rejection_rate": "arbitrary, needs tuning — the mis-fit guard fires on "
            "2-25% of sectors depending on scene, so a flag set by ANY rejection is on for "
            "every run and carries no signal; the rate is what distinguishes a hard scene",
            "accept_degraded_upstream": "C16 — consuming a DEGRADED (complete, quality-flagged) "
            "Stage 0 output is an explicit recorded decision, never a default",
            "coverage_config": "pilot_plan.md §11 decision 1 — R2, all six ring cameras",
        }
    )

    def as_dict(self) -> dict:
        out = asdict(self)
        out["ransac_candidate_z_band_m"] = list(self.ransac_candidate_z_band_m)
        return out


FILTERS: tuple[str, ...] = ("input", "post_ground", "post_range", "post_height")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def thin_stereo(cloud: np.ndarray, rings, stride: int) -> tuple[np.ndarray, int]:
    """Keep every `stride`-th point of each ring in `rings`; every other point
    stays. File order is preserved (a boolean mask, no sort), so the result is
    deterministic and re-runs reproduce it byte-for-byte. Returns
    (thinned cloud, points removed). stride 1 or no rings: the input, untouched.
    """
    if stride < 1:
        raise ValueError(f"stereo_stride must be >= 1, got {stride}")
    rings = tuple(rings)
    if stride == 1 or not rings:
        return cloud, 0
    keep = np.ones(cloud.shape[0], dtype=bool)
    ring_col = cloud[:, 4]
    for ring in rings:
        idx = np.flatnonzero(ring_col == ring)
        keep[idx[np.arange(idx.size) % stride != 0]] = False
    return cloud[keep], int(np.count_nonzero(~keep))


def read_pcd_bin(path: str) -> np.ndarray:
    """Read a nuScenes `.pcd.bin` as (N, 5): x, y, z, intensity, ring.

    The size check is not redundant with Stage 0's: a file that changed between
    the probe and this run must not reshape silently.
    """
    size = os.path.getsize(path)
    if size % POINT_RECORD_BYTES != 0:
        raise ValueError(f"{path}: size {size} is not a multiple of {POINT_RECORD_BYTES}")
    return np.fromfile(path, dtype=np.float32).reshape(-1, 5)


def write_pcd_bin(path: str, cloud: np.ndarray) -> int:
    """Write (N, 5) float32 back out, preserving the 20-byte record layout.

    Atomic: a partially written cloud is indistinguishable from a complete one
    to every reader downstream (§1.9).
    """
    if cloud.ndim != 2 or cloud.shape[1] != 5:
        raise ValueError(f"cloud must be (N, 5), got {cloud.shape}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    cloud.astype(np.float32, copy=False).tofile(tmp)
    os.replace(tmp, path)
    return int(cloud.shape[0])


def sector_index(points_xy: np.ndarray, n_sectors: int) -> np.ndarray:
    """Fixed azimuthal wedges in the ego frame, indexed 0..n_sectors-1 from -pi."""
    theta = np.arctan2(points_xy[:, 1], points_xy[:, 0])
    width = 2.0 * math.pi / n_sectors
    idx = np.floor((theta + math.pi) / width).astype(np.int64)
    return np.clip(idx, 0, n_sectors - 1)


def _sector_rng(cfg: IngestConfig, keyframe_token: str, sector: int) -> np.random.Generator:
    """Deterministic per (seed, keyframe, sector) — independent of iteration order.

    Seeding once per run and drawing sequentially would make a scene's ground
    planes depend on how many keyframes were processed before it, so a re-run of
    a single scene would not reproduce the full run's output.
    """
    token_seed = int(keyframe_token[:16], 16) if len(keyframe_token) >= 16 else abs(hash(keyframe_token))
    return np.random.default_rng([cfg.global_seed, token_seed, sector])


@dataclass
class SectorPlane:
    """z = a*x + b*y + d over one azimuthal wedge, with its own quality record."""

    sector: int
    a: float
    b: float
    d: float
    n_candidates: int
    n_inliers: int
    inlier_ratio: float
    tilt_deg: float
    fallback: str | None
    implausible_tilt: bool
    # When a fit is rejected, the rejected coefficients are kept verbatim. The
    # substitution is recorded, not hidden: a reader can see exactly what the
    # sector wanted to fit and decide whether the guard or the road is wrong.
    rejected_fit: dict | None = None

    def height_at(self, points_xy: np.ndarray) -> np.ndarray:
        return self.a * points_xy[:, 0] + self.b * points_xy[:, 1] + self.d

    def as_dict(self) -> dict:
        return asdict(self)


def _plane_from_three(p: np.ndarray) -> tuple[float, float, float] | None:
    """Plane through three points, as z = a*x + b*y + d. None if near-vertical."""
    normal = np.cross(p[1] - p[0], p[2] - p[0])
    nz = normal[2]
    if abs(nz) < 1e-6:
        return None  # a vertical plane has no z = f(x, y) form; reject the sample
    a = -normal[0] / nz
    b = -normal[1] / nz
    d = float(np.dot(normal, p[0]) / nz)
    return float(a), float(b), float(d)


def _least_squares_plane(points: np.ndarray) -> tuple[float, float, float] | None:
    if points.shape[0] < 3:
        return None
    A = np.column_stack([points[:, 0], points[:, 1], np.ones(points.shape[0])])
    try:
        coeffs, *_ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    except np.linalg.LinAlgError:
        return None
    return float(coeffs[0]), float(coeffs[1]), float(coeffs[2])


def _tilt_deg(a: float, b: float) -> float:
    """Angle between the fitted plane's normal and +z."""
    return math.degrees(math.atan(math.hypot(a, b)))


@dataclass
class ReferencePlane:
    """The robust whole-cloud ground plane: the substitute for any wedge that
    cannot or must not use its own fit, and the height reference the mis-fit
    guard compares every wedge against.

    Fitted by the same seeded RANSAC + inlier polish as the wedges, on every
    candidate the profile lets vote. A plain least-squares fit is NOT a ground
    plane here: the candidate band reaches above the road and every non-road
    candidate (legs, wheels, curbs, bodies) lies above it, so least squares
    drifted 0.26-0.39 m (p90 0.60 m) high on Dataset/A_nusc (2026-09-06).
    """

    a: float
    b: float
    d: float
    n_candidates: int
    n_inliers: int
    inlier_ratio: float
    tilt_deg: float
    method: str  # "ransac_polished" | "least_squares" (RANSAC found no model)

    def height_at(self, points_xy: np.ndarray) -> np.ndarray:
        return self.a * points_xy[:, 0] + self.b * points_xy[:, 1] + self.d

    def coeffs(self) -> tuple[float, float, float]:
        return (self.a, self.b, self.d)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class GroundFit:
    planes: list[SectorPlane]
    reference: ReferencePlane | None  # None only when no candidate exists at all


def _ransac_plane(
    candidates: np.ndarray, rng: np.random.Generator, cfg: IngestConfig
) -> tuple[tuple[float, float, float], int] | None:
    """Seeded RANSAC on `candidates` (N x 3), polished on the consensus set.

    Returns (coefficients, inlier count) or None when no non-vertical triple
    was found. The 3-point model is a hypothesis, not the estimate: an
    unpolished plane tilts with whichever triple happened to win.
    """
    n = int(candidates.shape[0])
    if n < 3:
        return None
    best: tuple[float, float, float] | None = None
    best_inliers = 0
    for _ in range(cfg.ransac_iterations):
        sample = candidates[rng.choice(n, size=3, replace=False)]
        coeffs = _plane_from_three(sample)
        if coeffs is None:
            continue
        a, b, d = coeffs
        residual = np.abs(candidates[:, 2] - (a * candidates[:, 0] + b * candidates[:, 1] + d))
        n_inliers = int(np.count_nonzero(residual <= cfg.ransac_distance_threshold_m))
        if n_inliers > best_inliers:
            best, best_inliers = coeffs, n_inliers
    if best is None:
        return None
    a, b, d = best
    residual = np.abs(candidates[:, 2] - (a * candidates[:, 0] + b * candidates[:, 1] + d))
    polished = _least_squares_plane(candidates[residual <= cfg.ransac_distance_threshold_m])
    if polished is not None:
        best = polished
        residual = np.abs(candidates[:, 2] - (best[0] * candidates[:, 0] + best[1] * candidates[:, 1] + best[2]))
        best_inliers = int(np.count_nonzero(residual <= cfg.ransac_distance_threshold_m))
    return best, best_inliers


def _reference_plane(candidates: np.ndarray, cfg: IngestConfig, keyframe_token: str) -> ReferencePlane | None:
    n = int(candidates.shape[0])
    if n < 3:
        return None
    # Seeded like a wedge, with the index one past the last wedge: the same
    # keyframe always yields the same reference, independent of run order.
    fit = _ransac_plane(candidates, _sector_rng(cfg, keyframe_token, cfg.n_sectors), cfg)
    if fit is not None:
        (a, b, d), n_inliers = fit
        method = "ransac_polished"
    else:
        coeffs = _least_squares_plane(candidates)
        if coeffs is None:
            return None
        a, b, d = coeffs
        n_inliers = 0
        method = "least_squares"
    return ReferencePlane(
        a=a, b=b, d=d, n_candidates=n, n_inliers=n_inliers,
        inlier_ratio=n_inliers / n, tilt_deg=_tilt_deg(a, b), method=method,
    )


def fit_ground_planes(
    accumulated: np.ndarray,
    cfg: IngestConfig,
    keyframe_token: str,
) -> GroundFit:
    """Sector-wise RANSAC on the ACCUMULATED cloud (§5.2), plus the robust
    whole-cloud reference plane every wedge is checked against.

    Candidates are restricted to `ransac_candidate_z_band_m` around the ego
    frame's ground plane. Without that restriction a sector dominated by a
    building facade or a bus flank fits the vehicle instead of the road, and the
    ground filter then deletes the road and keeps the wall — a failure that
    produces a clean-looking cloud.

    A sector with too few candidates does NOT get a silently substituted plane:
    it falls back to the reference plane, then to z = 0, and records which. A
    sector whose own fit is steeper than a road or sits more than
    `reject_height_disagreement_m` from the reference at its own centroid is
    rejected the same way, with the rejected coefficients kept verbatim.
    """
    idx = sector_index(accumulated[:, :2], cfg.n_sectors)
    z_lo, z_hi = cfg.ransac_candidate_z_band_m
    in_band = (accumulated[:, 2] >= z_lo) & (accumulated[:, 2] <= z_hi)
    # Profile-owned candidate sources (2026-09-06): only the rings that are
    # trusted for the road may vote, and only where they are reliable. The
    # fitted plane still filters EVERY point afterwards.
    if cfg.ground_fit_rings:
        in_band &= np.isin(accumulated[:, 4], np.asarray(cfg.ground_fit_rings, dtype=np.float64))
    if cfg.ground_fit_range_m is not None:
        r_lo, r_hi = cfg.ground_fit_range_m
        radius = np.hypot(accumulated[:, 0], accumulated[:, 1])
        in_band &= (radius >= r_lo) & (radius <= r_hi)

    reference = _reference_plane(accumulated[in_band][:, :3].astype(np.float64), cfg, keyframe_token)
    global_plane = reference.coeffs() if reference is not None else None

    planes: list[SectorPlane] = []
    for sector in range(cfg.n_sectors):
        mask = (idx == sector) & in_band
        candidates = accumulated[mask][:, :3].astype(np.float64)
        n_candidates = int(candidates.shape[0])

        if n_candidates < cfg.ransac_min_candidates:
            coeffs = global_plane
            fallback = "global_plane" if coeffs is not None else "z=0"
            if coeffs is None:
                coeffs = (0.0, 0.0, 0.0)
            a, b, d = coeffs
            planes.append(
                SectorPlane(
                    sector=sector, a=a, b=b, d=d, n_candidates=n_candidates, n_inliers=0,
                    inlier_ratio=0.0, tilt_deg=_tilt_deg(a, b), fallback=fallback,
                    implausible_tilt=False,
                )
            )
            continue

        fit = _ransac_plane(candidates, _sector_rng(cfg, keyframe_token, sector), cfg)
        fallback = None
        if fit is None:
            best = global_plane or (0.0, 0.0, 0.0)
            best_inliers = 0
            fallback = "global_plane" if global_plane else "z=0"
        else:
            best, best_inliers = fit

        a, b, d = best
        tilt = _tilt_deg(a, b)
        # The tilt of the sector's OWN best fit, frozen before any substitution:
        # the implausible-tilt flag is about what RANSAC found, and computing it
        # after the guard swapped in the (near-flat) global plane made the
        # manifest's "223 rejections / 0 implausible tilts" structural rather
        # than empirical (C18 [V], fixed 2026-08-12).
        fitted_tilt = tilt
        inlier_ratio = best_inliers / n_candidates if n_candidates else 0.0

        # Height disagreement with the reference, at this wedge's own centroid:
        # a flat plane through a truck bed or a crowd's knees has a fine tilt
        # and a fine inlier ratio and is still not the ground.
        centroid = candidates[:, :2].mean(axis=0, keepdims=True)
        if reference is not None:
            height_disagreement = float(abs((a * centroid[0, 0] + b * centroid[0, 1] + d)
                                            - reference.height_at(centroid)[0]))
        else:
            height_disagreement = 0.0

        # A fit steeper than a road can be, or one whose height is not the
        # road's, is not a ground plane — it is a facade, a stopped bus flank, a
        # barrier, a platform. Using it would carve a diagonal slab out of the
        # cloud, or strip the bottom off every object in the wedge. The guard
        # SUBSTITUTES the reference plane and records what it rejected; it never
        # edits the geometry silently.
        rejected_fit = None
        if fallback is None:
            reason = None
            if tilt > cfg.reject_tilt_deg:
                reason = "tilt"
            elif height_disagreement > cfg.reject_height_disagreement_m:
                reason = "height"
            elif cfg.reject_min_inlier_ratio > 0 and inlier_ratio < cfg.reject_min_inlier_ratio:
                reason = "inlier_ratio"
            if reason is not None:
                rejected_fit = {
                    "a": a, "b": b, "d": d, "tilt_deg": tilt, "inlier_ratio": inlier_ratio,
                    "height_disagreement_m": height_disagreement, "reason": reason,
                }
                fallback = f"rejected_{reason}"
                a, b, d = global_plane or (0.0, 0.0, 0.0)
                if global_plane is None:
                    fallback += "_no_global_plane"
                tilt = _tilt_deg(a, b)

        planes.append(
            SectorPlane(
                sector=sector, a=a, b=b, d=d, n_candidates=n_candidates, n_inliers=best_inliers,
                inlier_ratio=inlier_ratio, tilt_deg=tilt, fallback=fallback,
                # Flags the sector's own fit, not the surviving plane: after the
                # guard substitutes the global plane, the surviving tilt is flat
                # by construction and the flag would never fire.
                implausible_tilt=fitted_tilt > cfg.reject_tilt_deg,
                rejected_fit=rejected_fit,
            )
        )
    return GroundFit(planes=planes, reference=reference)


def fit_sector_planes(
    accumulated: np.ndarray,
    cfg: IngestConfig,
    keyframe_token: str,
) -> list[SectorPlane]:
    """The wedge planes only — see fit_ground_planes."""
    return fit_ground_planes(accumulated, cfg, keyframe_token).planes


def ground_distance(points: np.ndarray, planes: list[SectorPlane], sectors: np.ndarray) -> np.ndarray:
    """Signed z - ground(x, y), each point against its OWN sector's plane."""
    out = np.empty(points.shape[0], dtype=np.float64)
    for plane in planes:
        mask = sectors == plane.sector
        if mask.any():
            out[mask] = points[mask, 2] - plane.height_at(points[mask, :2])
    return out


# ---------------------------------------------------------------------------
# The filter ledger — the primary output (P1-13)
# ---------------------------------------------------------------------------


@dataclass
class FilterLedger:
    """input -> post_ground -> post_range -> post_height, per sector and total.

    Every count here is a survivor count after the named filter, so the
    difference between consecutive entries is exactly what that filter removed.
    Kept as counts rather than percentages: a ratio hides whether 12 % of 40
    points or 12 % of 400,000 disappeared.
    """

    cloud_kind: str
    n_sectors: int
    total: dict = field(default_factory=dict)
    per_sector: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        cloud_kind: str,
        sectors: np.ndarray,
        masks: dict,
        n_sectors: int,
    ) -> "FilterLedger":
        total = {name: int(np.count_nonzero(mask)) for name, mask in masks.items()}
        per_sector = []
        for sector in range(n_sectors):
            in_sector = sectors == sector
            per_sector.append(
                {
                    "sector": sector,
                    **{
                        name: int(np.count_nonzero(mask & in_sector))
                        for name, mask in masks.items()
                    },
                }
            )
        return cls(cloud_kind=cloud_kind, n_sectors=n_sectors, total=total, per_sector=per_sector)

    def removed(self) -> dict:
        names = list(self.total)
        return {
            f"{names[i]}->{names[i + 1]}": self.total[names[i]] - self.total[names[i + 1]]
            for i in range(len(names) - 1)
        }

    def as_dict(self) -> dict:
        return {
            "cloud_kind": self.cloud_kind,
            "n_sectors": self.n_sectors,
            "survivors": self.total,
            "removed_by_filter": self.removed(),
            "per_sector": self.per_sector,
            **({"extra": self.extra} if self.extra else {}),
        }


# ---------------------------------------------------------------------------
# Accumulation
# ---------------------------------------------------------------------------


@dataclass
class Accumulation:
    points: np.ndarray  # (N, 5) in ego frame at the ANCHOR time
    n_sweeps_actual: int
    window_ns: int
    truncated: bool
    ego_motion_m: float
    compensation: dict


def accumulate(
    sub: Substrate,
    sweeps: list[dict],
    anchor: dict,
    cfg: IngestConfig,
) -> Accumulation:
    """Ego-motion-compensated accumulation into the anchor's ego frame.

    Per sweep, three named hops — the same shape as §1.3's projection chain and
    for the same reason:

        point_sensor -> ego(t_sweep) -> nuscenes_global -> ego(t_anchor)

    The first hop is the ONE application of `T_ego_lidar` (§1.1 rule 3). Each
    sweep uses its OWN `ego_pose` and its OWN `calibrated_sensor`; reusing the
    anchor's pose for every sweep is the error that smears the whole cloud along
    the direction of travel and still looks like a point cloud.
    """
    anchor_pose = Transform.from_nuscenes(
        sub.by_token("ego_pose.json")[anchor["ego_pose_token"]],
        source_frame=EGO,
        parent_frame=NUSCENES_GLOBAL,
    )
    to_anchor_ego = anchor_pose.inverse_matrix()

    compensated: list[np.ndarray] = []
    sweep_origins: list[tuple] = []
    sweep_poses: list[Transform] = []
    for record in sweeps:
        raw, _ = thin_stereo(read_pcd_bin(sub.blob(record)), cfg.stereo_rings, cfg.stereo_stride)
        xyz = raw[:, :3].astype(np.float64)

        # hop 1 — sensor -> ego(t_sweep). Applied exactly once, here.
        t_ego_lidar = Transform.from_nuscenes(
            sub.by_token("calibrated_sensor.json")[record["calibrated_sensor_token"]],
            source_frame=LIDAR,
            parent_frame=EGO,
        )
        p_ego_sweep = apply_transform(t_ego_lidar.matrix(), xyz)

        # hop 2 — ego(t_sweep) -> nuscenes_global, with the sweep's OWN pose.
        sweep_pose = Transform.from_nuscenes(
            sub.by_token("ego_pose.json")[record["ego_pose_token"]],
            source_frame=EGO,
            parent_frame=NUSCENES_GLOBAL,
        )
        p_global = apply_transform(sweep_pose.matrix(), p_ego_sweep)

        # hop 3 — nuscenes_global -> ego(t_anchor).
        p_anchor = apply_transform(to_anchor_ego, p_global)

        compensated.append(np.column_stack([p_anchor, raw[:, 3:5].astype(np.float64)]))
        sweep_origins.append(sweep_pose.translation_m)
        sweep_poses.append(sweep_pose)

    points = np.concatenate(compensated, axis=0) if compensated else np.zeros((0, 5))
    window_ns = (
        int(anchor["timestamp"] * 1000 - min(r["timestamp"] for r in sweeps) * 1000) if sweeps else 0
    )
    # How far the ego actually travelled across the window — the quantity the
    # compensation corrects for, recorded so a reader can tell a stationary
    # keyframe (waiting at a light) from a moving one.
    origins = np.asarray(sweep_origins, dtype=np.float64) if sweep_origins else np.zeros((1, 3))
    ego_motion_m = float(np.linalg.norm(origins - origins[-1], axis=1).max())
    return Accumulation(
        points=points,
        n_sweeps_actual=len(sweeps),
        window_ns=window_ns,
        truncated=len(sweeps) < cfg.w_acc_count,
        ego_motion_m=ego_motion_m,
        compensation=_compensation_residual_m(sweep_poses, anchor_pose, to_anchor_ego),
    )


def _compensation_residual_m(
    sweep_poses: list[Transform],
    anchor_pose: Transform,
    to_anchor_ego: np.ndarray,
) -> dict:
    """Closed-form proof that the ego-motion hops run in the right direction.

    §5.2 asks for a static-structure sharpness check. Two cheap statistical
    proxies were tried and BOTH were measured to be confounded on this
    substrate, so neither is used:

      - total occupied voxels, compensated vs not: conflates alignment with
        spatial extent. At 4 m of ego motion the compensated cloud legitimately
        covers several more metres of road, so it occupies MORE voxels while
        being more correct, and the verdict inverts on exactly the fast-moving
        frames where compensation matters (7 keyframes of scene-0655).
      - hit rate against the anchor sweep's voxels: the road under the ego is
        self-similar across sweeps, so UNcompensated points score higher
        (measured ratio 0.63 on scene-0655) for a reason that has nothing to do
        with the transform.

    What replaces them is exact rather than statistical. Each sweep's sensor
    origin, carried through hops 2 and 3, must land at the ego displacement
    between that sweep's pose and the anchor's, expressed in the anchor's ego
    frame. That value is known in closed form:

        expected = R_anchor^T (t_sweep - t_anchor)

    A sign error, a swapped pose, or an un-inverted matrix moves the origin to
    the wrong side and the residual jumps from ~1e-12 m to metres. There is no
    threshold to guess: the correct answer is zero to floating-point precision.
    """
    residuals = []
    R_anchor = np.asarray(quaternion_to_rotation_matrix(anchor_pose.rotation_wxyz))
    t_anchor = np.asarray(anchor_pose.translation_m, dtype=np.float64)
    for pose in sweep_poses:
        expected = R_anchor.T @ (np.asarray(pose.translation_m, dtype=np.float64) - t_anchor)
        measured = apply_transform(to_anchor_ego, apply_transform(pose.matrix(), np.zeros((1, 3))))[0]
        residuals.append(float(np.linalg.norm(measured - expected)))
    return {
        "max_residual_m": max(residuals) if residuals else 0.0,
        "n_sweeps": len(residuals),
        "ok": (max(residuals) if residuals else 0.0) <= COMPENSATION_TOLERANCE_M,
        "tolerance_m": COMPENSATION_TOLERANCE_M,
    }


# ---------------------------------------------------------------------------
# Per-keyframe ingestion
# ---------------------------------------------------------------------------


def ingest_keyframe(
    sub: Substrate,
    scene: dict,
    sample: dict,
    lidar_by_time: list[dict],
    cfg: IngestConfig,
    spec: RegionSpec,
    out_dir: str,
) -> tuple[KeyframeRecord, dict]:
    """One keyframe: both clouds, the fitted planes, the ledger, the I-3 record."""
    channel_records = {
        sub.channel(r): r
        for r in sub.sample_data_by_scene[scene["token"]]
        if r["is_key_frame"] and r["sample_token"] == sample["token"]
    }
    anchor = channel_records["LIDAR_TOP"]
    anchor_ns = to_unix_ns(anchor["timestamp"], "unix_us")

    # --- sweeps inside W_acc, anchor included ------------------------------
    window_start_us = anchor["timestamp"] - cfg.w_acc_duration_ns // 1000
    sweeps = [r for r in lidar_by_time if window_start_us <= r["timestamp"] <= anchor["timestamp"]]
    acc = accumulate(sub, sweeps, anchor, cfg)

    # --- single sweep: T_ego_lidar applied exactly once ---------------------
    raw = read_pcd_bin(sub.blob(anchor))
    n_raw_pts = int(raw.shape[0])
    raw, n_stereo_thinned = thin_stereo(raw, cfg.stereo_rings, cfg.stereo_stride)
    t_ego_lidar = Transform.from_nuscenes(
        sub.by_token("calibrated_sensor.json")[anchor["calibrated_sensor_token"]],
        source_frame=LIDAR,
        parent_frame=EGO,
    )
    single = np.column_stack(
        [apply_transform(t_ego_lidar.matrix(), raw[:, :3].astype(np.float64)), raw[:, 3:5].astype(np.float64)]
    )
    n_lidar_pts = int(single.shape[0])

    # --- stereo fusion: ZED points into the SINGLE SWEEP --------------------
    # Same sensor -> ego hop as the LiDAR, applied exactly once, from each ZED's
    # own calibrated_sensor. Their clouds are already in a body frame (X fwd,
    # Y left, Z up), exactly like LIDAR_TOP's, so the transform is identical in
    # form. The ring column carries the source so nothing downstream has to
    # guess which points are stereo.
    n_stereo_pts = {}
    if cfg.fuse_stereo:
        for channel, tag in STEREO_CHANNELS.items():
            record = channel_records.get(channel)
            if record is None:
                continue
            stereo_raw = read_pcd_bin(sub.blob(record))
            t_ego_stereo = Transform.from_nuscenes(
                sub.by_token("calibrated_sensor.json")[record["calibrated_sensor_token"]],
                source_frame=LIDAR,
                parent_frame=EGO,
            )
            xyz = apply_transform(t_ego_stereo.matrix(), stereo_raw[:, :3].astype(np.float64))
            block = np.column_stack([
                xyz,
                stereo_raw[:, 3:4].astype(np.float64),          # Rec.709 luminance, NOT reflectivity
                np.full((xyz.shape[0], 1), float(tag)),         # provenance in the ring slot
            ])
            single = np.vstack([single, block])
            n_stereo_pts[channel] = int(block.shape[0])

    # --- fit on the accumulation, apply to the single sweep -----------------
    ground = fit_ground_planes(acc.points, cfg, sample["token"])
    planes = ground.planes

    ledgers = {}
    kept = {}
    for kind, cloud, prune in (
        ("single_sweep", single, True),
        ("accumulated", acc.points, cfg.prune_accumulated),
    ):
        sectors = sector_index(cloud[:, :2], cfg.n_sectors)
        m_input = np.ones(cloud.shape[0], dtype=bool)
        # Ground: discard |z - ground| < band. Points BELOW ground - band survive
        # by the letter of the spec; they are counted separately rather than
        # quietly dropped, because a large count there means the plane is wrong.
        distance = ground_distance(cloud, planes, sectors)
        m_ground = m_input & (np.abs(distance) >= cfg.ground_band_m)
        # Range: E membership is decided by in_region() and nowhere else (§1.10).
        m_range = m_ground & (in_region(cloud[:, 0], cloud[:, 1], spec) if prune else np.ones_like(m_ground))
        m_height = m_range & ((cloud[:, 2] <= cfg.height_cap_m) if prune else np.ones_like(m_range))

        ledger = FilterLedger.build(
            kind, sectors,
            {"input": m_input, "post_ground": m_ground, "post_range": m_range, "post_height": m_height},
            cfg.n_sectors,
        )
        ledger.extra = {
            "pruned": prune,
            "n_below_ground_band": int(np.count_nonzero(distance <= -cfg.ground_band_m)),
            "n_would_prune_range": int(np.count_nonzero(m_ground & ~in_region(cloud[:, 0], cloud[:, 1], spec))),
            "n_would_prune_height": int(np.count_nonzero(m_ground & (cloud[:, 2] > cfg.height_cap_m))),
        }
        ledgers[kind] = ledger
        kept[kind] = cloud[m_height]

    scene_dir = os.path.join(out_dir, "clouds", scene["name"])
    single_path = write_and_path(scene_dir, "single_sweep", sample["token"], kept["single_sweep"])
    acc_path = write_and_path(scene_dir, "accumulated", sample["token"], kept["accumulated"])

    cameras = {}
    for channel in RING_CAMERAS:
        record = channel_records[channel]
        cameras[channel] = CameraObservation(
            channel=channel,
            sample_data_token=record["token"],
            path=record["filename"],
            # MEASURED offset, not a nominal value (§1.2).
            dt_ns=to_unix_ns(record["timestamp"], "unix_us") - anchor_ns,
            ego_pose_token=record["ego_pose_token"],
            calibrated_sensor_token=record["calibrated_sensor_token"],
            width_px=record["width"],
            height_px=record["height"],
        )

    keyframe = KeyframeRecord(
        keyframe_token=sample["token"],
        scene_token=scene["token"],
        t_ns=anchor_ns,
        time_base="unix_ns",
        lidar_sample_data_token=anchor["token"],
        lidar_path=anchor["filename"],
        lidar_ego_pose_token=anchor["ego_pose_token"],
        lidar_calibrated_sensor_token=anchor["calibrated_sensor_token"],
        cameras=cameras,
        single_sweep_cloud=CloudArtifact(
            path=single_path, cloud_kind="single_sweep", frame=EGO,
            n_points=int(kept["single_sweep"].shape[0]), n_sweeps_actual=1, window_ns=0,
        ),
        accumulated_cloud=CloudArtifact(
            path=acc_path, cloud_kind="accumulated", frame=EGO,
            n_points=int(kept["accumulated"].shape[0]),
            n_sweeps_actual=acc.n_sweeps_actual, window_ns=acc.window_ns,
        ),
        coverage_config=cfg.coverage_config,
        is_first_in_scene=sample["token"] == scene["first_sample_token"],
    )

    diagnostics = {
        "keyframe_token": sample["token"],
        "scene": scene["name"],
        "t_ns": anchor_ns,
        "is_first_in_scene": keyframe.is_first_in_scene,
        "accumulation": {
            "n_sweeps_actual": acc.n_sweeps_actual,
            "n_sweeps_nominal": cfg.w_acc_count,
            "window_ns": acc.window_ns,
            "window_ns_nominal": cfg.w_acc_duration_ns,
            "truncated": acc.truncated,
            "ego_motion_m": acc.ego_motion_m,
        },
        "ego_compensation_check": acc.compensation,
        # What the fused single sweep is made of, before any filtering. Stage 6's
        # ">= 5 returns" gate now counts stereo points too, so the split has to be
        # on the record or the gate stops being auditable.
        "single_sweep_sources": {
            "n_lidar": n_lidar_pts,
            "n_stereo": n_stereo_pts,
            "n_total": n_lidar_pts + sum(n_stereo_pts.values()),
            "ring_tags": dict(STEREO_CHANNELS),
            "note": "stereo is ZED depth, not lidar: <= 20 m, error grows with range squared",
            # Fused-in stereo (rings declared by the profile) thinned at read
            # time; n_lidar above counts the cloud AFTER thinning.
            "stereo_thinning": {
                "rings": list(cfg.stereo_rings),
                "stride": cfg.stereo_stride,
                "n_raw_in_file": n_raw_pts,
                "n_removed": n_stereo_thinned,
            },
        },
        "ground_reference_plane": ground.reference.as_dict() if ground.reference is not None else None,
        "sector_planes": [p.as_dict() for p in planes],
        "ledgers": [ledgers["single_sweep"].as_dict(), ledgers["accumulated"].as_dict()],
    }
    return keyframe, diagnostics


def write_and_path(scene_dir: str, kind: str, token: str, cloud: np.ndarray) -> str:
    path = os.path.join(scene_dir, kind, f"{token}.pcd.bin")
    write_pcd_bin(path, cloud)
    return path


# ---------------------------------------------------------------------------
# Upstream binding
# ---------------------------------------------------------------------------


def load_allowlist(paths: Paths, allowlist_path: str, *, accept_degraded: bool = False) -> dict:
    """Read `usable_scenes.json` and REFUSE on a fingerprint mismatch (§1.8, §1.9).

    Also gates on Stage 0's completion marker (C16): absent means the probe did
    not finish and the allowlist may be stale; degraded (scenes excluded, or the
    partition unsatisfiable) is consumable only under `accept_degraded_upstream`.
    """
    if not os.path.isfile(allowlist_path):
        raise UpstreamRefusal(
            f"{allowlist_path} not found — run pipeline.stage0_data_probe.probe first. "
            "usable_scenes.json is the only scene list any stage reads."
        )
    stage0_dir = os.path.dirname(os.path.abspath(allowlist_path))
    marker = read_marker(stage0_dir)
    if marker is None:
        raise UpstreamRefusal(
            f"{stage0_dir} has no completion marker: the probe did not finish, and this allowlist "
            "is indistinguishable from a stale one (§1.9). Re-run pipeline.stage0_data_probe.probe"
        )
    if marker.degraded and not accept_degraded:
        raise UpstreamRefusal(
            f"Stage 0 completed DEGRADED ({'; '.join(marker.causes)}). Ingesting a reduced scene "
            "set is a recorded decision, not a default: re-run with --accept-degraded-upstream (C16)"
        )
    with open(allowlist_path, "r", encoding="utf-8") as fh:
        allowlist = json.load(fh)

    manifest = Record.from_dict(allowlist["manifest"])
    actual = metadata_fingerprint(paths)
    if manifest.metadata_fingerprint != actual:
        raise UpstreamRefusal(
            "metadata fingerprint mismatch — the allowlist was computed against different "
            f"bytes than the dataroot in front of us.\n  allowlist: {manifest.metadata_fingerprint}"
            f"\n  dataroot : {actual}\n  dataroot path: {paths.dataroot}"
        )
    if os.path.realpath(manifest.dataroot_realpath) != paths.dataroot:
        raise UpstreamRefusal(
            f"allowlist dataroot {manifest.dataroot_realpath!r} != configured {paths.dataroot!r}"
        )
    return allowlist


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    paths: Paths,
    allowlist: dict,
    cfg: IngestConfig,
    out_dir: str,
    scene_names: list[str] | None,
) -> tuple[dict, int]:
    # From here on the output tree is being rewritten: any marker still standing
    # describes the PREVIOUS run, so it comes down before the first write (C16).
    clear_markers(out_dir)
    sub = Substrate.load(paths)
    spec = region_spec_from_config({"coverage_config": cfg.coverage_config, "r_max_m": cfg.range_cap_m})
    scenes_by_name = {s["name"]: s for s in sub.tables["scene.json"]}

    selected = [s["name"] for s in allowlist["scenes"]]
    if scene_names:
        unknown = sorted(set(scene_names) - set(selected))
        if unknown:
            raise UpstreamRefusal(f"requested scene(s) not in the allowlist: {unknown}")
        selected = [n for n in selected if n in set(scene_names)]

    started = time.time()
    per_scene: list[dict] = []
    degraded = False

    for name in selected:
        scene = scenes_by_name[name]
        lidar = sorted(
            (r for r in sub.sample_data_by_scene[scene["token"]] if sub.channel(r) == "LIDAR_TOP"),
            key=lambda r: r["timestamp"],
        )
        samples = sub.scene_samples(scene)

        records: list[KeyframeRecord] = []
        diagnostics: list[dict] = []
        for sample in samples:
            keyframe, diag = ingest_keyframe(sub, scene, sample, lidar, cfg, spec, out_dir)
            records.append(keyframe)
            diagnostics.append(diag)

        scene_dir = os.path.join(out_dir, "scenes", name)
        os.makedirs(scene_dir, exist_ok=True)
        # I-3 through the raising boundary: validated on write AND on read-back.
        write_records(os.path.join(scene_dir, "keyframes.jsonl"), records)
        write_json_atomic(
            os.path.join(scene_dir, "filter_diagnostics.json"),
            {"spec": STAGE_SPEC, "scene": name, "config": cfg.as_dict(), "keyframes": diagnostics},
        )

        summary = summarise_scene(name, diagnostics, cfg)
        per_scene.append(summary)
        degraded = degraded or summary["degraded"]
        print(
            f"  {name}  {summary['n_keyframes']:>3} kf  "
            f"single {summary['single_sweep']['survivors']['input']:>7} -> "
            f"{summary['single_sweep']['survivors']['post_height']:>7} pts "
            f"({summary['single_sweep']['retained_fraction']:.1%})  "
            f"acc {summary['accumulated']['survivors']['input']:>8} -> "
            f"{summary['accumulated']['survivors']['post_height']:>8}  "
            + f"ego-comp residual {summary['compensation_max_residual_m']:.2e} m"
            + ("  DEGRADED" if summary["degraded"] else "")
        )

    manifest = {
        "spec": STAGE_SPEC,
        "stage": STAGE,
        "seed": cfg.global_seed,
        "config": cfg.as_dict(),
        "upstream": {
            "metadata_fingerprint": allowlist["manifest"]["metadata_fingerprint"],
            "fingerprint_spec": allowlist["manifest"]["fingerprint_spec"],
            "usable_scenes_spec": allowlist["spec"],
        },
        "paths": paths.as_dict(),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "elapsed_s": round(time.time() - started, 2),
        "scenes": per_scene,
        "totals": aggregate(per_scene),
    }
    return manifest, EXIT_DEGRADED if degraded else EXIT_OK


def summarise_scene(name: str, diagnostics: list[dict], cfg: IngestConfig) -> dict:
    """Roll the per-keyframe ledgers up, keeping the per-sector axis intact."""
    out: dict = {"scene": name, "n_keyframes": len(diagnostics)}
    for kind_index, kind in enumerate(("single_sweep", "accumulated")):
        survivors = {f: 0 for f in FILTERS}
        per_sector = [{"sector": s, **{f: 0 for f in FILTERS}} for s in range(cfg.n_sectors)]
        for diag in diagnostics:
            ledger = diag["ledgers"][kind_index]
            for f in FILTERS:
                survivors[f] += ledger["survivors"][f]
            for sector_row in ledger["per_sector"]:
                for f in FILTERS:
                    per_sector[sector_row["sector"]][f] += sector_row[f]
        removed = {
            f"{FILTERS[i]}->{FILTERS[i + 1]}": survivors[FILTERS[i]] - survivors[FILTERS[i + 1]]
            for i in range(len(FILTERS) - 1)
        }
        out[kind] = {
            "survivors": survivors,
            "removed_by_filter": removed,
            "retained_fraction": survivors["post_height"] / survivors["input"] if survivors["input"] else 0.0,
            "per_sector": per_sector,
            # The number the diagnostics exist for: if a sector retains almost
            # nothing, "detection found nothing there" and "the filter emptied
            # it" are the same observation until you look here.
            "sector_retained_fraction": [
                (row["post_height"] / row["input"]) if row["input"] else 0.0 for row in per_sector
            ],
        }

    residuals = [d["ego_compensation_check"]["max_residual_m"] for d in diagnostics]
    fallbacks = sum(
        1 for d in diagnostics for p in d["sector_planes"]
        if p["fallback"] and not p["fallback"].startswith("rejected_")
    )
    rejections = sum(1 for d in diagnostics for p in d["sector_planes"] if p.get("rejected_fit"))
    tilts = sum(1 for d in diagnostics for p in d["sector_planes"] if p["implausible_tilt"])
    bad_compensation = sum(1 for d in diagnostics if not d["ego_compensation_check"]["ok"])
    out.update(
        {
            "compensation_max_residual_m": max(residuals) if residuals else 0.0,
            "n_keyframes_failing_compensation": bad_compensation,
            "n_sector_fits_fallback": fallbacks,
            "n_sector_fits_rejected": rejections,
            "n_sector_fits_implausible_tilt": tilts,
            "n_keyframes_truncated_window": sum(1 for d in diagnostics if d["accumulation"]["truncated"]),
            "rejection_rate": rejections / max(1, len(diagnostics) * cfg.n_sectors),
            # A rejection is a HANDLED condition with the substitution recorded;
            # a fallback (no fit possible at all) and a compensation failure are not.
            "degraded": bool(
                bad_compensation
                or fallbacks
                or rejections / max(1, len(diagnostics) * cfg.n_sectors) > cfg.degraded_rejection_rate
            ),
        }
    )
    return out


def aggregate(per_scene: list[dict]) -> dict:
    totals: dict = {}
    for kind in ("single_sweep", "accumulated"):
        survivors = {f: sum(s[kind]["survivors"][f] for s in per_scene) for f in FILTERS}
        totals[kind] = {
            "survivors": survivors,
            "removed_by_filter": {
                f"{FILTERS[i]}->{FILTERS[i + 1]}": survivors[FILTERS[i]] - survivors[FILTERS[i + 1]]
                for i in range(len(FILTERS) - 1)
            },
            "retained_fraction": survivors["post_height"] / survivors["input"] if survivors["input"] else 0.0,
        }
    totals["n_keyframes"] = sum(s["n_keyframes"] for s in per_scene)
    totals["n_sector_fits_fallback"] = sum(s["n_sector_fits_fallback"] for s in per_scene)
    totals["n_sector_fits_rejected"] = sum(s["n_sector_fits_rejected"] for s in per_scene)
    totals["n_sector_fits_implausible_tilt"] = sum(s["n_sector_fits_implausible_tilt"] for s in per_scene)
    totals["n_keyframes_failing_compensation"] = sum(
        s["n_keyframes_failing_compensation"] for s in per_scene
    )
    totals["compensation_max_residual_m"] = max(
        (s["compensation_max_residual_m"] for s in per_scene), default=0.0
    )
    return totals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--allowlist", default=None, help="default <work_root>/stage0_data_probe/usable_scenes.json")
    parser.add_argument("--out-dir", default=None, help="default <work_root>/stage1_ingestion")
    parser.add_argument("--scenes", nargs="*", default=None, help="subset of allowlist scene names")
    parser.add_argument("--seed", type=int, default=None, help="override the global seed (recorded)")
    parser.add_argument(
        "--accept-degraded-upstream",
        action="store_true",
        help="consume a DEGRADED (complete, quality-flagged) Stage 0 output; recorded (C16)",
    )
    args = parser.parse_args(argv)

    try:
        paths = load_paths(args.paths)
    except PathValidationError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    allowlist_path = args.allowlist or os.path.join(
        paths.work_root, "stage0_data_probe", "usable_scenes.json"
    )
    out_dir = args.out_dir or os.path.join(paths.work_root, STAGE)
    assert_dataroot_read_only(paths, out_dir)

    cfg = IngestConfig(
        accept_degraded_upstream=args.accept_degraded_upstream,
        **({"global_seed": args.seed} if args.seed is not None else {}),
    )

    try:
        allowlist = load_allowlist(paths, allowlist_path, accept_degraded=cfg.accept_degraded_upstream)
        manifest, code = run(paths, allowlist, cfg, out_dir, args.scenes)
    except UpstreamRefusal as exc:
        print(f"REFUSING TO START: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (SchemaValidationError, PathValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)
    # Three-state marker (§1.9, C16): _SUCCESS = complete and clean;
    # _SUCCESS.degraded = complete, quality-flagged, causes named; absent (the
    # refusal paths above) = incomplete. "Complete but two scenes crossed a
    # quality threshold" withheld the marker before C16, and the whole pipeline
    # downstream of this stage read as broken.
    write_marker(
        out_dir,
        manifest["upstream"]["metadata_fingerprint"],
        degraded=code == EXIT_DEGRADED,
        causes=[
            f"{s['scene']}: rejection_rate {s['rejection_rate']:.3f} > {cfg.degraded_rejection_rate}"
            if s["rejection_rate"] > cfg.degraded_rejection_rate
            else f"{s['scene']}: compensation/fallback degradation"
            for s in manifest["scenes"]
            if s["degraded"]
        ],
    )

    t = manifest["totals"]
    print(f"keyframes            : {t['n_keyframes']}")
    for kind in ("single_sweep", "accumulated"):
        block = t[kind]
        print(f"{kind:<21}: {block['survivors']['input']} -> {block['survivors']['post_height']} "
              f"({block['retained_fraction']:.1%} retained)  removed {block['removed_by_filter']}")
    print(f"sector fits fallback : {t['n_sector_fits_fallback']}")
    print(f"sector fits rejected : {t['n_sector_fits_rejected']}  (mis-fit guard substituted the global plane)")
    print(f"implausible tilt     : {t['n_sector_fits_implausible_tilt']}")
    print(f"ego-comp failures    : {t['n_keyframes_failing_compensation']}"
          f"  (max residual {t['compensation_max_residual_m']:.3e} m,"
          f" tolerance {COMPENSATION_TOLERANCE_M:.0e} m)")
    print(f"wrote {out_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
