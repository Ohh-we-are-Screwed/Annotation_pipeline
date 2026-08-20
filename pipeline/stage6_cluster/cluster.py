#!/usr/bin/env python3
"""Stage 6 — per-instance BEV clustering + L-shape box fit (§5.7, §1.6, Phase 8).

Input is Stage 5's painted points: for each mask instance, the subset of the
**ground-filtered single-sweep** cloud (§1.4) that instance owns, in **ego frame**
(§1.1). Output is one oriented box per instance, `size` in nuScenes **[w, l, h]**
order (§3.2), yaw about +z from +x (ISO 8855).

**Clustering scope is per mask instance** (§1.6, reading (a)). "Class-conditional"
names *which epsilon is used* and nothing else. The rejected reading — pool a
class across the frame, cluster once, keep the largest — produces **one box per
class per frame**: in a parking row or gridlock (the regime this project exists
for) adjacent cars merge under any epsilon large enough to hold one car, and
every other instance of that class is silently discarded. The output is clean,
well-formed, and missing most objects.

**"Keep largest cluster" is the reprojection-ghost filter**, not a quality knob.
Stage 5 does no occlusion reasoning by design: the wall behind a car projects
into the car's mask and is painted as the car (§5.6). Those points are a second
cluster, further away, and dropping them is this stage's job. The count dropped
is recorded per instance, because "the ghost filter removed 80 % of the object"
and "it removed the ghost" are the same field.

**Determinism** (§1.9). DBSCAN's labels depend on the order points arrive in, and
that order comes from a file read. So the instance's points are put in a
canonical lexicographic order **before** clustering, neighbour queries return
sorted indices, and equal-size clusters are broken by lowest mean range, then by
lowest canonical index. Two runs on the same input give byte-identical boxes.
DBSCAN is implemented here rather than imported: it is 40 lines, scikit-learn is
not a dependency of this pipeline, and the ordering guarantees above are the
whole point.

**Epsilon comes from the priors file** — `comprehensive.md` §7.2's
`eps ~ 0.6 x mean footprint diagonal`, derived per class from the `priors` scene
subset — never from `Annotation_pipeline.md`'s hardcoded table (X-6), whose class
names (`cyclist`, `traffic_cone`, `car`) are not nuScenes categories and not
prompt phrases, so every lookup keyed on them misses silently and hands every
class the same default. A class with no prior here is recorded as
`eps_source: "config_fallback:..."` on **every affected box** and counted in the
manifest; the fallback is declared, not discovered.

**The near-square policy** (§5.7). Pedestrians, cones and barriers have a
*systematic* 90-degree yaw ambiguity, not an occasional one: the footprint is
square to within noise, so the fitter's chosen axis is arbitrary. Policy: the
long BEV side is the heading axis, which makes `w <= l` true by construction, and
a near-square footprint sets `yaw_ambiguous`. This matters downstream because
Stage 8 anchors the near face and grows the far one — under a 90-degree yaw error
it grows the box sideways into the neighbouring lane, deterministically and
invisibly (§5.9).

**Yaw is asserted against `conventions.py`, never trusted from the fitter.** Every
box re-projects its own points onto `(cos yaw, sin yaw)` and `(-sin yaw, cos yaw)`
and asserts the extents reproduce the stored `[w, l, h]` and the stored centre,
then round-trips yaw -> quaternion -> yaw. A sign or axis error in the fitter
produces boxes that are plausible everywhere except in the one comparison that is
made here.

**What this stage does NOT decide.** The 180-degree heading direction: a
symmetric footprint of points cannot say which end is the front, and inventing an
answer here would be a guess dressed as geometry. Stage 7's yaw-consistency
enforcement along tracks is the producer of that bit (§4, §5.8); until then
`yaw_rad` names an axis, and `yaw_axis_only: true` says so in every record.

    python3 -m pipeline.stage6_cluster.cluster [--paths configs/paths.yaml]

Exit codes:
    0  every keyframe clustered under contract
    1  ran, but at least one scene produced no box from any instance
    2  upstream contract broken; nothing was written
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.common.conventions import (  # noqa: E402
    EGO,
    quaternion_from_yaw_rad,
    wrap_to_pi_rad,
    yaw_rad_from_quaternion,
)
from pipeline.common.eval_region import R1_DEFAULT, R2_DEFAULT, RegionSpec, in_region  # noqa: E402
from pipeline.common.paths import (  # noqa: E402
    PathValidationError,
    Paths,
    assert_dataroot_read_only,
    load_paths,
    metadata_fingerprint,
)
from pipeline.common.manifest import (  # noqa: E402
    UpstreamRefusal,
    clear_markers,
    require_upstream,
    write_json_atomic,
    write_jsonl_atomic,
    write_marker,
)
from pipeline.stage1_ingestion.ingest import read_pcd_bin  # noqa: E402
from pipeline.stage6_cluster.priors import PRIORS_NAME, Priors, load_priors  # noqa: E402

STAGE = "stage6_cluster"
STAGE_SPEC = "dhakascenes-pilot/stage6_cluster/v1"

EXIT_OK = 0
EXIT_DEGRADED = 1
EXIT_REFUSED = 2

FIT_CRITERIA: tuple[str, ...] = ("closeness", "area")
MISSING_PRIOR_POLICIES: tuple[str, ...] = ("fallback", "refuse")

# §11 decision 3 / P1-5: epsilon is a TUNED quantity, and tuning it on the scored
# scenes is the leak the scene partition exists to prevent. The priors file this
# stage consumes must record that derivation scope — checked at load, not assumed
# from the filename.
PRIORS_SCENE_SUBSET = "priors"

# Per-instance outcomes. Emitted as `status` on every row, including the rows
# with no box: an instance that produced nothing is a reportable outcome and its
# disappearance from the file would be indistinguishable from it never existing.
STATUS_FIT = "fit"
STATUS_NO_POINTS = "no_points"
STATUS_BELOW_MIN_SAMPLES = "below_min_samples"
STATUS_ALL_NOISE = "all_noise"
STATUSES: tuple[str, ...] = (STATUS_FIT, STATUS_NO_POINTS, STATUS_BELOW_MIN_SAMPLES, STATUS_ALL_NOISE)


class ClusterContractError(RuntimeError):
    """A Stage 6 input, or a box this stage produced, violated the contract."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterConfig:
    """Stage 6 tunables. None of these may appear as a literal in the code below."""

    cloud_kind: str = "single_sweep"

    # --- DBSCAN (§1.6, §7.2) ---
    cluster_space: str = "bev_xy"
    min_samples: int = 3
    eps_fallback_m: float = 0.75
    on_missing_prior: str = "fallback"
    canonical_sort: str = "lexicographic_xyz_then_point_index"
    cluster_tie_break: str = "largest_then_lowest_mean_range_then_lowest_canonical_index"

    # --- the near-cut retry (C24) ---
    near_cut_enabled: bool = True
    near_cut_depth_m: float = 5.0
    near_cut_prior_multiple: float = 2.0

    # --- L-shape fit (§5.7) ---
    fit_criterion: str = "closeness"
    angle_step_rad: float = math.radians(1.0)
    refine_passes: int = 3
    refine_factor: float = 10.0
    closeness_d0_m: float = 0.01
    min_extent_m: float = 0.05

    # --- the near-square policy (§5.7) ---
    near_square_ratio: float = 0.80
    min_points_for_yaw: int = 10

    # --- assertions (§3.2) ---
    extent_assert_tol_m: float = 1e-6
    yaw_assert_tol_rad: float = 1e-9

    # --- diagnostics, not filters ---
    min_points_per_instance: int = 5
    record_eval_region: bool = True

    # --- upstream gate (C16) ---
    accept_degraded_upstream: bool = False

    # --- determinism (§1.9) ---
    global_seed: int = 20260812

    provenance: dict = field(
        default_factory=lambda: {
            "cloud_kind": (
                "single_sweep: §1.4. Clustering the accumulation smears dynamic objects along the "
                "direction of travel, inflating box length and biasing L-shape yaw toward the "
                "motion direction — which is exactly the quantity §7.3.7 and §8.2.1 (APH) treat as "
                "load-bearing"
            ),
            "cluster_space": (
                "bev_xy: comprehensive.md §7.3.6 clusters in BEV, and epsilon is defined from a "
                "footprint diagonal (§7.2) — a 2D quantity. Clustering in 3D with the same epsilon "
                "would silently split tall objects"
            ),
            "min_samples": "arbitrary pilot value, needs tuning; config per §5.7",
            "eps_fallback_m": (
                "declared fallback for a class with no prior. Recorded per box in eps_source and "
                "counted in the manifest — the X-6 failure is a SILENT default, not a default"
            ),
            "on_missing_prior": "fallback (declared, counted) or refuse (stop the run)",
            "canonical_sort": (
                "§1.6: DBSCAN's labels depend on input point order, which depends on file read "
                "order. The sort makes the run reproducible byte-for-byte (§1.9)"
            ),
            "near_cut_enabled": (
                "C24. Stage 5 paints with no occlusion reasoning (lift.py:64-69) and delegates the "
                "resulting far-surface points to THIS stage's keep-largest filter — which measurably "
                "cannot fire, because eps exceeds the object-to-background depth gap (p50 1.50 m) and "
                "chains the two into ONE cluster in 64% of oversized instances. The retry is the "
                "delegated filter finally being implemented, not a second one beside it"
            ),
            "near_cut_depth_m": (
                "measured, 2026-08-14, two independent implementations over all 10 scenes / 6104 "
                "boxes. Keep points within this distance of the instance's OWN nearest return: a "
                "mask is generated by the nearest surface along the ray, so the object lies at the "
                "near end of its own painted depth range. Prior-free by design — the class label is "
                "wrong on 37% of oversized instances, so a prior-derived budget is keyed to an "
                "unreliable label. On the 366 rows the Stage 9 spatial gate rejects, boxes placed on "
                "the real object at BEV IoU >= 0.5 go 1 -> 161. Arbitrary in magnitude, needs tuning"
            ),
            "near_cut_prior_multiple": (
                "the retry TRIGGER, mirroring stage9_qa/gate.py:spatial_gate exactly (sorted "
                "measured w,l against sorted prior mu, same 2.0 multiplier) so Stage 6 repairs "
                "precisely the boxes Stage 9 would reject and no others. Gating matters: applied "
                "unconditionally the same cut loses 306 boxes and drags mean IoU BELOW baseline; "
                "gated it loses 7 and raises precision@0.5 from 17.37% to 19.99%"
            ),
            "cluster_tie_break": (
                "§1.6's deterministic tie-break, in the order the spec states it. Range is measured "
                "from the EGO origin, not the LiDAR origin: the tie-break only has to be a total "
                "order, and Stage 8's near-face anchor — where the ~0.94 m offset does change an "
                "answer — takes the sensor origin from the calibration instead"
            ),
            "fit_criterion": (
                "closeness: Zhang et al. 2017's L-shape criterion, which fits the two VISIBLE faces "
                "rather than the convex hull. area is the min-area rectangle and is kept as a "
                "documented alternative, not as a fallback"
            ),
            "angle_step_rad": "1 deg coarse scan over [0, pi/2), refined; a rectangle is pi/2-periodic",
            "refine_passes": "3 passes at 1/10 step each: 1 deg -> 0.001 deg, deterministic",
            "closeness_d0_m": "Zhang's d0 floor; stops 1/d exploding for a point on an edge",
            "min_extent_m": (
                "I-4 requires strictly positive size (§A.1). A single-scanline cluster has zero "
                "height and a collinear one zero width; the clamp is recorded per box, never hidden"
            ),
            "near_square_ratio": (
                "§5.7: near-square footprints have a SYSTEMATIC 90 deg ambiguity. Arbitrary "
                "threshold, needs tuning; the flag it sets is what Stage 8 and Stage 9 read"
            ),
            "min_points_for_yaw": (
                "arbitrary; below this a footprint has no shape to fit and the yaw is flagged "
                "ambiguous regardless of the aspect ratio"
            ),
            "min_points_per_instance": (
                "§7.3.9 / §6.3's '>= 5 returns', counted on the SINGLE-SWEEP cloud pre-inflation "
                "(§1.4). Recorded here, gated in Stage 9 — Stage 6 drops nothing"
            ),
            "accept_degraded_upstream": "C16 — consuming a DEGRADED (complete, quality-flagged) "
            "Stage 5 output is an explicit recorded decision, never a default",
            "global_seed": "§1.9, one global seed, recorded (nothing here samples; recorded anyway)",
        }
    )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.cloud_kind != "single_sweep":
            errors.append(f"cloud_kind={self.cloud_kind!r}: §1.4 clusters the single-sweep cloud")
        if self.cluster_space != "bev_xy":
            errors.append(f"cluster_space={self.cluster_space!r} is not implemented; epsilon is a BEV quantity")
        if self.fit_criterion not in FIT_CRITERIA:
            errors.append(f"fit_criterion={self.fit_criterion!r} is not one of {FIT_CRITERIA}")
        if self.on_missing_prior not in MISSING_PRIOR_POLICIES:
            errors.append(f"on_missing_prior={self.on_missing_prior!r} is not one of {MISSING_PRIOR_POLICIES}")
        if self.min_samples < 1:
            errors.append(f"min_samples={self.min_samples} must be >= 1")
        if not self.eps_fallback_m > 0.0:
            errors.append(f"eps_fallback_m={self.eps_fallback_m} must be positive")
        if not self.near_cut_depth_m > 0.0:
            errors.append(f"near_cut_depth_m={self.near_cut_depth_m} must be positive")
        if not self.near_cut_prior_multiple >= 1.0:
            errors.append(
                f"near_cut_prior_multiple={self.near_cut_prior_multiple} must be >= 1: a trigger "
                "below the class mean would fire on correctly-sized boxes"
            )
        if not 0.0 < self.angle_step_rad <= math.pi / 4:
            errors.append(f"angle_step_rad={self.angle_step_rad} must be in (0, pi/4]")
        if self.refine_passes < 0:
            errors.append(f"refine_passes={self.refine_passes} must be >= 0")
        if self.refine_factor <= 1.0:
            errors.append(f"refine_factor={self.refine_factor} must be > 1")
        if not 0.0 < self.near_square_ratio <= 1.0:
            errors.append(f"near_square_ratio={self.near_square_ratio} must be in (0, 1]")
        if not self.min_extent_m > 0.0:
            errors.append(f"min_extent_m={self.min_extent_m} must be positive (I-4 size must be > 0)")
        return errors

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def region_for(coverage_config: str) -> RegionSpec:
    if coverage_config == "R1":
        return R1_DEFAULT
    if coverage_config == "R2":
        return R2_DEFAULT
    raise ClusterContractError(f"coverage_config={coverage_config!r} is not R1 or R2")


# ---------------------------------------------------------------------------
# DBSCAN — deterministic, BEV, per instance (§1.6, §1.9)
# ---------------------------------------------------------------------------


class _UniformGrid:
    """Fixed-radius neighbour queries over a uniform BEV grid of cell size eps.

    Cells are eps wide, so every neighbour within eps lies in the 3x3 block
    around a point's own cell. Returned indices are ASCENDING, which is half of
    what makes the labelling reproducible — the other half is the caller's
    canonical point order.
    """

    def __init__(self, points_xy: np.ndarray, eps_m: float) -> None:
        if not (math.isfinite(eps_m) and eps_m > 0.0):
            raise ClusterContractError(f"eps must be positive and finite, got {eps_m!r}")
        self._xy = points_xy
        self._eps2 = float(eps_m) * float(eps_m)
        self._cell = np.floor(points_xy / float(eps_m)).astype(np.int64)
        self._cells: dict[tuple[int, int], list[int]] = {}
        for index, (cx, cy) in enumerate(self._cell):
            self._cells.setdefault((int(cx), int(cy)), []).append(index)

    def query(self, index: int) -> np.ndarray:
        cx, cy = int(self._cell[index, 0]), int(self._cell[index, 1])
        candidates: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.extend(self._cells.get((cx + dx, cy + dy), ()))
        if not candidates:
            return np.empty(0, dtype=np.int64)
        candidate_index = np.asarray(sorted(candidates), dtype=np.int64)
        delta = self._xy[candidate_index] - self._xy[index]
        return candidate_index[np.einsum("ij,ij->i", delta, delta) <= self._eps2]


def dbscan_bev(points_xy: np.ndarray, eps_m: float, min_samples: int) -> np.ndarray:
    """Textbook DBSCAN over BEV points. Returns labels; -1 is noise.

    Written out rather than imported: scikit-learn is not a dependency of this
    pipeline, and the two properties this stage needs from it — ascending seed
    expansion and ascending neighbour order — are implementation details of any
    library version rather than guarantees of it. §1.9 requires byte-identical
    re-runs, so they are guarantees here.
    """
    points_xy = np.asarray(points_xy, dtype=np.float64)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ClusterContractError(f"points_xy must be (N, 2), got {points_xy.shape}")
    n = points_xy.shape[0]
    labels = np.full(n, -1, dtype=np.int64)
    if n == 0:
        return labels

    grid = _UniformGrid(points_xy, eps_m)
    visited = np.zeros(n, dtype=bool)
    next_label = 0
    for seed in range(n):  # ascending: the canonical order IS the label order
        if visited[seed]:
            continue
        visited[seed] = True
        neighbours = grid.query(seed)
        if neighbours.shape[0] < min_samples:
            continue  # noise for now; a later core point may claim it as a border
        labels[seed] = next_label
        queue = deque(int(j) for j in neighbours)
        while queue:
            current = queue.popleft()
            if not visited[current]:
                visited[current] = True
                expansion = grid.query(current)
                if expansion.shape[0] >= min_samples:
                    queue.extend(int(j) for j in expansion)
            if labels[current] < 0:
                labels[current] = next_label
        next_label += 1
    return labels


def canonical_order(points_xyz: np.ndarray, point_index: np.ndarray) -> np.ndarray:
    """The permutation §1.6 requires before any clustering happens.

    Lexicographic on (x, y, z), with the cloud's own row index as the final key
    so the order is total even for coincident points.
    """
    return np.lexsort(
        (point_index, points_xyz[:, 2], points_xyz[:, 1], points_xyz[:, 0])
    ).astype(np.int64)


@dataclass
class ClusterChoice:
    """Which cluster survived the ghost filter, and what it cost."""

    label: int
    member: np.ndarray  # (n,) bool over the canonically ordered points
    ledger: dict


def select_cluster(labels: np.ndarray, points_xyz: np.ndarray) -> ClusterChoice | None:
    """Keep the largest cluster; break ties exactly as §1.6 states.

    Ties are not hypothetical: two 40-point clusters of a car and the wall behind
    it are the ordinary case at range. Size, then lowest mean range (the near
    cluster is the object; the far one is the reprojection ghost), then lowest
    canonical index.
    """
    present = [int(v) for v in np.unique(labels) if v >= 0]
    if not present:
        return None
    ranked: list[tuple[int, float, int, int]] = []
    for label in present:
        member = labels == label
        xyz = points_xyz[member]
        mean_range = float(np.hypot(xyz[:, 0], xyz[:, 1]).mean())
        ranked.append(
            (
                -int(np.count_nonzero(member)),  # largest first
                mean_range,  # then nearest
                int(np.nonzero(member)[0][0]),  # then lowest canonical index
                label,
            )
        )
    ranked.sort()
    chosen = ranked[0]
    member = labels == chosen[3]
    sizes = {int(label): int(np.count_nonzero(labels == label)) for label in present}
    return ClusterChoice(
        label=chosen[3],
        member=member,
        ledger={
            "n_clusters": len(present),
            "cluster_sizes": [sizes[label] for label in present],
            "kept_label": chosen[3],
            "n_points_kept": sizes[chosen[3]],
            "n_points_noise": int(np.count_nonzero(labels < 0)),
            "n_points_other_clusters": int(
                sum(size for label, size in sizes.items() if label != chosen[3])
            ),
            "kept_mean_range_m": round(chosen[1], 4),
            "tie_at_size": sum(1 for r in ranked if r[0] == chosen[0]) > 1,
        },
    )


# ---------------------------------------------------------------------------
# L-shape fit (§5.7)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RectangleFit:
    """The fitter's own answer, before any [w, l, h] convention is applied."""

    theta_rad: float  # angle of the u axis, in [0, pi/2)
    extent_u_m: float
    extent_v_m: float
    centre_u_m: float
    centre_v_m: float
    criterion: str
    score: float
    area_m2: float
    n_angles_evaluated: int


def _project(points_xy: np.ndarray, theta_rad: float) -> tuple[np.ndarray, np.ndarray]:
    cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
    u = points_xy[:, 0] * cos_t + points_xy[:, 1] * sin_t
    v = -points_xy[:, 0] * sin_t + points_xy[:, 1] * cos_t
    return u, v


def _criterion_score(u: np.ndarray, v: np.ndarray, cfg: ClusterConfig) -> tuple[float, float]:
    """(criterion, -area). Higher is better on both, so the search is one comparison.

    The second term exists because the first one **plateaus**. `closeness` floors
    each point's edge distance at `closeness_d0_m`, so every angle that puts all
    points within a centimetre of an edge scores identically — which is the
    ordinary case for a flat wall or a well-sampled vehicle face, and the exactly
    correct angle sits somewhere inside that plateau. Breaking the tie on angle
    order alone would take the plateau's lower EDGE, tilting every such box by up
    to the plateau's width. Breaking it on area takes the tightest rectangle in
    the plateau, which is the answer the criterion was reaching for.
    """
    area = float((u.max() - u.min()) * (v.max() - v.min()))
    if cfg.fit_criterion == "area":
        # The min-area rectangle. Correct for a convex blob, wrong for an L: a
        # LiDAR return set is two faces, and the smallest enclosing rectangle of
        # two faces is pulled by whichever face has more spread.
        return (-area, -area)
    # Zhang et al. 2017 "closeness": reward points sitting ON an edge of the
    # rectangle. This is what makes it an L-shape fit — the two observed faces
    # dominate and the two unobserved ones fall where the observed ones imply.
    d_u = np.minimum(u - u.min(), u.max() - u)
    d_v = np.minimum(v - v.min(), v.max() - v)
    distance = np.minimum(d_u, d_v)
    return (float(np.sum(1.0 / np.maximum(distance, cfg.closeness_d0_m))), -area)


def fit_rectangle(points_xy: np.ndarray, cfg: ClusterConfig) -> RectangleFit:
    """Search theta over [0, pi/2), coarse then refined. Deterministic by construction.

    A rectangle is pi/2-periodic, so the search covers a quarter turn and the
    [w, l, h] convention below picks which of the two axes is the heading. Ties
    go to the tightest rectangle and then to the smaller angle; without a total
    order two runs of the same input can disagree by 90 degrees on a square
    footprint, which is a rotated box with identical dimensions and no way to
    notice.
    """
    step = float(cfg.angle_step_rad)
    lo, hi = 0.0, math.pi / 2.0
    best_theta = 0.0
    best_score = (-math.inf, -math.inf)
    evaluated = 0

    for _ in range(int(cfg.refine_passes) + 1):
        n_steps = max(1, int(round((hi - lo) / step)))
        for i in range(n_steps + 1):
            theta = lo + i * step
            if theta > hi + 1e-12:
                break
            u, v = _project(points_xy, theta)
            score = _criterion_score(u, v, cfg)
            evaluated += 1
            if score > best_score or (score == best_score and theta < best_theta):
                best_score = score
                best_theta = theta
        lo, hi = best_theta - step, best_theta + step
        step = step / float(cfg.refine_factor)

    u, v = _project(points_xy, best_theta)
    return RectangleFit(
        theta_rad=float(best_theta),
        extent_u_m=float(u.max() - u.min()),
        extent_v_m=float(v.max() - v.min()),
        centre_u_m=float(0.5 * (u.max() + u.min())),
        centre_v_m=float(0.5 * (v.max() + v.min())),
        criterion=cfg.fit_criterion,
        score=float(best_score[0]),
        area_m2=float(-best_score[1]),
        n_angles_evaluated=evaluated,
    )


# ---------------------------------------------------------------------------
# The box, and the assertions that stand behind it (§3.2, §5.7)
# ---------------------------------------------------------------------------


@dataclass
class Box:
    """One oriented box in ego frame. `size_wlh_m` is [w, l, h], w <= l, always."""

    translation_m: list
    size_wlh_m: list
    yaw_rad: float
    rotation_wxyz: list
    yaw_ambiguous: bool
    yaw_ambiguous_reasons: list
    axis_swapped: bool
    clamped_axes: list
    z_min_m: float
    z_max_m: float
    footprint_diagonal_m: float
    aspect_ratio_w_over_l: float
    fit: dict

    def as_dict(self) -> dict:
        return {
            "translation_m": [round(float(v), 4) for v in self.translation_m],
            "size_wlh_m": [round(float(v), 4) for v in self.size_wlh_m],
            "size_order": "w,l,h",
            "yaw_rad": round(float(self.yaw_rad), 6),
            "rotation_wxyz": [round(float(v), 9) for v in self.rotation_wxyz],
            # The 180 deg half of the heading is NOT decided here (see the module
            # docstring): a symmetric point set cannot say which end is the front.
            "yaw_axis_only": True,
            "yaw_ambiguous": bool(self.yaw_ambiguous),
            "yaw_ambiguous_reasons": list(self.yaw_ambiguous_reasons),
            "axis_swapped": bool(self.axis_swapped),
            "clamped_axes": list(self.clamped_axes),
            "z_min_m": round(float(self.z_min_m), 4),
            "z_max_m": round(float(self.z_max_m), 4),
            "footprint_diagonal_m": round(float(self.footprint_diagonal_m), 4),
            "aspect_ratio_w_over_l": round(float(self.aspect_ratio_w_over_l), 4),
            "fit": self.fit,
        }


def build_box(points_xyz: np.ndarray, fit: RectangleFit, cfg: ClusterConfig) -> Box:
    """Apply the [w, l, h] convention, the near-square policy, and the clamps.

    The long BEV side becomes the heading axis. That single choice is what makes
    `w <= l` hold by construction rather than by a later swap of two numbers —
    and a later swap of two numbers, without rotating yaw with them, is the
    90-degree rotation §3.2 exists to prevent.
    """
    if fit.extent_u_m >= fit.extent_v_m:
        length_m, width_m, yaw_rad, swapped = fit.extent_u_m, fit.extent_v_m, fit.theta_rad, False
    else:
        length_m, width_m, yaw_rad, swapped = (
            fit.extent_v_m,
            fit.extent_u_m,
            fit.theta_rad + math.pi / 2.0,
            True,
        )
    yaw_rad = wrap_to_pi_rad(yaw_rad)

    cos_t, sin_t = math.cos(fit.theta_rad), math.sin(fit.theta_rad)
    centre_x = fit.centre_u_m * cos_t - fit.centre_v_m * sin_t
    centre_y = fit.centre_u_m * sin_t + fit.centre_v_m * cos_t
    z_min, z_max = float(points_xyz[:, 2].min()), float(points_xyz[:, 2].max())

    _assert_geometry(points_xyz, (centre_x, centre_y), yaw_rad, width_m, length_m, cfg)

    clamped: list[str] = []
    height_m = z_max - z_min
    if width_m < cfg.min_extent_m:
        clamped.append("w")
        width_m = cfg.min_extent_m
    if length_m < cfg.min_extent_m:
        clamped.append("l")
        length_m = cfg.min_extent_m
    if height_m < cfg.min_extent_m:
        clamped.append("h")
        height_m = cfg.min_extent_m
    if width_m > length_m:
        raise ClusterContractError(
            f"w={width_m} > l={length_m} after the min-extent clamp; [w, l, h] order is broken (§3.2)"
        )

    aspect = width_m / length_m
    reasons: list[str] = []
    if aspect >= cfg.near_square_ratio:
        # Systematic, not occasional (§5.7): a square footprint's fitted axis is
        # arbitrary, and Stage 8 will anchor and grow along it.
        reasons.append(f"near_square_footprint:w/l={aspect:.3f}>={cfg.near_square_ratio}")
    if points_xyz.shape[0] < cfg.min_points_for_yaw:
        reasons.append(f"few_points:{points_xyz.shape[0]}<{cfg.min_points_for_yaw}")
    if clamped:
        reasons.append(f"clamped_extent:{','.join(clamped)}")

    return Box(
        translation_m=[centre_x, centre_y, 0.5 * (z_min + z_max)],
        size_wlh_m=[width_m, length_m, height_m],
        yaw_rad=yaw_rad,
        rotation_wxyz=list(quaternion_from_yaw_rad(yaw_rad)),
        yaw_ambiguous=bool(reasons),
        yaw_ambiguous_reasons=reasons,
        axis_swapped=swapped,
        clamped_axes=clamped,
        z_min_m=z_min,
        z_max_m=z_max,
        footprint_diagonal_m=float(math.hypot(width_m, length_m)),
        aspect_ratio_w_over_l=float(aspect),
        fit={
            "criterion": fit.criterion,
            "score": round(fit.score, 6),
            "tie_break_area_m2": round(fit.area_m2, 6),
            "theta_rad": round(fit.theta_rad, 9),
            "n_angles_evaluated": fit.n_angles_evaluated,
            "angle_step_rad": cfg.angle_step_rad,
            "refine_passes": cfg.refine_passes,
            "extent_u_m": round(fit.extent_u_m, 6),
            "extent_v_m": round(fit.extent_v_m, 6),
        },
    )


def _assert_geometry(
    points_xyz: np.ndarray,
    centre_xy: tuple[float, float],
    yaw_rad: float,
    width_m: float,
    length_m: float,
    cfg: ClusterConfig,
) -> None:
    """Re-derive the box from the stored yaw, under `conventions.py`'s definition.

    The fitter's internal axes are not evidence: what is stored is `yaw_rad`, and
    what must be true is that projecting the cluster onto `(cos yaw, sin yaw)`
    gives the stored LENGTH and onto `(-sin yaw, cos yaw)` gives the stored
    WIDTH, about the stored centre. A transposed pair, a sign error, or a
    quaternion built about the wrong axis fails here and nowhere else — every
    such box is otherwise a perfectly well-formed box.
    """
    heading = np.asarray([math.cos(yaw_rad), math.sin(yaw_rad)], dtype=np.float64)
    lateral = np.asarray([-math.sin(yaw_rad), math.cos(yaw_rad)], dtype=np.float64)
    xy = points_xyz[:, :2]
    along = xy @ heading
    across = xy @ lateral
    centre = np.asarray(centre_xy, dtype=np.float64)

    checks = (
        ("length", float(along.max() - along.min()), length_m),
        ("width", float(across.max() - across.min()), width_m),
        ("centre_along", float(0.5 * (along.max() + along.min())), float(centre @ heading)),
        ("centre_across", float(0.5 * (across.max() + across.min())), float(centre @ lateral)),
    )
    for name, measured, stored in checks:
        if abs(measured - stored) > cfg.extent_assert_tol_m:
            raise ClusterContractError(
                f"box geometry assertion failed on {name}: re-projecting the cluster onto the "
                f"stored yaw={yaw_rad!r} gives {measured!r}, the box stores {stored!r}. A [w, l] "
                "transposition or an axis-sign error rotates every box 90 deg while every value "
                "stays plausible (§3.2)"
            )

    round_trip = yaw_rad_from_quaternion(quaternion_from_yaw_rad(yaw_rad))
    if abs(wrap_to_pi_rad(round_trip - yaw_rad)) > cfg.yaw_assert_tol_rad:
        raise ClusterContractError(
            f"yaw -> quaternion -> yaw round trip moved {yaw_rad!r} to {round_trip!r}; I-4 stores a "
            "quaternion and Stage 6 produces a scalar, so this conversion is the unwitnessed point "
            "an axis or sign error hides in (§3.2)"
        )


# ---------------------------------------------------------------------------
# One instance, end to end
# ---------------------------------------------------------------------------


@dataclass
class InstanceResult:
    row: dict
    box: Box | None


def exceeds_prior_footprint(size_wlh_m, prior, multiple: float) -> bool | None:
    """Does this footprint exceed `multiple` x the class prior on either BEV axis?

    Deliberately the SAME test as `stage9_qa/gate.py:spatial_gate` — both axes
    sorted, both compared, one multiplier — so the boxes this stage repairs are
    exactly the boxes that stage rejects. Two thresholds that drift apart would
    leave a band of boxes nothing repairs and nothing catches.

    None when the class has no usable prior dims: not evaluable, so not retried
    (Stage 9 tiers those `flagged` on the same evidence).
    """
    if prior is None or prior.dims is None:
        return None
    short_meas, long_meas = sorted((float(size_wlh_m[0]), float(size_wlh_m[1])))
    short_mu, long_mu = sorted((float(prior.dims["w"]["mu"]), float(prior.dims["l"]["mu"])))
    return short_meas > multiple * short_mu or long_meas > multiple * long_mu


def cluster_instance(
    instance: dict,
    points_xyz: np.ndarray,
    point_index: np.ndarray,
    priors: Priors,
    region: RegionSpec,
    cfg: ClusterConfig,
    depth_m: np.ndarray | None = None,
) -> InstanceResult:
    """DBSCAN -> ghost filter -> L-shape fit, for one Stage 5 mask instance.

    `depth_m` is Stage 5's per-point distance from the camera that painted the
    point. It is what the near-cut retry (C24) reads; without it the retry is
    skipped and the stage behaves exactly as it did before.
    """
    class_name = instance["class_name"]
    eps_m, eps_source = priors.eps_bev(class_name, fallback_m=cfg.eps_fallback_m)
    if eps_source.startswith("config_fallback") and cfg.on_missing_prior == "refuse":
        raise UpstreamRefusal(
            f"class {class_name!r} has no epsilon in {priors.path}: {eps_source}. "
            "on_missing_prior=refuse; run the priors derivation or declare the fallback"
        )

    base = {
        "instance_id": instance["instance_id"],
        "channel": instance["channel"],
        "proposal_index": instance["proposal_index"],
        "class_name": class_name,
        "score": instance["score"],
        "n_mask_px": instance["n_mask_px"],
        "n_points_instance": int(points_xyz.shape[0]),
        "eps_m": round(float(eps_m), 5),
        "eps_source": eps_source,
        "min_samples": cfg.min_samples,
        "cluster_space": cfg.cluster_space,
        "canonical_sort": cfg.canonical_sort,
        "cluster_tie_break": cfg.cluster_tie_break,
        "cloud_kind": cfg.cloud_kind,
        "frame": EGO,
        "num_lidar_pts_basis": "single_sweep_ground_filtered_pre_inflation",
    }

    # Every row carries num_lidar_pts, including the ones with no box: a count of
    # zero is the measurement, and an absent field is a reader's default.
    base["num_lidar_pts"] = 0
    base["n_points_below_gate"] = True

    if points_xyz.shape[0] == 0:
        return InstanceResult({**base, "status": STATUS_NO_POINTS, "box": None, "cluster": None}, None)
    if points_xyz.shape[0] < cfg.min_samples:
        # Not silently promoted to a cluster of its own: DBSCAN with these
        # settings cannot find a core point here, and inventing one would fit a
        # box to two returns. The count survives to Stage 9's gate either way.
        return InstanceResult(
            {
                **base,
                "status": STATUS_BELOW_MIN_SAMPLES,
                "box": None,
                "cluster": {"n_points_noise": int(points_xyz.shape[0]), "n_clusters": 0},
            },
            None,
        )

    order = canonical_order(points_xyz, point_index)
    ordered_xyz = points_xyz[order]
    labels = dbscan_bev(ordered_xyz[:, :2], eps_m, cfg.min_samples)
    choice = select_cluster(labels, ordered_xyz)
    if choice is None:
        return InstanceResult(
            {
                **base,
                "status": STATUS_ALL_NOISE,
                "box": None,
                "cluster": {"n_points_noise": int(points_xyz.shape[0]), "n_clusters": 0},
            },
            None,
        )

    cluster_xyz = ordered_xyz[choice.member]
    fit = fit_rectangle(cluster_xyz[:, :2], cfg)
    box = build_box(cluster_xyz, fit, cfg)

    # --- the near-cut retry (C24) ------------------------------------------
    # Only for boxes that fail the prior-footprint test — the same test Stage 9
    # rejects on. Everything else takes the identical code path it always did
    # and is byte-identical to a pre-C24 run.
    near_cut = None
    if cfg.near_cut_enabled and depth_m is not None and depth_m.shape[0] == points_xyz.shape[0]:
        prior = priors.get(class_name)
        triggered = exceeds_prior_footprint(box.size_wlh_m, prior, cfg.near_cut_prior_multiple)
        near_cut = {
            "applied": False,
            "triggered": bool(triggered) if triggered is not None else None,
            "depth_m": cfg.near_cut_depth_m,
            "prior_multiple": cfg.near_cut_prior_multiple,
            "size_wlh_m_before": [round(float(v), 4) for v in box.size_wlh_m],
            "n_points_before": int(cluster_xyz.shape[0]),
        }
        if triggered:
            ordered_depth = depth_m[order]
            # The object is at the NEAR end of its own painted depth range: the
            # mask was generated by the surface facing the camera. Measured from
            # the instance's own nearest return, so no absolute depth or class
            # prior enters — see near_cut_depth_m's provenance.
            keep = ordered_depth <= float(ordered_depth.min()) + cfg.near_cut_depth_m
            near_cut["d_near_m"] = round(float(ordered_depth.min()), 4)
            near_cut["n_points_kept_by_cut"] = int(np.count_nonzero(keep))
            if int(np.count_nonzero(keep)) >= cfg.min_samples:
                cut_xyz = ordered_xyz[keep]
                cut_labels = dbscan_bev(cut_xyz[:, :2], eps_m, cfg.min_samples)
                cut_choice = select_cluster(cut_labels, cut_xyz)
                if cut_choice is not None:
                    retry_xyz = cut_xyz[cut_choice.member]
                    retry_fit = fit_rectangle(retry_xyz[:, :2], cfg)
                    retry_box = build_box(retry_xyz, retry_fit, cfg)
                    # Never accept a retry that grew the footprint: the cut is
                    # there to remove a far tail, and a larger box means it
                    # selected a different blob instead.
                    if (retry_box.size_wlh_m[0] * retry_box.size_wlh_m[1]
                            <= box.size_wlh_m[0] * box.size_wlh_m[1] + cfg.extent_assert_tol_m):
                        cluster_xyz, fit, box, choice = retry_xyz, retry_fit, retry_box, cut_choice
                        near_cut["applied"] = True
                    else:
                        near_cut["rejected_reason"] = "footprint_would_grow"
                else:
                    near_cut["rejected_reason"] = "all_noise_after_cut"
            else:
                near_cut["rejected_reason"] = "below_min_samples_after_cut"
            near_cut["size_wlh_m_after"] = [round(float(v), 4) for v in box.size_wlh_m]
            near_cut["n_points_after"] = int(cluster_xyz.shape[0])

    row = {
        **base,
        "status": STATUS_FIT,
        "num_lidar_pts": int(cluster_xyz.shape[0]),
        "n_points_below_gate": bool(cluster_xyz.shape[0] < cfg.min_points_per_instance),
        "cluster": choice.ledger,
        "near_cut": near_cut,
        "box": box.as_dict(),
    }
    if cfg.record_eval_region:
        row["in_region"] = bool(
            in_region(box.translation_m[0], box.translation_m[1], region, frame=EGO)
        )
        row["coverage_config"] = region.coverage_config
        row["range_m"] = round(float(math.hypot(box.translation_m[0], box.translation_m[1])), 4)
    return InstanceResult(row, box)


# ---------------------------------------------------------------------------
# One keyframe
# ---------------------------------------------------------------------------


def load_keyframe_points(
    lift_row: dict, stage5_dir: str, cfg: ClusterConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """(cloud, painted point_index, painted instance_id, per-point depth).

    Stage 5 stored indices, not coordinates (§5.6): the cloud is read here and
    the rows named there are taken from it, so there is exactly one copy of every
    point's position in the run and no rounded duplicate to disagree with it.

    `depth_m` is the distance from the camera that won the point, written by
    Stage 5 for exactly this purpose (lift.py:64-69 records that the far-surface
    points it does not filter are left to this stage, "the per-point depth is
    recorded so that filter has what it needs"). None on an older Stage 5 output
    that predates the field — the near-cut retry then simply does not run.
    """
    points_path = os.path.join(stage5_dir, lift_row["points_path"])
    if not os.path.isfile(points_path):
        raise UpstreamRefusal(f"{points_path} not found; run `python3 -m pipeline.stage5_lift.lift` first")
    with np.load(points_path) as npz:
        frame = str(npz["__frame__"][0])
        cloud_kind = str(npz["__cloud_kind__"][0])
        point_index = npz["point_index"].astype(np.int64)
        instance_id = npz["instance_id"].astype(np.int64)
        depth_m = npz["depth_m"].astype(np.float64) if "depth_m" in npz.files else None
    if frame != EGO:
        raise ClusterContractError(
            f"{points_path}: painted points claim frame={frame!r}, not {EGO!r}; every geometric "
            "record crossing a stage boundary is ego-frame (§1.1 rule 1)"
        )
    if cloud_kind != cfg.cloud_kind:
        raise ClusterContractError(
            f"{points_path}: cloud_kind={cloud_kind!r}, Stage 6 clusters {cfg.cloud_kind!r} (§1.4)"
        )

    cloud = read_pcd_bin(lift_row["cloud_path"])[:, :3].astype(np.float64)
    if cloud.shape[0] != int(lift_row["n_points_cloud"]):
        raise ClusterContractError(
            f"{lift_row['cloud_path']}: holds {cloud.shape[0]} points, Stage 5 recorded "
            f"{lift_row['n_points_cloud']}. The point indices in {points_path} name rows of a "
            "cloud that has since changed"
        )
    if point_index.size and int(point_index.max()) >= cloud.shape[0]:
        raise ClusterContractError(
            f"{points_path}: point index {int(point_index.max())} is out of range for a "
            f"{cloud.shape[0]}-point cloud"
        )
    return cloud, point_index, instance_id, depth_m


def cluster_keyframe(
    lift_row: dict,
    stage5_dir: str,
    priors: Priors,
    cfg: ClusterConfig,
) -> tuple[list[dict], dict]:
    """Every instance of one keyframe."""
    cloud, point_index, instance_id, depth_m = load_keyframe_points(lift_row, stage5_dir, cfg)
    region = region_for(lift_row["coverage_config"])

    rows: list[dict] = []
    totals = {
        "n_instances": 0,
        "n_boxes": 0,
        "n_no_points": 0,
        "n_below_min_samples": 0,
        "n_all_noise": 0,
        "n_yaw_ambiguous": 0,
        "n_points_painted": int(point_index.shape[0]),
        "n_points_in_boxes": 0,
        "n_points_dropped_as_ghost": 0,
        "n_points_noise": 0,
        "n_boxes_in_region": 0,
        "n_boxes_below_gate": 0,
        "n_eps_fallback": 0,
    }

    for instance in lift_row["instances"]:
        selected = instance_id == instance["instance_id"]
        rows_of_cloud = point_index[selected]
        result = cluster_instance(
            instance, cloud[rows_of_cloud], rows_of_cloud, priors, region, cfg,
            depth_m=None if depth_m is None else depth_m[selected],
        )
        row = {
            "spec": STAGE_SPEC,
            "keyframe_token": lift_row["keyframe_token"],
            "scene_token": lift_row["scene_token"],
            "t_ns": lift_row["t_ns"],
            "time_base": lift_row["time_base"],
            "coverage_config": lift_row["coverage_config"],
            "cloud_path": lift_row["cloud_path"],
            "points_path": lift_row["points_path"],
            **result.row,
        }
        rows.append(row)

        totals["n_instances"] += 1
        if row["eps_source"].startswith("config_fallback"):
            totals["n_eps_fallback"] += 1
        if row["status"] == STATUS_FIT:
            ledger = row["cluster"]
            totals["n_boxes"] += 1
            totals["n_points_in_boxes"] += ledger["n_points_kept"]
            totals["n_points_dropped_as_ghost"] += ledger["n_points_other_clusters"]
            totals["n_points_noise"] += ledger["n_points_noise"]
            totals["n_yaw_ambiguous"] += int(row["box"]["yaw_ambiguous"])
            totals["n_boxes_in_region"] += int(row.get("in_region", False))
            totals["n_boxes_below_gate"] += int(row["n_points_below_gate"])
        elif row["status"] == STATUS_NO_POINTS:
            totals["n_no_points"] += 1
        elif row["status"] == STATUS_BELOW_MIN_SAMPLES:
            totals["n_below_min_samples"] += 1
            totals["n_points_noise"] += row["cluster"]["n_points_noise"]
        elif row["status"] == STATUS_ALL_NOISE:
            totals["n_all_noise"] += 1
            totals["n_points_noise"] += row["cluster"]["n_points_noise"]
    return rows, totals


# ---------------------------------------------------------------------------
# Upstream
# ---------------------------------------------------------------------------


def load_upstream(paths: Paths, stage5_dir: str, priors_path: str, *, accept_degraded: bool = False):
    """Refuse to start unless Stage 5 COMPLETED on THIS substrate and the priors match it.

    Stage 5 goes through the C16 gate: absent marker refuses unconditionally,
    degraded marker refuses unless `accept_degraded` — the explicit, recorded
    opt-in, never a default.
    """
    current = metadata_fingerprint(paths)
    stage5, marker5 = require_upstream(
        stage5_dir,
        stage_name="Stage 5",
        module_hint="pipeline.stage5_lift.lift",
        current_fingerprint=current,
        accept_degraded=accept_degraded,
    )
    if stage5.get("cloud_kind") != "single_sweep":
        raise UpstreamRefusal(
            f"Stage 5 lifted cloud_kind={stage5.get('cloud_kind')!r}; §1.4 clusters the single sweep"
        )
    if stage5.get("frame") != EGO:
        raise UpstreamRefusal(f"Stage 5 output claims frame={stage5.get('frame')!r}, not {EGO!r}")

    priors = load_priors(priors_path)
    if not priors.metadata_fingerprint:
        # An empty fingerprint is not a match, it is an unbound file: the pilot
        # derivation always records one, so its absence means this file cannot
        # be audited against any substrate — including this one.
        raise UpstreamRefusal(
            f"{priors.path} records no derived_from.metadata_fingerprint; a priors file that does "
            "not bind itself to a substrate cannot be checked against this one"
        )
    if priors.metadata_fingerprint != current:
        # The priors carry class means and epsilons derived from GT on a specific
        # dataroot. Clustering this substrate with another one's epsilons is a
        # join across two datasets that produces a complete, valid-looking output.
        raise UpstreamRefusal(
            f"priors fingerprint mismatch: {priors.path} was derived against "
            f"{priors.metadata_fingerprint}, this dataroot is {current}"
        )
    subset = priors.derived_from.get("scene_subset")
    if subset != PRIORS_SCENE_SUBSET:
        # Same substrate is necessary, not sufficient: a priors file derived on
        # the run/eval scenes carries a matching fingerprint and tunes epsilon
        # on the scored set (P1-5, §11 decision 3).
        raise UpstreamRefusal(
            f"priors scene_subset={subset!r}: {priors.path} was not derived on the "
            f"{PRIORS_SCENE_SUBSET!r} partition, so its epsilons were tuned on scenes this "
            "pipeline scores (P1-5, §11 decision 3)"
        )
    return stage5, marker5, priors


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def read_lift_index(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise UpstreamRefusal(f"{path} not found; run `python3 -m pipeline.stage5_lift.lift` first")
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(
    paths: Paths,
    stage5_manifest: dict,
    stage5_marker,
    priors: Priors,
    cfg: ClusterConfig,
    stage5_dir: str,
    out_dir: str,
    scene_names: Sequence[str] | None,
) -> tuple[dict, int]:
    started = time.time()
    errors = cfg.validate()
    if errors:
        raise UpstreamRefusal("; ".join(errors))
    # Any marker still standing describes the PREVIOUS run of this stage; it
    # comes down before the first write (C16).
    clear_markers(out_dir)

    root = os.path.join(stage5_dir, "scenes")
    if not os.path.isdir(root):
        raise UpstreamRefusal(f"{root} not found; Stage 5 wrote no scenes")
    names = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
    if scene_names:
        missing = sorted(set(scene_names) - set(names))
        if missing:
            raise UpstreamRefusal(f"requested scene(s) not present in Stage 5 output: {missing}")
        names = [n for n in names if n in scene_names]

    per_scene: list[dict] = []
    degraded = False
    totals = {
        "n_keyframes": 0,
        "n_instances": 0,
        "n_boxes": 0,
        "n_no_points": 0,
        "n_below_min_samples": 0,
        "n_all_noise": 0,
        "n_yaw_ambiguous": 0,
        "n_points_painted": 0,
        "n_points_in_boxes": 0,
        "n_points_dropped_as_ghost": 0,
        "n_points_noise": 0,
        "n_boxes_in_region": 0,
        "n_boxes_below_gate": 0,
        "n_eps_fallback": 0,
    }
    fallback_classes: dict[str, int] = {}

    for scene_name in names:
        lift_rows = read_lift_index(os.path.join(root, scene_name, "lift.jsonl"))
        scene_rows: list[dict] = []
        scene_totals = {key: 0 for key in totals}

        for lift_row in lift_rows:
            rows, keyframe_totals = cluster_keyframe(lift_row, stage5_dir, priors, cfg)
            scene_rows.extend(rows)
            scene_totals["n_keyframes"] += 1
            for key, value in keyframe_totals.items():
                scene_totals[key] += value
            for row in rows:
                if row["eps_source"].startswith("config_fallback"):
                    fallback_classes[row["class_name"]] = fallback_classes.get(row["class_name"], 0) + 1

        write_jsonl_atomic(os.path.join(out_dir, "scenes", scene_name, "boxes.jsonl"), scene_rows)
        summary = {
            "scene": scene_name,
            **scene_totals,
            "boxes_per_keyframe": round(scene_totals["n_boxes"] / max(1, scene_totals["n_keyframes"]), 2),
            "ghost_fraction": round(
                scene_totals["n_points_dropped_as_ghost"] / max(1, scene_totals["n_points_painted"]), 5
            ),
            "yaw_ambiguous_fraction": round(
                scene_totals["n_yaw_ambiguous"] / max(1, scene_totals["n_boxes"]), 5
            ),
            # An instance that produced no box is reportable; a scene in which
            # nothing produced a box means the fit did not happen.
            "degraded": scene_totals["n_instances"] > 0 and scene_totals["n_boxes"] == 0,
        }
        per_scene.append(summary)
        degraded = degraded or summary["degraded"]
        for key in totals:
            totals[key] += scene_totals[key]
        print(
            f"  {scene_name}  {scene_totals['n_keyframes']:>3} kf  "
            f"{scene_totals['n_instances']:>5} inst  {scene_totals['n_boxes']:>5} boxes  "
            f"{summary['boxes_per_keyframe']:>6.2f}/kf  ghost {summary['ghost_fraction']:.3f}  "
            f"{scene_totals['n_yaw_ambiguous']:>4} yaw-ambiguous"
            + ("  DEGRADED" if summary["degraded"] else "")
        )

    manifest = {
        "spec": STAGE_SPEC,
        "stage": STAGE,
        "seed": cfg.global_seed,
        "config": cfg.as_dict(),
        "upstream": {
            "metadata_fingerprint": stage5_manifest["upstream"]["metadata_fingerprint"],
            "fingerprint_spec": stage5_manifest["upstream"]["fingerprint_spec"],
            "stage5_spec": stage5_manifest["spec"],
            # C16: a run built on accepted degradation says so in its provenance.
            "stage5_degraded": stage5_marker.degraded,
            "stage5_degraded_causes": list(stage5_marker.causes),
            "accepted_degraded_upstream": cfg.accept_degraded_upstream,
            "priors": {
                **priors.as_reference(),
                # P1-5 audit: the derivation scope, which as_reference() omits.
                "scene_subset": priors.derived_from.get("scene_subset"),
                "scenes": list(priors.derived_from.get("scenes", [])),
            },
        },
        "paths": paths.as_dict(),
        "frame": EGO,
        "cloud_kind": cfg.cloud_kind,
        "clustering": {
            "scope": "per_mask_instance",
            "scope_source": "§1.6 reading (a)",
            "rejected_reading": (
                "pooling a class across the frame yields ONE box per class per frame; adjacent cars "
                "merge under any epsilon large enough to hold one car and every other instance is "
                "silently discarded"
            ),
            "algorithm": "dbscan",
            "implementation": "pipeline.stage6_cluster.cluster.dbscan_bev (in-tree, ordering-guaranteed)",
            "eps_source": f"{priors.name} eps_bev (comprehensive.md §7.2 formula)",
            "eps_fallback_m": cfg.eps_fallback_m,
            "eps_fallback_classes": dict(sorted(fallback_classes.items())),
            "largest_cluster_rule": "the reprojection-ghost filter of §1.6 / §7.7, counted per instance",
        },
        "box_fit": {
            "method": "l_shape",
            "criterion": cfg.fit_criterion,
            "size_order": "w,l,h",
            "size_order_source": "§3.2 / A.1 — nuScenes order, asserted per box",
            "near_square_policy": (
                f"the long BEV side is the heading axis (so w <= l holds by construction); "
                f"w/l >= {cfg.near_square_ratio} sets yaw_ambiguous (§5.7)"
            ),
            "yaw_convention": "conventions.py: about +z, from +x, ISO 8855; asserted per box",
            "yaw_axis_only": True,
        },
        "known_gaps": [
            "the 180 deg heading direction is not decided here: a symmetric point set cannot say "
            "which end is the front. Stage 7's yaw-consistency enforcement along tracks is the "
            "producer of that bit (§4, §5.8)",
            "box height is the cluster's z-extent and is biased LOW: Stage 1's ground removal "
            "strips a 0.3 m band, so the bottom of every object is missing before this stage sees it",
            "no amodal reasoning: the box encloses the returns it was given. Stage 8 owns inflation "
            "(§5.9), and Stage 9 gates on the PRE-inflation dimensions recorded here (P1-12)",
        ],
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "elapsed_s": round(time.time() - started, 2),
        "scenes": per_scene,
        "totals": totals,
    }
    return manifest, EXIT_DEGRADED if degraded else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--stage5-dir", default=None, help="default <work_root>/stage5_lift")
    parser.add_argument("--priors", default=None, help=f"default <out_root>/priors/{PRIORS_NAME}.json")
    parser.add_argument("--out-dir", default=None, help="default <work_root>/stage6_cluster")
    parser.add_argument("--scenes", nargs="*", default=None, help="subset of Stage 5 scene names")
    parser.add_argument("--fit-criterion", default="closeness", choices=FIT_CRITERIA)
    parser.add_argument("--on-missing-prior", default="fallback", choices=MISSING_PRIOR_POLICIES)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="override the global seed (recorded)")
    parser.add_argument(
        "--accept-degraded-upstream",
        action="store_true",
        help="consume a DEGRADED (complete, quality-flagged) Stage 5 output; recorded (C16)",
    )
    args = parser.parse_args(argv)

    try:
        paths = load_paths(args.paths)
    except PathValidationError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    stage5_dir = args.stage5_dir or os.path.join(paths.work_root, "stage5_lift")
    priors_path = args.priors or os.path.join(paths.out_root, "priors", f"{PRIORS_NAME}.json")
    out_dir = args.out_dir or os.path.join(paths.work_root, STAGE)
    assert_dataroot_read_only(paths, out_dir)

    cfg = ClusterConfig(
        fit_criterion=args.fit_criterion,
        on_missing_prior=args.on_missing_prior,
        accept_degraded_upstream=args.accept_degraded_upstream,
        **({"min_samples": args.min_samples} if args.min_samples is not None else {}),
        **({"global_seed": args.seed} if args.seed is not None else {}),
    )

    try:
        stage5_manifest, stage5_marker, priors = load_upstream(
            paths, stage5_dir, priors_path, accept_degraded=cfg.accept_degraded_upstream
        )
        manifest, code = run(
            paths, stage5_manifest, stage5_marker, priors, cfg, stage5_dir, out_dir, args.scenes
        )
    except UpstreamRefusal as exc:
        print(f"REFUSING TO START: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (ClusterContractError, PathValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)
    # Three-state marker (§1.9, C16): clean / degraded-with-causes / absent.
    write_marker(
        out_dir,
        manifest["upstream"]["metadata_fingerprint"],
        degraded=code == EXIT_DEGRADED,
        causes=[
            f"{s['scene']}: {s['n_instances']} instance(s), 0 boxes"
            for s in manifest["scenes"]
            if s["degraded"]
        ],
    )

    t = manifest["totals"]
    print(f"keyframes            : {t['n_keyframes']}")
    print(f"instances            : {t['n_instances']}")
    print(
        f"boxes                : {t['n_boxes']}  "
        f"({t['n_no_points']} no points, {t['n_below_min_samples']} below min_samples, "
        f"{t['n_all_noise']} all noise)"
    )
    print(
        f"ghost filter         : {t['n_points_dropped_as_ghost']} points dropped from "
        f"{t['n_points_painted']} painted, {t['n_points_noise']} left as DBSCAN noise"
    )
    print(
        f"yaw ambiguous        : {t['n_yaw_ambiguous']} / {t['n_boxes']}  "
        "(near-square footprint, few points, or a clamped extent)"
    )
    print(f"boxes inside E       : {t['n_boxes_in_region']}  ({t['n_boxes_below_gate']} below the 5-return gate)")
    if manifest["clustering"]["eps_fallback_classes"]:
        print(f"eps fallback         : {manifest['clustering']['eps_fallback_classes']}")
    print(f"wrote {out_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
