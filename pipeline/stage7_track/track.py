#!/usr/bin/env python3
"""Stage 7 -- predict-then-match tracking, ICP velocity, Kalman fallback (S5.8, Phase 9).

**The 2 Hz problem.** nuScenes keyframes are 2 Hz; a vehicle at 40 km/h moves
~5.5 m between them, so 3D IoU between two un-propagated consecutive detections
is zero for most vehicles. **Resolution (S11 decision 5): predict-then-match.**
Every active track's Kalman state is propagated forward by its own estimated
velocity to the current keyframe's timestamp BEFORE any IoU is computed; IoU is
measured between that *predicted* box and each detection, never between two raw
detections. `iou_gate` is a config value carrying `derived for 2 Hz` provenance,
not a spec-copied constant tuned for a 10 Hz canonical rate.

**Matching.** Gate then score, per class: a (track, detection) pair is eligible
only if `class_name` matches and the predicted-vs-detection 3D IoU clears
`iou_gate`. Eligible pairs score `iou * cosine` (comprehensive.md's product
term) when a trusted appearance embedding exists on both sides, `iou` alone
otherwise -- declared per pair via `appearance_trusted`, never silently
substituted. `scipy.optimize.linear_sum_assignment` runs the Hungarian
algorithm over the cost matrix (ineligible pairs get a sentinel cost so the
solver never picks them); DBSCAN in Stage 6 is hand-rolled because scikit-learn
is not a dependency, but the Hungarian algorithm has no comparable dependency
cost here -- scipy is already on this substrate -- so it is not reimplemented.

**ICP, and which frame it registers in -- stated on every row.** LiDAR clusters
at two keyframes are each in *that keyframe's own* ego frame (S1.1), and ego
moves between them. Registering the two clusters directly measures the
object's motion *relative to the ego vehicle*, which is not what nuScenes'
absolute mAVE reports (pilot_plan.md S5.8). `icp_registration_frame` decides
which quantity is measured and is not a label bolted on after the fact:

  - `global_absolute` (default): both clusters are hopped into
    `nuscenes_global` via each keyframe's own `ego_pose` -- the same two-pose
    hop `conventions.project_lidar_to_image` and Stage 1's `accumulate()` use
    -- before ICP runs. The registered translation is the object's true
    world-frame displacement; velocity is absolute and mAVE-comparable. The
    Kalman filter's state also lives in this frame, so a track's velocity
    means the same thing whether it came from ICP or from the filter.
  - `ego_relative` (kept, declared inferior): no ego-motion correction. Two
    consecutive ego-frame point sets are registered as if they lived in one
    fixed frame. This is the literal version of the pitfall pilot_plan.md S5.8
    warns about, kept so the failure is measurable rather than argued about.

Every track record carries `icp_frame` (the config value) and
`velocity_semantics` (`"absolute_world_frame"` or `"relative_to_ego"`) so a
consumer never has to guess which quantity `velocity_mps` is.

**Kalman fallback, specified** (comprehensive.md S7.3.6, pilot_plan.md S5.8).
State `[x, y, z, vx, vy, vz, yaw]`, constant-velocity / constant-yaw process
model (no yaw-rate term -- a stated limitation, not an omission), Delta t from
the recorded `t_ns` (`time_base` explicit upstream). **Trigger:** a matched
detection with `num_lidar_pts < 15` skips ICP entirely; the reported velocity
is the filter's own predict/update estimate rather than a direct point-cluster
measurement. ICP additionally requires the previous match to have been in the
immediately preceding processed keyframe (`icp_require_adjacent_keyframe`) --
registering across a multi-frame gap folds a larger, more nonlinear motion into
one velocity estimate and is refused rather than silently attempted.

**Yaw-consistency enforcement along tracks** (S7.3.7, restored per S4/S11
decision 6 -- free geometry, the designed defence against the symmetric
3-wheeler/near-square-footprint 180-degree flip). Stage 6 stores yaw as an
AXIS, not a direction (`yaw_axis_only: true` there) -- a near-square or
few-point footprint cannot say which end is the front. Before a detection's
yaw is fed into the filter, `disambiguate_yaw()` picks whichever of
`{yaw, yaw + pi}` is closer to a reference: the track's own predicted velocity
heading when the track is moving faster than `yaw_consistency_min_speed_mps`,
else the track's previous smoothed yaw. A track's first-ever detection has
neither and is passed through unchanged, declared via `yaw_reference_source`.

**What this stage does not do.** No forward-backward smoothing over the whole
scene (S4: waived, offboard refinement, not needed to prove association
plumbing). No mask-pooled appearance embeddings (S7.1's stretch goal) -- crops
use the mask's tight box only. If the `reid_embedding` checkpoint is
unavailable and `--require-appearance` is not passed, the run proceeds on IoU
alone with `appearance_enabled: false` recorded prominently rather than
refusing the whole stage over one optional term.

    python3 -m pipeline.stage7_track.track [--paths configs/paths.yaml]

Exit codes:
    0  every scene produced at least one track that survived >= 3 hits
    1  ran, but at least one scene with detections confirmed zero tracks
    2  upstream contract broken, or --require-appearance and the model is
       unavailable; nothing was written
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.common.conventions import (  # noqa: E402
    EGO,
    NUSCENES_GLOBAL,
    Transform,
    apply_transform,
    quaternion_from_yaw_rad,
    wrap_to_pi_rad,
)
from pipeline.common.model_interfaces import (  # noqa: E402
    REID_EMBEDDING,
    CheckpointSpec,
    EmbeddingBatch,
    RoleContractError,
    register,
)
from pipeline.common.paths import (  # noqa: E402
    PathValidationError,
    Paths,
    assert_dataroot_read_only,
    load_paths,
    metadata_fingerprint,
)
from pipeline.common.schemas import KeyframeRecord, read_records  # noqa: E402
from pipeline.common.manifest import (  # noqa: E402
    UpstreamRefusal,
    clear_markers,
    require_upstream,
    write_json_atomic,
    write_jsonl_atomic,
    write_marker,
)
from pipeline.stage0_data_probe.probe import Substrate  # noqa: E402
from pipeline.stage1_ingestion.ingest import read_pcd_bin  # noqa: E402
from pipeline.stage3_proposals.proposals import ModelUnavailable  # noqa: E402
from pipeline.stage5_lift.lift import read_mask_index  # noqa: E402
from pipeline.stage6_cluster.cluster import (  # noqa: E402
    STATUS_FIT,
    canonical_order,
    dbscan_bev,
    select_cluster,
)

STAGE = "stage7_track"
STAGE_SPEC = "dhakascenes-pilot/stage7_track/v1"

EXIT_OK = 0
EXIT_DEGRADED = 1
EXIT_REFUSED = 2

ICP_FRAMES: tuple[str, ...] = ("global_absolute", "ego_relative")
MATCHERS: tuple[str, ...] = ("hungarian",)
IOU_MODES: tuple[str, ...] = ("bev", "3d")
KF_STATE_NAMES: tuple[str, ...] = ("x", "y", "z", "vx", "vy", "vz", "yaw")

TRACK_TENTATIVE = "tentative"
TRACK_CONFIRMED = "confirmed"
TRACK_DEAD = "dead"

REASON_MODEL_UNAVAILABLE = "model_unavailable"
REASON_CROP_TOO_SMALL = "crop_below_min_crop_px"
REASON_NOT_ADJACENT = "previous_match_not_adjacent_keyframe"
REASON_SPARSE_CURRENT = "current_cluster_below_kalman_fallback_min_points"
REASON_SPARSE_PREVIOUS = "previous_cluster_below_kalman_fallback_min_points"
REASON_NO_CLUSTER = "cluster_not_reconstructable"


class TrackContractError(RuntimeError):
    """A Stage 7 input, or a track record this stage produced, violated the contract."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackConfig:
    """Stage 7 tunables. None of these may appear as a literal in the code below."""

    cloud_kind: str = "single_sweep"

    # --- association (S11 decision 5, M-11) ---
    iou_mode: str = "bev"
    iou_gate: float = 0.05
    class_gate: bool = True
    matcher: str = "hungarian"
    ineligible_cost: float = 1.0e6

    # --- appearance (reid_embedding role) ---
    min_crop_px: int = 24
    require_appearance: bool = False
    # Default since 2026-08-14 (human-directed, after the 4-cell comparison in
    # Results/): DINOv3. It measured IDENTICAL to facebook/dinov2-small on every
    # accuracy metric -- 3D 75.1/71.7/72.2 and 2D 62.5/59.7/36.6 either way,
    # all six per-class numbers unchanged -- because appearance only re-ranks
    # candidates that already passed the IoU gate. Recorded so the next reader
    # does not mistake this default for a measured improvement.
    #   NOTE: a GATED hub repo. Stage 7 needs an HF_TOKEN whose account has been
    #   granted access, or the adapter takes its declared ModelUnavailable path
    #   (IoU-only tracking) rather than failing loudly.
    reid_model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    reid_revision: str | None = None
    device: str = "cuda"

    # --- birth / death ---
    min_hits_to_confirm: int = 1
    max_misses_before_death: int = 2
    stable_track_hits: int = 3

    # --- Kalman fallback (comprehensive.md S7.3.6) ---
    kalman_fallback_min_points: int = 15
    kf_process_noise_pos_m: float = 0.5
    kf_process_noise_vel_mps: float = 2.0
    kf_process_noise_yaw_rad: float = math.radians(5.0)
    kf_measurement_noise_pos_m: float = 0.3
    kf_measurement_noise_yaw_rad: float = math.radians(10.0)
    kf_init_pos_std_m: float = 1.0
    kf_init_vel_std_mps: float = 5.0
    kf_init_yaw_std_rad: float = math.radians(30.0)

    # --- ICP (comprehensive.md S7.3.7) ---
    icp_registration_frame: str = "global_absolute"
    icp_max_iterations: int = 25
    icp_convergence_tol_m: float = 1.0e-4
    icp_min_points_each_side: int = 15
    icp_require_adjacent_keyframe: bool = True
    icp_measurement_noise_vel_mps: float = 1.0

    # --- yaw-consistency (S7.3.7, restored) ---
    yaw_consistency_min_speed_mps: float = 1.0

    # --- upstream gate (C16) ---
    accept_degraded_upstream: bool = False

    # --- determinism (S1.9) ---
    global_seed: int = 20260812

    provenance: dict = field(
        default_factory=lambda: {
            "iou_mode": (
                "bev (deviates from comprehensive.md's literal '3D IoU'): height is already "
                "documented in this pipeline as the noisiest, most systematically biased axis of "
                "every box (Stage 6 -- ground removal strips a variable band before the box is "
                "even fit; here -- vz is by far the least observable component of the Kalman "
                "state, especially early in a track's life). Gating cross-frame IDENTITY on that "
                "axis makes association fail exactly where predict-then-match is supposed to "
                "rescue it: XY can be well predicted while a noisy vz alone drops the 3D overlap "
                "to zero. Both ious_bev and ious_3d are computed and recorded on every pair "
                "regardless of which one gates, so this is a measured choice, not a silent one"
            ),
            "iou_gate": (
                "derived for 2 Hz, arbitrary pilot value needing tuning -- not a spec-copied "
                "constant. predict-then-match exists precisely to make this gate reachable at "
                "all after a 5+ m inter-keyframe displacement (S11 decision 5)"
            ),
            "class_gate": "a car is never matched to a pedestrian regardless of geometry",
            "matcher": "Hungarian via scipy.optimize.linear_sum_assignment (M-11); scipy is "
            "already on this substrate, unlike scikit-learn (cf. Stage 6's hand-rolled DBSCAN)",
            "min_crop_px": "below this side length, appearance similarity is not trusted "
            "(pilot_plan.md S5.8); the pair falls back to IoU-only scoring, declared per pair",
            "require_appearance": "false: tracking degrades to IoU-only rather than refusing "
            "the whole stage when the reid_embedding checkpoint is unavailable",
            "min_hits_to_confirm": "birth == confirmed by default; raise to gate false tracks",
            "max_misses_before_death": "arbitrary, needs tuning -- how many keyframes a track "
            "coasts on prediction alone before being declared dead",
            "stable_track_hits": "the Phase 9 exit gate's >= 3-frame stable-track threshold, "
            "reported per scene, not a birth/death rule itself",
            "kalman_fallback_min_points": "comprehensive.md S7.3.6's sparse-cluster guard, "
            "verbatim: < 15 LiDAR returns on the CURRENT detection skips ICP",
            "icp_registration_frame": (
                "global_absolute (default): both clusters are hopped through nuscenes_global "
                "via each keyframe's own ego_pose before registration, so the measured velocity "
                "is the object's true world-frame motion and is nuScenes-mAVE-comparable. "
                "ego_relative: no ego-motion correction at all -- the literal version of the "
                "pitfall pilot_plan.md S5.8 warns about, kept so it is measurable, not argued"
            ),
            "icp_require_adjacent_keyframe": (
                "ICP only runs between the immediately preceding processed keyframe and the "
                "current one; registering across a coasted gap folds a larger, more nonlinear "
                "motion into one velocity estimate"
            ),
            "icp_measurement_noise_vel_mps": (
                "ICP's velocity estimate is fed into the filter as a MEASUREMENT (Kalman gain), "
                "never a direct overwrite of the state -- a single noisy registration on a sparse "
                "or texture-poor cluster must not hijack the whole track's future prediction"
            ),
            "yaw_consistency_min_speed_mps": (
                "below this speed a velocity heading is noise, not a direction; the previous "
                "track yaw is used as the disambiguation reference instead (S7.3.7)"
            ),
            "accept_degraded_upstream": "C16 -- consuming a DEGRADED (complete, quality-flagged) "
            "Stage 1, 4, 5, or 6 output is an explicit recorded decision, never a default",
            "global_seed": "S1.9, one global seed, recorded (nothing here samples; recorded anyway)",
        }
    )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.cloud_kind != "single_sweep":
            errors.append(f"cloud_kind={self.cloud_kind!r}: S1.4 clusters the single-sweep cloud")
        if self.iou_mode not in IOU_MODES:
            errors.append(f"iou_mode={self.iou_mode!r} is not one of {IOU_MODES}")
        if not 0.0 <= self.iou_gate <= 1.0:
            errors.append(f"iou_gate={self.iou_gate} must be in [0, 1]")
        if self.matcher not in MATCHERS:
            errors.append(f"matcher={self.matcher!r} is not one of {MATCHERS}")
        if self.icp_registration_frame not in ICP_FRAMES:
            errors.append(f"icp_registration_frame={self.icp_registration_frame!r} is not one of {ICP_FRAMES}")
        if self.min_hits_to_confirm < 1:
            errors.append(f"min_hits_to_confirm={self.min_hits_to_confirm} must be >= 1")
        if self.max_misses_before_death < 0:
            errors.append(f"max_misses_before_death={self.max_misses_before_death} must be >= 0")
        if self.kalman_fallback_min_points < 0:
            errors.append(f"kalman_fallback_min_points={self.kalman_fallback_min_points} must be >= 0")
        if self.icp_max_iterations < 1:
            errors.append(f"icp_max_iterations={self.icp_max_iterations} must be >= 1")
        if self.icp_measurement_noise_vel_mps <= 0.0:
            errors.append(f"icp_measurement_noise_vel_mps={self.icp_measurement_noise_vel_mps} must be positive")
        return errors

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# BEV rotated-rectangle IoU (S1.6-style oriented boxes; yaw is about +z only)
# ---------------------------------------------------------------------------


def _rect_corners(translation_m: Sequence[float], yaw_rad: float, length_m: float, width_m: float) -> np.ndarray:
    """(4, 2) BEV corners, CCW, box-local +x = heading (length), +y = lateral (width)."""
    cos_t, sin_t = math.cos(yaw_rad), math.sin(yaw_rad)
    dx, dy = length_m / 2.0, width_m / 2.0
    local = np.array([[dx, dy], [-dx, dy], [-dx, -dy], [dx, -dy]], dtype=np.float64)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)
    return local @ rot.T + np.asarray(translation_m[:2], dtype=np.float64)


def _polygon_area(poly: np.ndarray) -> float:
    if poly.shape[0] < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def _clip_by_edge(subject: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """One Sutherland-Hodgman clip against the half-plane left of edge a->b (clip poly is CCW)."""
    if subject.shape[0] == 0:
        return subject
    edge = b - a
    out: list[np.ndarray] = []
    n = subject.shape[0]
    for i in range(n):
        cur, prev = subject[i], subject[i - 1]
        cur_inside = np.cross(edge, cur - a) >= 0.0
        prev_inside = np.cross(edge, prev - a) >= 0.0
        if cur_inside:
            if not prev_inside:
                out.append(_segment_intersection(prev, cur, a, b))
            out.append(cur)
        elif prev_inside:
            out.append(_segment_intersection(prev, cur, a, b))
    return np.asarray(out, dtype=np.float64) if out else np.zeros((0, 2))


def _segment_intersection(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d1, d2 = p2 - p1, b - a
    denom = np.cross(d1, d2)
    if abs(denom) < 1e-15:
        return p2
    t = np.cross(a - p1, d2) / denom
    return p1 + t * d1


def bev_intersection_area(box_a: dict, box_b: dict) -> float:
    """Intersection area of two oriented BEV rectangles via Sutherland-Hodgman clipping."""
    poly_a = _rect_corners(box_a["translation_m"], box_a["yaw_rad"], box_a["size_wlh_m"][1], box_a["size_wlh_m"][0])
    poly_b = _rect_corners(box_b["translation_m"], box_b["yaw_rad"], box_b["size_wlh_m"][1], box_b["size_wlh_m"][0])
    clipped = poly_a
    n = poly_b.shape[0]
    for i in range(n):
        clipped = _clip_by_edge(clipped, poly_b[i], poly_b[(i + 1) % n])
        if clipped.shape[0] == 0:
            return 0.0
    return _polygon_area(clipped)


def bev_iou(box_a: dict, box_b: dict) -> float:
    """2D IoU of two oriented BEV footprints, height ignored entirely."""
    inter = bev_intersection_area(box_a, box_b)
    if inter <= 0.0:
        return 0.0
    area_a = box_a["size_wlh_m"][0] * box_a["size_wlh_m"][1]
    area_b = box_b["size_wlh_m"][0] * box_b["size_wlh_m"][1]
    union = area_a + area_b - inter
    return float(inter / union) if union > 0.0 else 0.0


def iou_3d(box_a: dict, box_b: dict) -> float:
    """3D IoU of two upright oriented boxes: BEV polygon intersection x height overlap.

    Both boxes are assumed upright (yaw about +z only, ISO 8855) -- true of every box this
    pipeline produces -- so the volume decomposes exactly into a BEV term and a 1D height term.
    """
    inter_bev = bev_intersection_area(box_a, box_b)
    if inter_bev <= 0.0:
        return 0.0
    za_lo, za_hi = box_a["translation_m"][2] - box_a["size_wlh_m"][2] / 2.0, box_a["translation_m"][2] + box_a["size_wlh_m"][2] / 2.0
    zb_lo, zb_hi = box_b["translation_m"][2] - box_b["size_wlh_m"][2] / 2.0, box_b["translation_m"][2] + box_b["size_wlh_m"][2] / 2.0
    z_overlap = max(0.0, min(za_hi, zb_hi) - max(za_lo, zb_lo))
    if z_overlap <= 0.0:
        return 0.0
    vol_a = box_a["size_wlh_m"][0] * box_a["size_wlh_m"][1] * box_a["size_wlh_m"][2]
    vol_b = box_b["size_wlh_m"][0] * box_b["size_wlh_m"][1] * box_b["size_wlh_m"][2]
    vol_inter = inter_bev * z_overlap
    vol_union = vol_a + vol_b - vol_inter
    return float(vol_inter / vol_union) if vol_union > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Frame handling: the global-vs-relative choice, made once, applied everywhere
# ---------------------------------------------------------------------------


def to_common_frame(translation_m: Sequence[float], yaw_rad: float, ego_pose: Transform, cfg: TrackConfig) -> tuple[np.ndarray, float]:
    """Ego(t) -> the registration frame this run is configured for.

    `ego_relative` is the identity: the declared-inferior alternate that treats
    every keyframe's ego frame as if it were the same fixed frame.
    """
    if cfg.icp_registration_frame == "ego_relative":
        return np.asarray(translation_m, dtype=np.float64), float(yaw_rad)
    xyz = apply_transform(ego_pose.matrix(), np.asarray([translation_m], dtype=np.float64))[0]
    yaw = wrap_to_pi_rad(float(yaw_rad) + ego_pose.yaw_rad)
    return xyz, yaw


def from_common_frame(translation: Sequence[float], yaw_rad: float, ego_pose: Transform, cfg: TrackConfig) -> tuple[np.ndarray, float]:
    """The registration frame -> ego(t). Inverse of `to_common_frame`."""
    if cfg.icp_registration_frame == "ego_relative":
        return np.asarray(translation, dtype=np.float64), float(yaw_rad)
    xyz = apply_transform(ego_pose.inverse_matrix(), np.asarray([translation], dtype=np.float64))[0]
    yaw = wrap_to_pi_rad(float(yaw_rad) - ego_pose.yaw_rad)
    return xyz, yaw


def rotate_vector_to_ego(vector_common: np.ndarray, ego_pose: Transform, cfg: TrackConfig) -> np.ndarray:
    """A velocity (a difference of positions) transforms by rotation only, never translation."""
    if cfg.icp_registration_frame == "ego_relative":
        return np.asarray(vector_common, dtype=np.float64)
    R = ego_pose.inverse_matrix()[:3, :3]
    return R @ np.asarray(vector_common, dtype=np.float64)


def points_to_common_frame(points_xyz: np.ndarray, ego_pose: Transform, cfg: TrackConfig) -> np.ndarray:
    if cfg.icp_registration_frame == "ego_relative":
        return points_xyz
    return apply_transform(ego_pose.matrix(), points_xyz)


# ---------------------------------------------------------------------------
# The constant-velocity / constant-yaw Kalman filter (comprehensive.md S7.3.6)
# ---------------------------------------------------------------------------


class ConstantVelocityYawKF:
    """State [x, y, z, vx, vy, vz, yaw] in the configured registration frame.

    No yaw-rate term: yaw is carried forward unchanged by the process model and
    corrected only by measurement updates. That is `comprehensive.md`'s 7-state
    vector exactly as specified, not a richer model substituted for it.
    """

    def __init__(self, x0: np.ndarray, cfg: TrackConfig) -> None:
        self.x = np.asarray(x0, dtype=np.float64).reshape(7)
        self.P = np.diag(
            [cfg.kf_init_pos_std_m**2] * 3 + [cfg.kf_init_vel_std_mps**2] * 3 + [cfg.kf_init_yaw_std_rad**2]
        )
        self._cfg = cfg

    def predict(self, dt_s: float) -> None:
        cfg = self._cfg
        F = np.eye(7)
        F[0, 3] = F[1, 4] = F[2, 5] = dt_s
        q_pos, q_vel, q_yaw = cfg.kf_process_noise_pos_m, cfg.kf_process_noise_vel_mps, cfg.kf_process_noise_yaw_rad
        Q = np.diag(
            [
                (q_pos * max(dt_s, 1e-6)) ** 2,
                (q_pos * max(dt_s, 1e-6)) ** 2,
                (q_pos * max(dt_s, 1e-6)) ** 2,
                (q_vel * max(dt_s, 1e-6)) ** 2,
                (q_vel * max(dt_s, 1e-6)) ** 2,
                (q_vel * max(dt_s, 1e-6)) ** 2,
                (q_yaw * max(dt_s, 1e-6)) ** 2,
            ]
        )
        self.x = F @ self.x
        self.x[6] = wrap_to_pi_rad(self.x[6])
        self.P = F @ self.P @ F.T + Q

    def update(self, measurement_xyz_yaw: np.ndarray) -> None:
        cfg = self._cfg
        H = np.zeros((4, 7))
        H[0, 0] = H[1, 1] = H[2, 2] = H[3, 6] = 1.0
        R = np.diag(
            [cfg.kf_measurement_noise_pos_m**2] * 3 + [cfg.kf_measurement_noise_yaw_rad**2]
        )
        z = np.asarray(measurement_xyz_yaw, dtype=np.float64).reshape(4)
        y = z - H @ self.x
        y[3] = wrap_to_pi_rad(y[3])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[6] = wrap_to_pi_rad(self.x[6])
        self.P = (np.eye(7) - K @ H) @ self.P

    def update_velocity(self, measured_velocity: np.ndarray) -> None:
        """Fold an ICP-measured velocity in as a Kalman measurement, not an overwrite.

        A direct assignment (`self.x[3:6] = measured_velocity`) would let ONE
        noisy registration -- ICP on a small, sparse cluster is exactly that --
        override the filter's whole running belief and propagate unchecked
        through every future `predict()`. Routing it through the same
        Kalman-gain machinery as every other measurement weighs it against the
        filter's current uncertainty instead.
        """
        cfg = self._cfg
        H = np.zeros((3, 7))
        H[0, 3] = H[1, 4] = H[2, 5] = 1.0
        R = np.diag([cfg.icp_measurement_noise_vel_mps**2] * 3)
        z = np.asarray(measured_velocity, dtype=np.float64).reshape(3)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[6] = wrap_to_pi_rad(self.x[6])
        self.P = (np.eye(7) - K @ H) @ self.P

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def yaw(self) -> float:
        return float(self.x[6])

    def as_dict(self) -> dict:
        return {
            "state": {name: round(float(v), 5) for name, v in zip(KF_STATE_NAMES, self.x)},
            "position_std_m": [round(float(math.sqrt(max(self.P[i, i], 0.0))), 4) for i in range(3)],
            "velocity_std_mps": [round(float(math.sqrt(max(self.P[i, i], 0.0))), 4) for i in range(3, 6)],
            "yaw_std_rad": round(float(math.sqrt(max(self.P[6, 6], 0.0))), 5),
        }


# ---------------------------------------------------------------------------
# ICP (point-to-point, Kabsch alignment). Hand-rolled: no point-cloud library
# is a dependency of this pipeline, and cluster sizes are small (tens-to-a-few-
# hundred points), so a brute-force nearest-neighbour correspondence per
# iteration is cheap and needs no external acceleration structure.
# ---------------------------------------------------------------------------


def icp_register(source_xyz: np.ndarray, target_xyz: np.ndarray, cfg: TrackConfig) -> dict | None:
    """Rigid transform aligning `source` onto `target`. None if either side is too sparse.

    Seeded with plain centroid alignment before the first correspondence
    search. Nearest-neighbour correspondence with NO initial alignment is
    wrong from the first iteration whenever the object's true displacement
    between keyframes exceeds its own cluster's spatial extent -- a fast
    vehicle can move further in 0.5 s than its own length -- so an unseeded
    start is not a simplification, it is a different, worse algorithm. A
    velocity-based seed (the track's own predicted displacement) was tried and
    dropped: on a track's first ICP attempt the filter's velocity is exactly
    the unreliable quantity ICP exists to measure, and seeding from it can
    steer the correspondence search away from the far better centroid seed.
    """
    if source_xyz.shape[0] < 3 or target_xyz.shape[0] < 3:
        return None
    src = source_xyz.astype(np.float64).copy()
    init_t = target_xyz.mean(axis=0) - src.mean(axis=0)
    src = src + init_t
    R_total = np.eye(3)
    t_total = init_t.copy()
    prev_mean = math.inf
    n_iter = 0
    converged = False
    for n_iter in range(1, cfg.icp_max_iterations + 1):
        d = np.linalg.norm(src[:, None, :] - target_xyz[None, :, :], axis=2)
        nn = np.argmin(d, axis=1)
        corr = target_xyz[nn]
        mean_dist = float(d[np.arange(src.shape[0]), nn].mean())

        src_c = src.mean(axis=0)
        tgt_c = corr.mean(axis=0)
        H = (src - src_c).T @ (corr - tgt_c)
        U, _, Vt = np.linalg.svd(H)
        det_sign = float(np.sign(np.linalg.det(Vt.T @ U.T))) or 1.0
        D = np.diag([1.0, 1.0, det_sign])
        R_step = Vt.T @ D @ U.T
        t_step = tgt_c - R_step @ src_c

        src = (R_step @ src.T).T + t_step
        R_total = R_step @ R_total
        t_total = R_step @ t_total + t_step

        if abs(prev_mean - mean_dist) < cfg.icp_convergence_tol_m:
            prev_mean = mean_dist
            converged = True
            break
        prev_mean = mean_dist

    return {
        "rotation": R_total,
        "translation_m": t_total,
        "n_iterations": n_iter,
        "converged": converged,
        "mean_residual_m": round(prev_mean, 6),
        "n_source_points": int(source_xyz.shape[0]),
        "n_target_points": int(target_xyz.shape[0]),
    }


# ---------------------------------------------------------------------------
# Reconstructing Stage 6's kept-cluster points for one instance
#
# Stage 6 persists the FITTED BOX, not the member point set. ICP needs that
# point set, so it is reconstructed here by re-running Stage 6's own pure
# functions (`canonical_order`, `dbscan_bev`, `select_cluster`) with the exact
# `eps_m` / `min_samples` Stage 6 recorded on the row -- bit-for-bit the same
# points Stage 6 fit its box to, not a re-derivation that could drift from it.
#
# That identity is conditional on replaying every step Stage 6 took, so the
# near-cut retry (C24) is replayed too, from the `near_cut` ledger the row
# carries. Reconstructing without it would hand ICP the untrimmed cluster for
# precisely the boxes Stage 6 repaired -- the one population where the two
# point sets differ, and the one where a silent drift would be least visible.
# ---------------------------------------------------------------------------


class CloudCache:
    """The current and immediately-previous processed keyframe's painted cloud.

    Bounded to two entries: ICP only ever needs "now" and "the last keyframe
    this track was matched in", and a scene's clouds are read once each.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = {}

    def get(
        self, keyframe_token: str, cloud_path: str, points_path: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        if keyframe_token not in self._entries:
            if not os.path.isfile(points_path):
                raise UpstreamRefusal(f"{points_path} not found; run `python3 -m pipeline.stage5_lift.lift` first")
            with np.load(points_path) as npz:
                frame = str(npz["__frame__"][0])
                point_index = npz["point_index"].astype(np.int64)
                instance_id = npz["instance_id"].astype(np.int64)
                # Needed to replay Stage 6's near-cut retry (C24); absent on a
                # Stage 5 output that predates the field, in which case no row
                # can have had the cut applied either.
                depth_m = npz["depth_m"].astype(np.float64) if "depth_m" in npz.files else None
            if frame != EGO:
                raise TrackContractError(f"{points_path}: painted points claim frame={frame!r}, not {EGO!r}")
            cloud = read_pcd_bin(cloud_path)[:, :3].astype(np.float64)
            if point_index.size and int(point_index.max()) >= cloud.shape[0]:
                raise TrackContractError(f"{points_path}: point index out of range for a {cloud.shape[0]}-point cloud")
            self._entries[keyframe_token] = (cloud, point_index, instance_id, depth_m)
        return self._entries[keyframe_token]

    def prune_to(self, keep_tokens: set) -> None:
        for token in list(self._entries):
            if token not in keep_tokens:
                del self._entries[token]


def applied_near_cut_depth_m(det: dict) -> float | None:
    """The near-cut depth Stage 6 actually APPLIED to this row, or None.

    Keyed on `applied`, not on `triggered`: a row whose retry was rejected
    (the footprint would have grown, or too few points survived) kept its
    original cluster, and replaying the cut on it would desynchronise the
    very rows Stage 6 deliberately left alone.
    """
    near_cut = det.get("near_cut")
    if not isinstance(near_cut, dict) or not near_cut.get("applied"):
        return None
    return float(near_cut["depth_m"])


def reconstruct_cluster_points(
    keyframe_token: str,
    cloud_path: str,
    points_path: str,
    instance_id: int,
    eps_m: float,
    min_samples: int,
    cache: CloudCache,
    near_cut_depth_m: float | None = None,
) -> np.ndarray | None:
    """The exact ego-frame points Stage 6's kept cluster held for this instance, or None.

    `near_cut_depth_m` replays Stage 6's near-cut retry (C24) for the rows that
    recorded it applied. Without it this function would hand ICP the UNTRIMMED
    cluster for exactly the boxes Stage 6 repaired — registering one point set
    against a box fitted to a different one.
    """
    cloud, point_index, instance_id_arr, depth_arr = cache.get(keyframe_token, cloud_path, points_path)
    selected = instance_id_arr == instance_id
    rows_of_cloud = point_index[selected]
    if rows_of_cloud.shape[0] < min_samples:
        return None
    points_xyz = cloud[rows_of_cloud]
    order = canonical_order(points_xyz, rows_of_cloud)
    ordered_xyz = points_xyz[order]
    if near_cut_depth_m is not None and depth_arr is not None:
        ordered_depth = depth_arr[selected][order]
        keep = ordered_depth <= float(ordered_depth.min()) + near_cut_depth_m
        if int(np.count_nonzero(keep)) < min_samples:
            return None
        ordered_xyz = ordered_xyz[keep]
    labels = dbscan_bev(ordered_xyz[:, :2], eps_m, min_samples)
    choice = select_cluster(labels, ordered_xyz)
    if choice is None:
        return None
    return ordered_xyz[choice.member]


# ---------------------------------------------------------------------------
# Yaw-consistency enforcement (S7.3.7)
# ---------------------------------------------------------------------------


def disambiguate_yaw(
    local_yaw_rad: float,
    predicted_local_yaw_rad: float | None,
    predicted_velocity_local: np.ndarray | None,
    cfg: TrackConfig,
) -> tuple[float, bool, str]:
    """Pick whichever of {yaw, yaw + pi} matches the track's own history.

    Stage 6's yaw is an axis, not a direction (systematic for near-square or
    few-point footprints, S5.7). Returns (disambiguated_yaw, flip_applied,
    reference_source). `reference_source` is `"none_first_observation"` when a
    track has no history to disambiguate against -- the raw axis-only yaw is
    returned unchanged in that case.
    """
    candidates = (local_yaw_rad, wrap_to_pi_rad(local_yaw_rad + math.pi))
    speed_mps = float(np.linalg.norm(predicted_velocity_local)) if predicted_velocity_local is not None else 0.0
    if predicted_velocity_local is not None and speed_mps >= cfg.yaw_consistency_min_speed_mps:
        reference = float(math.atan2(predicted_velocity_local[1], predicted_velocity_local[0]))
        source = "velocity_heading"
    elif predicted_local_yaw_rad is not None:
        reference = predicted_local_yaw_rad
        source = "previous_track_yaw"
    else:
        return local_yaw_rad, False, "none_first_observation"
    diffs = [abs(wrap_to_pi_rad(c - reference)) for c in candidates]
    idx = 0 if diffs[0] <= diffs[1] else 1
    return candidates[idx], idx == 1, source


# ---------------------------------------------------------------------------
# Appearance: the reid_embedding role, DINOv2 crop-CLS adapter
# ---------------------------------------------------------------------------


class Dinov2ReidAdapter:
    """DINOv2 as the `reid_embedding` role: CROP_CLS over the mask's tight box.

    Preprocessing belongs to the role, not the model (P1-2): this adapter's
    transform is a per-object crop resize, never Stage 2's whole-image
    transform, even though both roles may share a checkpoint.
    """

    def __init__(self, spec: CheckpointSpec, cfg: TrackConfig) -> None:
        self._spec = spec
        self._cfg = cfg
        self._device = cfg.device
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None

    @property
    def roles(self) -> tuple[str, ...]:
        return (REID_EMBEDDING,)

    @property
    def spec(self) -> CheckpointSpec:
        return self._spec

    @property
    def device(self) -> str:
        return self._device

    @property
    def min_crop_px(self) -> int:
        return self._cfg.min_crop_px

    def load(self) -> None:
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel
        except ImportError as exc:
            raise ModelUnavailable(f"{self._spec.model_id}: {exc}") from exc
        self._torch = torch
        kwargs: dict[str, Any] = {}
        if self._spec.revision:
            kwargs["revision"] = self._spec.revision
        try:
            self._processor = AutoImageProcessor.from_pretrained(self._spec.model_id, **kwargs)
            self._model = AutoModel.from_pretrained(self._spec.model_id, **kwargs).to(self._device).eval()
        except OSError as exc:
            # transformers signals a missing or undownloadable checkpoint as
            # OSError (hub HTTP errors subclass it too) -- the same "this role
            # is unavailable" fact as a missing import, so it takes the same
            # declared path: IoU-only fallback, or a clean refusal under
            # --require-appearance. Never a raw traceback.
            raise ModelUnavailable(f"{self._spec.model_id}: {exc}") from exc

    def unload(self) -> None:
        self._model = None
        self._processor = None
        if self._torch is not None and self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()

    def embed_crops(
        self,
        image: np.ndarray,
        boxes_xyxy_px: np.ndarray,
        *,
        masks: np.ndarray | None = None,
        ids: Sequence[str] | None = None,
    ) -> EmbeddingBatch:
        if self._model is None:
            raise RuntimeError("adapter is not loaded")
        torch = self._torch
        boxes = np.asarray(boxes_xyxy_px, dtype=np.float64).reshape(-1, 4)
        crops = []
        for x1, y1, x2, y2 in boxes:
            crop = image[max(0, int(y1)) : max(0, int(y2)), max(0, int(x1)) : max(0, int(x2))]
            crops.append(crop)
        inputs = self._processor(images=crops, return_tensors="pt").to(self._device)
        with torch.inference_mode():
            outputs = self._model(**inputs)
        vectors = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()  # CLS token
        model_input = inputs["pixel_values"].shape[-2:]
        return EmbeddingBatch(
            vectors=vectors,
            semantics="crop_cls",
            preprocessing=f"{self._spec.provider}:crop_cls_processor",
            model_input_size_px=(int(model_input[1]), int(model_input[0])),
            source_size_px=(int(image.shape[1]), int(image.shape[0])),
            ids=list(ids) if ids is not None else None,
        )


REID_PROVIDERS: dict[str, str] = {
    "dinov2": "dinov2_reid",
    "dinov3": "dinov3_reid",
}

REID_PROVENANCE: dict[str, str] = {
    "dinov2_reid": "pilot tier, S7.1",
    "dinov3_reid": (
        "2026-08-14, human-directed: DINOv3 (LVD-1689M) as the reid_embedding arm of the "
        "detector/re-ID comparison. Same role contract as DINOv2 -- AutoModel, CLS token of "
        "last_hidden_state, per-object crop -- so the adapter is shared and only the checkpoint "
        "differs. facebook/dinov3-vits16 is the size-matched counterpart of facebook/dinov2-small "
        "(both 384-d, both ~21M params); the patch size differs (16 vs 14) and is the model's own"
    ),
}


def infer_reid_provider(model_id: str) -> str:
    """model_id -> registered reid provider name.

    Derived, never hardcoded: recording `dinov2_reid` for a DINOv3 checkpoint
    would put a false provider in the manifest that decided the numbers, and the
    manifest is the only place a reader can see which model actually ran.
    """
    base = model_id.rsplit("/", 1)[-1].lower()
    for prefix, provider in REID_PROVIDERS.items():
        if base.startswith(prefix):
            return provider
    raise UpstreamRefusal(
        f"unknown reid_embedding checkpoint {model_id!r}: no registered provider claims it "
        f"(known prefixes: {sorted(REID_PROVIDERS)}). A checkpoint whose provider nobody "
        "established would be recorded under another model's name"
    )


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norm, 1e-12)


# ---------------------------------------------------------------------------
# Track state
# ---------------------------------------------------------------------------


@dataclass
class Track:
    track_id: int
    class_name: str
    kf: ConstantVelocityYawKF
    last_box: dict
    last_t_ns: int
    last_keyframe_token: str
    last_num_lidar_pts: int
    last_embedding: np.ndarray | None
    hits: int = 1
    misses: int = 0
    age: int = 1
    status: str = TRACK_TENTATIVE

    def confirm_if_ready(self, cfg: TrackConfig) -> None:
        if self.hits >= cfg.min_hits_to_confirm and self.status == TRACK_TENTATIVE:
            self.status = TRACK_CONFIRMED


def predicted_local_box(track: Track, ego_pose_cur: Transform, dt_s: float, cfg: TrackConfig) -> dict:
    """Propagate a track's Kalman state to `dt_s` ahead and express it in ego(t_cur).

    This IS the "propagate previous boxes by estimated velocity before computing
    IoU" step; the box returned here, never the track's raw last-observed box,
    is what the cost matrix measures IoU against.
    """
    track.kf.predict(dt_s)
    xyz_ego, yaw_ego = from_common_frame(track.kf.position, track.kf.yaw, ego_pose_cur, cfg)
    return {
        "translation_m": [float(v) for v in xyz_ego],
        "size_wlh_m": list(track.last_box["size_wlh_m"]),
        "yaw_rad": yaw_ego,
    }


# ---------------------------------------------------------------------------
# Association: cost matrix + Hungarian
# ---------------------------------------------------------------------------


@dataclass
class MatchResult:
    track_index_to_detection: dict[int, int]
    detection_index_to_track: dict[int, int]
    ious_bev: np.ndarray
    ious_3d: np.ndarray
    cosines: np.ndarray
    trusted: np.ndarray


def build_and_solve(
    tracks: Sequence[Track],
    predicted_boxes: Sequence[dict],
    detections: Sequence[dict],
    embeddings_by_instance: dict[int, np.ndarray],
    cfg: TrackConfig,
) -> MatchResult:
    """Gate then score, per S11 decision 5 / M-11.

    The GATE and the reported diagnostic are not necessarily the same
    dimensionality: `iou_mode` picks which IoU gates and scores the match
    (`bev` by default -- see the module docstring's association section for
    why), but BOTH the BEV and the full 3D IoU are always computed and
    recorded per pair, so a consumer can see what the road-not-taken would
    have measured.
    """
    n_t, n_d = len(tracks), len(detections)
    ious_bev = np.zeros((n_t, n_d))
    ious_3d = np.zeros((n_t, n_d))
    cosines = np.full((n_t, n_d), np.nan)
    trusted = np.zeros((n_t, n_d), dtype=bool)
    cost = np.full((n_t, n_d), cfg.ineligible_cost)

    for i, (track, pbox) in enumerate(zip(tracks, predicted_boxes)):
        for j, det in enumerate(detections):
            if cfg.class_gate and track.class_name != det["class_name"]:
                continue
            iou_bev_val = bev_iou(pbox, det["box"])
            iou_3d_val = iou_3d(pbox, det["box"])
            ious_bev[i, j] = iou_bev_val
            ious_3d[i, j] = iou_3d_val
            gate_iou = iou_bev_val if cfg.iou_mode == "bev" else iou_3d_val
            if gate_iou < cfg.iou_gate:
                continue
            score = gate_iou
            det_embedding = embeddings_by_instance.get(det["instance_id"])
            if track.last_embedding is not None and det_embedding is not None:
                cos = float(np.dot(track.last_embedding, det_embedding))
                cosines[i, j] = cos
                trusted[i, j] = True
                score = gate_iou * max(0.0, cos)
            cost[i, j] = -score

    track_to_det: dict[int, int] = {}
    det_to_track: dict[int, int] = {}
    if n_t and n_d:
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            if cost[r, c] < cfg.ineligible_cost:
                track_to_det[int(r)] = int(c)
                det_to_track[int(c)] = int(r)
    return MatchResult(track_to_det, det_to_track, ious_bev, ious_3d, cosines, trusted)


# ---------------------------------------------------------------------------
# One scene
# ---------------------------------------------------------------------------


def track_scene(
    scene_name: str,
    keyframe_tokens_sorted: Sequence[str],
    keyframes: dict[str, KeyframeRecord],
    rows_by_keyframe: dict[str, list[dict]],
    mask_index: dict[str, dict],
    substrate: Substrate,
    dataroot: str,
    stage5_dir: str,
    adapter: Dinov2ReidAdapter | None,
    cfg: TrackConfig,
) -> tuple[list[dict], dict]:
    ego_pose_table = substrate.by_token("ego_pose.json")
    cloud_cache = CloudCache()

    tracks: list[Track] = []
    next_track_id = 0
    out_rows: list[dict] = []
    totals = {
        "n_keyframes": 0,
        "n_detections": 0,
        "n_matched": 0,
        "n_births": 0,
        "n_deaths": 0,
        "n_icp_attempted": 0,
        "n_icp_succeeded": 0,
        "n_kalman_fallback": 0,
        "n_yaw_flips": 0,
        "n_appearance_trusted_pairs": 0,
        "n_tracks_confirmed": 0,
        "n_tracks_stable": 0,
    }

    for keyframe_index, keyframe_token in enumerate(keyframe_tokens_sorted):
        keyframe = keyframes[keyframe_token]
        ego_pose_cur = Transform.from_nuscenes(
            ego_pose_table[keyframe.lidar_ego_pose_token], source_frame=EGO, parent_frame=NUSCENES_GLOBAL
        )
        rows = rows_by_keyframe.get(keyframe_token, [])
        detections = [r for r in rows if r.get("status") == STATUS_FIT and r.get("box") is not None]
        detections.sort(key=lambda r: r["instance_id"])
        totals["n_keyframes"] += 1
        totals["n_detections"] += len(detections)

        embeddings_by_instance: dict[int, np.ndarray] = {}
        if adapter is not None and detections:
            embeddings_by_instance = _embed_keyframe_detections(
                keyframe, detections, mask_index.get(keyframe_token, {}), dataroot, adapter, cfg
            )

        active = [t for t in tracks if t.status != TRACK_DEAD]
        active.sort(key=lambda t: t.track_id)
        dt_by_track = {t.track_id: (keyframe.t_ns - t.last_t_ns) / 1.0e9 for t in active}
        predicted = [predicted_local_box(t, ego_pose_cur, dt_by_track[t.track_id], cfg) for t in active]

        match = build_and_solve(active, predicted, detections, embeddings_by_instance, cfg)
        matched_track_ids: set = set()
        matched_detection_indices: set = set()

        for track_pos, det_pos in match.track_index_to_detection.items():
            track = active[track_pos]
            det = detections[det_pos]
            matched_track_ids.add(track.track_id)
            matched_detection_indices.add(det_pos)
            if match.trusted[track_pos, det_pos]:
                totals["n_appearance_trusted_pairs"] += 1

            match_ious = {
                "bev": round(float(match.ious_bev[track_pos, det_pos]), 5),
                "3d": round(float(match.ious_3d[track_pos, det_pos]), 5),
            }
            match_cosine = match.cosines[track_pos, det_pos]
            row_out = _update_track_and_build_row(
                track, det, predicted[track_pos], ego_pose_cur, keyframe_index, keyframe_token,
                keyframes, keyframe_tokens_sorted, ego_pose_table, cloud_cache, stage5_dir,
                embeddings_by_instance,
                match_ious, None if np.isnan(match_cosine) else round(float(match_cosine), 5), cfg, totals,
            )
            out_rows.append(row_out)
            totals["n_matched"] += 1

        for det_pos, det in enumerate(detections):
            if det_pos in matched_detection_indices:
                continue
            track = _birth_track(
                det, next_track_id, ego_pose_cur, embeddings_by_instance.get(det["instance_id"]),
                cloud_cache, stage5_dir, cfg,
            )
            next_track_id += 1
            tracks.append(track)
            totals["n_births"] += 1
            out_rows.append(_birth_row(det, track))

        for track in active:
            if track.track_id in matched_track_ids:
                continue
            track.misses += 1
            track.age += 1
            track.last_box = predicted[[t.track_id for t in active].index(track.track_id)]
            track.last_t_ns = keyframe.t_ns
            track.last_keyframe_token = keyframe_token
            if track.misses > cfg.max_misses_before_death:
                track.status = TRACK_DEAD
                totals["n_deaths"] += 1

        for row in rows:
            if row.get("status") != STATUS_FIT or row.get("box") is None:
                out_rows.append(_passthrough_row(row))

        # ICP only ever needs "current" (just reconstructed above, possibly
        # several times) and "whatever a track carries on `last_box`" -- the
        # cache's own entries are never read again once this keyframe ends.
        cloud_cache.prune_to({keyframe_token})

    for track in tracks:
        if track.status == TRACK_CONFIRMED:
            totals["n_tracks_confirmed"] += 1
        if track.hits >= cfg.stable_track_hits:
            totals["n_tracks_stable"] += 1

    totals["n_tracks_total"] = len(tracks)
    return out_rows, totals


def _embed_keyframe_detections(
    keyframe: KeyframeRecord,
    detections: Sequence[dict],
    mask_row: dict,
    dataroot: str,
    adapter: Dinov2ReidAdapter,
    cfg: TrackConfig,
) -> dict[int, np.ndarray]:
    """One embedding per detection whose crop clears `min_crop_px`, keyed by instance_id."""
    box_lookup = {
        (c["channel"], c["proposal_index"]): tuple(c["mask_box_xyxy_px"])
        for c in mask_row.get("candidates", ())
        if c.get("kept")
    }
    by_channel: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = {}
    for det in detections:
        key = (det["channel"], det["proposal_index"])
        box = box_lookup.get(key)
        if box is None:
            continue
        w, h = box[2] - box[0], box[3] - box[1]
        if w < cfg.min_crop_px or h < cfg.min_crop_px:
            continue
        by_channel.setdefault(det["channel"], []).append((det["instance_id"], box))

    out: dict[int, np.ndarray] = {}
    from PIL import Image

    for channel, entries in by_channel.items():
        observation = keyframe.cameras.get(channel)
        if observation is None:
            continue
        with Image.open(os.path.join(dataroot, observation.path)) as im:
            image = np.asarray(im.convert("RGB"), dtype=np.uint8)
        instance_ids = [e[0] for e in entries]
        boxes = np.asarray([e[1] for e in entries], dtype=np.float64)
        batch = adapter.embed_crops(image, boxes, ids=[str(i) for i in instance_ids])
        vectors = _l2_normalize(np.asarray(batch.vectors, dtype=np.float64))
        for instance_id, vec in zip(instance_ids, vectors):
            out[instance_id] = vec
    return out


def _birth_track(
    det: dict,
    track_id: int,
    ego_pose_cur: Transform,
    embedding: np.ndarray | None,
    cloud_cache: CloudCache,
    stage5_dir: str,
    cfg: TrackConfig,
) -> Track:
    box = det["box"]
    xyz_common, yaw_common = to_common_frame(box["translation_m"], box["yaw_rad"], ego_pose_cur, cfg)
    kf = ConstantVelocityYawKF(
        np.array([xyz_common[0], xyz_common[1], xyz_common[2], 0.0, 0.0, 0.0, yaw_common]), cfg
    )
    # Reconstructed at birth too, not just on the first update: without this a
    # track's SECOND detection could never attempt ICP (no prior member points
    # to register against) even when both clusters are well past the sparse
    # guard.
    # `points_path` is RELATIVE to the Stage 5 out dir (lift.py records it via
    # os.path.relpath; Stage 6 re-emits it unchanged), so it is joined here
    # exactly as cluster.py's load_keyframe_points joins it. `cloud_path` is
    # absolute (Stage 1) and passes through unjoined.
    birth_box = dict(box)
    birth_box["_member_points"] = reconstruct_cluster_points(
        det["keyframe_token"], det.get("cloud_path", ""),
        os.path.join(stage5_dir, det.get("points_path", "")),
        det["instance_id"], float(det.get("eps_m", 0.0)), int(det.get("min_samples", 0)), cloud_cache,
        near_cut_depth_m=applied_near_cut_depth_m(det),
    )
    return Track(
        track_id=track_id,
        class_name=det["class_name"],
        kf=kf,
        last_box=birth_box,
        last_t_ns=det["t_ns"],
        last_keyframe_token=det["keyframe_token"],
        last_num_lidar_pts=int(det.get("num_lidar_pts", 0)),
        last_embedding=embedding,
        status=TRACK_CONFIRMED if cfg.min_hits_to_confirm <= 1 else TRACK_TENTATIVE,
    )


def _birth_row(det: dict, track: Track) -> dict:
    return {
        **det,
        "spec": STAGE_SPEC,
        "track_id": track.track_id,
        "track_event": "birth",
        "track_status": track.status,
        "track_hits": track.hits,
        "track_age": track.age,
        "track_misses": track.misses,
        "velocity_mps": [0.0, 0.0, 0.0],
        "velocity_source": "none_first_observation",
        "velocity_semantics": "undefined_first_observation",
        "icp": None,
        "icp_frame": None,
        "kalman": track.kf.as_dict(),
        "yaw_flip_applied": False,
        "yaw_reference_source": "none_first_observation",
        "predicted_box": None,
        "match_iou_bev": None,
        "match_iou_3d": None,
        "match_cosine": None,
        "appearance_trusted": False,
    }


def _passthrough_row(row: dict) -> dict:
    return {
        **row,
        "spec": STAGE_SPEC,
        "pre_track_spec": row.get("spec"),
        "track_id": None,
        "track_event": "no_box",
        "track_status": None,
        "track_hits": None,
        "track_age": None,
        "track_misses": None,
        "velocity_mps": None,
        "velocity_source": None,
        "velocity_semantics": None,
        "icp": None,
        "icp_frame": None,
        "kalman": None,
        "yaw_flip_applied": False,
        "yaw_reference_source": None,
        "predicted_box": None,
        "match_iou_bev": None,
        "match_iou_3d": None,
        "match_cosine": None,
        "appearance_trusted": False,
    }


def _update_track_and_build_row(
    track: Track,
    det: dict,
    predicted_box: dict,
    ego_pose_cur: Transform,
    keyframe_index: int,
    keyframe_token: str,
    keyframes: dict[str, KeyframeRecord],
    keyframe_tokens_sorted: Sequence[str],
    ego_pose_table: dict,
    cloud_cache: CloudCache,
    stage5_dir: str,
    embeddings_by_instance: dict[int, np.ndarray],
    match_ious: dict,
    match_cosine: float | None,
    cfg: TrackConfig,
    totals: dict,
) -> dict:
    box = det["box"]
    n_points_cur = int(det.get("num_lidar_pts", 0))

    # Predicted velocity heading, pre-update, is the yaw-consistency reference.
    predicted_velocity_local = rotate_vector_to_ego(track.kf.velocity, ego_pose_cur, cfg) if track.hits >= 1 else None
    disambiguated_yaw, flipped, yaw_source = disambiguate_yaw(
        box["yaw_rad"], predicted_box["yaw_rad"], predicted_velocity_local, cfg
    )
    if flipped:
        totals["n_yaw_flips"] += 1

    xyz_common, yaw_common = to_common_frame(box["translation_m"], disambiguated_yaw, ego_pose_cur, cfg)
    track.kf.update(np.array([xyz_common[0], xyz_common[1], xyz_common[2], yaw_common]))

    icp_ledger: dict | None = None
    velocity_source = "kalman_filter"
    dt_s = (det["t_ns"] - track.last_t_ns) / 1.0e9
    is_adjacent = keyframe_index > 0 and track.last_keyframe_token == keyframe_tokens_sorted[keyframe_index - 1]

    # Reconstructed once regardless of whether ICP runs: this run's member
    # points become the track's `_member_points` for the NEXT update's ICP
    # attempt either way, so there is no reason to fit the cluster twice.
    # `points_path` is stage5-out-dir-relative (see _birth_track), joined here
    # the same way cluster.py's load_keyframe_points joins it.
    cur_points = reconstruct_cluster_points(
        keyframe_token, det.get("cloud_path", ""),
        os.path.join(stage5_dir, det.get("points_path", "")),
        det["instance_id"], float(det.get("eps_m", 0.0)), int(det.get("min_samples", 0)), cloud_cache,
        near_cut_depth_m=applied_near_cut_depth_m(det),
    )
    # The previous cluster's own member points, reconstructed with ITS OWN
    # eps_m/min_samples when it was current, and carried on the track since.
    prev_points = track.last_box.get("_member_points")

    if n_points_cur < cfg.kalman_fallback_min_points:
        icp_ledger = {"attempted": False, "reason": REASON_SPARSE_CURRENT}
    elif track.last_num_lidar_pts < cfg.icp_min_points_each_side:
        icp_ledger = {"attempted": False, "reason": REASON_SPARSE_PREVIOUS}
    elif cfg.icp_require_adjacent_keyframe and not is_adjacent:
        icp_ledger = {"attempted": False, "reason": REASON_NOT_ADJACENT}
    elif prev_points is None or cur_points is None:
        icp_ledger = {"attempted": False, "reason": REASON_NO_CLUSTER}
    else:
        prev_keyframe = keyframes[track.last_keyframe_token]
        prev_ego_pose = Transform.from_nuscenes(
            ego_pose_table[prev_keyframe.lidar_ego_pose_token], source_frame=EGO, parent_frame=NUSCENES_GLOBAL
        )
        totals["n_icp_attempted"] += 1
        prev_common = points_to_common_frame(prev_points, prev_ego_pose, cfg)
        cur_common = points_to_common_frame(cur_points, ego_pose_cur, cfg)
        result = icp_register(prev_common, cur_common, cfg)
        if result is None:
            icp_ledger = {"attempted": True, "succeeded": False, "reason": "too_few_points_for_rigid_fit"}
        else:
            totals["n_icp_succeeded"] += 1
            # icp_register's (R, t) satisfies aligned = R @ src + t, so the
            # cluster's actual displacement is (R @ mu + t) - mu for the
            # previous centroid mu -- t alone is exact only when R == I. In
            # the default global_absolute frame mu is a nuScenes WORLD
            # coordinate (hundreds of metres from the origin), so dropping
            # the (R - I) @ mu term folds any residual rotation's lever arm
            # about the world origin into the velocity measurement.
            prev_centroid = prev_common.mean(axis=0)
            displacement_m = result["translation_m"] + result["rotation"] @ prev_centroid - prev_centroid
            velocity_common = displacement_m / max(dt_s, 1e-6)
            track.kf.update_velocity(velocity_common)
            velocity_source = "icp"
            icp_ledger = {
                "attempted": True,
                "succeeded": True,
                "n_iterations": result["n_iterations"],
                "converged": result["converged"],
                "mean_residual_m": result["mean_residual_m"],
                "n_source_points": result["n_source_points"],
                "n_target_points": result["n_target_points"],
            }

    if velocity_source != "icp":
        totals["n_kalman_fallback"] += 1

    velocity_local = rotate_vector_to_ego(track.kf.velocity, ego_pose_cur, cfg)
    velocity_semantics = "absolute_world_frame" if cfg.icp_registration_frame == "global_absolute" else "relative_to_ego"

    updated_box = dict(box)
    updated_box["yaw_rad"] = wrap_to_pi_rad(disambiguated_yaw)
    updated_box["rotation_wxyz"] = list(quaternion_from_yaw_rad(updated_box["yaw_rad"]))
    updated_box["_member_points"] = cur_points

    track.last_box = updated_box
    track.last_t_ns = det["t_ns"]
    track.last_keyframe_token = keyframe_token
    track.last_num_lidar_pts = n_points_cur
    if embeddings_by_instance.get(det["instance_id"]) is not None:
        track.last_embedding = embeddings_by_instance[det["instance_id"]]
    track.hits += 1
    track.age += 1
    track.misses = 0
    track.confirm_if_ready(cfg)

    out_box = {k: v for k, v in updated_box.items() if k != "_member_points"}
    row = {
        **det,
        "spec": STAGE_SPEC,
        "box": out_box,
        "track_id": track.track_id,
        "track_event": "matched",
        "track_status": track.status,
        "track_hits": track.hits,
        "track_age": track.age,
        "track_misses": track.misses,
        "velocity_mps": [round(float(v), 4) for v in velocity_local],
        "velocity_source": velocity_source,
        "velocity_semantics": velocity_semantics,
        "icp": icp_ledger,
        "icp_frame": cfg.icp_registration_frame,
        "kalman": track.kf.as_dict(),
        "yaw_flip_applied": bool(flipped),
        "yaw_reference_source": yaw_source,
        "predicted_box": {
            "translation_m": [round(float(v), 4) for v in predicted_box["translation_m"]],
            "yaw_rad": round(float(predicted_box["yaw_rad"]), 6),
        },
        "match_iou_bev": match_ious["bev"],
        "match_iou_3d": match_ious["3d"],
        "match_cosine": match_cosine,
        "appearance_trusted": match_cosine is not None,
    }
    return row


# ---------------------------------------------------------------------------
# Upstream
# ---------------------------------------------------------------------------


def load_upstream(
    paths: Paths,
    stage1_dir: str,
    stage4_dir: str,
    stage5_dir: str,
    stage6_dir: str,
    *,
    accept_degraded: bool = False,
):
    """Refuse to start unless ALL upstreams COMPLETED on THIS substrate.

    Stage 7 joins four upstream artifacts -- Stage 1's keyframes and clouds,
    Stage 4's mask boxes for the reid crops, Stage 5's painted point indices
    (read per keyframe by `reconstruct_cluster_points`), Stage 6's fitted
    boxes. All four go through the C16 gate: absent marker refuses
    unconditionally, degraded marker refuses unless `accept_degraded` -- one
    flag for all, because accepting one degraded upstream and refusing another
    is not a meaningful position when the join needs every one of them. (Stage
    5 was previously consumed here with no gate at all; its files are read by
    this stage directly, so it is gated like the other three.)
    """
    current = metadata_fingerprint(paths)
    _stage1_manifest, stage1_marker = require_upstream(
        stage1_dir,
        stage_name="Stage 1",
        module_hint="pipeline.stage1_ingestion.ingest",
        current_fingerprint=current,
        accept_degraded=accept_degraded,
    )
    _stage4_manifest, stage4_marker = require_upstream(
        stage4_dir,
        stage_name="Stage 4",
        module_hint="pipeline.stage4_masks.masks",
        current_fingerprint=current,
        accept_degraded=accept_degraded,
    )
    _stage5_manifest, stage5_marker = require_upstream(
        stage5_dir,
        stage_name="Stage 5",
        module_hint="pipeline.stage5_lift.lift",
        current_fingerprint=current,
        accept_degraded=accept_degraded,
    )
    stage6_manifest, stage6_marker = require_upstream(
        stage6_dir,
        stage_name="Stage 6",
        module_hint="pipeline.stage6_cluster.cluster",
        current_fingerprint=current,
        accept_degraded=accept_degraded,
    )
    return stage6_manifest, stage1_marker, stage4_marker, stage5_marker, stage6_marker


def read_box_rows(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise UpstreamRefusal(f"{path} not found; run `python3 -m pipeline.stage6_cluster.cluster` first")
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    paths: Paths,
    stage6_manifest: dict,
    stage1_marker,
    stage4_marker,
    stage5_marker,
    stage6_marker,
    cfg: TrackConfig,
    stage1_dir: str,
    stage4_dir: str,
    stage5_dir: str,
    stage6_dir: str,
    out_dir: str,
    scene_names: Sequence[str] | None,
    appearance_enabled: bool,
    appearance_unavailable_reason: str | None,
) -> tuple[dict, int]:
    started = time.time()
    errors = cfg.validate()
    if errors:
        raise UpstreamRefusal("; ".join(errors))
    # Any marker still standing describes the PREVIOUS run of this stage; it
    # comes down before the first write (C16).
    clear_markers(out_dir)

    substrate = Substrate.load(paths)
    adapter: Dinov2ReidAdapter | None = None
    if appearance_enabled:
        spec = CheckpointSpec(
            role=REID_EMBEDDING, provider=infer_reid_provider(cfg.reid_model_id),
            model_id=cfg.reid_model_id, revision=cfg.reid_revision,
            provenance=REID_PROVENANCE.get(infer_reid_provider(cfg.reid_model_id), ""),
        )
        spec_errors = spec.validate(prefix="checkpoint: ")
        if spec_errors:
            raise UpstreamRefusal("; ".join(spec_errors) + " -- pass --reid-revision")
        adapter = Dinov2ReidAdapter(spec, cfg)
        adapter.load()

    root = os.path.join(stage6_dir, "scenes")
    if not os.path.isdir(root):
        raise UpstreamRefusal(f"{root} not found; Stage 6 wrote no scenes")
    names = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
    if scene_names:
        missing = sorted(set(scene_names) - set(names))
        if missing:
            raise UpstreamRefusal(f"requested scene(s) not present in Stage 6 output: {missing}")
        names = [n for n in names if n in scene_names]

    per_scene: list[dict] = []
    degraded = False
    totals: dict = {}

    for scene_name in names:
        rows = read_box_rows(os.path.join(root, scene_name, "boxes.jsonl"))
        keyframe_records = read_records(
            os.path.join(stage1_dir, "scenes", scene_name, "keyframes.jsonl"), expect_type=KeyframeRecord
        )
        keyframes = {k.keyframe_token: k for k in keyframe_records}
        mask_index = read_mask_index(os.path.join(stage4_dir, "scenes", scene_name, "masks.jsonl"))

        rows_by_keyframe: dict[str, list[dict]] = {}
        for row in rows:
            rows_by_keyframe.setdefault(row["keyframe_token"], []).append(row)
        keyframe_tokens_sorted = sorted(rows_by_keyframe, key=lambda tok: keyframes[tok].t_ns if tok in keyframes else 0)

        scene_rows, scene_totals = track_scene(
            scene_name, keyframe_tokens_sorted, keyframes, rows_by_keyframe, mask_index,
            substrate, paths.dataroot, stage5_dir, adapter, cfg,
        )
        # Strip the internal member-point cache before writing: it exists only
        # to carry ICP's source cluster from one keyframe to the next in
        # memory and is never a Stage 7 output field.
        for row in scene_rows:
            if isinstance(row.get("box"), dict):
                row["box"].pop("_member_points", None)

        write_jsonl_atomic(os.path.join(out_dir, "scenes", scene_name, "boxes.jsonl"), scene_rows)

        summary = {
            "scene": scene_name,
            **scene_totals,
            "degraded": scene_totals["n_detections"] > 0 and scene_totals["n_tracks_confirmed"] == 0,
        }
        per_scene.append(summary)
        degraded = degraded or summary["degraded"]
        for key, value in scene_totals.items():
            totals[key] = totals.get(key, 0) + value
        print(
            f"  {scene_name}  {scene_totals['n_keyframes']:>3} kf  {scene_totals['n_detections']:>5} det  "
            f"{scene_totals['n_matched']:>5} matched  {scene_totals['n_births']:>4} births  "
            f"{scene_totals['n_deaths']:>4} deaths  {scene_totals['n_tracks_stable']:>4} stable(>=3 hits)"
            + ("  DEGRADED" if summary["degraded"] else "")
        )

    if adapter is not None:
        adapter.unload()

    manifest = {
        "spec": STAGE_SPEC,
        "stage": STAGE,
        "seed": cfg.global_seed,
        "config": cfg.as_dict(),
        "upstream": {
            "metadata_fingerprint": stage6_manifest["upstream"]["metadata_fingerprint"],
            "fingerprint_spec": stage6_manifest["upstream"]["fingerprint_spec"],
            "stage6_spec": stage6_manifest["spec"],
            # C16: a run built on accepted degradation says so in its provenance.
            "stage1_degraded": stage1_marker.degraded,
            "stage1_degraded_causes": list(stage1_marker.causes),
            "stage4_degraded": stage4_marker.degraded,
            "stage4_degraded_causes": list(stage4_marker.causes),
            "stage5_degraded": stage5_marker.degraded,
            "stage5_degraded_causes": list(stage5_marker.causes),
            "stage6_degraded": stage6_marker.degraded,
            "stage6_degraded_causes": list(stage6_marker.causes),
            "accepted_degraded_upstream": cfg.accept_degraded_upstream,
        },
        "paths": paths.as_dict(),
        "frame": EGO,
        "association": {
            "regime": "predict_then_match_2hz",
            "regime_source": "S11 decision 5, pilot_plan.md S5.8",
            "iou_mode": cfg.iou_mode,
            "iou_mode_note": (
                "gates and scores on ious_bev by default -- comprehensive.md's literal '3D IoU' "
                "is still computed and recorded per pair as ious_3d, not used to gate; see config "
                "provenance for the height-bias reasoning"
            ),
            "iou_gate": cfg.iou_gate,
            "iou_gate_provenance": "derived_for_2hz",
            "matcher": cfg.matcher,
            "matcher_implementation": "scipy.optimize.linear_sum_assignment",
            "score": "iou * cosine when appearance is trusted on the pair, iou alone otherwise",
        },
        "appearance": {
            "enabled": appearance_enabled,
            "min_crop_px": cfg.min_crop_px,
            "unavailable_reason": appearance_unavailable_reason,
        },
        "icp": {
            "frame": cfg.icp_registration_frame,
            "frame_meaning": (
                "global_absolute: both clusters hopped through nuscenes_global via each "
                "keyframe's own ego_pose before registration; velocity is absolute and "
                "nuScenes-mAVE-comparable. ego_relative: no ego-motion correction; velocity is "
                "contaminated by the ego vehicle's own motion between the two keyframes"
            ),
            "kalman_fallback_trigger": f"num_lidar_pts < {cfg.kalman_fallback_min_points} on the current cluster",
            "kalman_fallback_trigger_source": "comprehensive.md S7.3.6",
            "require_adjacent_keyframe": cfg.icp_require_adjacent_keyframe,
        },
        "kalman": {
            "state": list(KF_STATE_NAMES),
            "process_model": "constant_velocity_constant_yaw",
            "no_yaw_rate_term": True,
        },
        "yaw_consistency": {
            "rule": "pick argmin_{c in {yaw, yaw+pi}} |c - reference|",
            "reference": "predicted velocity heading when speed >= yaw_consistency_min_speed_mps, "
            "else the track's previous smoothed yaw; undetermined on a track's first detection",
            "rule_source": "S7.3.7, restored per S4 / S11 decision 6",
        },
        "known_gaps": [
            "no forward-backward smoothing over the whole scene (S4: waived, offboard "
            "refinement, not needed to prove association plumbing)",
            "reid embeddings use the mask's tight box only, not mask-pooled patch tokens",
            "ICP is skipped across a coasted (missed-frame) gap by default; those tracks fall "
            "back to the Kalman filter's own velocity estimate for that update",
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
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--stage1-dir", default=None, help="default <work_root>/stage1_ingestion")
    parser.add_argument("--stage4-dir", default=None, help="default <work_root>/stage4_masks")
    parser.add_argument("--stage5-dir", default=None, help="default <work_root>/stage5_lift")
    parser.add_argument("--stage6-dir", default=None, help="default <work_root>/stage6_cluster")
    parser.add_argument("--out-dir", default=None, help="default <work_root>/stage7_track")
    parser.add_argument("--scenes", nargs="*", default=None, help="subset of Stage 6 scene names")
    parser.add_argument("--icp-frame", default="global_absolute", choices=ICP_FRAMES)
    parser.add_argument("--iou-mode", default="bev", choices=IOU_MODES)
    parser.add_argument("--require-appearance", action="store_true")
    parser.add_argument(
        "--reid-model-id",
        default=None,
        help="reid_embedding checkpoint; default facebook/dinov3-vits16-pretrain-lvd1689m. The provider name is "
             "derived from it (dinov2-* -> dinov2_reid, dinov3-* -> dinov3_reid), so --reid-revision "
             "must be the hub sha OF THIS id",
    )
    parser.add_argument("--reid-revision", default=None, help="hub commit sha for the reid checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="override the global seed (recorded)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--accept-degraded-upstream",
        action="store_true",
        help="consume a DEGRADED (complete, quality-flagged) Stage 1, 4, 5, or 6 output; recorded (C16)",
    )
    args = parser.parse_args(argv)

    try:
        paths = load_paths(args.paths)
    except PathValidationError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    stage1_dir = args.stage1_dir or os.path.join(paths.work_root, "stage1_ingestion")
    stage4_dir = args.stage4_dir or os.path.join(paths.work_root, "stage4_masks")
    stage5_dir = args.stage5_dir or os.path.join(paths.work_root, "stage5_lift")
    stage6_dir = args.stage6_dir or os.path.join(paths.work_root, "stage6_cluster")
    out_dir = args.out_dir or os.path.join(paths.work_root, STAGE)
    assert_dataroot_read_only(paths, out_dir)

    cfg = TrackConfig(
        icp_registration_frame=args.icp_frame,
        iou_mode=args.iou_mode,
        require_appearance=args.require_appearance,
        reid_revision=args.reid_revision,
        **({"reid_model_id": args.reid_model_id} if args.reid_model_id else {}),
        device=args.device,
        accept_degraded_upstream=args.accept_degraded_upstream,
        **({"global_seed": args.seed} if args.seed is not None else {}),
    )

    appearance_enabled = True
    appearance_unavailable_reason = None
    try:
        stage6_manifest, stage1_marker, stage4_marker, stage5_marker, stage6_marker = load_upstream(
            paths, stage1_dir, stage4_dir, stage5_dir, stage6_dir,
            accept_degraded=cfg.accept_degraded_upstream,
        )
        try:
            manifest, code = run(
                paths, stage6_manifest, stage1_marker, stage4_marker, stage5_marker, stage6_marker,
                cfg, stage1_dir, stage4_dir, stage5_dir, stage6_dir, out_dir,
                args.scenes, appearance_enabled, appearance_unavailable_reason,
            )
        except ModelUnavailable as exc:
            if cfg.require_appearance:
                raise
            print(f"appearance model unavailable, continuing IoU-only: {exc}", file=sys.stderr)
            manifest, code = run(
                paths, stage6_manifest, stage1_marker, stage4_marker, stage5_marker, stage6_marker,
                cfg, stage1_dir, stage4_dir, stage5_dir, stage6_dir, out_dir,
                args.scenes, False, str(exc),
            )
    except UpstreamRefusal as exc:
        print(f"REFUSING TO START: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except ModelUnavailable as exc:
        print(f"REFUSING TO START: model unavailable: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (TrackContractError, RoleContractError, PathValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)
    # Three-state marker (S1.9, C16): clean / degraded-with-causes / absent.
    write_marker(
        out_dir,
        manifest["upstream"]["metadata_fingerprint"],
        degraded=code == EXIT_DEGRADED,
        causes=[
            f"{s['scene']}: {s['n_detections']} detection(s), 0 confirmed tracks"
            for s in manifest["scenes"]
            if s["degraded"]
        ],
    )

    t = manifest["totals"]
    print(f"keyframes            : {t.get('n_keyframes', 0)}")
    print(f"detections           : {t.get('n_detections', 0)}  ({t.get('n_matched', 0)} matched)")
    print(f"births / deaths      : {t.get('n_births', 0)} / {t.get('n_deaths', 0)}")
    print(f"tracks confirmed     : {t.get('n_tracks_confirmed', 0)} / {t.get('n_tracks_total', 0)}"
          f"  ({t.get('n_tracks_stable', 0)} with >= {cfg.stable_track_hits} hits)")
    print(f"ICP attempted        : {t.get('n_icp_attempted', 0)}  succeeded {t.get('n_icp_succeeded', 0)}"
          f"  (frame={cfg.icp_registration_frame})")
    print(f"Kalman fallback      : {t.get('n_kalman_fallback', 0)}  (< {cfg.kalman_fallback_min_points} pts)")
    print(f"yaw flips applied    : {t.get('n_yaw_flips', 0)}")
    print(f"appearance           : enabled={manifest['appearance']['enabled']}  "
          f"trusted pairs={t.get('n_appearance_trusted_pairs', 0)}")
    print(f"wrote {out_dir}")
    return code


register("dinov2_reid", REID_EMBEDDING, Dinov2ReidAdapter)
register("dinov3_reid", REID_EMBEDDING, Dinov2ReidAdapter)


if __name__ == "__main__":
    raise SystemExit(main())
