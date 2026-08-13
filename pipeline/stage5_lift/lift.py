#!/usr/bin/env python3
"""Stage 5 — 2D->3D lift (§5.6, Phase 7).

The four-hop chain of §1.3 applied to **ground-filtered single-sweep** points
(§1.4) in **ego frame** (§1.1), indexing Stage 4's masks at **original
resolution** (§1.5), with the **six per-camera frusta unioned** for R2 coverage
and the **multi-camera contest rule** of §1.5 rule 5.

What this stage does and does not move. The cloud it reads is already in the ego
frame at the LiDAR anchor time; Stage 1 applied `T_ego_lidar` exactly once and
nothing here re-applies it (§1.1 rule 3). The lift does not transform the cloud
at all: it decides, for each point, **which mask instance owns it**. Every
painted point keeps its Stage 1 coordinates, in `frame: "ego"` at `t_ns`. The
projection is scaffolding for a labelling decision, not a change of
representation, and saying so is the difference between Stage 6 clustering the
points it thinks it has and clustering something silently re-based.

The chain itself is not reimplemented here. `conventions.project_lidar_to_image`
is the one implementation (§1.3); this module calls it once per camera per
keyframe and owns only the guards, the union, and the contest.

**The four guards** (§5.6), each counted separately in every record:

  1. `z <= 0` — culled BEFORE the perspective divide. A point behind the camera
     divided by its own negative z lands at a valid-looking mirrored pixel,
     inside the image, and paints itself with whatever mask is there. This is
     the guard whose absence produces a plausible, fully populated, wrong
     output.
  2. Near-zero depth — `0 < z <= min_depth_m`. Here the divide does not lie, it
     explodes: a few points acquire pixel coordinates of order 1e6 and land
     out-of-bounds anyway, so the guard looks redundant until a point sits at
     exactly z = 0 and produces inf/nan that propagates into the mask index.
  3. Out of bounds — in front of the camera, outside `[0, W) x [0, H)`. Not an
     error: it is most of the cloud, and its count is the frustum's shape.
  4. Deterministic overlap resolution — below.

Guards 1 and 2 are one comparison inside the chain, which is why they are
counted here by running the chain at an epsilon depth and applying the real
`min_depth_m` in stage code: two numbers, one implementation, no second copy of
the transform.

**The contest, and the reading it commits to.** §1.5 rule 5: a 3D point visible
in two overlapping cameras takes its class from the camera whose principal axis
is closest to the point's bearing, ties broken by a fixed camera-priority list.
"Closest bearing" is computed as the cosine between the point's camera-frame ray
and that camera's optical axis — identical to the ego-frame formulation and
immune to the fact that the two cameras have two different ego poses (§1.2), so
"the point's bearing" is not one vector but two.

The rule as written does not say whether an *unlabelled* camera may win. Two
readings:

  - **`labelled_cameras`** (default): the contest runs over the cameras that
    actually put the point inside a mask. A point painted by CAM_BACK_LEFT is
    labelled even if CAM_BACK saw it better and proposed nothing there.
  - **`all_visible_cameras`** (strict): the nearest-axis camera wins even when
    it has no mask at that pixel, so the point goes unlabelled.

The strict reading makes labelling depend on whether Stage 3 fired in the other
camera, which is a detection outcome leaking into a geometry rule. The default
is the first, and the number of points the two readings disagree on is measured
and recorded per keyframe rather than argued about.

**What this stage deliberately does not do.** No occlusion reasoning. A mask is
a 2D region; the far wall behind a car projects into the car's mask and is
painted as the car. Those are the reprojection ghosts §1.6 assigns to Stage 6's
"keep largest cluster" filter, and pre-empting them here with a depth heuristic
would move a documented filter into an undocumented one. The per-point depth is
recorded so Stage 6 has it.

    python3 -m pipeline.stage5_lift.lift [--paths configs/paths.yaml]

Exit codes:
    0  every keyframe lifted under contract
    1  ran, but at least one keyframe painted nothing
    2  upstream contract broken; nothing was written
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.common.conventions import (  # noqa: E402
    CAMERA,
    EGO,
    MIN_DEPTH_M,
    NUSCENES_GLOBAL,
    Transform,
    project_lidar_to_image,
)
from pipeline.common.eval_region import R1_DEFAULT, R2_DEFAULT, RegionSpec, in_region  # noqa: E402
from pipeline.common.paths import (  # noqa: E402
    PathValidationError,
    Paths,
    assert_dataroot_read_only,
    load_paths,
    metadata_fingerprint,
)
from pipeline.common.schemas import (  # noqa: E402
    IMAGE_HEIGHT_PX,
    IMAGE_WIDTH_PX,
    RING_CAMERAS,
    KeyframeRecord,
    read_records,
)
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

STAGE = "stage5_lift"
STAGE_SPEC = "dhakascenes-pilot/stage5_lift/v1"

EXIT_OK = 0
EXIT_DEGRADED = 1  # ran, but at least one keyframe with instances painted no points
EXIT_REFUSED = 2  # upstream contract broken; nothing was written

# The `z <= 0` test, run as `z <= BEHIND_CAMERA_EPS_M`. The chain culls behind-
# camera and near-zero-depth points with a single comparison; running it at this
# epsilon and applying `min_depth_m` afterwards separates the two counts without
# a second copy of the transform. Points in (0, eps] are counted as behind, which
# is a band of width 1e-12 m.
BEHIND_CAMERA_EPS_M = 1e-12

WITHIN_CAMERA_RULES: tuple[str, ...] = ("smallest_mask", "highest_score")
CONTEST_SCOPES: tuple[str, ...] = ("labelled_cameras", "all_visible_cameras")


class LiftContractError(RuntimeError):
    """A lift input or intermediate violated the Stage 5 contract."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiftConfig:
    """Stage 5 tunables. None of these may appear as a literal in the code below."""

    # --- which cloud (§1.4) ---
    cloud_kind: str = "single_sweep"

    # --- the four guards (§5.6) ---
    min_depth_m: float = MIN_DEPTH_M
    behind_camera_eps_m: float = BEHIND_CAMERA_EPS_M
    image_width_px: int = IMAGE_WIDTH_PX
    image_height_px: int = IMAGE_HEIGHT_PX

    # --- deterministic overlap resolution (§1.5 rule 5) ---
    contest_scope: str = "labelled_cameras"
    within_camera_rule: str = "smallest_mask"
    camera_priority: tuple[str, ...] = RING_CAMERAS

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
                "single_sweep: comprehensive.md §7.3.5 and §1.4. The accumulated cloud is "
                "ego-motion compensated only, so dynamic objects smear across the window and a "
                "single-frame mask paints the smear — box length inflates along the direction of "
                "travel and reads as 'fast objects are harder'"
            ),
            "min_depth_m": "conventions.MIN_DEPTH_M; the near-zero-depth guard (§1.3, §5.6)",
            "behind_camera_eps_m": (
                "the z <= 0 test, run at 1e-12 so the behind-camera and near-zero-depth counts "
                "separate without a second implementation of the projection chain"
            ),
            "contest_scope": (
                "§1.5 rule 5 does not say whether an unlabelled camera may win the contest. "
                "labelled_cameras is the committed reading; the disagreement with the strict "
                "reading is counted per keyframe, not argued"
            ),
            "within_camera_rule": (
                "undefined by the spec, and reachable: instance masks overlap within one image. "
                "smallest_mask picks the innermost instance, which is the pedestrian standing in "
                "front of the bus rather than the bus"
            ),
            "camera_priority": "§1.5 rule 5's fixed list; the final tie-break, and nothing else",
            "min_points_per_instance": (
                "comprehensive.md §7.3.9 / §6.3, counted on the SINGLE-SWEEP cloud pre-inflation "
                "(§1.4). Recorded here, gated in Stage 9 — Stage 5 drops nothing"
            ),
            "accept_degraded_upstream": "C16 — consuming a DEGRADED (complete, quality-flagged) "
            "Stage 1 or Stage 4 output is an explicit recorded decision, never a default",
            "global_seed": "§1.9, one global seed, recorded",
        }
    )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.cloud_kind != "single_sweep":
            errors.append(
                f"cloud_kind={self.cloud_kind!r}: §1.4 and comprehensive.md §7.3.5 lift the "
                "single-sweep cloud; lifting the accumulation is the rev 1 defect"
            )
        if self.contest_scope not in CONTEST_SCOPES:
            errors.append(f"contest_scope={self.contest_scope!r} is not one of {CONTEST_SCOPES}")
        if self.within_camera_rule not in WITHIN_CAMERA_RULES:
            errors.append(
                f"within_camera_rule={self.within_camera_rule!r} is not one of {WITHIN_CAMERA_RULES}"
            )
        if not 0.0 < self.behind_camera_eps_m < self.min_depth_m:
            errors.append(
                f"behind_camera_eps_m={self.behind_camera_eps_m!r} must be positive and below "
                f"min_depth_m={self.min_depth_m!r}"
            )
        if set(self.camera_priority) != set(RING_CAMERAS):
            errors.append(
                f"camera_priority must be a permutation of {RING_CAMERAS}, got {self.camera_priority}"
            )
        return errors

    def as_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["camera_priority"] = list(self.camera_priority)
        return payload


def region_for(coverage_config: str) -> RegionSpec:
    """E for a keyframe's declared coverage, from the two configured defaults."""
    if coverage_config == "R1":
        return R1_DEFAULT
    if coverage_config == "R2":
        return R2_DEFAULT
    raise LiftContractError(f"coverage_config={coverage_config!r} is not R1 or R2")


# ---------------------------------------------------------------------------
# Instances — Stage 4's surviving masks, given keyframe-stable ids
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Instance:
    """One mask that survived Stage 4's cross-camera IoA-NMS."""

    instance_id: int
    channel: str
    proposal_index: int
    class_name: str
    score: float
    n_mask_px: int


def build_instances(candidates: Sequence[dict], cfg: LiftConfig) -> list[Instance]:
    """Assign keyframe-stable instance ids to the kept masks, deterministically.

    Order is the fixed camera-priority list, then the proposal index inside that
    camera — never the order Stage 4 happened to emit. Instance ids are the join
    key Stage 6 clusters by and Stage 7 tracks through; an id that permutes
    between two runs of the same input silently permutes every downstream box.
    """
    priority = {name: i for i, name in enumerate(cfg.camera_priority)}
    kept = [c for c in candidates if c.get("kept")]
    kept.sort(key=lambda c: (priority.get(c["channel"], len(priority)), c["channel"], c["proposal_index"]))
    return [
        Instance(
            instance_id=i,
            channel=c["channel"],
            proposal_index=int(c["proposal_index"]),
            class_name=c["class_name"],
            score=float(c["score"]),
            n_mask_px=int(c["n_mask_px"]),
        )
        for i, c in enumerate(kept)
    ]


def within_camera_order(instances: Sequence[Instance], rule: str) -> list[Instance]:
    """The order in which overlapping masks in ONE image claim a pixel.

    First claim wins, so this order IS the within-camera occlusion policy.
    `smallest_mask` claims from the inside out: where a pedestrian mask sits
    inside a bus mask, the pedestrian's points are the pedestrian's. Under
    `highest_score` the bus takes them, and the pedestrian arrives at Stage 6 as
    a handful of survivors or nothing at all.
    """
    if rule == "smallest_mask":
        key = lambda inst: (inst.n_mask_px, -inst.score, inst.proposal_index)  # noqa: E731
    elif rule == "highest_score":
        key = lambda inst: (-inst.score, inst.n_mask_px, inst.proposal_index)  # noqa: E731
    else:
        raise LiftContractError(f"within_camera_rule={rule!r} is not one of {WITHIN_CAMERA_RULES}")
    return sorted(instances, key=key)


# ---------------------------------------------------------------------------
# Mask access
# ---------------------------------------------------------------------------


class MaskFile:
    """Stage 4's per-keyframe npz, read one mask at a time.

    Masks are bit-packed along the last axis (§5.5), so unpacking needs the
    original width — which is stored IN the file rather than assumed by the
    reader, and is asserted against the 1600 x 900 contract here (§1.5 rule 4).
    A whole channel unpacked at once is `n_masks x 1.4 MB`; one mask at a time
    is 1.4 MB, and Stage 5 never needs two.
    """

    def __init__(self, path: str) -> None:
        if not os.path.isfile(path):
            raise UpstreamRefusal(f"{path} not found; run `python3 -m pipeline.stage4_masks.masks` first")
        self._path = path
        self._npz = np.load(path)
        self.width_px = int(self._npz["__width_px__"][0])
        self.height_px = int(self._npz["__height_px__"][0])
        self.bit_packed = bool(int(self._npz["__bit_packed__"][0]))
        if (self.width_px, self.height_px) != (IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX):
            raise LiftContractError(
                f"{path}: masks are {self.width_px}x{self.height_px}, not "
                f"{IMAGE_WIDTH_PX}x{IMAGE_HEIGHT_PX}; Stage 5 indexes them with pixel coordinates "
                "it does not own otherwise (§1.5 rule 4)"
            )

    def channels(self) -> list[str]:
        return sorted(k for k in self._npz.files if not k.startswith("__"))

    def n_masks(self, channel: str) -> int:
        return int(self._npz[channel].shape[0])

    def mask(self, channel: str, index: int) -> np.ndarray:
        """One (H, W) boolean mask at original resolution."""
        if channel not in self._npz.files:
            raise LiftContractError(f"{self._path}: no masks for channel {channel!r}")
        stack = self._npz[channel]
        if not 0 <= index < stack.shape[0]:
            raise LiftContractError(
                f"{self._path}: {channel} mask index {index} out of range for {stack.shape[0]} masks; "
                "Stage 4 indexes masks by proposal position and a shifted index relabels every object"
            )
        row = stack[index]
        out = np.unpackbits(row, axis=-1, count=self.width_px).astype(bool) if self.bit_packed else row.astype(bool)
        if out.shape != (self.height_px, self.width_px):
            raise LiftContractError(
                f"{self._path}: {channel}[{index}] unpacked to {out.shape}, expected "
                f"{(self.height_px, self.width_px)}"
            )
        return out

    def close(self) -> None:
        self._npz.close()


# ---------------------------------------------------------------------------
# One camera: the chain, the guards, and the within-camera claim
# ---------------------------------------------------------------------------


@dataclass
class CameraLift:
    """What one camera contributes to one keyframe's lift."""

    channel: str
    point_index: np.ndarray  # (M,) int64 into the single-sweep cloud, in front + in image
    uv_px: np.ndarray  # (M, 2) float64, absolute pixels at 1600x900
    depth_m: np.ndarray  # (M,) float64, camera-frame z
    axis_cos: np.ndarray  # (M,) float64, cos(angle between the ray and the optical axis)
    instance_id: np.ndarray  # (M,) int64, -1 where the point landed in no mask
    guards: dict
    ego_translation_delta_m: float


def principal_axis_cos(points_camera_m: np.ndarray) -> np.ndarray:
    """cos of the angle between each point's ray and the camera's optical axis.

    The optical axis is +z in camera frame, so this is `z / |p|` and needs no
    extrinsic at all. That is the point: §1.5 rule 5's "principal axis closest to
    the point's bearing" is an angle between two directions, and computing it in
    camera frame sidesteps the question of WHICH ego frame the bearing is
    measured in — the six cameras have six different ego poses within one
    keyframe (§1.2), so the ego-frame formulation has six answers and this one
    has the same answer as all of them.
    """
    norm = np.linalg.norm(points_camera_m, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = np.where(norm > 0.0, points_camera_m[:, 2] / norm, -1.0)
    return np.clip(cos, -1.0, 1.0)


def lift_camera(
    points_ego_m: np.ndarray,
    ego_pose_at_lidar: Transform,
    observation,
    ego_pose_table: dict,
    calibrated_table: dict,
    masks: MaskFile,
    instances: Sequence[Instance],
    cfg: LiftConfig,
) -> CameraLift:
    """Project the cloud into one camera, apply the guards, claim pixels."""
    channel = observation.channel
    ego_pose_at_camera = Transform.from_nuscenes(
        ego_pose_table[observation.ego_pose_token], source_frame=EGO, parent_frame=NUSCENES_GLOBAL
    )
    calibrated = calibrated_table[observation.calibrated_sensor_token]
    extrinsic = Transform.from_nuscenes(calibrated, source_frame=CAMERA, parent_frame=EGO)
    intrinsic = np.asarray(calibrated["camera_intrinsic"], dtype=np.float64)
    if intrinsic.shape != (3, 3):
        raise LiftContractError(
            f"{channel}: camera_intrinsic is {intrinsic.shape}, not (3, 3); a camera with no "
            "intrinsic cannot be projected into and must not be silently skipped"
        )
    if (observation.width_px, observation.height_px) != (cfg.image_width_px, cfg.image_height_px):
        raise LiftContractError(
            f"{channel}: keyframe declares {observation.width_px}x{observation.height_px}, Stage 5 "
            f"indexes masks at {cfg.image_width_px}x{cfg.image_height_px} (§1.5 rule 1)"
        )

    # Guards 1 and 2. The chain culls at `behind_camera_eps_m`, so its own cull
    # count is the `z <= 0` guard; the real near-zero-depth guard is applied to
    # the survivors here. One chain, two counts (§1.3).
    projection = project_lidar_to_image(
        points_ego_m,
        ego_pose_at_lidar,
        ego_pose_at_camera,
        extrinsic,
        intrinsic,
        (cfg.image_width_px, cfg.image_height_px),
        min_depth_m=cfg.behind_camera_eps_m,
    )
    deep_enough = projection.depth_m > cfg.min_depth_m

    # Guard 3. Out of bounds is not an error: it is most of the cloud, and its
    # count is the shape of this camera's frustum.
    visible = deep_enough & projection.in_image
    point_index = projection.source_index[visible]
    uv_px = projection.uv_px[visible]
    depth_m = projection.depth_m[visible]
    axis_cos = principal_axis_cos(projection.points_camera_m[visible])

    # The within-camera claim. `u`/`v` are safe to floor without clipping: the
    # in-image test already bounded them to [0, W) x [0, H).
    u_idx = np.floor(uv_px[:, 0]).astype(np.int64)
    v_idx = np.floor(uv_px[:, 1]).astype(np.int64)
    instance_id = np.full(point_index.shape[0], -1, dtype=np.int64)
    n_claimed_by = []
    for inst in within_camera_order([i for i in instances if i.channel == channel], cfg.within_camera_rule):
        if inst.proposal_index >= masks.n_masks(channel):
            raise LiftContractError(
                f"{channel}: instance {inst.instance_id} names proposal {inst.proposal_index} but the "
                f"mask file holds {masks.n_masks(channel)} masks for this channel"
            )
        unclaimed = instance_id < 0
        if not unclaimed.any():
            break
        hit = np.zeros(instance_id.shape[0], dtype=bool)
        mask = masks.mask(channel, inst.proposal_index)
        hit[unclaimed] = mask[v_idx[unclaimed], u_idx[unclaimed]]
        instance_id[hit] = inst.instance_id
        n_claimed_by.append({"instance_id": inst.instance_id, "n_points": int(np.count_nonzero(hit))})

    n_input = int(points_ego_m.shape[0])
    guards = {
        "n_input": n_input,
        # Guard 1: behind the camera, culled BEFORE the divide.
        "n_culled_behind_camera": int(projection.n_culled_behind_camera),
        # Guard 2: in front, but within min_depth_m of the optical centre.
        "n_culled_near_zero_depth": int(np.count_nonzero(~deep_enough)),
        # Guard 3: in front and deep enough, but off the sensor.
        "n_culled_out_of_bounds": int(np.count_nonzero(deep_enough & ~projection.in_image)),
        "n_in_frustum": int(point_index.shape[0]),
        "n_painted": int(np.count_nonzero(instance_id >= 0)),
        "min_depth_m": cfg.min_depth_m,
        "behind_camera_eps_m": cfg.behind_camera_eps_m,
        "claims": n_claimed_by,
    }
    if guards["n_culled_behind_camera"] + guards["n_culled_near_zero_depth"] + guards[
        "n_culled_out_of_bounds"
    ] + guards["n_in_frustum"] != n_input:
        raise LiftContractError(f"{channel}: guard counts do not partition {n_input} input points")

    return CameraLift(
        channel=channel,
        point_index=point_index,
        uv_px=uv_px,
        depth_m=depth_m,
        axis_cos=axis_cos,
        instance_id=instance_id,
        guards=guards,
        ego_translation_delta_m=projection.ego_translation_delta_m,
    )


# ---------------------------------------------------------------------------
# The frustum union and the multi-camera contest (§1.5 rule 5, §5.6)
# ---------------------------------------------------------------------------


@dataclass
class Contest:
    """The resolved assignment over the whole cloud, plus what it cost."""

    point_index: np.ndarray  # (P,) int64, painted points only
    instance_id: np.ndarray  # (P,) int64
    channel_index: np.ndarray  # (P,) int64 into `channels`
    uv_px: np.ndarray  # (P, 2) float64, from the WINNING camera
    depth_m: np.ndarray  # (P,) float64, from the winning camera
    axis_cos: np.ndarray  # (P,) float64
    n_cameras_labelled: np.ndarray  # (P,) int64, how many cameras had a mask on this point
    visible_point_index: np.ndarray  # (V,) int64, the frustum UNION
    visible_n_cameras: np.ndarray  # (V,) int64
    channels: tuple[str, ...]
    ledger: dict


def resolve_contest(lifts: Sequence[CameraLift], n_points: int, cfg: LiftConfig) -> Contest:
    """Union the six frusta, then give each point exactly one owner.

    Cameras are visited in the fixed priority order and the comparison is
    strictly greater-than, so an exact cosine tie is decided by that list and by
    nothing else — which is the whole content of §1.5 rule 5's tie-break. Rev 1
    left it undefined, and undefined here means the label depends on dictionary
    iteration order: clean output, plausible output, different on every run.
    """
    order = {name: i for i, name in enumerate(cfg.camera_priority)}
    ordered = sorted(lifts, key=lambda lift: (order.get(lift.channel, len(order)), lift.channel))
    channels = tuple(lift.channel for lift in ordered)

    best_cos_labelled = np.full(n_points, -np.inf)
    winner_camera_labelled = np.full(n_points, -1, dtype=np.int64)
    winner_instance = np.full(n_points, -1, dtype=np.int64)
    winner_uv = np.zeros((n_points, 2))
    winner_depth = np.zeros(n_points)
    best_cos_visible = np.full(n_points, -np.inf)
    winner_camera_visible = np.full(n_points, -1, dtype=np.int64)
    n_cameras_visible = np.zeros(n_points, dtype=np.int64)
    n_cameras_labelled = np.zeros(n_points, dtype=np.int64)

    for camera_index, lift in enumerate(ordered):
        idx = lift.point_index
        n_cameras_visible[idx] += 1

        # The union contest: over every camera that sees the point at all.
        better = lift.axis_cos > best_cos_visible[idx]
        selected = idx[better]
        best_cos_visible[selected] = lift.axis_cos[better]
        winner_camera_visible[selected] = camera_index

        # The labelled contest: over the cameras that put it inside a mask.
        labelled = lift.instance_id >= 0
        idx_l = idx[labelled]
        n_cameras_labelled[idx_l] += 1
        cos_l = lift.axis_cos[labelled]
        better_l = cos_l > best_cos_labelled[idx_l]
        selected_l = idx_l[better_l]
        best_cos_labelled[selected_l] = cos_l[better_l]
        winner_camera_labelled[selected_l] = camera_index
        winner_instance[selected_l] = lift.instance_id[labelled][better_l]
        winner_uv[selected_l] = lift.uv_px[labelled][better_l]
        winner_depth[selected_l] = lift.depth_m[labelled][better_l]

    painted = winner_instance >= 0
    # The two readings of §1.5 rule 5 differ exactly where the nearest-axis
    # camera overall is NOT the nearest-axis camera among those that labelled the
    # point: under the strict reading those points go unlabelled. Measured, not
    # argued (see the module docstring).
    reading_differs = painted & (winner_camera_visible != winner_camera_labelled)
    if cfg.contest_scope == "all_visible_cameras":
        painted = painted & ~reading_differs

    point_index = np.nonzero(painted)[0]
    visible_point_index = np.nonzero(n_cameras_visible > 0)[0]

    ledger = {
        "rule": "principal_axis_cosine",
        "rule_source": "§1.5 rule 5",
        "contest_scope": cfg.contest_scope,
        "tie_break": list(cfg.camera_priority),
        "within_camera_rule": cfg.within_camera_rule,
        "n_points_cloud": int(n_points),
        "n_points_visible_union": int(visible_point_index.shape[0]),
        "n_points_multi_camera": int(np.count_nonzero(n_cameras_visible > 1)),
        "n_points_painted": int(point_index.shape[0]),
        "n_points_contested": int(np.count_nonzero(n_cameras_labelled > 1)),
        "n_points_reading_differs": int(np.count_nonzero(reading_differs)),
        "union_fraction": round(float(visible_point_index.shape[0]) / max(1, n_points), 5),
        "painted_fraction_of_union": round(
            float(point_index.shape[0]) / max(1, int(visible_point_index.shape[0])), 5
        ),
    }
    return Contest(
        point_index=point_index,
        instance_id=winner_instance[point_index],
        channel_index=winner_camera_labelled[point_index],
        uv_px=winner_uv[point_index],
        depth_m=winner_depth[point_index],
        axis_cos=best_cos_labelled[point_index],
        n_cameras_labelled=n_cameras_labelled[point_index],
        visible_point_index=visible_point_index,
        visible_n_cameras=n_cameras_visible[visible_point_index],
        channels=channels,
        ledger=ledger,
    )


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def write_points_npz(path: str, contest: Contest, cfg: LiftConfig) -> str:
    """One npz per keyframe: the painted points, plus the frustum union.

    `point_index` indexes Stage 1's single-sweep cloud directly. The coordinates
    are NOT copied: Stage 6 reads the same `.pcd.bin` and takes the rows named
    here, so there is exactly one copy of every point's position in the run and
    no way for a rounded duplicate to disagree with it.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    payload = {
        "point_index": contest.point_index.astype(np.int32),
        "instance_id": contest.instance_id.astype(np.int32),
        "channel_index": contest.channel_index.astype(np.int8),
        "uv_px": contest.uv_px.astype(np.float32),
        "depth_m": contest.depth_m.astype(np.float32),
        "axis_cos": contest.axis_cos.astype(np.float32),
        "n_cameras_labelled": contest.n_cameras_labelled.astype(np.int8),
        "visible_point_index": contest.visible_point_index.astype(np.int32),
        "visible_n_cameras": contest.visible_n_cameras.astype(np.int8),
        "__channels__": np.asarray(contest.channels),
        "__frame__": np.asarray([EGO]),
        "__cloud_kind__": np.asarray([cfg.cloud_kind]),
    }
    # np.savez_compressed APPENDS ".npz" to any name that does not already end in
    # it, so the temp name must carry the suffix or the rename below looks for a
    # file that was never created.
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    if not os.path.isfile(tmp):
        raise RuntimeError(f"{tmp}: np.savez_compressed did not write the name it was given")
    os.replace(tmp, path)
    return path


def instance_rows(
    instances: Sequence[Instance],
    contest: Contest,
    points_ego_m: np.ndarray,
    region: RegionSpec,
    cfg: LiftConfig,
) -> list[dict]:
    """Per-instance summary: the point set Stage 6 will cluster.

    `n_points` is counted on the SINGLE-SWEEP cloud, pre-inflation, which is what
    comprehensive.md §7.3.9's ">= 5 LiDAR returns" gate is defined against
    (§1.4). Counting it on the accumulation loosens the gate by roughly the
    accumulation factor; the gate still fires, still tiers boxes, and no longer
    means what the spec says.
    """
    rows: list[dict] = []
    for inst in instances:
        selected = contest.instance_id == inst.instance_id
        n_points = int(np.count_nonzero(selected))
        row = {
            "instance_id": inst.instance_id,
            "channel": inst.channel,
            "proposal_index": inst.proposal_index,
            "class_name": inst.class_name,
            "score": round(inst.score, 5),
            "n_mask_px": inst.n_mask_px,
            "n_points": n_points,
            "n_points_below_gate": bool(n_points < cfg.min_points_per_instance),
            "cloud_kind": cfg.cloud_kind,
            "frame": EGO,
        }
        if n_points:
            xyz = points_ego_m[contest.point_index[selected]]
            row["centroid_ego_m"] = [round(float(v), 4) for v in xyz.mean(axis=0)]
            row["depth_m"] = {
                "min": round(float(contest.depth_m[selected].min()), 3),
                "median": round(float(np.median(contest.depth_m[selected])), 3),
                "max": round(float(contest.depth_m[selected].max()), 3),
            }
            if cfg.record_eval_region:
                inside = in_region(xyz[:, 0], xyz[:, 1], region, frame=EGO)
                row["n_points_in_region"] = int(np.count_nonzero(inside))
                row["coverage_config"] = region.coverage_config
        else:
            # A kept mask that painted nothing is a real, reportable outcome —
            # ground removal stripped the object, or the mask sits on sky — and
            # it is emitted rather than filtered so the count survives to Stage 9.
            row["centroid_ego_m"] = None
            row["depth_m"] = None
            if cfg.record_eval_region:
                row["n_points_in_region"] = 0
                row["coverage_config"] = region.coverage_config
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Upstream
# ---------------------------------------------------------------------------


def load_upstream(paths: Paths, stage1_dir: str, stage4_dir: str, *, accept_degraded: bool = False):
    """Refuse to start unless BOTH upstreams COMPLETED on THIS substrate.

    Stage 5 is the first stage that joins two upstream artifacts — Stage 1's
    clouds and poses, Stage 4's masks — and a join is exactly where two runs
    against two different substrates produce an output that validates. Both go
    through the C16 gate: absent marker refuses unconditionally, degraded marker
    refuses unless `accept_degraded` — one flag for both, because accepting one
    degraded upstream and refusing the other is not a meaningful position when
    the join needs both.
    """
    current = metadata_fingerprint(paths)
    stage1, marker1 = require_upstream(
        stage1_dir,
        stage_name="Stage 1",
        module_hint="pipeline.stage1_ingestion.ingest",
        current_fingerprint=current,
        accept_degraded=accept_degraded,
    )
    stage4, marker4 = require_upstream(
        stage4_dir,
        stage_name="Stage 4",
        module_hint="pipeline.stage4_masks.masks",
        current_fingerprint=current,
        accept_degraded=accept_degraded,
    )
    if stage4["image_size_px"] != [IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX]:
        raise UpstreamRefusal(
            f"Stage 4 masks are {stage4['image_size_px']}, Stage 5 indexes at "
            f"{[IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX]} (§1.5 rule 4)"
        )
    return stage1, marker1, stage4, marker4


def read_mask_index(path: str) -> dict[str, dict]:
    if not os.path.isfile(path):
        raise UpstreamRefusal(f"{path} not found; run `python3 -m pipeline.stage4_masks.masks` first")
    with open(path, "r", encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return {row["keyframe_token"]: row for row in rows}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def lift_keyframe(
    keyframe: KeyframeRecord,
    mask_row: dict,
    stage4_dir: str,
    substrate: Substrate,
    cfg: LiftConfig,
) -> tuple[Contest, list[Instance], list[dict], np.ndarray, dict]:
    """One keyframe: read the cloud, project into six cameras, resolve, summarise."""
    cloud_artifact = keyframe.single_sweep_cloud if cfg.cloud_kind == "single_sweep" else None
    if cloud_artifact is None:
        raise LiftContractError(f"cloud_kind={cfg.cloud_kind!r} has no artifact on the keyframe record")
    if cloud_artifact.frame != EGO:
        raise LiftContractError(
            f"{keyframe.keyframe_token}: cloud is in frame {cloud_artifact.frame!r}, not {EGO!r}; "
            "Stage 1 applies T_ego_lidar exactly once and Stage 5 does not re-apply it (§1.1 rule 3)"
        )
    points_ego_m = read_pcd_bin(cloud_artifact.path)[:, :3].astype(np.float64)
    if points_ego_m.shape[0] != cloud_artifact.n_points:
        raise LiftContractError(
            f"{keyframe.keyframe_token}: cloud holds {points_ego_m.shape[0]} points, the record "
            f"claims {cloud_artifact.n_points}"
        )

    ego_pose_at_lidar = Transform.from_nuscenes(
        substrate.by_token("ego_pose.json")[keyframe.lidar_ego_pose_token],
        source_frame=EGO,
        parent_frame=NUSCENES_GLOBAL,
    )
    ego_pose_table = substrate.by_token("ego_pose.json")
    calibrated_table = substrate.by_token("calibrated_sensor.json")

    instances = build_instances(mask_row.get("candidates", ()), cfg)
    masks = MaskFile(os.path.join(stage4_dir, mask_row["mask_path"]))
    try:
        lifts: list[CameraLift] = []
        for channel in cfg.camera_priority:
            observation = keyframe.cameras.get(channel)
            if observation is None:
                # A missing camera is upstream's business (I-3 already rejects an
                # incomplete R2 keyframe); what matters here is that the union is
                # over the cameras that exist, and that the record says so.
                continue
            lifts.append(
                lift_camera(
                    points_ego_m,
                    ego_pose_at_lidar,
                    observation,
                    ego_pose_table,
                    calibrated_table,
                    masks,
                    instances,
                    cfg,
                )
            )
    finally:
        masks.close()

    contest = resolve_contest(lifts, points_ego_m.shape[0], cfg)
    region = region_for(keyframe.coverage_config)
    rows = instance_rows(instances, contest, points_ego_m, region, cfg)

    frusta = {
        "n_cameras": len(lifts),
        "cameras": [lift.channel for lift in lifts],
        "per_camera": {
            lift.channel: {
                **lift.guards,
                "ego_translation_delta_m": round(lift.ego_translation_delta_m, 6),
            }
            for lift in lifts
        },
        # Zero across every camera means the two poses of §1.3's middle hops are
        # the same record: the chain still runs, still lands points on the image,
        # and quietly omits ego motion between the two capture times.
        "max_ego_translation_delta_m": round(
            max((lift.ego_translation_delta_m for lift in lifts), default=0.0), 6
        ),
    }
    return contest, instances, rows, points_ego_m, frusta


def run(
    paths: Paths,
    stage1_manifest: dict,
    stage1_marker,
    stage4_manifest: dict,
    stage4_marker,
    cfg: LiftConfig,
    stage1_dir: str,
    stage4_dir: str,
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

    substrate = Substrate.load(paths)
    root = os.path.join(stage4_dir, "scenes")
    if not os.path.isdir(root):
        raise UpstreamRefusal(f"{root} not found; Stage 4 wrote no scenes")
    names = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
    if scene_names:
        missing = sorted(set(scene_names) - set(names))
        if missing:
            raise UpstreamRefusal(f"requested scene(s) not present in Stage 4 output: {missing}")
        names = [n for n in names if n in scene_names]

    per_scene: list[dict] = []
    degraded = False
    totals = {
        "n_keyframes": 0,
        "n_instances": 0,
        "n_instances_below_gate": 0,
        "n_instances_empty": 0,
        "n_points_cloud": 0,
        "n_points_visible_union": 0,
        "n_points_painted": 0,
        "n_points_contested": 0,
        "n_points_reading_differs": 0,
        "n_culled_behind_camera": 0,
        "n_culled_near_zero_depth": 0,
        "n_culled_out_of_bounds": 0,
    }

    for scene_name in names:
        keyframes = read_records(
            os.path.join(stage1_dir, "scenes", scene_name, "keyframes.jsonl"), expect_type=KeyframeRecord
        )
        mask_index = read_mask_index(os.path.join(root, scene_name, "masks.jsonl"))
        index_rows: list[dict] = []
        scene_totals = {key: 0 for key in totals}
        scene_max_delta = 0.0

        for keyframe in keyframes:
            mask_row = mask_index.get(keyframe.keyframe_token)
            if mask_row is None:
                raise UpstreamRefusal(
                    f"{scene_name}: keyframe {keyframe.keyframe_token} has no Stage 4 masks; the two "
                    "upstreams describe different keyframe sets"
                )
            contest, instances, rows, points_ego_m, frusta = lift_keyframe(
                keyframe, mask_row, stage4_dir, substrate, cfg
            )
            points_path = os.path.join(out_dir, "scenes", scene_name, "points", f"{keyframe.keyframe_token}.npz")
            write_points_npz(points_path, contest, cfg)

            n_empty = sum(1 for r in rows if r["n_points"] == 0)
            n_below = sum(1 for r in rows if r["n_points_below_gate"])
            index_rows.append(
                {
                    "spec": STAGE_SPEC,
                    "keyframe_token": keyframe.keyframe_token,
                    "scene_token": keyframe.scene_token,
                    "t_ns": keyframe.t_ns,
                    "time_base": keyframe.time_base,
                    "frame": EGO,
                    "coverage_config": keyframe.coverage_config,
                    "is_first_in_scene": keyframe.is_first_in_scene,
                    "cloud_kind": cfg.cloud_kind,
                    "cloud_path": keyframe.single_sweep_cloud.path,
                    "n_points_cloud": int(points_ego_m.shape[0]),
                    "points_path": os.path.relpath(points_path, out_dir),
                    "mask_path": os.path.join(
                        os.path.relpath(stage4_dir, out_dir), mask_row["mask_path"]
                    ),
                    "channels": list(contest.channels),
                    "frusta": frusta,
                    "contest": contest.ledger,
                    "instances": rows,
                    "n_instances": len(rows),
                    "n_instances_empty": n_empty,
                    "n_instances_below_gate": n_below,
                }
            )

            ledger = contest.ledger
            scene_totals["n_keyframes"] += 1
            scene_totals["n_instances"] += len(rows)
            scene_totals["n_instances_below_gate"] += n_below
            scene_totals["n_instances_empty"] += n_empty
            scene_totals["n_points_cloud"] += ledger["n_points_cloud"]
            scene_totals["n_points_visible_union"] += ledger["n_points_visible_union"]
            scene_totals["n_points_painted"] += ledger["n_points_painted"]
            scene_totals["n_points_contested"] += ledger["n_points_contested"]
            scene_totals["n_points_reading_differs"] += ledger["n_points_reading_differs"]
            for key in ("n_culled_behind_camera", "n_culled_near_zero_depth", "n_culled_out_of_bounds"):
                scene_totals[key] += sum(g[key] for g in frusta["per_camera"].values())
            scene_max_delta = max(scene_max_delta, frusta["max_ego_translation_delta_m"])

        if scene_totals["n_keyframes"] and scene_max_delta == 0.0:
            # Not a tolerance: an ego that never moved by any amount across a
            # whole scene, in double precision, means every camera was handed the
            # LiDAR's own pose and hops 1 and 2 of §1.3 cancelled.
            raise LiftContractError(
                f"{scene_name}: |ego(t_cam) - ego(t_lidar)| is exactly 0 for every camera of every "
                "keyframe. The two middle hops of the projection chain have been handed the same "
                "ego_pose twice, so the chain silently omits ego motion between capture times (§1.3)"
            )

        write_jsonl_atomic(os.path.join(out_dir, "scenes", scene_name, "lift.jsonl"), index_rows)
        summary = {
            "scene": scene_name,
            **scene_totals,
            "union_fraction": round(
                scene_totals["n_points_visible_union"] / max(1, scene_totals["n_points_cloud"]), 5
            ),
            "points_per_instance": round(
                scene_totals["n_points_painted"] / max(1, scene_totals["n_instances"]), 2
            ),
            "max_ego_translation_delta_m": round(scene_max_delta, 6),
            # An instance that painted nothing is reportable, not fatal; a scene
            # in which NOTHING was painted means the lift did not happen.
            "degraded": scene_totals["n_instances"] > 0 and scene_totals["n_points_painted"] == 0,
        }
        per_scene.append(summary)
        degraded = degraded or summary["degraded"]
        for key in totals:
            totals[key] += scene_totals[key]
        print(
            f"  {scene_name}  {scene_totals['n_keyframes']:>3} kf  "
            f"{scene_totals['n_instances']:>5} inst  {scene_totals['n_points_painted']:>7} painted  "
            f"{summary['points_per_instance']:>7.2f}/inst  union {summary['union_fraction']:.3f}  "
            f"{scene_totals['n_instances_below_gate']:>4} < {cfg.min_points_per_instance} pts"
            + ("  DEGRADED" if summary["degraded"] else "")
        )

    manifest = {
        "spec": STAGE_SPEC,
        "stage": STAGE,
        "seed": cfg.global_seed,
        "config": cfg.as_dict(),
        "upstream": {
            "metadata_fingerprint": stage1_manifest["upstream"]["metadata_fingerprint"],
            "fingerprint_spec": stage1_manifest["upstream"]["fingerprint_spec"],
            "stage1_spec": stage1_manifest["spec"],
            "stage4_spec": stage4_manifest["spec"],
            "stage3_prompt_caption_sha256": stage4_manifest["upstream"]["prompt_caption_sha256"],
            # C16: a run built on accepted degradation says so in its provenance.
            "stage1_degraded": stage1_marker.degraded,
            "stage1_degraded_causes": list(stage1_marker.causes),
            "stage4_degraded": stage4_marker.degraded,
            "stage4_degraded_causes": list(stage4_marker.causes),
            "accepted_degraded_upstream": cfg.accept_degraded_upstream,
        },
        "paths": paths.as_dict(),
        "frame": EGO,
        "cloud_kind": cfg.cloud_kind,
        "image_size_px": [cfg.image_width_px, cfg.image_height_px],
        "projection": {
            "chain": "point_ego(t_lidar) -> nuscenes_global -> point_ego(t_cam) -> camera -> pixel",
            "implementation": "pipeline.common.conventions.project_lidar_to_image",
            "guards": [
                "z <= 0, culled before the perspective divide",
                f"near-zero depth, z <= {cfg.min_depth_m} m",
                "out of bounds, outside [0, W) x [0, H)",
                "deterministic overlap resolution: principal-axis cosine, "
                "tie-broken by the fixed camera-priority list",
            ],
        },
        "contest": {
            "rule": "principal_axis_cosine",
            "scope": cfg.contest_scope,
            "alternate_scope": [s for s in CONTEST_SCOPES if s != cfg.contest_scope],
            "n_points_reading_differs": totals["n_points_reading_differs"],
            "within_camera_rule": cfg.within_camera_rule,
            "camera_priority": list(cfg.camera_priority),
        },
        "known_gaps": [
            "no occlusion reasoning: background points inside a mask are painted with it. These are "
            "the reprojection ghosts §1.6 assigns to Stage 6's keep-largest-cluster filter; the "
            "per-point depth is recorded so that filter has what it needs",
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
    parser.add_argument("--out-dir", default=None, help="default <work_root>/stage5_lift")
    parser.add_argument("--scenes", nargs="*", default=None, help="subset of Stage 4 scene names")
    parser.add_argument("--contest-scope", default="labelled_cameras", choices=CONTEST_SCOPES)
    parser.add_argument("--within-camera-rule", default="smallest_mask", choices=WITHIN_CAMERA_RULES)
    parser.add_argument("--seed", type=int, default=None, help="override the global seed (recorded)")
    parser.add_argument(
        "--accept-degraded-upstream",
        action="store_true",
        help="consume a DEGRADED (complete, quality-flagged) Stage 1 or 4 output; recorded (C16)",
    )
    args = parser.parse_args(argv)

    try:
        paths = load_paths(args.paths)
    except PathValidationError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    stage1_dir = args.stage1_dir or os.path.join(paths.work_root, "stage1_ingestion")
    stage4_dir = args.stage4_dir or os.path.join(paths.work_root, "stage4_masks")
    out_dir = args.out_dir or os.path.join(paths.work_root, STAGE)
    assert_dataroot_read_only(paths, out_dir)

    cfg = LiftConfig(
        contest_scope=args.contest_scope,
        within_camera_rule=args.within_camera_rule,
        accept_degraded_upstream=args.accept_degraded_upstream,
        **({"global_seed": args.seed} if args.seed is not None else {}),
    )

    try:
        stage1_manifest, stage1_marker, stage4_manifest, stage4_marker = load_upstream(
            paths, stage1_dir, stage4_dir, accept_degraded=cfg.accept_degraded_upstream
        )
        manifest, code = run(
            paths, stage1_manifest, stage1_marker, stage4_manifest, stage4_marker,
            cfg, stage1_dir, stage4_dir, out_dir, args.scenes,
        )
    except UpstreamRefusal as exc:
        print(f"REFUSING TO START: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (LiftContractError, PathValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)
    # Three-state marker (§1.9, C16): clean / degraded-with-causes / absent.
    write_marker(
        out_dir,
        manifest["upstream"]["metadata_fingerprint"],
        degraded=code == EXIT_DEGRADED,
        causes=[
            f"{s['scene']}: {s['n_instances']} instance(s), 0 points painted"
            for s in manifest["scenes"]
            if s["degraded"]
        ],
    )

    t = manifest["totals"]
    print(f"keyframes            : {t['n_keyframes']}")
    print(f"single-sweep points  : {t['n_points_cloud']}")
    print(
        f"frustum union        : {t['n_points_visible_union']}  "
        f"({t['n_points_visible_union'] / max(1, t['n_points_cloud']):.3f} of the cloud, "
        f"{len(RING_CAMERAS)} cameras)"
    )
    print(f"painted              : {t['n_points_painted']}  into {t['n_instances']} instances")
    print(f"contested points     : {t['n_points_contested']}  (resolved by principal-axis cosine)")
    print(
        f"reading disagreement : {t['n_points_reading_differs']}  "
        f"(points the {CONTEST_SCOPES[1]} reading would drop)"
    )
    print(
        f"culled               : {t['n_culled_behind_camera']} behind camera, "
        f"{t['n_culled_near_zero_depth']} near-zero depth, {t['n_culled_out_of_bounds']} out of bounds"
    )
    print(
        f"instances < {cfg.min_points_per_instance} points : {t['n_instances_below_gate']}  "
        f"({t['n_instances_empty']} painted nothing at all)"
    )
    print(f"wrote {out_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
