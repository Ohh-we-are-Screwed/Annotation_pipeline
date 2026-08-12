"""Frames, time base, and the one projection chain (§1.1–§1.3, §1.6).

This module is the single source of truth for three things that are otherwise
re-derived — differently — in every stage that needs them:

  1. **Frames.** The ego frame (ISO 8855: x forward, y left, z up) is canonical
     for all cross-stage geometry. Every geometric record carries an explicit
     `frame`; a record with no frame is invalid, not defaulted (§1.1 rule 2).
  2. **Time.** nuScenes ships microseconds, Unix epoch. The pilot stores int64
     nanoseconds. `to_unix_ns()` is the ONLY function in the codebase that
     multiplies or divides a timestamp (§1.2 rule 2).
  3. **Projection.** `project_lidar_to_image()` is the only implementation of
     the four-hop chain. Stages call it; they do not compose their own
     transforms (§1.3).

Measured facts this module is built around, verified against the substrate by
`scripts/probe_substrate.py`, not assumed:

  - `LIDAR_TOP`'s `calibrated_sensor` rotation is a yaw of about -89.9 deg: the
    raw sensor frame has +x to the vehicle's RIGHT and +y FORWARD. Sensor frame
    is not ego frame, and nothing here pretends otherwise.
  - Every `sample_data` record carries its OWN `ego_pose`. Within one keyframe
    the LiDAR and a given camera are captured up to ~48 ms apart, so the two
    middle hops of the chain are load-bearing, not ceremony.
  - nuScenes `camera_intrinsic` carries no distortion coefficients; the imagery
    is already rectified. Undistortion is a no-op on this substrate and is
    recorded as such, never as evidence that the distortion path was tested.

No GPU, no models: numpy and stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "FRAMES",
    "EGO",
    "LIDAR",
    "CAMERA",
    "NUSCENES_GLOBAL",
    "TIME_BASES",
    "INTERNAL_TIME_BASE",
    "US_PER_NS",
    "MIN_DEPTH_M",
    "FrameError",
    "TimeBaseError",
    "check_frame",
    "check_time_base",
    "to_unix_ns",
    "Transform",
    "quaternion_to_rotation_matrix",
    "quaternion_from_yaw_rad",
    "yaw_rad_from_quaternion",
    "wrap_to_pi_rad",
    "transform_matrix",
    "apply_transform",
    "Projection",
    "project_lidar_to_image",
]

# ---------------------------------------------------------------------------
# §1.1 Frames
# ---------------------------------------------------------------------------

EGO = "ego"
LIDAR = "lidar"
CAMERA = "camera"
# nuScenes' "global" is a per-map arbitrary frame. It is NOT the published-origin
# local ENU of comprehensive.md §2.3, and it is never called "global" here: the
# name is the deviation record (§1.1 rule 4).
NUSCENES_GLOBAL = "nuscenes_global"

FRAMES: tuple[str, ...] = (EGO, LIDAR, CAMERA, NUSCENES_GLOBAL)


class FrameError(ValueError):
    """A geometric record carried an absent or unknown frame."""


def check_frame(frame: object, *, field: str = "frame") -> str:
    """Validate a frame tag. There is no default; absence is an error."""
    if frame is None:
        raise FrameError(f"{field} is required; a geometric record with no frame is invalid")
    if frame not in FRAMES:
        raise FrameError(f"{field}={frame!r} is not one of {FRAMES}")
    return str(frame)


# ---------------------------------------------------------------------------
# §1.2 Time base
# ---------------------------------------------------------------------------

# "gps_ns" is declared but unproduced: the pilot does not have GPS time and does
# not claim it. The divergence from comprehensive.md §2.3 is a recorded pilot
# deviation, not a silently absorbed one (§1.2 rule 3).
TIME_BASES: tuple[str, ...] = ("unix_us", "unix_ns", "gps_ns")
INTERNAL_TIME_BASE = "unix_ns"

US_PER_NS = 1_000  # nanoseconds per microsecond

_INT64_MAX = 2**63 - 1
_INT64_MIN = -(2**63)


class TimeBaseError(ValueError):
    """A temporal record carried an absent, unknown, or unconvertible time base."""


def check_time_base(time_base: object, *, field: str = "time_base") -> str:
    if time_base is None:
        raise TimeBaseError(f"{field} is required; a temporal record with no time base is invalid")
    if time_base not in TIME_BASES:
        raise TimeBaseError(f"{field}={time_base!r} is not one of {TIME_BASES}")
    return str(time_base)


def to_unix_ns(timestamp: int, time_base: str) -> int:
    """Convert a timestamp to int64 Unix nanoseconds. The ONLY such conversion.

    Nothing else in the pipeline multiplies or divides a timestamp (§1.2 rule 2).
    A 1000x unit error committed here is committed exactly once and is testable
    against a real `sample_data` pair with a physically known offset; the same
    error spread across nine stages is not.

    `gps_ns` is rejected on purpose: converting it would require a leap-second
    table and a claim about the clock that the pilot cannot support.
    """
    check_time_base(time_base)
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, np.integer)):
        raise TimeBaseError(f"timestamp must be an integer, got {type(timestamp).__name__}")
    value = int(timestamp)
    if time_base == "unix_us":
        value *= US_PER_NS
    elif time_base == "gps_ns":
        raise TimeBaseError(
            "refusing to convert gps_ns: the pilot has no GPS time source and no leap-second "
            "table; records claiming gps_ns are a provenance error, not a unit problem"
        )
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise TimeBaseError(f"converted timestamp {value} does not fit int64")
    return value


# ---------------------------------------------------------------------------
# Rotations and rigid transforms
# ---------------------------------------------------------------------------

# nuScenes stores quaternions as [w, x, y, z]. Every quaternion in this codebase
# is wxyz; the field names say so.
_QUAT_NORM_TOL = 1e-6


def quaternion_to_rotation_matrix(q_wxyz: Sequence[float]) -> np.ndarray:
    """Rotation matrix from a nuScenes-order [w, x, y, z] quaternion."""
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < _QUAT_NORM_TOL:
        raise ValueError(f"quaternion is not normalisable: {q_wxyz!r}")
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_rad_from_quaternion(q_wxyz: Sequence[float]) -> float:
    """Yaw about +z, measured from +x, radians, wrapped to (-pi, pi].

    ISO 8855 convention (§1.1). Stage 6 produces a scalar yaw and I-4 stores a
    quaternion; §3.2 requires the yaw -> quaternion -> yaw round trip to be an
    explicit, tested conversion rather than an inline expression repeated at
    each site, because an axis or sign error there is invisible in the output.
    """
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm < _QUAT_NORM_TOL:
        raise ValueError(f"quaternion is not normalisable: {q_wxyz!r}")
    w, x, y, z = q / norm
    return wrap_to_pi_rad(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def quaternion_from_yaw_rad(yaw_rad: float) -> tuple[float, float, float, float]:
    """[w, x, y, z] for a pure yaw about +z. Inverse of yaw_rad_from_quaternion."""
    if not math.isfinite(yaw_rad):
        raise ValueError(f"yaw_rad must be finite, got {yaw_rad!r}")
    half = 0.5 * float(yaw_rad)
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def wrap_to_pi_rad(angle_rad: float) -> float:
    """Wrap to (-pi, pi]. Yaw differences are compared only after wrapping."""
    wrapped = math.remainder(float(angle_rad), 2.0 * math.pi)
    return math.pi if wrapped == -math.pi else wrapped


@dataclass(frozen=True)
class Transform:
    """A nuScenes rigid transform: SOURCE -> PARENT.

    Both `ego_pose` (ego -> nuscenes_global) and `calibrated_sensor`
    (sensor -> ego) are stored by nuScenes in this direction. Projection needs
    the inverses, and taking the wrong direction is the classic error this type
    exists to make impossible to commit silently: ask for `.inverse_matrix()`
    by name, or do not get an inverse.
    """

    translation_m: tuple[float, float, float]
    rotation_wxyz: tuple[float, float, float, float]
    source_frame: str
    parent_frame: str

    @classmethod
    def from_nuscenes(
        cls,
        record: dict,
        *,
        source_frame: str,
        parent_frame: str,
    ) -> "Transform":
        """Build from a raw `ego_pose` / `calibrated_sensor` dict."""
        check_frame(source_frame, field="source_frame")
        check_frame(parent_frame, field="parent_frame")
        t = tuple(float(v) for v in record["translation"])
        q = tuple(float(v) for v in record["rotation"])
        if len(t) != 3 or len(q) != 4:
            raise ValueError("nuScenes transform needs translation[3] and rotation[4]")
        return cls(translation_m=t, rotation_wxyz=q, source_frame=source_frame, parent_frame=parent_frame)

    def matrix(self) -> np.ndarray:
        """4x4 homogeneous SOURCE -> PARENT."""
        return transform_matrix(self.translation_m, self.rotation_wxyz, inverse=False)

    def inverse_matrix(self) -> np.ndarray:
        """4x4 homogeneous PARENT -> SOURCE."""
        return transform_matrix(self.translation_m, self.rotation_wxyz, inverse=True)

    @property
    def yaw_rad(self) -> float:
        return yaw_rad_from_quaternion(self.rotation_wxyz)


def transform_matrix(
    translation_m: Sequence[float],
    rotation_wxyz: Sequence[float],
    *,
    inverse: bool = False,
) -> np.ndarray:
    """4x4 homogeneous transform. `inverse=True` gives parent -> source."""
    R = quaternion_to_rotation_matrix(rotation_wxyz)
    t = np.asarray(translation_m, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(t)):
        raise ValueError(f"translation_m must be finite, got {translation_m!r}")
    T = np.eye(4, dtype=np.float64)
    if inverse:
        T[:3, :3] = R.T
        T[:3, 3] = -R.T @ t
    else:
        T[:3, :3] = R
        T[:3, 3] = t
    return T


def apply_transform(T: np.ndarray, points_m: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to an (N, 3) point array."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"transform must be 4x4, got {T.shape}")
    p = np.asarray(points_m, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {p.shape}")
    return p @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------------------
# §1.3 The projection chain
# ---------------------------------------------------------------------------

# Points at or behind the image plane are culled BEFORE the perspective divide.
# A point with z < 0 divided by its own z lands at a valid-looking mirrored
# pixel, inside the image, and paints itself with whatever mask is there.
# The guard band also covers z ~ 0, where the divide explodes rather than lies.
MIN_DEPTH_M = 1e-3


@dataclass(frozen=True)
class Projection:
    """Result of `project_lidar_to_image`, on the points that survived the cull.

    Arrays are parallel and indexed by the surviving points, in input order.
    `source_index` maps each row back to the input array so a caller can paint
    the original cloud without reconstructing the mask itself.
    """

    uv_px: np.ndarray  # (M, 2) float64, absolute pixels, ORIGINAL resolution (§1.5 rule 1)
    depth_m: np.ndarray  # (M,) float64, camera-frame z, strictly > min_depth_m
    source_index: np.ndarray  # (M,) int64, row index into the input points
    in_image: np.ndarray  # (M,) bool, uv falls inside [0, W) x [0, H)
    points_camera_m: np.ndarray  # (M, 3) float64, camera frame, pre-divide
    n_input: int
    n_culled_behind_camera: int
    ego_translation_delta_m: float  # |ego(t_cam) - ego(t_lidar)|, the middle hops' effect

    @property
    def n_visible(self) -> int:
        """Points that are in front of the camera AND inside the image."""
        return int(np.count_nonzero(self.in_image))


def project_lidar_to_image(
    points_ego_at_lidar_m: np.ndarray,
    ego_pose_at_lidar: Transform,
    ego_pose_at_camera: Transform,
    camera_extrinsic: Transform,
    camera_intrinsic_k: np.ndarray,
    image_size_px: tuple[int, int],
    *,
    min_depth_m: float = MIN_DEPTH_M,
) -> Projection:
    """The four-hop chain, written as four hops.

        point_ego(t_lidar) -> nuscenes_global -> point_ego(t_cam) -> camera -> pixel

    Inputs
      points_ego_at_lidar_m : (N, 3) points already in EGO frame at the LiDAR
          anchor time. Stage 1 applies `T_ego_lidar` exactly once; this function
          does not re-apply it (§1.1 rule 3).
      ego_pose_at_lidar     : ego(t_lidar) -> nuscenes_global, from the LiDAR
          `sample_data`'s own `ego_pose`.
      ego_pose_at_camera    : ego(t_cam) -> nuscenes_global, from the CAMERA
          `sample_data`'s own `ego_pose`. These are different records; that is
          the entire point of hops 1 and 2.
      camera_extrinsic      : camera -> ego(t_cam), i.e. nuScenes
          `calibrated_sensor` as stored. It is INVERTED here.
      camera_intrinsic_k    : (3, 3) K, applied after the extrinsics, never
          folded into them.

    The two middle hops carry ego motion between the two capture times. Dropping
    them ("project through calibrated intrinsics and extrinsics") still lands
    points on the image, still hits masks, still paints points: near-range
    objects simply acquire a systematic several-pixel offset in the direction of
    travel, points bleed across mask boundaries onto neighbouring objects, and
    every cluster is biased. At 50 km/h and 35 ms that is ~0.5 m of silent lie.
    A synthetic-point test with a single static calibration cannot detect it,
    which is why `ego_translation_delta_m` is returned: a caller that measures
    zero on real data has been handed the same pose twice.
    """
    points = np.asarray(points_ego_at_lidar_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_ego_at_lidar_m must be (N, 3), got {points.shape}")
    if not np.all(np.isfinite(points)):
        raise ValueError("points_ego_at_lidar_m contains non-finite values")

    for name, tf, source, parent in (
        ("ego_pose_at_lidar", ego_pose_at_lidar, EGO, NUSCENES_GLOBAL),
        ("ego_pose_at_camera", ego_pose_at_camera, EGO, NUSCENES_GLOBAL),
        ("camera_extrinsic", camera_extrinsic, CAMERA, EGO),
    ):
        if not isinstance(tf, Transform):
            raise TypeError(f"{name} must be a Transform, got {type(tf).__name__}")
        if (tf.source_frame, tf.parent_frame) != (source, parent):
            raise FrameError(
                f"{name} must be {source} -> {parent}, got "
                f"{tf.source_frame} -> {tf.parent_frame}"
            )

    K = np.asarray(camera_intrinsic_k, dtype=np.float64)
    if K.shape != (3, 3):
        raise ValueError(f"camera_intrinsic_k must be (3, 3), got {K.shape}")
    if not np.all(np.isfinite(K)) or K[2, 2] == 0.0:
        raise ValueError("camera_intrinsic_k must be finite with a non-zero K[2][2]")

    width_px, height_px = (int(image_size_px[0]), int(image_size_px[1]))
    if width_px <= 0 or height_px <= 0:
        raise ValueError(f"image_size_px must be positive, got {image_size_px!r}")
    if not math.isfinite(min_depth_m) or min_depth_m <= 0.0:
        raise ValueError(f"min_depth_m must be a positive finite value, got {min_depth_m!r}")

    # --- hop 1: ego(t_lidar) -> nuscenes_global --------------------------------
    p_global = apply_transform(ego_pose_at_lidar.matrix(), points)

    # --- hop 2: nuscenes_global -> ego(t_cam) ----------------------------------
    p_ego_cam = apply_transform(ego_pose_at_camera.inverse_matrix(), p_global)

    # --- hop 3: ego(t_cam) -> camera -------------------------------------------
    # nuScenes gives sensor -> ego; projection needs ego -> sensor.
    p_camera = apply_transform(camera_extrinsic.inverse_matrix(), p_ego_cam)

    # --- cull before the divide -------------------------------------------------
    depth_m = p_camera[:, 2]
    keep = depth_m > min_depth_m
    kept_index = np.nonzero(keep)[0].astype(np.int64)
    p_camera_kept = p_camera[keep]
    depth_kept = p_camera_kept[:, 2]

    # --- hop 4: camera -> pixel -------------------------------------------------
    uv_h = p_camera_kept @ K.T  # (M, 3), K applied after the extrinsics
    uv_px = uv_h[:, :2] / uv_h[:, 2:3]

    in_image = (
        (uv_px[:, 0] >= 0.0)
        & (uv_px[:, 0] < float(width_px))
        & (uv_px[:, 1] >= 0.0)
        & (uv_px[:, 1] < float(height_px))
    )

    delta = np.asarray(ego_pose_at_camera.translation_m, dtype=np.float64) - np.asarray(
        ego_pose_at_lidar.translation_m, dtype=np.float64
    )

    return Projection(
        uv_px=uv_px,
        depth_m=depth_kept,
        source_index=kept_index,
        in_image=in_image,
        points_camera_m=p_camera_kept,
        n_input=int(points.shape[0]),
        n_culled_behind_camera=int(points.shape[0] - kept_index.size),
        ego_translation_delta_m=float(np.linalg.norm(delta)),
    )
