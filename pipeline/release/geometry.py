"""Release geometry: box frame changes, corners, tokens, quaternion helpers.

Moved out of `scripts/export_release.py` so the release post-processing modules
can share them without importing the exporter script. Conventions are the
pipeline's: quaternions are [w, x, y, z], size is [w, l, h] (nuScenes order,
`l` along heading), and `normalise_quat` canonicalises the sign to w >= 0.

A degenerate quaternion raises `ValueError` here; the exporter wraps that in its
own `ExportError`.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

from pipeline.common.conventions import (
    Transform,
    apply_transform,
    quaternion_to_rotation_matrix,
    yaw_rad_from_quaternion,
)


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------


def make_token(*parts: Any) -> str:
    """Deterministic 32-hex token, the nuScenes shape, from a namespace + parts."""
    h = hashlib.md5("\x1f".join(str(p) for p in parts).encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of two [w, x, y, z] quaternions (a then b applied = a*b)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=np.float64)


def normalise_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = float(np.linalg.norm(q))
    if not math.isfinite(n) or n < 1e-9:
        raise ValueError(f"degenerate quaternion {q.tolist()}")
    q = q / n
    if q[0] < 0:  # canonical sign, w >= 0
        q = -q
    return q


def box_ego_to_global(translation_ego: list, rotation_ego_wxyz: list,
                      ego_pose: Transform) -> tuple[list[float], list[float]]:
    t = apply_transform(ego_pose.matrix(), np.asarray([translation_ego], dtype=np.float64))[0]
    q = quat_multiply(normalise_quat(np.asarray(ego_pose.rotation_wxyz)),
                      normalise_quat(np.asarray(rotation_ego_wxyz)))
    return [float(v) for v in t], [float(v) for v in normalise_quat(q)]


def box_global_to_ego(translation_global: list, rotation_global_wxyz: list,
                      ego_pose: Transform) -> tuple[list[float], list[float]]:
    t = apply_transform(ego_pose.inverse_matrix(), np.asarray([translation_global], dtype=np.float64))[0]
    qe = normalise_quat(np.asarray(ego_pose.rotation_wxyz))
    qe_inv = np.array([qe[0], -qe[1], -qe[2], -qe[3]])
    q = quat_multiply(qe_inv, normalise_quat(np.asarray(rotation_global_wxyz)))
    return [float(v) for v in t], [float(v) for v in normalise_quat(q)]


def box_corners_ego(translation: list, size_wlh: list, rotation_wxyz: list) -> np.ndarray:
    """(8, 3) corners, the nuScenes-devkit `Box.corners()` layout."""
    w, l, h = (float(v) for v in size_wlh)
    x = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    local = np.stack([x, y, z], axis=1)
    R = quaternion_to_rotation_matrix(rotation_wxyz)
    return local @ R.T + np.asarray(translation, dtype=np.float64)


def yaw_of(q_wxyz) -> float:
    return yaw_rad_from_quaternion([float(v) for v in q_wxyz])


def slerp(q0, q1, s: float) -> np.ndarray:
    """Shortest-arc spherical interpolation of unit [w,x,y,z] quaternions."""
    a = normalise_quat(np.asarray(q0, dtype=np.float64))
    b = normalise_quat(np.asarray(q1, dtype=np.float64))
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 1.0 - 1e-9:
        return normalise_quat(a + s * (b - a))
    theta = math.acos(min(1.0, dot))
    sin_t = math.sin(theta)
    return normalise_quat((math.sin((1.0 - s) * theta) / sin_t) * a + (math.sin(s * theta) / sin_t) * b)


def points_in_box(points_xyz: np.ndarray, translation, size_wlh, rotation_wxyz) -> int:
    """Count points strictly inside the oriented box (size [w, l, h]: l along heading)."""
    if points_xyz.size == 0:
        return 0
    R = quaternion_to_rotation_matrix([float(v) for v in rotation_wxyz])
    local = (np.asarray(points_xyz[:, :3], dtype=np.float64) - np.asarray(translation, dtype=np.float64)) @ R
    w, l, h = (float(v) for v in size_wlh)
    inside = (np.abs(local[:, 0]) < l / 2) & (np.abs(local[:, 1]) < w / 2) & (np.abs(local[:, 2]) < h / 2)
    return int(inside.sum())
