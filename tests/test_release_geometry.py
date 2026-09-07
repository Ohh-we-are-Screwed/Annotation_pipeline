"""Tests for pipeline/release/geometry.py — the exporter's geometry helpers.

    /home/mt/miniconda3/envs/ano_pipe/bin/python -m pytest tests/test_release_geometry.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.conventions import EGO, NUSCENES_GLOBAL, Transform, quaternion_from_yaw_rad  # noqa: E402
from pipeline.release.geometry import (  # noqa: E402
    box_corners_ego, box_ego_to_global, box_global_to_ego, make_token, normalise_quat,
    points_in_box, slerp, yaw_of,
)


def _pose(x, y, yaw):
    return Transform.from_nuscenes({"translation": [x, y, 0.0], "rotation": list(quaternion_from_yaw_rad(yaw))},
                                   source_frame=EGO, parent_frame=NUSCENES_GLOBAL)


def test_ego_global_round_trip():
    pose = _pose(100.0, -20.0, 0.7)
    t, q = [5.0, 1.0, -1.5], list(quaternion_from_yaw_rad(0.3))
    tg, qg = box_ego_to_global(t, q, pose)
    te, qe = box_global_to_ego(tg, qg, pose)
    assert np.allclose(te, t, atol=1e-9)
    assert np.allclose(qe, normalise_quat(np.asarray(q)), atol=1e-9)


def test_make_token_is_deterministic_32_hex():
    a, b = make_token("instance", "s", "1"), make_token("instance", "s", "1")
    assert a == b and len(a) == 32 and int(a, 16) >= 0
    assert make_token("instance", "s", "2") != a


def test_slerp_endpoints_and_shortest_arc():
    q0 = np.asarray(quaternion_from_yaw_rad(0.0))
    q1 = np.asarray(quaternion_from_yaw_rad(math.radians(170)))
    assert np.allclose(slerp(q0, q1, 0.0), q0)
    assert abs(yaw_of(slerp(q0, q1, 1.0)) - math.radians(170)) < 1e-9
    mid = yaw_of(slerp(q0, q1, 0.5))
    assert abs(mid - math.radians(85)) < 1e-9
    # 350 deg is -10 deg: the short way round passes through -5, not 175
    q2 = np.asarray(quaternion_from_yaw_rad(math.radians(-10)))
    assert abs(yaw_of(slerp(q0, q2, 0.5)) - math.radians(-5)) < 1e-9


def test_points_in_box_counts_only_inside_oriented_box():
    yaw = math.radians(90)
    q = list(quaternion_from_yaw_rad(yaw))
    # width 1 (across heading), length 4 (along heading, now along +y), height 2
    pts = np.array([[0.0, 1.9, 0.0], [0.0, 2.1, 0.0], [0.4, 0.0, 0.9], [0.6, 0.0, 0.0], [0.0, 0.0, 1.1]])
    assert points_in_box(pts, [0.0, 0.0, 0.0], [1.0, 4.0, 2.0], q) == 2


def test_corners_match_size_order():
    c = box_corners_ego([0, 0, 0], [1.0, 4.0, 2.0], list(quaternion_from_yaw_rad(0.0)))
    assert c[:, 0].max() - c[:, 0].min() == pytest.approx(4.0)   # length along heading (+x)
    assert c[:, 1].max() - c[:, 1].min() == pytest.approx(1.0)   # width across
    assert c[:, 2].max() - c[:, 2].min() == pytest.approx(2.0)
