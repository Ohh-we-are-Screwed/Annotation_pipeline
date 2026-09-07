from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.conventions import EGO, NUSCENES_GLOBAL, Transform, quaternion_from_yaw_rad  # noqa: E402
from pipeline.release.config import StitchConfig  # noqa: E402
from pipeline.release.frames import SceneFrames  # noqa: E402
from pipeline.release.stitch import stitch_scene  # noqa: E402

CFG = StitchConfig(max_gap_keyframes=3, base_gate_m=2.0, gap_slack_m=1.0, size_ratio_max=2.0,
                   class_agnostic=False, interpolated_min_lidar_points=5)
DT_NS = 400_000_000


def _frames(n=8, ego_speed=0.0):
    toks = [f"s{i}" for i in range(n)]
    poses = {t: Transform.from_nuscenes({"translation": [ego_speed * 0.4 * i, 0.0, 0.0],
                                         "rotation": list(quaternion_from_yaw_rad(0.0))},
                                        source_frame=EGO, parent_frame=NUSCENES_GLOBAL)
             for i, t in enumerate(toks)}
    return SceneFrames("sc", "chunk_t", toks, [i * DT_NS for i in range(n)], poses)


def _rec(i, track, x, y=0.0, cat="a car", tier="auto_accept", size=(1.8, 4.5, 1.6), yaw=0.0):
    return {
        "token": f"s{i}:CAM_FRONT:{track}", "sample_token": f"s{i}",
        "instance_token": f"pilot-track:sc:{track}", "category": cat, "frame": "ego",
        "t_ns": i * DT_NS, "time_base": "unix_ns",
        "translation_m": [x, y, 0.0], "size_wlh_m": list(size),
        "rotation_wxyz": list(quaternion_from_yaw_rad(yaw)), "num_lidar_pts": 20,
        "num_lidar_pts_basis": "single_sweep_ground_filtered_pre_inflation",
        "provenance": {"source": "pipeline", "tier": tier, "gates": {"conf": 0.9}, "verified_by": None,
                       "verification_pass": 0},
        "coverage_config": "R2", "velocity_mps": [0.0, 0.0], "track_id": str(track), "split": None,
        "attribute": None, "visibility": None,
    }


def _chains(rows):
    out = {}
    for r in rows:
        out.setdefault(r["stitch_chain_id"], []).append(r["sample_token"])
    return {k: sorted(v, key=lambda t: int(t[1:])) for k, v in out.items()}


def test_gap1_join_of_a_moving_car():
    # track 1 at 10 m/s for frames 0-2, track 2 continues at frames 3-5
    rows = [_rec(i, 1, 10.0 + 4.0 * i) for i in range(3)] + [_rec(i, 2, 10.0 + 4.0 * i) for i in range(3, 6)]
    out, st = stitch_scene(rows, _frames(), None, CFG)
    ch = _chains(out)
    assert len(ch) == 1 and st.n_chains == 1 and st.joins_by_gap == {1: 1}
    assert st.n_interpolated == 0
    assert all(r["instance_token"] == "chain:sc:1" for r in out)
    assert {r["stitch_track_id_pre"] for r in out} == {"1", "2"}


def test_gap3_join_interpolates_two_rows_and_inherits_worse_tier():
    rows = [_rec(i, 1, 1.0 * i, tier="auto_accept") for i in range(3)]          # frames 0,1,2 (1 m/s)
    rows += [_rec(i, 2, 1.0 * i, tier="flagged") for i in range(5, 8)]          # frames 5,6,7
    out, st = stitch_scene(rows, _frames(), None, CFG)
    assert st.joins_by_gap == {3: 1} and st.n_interpolated == 2
    interp = sorted((r for r in out if r["stitch_interpolated"]), key=lambda r: r["t_ns"])
    assert [r["sample_token"] for r in interp] == ["s3", "s4"]
    assert interp[0]["translation_m"][0] == pytest.approx(3.0) and interp[1]["translation_m"][0] == pytest.approx(4.0)
    assert interp[0]["provenance"]["tier"] == "flagged" and interp[0]["stitch_tier_basis"] == "inherited_from_endpoints"
    assert interp[0]["token"] == "s3:INTERP:1" and interp[0]["num_lidar_pts_basis"] == "unavailable"
    # 0 because there is no cloud to count against here. The stitcher writes the
    # row either way; whether a row with 0 returns SHIPS is the tier filter's
    # question, and tiers.partition sends it to the sidecar as
    # `interpolated_below_point_floor` (tests/test_release_tiers.py).
    assert interp[0]["num_lidar_pts"] == 0
    assert interp[0]["track_id"] is None and interp[0]["velocity_mps"] is None


def test_class_mismatch_and_gate_refuse():
    rows = [_rec(0, 1, 0.0), _rec(1, 2, 0.0, cat="a bus")]                 # class differs
    rows += [_rec(3, 3, 50.0), _rec(4, 4, 60.0)]                           # 10 m apart, no velocity
    out, st = stitch_scene(rows, _frames(), None, CFG)
    assert st.n_chains == 4 and st.joins_by_gap == {}


def test_back_prediction_joins_single_frame_predecessor():
    # a single-frame fragment (no velocity) followed by a moving fragment whose back-prediction lands on it
    rows = [_rec(0, 1, 0.0)] + [_rec(i, 2, 4.0 * i) for i in range(1, 4)]   # 10 m/s
    out, st = stitch_scene(rows, _frames(), None, CFG)
    assert st.n_chains == 1


def test_stationary_assumption_refuses_fast_single_frames():
    rows = [_rec(0, 1, 0.0), _rec(1, 2, 4.0)]     # 4 m apart, neither has a velocity
    out, st = stitch_scene(rows, _frames(), None, CFG)
    assert st.n_chains == 2


def test_interpolation_is_in_each_keyframes_ego_frame():
    # ego drives +x at 5 m/s; object is stationary at global x=20
    fr = _frames(ego_speed=5.0)
    rows = [_rec(0, 1, 20.0), _rec(3, 2, 20.0 - 5.0 * 0.4 * 3)]
    out, st = stitch_scene(rows, fr, None, CFG)
    assert st.n_chains == 1 and st.n_interpolated == 2
    i1 = next(r for r in out if r["sample_token"] == "s1")
    assert i1["translation_m"][0] == pytest.approx(20.0 - 2.0)


def test_deterministic_and_idempotent_on_singletons():
    rows = [_rec(i, i, 3.0 * i, y=float(i)) for i in range(4)]     # 3.16 m apart each step, stationary assumption
    a, sa = stitch_scene([dict(r) for r in rows], _frames(), None, CFG)
    b, sb = stitch_scene([dict(r) for r in rows], _frames(), None, CFG)
    assert [r["stitch_chain_id"] for r in a] == [r["stitch_chain_id"] for r in b]
    assert sa.n_chains == 4
