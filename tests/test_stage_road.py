"""Tests for pipeline/stage_road/road.py — the road-surface stage (C33).

Run: /home/mt/miniconda3/envs/ano_pipe/bin/python -m pytest tests/test_stage_road.py -v

House posture (same as Stage 5): the pure pieces — the union rule, the plane
gate, the two npz formats, marker composition, the projection paint — are
unit-tested with synthetic data and no GPU; the model-bound driver is
exercised only by a real run.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pipeline.common.manifest import UpstreamRefusal, write_marker  # noqa: E402
from pipeline.stage_road.road import (  # noqa: E402
    carve_raised,
    compose_marker,
    paint_cameras,
    plane_gate,
    resolve_sidewalk,
    run,
    union_masks,
    write_points_npz,
    write_road_masks_npz,
)

W, H = 1280, 720  # the substrate contract (schemas.IMAGE_WIDTH_PX/HEIGHT_PX)


class TestUnionMasks:
    def test_or_of_instances(self):
        a = np.zeros((4, 4), bool); a[0, :] = True
        b = np.zeros((4, 4), bool); b[:, 0] = True
        got = union_masks([a, b])
        assert got.dtype == bool and got.sum() == 7

    def test_empty_is_none(self):
        # "no region returned" must stay distinguishable from "empty mask":
        # the spike showed an absent firing means "the prompt missed", not
        # "no road in this frame"
        assert union_masks([]) is None


class TestPlaneGate:
    """The occlusion policy: a hole in the 2D mask does not stop a wall past
    the road edge from being claimed along the ray — the gate must."""

    def _road(self, n=800, z=-2.15, seed=0):
        rng = np.random.default_rng(seed)
        pts = np.column_stack([
            rng.uniform(2, 18, n), rng.uniform(-6, 6, n),
            rng.normal(z, 0.02, n)])
        return pts

    def test_rejects_wall_points_inside_the_mask(self):
        road = self._road()
        wall = np.column_stack([
            np.full(150, 11.0), np.linspace(-2, 2, 150),
            np.linspace(-1.5, 1.0, 150)])
        pts = np.vstack([road, wall])
        cand = np.arange(len(pts))
        got = plane_gate(pts, cand)
        assert got["refusal"] is None
        assert got["n_rejected_off_plane"] == 150
        assert set(got["kept_index"]) == set(range(len(road)))

    def test_refuses_thin_anchor(self):
        pts = self._road(n=50)
        got = plane_gate(pts, np.arange(50))
        assert got["refusal"] is not None and "anchor" in got["refusal"]
        assert got["kept_index"].size == 0

    def test_refuses_a_steep_fit(self):
        # z = 0.3x is ~16.7 deg of tilt — no road plane, refuse rather than gate
        rng = np.random.default_rng(1)
        x = rng.uniform(2, 18, 800); y = rng.uniform(-6, 6, 800)
        pts = np.column_stack([x, y, 0.3 * x - 4.0])
        got = plane_gate(pts, np.arange(800))
        assert got["refusal"] is not None and "tilt" in got["refusal"]

    def test_keeps_a_gently_sloped_road(self):
        # the ground fixture must not be flat: Dhaka roads have camber; a
        # 1.7 deg slope is a road, not a refusal
        rng = np.random.default_rng(2)
        x = rng.uniform(2, 18, 800); y = rng.uniform(-6, 6, 800)
        pts = np.column_stack([x, y, 0.03 * x - 2.3 + rng.normal(0, 0.02, 800)])
        got = plane_gate(pts, np.arange(800))
        assert got["refusal"] is None
        assert got["kept_index"].size >= 750


class TestNpzFormats:
    def test_masks_npz_reads_back_through_stage5_maskfile(self, tmp_path):
        # byte-for-byte Stage 4's layout: lift.py's MaskFile must read it
        # unchanged, one mask per channel
        from pipeline.stage5_lift.lift import MaskFile
        m1 = np.zeros((H, W), bool); m1[:100, :200] = True
        m2 = np.zeros((H, W), bool); m2[-50:, -60:] = True
        path = str(tmp_path / "kf0.npz")
        write_road_masks_npz(path, {"CAM_FRONT": m1, "CAM_BACK": m2})
        mf = MaskFile(path)
        assert mf.channels() == ["CAM_BACK", "CAM_FRONT"]
        assert mf.n_masks("CAM_FRONT") == 1
        assert np.array_equal(mf.mask("CAM_FRONT", 0), m1)
        assert np.array_equal(mf.mask("CAM_BACK", 0), m2)
        mf.close()

    def test_points_npz_names_its_basis(self, tmp_path):
        path = str(tmp_path / "kf0_points.npz")
        write_points_npz(
            path,
            road_point_index=np.array([3, 5, 9], np.int32),
            n_cameras_road=np.array([1, 2, 8], np.int8),
            seen_point_index=np.array([1, 3, 5, 9], np.int32),
            n_points_raw=19968,
            lidar_sample_data_token="sd_lidar_0",
        )
        z = np.load(path)
        assert list(z["road_point_index"]) == [3, 5, 9]
        assert z["road_point_index"].dtype == np.int32
        assert z["n_cameras_road"].dtype == np.int8
        assert list(z["seen_point_index"]) == [1, 3, 5, 9]
        assert int(z["__n_points_raw__"][0]) == 19968
        assert str(z["__lidar_sample_data_token__"]) == "sd_lidar_0"
        assert str(z["__frame__"]) == "ego"
        # load-bearing: nothing else in the repo has this basis, and a consumer
        # that guesses wrong produces a plausible, fully populated, wrong answer
        assert str(z["__basis__"]) == "raw_lidar_top_file_order"


def _forward_camera_record():
    """A camera->ego calibrated_sensor record looking down ego +x."""
    from pyquaternion import Quaternion
    r = np.array([[0.0, 0.0, 1.0],
                  [-1.0, 0.0, 0.0],
                  [0.0, -1.0, 0.0]])
    q = Quaternion(matrix=r)
    return {"translation": [0.0, 0.0, 0.0],
            "rotation": [q.w, q.x, q.y, q.z],
            "camera_intrinsic": [[500.0, 0.0, 640.0],
                                 [0.0, 500.0, 360.0],
                                 [0.0, 0.0, 1.0]]}


class _Obs:
    def __init__(self, channel, ego_pose_token, calibrated_sensor_token):
        self.channel = channel
        self.ego_pose_token = ego_pose_token
        self.calibrated_sensor_token = calibrated_sensor_token
        self.width_px, self.height_px = W, H


class TestPaintCameras:
    def test_no_cross_camera_suppression(self):
        """The deliberate absence of Stage 4's IoA-NMS: a road seen by every
        camera keeps every camera's claim, and n_cameras_road counts them."""
        identity_pose = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}
        cal = _forward_camera_record()
        ego_pose_table = {"ep0": identity_pose}
        calibrated_table = {"csA": cal, "csB": dict(cal)}
        rng = np.random.default_rng(3)
        pts = np.column_stack([
            rng.uniform(5, 20, 300), rng.uniform(-1, 1, 300),
            np.full(300, -2.0)])
        full = np.ones((H, W), bool)
        seen, road, n_cams = paint_cameras(
            pts,
            observations=[_Obs("CAM_FRONT", "ep0", "csA"), _Obs("CAM_BACK", "ep0", "csB")],
            road_mask_by_channel={"CAM_FRONT": full, "CAM_BACK": full},
            lidar_ego_pose=identity_pose,
            ego_pose_table=ego_pose_table,
            calibrated_table=calibrated_table,
        )
        assert road.size == 300          # every point claimed
        assert (n_cams == 2).all()       # by BOTH cameras, no contest
        assert set(road) <= set(seen)

    def test_a_masked_out_camera_claims_nothing(self):
        identity_pose = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}
        cal = _forward_camera_record()
        pts = np.array([[10.0, 0.0, -2.0]])
        seen, road, n_cams = paint_cameras(
            pts,
            observations=[_Obs("CAM_FRONT", "ep0", "csA")],
            road_mask_by_channel={"CAM_FRONT": np.zeros((H, W), bool)},
            lidar_ego_pose=identity_pose,
            ego_pose_table={"ep0": identity_pose},
            calibrated_table={"csA": cal},
        )
        assert seen.size == 1 and road.size == 0


class TestZedCarve:
    """The v2 refinement: ZED points sitting above the road plane carve the
    SAM road mask where the prompt leaked onto a sidewalk (C33 v2)."""

    def test_carves_around_raised_points_only(self):
        mask = np.zeros((100, 200), bool)
        mask[40:90, 20:180] = True                       # the SAM road claim
        raised_vu = np.array([[50, 30], [52, 34], [55, 31]])  # sidewalk support
        carved, n_px = carve_raised(mask, raised_vu, radius_px=5)
        assert not carved[50, 30] and not carved[55, 31]
        assert carved[50, 100]                            # far road untouched
        assert n_px == (mask & ~carved).sum() and n_px > 0

    def test_no_raised_points_is_identity(self):
        mask = np.zeros((50, 50), bool); mask[10:40, 10:40] = True
        carved, n_px = carve_raised(mask, np.zeros((0, 2), int), radius_px=5)
        assert n_px == 0 and np.array_equal(carved, mask)


class TestResolveSidewalk:
    """Overlap rules. Measured on the spike: on CAM_RIGHT the sidewalk prompt
    over-claims the entire road, so blind subtraction would delete real road —
    without geometry, ROAD keeps its pixels and sidewalk takes the rest."""

    def test_without_zed_road_keeps_contested_pixels(self):
        road = np.zeros((10, 10), bool); road[:, :6] = True
        sw = np.zeros((10, 10), bool); sw[:, 4:] = True   # over-claims the road
        got = resolve_sidewalk(road, sw)
        assert not (got & road).any()                      # disjoint
        assert got[:, 6:].all() and not got[:, :6].any()   # only the non-road part

    def test_carved_road_pixels_may_become_sidewalk(self):
        road = np.zeros((10, 10), bool); road[:, :6] = True
        carved = road.copy(); carved[:, 4:6] = False       # ZED carved these
        sw = np.zeros((10, 10), bool); sw[:, 4:] = True
        got = resolve_sidewalk(carved, sw)
        assert got[:, 4:6].all()                           # released to sidewalk


class TestMarkerComposition:
    """Copies 3b/3m/3c, NOT 4/5 — the road stage must not launder an accepted
    degraded upstream into a clean marker."""

    def test_upstream_causes_ride_with_prefix(self):
        degraded, causes = compose_marker(("chunk_0000: upstream sadness",), {})
        assert degraded is True
        assert causes == ("upstream: chunk_0000: upstream sadness",)

    def test_plane_refusals_are_a_cause(self):
        degraded, causes = compose_marker((), {"chunk_0000": 3})
        assert degraded is True
        assert any("3 keyframe(s) with no fittable road plane" in c for c in causes)

    def test_clean_is_clean(self):
        assert compose_marker((), {}) == (False, ())


def _stage1_tree(tmp_path, *, marker: str | None, causes=()):
    d = tmp_path / "stage1_ingestion"
    d.mkdir()
    with open(d / "run_manifest.json", "w") as fh:
        json.dump({"spec": "dhakascenes-pilot/stage1_ingest/v1", "stage": "stage1_ingestion"}, fh)
    if marker == "clean":
        write_marker(str(d), "fp0", degraded=False, causes=())
    elif marker == "degraded":
        write_marker(str(d), "fp0", degraded=True, causes=tuple(causes))
    return str(d)


class FakeSegmenter:
    def __call__(self, image):  # pragma: no cover - never reached in refusal tests
        raise AssertionError("segmenter must not run when the upstream gate refuses")


class TestRefusals:
    def test_refuses_absent_marker(self, tmp_path):
        src = _stage1_tree(tmp_path, marker=None)
        with pytest.raises(UpstreamRefusal):
            run(src, str(tmp_path / "stage_road"), dataroot=str(tmp_path),
                segmenter=FakeSegmenter())

    def test_refuses_degraded_upstream_without_flag(self, tmp_path):
        src = _stage1_tree(tmp_path, marker="degraded", causes=("chunk_0000: rejection_rate",))
        with pytest.raises(UpstreamRefusal) as exc:
            run(src, str(tmp_path / "stage_road"), dataroot=str(tmp_path),
                segmenter=FakeSegmenter())
        assert "--accept-degraded-upstream" in str(exc.value)

    def test_max_keyframes_refuses_a_non_empty_out_dir(self, tmp_path):
        # check.py's --max-rows guard, ported: a bounded probe pointed at a
        # COMPLETE tree would silently truncate it under the previous run's
        # marker (the marker is deliberately NOT written by a partial run,
        # and deliberately NOT cleared before this refusal)
        src = _stage1_tree(tmp_path, marker="clean")
        out = tmp_path / "stage_road"
        (out / "scenes" / "chunk_0000").mkdir(parents=True)
        with pytest.raises(UpstreamRefusal) as exc:
            run(src, str(out), dataroot=str(tmp_path),
                segmenter=FakeSegmenter(), max_keyframes=2)
        assert "--max-keyframes" in str(exc.value)
