"""stage6_stereo_box: one mask -> one box from stereo points, no clustering.
Synthetic: a box of known pose sampled as noisy stereo points, with a background
plane behind it; the MAD trim must reject the plane, the near face must sit at
the robust depth, the bottom on the ground plane, the yaw axis along the box."""
from __future__ import annotations
import math, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.stage6_stereo_box.stereo_box import box_from_stereo, DEFAULT_CFG, _clamp_extent  # noqa: E402

# front ZED: optical frame x-right y-down z-forward, mounted 0.8 m ahead, 0.7 m below ego origin
K = np.array([[953.16, 0, 656.28], [0, 953.16, 375.74], [0, 0, 1.0]])
R_opt_to_ego = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])   # optical -> body (x fwd, y left, z up)
T_EGO_CAM = np.eye(4); T_EGO_CAM[:3, :3] = R_opt_to_ego; T_EGO_CAM[:3, 3] = [0.8, 0.0, -0.7]
GROUND = (0.0, 0.0, -2.4)   # z = -2.4 everywhere
PRIOR = {"w": (1.15, 0.115), "l": (2.40, 0.24), "h": (1.75, 0.175)}   # (mu, sigma)


def _rickshaw(center=(12.0, 1.0), yaw=math.radians(20), n=800, seed=0):
    """VISIBLE surface of a 1.15 x 2.40 x 1.75 box standing on the ground, ego frame: the end face
    nearest the camera (60 % of points) plus the side face turned toward it (40 %) — what a stereo
    camera actually sees — with 15 cm range noise along the camera ray and a wall 6 m behind."""
    rng = np.random.default_rng(seed)
    w, l, h = 1.15, 2.40, 1.75
    n_end = int(0.6 * n); n_side = n - n_end
    u = np.concatenate([np.full(n_end, -l / 2), rng.uniform(-l / 2, l / 2, n_side)])
    v = np.concatenate([rng.uniform(-w / 2, w / 2, n_end), np.full(n_side, w / 2)])
    z = rng.uniform(0, h, n)
    xy = np.column_stack([u, v]) @ np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    pts = np.column_stack([xy[:, 0] + center[0], xy[:, 1] + center[1], z + GROUND[2]])
    ray = pts - T_EGO_CAM[:3, 3]; ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    pts += ray * rng.normal(0, 0.15, (n, 1))                       # range noise 15 cm
    wall = np.column_stack([np.full(120, center[0] + 6.0), rng.uniform(-2, 4, 120), rng.uniform(GROUND[2], GROUND[2] + 3, 120)])
    allp = np.vstack([pts, wall]); rings = np.full(len(allp), 101.0)
    return allp, rings


def test_clamp_extent_three_branches():
    """`_clamp_extent` (spec 4.2 step 5, controller ruling R20): a measurement
    below mu-k*sigma is unreliable, not small -> the prior mean; inside the
    band -> the measurement stands; above mu+k*sigma -> capped there."""
    mu, sigma, k = 1.0, 0.1, 2.0                                    # band = [0.8, 1.2]
    val, rule = _clamp_extent(0.5, mu, sigma, k)
    assert val == mu and rule == "low_to_mu"
    val, rule = _clamp_extent(1.05, mu, sigma, k)
    assert val == 1.05 and rule is None
    val, rule = _clamp_extent(2.0, mu, sigma, k)
    assert val == mu + k * sigma and rule == "high"


def test_recovers_pose_and_rejects_wall():
    pts, rings = _rickshaw()
    box, status, st = box_from_stereo(pts, rings, K=K, T_ego_cam=T_EGO_CAM, prior=PRIOR, ground_abd=GROUND, cfg=DEFAULT_CFG)
    assert status == "fit" and box is not None
    assert st["n_stereo_kept"] < st["n_stereo_pts"]                 # the wall was trimmed
    tx, ty, tz = box["translation_m"]
    # 0.4 m: 15 cm range noise plus the p20-vs-true-near-face residual on an oblique view
    assert abs(tx - 12.0) < 0.4 and abs(ty - 1.0) < 0.4, (tx, ty)
    w, l, h = box["size_wlh_m"]
    assert l == 2.40 and 0.9 <= w <= 1.4 and 1.4 <= h <= 2.1
    assert abs(box["z_min_m"] - GROUND[2]) < 1e-9 and abs(box["z_max_m"] - (GROUND[2] + h)) < 1e-9
    yaw = box["yaw_rad"] % math.pi
    # 15 deg: the principal axis of an L-shaped (end + side) footprint is biased toward the long leg
    d = abs(yaw - math.radians(20)) % math.pi
    assert min(d, math.pi - d) < math.radians(15), yaw
    assert box["yaw_axis_only"] is True and box["yaw_ambiguous"] is True
    assert box["size_order"] == "w,l,h" and w <= l
    # clamp direction recorded per axis (task 5c): present and shaped {w, h}
    assert set(st["clamp"]) == {"w", "h"}
    assert st["clamp"]["w"] in (None, "low_to_mu", "high")
    assert st["clamp"]["h"] in (None, "low_to_mu", "high")


def test_too_few_points_and_beyond_cap():
    pts, rings = _rickshaw(n=10)
    box, status, _ = box_from_stereo(pts[:10], rings[:10], K=K, T_ego_cam=T_EGO_CAM, prior=PRIOR, ground_abd=GROUND, cfg=DEFAULT_CFG)
    assert box is None and status == "too_few_stereo"
    pts, rings = _rickshaw(center=(30.0, 0.0))
    box, status, _ = box_from_stereo(pts, rings, K=K, T_ego_cam=T_EGO_CAM, prior=PRIOR, ground_abd=GROUND, cfg={**DEFAULT_CFG, "stereo_range_cap_m": 25.0})
    assert box is None and status == "beyond_stereo_cap"



def test_side_on_pushes_by_the_half_width_not_the_half_length():
    """Crossing traffic. At yaw 90 deg the camera sees the 2.40 m SIDE face, so the
    distance from the near face to the centre is w/2 (~0.6 m), not l/2 (1.2 m).
    Under the old `push l/2 along the ray` rule the box lands at x = 12.578 against
    a truth of 12.0 — 0.58 m too far away, outside the brief's 0.4 m bar."""
    pts, rings = _rickshaw(yaw=math.radians(90))
    box, status, st = box_from_stereo(pts, rings, K=K, T_ego_cam=T_EGO_CAM, prior=PRIOR,
                                      ground_abd=GROUND, cfg=DEFAULT_CFG)
    assert status == "fit"
    tx, ty, _ = box["translation_m"]
    assert abs(tx - 12.0) < 0.4 and abs(ty - 1.0) < 0.4, (tx, ty)
    assert 80.0 <= st["theta_deg"] <= 100.0, st["theta_deg"]        # the ray crosses the length axis
    assert st["push_m"] < 0.9, st["push_m"]                         # ~w/2, nowhere near l/2 = 1.2

def test_lidar_refines_depth_when_present():
    pts, rings = _rickshaw()
    # add 8 LiDAR points on the near face, 0.4 m closer than the (biased) stereo median would say
    near = pts[:8].copy(); near[:, 0] -= 0.4
    allp = np.vstack([pts, near]); allr = np.concatenate([rings, np.zeros(8)])
    _, status, st = box_from_stereo(allp, allr, K=K, T_ego_cam=T_EGO_CAM, prior=PRIOR, ground_abd=GROUND, cfg=DEFAULT_CFG)
    assert status == "fit" and st["depth_source"] == "lidar_refined"
    # NOT asserted: n_lidar_in_box >= 1. These 8 points are copies of synthetic
    # surface points that sit exactly on the box's true length-axis boundary
    # (u = -l/2), so whether they land inside the FITTED box is a coin flip on
    # sub-cm noise, not a property of lidar refinement. Under task 5c (p1/p99 +
    # asymmetric clamp) w_meas grew slightly, the ray-projection push grew with
    # it, and the box centre moved ~1.7 cm further along the ray -- enough to
    # flip the one point that used to land inside (n_lidar_in_box: 1 -> 0 on
    # this fixture/seed). depth_source is the property this test is for.


def test_pedestrian_prior_keeps_w_le_l_by_swapping():
    ped = {"w": (0.77, 0.077), "l": (0.76, 0.076), "h": (1.72, 0.172)}
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.normal(8.0, 0.15, 300), rng.normal(0.0, 0.3, 300), rng.uniform(GROUND[2], GROUND[2] + 1.7, 300)])
    box, status, _ = box_from_stereo(pts, np.full(300, 101.0), K=K, T_ego_cam=T_EGO_CAM, prior=ped, ground_abd=GROUND, cfg=DEFAULT_CFG)
    assert status == "fit"
    w, l, _ = box["size_wlh_m"]
    assert w <= l



class _StubPriors:
    """The two `Priors` methods this stage calls, and nothing else."""

    class _ClassPrior:
        dims = {"w": {"mu": 1.15, "sigma": 0.115}, "l": {"mu": 2.40, "sigma": 0.24},
                "h": {"mu": 1.75, "sigma": 0.175}}

        def mu(self, axis):
            return self.dims[axis]["mu"]

        def sigma(self, axis):
            return self.dims[axis]["sigma"]

    def get(self, class_name):
        return self._ClassPrior() if class_name == "rickshaw" else None

    def eps_bev(self, class_name, *, fallback_m):
        return fallback_m, f"config_fallback:class_absent_from_priors:{class_name}"


def _keyframe_fixture(tmp_path, instances, n_copies):
    """A Stage 5 keyframe on disk: one cloud, one npz, one lift row."""
    pts, rings = _rickshaw()
    cloud = np.zeros((len(pts), 5), dtype=np.float32)
    cloud[:, :3] = pts
    cloud[:, 4] = rings
    cloud_path = tmp_path / "cloud.pcd.bin"
    cloud.tofile(str(cloud_path))
    idx = np.arange(len(pts), dtype=np.int64)
    np.savez(tmp_path / "kf.npz", point_index=np.concatenate([idx] * n_copies),
             instance_id=np.repeat([i["instance_id"] for i in instances], len(pts)))
    return {"keyframe_token": "kf", "scene_token": "sc", "t_ns": 1, "time_base": "unix_ns",
            "coverage_config": "R3", "cloud_path": str(cloud_path), "points_path": "kf.npz",
            "instances": instances}


def test_status_per_instance_and_the_disabled_front_channel(tmp_path):
    """box_keyframe's routing: R3 candidacy, active_channels, prior, then the fit.

    CAM_FRONT is a ZED channel but is NOT in active_channels on this run (its
    export extrinsics are pitched), so its instances must come back
    `channel_disabled` with no box — not `fit`, and not silently dropped."""
    from pipeline.stage6_stereo_box.stereo_box import DEFAULT_CFG, box_keyframe

    instances = [{"instance_id": i, "channel": ch, "proposal_index": i, "class_name": cls,
                  "score": 0.9, "n_mask_px": 500}
                 for i, ch, cls in [(1, "CAM_BACK", "rickshaw"), (2, "CAM_FRONT", "rickshaw"),
                                    (3, "CAM_FRONT_LEFT", "rickshaw"), (4, "CAM_BACK", "unknown_class")]]
    lift_row = _keyframe_fixture(tmp_path, instances, 4)
    calibs = {"CAM_BACK": (K, T_EGO_CAM), "CAM_FRONT": (K, T_EGO_CAM)}
    rows, totals = box_keyframe(lift_row, str(tmp_path), calibs, {"kf": GROUND}, _StubPriors(),
                                {**DEFAULT_CFG, "active_channels": ["CAM_BACK"]})

    assert [r["status"] for r in rows] == ["fit", "channel_disabled", "out_of_r3", "no_prior"]
    assert rows[0]["box"] is not None and rows[0]["num_lidar_pts"] > 0
    assert rows[0]["stereo"]["n_stereo_in_box"] > 0 and rows[0]["stereo"]["zed_ring"] == 101
    assert all(r["box"] is None and r["stereo"] is None for r in rows[1:])
    assert totals["n_fit"] == 1 and totals["n_channel_disabled"] == 1
    assert totals["n_out_of_r3"] == 1 and totals["n_no_prior"] == 1 and totals["n_instances"] == 4
    # the row envelope Stages 7/8/9 read is Stage 6's, unchanged
    for key in ("spec", "keyframe_token", "scene_token", "t_ns", "time_base", "coverage_config",
                "cloud_path", "points_path", "eps_m", "eps_source", "min_samples", "cluster",
                "near_cut", "num_lidar_pts", "num_lidar_pts_basis", "n_points_below_gate", "frame"):
        assert key in rows[0], key


def test_missing_ground_plane_is_recorded_not_guessed(tmp_path):
    """The box bottom is SNAPPED to Stage 1's plane, so a missing one is a status."""
    from pipeline.stage6_stereo_box.stereo_box import DEFAULT_CFG, box_keyframe

    instances = [{"instance_id": 1, "channel": "CAM_BACK", "proposal_index": 0,
                  "class_name": "rickshaw", "score": 0.9, "n_mask_px": 500}]
    lift_row = _keyframe_fixture(tmp_path, instances, 1)
    rows, totals = box_keyframe(lift_row, str(tmp_path), {"CAM_BACK": (K, T_EGO_CAM)}, {},
                                _StubPriors(), {**DEFAULT_CFG, "active_channels": ["CAM_BACK"]})
    assert rows[0]["status"] == "no_ground_plane" and rows[0]["box"] is None
    assert totals["n_no_ground_plane"] == 1
