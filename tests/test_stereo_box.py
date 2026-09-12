"""stage6_stereo_box: one mask -> one box from stereo points, no clustering.
Synthetic: a box of known pose sampled as noisy stereo points, with a background
plane behind it; the MAD trim must reject the plane, the near face must sit at
the robust depth, the bottom on the ground plane, the yaw axis along the box."""
from __future__ import annotations
import math, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.common.schemas import IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX  # noqa: E402
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
    box, status, st = box_from_stereo(pts[:10], rings[:10], K=K, T_ego_cam=T_EGO_CAM, prior=PRIOR, ground_abd=GROUND, cfg=DEFAULT_CFG)
    assert box is None and status == "too_few_stereo"
    # a non-fit row still carries `clamp` with the same {w, h} shape, unset (task 5c fix round 1)
    assert st["clamp"] == {"w": None, "h": None}
    pts, rings = _rickshaw(center=(30.0, 0.0))
    box, status, st = box_from_stereo(pts, rings, K=K, T_ego_cam=T_EGO_CAM, prior=PRIOR, ground_abd=GROUND, cfg={**DEFAULT_CFG, "stereo_range_cap_m": 25.0})
    assert box is None and status == "beyond_stereo_cap"
    assert st["range_gate_m"] > 25.0, st["range_gate_m"]             # the FACE is past the cap


def test_range_gate_tests_the_near_face_not_the_prior_extrapolated_centre():
    """Controller ruling R25. A bus turned to face the camera has its rear face at
    21 m — well inside the 25 m cap, and that face is what the stereo actually
    measured — but the class prior then pushes the centre to ~26.6 m. Gating the
    centre threw such a box away for having a LONG prior, not for having bad
    points, and took 70 of chunk_0010's 114 fitted bus boxes with it."""
    pts, rings = _flat_face(2.9, 3.3, depth=21.0 - T_EGO_CAM[0, 3])
    box, status, st = box_from_stereo(pts, rings, K=K, T_ego_cam=T_EGO_CAM, prior=BUS_PRIOR,
                                      ground_abd=GROUND, cfg=DEFAULT_CFG)
    assert status == "fit", (status, st["range_gate_m"])
    assert st["single_face"]["matched"] == "w" and st["push_m"] > 5.0
    assert abs(st["range_gate_m"] - 21.0) < 0.5, st["range_gate_m"]  # gated on the face
    assert math.hypot(*box["translation_m"][:2]) > 25.0              # the centre is past the cap



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
    assert st["n_stereo_in_box"] > 0                                # in-box counting stays exercised
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


BUS_PRIOR = {"w": (2.96, 0.296), "l": (11.19, 1.119), "h": (3.44, 0.344)}   # priors.json, "a bus"


def _flat_face(width, height, depth=15.0, n=800, jitter=0.05, seed=3):
    """ONE face of an object seen straight on: a vertical plane `width` m across and
    `height` m tall at `depth` m in front of the camera, jittered in depth only.

    No second face, so the footprint is a STRIP whose longer extent is the face's
    width — not the object's length. This is the head-on bus of keyframe 575,
    chunk_0010: "the longer visible extent is the length axis" laid an 11.19 m
    prior ACROSS the road."""
    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        T_EGO_CAM[0, 3] + depth + rng.normal(0, jitter, n),
        rng.uniform(-width / 2, width / 2, n),
        rng.uniform(0, height, n) + GROUND[2],
    ])
    return pts, np.full(n, 100.0)


def test_single_face_head_on_bus_lays_the_length_along_the_ray():
    """A bus rear face (2.9 m wide, 3.3 m tall) at 15 m, nothing else visible.
    The visible strip's width matches mu_w (2.96) far better than mu_l (11.19),
    so it is the REAR face and the length axis runs perpendicular to it — i.e.
    along the viewing ray. Under the pre-fix rule ("the longer visible extent is
    the length axis") the 11.19 m length is laid across the road instead."""
    pts, rings = _flat_face(2.9, 3.3)
    box, status, st = box_from_stereo(pts, rings, K=K, T_ego_cam=T_EGO_CAM, prior=BUS_PRIOR,
                                      ground_abd=GROUND, cfg=DEFAULT_CFG)
    assert status == "fit"
    assert box["fit"]["yaw_source"] == "single_face_prior_match", box["fit"]["yaw_source"]
    assert st["single_face"]["matched"] == "w", st["single_face"]
    assert st["single_face"]["e_minor_m"] < st["single_face"]["e_major_m"]
    assert st["frame_truncated"] is False                           # fully in frame
    d = abs(box["yaw_rad"] % math.pi - st["ray_yaw_rad"] % math.pi) % math.pi
    assert min(d, math.pi - d) < math.radians(15), (box["yaw_rad"], st["ray_yaw_rad"])
    assert abs(st["push_m"] - 11.19 / 2) < 0.5, st["push_m"]        # l/2, not w/2


def test_single_face_side_only_rickshaw_keeps_the_length_across_the_ray():
    """The other single-face case: only the 2.40 m SIDE is visible. Its width
    matches mu_l, so the strip IS the length axis and yaw stays lateral."""
    pts, rings = _flat_face(2.4, 1.6)
    box, status, st = box_from_stereo(pts, rings, K=K, T_ego_cam=T_EGO_CAM, prior=PRIOR,
                                      ground_abd=GROUND, cfg=DEFAULT_CFG)
    assert status == "fit"
    assert st["single_face"]["matched"] == "l", st["single_face"]
    d = abs(box["yaw_rad"] % math.pi - math.pi / 2) % math.pi        # lateral: across the ray
    assert min(d, math.pi - d) < math.radians(15), box["yaw_rad"]


CAR_PRIOR = {"w": (1.93, 0.193), "l": (4.63, 0.463), "h": (1.56, 0.156)}   # priors.json, "a car"


def _truncated_car_side(depth=3.5, visible_m=1.28, y=-2.4, n=800, jitter=0.05, seed=5):
    """The right-edge car of keyframe 895d7483, CAM_FRONT, chunk_0010: only 1.28 m
    of a 4.63 m car's SIDE is inside the frame, at 3.5 m. The length axis runs
    along ego x (yaw 0) and the strip runs off the RIGHT image border, so its
    extent is a LOWER bound — 1.28 m is closer in log-ratio to mu_w (1.93) than to
    mu_l (4.63) and the single-face width match calls the side a rear face."""
    rng = np.random.default_rng(seed)
    x0 = T_EGO_CAM[0, 3] + depth
    pts = np.column_stack([
        rng.uniform(x0, x0 + visible_m, n),
        y + rng.normal(0, jitter, n),
        rng.uniform(0, 1.56, n) + GROUND[2],
    ])
    return pts, np.full(n, 101.0)


def _u_range(pts):
    """The fixture's projected column range, so the test's premise is measured."""
    cam = (np.linalg.inv(T_EGO_CAM) @ np.column_stack([pts, np.ones(len(pts))]).T).T[:, :3]
    u = K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]
    return float(u.min()), float(u.max())


def test_frame_truncated_single_face_falls_back_to_ego_forward():
    """A frame-truncated object's visible extent is a LOWER bound, so the
    single-face width match is invalid. `truncated=True` must skip it and fall
    back to ego forward; `truncated=False` must still reproduce the defect, so
    the guard is pinned to the case it was written for."""
    pts, rings = _truncated_car_side()
    assert _u_range(pts)[1] >= IMAGE_WIDTH_PX - 1 - DEFAULT_CFG["truncation_margin_px"], _u_range(pts)

    box, status, st = box_from_stereo(pts, rings, K=K, T_ego_cam=T_EGO_CAM, prior=CAR_PRIOR,
                                      ground_abd=GROUND, cfg=DEFAULT_CFG)
    assert status == "fit" and st["single_face"]["matched"] == "w", st["single_face"]
    d = box["yaw_rad"] % math.pi
    assert min(d, math.pi - d) > math.radians(15), box["yaw_rad"]     # the defect: across the lane
    assert st["frame_truncated"] is False

    box, status, st = box_from_stereo(pts, rings, K=K, T_ego_cam=T_EGO_CAM, prior=CAR_PRIOR,
                                      ground_abd=GROUND, cfg=DEFAULT_CFG, truncated=True)
    assert status == "fit"
    assert box["fit"]["yaw_source"] == "truncated_ego_forward", box["fit"]["yaw_source"]
    assert "frame_truncated" in box["yaw_ambiguous_reasons"], box["yaw_ambiguous_reasons"]
    assert st["frame_truncated"] is True
    d = box["yaw_rad"] % math.pi
    assert min(d, math.pi - d) < math.radians(15), box["yaw_rad"]     # ego forward, along the lane


def _write_masks(tmp_path, channel, spans):
    """A Stage 4 mask npz: one bit-packed mask per (col_lo, col_hi) span."""
    stack = np.zeros((len(spans), IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX), dtype=bool)
    for i, (lo, hi) in enumerate(spans):
        stack[i, 100:200, lo:hi + 1] = True
    np.savez(str(tmp_path / "masks.npz"), **{
        channel: np.packbits(stack, axis=-1),
        "__width_px__": np.asarray([IMAGE_WIDTH_PX], dtype=np.int32),
        "__height_px__": np.asarray([IMAGE_HEIGHT_PX], dtype=np.int32),
        "__bit_packed__": np.asarray([1], dtype=np.int8),
    })
    return "masks.npz"


def test_driver_reads_the_truncation_flag_from_the_mask(tmp_path):
    """box_keyframe decides truncation from the INSTANCE's own mask columns: one
    touching the last column is truncated, a centred one is not."""
    from pipeline.stage6_stereo_box.stereo_box import DEFAULT_CFG, box_keyframe

    instances = [{"instance_id": i, "channel": "CAM_BACK", "proposal_index": i - 1,
                  "class_name": "rickshaw", "score": 0.9, "n_mask_px": 500} for i in (1, 2)]
    lift_row = _keyframe_fixture(tmp_path, instances, 2)
    lift_row["mask_path"] = _write_masks(tmp_path, "CAM_BACK",
                                         [(1100, IMAGE_WIDTH_PX - 1), (600, 700)])
    rows, totals = box_keyframe(lift_row, str(tmp_path), {"CAM_BACK": (K, T_EGO_CAM)},
                                {"kf": GROUND}, _StubPriors(),
                                {**DEFAULT_CFG, "active_channels": ["CAM_BACK"]})
    assert [r["status"] for r in rows] == ["fit", "fit"]
    assert [r["stereo"]["truncation_source"] for r in rows] == ["mask", "mask"]
    assert rows[0]["stereo"]["frame_truncated"] is True
    assert rows[1]["stereo"]["frame_truncated"] is False
    assert "n_truncated_yaw" in totals


def test_driver_falls_back_to_the_points_when_the_mask_is_unreadable(tmp_path):
    """No mask file -> the owned points' projected u-range, recorded as such."""
    from pipeline.stage6_stereo_box.stereo_box import DEFAULT_CFG, box_keyframe

    instances = [{"instance_id": 1, "channel": "CAM_BACK", "proposal_index": 0,
                  "class_name": "rickshaw", "score": 0.9, "n_mask_px": 500}]
    lift_row = _keyframe_fixture(tmp_path, instances, 1)
    lift_row["mask_path"] = "no_such_masks.npz"
    rows, _ = box_keyframe(lift_row, str(tmp_path), {"CAM_BACK": (K, T_EGO_CAM)},
                           {"kf": GROUND}, _StubPriors(),
                           {**DEFAULT_CFG, "active_channels": ["CAM_BACK"]})
    assert rows[0]["status"] == "fit"
    assert rows[0]["stereo"]["truncation_source"] == "points"
    assert rows[0]["stereo"]["frame_truncated"] is False           # the rickshaw is mid-frame


def test_known_gaps_sentence_reflects_active_channels():
    """The manifest's first known_gaps sentence must not lie about which ZED channel
    is boxed: it names the disabled channel when one is excluded, and says CAM_FRONT
    IS boxed (with the pitch-error caveat) once both channels are active."""
    from pipeline.stage6_stereo_box.stereo_box import _stereo_channel_gap_sentence

    only_back = _stereo_channel_gap_sentence(["CAM_BACK"])
    assert "CAM_FRONT" in only_back
    assert "NOT boxed" in only_back
    assert "channel_disabled" in only_back

    both = _stereo_channel_gap_sentence(["CAM_FRONT", "CAM_BACK"])
    assert "CAM_FRONT" in both
    assert "IS boxed" in both
    assert "NOT boxed" not in both


def test_n_fit_by_channel_tallies_per_channel_fit_counts():
    """totals["n_fit_by_channel"] is the manifest's numeric evidence for the
    known_gaps prose: it must count fitted boxes per ZED channel, not just overall."""
    from pipeline.stage6_stereo_box.stereo_box import _accumulate_fit_by_channel

    n_fit_by_channel = {"CAM_FRONT": 0, "CAM_BACK": 0}
    rows = [
        {"status": "fit", "channel": "CAM_FRONT"},
        {"status": "fit", "channel": "CAM_FRONT"},
        {"status": "fit", "channel": "CAM_BACK"},
        {"status": "channel_disabled", "channel": "CAM_FRONT"},
        {"status": "out_of_r3", "channel": "CAM_SIDE"},
    ]
    _accumulate_fit_by_channel(n_fit_by_channel, rows)
    assert n_fit_by_channel == {"CAM_FRONT": 2, "CAM_BACK": 1}
