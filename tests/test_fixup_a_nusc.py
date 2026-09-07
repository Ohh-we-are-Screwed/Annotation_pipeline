"""scripts/fixup_a_nusc.py — drop keyframes that lack a required channel.

The 2026-09-04 export (Dataset/A_nusc) has ~5 % of keyframes with at least one
camera image missing (CAM_LEFT worst). Stage 0's channels_complete is
all-or-nothing per SCENE and each chunk is one scene, so one such keyframe
refuses 750; Stage 1 would KeyError on the missing channel if it got that far.
The pilot's own precedent is a data-side fixup into a NEW version dir beside
the original (v1.0-dhaka-fixed, 2026-08-30: "drop the 2 CAM_RIGHT-less
samples"). This is that, generalised, and non-destructive: the source version
dir is never written.
"""

from __future__ import annotations

import copy
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.fixup_a_nusc import TABLES, drop_incomplete_samples, main  # noqa: E402

REQ = ("LIDAR_TOP", "CAM_A", "CAM_B")


def _tables(missing: dict[str, tuple[str, ...]] | None = None, n: int = 3) -> dict[str, list]:
    """n chained samples s1..sn, each with a keyframe sample_data row per channel
    except the channels named in `missing` for that sample."""
    missing = missing or {}
    sensors = [{"token": f"sen_{c}", "channel": c, "modality": "lidar" if c == "LIDAR_TOP" else "camera"} for c in REQ]
    cal = [{"token": f"cs_{c}", "sensor_token": f"sen_{c}"} for c in REQ]
    samples, sd = [], []
    for i in range(1, n + 1):
        tok = f"s{i}"
        samples.append({"token": tok, "timestamp": 1000 * i, "scene_token": "sc",
                        "prev": f"s{i-1}" if i > 1 else "", "next": f"s{i+1}" if i < n else ""})
        for c in REQ:
            if c in missing.get(tok, ()):
                continue
            sd.append({"token": f"{tok}_{c}", "sample_token": tok, "calibrated_sensor_token": f"cs_{c}",
                       "ego_pose_token": f"ep_{tok}", "timestamp": 1000 * i, "is_key_frame": True,
                       "filename": f"samples/{c}/{i:06d}.x", "prev": "", "next": ""})
    scene = [{"token": "sc", "name": "chunk_x", "nbr_samples": n,
              "first_sample_token": "s1", "last_sample_token": f"s{n}", "log_token": "lg", "description": ""}]
    t = {name: [] for name in TABLES}
    t.update({"sensor": sensors, "calibrated_sensor": cal, "sample": samples, "sample_data": sd, "scene": scene,
              "ego_pose": [{"token": f"ep_s{i}", "timestamp": 1000 * i, "translation": [float(i), 0.0, 0.0],
                            "rotation": [1.0, 0.0, 0.0, 0.0], "identity": False} for i in range(1, n + 1)],
              "log": [{"token": "lg"}]})
    return t


class TestDropIncompleteSamples:
    def test_complete_tables_are_returned_unchanged_and_nothing_dropped(self):
        t = _tables()
        out, dropped = drop_incomplete_samples(t, REQ)
        assert dropped == []
        assert out == t

    def test_input_is_not_mutated(self):
        t = _tables(missing={"s2": ("CAM_B",)})
        before = copy.deepcopy(t)
        drop_incomplete_samples(t, REQ)
        assert t == before

    def test_middle_sample_missing_one_camera_is_dropped_and_chain_relinked(self):
        out, dropped = drop_incomplete_samples(_tables(missing={"s2": ("CAM_B",)}), REQ)
        assert dropped == ["s2"]
        by = {s["token"]: s for s in out["sample"]}
        assert list(by) == ["s1", "s3"]
        assert (by["s1"]["prev"], by["s1"]["next"]) == ("", "s3")
        assert (by["s3"]["prev"], by["s3"]["next"]) == ("s1", "")

    def test_dropped_samples_lose_every_sample_data_row(self):
        out, _ = drop_incomplete_samples(_tables(missing={"s2": ("CAM_B",)}), REQ)
        assert {r["sample_token"] for r in out["sample_data"]} == {"s1", "s3"}
        # the rows that DID exist for s2 (its lidar and CAM_A) are gone too
        assert not any(r["token"].startswith("s2_") for r in out["sample_data"])

    def test_scene_counts_and_endpoints_follow(self):
        out, _ = drop_incomplete_samples(_tables(missing={"s1": ("LIDAR_TOP",), "s3": ("CAM_A", "CAM_B")}), REQ)
        (scene,) = out["scene"]
        assert scene["nbr_samples"] == 1
        assert (scene["first_sample_token"], scene["last_sample_token"]) == ("s2", "s2")
        (s2,) = out["sample"]
        assert (s2["prev"], s2["next"]) == ("", "")

    def test_a_row_for_an_unrequired_channel_does_not_make_a_sample_complete(self):
        # RADAR-style extra channels are ignored in both directions: their
        # presence never substitutes for a missing required one.
        t = _tables(missing={"s2": ("CAM_B",)})
        t["sensor"].append({"token": "sen_R", "channel": "RADAR", "modality": "radar"})
        t["calibrated_sensor"].append({"token": "cs_R", "sensor_token": "sen_R"})
        t["sample_data"].append({"token": "s2_R", "sample_token": "s2", "calibrated_sensor_token": "cs_R",
                                 "ego_pose_token": "ep_s2", "timestamp": 2000, "is_key_frame": True,
                                 "filename": "samples/RADAR/2.x", "prev": "", "next": ""})
        _, dropped = drop_incomplete_samples(t, REQ)
        assert dropped == ["s2"]

    # --- a referenced blob that is not on disk is a missing channel ------------
    # chunk_0006's sample_data names samples/CAM_FRONT/000100.jpg, which the
    # exporter never wrote; Stage 0's files_resolve fails the scene on it.
    def test_a_channel_whose_blob_is_missing_counts_as_missing(self):
        t = _tables()
        gone = {"samples/CAM_B/000002.x"}
        out, dropped = drop_incomplete_samples(t, REQ, blob_exists=lambda fn: fn not in gone)
        assert dropped == ["s2"]
        assert [s["token"] for s in out["sample"]] == ["s1", "s3"]

    def test_blob_check_defaults_to_trusting_the_tables(self):
        _, dropped = drop_incomplete_samples(_tables(), REQ)
        assert dropped == []

    def test_untouched_tables_pass_through_identically(self):
        t = _tables(missing={"s2": ("CAM_B",)})
        out, _ = drop_incomplete_samples(t, REQ)
        for name in TABLES:
            if name not in ("sample", "sample_data", "scene"):
                assert out[name] == t[name], name


class TestMain:
    def _write_root(self, tmp_path, tables, version="v1.0-x", blobs=True):
        """Tables on disk, and (by default) a real file for every sample_data
        row — main() checks blob existence, so a root with no blobs would
        drop everything."""
        vd = tmp_path / version
        vd.mkdir()
        for name in TABLES:
            (vd / f"{name}.json").write_text(json.dumps(tables[name]))
        (tmp_path / "samples").mkdir()
        if blobs:
            for row in tables["sample_data"]:
                p = tmp_path / row["filename"]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"x")
        return tmp_path

    def test_writes_a_new_version_dir_with_all_tables_and_leaves_the_source_alone(self, tmp_path):
        t = _tables(missing={"s2": ("CAM_B",)})
        root = self._write_root(tmp_path, t)
        src_before = {n: (root / "v1.0-x" / f"{n}.json").read_bytes() for n in TABLES}
        rc = main(["--dataroot", str(root), "--version", "v1.0-x", "--out-version", "v1.0-x-fixed",
                   "--required-channels", *REQ])
        assert rc == 0
        out = root / "v1.0-x-fixed"
        assert {f"{n}.json" for n in TABLES} <= {p.name for p in out.iterdir()}
        assert (out / "fixup_meta.json").is_file()  # beside the tables, never one of them
        assert [s["token"] for s in json.loads((out / "sample.json").read_text())] == ["s1", "s3"]
        assert {n: (root / "v1.0-x" / f"{n}.json").read_bytes() for n in TABLES} == src_before

    def test_main_drops_samples_whose_blobs_are_absent_from_the_dataroot(self, tmp_path):
        # Tables are complete; only s2's CAM_B file is missing on disk.
        t = _tables()
        root = self._write_root(tmp_path, t)
        (root / "samples/CAM_B/000002.x").unlink()
        rc = main(["--dataroot", str(root), "--version", "v1.0-x", "--out-version", "v1.0-x-fixed",
                   "--required-channels", *REQ])
        assert rc == 0
        assert [s["token"] for s in json.loads((root / "v1.0-x-fixed" / "sample.json").read_text())] == ["s1", "s3"]

    def test_refuses_to_overwrite_an_existing_out_version(self, tmp_path):
        root = self._write_root(tmp_path, _tables())
        (root / "v1.0-x-fixed").mkdir()
        rc = main(["--dataroot", str(root), "--version", "v1.0-x", "--out-version", "v1.0-x-fixed",
                   "--required-channels", *REQ])
        assert rc == 2

    def test_refuses_when_every_sample_would_be_dropped(self, tmp_path):
        root = self._write_root(tmp_path, _tables(missing={f"s{i}": ("CAM_A",) for i in (1, 2, 3)}))
        rc = main(["--dataroot", str(root), "--version", "v1.0-x", "--out-version", "v1.0-x-fixed",
                   "--required-channels", *REQ])
        assert rc == 2
        assert not (root / "v1.0-x-fixed").exists()


# ---------------------------------------------------------------------------
# Per-camera ego poses. The exporter hands every sensor of a sample the ONE
# ego_pose of its LiDAR, so |ego(t_cam) - ego(t_lidar)| is exactly 0 and
# Stage 5 refuses the scene ("the chain silently omits ego motion between
# capture times", §1.3). The pilot's fixup rebuilt ego_pose rows per sensor;
# this interpolates the LiDAR ego trajectory at each camera's own timestamp —
# translation linearly, rotation by slerp — and gives the row its own token.
# ---------------------------------------------------------------------------
from scripts.fixup_a_nusc import interpolate_camera_ego_poses  # noqa: E402


def _traj_tables(cam_offsets_us: dict[str, int], yaw_deg_per_sample=(0, 0, 0)):
    """3 samples, LiDAR at t = 0, 1000, 2000 us moving +1 m/x per sample; one
    camera CAM_A whose timestamp is lidar + cam_offsets_us[sample]."""
    import math
    t = _tables(n=3)
    t["sample_data"] = [r for r in t["sample_data"] if "CAM_B" not in r["filename"]]
    t["sensor"] = [s for s in t["sensor"] if s["channel"] != "CAM_B"]
    t["calibrated_sensor"] = [c for c in t["calibrated_sensor"] if c["token"] != "cs_CAM_B"]
    t["ego_pose"] = []
    for i in range(3):
        tok = f"s{i+1}"
        half = math.radians(yaw_deg_per_sample[i]) / 2
        t["ego_pose"].append({"token": f"ep_{tok}", "timestamp": 1000 * i,
                              "translation": [float(i), 0.0, 0.0],
                              "rotation": [math.cos(half), 0.0, 0.0, math.sin(half)], "identity": False})
        for r in t["sample_data"]:
            if r["sample_token"] != tok:
                continue
            r["ego_pose_token"] = f"ep_{tok}"
            r["timestamp"] = 1000 * i + (cam_offsets_us.get(tok, 0) if "CAM_A" in r["filename"] else 0)
    return t


def _cam_rows(tables):
    return [r for r in tables["sample_data"] if "CAM_A" in r["filename"]]


class TestInterpolateCameraEgoPoses:
    def test_every_camera_row_gets_its_own_pose_at_its_own_timestamp(self):
        out, n = interpolate_camera_ego_poses(_traj_tables({"s1": 200, "s2": 200, "s3": 200}))
        assert n == 3
        ep = {e["token"]: e for e in out["ego_pose"]}
        lidar_tokens = {r["ego_pose_token"] for r in out["sample_data"] if "LIDAR_TOP" in r["filename"]}
        for r in _cam_rows(out):
            assert r["ego_pose_token"] not in lidar_tokens
            assert ep[r["ego_pose_token"]]["timestamp"] == r["timestamp"]
        assert len(out["ego_pose"]) == 3 + 3

    def test_translation_is_linear_between_bracketing_lidar_poses(self):
        out, _ = interpolate_camera_ego_poses(_traj_tables({"s1": 200, "s2": 500}))
        ep = {e["token"]: e for e in out["ego_pose"]}
        by = {r["sample_token"]: ep[r["ego_pose_token"]] for r in _cam_rows(out)}
        assert by["s1"]["translation"] == pytest.approx([0.2, 0.0, 0.0])
        assert by["s2"]["translation"] == pytest.approx([1.5, 0.0, 0.0])

    def test_edge_of_scene_extrapolates_along_the_last_segment(self):
        out, _ = interpolate_camera_ego_poses(_traj_tables({"s3": 300, "s1": -100}))
        ep = {e["token"]: e for e in out["ego_pose"]}
        by = {r["sample_token"]: ep[r["ego_pose_token"]] for r in _cam_rows(out)}
        assert by["s3"]["translation"] == pytest.approx([2.3, 0.0, 0.0])
        assert by["s1"]["translation"] == pytest.approx([-0.1, 0.0, 0.0])

    def test_rotation_is_slerped(self):
        import math
        out, _ = interpolate_camera_ego_poses(_traj_tables({"s1": 500}, yaw_deg_per_sample=(0, 90, 90)))
        ep = {e["token"]: e for e in out["ego_pose"]}
        (r,) = [r for r in _cam_rows(out) if r["sample_token"] == "s1"]
        w, x, y, z = ep[r["ego_pose_token"]]["rotation"]
        yaw = math.degrees(2 * math.atan2(z, w))
        assert yaw == pytest.approx(45.0, abs=1e-6)
        assert (x, y) == pytest.approx((0.0, 0.0))

    def test_lidar_rows_and_original_poses_are_untouched(self):
        t = _traj_tables({"s2": 200})
        out, _ = interpolate_camera_ego_poses(t)
        lidar_before = [r for r in t["sample_data"] if "LIDAR_TOP" in r["filename"]]
        lidar_after = [r for r in out["sample_data"] if "LIDAR_TOP" in r["filename"]]
        assert lidar_after == lidar_before
        assert out["ego_pose"][:3] == t["ego_pose"]

    def test_deterministic_tokens_and_idempotent(self):
        t = _traj_tables({"s1": 200, "s2": 200, "s3": 200})
        a, na = interpolate_camera_ego_poses(t)
        b, nb = interpolate_camera_ego_poses(t)
        assert a == b and na == nb == 3
        again, n_again = interpolate_camera_ego_poses(a)
        assert n_again == 0 and again == a

    def test_input_not_mutated(self):
        t = _traj_tables({"s1": 200})
        before = copy.deepcopy(t)
        interpolate_camera_ego_poses(t)
        assert t == before

    def test_scene_with_a_single_lidar_pose_is_left_shared(self):
        t = _traj_tables({"s1": 200})
        keep = {"s1"}
        t["sample"] = [s for s in t["sample"] if s["token"] in keep]
        t["sample_data"] = [r for r in t["sample_data"] if r["sample_token"] in keep]
        out, n = interpolate_camera_ego_poses(t)
        assert n == 0 and out == t


class TestMainInterpolatesToo:
    def test_written_camera_rows_carry_their_own_ego_pose(self, tmp_path):
        t = _traj_tables({"s1": 200, "s2": 200, "s3": 200})
        root = TestMain()._write_root(tmp_path, t)
        rc = main(["--dataroot", str(root), "--version", "v1.0-x", "--out-version", "v1.0-x-fixed",
                   "--required-channels", "LIDAR_TOP", "CAM_A"])
        assert rc == 0
        sd = json.loads((root / "v1.0-x-fixed" / "sample_data.json").read_text())
        ep = {e["token"] for e in json.loads((root / "v1.0-x-fixed" / "ego_pose.json").read_text())}
        lidar = {r["sample_token"]: r["ego_pose_token"] for r in sd if "LIDAR_TOP" in r["filename"]}
        for r in sd:
            if "CAM_A" in r["filename"]:
                assert r["ego_pose_token"] != lidar[r["sample_token"]]
                assert r["ego_pose_token"] in ep


# ---------------------------------------------------------------------------
# Camera extrinsics: body frame -> optical convention. The exporter writes
# camera rotations in the vehicle body convention (CAM_FRONT ~ identity,
# CAM_BACK ~ 180 deg yaw). Every projection in the pipeline assumes nuScenes'
# optical convention (z forward, x right, y down): measured 2026-09-06 on
# chunk_0006, 0.0 % of LiDAR points landed inside ANY camera image, Stage 5
# painted 672 of 30 M points, 4,640 of 4,730 instances were empty, and the
# road stage found 5-11 candidates per keyframe instead of thousands. The
# pilot's fixup did this same conversion ("camera rotations -> optical
# convention", handover §4). Rows are stamped so a second pass is a no-op.
# ---------------------------------------------------------------------------
from scripts.fixup_a_nusc import BODY_TO_OPTICAL_WXYZ, cameras_to_optical_convention  # noqa: E402


def _cal_tables():
    t = _tables(n=1)
    t["calibrated_sensor"] = [
        {"token": "cs_LIDAR_TOP", "sensor_token": "sen_LIDAR_TOP", "rotation": [1.0, 0.0, 0.0, 0.0],
         "translation": [0.0, 0.0, 1.8], "camera_intrinsic": []},
        {"token": "cs_CAM_A", "sensor_token": "sen_CAM_A", "rotation": [1.0, 0.0, 0.0, 0.0],
         "translation": [0.8, 0.0, 1.2], "camera_intrinsic": [[900, 0, 640], [0, 900, 360], [0, 0, 1]]},
        {"token": "cs_CAM_B", "sensor_token": "sen_CAM_B", "rotation": [0.0, 0.0, 0.0, 1.0],  # 180 deg yaw
         "translation": [-0.8, 0.0, 1.2], "camera_intrinsic": [[900, 0, 640], [0, 900, 360], [0, 0, 1]]},
    ]
    return t


def _q2m(q):
    w, x, y, z = q
    return [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]


class TestCamerasToOpticalConvention:
    def test_identity_body_camera_becomes_the_nuscenes_front_camera(self):
        out, n = cameras_to_optical_convention(_cal_tables())
        cam = next(c for c in out["calibrated_sensor"] if c["token"] == "cs_CAM_A")
        assert n == 2
        assert cam["rotation"] == pytest.approx([0.5, -0.5, 0.5, -0.5])
        assert cam["frame_convention"] == "optical"

    def test_optical_z_axis_points_where_body_x_pointed(self):
        # The camera's forward axis: optical +z expressed in ego must equal the
        # body +x it replaced (front camera looks forward, back camera backward).
        out, _ = cameras_to_optical_convention(_cal_tables())
        for tok, forward in (("cs_CAM_A", [1, 0, 0]), ("cs_CAM_B", [-1, 0, 0])):
            R = _q2m(next(c for c in out["calibrated_sensor"] if c["token"] == tok)["rotation"])
            z_in_ego = [R[i][2] for i in range(3)]
            assert z_in_ego == pytest.approx(forward, abs=1e-9)
            y_in_ego = [R[i][1] for i in range(3)]
            assert y_in_ego == pytest.approx([0, 0, -1], abs=1e-9)  # optical y points DOWN

    def test_lidar_and_translations_are_untouched(self):
        t = _cal_tables()
        out, _ = cameras_to_optical_convention(t)
        lidar = next(c for c in out["calibrated_sensor"] if c["token"] == "cs_LIDAR_TOP")
        assert lidar == next(c for c in t["calibrated_sensor"] if c["token"] == "cs_LIDAR_TOP")
        for c_out, c_in in zip(out["calibrated_sensor"], t["calibrated_sensor"]):
            assert c_out["translation"] == c_in["translation"]

    def test_second_pass_is_a_no_op(self):
        once, n1 = cameras_to_optical_convention(_cal_tables())
        twice, n2 = cameras_to_optical_convention(once)
        assert n1 == 2 and n2 == 0 and twice == once

    def test_input_not_mutated(self):
        t = _cal_tables()
        before = copy.deepcopy(t)
        cameras_to_optical_convention(t)
        assert t == before

    def test_constant_is_the_classic_body_to_optical_quaternion(self):
        assert BODY_TO_OPTICAL_WXYZ == pytest.approx([0.5, -0.5, 0.5, -0.5])


# ---------------------------------------------------------------------------
# Swapped camera channels. Measured 2026-09-06 on chunk_0000 by optical flow:
# the stream the exporter filed as CAM_LEFT faces RIGHT (+68.6 px rearward
# drift) and CAM_RIGHT faces LEFT (-80.6 px), while both calibrations say the
# opposite — so every projection into those two cameras landed on the wrong
# side of the car and the 3D boxes came out scrambled (operator, 09:10). The
# fix pairs each image row with the calibration that matches its content:
# the two rows of a sample exchange calibrated_sensor_token; filename,
# timestamp and ego_pose stay with the row (they belong to the file).
# ---------------------------------------------------------------------------
from scripts.fixup_a_nusc import swap_camera_channels  # noqa: E402


def _two_cam_tables():
    t = _tables(n=2)  # channels LIDAR_TOP, CAM_A, CAM_B
    return t


class TestSwapCameraChannels:
    def test_rows_exchange_calibration_but_keep_file_time_and_pose(self):
        t = _two_cam_tables()
        out, n = swap_camera_channels(t, "CAM_A", "CAM_B")
        assert n == 4  # 2 samples x 2 rows
        by_tok = {r["token"]: r for r in out["sample_data"]}
        for r in t["sample_data"]:
            o = by_tok[r["token"]]
            assert (o["filename"], o["timestamp"], o["ego_pose_token"]) == (r["filename"], r["timestamp"], r["ego_pose_token"])
            if "CAM_A" in r["filename"]:
                assert o["calibrated_sensor_token"] == "cs_CAM_B"
            elif "CAM_B" in r["filename"]:
                assert o["calibrated_sensor_token"] == "cs_CAM_A"
            else:
                assert o["calibrated_sensor_token"] == r["calibrated_sensor_token"]

    def test_channel_now_matches_content_via_the_sensor_table(self):
        from scripts.fixup_a_nusc import channels_per_sample
        out, _ = swap_camera_channels(_two_cam_tables(), "CAM_A", "CAM_B")
        # every sample still has both channels — the set is unchanged, the pairing is not
        assert all({"CAM_A", "CAM_B"} <= chans for chans in channels_per_sample(out).values())

    def test_sample_missing_one_of_the_pair_is_left_alone(self):
        t = _two_cam_tables()
        t["sample_data"] = [r for r in t["sample_data"] if not (r["sample_token"] == "s2" and "CAM_B" in r["filename"])]
        out, n = swap_camera_channels(t, "CAM_A", "CAM_B")
        assert n == 2
        s2 = [r for r in out["sample_data"] if r["sample_token"] == "s2" and "CAM_A" in r["filename"]]
        assert s2[0]["calibrated_sensor_token"] == "cs_CAM_A"

    def test_swap_is_its_own_inverse_and_input_not_mutated(self):
        t = _two_cam_tables()
        before = copy.deepcopy(t)
        once, _ = swap_camera_channels(t, "CAM_A", "CAM_B")
        twice, _ = swap_camera_channels(once, "CAM_A", "CAM_B")
        assert twice == t and t == before

    def test_unknown_channel_refuses(self):
        with pytest.raises(ValueError, match="CAM_Z"):
            swap_camera_channels(_two_cam_tables(), "CAM_A", "CAM_Z")


class TestMainSwaps:
    def test_flag_swaps_and_records_it(self, tmp_path):
        t = _two_cam_tables()
        root = TestMain()._write_root(tmp_path, t)
        rc = main(["--dataroot", str(root), "--version", "v1.0-x", "--out-version", "v1.0-x-fixed",
                   "--required-channels", "LIDAR_TOP", "CAM_A", "CAM_B", "--swap-channels", "CAM_A", "CAM_B"])
        assert rc == 0
        sd = json.loads((root / "v1.0-x-fixed" / "sample_data.json").read_text())
        assert all(r["calibrated_sensor_token"] == "cs_CAM_B" for r in sd if "CAM_A/" in r["filename"])
        meta = json.loads((root / "v1.0-x-fixed" / "fixup_meta.json").read_text())
        assert meta["swapped_channels"] == [["CAM_A", "CAM_B"]]


def test_tables_is_the_thirteen_nuscenes_tables():
    assert len(TABLES) == 13
    assert {"sample", "sample_data", "scene", "sensor", "calibrated_sensor"} <= set(TABLES)


@pytest.mark.parametrize("bad", [None, []])
def test_required_channels_must_be_non_empty(bad):
    with pytest.raises(ValueError):
        drop_incomplete_samples(_tables(), bad)
