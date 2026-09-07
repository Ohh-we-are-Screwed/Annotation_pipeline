"""Day-1 chunked run (2026-09-06): the two repo changes it needs.

1. A `dhaka6` substrate profile — the 2026-09-04 capture rig: the Dhaka ring
   minus its two rear corners, 1280x720. Without it Stage 0 excludes every
   scene on `channels_complete` (the `dhaka` profile demands CAM_BACK_LEFT /
   CAM_BACK_RIGHT, which this rig does not carry).
2. `CVAT_PIPELINE_3D_PROJECT` — the 3D publish's project name becomes env-
   overridable exactly the way the 2D publish's CVAT_PIPELINE_PROJECT already
   is, so each chunk can land in its own 3D project. Unset keeps the legacy
   name byte-identical.

Both values are resolved ONCE at import (schemas.py: "resolved ONCE at
import"; cvat_setup_3d.py: module constant), so each case runs in a fresh
interpreter with the env it needs rather than mutating a shared module.
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.schemas import SUBSTRATE_PROFILES  # noqa: E402

A_NUSC_RING = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_LEFT",
    "CAM_RIGHT",
    "CAM_BACK",
)


def _in_fresh_interpreter(code: str, **env_overrides: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("DHAKASCENES_SUBSTRATE", "CVAT_PIPELINE_3D_PROJECT")}
    env.update(env_overrides)
    env["PYTHONPATH"] = ROOT
    out = subprocess.run(
        [sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


class TestDhaka6Profile:
    def test_profile_exists_with_the_six_a_nusc_cameras(self):
        assert "dhaka6" in SUBSTRATE_PROFILES
        assert set(SUBSTRATE_PROFILES["dhaka6"]["ring_cameras"]) == set(A_NUSC_RING)
        assert len(SUBSTRATE_PROFILES["dhaka6"]["ring_cameras"]) == 6

    def test_profile_pins_1280x720(self):
        p = SUBSTRATE_PROFILES["dhaka6"]
        assert (p["image_width_px"], p["image_height_px"]) == (1280, 720)

    def test_env_selects_it_and_required_channels_follow(self):
        got = _in_fresh_interpreter(
            "from pipeline.common import schemas as s;"
            "print(s.SUBSTRATE, sorted(s.RING_CAMERAS), s.IMAGE_WIDTH_PX, s.IMAGE_HEIGHT_PX,"
            " 'LIDAR_TOP' in s.REQUIRED_CHANNELS, 'CAM_BACK_LEFT' in s.REQUIRED_CHANNELS)",
            DHAKASCENES_SUBSTRATE="dhaka6",
        )
        assert got == f"dhaka6 {sorted(A_NUSC_RING)} 1280 720 True False"

    # --- accumulation contract is a PROFILE property (2026-09-06) ---------------
    # The day-1 export writes no sweeps: one LIDAR_TOP record per keyframe,
    # already fused. Stage 0's sweeps_cover_window (expects 0.8 x 5 records in
    # the 0.5 s window) refused every chunk on that alone. The count/duration
    # were module literals in probe.py and IngestConfig; they are now declared
    # per profile so a sweep-less substrate says so, and dhaka/nuscenes keep 5 /
    # 0.5 s byte-for-byte.
    ACC_CODE = (
        "from pipeline.common import schemas as s;"
        "from pipeline.stage0_data_probe import probe as p;"
        "from pipeline.stage1_ingestion.ingest import IngestConfig as C;"
        "print(s.W_ACC_COUNT, s.W_ACC_DURATION_NS, p.W_ACC_COUNT, p.W_ACC_DURATION_NS,"
        " C().w_acc_count, C().w_acc_duration_ns)"
    )

    def test_dhaka6_declares_single_sweep_accumulation(self):
        p = SUBSTRATE_PROFILES["dhaka6"]
        assert (p["w_acc_count"], p["w_acc_duration_ns"]) == (1, 0)

    def test_legacy_profiles_declare_the_pilot_window(self):
        for name in ("dhaka", "nuscenes"):
            p = SUBSTRATE_PROFILES[name]
            assert (p["w_acc_count"], p["w_acc_duration_ns"]) == (5, 500_000_000), name

    def test_probe_and_stage1_read_the_profile_window(self):
        assert _in_fresh_interpreter(self.ACC_CODE, DHAKASCENES_SUBSTRATE="dhaka6") == "1 0 1 0 1 0"
        assert _in_fresh_interpreter(self.ACC_CODE) == "5 500000000 5 500000000 5 500000000"

    # --- parse bands are a PROFILE property too (2026-09-06) ------------------
    # Stage 0's files_parse gates every blob on a size band measured on the
    # pilot substrate: 34-35k points per cloud, 20 kB-4 MB per JPEG. The day-1
    # export's fused clouds run 37,920-490,859 points and its re-encoded side
    # cameras go down to 15,027 bytes (925 of 26,799 JPEGs under 20 kB) — so
    # 93 of 94 clouds in chunk_0006 failed the pilot band. Measured 2026-09-06
    # over all 4,536 clouds / 26,799 JPEGs; the dhaka6 bands carry headroom
    # above those extremes and the legacy profiles keep the pilot's numbers.
    BAND_CODE = (
        "from pipeline.stage0_data_probe import probe as p;"
        "print(p.PCD_MIN_POINTS, p.PCD_MAX_POINTS, p.JPEG_MIN_BYTES, p.JPEG_MAX_BYTES)"
    )

    def test_dhaka6_bands_cover_the_measured_extremes(self):
        p = SUBSTRATE_PROFILES["dhaka6"]
        lo, hi = p["pcd_point_band"]
        assert lo <= 37_920 and hi >= 490_859
        lo, hi = p["jpeg_byte_band"]
        assert lo <= 15_027 and hi >= 428_741

    def test_legacy_profiles_keep_the_pilot_bands(self):
        for name in ("dhaka", "nuscenes"):
            p = SUBSTRATE_PROFILES[name]
            assert tuple(p["pcd_point_band"]) == (10_000, 300_000), name
            assert tuple(p["jpeg_byte_band"]) == (20_000, 4_000_000), name

    def test_probe_reads_the_profile_bands(self):
        assert _in_fresh_interpreter(self.BAND_CODE) == "10000 300000 20000 4000000"
        got = [int(x) for x in _in_fresh_interpreter(self.BAND_CODE, DHAKASCENES_SUBSTRATE="dhaka6").split()]
        assert got[0] <= 37_920 and got[1] >= 490_859 and got[2] <= 15_027 and got[3] >= 428_741

    # --- the manifest must ACCEPT the window the profile declares -------------
    # SubstrateManifest.validate() floored w_acc_duration_ns at 1, so Stage 0
    # under dhaka6 computed a clean verdict and then refused to write it
    # ("w_acc_duration_ns must be >= 1, got 0", 2026-09-06). Zero is now a
    # declared, meaningful value: the anchor alone. Negative stays invalid.
    @staticmethod
    def _manifest(duration_ns: int):
        from pipeline.common.schemas import SubstrateManifest
        return SubstrateManifest(
            dataroot_realpath="/x", version="v1.0-dhaka-fixed", metadata_fingerprint="f" * 64,
            fingerprint_spec="spec", required_channels=["LIDAR_TOP", "CAM_FRONT"],
            camera_subset=["CAM_FRONT"], coverage_config="R2", usable_scene_tokens=["t"],
            w_acc_count=1, w_acc_duration_ns=duration_ns,
        )

    def test_manifest_accepts_a_zero_length_window(self):
        assert [v for v in self._manifest(0).validate() if "w_acc_duration_ns" in v] == []

    def test_manifest_still_rejects_a_negative_window(self):
        assert any("w_acc_duration_ns" in v for v in self._manifest(-1).validate())

    # --- stereo thinning is a PROFILE property (2026-09-06) --------------------
    # The day-1 LIDAR_TOP is a fused cloud: Mid-360 rings 0-3 (39,936 points)
    # plus two ZED depth clouds as rings 100/101 (350,595 points — 8.8x the
    # LiDAR). One parked car in front of a ZED camera painted ~38k points per
    # keyframe; sklearn DBSCAN's neighbour graph on such an instance reached
    # 53 GB RSS and 14 GB of swap and never finished (Stage 6, 47 min, killed).
    # Stage 1 now keeps every `stereo_stride`-th point of each stereo ring —
    # deterministic, recorded — so stereo density matches the LiDAR's.
    def test_dhaka6_declares_its_stereo_rings_and_a_stride(self):
        p = SUBSTRATE_PROFILES["dhaka6"]
        assert tuple(p["stereo_rings"]) == (100, 101)
        assert p["stereo_stride"] >= 4

    def test_legacy_profiles_thin_nothing(self):
        for name in ("dhaka", "nuscenes"):
            p = SUBSTRATE_PROFILES[name]
            assert tuple(p["stereo_rings"]) == () and p["stereo_stride"] == 1, name

    def test_stage1_config_reads_the_profile_thinning(self):
        code = ("from pipeline.stage1_ingestion.ingest import IngestConfig as C;"
                "c=C(); print(tuple(c.stereo_rings), c.stereo_stride)")
        assert _in_fresh_interpreter(code) == "() 1"
        rings, stride = _in_fresh_interpreter(code, DHAKASCENES_SUBSTRATE="dhaka6").rsplit(" ", 1)
        assert rings == "(100, 101)" and int(stride) >= 4

    # --- ground-plane candidate band is a PROFILE property (2026-09-06) --------
    # Stage 1's RANSAC ground fit takes candidates from z in [-1.5, +1.5] m,
    # written for an ego frame with z=0 at the ground. The day-1 rig's ego
    # origin IS the LiDAR, ~2.3 m up: measured on chunk_0000 keyframe 100 the
    # road is at z = -2.0..-2.75 (LiDAR rings) / -2.5..-3.0 (ZED). The band
    # never held the ground, RANSAC "found" planes at -0.3..-0.6 m through the
    # scene, the +-0.3 m removal slab cut pedestrians' torsos and car roofs
    # (heights 55-65 % of true), and the surviving road points elongated
    # 84-93 % of boxes along the viewing ray. The operator saw "cuboids way
    # out of proportion" — this is it.
    def test_dhaka6_ground_band_holds_the_measured_road(self):
        lo, hi = SUBSTRATE_PROFILES["dhaka6"]["ground_z_band_m"]
        assert lo <= -3.0 and hi >= -2.0 and hi < 0.0

    def test_legacy_profiles_keep_the_iso8855_band(self):
        for name in ("dhaka", "nuscenes"):
            assert tuple(SUBSTRATE_PROFILES[name]["ground_z_band_m"]) == (-1.5, 1.5), name

    def test_stage1_config_reads_the_profile_band(self):
        code = ("from pipeline.stage1_ingestion.ingest import IngestConfig as C;"
                "print(list(C().ransac_candidate_z_band_m))")
        assert _in_fresh_interpreter(code) == "[-1.5, 1.5]"
        lo, hi = eval(_in_fresh_interpreter(code, DHAKASCENES_SUBSTRATE="dhaka6"))
        assert lo <= -3.0 and hi < 0.0

    def test_existing_profiles_are_untouched(self):
        # The default stays `dhaka`, eight cameras, and nuscenes stays 1600x900:
        # every archived number keeps meaning what it meant.
        assert len(SUBSTRATE_PROFILES["dhaka"]["ring_cameras"]) == 8
        assert (SUBSTRATE_PROFILES["nuscenes"]["image_width_px"],
                SUBSTRATE_PROFILES["nuscenes"]["image_height_px"]) == (1600, 900)
        assert _in_fresh_interpreter("from pipeline.common import schemas as s; print(s.SUBSTRATE)") == "dhaka"


class TestCvat3dProjectEnvOverride:
    LEGACY = "OUR PIPELINE — machine pre-annotations (3D)"
    CODE = "from scripts import cvat_setup_3d as m; print(m.OURS_PROJECT + '|' + m.GT_PROJECT)"

    def test_unset_keeps_the_legacy_name_byte_identical(self):
        ours, gt = _in_fresh_interpreter(self.CODE).split("|")
        assert ours == self.LEGACY
        assert gt == "nuScenes GT — HUMAN answer key (3D)"

    def test_env_overrides_ours_only(self):
        ours, gt = _in_fresh_interpreter(self.CODE, CVAT_PIPELINE_3D_PROJECT="day1_chunk_0006 (3D)").split("|")
        assert ours == "day1_chunk_0006 (3D)"
        # C13: the answer-key project is never renamed by the pipeline's own knob.
        assert gt == "nuScenes GT — HUMAN answer key (3D)"

    def test_empty_env_falls_back_to_legacy(self):
        # An exported-but-empty variable must not create a project named "".
        ours, _ = _in_fresh_interpreter(self.CODE, CVAT_PIPELINE_3D_PROJECT="").split("|")
        assert ours == self.LEGACY
