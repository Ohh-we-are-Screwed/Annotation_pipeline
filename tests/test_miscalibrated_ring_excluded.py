"""The rear ZED (LIDAR_TOP ring 100) is miscalibrated and must not reach the boxes.

Measured 2026-09-10 on Dataset/A_nusc/chunk_0000 (30 keyframes, RANSAC road
plane per ring group, 3-15 m):

    group             tilt deg   tilt dir deg   z at origin
    lidar 0-3             2.70          -80.3         -2.43
    ZED front (101)       2.27          -59.7         -2.55     agrees
    ZED rear  (100)       6.76         +162.6         -2.42     4.06 deg OFF

The rear ZED's road plane is tilted 4.06 deg away from the LiDAR's, in a tilt
DIRECTION 117 deg apart -- it is not a z offset that a translation could absorb,
and not a common pitch that a single rotation could. Over the 3-25 m band that
is 0.45 m to 1.75 m of vertical disagreement, growing with range.

The consequence, measured on work_b/chunk_0000's 74,369 shipped cuboids: every
object's point set is the superposition of two mutually tilted copies of the
scene, so the cluster elongates along the viewing ray, the L-shape fit takes
that ray as the box's LENGTH, and the yaw it derives is the bearing to the ego
vehicle -- 72-79 % of vehicle boxes point within 20 deg of their own bearing.
Widths and heights land at ~55 % of the class mean; 97 % of car boxes are
flatter than their own footprint.

The dhaka6 profile already knew (`ground_fit_rings` excludes 100, calling it
"a miscalibration the exporter baked into LIDAR_TOP"), but the exclusion only
stopped ring 100 from VOTING for the ground plane. Its points still reached
Stage 5's lift, Stage 6's DBSCAN and the fitted boxes. This closes that gap at
the one place where the cloud is already filtered before it is written, so
`point_index` keeps meaning what Stages 6-8 and the road export think it means.

Dropping ring 100 costs 38 % of object points and leaves the median car with 28
(from 49); 63 % of cars keep the >= 15 returns that Stage 8 treats as measured.
LiDAR alone was not an option: it leaves the median car with 9.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.stage1_ingestion.ingest import drop_rings  # noqa: E402


def _in_fresh_interpreter(code: str, **env_overrides: str) -> str:
    """Profile constants resolve ONCE at import, so each case needs its own."""
    env = {k: v for k, v in os.environ.items() if k != "DHAKASCENES_SUBSTRATE"}
    env.update(env_overrides)
    env.setdefault("PYTHONPATH", ROOT)
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT
    )
    assert out.returncode == 0, f"subprocess failed:\n{out.stdout}\n{out.stderr}"
    return out.stdout.strip()


def _cloud(rings: list[int]) -> np.ndarray:
    """(N, 5) x, y, z, intensity, ring -- the nuScenes .pcd.bin layout."""
    n = len(rings)
    return np.column_stack(
        [
            np.arange(n, dtype=np.float64),  # x doubles as a file-order marker
            np.zeros(n),
            np.zeros(n),
            np.zeros(n),
            np.asarray(rings, dtype=np.float64),
        ]
    )


class TestProfileDeclaresTheMiscalibratedRing:
    def test_dhaka6_excludes_the_rear_zed(self):
        code = (
            "from pipeline.common.schemas import MISCALIBRATED_RINGS;"
            "print(tuple(MISCALIBRATED_RINGS))"
        )
        assert _in_fresh_interpreter(code, DHAKASCENES_SUBSTRATE="dhaka6") == "(100,)"

    def test_other_substrates_exclude_nothing(self):
        code = (
            "from pipeline.common.schemas import MISCALIBRATED_RINGS;"
            "print(tuple(MISCALIBRATED_RINGS))"
        )
        for substrate in ("dhaka", "nuscenes"):
            got = _in_fresh_interpreter(code, DHAKASCENES_SUBSTRATE=substrate)
            assert got == "()", f"{substrate} should exclude no ring, got {got}"

    def test_the_excluded_ring_is_not_a_ground_voter_either(self):
        """The two lists must not contradict: a ring we do not trust for
        geometry cannot be trusted to define the ground plane."""
        code = (
            "from pipeline.common.schemas import MISCALIBRATED_RINGS, GROUND_FIT_RINGS;"
            "print(sorted(set(MISCALIBRATED_RINGS) & set(GROUND_FIT_RINGS)))"
        )
        assert _in_fresh_interpreter(code, DHAKASCENES_SUBSTRATE="dhaka6") == "[]"


class TestDropRings:
    def test_removes_exactly_the_named_rings(self):
        cloud = _cloud([0, 1, 100, 101, 100, 2])
        kept, removed = drop_rings(cloud, (100,))
        assert removed == 2
        assert kept[:, 4].tolist() == [0.0, 1.0, 101.0, 2.0]

    def test_preserves_file_order(self):
        """Determinism (§1.9): a boolean mask, never a sort. The x column
        carries the original index, so the survivors must stay ascending."""
        rings = [0, 100, 1, 100, 101, 100, 2, 3]
        kept, _ = drop_rings(_cloud(rings), (100,))
        assert kept[:, 0].tolist() == sorted(kept[:, 0].tolist())
        assert kept[:, 0].tolist() == [0.0, 2.0, 4.0, 6.0, 7.0]

    def test_empty_exclusion_is_the_input_untouched(self):
        cloud = _cloud([0, 1, 100, 101])
        kept, removed = drop_rings(cloud, ())
        assert removed == 0
        assert np.array_equal(kept, cloud)

    def test_dropping_every_ring_leaves_an_empty_cloud_not_an_error(self):
        """Stage 1 reports an empty cloud through its ledger; it must not raise
        here, or a single degenerate keyframe aborts an eight-hour chunk."""
        kept, removed = drop_rings(_cloud([100, 100]), (100,))
        assert removed == 2
        assert kept.shape == (0, 5)

    def test_ring_column_is_compared_exactly(self):
        """Ring 100 and ring 10 are different sensors on different rigs; a
        prefix or float-tolerance match would silently take both."""
        kept, removed = drop_rings(_cloud([10, 100, 1000]), (100,))
        assert removed == 1
        assert kept[:, 4].tolist() == [10.0, 1000.0]


class TestIngestConfigWiresItUp:
    def test_config_defaults_to_the_profile(self):
        code = (
            "from pipeline.stage1_ingestion.ingest import IngestConfig;"
            "print(tuple(IngestConfig().miscalibrated_rings))"
        )
        assert _in_fresh_interpreter(code, DHAKASCENES_SUBSTRATE="dhaka6") == "(100,)"

    def test_config_carries_provenance_for_the_exclusion(self):
        """Every profile-owned number in this pipeline says where it came from;
        this one is a measurement and must name it."""
        code = (
            "from pipeline.stage1_ingestion.ingest import IngestConfig;"
            "p = IngestConfig().provenance;"
            "print('miscalibrated_rings' in p)"
        )
        assert _in_fresh_interpreter(code, DHAKASCENES_SUBSTRATE="dhaka6") == "True"
