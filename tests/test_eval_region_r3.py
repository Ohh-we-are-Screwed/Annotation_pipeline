"""coverage_config R3 = the two ZED frusta (2026-09-12): front +-h, rear pi+-h."""
from __future__ import annotations
import math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.common import eval_region as er  # noqa: E402
from pipeline.common.eval_region import in_region, region_spec_from_config  # noqa: E402


def test_r3_is_a_known_config_with_two_wedges():
    assert "R3" in er.COVERAGE_CONFIGS
    spec = er.R3_DEFAULT
    assert spec.coverage_config == "R3"
    assert abs(spec.azimuth_measure_rad - 4 * er.R3_HALF_WIDTH_RAD) < 1e-9
    assert spec.r_max_m == er.STEREO_RANGE_CAP_DEFAULT_M == 25.0


def test_r3_membership():
    spec = er.R3_DEFAULT
    r = 10.0
    for deg, inside in ((0, True), (180, True), (25, True), (-25, True), (155, True), (-155, True),
                        (90, False), (-90, False), (60, False), (120, False)):
        th = math.radians(deg)
        got = bool(in_region([r * math.cos(th)], [r * math.sin(th)], spec, frame="ego")[0])
        assert got is inside, f"{deg} deg: expected {inside}, got {got}"
    assert not in_region([30.0], [0.0], spec, frame="ego")[0]  # beyond the cap


def test_r3_from_config_dict():
    spec = region_spec_from_config({"coverage_config": "R3", "r_max_m": 18.0})
    assert spec.coverage_config == "R3" and spec.r_max_m == 18.0
    assert abs(spec.azimuth_measure_rad - 4 * er.R3_HALF_WIDTH_RAD) < 1e-9
