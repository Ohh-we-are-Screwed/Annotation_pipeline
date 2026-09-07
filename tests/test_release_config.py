from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.release.config import ReleaseConfigError, load_release_config  # noqa: E402

DEFAULT = os.path.join(ROOT, "configs", "release.yaml")


def test_default_config_loads_with_benchmark_values():
    cfg = load_release_config(DEFAULT)
    assert cfg.stitch.max_gap_keyframes == 3
    assert cfg.stitch.base_gate_m == 2.0
    assert cfg.attributes.moving_speed_threshold_mps == 0.5
    assert cfg.strata.illumination_bin_edges == [45.0, 75.0, 95.0]
    assert cfg.strata.illumination_bin_names == ["dark", "night", "dusk", "day"]
    assert cfg.strata.density_bin_names == ["Low", "Medium", "High", "Extreme"]
    assert cfg.double.fraction == 0.05 and cfg.double.seed == 20260812
    assert len(cfg.sha256) == 64
    assert cfg.benchmark_source["path"].endswith("benchmark_v1.0.yaml")


def test_bad_values_are_all_reported(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(
        "spec: dhakascenes/release_config/v1\n"
        "benchmark_source: {path: x, sha256: y}\n"
        "stitch: {max_gap_keyframes: 0, base_gate_m: -1, gap_slack_m: 1, size_ratio_max: 0.5, class_agnostic: false}\n"
        "attributes: {moving_speed_threshold_mps: 0.5, max_time_diff_s: 1.5}\n"
        "strata: {density_radius_m: 30, density_quantiles: [0.25, 0.5, 0.75], illumination_channel: CAM_FRONT,\n"
        "  illumination_saturation_ignore_above: 250, illumination_bin_edges: [45, 75], illumination_bin_names: [a, b, c, d],\n"
        "  density_bin_names: [Low, Medium, High, Extreme]}\n"
        "double: {fraction: 1.5, seed: 1}\n")
    with pytest.raises(ReleaseConfigError) as exc:
        load_release_config(str(p))
    msg = str(exc.value)
    for needle in ("max_gap_keyframes", "base_gate_m", "size_ratio_max", "illumination_bin_edges", "fraction"):
        assert needle in msg


def test_unknown_key_is_an_error(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(open(DEFAULT).read() + "\nextra_key: 1\n")
    with pytest.raises(ReleaseConfigError):
        load_release_config(str(p))
