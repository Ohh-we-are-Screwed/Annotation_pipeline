from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.conventions import EGO, NUSCENES_GLOBAL, Transform, quaternion_from_yaw_rad  # noqa: E402
from pipeline.release.config import StrataConfig  # noqa: E402
from pipeline.release.frames import SceneFrames  # noqa: E402
from pipeline.release.strata import bin_of, compute_strata, density_per_keyframe, luma_of_image, quantile_edges  # noqa: E402

CFG = StrataConfig(density_radius_m=30.0, density_quantiles=[0.25, 0.5, 0.75],
                   density_bin_names=["Low", "Medium", "High", "Extreme"], illumination_channel="CAM_FRONT",
                   illumination_saturation_ignore_above=250, illumination_bin_edges=[45.0, 75.0, 95.0],
                   illumination_bin_names=["dark", "night", "dusk", "day"])


def _frames(n=4):
    toks = [f"s{i}" for i in range(n)]
    poses = {t: Transform.from_nuscenes({"translation": [10.0 * i, 0.0, 0.0], "rotation": list(quaternion_from_yaw_rad(0))},
                                        source_frame=EGO, parent_frame=NUSCENES_GLOBAL) for i, t in enumerate(toks)}
    return SceneFrames("sc", "chunk_t", toks, [i * 400_000_000 for i in range(n)], poses)


def test_density_counts_within_radius_of_ego():
    fr = _frames()
    centers = {"s0": [np.array([5.0, 0, 0]), np.array([29.0, 0, 0]), np.array([31.0, 0, 0])], "s1": []}
    rho = density_per_keyframe(fr, centers, CFG)
    assert rho["s0"] == 2 / (math.pi * 900) and rho["s1"] == 0.0 and rho["s3"] == 0.0


def test_luma_ignores_saturated_pixels(tmp_path):
    img = np.zeros((4, 4, 3), np.uint8)
    img[:2] = 255                        # saturated half is ignored
    img[2:] = (100, 100, 100)
    p = tmp_path / "f.jpg"
    Image.fromarray(img).save(p, quality=100)
    assert abs(luma_of_image(str(p), 250) - 100.0) < 3.0
    assert luma_of_image(str(tmp_path / "missing.jpg"), 250) is None


def test_edges_and_bins():
    assert quantile_edges([1, 2, 3, 4, 5], [0.5]) == [3.0]
    assert bin_of(44.9, [45, 75, 95], ["dark", "night", "dusk", "day"]) == "dark"
    assert bin_of(45.0, [45, 75, 95], ["dark", "night", "dusk", "day"]) == "night"
    assert bin_of(200.0, [45, 75, 95], ["dark", "night", "dusk", "day"]) == "day"
    assert bin_of(None, [45], ["a", "b"]) is None


def test_compute_strata_end_to_end(tmp_path):
    fr = _frames()
    paths = {}
    for i, t in enumerate(fr.tokens):
        p = tmp_path / f"{t}.jpg"
        Image.fromarray(np.full((2, 2, 3), 30 * i + 20, np.uint8)).save(p)
        paths[t] = str(p)
    centers = {t: [np.array([10.0 * i + 1.0, 0, 0])] * i for i, t in enumerate(fr.tokens)}
    st = compute_strata(fr, centers, paths, CFG)
    assert len(st.density_edges) == 3 and set(st.density_bin.values()) <= set(CFG.density_bin_names)
    assert st.illumination_bin["s0"] == "dark" and st.illumination_bin["s3"] == "day"
