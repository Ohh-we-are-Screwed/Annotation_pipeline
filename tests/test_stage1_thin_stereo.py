"""Stage 1 `thin_stereo` — keep every k-th point of each stereo ring (2026-09-06).

The day-1 fused LIDAR_TOP carries ZED depth points as rings 100/101 at 8.8x
the Mid-360's density; a single close instance then holds ~40k points and
Stage 6's DBSCAN neighbour graph exhausts 62 GB of RAM. The thinning is a
substrate property (schemas.SUBSTRATE_PROFILES[...]["stereo_rings" /
"stereo_stride"]), applied ONCE at ingestion to every cloud Stage 1 reads, so
Stage 5/6/7/8 all see the same thinned cloud. Deterministic: file order, every
stride-th point per ring, no randomness. Non-stereo rings are never touched.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.stage1_ingestion.ingest import thin_stereo  # noqa: E402


def _cloud():
    # 10 LiDAR points (ring 0/1), 16 on ring 100, 8 on ring 101 — x encodes file order.
    rows = []
    for i in range(10):
        rows.append([i, 0, 0, 1, i % 2])
    for i in range(16):
        rows.append([100 + i, 0, 0, 1, 100])
    for i in range(8):
        rows.append([200 + i, 0, 0, 1, 101])
    return np.array(rows, dtype=np.float32)


def test_identity_when_no_stereo_rings_or_stride_one():
    c = _cloud()
    out, removed = thin_stereo(c, (), 8)
    assert removed == 0 and np.array_equal(out, c)
    out, removed = thin_stereo(c, (100, 101), 1)
    assert removed == 0 and np.array_equal(out, c)


def test_keeps_every_stride_th_point_per_stereo_ring_in_file_order():
    out, removed = thin_stereo(_cloud(), (100, 101), 4)
    lidar = out[out[:, 4] < 100]
    assert len(lidar) == 10 and np.array_equal(lidar[:, 0], np.arange(10))  # untouched, in order
    r100 = out[out[:, 4] == 100][:, 0].tolist()
    r101 = out[out[:, 4] == 101][:, 0].tolist()
    assert r100 == [100, 104, 108, 112]
    assert r101 == [200, 204]
    assert removed == (16 - 4) + (8 - 2)


def test_overall_order_is_preserved_and_shape_kept():
    out, _ = thin_stereo(_cloud(), (100, 101), 2)
    assert out.shape[1] == 5 and out.dtype == np.float32
    assert np.all(np.diff(out[:, 0]) > 0)  # still in file order


def test_ring_not_in_the_stereo_set_is_never_thinned():
    out, removed = thin_stereo(_cloud(), (101,), 8)
    assert len(out[out[:, 4] == 100]) == 16 and len(out[out[:, 4] == 101]) == 1
    assert removed == 7


def test_deterministic():
    a, _ = thin_stereo(_cloud(), (100, 101), 3)
    b, _ = thin_stereo(_cloud(), (100, 101), 3)
    assert np.array_equal(a, b)


def test_input_not_mutated():
    c = _cloud()
    before = c.copy()
    thin_stereo(c, (100, 101), 4)
    assert np.array_equal(c, before)


# ---------------------------------------------------------------------------
# Ground fit on LiDAR rings only (2026-09-06). The fused ZED depth points put
# the road 0.2 m (front ZED) to 0.9 m (rear ZED) BELOW where the LiDAR puts
# it — a per-camera extrinsic bias the exporter baked into LIDAR_TOP. A sector
# dominated by ZED points then fits a plane up to 0.6 m too low. The LiDAR is
# the accurate sensor for the road: under the profile flag, stereo rings are
# never RANSAC candidates; the fitted plane still filters every point.
# ---------------------------------------------------------------------------
def _two_level_cloud(seed=0, n=4000, r_max=20.0):
    """LiDAR rings 0-3 on a road at -2.3; rear ZED (ring 100) 0.7 m lower; front
    ZED (ring 101) on the road — the measured 2026-09-06 situation."""
    rng = np.random.default_rng(seed)
    ang = rng.uniform(-np.pi, np.pi, n); rad = rng.uniform(1.0, r_max, n)
    xy = np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])
    mk = lambda z, ring: np.column_stack([xy, np.full(n, z) + rng.normal(0, 0.02, n), np.ones(n), np.full(n, float(ring))])
    return np.vstack([mk(-2.3, 0), mk(-2.3, 2), mk(-3.0, 100), mk(-2.3, 101)]).astype(np.float64)


def test_ground_fit_uses_only_the_profiles_candidate_rings():
    from pipeline.stage1_ingestion.ingest import IngestConfig, fit_sector_planes
    cloud = _two_level_cloud()
    cfg = IngestConfig(ransac_candidate_z_band_m=(-3.5, -1.0), stereo_rings=[100, 101], stereo_stride=1,
                       ground_fit_rings=[0, 1, 2, 3, 101], ground_fit_range_m=[3.0, 12.0])
    planes = fit_sector_planes(cloud, cfg, "kf")
    assert all(abs(p.d + 2.3) < 0.1 for p in planes), [round(p.d, 2) for p in planes]


def test_ground_fit_candidates_respect_the_range_window():
    # Put the road ONLY beyond 12 m for the allowed rings: the window empties the
    # candidate set, so every sector must fall back (recorded), not fit garbage.
    from pipeline.stage1_ingestion.ingest import IngestConfig, fit_sector_planes
    cloud = _two_level_cloud()
    r = np.hypot(cloud[:, 0], cloud[:, 1])
    cloud = cloud[(r > 13) | (cloud[:, 4] == 100)]
    cfg = IngestConfig(ransac_candidate_z_band_m=(-3.5, -1.0), stereo_rings=[100, 101], stereo_stride=1,
                       ground_fit_rings=[0, 1, 2, 3, 101], ground_fit_range_m=[3.0, 12.0])
    planes = fit_sector_planes(cloud, cfg, "kf")
    assert all(p.fallback is not None for p in planes)


def test_legacy_config_lets_every_ring_vote():
    # Only the rear ZED (ring 100) is present: legacy (no ring gate) fits its
    # plane; the dhaka6 gate has no candidates and every sector falls back.
    from pipeline.stage1_ingestion.ingest import IngestConfig, fit_sector_planes
    cloud = _two_level_cloud()
    cloud = cloud[cloud[:, 4] == 100]
    legacy = IngestConfig(ransac_candidate_z_band_m=(-3.5, -1.0), stereo_rings=[100, 101], stereo_stride=1)
    assert list(legacy.ground_fit_rings) == [] and legacy.ground_fit_range_m is None
    assert all(abs(p.d + 3.0) < 0.1 for p in fit_sector_planes(cloud, legacy, "kf"))
    gated = IngestConfig(ransac_candidate_z_band_m=(-3.5, -1.0), stereo_rings=[100, 101], stereo_stride=1,
                         ground_fit_rings=[0, 1, 2, 3, 101], ground_fit_range_m=[3.0, 12.0])
    assert all(p.fallback is not None for p in fit_sector_planes(cloud, gated, "kf"))


def test_profile_declares_the_ground_fit_sources():
    from pipeline.common.schemas import SUBSTRATE_PROFILES
    d6 = SUBSTRATE_PROFILES["dhaka6"]
    assert tuple(d6["ground_fit_rings"]) == (0, 1, 2, 3, 101) and tuple(d6["ground_fit_range_m"]) == (3.0, 12.0)
    for name in ("dhaka", "nuscenes"):
        assert tuple(SUBSTRATE_PROFILES[name]["ground_fit_rings"]) == () and SUBSTRATE_PROFILES[name]["ground_fit_range_m"] is None


def test_config_with_thinning_survives_the_json_round_trip():
    # Stage 1 crashed live (2026-09-06 02:48): the config is serialised into
    # filter_diagnostics.json and write_json_atomic refuses a payload whose
    # JSON round trip differs — a tuple-typed stereo_rings comes back a list.
    import json
    from pipeline.stage1_ingestion.ingest import IngestConfig
    cfg = IngestConfig(stereo_rings=[100, 101], stereo_stride=8)
    payload = {"config": cfg.as_dict()}
    assert json.loads(json.dumps(payload)) == payload


@pytest.mark.parametrize("stride", [0, -1])
def test_rejects_a_non_positive_stride(stride):
    with pytest.raises(ValueError):
        thin_stereo(_cloud(), (100,), stride)
