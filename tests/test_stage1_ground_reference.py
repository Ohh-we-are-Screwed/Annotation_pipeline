"""Stage 1 ground fit: the whole-cloud reference plane and the mis-fit guard.

Background (2026-09-06, Dataset/A_nusc chunk_0000/0001): when a wedge's own
RANSAC fit was rejected, Stage 1 substituted a plain least-squares plane over
every candidate in the search band. The band reaches 1.35 m above the road,
every non-road candidate (legs, wheels, curbs, bodies) sits ABOVE it, and least
squares has no outlier rejection — so the substitute landed 0.26-0.39 m
(p90 0.60 m) above the road in 21 % of all wedge fits, stripping the bottom of
every object in those wedges. Meanwhile the inlier-ratio gate rejected fits
that were within 0.1 m of the road because a crowded band lowers the ratio.

The rules under test:
  1. the reference (fallback) plane is fitted robustly — it is the road, not
     the mean of the clutter;
  2. a wedge fit is rejected on what actually harms the filter — tilt, or
     disagreeing with the reference plane in height — not on clutter ratio;
  3. the reference plane is part of the diagnostics record.
"""
from __future__ import annotations

import numpy as np

ROAD_Z = -2.3
BAND = (-3.5, -1.0)


def _cloud(seed=0, n_road=6000, n_clutter=0, r_max=20.0, clutter_z=(-2.15, -1.05)):
    """Ring-0 road at ROAD_Z plus optional clutter uniformly ABOVE it, inside the band."""
    rng = np.random.default_rng(seed)

    def _xy(n):
        ang = rng.uniform(-np.pi, np.pi, n)
        rad = rng.uniform(1.0, r_max, n)
        return np.column_stack([rad * np.cos(ang), rad * np.sin(ang)])

    road = _xy(n_road)
    road = np.column_stack([road, ROAD_Z + rng.normal(0, 0.02, n_road), np.ones(n_road), np.zeros(n_road)])
    parts = [road]
    if n_clutter:
        cl = _xy(n_clutter)
        cl = np.column_stack([cl, rng.uniform(*clutter_z, n_clutter), np.ones(n_clutter), np.zeros(n_clutter)])
        parts.append(cl)
    return np.vstack(parts).astype(np.float64)


def _cfg(**kw):
    from pipeline.stage1_ingestion.ingest import IngestConfig
    base = dict(ransac_candidate_z_band_m=BAND, stereo_rings=[100, 101], stereo_stride=1,
                ground_fit_rings=[0, 1, 2, 3, 101], ground_fit_range_m=[3.0, 12.0])
    base.update(kw)
    return IngestConfig(**base)


def _sector_of(cloud):
    from pipeline.stage1_ingestion.ingest import sector_index
    return sector_index(cloud[:, :2], 8)


def test_reference_plane_ignores_clutter_above_the_road():
    # 45 % of the candidates are clutter above the road. Least squares would put
    # the plane ~0.3 m high; a robust fit finds the road.
    from pipeline.stage1_ingestion.ingest import fit_ground_planes
    cloud = _cloud(n_road=6000, n_clutter=5000)
    fit = fit_ground_planes(cloud, _cfg(), "kf")
    assert abs(fit.reference.d - ROAD_Z) < 0.06, fit.reference
    assert fit.reference.tilt_deg < 2.0


def test_wedge_with_too_few_candidates_falls_back_to_the_road_not_the_clutter_mean():
    from pipeline.stage1_ingestion.ingest import fit_sector_planes
    cloud = _cloud(n_road=6000, n_clutter=5000)
    sec = _sector_of(cloud)
    # Starve sector 0 of road points: only 10 remain (below ransac_min_candidates).
    road0 = np.flatnonzero((sec == 0) & (np.abs(cloud[:, 2] - ROAD_Z) < 0.1))
    cloud = np.delete(cloud, road0[10:], axis=0)
    # And drop its clutter too, so the wedge is genuinely empty.
    sec = _sector_of(cloud)
    cloud = np.delete(cloud, np.flatnonzero((sec == 0) & (cloud[:, 2] > ROAD_Z + 0.1)), axis=0)
    planes = fit_sector_planes(cloud, _cfg(), "kf")
    p0 = planes[0]
    assert p0.fallback == "global_plane"
    assert abs(p0.d - ROAD_Z) < 0.06, p0


def test_good_fit_with_a_low_inlier_ratio_is_kept():
    # Crowded band: ~55 % clutter gives inlier ratios ~0.4-0.45 on the road; with
    # 70 % clutter they drop under the old 0.35 gate. The fit is still the road.
    from pipeline.stage1_ingestion.ingest import fit_sector_planes
    cloud = _cloud(n_road=3000, n_clutter=7000)
    planes = fit_sector_planes(cloud, _cfg(), "kf")
    assert all(p.fallback is None for p in planes), [(p.sector, p.fallback, round(p.inlier_ratio, 2)) for p in planes]
    assert all(abs(p.d - ROAD_Z) < 0.06 for p in planes), [round(p.d, 2) for p in planes]
    assert min(p.inlier_ratio for p in planes) < 0.35  # the case the old gate rejected


def test_flat_slab_above_the_road_is_rejected_for_height_and_replaced_by_the_reference():
    # Sector 0's candidates are a flat platform 0.8 m above the road (a truck
    # bed, a stage): flat, high inlier ratio, and NOT the ground. The guard
    # must reject it for disagreeing with the reference and substitute the road.
    from pipeline.stage1_ingestion.ingest import fit_sector_planes
    cloud = _cloud(n_road=6000)
    sec = _sector_of(cloud)
    in0 = sec == 0
    cloud[in0, 2] = ROAD_Z + 0.8 + np.random.default_rng(1).normal(0, 0.02, int(in0.sum()))
    planes = fit_sector_planes(cloud, _cfg(), "kf")
    p0 = planes[0]
    assert p0.fallback == "rejected_height", p0
    assert p0.rejected_fit["reason"] == "height"
    assert abs(p0.rejected_fit["d"] - (ROAD_Z + 0.8)) < 0.06
    assert p0.rejected_fit["height_disagreement_m"] > 0.7
    assert abs(p0.d - ROAD_Z) < 0.06
    assert all(p.fallback is None for p in planes[1:])


def test_tilt_guard_still_fires():
    from pipeline.stage1_ingestion.ingest import fit_sector_planes
    cloud = _cloud(n_road=6000)
    sec = _sector_of(cloud)
    in0 = sec == 0
    # A 30-degree ramp through the road's own height at the wedge centroid.
    cx, cy = cloud[in0, 0].mean(), cloud[in0, 1].mean()
    cloud[in0, 2] = ROAD_Z + np.tan(np.radians(30)) * (cloud[in0, 0] - cx)
    planes = fit_sector_planes(cloud, _cfg(), "kf")
    assert planes[0].fallback == "rejected_tilt"
    assert planes[0].implausible_tilt


def test_reference_plane_is_deterministic():
    from pipeline.stage1_ingestion.ingest import fit_ground_planes
    cloud = _cloud(n_road=4000, n_clutter=4000)
    a = fit_ground_planes(cloud, _cfg(), "kf-x").reference
    b = fit_ground_planes(cloud, _cfg(), "kf-x").reference
    assert (a.a, a.b, a.d, a.n_inliers) == (b.a, b.b, b.d, b.n_inliers)


def test_reference_plane_record_is_json_ready_and_counts_its_evidence():
    from pipeline.stage1_ingestion.ingest import fit_ground_planes
    cloud = _cloud(n_road=4000, n_clutter=2000)
    ref = fit_ground_planes(cloud, _cfg(), "kf").reference
    rec = ref.as_dict()
    for key in ("a", "b", "d", "n_candidates", "n_inliers", "inlier_ratio", "tilt_deg", "method"):
        assert key in rec, key
    assert rec["method"] == "ransac_polished"
    assert rec["n_candidates"] > rec["n_inliers"] > 0


def test_height_disagreement_tolerance_is_a_recorded_config_knob():
    from pipeline.stage1_ingestion.ingest import IngestConfig
    cfg = IngestConfig()
    assert cfg.reject_height_disagreement_m > 0
    assert "reject_height_disagreement_m" in cfg.provenance
    d = cfg.as_dict() if hasattr(cfg, "as_dict") else cfg.__dict__
    assert "reject_height_disagreement_m" in d
