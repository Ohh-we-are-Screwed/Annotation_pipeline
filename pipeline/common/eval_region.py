"""The evaluation region E and the density statistic rho (§1.10, spec §3.6, §8.3.1).

`in_region()` is the ONLY place in the codebase that decides whether a point is
inside E, and `rho()` is the only place that computes density. Both derive their
answer from a `RegionSpec` built from `coverage_config`; neither contains a
constant angle or radius of its own.

Why that matters: a locally reimplemented count is exactly where an R1/R2
decision silently distorts a metric. R1's E is a 110-degree frontal wedge and
R2's is a full annulus; the same box list yields different densities under each,
and rho is *count over area* precisely so the two remain comparable. A stage
that filters with its own `abs(theta) < 0.96` publishes a number that is no
longer the dataset's rho and looks identical in the output file.

Pilot decision (§11, decision 1): `coverage_config: R2`, all six ring cameras.
A CAM_FRONT-only run is legal, must record `coverage_config: R1`, and is a
different experiment rather than a cheaper version of the same one.

Frame: ego, ISO 8855 (x forward, y left, theta from +x about +z). Passing
LiDAR-frame coordinates here selects the vehicle's right-hand sector under R1
and silently mis-areas every wedge under R2 — hence `expect_frame`.

No GPU, no models: numpy and stdlib only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from pipeline.common.conventions import EGO, check_frame

__all__ = [
    "COVERAGE_CONFIGS",
    "RegionSpec",
    "Density",
    "R1_DEFAULT",
    "R2_DEFAULT",
    "region_spec_from_config",
    "in_region",
    "rho",
]

COVERAGE_CONFIGS: tuple[str, ...] = ("R1", "R2", "R3")

# --- inherited values, flagged (§10) ---------------------------------------
# 50 m range cap: operator decision 2026-09-07 — annotate to the benchmark's
#   evaluation range (benchmark_v1.0.yaml class_range 50/40/30 m). The Stage 9
#   point floor (>= 5 single-sweep returns) decides which far boxes survive and
#   the release's delivery note reports the effective per-class range. History:
#   spec §3.6 gave 40 m (Mid-360 10 %-reflectivity range); cut to 30 m on
#   2026-08-30 because the ZED densification caps at 20 m and boxes beyond
#   ~30 m were lidar-sparse. Runs before and after each change are NOT
#   comparable: E is the denominator of every density metric.
# 30 m rho radius: spec §8.3.1, verbatim.
# 55 deg R1 half-width: spec §3.6, verbatim.
_R_MAX_M = 50.0
_RHO_RADIUS_M = 30.0
_R1_HALF_WIDTH_RAD = math.radians(55.0)

# R3 (2026-09-12): the two ZED 2i frusta. h = half the rectified horizontal FOV
# (calibration.json h_fov_deg 67.748); the cap is the spike's default until
# docs/evidence/2026-09-12-stereo-vs-lidar-*.md says otherwise.
R3_HALF_WIDTH_RAD = math.radians(67.748 / 2.0)
STEREO_RANGE_CAP_DEFAULT_M = 25.0
_R3_BLIND_WEDGES = (
    (R3_HALF_WIDTH_RAD, math.pi - R3_HALF_WIDTH_RAD),        # left side
    (-math.pi + R3_HALF_WIDTH_RAD, -R3_HALF_WIDTH_RAD),      # right side
)

_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# Azimuth interval algebra
#
# The admitted azimuth set is computed ONCE per spec and answers both questions
# — is this bearing inside E, and how much azimuth does E span. Deriving the
# two separately is how a blind wedge ends up excluded from the numerator and
# still present in the denominator.
# ---------------------------------------------------------------------------


def _split_wrapping(lo: float, hi: float) -> list[tuple[float, float]]:
    """Normalise one (start, end) sweep into non-wrapping intervals on [-pi, pi]."""
    span = (hi - lo) % _TWO_PI
    if span == 0.0 and hi != lo:
        span = _TWO_PI  # a full-circle wedge, stated as such
    start = math.remainder(lo, _TWO_PI)
    end = start + span
    if end <= math.pi:
        return [(start, end)]
    return [(start, math.pi), (-math.pi, end - _TWO_PI)]


def _subtract(base: list[tuple[float, float]], cut: tuple[float, float]) -> list[tuple[float, float]]:
    lo, hi = cut
    out: list[tuple[float, float]] = []
    for a, b in base:
        if hi <= a or lo >= b:
            out.append((a, b))
            continue
        if a < lo:
            out.append((a, min(lo, b)))
        if b > hi:
            out.append((max(hi, a), b))
    return [(a, b) for a, b in out if b > a]


def _admitted_azimuth(spec: "RegionSpec") -> tuple[tuple[float, float], ...]:
    if spec.coverage_config in ("R2", "R3"):
        base: list[tuple[float, float]] = [(-math.pi, math.pi)]
    else:
        half = float(spec.azimuth_half_width_rad)
        base = [(-half, half)]
    for wedge in spec.blind_wedges_rad:
        for piece in _split_wrapping(*wedge):
            base = _subtract(base, piece)
    return tuple(sorted(base))


@dataclass(frozen=True)
class RegionSpec:
    """E, as data. Built from config; never hand-constructed inside a stage.

    E = { (r, theta) : 0 < r <= r_max_m, theta in admitted azimuth set }

    The admitted azimuth set is the coverage config's base span minus the
    documented blind wedges:

      R1 : base = [-azimuth_half_width_rad, +azimuth_half_width_rad]
      R2 : base = the full circle

    `blind_wedges_rad` are (start, end) pairs in ego-frame radians, swept
    counter-clockwise from start to end; a pair may wrap through +/-pi. Under R2
    the spec targets <= 15 deg of total blind wedge from the mounting. On this
    substrate the nuScenes ring has no documented blind wedge, so the default is
    empty — which is a measured claim about nuScenes, not a modelling shortcut,
    and `azimuth_measure_rad` will read as the full 2*pi to prove it.

    Vertical extent is deliberately absent: `in_region(x, y, spec)` decides
    membership in the BEV region only. The height cap is a separate config
    filter applied to points, not part of E (spec §3.6 defers vertical extent to
    the LiDAR FOV).
    """

    coverage_config: str
    r_max_m: float = _R_MAX_M
    rho_radius_m: float = _RHO_RADIUS_M
    azimuth_half_width_rad: float | None = _R1_HALF_WIDTH_RAD  # R1 only
    blind_wedges_rad: tuple[tuple[float, float], ...] = ()
    provenance: str = "spec §3.6"

    def __post_init__(self) -> None:
        if self.coverage_config not in COVERAGE_CONFIGS:
            raise ValueError(
                f"coverage_config={self.coverage_config!r} is not one of {COVERAGE_CONFIGS}"
            )
        for name in ("r_max_m", "rho_radius_m"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite value, got {value!r}")
        if self.coverage_config == "R1":
            half = self.azimuth_half_width_rad
            if half is None or not math.isfinite(half) or not 0.0 < half <= math.pi:
                raise ValueError(
                    f"R1 requires azimuth_half_width_rad in (0, pi], got {half!r}"
                )
        if self.coverage_config == "R3" and not self.blind_wedges_rad:
            raise ValueError(
                "R3 requires the two side blind wedges; use R3_DEFAULT or region_spec_from_config"
            )
        for lo, hi in self.blind_wedges_rad:
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError(f"blind wedge must be finite, got {(lo, hi)!r}")
        # Cache the admitted azimuth set once: it is pure geometry and both
        # membership and area must be answered from the SAME set, or a wedge
        # excluded from the count still contributes to the denominator.
        object.__setattr__(self, "_admitted", _admitted_azimuth(self))

    @property
    def admitted_azimuth_rad(self) -> tuple[tuple[float, float], ...]:
        """Disjoint, ascending, non-wrapping intervals on [-pi, pi]."""
        return getattr(self, "_admitted")

    @property
    def azimuth_measure_rad(self) -> float:
        """Total admitted azimuth. 2*pi for an unobstructed R2 ring."""
        return float(sum(hi - lo for lo, hi in self.admitted_azimuth_rad))

    @property
    def blind_wedge_measure_rad(self) -> float:
        base = _TWO_PI if self.coverage_config in ("R2", "R3") else 2.0 * float(self.azimuth_half_width_rad)
        return float(base - self.azimuth_measure_rad)

    def area_m2(self, radius_m: float | None = None) -> float:
        """Area of E intersected with a disc of `radius_m` (default r_max_m).

        This is rho's denominator. It is derived from the admitted azimuth set,
        so a blind wedge shrinks the count and the area together and rho stays
        comparable across R1/R2 and across blind-wedge layouts.
        """
        r = float(self.r_max_m if radius_m is None else radius_m)
        if not math.isfinite(r) or r <= 0.0:
            raise ValueError(f"radius_m must be a positive finite value, got {radius_m!r}")
        effective = min(r, float(self.r_max_m))
        return 0.5 * self.azimuth_measure_rad * effective * effective

    def as_dict(self) -> dict:
        """Serialisable form; recorded alongside every rho so E is reconstructible."""
        return {
            "coverage_config": self.coverage_config,
            "r_max_m": float(self.r_max_m),
            "rho_radius_m": float(self.rho_radius_m),
            "azimuth_half_width_rad": (
                None if self.azimuth_half_width_rad is None else float(self.azimuth_half_width_rad)
            ),
            "blind_wedges_rad": [[float(a), float(b)] for a, b in self.blind_wedges_rad],
            "azimuth_measure_rad": self.azimuth_measure_rad,
            "provenance": self.provenance,
        }


# The two configs as the spec defines them. R2 is the pilot default.
R1_DEFAULT = RegionSpec(coverage_config="R1")
R2_DEFAULT = RegionSpec(
    coverage_config="R2",
    azimuth_half_width_rad=None,
    provenance="spec §3.6; nuScenes ring has no documented blind wedge (measured, v1.0-mini)",
)
R3_DEFAULT = RegionSpec(
    coverage_config="R3", r_max_m=STEREO_RANGE_CAP_DEFAULT_M, azimuth_half_width_rad=None,
    blind_wedges_rad=_R3_BLIND_WEDGES,
    provenance="2026-09-12 approach A: the two ZED frusta; cap = STEREO_RANGE_CAP_DEFAULT_M until measured",
)


def region_spec_from_config(config: dict) -> RegionSpec:
    """Build E from the eval-region block of `pipeline_pilot.yaml`.

    `coverage_config` is required and has no default: the choice determines the
    region, rho's normalisation, whether the multi-camera union is needed at
    all, and roughly 6x of GPU time. A default here would make that decision by
    omission, which is how it went unmade in rev 1.
    """
    if not isinstance(config, dict):
        raise TypeError(f"eval-region config must be a mapping, got {type(config).__name__}")
    if "coverage_config" not in config:
        raise ValueError("eval-region config must state coverage_config explicitly (R1, R2 or R3)")

    coverage = config["coverage_config"]
    known = {
        "coverage_config",
        "r_max_m",
        "rho_radius_m",
        "azimuth_half_width_rad",
        "blind_wedges_rad",
        "provenance",
    }
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(f"unknown eval-region config key(s): {unknown}")

    half = config.get("azimuth_half_width_rad", _R1_HALF_WIDTH_RAD if coverage == "R1" else None)
    wedges_in = config.get("blind_wedges_rad", ()) or ()
    if not wedges_in and coverage == "R3":
        wedges = _R3_BLIND_WEDGES
    else:
        wedges = tuple((float(lo), float(hi)) for lo, hi in wedges_in)
    r_max_default = STEREO_RANGE_CAP_DEFAULT_M if coverage == "R3" else _R_MAX_M
    return RegionSpec(
        coverage_config=str(coverage),
        r_max_m=float(config.get("r_max_m", r_max_default)),
        rho_radius_m=float(config.get("rho_radius_m", _RHO_RADIUS_M)),
        azimuth_half_width_rad=None if half is None else float(half),
        blind_wedges_rad=wedges,
        provenance=str(config.get("provenance", "configs/pipeline_pilot.yaml")),
    )


# ---------------------------------------------------------------------------
# Membership and density
# ---------------------------------------------------------------------------


def in_region(x_m, y_m, spec: RegionSpec, *, frame: str = EGO):
    """Membership in E. The single implementation; nothing else decides this.

    `x_m` / `y_m` are ego-frame BEV coordinates, scalar or array-like of the
    same shape. Returns a bool for scalar input and a bool ndarray otherwise.

    r = 0 is excluded (spec §3.6 writes 0 < r), interval boundaries are
    inclusive on the admitted side, and non-finite coordinates fall out as False
    rather than raising — a NaN centre is a defect in the producing stage and is
    counted there, not silently admitted here.
    """
    if check_frame(frame) != EGO:
        raise ValueError(
            f"in_region requires ego-frame coordinates (ISO 8855), got frame={frame!r}; "
            "LiDAR-frame x/y select the wrong sector of the world under R1 and mis-place "
            "every blind wedge under R2"
        )
    x = np.asarray(x_m, dtype=np.float64)
    y = np.asarray(y_m, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"x_m and y_m must have the same shape, got {x.shape} and {y.shape}")
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)

    with np.errstate(invalid="ignore"):
        r = np.hypot(x, y)
        inside = (r > 0.0) & (r <= float(spec.r_max_m))
        theta = np.arctan2(y, x)
        admitted = np.zeros(theta.shape, dtype=bool)
        for lo, hi in spec.admitted_azimuth_rad:
            admitted |= (theta >= lo) & (theta <= hi)
        inside &= admitted
    inside &= np.isfinite(x) & np.isfinite(y)

    return bool(inside[0]) if scalar else inside


@dataclass(frozen=True)
class Density:
    """One frame's rho, with everything needed to reproduce it.

    Recorded whole. A bare float loses which E produced it, and the R1/R2
    distinction is exactly what makes two rho values incomparable.
    """

    rho_per_m2: float
    n_agents: float
    area_m2: float
    radius_m: float
    coverage_config: str
    spec: dict = field(repr=False, default_factory=dict)


def rho(centers_xy_m, spec: RegionSpec, *, n_min=None, frame: str = EGO) -> Density:
    """Density: annotated agents inside E within `spec.rho_radius_m`, over area.

        rho(frame) = |{ boxes : centre in E, r <= R }| / area(E ∩ disc(R))

    Count over area, per spec §8.3.1 — never a bare count. The area
    normalisation is what makes rho comparable across R1/R2 and across
    blind-wedge layouts.

    `centers_xy_m` is (N, 2) ego-frame box centres. `n_min` optionally supplies
    per-centre weights so a group/ignore box contributes its annotated
    lower-bound count instead of 1 (spec §A.2). Group boxes have no producer in
    the pilot (§4, waived), so `n_min` is normally None; the parameter exists so
    that when they do appear the term is applied here rather than bolted on by a
    caller that also owns the denominator.
    """
    centers = np.asarray(centers_xy_m, dtype=np.float64)
    if centers.size == 0:
        centers = centers.reshape(0, 2)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"centers_xy_m must be (N, 2), got {centers.shape}")

    radius_m = float(spec.rho_radius_m)
    inside = in_region(centers[:, 0], centers[:, 1], spec, frame=frame)
    within = inside & (np.hypot(centers[:, 0], centers[:, 1]) <= radius_m)

    if n_min is None:
        n_agents = float(np.count_nonzero(within))
    else:
        weights = np.asarray(n_min, dtype=np.float64).reshape(-1)
        if weights.shape[0] != centers.shape[0]:
            raise ValueError(
                f"n_min must have one entry per centre, got {weights.shape[0]} for {centers.shape[0]}"
            )
        if np.any(weights < 0) or not np.all(np.isfinite(weights)):
            raise ValueError("n_min entries must be finite and non-negative")
        n_agents = float(weights[within].sum())

    area_m2 = spec.area_m2(radius_m)
    return Density(
        rho_per_m2=n_agents / area_m2,
        n_agents=n_agents,
        area_m2=area_m2,
        radius_m=radius_m,
        coverage_config=spec.coverage_config,
        spec=spec.as_dict(),
    )
