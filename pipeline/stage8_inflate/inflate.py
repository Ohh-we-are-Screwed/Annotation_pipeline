#!/usr/bin/env python3
"""Stage 8 — prior-anchored amodal inflation (§5.9, M-12, Phase 8).

`comprehensive.md` §7.3.8 is one line — "sparse boxes inflated toward `priors_v1`
class means, anchored to the LiDAR-return surface (shift away from ego)" — and
A.4 supplies `dims: {mu, sigma}` and nothing else. That leaves **how much, when,
along which axes, and how far is too far** undefined (M-12), and an undefined
inflation still produces boxes: bigger ones, uniformly, with no field saying so.
Every one of those four is an explicit, recorded config value here.

**The anchor, stated correctly.** LiDAR sees surfaces, so the observed face of an
object is always the **near** face — the one pointing at the sensor. The rule is
therefore *hold the near face still and grow the far face*, not the vaguer "shift
outward". The two differ by exactly half the growth on every box, in the
direction of the sensor, and "shift outward" is the reading that walks the
measured surface off the returns that produced it.

Three guards stand behind that (§5.9):

  1. **The near face may not move.** It is re-derived after inflation and
     compared to its own pre-inflation position; a mismatch is a defect in this
     module's arithmetic and stops the run rather than shipping.
  2. **The far face must move away from the sensor**, never toward it. This is
     the "no growth back through the sensor-facing surface" condition, checked as
     a distance comparison rather than assumed from the sign algebra.
  3. **The box may not grow to enclose the sensor.** A box whose footprint
     swallows the LiDAR origin is not an amodal completion, it is a fitting
     failure; the inflation is reverted whole and the reason recorded.

**Which face is near is a decision, and near the box's own mid-plane it is a coin
flip.** When the sensor sits within `near_face_margin_m` of an axis' mid-plane,
the anchor is ambiguous: the axis grows symmetrically and says so
(`anchor: "symmetric_ambiguous"`), because committing to a near face there means
committing to a sign derived from millimetres of noise.

**Height anchors the top, not the near face.** Stage 1's ground removal strips a
0.3 m band, so the *bottom* of every object is the missing part and the top is
the measured one. Growth is downward. This is a different anchor from the two BEV
axes and is recorded as such rather than folded into "anchor the near face".

**The yaw failure this stage amplifies.** Under a 90-degree yaw error from Stage
6 — systematic for near-square footprints (§5.7) — anchoring and growing pushes
the box sideways into the neighbouring lane, deterministically and invisibly.
Stage 6's `yaw_ambiguous` flag travels into every record here and is counted in
the manifest; `inflate_when_yaw_ambiguous: false` turns the amplification off at
the cost of leaving those boxes uninflated.

**The prior is not automatically in the box's convention.** nuScenes `size` is
[w, l, h] with `l` measured along the object's OWN heading; Stage 6 defines the
heading as the long BEV side (§5.7), so every box it produces has `w <= l`. For a
class whose GT heading crosses its long side — measured on this substrate,
`a road barrier` has mean w 2.42 m and mean l 0.58 m — the two conventions name
different axes, and blending width toward the prior's width would grow a 0.6 m
box toward 2.4 m *across its own fitted axis*. `prior_axis_mapping` decides which
reading is used and is recorded on every box.

**Every inflated box records `inflated: bool` and `inflation_fraction`** (§5.9).
Without them, a run where ground removal stripped most object points produces
boxes that are ~90 % prior and 10 % measurement and nothing in the output says
so. `inflation_fraction` is defined once, here, as the fraction of the shipped
box's VOLUME that is not measured: `1 - V_measured / V_inflated`. Per-axis
fractions are recorded alongside it, because a box that grew only in height and
one that grew only in length are different objects with the same scalar.

**Stage 9 gates on the measured dimensions, not these** (P1-12). §7.3.9's spatial
sanity check is "BEV box exceeds 2x class prior" and this stage grows boxes
*toward* the prior, so any box through here passes that gate more easily — the
gate is weakest exactly where it is needed. That is an inherited spec flaw; the
pilot surfaces it by shipping `box_measured` next to `box` in every row and
saying which one the gate must read.

**Stage 7 is not in this chain yet.** When it lands, it must run BEFORE this
stage: it enforces yaw consistency along tracks (§4), and the yaw is what decides
which face is near. `--boxes-dir` points at whatever produced the boxes and the
producing spec is recorded in the manifest.

    python3 -m pipeline.stage8_inflate.inflate [--paths configs/paths.yaml]

Exit codes:
    0  every triggered box resolved under contract
    1  ran, but at least one scene triggered inflation and could apply none
    2  upstream contract broken; nothing was written
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.common.conventions import EGO  # noqa: E402
from pipeline.common.paths import (  # noqa: E402
    PathValidationError,
    Paths,
    assert_dataroot_read_only,
    load_paths,
    metadata_fingerprint,
)
from pipeline.common.schemas import KeyframeRecord, read_records  # noqa: E402
from pipeline.common.manifest import (  # noqa: E402
    UpstreamRefusal,
    clear_markers,
    require_upstream,
    write_json_atomic,
    write_jsonl_atomic,
    write_marker,
)
from pipeline.stage0_data_probe.probe import Substrate  # noqa: E402
from pipeline.stage6_cluster.cluster import STATUS_FIT  # noqa: E402
from pipeline.stage6_cluster.priors import PRIORS_NAME, ClassPrior, Priors, load_priors  # noqa: E402

STAGE = "stage8_inflate"
STAGE_SPEC = "dhakascenes-pilot/stage8_inflate/v1"

EXIT_OK = 0
EXIT_DEGRADED = 1
EXIT_REFUSED = 2

TRIGGERS: tuple[str, ...] = ("point_count",)
BLENDS: tuple[str, ...] = ("point_count_weighted", "to_mean")
SENSOR_ORIGINS: tuple[str, ...] = ("lidar_calibrated_sensor", "ego_origin")
PRIOR_AXIS_MAPPINGS: tuple[str, ...] = ("sorted_short_long", "as_stated")

# The nuScenes [w, l, h] order (§3.2), named once. `w` is lateral (the box's own
# +y), `l` is longitudinal (along yaw), `h` is vertical.
AXES: tuple[str, ...] = ("w", "l", "h")

# Why a box was not inflated. Recorded per row: "not inflated" and "inflated by
# zero" are the same number and different facts.
REASON_NOT_TRIGGERED = "not_triggered"
REASON_NO_BOX = "no_box"
REASON_NO_PRIOR = "no_prior_for_class"
REASON_PRIOR_NO_DIMS = "prior_has_no_dims"
REASON_YAW_AMBIGUOUS = "yaw_ambiguous_and_disabled"
REASON_ENCLOSES_SENSOR = "would_enclose_sensor"
REASON_SENSOR_INSIDE = "sensor_inside_measured_box"
REASON_NO_GROWTH = "measured_exceeds_prior_on_every_axis"


class InflationContractError(RuntimeError):
    """An inflation input, or an inflated box, violated the Stage 8 contract."""


# ---------------------------------------------------------------------------
# Config — M-12's four undefined quantities, made explicit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InflationConfig:
    """Stage 8 tunables. None of these may appear as a literal in the code below."""

    # --- WHEN (the trigger) ---
    trigger: str = "point_count"
    trigger_max_points: int = 15

    # --- HOW MUCH (the blend) ---
    blend: str = "point_count_weighted"
    one_sided: bool = True

    # --- ALONG WHICH AXES ---
    axes: tuple[str, ...] = AXES
    prior_axis_mapping: str = "sorted_short_long"
    bev_anchor: str = "near_face"
    height_anchor: str = "top"
    near_face_margin_m: float = 0.05
    sensor_origin: str = "lidar_calibrated_sensor"

    # --- THE CLAMP ---
    max_growth_ratio: float = 3.0
    max_growth_m: float = 3.0
    clamp_sigma: float = 0.0

    # --- the Stage 6 failure this stage amplifies (§5.9) ---
    inflate_when_yaw_ambiguous: bool = True

    # --- assertions ---
    guard_tol_m: float = 1e-9

    # --- upstream gate (C16) ---
    accept_degraded_upstream: bool = False

    # --- determinism (§1.9) ---
    global_seed: int = 20260812

    provenance: dict = field(
        default_factory=lambda: {
            "trigger": (
                "M-12: A.4 defines no trigger. point_count is the only signal this stage has that "
                "correlates with amodal incompleteness"
            ),
            "trigger_max_points": (
                "comprehensive.md §7.3.6's sparse-cluster guard (< 15 pts). Inherited, unvalidated "
                "on this substrate (§10) — and NOT the same number as §7.3.9's >= 5 gate, which "
                "Stage 9 owns"
            ),
            "blend": (
                "M-12: 'inflate toward the class mean' admits all-the-way-to-mean and "
                "point-count-weighted. point_count_weighted makes a 14-return box move a little and "
                "a 1-return box move almost all the way; to_mean makes both the prior"
            ),
            "one_sided": (
                "inflation never SHRINKS a measured dimension. A box larger than its class mean is "
                "evidence — a truck labelled as a car, or a merged cluster — and pulling it down to "
                "the mean deletes that evidence and hides the upstream error"
            ),
            "axes": "M-12's per-axis applicability, stated rather than assumed to be all three",
            "prior_axis_mapping": (
                "nuScenes' `l` is the extent along the object's OWN heading; Stage 6's is the LONG "
                "BEV side (§5.7). Measured on this substrate, 'a road barrier' has mean w 2.42 and "
                "mean l 0.58, so for that class the two conventions name different axes. "
                "sorted_short_long maps the prior's short/long means onto w/l; as_stated takes A.4 "
                "literally and grows the box across its own fitted axis"
            ),
            "bev_anchor": (
                "§5.9: LiDAR sees surfaces, so the observed face is the NEAR one. Anchor it and "
                "grow the far face — not 'shift outward', which moves the measured surface"
            ),
            "height_anchor": (
                "top: Stage 1's ground removal strips a 0.3 m band, so the BOTTOM is the missing "
                "part of every object and the top is the measured one. Growth is downward"
            ),
            "near_face_margin_m": (
                "below this, the sensor is on the box's mid-plane and 'which face is near' is a "
                "sign derived from noise; the axis grows symmetrically and is flagged"
            ),
            "sensor_origin": (
                "lidar_calibrated_sensor: the LiDAR origin in ego frame, not the ego origin. On "
                "this substrate they differ by ~0.94 m forward and ~1.84 m up, which is the same "
                "order as the near-face decision for a close object"
            ),
            "max_growth_ratio": "M-12's clamp, multiplicative; arbitrary, needs tuning",
            "max_growth_m": "M-12's clamp, absolute; arbitrary, needs tuning",
            "clamp_sigma": (
                "target <= mu + clamp_sigma * sigma. 0.0 means the class mean is the ceiling: a "
                "blend toward mu can never overshoot it, and the assertion says so"
            ),
            "inflate_when_yaw_ambiguous": (
                "§5.9: under a 90 deg yaw error inflation grows the box sideways into the "
                "neighbouring lane. Left ON by default so the count is visible in the output rather "
                "than absent from it; the flag travels into every record"
            ),
            "accept_degraded_upstream": "C16 — consuming a DEGRADED (complete, quality-flagged) "
            "Stage 1 or box-producer output is an explicit recorded decision, never a default",
            "global_seed": "§1.9, one global seed, recorded (nothing here samples; recorded anyway)",
        }
    )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.trigger not in TRIGGERS:
            errors.append(f"trigger={self.trigger!r} is not one of {TRIGGERS}")
        if self.blend not in BLENDS:
            errors.append(f"blend={self.blend!r} is not one of {BLENDS}")
        if self.sensor_origin not in SENSOR_ORIGINS:
            errors.append(f"sensor_origin={self.sensor_origin!r} is not one of {SENSOR_ORIGINS}")
        if self.trigger_max_points < 1:
            errors.append(f"trigger_max_points={self.trigger_max_points} must be >= 1")
        unknown = [axis for axis in self.axes if axis not in AXES]
        if unknown:
            errors.append(f"axes {unknown} are not among {AXES} ([w, l, h] order, §3.2)")
        if self.prior_axis_mapping not in PRIOR_AXIS_MAPPINGS:
            errors.append(
                f"prior_axis_mapping={self.prior_axis_mapping!r} is not one of {PRIOR_AXIS_MAPPINGS}"
            )
        if self.bev_anchor != "near_face":
            errors.append(
                f"bev_anchor={self.bev_anchor!r}: §5.9 states the near face is the anchor; no other "
                "rule is implemented, and a silently different one moves the measured surface"
            )
        if self.height_anchor not in ("top", "bottom", "symmetric"):
            errors.append(f"height_anchor={self.height_anchor!r} is not one of top/bottom/symmetric")
        if self.near_face_margin_m < 0.0:
            errors.append(f"near_face_margin_m={self.near_face_margin_m} must be >= 0")
        if self.max_growth_ratio < 1.0:
            errors.append(f"max_growth_ratio={self.max_growth_ratio} must be >= 1 (inflation grows)")
        if self.max_growth_m < 0.0:
            errors.append(f"max_growth_m={self.max_growth_m} must be >= 0")
        if self.clamp_sigma < 0.0:
            errors.append(f"clamp_sigma={self.clamp_sigma} must be >= 0")
        return errors

    def as_dict(self) -> dict:
        payload = dict(self.__dict__)
        payload["axes"] = list(self.axes)
        return payload


# ---------------------------------------------------------------------------
# The blend, and the clamp (M-12)
# ---------------------------------------------------------------------------


def blend_weight(n_points: int, cfg: InflationConfig) -> float:
    """How far toward the class mean this box travels, in [0, 1].

    `to_mean` is 1.0 for every triggered box: a 14-return box becomes the prior
    exactly as hard as a 1-return box does. `point_count_weighted` scales it by
    how little was measured, which is the quantity the trigger is testing in the
    first place.
    """
    if cfg.blend == "to_mean":
        return 1.0
    return float(min(1.0, max(0.0, 1.0 - n_points / float(cfg.trigger_max_points))))


@dataclass
class AxisPlan:
    """One axis' arithmetic, before any of it is applied."""

    axis: str
    measured_m: float
    mu_m: float
    sigma_m: float
    target_m: float
    delta_m: float
    clamps: list

    @property
    def grows(self) -> bool:
        return self.delta_m > 0.0


def prior_extents(prior: ClassPrior, cfg: InflationConfig) -> tuple[dict, dict, str]:
    """Express the prior in Stage 6's box convention. Measured, not cosmetic.

    nuScenes `size` is [w, l, h] with `l` along the object's OWN heading, and on
    this substrate `a road barrier` has mean w 2.42 and mean l 0.58: a fence
    panel's heading points through its thin direction, so the class' long side
    lives in `w`. Stage 6 defines the heading as the LONG BEV side (§5.7), which
    makes `w <= l` true of every box it produces. For such a class the two
    conventions therefore name DIFFERENT axes, and blending the box's width
    toward the prior's width would grow a 0.6 m-wide fitted box toward 2.4 m
    across its own fitted axis — §5.9's sideways growth, arrived at through the
    prior instead of through a yaw error.

    `sorted_short_long` maps the prior's short and long means onto the box's `w`
    and `l`. `as_stated` takes A.4 literally and is kept so the difference can be
    measured rather than argued; the choice is recorded on every box.
    """
    mu = {axis: float(prior.mu(axis)) for axis in AXES}
    sigma = {axis: float(prior.sigma(axis)) for axis in AXES}
    if cfg.prior_axis_mapping == "as_stated" or mu["w"] <= mu["l"]:
        return mu, sigma, "as_stated"
    swapped_mu = {"w": mu["l"], "l": mu["w"], "h": mu["h"]}
    swapped_sigma = {"w": sigma["l"], "l": sigma["w"], "h": sigma["h"]}
    return swapped_mu, swapped_sigma, f"sorted_short_long:swapped:w_gt_l={mu['w'] / mu['l']:.3f}"


def plan_axis(
    axis: str,
    measured_m: float,
    mu: float,
    sigma: float,
    alpha: float,
    cfg: InflationConfig,
    ceiling_m: float | None = None,
) -> AxisPlan:
    """Blend toward mu, then clamp. Never shrinks (`one_sided`)."""
    mu = float(mu)
    sigma = float(sigma)
    clamps: list[str] = []

    if cfg.one_sided and measured_m >= mu:
        return AxisPlan(axis, measured_m, mu, sigma, measured_m, 0.0, ["measured_exceeds_mu"])

    target = measured_m + alpha * (mu - measured_m)

    ceiling = mu + cfg.clamp_sigma * sigma
    if target > ceiling:
        # Unreachable while the blend is a convex combination with mu; asserted
        # rather than assumed, because a future blend function that overshoots
        # would otherwise pass every other check in this file.
        clamps.append("clamp_sigma")
        target = ceiling
    if target > measured_m * cfg.max_growth_ratio:
        clamps.append("max_growth_ratio")
        target = measured_m * cfg.max_growth_ratio
    if target - measured_m > cfg.max_growth_m:
        clamps.append("max_growth_m")
        target = measured_m + cfg.max_growth_m
    if ceiling_m is not None and target > ceiling_m:
        clamps.append("w_le_l")
        target = ceiling_m

    target = max(target, measured_m)
    return AxisPlan(axis, measured_m, mu, sigma, target, target - measured_m, clamps)


# ---------------------------------------------------------------------------
# Anchoring (§5.9) and its three guards
# ---------------------------------------------------------------------------


@dataclass
class AxisAnchor:
    """Where the growth went, and the evidence that it went the right way."""

    axis: str
    anchor: str
    near_face_sign: float
    centre_shift_m: float
    near_face_before_m: float
    near_face_after_m: float
    far_face_before_m: float
    far_face_after_m: float


def anchor_growth(
    axis: str,
    centre_along_m: float,
    extent_m: float,
    delta_m: float,
    sensor_along_m: float,
    cfg: InflationConfig,
) -> AxisAnchor:
    """Hold the near face, push the far one out. All quantities are 1D along one axis.

    `centre_along_m` and `sensor_along_m` are the box centre and the sensor
    origin projected onto this axis' unit vector. Working in that 1D coordinate
    is what keeps the near-face decision from silently depending on the frame the
    caller happened to hand over.
    """
    offset = sensor_along_m - centre_along_m
    if abs(offset) <= cfg.near_face_margin_m:
        # The sensor is on the mid-plane: no face is meaningfully nearer, and a
        # sign taken from millimetres would flip between two runs of two nearly
        # identical clusters.
        sign = 0.0
        shift = 0.0
    else:
        sign = 1.0 if offset > 0.0 else -1.0
        shift = -sign * delta_m / 2.0

    half_before = extent_m / 2.0
    half_after = (extent_m + delta_m) / 2.0
    centre_after = centre_along_m + shift
    if sign == 0.0:
        # Symmetric growth: both faces move, and "near"/"far" are recorded as the
        # +/- faces so the numbers below still mean something definite.
        near_before, far_before = centre_along_m + half_before, centre_along_m - half_before
        near_after, far_after = centre_after + half_after, centre_after - half_after
        anchor = "symmetric_ambiguous"
    else:
        near_before = centre_along_m + sign * half_before
        far_before = centre_along_m - sign * half_before
        near_after = centre_after + sign * half_after
        far_after = centre_after - sign * half_after
        anchor = "near_face"

        # Guard 1: the anchored face did not move.
        if abs(near_after - near_before) > cfg.guard_tol_m:
            raise InflationContractError(
                f"axis {axis}: the near face moved from {near_before!r} to {near_after!r} while "
                f"growing by {delta_m!r}. §5.9's rule is anchor-the-near-face; a near face that "
                "moves is 'shift outward', which walks the measured surface off its own returns"
            )
        # Guard 2: the far face moved AWAY from the sensor, never back through it.
        if abs(sensor_along_m - far_after) + cfg.guard_tol_m < abs(sensor_along_m - far_before):
            raise InflationContractError(
                f"axis {axis}: the far face moved from {far_before!r} to {far_after!r}, toward the "
                f"sensor at {sensor_along_m!r}. Growth back through the sensor-facing surface (§5.9)"
            )

    return AxisAnchor(
        axis=axis,
        anchor=anchor,
        near_face_sign=sign,
        centre_shift_m=shift,
        near_face_before_m=near_before,
        near_face_after_m=near_after,
        far_face_before_m=far_before,
        far_face_after_m=far_after,
    )


def footprint_contains(
    centre_xy: np.ndarray,
    heading: np.ndarray,
    lateral: np.ndarray,
    length_m: float,
    width_m: float,
    point_xy: np.ndarray,
) -> bool:
    """Is `point_xy` inside the box's BEV footprint? Guard 3's predicate."""
    delta = point_xy - centre_xy
    return bool(
        abs(float(delta @ heading)) <= length_m / 2.0 and abs(float(delta @ lateral)) <= width_m / 2.0
    )


# ---------------------------------------------------------------------------
# One box
# ---------------------------------------------------------------------------


def inflate_box(
    box: dict,
    n_points: int,
    prior: ClassPrior | None,
    sensor_xyz_m: np.ndarray,
    cfg: InflationConfig,
) -> tuple[dict, dict]:
    """(inflated box, inflation ledger) for one Stage 6 box.

    The returned box is a new dict; the caller keeps the measured one verbatim so
    Stage 9's spatial gate has something to gate on that this stage did not touch
    (P1-12).
    """
    width_m, length_m, height_m = (float(v) for v in box["size_wlh_m"])
    centre = np.asarray(box["translation_m"], dtype=np.float64)
    yaw = float(box["yaw_rad"])
    heading = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
    lateral = np.asarray([-math.sin(yaw), math.cos(yaw)], dtype=np.float64)

    ledger: dict = {
        "trigger": cfg.trigger,
        "trigger_max_points": cfg.trigger_max_points,
        "n_points": int(n_points),
        "triggered": bool(n_points < cfg.trigger_max_points),
        "blend": cfg.blend,
        "alpha": 0.0,
        "inflated": False,
        "inflation_fraction": 0.0,
        "inflation_fraction_definition": "1 - V_measured / V_inflated",
        "inflation_fraction_per_axis": {axis: 0.0 for axis in AXES},
        "axes": {},
        "sensor_origin_ego_m": [round(float(v), 4) for v in sensor_xyz_m],
        "sensor_origin_source": cfg.sensor_origin,
        "yaw_ambiguous": bool(box.get("yaw_ambiguous", False)),
        "reason": None,
        "prior": None,
    }

    if not ledger["triggered"]:
        ledger["reason"] = REASON_NOT_TRIGGERED
        return dict(box), ledger
    if prior is None:
        ledger["reason"] = REASON_NO_PRIOR
        return dict(box), ledger
    if prior.dims is None:
        ledger["reason"] = REASON_PRIOR_NO_DIMS
        ledger["prior"] = {"class_name": prior.class_name, "n_instances": prior.n_instances, "gaps": list(prior.gaps)}
        return dict(box), ledger
    if ledger["yaw_ambiguous"] and not cfg.inflate_when_yaw_ambiguous:
        ledger["reason"] = REASON_YAW_AMBIGUOUS
        return dict(box), ledger

    mu, sigma, axis_mapping = prior_extents(prior, cfg)
    ledger["prior"] = {
        "class_name": prior.class_name,
        "source": prior.source,
        "n_instances": prior.n_instances,
        "mu_wlh_m_as_stated": [prior.mu("w"), prior.mu("l"), prior.mu("h")],
        "sigma_wlh_m_as_stated": [prior.sigma("w"), prior.sigma("l"), prior.sigma("h")],
        "mu_wlh_m_used": [mu["w"], mu["l"], mu["h"]],
        "axis_mapping": axis_mapping,
        "gaps": list(prior.gaps),
    }
    alpha = blend_weight(n_points, cfg)
    ledger["alpha"] = round(alpha, 6)

    # Length first, then width capped by it: the [w, l, h] convention is w <= l
    # (§3.2, §5.7), and two axes blended independently toward two independent
    # means can cross. Capping is the only repair that does not rotate the box.
    plans: dict[str, AxisPlan] = {}
    plans["l"] = (
        plan_axis("l", length_m, mu["l"], sigma["l"], alpha, cfg)
        if "l" in cfg.axes
        else AxisPlan("l", length_m, mu["l"], sigma["l"], length_m, 0.0, ["axis_disabled"])
    )
    plans["w"] = (
        plan_axis("w", width_m, mu["w"], sigma["w"], alpha, cfg, ceiling_m=plans["l"].target_m)
        if "w" in cfg.axes
        else AxisPlan("w", width_m, mu["w"], sigma["w"], width_m, 0.0, ["axis_disabled"])
    )
    plans["h"] = (
        plan_axis("h", height_m, mu["h"], sigma["h"], alpha, cfg)
        if "h" in cfg.axes
        else AxisPlan("h", height_m, mu["h"], sigma["h"], height_m, 0.0, ["axis_disabled"])
    )

    if not any(plan.grows for plan in plans.values()):
        ledger["reason"] = REASON_NO_GROWTH
        ledger["axes"] = {axis: _axis_record(plan, None) for axis, plan in plans.items()}
        return dict(box), ledger

    sensor_xy = sensor_xyz_m[:2]
    # Guard 3, first half: a measured box that already contains the sensor is a
    # fitting failure (a mask that painted points all around the ego), and
    # anchoring inside it is meaningless — there is no near face.
    if footprint_contains(centre[:2], heading, lateral, length_m, width_m, sensor_xy):
        ledger["reason"] = REASON_SENSOR_INSIDE
        ledger["axes"] = {axis: _axis_record(plan, None) for axis, plan in plans.items()}
        return dict(box), ledger

    anchors: dict[str, AxisAnchor] = {}
    anchors["l"] = anchor_growth(
        "l", float(centre[:2] @ heading), length_m, plans["l"].delta_m, float(sensor_xy @ heading), cfg
    )
    anchors["w"] = anchor_growth(
        "w", float(centre[:2] @ lateral), width_m, plans["w"].delta_m, float(sensor_xy @ lateral), cfg
    )
    anchors["h"] = _anchor_height(centre[2], height_m, plans["h"].delta_m, cfg)

    centre_after = centre + np.concatenate(
        (anchors["l"].centre_shift_m * heading + anchors["w"].centre_shift_m * lateral,
         [anchors["h"].centre_shift_m])
    )
    width_after = plans["w"].target_m
    length_after = plans["l"].target_m
    height_after = plans["h"].target_m

    # Guard 3, second half: the grown footprint must not swallow the sensor.
    if footprint_contains(centre_after[:2], heading, lateral, length_after, width_after, sensor_xy):
        ledger["reason"] = REASON_ENCLOSES_SENSOR
        ledger["axes"] = {axis: _axis_record(plans[axis], anchors[axis]) for axis in AXES}
        return dict(box), ledger

    if width_after > length_after + cfg.guard_tol_m:
        raise InflationContractError(
            f"inflated box has w={width_after} > l={length_after}; [w, l, h] order is broken (§3.2) "
            "and the w_le_l cap did not hold"
        )

    volume_before = width_m * length_m * height_m
    volume_after = width_after * length_after * height_after
    inflated_box = dict(box)
    inflated_box["translation_m"] = [round(float(v), 4) for v in centre_after]
    inflated_box["size_wlh_m"] = [round(width_after, 4), round(length_after, 4), round(height_after, 4)]
    inflated_box["z_min_m"] = round(float(centre_after[2] - height_after / 2.0), 4)
    inflated_box["z_max_m"] = round(float(centre_after[2] + height_after / 2.0), 4)
    inflated_box["footprint_diagonal_m"] = round(float(math.hypot(width_after, length_after)), 4)
    inflated_box["aspect_ratio_w_over_l"] = round(width_after / length_after, 4)
    # yaw, rotation_wxyz and the yaw flags are NOT touched: inflation is a
    # per-axis extent change about a fixed orientation. A stage that also nudged
    # the yaw would make the near-face anchor unverifiable after the fact.

    ledger["inflated"] = True
    ledger["axes"] = {axis: _axis_record(plans[axis], anchors[axis]) for axis in AXES}
    ledger["inflation_fraction"] = round(1.0 - volume_before / volume_after, 6)
    ledger["inflation_fraction_per_axis"] = {
        axis: round((plans[axis].target_m - plans[axis].measured_m) / plans[axis].target_m, 6)
        for axis in AXES
    }
    ledger["volume_measured_m3"] = round(volume_before, 6)
    ledger["volume_inflated_m3"] = round(volume_after, 6)
    return inflated_box, ledger


def _anchor_height(centre_z_m: float, height_m: float, delta_m: float, cfg: InflationConfig) -> AxisAnchor:
    """Height's anchor is the TOP face, so growth goes down toward the stripped ground band."""
    if cfg.height_anchor == "top":
        shift = -delta_m / 2.0
        sign = 1.0
    elif cfg.height_anchor == "bottom":
        shift = delta_m / 2.0
        sign = -1.0
    else:
        shift = 0.0
        sign = 0.0
    half_before, half_after = height_m / 2.0, (height_m + delta_m) / 2.0
    centre_after = centre_z_m + shift
    return AxisAnchor(
        axis="h",
        anchor=f"height_{cfg.height_anchor}",
        near_face_sign=sign,
        centre_shift_m=shift,
        near_face_before_m=centre_z_m + sign * half_before,
        near_face_after_m=centre_after + sign * half_after,
        far_face_before_m=centre_z_m - sign * half_before,
        far_face_after_m=centre_after - sign * half_after,
    )


def _axis_record(plan: AxisPlan, anchor: AxisAnchor | None) -> dict:
    record = {
        "measured_m": round(plan.measured_m, 4),
        "mu_m": round(plan.mu_m, 4),
        "sigma_m": round(plan.sigma_m, 4),
        "target_m": round(plan.target_m, 4),
        "delta_m": round(plan.delta_m, 4),
        "clamps": list(plan.clamps),
        "applied": False,
    }
    if anchor is not None:
        record.update(
            {
                "applied": True,
                "anchor": anchor.anchor,
                "near_face_sign": anchor.near_face_sign,
                "centre_shift_m": round(anchor.centre_shift_m, 4),
                "near_face_m": [round(anchor.near_face_before_m, 4), round(anchor.near_face_after_m, 4)],
                "far_face_m": [round(anchor.far_face_before_m, 4), round(anchor.far_face_after_m, 4)],
            }
        )
    return record


# ---------------------------------------------------------------------------
# Sensor origin (§5.9: the surface the anchor is measured from)
# ---------------------------------------------------------------------------


def sensor_origins(stage1_dir: str, scene_name: str, substrate: Substrate, cfg: InflationConfig) -> dict:
    """keyframe_token -> LiDAR origin in ego frame.

    nuScenes' `calibrated_sensor.translation` for LIDAR_TOP IS the sensor origin
    in the ego frame (the transform is sensor -> ego), so no inversion is
    involved and none is performed. On this substrate the LiDAR sits ~0.94 m
    forward and ~1.84 m above the ego origin; for an object 3 m away that offset
    is the same order as the near-face decision itself, which is why the ego
    origin is a config choice and not the default.
    """
    path = os.path.join(stage1_dir, "scenes", scene_name, "keyframes.jsonl")
    if not os.path.isfile(path):
        raise UpstreamRefusal(f"{path} not found; run `python3 -m pipeline.stage1_ingestion.ingest` first")
    calibrated = substrate.by_token("calibrated_sensor.json")
    out: dict[str, np.ndarray] = {}
    for keyframe in read_records(path, expect_type=KeyframeRecord):
        if cfg.sensor_origin == "ego_origin":
            out[keyframe.keyframe_token] = np.zeros(3, dtype=np.float64)
            continue
        record = calibrated.get(keyframe.lidar_calibrated_sensor_token)
        if record is None:
            raise InflationContractError(
                f"{keyframe.keyframe_token}: calibrated_sensor "
                f"{keyframe.lidar_calibrated_sensor_token!r} is not in the substrate"
            )
        out[keyframe.keyframe_token] = np.asarray(record["translation"], dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# Upstream
# ---------------------------------------------------------------------------


def load_upstream(
    paths: Paths, stage1_dir: str, boxes_dir: str, priors_path: str, *, accept_degraded: bool = False
):
    """Refuse to start unless the box producer finished on THIS substrate, with THESE priors.

    Both upstreams go through the C16 gate: absent marker refuses unconditionally,
    degraded marker refuses unless `accept_degraded` — one flag for both, because
    accepting one degraded upstream and refusing the other is not a meaningful
    position when the join needs both.
    """
    current = metadata_fingerprint(paths)
    upstream, boxes_marker = require_upstream(
        boxes_dir,
        stage_name="the box producer",
        module_hint="pipeline.stage6_cluster.cluster",
        current_fingerprint=current,
        accept_degraded=accept_degraded,
    )
    _, stage1_marker = require_upstream(
        stage1_dir,
        stage_name="Stage 1",
        module_hint="pipeline.stage1_ingestion.ingest",
        current_fingerprint=current,
        accept_degraded=accept_degraded,
    )
    if upstream.get("frame") != EGO:
        raise UpstreamRefusal(f"upstream boxes claim frame={upstream.get('frame')!r}, not {EGO!r}")

    priors = load_priors(priors_path)
    if priors.metadata_fingerprint and priors.metadata_fingerprint != current:
        raise UpstreamRefusal(
            f"priors fingerprint mismatch: {priors.path} was derived against "
            f"{priors.metadata_fingerprint}, this dataroot is {current}"
        )
    upstream_priors = upstream.get("upstream", {}).get("priors", {})
    if upstream_priors.get("sha256") and upstream_priors["sha256"] != priors.sha256:
        # Stage 6 chose epsilon from one priors file and Stage 8 would inflate
        # toward another's means. Both runs succeed; the boxes are a blend of two
        # class definitions and nothing in the output says which.
        raise UpstreamRefusal(
            f"priors mismatch: the box producer used {upstream_priors.get('name')}@"
            f"{upstream_priors['sha256'][:16]}, this run was given {priors.name}@{priors.sha256[:16]}"
        )
    return upstream, boxes_marker, stage1_marker, priors


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def read_boxes(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise UpstreamRefusal(f"{path} not found; run `python3 -m pipeline.stage6_cluster.cluster` first")
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def inflate_scene(
    rows: Sequence[dict],
    origins: dict,
    priors: Priors,
    cfg: InflationConfig,
) -> tuple[list[dict], dict]:
    out: list[dict] = []
    totals = {
        "n_rows": 0,
        "n_boxes": 0,
        "n_triggered": 0,
        "n_inflated": 0,
        "n_yaw_ambiguous_inflated": 0,
        "n_no_prior": 0,
        "n_prior_no_dims": 0,
        "n_enclose_sensor_refused": 0,
        "n_sensor_inside_refused": 0,
        "n_no_growth": 0,
        "n_anchor_ambiguous_axes": 0,
        "n_clamped_axes": 0,
    }
    fraction_sum = 0.0

    for row in rows:
        totals["n_rows"] += 1
        measured_box = row.get("box")
        if row.get("status") != STATUS_FIT or measured_box is None:
            out.append(
                {
                    **row,
                    "spec": STAGE_SPEC,
                    "pre_inflation_spec": row.get("spec"),
                    "box_measured": None,
                    "inflated": False,
                    "inflation_fraction": 0.0,
                    "inflation": {"inflated": False, "inflation_fraction": 0.0, "reason": REASON_NO_BOX},
                }
            )
            continue

        totals["n_boxes"] += 1
        origin = origins.get(row["keyframe_token"])
        if origin is None:
            raise InflationContractError(
                f"keyframe {row['keyframe_token']} has no Stage 1 record, so the sensor origin the "
                "near-face anchor is measured from is unknown (§5.9)"
            )
        prior = priors.get(row["class_name"])
        inflated_box, ledger = inflate_box(
            measured_box, int(row.get("num_lidar_pts", 0)), prior, origin, cfg
        )

        if ledger["triggered"]:
            totals["n_triggered"] += 1
        if ledger["inflated"]:
            totals["n_inflated"] += 1
            fraction_sum += ledger["inflation_fraction"]
            totals["n_yaw_ambiguous_inflated"] += int(ledger["yaw_ambiguous"])
            for axis_record in ledger["axes"].values():
                totals["n_anchor_ambiguous_axes"] += int(
                    axis_record.get("anchor") == "symmetric_ambiguous"
                )
                totals["n_clamped_axes"] += int(bool(axis_record["clamps"]))
        elif ledger["reason"] == REASON_NO_PRIOR:
            totals["n_no_prior"] += 1
        elif ledger["reason"] == REASON_PRIOR_NO_DIMS:
            totals["n_prior_no_dims"] += 1
        elif ledger["reason"] == REASON_ENCLOSES_SENSOR:
            totals["n_enclose_sensor_refused"] += 1
        elif ledger["reason"] == REASON_SENSOR_INSIDE:
            totals["n_sensor_inside_refused"] += 1
        elif ledger["reason"] == REASON_NO_GROWTH:
            totals["n_no_growth"] += 1

        out.append(
            {
                **row,
                "spec": STAGE_SPEC,
                "pre_inflation_spec": row.get("spec"),
                # `box` is what ships; `box_measured` is what Stage 9's spatial
                # gate must read (P1-12). Both are always present, so neither can
                # be inferred by absence.
                "box": inflated_box,
                "box_measured": measured_box,
                "inflated": ledger["inflated"],
                "inflation_fraction": ledger["inflation_fraction"],
                "inflation": ledger,
            }
        )

    totals["mean_inflation_fraction"] = round(fraction_sum / max(1, totals["n_inflated"]), 6)
    return out, totals


def run(
    paths: Paths,
    upstream_manifest: dict,
    boxes_marker,
    stage1_marker,
    priors: Priors,
    cfg: InflationConfig,
    stage1_dir: str,
    boxes_dir: str,
    out_dir: str,
    scene_names: Sequence[str] | None,
) -> tuple[dict, int]:
    started = time.time()
    errors = cfg.validate()
    if errors:
        raise UpstreamRefusal("; ".join(errors))
    # Any marker still standing describes the PREVIOUS run of this stage; it
    # comes down before the first write (C16).
    clear_markers(out_dir)

    substrate = Substrate.load(paths)
    root = os.path.join(boxes_dir, "scenes")
    if not os.path.isdir(root):
        raise UpstreamRefusal(f"{root} not found; the box producer wrote no scenes")
    names = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
    if scene_names:
        missing = sorted(set(scene_names) - set(names))
        if missing:
            raise UpstreamRefusal(f"requested scene(s) not present in the box producer's output: {missing}")
        names = [n for n in names if n in scene_names]

    per_scene: list[dict] = []
    degraded = False
    totals: dict = {}

    for scene_name in names:
        rows = read_boxes(os.path.join(root, scene_name, "boxes.jsonl"))
        origins = sensor_origins(stage1_dir, scene_name, substrate, cfg)
        inflated_rows, scene_totals = inflate_scene(rows, origins, priors, cfg)
        write_jsonl_atomic(os.path.join(out_dir, "scenes", scene_name, "inflated.jsonl"), inflated_rows)

        summary = {
            "scene": scene_name,
            **scene_totals,
            "triggered_fraction": round(scene_totals["n_triggered"] / max(1, scene_totals["n_boxes"]), 5),
            "inflated_fraction": round(scene_totals["n_inflated"] / max(1, scene_totals["n_boxes"]), 5),
            # Triggering and being unable to act on it is the reportable state:
            # it means the boxes that most need a prior are the ones without one.
            "degraded": scene_totals["n_triggered"] > 0 and scene_totals["n_inflated"] == 0,
        }
        per_scene.append(summary)
        degraded = degraded or summary["degraded"]
        for key, value in scene_totals.items():
            if key == "mean_inflation_fraction":
                continue
            totals[key] = totals.get(key, 0) + value
        print(
            f"  {scene_name}  {scene_totals['n_boxes']:>5} boxes  "
            f"{scene_totals['n_triggered']:>5} triggered  {scene_totals['n_inflated']:>5} inflated  "
            f"mean fraction {scene_totals['mean_inflation_fraction']:.3f}  "
            f"{scene_totals['n_no_prior'] + scene_totals['n_prior_no_dims']:>4} without a prior"
            + ("  DEGRADED" if summary["degraded"] else "")
        )

    weighted = sum(
        s["mean_inflation_fraction"] * s["n_inflated"] for s in per_scene
    ) / max(1, totals.get("n_inflated", 0))
    totals["mean_inflation_fraction"] = round(weighted, 6)

    manifest = {
        "spec": STAGE_SPEC,
        "stage": STAGE,
        "seed": cfg.global_seed,
        "config": cfg.as_dict(),
        "upstream": {
            "metadata_fingerprint": upstream_manifest["upstream"]["metadata_fingerprint"],
            "fingerprint_spec": upstream_manifest["upstream"]["fingerprint_spec"],
            "boxes_spec": upstream_manifest["spec"],
            "boxes_stage": upstream_manifest.get("stage"),
            "priors": priors.as_reference(),
            # C16: a run built on accepted degradation says so in its provenance.
            "boxes_degraded": boxes_marker.degraded,
            "boxes_degraded_causes": list(boxes_marker.causes),
            "stage1_degraded": stage1_marker.degraded,
            "stage1_degraded_causes": list(stage1_marker.causes),
            "accepted_degraded_upstream": cfg.accept_degraded_upstream,
            "note": (
                "Stage 7 (tracking) is not in this chain yet. When it lands it must run BEFORE this "
                "stage: it enforces yaw consistency along tracks (§4), and yaw is what decides which "
                "face is near"
            ),
        },
        "paths": paths.as_dict(),
        "frame": EGO,
        "inflation": {
            "rule": "anchor the near (sensor-facing) face, grow the far face toward the class mean",
            "rule_source": "§5.9 / comprehensive.md §7.3.8",
            "trigger": f"{cfg.trigger} < {cfg.trigger_max_points}",
            "blend": cfg.blend,
            "axes": list(cfg.axes),
            "prior_axis_mapping": cfg.prior_axis_mapping,
            "bev_anchor": cfg.bev_anchor,
            "height_anchor": cfg.height_anchor,
            "clamp": {
                "max_growth_ratio": cfg.max_growth_ratio,
                "max_growth_m": cfg.max_growth_m,
                "clamp_sigma": cfg.clamp_sigma,
                "one_sided": cfg.one_sided,
                "w_le_l": "width is capped at the inflated length; the box is never rotated to fix it",
            },
            "guards": [
                "the anchored near face may not move",
                "the far face may not move toward the sensor (no growth back through the surface)",
                "the inflated footprint may not enclose the sensor origin",
                "a measured box that already contains the sensor is not inflated at all",
            ],
            "inflation_fraction": "1 - V_measured / V_inflated, per box; per-axis fractions alongside",
            "sensor_origin": cfg.sensor_origin,
        },
        "downstream_contract": {
            "box": "the shipped box, inflated where triggered",
            "box_measured": (
                "the Stage 6 box, untouched. §7.3.9's spatial gate must read THIS one: the gate is "
                "'BEV box exceeds 2x class prior' and this stage grows boxes toward that prior, so "
                "gating on the inflated box makes the gate weakest exactly where it is needed (P1-12)"
            ),
            "num_lidar_pts": (
                "unchanged, and still counted on the single-sweep ground-filtered PRE-inflation "
                "cloud (§1.4); inflation adds no returns"
            ),
        },
        "known_gaps": [
            "under a 90 deg yaw error from Stage 6 — systematic for near-square footprints (§5.7) — "
            "inflation grows the box into the neighbouring lane. The yaw_ambiguous flag travels into "
            "every record and the count is in totals; the geometry itself cannot detect the error",
            "the class means come from the priors file's source, which in the pilot is "
            f"{priors.source!r}, not 'S0'. A box inflated toward a nuScenes mean is a nuScenes-shaped "
            "box; the release builder's source guard is what stops it shipping (§6)",
            "no ground plane is consulted when growing downward: Stage 1 fits it per sector and this "
            "stage does not re-derive it, so an inflated box may extend below the road surface",
            "nuScenes' `l` is measured along the object's own heading and Stage 6's is the long BEV "
            "side, so for a class whose GT heading crosses its long side ('a road barrier': mean w "
            "2.42, mean l 0.58) the two conventions name different axes. prior_axis_mapping="
            f"{cfg.prior_axis_mapping!r} resolves it and the choice is recorded on every box; what "
            "it cannot resolve is that such a class' FITTED yaw is 90 deg from its GT yaw",
        ],
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "elapsed_s": round(time.time() - started, 2),
        "scenes": per_scene,
        "totals": totals,
    }
    return manifest, EXIT_DEGRADED if degraded else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--stage1-dir", default=None, help="default <work_root>/stage1_ingestion")
    parser.add_argument(
        "--boxes-dir",
        default=None,
        help="whatever produced the boxes; default <work_root>/stage6_cluster (Stage 7 when it lands)",
    )
    parser.add_argument("--priors", default=None, help=f"default <out_root>/priors/{PRIORS_NAME}.json")
    parser.add_argument("--out-dir", default=None, help="default <work_root>/stage8_inflate")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--blend", default="point_count_weighted", choices=BLENDS)
    parser.add_argument("--trigger-max-points", type=int, default=None)
    parser.add_argument("--sensor-origin", default="lidar_calibrated_sensor", choices=SENSOR_ORIGINS)
    parser.add_argument("--prior-axis-mapping", default="sorted_short_long", choices=PRIOR_AXIS_MAPPINGS)
    parser.add_argument(
        "--no-inflate-yaw-ambiguous",
        action="store_true",
        help="leave near-square boxes uninflated instead of growing them along a fitted axis (§5.9)",
    )
    parser.add_argument("--seed", type=int, default=None, help="override the global seed (recorded)")
    parser.add_argument(
        "--accept-degraded-upstream",
        action="store_true",
        help="consume a DEGRADED (complete, quality-flagged) Stage 1 or box-producer output; recorded (C16)",
    )
    args = parser.parse_args(argv)

    try:
        paths = load_paths(args.paths)
    except PathValidationError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    stage1_dir = args.stage1_dir or os.path.join(paths.work_root, "stage1_ingestion")
    boxes_dir = args.boxes_dir or os.path.join(paths.work_root, "stage6_cluster")
    priors_path = args.priors or os.path.join(paths.out_root, "priors", f"{PRIORS_NAME}.json")
    out_dir = args.out_dir or os.path.join(paths.work_root, STAGE)
    assert_dataroot_read_only(paths, out_dir)

    cfg = InflationConfig(
        blend=args.blend,
        sensor_origin=args.sensor_origin,
        prior_axis_mapping=args.prior_axis_mapping,
        inflate_when_yaw_ambiguous=not args.no_inflate_yaw_ambiguous,
        accept_degraded_upstream=args.accept_degraded_upstream,
        **({"trigger_max_points": args.trigger_max_points} if args.trigger_max_points is not None else {}),
        **({"global_seed": args.seed} if args.seed is not None else {}),
    )

    try:
        upstream_manifest, boxes_marker, stage1_marker, priors = load_upstream(
            paths, stage1_dir, boxes_dir, priors_path, accept_degraded=cfg.accept_degraded_upstream
        )
        manifest, code = run(
            paths, upstream_manifest, boxes_marker, stage1_marker, priors,
            cfg, stage1_dir, boxes_dir, out_dir, args.scenes,
        )
    except UpstreamRefusal as exc:
        print(f"REFUSING TO START: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (InflationContractError, PathValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)
    # Three-state marker (§1.9, C16): clean / degraded-with-causes / absent.
    write_marker(
        out_dir,
        manifest["upstream"]["metadata_fingerprint"],
        degraded=code == EXIT_DEGRADED,
        causes=[
            f"{s['scene']}: {s['n_triggered']} triggered, 0 inflated"
            for s in manifest["scenes"]
            if s["degraded"]
        ],
    )

    t = manifest["totals"]
    print(f"rows                 : {t.get('n_rows', 0)}  ({t.get('n_boxes', 0)} carry a box)")
    print(
        f"triggered            : {t.get('n_triggered', 0)}  "
        f"({cfg.trigger} < {cfg.trigger_max_points})"
    )
    print(
        f"inflated             : {t.get('n_inflated', 0)}  "
        f"mean fraction {t.get('mean_inflation_fraction', 0.0):.3f}  "
        f"({t.get('n_yaw_ambiguous_inflated', 0)} of them yaw-ambiguous)"
    )
    print(
        f"not inflated         : {t.get('n_no_prior', 0)} no prior, "
        f"{t.get('n_prior_no_dims', 0)} prior without dims, {t.get('n_no_growth', 0)} already >= mean"
    )
    print(
        f"sensor guards        : {t.get('n_enclose_sensor_refused', 0)} would have enclosed the "
        f"sensor, {t.get('n_sensor_inside_refused', 0)} already contained it"
    )
    print(
        f"anchor ambiguous     : {t.get('n_anchor_ambiguous_axes', 0)} axes grew symmetrically "
        f"(sensor within {cfg.near_face_margin_m} m of the mid-plane)"
    )
    print(f"clamped axes         : {t.get('n_clamped_axes', 0)}")
    print(f"wrote {out_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
