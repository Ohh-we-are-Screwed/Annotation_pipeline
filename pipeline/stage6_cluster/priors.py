#!/usr/bin/env python3
"""`priors_pilot_v0.json` — the A.4 prior file Stages 6 and 8 consume (§6, Phase 8).

The pilot has no annotators, so it cannot produce `comprehensive.md` §7.2's S0.
What it can do is derive the SAME SHAPE of file from nuScenes' own human labels
and prove the **consumption** side of the contract: Stage 6 takes `eps_bev` from
here, Stage 8 takes `dims.{w,l,h}.mu` from here, and both read them through the
interface the real S0 output will use.

**Two guards that keep that from becoming a lie** (§6, P1-11):

  1. This file sets `source: "nuscenes_gt_pilot"`, never `"S0"`. §7.2 is explicit
     that nuScenes/KITTI values are "initialization only, deleted after S0", and
     this writes a file in the same schema, in the same directory shape, through
     the same interface. The only thing separating it from a real prior is the
     string, so the string is load-bearing.
  2. `assert_release_source()` is five lines that stop a nuScenes-shaped prior
     reaching a Dhaka release. The release builder calls it; nothing else does.

**Derived under the pipeline's own constraints, not the annotation set's** (§6).
nuScenes GT includes boxes beyond the 40 m cap and boxes with **zero** LiDAR
returns. The pipeline never sees either: it clusters points, so an object with no
points has no cluster, and E stops at 40 m. Priors drawn from the full population
therefore pull inflation toward a mean the sensor never measures — a bias with no
symptom, because every inflated box still looks like a box. So the population is
filtered to: centre inside E (`eval_region.in_region`, the one implementation),
`num_lidar_pts >= min_lidar_pts`, and the `priors` scene subset ONLY (§11
decision 3 — deriving priors on the scenes the pipeline is scored on is the
"tuned on the eval set" defect this partition exists to prevent).

**Keyed by prompt phrase, not by nuScenes category** (X-6). `class_name`
everywhere downstream is the taxonomy's prompt phrase ("a car"), because that is
what Stage 3's caption spans produce. A prior file keyed on dotted categories, or
on `Annotation_pipeline.md`'s invented names (`cyclist`, `traffic_cone`), misses
every lookup silently and hands each unmatched class the same default epsilon —
clean output, one epsilon, wrong clusters. Every phrase in the taxonomy therefore
gets an entry here, INCLUDING the ones with no GT instances in the subset: their
block is present with `dims: null`, `eps_bev: null`, `n_instances: 0`. A consumer
that finds a null knows the gap exists; a consumer that finds a missing key does
not know it looked.

`eps_bev` is §7.2's formula — `0.6 x mean footprint diagonal` — with the mean
taken over per-instance diagonals `sqrt(w^2 + l^2)`, not over the mean w and l.
The two differ by Jensen and the difference is recorded rather than assumed away.

`min_pts_L1`, `min_pts_L2` and `conf_thresh` are **null on purpose**. §6.3 wants
LEVEL_1/2 thresholds "set for *this* sensor's density ... not copied from a
64-beam dataset"; deriving them from nuScenes' 32-beam returns is exactly that
copy. The measured distribution is recorded in a `measured` block as a
diagnostic, where it cannot be mistaken for a threshold.

    python3 -m pipeline.stage6_cluster.priors [--paths configs/paths.yaml]

Exit codes:
    0  priors written for every taxonomy phrase
    1  written, but at least one phrase has no usable instances
    2  upstream contract broken; nothing was written
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.common.conventions import EGO, NUSCENES_GLOBAL, Transform, apply_transform  # noqa: E402
from pipeline.common.eval_region import R1_DEFAULT, R2_DEFAULT, RegionSpec  # noqa: E402
from pipeline.common.eval_region import in_region  # noqa: E402
from pipeline.common.paths import (  # noqa: E402
    FINGERPRINT_SPEC,
    PathValidationError,
    Paths,
    assert_dataroot_read_only,
    load_paths,
    metadata_fingerprint,
)
from pipeline.stage0_data_probe.probe import PARTITION, Substrate, write_json_atomic  # noqa: E402
from pipeline.stage3_proposals.proposals import UpstreamRefusal, load_taxonomy  # noqa: E402

PRIORS_SPEC = "dhakascenes-pilot/priors/v1"  # the A.4 shape
PRIORS_NAME = "priors_pilot_v0"
PILOT_SOURCE = "nuscenes_gt_pilot"
RELEASE_SOURCE = "S0"

EXIT_OK = 0
EXIT_INCOMPLETE = 1  # written, but at least one phrase has no usable instances
EXIT_REFUSED = 2

# The three A.4 dimension axes, in nuScenes [w, l, h] order (§3.2). The order is
# named once here and reused, because a [h, w, l] slip anywhere on this path
# rotates every inflated box while every value stays plausible.
DIM_AXES: tuple[str, ...] = ("w", "l", "h")

# The transposed-file guard, and why it is not simply "w <= l".
#
# nuScenes `size` is [w, l, h] where `l` is the extent along the object's OWN
# heading. Measured on this substrate, two classes legitimately have w > l:
#
#   'a child'       mean w 0.519, mean l 0.514  — round in plan view; which of
#                   two nearly equal numbers is larger is noise.
#   'a road barrier' mean w 2.421, mean l 0.583 — a fence panel's heading points
#                   through its thin direction, so its LONG side is its width.
#
# So a blanket w <= l rejects real classes. What a transposed read of `size`
# actually looks like is a VEHICLE with its length in the width slot — a car at
# w 4.6, l 1.9. Vehicles are long along their heading under every convention, so
# they are the discriminating population and the guard runs on them.
VEHICLE_CATEGORY_PREFIX = "vehicle."
W_LE_L_TOLERANCE = 0.05


class PriorsError(RuntimeError):
    """A priors file is malformed, or a prior was asked for something it cannot answer."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorsConfig:
    """Derivation tunables. Every one of these changes the numbers, so every one is recorded."""

    scene_subset: str = "priors"
    coverage_config: str = "R2"

    # --- the pipeline's own constraints, applied to the GT population (§6) ---
    min_lidar_pts: int = 5
    restrict_to_eval_region: bool = True

    # --- aggregation ---
    min_instances_for_dims: int = 5
    eps_scale: float = 0.6

    # --- determinism (§1.9) ---
    global_seed: int = 20260812

    provenance: dict = field(
        default_factory=lambda: {
            "scene_subset": (
                "§11 decision 3: the `priors` subset ONLY. Deriving priors on the `run` scenes is "
                "the tuned-on-the-eval-set defect the partition exists to prevent"
            ),
            "coverage_config": "§1.10 decision 1: R2, and E is derived from it, never hardcoded",
            "min_lidar_pts": (
                "§6.3's '>= 5 LiDAR points in the single-sweep frame'. nuScenes GT contains boxes "
                "with ZERO returns; the pipeline clusters points and can never produce one, so "
                "including them pulls the mean toward objects the sensor never measured"
            ),
            "restrict_to_eval_region": (
                "§6: priors are derived under the same E and 40 m cap the pipeline operates under. "
                "in_region() is the only membership test (§1.10)"
            ),
            "min_instances_for_dims": (
                "arbitrary, needs tuning. Below it the block carries dims: null rather than a mean "
                "of two boxes — a prior with n=2 inflates as confidently as one with n=2000"
            ),
            "eps_scale": "comprehensive.md §7.2: eps ~ 0.6 x mean footprint diagonal",
            "global_seed": "§1.9, one global seed, recorded (nothing here samples; recorded anyway)",
        }
    )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.scene_subset not in PARTITION:
            errors.append(f"scene_subset={self.scene_subset!r} is not one of {sorted(PARTITION)}")
        if self.coverage_config not in ("R1", "R2"):
            errors.append(f"coverage_config={self.coverage_config!r} is not R1 or R2")
        if self.min_lidar_pts < 0:
            errors.append(f"min_lidar_pts={self.min_lidar_pts} must be >= 0")
        if self.min_instances_for_dims < 2:
            errors.append(
                f"min_instances_for_dims={self.min_instances_for_dims} must be >= 2; a sigma over "
                "one sample is 0.0, which reads downstream as a perfectly known prior"
            )
        if not 0.0 < self.eps_scale <= 2.0:
            errors.append(f"eps_scale={self.eps_scale} is outside the plausible (0, 2] band")
        return errors

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def region_for(coverage_config: str) -> RegionSpec:
    if coverage_config == "R1":
        return R1_DEFAULT
    if coverage_config == "R2":
        return R2_DEFAULT
    raise PriorsError(f"coverage_config={coverage_config!r} is not R1 or R2")


# ---------------------------------------------------------------------------
# The loaded file — the interface Stages 6 and 8 hold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClassPrior:
    """One A.4 class block, after validation.

    `dims` is `None` when the subset held too few instances to justify a mean.
    That is a value, not an absence: a consumer that receives `None` knows the
    prior does not exist and must say so in its own record, which is the whole
    difference between "no prior for this class" and "inflated to a mean derived
    from two boxes".
    """

    class_name: str  # the taxonomy prompt phrase — the key everything downstream uses
    category: str  # the nuScenes category it came from, for traceability only
    n_instances: int
    dims: dict | None  # {"w": {"mu": float, "sigma": float}, "l": {...}, "h": {...}}
    eps_bev: float | None
    min_pts_L1: int | None
    min_pts_L2: int | None
    conf_thresh: float | None
    source: str
    gaps: tuple[str, ...] = ()

    def mu(self, axis: str) -> float | None:
        if self.dims is None:
            return None
        return float(self.dims[axis]["mu"])

    def sigma(self, axis: str) -> float | None:
        if self.dims is None:
            return None
        return float(self.dims[axis]["sigma"])


@dataclass(frozen=True)
class Priors:
    """`priors_pilot_v0.json`, loaded, validated, and bound to its substrate."""

    path: str
    sha256: str
    spec: str
    name: str
    source: str
    metadata_fingerprint: str
    classes: dict  # class_name -> ClassPrior
    eps_scale: float
    derived_from: dict
    payload: dict = field(repr=False, default_factory=dict)

    def get(self, class_name: str) -> ClassPrior | None:
        """The prior for a phrase, or None when the phrase is not in the file at all.

        A missing key and a present-but-empty block are different failures and
        callers must be able to tell them apart: the first means the priors file
        and the taxonomy disagree about what the classes ARE, the second means
        the subset held no instances of a class that exists.
        """
        return self.classes.get(class_name)

    def eps_bev(self, class_name: str, *, fallback_m: float) -> tuple[float, str]:
        """(epsilon, provenance). Never silently defaults — the provenance says which (X-6)."""
        prior = self.classes.get(class_name)
        if prior is None:
            return float(fallback_m), f"config_fallback:class_absent_from_priors:{class_name}"
        if prior.eps_bev is None:
            return float(fallback_m), f"config_fallback:{','.join(prior.gaps) or 'no_eps'}"
        return float(prior.eps_bev), f"priors:{self.name}@{self.sha256[:16]}"

    def as_reference(self) -> dict:
        """The block every consuming stage's manifest records."""
        return {
            "path": self.path,
            "sha256": self.sha256,
            "spec": self.spec,
            "name": self.name,
            "source": self.source,
            "metadata_fingerprint": self.metadata_fingerprint,
            "n_classes": len(self.classes),
            "n_classes_with_dims": sum(1 for p in self.classes.values() if p.dims is not None),
            "eps_scale": self.eps_scale,
        }


def _check_dim_block(block: object, where: str, category: str) -> dict | None:
    if block is None:
        return None
    if not isinstance(block, dict):
        raise PriorsError(f"{where}: dims must be a mapping or null, got {type(block).__name__}")
    missing = [axis for axis in DIM_AXES if axis not in block]
    if missing:
        raise PriorsError(f"{where}: dims is missing axis/axes {missing}; A.4 order is [w, l, h] (§3.2)")
    out: dict = {}
    for axis in DIM_AXES:
        entry = block[axis]
        if not isinstance(entry, dict) or "mu" not in entry or "sigma" not in entry:
            raise PriorsError(f"{where}: dims[{axis!r}] must be {{mu, sigma}}")
        mu = float(entry["mu"])
        sigma = float(entry["sigma"])
        if not math.isfinite(mu) or mu <= 0.0:
            raise PriorsError(f"{where}: dims[{axis!r}].mu must be positive and finite, got {mu!r}")
        if not math.isfinite(sigma) or sigma < 0.0:
            raise PriorsError(f"{where}: dims[{axis!r}].sigma must be finite and >= 0, got {sigma!r}")
        out[axis] = {"mu": mu, "sigma": sigma}
    if category.startswith(VEHICLE_CATEGORY_PREFIX) and out["w"]["mu"] > out["l"]["mu"] * (
        1.0 + W_LE_L_TOLERANCE
    ):
        # Not a stylistic preference: a vehicle whose mean width exceeds its mean
        # length is the signature of a [l, w, h] read of `size`, and inflating a
        # car's width toward its length grows the box a metre sideways into the
        # next lane (§5.9). Non-vehicle classes are exempt for the measured
        # reason recorded above the tolerance constant.
        raise PriorsError(
            f"{where}: dims.w.mu={out['w']['mu']} exceeds dims.l.mu={out['l']['mu']} by more than "
            f"{W_LE_L_TOLERANCE:.0%} for vehicle category {category!r}; A.4 size order is [w, l, h] "
            "(§3.2) and a swapped file inflates every box sideways"
        )
    return out


def load_priors(path: str) -> Priors:
    """Read and validate. The only reader of a priors file in the pipeline."""
    if not os.path.isfile(path):
        raise UpstreamRefusal(
            f"{path} not found; run `python3 -m pipeline.stage6_cluster.priors` first. "
            "Stage 6 takes epsilon and Stage 8 takes class means from this file; there is no "
            "default for either (X-6)"
        )
    with open(path, "rb") as fh:
        raw = fh.read()
    payload = json.loads(raw.decode("utf-8"))
    if payload.get("spec") != PRIORS_SPEC:
        raise PriorsError(f"{path}: spec={payload.get('spec')!r}, expected {PRIORS_SPEC!r}")
    source = payload.get("source")
    if not isinstance(source, str) or not source:
        raise PriorsError(f"{path}: source is required (A.4)")
    blocks = payload.get("classes")
    if not isinstance(blocks, dict) or not blocks:
        raise PriorsError(f"{path}: classes must be a non-empty mapping of class_name -> block")

    classes: dict[str, ClassPrior] = {}
    for class_name, block in sorted(blocks.items()):
        where = f"{path}: classes[{class_name!r}]"
        if not isinstance(block, dict):
            raise PriorsError(f"{where} must be a mapping")
        eps = block.get("eps_bev")
        if eps is not None:
            eps = float(eps)
            if not math.isfinite(eps) or eps <= 0.0:
                raise PriorsError(f"{where}: eps_bev must be positive and finite, got {eps!r}")
        n_instances = int(block.get("n_instances", 0))
        if n_instances < 0:
            raise PriorsError(f"{where}: n_instances must be >= 0")
        category = str(block.get("category", ""))
        classes[class_name] = ClassPrior(
            class_name=class_name,
            category=category,
            n_instances=n_instances,
            dims=_check_dim_block(block.get("dims"), where, category),
            eps_bev=eps,
            min_pts_L1=None if block.get("min_pts_L1") is None else int(block["min_pts_L1"]),
            min_pts_L2=None if block.get("min_pts_L2") is None else int(block["min_pts_L2"]),
            conf_thresh=None if block.get("conf_thresh") is None else float(block["conf_thresh"]),
            source=str(block.get("source", source)),
            gaps=tuple(block.get("gaps", ())),
        )

    return Priors(
        path=os.path.realpath(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        spec=str(payload["spec"]),
        name=str(payload.get("name", os.path.basename(path))),
        source=source,
        metadata_fingerprint=str(payload.get("derived_from", {}).get("metadata_fingerprint", "")),
        classes=classes,
        eps_scale=float(payload.get("eps_scale", 0.6)),
        derived_from=dict(payload.get("derived_from", {})),
        payload=payload,
    )


def assert_release_source(priors: Priors) -> None:
    """The release-builder guard (§6, P1-11 guard 2). Five lines, and the reason for them.

    `priors_pilot_v0.json` is nuScenes-shaped: Boston and Singapore sedans,
    32-beam returns, no rickshaw, no CNG, no human-hauler. It is written in the
    A.4 schema, in the A.4 directory, and consumed through the A.4 interface —
    which is exactly what makes it capable of reaching a Dhaka release unnoticed.
    """
    if priors.source != RELEASE_SOURCE:
        raise PriorsError(
            f"{priors.path}: source={priors.source!r}, release requires {RELEASE_SOURCE!r}. "
            "comprehensive.md §7.2: nuScenes/KITTI values are initialization only and are deleted "
            "after S0; a nuScenes-derived prior must never ship in a DhakaScenes release"
        )


# ---------------------------------------------------------------------------
# Derivation from nuScenes GT
# ---------------------------------------------------------------------------


@dataclass
class Sample:
    """One GT box, already reduced to what a prior needs."""

    category: str
    w_m: float
    l_m: float
    h_m: float
    num_lidar_pts: int
    range_m: float


def lidar_anchors(substrate: Substrate, scene: dict) -> dict:
    """sample_token -> the scene's LIDAR_TOP keyframe `sample_data` records.

    `sample_annotation.translation` is in the nuScenes global frame; E is defined
    in the ego frame (§1.1, §3.6). The conversion needs the ego pose at the LiDAR
    anchor time, and the LiDAR's own `sample_data` is the only record that
    carries it (§1.2: each `sample_data` has its OWN ego pose).

    Built by scanning the scene's `sample_data`, not by reading `sample["data"]`:
    that field is assembled by the nuScenes devkit at load time and does not
    exist in the on-disk table, so a lookup through it finds nothing and reads as
    a missing channel.
    """
    return {
        record["sample_token"]: record
        for record in substrate.sample_data_by_scene.get(scene["token"], ())
        if record["is_key_frame"] and substrate.channel(record) == "LIDAR_TOP"
    }


def collect_samples(
    substrate: Substrate,
    scene_names: list[str],
    cfg: PriorsConfig,
) -> tuple[list[Sample], dict]:
    """Every GT box in the subset, filtered exactly as the pipeline is (§6)."""
    region = region_for(cfg.coverage_config)
    by_name = {s["name"]: s for s in substrate.tables["scene.json"]}
    category_name = {c["token"]: c["name"] for c in substrate.tables["category.json"]}
    ego_pose_table = substrate.by_token("ego_pose.json")

    collected: list[Sample] = []
    ledger = {
        "n_annotations": 0,
        "n_dropped_no_lidar_pts": 0,
        "n_dropped_outside_region": 0,
        "n_dropped_unknown_category": 0,
        "n_kept": 0,
        "n_keyframes": 0,
        "n_scenes": 0,
    }

    for name in scene_names:
        scene = by_name.get(name)
        if scene is None:
            raise UpstreamRefusal(f"scene {name!r} is in the {cfg.scene_subset!r} partition but not in scene.json")
        ledger["n_scenes"] += 1
        anchors = lidar_anchors(substrate, scene)
        for sample in substrate.scene_samples(scene):
            ledger["n_keyframes"] += 1
            lidar_sd = anchors.get(sample["token"])
            if lidar_sd is None:
                raise PriorsError(
                    f"sample {sample['token']}: no LIDAR_TOP keyframe sample_data; Stage 0's "
                    "channels_complete predicate should have excluded this scene"
                )
            ego_pose = Transform.from_nuscenes(
                ego_pose_table[lidar_sd["ego_pose_token"]],
                source_frame=EGO,
                parent_frame=NUSCENES_GLOBAL,
            )
            inverse = ego_pose.inverse_matrix()
            for ann in substrate.annotations_by_sample.get(sample["token"], ()):
                ledger["n_annotations"] += 1
                instance = substrate.instance_by_token.get(ann["instance_token"])
                category = category_name.get(instance["category_token"], "") if instance else ""
                if not category:
                    ledger["n_dropped_unknown_category"] += 1
                    continue
                num_lidar_pts = int(ann.get("num_lidar_pts", 0))
                if num_lidar_pts < cfg.min_lidar_pts:
                    ledger["n_dropped_no_lidar_pts"] += 1
                    continue
                centre_ego = apply_transform(
                    inverse, np.asarray([ann["translation"]], dtype=np.float64)
                )[0]
                if cfg.restrict_to_eval_region and not in_region(
                    centre_ego[0], centre_ego[1], region, frame=EGO
                ):
                    ledger["n_dropped_outside_region"] += 1
                    continue
                # nuScenes `size` is [w, l, h] — the same order the pilot stores
                # (§3.2). Nothing is reordered here, and nothing may be.
                w_m, l_m, h_m = (float(v) for v in ann["size"])
                collected.append(
                    Sample(
                        category=category,
                        w_m=w_m,
                        l_m=l_m,
                        h_m=h_m,
                        num_lidar_pts=num_lidar_pts,
                        range_m=float(math.hypot(centre_ego[0], centre_ego[1])),
                    )
                )
                ledger["n_kept"] += 1
    return collected, ledger


def _stats(values: np.ndarray) -> dict:
    # ddof=1: these are a sample of the class, not the population. With n < 2 the
    # sample sigma is undefined; min_instances_for_dims keeps that unreachable,
    # and the guard here says so rather than emitting 0.0 as if it were measured.
    if values.shape[0] < 2:
        raise PriorsError("a sigma over fewer than 2 instances is undefined, not 0.0")
    return {"mu": float(values.mean()), "sigma": float(values.std(ddof=1))}


def build_class_blocks(
    samples: list[Sample],
    category_to_phrase: dict,
    cfg: PriorsConfig,
) -> tuple[dict, list[str]]:
    """One A.4 block per taxonomy phrase — including the phrases with no instances."""
    by_category: dict[str, list[Sample]] = {}
    for sample in samples:
        by_category.setdefault(sample.category, []).append(sample)

    unmapped = sorted(set(by_category) - set(category_to_phrase))
    if unmapped:
        # The taxonomy is the class space. A category with GT boxes and no phrase
        # would contribute to no prior at all and never be missed.
        raise UpstreamRefusal(
            f"nuScenes categories with GT in the subset but no prompt phrase: {unmapped}. "
            "The taxonomy file is the class space (§0.3); a category outside it has no consumer"
        )

    blocks: dict[str, dict] = {}
    empty: list[str] = []
    for category, phrase in sorted(category_to_phrase.items(), key=lambda kv: kv[1]):
        rows = by_category.get(category, [])
        n = len(rows)
        gaps: list[str] = []
        dims = None
        eps_bev = None
        measured: dict = {"n_instances": n}

        if n:
            w = np.asarray([r.w_m for r in rows], dtype=np.float64)
            l = np.asarray([r.l_m for r in rows], dtype=np.float64)  # noqa: E741
            h = np.asarray([r.h_m for r in rows], dtype=np.float64)
            pts = np.asarray([r.num_lidar_pts for r in rows], dtype=np.float64)
            rng = np.asarray([r.range_m for r in rows], dtype=np.float64)
            diagonal = np.hypot(w, l)
            measured.update(
                {
                    # Diagnostics, deliberately NOT thresholds. §6.3 wants LEVEL_1/2
                    # point counts set for the Mid-360's density; these are a
                    # 32-beam spinning sensor's, and putting them in min_pts_L1
                    # would be precisely the copy §6.3 forbids.
                    "num_lidar_pts_p05": float(np.percentile(pts, 5)),
                    "num_lidar_pts_p50": float(np.percentile(pts, 50)),
                    "num_lidar_pts_p95": float(np.percentile(pts, 95)),
                    "range_m_p50": float(np.percentile(rng, 50)),
                    "range_m_max": float(rng.max()),
                    "footprint_diagonal_mean_m": float(diagonal.mean()),
                    # Recorded because eps is 0.6 x mean(diagonal), NOT
                    # 0.6 x diagonal(mean w, mean l). Jensen separates them and
                    # the gap belongs in the file, not in a reader's assumption.
                    "footprint_diagonal_of_means_m": float(math.hypot(w.mean(), l.mean())),
                }
            )
            if n >= cfg.min_instances_for_dims:
                dims = {"w": _stats(w), "l": _stats(l), "h": _stats(h)}
                eps_bev = float(cfg.eps_scale * diagonal.mean())
                ratio = dims["w"]["mu"] / dims["l"]["mu"]
                if category.startswith(VEHICLE_CATEGORY_PREFIX) and ratio > 1.0 + W_LE_L_TOLERANCE:
                    raise PriorsError(
                        f"{phrase!r} ({category}): mean w ({dims['w']['mu']:.3f}) exceeds mean l "
                        f"({dims['l']['mu']:.3f}) over {n} GT boxes by more than "
                        f"{W_LE_L_TOLERANCE:.0%}. nuScenes size is [w, l, h] and a vehicle is long "
                        "along its heading; this reads as a transposed read of the size field (§3.2)"
                    )
                if ratio > 1.0:
                    # Real, and load-bearing for Stage 8. nuScenes' `l` is the
                    # extent along the object's OWN heading, and a barrier's
                    # heading crosses its panel — so this class' long side lives
                    # in `w`. Stage 6 defines the heading as the LONG side (§5.7,
                    # w <= l by construction), so the two conventions name
                    # different axes here and a consumer that blends w toward w
                    # would grow the box across the fitted axis.
                    gaps.append(f"w_gt_l:{ratio:.3f}")
            else:
                gaps.append(f"too_few_instances:{n}<{cfg.min_instances_for_dims}")
        else:
            gaps.append("no_instances_in_subset")
            empty.append(phrase)

        blocks[phrase] = {
            "category": category,
            "dims": dims,
            "w_gt_l": bool(dims is not None and dims["w"]["mu"] > dims["l"]["mu"]),
            "eps_bev": eps_bev,
            # A.4 fields with no honest pilot producer. Null, never a copied value.
            "min_pts_L1": None,
            "min_pts_L2": None,
            "conf_thresh": None,
            "source": PILOT_SOURCE,
            "n_instances": n,
            "gaps": gaps,
            "measured": measured,
        }
    return blocks, empty


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def read_allowlist(stage0_dir: str) -> dict:
    path = os.path.join(stage0_dir, "usable_scenes.json")
    if not os.path.isfile(path):
        raise UpstreamRefusal(f"{path} not found; run `python3 -m pipeline.stage0_data_probe.probe` first")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def subset_scene_names(allowlist: dict, cfg: PriorsConfig) -> list[str]:
    """The `priors` subset, intersected with the allowlist — and never quietly reduced.

    A partition scene that failed a Stage 0 predicate must stop the derivation:
    silently deriving from three scenes instead of four changes every mean in the
    file, and the file records `scenes` either way.
    """
    usable = {s["name"] for s in allowlist.get("scenes", ())}
    wanted = list(PARTITION[cfg.scene_subset])
    missing = [name for name in wanted if name not in usable]
    if missing:
        raise UpstreamRefusal(
            f"{cfg.scene_subset!r} partition scene(s) {missing} are not in the Stage 0 allowlist. "
            "The partition is fixed (§11 decision 3) and is not re-drawn to work around an "
            "exclusion; fix the substrate or record a new partition"
        )
    return wanted


def run(paths: Paths, cfg: PriorsConfig, stage0_dir: str, taxonomy_path: str) -> tuple[dict, int]:
    started = time.time()
    errors = cfg.validate()
    if errors:
        raise UpstreamRefusal("; ".join(errors))

    allowlist = read_allowlist(stage0_dir)
    fingerprint = metadata_fingerprint(paths)
    recorded = allowlist.get("manifest", {}).get("metadata_fingerprint")
    if recorded != fingerprint:
        raise UpstreamRefusal(
            f"metadata fingerprint mismatch: the allowlist was computed against {recorded}, this "
            f"dataroot ({paths.dataroot}) is {fingerprint}"
        )

    taxonomy = load_taxonomy(taxonomy_path)
    substrate = Substrate.load(paths)
    scene_names = subset_scene_names(allowlist, cfg)
    samples, ledger = collect_samples(substrate, scene_names, cfg)
    if not samples:
        raise UpstreamRefusal(
            f"no GT boxes survived the pipeline's own filters over the {cfg.scene_subset!r} subset "
            f"(>= {cfg.min_lidar_pts} returns, inside E). A priors file of all-nulls would inflate "
            "nothing and say nothing"
        )

    blocks, empty = build_class_blocks(samples, dict(taxonomy.category_to_phrase), cfg)
    region = region_for(cfg.coverage_config)

    payload = {
        "spec": PRIORS_SPEC,
        "name": PRIORS_NAME,
        "version": 0,
        # Guard 1 (§6, P1-11). The only thing separating this file from a real
        # S0 prior is this string, which is why it is not a default anywhere.
        "source": PILOT_SOURCE,
        "source_note": (
            "Derived from nuScenes v1.0-mini human GT, NOT from comprehensive.md §7.2's S0. §7.2 is "
            "explicit that nuScenes/KITTI values are initialization only and are deleted after S0. "
            "This file exists to prove the CONSUMPTION side of the A.4 contract; the production "
            "side (real annotation effort on indigenous classes) is untouched"
        ),
        "release_guard": (
            "pipeline.stage6_cluster.priors.assert_release_source() rejects any priors file whose "
            "source is not 'S0'. The release builder calls it"
        ),
        "license_note": (
            "[VERIFY] the nuScenes licence (CC BY-NC-SA class) bears on redistributing GT-derived "
            "statistics. Check before this file lands in any public repo (§6)"
        ),
        "eps_formula": "eps_bev = eps_scale * mean_i sqrt(w_i^2 + l_i^2)  (comprehensive.md §7.2)",
        "eps_scale": cfg.eps_scale,
        "derived_from": {
            "dataroot_realpath": paths.dataroot,
            "version": paths.version,
            "metadata_fingerprint": fingerprint,
            "fingerprint_spec": FINGERPRINT_SPEC,
            "scene_subset": cfg.scene_subset,
            "scenes": scene_names,
            "taxonomy": taxonomy.as_dict(),
            "eval_region": region.as_dict(),
            "filters": {
                "min_lidar_pts": cfg.min_lidar_pts,
                "restrict_to_eval_region": cfg.restrict_to_eval_region,
                "min_instances_for_dims": cfg.min_instances_for_dims,
                "note": (
                    "the GT population is filtered to what the pipeline can actually produce: a "
                    "cluster needs returns, and E stops at 40 m (§6)"
                ),
            },
            "config": cfg.as_dict(),
        },
        "unproduced_fields": {
            "min_pts_L1": (
                "§6.3 requires LEVEL_1/2 point thresholds set for THIS sensor's density; nuScenes "
                "is a 32-beam spinning LiDAR and copying its counts is the error §6.3 names. The "
                "measured distribution is in classes[*].measured as a diagnostic"
            ),
            "min_pts_L2": "as min_pts_L1",
            "conf_thresh": (
                "per-class proposal thresholds are tuned on the `tuning` subset in Stage 3 and live "
                "in configs/taxonomy_pilot_nuscenes.yaml; they are not derivable from GT geometry"
            ),
        },
        "classes": blocks,
        "classes_without_instances": empty,
        "totals": {
            **ledger,
            "n_classes": len(blocks),
            "n_classes_with_dims": sum(1 for b in blocks.values() if b["dims"] is not None),
            "n_classes_without_instances": len(empty),
        },
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "elapsed_s": round(time.time() - started, 2),
    }
    return payload, EXIT_INCOMPLETE if empty else EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default="configs/paths.yaml")
    parser.add_argument("--stage0-dir", default=None, help="default <work_root>/stage0_data_probe")
    parser.add_argument("--taxonomy", default="configs/taxonomy_pilot_nuscenes.yaml")
    parser.add_argument("--out", default=None, help=f"default <out_root>/priors/{PRIORS_NAME}.json")
    parser.add_argument("--min-lidar-pts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        paths = load_paths(args.paths)
    except PathValidationError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    stage0_dir = args.stage0_dir or os.path.join(paths.work_root, "stage0_data_probe")
    out_path = args.out or os.path.join(paths.out_root, "priors", f"{PRIORS_NAME}.json")
    assert_dataroot_read_only(paths, out_path)

    cfg = PriorsConfig(
        **({"min_lidar_pts": args.min_lidar_pts} if args.min_lidar_pts is not None else {}),
        **({"global_seed": args.seed} if args.seed is not None else {}),
    )

    try:
        payload, code = run(paths, cfg, stage0_dir, args.taxonomy)
    except UpstreamRefusal as exc:
        print(f"REFUSING TO START: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (PriorsError, PathValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    write_json_atomic(out_path, payload)
    # Read-back through the real loader: the file every stage consumes is proven
    # loadable by the code that will consume it, at the moment it is written.
    priors = load_priors(out_path)

    totals = payload["totals"]
    print(f"subset               : {payload['derived_from']['scene_subset']} {payload['derived_from']['scenes']}")
    print(f"GT annotations       : {totals['n_annotations']} over {totals['n_keyframes']} keyframes")
    print(
        f"kept                 : {totals['n_kept']}  "
        f"(dropped {totals['n_dropped_no_lidar_pts']} with < {cfg.min_lidar_pts} returns, "
        f"{totals['n_dropped_outside_region']} outside E)"
    )
    print(f"classes with dims    : {totals['n_classes_with_dims']} / {totals['n_classes']}")
    if payload["classes_without_instances"]:
        print(f"no instances         : {payload['classes_without_instances']}")
    for name in sorted(priors.classes):
        prior = priors.classes[name]
        if prior.dims is None:
            print(f"  {name:<32} n={prior.n_instances:<5} {'; '.join(prior.gaps)}")
        else:
            print(
                f"  {name:<32} n={prior.n_instances:<5} "
                f"w={prior.mu('w'):.2f} l={prior.mu('l'):.2f} h={prior.mu('h'):.2f}  "
                f"eps={prior.eps_bev:.3f}"
            )
    print(f"source               : {priors.source}  (release requires {RELEASE_SOURCE!r})")
    print(f"wrote {out_path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
