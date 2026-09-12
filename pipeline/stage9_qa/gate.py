#!/usr/bin/env python3
"""Stage 9 — QA gating: gate vector -> tier -> the I-4 pre-label export (§5.10).

Consumes Stage 8's `inflated.jsonl` (every row still carries the PRE-inflation
box as `box_measured`) plus the priors file, and writes one
`prelabels.jsonl` per scene through `schemas.write_records()` — the raising
§1.7 boundary — so every record that lands on disk has passed the
AnnotationRecord/Provenance/GateVector validators, including the
`num_lidar_pts_basis` check and the tier≠source rules.

The silent failures this stage is built around (§5.10, P1-12, §1.4):

  1. **Spatial gate on POST-inflation dimensions.** Stage 8 inflates *toward*
     the class prior, so a box that has been through Stage 8 passes a
     "exceeds 2x class prior" test more easily — the gate is weakest exactly
     where it is needed. This is an INHERITED spec flaw (`comprehensive.md`
     §7.3.9 defines the gate, §7.3.8 inflates first) that the pilot surfaces
     rather than reproduces: `spatial_gate()` reads `box_measured` and never
     the shipped box, both dimension sets are already recorded in the Stage 8
     row, and the manifest states which one was gated.
  2. **A return count from the accumulation.** "≥ 5 returns" over a ~10-sweep
     accumulation is a ~10x looser gate that still LOOKS like it fired. Stage 6
     counted on the single-sweep, ground-filtered, pre-inflation cloud and
     labelled the count (`num_lidar_pts_basis`); this stage REFUSES any row
     whose basis says otherwise, and the schema validator re-checks at write.
  3. **`tier` conflated with `source`.** Every record here is
     `source: "pipeline"` regardless of tier; `tier: auto_accept` never
     upgrades provenance (§1.7 rule 3). Enforced by `Provenance.validate()`,
     not by this module remembering.

The drivable term of `spatial_ok` is OFF (§11, decision 7, locked at this
phase): the spatial gate below is the 2x-class-prior size term ONLY, and
`spatial_ok_source` says so on every record rather than leaving a reader to
guess which terms an `spatial_ok: true` includes.

Tier rule (config, stated here because `comprehensive.md` §7.4 tunes the
thresholds but never states the mapping):
    rejected     a PHYSICAL gate failed — returns below the floor, or the
                 measured footprint exceeds the 2x class prior
    flagged      physical gates pass but confidence is below the cutoff, or
                 the spatial gate could not be evaluated (no prior dims)
    auto_accept  every gate passed
Nothing ships as ground truth from this stage (§7.4); `auto_accept` is a
review-priority tier, not an accepted label.

    python3 -m pipeline.stage9_qa.gate [--scenes scene-0061]

Exit codes:
    0  every scene gated, every gate evaluable
    1  complete but quality-flagged (`_SUCCESS.degraded` carries causes):
       a scene produced zero pre-labels, or boxes were tiered `flagged`
       because their class has no usable prior
    2  upstream contract broken (missing/degraded-unaccepted Stage 8, priors
       mismatch, wrong count basis, invalid record); nothing marked complete
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.common.manifest import (  # noqa: E402
    UpstreamRefusal,
    boxes_source, num_lidar_pts_basis_detail,
    clear_markers,
    require_upstream,
    write_json_atomic,
    write_marker,
)
from pipeline.common.paths import (  # noqa: E402
    PathValidationError,
    Paths,
    assert_dataroot_read_only,
    load_paths,
    metadata_fingerprint,
)
from pipeline.common.conventions import EGO  # noqa: E402
from pipeline.common.schemas import (  # noqa: E402
    AnnotationRecord,
    GateVector,
    Provenance,
    SchemaValidationError,
    write_records,
)
from pipeline.stage6_cluster.priors import PRIORS_NAME, Priors, load_priors  # noqa: E402

STAGE = "stage9_qa"
STAGE_SPEC = "dhakascenes-pilot/stage9_qa/v1"

EXIT_OK = 0
EXIT_DEGRADED = 1
EXIT_REFUSED = 2

# The only basis under which "n returns" means what §1.4 / §5.10 mean.
REQUIRED_PTS_BASIS = "single_sweep_ground_filtered_pre_inflation"

TIER_AUTO = "auto_accept"
TIER_FLAGGED = "flagged"
TIER_REJECTED = "rejected"


class GateContractError(RuntimeError):
    """A Stage 9 input violated the contract this stage assumes."""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GateConfig:
    """Stage 9 tunables. None of these may appear as a literal in the code below."""

    conf_gate: float = 0.5
    min_lidar_returns: int = 5
    spatial_multiplier: float = 2.0
    # BEV dims compared sorted (short vs short, long vs long) — the same
    # convention Stage 8's default `prior_axis_mapping=sorted_short_long` uses,
    # so the gate and the inflation read the prior the same way and a [w,l]
    # swap in either place cannot silently pass the gate.
    prior_axis_mapping: str = "sorted_short_long"
    drivable_term: str = "off"

    # --- upstream gate (C16) ---
    accept_degraded_upstream: bool = False

    # --- determinism (§1.9): recorded for manifest uniformity; no RNG here ---
    global_seed: int = 20260812

    provenance: dict = field(
        default_factory=lambda: {
            "conf_gate": (
                "arbitrary, needs tuning. comprehensive.md §7.4 says QA thresholds are tuned "
                "to maximize auto-accept precision at a fixed target; the pilot never tunes "
                "them (owed on the `tuning` split, §11 decision 3, alongside the Stage 3 "
                "thresholds — C21 open item)"
            ),
            "min_lidar_returns": (
                "comprehensive.md §7.3.9 'LiDAR-return >= 5'; counted single-sweep, "
                "ground-filtered, pre-inflation (§1.4) — basis asserted, not assumed"
            ),
            "spatial_multiplier": "comprehensive.md §7.3.9 'BEV box exceeds 2x class prior'",
            "prior_axis_mapping": (
                "matches Stage 8's default prior_axis_mapping=sorted_short_long, so gate and "
                "inflation agree on which prior axis is which (schemas.py §3.2 [w,l] trap)"
            ),
            "drivable_term": (
                "OFF, §11 decision 7 (locked at Phase 10): the map expansion is on disk so it "
                "is implementable, but comprehensive.md §1.4 excludes HD maps from v1.0 — a "
                "pilot component with no production counterpart teaches nothing. spatial_ok "
                "below is the size term only, and spatial_ok_source says so on every record"
            ),
            "tier_rule": (
                "rejected = physical gate failure (returns|spatial); flagged = conf below "
                "gate or spatial unevaluable; auto_accept = all pass. Stated here because "
                "comprehensive.md names the tiers but not the mapping; arbitrary, recorded"
            ),
            "accept_degraded_upstream": "C16 — consuming a DEGRADED upstream is an explicit "
            "recorded decision, never a default",
            "global_seed":
              "§1.9 uniformity; this stage is a pure function of its inputs "
            "and draws no random numbers",
        }
    )

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not 0.0 <= self.conf_gate <= 1.0:
            errors.append(f"conf_gate={self.conf_gate} must be in [0, 1]")
        if self.min_lidar_returns < 1:
            errors.append(f"min_lidar_returns={self.min_lidar_returns} must be >= 1")
        if self.spatial_multiplier <= 1.0:
            errors.append(
                f"spatial_multiplier={self.spatial_multiplier} must be > 1 (a multiplier at or "
                "below 1 rejects every box at least as large as its class mean)"
            )
        if self.prior_axis_mapping != "sorted_short_long":
            errors.append(
                f"prior_axis_mapping={self.prior_axis_mapping!r}: only 'sorted_short_long' is "
                "implemented (it is Stage 8's default; adding a mode here without adding it "
                "there would gate and inflate under different axis conventions)"
            )
        if self.drivable_term != "off":
            errors.append(
                "drivable_term must be 'off' (§11 decision 7, locked at Phase 10). Enabling it "
                "requires a real map lookup plus no_production_counterpart marking — not a flag"
            )
        return errors

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# The gates. Each is a pure function of one row (+ prior, + config).
# ---------------------------------------------------------------------------


def returns_gate(row: dict, cfg: GateConfig) -> bool:
    """`num_lidar_pts >= floor`, on the asserted single-sweep basis (§1.4).

    The basis check REFUSES rather than flags: a row counted on the
    accumulation is not a borderline box, it is a different measurement
    wearing this one's field name.
    """
    basis = row.get("num_lidar_pts_basis")
    if basis != REQUIRED_PTS_BASIS:
        raise GateContractError(
            f"{row.get('keyframe_token')}/{row.get('channel')}/{row.get('proposal_index')}: "
            f"num_lidar_pts_basis={basis!r}, need {REQUIRED_PTS_BASIS!r}. A count from the "
            "accumulation loosens the >=5-return gate ~10x on this substrate while the gate "
            "still appears to fire (§1.4)"
        )
    return int(row["num_lidar_pts"]) >= cfg.min_lidar_returns


def spatial_gate(row: dict, prior, cfg: GateConfig) -> bool | None:
    """The 2x-class-prior size term, on PRE-inflation BEV dimensions (P1-12).

    Reads `box_measured` and never `box`: Stage 8 inflates toward the prior,
    so the shipped box passes this gate more easily by construction. Returns
    None ("not evaluable") when the class has no usable prior dims — the
    caller tiers that `flagged` and the run degrades, because a gate that
    silently passes what it cannot measure is not a gate.
    """
    if prior is None or prior.dims is None:
        return None
    measured = row["box_measured"]["size_wlh_m"]
    short_meas, long_meas = sorted((float(measured[0]), float(measured[1])))
    short_mu, long_mu = sorted((float(prior.dims["w"]["mu"]), float(prior.dims["l"]["mu"])))
    exceeds = (
        short_meas > cfg.spatial_multiplier * short_mu
        or long_meas > cfg.spatial_multiplier * long_mu
    )
    return not exceeds


def tier_of(conf: float, lidar_ok: bool, spatial_ok: bool | None, cfg: GateConfig) -> str:
    """The stated tier rule (see module docstring / config provenance)."""
    if not lidar_ok or spatial_ok is False:
        return TIER_REJECTED
    if conf < cfg.conf_gate or spatial_ok is None:
        return TIER_FLAGGED
    return TIER_AUTO


SPATIAL_SOURCE = (
    "bev_size_vs_{mult}x_class_prior;dims=pre_inflation(box_measured);"
    "axis=sorted_short_long;drivable_term=off(S11-d7)"
)


def gate_row(row: dict, prior, cfg: GateConfig) -> tuple[GateVector, str]:
    conf = float(row["score"])
    lidar_ok = returns_gate(row, cfg)
    spatial = spatial_gate(row, prior, cfg)
    gates = GateVector(
        conf=conf,
        lidar_pts_ok=lidar_ok,
        spatial_ok=spatial,
        spatial_ok_source=(
            SPATIAL_SOURCE.format(mult=cfg.spatial_multiplier) if spatial is not None else None
        ),
    )
    return gates, tier_of(conf, lidar_ok, spatial, cfg)


# ---------------------------------------------------------------------------
# Row -> I-4 AnnotationRecord
# ---------------------------------------------------------------------------


def record_of(row: dict, gates: GateVector, tier: str) -> AnnotationRecord:
    """One §A.1 record from one Stage 8 row.

    The SHIPPED (post-inflation) box is what the pre-label carries — that is
    the geometry a reviewer corrects — while the gate vector beside it was
    computed on the measured one. Both live in the Stage 8 row; the token
    points back at it.
    """
    box = row["box"]
    token = f"{row['keyframe_token']}:{row['channel']}:{row['proposal_index']}"
    track_id = row.get("track_id")
    if track_id is not None:
        instance_token = f"pilot-track:{row['scene_token']}:{track_id}"
    else:
        # An untracked detection is its own instance; inventing a shared one
        # would fabricate identity the pipeline never established.
        instance_token = f"pilot-det:{token}"

    velocity = row.get("velocity_mps")
    if velocity is not None:
        if not isinstance(velocity, (list, tuple)) or len(velocity) != 3:
            raise GateContractError(
                f"{token}: velocity_mps={velocity!r}, expected a 3-vector from Stage 7"
            )
        # I-4 carries BEV velocity (2-vector, §A.1); Stage 7's vz is dropped
        # here, not averaged in.
        velocity = [float(velocity[0]), float(velocity[1])]

    return AnnotationRecord(
        token=token,
        sample_token=row["keyframe_token"],
        instance_token=instance_token,
        category=row["class_name"],
        frame=row["frame"],
        t_ns=int(row["t_ns"]),
        time_base=row["time_base"],
        translation_m=[float(v) for v in box["translation_m"]],
        size_wlh_m=[float(v) for v in box["size_wlh_m"]],
        rotation_wxyz=[float(v) for v in box["rotation_wxyz"]],
        num_lidar_pts=int(row["num_lidar_pts"]),
        num_lidar_pts_basis=row["num_lidar_pts_basis"],
        provenance=Provenance(source="pipeline", tier=tier, gates=gates),
        coverage_config=row["coverage_config"],
        velocity_mps=velocity,
        track_id=str(track_id) if track_id is not None else None,
        split=None,
    )


# ---------------------------------------------------------------------------
# Upstream
# ---------------------------------------------------------------------------


def load_upstream(
    paths: Paths, stage8_dir: str, priors_path: str, *, accept_degraded: bool = False
):
    """The C16 gate over Stage 8, plus the priors-identity join.

    The priors check mirrors Stage 8's own: the file this gate multiplies by 2
    must be byte-identical to the file the chain clustered and inflated with —
    otherwise "exceeds 2x class prior" tests against a different class
    definition than the one that produced the box, and nothing in the output
    says which.
    """
    current = metadata_fingerprint(paths)
    manifest, marker = require_upstream(
        stage8_dir,
        stage_name="Stage 8",
        module_hint="pipeline.stage8_inflate.inflate",
        current_fingerprint=current,
        accept_degraded=accept_degraded,
    )
    if manifest.get("frame") != EGO:
        raise UpstreamRefusal(f"Stage 8 claims frame={manifest.get('frame')!r}, not {EGO!r}")

    priors = load_priors(priors_path)
    if priors.metadata_fingerprint and priors.metadata_fingerprint != current:
        raise UpstreamRefusal(
            f"priors fingerprint mismatch: {priors.path} was derived against "
            f"{priors.metadata_fingerprint}, this dataroot is {current}"
        )
    upstream_priors = manifest.get("upstream", {}).get("priors", {})
    if upstream_priors.get("sha256") and upstream_priors["sha256"] != priors.sha256:
        raise UpstreamRefusal(
            f"priors mismatch: the chain ran with {upstream_priors.get('name')}@"
            f"{upstream_priors['sha256'][:16]}, this run was given {priors.name}@{priors.sha256[:16]}"
        )
    return manifest, marker, priors


def scene_names_from_stage8(stage8_dir: str, wanted: Sequence[str] | None) -> list[str]:
    root = os.path.join(stage8_dir, "scenes")
    if not os.path.isdir(root):
        raise UpstreamRefusal(f"{root} not found; Stage 8 wrote no scenes")
    names = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n)))
    if wanted:
        missing = sorted(set(wanted) - set(names))
        if missing:
            raise UpstreamRefusal(f"requested scene(s) not present in Stage 8 output: {missing}")
        names = [n for n in names if n in wanted]
    return names


def read_rows(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise UpstreamRefusal(f"{path} not found; Stage 8's scene tree is incomplete")
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run(
    paths: Paths,
    stage8_manifest: dict,
    stage8_marker,
    priors: Priors,
    cfg: GateConfig,
    stage8_dir: str,
    out_dir: str,
    scene_names: Sequence[str] | None,
) -> tuple[dict, int]:
    started = time.time()
    errors = cfg.validate()
    if errors:
        raise UpstreamRefusal("; ".join(errors))
    clear_markers(out_dir)

    names = scene_names_from_stage8(stage8_dir, scene_names)

    per_scene: dict[str, dict] = {}
    degraded_causes: list[str] = []
    missing_prior_classes: set[str] = set()
    totals = {
        "n_rows": 0,
        "n_no_box": 0,
        "n_prelabels": 0,
        "n_spatial_unevaluable": 0,
        TIER_AUTO: 0,
        TIER_FLAGGED: 0,
        TIER_REJECTED: 0,
    }

    for scene in names:
        rows = read_rows(os.path.join(stage8_dir, "scenes", scene, "inflated.jsonl"))
        records: list[AnnotationRecord] = []
        no_box: dict[str, int] = {}
        tiers = {TIER_AUTO: 0, TIER_FLAGGED: 0, TIER_REJECTED: 0}
        n_unevaluable = 0
        seen_tokens: set[str] = set()

        for row in rows:
            if not row.get("box"):
                status = str(row.get("status", "no_box"))
                no_box[status] = no_box.get(status, 0) + 1
                continue
            prior = priors.classes.get(row["class_name"])
            if prior is None or prior.dims is None:
                missing_prior_classes.add(row["class_name"])
                n_unevaluable += 1
            gates, tier = gate_row(row, prior, cfg)
            record = record_of(row, gates, tier)
            if record.token in seen_tokens:
                raise GateContractError(
                    f"{scene}: duplicate pre-label token {record.token!r}; "
                    "(keyframe, channel, proposal_index) stopped being unique upstream"
                )
            seen_tokens.add(record.token)
            tiers[tier] += 1
            records.append(record)

        out_path = os.path.join(out_dir, "scenes", scene, "prelabels.jsonl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        # allow_empty declared: a scene whose every cluster failed to fit has no
        # pre-labels; the empty file plus the degraded cause below IS the record.
        write_records(out_path, records, expect_type=AnnotationRecord, allow_empty=True)

        if not records:
            degraded_causes.append(f"{scene}: 0 pre-labels ({len(rows)} rows, all without a box)")

        per_scene[scene] = {
            "n_rows": len(rows),
            "n_no_box": sum(no_box.values()),
            "no_box_by_status": dict(sorted(no_box.items())),
            "n_prelabels": len(records),
            "n_spatial_unevaluable": n_unevaluable,
            "tiers": tiers,
        }
        totals["n_rows"] += len(rows)
        totals["n_no_box"] += sum(no_box.values())
        totals["n_prelabels"] += len(records)
        totals["n_spatial_unevaluable"] += n_unevaluable
        for t, n in tiers.items():
            totals[t] += n

    if missing_prior_classes:
        degraded_causes.append(
            "spatial gate unevaluable (no prior dims) for: "
            + ", ".join(sorted(missing_prior_classes))
        )

    code = EXIT_DEGRADED if degraded_causes else EXIT_OK
    manifest = {
        "spec": STAGE_SPEC,
        "stage": STAGE,
        # The Stage 6 producer the gated boxes descend from, via stages 7 and 8.
        "boxes_source": boxes_source(stage8_manifest),
        "num_lidar_pts_basis_detail": num_lidar_pts_basis_detail(stage8_manifest),
        "seed": cfg.global_seed,
        "config": cfg.as_dict(),
        "frame": EGO,
        "upstream": {
            "metadata_fingerprint": stage8_manifest["upstream"]["metadata_fingerprint"],
            "fingerprint_spec": stage8_manifest["upstream"].get("fingerprint_spec"),
            "stage8_spec": stage8_manifest["spec"],
            # C16: the whole degradation chain, not just the last link.
            "stage8_degraded": stage8_marker.degraded,
            "stage8_degraded_causes": list(stage8_marker.causes),
            "accepted_degraded_upstream": cfg.accept_degraded_upstream,
            "priors": {
                "name": priors.name,
                "sha256": priors.sha256,
                "source": priors.source,
                "path": priors.path,
            },
        },
        "gated_dimensions": (
            "pre-inflation (`box_measured`), P1-12: Stage 8 inflates toward the prior, so the "
            "shipped box passes a 2x-prior test more easily by construction. Both dimension "
            "sets are in the Stage 8 row; the shipped box is what the pre-label carries"
        ),
        "returns_basis": REQUIRED_PTS_BASIS,
        "tier_rule": cfg.provenance["tier_rule"],
        "not_ground_truth": (
            "nothing ships as ground truth from this stage (comprehensive.md §7.4): every "
            "record is source='pipeline', tier is a review-priority signal, and "
            "allow_human_provenance=False is pilot-wide (§1.7)"
        ),
        "numpy_version": np.__version__,
        "python_version": sys.version.split()[0],
        "elapsed_s": round(time.time() - started, 2),
        "scenes": per_scene,
        "totals": totals,
        "degraded_causes": degraded_causes,
    }
    return manifest, code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--stage8-dir", default=None, help="default <work_root>/stage8_inflate")
    parser.add_argument("--priors", default=None, help="default <out_root>/priors/{}.json".format(PRIORS_NAME))
    parser.add_argument("--out-dir", default=None, help="default <work_root>/stage9_qa")
    parser.add_argument("--scenes", nargs="*", default=None, help="subset of Stage 8 scene names")
    parser.add_argument("--conf-gate", type=float, default=None)
    parser.add_argument("--min-lidar-returns", type=int, default=None)
    parser.add_argument("--spatial-multiplier", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None, help="override the recorded seed (no RNG here)")
    parser.add_argument(
        "--accept-degraded-upstream",
        action="store_true",
        help="consume a DEGRADED (complete, quality-flagged) Stage 8 output; recorded (C16)",
    )
    args = parser.parse_args(argv)

    try:
        paths = load_paths(args.paths)
    except PathValidationError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    stage8_dir = args.stage8_dir or os.path.join(paths.work_root, "stage8_inflate")
    priors_path = args.priors or os.path.join(paths.out_root, "priors", f"{PRIORS_NAME}.json")
    out_dir = args.out_dir or os.path.join(paths.work_root, STAGE)
    assert_dataroot_read_only(paths, out_dir)

    cfg = GateConfig(
        accept_degraded_upstream=args.accept_degraded_upstream,
        **({"conf_gate": args.conf_gate} if args.conf_gate is not None else {}),
        **({"min_lidar_returns": args.min_lidar_returns} if args.min_lidar_returns is not None else {}),
        **({"spatial_multiplier": args.spatial_multiplier} if args.spatial_multiplier is not None else {}),
        **({"global_seed": args.seed} if args.seed is not None else {}),
    )

    try:
        stage8_manifest, stage8_marker, priors = load_upstream(
            paths, stage8_dir, priors_path, accept_degraded=cfg.accept_degraded_upstream
        )
        manifest, code = run(
            paths, stage8_manifest, stage8_marker, priors, cfg, stage8_dir, out_dir, args.scenes
        )
    except UpstreamRefusal as exc:
        print(f"REFUSING TO START: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    except (GateContractError, SchemaValidationError, PathValidationError) as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)
    # Three-state marker (§1.9, C16): clean / degraded-with-causes / absent.
    write_marker(
        out_dir,
        manifest["upstream"]["metadata_fingerprint"],
        degraded=code == EXIT_DEGRADED,
        causes=manifest["degraded_causes"],
    )

    t = manifest["totals"]
    print(f"rows                 : {t['n_rows']}  ({t['n_no_box']} without a box, excluded)")
    print(f"pre-labels           : {t['n_prelabels']}  (I-4, source='pipeline', schemas.write_records)")
    print(
        f"tiers                : auto_accept {t[TIER_AUTO]}   flagged {t[TIER_FLAGGED]}   "
        f"rejected {t[TIER_REJECTED]}"
    )
    print(
        f"gates                : conf >= {cfg.conf_gate}, returns >= {cfg.min_lidar_returns} "
        f"(single-sweep), BEV <= {cfg.spatial_multiplier}x prior (pre-inflation dims)"
    )
    if t["n_spatial_unevaluable"]:
        print(f"spatial unevaluable  : {t['n_spatial_unevaluable']}  (no prior dims -> flagged)")
    for cause in manifest["degraded_causes"]:
        print(f"degraded             : {cause}")
    print(f"wrote {out_dir}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
