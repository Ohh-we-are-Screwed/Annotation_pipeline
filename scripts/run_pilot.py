#!/usr/bin/env python3
"""End-to-end pilot driver over the `run` subset (Phase 10; plan §2, §1.9).

Relationship to `scripts/run_stages.sh`: the shell wrapper is OPERATOR
convenience — env loading, the flock, CVAT publishing, eval/viz. THIS script
is the contract-level driver the plan mandates, and it owns three things the
wrapper does not:

  1. **The allowlist binding (§1.8).** It reads `usable_scenes.json` and
     REFUSES to start when the dataroot in front of it does not fingerprint to
     the one the allowlist was derived from. An allowlist for a different
     substrate is not a weaker allowlist; it is no allowlist.
  2. **Per-scene failure isolation (§1.9).** A stage failure with several
     scenes in flight is re-probed one scene at a time; scenes that fail land
     in `failures.json` with the failing stage and the stderr tail, the
     SURVIVORS are re-run together (so the stage's own manifest and marker
     describe a coherent final state), and the chain continues. One corrupted
     scene must cost that scene, not the run — exercised deliberately at the
     Phase 10 gate.
  3. **The aggregate `run_manifest.json`**, written atomically, carrying the
     §1.9 identity set for the RUN as a whole: metadata fingerprint,
     `usable_scenes.json` hash, scene partition, git commit (+dirty flag),
     package versions, per-stage exit/marker states with degraded causes, and
     the §13.2 claim-hygiene banner.

Exit-code contract per stage (same three states the stages declare):
    rc 0 + clean marker      OK
    rc 1 + degraded marker   DEGRADED — complete, quality-flagged; the chain
                             continues and every later stage gets
                             `--accept-degraded-upstream` (C16, recorded)
    rc 1,  no marker         CRASHED (Python tracebacks exit 1 too) — isolate
    rc >= 2                  REFUSED, nothing written — isolate
A marker/exit-code disagreement in the other direction (rc 0, no clean
marker) is BROKEN and aborts: it means the stage and this driver disagree
about what "done" means, and continuing would consume an output the stage
itself did not vouch for.

    python -m scripts.run_pilot                    # stages 3..9, `run` subset
    python -m scripts.run_pilot --scenes scene-0757
    python -m scripts.run_pilot --steps 8 9        # resume a completed prefix
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.manifest import read_marker, write_json_atomic  # noqa: E402
from pipeline.common.paths import PathValidationError, load_paths, metadata_fingerprint  # noqa: E402

DRIVER_SPEC = "dhakascenes-pilot/run_pilot/v1"

EXIT_OK = 0
EXIT_FLAGGED = 1  # completed, but degraded stages and/or isolated scene failures
EXIT_REFUSED = 2

# §13.2 — the claim-hygiene banner (C10). Lives verbatim in README.md and in
# every figure header; the aggregate manifest carries it too.
BANNER = (
    "This demonstrates pipeline plumbing only. Label quality is not evidence of "
    "anything; models are deliberately under-tier; the substrate is nuScenes "
    "v1.0-mini, not Dhaka."
)

# Checkpoint pins — identical defaults to scripts/run_stages.sh, overridable
# through the same environment variables so the two drivers cannot silently
# diverge on which model a stage loads.
PROPOSAL_MODEL_ID = os.environ.get("PROPOSAL_MODEL_ID", "iSEE-Laboratory/llmdet_large")
PROPOSAL_REVISION = os.environ.get("PROPOSAL_REVISION", "bec37f296f05b22f6c6b39bc05a6c611239f4e31")
MASK_MODEL_ID = os.environ.get("MASK_MODEL_ID", "facebook/sam3")
MASK_REVISION = os.environ.get("MASK_REVISION", "3c879f39826c281e95690f02c7821c4de09afae7")
REID_REVISION = os.environ.get("REID_REVISION", "ed25f3a31f01632728cabb09d1542f84ab7b0056")

STAGE_DIRS = {
    "3": "stage3_proposals",
    "4": "stage4_masks",
    "5": "stage5_lift",
    "6": "stage6_cluster",
    "7": "stage7_track",
    "8": "stage8_inflate",
    "9": "stage9_qa",
}
CHAIN = ("3", "4", "5", "6", "7", "8", "9")

PACKAGES_OF_RECORD = (
    "numpy", "scipy", "scikit-learn", "torch", "transformers", "umap-learn", "hdbscan",
)


def stage_cmd(step: str, scenes: list[str], accept_degraded: bool, work_root: str) -> list[str]:
    """The exact argv for one stage, single-sourced so probe and re-run agree."""
    acc = ["--accept-degraded-upstream"] if accept_degraded else []
    scn = ["--scenes", *scenes]
    py = [sys.executable, "-m"]
    if step == "3":
        return py + ["pipeline.stage3_proposals.proposals", *scn, *acc,
                     "--model-id", PROPOSAL_MODEL_ID, "--revision", PROPOSAL_REVISION]
    if step == "4":
        return py + ["pipeline.stage4_masks.masks", *scn, *acc,
                     "--model-id", MASK_MODEL_ID, "--revision", MASK_REVISION]
    if step == "5":
        return py + ["pipeline.stage5_lift.lift", *scn, *acc]
    if step == "6":
        return py + ["pipeline.stage6_cluster.cluster", *scn, *acc]
    if step == "7":
        return py + ["pipeline.stage7_track.track", *scn, *acc, "--reid-revision", REID_REVISION]
    if step == "8":
        boxes = os.path.join(work_root, "stage7_track")
        args = py + ["pipeline.stage8_inflate.inflate", *scn, *acc]
        # Stage 8 prefers tracked boxes when Stage 7 completed (same rule as the
        # wrapper); its own default is stage6.
        if read_marker(boxes) is not None:
            args += ["--boxes-dir", boxes]
        return args
    if step == "9":
        return py + ["pipeline.stage9_qa.gate", *scn, *acc]
    raise ValueError(f"unknown step {step!r}")


def run_stage(step: str, scenes: list[str], accept_degraded: bool, work_root: str, log) -> dict:
    """Run one stage over `scenes`; classify rc x marker into a verdict."""
    stage_dir = os.path.join(work_root, STAGE_DIRS[step])
    cmd = stage_cmd(step, scenes, accept_degraded, work_root)
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = round(time.time() - started, 1)
    marker = read_marker(stage_dir)

    if proc.returncode == 0:
        verdict = "OK" if (marker is not None and not marker.degraded) else "BROKEN"
    elif proc.returncode == 1:
        verdict = "DEGRADED" if (marker is not None and marker.degraded) else "CRASHED"
    else:
        verdict = "REFUSED"

    tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-8:])
    log(f"  stage {step} [{','.join(scenes)}] -> rc {proc.returncode} {verdict} ({elapsed}s)")
    if verdict in ("CRASHED", "REFUSED", "BROKEN") and tail:
        log("    " + tail.replace("\n", "\n    "))
    return {
        "step": step,
        "scenes": list(scenes),
        "cmd": cmd,
        "rc": proc.returncode,
        "verdict": verdict,
        "marker": None if marker is None else marker.state,
        "degraded_causes": list(marker.causes) if marker is not None else [],
        "elapsed_s": elapsed,
        "stderr_tail": tail if verdict in ("CRASHED", "REFUSED", "BROKEN") else "",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="default: the `run` subset from usable_scenes.json (§11 d3)")
    ap.add_argument("--steps", nargs="*", default=list(CHAIN), choices=list(CHAIN),
                    help="subset of the chain, e.g. `--steps 8 9` to resume")
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, flush=True)

    try:
        paths = load_paths(args.paths)
    except PathValidationError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_REFUSED

    started_at = time.time()
    current = metadata_fingerprint(paths)

    # --- §1.8: the allowlist binding -------------------------------------
    allowlist_path = os.path.join(paths.work_root, "stage0_data_probe", "usable_scenes.json")
    if not os.path.isfile(allowlist_path):
        print(f"REFUSING TO START: {allowlist_path} not found; run "
              "`python3 -m pipeline.stage0_data_probe.probe` first", file=sys.stderr)
        return EXIT_REFUSED
    raw = open(allowlist_path, "rb").read()
    allowlist = json.loads(raw)
    allow_fp = allowlist["manifest"]["metadata_fingerprint"]
    if allow_fp != current:
        print(
            f"REFUSING TO START: usable_scenes.json was derived against {allow_fp}, this "
            f"dataroot fingerprints to {current}. The allowlist describes a different "
            "substrate (§1.8); re-run Stage 0 rather than trusting a foreign allowlist",
            file=sys.stderr,
        )
        return EXIT_REFUSED

    usable = {s["name"] for s in allowlist["scenes"]}
    partition = allowlist.get("partition", {}).get("subsets", {})
    run_subset = list(partition.get("run", {}).get("scenes", [])) or sorted(usable)
    scenes = list(args.scenes) if args.scenes else run_subset
    outside = sorted(set(scenes) - usable)
    if outside:
        print(f"REFUSING TO START: scene(s) {outside} are not on the Stage 0 allowlist",
              file=sys.stderr)
        return EXIT_REFUSED

    # Stage 1 must exist and describe this substrate; the driver does not
    # re-run ingestion implicitly (it covers ALL scenes and is expensive —
    # an explicit `run_stages.sh 0 1` owns that).
    stage1_marker = read_marker(os.path.join(paths.work_root, "stage1_ingestion"))
    if stage1_marker is None:
        print("REFUSING TO START: Stage 1 has no completion marker; run stages 0/1 first",
              file=sys.stderr)
        return EXIT_REFUSED
    if stage1_marker.fingerprint != current:
        print(f"REFUSING TO START: Stage 1 completed against {stage1_marker.fingerprint}, "
              f"this dataroot is {current}", file=sys.stderr)
        return EXIT_REFUSED

    log(f"pilot driver: scenes {scenes}  steps {list(args.steps)}")
    log(f"substrate {current[:16]}…  allowlist sha256 {hashlib.sha256(raw).hexdigest()[:16]}…")

    # C16: arm --accept-degraded-upstream as soon as anything upstream is
    # degraded — including markers already on disk from stages outside --steps.
    accept_degraded = stage1_marker.degraded
    for step in CHAIN:
        if step not in args.steps:
            marker = read_marker(os.path.join(paths.work_root, STAGE_DIRS[step]))
            if marker is not None and marker.degraded:
                accept_degraded = True

    step_results: list[dict] = []
    failures: list[dict] = []
    aborted = ""

    for step in [s for s in CHAIN if s in args.steps]:
        if not scenes:
            aborted = f"no scenes left before stage {step}"
            break
        result = run_stage(step, scenes, accept_degraded, paths.work_root, log)
        if result["verdict"] == "DEGRADED":
            accept_degraded = True
        if result["verdict"] == "BROKEN":
            step_results.append(result)
            aborted = (f"stage {step}: rc 0 without a clean marker — the stage and this driver "
                       "disagree about what 'done' means")
            break
        if result["verdict"] in ("CRASHED", "REFUSED"):
            # --- §1.9 per-scene isolation --------------------------------
            log(f"  stage {step} failed with {len(scenes)} scene(s); probing one at a time")
            survivors: list[str] = []
            for scene in scenes:
                probe = run_stage(step, [scene], accept_degraded, paths.work_root, log)
                if probe["verdict"] in ("OK", "DEGRADED"):
                    survivors.append(scene)
                    if probe["verdict"] == "DEGRADED":
                        accept_degraded = True
                else:
                    failures.append({
                        "scene": scene, "stage": step, "rc": probe["rc"],
                        "verdict": probe["verdict"], "stderr_tail": probe["stderr_tail"],
                    })
            if not survivors:
                step_results.append(result)
                aborted = f"stage {step}: every scene failed individually"
                break
            # Re-run the survivors TOGETHER so the stage's own manifest and
            # marker describe the set the chain continues with.
            result = run_stage(step, survivors, accept_degraded, paths.work_root, log)
            if result["verdict"] not in ("OK", "DEGRADED"):
                step_results.append(result)
                aborted = f"stage {step}: survivor re-run failed ({result['verdict']})"
                break
            if result["verdict"] == "DEGRADED":
                accept_degraded = True
            scenes = survivors
        step_results.append(result)

    # --- the aggregate manifest (§1.9), atomic ---------------------------
    def pkg_versions() -> dict:
        import importlib.metadata as im
        out = {"python": sys.version.split()[0]}
        for name in PACKAGES_OF_RECORD:
            try:
                out[name] = im.version(name)
            except im.PackageNotFoundError:
                out[name] = None
        return out

    def git_identity() -> dict:
        try:
            here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            commit = subprocess.run(["git", "-C", here, "rev-parse", "HEAD"],
                                    capture_output=True, text=True).stdout.strip()
            dirty = bool(subprocess.run(["git", "-C", here, "status", "--porcelain"],
                                        capture_output=True, text=True).stdout.strip())
            return {"commit": commit or None, "dirty": dirty}
        except OSError:
            return {"commit": None, "dirty": None}

    out_dir = os.path.join(paths.work_root, "pilot_run")
    completed = not aborted
    manifest = {
        "spec": DRIVER_SPEC,
        "banner": BANNER,
        "metadata_fingerprint": current,
        "usable_scenes_sha256": hashlib.sha256(raw).hexdigest(),
        "scene_partition": partition,
        "scenes_requested": list(args.scenes) if args.scenes else run_subset,
        "scenes_completed": scenes if completed else [],
        "steps": step_results,
        "failures": failures,
        "aborted": aborted or None,
        "accepted_degraded_upstream": accept_degraded,
        "checkpoints": {
            "proposal_2d": {"model_id": PROPOSAL_MODEL_ID, "revision": PROPOSAL_REVISION},
            "mask_2d": {"model_id": MASK_MODEL_ID, "revision": MASK_REVISION},
            "reid_embedding": {"revision": REID_REVISION},
        },
        "git": git_identity(),
        "packages": pkg_versions(),
        "elapsed_s": round(time.time() - started_at, 1),
    }
    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)
    write_json_atomic(os.path.join(out_dir, "failures.json"),
                      {"spec": DRIVER_SPEC, "failures": failures})

    log("")
    log(f"scenes completed  : {scenes if completed else '(aborted)'}")
    log(f"scene failures    : {len(failures)}" + (f"  -> {out_dir}/failures.json" if failures else ""))
    degraded_steps = [r["step"] for r in step_results if r["verdict"] == "DEGRADED"]
    if degraded_steps:
        log(f"degraded stages   : {degraded_steps} (complete, quality-flagged; causes in markers)")
    if aborted:
        log(f"ABORTED           : {aborted}")
    log(f"wrote {out_dir}/run_manifest.json")

    if aborted:
        return EXIT_REFUSED
    return EXIT_FLAGGED if (failures or degraded_steps or accept_degraded) else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
