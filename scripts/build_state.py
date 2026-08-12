#!/usr/bin/env python3
"""Generate docs/BUILD_STATE.md from disk evidence. Never hand-edit the output.

Exists to close DECISIONS.md C2: pilot_plan.md's hand-written status header went
stale the day it was written. This generator derives each phase's state from
things that cannot drift: module presence, run manifests, _SUCCESS markers, and
pytest results.

The state ladder, and the rule that keeps it honest (BUILD_PROMPT.md §9):

    NOT STARTED  ->  CODE WRITTEN  ->  RUN  ->  GATE PASSED

    CODE WRITTEN   the phase's modules exist on disk
    RUN            the phase's output artifacts exist (manifest / marker / file)
    GATE PASSED    the phase's pytest selection exists, was collected non-empty,
                   and passed. No tests -> no gate, no matter what else exists.

Silent failure this file guards against: "code written" being read as "done".
Phases 2-8 below will show CODE WRITTEN or RUN while their gates stay unpassed
until tests/ lands (DECISIONS.md C4) — that asymmetry is the point, not a bug.

Usage: python scripts/build_state.py [--check]
  --check    exit 1 if docs/BUILD_STATE.md is stale (for hooks/CI), write nothing.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "docs", "BUILD_STATE.md")

# Write roots come from configs/paths.yaml; parsed leniently so this script
# stays stdlib-only and runnable before the environment exists.
def _paths() -> dict:
    vals = {}
    with open(os.path.join(REPO, "configs", "paths.yaml")) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if ":" in line:
                k, _, v = line.partition(":")
                if v.strip():
                    vals[k.strip()] = v.strip()
    return vals


PATHS = _paths()
WORK = PATHS.get("work_root", "/home/mt/dhakascenes/work")
OUT_ROOT = PATHS.get("out_root", "/home/mt/dhakascenes/out")
PROBE_OUT = PATHS.get("probe_out_root", "/home/mt/dhakascenes/probe_out")

# Per-phase evidence. code: repo-relative modules that must all exist.
# run: absolute artifacts that must all exist. tests: pytest -k selection
# (empty selection = no gate defined yet, so the gate cannot pass).
PHASES: list[dict] = [
    dict(n=1, title="Substrate truth + path contract",
         code=["pipeline/common/paths.py", "scripts/probe_substrate.py"],
         run=[os.path.join(PROBE_OUT, "phase1_substrate.json")],
         tests="phase1 or paths_contract"),
    dict(n=2, title="pipeline/common/ — the representation contract",
         code=["pipeline/common/conventions.py", "pipeline/common/schemas.py",
               "pipeline/common/eval_region.py", "pipeline/common/manifest.py",
               "pipeline/common/model_interfaces.py",
               "configs/pipeline_pilot.yaml", "configs/models_pilot.yaml",
               "configs/models_production.yaml"],
         run=[],  # contract phase: RUN == its tests exist and pass; gate is the run
         tests="unit and (common or conventions or schemas or eval_region or manifest or env_contract)"),
    dict(n=3, title="Stage 0 probe + scene partition",
         code=["pipeline/stage0_data_probe/probe.py"],
         run=[os.path.join(WORK, "stage0_data_probe", "usable_scenes.json"),
              os.path.join(WORK, "stage0_data_probe", "probe_report.json")],
         tests="stage0"),
    dict(n=4, title="Stage 1 ingestion",
         code=["pipeline/stage1_ingestion/ingest.py"],
         run=[os.path.join(WORK, "stage1_ingestion", "run_manifest.json")],
         tests="stage1 or ingest"),
    dict(n=5, title="Model registry + VRAM (5a paper, 5b measured)",
         code=["pipeline/common/model_interfaces.py",
               "scripts/check_vram_paper.py", "scripts/measure_vram.py",
               "configs/models_pilot.yaml"],
         run=[os.path.join(WORK, "vram", "run_manifest.json")],
         tests="vram or registry or model_interfaces"),
    dict(n=6, title="Stages 3+4 — proposals + masks",
         code=["pipeline/stage3_proposals/proposals.py",
               "pipeline/stage4_masks/masks.py",
               "configs/taxonomy_pilot_nuscenes.yaml"],
         run=[os.path.join(WORK, "stage3_proposals", "run_manifest.json"),
              os.path.join(WORK, "stage4_masks", "run_manifest.json")],
         tests="stage3 or stage4 or proposals or masks"),
    dict(n=7, title="Stage 5 lift (paint-inside-GT is the gate number)",
         code=["pipeline/stage5_lift/lift.py"],
         run=[os.path.join(WORK, "stage5_lift", "run_manifest.json")],
         tests="stage5 or lift or paint_inside_gt or gt_reprojection"),
    dict(n=8, title="Stages 6+8 — cluster, priors, inflation",
         code=["pipeline/stage6_cluster/cluster.py",
               "pipeline/stage6_cluster/priors.py",
               "pipeline/stage8_inflate/inflate.py"],
         run=[os.path.join(OUT_ROOT, "priors", "priors_pilot_v0.json"),
              os.path.join(WORK, "stage6_cluster", "run_manifest.json"),
              os.path.join(WORK, "stage8_inflate", "run_manifest.json")],
         tests="stage6 or stage8 or cluster or priors or inflate"),
    dict(n=9, title="Stages 7+2 — tracking + OOD branch",
         code=["pipeline/stage7_track/track.py", "pipeline/stage2_ood/ood.py"],
         run=[os.path.join(WORK, "stage7_track", "run_manifest.json"),
              os.path.join(WORK, "stage2_ood", "run_manifest.json")],
         tests="stage7 or stage2 or track or ood"),
    dict(n=10, title="Stage 9 QA + end-to-end + claim hygiene",
         code=["pipeline/stage9_qa/qa.py", "scripts/run_pilot.py",
               "probes/indigenous_prompt_probe",
               "configs/taxonomy_probe_indigenous.yaml"],
         run=[os.path.join(OUT_ROOT, "run_manifest.json")],
         tests="stage9 or end_to_end or probe_separation or claim_hygiene"),
]


def _pytest_gate(selection: str) -> tuple[str, str]:
    """Run the phase's test selection. Returns (verdict, detail).

    verdict: PASS | FAIL | NO-TESTS | NO-HARNESS
    A selection that collects zero tests is NO-TESTS, not PASS — pytest exit
    code 5 exists precisely so an empty selection cannot masquerade as green.
    """
    if not os.path.isdir(os.path.join(REPO, "tests")):
        return "NO-HARNESS", "tests/ does not exist"
    env = dict(os.environ, PYTHONNOUSERSITE="1")
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-k", selection, "-q",
         "--no-header", "-x", "--timeout", "600"],
        capture_output=True, text=True, cwd=REPO, env=env,
    )
    tail = (r.stdout.strip().splitlines() or [""])[-1]
    if r.returncode == 0:
        return "PASS", tail
    if r.returncode == 5:
        return "NO-TESTS", "selection collected nothing"
    return "FAIL", tail


def phase_state(p: dict, run_tests: bool) -> dict:
    code_missing = [c for c in p["code"] if not os.path.exists(os.path.join(REPO, c))]
    run_missing = [a for a in p["run"] if not os.path.exists(a)]
    code_ok = not code_missing
    run_ok = code_ok and p["run"] and not run_missing

    gate, detail = ("SKIPPED", "run with tests enabled") if not run_tests \
        else _pytest_gate(p["tests"])

    if not code_ok:
        state = "NOT STARTED" if len(code_missing) == len(p["code"]) else "CODE PARTIAL"
    elif gate == "PASS" and (run_ok or not p["run"]):
        state = "GATE PASSED"
    elif run_ok:
        state = "RUN (ungated)"
    else:
        state = "CODE WRITTEN"

    return dict(state=state, gate=gate, gate_detail=detail,
                code_missing=code_missing, run_missing=run_missing)


def render(states: list[dict]) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, cwd=REPO
                                ).stdout.strip() or "(no commit)"
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True, cwd=REPO
                                    ).stdout.strip())
    except OSError:
        commit, dirty = "(no git)", False

    lines = [
        "# BUILD STATE — generated, do not edit",
        "",
        f"Generated {now} by `scripts/build_state.py` at commit `{commit}`"
        + (" (dirty tree)" if dirty else "") + ".",
        "",
        "> **This demonstrates pipeline plumbing only. Label quality is not evidence",
        "> of anything; models are deliberately under-tier; the substrate is nuScenes",
        "> v1.0-mini, not Dhaka.**",
        "",
        "`CODE WRITTEN ≠ RUN ≠ GATE PASSED`. A phase whose artifacts exist but whose",
        "tests do not is **RUN (ungated)** — its outputs are unverified (DECISIONS C7).",
        "",
        "| Phase | Title | State | Gate | Missing |",
        "|---|---|---|---|---|",
    ]
    for p, s in zip(PHASES, states):
        missing = ", ".join(
            [f"`{m}`" for m in s["code_missing"]]
            + [f"`{os.path.basename(m)}`" for m in s["run_missing"]]
        ) or "—"
        lines.append(f"| {p['n']} | {p['title']} | **{s['state']}** "
                     f"| {s['gate']} | {missing} |")

    lines += [
        "",
        "## Deferred verifications (may not be silently forgotten)",
        "",
        "- **C1**: the claim *\"runs on 4 GB VRAM\"* requires one end-to-end run on",
        "  the physical 3050ti laptop. Every run on this machine is under a synthetic",
        "  4096 MiB cap on a 24 GB RTX 4090 and supports only the capped claim.",
        "- **C7**: artifacts produced before tests existed remain unverified until",
        "  their phase's retro-fitted tests pass against them.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if BUILD_STATE.md is stale; write nothing")
    ap.add_argument("--no-tests", action="store_true",
                    help="skip pytest gates (fast; gates show SKIPPED)")
    args = ap.parse_args()

    states = [phase_state(p, run_tests=not args.no_tests) for p in PHASES]
    text = render(states)

    if args.check:
        try:
            with open(OUT) as f:
                current = f.read()
        except FileNotFoundError:
            current = ""
        # Compare everything below the timestamp line.
        strip = lambda t: "\n".join(t.splitlines()[3:])
        if strip(current) != strip(text):
            print("BUILD_STATE.md is stale — regenerate with scripts/build_state.py",
                  file=sys.stderr)
            return 1
        return 0

    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, OUT)
    print(f"wrote {OUT}")
    for p, s in zip(PHASES, states):
        print(f"  phase {p['n']:>2}  {s['state']:<14} gate={s['gate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
