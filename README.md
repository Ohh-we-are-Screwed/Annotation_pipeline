# DhakaScenes Pilot

> **This demonstrates pipeline plumbing only. Label quality is not evidence of
> anything; models are deliberately under-tier; the substrate is nuScenes
> v1.0-mini, not Dhaka.**

The banner above is required by `docs/pilot_plan.md` §13.2 in this README, in every
`run_manifest.json`, and in the header of every exported figure. It is not ceremonial:
see §13.2 for what a completed pilot supports and what it does not.

## What this is

A pilot implementation of the DhakaScenes auto-annotation pipeline
(`docs/comprehensive.md` §7) on nuScenes v1.0-mini as a stand-in substrate, built to
prove the *plumbing* — stage contracts, representation conventions, provenance
enforcement — with deliberately small model checkpoints. Governing documents and
their precedence are listed in [`docs/BUILD_PROMPT.md`](docs/BUILD_PROMPT.md) §1;
the bible is [`docs/pilot_plan.md`](docs/pilot_plan.md) (rev 2).

Build state: [`docs/BUILD_STATE.md`](docs/BUILD_STATE.md) (generated — do not edit).
Decision log: [`docs/DECISIONS.md`](docs/DECISIONS.md).

## Environment

One interpreter, always:

```bash
export PATH="/home/mt/miniconda3/bin:$PATH"
conda activate ano_pipe          # or: conda env create -f environment.yml
export PYTHONNOUSERSITE=1        # mandatory — see .env.example
cp .env.example .env             # then fill the SECRET block
pip install -r requirements-torch.txt
pip install -r requirements.txt
pip install --no-deps -r requirements-devkit.txt
```

Every dependency is exact-pinned with a provenance note in `requirements.txt`
(the index; it names the other files and the reason each split exists).
`requirements-lock.txt` is regenerated after any change.

## Hardware (decisions C1, C19)

Two machines: a 3050ti laptop (4 GB VRAM — the plan's target, where the pilot must
ultimately run) and a remote RTX 4090 (24 GB — where work happens now). Since C19
(2026-08-13) the **default tier on the 4090 is the best-locally-runnable one**
(LLMDet-large + SAM 2.1/SAM 3, `DHAKASCENES_VRAM_CAP_MIB=22000`); the 4 GB pilot
tier (grounding-dino-tiny + MobileSAM under a synthetic 4096 MiB ceiling) stays
selectable and remains the **only** tier that can support the "runs on 4 GB VRAM"
claim — and that claim still needs one end-to-end run on the physical laptop.
See `docs/DECISIONS.md` C1 and C19.

## Running

```bash
scripts/run_stages.sh        # stages 3 -> 4 -> 5 with the C19 default models
```

Per-stage CLIs, model/revision pins, gate semantics (`--accept-degraded-upstream`),
and the one-scene trial pattern: [`docs/RUNNING.md`](docs/RUNNING.md).

## Layout

```
pipeline/common/    representation contract (§1) — frames, time, schemas, paths, eval region
pipeline/stageN_*/  the ten pipeline stages (§5)
configs/            every tunable, each with a provenance note (§10)
scripts/            probe_substrate, measure_vram, build_state, drivers
docs/               the governing documents + this build's ledgers
tests/              unit (GPU-free, synthetic) / integration (opt-in via env var)
```

The nuScenes substrate and all generated outputs live **outside** version control
(`.gitignore`); the substrate is CC BY-NC-SA 4.0 and non-redistributable, and
GT-derived statistics (the priors) are licence-covered derived data (DECISIONS C12).
