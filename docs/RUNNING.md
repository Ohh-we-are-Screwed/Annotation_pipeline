# Running the pipeline

> **This demonstrates pipeline plumbing only. Label quality is not evidence of
> anything; the substrate is nuScenes v1.0-mini, not Dhaka.**

How to actually execute the stages on this machine (the 4090 box), with the
C19 model tier. Written 2026-08-13. Companion to `README.md` (environment
setup) and `docs/DECISIONS.md` (why everything is the way it is).

## The short version

```bash
conda activate ano_pipe            # or use the interpreter path directly
scripts/run_stages.sh              # stages 3-8, then eval + viz, then publish to CVAT
```

That is the whole run, end to end: the six model stages, the COCO exports and
metrics, the diagnostic renders, and the CVAT republish of the 2D review
tasks. Each stage is also a standalone CLI (below) — the wrapper only chains
them. It is **not** the Phase-10 orchestrator (`scripts/run_pilot.py`, still
unwritten); every stage writes its own `run_manifest.json` and success marker
exactly as if run by hand.

Subsets and flags:

```bash
scripts/run_stages.sh 4 5 6 7 8 eval cvat   # resume after a completed stage 3
scripts/run_stages.sh cvat                  # republish from what is on disk
scripts/run_stages.sh --no-cvat             # full chain, leave CVAT alone
scripts/run_stages.sh --scenes scene-0061   # one scene, end to end
```

### How it self-moderates

The stage CLIs have a **three-state** exit code and the wrapper honours all
three — the previous version collapsed them into pass/fail and stopped the
chain every time Stage 3 flagged a scene:

| rc | meaning | wrapper's response |
|---|---|---|
| 0 | clean | continue |
| 1 | **DEGRADED** — ran to completion, output complete, some scenes quality-flagged | record it, continue, and pass `--accept-degraded-upstream` to every later stage |
| 2 | **REFUSED** — contract broken or checkpoint missing, **nothing written** | abort the chain and suppress the CVAT publish |

The C16 consequence is deliberate and worth stating: once anything upstream is
degraded, the wrapper opts in to it for the rest of the chain. A human still
opted in — once, by running the script — and every consumer records
`upstream.accepted_degraded_upstream: true` in its own manifest, so the
provenance survives. What is gone is the per-stage prompt between a degraded
Stage 1 and a published CVAT task.

Concurrency and overwriting: the wrapper takes `flock` on
`<work_root>/.run_stages.lock` and refuses to start if another run holds it,
tees everything to `<work_root>/logs/run_<timestamp>.log`, rewrites each stage
tree in place, and **deletes and recreates** the `— OUR PIPELINE output` CVAT
tasks. Human edits inside those tasks do not survive a republish; the
`— nuScenes HUMAN answer key` tasks are never touched (C13).

## What runs, and what doesn't (today)

| Stage | Module | State |
|---|---|---|
| 0 probe | `pipeline/stage0_data_probe/probe.py` | already run (see `work_root`) |
| 1 ingestion | `pipeline/stage1_ingestion/ingest.py` | already run — **DEGRADED** (2 of 10 scenes exceeded the sector-rejection threshold; C16) |
| 2 OOD branch | `pipeline/stage2_ood/ood.py` | peer-owned (C18); still hard-refuses a degraded upstream |
| **3 proposals** | `pipeline/stage3_proposals/proposals.py` | runnable — this doc |
| **4 masks** | `pipeline/stage4_masks/masks.py` | runnable — this doc |
| **5 lift** | `pipeline/stage5_lift/lift.py` | runnable — this doc |
| **6 cluster** | `pipeline/stage6_cluster/cluster.py` | runnable — in the wrapper chain |
| **7 track** | `pipeline/stage7_track/track.py` | runnable — in the wrapper chain |
| **8 inflate** | `pipeline/stage8_inflate/inflate.py` | runnable — in the wrapper chain, boxes from Stage 7 |
| 9 QA | — | not started (Phase 10) |

Stages 3–8 are migrated to `pipeline/common/manifest.py`'s three-state markers
(C16), so they can consume the degraded Stage 1 output — but only when you say
so explicitly (`--accept-degraded-upstream`, recorded in the consumer's own
manifest; the wrapper passes it for you once something upstream is flagged).

Stage 8 takes its boxes from whatever produced them: the wrapper points it at
`stage7_track` when Stage 7 has a marker, and falls back to `stage6_cluster`
when it does not.

## Environment, in one screen

```bash
# interpreter — ALWAYS this one (C15 was caused by the wrong one)
PY=/home/mt/miniconda3/envs/ano_pipe/bin/python

# VRAM budget — the stage CLIs read the env var, not .env. Source it or export it:
export DHAKASCENES_VRAM_CAP_MIB=22000   # C19 production budget on the 4090
#                              4096     # C1 pilot budget (4 GB laptop contract)

# Hugging Face auth — needed once, only for the gated facebook/sam3 repo:
$PY -m huggingface_hub.cli auth login   # or: <env>/bin/hf auth login
```

`scripts/run_stages.sh` exports the cap from `.env` automatically if you
haven't. If the cap is unset entirely, the run still works but its fit claims
carry `verified: false` (C1 semantics — the physical card is the ceiling).

All checkpoints live in `~/.cache/huggingface/hub` after first download; every
later run is offline.

## The models (C19, decided 2026-08-13)

| Role | Default checkpoint | Pinned revision | Measured on this 4090 |
|---|---|---|---|
| `proposal_2d` (Stage 3) | `iSEE-Laboratory/llmdet_large` | `bec37f296f05b22f6c6b39bc05a6c611239f4e31` | 7.5 GiB fp32, ~350 ms / 1600×900 frame |
| `mask_2d` (Stage 4) | `facebook/sam3` — tracker path (gated; access granted 2026-08-13) | `3c879f39826c281e95690f02c7821c4de09afae7` | 2.1 GiB, ~0.7 s (smoke) |
| `mask_2d` (ungated alternate) | `facebook/sam2.1-hiera-large` | `665f8e2ad61cf5f53d65644ff27c8ee525124610` | 1.5 GiB, ~0.13 s / keyframe |
| pilot fallback | `IDEA-Research/grounding-dino-tiny` + MobileSAM `vit_t` | see `docs/DECISIONS.md` C1 | fits the 4096 MiB cap |

Why these: LLMDet-large is the best zero-shot open-vocabulary detector in the
Grounding-DINO family that runs locally (LVIS minival 51.1 AP / 45.1 AP-rare
vs 28.8 / 18.8 for the tiny pilot model). SAM 3's tracker path beats SAM 2.1-L
on video propagation (SA-V J&F 84.4 vs 78.4) and is API-compatible with it.
Full reasoning and caveats: `docs/DECISIONS.md` C19.

**The `--revision` flag is mandatory.** An unpinned hub id tracks the model's
default branch, which the `CheckpointSpec` contract refuses (§7.2) — the pins
above are the snapshots actually on disk.

## Stage-by-stage

Every command below is run from the repo root with the `ano_pipe` interpreter
and the cap exported. Outputs land under `work_root` from `configs/paths.yaml`
(`/home/mt/dhakascenes/work`), one directory per stage, cleared before write.

### Stage 3 — open-vocabulary 2D proposals

```bash
$PY pipeline/stage3_proposals/proposals.py \
    --revision bec37f296f05b22f6c6b39bc05a6c611239f4e31 \
    --accept-degraded-upstream
```

Reads the Stage 1 keyframe index, prompts LLMDet-large with the taxonomy
phrases (`configs/taxonomy_pilot_nuscenes.yaml`, order-hashed into every
record), writes per-scene proposal JSONL + `run_manifest.json`. Switch models
with `--model-id IDEA-Research/grounding-dino-tiny --revision <sha>`; per-class
thresholds live in the taxonomy file, not the CLI.

### Stage 4 — box-prompted masks + cross-camera IoA-NMS

```bash
$PY pipeline/stage4_masks/masks.py \
    --revision 3c879f39826c281e95690f02c7821c4de09afae7 \
    --accept-degraded-upstream
# SAM 2.1-L instead:        --model-id facebook/sam2.1-hiera-large --revision 665f8e2ad61cf5f53d65644ff27c8ee525124610
# MobileSAM (pilot tier):   --model-id mobile_sam_vit_t --checkpoint <weights.pt> --revision <file-id>
```

One mask per Stage 3 box, in order, asserted at 1600×900; ego-frame angular
IoA-NMS suppresses cross-camera duplicates. The adapter is chosen from the
model id (`--provider` overrides: `mobile_sam` / `sam2_video` / `sam3_tracker`).

### Stage 5 — 2D→3D lift

```bash
$PY pipeline/stage5_lift/lift.py --accept-degraded-upstream
```

CPU-bound; paints the accumulated clouds through the Stage 4 masks and writes
the paint metrics (the Phase-7 gate number).

## Trial-run pattern (do this before a full run)

One scene into a scratch directory leaves the real work tree untouched:

```bash
$PY pipeline/stage3_proposals/proposals.py --scenes scene-0061 \
    --out-dir /tmp/trial_stage3 --revision <sha> --accept-degraded-upstream
$PY pipeline/stage4_masks/masks.py --stage3-dir /tmp/trial_stage3 \
    --out-dir /tmp/trial_stage4 --revision <sha> --accept-degraded-upstream
```

## Reading the results

- `work_root/<stage>/run_manifest.json` — config, checkpoint id + revision,
  seed, VRAM cap block, totals. The manifest is the record; stdout is not.
- `_SUCCESS` = clean; `_SUCCESS.degraded` = complete but quality-flagged
  (carries the failing scenes); no marker = incomplete, downstream refuses.
- A full-substrate run is 10 scenes / 404 keyframes / 2 424 images; Stage 3 at
  ~350 ms/image is ~15 min, Stage 4 ~10–15 min, Stage 5 a few minutes.

## Standing caveats

- Stage 3's per-class thresholds (default 0.40) were tuned under
  transformers 4.46 numerics and are flagged **unvalidated** after the 5.15.0
  upgrade (fast image processors resize slightly differently) — C19.
- Any number produced on this substrate is plumbing evidence, not label
  quality evidence (banner above; `docs/pilot_plan.md` §13.2).
- The 4 GB claim (C1) still requires pilot-tier models under a 4096 cap on the
  physical laptop; nothing run at 22000 supports it.
