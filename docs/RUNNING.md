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

# Hugging Face auth — needed once, for the gated facebook/sam3 repo and (C26)
# facebook/sam3.1; the project account holds both grants:
$PY -m huggingface_hub.cli auth login   # or: <env>/bin/hf auth login
```

`scripts/run_stages.sh` exports the cap from `.env` automatically if you
haven't. If the cap is unset entirely, the run still works but its fit claims
carry `verified: false` (C1 semantics — the physical card is the ceiling).

All checkpoints live in `~/.cache/huggingface/hub` after first download; every
later run is offline.

## The models (C19 2026-08-13; Stage 3 superseded by C23, 2026-08-14)

| Role | Default checkpoint | Pinned revision | Measured on this 4090 |
|---|---|---|---|
| `proposal_2d` (Stage 3) | `yolo11x.pt` (ultralytics) — a FILE, `$YOLO11_CHECKPOINT` | `v8.3.0` (release tag; bytes hashed into the manifest) | 747 MiB fp32, ~31 ms / 1600×900 frame |
| `mask_2d` (Stage 4) | `facebook/sam3` — tracker path (gated; access granted 2026-08-13) | `3c879f39826c281e95690f02c7821c4de09afae7` | 2.1 GiB, ~0.7 s (smoke) |
| `mask_2d` (ungated alternate) | `facebook/sam2.1-hiera-large` | `665f8e2ad61cf5f53d65644ff27c8ee525124610` | 1.5 GiB, ~0.13 s / keyframe |
| `mask_2d` (selectable, C26) | `facebook/sam3.1` — Object Multiplex via `facebookresearch/sam3@8f0b7f4` (NOT transformers; gated, access held) | `daa63191845a41281374e725f4c9e51c7a824460` + ckpt sha256 `0567debe…` | 3.50 GB (3.26 GiB) ckpt, 7 456 MiB peak; adapter mode: 0.78 s / 32-box image, 0.106 s per forward video frame, 0.139 s reverse (harness mode agrees to 0.02 s) |
| `proposal_2d` (open-vocab, retained) | `iSEE-Laboratory/llmdet_large` | `bec37f296f05b22f6c6b39bc05a6c611239f4e31` | 7.5 GiB fp32, ~350 ms / frame |
| pilot fallback | `IDEA-Research/grounding-dino-tiny` + MobileSAM `vit_t` | see `docs/DECISIONS.md` C1 | fits the 4096 MiB cap |

Why these: **YOLO11x + SAM 3** is the pairing since C23 — detection by a
closed-vocabulary COCO-80 detector, segmentation by SAM 3 from those boxes.
Two class-histogram inversions on this substrate (C21) came from the caption
mechanism itself — per-token logits over a concatenated prompt — not from the
checkpoint, and a class assignment that is an argmax over the model's own
trained classes cannot fail that way. SAM 3's tracker path beats SAM 2.1-L on
video propagation (SA-V J&F 84.4 vs 78.4) and is API-compatible with it.
SAM 3.1 (C26) is selectable — `--provider sam31_multiplex`, or Stage 3b's
`TRACK2D_MODEL_ID=facebook/sam3.1` — gated by `scripts/smoke_sam31.py` (both
modes PASS on this 4090); promotion to default waits on the C26 Gate (2) A/B.

**What YOLO11x costs, up front:** four of the taxonomy's ten phrases have no
COCO source — `a road barrier`, `a traffic cone`, `a construction vehicle`,
`a trailer` — 21.3% of this substrate's GT, recall 0 by construction. Stage 3
prints the set before the run and records it as
`class_map.unreachable_phrases`. The open-vocabulary adapter is untouched and
one flag away for anything that needs a novel class:
`--model-id iSEE-Laboratory/llmdet_large --revision <sha>`.

Scores are NOT comparable between the two: records carry
`score_aggregation: yolo_class_confidence` under YOLO and the caption
aggregation under Grounding DINO. A threshold table tuned under one is
meaningless under the other. Full reasoning: `docs/DECISIONS.md` C19 and C23.

**The `--revision` flag is mandatory.** An unpinned hub id tracks the model's
default branch, which the `CheckpointSpec` contract refuses (§7.2) — the pins
above are the snapshots actually on disk. For a local weights file it is the
release tag the file came from; the file's own sha256 is hashed into the
manifest alongside it.

## Stage-by-stage

Every command below is run from the repo root with the `ano_pipe` interpreter
and the cap exported. Outputs land under `work_root` from `configs/paths.yaml`
(`/home/mt/dhakascenes/work`), one directory per stage, cleared before write.

### Stage 3 — 2D proposals (YOLO11x by default, C23)

```bash
$PY pipeline/stage3_proposals/proposals.py \
    --model-id "$YOLO11_CHECKPOINT" --revision v8.3.0 \
    --accept-degraded-upstream
# open-vocabulary instead:  --model-id iSEE-Laboratory/llmdet_large --revision bec37f296f05b22f6c6b39bc05a6c611239f4e31
# pilot tier:               --model-id IDEA-Research/grounding-dino-tiny --revision <sha>
```

Reads the Stage 1 keyframe index, runs the detector over all six ring cameras,
writes per-scene proposal JSONL + `run_manifest.json`. The class space is the
taxonomy's phrase set (`configs/taxonomy_pilot_nuscenes.yaml`, order-hashed
into every record) under BOTH providers — YOLO's COCO ids reach it through
`configs/coco_to_phrase_nuscenes.yaml`, which is asserted total against the
checkpoint's own `model.names` at load, so a non-COCO-80 checkpoint is refused
rather than silently relabelled. Per-class thresholds live in the taxonomy
file, not the CLI.

The provider is inferred from `--model-id` (a basename starting with `yolo` →
the ultralytics adapter) and can be forced with `--provider`. ultralytics ships
weights as a file: `YOLO("yolo11x.pt")` on a missing file downloads into the
CWD, i.e. into the repo tree (§1.8), so the adapter refuses anything that is
not an existing path.

### Stage 3 arm A / arm B — the two-detector proposal design (2026-08-26, IN BUILD)

**Status: arm B is being trained; the merge step does not exist yet.** Stage 3
today is arm A alone, exactly as documented above. Nothing below runs in the
pipeline; this section records the design so the build has a target.

The problem it solves is C25's, from the other side. A closed-vocabulary
detector cannot emit a class its source vocabulary lacks, and COCO has no word
for a cycle rickshaw or a CNG auto-rickshaw — the two most abundant vehicle
types on a Dhaka road. Recall on them is 0 by construction, and unlike the four
C25 phrases this is not fixable by holding them out of a denominator: the
objects have to be *found*.

| | Arm A | Arm B |
|---|---|---|
| Checkpoint | `yolo11x.pt`, ultralytics `v8.3.0` | `dhakascenes/yolo11x-rsud20k-armb` |
| Weights | COCO-80, stock | YOLO11x fine-tuned on RSUD20K |
| Emits | the COCO phrase set, via `coco_to_phrase_nuscenes.yaml` | `rickshaw`, `cng` |
| Status | **frozen** | in build (`local_yolox_build/`) |

**Arm A is frozen, and that is the point.** Every cell in [`Results/`](../Results/)
was measured with stock YOLO11x as `proposal_2d`. Fine-tuning that checkpoint
would invalidate the whole archived ablation and force a re-run; adding a second
arm leaves arm A as the baseline and makes arm B's contribution a measurable
delta against numbers that are already on disk.

**Arm B trains 5 classes and ships 2.** RSUD20K labels thirteen; arm B trains on
`person, rickshaw, cng, car, motorcycle` (85.0% of the corpus's 118,810 boxes)
and emits only `rickshaw` and `cng`. The other three are not padding — they are
the boundary the model must not cross. Training on the shipped pair alone would
turn 32,884 pedestrians, 18,117 cars and 14,801 motorcycles into undifferentiated
background, so the model would learn "not-rickshaw" rather than "car", and
`car ↔ cng` / `person ↔ rickshaw` are precisely the confusions arm B exists to
resolve. Motorcycle earns its place on geometry: among everything COCO knows it
is the nearest shape to a three-wheeler, and it is the largest class a
person/car-only subset would discard.

That matters more than it looks, because **arm B's precision is load-bearing for
the merge**. The authority rule lets an arm B `cng` claim suppress an arm A
`car`; if arm B false-positives on a real car, the rule deletes arm A's correct
label. Training the confusable classes is what keeps that rule safe.

| subset | boxes kept | COCO warm-start rows |
|---|---:|---:|
| rickshaw + cng only | 35,203 (29.6%) | 0 |
| **+ person, car, motorcycle (arm B)** | **101,005 (85.0%)** | **3** |
| all 13 | 118,810 (100%) | 6 |

The subset needs **no derived dataset**: ultralytics' own train-time `classes=`
argument does it in memory. Verified in the installed 8.4.120 source —
`BaseDataset.__init__` calls `update_labels(include_class=classes)`, which drops
label *rows* outside the list while keeping every *image* (one that loses all
its boxes stays in as an explicit negative), and `build.py` plumbs `cfg.classes`
into both the train and val datasets. Ids are **not remapped** — rickshaw stays
1, cng stays 3 — and the filter the run trained with is recorded in the run's
own `args.yaml`, which `evaluate.py` reads back rather than trusting a config on
disk (C25's rule, applied to training). Measured equivalence on val: 6,374 of
7,385 boxes survive, 6 images become negatives, 1,004 of 1,004 images kept.

The ship-time filter is computed **by name, not by index**, in
`scripts/predict_armb.py:ship_indices()` — `[1, 3]` on this corpus — and a
checkpoint missing a shipped name is refused rather than silently emitting the
wrong class. Build tree, configs and recipe:
[`local_yolox_build/`](../local_yolox_build/), plan at
[`docs/superpowers/plans/2026-08-26-rsud20k-yolo11x-finetune.md`](superpowers/plans/2026-08-26-rsud20k-yolo11x-finetune.md).

**The `cng` spelling is a display string, not a phrase.** It lives in exactly
one place — `names:` in `local_yolox_build/configs/rsud20k_yolo11x.yaml` — and
is baked into the checkpoint's `model.names`, which the model never reads. Three
layers, each a table that already exists or mirrors one that does:

```
model.names       "cng"                  ← checkpoint; a lookup id
    │ rsud20k_to_phrase_dhaka.yaml       ← NOT WRITTEN YET; mirrors coco_to_phrase_*
phrase            "an auto rickshaw"     ← what Stage 6 ε, Stage 8 priors, Stage 7
    │                                       matching and the CVAT category id key on
    │ configs/release_category_map.yaml  ← exists; `cng_autorickshaw` already listed
release class     "cng_autorickshaw"     ← what the shipped dataset calls it
```

The phrase stays `an auto rickshaw` and not `a cng` because two consumers read
it as natural language: the review tool's SigLIP suggester builds
`"a photo of {phrase}"` (`review_fix_sam31.py`), and gate S1's text arms prompt
with it. `cng` is Bangladeshi usage a text encoder has not seen; `auto rickshaw`
is web-common. The many-to-one map layer C21 forced for dotted nuScenes
categories is what keeps the two spellings from ever colliding.

**The merge, when it is built.** Run both arms over the same frames, then a
merge step emitting Stage 3's exact schema in Stage 3's row order plus additive
per-box keys — the Stage 3b trick (C27), used a second time, so Stage 4
consumes it unchanged via `--stage3-dir`:

```
stage3_proposals/    arm A, unchanged
stage3_finetuned/    arm B
stage3_merged/       ← merge; Stage 4 points --stage3-dir here
```

Arbitration is a **class-pair table, not a geometric test**, because geometry
cannot separate the two overlap cases: a CNG that arm A calls `car` is one
object with two names, while a rickshaw puller that arm A calls `person` is two
objects with overlapping boxes, and both have the same IoU signature.

| Arm A class on the same region | vs `rickshaw` / `cng` | Why |
|---|---|---|
| `car`, `truck`, `bus` | suppress arm A | COCO has no word for the object; the claim cannot be right at any confidence |
| `motorcycle`, `bicycle` | suppress arm A | same — a three-wheeler forced onto a two-wheeler label |
| `person` | **keep both** | the puller/rider is a separate object, per nuScenes' rider convention |

Suppressed boxes are retained with `suppressed_by`, never dropped — the Stage 4
pattern, so §8.4's reviewer sees what the contest removed. Arbitration is by
*vocabulary authority*, never by score: arm A is confident on its wrong answers
(`car` at 0.85 beats `cng` at 0.55), and a score contest would systematically
pick the wrong label on the most abundant indigenous class, which is C21's
failure rebuilt in a new mechanism.

**What the merge cannot settle, it must not guess.** Metric extent separates a
CNG (~2.7 m) from a car (~4.5 m), but that only exists at Stage 6, and Stage 6's
ε is class-conditional — assume the wrong class, get the wrong ε, and the
cluster manufactures its own confirmation. Contested boxes therefore carry the
pair forward rather than resolving at Stage 3, and the residual falls to the
pre-registered fallback in `comprehensive.md` §7.2: merged-pair pre-labelling,
split by humans at review, taxonomy unchanged.

**Arm B's training labels are 80.7% machine-generated, and that is recorded,
not absorbed.** RSUD20K's `images/train` (18,681) ships as the union of 3,985
human-labelled and 14,696 YOLOv6-M6 pseudo-labelled images, merged, with no
separable directory — from the download alone they are indistinguishable by
filename, index range or box density. The upstream repo's `csv/split-names.csv`
does separate them, so a hashed copy lives in `local_yolox_build/artifacts/`
and `scripts/label_provenance.py` reconciles it against disk:

| | images | instances | boxes/image |
|---|---:|---:|---:|
| human | 3,985 | 22,884 | 5.74 |
| machine (YOLOv6-M6) | 14,696 | 95,926 | 6.53 |
| **union (what arm B trains on)** | **18,681** | **118,810** | **6.36** |

For arm B's two shipped classes: `rickshaw` 4,410 human / 18,284 machine,
`cng` 2,308 human / 10,201 machine.

Three things follow, each worth stating rather than assuming. The union is the
*right* choice — the dataset's own Fig. 6 ablation gives it +5.7 to +7.8 test
mAP over human-only, and the teacher is an in-domain YOLOv6-M6, i.e. self-
training, not the off-the-shelf zero-shot labelling the same paper's Table 3
measures at 9–16% mAP. The machine labels are *denser* than the human ones
(6.53 vs 5.74 boxes/image), which refutes the obvious worry that a confidence
threshold silently dropped small or distant objects. And the label chain for
arm B is **third-generation** — human → YOLOv6-M6 → YOLO11x — while RSUD20K's
own val/test splits are themselves model-seeded and human-*refined* (48 s →
8 s per image), not human-authored. Any claim phrased "measured against human
ground truth" is wrong for this data; "model-seeded, human-refined reference
labels" is accurate.

**Licence.** RSUD20K is **CC BY-NC 4.0 — research and non-commercial only**.
Arm B inherits it, and so do labels arm B produces. This owes a `DECISIONS.md`
entry before any public DhakaScenes release.

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
model id (`--provider` overrides: `mobile_sam` / `sam2_video` / `sam3_tracker` /
`sam31_multiplex`). Since C26 an unrecognised `--model-id` is REFUSED rather than
falling through to `sam2_video`.

**CVAT publishing (C27).** `--no-cvat` now suppresses **both** `cvat` and
`cvat3d`, and it is what keeps `--clean-slate` from purging the server — before
C27, `--clean-slate --no-cvat` still deleted every task and project. Publishing
into a project created before C27 now REFUSES: the Stage 3b attributes
(`source` / `track_id` / `hops`) are declared in `cvat_setup.py`'s label schema,
CVAT only accepts label definitions at project creation, and patching them would
delete annotations. Delete that project in the CVAT UI and re-publish (tasks are
rebuilt from `work_root`; CVAT-side edits inside them are lost, as `--replace`
already loses them), or publish without provenance via
`--accept-missing-attributes`. The answer-key project is unaffected — the GT
export writes no attributes.

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
