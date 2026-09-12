# Running the pipeline

> **This demonstrates pipeline plumbing only. Label quality is not evidence of
> anything; the substrate is nuScenes v1.0-mini, not Dhaka.**

How to actually execute the stages on this machine, with the C19 model tier.
Written 2026-08-13 for the 4090 box and the nuScenes pilot. Companion to
`README.md` (environment setup) and `docs/DECISIONS.md` (why everything is the
way it is).

> **Revision 2026-09-12.** Production has moved to the Blackwell box and to the
> Dhaka substrate, and the chain has grown a step. If you are running the Dhaka
> exports, skip to **["The 2026-09-12 Dhaka chain"](#the-2026-09-12-dhaka-chain-0-1-3-3f-3m-4-5-6s-7-8-release)** —
> it covers step `6s`, its environment knobs, the batch runner and its dashboard,
> the two viewers, the GT-free evaluation, `--from-table` priors, and the gotchas.
> The sections before it are the nuScenes-pilot instructions and still describe
> the 4090 box, including its interpreter path.

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
| **6 cluster** | `pipeline/stage6_cluster/cluster.py` | runnable — in the wrapper chain; the DBSCAN producer |
| **6s stereo box** | `pipeline/stage6_stereo_box/stereo_box.py` | runnable — **opt-in** (2026-09-12), one box per SAM mask from the ZED stereo points it owns; excludes `6` |
| **7 track** | `pipeline/stage7_track/track.py` | runnable — in the wrapper chain |
| **8 inflate** | `pipeline/stage8_inflate/inflate.py` | runnable — in the wrapper chain, boxes from Stage 7 |
| **9 QA + release** | `pipeline/stage9_qa/` + `scripts/export_release.py` | runnable — step `release` runs both |
| road surface | `pipeline/stage_road/` | runnable — step `road` (C33), before `release` |

Stages 3–8 are migrated to `pipeline/common/manifest.py`'s three-state markers
(C16), so they can consume the degraded Stage 1 output — but only when you say
so explicitly (`--accept-degraded-upstream`, recorded in the consumer's own
manifest; the wrapper passes it for you once something upstream is flagged).

Stage 8 takes its boxes from whatever produced them: the wrapper points it at
`stage7_track` when Stage 7 has a marker, and falls back to the Stage 6 producer
otherwise. Since 2026-09-12 that fallback is itself a choice — `stage6_stereo_box`
wins over `stage6_cluster` when it has a marker and is at least as fresh — and
Stage 7 is handed the same choice (`--stage6-dir "$(boxes_dir_for_stage6)"`).
Stages 7, 8 and 9 each record `boxes_source` in their own manifest, so Stage 9
still names `stage6_stereo_box` three stages downstream.

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

### Stage 3 arm A / arm B — the two-detector proposal design (2026-08-26; built 2026-08-27)

**Status: built, opt-in.** `scripts/run_stages.sh 3 3b 3f 3m 4 …` runs the
two-arm chain: `3f` writes `stage3_finetuned/` (the same `proposals.py` driver,
arm B weights + `configs/rsud20k_to_phrase_dhaka.yaml` +
`configs/taxonomy_pilot_dhaka.yaml`), `3m` writes `stage3_merged/`
([`pipeline/stage3_merge/merge.py`](../pipeline/stage3_merge/merge.py)), and
Stage 4 reads the merged tree when it is fresh (`select_stage3_dir_for_4`).
Without `3f`/`3m` nothing changes: arm A alone remains the default until the
two arm B thresholds are tuned — they currently ride the 0.40 default, the same
standing caveat as every other per-class threshold, and arm B's precision is
load-bearing for the merge.

Measured on one scene of the pilot substrate (scene-0061, 234 images,
2026-08-27), which is **plumbing evidence, not label-quality evidence** — there
are no rickshaws in Boston:

| | arm A | arm B | merged |
|---|---:|---:|---:|
| proposals | 1 218 | 3 | 1 220 |
| reachable phrases | 6 | 2 | **8** (union) |
| unreachable phrases | 6 | 10 | **4** |

One arbitration fired, and it is the case the table exists for: arm A `a car`
at **0.768** was suppressed by arm B `an auto rickshaw` at **0.585**, IoU 0.814,
in CAM_BACK_RIGHT. A score contest would have kept the car. On this substrate
the arm B claim is near-certainly the false one — which is the point of
recording the suppressed box in the row's `merge.suppressed_arm_a` ledger
rather than dropping it. Stage 4 then consumed `stage3_merged/` with no code
change (1 220 masks, 1 145 kept after cross-camera IoA-NMS, 0 empty), which is
the C27 claim demonstrated rather than asserted.

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
| Status | **frozen** | built — `local_yolox_build/artifacts/yolo11x-rsud20k-armb.pt` (sha256 + best epoch in `artifacts/armb_provenance.json`) |

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
    │ rsud20k_to_phrase_dhaka.yaml       ← configs/; mirrors coco_to_phrase_*
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
| `bicycle` (C34 protection, opt-in) | **arm B rickshaw wins on overlap (default); C34 protection via `--protect-arm-a`** | C34 (2026-09-02) protected a confident arm A `bicycle` from an overlapping `rickshaw`; C35 (2026-09-12, supersedes C34) reverses this: measured on chunk_0010, 377 of 1,709 surviving bicycles were protected despite an overlapping arm B rickshaw, and a quarter of "bicycle" 3D boxes measured 1.2 m wide (rickshaw width). The default is now unprotected, same as the row above. `--protect-arm-a PHRASE:MIN_SCORE` (e.g. `"a bicycle:0.40"`) restores C34 verbatim; `--no-protect-arm-a` is explicit C28 |
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

---

## The 2026-09-12 Dhaka chain: `0 1 3 3f 3m 4 5 6s 7 8 release`

Everything above this line was written for the nuScenes pilot on the 4090 box.
This section is the chain that runs today on the Dhaka substrate, on the
Blackwell box. Architecture and the *why* of each step:
[`docs/Annotation_pipeline.md`](Annotation_pipeline.md).

```bash
export PY=/home/saif/miniconda3/envs/ano_pipe/bin/python   # see "Gotchas" — the default is wrong
export PYTHONNOUSERSITE=1
export DHAKASCENES_PATHS_CONFIG=configs/paths_zami_20260911.yaml
export DHAKASCENES_SUBSTRATE=dhaka6

STEREO_STRIDE=1 COVERAGE_CONFIG=R3 \
scripts/run_stages.sh 0 1 3 3f 3m 4 5 6s 7 8 release \
    --scenes dhaka_20260911_141259_chunk_0010 --no-cvat
```

### Step `6s` — per-mask stereo boxes on the ZED frusta

`6s` is **opt-in**, like `3f` / `3m` / `3b` / `3c`: it is absent from the default
list and from `all`, and has to be typed. It writes `stage6_stereo_box/` and
`boxes_dir()` — the source Stage 7, Stage 8 and the release all read — prefers it
over `stage6_cluster` once it exists and is at least as fresh. **`6` and `6s` are
mutually exclusive in one run and the wrapper refuses both** (`run_stages.sh --help`).

Standalone, if you want the stage without the wrapper:

```bash
$PY -m pipeline.stage6_stereo_box.stereo_box --accept-degraded-upstream
```

Stage 5 is DEGRADED on every scene of this export
(`ego_motion_between_capture_times_absent`), so the wrapper will be passing
`--accept-degraded-upstream` from Stage 5 onward whether you type it or not.

### Environment knobs for the stereo chain

Declared in `scripts/run_stages.sh --help`; each is empty by default, and empty
means *pass nothing*, i.e. the stage's own default.

| variable | forwarded as | notes |
|---|---|---|
| `STEREO_STRIDE=N` | Stage 1 `--stereo-stride N` | profile default is **8** (it existed to keep Stage 6's DBSCAN alive); **approach A runs 1** |
| `COVERAGE_CONFIG=R1\|R2\|R3` | Stage 1 `--coverage-config` | Stage 1's own default is R2; the stereo chain wants **R3** (the two ZED frusta, 25 m cap) |
| `STEREO_Z_CORR="RING:METRES …"` | one Stage 1 `--stereo-z-correction` per item | space-separated, repeatable. **Unset on this export** — neither ring qualifies |
| `STEREO_PITCH_CORR="RING:DEG:PX:PZ …"` | one Stage 1 `--stereo-pitch-correction` per item | space-separated, repeatable. **Unset on this export** — the front ZED's defect is window-dependent, so no single angle is right (`configs/stereo_box.yaml`) |
| `EXPORT_ROOT=<path>` | export parent | default is this repository's `export/` |
| `EXPORT_NAME=<name>` | export subfolder | default `<dataroot name>_<work name>` |
| `RELEASE_BLOBS=hardlink\|copy\|symlink` | release blob strategy | default `hardlink`; **use `copy` when the export disk is not the dataroot's disk**, which is every batch chunk |

Both stereo corrections are applied by **Stage 1 only**; the box stage never
re-applies them. The *camera-pose* correction in `configs/stereo_box.yaml`
(`camera_pose_pitch_correction`) is a different thing entirely: it is read by the
viewers and the evaluation and by nothing under `pipeline/`.

### Priors — author them, do not transfer them

Stage 6s and Stage 8 **refuse** without `<out_root>/priors/priors_pilot_v0.json`,
and no step in `run_stages.sh` creates it (clean-slate calls it "an INPUT to
Stage 6, not an output of it"). Author it:

```bash
$PY scripts/author_priors_dhaka.py --from-table --paths configs/paths_zami_20260911.yaml
```

`--from-table` needs no template file: the six nuScenes-derived phrases take
population means (source `nuscenes_population_mean_LITERATURE_not_measured`) and
the two indigenous classes take the operator's 2026-09-12 values — **`a rickshaw`
and `an auto rickshaw` l = 2.40 m** (w/h stay at 1.15×1.75 and 1.30×1.75 until
measured), source `operator_stated_2026-09-12_not_measured_on_this_data`.

The file is **bound to its dataroot's metadata fingerprint and refused by any
other**, so one session's priors cannot serve another's. It is idempotent: a file
already bound to the same fingerprint is left byte-identical.

---

## Running the whole batch: `run_all_chunks.py` + `batch_status.py`

38 scenes across four nuScenes roots, numbered globally 1–38 in session order so
one number names one scene for the whole batch.

```bash
# what WOULD run: writes the configs, prints every command, runs nothing
$PY scripts/run_all_chunks.py --dry-run --chunks 1-38

# the real thing
PYTHONNOUSERSITE=1 $PY scripts/run_all_chunks.py --workers 4 --chunks 1-38

# the dashboard, in another window
$PY scripts/batch_status.py --port 8766      # then browse 127.0.0.1:8766
```

`--steps` overrides the step list (default `0 1 3 3f 3m 4 5 6s 7 8 release`);
`--chunks` takes `1-38` or `1,5,7`; `--stagger` is the gap between wrapper starts
(90 s by default — Stage 1 writes ground-filtered clouds to the same HDD for
every chunk, and four of them opening at once is where that disk falls over).

**Isolated roots per chunk.** One generated paths config per chunk gives it its
own `work_root` / `out_root` / `probe_out_root`. That is what makes `--workers 4`
safe: the wrapper takes a `flock` per work root, so two chunks sharing one would
serialise, or overwrite each other's stage trees.

**Resume is the default, at two levels**, because `run_stages.sh` does not resume
on its own — it runs every step it is handed, marker or no marker:

| level | rule | disable with |
|---|---|---|
| chunk | a chunk whose release wrote `boxes/release_meta.json` **and** whose recorded state is `done`/`degraded` is not dispatched at all | `--no-resume` |
| step | a dispatched chunk gets the requested steps **from the first one whose marker is missing onward**. A marker after a hole does not count — a gap upstream makes everything below it suspect | `--no-step-resume` |

`steps_run` and `steps_skipped_by_marker` on each chunk record which was which.
SIGINT/SIGTERM stops dispatching and lets running wrappers finish: they are
started in their own session, so Ctrl-C in this terminal does not reach them.

**Where things live.**

```
<repo>/configs/batch_20260912/chunk_NN.yaml   generated per chunk, git-ignored
<out_root>/priors/priors_pilot_v0.json        authored/rebound per chunk, every time
<ssd>/exports/manifest.json                   the static chunk -> scene map
<ssd>/exports/status.json                     rewritten atomically per event
<ssd>/exports/events.jsonl                    append-only event log
<ssd>/exports/chunk_NN/boxes/                 the release, written by the export itself
<dataroot>/sweeps/                            created if missing — the ONLY write
                                              this program makes into an export
```

The batch running on 2026-09-12 writes to
**`/mnt/exoshdd/dhakascenes_batch_20260912/exports/chunk_NN/boxes/`**, with
`RELEASE_BLOBS=copy` in the chunk overlay so the blobs are **real files on the
operator's disk**, not links into a dataroot on another filesystem. The runner's
per-chunk overlay also sets `STEREO_STRIDE=1`, `COVERAGE_CONFIG=R3`,
`DHAKASCENES_SUBSTRATE=dhaka6`, `EXPORT_ROOT`, `EXPORT_NAME=chunk_NN` and `PY`.

`batch_status.py` serves one page plus `/status.json` (the runner's file,
verbatim) and `/live.json` (status + SSD free/total). It never writes, never runs
anything and never touches a work root, so it is safe to start, kill and restart
mid-batch.

---

## Looking at the results

### 3D viewer — cloud + wireframes + both ZED images

```bash
$PY scripts/view_boxes_3d.py --scene dhaka_20260911_141259_chunk_0010 \
    --out /mnt/hdd/dhakascenes/viewer_zami/chunk_0010 --serve 8767
# --boxes-dir defaults to <work_root>/stage6_stereo_box
# --config defaults to configs/stereo_box.yaml (range cap + image-space pose correction)
```

Read-only, no build step, no npm: one exported JSON per keyframe plus one HTML
file, served by `http.server` on localhost. Orbit controls, points coloured by
ring, class-coloured box wireframes with a heading arrow, both ZED panels with the
boxes projected client-side (so the projection is checkable against the cloud), a
25 m ring for the range cap, a LiDAR ground disc with 5 m rings, per-ring toggles
(**front ZED off by default**), colour-by-height and a per-ring height readout.
Hover shows the row's `stereo` block.

### 2D viewer — six ring cameras per keyframe

```bash
$PY scripts/view_2d.py --scene dhaka_20260911_170051_chunk_0005 \
    --out /mnt/hdd/dhakascenes/viewer_zami/chunk_33_2d --workers 8 --serve 8768
# --proposals-dir default <work_root>/stage3_merged
# --masks-dir     default <work_root>/stage4_masks   ('' disables polygons)
# --boxes-dir     default <work_root>/stage7_track   (absent tree = no outcome badges)
```

Stage 3m proposals (**solid** = arm A, **dashed** = arm B), SAM masks as polygons,
and the 3D outcome per box joined on `(keyframe_token, channel, proposal_index)`:
green `fit`, grey `out_of_r3` (**the expected answer on the four non-ZED cameras,
not a failure**), amber `too_few_stereo` / `beyond_stereo_cap` / `no_points`, red
for anything the page has never seen. `fit` rows also get the box projected back
into its own image through the same helpers the 3D viewer uses.

Everything downstream of Stage 3m is optional: no `stage7_track` tree → no badges,
no wireframes. `--max-width` and `--poly-budget` are the size knobs — a
1,494-keyframe chunk exports 8,964 JPEGs / 657 MB in 87.5 s at 8 threads.

### GT-free evaluation

```bash
$PY scripts/eval_stereo_box.py --scene dhaka_20260911_141259_chunk_0010 \
    --calib-evidence docs/evidence/2026-09-12-stereo-vs-lidar-chunk_0010.json \
    --out-json docs/evidence/2026-09-12-stereo-box-a-chunk_0010-both-frusta.json \
    --out-md   docs/evidence/2026-09-12-stereo-box-a-chunk_0010-both-frusta.md
```

**There is no ground truth on this substrate.** Every number it prints is a
consistency signal, not accuracy — read the distributions and their movement
between runs, never a single value. It reports the status histogram and manifest
totals, clamp rates *and* directions, `depth_source`, yaw-ambiguity reasons,
support and depth distributions, the per-camera pitch residual, and reprojection
IoU (the box's projected silhouette hull against the SAM mask that produced it).

The markdown is rendered from the JSON of the same name — **no number in
`docs/evidence/` is typed by hand** — and the caveat wording follows
`active_channels`, so the doc cannot claim a frustum was dropped when it was not.

---

## Gotchas learned the hard way, 2026-09-12

1. **`PY` defaults to a path that does not exist on this box.**
   `scripts/run_stages.sh:216` is
   `PY="${PY:-/home/mt/miniconda3/envs/ano_pipe/bin/python}"` — the 4090 box's
   user. Several other defaults (`PROPOSAL_MODEL_ID`, `VLM_GGUF`,
   `LLAMA_SERVER_BIN`, `YOLO_CONFIG_DIR`) point at `/home/mt/…` too.
   **Always `export PY=/home/saif/miniconda3/envs/ano_pipe/bin/python`**; the
   batch runner passes it in its overlay for you.

2. **`.env` values override the shell environment inside the batch runner.**
   `run_all_chunks.py`'s precedence is `os.environ < .env < the chunk overlay`
   — deliberately, because `.env` sets `PYTHONNOUSERSITE`,
   `DHAKASCENES_SUBSTRATE` and `DHAKASCENES_PATHS_CONFIG`, every one of which the
   per-chunk overlay must win. The consequence for you: exporting a variable in
   your shell does **not** change what a chunk runs with if `.env` also sets it.
   Edit `.env`, or pass it through the overlay. (This collision killed the first
   chunk of the first launch.)

3. **`prepare_run_exports.py` refuses a work root whose export links point
   elsewhere.** It checks every collision before moving anything, and raises
   `"<path> already points elsewhere: <target>"` for any of `cvat_export`,
   `cvat_export_gt`, `cvat_export_3d`, `cvat_export_3d_double`,
   `cvat_export_road` that is already a symlink to a different destination — and
   `"both <src> and <dst> exist; refusing to overwrite"` when both are real
   directories. It never merges two runs. Point `--export-name` at the run you
   actually mean, or clear the stale symlink.

4. **`pkill -f <pattern>` matches the `pkill` process itself.** Its own command
   line contains the pattern, so a broad `-f` pattern can kill the shell or the
   pipeline that issued it. Use `pgrep -f` first and read the list, or narrow the
   pattern so it cannot match `pkill -f …`.

5. **`run_stages.sh` does not resume.** It runs every step it is handed, marker
   or no marker. Step-level resume is the batch runner's job (above); by hand,
   type the step list you want.

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
