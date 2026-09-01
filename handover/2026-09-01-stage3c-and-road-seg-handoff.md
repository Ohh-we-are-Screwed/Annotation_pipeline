# Handoff — Stage 3c (VLM label check) and the road-surface stage design

**Date:** 2026-09-01 · **Repo:** `/home/mt/Zami/Annotation_pipeline` · **Branch:** `main`
**Nothing is committed.** Continues `2026-08-30-dhaka-pilot-handoff.md`; read that first
for the substrate, the rig geometry and the priors file. This document covers only what
changed on 2026-09-01.

---

## 1. Read this first

Two pieces of work, one shipped and one designed:

1. **Stage 3c** — a new opt-in stage that re-classifies every Stage-3 proposal crop with
   a local vision-language model and rewrites the label where the model confidently
   disagrees. Code is written, tested (21 tests) and wired into `run_stages.sh`.
   **It has never run at scale.** Only a 5-row smoke test against a live server.
2. **A road-surface segmentation stage** — designed, not built. Four design decisions
   are locked (§5), a feasibility spike is done and passed (§6), and a 6-scout design
   workflow was still in flight when this was written (§5.4).

Along the way two findings landed that matter more than either piece of work:
**arm B has never run** (§2) and **Stage 1's ground plane is not the road** (§4).

**Companion document:** `2026-09-01-road-seg-investigation.md` — 1,485 lines of raw
investigation output (7 scout reports + synthesis + adversarial critique) behind §5.
This file summarises it; that file is the evidence.

### Current state

| | |
|---|---|
| Work root | `/home/mt/dhakascenes/work` |
| Dataroot | `/home/mt/dhakascenes/data/pilot_1632` (`v1.0-dhaka-fixed`) |
| Live Stage 3 tree | `stage3_proposals` — 18,416 rows, 37,322 boxes, `_SUCCESS.degraded` |
| `stage3_finetuned` | **MISSING — never built** |
| `stage3_merged` | **MISSING — never built** |
| `stage3_checked` | **MISSING — never built** |
| GPU | RTX 4090 24 GB, idle (302 MiB) at time of writing |
| `handover/` itself | untracked in git, along with `pipeline/stage3c_check/` and `tests/test_stage3c_check.py` |

---

## 2. Arm B has never been used — the CNG/rickshaw vocabulary is not in any output

This was the reported symptom: *"I saw a CNG being labelled as a car."* It is not a
borderline misclassification. **Arm A cannot label a CNG at all.**

Three independent confirmations:

1. `OPT_IN_STEPS=(3b 3f 3m 3c)` — `scripts/run_stages.sh:240`. Steps 3f/3m are accepted
   by name but never run unless typed on the command line, and `all` does not expand to
   them.
2. `stage3_finetuned/` and `stage3_merged/` do not exist under the work root. The last
   full run executed `3b` but not `3f 3m`.
3. Every row in `stage3_proposals` carries
   `checkpoint.model_id = /home/mt/dhakascenes/cache/checkpoints/yolo11x.pt` with the
   COCO→nuScenes class map. That class space contains **no rickshaw and no CNG**, so
   every three-wheeler is forced onto `car` / `truck` / `motorcycle` by construction.

The fine-tune artifact `local_yolox_build/artifacts/yolo11x-rsud20k-armb.pt` (228 MB,
built 2026-08-27) exists and has never been through a completed pipeline run.

**The fix is to include `3f 3m` in the next run.** Nothing is broken; the arm was simply
never invoked. `merge.py`'s vocabulary-authority table then suppresses the arm-A
`car`/`truck` box wherever arm B says three-wheeler.

> **Note on framing.** Arm A and arm B are two branches that *merge*. After 3m there is
> exactly **one** annotated dataset, and that single tree is what flows into 3c, then
> Stage 4, then tracking. 3c is not a per-arm step.

---

## 3. Stage 3c — VLM label check

`pipeline/stage3c_check/check.py` (new) · `tests/test_stage3c_check.py` (new, 21 tests)
Step token `3c`, opt-in. Spec `dhakascenes-pilot/stage3c_check/v1`.

### What it does

Consumes the ONE tree Stage 4 would otherwise read (`stage3_merged` when 3f/3m ran,
else 3b, else 3), re-classifies every proposal crop with Nemotron 3 Nano Omni, and
writes `stage3_checked/` in Stage 3's exact schema and row order plus additive keys —
the Stage 3b trick (C27) used a third time — so Stage 4 consumes it unchanged via
`--stage3-dir`.

### Design decisions, and why

- **The VLM is asked BLIND.** The current label is never in the prompt. Showing a model
  the label to "verify" invites yes-bias; the point is an independent second opinion.
- **The class space never widens here.** The caption must equal the input manifest's
  caption byte-for-byte. Widening is the merge's job (C28).
- **A relabel rewrites three arrays and nothing else** — `class_names`,
  `nuscenes_categories`, `phrase_char_spans` at that index. Geometry, scores, track ids
  and merge ledgers ride through byte-identical (C27).
- **Scores keep arm semantics.** A relabelled box's score is still the ORIGINAL
  detector's confidence in the ORIGINAL class; the VLM emits no calibrated score to
  replace it. This is recorded in the manifest rather than hidden.
- **Boxes under `--min-side-px` (default 32) never reach the model.** A 10 px crop is
  noise and a confident answer on noise is the failure mode this stage exists to remove.
  Skips are recorded per box, not silently.
- **All classes are checked**, not just the vehicle family — operator decision.
  There is no class gate in the code, only the size floor.
- **The GPU rule is enforced in code.** `assert_gpu_exclusive()` (`check.py:291`) runs
  `nvidia-smi --query-compute-apps` and refuses to load the model if anything else is
  computing. Override is `--allow-shared-gpu`, deliberately explicit.

### Audit trail

Every box gets a verdict in the row's `vlm_check` block, one of `confirmed`,
`relabeled`, `unclear`, `skipped_small`, `error`. The run manifest aggregates an
`original -> new` confusion table so one glance shows what the checker actually did.

### Model and serving

| | |
|---|---|
| Model | `unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF`, `UD-Q4_K_XL` |
| GGUF | `/home/mt/dhakascenes/cache/checkpoints/nemotron-omni/…UD-Q4_K_XL.gguf` (23.9 GB) |
| Vision projector | `mmproj-BF16.gguf` (1.6 GB) — same dir |
| Server | `llama.cpp` built with CUDA at `/home/mt/dhakascenes/tools/llama.cpp` |
| Binaries | `build/bin/llama-server`, `build/bin/llama-mtmd-cli` |
| Port | **8092** (8090 was already taken on this host) |
| Fit knob | `--n-cpu-moe 8` — routed experts to system RAM; Mamba/attention/shared expert stay on GPU |

`check.py` starts and stops its own `llama-server`; `--server-url` bypasses that and
talks to one you started yourself.

Two build notes, in case it must be rebuilt: llama.cpp's CMake tries to download a
WebUI `dist.tar.gz` from HuggingFace and the build **fails with a misleading success
message** if that download 404s. A stub `dist` dir was created to get past it; `HF_ENABLED=OFF`
is the cleaner fix.

### Smoke test — the only evidence that exists

5 rows / 49 boxes, live server, real Dhaka frames:

| | |
|---|---|
| Checked | 29 (20 skipped, under the 32 px floor) |
| Confirmed | 17 |
| **Relabeled** | **7** |
| Unclear | 5 |
| Errors | 0 |

Confusion: `a car → an auto rickshaw` ×4, `a truck → an auto rickshaw` ×2,
`a pedestrian → a motorcycle` ×1.

So it found exactly the reported failure — CNGs sitting under `car` and `truck` — and
corrected six of them **from arm A alone**, without arm B having run.

**The seventh is suspect.** `pedestrian → motorcycle` runs against the prompt's explicit
rider rule (*"if the crop is centred on a person, including one riding a vehicle, the
label is the person's"*), and it is the direction that costs a real pedestrian label.
Treat it as a probable false positive and check the audit block after any real run.

### Status

- 21/21 tests pass (`tests/test_stage3c_check.py`); 48/48 across 3c + merge + armB configs.
- **Never run at scale.** The live tree is 18,416 rows / 37,322 boxes → roughly 22k VLM
  calls after the size floor. **No throughput was measured** — the smoke manifest
  recorded no per-scene timing. Time one scene before committing to a full pass.

---

## 4. Stage 1's fitted ground plane is NOT the road — verified

**The frame contract disagrees with itself.** `docs/comprehensive.md:152` fixes the ego
origin at the rear axle projected to **ground**. But `LIDAR_TOP`'s `calibrated_sensor` is
translation `[0,0,0]`, rotation identity — so ego `z=0` sits **at the sensor**, and the
road sits at `z ≈ −2.15 m`. `ingest.py:133` then restricts RANSAC candidates to
`z ∈ [−1.5, +1.5]` with the provenance note *"derived from ISO 8855 ego geometry
(z=0 at ground)"*. The band was chosen under the first reading and is applied under the
second, so **it excludes the road entirely**.

Measured over all 18,416 persisted sector fits in
`stage1_ingestion/scenes/chunk_0000/filter_diagnostics.json`:

| | |
|---|---|
| Plane `d` range | `−0.773 … +0.466`, median `−0.228` |
| Planes at road depth (`d < −1.7`) | **0 of 18,416** |
| Sector fits rejected → global fallback | **14,066 / 18,416 (76.4%)** |

That 76.4% is exactly why Stage 1 carries `_SUCCESS.degraded`.

### This reconciles with the standing diagnosis — it does not contradict it

`2026-08-30-dhaka-pilot-handoff.md` §8.4 already records this same 76.4%:

> **Stage 1 is DEGRADED on 76.4% sector rejection.** This is legitimate — rejected fits
> look like tilt 44.1° at inlier ratio 0.878, i.e. a *confident* fit to a wall or bus
> flank filling the sector. The guard substitutes the global plane. Do **not** loosen
> the thresholds.

**Both accounts are true, and they are the same phenomenon.** A candidate band that
excludes the road forces RANSAC to fit whatever *is* in the band — and at
`z ∈ [−1.5, +1.5]` with the road 2.15 m below, what is in the band is walls and bus
flanks. §8.4 observed the symptom; the band is the cause.

The standing instruction remains correct and should be obeyed: **do not loosen the
thresholds.** Loosening them would *accept* the wall fits. The candidate band is the
thing that is wrong, and it is a different knob.

There is no `DECISIONS.md` entry and no `conformance.yaml` row covering the band/frame
mismatch itself (`conformance.yaml` flags the related height-cap-on-raw-z issue but not
this). **The falsifiable next step:** fix the band, then re-measure the rejection rate.
If it collapses, the band was the cause and §8.4's observation was the symptom. If it
does not, the wall-and-bus-flank diagnosis is independent and the band fix is
insufficient. Either result is a `DECISIONS.md` entry.

### Two consequences

1. **`ground_band_m = 0.30` is carving a ~0.6 m slab through vehicle roofs**, not
   stripping the road. Median plane `d = −0.228` ± 0.30 deletes `z ∈ [−0.53, +0.07]`;
   the road is at `z ≈ −2.15`. So the deleted slab sits **1.62–2.22 m above the road** —
   roof height for cars, rickshaws and CNGs. It removes 12.9% of the fused single-sweep
   cloud and 35.5% of the accumulated cloud.

   **Hypothesis worth measuring, NOT an established finding:** Stage 6 takes
   `height_m = z_max − z_min` from the cluster's own z-extent (`cluster.py:631,636`), so
   deleting points at roof height could truncate `z_max` and bias box heights *down* for
   exactly the indigenous three-wheelers this dataset exists to characterise. **Nobody
   has measured this.** Note the prior handover attributes undersized boxes to a
   different mechanism entirely (~10 lidar points per box, addressed by ZED fusion, §7
   there). Do not assert the causal chain without the measurement — histogram the z of
   ground-band-deleted points, and compare Stage 6 `height_m` per class on a scratch run
   with ground removal disabled.
2. **The road points all survive** into the filtered cloud. ~52% of a sweep lies in the
   true road band `[−2.6, −1.7]`, 89.5% of it from ZED stereo. This is why the road
   stage's 2D→3D lift is viable today without touching Stage 1.

### Carry this into any claim

- Any design that says *"reuse Stage 1's ground plane"* is **wrong on this substrate**.
- Fixing Stage 1 changes `ground_band` membership and therefore every downstream point
  count, `num_lidar_pts`, cluster and box in the run. That is a decision above the road
  stage's pay grade and must be named explicitly, not made in passing.
- Other `z`-referenced constants were chosen under the same ambiguous reading and should
  be audited together: `ransac_candidate_z_band_m`, `ground_band_m`, `height_cap_m 4.0`.

---

## 5. The road-surface segmentation stage — design state

Designed, **not built**. No code exists.

### 5.1 Locked decisions (operator, 2026-09-01)

1. **Purpose: a released dataset layer.** The road surface ships as an annotation
   product alongside the 3D boxes. It therefore needs a human review path and a landing
   place in the release format.
2. **Representation: 2D per-camera mask per keyframe, lifted onto LiDAR point indices**
   the way Stage 5 paints instance masks — yielding nuScenes-lidarseg-style per-point
   surface labels as the 3D product.
3. **Class space: exactly ONE class, "road surface."** Not a multi-class semantic set.
4. **Producer: SAM 3's text prompt.** See §6 — the spike settled which checkpoint and
   which prompt.

A global HD-map layer was ruled out: ego-pose carries ~15–20 m unexplained z drift and
is explicitly *"not a global map."* Everything is keyframe-local.

### 5.2 Hard constraints the design must survive

- **There is no road ground truth of any kind.** `sample_annotation.json`,
  `category.json`, `instance.json`, `attribute.json`, `visibility.json` are all `[]`.
  No `maps/`, no lidarseg, no panoptic. `eval_2d`/`eval_3d` score 0.0 against 0 GT.
  Human review in CVAT is the **only** available route to a quality claim.
- **Stage 4's cross-camera IoA-NMS at 0.5 would delete the road in 5 of 6 cameras.** It
  exists to remove duplicate views of one object; a road genuinely *is* the same surface
  in every camera. The road stage must not route through that contest.
- **Reusing Stage 4 with a synthetic full-image box does not work.** The box would have
  to be appended to Stage 3's `boxes_xyxy_px`, and `build_instances` filters on `kept`,
  not on class — so the road region would be lifted, clustered against a missing prior,
  tracked, inflated and gated as an object. Suppressing it would need a class filter in
  Stages 5, 6, 7, 8, 9 and two exporters. A separate stage is genuinely cleaner.
- **No record type can express a non-instance region.** `AnnotationRecord` requires
  `instance_token`, positive `size_wlh_m` and a quaternion. `GroupAnnotation` is shape
  only, with no producer and no consumer.
- **`export_cvat_3d.py:203`** hard-indexes `label_id[row['class_name']]` with no
  fallback, at severity `fatal` — any new class name reaching the Stage 8 tree aborts
  the chain at publish time, after every GPU stage has already run.
- **Stage 9's `drivable_term` is hard-locked OFF** by a `validate()` error
  (`gate.py:192-196`), recorded as *"§11 decision 7, locked at Phase 10."* Stage 9 is
  not in `run_stages.sh` at all and has never run on this substrate.
- **`release_category_map.yaml`** has 18 classes, all instance objects, no surface slot;
  `export_release.py` refuses any unmapped source string.

### 5.3 Class-space mechanics (already verified as workable)

`phrase_char_spans` are a property of the caption **string**, computed while
concatenating rather than by `str.find`, and the token-level `span_map` is already
nullable — the closed-vocabulary YOLO adapter returns `None`. So a surface phrase would
load, caption and span without modification. What breaks is everything *downstream* of
the phrase, per §5.2.

### 5.4 Where the design investigation lives

The morning workflow (`wf_10826a32-49f`) **completed**: 7 scouts, a synthesis, and an
adversarial critique. Its output is the single most valuable artifact from 2026-09-01
and is **not** reproduced in full here — read it directly (§8 has the command). It
covers publication/export, taxonomy machinery, Stage 4, the downstream chain, QA and
ground truth, and an external model/licence survey, plus a synthesis carrying 16 hard
constraints and a critique carrying 10 gaps and 7 falsified claims.

A second workflow (`road-seg-design`, `wf_47215790-e26`) was launched in the afternoon
to close the same ground with the four decisions already locked. **It had not returned
when this was written** and is largely redundant with the morning run — check it, but
do not block on it.

### 5.5 What the critique found — read before designing anything

The adversarial pass falsified several claims and surfaced gaps that change the design.
The load-bearing ones, each verified:

- **SAM 3's square resize collides with a STANDING prohibition.** `docs/DECISIONS.md:273`
  (C13, Status: STANDING) lists *"Square resizes forbidden"*, and `conformance.yaml` row
  `1.5-r3` claims *"No square resize of non-square imagery anywhere; letterbox or
  resize-shortest-side only"*. SAM 3's cached processor config is
  `size {1008, 1008}, default_to_square: true, mask_size {288, 288}`. This is **not a
  free design choice** — adopting the spike's route as-is either violates C13 or
  falsifies that conformance row. Determine how Stage 4's existing transformers SAM 3
  path is reconciled with C13 today (it appears to lean on `post_process_masks(...,
  original_sizes)` mapping back to 1280×720 rather than avoiding the square resize),
  then either amend C13 with that reading recorded or override the processor to
  letterbox. Either way it is a decision entry, not a config line.
  *Direction note:* 1280×720 → 1008×1008 **stretches vertically 1.40× and compresses
  horizontally 0.79×**. Earlier prose in this project's design notes had this inverted.
- **The lift has no occlusion reasoning, and road is the worst case.** `lift.py:64-65`
  states it outright: *"No occlusion reasoning. A mask is a 2D region; the far wall
  behind a car projects into the car's mask and is claimed by it."* Stage 6's mitigation
  is a keep-largest-cluster filter, which is meaningless for a non-compact surface. So
  every 3D point along the ray through a road pixel — road 30 m away, the underside of a
  bus, a wall past the road edge — lands inside the mask. **The chosen 2D→3D lift needs
  an explicit depth-ordering policy** (accept a point as road only if its range matches
  the ground-surface intersection range for its pixel within a tolerance), or the product
  stops at 2D. Do not inherit Stage 5's behaviour silently.
- **Stage 5's npz is NOT reusable as a shortcut.** Measured on a real file: `uv_px` has
  shape (1397, 2) — pixel coordinates exist only for instance-claimed points, not for the
  72,287 visible ones, and `visible_n_cameras` is a *count*, not a camera identity. A road
  stage cannot skip re-projection; it would run the full 4-hop chain itself.
- **Six of eight cameras cannot support a 3D road claim.** Road is viewed at grazing
  incidence, so pitch error maps to range error with large gain. Six pitches are
  ASSUMED; four cameras have FOV-derived nominal intrinsics with no distortion model and
  principal point assumed at image centre. `CAM_LEFT`/`CAM_RIGHT` are additionally
  motion-blurred in every frame. **Recommendation: scope 3D road to `CAM_FRONT` and
  `CAM_BACK`** — the two ZED-referenced, distortion-modelled channels — unless a
  target-based calibration happens first, and record the rest as a capability gap.
- **The model-role vocabulary is a closed tuple.** `model_interfaces.py:116-121` —
  `ROLES = (embedding_ood, proposal_2d, mask_2d, reid_embedding)`; `register()` raises on
  anything else. A `surface_2d` role needs matching `ROLE_PROTOCOLS` and `ROLE_METHODS`
  entries. **Do not reuse `MaskResult`** — its validator is box-count-shaped
  (*"one mask per box, in order"*), so a stuff payload either carries `n_boxes=1`
  (a semantic lie) or passes validation while meaning something different.
- **A new stage must be registered in 5 places.** `run_stages.sh` at `ALL_STEPS` /
  `OPT_IN_STEPS` / the argument case / the clean-slate directory list / the marker
  summary loop — **plus** `run_pilot.py` (`STAGE_DIRS`, `CHAIN`, `stage_cmd`), or the two
  orchestrators disagree the way they already do about step 9. Omission from the
  clean-slate list leaves a stale tree describing a previous run beside a fresh one.
- **Do not touch the shared phrase taxonomy.** Adding a road phrase makes `priors.py`
  emit a dims/eps_bev block for it, Stage 8 look up a dimension prior, Stage 9 apply a
  2× BEV spatial gate, and `export_release.py` refuse the export — and the CVAT
  exporter's `sorted(set(phrases))` numbering **renumbers existing category ids**
  (measured: `"a road surface"` renumbers 4 of them). Give the road stage its own class
  space.
- **Temporal redundancy is unexploited.** The substrate is one continuous 230 s drive at
  10 Hz — ~0.50 m of ego motion between consecutive keyframes. Consecutive road masks are
  near-duplicates. Segmenting every keyframe independently buys little over striding and
  propagating, and gives no temporal-consistency guarantee — and IoU between consecutive
  road masks is one of the very few quality numbers obtainable **without ground truth**.
- **A free geometric falsifier is being left on the table.** Nothing in the repo computes
  a horizon row. On a mostly-flat road with known pitch and `fy`, road pixels above the
  horizon are wrong by construction — so `road_px_above_horizon` is a self-checking
  degradation cause costing about a dozen lines.
- **Daytime only.** CAM_FRONT luminance sampled across the session is uniformly daytime,
  while `comprehensive.md:293` requires *"night+dark ≥ 25% (differentiator, not token)"*.
  Any prompt or threshold tuned here is unvalidated for the condition the release is
  contractually differentiated on. Record it as a manifest capability gap on day one.

### 5.6 The approach nobody proposed: ZED-stereo-first

The critique's strongest contribution. All three synthesised approaches either ignore
the ZED passes or explicitly exclude them — yet **ZED is the best-characterised ground
signal on this rig**, not the worst:

- Its extrinsic correction was validated *specifically against lidar ground agreement*,
  moving from −0.89/−1.29 m to +0.01/+0.04 m. That is the **only** ground-referenced
  calibration measurement anyone has taken here.
- It contributes ~81k of ~101k points per keyframe at ≤20 m — exactly the band the road
  occupies. On the one keyframe measured, the road band is 38,627 stereo vs 4,511 lidar
  (**89.5% stereo**), so a stereo surface fit is far better conditioned than the 8-sector
  RANSAC currently failing 76.4% of the time.
- A ZED point has a **native pixel correspondence in its co-located camera**, so a road
  surface built from ZED and painted into `CAM_FRONT`/`CAM_BACK` never touches the six
  assumed-pitch cameras and never re-runs the 4-hop projection chain.
- Provenance already rides in the ring column (tags 10/11) — separable for free, no
  schema change, no dataroot re-read.

The shape: fit a ground surface per keyframe from ring-10/11 points inside the co-located
camera's frustum; use it as the **depth-ordering falsifier** for the SAM 3 road mask in
that same camera (closing the occlusion gap above); and publish the agreement statistic
between the two independent estimates as the stage's headline quality number.

**That last part is why it matters.** It is the only proposal that yields a real,
falsifiable quality metric on a substrate with zero ground truth, because it cross-checks
two independent sensing modalities rather than reporting self-consistency. Honest limits:
≤20 m range, front/back only — a near-field two-camera product. On this rig that is what
the calibration actually supports.

### 5.7 The investigation output is in the repo

**`handover/2026-09-01-road-seg-investigation.md`** — all 7 scout reports, the synthesis
(architectural map, 3 approaches, tensions, 16 hard constraints, 12 operator questions)
and the adversarial critique (10 gaps, 7 falsified claims, the missing approach), 1,485
lines. Extracted verbatim from the workflow journal because the session directory it
lived in is not durable.

It is **raw agent output, unedited**. Two reading rules: every claim needs verification,
and the critique section explicitly falsifies claims made in the synthesis above it —
where they disagree, the critique cites measurements, but check them yourself. §5.5 and
§5.6 above are only the load-bearing summary.

Should you need to re-extract it (e.g. for the afternoon run `wf_47215790-e26`):

```bash
# NB: the session id is a literal path component — a glob will NOT expand in a
# bash assignment, so resolve it first.
W=$(ls -d ~/.claude/projects/-home-mt-Zami-Annotation-pipeline/*/subagents/workflows/wf_10826a32-49f | head -1)
$PY -c "
import json, sys
for l in open(sys.argv[1] + '/journal.jsonl'):
    e = json.loads(l)
    if e.get('type') == 'result':
        print(json.dumps(e['result'], indent=1))
" "$W"
```

---

## 6. The SAM 3 road spike — passed

Throwaway probe, **not pipeline code**. Scripts are in the session scratchpad and are
not preserved. It answered one question: does SAM 3's text-prompt path yield a usable
road mask on Dhaka frames? **Yes.**

### Route

Not the raw `sam3` package and **not** the `facebook/sam3.1` multiplex tracker that
Stage 4 drives. The working route is transformers' image model:

```python
from transformers import Sam3Processor, Sam3Model   # transformers 5.15.0
proc(images=img, text="paved road", return_tensors="pt")
proc.post_process_instance_segmentation(out, threshold=0.4, mask_threshold=0.5,
                                        target_sizes=[(H, W)])
```

Two gotchas worth recording:

1. The cached `facebook/sam3` snapshot ships `processor_config.json` for the **video**
   processor (Stage 4's download) and no `preprocessor_config.json`, so
   `Sam3Processor.from_pretrained` fails. Build the pair by hand from the embedded
   `image_processor` block (size 1008, mask_size 288, mean/std 0.5) plus
   `AutoTokenizer.from_pretrained("facebook/sam3")`.
2. `facebook/sam3` is **gated**. The token at `~/.cache/huggingface/token` works, but
   setting `HF_HOME` to the project cache hides it — pass `HF_TOKEN` explicitly. Five
   small tokenizer files (~5 MB) were pulled into the cache on 2026-09-01.

### Results — 60 images across all 8 cameras, ~1,200 keyframes of `chunk_0000`

| Prompt | Fires on | Mean coverage | p10 coverage |
|---|---|---|---|
| `"road"` | 49/60 (82%) | 18.0% | **0.0%** |
| `"paved road"` | **60/60 (100%)** | 22.7% | 12.6% |

**Cost: 88 ms/image → ~27 min for all 18,416 images, at 2.04 GiB peak VRAM.**
Half Stage 4's wall-clock and a twelfth of its memory.

**This measurement supersedes an estimate that was in circulation and wrong.** The
design synthesis costed the stage at ~0.7 s/image → *"~27 min front-only, ~3.6 h
all-8"*, taking 0.7 s from `docs/RUNNING.md:111`'s `~0.7 s (smoke)` — a smoke-test
latency including warmup, not a throughput measurement. The real figure is 8× better,
and 27 min is for **all 8 cameras**, not front-only. Against the measured chain total
(3b 6,131 + 4 3,562 + 5 151 + 6 8,260 + 7 7,909 + 8 5 = 26,018 s ≈ **7.2 h**), an
all-camera road stage is **~6% of the run**, not +50%. Camera scope should therefore be
decided on *calibration quality* (§5.5), not on compute.

### Four findings that shape the design

1. **The prompt must be `"paved road"`, not `"road"`.** Plain `road` returns *zero
   instances* on 11 of 60 images. An empty mask reads downstream as "no road in this
   frame" when it means "the prompt missed" — the exact silent-wrongness shape this
   pipeline refuses elsewhere. This is a load-bearing constant and belongs recorded with
   its rationale, the way `taxonomy_pilot_dhaka.yaml` records why the phrase is
   `"an auto rickshaw"` and not `"a cng"`.
2. **Vehicles are cut out, not included.** The masks have clean holes around cars and
   CNGs; it segments *visible road surface*, not road region. Painting these onto LiDAR
   points will not contaminate vehicle points with a road label.
3. **It is instance grounding, not stuff** — 1–3 regions per image, a divided road
   returning each carriageway separately. The design needs an explicit, recorded union
   rule rather than an assumption of one region.
4. **Different checkpoint from Stage 4**, so the road stage registers its own provider
   rather than reusing Stage 4's `mask_2d` role.

### What the spike did NOT establish

- **No quality measurement.** There is no ground truth; coverage percentages are not
  accuracy. A 22.7% mean coverage says nothing about whether the right 22.7% was chosen.
- 60 images from **one scene**, all daytime. No claim about other captures, night or rain.
- Only 4 prompt phrasings tried, at one threshold pair (0.4 / 0.5). Untuned.
- Nothing was tested about **cross-camera agreement**, or about the lift to LiDAR points.
- **The probe ran straight through the C13 square-resize prohibition** (§5.5) without
  addressing it. It used `Sam3Processor` at its stock `1008×1008, default_to_square:
  true`, relying on `post_process_instance_segmentation(target_sizes=...)` to map back to
  1280×720. The masks shown are the output of that path. Whether that satisfies C13 is
  **undecided**, and the 288×288 internal mask resolution means the road boundary is
  being decided at roughly 2.5 px of native vertical resolution — which matters far more
  for a wide shallow band than for a compact object.
- **Finding 2 does not solve the occlusion problem.** Vehicles being cut out of the *2D*
  mask is necessary but nowhere near sufficient: the lift paints every point along the
  ray, so a hole in the 2D mask does not stop a wall 40 m past the road edge from being
  claimed. See §5.5.

---

## 7. Known limitations — carry these into any claim

1. **Stage 3c has never run at scale.** One 5-row smoke test is the entire evidence base.
   No throughput number exists.
2. **3c's relabels are unmeasured.** 7 relabels on 49 boxes, of which at least one looks
   wrong. There is no ground truth to score against, and no human has reviewed them.
3. **The two arm-B phrase thresholds are still untuned** — `thresholds: {}` with
   `default_threshold: 0.40` in `configs/taxonomy_pilot_dhaka.yaml`. Arm B precision is
   load-bearing because an arm-B false positive **deletes** an arm-A label. The file
   itself records this tuning as owed BEFORE any label-quality claim.
4. **Stage 1 is degraded and the cause is now understood but unfixed** (§4). Everything
   downstream inherits that provenance.
5. **The road stage is a design, not code**, and the design is not finished: the
   occlusion policy, the C13 square-resize collision, and the camera scope are all
   open (§5.5). The SAM 3 spike proves feasibility, not a design.
6. **The road spike is daytime-only** on one scene, and `comprehensive.md:293` requires
   night+dark ≥ 25%. Any prompt or threshold from §6 is unvalidated for that condition.
7. **`paint_metrics.py` — the documented "single strongest correctness signal" — still
   crashes** (`ValueError` at `paint_metrics.py:187`) and writes no file. The pipeline
   has no working geometry-correctness number at all.
8. Everything in this document is **uncommitted**.

---

## 8. Useful commands

```bash
# environment: THIS interpreter, always
PY=/home/mt/miniconda3/envs/ano_pipe/bin/python

# the tests added on 2026-09-01
$PY -m pytest tests/test_stage3c_check.py tests/test_stage3_merge.py tests/test_armb_configs.py -q

# build arm B and merge it in — this is what introduces CNG/rickshaw (§2)
scripts/run_stages.sh 3f 3m

# then the VLM check over the merged tree; time ONE scene first (§3)
scripts/run_stages.sh 3c --scenes chunk_0000

# what 3c actually changed
$PY -c "import json; m=json.load(open('/home/mt/dhakascenes/work/stage3_checked/run_manifest.json')); \
print(json.dumps(m['totals'], indent=1))"

# confirm the GPU is free before any VLM/SAM work
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader

# re-verify the Stage 1 plane finding (§4)
$PY -c "
import json; d=json.load(open('/home/mt/dhakascenes/work/stage1_ingestion/scenes/chunk_0000/filter_diagnostics.json'))
ds=[p['d'] for kf in d['keyframes'] for p in (kf.get('sector_planes') or []) if 'd' in p]
print('n', len(ds), 'min', min(ds), 'max', max(ds), 'below -1.7:', sum(1 for x in ds if x < -1.7))"
```

Weights added on 2026-09-01: Nemotron 3 Nano Omni GGUF + `mmproj-BF16` in
`/home/mt/dhakascenes/cache/checkpoints/nemotron-omni/`. `llama.cpp` (CUDA, arch 89)
built at `/home/mt/dhakascenes/tools/llama.cpp`.

---

## 9. Suggested next steps

1. **Run `3f 3m`.** This is the actual fix for the reported CNG-labelled-as-car problem
   and costs nothing but time (§2). Everything else is secondary to it.
2. **Tune the two arm-B thresholds** on the `tuning` scene subset before any label
   quality claim (§7.3). Note the probe reports the priors/tuning/run partition as
   unsatisfiable with `chunk_0000` unassigned — that has to be resolved first.
3. **Time one scene of 3c**, then decide on the full pass. Read the confusion table and
   the `pedestrian → motorcycle` class of relabel before trusting it (§3).
4. **Decide whether Stage 1's frame bug is in scope** (§4). It is a genuine defect,
   it explains the degraded marker, and fixing it invalidates every downstream artifact.
   The road stage does **not** need it fixed; the object chain might.
5. **Read the morning workflow's synthesis and critique** (§5.7) before writing any road
   design. It is the densest artifact from this day and §5.5/§5.6 are only its summary.
6. **Settle the four open road questions**, in this order — each blocks the design:
   (a) the C13 square-resize collision, which is a `DECISIONS.md` entry, not a config
   line; (b) modal vs amodal road, which decides the occlusion policy and whether the
   CVAT exporter's hole-filling is a bug or a feature; (c) camera scope — the honest
   answer is `CAM_FRONT`/`CAM_BACK` on calibration grounds, and compute is no longer an
   argument (§6); (d) whether the ZED-first cross-check (§5.6) is the design, given it is
   the only route to a falsifiable quality number on a substrate with zero GT.
7. **Cheap, high-value, independent of all the above:** compute the horizon row per
   camera and emit `road_px_above_horizon` (§5.5). About a dozen lines, and one of only
   two or three quality signals obtainable here without ground truth.
8. **Commit something.** `pipeline/stage3c_check/`, `tests/test_stage3c_check.py` and the
   `run_stages.sh` 3c wiring are a coherent, tested unit. The `handover/` directory is
   also untracked.
