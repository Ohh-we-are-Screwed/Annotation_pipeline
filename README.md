# DhakaScenes Auto-Annotation Pipeline — v1.0.0

**A provenance-enforcing, open-weights, zero-LLM 3D auto-annotation pipeline for
unstructured urban traffic, validated end-to-end on nuScenes v1.0-mini.**

> **CLAIM-HYGIENE BANNER — mandatory, reproduced verbatim in every
> `run_manifest.json` and in the header of every exported figure
> (`docs/pilot_plan.md` §13.2).**
>
> **This demonstrates pipeline plumbing only. Label quality is not evidence of
> anything; models are deliberately under-tier; the substrate is nuScenes
> v1.0-mini, not Dhaka.**
>
> The banner is not ceremonial. Every number in this document is a *descriptive
> statistic over 10 scenes of a stand-in substrate*, produced to show that the
> machinery computes the quantity it claims to compute. None of it is a
> measurement of DhakaScenes label quality, because DhakaScenes has not been
> collected yet.

---

## Release status — what v1.0.0 means

`v1.0.0` freezes the **reference implementation of the annotation pipeline**:
stages 0–9 implemented, stages 0–8 executed end-to-end over the full substrate,
four-cell model ablation completed and archived under [`Results/`](Results/),
evaluation harness written and run, and every design contradiction recorded in a
27-entry decision register. It is the artifact a methods paper is written from.

It does **not** mean the DhakaScenes *dataset* is at v1.0, and it does not
promote any number here to a quality claim. The repository's own self-audit
([`docs/CONFORMANCE.md`](docs/CONFORMANCE.md), 284 rows) still reports 50
`VIOLATES` and 57 `ABSENT` rows against the governing plan, and the 4 GB VRAM
claim (decision C1) still owes one end-to-end run on the physical laptop. Those
are stated in [§12](#12-conformance-ledger--self-audit) and
[§13](#13-limitations-and-threats-to-validity) rather than resolved by
versioning. Freezing the version is a statement that *the implementation is
complete and its gaps are enumerated*, not that the gaps are closed.

---

## Contents

1. [Problem statement](#1-problem-statement)
2. [Contributions](#2-contributions)git remote add origin https://git.zamiulrashid.online/zamiul/Thesis.git
git push -u origin main
3. [System architecture](#3-system-architecture)
4. [The representation contract](#4-the-representation-contract)
5. [Method — stage by stage](#5-method--stage-by-stage)
6. [Models, roles and hardware tiers](#6-models-roles-and-hardware-tiers)
7. [Class space and taxonomy](#7-class-space-and-taxonomy)
8. [Experimental setup](#8-experimental-setup)
9. [Evaluation methodology](#9-evaluation-methodology)
10. [Results](#10-results)
11. [Reproducibility and provenance machinery](#11-reproducibility-and-provenance-machinery)
12. [Conformance ledger — self-audit](#12-conformance-ledger--self-audit)
13. [Limitations and threats to validity](#13-limitations-and-threats-to-validity)
14. [Decision register](#14-decision-register)
15. [Repository layout](#15-repository-layout)
16. [Installation and running](#16-installation-and-running)
17. [Roadmap to DhakaScenes v1](#17-roadmap-to-dhakascenes-v1)
18. [Licence, citation, provenance of documents](#18-licence-citation-provenance-of-documents)
- [Appendix A — metric definitions](#appendix-a--metric-definitions)
- [Appendix B — glossary](#appendix-b--glossary)

---

## 1. Problem statement

### 1.1 The dataset gap

Every large 3D driving benchmark — KITTI, nuScenes, Waymo Open, Argoverse 2, ZOD
— was collected in *structured* traffic: lane discipline, homogeneous vehicle
classes, low agent density. South Asian urban traffic violates all three
assumptions simultaneously. Dhaka's road population is dominated by vehicle
types that are **absent from every existing 3D taxonomy** (CNG auto-rickshaws,
battery rickshaws, cycle rickshaws, tempos, human haulers), moving in densities
where inter-agent gaps fall below the spatial resolution most detection stacks
assume.

`DhakaScenes` (planned; see [`docs/comprehensive.md`](docs/comprehensive.md)) is
a multimodal driving dataset targeting that gap: synchronized 360° solid-state
LiDAR, a global-shutter camera ring, and centimetre-class PPK ground-truth
trajectory, with 3D boxes and tracks over an indigenous-vehicle taxonomy, plus
**density-stratified evaluation** — every headline metric reported across
measured traffic-density bins rather than as a single whole-set average.

### 1.2 The annotation gap

A 3D dataset is bounded by its annotation cost, not its capture cost. Manual 3D
cuboid annotation runs at roughly one object-track per several minutes of
annotator time; at Dhaka densities (tens of agents per keyframe) a manually
annotated release is not reachable on an academic budget.

The standard answer is auto-annotation with human verification. The standard
*failure* of auto-annotation is that it produces output that is **wrong and
plausible at the same time**: a swapped channel order, a re-applied extrinsic, a
square resize with a uniform inverse, a phrase-span off-by-one — each of these
yields boxes that look like boxes, masks that look like masks, and metrics that
look like metrics, with no crash anywhere in the chain. A pipeline that fails
loudly is a nuisance; a pipeline that fails silently poisons a dataset.

### 1.3 What this repository is

This repository is the **annotation pipeline itself**, built and validated ahead
of capture, on nuScenes v1.0-mini as a stand-in substrate. nuScenes is the right
stand-in for one specific reason: it ships **human 3D ground truth**, so every
stage of the machinery can be scored against an answer key that the pipeline
never saw. What it *cannot* validate is anything about Dhaka — the substrate is
Boston and Singapore, structured traffic, a 32-beam spinning LiDAR rather than a
non-repetitive solid-state one.

The engineering thesis being tested here is therefore not "can a pipeline
produce labels" (obviously yes) but:

> **Can an auto-annotation pipeline be built so that every silent failure mode
> is converted into either an impossibility, a loud refusal, or a recorded
> number?**

Everything below is organised around that question.

---

## 2. Contributions

Each is falsifiable and mapped to an artifact in this repository.

| # | Contribution | Evidence in this repo |
|---|---|---|
| **A** | A ten-stage annotation pipeline (2D proposal → segmentation → 3D lift → clustering → tracking → amodal inflation → QA gating) implemented against **explicit interface contracts**, where every cross-stage record is schema-validated on write *and* on read-back | [`pipeline/common/schemas.py`](pipeline/common/schemas.py) (I-1…I-5 + raising write boundary); [§4](#4-the-representation-contract) |
| **B** | A **single-source-of-truth representation contract** — one frame convention, one time base, one projection chain, one evaluation-region definition — enforced by construction rather than by review | [`pipeline/common/conventions.py`](pipeline/common/conventions.py), [`pipeline/common/eval_region.py`](pipeline/common/eval_region.py) |
| **C** | A **role/provider registry** that makes model tier a configuration choice rather than a rewrite, including temporal state in the mask role's signature before any temporal provider existed | [`pipeline/common/model_interfaces.py`](pipeline/common/model_interfaces.py); [§6](#6-models-roles-and-hardware-tiers) |
| **D** | **Three-state completion semantics** (`_SUCCESS` / `_SUCCESS.degraded` / absent) that distinguish *incomplete output* from *quality-flagged output*, with degraded consumption possible only under an explicitly recorded opt-in | [`pipeline/common/manifest.py`](pipeline/common/manifest.py); decision C16 |
| **E** | **Detector-reachability-aware evaluation**: ground-truth classes the configured detector cannot emit are held out of the recall denominator, counted and named, derived from the *run's own manifest* rather than the config on disk | [`pipeline/common/class_space.py`](pipeline/common/class_space.py); decision C25 |
| **F** | A **12 Hz identity-propagation stage** (Stage 3b) that recovers detector misses between 2 Hz keyframes via video-tracker propagation, emitting recovered boxes with per-box provenance (`source`, `track_id`, `hops`, decayed score) while keeping every pipeline output at 2 Hz | [`pipeline/stage3b_track2d/track2d.py`](pipeline/stage3b_track2d/track2d.py); decision C27 |
| **G** | A **two-detector Stage 3**: a COCO-frozen arm A plus an RSUD20K fine-tuned arm B that emits the indigenous classes COCO cannot name, merged by **vocabulary authority rather than score**, with every suppressed box retained in an auditable per-row ledger | [`pipeline/stage3_merge/merge.py`](pipeline/stage3_merge/merge.py); decision C28 |
| **H** | A **conformance ledger**: 284 machine-validated rows mapping every checkable assertion of the governing plan to evidence about this repository, with rules that make a tidy-looking but hollow ledger fail validation | [`docs/conformance.yaml`](docs/conformance.yaml), [`scripts/check_conformance.py`](scripts/check_conformance.py) |
| **I** | A **contradiction register** (27 entries) recording every conflict between plan, machine and code — including the ones that make the project look worse | [`docs/DECISIONS.md`](docs/DECISIONS.md) |
| **J** | An end-to-end **model ablation** over the proposal × re-ID cross product, each cell a clean-slate rebuild, with model identity read from each run's own manifests | [`Results/`](Results/), [`scripts/run_matrix.sh`](scripts/run_matrix.sh); [§10](#10-results) |

---

## 3. System architecture

### 3.1 Position within the DhakaScenes system

The full system (`docs/comprehensive.md` §2.1) spans capture → trajectory →
ingestion → annotation → human verification → release → benchmark. **This
repository implements the annotation subsystem and its evaluation**, i.e. the
segment between interface contracts I-3 and I-5, plus a stand-in for I-1/I-2
sourced from nuScenes.

```
 [I-1] session bundle          ── stand-in: nuScenes sample_data via Stage 0 allowlist
 [I-2] ego_pose + quality      ── stand-in: nuScenes ego_pose (quality fields = None, never 0.0)
        │
        ▼
 [I-3] keyframe index + clouds + images + poses        ← Stage 1 produces this
        │
        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  ANNOTATION PIPELINE  (this repository, stages 2–8)          │
 └──────────────────────────────────────────────────────────────┘
        │
        ▼
 [I-4] prelabels.jsonl → CVAT import                   ← Stage 9 produces this
        │
        ▼
 [I-5] human-verified labels export                    ← CVAT review loop
```

### 3.2 Stage dataflow

```
                        configs/paths.yaml  ──►  Paths (frozen, validated once)
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ STAGE 0 — SUBSTRATE PROBE                          no GPU, no devkit     │
  │   six named predicates: version_matches · channels_complete ·            │
  │   files_resolve · files_parse · sweeps_cover_window · token_graph_closed │
  │   ──► usable_scenes.json (fingerprint-bound allowlist) + scene partition │
  └───────────────────────────────┬──────────────────────────────────────────┘
                                  ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ STAGE 1 — INGESTION                                no GPU                │
  │   sensor→ego hop applied EXACTLY ONCE · ego-motion-compensated           │
  │   accumulation (W_acc sweeps) · sector RANSAC ground plane FIT on the    │
  │   accumulation, APPLIED to the single sweep · per-filter per-sector      │
  │   point ledger                                                          │
  │   ──► [I-3] keyframes + single-sweep & accumulated clouds (ego frame)    │
  └───────┬─────────────────────────────────────────────────────┬───────────┘
          │                                                     │
          │                             ┌───────────────────────▼───────────┐
          │                             │ STAGE 2 — OOD / LONG-TAIL BRANCH  │
          │                             │  DINOv2 embeddings → HDBSCAN      │
          │                             │  GLOSH on RAW embeddings          │
          │                             │  (UMAP is visualisation ONLY)     │
          │                             │  feeds nothing downstream         │
          │                             └───────────────────────────────────┘
          ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ STAGE 3 — 2D PROPOSALS      6 ring cameras, 1600×900, xyxy abs px        │
  │   default: YOLO11x (closed-vocab COCO-80) → phrase via class map         │
  │   retained: Grounding-DINO / LLMDet (open-vocab, phrase = prompt)        │
  │   ──► proposals.jsonl  (+ class_map.unreachable_phrases in manifest)     │
  └───────┬──────────────────────────────────────────────────────────────────┘
          │            ┌─────────────────────────────────────────────────┐
          ├───────────►│ STAGE 3b — 12 Hz IDENTITY PROPAGATION (opt-in)  │
          │            │  Phase A: re-run Stage 3's own detector on the   │
          │            │           sweep frames (models never co-resident)│
          │            │  Phase B: SAM-video propagation, ≤16-frame       │
          │            │           window, state re-init at boundaries    │
          │            │  ──► Stage 3 schema + {source, track_id, hops}   │
          │            │      every row lands AT A KEYFRAME (2 Hz output) │
          │            └───────────────────┬─────────────────────────────┘
          ▼                                ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ STAGE 4 — BOX-PROMPTED MASKS + CROSS-CAMERA IoA-NMS                      │
  │   SAM 3 / SAM 2.1-L / SAM 3.1 multiplex / MobileSAM (pilot tier)         │
  │   masks asserted at 1600×900 · one mask per box, same order              │
  │   IoA computed in EGO (azimuth, elevation) footprints — NOT in pixels    │
  │   ──► masks.jsonl (kept + suppressed, contest recorded)                  │
  └───────┬──────────────────────────────────────────────────────────────────┘
          ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ STAGE 5 — 2D→3D LIFT                               CPU                   │
  │   four-hop chain (conventions.project_lidar_to_image, one impl.)         │
  │   four guards: z≤0 cull · near-zero depth · out-of-bounds · multi-camera │
  │   contest by nearest principal axis (ties → fixed camera priority)       │
  │   points KEEP their Stage 1 ego coordinates — the lift labels, not moves │
  │   ──► painted.jsonl (per-instance point subsets) + paint diagnostics     │
  └───────┬──────────────────────────────────────────────────────────────────┘
          ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ STAGE 6 — PER-INSTANCE BEV CLUSTERING + L-SHAPE FIT                      │
  │   DBSCAN per MASK INSTANCE (ε from priors, class-conditional)            │
  │   hand-rolled + canonically ordered → byte-identical across runs         │
  │   keep-largest-cluster = reprojection-ghost filter (count recorded)      │
  │   near-square policy → yaw_ambiguous · yaw re-asserted against conventions│
  │   ──► boxes.jsonl  (size in nuScenes [w,l,h], yaw about +z from +x)      │
  └───────┬──────────────────────────────────────────────────────────────────┘
          ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ STAGE 7 — PREDICT-THEN-MATCH TRACKING                                    │
  │   Kalman [x,y,z,vx,vy,vz,yaw] propagated to t BEFORE IoU is computed     │
  │   (2 Hz ⇒ raw-detection IoU is ~0 for moving vehicles)                   │
  │   Hungarian over iou·cos(DINOv2/v3 re-ID) — appearance_trusted per pair  │
  │   ICP in nuscenes_global (absolute, mAVE-comparable) · Kalman fallback   │
  │   when num_lidar_pts < 15 · yaw-consistency enforcement along tracks     │
  │   ──► tracks.jsonl (velocity_semantics stated per row)                   │
  └───────┬──────────────────────────────────────────────────────────────────┘
          ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ STAGE 8 — PRIOR-ANCHORED AMODAL INFLATION                                │
  │   hold the NEAR face still, grow the FAR face (never "shift outward")    │
  │   three guards: near face immobile · far face moves away · box may not   │
  │   enclose the sensor · height anchors the TOP (ground band removed below)│
  │   every box records inflated + inflation_fraction + anchor per axis      │
  │   ──► inflated.jsonl (PRE-inflation box retained as box_measured)        │
  └───────┬──────────────────────────────────────────────────────────────────┘
          ▼
  ┌──────────────────────────────────────────────────────────────────────────┐
  │ STAGE 9 — QA GATING → [I-4]                                              │
  │   gate vector → label tier → prelabels.jsonl through the raising         │
  │   schema boundary (tier ≠ source rules, num_lidar_pts_basis check)       │
  └───────┬──────────────────────────────────────────────────────────────────┘
          ▼
     CVAT REVIEW  ── two SEPARATE projects: "— OUR PIPELINE output" and
                     "— nuScenes HUMAN answer key" (never merged, never mixed)
          ▼
     EVALUATION   ── eval_2d · eval_3d · paint_metrics · compare_runs
```

### 3.3 Interface contracts

Every arrow above is a named, versioned contract. A stage may change internally
at will; it may not change its contract without a version bump. The record
definitions live in one file, [`pipeline/common/schemas.py`](pipeline/common/schemas.py),
and are validated **on write and again on read-back** — serialization through
plain dicts was the open path by which unvalidated records reached downstream
consumers.

| Contract | Producer → Consumer | Record |
|---|---|---|
| I-1 | substrate → Stage 0 | session/sample_data bundle (stand-in: nuScenes tables, allowlisted) |
| I-2 | trajectory → Stage 1 | `ego_pose` + quality fields (**absent quality is `None`, never `0.0`**) |
| I-3 | Stage 1 → Stages 2–5 | keyframe index, single-sweep + accumulated clouds, poses |
| I-4 | Stage 9 → CVAT | `prelabels.jsonl` — annotation records with tier, gate vector, provenance |
| I-5 | CVAT → release builder | human-verified label export |

**Rule that gives I-2 its teeth:** nuScenes supplies none of I-2's quality
fields. Writing `sigma_pos_m = 0.0` to satisfy a dataclass would make every
consumer read the stand-in pose as *better than PPK*. So absent quality is
`None`, and consumers fail closed through `require_*()` accessors rather than
defaulting.

---

## 4. The representation contract

Six conventions are defined exactly once and consumed everywhere. Each exists
because the alternative — every stage re-deriving it — produced, or would
produce, a specific silent failure.

### 4.1 Frames (§1.1)

Canonical frame for all cross-stage geometry is **ego, ISO 8855**: x forward,
y left, z up, yaw about +z from +x, radians. Rules:

1. Every geometric record carries an explicit `frame`. A record with no frame is
   **invalid**, not defaulted.
2. The sensor→ego hop is applied **exactly once**, in Stage 1, and nowhere else.
   `LIDAR_TOP`'s extrinsic is a ≈ −89.9° yaw; a second application rotates the
   world by another 90° and leaves every downstream value plausible.
3. Calibration is **per scene, never a global constant** (decision C14, verified
   independently twice): `CAM_FRONT` has two distinct intrinsics across the 10
   scenes (fx 1266.417 ×6, 1252.813 ×4) and `LIDAR_TOP` two distinct yaws
   (−89.883° ×6, −90.031° ×4). Nothing may cache one scene's K or `T_ego_lidar`
   for another.

### 4.2 Time (§1.2)

nuScenes ships microseconds (Unix epoch); the pipeline stores **int64
nanoseconds**. `conventions.to_unix_ns()` is the only function in the codebase
that multiplies or divides a timestamp. Every `sample_data` record carries its
**own** `ego_pose`: within one keyframe the LiDAR and a given camera are captured
up to ~48 ms apart, so the two middle hops of the projection chain are
load-bearing, not ceremony. `CAM_BACK_LEFT` can fire up to +1.20 ms *after* the
LiDAR anchor — "cameras fire before the anchor" is a tendency, not an invariant.

### 4.3 Projection (§1.3)

`conventions.project_lidar_to_image()` is the **single implementation** of the
four-hop chain `ego(t_lidar) → global → ego(t_cam) → camera → pixel`. Stages call
it; they never compose their own transforms. nuScenes `camera_intrinsic` carries
no distortion coefficients (the imagery is rectified), so undistortion is a no-op
on this substrate — recorded as such, never as evidence that the distortion path
was tested.

### 4.4 The 2D contract (§1.5)

Every 2D quantity crossing a stage boundary is **absolute pixels at 1600×900,
`xyxy`** — never `cxcywh`, never normalised. Each model adapter owns both
directions of its own transform; no transform leaks into stage code. Enforced in
the adapter layer, not by review:

- a non-aspect-preserving (square) resize with a uniform inverse is **rejected by
  `validate()`** — it stretches every box along one axis, every mask still looks
  like a mask, and every 3D extent is wrong in one axis, consistently;
- the YOLO path letterboxes (padding is legitimate there) and asserts instead
  that the letterbox scale is **uniform**;
- channel order is the adapter's job: Ultralytics reads a numpy source as BGR and
  swaps it itself, so passing RGB straight through would send every image to the
  network with red and blue exchanged — detections still appear, scores still
  look like scores, recall is quietly worse everywhere.

### 4.5 Determinism (§1.9)

Two runs on the same input must produce byte-identical output.

- **RANSAC** (Stage 1) is the first geometric operation in the pipeline and is
  stochastic; unseeded, it yields a different ground plane, different retained
  points, different clusters and different boxes on every run. The RNG is seeded
  per `(global seed, keyframe token, sector)`, so the stream does not depend on
  processing order, and the derivation is recorded.
- **DBSCAN** (Stage 6) label assignment depends on point arrival order, which
  comes from a file read. Points are placed in canonical lexicographic order
  *before* clustering, neighbour queries return sorted indices, and equal-size
  clusters are broken by lowest mean range, then lowest canonical index. DBSCAN
  is implemented in-repo (≈40 lines) rather than imported, precisely so those
  ordering guarantees exist.

### 4.6 Completion semantics (§1.9, decision C16)

Three states, not two:

| Marker | Meaning | Downstream |
|---|---|---|
| `_SUCCESS` | complete, clean | consumable |
| `_SUCCESS.degraded` | complete, quality-flagged; **carries the causes** | consumable **only** under an explicit `--accept-degraded-upstream`, recorded in the consumer's own manifest |
| *(absent)* | incomplete | refused, unconditionally |

This exists because of a real failure: Stage 1 ran to completion — 404/404
keyframes written, validated, atomic — and then withheld `_SUCCESS` because two
scenes crossed a sector-rejection threshold whose own provenance reads
"arbitrary, needs tuning". Every downstream stage hard-refused, so a *quality*
signal presented as an *incompleteness* signal read as a broken pipeline.
`clear_markers()` additionally runs before a stage writes its first byte, so a
mid-run refusal cannot leave the previous run's `_SUCCESS` standing over a
partially rewritten tree.

### 4.7 Evaluation region E and density ρ (§1.10)

`eval_region.in_region()` is the only place that decides whether a point is
inside E, and `rho()` the only place that computes density; both derive from a
`RegionSpec` built from `coverage_config`, and neither contains a constant angle
or radius of its own. R1 is a 110° frontal wedge; **R2 (all six ring cameras) is
the pilot's locked choice**. The same box list yields different densities under
each, and ρ is *count over area* precisely so the two remain comparable. A stage
that filtered with its own hardcoded `abs(theta) < 0.96` would publish a number
that is no longer the dataset's ρ and looks identical in the output file.

ρ is the statistic the DhakaScenes paper's **density-stratified evaluation**
contribution is built on, which is why its definition is centralised here rather
than at the benchmark layer.

---

## 5. Method — stage by stage

Each stage below lists what it consumes, what it produces, the method, and — the
part that matters for the paper — **the silent failure it is built to prevent**.

### Stage 0 — Substrate probe

*No GPU, no models, no `nuscenes-devkit`: stdlib only. Verifying the substrate
with the library that assumes the substrate is circular.*

| | |
|---|---|
| **In** | dataroot, `configs/paths.yaml` |
| **Out** | `usable_scenes.json` (the only scene list any later stage reads), `probe_report.json`, the stratified scene partition |
| **Method** | six named, individually reportable predicates: `version_matches`, `channels_complete`, `files_resolve`, `files_parse` (`size % 20 == 0` and a sane band for `.pcd.bin`; SOI/EOI markers and a sane band for JPEG), `sweeps_cover_window`, `token_graph_closed` |
| **Prevents** | a probe that checks file existence passes trivially on a complete substrate and catches nothing. A scene failing any predicate is excluded **loudly with the failing predicate named**, never repaired, substituted, or quietly skipped at Stage 5 |

`usable_scenes.json` carries the dataroot realpath, the metadata fingerprint, the
required-channel set and the `W_acc` actually used — each of which changes the
answer — so every downstream artifact is bound to the exact substrate bytes it
was computed from.

### Stage 1 — Ingestion, ground removal, filter ledger

| | |
|---|---|
| **In** | Stage 0 allowlist |
| **Out** | [I-3]: keyframe index; **two** clouds per keyframe (single-sweep and ego-motion-compensated accumulation); per-filter, per-sector point counts |
| **Method** | sensor→ego hop applied once; accumulation compensated using **each sweep's own `ego_pose`**; sector RANSAC ground plane **fit on the accumulation, applied to the single sweep** |
| **Prevents** | (a) double-applied extrinsics; (b) lifting from a smeared cloud — an accumulated cloud is ego-motion compensated *only*, so dynamic objects smear across the window; (c) a silently loosened point gate — fitting *and* lifting on the accumulated cloud loosens the "≥ 5 returns" gate by roughly the accumulation factor (~10×) while still appearing to fire |

The **per-filter, per-sector point ledger** (input → post-ground → post-range →
post-height, both clouds) is a primary output rather than a log line: without it
there is no way to distinguish "few objects because detection is bad" from "few
objects because filtering deleted them". With a 0.3 m ground band on a 32-beam
LiDAR, the second is a live possibility — that band removes all wheel returns
from every vehicle and most of a traffic cone's body.

**Parameter-provenance warning carried in the config and in every output:** 0.3 m
/ 40 m / 4 m were chosen for a Livox Mid-360 on Dhaka roads and are applied here
to a 32-beam spinning LiDAR on Boston and Singapore roads. Nothing corrects for
that; the diagnostics are what make it measurable.

### Stage 2 — OOD / long-tail discovery *(branch; feeds nothing)*

| | |
|---|---|
| **In** | Stage 1 keyframes |
| **Out** | per-image embedding rows, cluster labels, `is_ood` flags, 2-D coordinates for plotting |
| **Method** | DINOv2 image embeddings → HDBSCAN; **the outlier call is GLOSH (`outlier_scores_`) computed on the RAW 384-D embeddings**; UMAP output is written for visualisation and **never read by the flagging decision** |
| **Prevents** | the standard conflation. HDBSCAN membership in a UMAP projection has no formal relationship to outlyingness in the embedding space UMAP preserves neither density nor global structure of. `flag_ood()` takes precomputed scores and a threshold and has **no UMAP coordinate in its argument list** — that signature is the enforcement mechanism, not a comment promising one |

Sampling is over **images, not keyframes**: 1-in-10 over 404 keyframes gives 40
samples, on which HDBSCAN returns one cluster plus noise and any "not zero, not
thousands" success criterion is unfalsifiable. 404 keyframes × 6 cameras ≈ 2,424
images; 1-in-10 over *that* flat list gives ≈242. Degenerate-N is a declared
failure mode: `n_neighbors` / `min_cluster_size` are clamped when the sample is
smaller, the clamp is recorded, and the run degrades rather than raising.

This stage is deliberately a **branch**: nuScenes' taxonomy is already known, so
there is no confirmed-novel-label loop for it to feed. Running it proves the
embedding → projection → clustering path end to end and nothing else. On
DhakaScenes it becomes the taxonomy-discovery front-end.

### Stage 3 — 2D proposals

| | |
|---|---|
| **In** | Stage 1 keyframes, six ring cameras at 1600×900 |
| **Out** | `proposals.jsonl` — boxes in absolute `xyxy` pixels, labelled with **taxonomy phrases**; `run_manifest.json` recording the class map, its sha256, and `class_map.unreachable_phrases` |
| **Providers** | **YOLO11x** (default since C23; closed-vocabulary COCO-80, phrase via `configs/coco_to_phrase_nuscenes.yaml`) · **LLMDet-large / Grounding-DINO** (open-vocabulary, phrase *is* the prompt) · Grounding-DINO-tiny (4 GB pilot tier) |
| **Prevents** | five distinct silent failures, below |

1. **Dotted category names as prompts.** `vehicle.emergency.ambulance` tokenises
   to nonsense and returns near-random but *non-empty* detections. Prompt strings
   come from the mapping table always, and `PromptConfig` rejects a dotted string
   by pattern.
2. **Phrase-span bookkeeping** (the sharpest one). Grounding DINO emits per-token
   logits over the *concatenated* caption; recovering which phrase a box belongs
   to is token-span arithmetic, and multi-word classes make it error-prone. A
   span bug yields well-placed boxes with the **wrong label**, which then selects
   the wrong DBSCAN ε, the wrong dimension prior and the wrong inflation target —
   with every downstream stage running perfectly. The adapter builds the mapping
   from the tokenizer's own character offsets and asserts it is **total** before
   a single image runs; `build_phrase_span_map()` is pure and testable without a
   GPU or a network.
3. **Square resize** — see [§4.4](#44-the-2d-contract-15).
4. **Channel order** — see [§4.4](#44-the-2d-contract-15).
5. **A COCO name reaching the record.** `person` and `motorcycle` are not
   `a pedestrian` and `a motorcycle`; every downstream lookup keyed on the phrase
   would miss and fall back to a default. The class map is asserted **total
   against the checkpoint's own `model.names` at load**, so a checkpoint that is
   not COCO-80 is refused rather than silently relabelling every box.

**Scores are not comparable across providers.** Records carry
`score_aggregation: yolo_class_confidence` under YOLO and the caption aggregation
under Grounding DINO. A threshold table tuned under one is meaningless under the
other, and the field says so in every row.

### Stage 3f / 3m — arm B and the two-detector merge *(opt-in; decision C28)*

A closed-vocabulary detector cannot emit a class its source vocabulary lacks, and
COCO has no word for a cycle rickshaw or a CNG auto-rickshaw — the two most
abundant vehicle types on a Dhaka road. Unlike C25's four unreachable phrases,
this is not fixable by holding them out of a denominator: the objects have to be
*found*.

| | | |
|---|---|---|
| **In** | arm A's tree (Stage 3 or Stage 3b) + arm B's own Stage 3 tree | |
| **Out** | Stage 3's **exact schema in Stage 3's row order**, plus additive keys, in `stage3_merged/`. Stage 4 consumes it unchanged via `--stage3-dir` | |
| **Method** | `3f` runs the **unmodified** Stage 3 driver with the RSUD20K fine-tune, a superset taxonomy and its own class map, into its own tree — **arm A is frozen**, so every archived `Results/` number stands. `3m` merges by a **class-pair authority table, never by score**: an arm B `rickshaw`/`cng` claim suppresses an overlapping arm A `car`/`truck`/`bus`/`motorcycle`/`bicycle`, keeps both against `person`, and counts anything else as an out-of-table overlap. Suppressed arm A boxes move whole into the row's `merge.suppressed_arm_a` ledger — retained and auditable, but out of the arrays, because Stage 4 masks every array box in order | |

Arm B and every label it produces inherit RSUD20K's **CC BY-NC 4.0** licence
(research / non-commercial only); each merged box carries `proposal_arm` so a
release build can say which arm produced it.

### Stage 3b — 12 Hz identity propagation *(opt-in; decision C27)*

Stage 3 runs at the substrate's 2 Hz keyframe rate. Between two consecutive
keyframes the substrate holds ~5 more frames of the same camera at 12 Hz — 14,008
camera frames in total across 60 (scene, camera) chains, all of which walk clean.
Nothing in the pipeline had ever looked at them.

| | |
|---|---|
| **In** | Stage 3 proposals + the 12 Hz sweep frames |
| **Out** | Stage 3's **exact schema in Stage 3's row order**, plus additive keys, in this stage's own directory. Stage 4 consumes it unchanged via `--stage3-dir` |
| **Method** | **Phase A** re-runs Stage 3's *own* detector (identity read from Stage 3's manifest, never re-chosen) over the sweep frames, so a track born mid-gap has real evidence behind it. **Phase B** loads the SAM video tracker and propagates every keyframe box across the window. The two models are **never resident together** — Phase A unloads and empties the CUDA cache before Phase B loads, because C1's 4 GiB ceiling is the binding contract and two ~3 GB residents do not fit under it |
| **Recovery** | an object found at *t* and missed at *t+1* is re-emitted from the propagated mask's tight box, tagged `box_sources == "recovered"`, with score `last_yolo_score × decay^hops` and the hop count. **A recovered box is never presented as a detection**: provenance travels in the row, per box |
| **Identity** | every box carries a `track_ids` entry stable across keyframes for one (scene, channel). Stage 7 associates in 3D and does not read this; it exists because the recovery decision needs it and because a 2D identity is what makes the 12 Hz artifact auditable |

**The invariant that keeps this honest:** 12 Hz sweep frames are connective
tissue for 2D identity only. **No LiDAR or evaluation quantity exists
off-keyframe** — every row lands at a keyframe, so pipeline outputs stay at 2 Hz
and `pilot_plan.md` §11 decision 5 is not reopened. Nothing downstream needs to
know Stage 3b ran, which is the property that makes the A/B measurable at all.

Window failures are isolated, never fatal: a window that raises is treated as a
**block boundary** — its camera's live tracks are retired, the next keyframe's
boxes seed fresh ones, the failure is counted and named in the manifest, and the
run degrades. A Phase B **preflight** proves the tracker loadable and rebuilds
Phase A's detector *before* `clear_markers()` and before the first byte, so an
unavailable model costs neither the previous run's marker nor 8 minutes of sweep
detection over 14k images.

*Measured, one-scene trial (scene-0061, six cameras, SAM 3 tracker path):* 273
detections → 344 boxes (**+71 recovered, +26 %**), 76 tracks, 228 windows, 0
failed, 0 missing anchors, 436 s. Stage 4 consumed the output with **no code
change** and kept 67 of the 71 recovered boxes through cross-camera IoA-NMS.
Recovery scores reproduce the decay rule exactly (0.44605 → 0.31223 = 0.44605 ×
0.7¹). The full A/B matrix ([`scripts/run_matrix_track2d.sh`](scripts/run_matrix_track2d.sh),
five cells) is specified and runnable; its cells are **not** in `Results/` yet —
see [§13](#13-limitations-and-threats-to-validity).

### Stage 4 — Box-prompted masks + cross-camera IoA-NMS

| | |
|---|---|
| **In** | Stage 3 (or Stage 3b) proposals |
| **Out** | `masks.jsonl` — one mask per box, kept/suppressed with the contest recorded |
| **Providers** | SAM 3 (tracker path, default) · SAM 2.1-hiera-large (ungated alternate) · SAM 3.1 Object Multiplex (selectable, C26; reached through Meta's own `sam3` package because no transformers integration exists and `facebook/sam3.1`'s `config.json` is a stale SAM 3 copy that would silently load SAM 3's classes) · MobileSAM (~10 M params, 4 GB pilot tier) |

Three responsibilities, each with the silent failure it blocks:

1. **Masks come back at 1600×900, asserted rather than assumed.** MobileSAM
   prompts live in a 1024-longest-side transformed space and its decoder emits
   1024-space logits; skip the inverse transform and Stage 5 indexes a mask with
   pixel coordinates it does not own — painted points wrong by a scale factor
   everywhere, no crash, no empty output.
2. **One mask per box, in the same order.** Silently permuted masks assign every
   object its neighbour's class.
3. **IoA-NMS > 0.5 across overlapping cameras.** Its absence is what makes
   duplicate assignment reachable: the ring cameras overlap, one car is proposed
   twice and lifted twice, and the two boxes disagree slightly.

**Where the IoA is computed, and why it is not pixels.** Two boxes in two
different cameras live in two different image planes; their pixel coordinates are
not comparable, and a pixel-space IoA between them is a number the code will
happily produce and that means nothing — it suppresses whatever shares image
coordinates, which for a ring rig is "objects at the same bearing relative to two
different optical axes", i.e. arbitrary. So overlap is computed where the two
cameras *do* share a frame: each mask's tight box is back-projected through K⁻¹
and the camera→ego rotation into an **(azimuth, elevation) footprint in ego
frame**, and IoA is area-over-area there. Within one camera the two formulations
coincide; across cameras only this one exists.

The footprint uses the **mask's** tight box, not Stage 3's proposal box — SAM
routinely tightens a loose proposal, and a tighter footprint is a strictly better
overlap test. That costs one segmentation for a mask that may then be suppressed,
which is the trade, recorded in the manifest.

### Stage 5 — 2D→3D lift

| | |
|---|---|
| **In** | Stage 1 ground-filtered **single-sweep** cloud (ego frame) + Stage 4 masks (original resolution) |
| **Out** | per mask-instance point subsets, each point keeping its Stage 1 ego coordinates and `t_ns` |
| **Method** | six per-camera frusta unioned for R2 coverage; the four guards; the multi-camera contest |

**What this stage does and does not move.** The lift does not transform the cloud
at all: it decides, for each point, *which mask instance owns it*. The projection
is scaffolding for a labelling decision, not a change of representation — and
saying so explicitly is the difference between Stage 6 clustering the points it
thinks it has and clustering something silently re-based.

**Four guards, each counted separately in every record:**

1. `z ≤ 0` — culled **before** the perspective divide. A point behind the camera
   divided by its own negative z lands at a valid-looking mirrored pixel, inside
   the image, and paints itself with whatever mask is there. This is the guard
   whose absence produces a plausible, fully populated, wrong output.
2. Near-zero depth (`0 < z ≤ min_depth_m`) — here the divide does not lie, it
   explodes; the guard looks redundant until a point sits at exactly z = 0 and
   produces inf/nan that propagates into the mask index.
3. Out of bounds — not an error: it is most of the cloud, and its count *is* the
   frustum's shape.
4. Deterministic overlap resolution — the contest.

**The contest, and the reading it commits to.** A 3D point visible in two
overlapping cameras takes its class from the camera whose principal axis is
closest to the point's bearing, ties broken by a fixed camera-priority list.
"Closest bearing" is the cosine between the point's camera-frame ray and that
camera's optical axis — identical to the ego-frame formulation and immune to the
fact that the two cameras have two different ego poses. Two readings of the rule
exist and the choice is a recorded config value:

- `labelled_cameras` **(default)** — the contest runs over cameras that actually
  put the point inside a mask;
- `all_visible_cameras` (strict) — the nearest-axis camera wins even with no mask
  there, so the point goes unlabelled. Rejected as default because it makes
  labelling depend on whether Stage 3 fired in the *other* camera: a detection
  outcome leaking into a geometry rule.

Stage 5 does **no occlusion reasoning by design** — the wall behind a car projects
into the car's mask and is painted as the car. Those points are a second, deeper
cluster, and removing them is delegated to Stage 6 (see decision C24 for what
that delegation cost).

### Stage 6 — Per-instance BEV clustering + L-shape fit

| | |
|---|---|
| **In** | Stage 5 painted points |
| **Out** | one oriented box per mask instance; `size` in nuScenes **[w, l, h]**, yaw about +z from +x |
| **Method** | DBSCAN **per mask instance** with class-conditional ε from the priors file; keep-largest-cluster; L-shape fit; near-square policy; yaw re-assertion |

**Clustering scope is per mask instance.** "Class-conditional" names *which ε is
used* and nothing else. The rejected reading — pool a class across the frame,
cluster once, keep the largest — produces **one box per class per frame**: in a
parking row or gridlock (exactly the regime this project exists for) adjacent
cars merge under any ε large enough to hold one car, and every other instance of
that class is silently discarded. The output is clean, well-formed, and missing
most objects.

**"Keep largest cluster" is the reprojection-ghost filter**, not a quality knob —
and the count it drops is recorded per instance, because "the ghost filter
removed 80 % of the object" and "it removed the ghost" are the same field.

**ε comes from the priors file** — `0.6 × mean footprint diagonal`, per class,
derived from the `priors` scene subset — never from a hardcoded table whose class
names (`cyclist`, `traffic_cone`, `car`) are neither nuScenes categories nor
prompt phrases, and would therefore miss every lookup silently and hand every
class the same default. A class with no prior is recorded as
`eps_source: "config_fallback:…"` on **every affected box**.

**The near-square policy.** Pedestrians, cones and barriers have a *systematic*
90° yaw ambiguity, not an occasional one: the footprint is square to within
noise, so the fitter's chosen axis is arbitrary. Policy: the long BEV side is the
heading axis (making `w ≤ l` true by construction), and a near-square footprint
sets `yaw_ambiguous`. This matters downstream because Stage 8 anchors the near
face and grows the far one — under a 90° yaw error it grows the box sideways into
the neighbouring lane, deterministically and invisibly.

**Yaw is asserted against `conventions.py`, never trusted from the fitter.** Every
box re-projects its own points onto `(cos yaw, sin yaw)` and `(−sin yaw, cos yaw)`,
asserts the extents reproduce the stored `[w, l, h]` and centre, then round-trips
yaw → quaternion → yaw.

**What this stage does not decide:** the 180° heading direction. A symmetric
footprint of points cannot say which end is the front, and inventing an answer
here would be a guess dressed as geometry. Stage 7's yaw-consistency enforcement
along tracks is where that is resolved.

### Stage 7 — Predict-then-match tracking

| | |
|---|---|
| **In** | Stage 6 boxes + the painted clusters + re-ID crops |
| **Out** | `tracks.jsonl` with velocity, `velocity_semantics`, `icp_frame`, `appearance_trusted` per pair |

**The 2 Hz problem.** nuScenes keyframes are 2 Hz; a vehicle at 40 km/h moves
~5.5 m between them, so 3D IoU between two *un-propagated* consecutive detections
is zero for most vehicles. **Resolution: predict-then-match.** Every active
track's Kalman state is propagated forward by its own estimated velocity to the
current keyframe's timestamp **before** any IoU is computed; IoU is measured
between that *predicted* box and each detection, never between two raw
detections. `iou_gate` carries `derived for 2 Hz` provenance rather than being a
constant copied from a 10 Hz spec.

**Matching.** Gate then score, per class: a (track, detection) pair is eligible
only if `class_name` matches and predicted-vs-detection 3D IoU clears `iou_gate`.
Eligible pairs score `iou × cosine` when a trusted appearance embedding exists on
both sides, `iou` alone otherwise — declared per pair via `appearance_trusted`,
never silently substituted. `scipy.optimize.linear_sum_assignment` runs the
Hungarian algorithm over the cost matrix.

**ICP, and which frame it registers in — stated on every row.** LiDAR clusters at
two keyframes are each in *that keyframe's own* ego frame, and ego moves between
them. Registering them directly measures the object's motion **relative to the
ego vehicle**, which is not what nuScenes' absolute mAVE reports.

- `global_absolute` **(default)** — both clusters are hopped into
  `nuscenes_global` via each keyframe's own `ego_pose` before ICP runs. The
  registered translation is true world-frame displacement; velocity is absolute
  and mAVE-comparable, and the Kalman state lives in the same frame so a track's
  velocity means the same thing whichever produced it.
- `ego_relative` — kept and **declared inferior**: the literal version of the
  pitfall, retained so the failure is measurable rather than argued about.

**Kalman fallback, specified.** State `[x, y, z, vx, vy, vz, yaw]`,
constant-velocity / constant-yaw process model (no yaw-rate term — a *stated
limitation*, not an omission), Δt from the recorded `t_ns`. Trigger: a matched
detection with `num_lidar_pts < 15` skips ICP entirely. ICP additionally requires
the previous match to have been in the immediately preceding processed keyframe —
registering across a multi-frame gap folds a larger, more nonlinear motion into
one velocity estimate, and is refused rather than silently attempted.

**Yaw-consistency enforcement along tracks** is the designed defence against the
symmetric-footprint 180° flip, and is free geometry: a track that is consistent in
motion but flips heading between keyframes is corrected, and the flips are
counted (991–995 per run on this substrate, [§10](#10-results)).

### Stage 8 — Prior-anchored amodal inflation

The governing spec is one line — "sparse boxes inflated toward class means,
anchored to the LiDAR-return surface" — and the prior file supplies `dims: {mu,
sigma}` and nothing else. That leaves **how much, when, along which axes, and how
far is too far** undefined; and an undefined inflation still produces boxes:
bigger ones, uniformly, with no field saying so. All four are explicit, recorded
config values here.

**The anchor, stated correctly.** LiDAR sees surfaces, so the observed face is
always the **near** face. The rule is *hold the near face still and grow the far
face*, not the vaguer "shift outward" — the two differ by exactly half the growth
on every box, in the direction of the sensor, and "shift outward" is the reading
that walks the measured surface off the returns that produced it.

Three guards:

1. **The near face may not move** — re-derived after inflation and compared to
   its own pre-inflation position; a mismatch stops the run.
2. **The far face must move away from the sensor** — checked as a distance
   comparison, not assumed from sign algebra.
3. **The box may not grow to enclose the sensor** — a footprint that swallows the
   LiDAR origin is a fitting failure, not an amodal completion; the inflation is
   reverted whole and the reason recorded.

**Ambiguous anchors are declared.** When the sensor sits within
`near_face_margin_m` of an axis' mid-plane, the anchor is a coin flip; the axis
grows symmetrically and says so (`anchor: "symmetric_ambiguous"`).

**Height anchors the top, not the near face** — Stage 1's ground removal strips a
0.3 m band, so the *bottom* is the missing part; growth is downward, recorded as
a different anchor rather than folded into "anchor the near face".

**The prior is not automatically in the box's convention.** nuScenes `size` is
[w, l, h] with `l` along the object's own heading; Stage 6 defines heading as the
long BEV side, so every box has `w ≤ l`. For a class whose GT heading crosses its
long side — measured on this substrate, `a road barrier` has mean w 2.42 m and
mean l 0.58 m — the two conventions name different axes, and blending width
toward the prior's width would grow a 0.6 m box toward 2.4 m *across its own
fitted axis*. `prior_axis_mapping` decides the reading and is recorded on every
box.

Every inflated box records `inflated: bool` and `inflation_fraction`. Without
them, a run where ground removal stripped most object points produces boxes that
are ~90 % prior and 10 % measurement, and nothing in the output says so.

### Stage 9 — QA gating → I-4

Consumes Stage 8's `inflated.jsonl` (every row still carrying the pre-inflation
box as `box_measured`) plus the priors, and writes one `prelabels.jsonl` per
scene through `schemas.write_records()` — the **raising** boundary — so every
record that lands on disk has passed the AnnotationRecord / Provenance /
GateVector validators, including the `num_lidar_pts_basis` check and the
tier ≠ source rules.

**An inherited spec flaw the pilot surfaces rather than hides:** the spatial gate
tests post-inflation dimensions, but Stage 8 inflates *toward* the class prior, so
a box that has been through Stage 8 passes an "exceeds 2× class prior" test more
easily. The gate is weakest exactly where it is needed. Recorded, with the
pre-inflation box retained in every row so a corrected gate can be computed after
the fact.

### Release export — `prelabels.jsonl` → nuScenes tables (added 2026-08-23)

`scripts/export_release.py` is the writer the table in [§3.3](#33-interface-contracts)
called "release builder": it turns I-4/I-5 records (ego frame, phrase categories,
`track_id`, `num_lidar_pts`) into the five nuScenes annotation tables that
`dataset_benchmark` (`dbench`) reads — `sample_annotation`, `instance`,
`category`, `attribute`, `visibility` — and writes them into a **new** root beside
a copy of the source tables, with blobs symlinked (or `--blobs copy`). The source
dataroot is never modified.

```
python scripts/export_release.py --prelabels <prelabels.jsonl | stage9 out dir> \
    --dataroot <nuScenes root> --version v1.0-dhaka --out <new root> \
    [--mapper configs/release_category_map.yaml] [--human-verified-scenes scenes.txt]
```

What it does, and where each choice is recorded:

| Field | Rule |
|---|---|
| `translation` / `rotation` | ego → global through the keyframe's **LIDAR_TOP** `ego_pose` (the pose dbench reads back); `[w,x,y,z]`; cross-checked against `nuscenes-devkit` `Box.rotate/translate` at run time (`release_meta.json: devkit_cross_check`) |
| `size` | `size_wlh_m` verbatim — already `[w, l, h]` |
| `instance_token` | hash(scene, `track_id`); untracked records are singleton instances. `prev`/`next` chained by sample timestamp; `instance.first/last_annotation_token`, `nbr_annotations` filled |
| `category` | phrase → the 18 dbench names via `configs/release_category_map.yaml` (strict: unmapped or `unresolved` strings abort with the list; `static-obstacle` is deliberately unresolved — audit B1) |
| `visibility_token` | fraction of the 8 corners inside any camera image of the sample, via `conventions.project_lidar_to_image` → nuScenes tokens `1..4`. This is a **field-of-view proxy, not occlusion**. If any camera lacks intrinsics/size the export falls back to `"4"` and `release_meta.json: visibility.basis = assumed_full` |
| `attribute_tokens` | `record.attribute` ∈ {moving, stopped, parked} → `vehicle.*` / `pedestrian.*` / `cycle.*`; absent → `[]`. Nothing is inferred from velocity |
| `num_lidar_pts` | verbatim; basis (`single_sweep_ground_filtered_pre_inflation`, **not** nuScenes' with-ground count) carried per annotation and in `release_meta.json` |
| `num_radar_pts` | `0` — the rig has no radar (dbench B3) |

`release_meta.json` records pipeline/schema versions, the Stage 9 `run_manifest.json`
(if the input is a Stage 9 directory), sha256 of every input and of the mapper,
per-class/per-scene counts and tiers, and a `human_verified` flag per scene from
`--human-verified-scenes`. Tests: `tests/test_export_release.py` builds a 2-sample
synthetic root, asserts the global→ego round trip and the token chains, and runs
`dbench ingest validate --version v1.0-dhaka` on the output (0 errors required).

### The priors file

The pilot has no annotators, so it cannot produce the real seed-set S0. What it
can do is derive **the same shape of file** from nuScenes' own human labels and
prove the *consumption* side of the contract: Stage 6 takes `eps_bev` from it,
Stage 8 takes `dims.{w,l,h}.mu`, both through the interface the real S0 output
will use. Two guards keep that from becoming a lie:

1. The file sets `source: "nuscenes_gt_pilot"`, **never** `"S0"`. The only thing
   separating it from a real prior is that string, so the string is load-bearing.
2. `assert_release_source()` is five lines that stop a nuScenes-shaped prior from
   reaching a Dhaka release. The release builder calls it; nothing else does.

Priors are derived **under the pipeline's own constraints**, not the annotation
set's: centre inside E, `num_lidar_pts ≥ min_lidar_pts`, and the `priors` scene
subset only. nuScenes GT includes boxes beyond the 40 m cap and boxes with *zero*
LiDAR returns; the pipeline never sees either, so priors drawn from the full
population would pull inflation toward a mean the sensor never measures — a bias
with no symptom, because every inflated box still looks like a box.

Every taxonomy phrase gets an entry **including phrases with no GT instances**,
whose block is present with `dims: null`, `eps_bev: null`, `n_instances: 0`. A
consumer that finds a null knows the gap exists; a consumer that finds a missing
key does not know it looked. `min_pts_L1`, `min_pts_L2` and `conf_thresh` are
**null on purpose** — deriving sensor-density thresholds from a 32-beam substrate
for a solid-state target is exactly the copy the spec forbids; the measured
distribution is recorded in a `measured` block where it cannot be mistaken for a
threshold.

---

## 6. Models, roles and hardware tiers

### 6.1 Four roles, two tiers

Stages 6, 8 and 9 are purely geometric: no role, identical code on both tiers.

| Role | Stage | 4 GB pilot tier | Best-locally-runnable tier (default on the 4090) |
|---|---|---|---|
| `embedding_ood` | 2 | DINOv2 ViT-S/14 | DINOv2 ViT-L/14 |
| `proposal_2d` | 3 | Grounding-DINO-tiny | **YOLO11x** (C23); LLMDet-large retained |
| `mask_2d` | 4 | MobileSAM (~10 M params, ~40 MB) | **SAM 3** tracker path; SAM 2.1-L; SAM 3.1 multiplex |
| `reid_embedding` | 7 | DINOv2 ViT-S/14 | DINOv2 ViT-S/14 or **DINOv3 ViT-S/16** |

Three interface properties the registry was built to express on day one — each
cheap then, a stage rewrite later:

1. **`mask_2d` takes temporal state and a window from the start.** MobileSAM has
   no propagation and its adapter ignores both arguments; SAM 2.1/3 is
   `(frames, boxes, memory_state) → masks over time`. Without those parameters in
   the signature, the swap is a Stage 4 rewrite rather than a config edit. This
   is what made Stage 3b (C27) a provider change rather than a redesign.
2. **One provider may register against several roles** — SAM 3 is a unified
   proposal + mask + track engine, which a one-model-per-role registry cannot
   express.
3. **`proposal_2d` may return masks**, in which case Stage 4 is a pass-through.

**Preprocessing belongs to the role, not to the model.** `embedding_ood` embeds a
whole image (CLS token, ~518×518); `reid_embedding` embeds a small object crop,
ideally mask-pooled patch tokens. Same weights, different transform, different
output semantics. A registry that hands back "the DINOv2 model" and lets Stage 7
inherit Stage 2's transform upsamples a 28×28 crop ~20× and produces a similarity
dominated by interpolation artifacts. Hence `EmbeddingBatch.semantics` and
`.preprocessing` are required fields and the two roles are separate registry
entries even when they share a checkpoint.

### 6.2 Measured checkpoints (this 4090, 1600×900)

| Role | Checkpoint | Pinned revision | Measured |
|---|---|---|---|
| `proposal_2d` | `yolo11x.pt` (ultralytics) | `v8.3.0` (bytes hashed into the manifest) | 747 MiB fp32, ~31 ms/frame |
| `mask_2d` | `facebook/sam3` (tracker path, gated) | `3c879f39826c…` | 2.1 GiB, ~0.7 s smoke |
| `mask_2d` | `facebook/sam2.1-hiera-large` (ungated) | `665f8e2ad61c…` | 1.5 GiB, ~0.13 s/keyframe |
| `mask_2d` | `facebook/sam3.1` Object Multiplex (C26) | `daa63191845a…` + ckpt sha256 `0567debe…` | 3.26 GiB ckpt, 7,456 MiB peak; 0.78 s / 32-box image, 0.106 s per forward video frame |
| `proposal_2d` (open-vocab) | `iSEE-Laboratory/llmdet_large` | `bec37f296f05…` | 7.5 GiB fp32, ~350 ms/frame |
| pilot fallback | `grounding-dino-tiny` + MobileSAM `vit_t` | see C1 | fits the 4096 MiB cap |

**`--revision` is mandatory.** An unpinned hub id tracks the model's default
branch, which the `CheckpointSpec` contract refuses. For a local weights file the
revision is the release tag the file came from, and the file's own sha256 is
hashed into the manifest alongside it.

### 6.3 The VRAM contract (decision C1)

Two machines: a 3050ti laptop (4 GB — the plan's target, where the pipeline must
ultimately run) and an RTX 4090 (24 GB — where work happens). Since C19 the
default tier on the 4090 is the best-locally-runnable one
(`DHAKASCENES_VRAM_CAP_MIB=22000`), while the 4 GB pilot tier stays selectable.

Every GPU process enforces the cap synthetically
(`DHAKASCENES_VRAM_CAP_MIB` → `torch.cuda.set_per_process_memory_fraction`) and
every manifest records:

```json
"vram_cap": {"value_mib": 22000, "enforced": "synthetic",
             "physical_device_mib": 24079, "device_name": "NVIDIA GeForce RTX 4090"}
```

**The "runs on 4 GB VRAM" claim is quotable only after one end-to-end run on the
physical 3050ti**, and remains open. A synthetic cap reproduces neither
fragmentation, nor the laptop's display-server allocation, nor thermals; uncapped
runs carry `verified: false`. `scripts/measure_vram.py` measures every role two
ways — `max_memory_reserved()` *and* `mem_get_info()` free-deltas — and believes
the larger, because on a 4 GB device the gap between "allocated" and "occupied"
*is* the margin being budgeted; other-process allocation is measured via
`nvidia-smi` **before** this process creates a CUDA context, since `mem_get_info()`
creates the context it was meant to measure.

---

## 7. Class space and taxonomy

### 7.1 Phrases, not category names

The class space is a set of **natural-language prompt phrases**
(`configs/taxonomy_pilot_nuscenes.yaml`), and that phrase — not a category name,
not a COCO name — is what Stage 6 keys an ε on, Stage 8 keys a prior on, Stage 7
gates a match on, and the CVAT export keys a category id on. Both providers
arrive at the same space by different routes:

```
  Grounding DINO / LLMDet   phrases ARE the prompt; the caption is the input
  YOLO11x                   COCO class id → phrase, via coco_to_phrase_nuscenes.yaml
```

`prompt_phrase` values are lowercase, dot-free noun phrases, and
`PromptConfig.validate()` rejects a dotted string by pattern. The map is
**many-to-one by design**; the phrase set is the deduplicated set of values in
first-appearance order, and Stage 3 refuses a caption that repeats a phrase (two
spans competing for one class). Prompt order is fixed, documented and hashed into
every record, because with a caption provider the caption's composition is part
of the measurement.

### 7.2 The class collapse (decision C21) — a negative result worth publishing

Revision 1 prompted all 23 nuScenes mini categories individually. Measured on the
first full run, the resulting class assignment was not merely noisy but
**inverted**:

- **Five phrases had zero instances in the substrate** — *an animal, a stroller,
  a wheelchair, an ambulance, a police car*. "a police car" was nevertheless the
  single **most-predicted class (1,869 boxes)** at 0.0 % class-correct *by
  construction* — and at 13.2 % localization it held the **best-placed boxes in
  the whole run**. Real objects, impossible name.
- Two more were near-absent: "a bicycle rack" 680 predictions against 54 GT
  boxes; "a trash bin" 973 against 82.
- Meanwhile `vehicle.car` is 41 % of all ground truth and drew 1,193.

**Mechanism.** `phrase_scores()` takes the max over a phrase's tokens. "a police
car" contains the token *car*, which fires on any car, and a max never requires
the modifier to fire; a longer phrase also gets more draws at that max. Every
over-predicted class was a 3+ token phrase and every under-predicted one a plain
2-token noun. The authoring note in the config had predicted the **opposite**
bias, so nothing compensated for it.

**Resolution.** The class space became the official **nuScenes detection
benchmark's 10 classes** — a published convention, deliberately *not* a set tuned
against our own error histogram. It covers 99.2 % of this substrate's GT
(18,389 / 18,538).

**What it costs, recorded rather than glossed:** the pipeline can no longer
distinguish a child, a construction worker or a police officer from any other
pedestrian — that distinction is gone from the **output**, not merely from the
metric. And a class space collapsed onto nuScenes' benchmark is *further* from
the Dhaka target taxonomy, not closer. This file is a diagnostic instrument for
obtaining a trustworthy measurement on this substrate; it is not the delivered
class list.

| Phrase | Share of substrate GT | Reachable under YOLO11x |
|---|---:|---|
| a car | 41.1 % | ✓ |
| a pedestrian (5 subtypes collapsed) | 27.2 % | ✓ |
| a road barrier | 12.5 % | ✗ |
| a traffic cone | 7.4 % | ✗ |
| a truck | 3.5 % | ✓ |
| a motorcycle | 2.5 % | ✓ |
| a bus (bendy + rigid) | 2.2 % | ✓ |
| a bicycle | 1.3 % | ✓ |
| a construction vehicle | 1.1 % | ✗ |
| a trailer | 0.3 % | ✗ |

Excluded categories are **listed, not deleted**, each with its reason, so the
omission is auditable and `export_gt_coco.py` can report how much GT it excludes
rather than silently scoring the pipeline against objects it never prompted for.

### 7.3 Detector reachability (decision C25) — an evaluation contribution

A closed-vocabulary detector cannot produce a class its source vocabulary does
not contain. Under YOLO11x, four of the ten phrases have **no COCO source class
at all** — together **21.3 % of this substrate's GT** (1,579 of the eligible
boxes, 19.9 %). Recall on them is 0 **by construction**.

Scoring those four as misses does not measure the pipeline; it measures the gap
between two vocabularies, and it does it silently. A ten-class recall is then
four structural zeros averaged with six real numbers, and every summary built on
it understates the pipeline by a fixed, invisible amount.

The resolution, and the property that makes it trustworthy:

> **The reachable set is a property of the RUN, not of the checkout.** It is read
> from Stage 3's `run_manifest.json` (`class_map.phrases_in_use`, with its
> sha256), never from the config file on disk — which describes whatever the
> working tree happens to hold now, and is how an evaluation comes to describe a
> run that never happened. The config is a *declared fallback*, used only when no
> manifest exists, and the choice is recorded in `source`.

`eval_3d`, `paint_metrics` and `export_gt_coco` hold unreachable-class GT out of
the denominator, **counted and named** in the report. `--include-unreachable`
restores the old behaviour explicitly and stamps the choice into
`class_space.source`.

**Measured effect:** 3D GT recall 55.9 % → 68.9 %. And it cuts both ways —
`a truck` localization precision *falls* 50.4 % → 41.5 %, because truck boxes had
been collecting credit for landing on trailers and construction vehicles.

---

## 8. Experimental setup

### 8.1 Substrate

| Property | Value |
|---|---|
| Dataset | nuScenes **v1.0-mini** (CC BY-NC-SA 4.0, non-redistributable) |
| Scenes | 10 |
| City split | **6 Singapore / 4 Boston** (Boston: 0103, 0553, 0655, 0757) — verified twice; the governing plan's "5/5" is an erratum (C14) |
| Illumination | 3 night scenes (1077, 1094, 1100); one after rain |
| Keyframes | 404 @ 2 Hz |
| Camera images (keyframes) | 2,424 (404 × 6 ring cameras) |
| Camera frames total @ 12 Hz | 14,008 across 60 (scene, camera) chains — all chains walk clean; 2,364 inter-keyframe windows, lengths 4–8 (median 7) |
| LiDAR | `LIDAR_TOP`, 32-beam spinning |
| GT 3D boxes | 18,538 total; 18,389 (99.2 %) inside the 10-phrase class space |
| Coverage config | **R2** (all six ring cameras) |
| Metadata tables | 13, frozen; a 14th appearing invalidates every fingerprint |

### 8.2 Scene partition (decision 3, locked at Phase 3)

Disjoint, stratified by city and illumination. **Every subset spans both
cities**, and each contains at least one night scene.

| Subset | Scenes | Purpose |
|---|---|---|
| `priors` | 0061, 0103, 0553, 1077 | derive `eps_bev` and `dims.mu`. Deriving priors on the scenes the pipeline is scored on is the "tuned on the eval set" defect this partition exists to prevent |
| `tuning` | 0655, 1094 | per-class confidence thresholds only |
| `run` | 0757, 0796, 0916, 1100 | the scored subset |

**The partition constrains what may be tuned where; it does not make 10 scenes a
sample size.** Metrics in [§10](#10-results) are reported over all 10 scenes and
are labelled *descriptive*, never *measurement*, in every metrics file's own
`caveat` field.

### 8.3 Ablation design

Two matrices, each cell a **clean-slate rebuild of stages 0–8**.

**Matrix 1 — `proposal_2d` × `reid_embedding`** ([`scripts/run_matrix.sh`](scripts/run_matrix.sh), completed, archived in [`Results/`](Results/)):

| Cell | Detector | Re-ID |
|---|---|---|
| `yolo11x_dinov2` | YOLO11x (COCO-80, 6/80 mapped) | DINOv2-small |
| `yolo11x_dinov3` | YOLO11x | DINOv3 ViT-S/16 |
| `yolov8x-oiv7_dinov2` | YOLOv8x-OIV7 (Open Images, 13/601 mapped) | DINOv2-small |
| `yolov8x-oiv7_dinov3` | YOLOv8x-OIV7 | DINOv3 ViT-S/16 |

Stages 0–6 do not depend on the re-ID checkpoint, so two of the four cells could
have reused the other two's trees — and that is exactly what is **not** done. A
comparison whose arms were produced by different amounts of recomputation invites
the question of whether the difference *is* the reuse; a clean slate per cell
costs ~12 minutes and removes the question.

**Matrix 2 — Stage 3b A/B** ([`scripts/run_matrix_track2d.sh`](scripts/run_matrix_track2d.sh), specified and runnable, **cells not yet archived**):

| Cell | Configuration | Question |
|---|---|---|
| `yolo11x_3b_base` | stages 0–8, no Stage 3b | the baseline |
| `yolo11x_3b_fill` | + Stage 3b, recovery only | what does gap-fill add? |
| `yolo11x_3b_refine` | + `--refine-boxes` | may 3b *replace* a detected box with its propagated mask box? |
| `yolo11x_3b_sam31` | 3b tracker on SAM 3.1 | varies the model that **proposes** recovered boxes |
| `yolo11x_mask2d_sam31` | mask_2d on SAM 3.1, **no** 3b | varies the model that **segments** every box (the C26 promotion gate) |

Cells 4 and 5 answer different questions and neither substitutes for the other.
No cell publishes to CVAT: `run_stages.sh`'s `cvat` step deletes and recreates
every pipeline task, so running it four times over would leave the review server
describing whichever cell ran last.

### 8.4 Human review loop (CVAT)

Two **separate CVAT projects**, never merged:

- `— OUR PIPELINE output`: Stage 3 boxes as rectangles + Stage 4 masks as
  polygons (cv2 contours), suppressed duplicates exported bbox-only with
  `attributes.suppressed = true` so the reviewer sees what the contest removed;
  every annotation carries its Stage 3b provenance (`attributes.source`,
  `track_id`, `hops`).
- `— nuScenes HUMAN answer key`: the dataset's own labels projected 3D→2D per
  camera, over the **same images in the same frame order**, every label painted
  one uniform green. Never touched by any pipeline run (decision C13).

**Provenance is the project, not the task name.** Same-named tasks in one flat
list, distinguishable only by suffix, proved genuinely confusing to review. A
project's label schema is written once at creation, and an attribute the schema
does not name is dropped by the importer *without an error* — so an existing
project is checked against what the export actually carries and **refused** when
it is short, rather than publishing provenance that is not there.

Tasks are created **from the CVAT share** (the mounted nuScenes dataroot); no
image bytes are copied.

---

## 9. Evaluation methodology

Four instruments, each measuring a different link of the chain. All of them read
the reachable class set from the **run's own Stage 3 manifest** ([§7.3](#73-detector-reachability-decision-c25--an-evaluation-contribution)) and record their
own caveat string into their output.

### 9.1 Paint-inside-GT (`scripts/paint_metrics.py`) — the strongest signal available

> *"Of the LiDAR points Stage 5 painted, what fraction lies inside **any** nuScenes
> ground-truth 3D box of the same keyframe?"*

This is the single strongest correctness signal in the whole pilot, and it uses
data already on disk. Painted points are ego-frame at the LiDAR anchor time; GT
boxes are global-frame; the comparison transforms the points through the **same
LiDAR `ego_pose` Stage 5's chain used**, so a pose or frame error upstream shows
up here as a *collapsed rate*, not as a plausible one.

Three views, because the headline alone can lie:

| View | Definition |
|---|---|
| `paint_inside_gt` | painted points inside any GT box / painted points |
| `base_rate` | **all** single-sweep points inside any GT box / all points — the enrichment denominator. Painting at the base rate means the lift did nothing |
| `class_correct` | painted points inside a GT box whose category maps back to the predicted phrase, per class |

Plus **GT coverage**: GT boxes within 40 m carrying ≥ 5 LiDAR returns that
received ≥ 5 painted points.

### 9.2 3D box test (`scripts/eval_3d.py`)

**nuScenes-style centre-distance matching** (greedy, BEV centre distance ≤ 2 m,
per keyframe) rather than IoU matching, because IoU would **double-punish the two
known failure modes**: an under-tier detector's boxes are on the right object but
loosely sized, and a 90°-ambiguous yaw makes IoU collapse while the object is
still correctly found.

Reported over matched pairs only:

| Metric | Definition |
|---|---|
| **ATE** | mean BEV centre error (m) |
| **ASE** | mean (1 − IoU of the two boxes after aligning centre **and** yaw) — pure size error |
| **AOE** | mean \|yaw error\| mod π, **on boxes whose yaw the fitter trusts** (`yaw_ambiguous` excluded; mod π because Stage 6 declares yaw axis-only) |

Precision is split **localization** vs **class-aware**. Recall is against GT boxes
within 40 m carrying ≥ 5 LiDAR points — the same eligibility rule
`paint_metrics.py` uses, so the two numbers are readable together.

### 9.3 2D detection test (`scripts/eval_2d.py`)

Compares the two COCO exports on disk — pipeline output vs. the GT twin — image
by image, matched by `file_name`, greedy IoU, predictions sorted by score.

Two corrections that materially change the numbers, both recorded in the output:

1. **Deduplication.** `cvat_export/` is a CVAT *review* export, not a detection
   result set: every kept proposal appears twice, as the Stage 3 rectangle and
   the Stage 4 mask polygon sharing one bbox. v1 of the script counted both, so
   10,433 kept boxes arrived as 20,866 "predictions", the twins competed for the
   same GT under greedy matching, and **precision could not exceed 50 %**. One row
   per proposal is now kept and the drop is recorded in a `dedup` block.
2. **Amodal vs modal, so an IoU sweep instead of one gate.** Each GT box is the
   clipped AABB of a *projected 3D cuboid* — occluded extent included — while a
   2D detector emits a box around *visible pixels*. IoU between the two is
   depressed **systematically, not randomly**. Results are therefore reported at
   IoU 0.3 / 0.5 / 0.75 as well as the single gate.

GT recall is additionally split by GT box size (≥ 32×32 px ≈ "visibly present"),
because the projected answer key includes every human-labelled object at any
distance and occlusion, which **no** 2D detector at 1600×900 could fully recall.
By default nothing is filtered out of the GT — an object 90 % behind a bus, or
with zero LiDAR returns, is still in the denominator. That is a deliberate, harsh
baseline; `--min-visibility` / `--min-lidar-pts` narrow it, both default off so
previously recorded numbers stay reproducible, and whatever was used is written
into each `instances.json`.

### 9.4 Threshold tuning (`scripts/tune_thresholds.py`)

Runs on the **`tuning` split only**, against a Stage 3 output produced with a
deliberately low default threshold (a sweep can only raise a floor it can see).
Three cases, each flagged distinctly:

- class has GT here and true positives → sweep, maximise **F0.5**
  (precision-weighted: the QA gate wants clean auto-accepts more than coverage)
  at IoU ≥ 0.3 (the GT is amodal, so 0.5 punishes a tightness the detector cannot
  express);
- class has GT here but **zero** true positives → nothing to optimise; threshold
  set to the 95th percentile of the class's all-false-positive score
  distribution, suppressing ~95 % of its junk. **Flagged**;
- class **absent** from the tuning split's GT → same rule, flagged louder — the
  choice is informed only by where the class fires falsely.

The script **only prints**. Writing the result into the taxonomy file is a
human-reviewed edit with this script's output as the provenance.

### 9.5 What is *not* evaluated

No mAP/NDS, no AMOTA. Those require a full detection benchmark protocol over a
proper split; on 10 scenes of a stand-in substrate they would be a number with
the shape of a benchmark result and none of its meaning. The instruments above
were chosen because each isolates one link of the chain.

---

## 10. Results

**Read every number in this section through the banner at the top of this file.**
All four cells ran the full substrate (10 scenes / 404 keyframes / 2,424 images),
clean-slate, under `DHAKASCENES_VRAM_CAP_MIB=22000` on an RTX 4090. Model
identity in each row is read from that run's own stage manifests, never from the
directory name. Source: [`Results/COMPARISON.md`](Results/COMPARISON.md).

### 10.1 Configuration of each cell

| Cell | `proposal_2d` | source classes mapped | `reid_embedding` |
|---|---|---|---|
| `yolo11x_dinov2` | `yolo11x.pt` (sha `7bc158aa95c0`) | 6 / 80 COCO | `facebook/dinov2-small` @ `ed25f3a31f01` |
| `yolo11x_dinov3` | `yolo11x.pt` | 6 / 80 COCO | `facebook/dinov3-vits16-…` @ `114c13799502` |
| `yolov8x-oiv7_dinov2` | `yolov8x-oiv7.pt` (sha `89acc72b5b4d`) | 13 / 601 OIV7 | `facebook/dinov2-small` |
| `yolov8x-oiv7_dinov3` | `yolov8x-oiv7.pt` | 13 / 601 OIV7 | `facebook/dinov3-vits16-…` |

All four reach the **same six phrases** and leave the **same four unreachable**
(`a road barrier`, `a traffic cone`, `a construction vehicle`, `a trailer`).
`mask_2d` is `facebook/sam3` @ `3c879f39826c` in every cell.

### 10.2 Headline table

| Metric | `yolo11x_dinov2` | `yolo11x_dinov3` | `yolov8x-oiv7_dinov2` | `yolov8x-oiv7_dinov3` |
|---|---|---|---|---|
| **3D boxes vs. human 3D answer key** | | | | |
| precision — localization | 75.1 % | 75.1 % | **82.0 %** | **82.0 %** |
| precision — class-aware | 71.7 % | 71.7 % | **78.8 %** | **78.8 %** |
| GT recall (reachable classes) | **72.2 %** | **72.2 %** | 38.3 % | 38.3 % |
| ATE (m) ↓ | **0.618** | **0.618** | 0.692 | 0.692 |
| ASE ↓ | 0.777 | 0.777 | **0.682** | **0.682** |
| AOE (rad) ↓ | 0.395 | 0.395 | **0.279** | **0.279** |
| boxes shipped | 6,104 | 6,104 | 2,965 | 2,965 |
| matched to a GT box | 4,586 | 4,586 | 2,432 | 2,432 |
| **2D proposals vs. human 2D answer key** | | | | |
| precision — localization | 62.5 % | 62.5 % | **83.6 %** | **83.6 %** |
| precision — class-aware | 59.7 % | 59.7 % | **80.9 %** | **80.9 %** |
| GT recall (all GT) | **36.6 %** | **36.6 %** | 16.6 % | 16.6 % |
| GT recall (≥ 32 px) | **38.8 %** | **38.8 %** | 18.2 % | 18.2 % |
| predictions counted | 10,150 | 10,150 | 3,438 | 3,438 |
| **Paint / lift geometry** | | | | |
| GT coverage rate | **74.7 %** | **74.7 %** | 43.9 % | 43.9 % |
| painted points inside a GT box | 87.8 % | 87.8 % | **89.3 %** | **89.3 %** |
| enrichment over base rate | 6.93× | 6.93× | **7.05×** | **7.05×** |
| points painted | 482,074 | 482,074 | 461,589 | 461,589 |
| **Tracking (Stage 7)** | | | | |
| tracks total | 2,956 | 2,958 | 1,147 | 1,148 |
| tracks ≥ 3 hits | **430** | 424 | 246 | 245 |
| matched pairs | 3,148 | 3,146 | 1,818 | 1,817 |
| appearance-trusted pairs | 3,121 | 3,119 | 1,818 | 1,817 |
| yaw flips applied | 991 | 995 | 567 | 565 |
| **Runtime** | | | | |
| Stage 3 proposals (s) | 80.5 | 80.2 | 104.0 | 104.1 |
| Stage 7 track (s) | 85.7 | 86.7 | 65.2 | 66.3 |

Stage 4 (SAM 3 over all kept proposals) is the dominant cost at **394.6 s**;
Stage 5 lift is **23.5 s** (CPU).

### 10.3 Per-class 3D localization precision (boxes shipped in parentheses)

| Phrase | `yolo11x` | `yolov8x-oiv7` |
|---|---|---|
| a bicycle | 22.8 % (162) | 26.4 % (72) |
| a bus | 65.4 % (159) | 74.8 % (111) |
| a car | 75.0 % (3,316) | 81.7 % (2,202) |
| a motorcycle | 88.7 % (71) | 100.0 % (22) |
| a pedestrian | 82.2 % (1,900) | 95.5 % (356) |
| a truck | 67.1 % (496) | 84.2 % (202) |

### 10.4 Findings

**F1 — The detector choice is a precision/recall operating point, not a quality
ranking.** YOLOv8x-OIV7 wins every precision column and loses every recall column
by roughly a factor of two (3D GT recall 72.2 % vs 38.3 %; GT coverage 74.7 % vs
43.9 %). It ships less than half as many boxes (2,965 vs 6,104) and those boxes
are better. For a **pre-annotation** pipeline whose output a human corrects,
recall is the more expensive side to lose: a missing object must be drawn from
scratch, a wrong box is adjusted. Neither cell is "better" without stating which
cost dominates — and that is a workflow decision, not a metric one.

**F2 — The re-ID checkpoint changes almost nothing measurable here.** DINOv2-small
and DINOv3 ViT-S/16 produce **identical** 3D, 2D and paint metrics (Stages 0–6 do
not consume the re-ID model at all, and Stage 7 does not feed back into them),
and differ only in Stage 7 internals: 2,956 vs 2,958 tracks, 430 vs 424 tracks
with ≥ 3 hits, 991 vs 995 yaw flips. **The appearance term is not the binding
constraint on this substrate** — 3,121 of 3,148 pairs (99.1 %) were
appearance-trusted, so the term was almost always available and still moved
nothing. This is a clean negative result: at 2 Hz with predict-then-match, the
IoU gate is doing the association work. It is also exactly the regime where a
denser, more homogeneous scene (Dhaka) would be expected to differ, which makes
this a pre-registered hypothesis for the real substrate rather than a settled
question.

**F3 — The lift geometry is sound; the detector is the bottleneck.** 87.8 % of
painted points land inside a human 3D box against a **12.7 % base rate** — a
**6.93× enrichment**. A pose error, a double-applied extrinsic, a mask-scale
error or a frame confusion anywhere in Stages 1–5 would collapse this number
rather than degrade it gracefully. The four-hop chain, the ego-frame invariant
and the mask-resolution assertion are therefore *demonstrated*, not merely
asserted.

**F4 — 2D and 3D precision diverge in an informative direction.** For `yolo11x`,
2D localization precision is 62.5 % while 3D localization precision is 75.1 %.
Boxes that survive segmentation, lifting, ghost-filtering and clustering are a
*cleaner* population than the raw proposals: a false 2D box that paints no
coherent LiDAR cluster produces no 3D box at all. The geometric stages act as a
filter, and the size of that effect (+12.6 pp) is a measurement of it.

**F5 — Class-aware precision trails localization precision by ~3 pp in 3D
(71.7 % vs 75.1 %) but the gap is class-dependent and large where it matters.**
`a truck` is 67.1 % localization and 42.1 % class-aware — trucks land on real
objects that are frequently *not* trucks. Paint-level class-correctness shows the
same structure (`a truck`: 91.4 % inside any GT box, 58.3 % class-correct;
`a bicycle`: 56.2 % / 15.1 %). **The under-tier detector places boxes far better
than it names them**, which is precisely the split the QA-gate/tier policy exists
to exploit: geometry can be auto-accepted at a higher tier than class.

**F6 — Yaw flips are common, and that is the defence working.** 991–995
yaw-consistency corrections were applied across ~3,150 matched pairs (≈ 31 %).
Stage 6 declares yaw axis-only and refuses to guess heading direction; Stage 7's
along-track enforcement is what resolves it. Without that stage, roughly a third
of matched detections would carry a 180° heading error into Stage 8, which would
then anchor the wrong face.

### 10.5 Diagnostic finding: the ghost filter cannot fire (decision C24)

Stage 5 performs **no occlusion reasoning by design** and delegates far-surface
removal to Stage 6's keep-largest-cluster rule. Measured over all 10 scenes and
6,104 fitted boxes: **366 (6.0 %) exceed 2× their class prior on a BEV axis**. Of
those oversized instances, **64.2 % have exactly one DBSCAN cluster** — object and
far surface chained together, because ε (2.995 m for `a car`, 4.666 m for
`a truck`) exceeds the p50 1.50 m depth gap between them. **No
cluster-*selection* rule can separate them.** The far tail is a p50 9.2 % of the
cluster's points sitting a median +5.96 m deeper along the camera ray, and it
sets the length: a "car" 16.49 m long where the object is 4.00 m. Published to
CVAT this reads as several boxes stacked on one vehicle.

The masks are innocent — this is a clustering-scale problem, not a segmentation
one. It is recorded as an open defect with a measured mechanism rather than a
tuning note, and it is the highest-value single fix available to the next
revision (a depth-gap split along the camera ray, before ε is applied).

---

## 11. Reproducibility and provenance machinery

### 11.1 The manifest is the record; stdout is not

Every stage writes `run_manifest.json` carrying: resolved config and its hash,
seed and its derivation, checkpoint ids + hub revisions + weight sha256s, class
map path + sha256 + reachable/unreachable phrase sets, dataroot fingerprint,
`usable_scenes.json` hash, scene partition, `W_acc` count and duration, image
resolution and prompt configuration, the VRAM cap block, per-stage input/output
counts, elapsed time, and the upstream block (including
`accepted_degraded_upstream`).

### 11.2 The metadata fingerprint

`metadata_fingerprint()` binds every downstream artifact to the exact bytes of
the 13 v1.0-mini metadata tables it was computed from. The table set is **frozen**:
a 14th file appearing in the version directory, or one of these missing, is a
substrate change and must invalidate every fingerprint computed under the old
set. `usable_scenes.json` carries the fingerprint and the dataroot realpath, and
every consumer verifies the match.

### 11.3 The path contract

One `configs/paths.yaml`, resolved **exactly once** into a frozen `Paths` object;
`validate_paths()` runs at process start before any stage touches disk. Its
invariants:

- dataroot is read-only by convention; nothing in the pipeline opens a dataroot
  path for writing;
- the on-disk version directory name equals `version` **verbatim** — mini
  metadata against trainval blobs yields zero token matches, which reads
  downstream as "no usable scenes" rather than as a config error;
- `work_root`, `out_root` and `probe_out_root` are pairwise `commonpath`-disjoint
  from dataroot;
- write roots default **outside the repository**. A symlink into the project tree
  is dereferenced by `shutil.copytree`, `rsync -L` and a Docker build context, and
  intermediate point clouds then fill the repo disk.

No module may re-derive a path from string concatenation off `cwd`. (Related:
Ultralytics' `YOLO("yolo11x.pt")` on a missing file downloads **into the current
working directory**, i.e. into the repo tree — so the adapter refuses anything
that is not an existing path.)

### 11.4 Atomic writes and read-back validation

All persistence goes through `write_json_atomic` / `write_records()`, which
validate **on write and again on read-back** before the file lands. Records
round-tripping through plain dicts get reconstructed downstream, and that was the
path by which a validator could be skipped entirely. `validate() -> [errors]` is
retained for the legitimate check-then-decide case (one scene's failure lands in
`failures.json`; it does not abort the run), but the write boundary **raises** —
because a non-raising design makes "just don't inspect the returned list" the
easiest bypass of the invariant it exists to protect.

### 11.5 Independent re-measurement

[`scripts/collect_evidence.py`](scripts/collect_evidence.py) re-derives the
substrate facts **sharing no code with `pipeline/`** — quaternion→yaw, JPEG
header parsing and the metadata fingerprint are reimplemented, so a bug in
`pipeline/common/` cannot ratify itself. Its output is MEASUREMENT-class evidence
for the conformance ledger. Likewise
[`scripts/probe_substrate.py`](scripts/probe_substrate.py) verifies the substrate
with **no GPU, no models and no `nuscenes-devkit`**: metadata parsed as plain
JSON, JPEG dimensions from the SOF marker, point clouds from struct arithmetic.

### 11.6 Environment pinning

Every dependency is exact-pinned with a provenance note.
`requirements.txt` is the index and names the other files and the reason each
split exists; `requirements-lock.txt` is regenerated after any change;
`PYTHONNOUSERSITE=1` is mandatory. This is not hygiene theatre — decision C15
records a full set of Stage 1 artifacts produced by the **wrong interpreter**
(system Python 3.10.12 / numpy 1.21.5 instead of the environment's), discovered
only because the manifest recorded package versions.

---

## 12. Conformance ledger — self-audit

[`docs/conformance.yaml`](docs/conformance.yaml) maps **every checkable assertion
in the governing plan** to evidence about this repository;
[`scripts/check_conformance.py`](scripts/check_conformance.py) validates it and
renders [`docs/CONFORMANCE.md`](docs/CONFORMANCE.md).

As of commit `4614a9c`, **284 rows**:

| Status | Rows | Meaning |
|---|---:|---|
| CONFORMS | 39 | demonstrated by TEST or MEASUREMENT |
| PLAUSIBLE | 135 | code appears to implement it; **nothing demonstrates it** |
| VIOLATES | 50 | implemented contrary to the claim |
| ABSENT | 57 | not implemented |
| UNVERIFIABLE | 2 | not checkable on this substrate |
| N/A | 1 | waived |

Evidence classes: CODE-SITE 209, MEASUREMENT 36, ARTIFACT 35, TEST 4.

Six rules keep the ledger from degenerating into a reading exercise:

| Rule | Enforcement |
|---|---|
| **R1** | `CONFORMS` requires TEST or MEASUREMENT evidence (except pure existence claims, where ARTIFACT is legal). *"I read the code and it does this"* (CODE-SITE) supports **PLAUSIBLE at most** |
| **R2** | No row has an empty `evidence_ref` — even `ABSENT` rows name the path that should exist |
| **R3** | Every audit finding ID (P0-1…9, P1-1…15, M-1…15, X-1…10) appears in at least one row's `closes` |
| **R4** | Every top-level plan section appears in at least one row's `source` |
| **R5** | IDs are unique and match the declared patterns |
| **R6** | Enum fields hold only declared values |

R3 and R4 are the load-bearing ones: they guard against a ledger that *looks*
complete — hundreds of rows, tidy statuses — while quietly skipping the sections
or findings where the code is weakest. **Coverage is asserted, not assumed.**

**Reading order is VIOLATES first.** A PLAUSIBLE row is an open question, not a
pass; it converts to CONFORMS only when a test or measurement lands. That 135
rows sit at PLAUSIBLE and only 4 rows carry TEST evidence is the single largest
honest weakness of this release ([§13](#13-limitations-and-threats-to-validity)).

---

## 13. Limitations and threats to validity

Drafted as the paper's limitations section, kept honest.

### 13.1 Substrate

1. **The substrate is not the target.** nuScenes v1.0-mini is Boston and
   Singapore, structured traffic, a 32-beam spinning LiDAR. DhakaScenes is Dhaka,
   unstructured traffic, a 360°×59° non-repetitive solid-state LiDAR. **Nothing
   measured here transfers as a performance claim**, and the sensor difference
   alone changes point density, sweep geometry and ground-fit behaviour.
2. **Ten scenes are not a sample.** Every metrics file carries
   `"caveat": "descriptive on 10 scenes … never a measurement"` in its own JSON.
   The disjoint priors/tuning/run partition constrains what may be tuned where; it
   does not manufacture statistical power.
3. **Inherited parameters are applied out of domain.** The 0.3 m ground band,
   40 m range cap and 4 m height cap were chosen for a Livox Mid-360 on Dhaka
   roads. On a 32-beam spinning LiDAR the 0.3 m band removes all wheel returns
   from every vehicle and most of a traffic cone's body. Sector RANSAC mis-fits on
   ramps, speed bumps and cambered roads, and a single plane per sector cannot
   represent a curb. Nothing corrects for this; the per-filter/per-sector ledger
   is what makes it measurable.

### 13.2 Method

4. **Four of ten classes are unreachable** under the default detector — 21.3 % of
   substrate GT, recall 0 by construction. This is a capability gap of a
   closed-vocabulary detector, declared and counted, not a bug. The
   open-vocabulary path is one `--model-id` away and is the route to a Dhaka
   taxonomy that COCO has never heard of.
5. **Per-class confidence thresholds are still empty.** Every class falls back to
   the inherited default 0.40, which decision C19 flags **unvalidated** since the
   transformers 5.15.0 upgrade changed how fast image processors resize. This is
   the largest known untuned quantity in the pipeline. `tune_thresholds.py` exists
   and runs on the tuning split; writing its output back is an owed, human-reviewed
   edit.
6. **The far-surface defect (C24) is open** — see [§10.5](#105-diagnostic-finding-the-ghost-filter-cannot-fire-decision-c24). 6.0 % of shipped boxes exceed 2× their class prior;
   the delegated filter provably cannot fix it.
7. **The QA spatial gate tests post-inflation dimensions**, which is exactly
   backwards; inherited from the spec, surfaced rather than silently corrected,
   with `box_measured` retained in every row so a corrected gate is computable
   after the fact.
8. **Stage 7's process model has no yaw-rate term** — constant-velocity /
   constant-yaw. A stated limitation, not an omission.
9. **Stage 2 is a branch that feeds nothing** in this configuration, and on 242
   sampled images sits at the edge of where HDBSCAN parameters are meaningful.
10. **Stage 9 has not been run end-to-end**; the module exists and the schema
    boundary is enforced, but the Phase-10 orchestrator (`run_pilot.py`) is
    unwritten and `run_stages.sh` chains stages 3–8 plus eval.

### 13.3 Evidence

11. **Zero unit tests exist.** 135 conformance rows sit at PLAUSIBLE on CODE-SITE
    evidence, and only 4 rows carry TEST evidence. The module docstrings name the
    silent failures with unusual precision and the assertions are in the code, but
    *"the code contains an assertion"* is not the same evidence class as *"a test
    demonstrates the assertion fires"*. **This is the single most consequential gap
    in the release** and the first item on the roadmap.
12. **The 4 GB VRAM claim is unverified on hardware.** Every run reported here was
    produced under a synthetic 22,000 MiB cap on a 24 GB card. A synthetic cap
    reproduces neither fragmentation, nor the laptop's display-server allocation,
    nor thermals.
13. **The Stage 3b A/B is specified but not archived.** The one-scene trial
    (+26 % boxes recovered, deterministic, consumed by Stage 4 with no code change)
    is real; the five-cell matrix has not been run to `Results/`, so **no
    full-substrate claim about gap-fill's effect on recall or precision exists yet**.
14. **SAM 3.1 is gated but not promoted.** Both adapter modes pass the smoke test
    on this card; promotion to the `mask_2d` default waits on the C26 Gate (2)
    A/B cell, which is cell 5 of the unrun matrix above.
15. **Phase ordering was violated during the build.** Code for later stages was
    written before earlier phase gates passed (conformance row `12-r1`). The
    ledger records it rather than retro-fitting a clean history.

---

## 14. Decision register

[`docs/DECISIONS.md`](docs/DECISIONS.md) holds one entry per contradiction
between the plan, the machine and the code, in a fixed format —
`Status / Plan says / Disk says / Resolution / Because / Recorded in / Gate`.
**Nothing is closed by silence.** Statuses: `RESOLVED`,
`DEFERRED-TO-PHASE-<n>`, `ESCALATED-TO-HUMAN`, `STANDING` (a tripwire to
preserve, not a decision).

| ID | Subject | Outcome |
|---|---|---|
| C1 | 4 GB plan vs 24 GB card | two-machine fleet; synthetic cap enforced and recorded; 4 GB claim quotable only after a physical-laptop run |
| C2 | The plan's status header was false | `BUILD_STATE.md` **generated** from disk evidence, never hand-maintained |
| C3 | `manifest.py` absent, responsibilities leaked into stages | markers, atomic writers and the upstream gate consolidated |
| C4 | No tests; no phase gate passed | recorded; still open ([§13.3](#133-evidence)) |
| C5 | Manifest missing four mandatory fields | fields specified; partially closed |
| C6 | Configs referenced by code that cannot run without them | configs authored with provenance notes |
| C7 | Phase order already violated | recorded, not retro-fitted |
| C8 | What does ρ count? | resolved in `eval_region.py` as count-over-area with one implementation |
| C9 | Stage 9, probe and driver did not exist | Stage 9 + probes written; `run_pilot.py` still absent |
| C10 | Claim-hygiene banner in exactly one file | banner propagated to manifests and figures |
| C11 | Repository structure violates the plan | recorded (rows `2-r11`, `2-r12`) |
| C12 | nuScenes licence | CC BY-NC-SA 4.0; substrate and GT-derived priors are non-redistributable |
| C13 | Deliberate tripwires | **STANDING** — preserve, do not normalise (e.g. the answer-key CVAT project is never touched) |
| C14 | Substrate errata | calibration is **per scene**; city split 6/4 not 5/5; `animal` is undotted |
| C15 | Stage 1 artifacts produced by the wrong interpreter | regeneration mandated; caught by manifest package versions |
| C16 | Pipeline blocked by its own gate semantics | three-state markers + explicit degraded opt-in |
| C17 | C1's cap bindings existed only as words | cap enforced in every GPU process, recorded in every manifest |
| C18 | Peer code-sweep intake | findings merged into the register |
| C19 | Default model tier on the 4090 | best-locally-runnable becomes default; 4 GB tier stays selectable and stays the only tier that can support the 4 GB claim |
| C20 | The chain past Stage 5 was blocked by gate semantics, not physics | unblocked by C16 |
| C21 | The 2.4 % detection number was a broken ruler + a class space that does not exist here | class space collapsed to the nuScenes benchmark's 10; mechanism of the inversion documented ([§7.2](#72-the-class-collapse-decision-c21--a-negative-result-worth-publishing)) |
| C22 | Threshold tuning on the tuning split | `tune_thresholds.py`; write-back still owed |
| C23 | Stage 3 moves to YOLO11x + SAM 3 | closed-vocab detection + box-prompted segmentation; open-vocab retained one flag away; the cost (4 unreachable phrases) printed before every run |
| C24 | Stage 6's far-surface filter cannot fire | measured mechanism, open defect ([§10.5](#105-diagnostic-finding-the-ghost-filter-cannot-fire-decision-c24)) |
| C25 | Unreachable GT classes leave the recall denominator | reachable set read from the **run's** manifest; counted and named; 3D recall 55.9 % → 68.9 % |
| C26 | SAM 3.1 Object Multiplex as a selectable `mask_2d` provider | reached via Meta's own `sam3` package (transformers has no integration and `config.json` is a stale SAM 3 copy); smoke-gated; promotion pending |
| C27 | Stage 3b — 12 Hz gap-fill | sweep frames are connective tissue for 2D identity only; **all pipeline output stays at 2 Hz**; Gate (2) open |

---

## 15. Repository layout

```
pipeline/common/          the representation contract (§4)
  conventions.py          frames, time base, THE projection chain
  schemas.py              I-1…I-5 records, validators, raising write boundary
  manifest.py             three-state markers, atomic writers, upstream gate
  paths.py                path contract + metadata fingerprint
  eval_region.py          evaluation region E and density ρ — one implementation
  class_space.py          detector reachability, read from the run's manifest
  model_interfaces.py     role Protocols + provider registry
  sequences.py            12 Hz frame sequencing for Stage 3b

pipeline/stage0_data_probe/   six substrate predicates → usable_scenes.json
pipeline/stage1_ingestion/    both clouds, ground removal, filter ledger
pipeline/stage2_ood/          DINOv2 → HDBSCAN GLOSH (branch)
pipeline/stage3_proposals/    2D proposals, six ring cameras
pipeline/stage3b_track2d/     12 Hz identity propagation + box recovery
pipeline/stage4_masks/        box-prompted masks + ego-frame IoA-NMS
pipeline/stage5_lift/         2D→3D lift, four guards, camera contest
pipeline/stage6_cluster/      per-instance DBSCAN + L-shape fit (+ priors.py)
pipeline/stage7_track/        predict-then-match, ICP, Kalman, yaw consistency
pipeline/stage8_inflate/      prior-anchored amodal inflation
pipeline/stage9_qa/           gate vector → tier → prelabels [I-4]

configs/                  every tunable, each with a provenance note
  paths.yaml                    the path contract
  taxonomy_pilot_nuscenes.yaml  category → phrase, thresholds, exclusions
  coco_to_phrase_nuscenes.yaml  COCO-80 → phrase (YOLO11 provider)
  oiv7_to_phrase_nuscenes.yaml  Open Images 601 → phrase (YOLOv8-OIV7 provider)

scripts/                  probes, evaluation, export, review, drivers
  probe_substrate.py  collect_evidence.py  measure_vram.py  build_state.py
  eval_2d.py  eval_3d.py  paint_metrics.py  tune_thresholds.py
  export_gt_coco.py  export_cvat_coco.py  cvat_setup.py  cvat_setup_3d.py
  export_cvat_3d.py  cvat_purge.py  render_annotations.py  render_boxes_3d.py
  render_boxes_bev.py  save_run_results.py  compare_runs.py
  check_conformance.py  smoke_sam31.py
  run_stages.sh  run_matrix.sh  run_matrix_track2d.sh  run_pilot.py

docs/                     governing documents + this build's ledgers
  comprehensive.md        the full DhakaScenes system + paper plan
  Annotation_pipeline.md  the annotation-pipeline reference design
  pilot_plan.md           the pilot bible (rev 2)
  BUILD_PROMPT.md         build rules, precedence, verification order
  DECISIONS.md            the contradiction register (27 entries)
  conformance.yaml        284-row evidence ledger (machine-validated)
  CONFORMANCE.md          rendered ledger — do not edit
  BUILD_STATE.md          generated from disk evidence — do not edit
  GAP_ANALYSIS.md         known pitfalls of the reference design
  RUNNING.md              operational runbook
  CVAT_GUIDE.md           review-loop instructions
  evidence/               independent re-measurement artifacts

Results/                  archived per-run metrics + run_config.json
```

Approximate implementation size: **≈28.9k lines of Python** — `pipeline/` ≈21.7k
(of which `common/` 4.3k is the contract layer), `scripts/` ≈7.1k.

**The substrate and every generated output live outside version control.** The
nuScenes blob root is CC BY-NC-SA 4.0 and non-redistributable, and GT-derived
statistics (the priors) are licence-covered derived data (decision C12).
`work_root`, `out_root` and `probe_out_root` default to `/home/mt/dhakascenes/*`.

---

## 16. Installation and running

### 16.1 Environment

One interpreter, always.

```bash
export PATH="/home/mt/miniconda3/bin:$PATH"
conda activate ano_pipe            # or: conda env create -f environment.yml
export PYTHONNOUSERSITE=1          # mandatory
cp .env.example .env               # then fill the SECRET block
pip install -r requirements-torch.txt
pip install -r requirements.txt
pip install --no-deps -r requirements-devkit.txt
```

```bash
PY=/home/mt/miniconda3/envs/ano_pipe/bin/python   # C15: always this one
export DHAKASCENES_VRAM_CAP_MIB=22000             # C19 budget on the 4090
#                                4096             # C1 pilot budget (4 GB laptop)
$PY -m huggingface_hub.cli auth login             # once, for the gated SAM repos
```

Checkpoints land in `~/.cache/huggingface/hub`; every later run is offline.

### 16.2 Full chain

```bash
scripts/run_stages.sh                        # stages 3-8 → eval + viz → CVAT publish
scripts/run_stages.sh 4 5 6 7 8 eval cvat    # resume after a completed stage 3
scripts/run_stages.sh --scenes scene-0061    # one scene, end to end
scripts/run_stages.sh --no-cvat              # full chain, leave the review server alone
```

The wrapper honours the **three-state exit code**:

| rc | meaning | wrapper response |
|---|---|---|
| 0 | clean | continue |
| 1 | **DEGRADED** — complete output, some scenes quality-flagged | record it, continue, and pass `--accept-degraded-upstream` to every later stage |
| 2 | **REFUSED** — contract broken or checkpoint missing, **nothing written** | abort the chain and suppress the CVAT publish |

Once anything upstream is degraded the wrapper opts in for the rest of the chain
— a human still opted in *once*, by running the script, and every consumer records
`upstream.accepted_degraded_upstream: true` in its own manifest. It takes `flock`
on `<work_root>/.run_stages.lock`, tees to `<work_root>/logs/run_<timestamp>.log`,
rewrites each stage tree in place, and **deletes and recreates** the pipeline CVAT
tasks (human edits inside them do not survive a republish; the answer-key tasks
are never touched).

### 16.3 Individual stages

```bash
$PY pipeline/stage3_proposals/proposals.py --model-id "$YOLO11_CHECKPOINT" \
      --revision v8.3.0 --accept-degraded-upstream
$PY pipeline/stage4_masks/masks.py --revision 3c879f39826c281e95690f02c7821c4de09afae7 \
      --accept-degraded-upstream
$PY pipeline/stage5_lift/lift.py --accept-degraded-upstream
```

**Trial pattern — always do this before a full run.** One scene into a scratch
directory leaves the real work tree untouched:

```bash
$PY pipeline/stage3_proposals/proposals.py --scenes scene-0061 \
      --out-dir /tmp/trial_stage3 --revision <sha> --accept-degraded-upstream
$PY pipeline/stage4_masks/masks.py --stage3-dir /tmp/trial_stage3 \
      --out-dir /tmp/trial_stage4 --revision <sha> --accept-degraded-upstream
```

### 16.4 Evaluation, ablation and audit

```bash
$PY -m scripts.paint_metrics                       # the Phase-7 gate number
$PY -m scripts.eval_3d                             # 3D vs the human answer key
$PY -m scripts.eval_2d                             # 2D vs the projected answer key
$PY scripts/save_run_results.py --label <name>     # lift metrics out of work_root
$PY scripts/compare_runs.py --markdown             # tabulate Results/

scripts/run_matrix.sh                              # detector × re-ID, 4 clean-slate cells
scripts/run_matrix_track2d.sh                      # Stage 3b A/B, 5 clean-slate cells

$PY scripts/check_conformance.py                   # validate ledger + render CONFORMANCE.md
$PY scripts/collect_evidence.py                    # independent substrate re-measurement
$PY scripts/build_state.py                         # regenerate BUILD_STATE.md
```

`work_root` holds **exactly one run at a time** — the next `--clean-slate` erases
it. `save_run_results.py` is what lifts a run's numbers out **with the identity of
the models that produced them**, built from the stage manifests rather than from
the command line that invoked it.

### 16.5 Full-run cost

10 scenes / 404 keyframes / 2,424 images on an RTX 4090: Stage 3 ≈ 80 s (YOLO11x)
or ≈ 15 min (LLMDet at ~350 ms/frame), Stage 4 ≈ 6.5 min, Stage 5 ≈ 25 s,
Stage 7 ≈ 85 s. A clean-slate matrix cell is ≈ 12 min.

---

## 17. Roadmap to DhakaScenes v1

Ordered by what most changes the trustworthiness of the next release.

1. **Tests.** Convert the 135 PLAUSIBLE conformance rows into TEST or MEASUREMENT
   evidence, starting with the silent-failure assertions each module docstring
   already names: phrase-span totality, mask resolution round-trip, the `z ≤ 0`
   guard, DBSCAN determinism, near-face immobility, the time-base single-conversion
   rule. GPU-free unit tier + env-gated integration tier.
2. **Fix C24** — split the far surface along the camera ray *before* ε is applied,
   since no cluster-selection rule can separate an object from a wall that ε has
   already chained to it.
3. **Run the Stage 3b matrix to `Results/`**, closing C26 Gate (2) and C27's open
   gate with full-substrate numbers rather than a one-scene trial.
4. **Write back the tuned thresholds** from the tuning split, with provenance,
   closing the largest untuned quantity in the pipeline.
5. **One end-to-end run on the physical 3050ti** under a 4096 MiB cap with
   pilot-tier models, which is the only thing that can make the 4 GB claim
   quotable.
6. **Stage 9 end to end + the Phase-10 orchestrator** (`run_pilot.py`), producing
   [I-4] `prelabels.jsonl` for a full review cycle, and a corrected spatial gate
   computed on `box_measured`.
7. **Swap the substrate**: capture stack → I-1/I-2 real (PPK trajectory with
   published per-frame σ), open-vocabulary `proposal_2d` for the indigenous
   taxonomy, real S0 seed set replacing `nuscenes_gt_pilot` priors (and
   `assert_release_source()` earning its five lines).
8. **Density-stratified evaluation** on real ρ bins — the contribution `eval_region.py`
   has been holding the definition for since Phase 2.

---

## 18. Licence, citation, provenance of documents

**Code.** This repository's own source. See the repository licence file for terms.

**Substrate.** nuScenes v1.0-mini is © Motional, released under **CC BY-NC-SA
4.0** and **is not redistributable**. It is not in this repository and must be
obtained from nuscenes.org. GT-derived statistics — including
`priors_pilot_v0.json` — are licence-covered derived data (decision C12) and are
written outside the repository tree.

**Model weights** are each governed by their own upstream licence; `facebook/sam3`
and `facebook/sam3.1` are **gated repositories** requiring an access grant.

**Governing documents and their precedence** are listed in
[`docs/BUILD_PROMPT.md`](docs/BUILD_PROMPT.md) §1. In short:
[`docs/comprehensive.md`](docs/comprehensive.md) is the full system and paper
plan; [`docs/Annotation_pipeline.md`](docs/Annotation_pipeline.md) is the
annotation-pipeline reference design;
[`docs/pilot_plan.md`](docs/pilot_plan.md) (rev 2) is the pilot bible;
[`docs/DECISIONS.md`](docs/DECISIONS.md) records where they conflict and which
one won.

**Citation.** The DhakaScenes paper is in preparation. Until it appears, cite this
repository by name, version tag (`v1.0.0`) and commit hash — the commit is what
binds a number to the code that produced it.

---

## Appendix A — metric definitions

Every metric below is computed by the named script, whose output JSON carries its
own `caveat` string and the class-space block of the run being scored.

| Metric | Script | Definition |
|---|---|---|
| `paint_inside_gt_rate` | `paint_metrics.py` | painted LiDAR points inside **any** GT 3D box of the same keyframe ÷ painted points |
| `base_rate_all_points` | `paint_metrics.py` | **all** single-sweep points inside any GT box ÷ all points |
| `enrichment` | `paint_metrics.py` | `paint_inside_gt_rate ÷ base_rate_all_points` |
| `class_correct_rate` | `paint_metrics.py` | painted points inside a GT box whose category maps back to the predicted phrase ÷ points painted with that phrase |
| `gt_coverage.rate` | `paint_metrics.py` | eligible GT boxes receiving ≥ 5 painted points ÷ eligible GT boxes. Eligible = centre ≤ 40 m **and** `num_lidar_pts ≥ 5` |
| `precision_localization` (3D) | `eval_3d.py` | predicted boxes matching **any** GT box ÷ predicted boxes. Match = greedy BEV centre distance ≤ 2 m, score order, per keyframe |
| `precision_class_aware` (3D) | `eval_3d.py` | as above **and** the GT category maps to the predicted phrase |
| `gt_recall` (3D) | `eval_3d.py` | matched eligible GT boxes ÷ eligible GT boxes **of reachable classes** (C25) |
| `ate_m_mean` | `eval_3d.py` | mean BEV centre error over matched pairs (m) |
| `ase_mean` | `eval_3d.py` | mean (1 − IoU after aligning centre and yaw) — pure size error |
| `aoe_rad_mean_trusted_yaw` | `eval_3d.py` | mean \|yaw error\| mod π over matched pairs whose yaw is **not** `yaw_ambiguous` |
| `precision_localization` (2D) | `eval_2d.py` | deduplicated predictions matching any GT box at the stated IoU ÷ predictions |
| `gt_recall_all` / `gt_recall_visible` | `eval_2d.py` | matched GT ÷ all GT, and ÷ GT with area ≥ 32×32 px |
| `iou_sweep` | `eval_2d.py` | all four 2D quantities at IoU 0.3 / 0.5 / 0.75, because the answer key is amodal and the detector modal |
| ρ (density) | `eval_region.py` | object count ÷ area of the evaluation region E under the recorded `coverage_config` |
| `inflation_fraction` | `stage8_inflate` | fraction of the shipped box volume attributable to inflation rather than measurement |

---

## Appendix B — glossary

| Term | Meaning |
|---|---|
| **Substrate** | the dataset standing in for DhakaScenes — here nuScenes v1.0-mini |
| **Keyframe** | an annotated 2 Hz sample; every pipeline output lands at one |
| **Sweep frame** | a non-keyframe sensor reading; cameras at 12 Hz. Connective tissue for 2D identity only |
| **Phrase** | the natural-language class name (`"a car"`) that every stage keys on |
| **Reachable / unreachable** | whether the configured detector's source vocabulary can produce a given phrase at all |
| **E** | the evaluation region; R1 = 110° frontal wedge, R2 = full annulus (the pilot's choice) |
| **ρ** | object density, count over the area of E — the basis of density-stratified evaluation |
| **W_acc** | the sweep accumulation window (count and duration, both recorded) |
| **Painted point** | a LiDAR point assigned to a mask instance by Stage 5; it keeps its Stage 1 ego coordinates |
| **Ghost** | reprojected far-surface points inside an object's mask (the wall behind the car) |
| **Amodal box** | a box covering an object's full extent including occluded parts |
| **Near / far face** | the box face pointing at the sensor / away from it; inflation holds the near face still |
| **`yaw_ambiguous`** | a near-square BEV footprint whose heading axis the fitter cannot determine |
| **Degraded** | complete output that is quality-flagged; consumable only under an explicit, recorded opt-in |
| **Tier** | a label's confidence class, deciding whether it is auto-accepted or routed to a human |
| **Recovered box** | a Stage 3b box no detector emitted, carried in from a neighbouring keyframe by propagation, with `source`, `hops` and a decayed score |
| **CODE-SITE / TEST / MEASUREMENT / ARTIFACT** | conformance evidence classes; only the last three can support `CONFORMS` |
