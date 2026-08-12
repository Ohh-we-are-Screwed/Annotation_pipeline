# DhakaScenes Pilot — Pre-Implementation Technical Audit

**Reviewer role:** independent audit. No code written, no redesign proposed.
**Documents traced:** `comprehensive.md` (v1.0, authoritative), `GAP_ANALYSIS.md` (2026-08-07 scan), `Annotation_pipeline.md` (superseded), `pilot_plan.md` (under review).
**Unresolved inputs treated as placeholders:** `NUSCENES_DATAROOT`, `PILOT_OUTPUT_ROOT`.

---

## Executive Verdict

# READY WITH REQUIRED CHANGES

The architecture is sound and should not be redesigned. Contracts-first ordering (`pipeline/common/` before any stage), the role-registry indirection, the pilot-only Stage 0 probe, and the "name the silent-failure mode before writing the test" discipline are all correct instincts, and several of them are better than what the governing spec itself specifies.

The plan is **not** ready for stage code, for one structural reason: **it specifies stage responsibilities but almost never specifies the representation those stages exchange.** Coordinate frame, time base, 2D pixel space, cloud identity (single-sweep vs accumulated), clustering scope, and units are left implicit at nearly every stage boundary. Every one of those gaps produces output that runs, looks plausible, and is wrong — which is precisely the failure class the pilot exists to rule out.

Nine issues are classified P0. **All nine are resolvable inside `pipeline/common/` — the module the plan already sequences first.** None requires re-architecting. That is why this is not a NOT READY verdict. But P0-1 through P0-9 must be closed as written decisions in `conventions.py` / `schemas.py` before Stage 1 code is written, not discovered during it.

Two further points frame the whole review:

1. **The pilot's substrate is not neutral.** nuScenes' 2 Hz keyframe rate, its ~-90° LiDAR-to-ego frame rotation, its per-`sample_data` ego poses, its already-undistorted imagery, and its microsecond Unix timestamps each differ from what `comprehensive.md` assumes. Several of these silently change what a stage is testing. The plan anticipates the taxonomy substitution and the partial-blob problem well; it does not anticipate the geometric and temporal substitutions at all.
2. **This pilot is one document away from a publication path.** `comprehensive.md` §11.6 pivot B is literally "the annotation-pipeline paper, validated on nuScenes." Any number this pilot produces against nuScenes ground truth is therefore at risk of being read as pivot-B evidence — while being derived from priors extracted from that same ground truth. The claim-hygiene section (§30 below) is not ceremonial.

---

## A. Governing-Spec Alignment

| Area | `comprehensive.md` requires | Pilot proposes | Status | Note |
|---|---|---|---|---|
| Stage set | §7.3 stages 1–10 | Stages 0–9; stage 10 absent | **PARTIALLY ALIGNED** | Attribute pre-fill silently dropped; see M-1 |
| Stage order | 1→10 fixed | 0→9, same order | **ALIGNED** | Stage 0 is valid pilot-specific scaffolding |
| I-1 | MCAP bundle + `session_meta.json` + calib id | nuScenes `sample_data` via allowlist | **PILOT-SPECIFIC AND VALID** | Shape preserved; see C-1 for the missing manifest analogue |
| I-2 | PPK+RTS only; σ_pos, σ_yaw, fix_type, stationary | nuScenes `ego_pose.json`, "honestly downgraded" | **PARTIALLY ALIGNED** | nuScenes supplies *none* of the quality fields. Downgrade is asserted, not mechanised — P1-6 |
| I-3 | keyframes + **single-sweep AND accumulated** clouds + undistorted images + Δt per association | keyframe pack + accumulated cloud only | **CONTRADICTION** | P0-3 |
| I-4 | prelabels + gate vector + tier, `source:"pipeline"` | same | **ALIGNED** | Schema fine; enforcement path is not — P0-7 |
| I-5 | CVAT → verified; val/test + `pipeline_accepted` = build error | simulated only | **PILOT-SPECIFIC AND VALID** | But the pilot has no split field, making the test vacuous — P0-7 |
| I-6 / I-7 | release schema; results CSV + **config snapshot + seed** | shape defined, not exercised | **PARTIALLY ALIGNED** | I-7's config+seed requirement is the reproducibility contract the pilot needs *now* — P1-8 |
| Frames (§2.3) | ISO 8855 body frame, x fwd / y left / z up | `conventions.py` named, never defined per stage | **MISSING** | P0-1 |
| Time (§2.3) | GPS time, int64 **ns** | nuScenes µs Unix, no `time_base` | **CONTRADICTION** | P1-7 |
| Units (§2.3) | m, rad; yaw about +z from +x | not restated; A.3 uses `sigma_yaw_deg` | **PARTIALLY ALIGNED** | Inherited inconsistency in the spec itself — P2-4 |
| Eval region E (§3.6) | R1 or R2, coded not prose; `coverage_config` on every table | `eval_region.py` planned; R1/R2 never chosen | **MISSING** | P1-4 |
| Ground removal (§7.3.1) | sector RANSAC over accumulation, \|z−g\|<0.3, >40 m, >4 m | identical | **ALIGNED** | Parameters copied without pilot validation or diagnostics — P1-13 |
| Lift (§7.3.5) | ground-filtered **single-sweep**; per-camera frusta unioned | accumulated cloud; no union rule | **CONTRADICTION** | P0-3, P0-5 |
| Masks (§7.3.4) | SAM 2.1 + propagation + **IoA-NMS > 0.5 across cameras** | MobileSAM, no propagation, no IoA-NMS | **PARTIALLY ALIGNED** | Propagation loss is declared and acceptable; IoA-NMS omission is undeclared — M-2 |
| Tracking (§7.3.7) | IoU × DINOv2, ICP, **forward-backward smoothing**, **yaw consistency along tracks** | IoU × DINOv2, ICP, Kalman fallback | **PARTIALLY ALIGNED** | Both smoothing terms dropped undeclared; yaw consistency is the 180°-flip defence and is free — M-3 |
| Priors (§7.2) | S0-derived; nuScenes/KITTI values "initialization only, deleted after S0" | nuScenes GT as S0-equivalent; draft ε table as "starting point" | **PILOT-SPECIFIC AND VALID**, with leak risk | Valid substitution; needs a `source` marker and a release-time guard — P1-11 |
| S1 separability | on S0, indigenous classes, pass criteria | explicitly *not* attempted; informational probe only | **ALIGNED** | Correctly scoped. Containment is under-specified — P1-10 |
| QA (§7.3.9, §7.4) | gate vector → tier; nothing ships as GT | same + F1 invariant test | **PARTIALLY ALIGNED** | Gate inputs change meaning upstream — P0-3, P1-12 |
| Model policy (§7.1) | open-weights, locally runnable, `[VERIFY]` before quoting | roles + `verified:` flag | **ALIGNED** | Good. The plan's own §5 table then violates it — P0-8 |
| Group boxes (§6.3) | `group_annotation`, ρ uses `n_min` | dataclass listed; no producer/consumer | **MISSING** | Declare out of scope — M-4 |

**Does the pilot still test the same pipeline shape?** Yes for stages 0–4 and 8–9. **Not yet** for stages 5–7: the accumulated-cloud substitution (P0-3), the missing two-pose reprojection chain (P0-2), and the 2 Hz association regime (P1-1) each change what those stages are actually exercising, not merely how well.

---

## B. Contract-Spine Audit (I-1 → I-7)

### I-1 — capture → everything
Produced by nuScenes blob + Stage 0. Consumed by Stage 1. Required: resolvable per-channel file paths, `calibrated_sensor`, `ego_pose`, `sensor`. **Available?** Yes. **Substitution valid?** Yes. **Gap:** `comprehensive.md` I-1 says "a bag missing a mandatory topic fails ingestion" — the pilot's analogue is the required-channel set, and it is never enumerated (P0-6). `session_meta.json` has no pilot equivalent; the dataroot fingerprint should fill that role (P0-9).

### I-2 — trajectory → ingestion/release
Produced by nuScenes `ego_pose.json`. Consumed by Stage 1 (accumulation), Stage 5 (reprojection), Stage 7 (velocity).
**Fields required:** `t`, translation, rotation, `σ_pos`, `σ_yaw`, `fix_type`, `stationary_flag`.
**Actually available from nuScenes:** translation, rotation, timestamp. **Nothing else.**
**Could a downstream stage read a pilot field as production-quality?** *Yes — this is the sharpest instance in the plan.* If σ fields are filled with `0.0` to satisfy the dataclass, every consumer reads "perfect pose." They must be `null` with a mandatory `pose_source: "nuscenes_ego_pose"` and `quality_known: false`, and any consumer must fail closed on `null` rather than defaulting. `stationary_flag` can be honestly derived from ego speed; `fix_type` cannot and must stay `null`.
Also semantic: nuScenes "global" is a per-map arbitrary frame, not the published-origin local ENU of §2.3. Harmless in-pilot, misleading if the field name is reused verbatim.

### I-3 — ingestion → annotation
**Required:** `kf_id`, `t`, per-sensor sample refs, **Δt per association**, single-sweep `.bin`, accumulated `.bin`, accumulation window per cloud, undistorted images.
**Gaps:** single-sweep cloud not produced (P0-3); accumulation window recorded is required by spec and not mentioned by the plan — it matters because clouds at scene starts have fewer sweeps than the nominal window, changing point density and therefore cluster size and box dimensions at every scene boundary. Undistortion is a **no-op on nuScenes** (imagery ships rectified, `calibrated_sensor.json` carries `camera_intrinsic` only, no distortion coefficients) — so a production-path component goes entirely untested and must be recorded as `undistorted: true, method: "nuscenes_native"` rather than silently passing through.

### I-4 — annotation → CVAT
**Required (A.1):** token graph, translation, size `[w,l,h]`, rotation **quaternion**, velocity, `attribute`, `visibility`, `num_lidar_pts`, provenance{source, gates, tier}.
**Gaps:** Stage 6 produces a scalar yaw; A.1 wants a quaternion — an unwitnessed conversion point where an axis or sign error hides invisibly (needs a yaw→quat→yaw round-trip test). `attribute` has no producer (M-1). `num_lidar_pts` is semantically undefined in the pilot: nuScenes' own `num_lidar_pts` counts single-sweep points *including ground*; if the pilot counts ground-removed accumulated points the same field name carries a ~5× different quantity, and the §7.3.9 "≥5 returns" gate loosens accordingly (P0-3). `visibility` has no producer.

### I-5 — CVAT → release
Simulated. The invariant to enforce is not only the spec's *val/test × `pipeline_accepted`* rule but also its converse: `source ∈ {human_verified, human_created} ⇒ verified_by is not None ∧ verification_pass ≥ 1`. The plan states the first and tests neither positively (P0-7).

### I-6 / I-7
Shape-only, correct scoping. I-7's "config snapshot + seed" is the reproducibility requirement the pilot needs immediately (P1-8).

**Cross-cutting contract gap:** no contract declares its **coordinate frame** or **time base** as a field. Adding `frame: "ego" | "lidar" | "global"` and `time_base: "unix_us" | "gps_ns"` to every geometric and temporal record turns six of the P0/P1 findings below into assertion failures instead of silent corruption. This is the single highest-leverage change in the review.

---

## Critical Findings — P0 (blocking)

### P0-1 — No stage declares its coordinate frame, and nuScenes' LiDAR frame is not the ego frame

**Location:** `common/conventions.py` (planned), Stages 1, 5, 6, 7, 8.
**Problem:** `comprehensive.md` §2.3 fixes the body frame as ISO 8855 (x forward, y left, z up) and yaw about +z from +x. The pilot plan names `conventions.py` but never states, for any stage, which frame its points, boxes, or yaw values live in. nuScenes `LIDAR_TOP` is **not** aligned to the ego frame: its `calibrated_sensor` rotation is approximately a −90° yaw (quaternion ≈ `[0.708, −0.006, 0.011, −0.706]` — verify against your blob), i.e. the sensor's **+x points to the vehicle's right and +y points forward**.
**Why it matters:** every one of these runs cleanly and produces wrong output:
- Eval region E under R1 (`|θ| ≤ 55°` from +x) applied to raw LiDAR-frame points selects the **right-hand sector of the vehicle**, not the frontal sector. Every downstream count, ρ value, and per-region claim is then about the wrong 110° of the world.
- Stage 6 yaw is off by 90°, so Stage 8 inflates length along the width axis and boxes grow sideways into adjacent lanes.
- Stage 7 velocity vectors are rotated 90°.
- Ground removal survives (z is up in both frames), so the one filter that *would* have crashed does not.
**Evidence:** `comprehensive.md` §2.3, §3.6; nuScenes `calibrated_sensor.json` for `LIDAR_TOP`; pilot plan §1 line 34 and §3 Stages 1/5/6, which name no frame anywhere.
**Resolution:** In `conventions.py`, declare **ego frame is canonical for all cross-stage geometry**; every artifact carries an explicit `frame` field validated on write; Stage 1 transforms LiDAR points to ego once, and no later stage re-applies `T_ego_lidar`. Add a test that loads one real `calibrated_sensor` record for `LIDAR_TOP` and asserts the rotation is *not* identity — a test that would fail loudly the day someone assumes it is.

### P0-2 — Reprojection needs two ego poses; the plan implies one static extrinsic

**Location:** Stage 5; Stage 1 accumulation.
**Problem:** The plan describes Stage 5 as "projects points through calibrated intrinsics/extrinsics into each camera frame." In nuScenes, **every `sample_data` record has its own `ego_pose`**, and within one `sample` the LiDAR and each camera are captured tens of milliseconds apart. The correct chain is `lidar → ego(t_lidar) → global → ego(t_cam) → camera → pixel`. Dropping the two middle hops — the natural reading of "calibrated intrinsics/extrinsics" — omits the ego motion between the two timestamps.
**Why it matters:** at 50 km/h and a 30 ms offset the ego moves ~0.4 m. The projection still lands on the image, still hits masks, still paints points. Near-range objects acquire a systematic several-pixel offset, points bleed across mask boundaries onto neighbouring objects, and the resulting clusters are biased along the direction of travel. This is the canonical nuScenes reprojection bug and it is **invisible to the plan's proposed test**, which uses a single synthetic point and one static calibration.
**Evidence:** nuScenes per-`sample_data` `ego_pose` records; `comprehensive.md` §3.5.4 makes the LiDAR frame the anchor and requires the real Δt be stored precisely because it is nonzero; pilot plan §3 Stage 5.
**Resolution:** Specify the four-hop chain in `conventions.py` as the single implementation of `project_lidar_to_image()`. Add the validation test in T-5 below (project nuScenes GT box centres and assert they land inside the object) — a synthetic point test cannot catch a directionally-plausible transform.

### P0-3 — Stage 5 consumes the accumulated cloud; the spec says single-sweep

**Location:** Pilot §3 Stage 1 ("the ground-filtered cloud is what downstream stages, especially Stage 5's reprojection, consume") vs `comprehensive.md` §7.3.5 ("project ground-filtered **single-sweep** points") and I-3 (which mandates *both* clouds).
**Problem:** An accumulated cloud is ego-motion-compensated only. Dynamic objects smear across the accumulation window. Painting a smeared cloud with a single-frame mask and clustering the result produces elongated clusters for every moving object.
**Why it matters, three ways:**
1. Box length for moving vehicles is systematically inflated along the direction of travel — plausible boxes, wrong dimensions, and worse for fast objects, which reads as a "hard cases are harder" result rather than a bug.
2. Yaw from L-shape fitting is biased toward the motion direction, corrupting the exact quantity `comprehensive.md` §7.3.7 and §8.2.1 (APH) treat as load-bearing.
3. The §7.3.9 QA gate "LiDAR-return ≥ 5" and §6.3's "≥ 5 points in the **single-sweep** frame" both become ~5× looser. The gate still fires, still tiers boxes, and no longer means what the spec defines.
**Resolution:** Produce both clouds per I-3. Fit the ground plane on the accumulation (spec-faithful) but **apply the resulting plane to the single-sweep cloud** and lift from single-sweep. Count `num_lidar_pts` on single-sweep, pre-inflation. Keep the accumulated cloud for visualisation and the secondary track only. If the pilot deliberately deviates, it must be recorded as a deviation, not inherited by silence.

### P0-4 — The 2D coordinate space between Stages 3, 4 and 5 is undefined

**Location:** Stage 3 output → Stage 4 input → Stage 5 mask lookup.
**Problem:** nuScenes imagery is **1600×900**. Grounding DINO emits boxes in normalised `cxcywh` at its own resized input scale; MobileSAM takes box prompts in a 1024-longest-side transformed space and returns masks at 1024 before postprocessing. The plan's §5 VRAM fallback proposes "640×640 instead of 800×800" — a **square** resize of a 16:9 image. Nothing in the plan states what coordinate space a Stage 3 box is in, whether aspect ratio is preserved, or that Stage 5 must index masks in original 1600×900 pixels.
**Why it matters:** a non-aspect-preserving resize whose inverse assumes uniform scale stretches every box along one axis. The boxes still look like boxes, SAM still returns masks, points still get painted — and every object's 3D extent is wrong in one axis, consistently. This is exactly the plan's own §26-class "Stage 4 changes image coordinates before Stage 5 uses them," and it is currently unguarded.
**Resolution:** Declare **absolute pixel coordinates in original image resolution** as the contract at every 2D boundary; require each model adapter to own its own forward and inverse transform and to return original-resolution outputs; forbid square resizes of non-square imagery (letterbox with recorded padding, or resize-shortest-side). Add a test asserting a box at a known original-resolution location survives Stage 3 → Stage 4 round-trip within a pixel.

### P0-5 — DBSCAN scope is ambiguous: per-mask-instance or per-class-per-frame?

**Location:** Stage 6; inherited ambiguity from `comprehensive.md` §7.3.6 and `Annotation_pipeline.md` §3.5.
**Problem:** "Class-conditional DBSCAN on the reprojected points, keep largest cluster" admits two readings. **(a)** Cluster the points of each mask instance separately, using that class's ε, and keep the largest cluster to discard reprojection ghosts. **(b)** Pool all points of a class across the frame, cluster once, and keep the largest cluster.
**Why it matters:** reading (b) yields **one box per class per frame**. In a nuScenes parking row or a gridlock scene — the exact regime this project is about — adjacent cars merge into one component under any ε large enough to hold a single car, so (b) both merges neighbours and discards every other instance of that class in the frame. Output is clean, well-formed, and silently missing most objects. The two readings differ in output count by an order of magnitude.
**Resolution:** Fix reading (a) explicitly in the contract: DBSCAN runs **per mask instance**, "class-conditional" refers to which ε is used, and "keep largest cluster" is the ghost-point filter. Add a test with two same-class instances 1 m apart asserting two boxes, not one. Also fix a deterministic tie-break for equal-size clusters — DBSCAN label order depends on input point order, which depends on file read order.

### P0-6 — Stage 0's probe is necessary but not sufficient

**Location:** Stage 0; pilot §0.1, §7 Stage 0 test.
**Problem:** Cross-referencing `scene.json` / `sample.json` / `sample_data.json` against files on disk catches missing files and nothing else. The failure modes it misses are the ones that reach Stage 5 before surfacing:
- **Referential integrity is unchecked.** The probe never validates `sample_data → ego_pose`, `sample_data → calibrated_sensor`, `sample → sample_annotation → instance → category`, or the `prev`/`next` chains. A mismatched or partially-extracted metadata tarball surfaces as a `KeyError` inside Stage 5, not at Stage 0. Stage 8's priors also depend on `sample_annotation`, which the probe does not look at at all.
- **Sweeps are not mentioned.** Stage 1's accumulation reads `sweeps/LIDAR_TOP/*.pcd.bin`. If "every sensor file resolves" means keyframe records only, accumulation hits missing sweeps mid-run — the precise failure Stage 0 exists to prevent.
- **File integrity is unchecked.** A truncated `.pcd.bin` whose length is still a multiple of 20 bytes reshapes to `(-1, 5)` without error and yields silently fewer points. A truncated JPEG often decodes to a partial image. Both produce plausible downstream output.
- **The required channel set is never enumerated.** nuScenes has 6 cameras, 1 LiDAR and 5 RADARs. If "every sensor file" includes RADAR, scenes may be excluded for irrelevant reasons; if it silently means "the cameras I happen to use," the allowlist does not mean what its name says.
- **Metadata/blob co-location is assumed.** The meta tarball extracts `v1.0-trainval/*.json`; blobs extract `samples/`, `sweeps/`, `maps/`. If they were extracted to different roots the plan has no way to express that.
**"What must exist for a scene to be fully usable by every downstream stage?"** The plan does not answer this, and the answer is the definition of `fully_present`. It should be: every keyframe's records for the **declared required channel set**, plus every LiDAR sweep within `W_acc` of every keyframe, plus every referenced `ego_pose` / `calibrated_sensor` token, plus `sample_annotation`+`instance`+`category` closure for every sample, plus a size/parse check on each file, plus an intact `prev`/`next` chain. Anything less is `partially_present`.
**Resolution:** Make the bucket definitions executable predicates in the probe, one per condition, and have the probe emit *which* predicate failed per scene rather than a bare bucket label. Record the required-channel set and `W_acc` used, in the output, since both change the answer.

### P0-7 — The provenance invariant is not enforceable as designed

**Location:** `common/schemas.py`, Stage 9.
**Problem:** Three independent gaps:
1. **The plan chooses non-raising validators** ("returns a list of errors rather than raising — so a stage can check-then-decide"). The most likely bypass of any invariant is then simply not inspecting the returned list. The plan's own §2 rationale — that retrofitted invariants "quietly don't get enforced somewhere" — argues against its own design choice here.
2. **The pilot has no split field populated.** The stated invariant is "no `pipeline_accepted` in `val`/`test`." With every record's split `None`, the invariant is *vacuously true* and the proposed Stage 9 test passes against a completely broken validator.
3. **The invariant the user actually needs is a different one.** Preventing `pilot-generated → auto_accept → human_verified` requires `source ∈ {human_verified, human_created} ⇒ verified_by ≠ null ∧ verification_pass ≥ 1`, which the plan never states. Note also that the plan's §3 Stage 9 prose conflates `tier` (`auto_accept`) with `source` (`pipeline_accepted`) — they are different fields with different rules.
**Indirect bypass paths:** serialization is the open one. If records round-trip through plain dicts or JSON and are reconstructed downstream without revalidation, or if a `GroupAnnotation`/release-builder path constructs records from dicts, the validator is never consulted.
**Resolution:** Keep `validate() → [errors]` for check-then-decide, and add a single `write_records()` boundary that **raises** on any invalid record — all persistence goes through it, on write *and* on read-back. Add a pilot-wide `allow_human_provenance: false` config that makes any human-provenance record a hard error. Test **positively**: construct a `val` + `pipeline_accepted` record and assert it is rejected; construct `human_verified` with `verified_by=None` and assert it is rejected.

### P0-8 — The VRAM budget rests on two wrong figures and a measurement API that under-reports

**Location:** §5 budget table; §8 build step 3.
**Problem — the numbers:**
- **Grounding DINO Tiny is listed at "~172 MB" weight file.** Grounding DINO-T has **~172 M parameters**; the FP32 checkpoint is roughly **690 MB**. This looks like a parameters→megabytes transcription error, and it is a 4× understatement of the largest model in the budget — the one the plan itself identifies as the binding constraint.
- **"SAM-ViT-B (~375M params, ~360MB weights)"** — SAM ViT-B is **~91 M parameters** with a ~375 MB checkpoint; params and megabytes have been swapped. The stated "9× parameter difference" versus MobileSAM's ~10 M happens to be right, so the *decision* survives, but the *justification as written* is wrong.
- DINOv2 ViT-S/14 (~21 M, ~85 MB) and MobileSAM (~10 M, ~40 MB) are correct.
**Problem — the method:** the plan mandates validating with `torch.cuda.max_memory_allocated()`. That counter excludes the CUDA context (~300 MB), excludes reserved-but-unallocated allocator blocks, excludes cuBLAS/cuDNN workspaces, and excludes other processes on the card. On a 4 GB device the gap between "allocated" and "actually occupied" is exactly the margin being budgeted. A checkpoint that "passes" this check can still OOM.
**Why it matters:** this check is build step 3 and gates checkpoint selection. A wrong input and a wrong meter produce a confident, documented, wrong go-decision.
**Resolution:** Correct both figures with the `verified:` flag the plan already designed. Measure with `torch.cuda.max_memory_reserved()` **and** `torch.cuda.mem_get_info()` free-delta around a representative forward pass at the **real** input resolution and **real** 23-phrase prompt, on the actual card, with the display-server allocation measured first. State the hard-stop behaviour when measured > `hard_ceiling_mb` (the plan defines the ceiling but not the consequence). Recommend `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` given the variable-length text tensors.

### P0-9 — The dataroot/output-path contract does not exist

**Location:** §1 directory layout (`data/raw/`, `data/work/`, `data/out/`), §8 build order.
**Problem:** The plan expresses the path model as a comment in a directory tree — "pointer/symlink to your real nuScenes dataroot, never copied in." Once `NUSCENES_DATAROOT` and `PILOT_OUTPUT_ROOT` are supplied, nothing specifies how they enter the code, how they are validated, or what prevents the failure modes the layout is trying to avoid. Specifically unaddressed:
- No startup validation that the dataroot contains `v1.0-trainval/`, `samples/`, `sweeps/`, is readable, and matches the expected metadata version (mini metadata against trainval blobs produces zero token matches, which reads as "no usable scenes").
- No separation of `dataroot` from `meta_root` for the split-extraction case.
- **No staleness guard on `usable_scenes.json`.** It is computed against one dataroot and consumed by nine stages. Pointed at a different or fuller dataroot, it silently excludes good scenes or admits missing ones. It must record the dataroot realpath and a metadata fingerprint, and every consumer must verify the match.
- Nothing prevents accidental copying. A symlink inside the project tree is copied by `shutil.copytree`, `rsync -L`, or a Docker build context. `data/work/` under the repo can fill the disk with intermediate clouds.
- Nothing asserts the write roots are disjoint from the dataroot.
**Resolution:** A single `configs/paths.yaml` (or env vars) resolved once into a `Paths` object; a `validate_paths()` that checks structure, readability, version-directory name, and `commonpath` disjointness between dataroot and both write roots; all dataroot access read-only by convention with a test asserting no module opens a dataroot path for writing; `data/` git-ignored and work/out defaulting **outside** the repo; the dataroot fingerprint recorded in every stage manifest. Changing either path must then be a config edit with no pipeline-logic change — which is the property the plan intends but does not yet specify.

---

## Critical Findings — P1 (resolve before the first full pilot run)

### P1-1 — 2 Hz keyframes break the 3D-IoU association regime the pilot is meant to test
`comprehensive.md` assumes a 10 Hz canonical rate. nuScenes keyframes are **2 Hz**. A vehicle at 40 km/h moves ~5.5 m between keyframes — 3D IoU between consecutive detections of the same object is **zero** for most vehicles. The association term `3D IoU × DINOv2 cosine` therefore collapses to appearance-only, and the IoU half of the mechanism is never exercised at all. Worse, appearance is also degraded: distant nuScenes crops are a few tens of pixels, and DINOv2's patch-14 tokenisation reduces a 28×28 crop to a 2×2 patch grid, so cosine similarity approaches degeneracy exactly where IoU has already failed.
This is a pilot shortcut that invalidates what the stage is supposed to prove. **Resolution:** either (a) predict-then-match — propagate the previous box by its estimated velocity before computing IoU, and state that the gate threshold is derived for 2 Hz, or (b) run Stage 7 over nuScenes' 12 Hz camera cadence with interpolated LiDAR anchors. Either way, record the effective inter-frame Δt in the track record and make the IoU gate a config value with documented provenance, not a spec-copied constant. Also state a minimum crop size below which appearance similarity is not trusted.

### P1-2 — The shared-DINOv2 rationale contradicts the serial-execution policy
§4.2 mandates "serial, one role loaded at a time, cache cleared between stages." §4.4 says loading DINOv2 twice would "burn ~700 MB" and calls this "the single easiest way to blow the 4 GB budget." **Both cannot be true.** Stage 2 and Stage 7 are separated by Stages 3–6; under the serial policy the Stage 2 instance is freed long before Stage 7, so two copies can never be resident, and the marginal cost of reloading is a few seconds of disk I/O, not 700 MB. Conversely, if the shared-instance concern is real, the pipeline is not stage-sequential — which contradicts the staged-artifact architecture and the build order.
**Also — "same role instance" hides an interface difference.** `embedding_ood` embeds a whole frame (global/CLS token, ~518×518 input); `reid_embedding` embeds a small object crop (ideally mask-pooled patch tokens). Same weights, **different preprocessing and different output semantics**. If the registry hands back "the DINOv2 model" and Stage 7 inherits Stage 2's transform, tiny crops get upsampled 20× and similarity is dominated by interpolation artifacts.
**Resolution:** Drop the zero-marginal-cost claim; keep the registry's singleton behaviour as a convenience only. Make preprocessing part of the **role** interface, not the model, so `embedding_ood` and `reid_embedding` are distinct interfaces that happen to share weights. Add an explicit teardown that asserts `torch.cuda.memory_allocated() ≈ 0` between stages — `empty_cache()` alone does not free memory still referenced by a live Python object, and the registry can enforce singletons but cannot enforce ceilings.

### P1-3 — Role interfaces will not survive the most likely production swap
`GAP_ANALYSIS.md` §6 recommends revising `comprehensive.md` §7.1 to prefer **SAM 3/3.1 as a unified proposal + mask + track engine**, with Grounding DINO retained only as a comparison arm. If that happens, one model fills `proposal_2d`, `mask_2d`, and part of `reid_embedding` — and the pilot's one-model-per-role registry cannot express it. The "swap models later = edit which YAML the registry loads" claim fails for the single most likely production choice.
Three concrete places "same role" ≠ "same interface":
- **`mask_2d`:** MobileSAM is `(image, boxes) → masks`. SAM 2.1 is `(video, boxes, memory_state) → masks over time` with a propagation window and state re-init at block boundaries (`comprehensive.md` §7.3.4). A per-image interface makes the production swap a Stage 4 rewrite.
- **`proposal_2d`:** Grounding DINO returns normalised `cxcywh` + per-token logits requiring phrase-span mapping. SAM 3 returns concept-prompted instance masks with scores, and DINO-X-class models return box+mask+caption jointly — collapsing the Stage 3/Stage 4 boundary.
- **`embedding_ood` / `reid_embedding`:** see P1-2.
**Resolution:** Define `mask_2d` with an **optional temporal-state parameter and window size from day one**, which MobileSAM's adapter ignores. Allow one provider to register against multiple roles (composite provider). Allow `proposal_2d` to optionally return masks, with Stage 4 becoming a pass-through when it does. None of this costs anything now; all of it is a rewrite later.

### P1-4 — R1/R2 coverage config and the camera set are undecided
`comprehensive.md` §2.3 requires `coverage_config ∈ {R1, R2}` on every release table, and §3.6 defines E differently for each. The pilot plan mentions neither, and never says whether Stages 2–5 run on all six nuScenes cameras or only `CAM_FRONT`. This single unmade decision determines: the eval region and therefore ρ's area normalisation; whether Stage 5 needs the multi-camera union and IoA-NMS at all; what "OOD frame" means in Stage 2 (a frame? a camera image?); and roughly a 6× difference in GPU time. nuScenes' 6-camera ring is closest to R2, which is also `comprehensive.md`'s default. **Resolution:** decide explicitly, record `coverage_config` in every output record, and derive E from it in `eval_region.py` rather than from a constant.

### P1-5 — Priors, threshold tuning, and the pilot run all draw on the same scenes
The pilot derives dimension priors and DBSCAN ε from nuScenes `sample_annotation.json` (§6), tunes the Grounding DINO threshold "empirically on a handful of pilot frames" (§3 Stage 3), and then runs end-to-end over usable scenes — with no statement that these three sets are disjoint. `GAP_ANALYSIS.md` §3 names this exact pitfall ("tuning prompts on S0 and evaluating on S0. Split S0 for prompt-tuning vs. S1 measurement"), and the pilot reproduces it.
For plumbing alone this is tolerable; it becomes unacceptable the moment any number is compared to nuScenes GT, because Stage 8 inflates boxes toward means derived from the same boxes being scored. **Resolution:** partition `usable_scenes.json` into disjoint `priors`, `tuning`, and `run` subsets at Stage 0, record the partition and its seed in the manifest, and carry the habit into production. Cheap now, structural later.
Related: nuScenes GT includes boxes beyond the 40 m eval cap and boxes with zero LiDAR points. If priors are derived from the full annotation population but the pipeline only ever sees objects inside E within 40 m, the prior population and the observable population differ — inflation then pulls toward a mean the sensor never measures. Derive priors under the same E and range constraints the pipeline operates under.

### P1-6 — nuScenes ego poses carry none of I-2's quality fields
See I-2 above. The plan says the quality field is "honestly downgraded, not faked as PPK," which is the right intent, but there is no mechanism — and the failure mode is filling `σ_pos = 0.0` to satisfy a dataclass, which reads downstream as *better* than PPK. **Resolution:** `null` + `pose_source` + `quality_known: false`, consumers fail closed.

### P1-7 — Time base, units, and the Δt sanity threshold
nuScenes timestamps are **microseconds, Unix epoch**. `comprehensive.md` §2.3 mandates **int64 nanoseconds, GPS time**. Two consequences:
- A µs value in a field named/typed for ns is a 1000× error that no downstream stage can detect. **The plan's proposed test cannot catch it**: it constructs "known synthetic timestamps" and asserts Δt is computed correctly — test and code share the same wrong unit, so the test passes. This is the clearest example in the plan of a test that would pass against a broken implementation.
- The proposed guard flags "implausibly large Δt (e.g., hundreds of milliseconds)." Real nuScenes camera-to-LiDAR offsets are on the order of **tens** of milliseconds, because cameras are triggered as the LiDAR sweeps through each camera's FOV. A hundreds-of-ms threshold never fires; a naively tightened one fires constantly on `CAM_BACK`.
**Resolution:** add `time_base` to every temporal record; convert at exactly one place; test the conversion against a **real** nuScenes record pair with a physically known offset, not a synthetic one; derive the Δt threshold **per camera** from the measured distribution over usable scenes and record it as config with provenance. Also fix the anchor explicitly: the LiDAR `sample_data.timestamp`, never `sample.timestamp`.

### P1-8 — Nothing in the pilot is reproducible as specified
`comprehensive.md` I-7 already requires "config snapshot + seed" with every result. The pilot has no seed policy at all, and at least four sources of run-to-run variation:
- **Sector-wise RANSAC is stochastic.** Unseeded, it fits a different ground plane every run → different retained points → different clusters → different boxes. This alone makes the pipeline non-reproducible end to end, and it is the very first geometric operation.
- **UMAP** is stochastic; setting `random_state` forces single-threaded execution and changes results, so seeded and unseeded runs are not comparable.
- **DBSCAN** is deterministic but its cluster *labels* depend on point order; "keep largest cluster" needs a deterministic tie-break (P0-5).
- cuDNN autotuning and TF32 change numerics between runs.
**Resolution:** one global seed threaded into RANSAC, UMAP, and any sampling; a `run_manifest.json` per stage recording seed, config hash, git commit, package versions, CUDA/PyTorch versions, checkpoint SHA-256, dataroot fingerprint, `usable_scenes.json` hash, priors version, and `W_acc`. Add a determinism test: run one scene twice, byte-compare outputs.

### P1-9 — No failure-recovery or completion semantics
The plan does not say what happens when a scene fails, a frame fails, a file is corrupt, a model OOMs, a checkpoint is unavailable, a stage is interrupted mid-write, or the pipeline is rerun. Without this, a partially-written stage output is indistinguishable from a complete one, and the next stage consumes it — producing a partially valid dataset that looks whole. **Resolution:** atomic writes (temp + rename); a per-stage `manifest.json` recording inputs consumed, outputs produced, counts, and the upstream manifest hash; a `_SUCCESS` marker; a downstream refusal to start when the upstream manifest is missing or its input fingerprint differs; per-scene isolation so one failure lands in `failures.json` rather than aborting the run; idempotent re-run keyed on the manifest. OOM specifically should be a hard stop with the offending stage/role/resolution logged, never a silent retry at lower resolution — that would change the output distribution mid-run.

### P1-10 — Indigenous-prompt containment is stated but not mechanised
The plan correctly scopes the indigenous-class prompting as informational (§0.2, §6) and correctly refuses to call it S1. The containment is a naming convention. The realistic contamination path: Grounding DINO prompted with "cycle rickshaw" on Boston/Singapore imagery **will** return boxes on something, with confidences. A screenshot or a JSON file of those boxes is one context-loss away from being read as evidence that the model separates indigenous classes — the precise claim `GAP_ANALYSIS.md` G3 says nobody has measured.
**Resolution:** hard separation, not labelling. Separate output root; every record carries `experiment: "indigenous_prompt_probe"` and `not_evidence_for: "S1"`; no probe output ever enters `data/out/` or the priors; a test asserting no indigenous prompt string appears in the pilot taxonomy array and no probe record appears in the main outputs. Any writeup sentence about it must carry the explicit form: *"prompt-injection mechanism check on non-Dhaka imagery; not a separability result."*

### P1-11 — nuScenes-derived priors can be mistaken for S0 priors
`comprehensive.md` §7.2 is explicit that nuScenes/KITTI values are "initialization only, **deleted after S0**." The pilot writes `priors/priors_pilot_v0.json` in the A.4 schema, in the same directory shape, consumed through the same interface. A.4 carries `source:"S0"`. **Resolution:** the pilot's file must set `source: "nuscenes_gt_pilot"`, and the release builder must reject any priors file whose `source` is not `S0` — a five-line guard that prevents a nuScenes-shaped prior from reaching a Dhaka release. Note also the nuScenes licence (CC BY-NC-SA class — verify) bears on redistributing GT-derived statistics; check before the priors file lands in a public repo.

### P1-12 — Stage 8 systematically defeats Stage 9's spatial-sanity gate
`comprehensive.md` §7.3.9 defines the spatial gate as "BEV box exceeds 2× class prior." Stage 8 inflates sparse boxes **toward the class prior mean**. Any box that passes through Stage 8 is by construction closer to the prior than before, so it passes the spatial gate more easily — the gate's discriminative power is highest exactly on the boxes Stage 8 has already corrected, and lowest where it is needed. Ordering is the same in the spec, so this is an inherited flaw the pilot should surface rather than reproduce silently.
**Resolution:** run the spatial gate on **pre-inflation** dimensions, or record both and gate on the measured one; add an `inflated: bool` + `inflation_fraction` field so a downstream reader can tell a measured box from a prior-shaped one. Without that field, a pilot run where ground removal stripped most object points produces boxes that are ~90% prior and 10% measurement, and nothing in the output says so.

### P1-13 — Ground removal has no diagnostics and parameters are spec-copied
The 0.3 m / 40 m / 4 m parameters are transferred from `comprehensive.md` §7.3.1 — where they were chosen for a Livox Mid-360 on Dhaka roads — to a 32-beam spinning LiDAR on Boston/Singapore roads, without pilot validation and without being marked as pilot defaults needing provenance. Concrete risks: a 0.3 m band removes all wheel returns from every vehicle and most of a traffic cone's body, pushing small classes below the ≥5-point gate; sector-wise RANSAC on ramps, speed bumps and cambered roads mis-fits, removing whole small objects in one sector while leaving a ground shelf in another; and single-plane-per-sector cannot represent a curb.
**Resolution:** emit per-filter point-count diagnostics (input → post-ground → post-range → post-height, per frame, per sector) as a first-class Stage 1 output. Without it there is no way to distinguish "the pipeline found few objects because detection is bad" from "because filtering deleted them." Treat all three thresholds as config with recorded provenance. Also state the RANSAC sector count, distance threshold, and iteration count — none appear in either document.

### P1-14 — VRAM mitigations silently change what the pilot measures
Two fallbacks in §5 have unstated semantic costs:
- **Reducing input resolution** from 1600×900-derived scales to 640×640 disproportionately destroys small-object recall, which is precisely what a dense-traffic pipeline is about. A pilot that quietly ran at 640 has not demonstrated the same plumbing under load.
- **Chunking the 23-phrase prompt** to save VRAM — the obvious next mitigation — **changes confidence semantics**, because Grounding DINO's scores derive from token-level logits over the full prompt. Scores from a 6-phrase chunk are not comparable to scores from a 23-phrase prompt, so a threshold tuned under one regime is meaningless under the other.
**Resolution:** record the actual resolution and prompt configuration in every Stage 3 output record; forbid mid-run changes; if chunking is used, re-tune and re-record the threshold per chunking configuration.

### P1-15 — No policy on computing metrics against nuScenes GT
The pilot will be sitting on nuScenes ground truth and a pipeline that emits boxes. Computing mAP is a two-line temptation, and the resulting number would be structurally comparable-looking to VESPA's published 46.76 % mAP on nuScenes (`GAP_ANALYSIS.md` §6) — while being produced by a Tiny-tier model, from priors derived from the same GT, on scenes that also supplied the threshold tuning. **Resolution:** decide now. Either forbid it in the pilot, or permit it only on the disjoint `run` subset (P1-5), reported with the leakage, the model tier, and the pilot's purpose stated in the same sentence, and never as a pipeline-quality figure.

---

## Stage-by-Stage Analysis

### Stage 0 — Data probe · **NEEDS REVISION**
**Purpose:** prove the partial blob can be reconciled to a trustworthy allowlist before any GPU work.
**In:** metadata JSONs + files on disk. **Out:** `usable_scenes.json`. **Depends on:** dataroot resolution and a declared required-channel set — neither specified (P0-6, P0-9).
**Silent failure:** partially-present scene admitted as usable; truncated `.pcd.bin` reshaping cleanly; sweeps unchecked; broken token references surfacing three stages later; stale allowlist reused against a different dataroot.
**Hard failure (should stop):** metadata version mismatch; zero usable scenes; dataroot missing `samples/` or `v1.0-trainval/`.
**Missing validation:** referential integrity of the token graph; file-size/parse checks; sweep coverage within `W_acc`; `sample_annotation` closure (Stage 8 needs it); required-channel enumeration.
**Test adequacy:** the proposed three-scene fixture is the right idea and catches the headline misclassification. It does not touch any of the above. Add fixtures for a truncated point file, a dangling `ego_pose` token, and a scene missing sweeps for its first keyframe.
The plan's instinct — exclude partial scenes loudly rather than skip frames quietly — is correct and should be kept.

### Stage 1 — Ingestion · **NEEDS REVISION**
**Purpose:** produce I-3; prove keyframe association and cloud construction.
**In:** allowlist + sample/sample_data/ego_pose/calibrated_sensor. **Out:** keyframe pack + clouds.
**Depends on:** frame convention (P0-1), time base (P1-7), `W_acc` policy.
**Silent failure:** points left in the LiDAR frame; µs written into an ns field; accumulation using the wrong ego pose per sweep; short accumulation at scene starts changing point density without a recorded `n_sweeps_actual`; ground plane mis-fit deleting small objects; single-sweep cloud never produced (P0-3).
**Hard failure:** missing `ego_pose` for a sweep; RANSAC failing to converge in a sector.
**Missing validation:** point-count diagnostics per filter (P1-13); an assertion that the accumulated cloud's ego-compensated static structure is consistent (e.g. building facades stay thin across sweeps — a cheap, decisive check that ego compensation is applied in the right direction).
**Test adequacy:** the Δt test cannot catch a unit error (P1-7). The ground-removal test uses a perfect z=0 plane, which a **global**-plane implementation passes identically — so it does not test the "sector-wise" property at all. Use a fixture with per-sector slope.
**Unresolved decision:** `W_acc` at 20 Hz — preserve the spec's 5-sweep *count* (0.25 s) or its ~0.5 s *duration* (10 sweeps)? The plan notes the cadence difference but does not choose. Preserve duration and record both.

### Stage 2 — OOD / long-tail discovery · **SOUND WITH CAVEATS**
**Purpose:** prove the embedding → projection → clustering path runs. Explicitly not discovery.
**Silent failure:** UMAP's non-determinism producing a different flagged set each run (P1-8); flagging driven by image brightness or camera identity rather than content; "OOD" decided in 2-D UMAP space rather than embedding space.
**Substantive caveat:** this is **not OOD detection**; it is clustering of a non-metric 2-D projection. UMAP does not preserve density or global structure, so HDBSCAN membership in UMAP space carries no formal relationship to outlyingness in the 384-D embedding space. "Dominant manifold" is undefined in both documents. If any OOD claim is wanted, decide in embedding space — HDBSCAN's GLOSH outlier scores computed on the original embeddings, with UMAP retained for visualisation only.
**Scale problem:** a single blob chunk at 1-in-10 sampling yields a few hundred frames. HDBSCAN on a few hundred points typically returns one cluster plus noise, so the plan's success criterion ("not zero, not thousands") is close to unfalsifiable. State the expected sample count and check it is large enough before running.
**Missing decision:** does Stage 2 embed a *keyframe* (which camera?) or an *image*? This changes what an "OOD frame" is (P1-4).
**Test adequacy:** the unit test as described feeds synthetic vectors through UMAP — which degenerates on tiny inputs (`n_neighbors` must be < n_samples) and will either fail spuriously or exercise a bypassed path. Test the flagging rule on precomputed cluster labels/outlier scores instead, and keep UMAP out of the unit test entirely. The subset/no-invented-IDs assertion is genuinely good; keep it.
Note also that Stage 2's output is consumed by nothing in the pilot, which contradicts §1's "0→9 is a real dependency chain." Harmless, but it means Stage 2 can be built and run out of order, and it means DINOv2 is loaded, freed, and reloaded at Stage 7 (P1-2).

### Stage 3 — 2D proposals · **NEEDS REVISION**
**Silent failure — the important one:** **wrong class assignment from phrase-span mapping.** Grounding DINO produces per-token logits over the concatenated prompt; recovering which *phrase* a box belongs to requires correct token-span bookkeeping, and multi-word classes make it error-prone. A span-mapping bug yields well-placed boxes with wrong labels — which then select the wrong DBSCAN ε, the wrong dimension prior, and the wrong inflation target. Everything downstream runs perfectly.
**Second silent failure:** feeding nuScenes' **dotted hierarchical category names** (`vehicle.emergency.ambulance`, `human.pedestrian.adult`) directly as text prompts. They tokenise into nonsense. The plan says "nuScenes' 23-class taxonomy as the pilot prompt set" with no mapping layer, and its proposed test — "does the pilot correctly build nuScenes' 23-class prompt set" — would **pass** while asserting exactly the wrong thing.
**Also:** a single threshold across 23 classes is not defensible — Grounding DINO scores are not calibrated across phrases and longer phrases score systematically lower. `comprehensive.md` §7.3.3 specifies per-class thresholds with a 0.40 global *default*; the pilot proposes one tuned scalar. Acceptable for plumbing if the interface accepts a per-class dict with a default, so production does not need a Stage 3 change.
**Missing:** deduplication of overlapping/duplicate proposals; output coordinate space (P0-4); the `[VERIFY]` on "23 classes" itself (confirm against your `category.json`).
**Missing test:** assert prompt strings are natural-language phrases, that the class returned for a synthetic single-phrase prompt is that phrase, and that box coordinates are in original image resolution.

### Stage 4 — Masks · **SOUND WITH CAVEATS**
The MobileSAM propagation loss is correctly identified and correctly accepted — it is a capability gap, not a shape change, and the plan says so plainly. Two additions:
**IoA-NMS across overlapping cameras (`comprehensive.md` §7.3.4) is omitted without comment.** It is pure geometry, costs nothing, and its absence is what makes P0-5's duplicate-assignment problem reachable. Restore it or declare it out of scope.
**Silent failure:** box prompts passed in the wrong coordinate space (MobileSAM expects transformed 1024-longest-side coordinates); masks returned at 1024 and not resized back to 1600×900 before Stage 5 indexes them.
**Test adequacy:** "one mask per box, same order" is a genuinely strong test — it catches the misalignment that would silently assign the wrong class to the wrong region. Add a mask-resolution assertion.
**On the plan's own open question (§9):** MobileSAM's lack of propagation does **not** by itself break Stage 7, because association operates on 3D clusters and crop embeddings, not propagated masks. The real damage is mask flicker → unstable per-frame clusters → fragmented tracks. So association-logic unit tests are *not* sufficient on their own, but the missing piece is small: **the minimum integration test is to run Stages 3→6 over ~10 consecutive keyframes of one usable scene and assert that at least one object yields a track of ≥3 frames with a stable ID.** If that fails, Stage 7 has no real input and the association logic is being tested only against fabricated matrices.

### Stage 5 — Reprojection · **BLOCKED**
Blocked on P0-1, P0-2, P0-3, P0-4 — the frame, the pose chain, the cloud identity, and the pixel space are all undefined, and each independently produces plausible wrong output.
**Additional missing validation, beyond the single known-point test:**
- points behind the camera (`z ≤ 0` must be culled **before** the divide — otherwise they project to valid-looking mirrored pixels);
- near-zero depth (division blow-up);
- points outside image bounds;
- a point visible in two overlapping cameras — which class wins? Currently undefined, so the answer depends on camera iteration order, i.e. non-deterministic labelling;
- `comprehensive.md` §7.3.5's per-camera frustum **union** rule, absent from the pilot;
- distortion: nuScenes ships **no distortion coefficients** and pre-rectified imagery, so the production undistortion path is never exercised. Record this explicitly rather than letting "we did reprojection on real calibration" imply the distortion path was tested.
**The decisive test the plan is missing:** project the centres of nuScenes **ground-truth 3D boxes** into each camera and assert they land inside the object / inside the box's 2D extent, and separately assert that mask-painted points fall inside the corresponding GT box at a high rate. A synthetic point with synthetic calibration cannot detect a transform that is mathematically valid but directionally wrong; this one can, using data already on disk.

### Stage 6 — Clustering + box fit · **NEEDS REVISION**
Blocked on P0-5 (clustering scope). Further:
**Silent failure:** yaw convention drift between the L-shape fitter's output and `conventions.py`; `size` ordering — nuScenes is `[w, l, h]`, KITTI is `[h, w, l]`, and many L-shape implementations return `(length, width)`; a swap rotates every box 90° while all values stay plausible. **90° ambiguity** on near-square footprints (pedestrians, cones, barriers) is systematic, not occasional.
**Missing:** deterministic tie-break for equal-size clusters; a stated policy for the near-square case (e.g. enforce `w ≤ l` and record ambiguity); ε derivation formula (`comprehensive.md` §7.2 gives `ε ≈ 0.6 × mean footprint diagonal` — use it rather than the `Annotation_pipeline.md` hardcoded table, whose class names are not nuScenes categories and whose provenance claim is unverified).
**Test adequacy:** the synthetic-rectangle test is good but must use a **deliberately non-square** rectangle at a **non-trivial yaw** (e.g. 30°), and must assert against `conventions.py`'s yaw definition, not against the fitter's own. Add a yaw→quaternion→yaw round-trip test for the A.1 conversion.

### Stage 7 — Tracking · **NEEDS REVISION**
Blocked in practice by P1-1 (2 Hz kills the IoU term).
**Association logic is under-specified:** the plan does not state the matching algorithm (greedy vs Hungarian), the order of gating vs scoring, how the similarity and IoU terms combine (product per `comprehensive.md`, or weighted sum), the gate thresholds, or the handling of one-to-many and many-to-one candidates. Track **birth** and **death** rules are absent entirely; so is occlusion handling.
**ICP:** meaningfulness is doubtful with the proposed inputs. On smeared accumulated clusters (P0-3), across 0.5 s gaps, on sparse ground-removed points, ICP will converge to something — and that something will be reported as velocity. **Frame matters critically here:** ICP between two ego-frame clusters measures motion *relative to the ego vehicle*, not absolute velocity, and nuScenes mAVE is absolute. Register in the global frame or subtract ego motion explicitly, and state which.
**Kalman fallback is not specified at all** — no state vector, no motion model, no process/measurement noise, no initialisation. `comprehensive.md` is equally thin. At minimum: constant-velocity state `[x, y, z, vx, vy, vz, yaw]`, Δt from the recorded timestamps (which base? — P1-7), and the `< 15 points` trigger from §7.3.6.
**Missing from the spec's stage 7:** forward-backward smoothing over the scene and **yaw-consistency enforcement along tracks** — the latter being the designed defence against 180° flips on symmetric vehicles, which is free geometry and directly relevant to the pilot's L-shape fitter.
**Test adequacy:** fabricated-similarity-matrix tests for the matcher are right. The min-point Kalman-fallback test is right. Both miss that the *inputs* may be in inconsistent frames or at inconsistent timestamps — add an integration assertion that association inputs carry matching `frame` and comparable `t`.

### Stage 8 — Inflation · **SOUND WITH CAVEATS**
**Is "inflate toward class mean" well-defined?** Not from the A.4 schema alone. A.4 supplies `dims:{mu, sigma}` per class and nothing about *how much* to inflate, *when*, or along which axes. To be deterministic it needs: a trigger (point count below threshold), a blend function (all-the-way-to-mean vs point-count-weighted), per-axis applicability, and a clamp. Add these as `priors_v*.json` fields or as explicit config with recorded provenance.
**Geometric caveats:** anchoring "away from ego" is well-defined *given* a correct yaw and a declared frame — but under a 90° yaw error (Stage 6) inflation grows the box sideways into the neighbouring lane, deterministically and invisibly. LiDAR visibility bias means the observed face is always the near face, so anchoring the near face and growing the far face is the right rule; state it that way rather than as "shift outward." Guard against growth back through the sensor-facing surface.
**Cross-stage:** defeats the Stage 9 spatial gate (P1-12); must emit `inflated` / `inflation_fraction`.
**Test adequacy:** the plan's directional test — that the box shifts away from the LiDAR surface rather than growing symmetrically through it — is one of the strongest tests in the document. Keep it. Add a case with a deliberately wrong yaw to document the failure mode.

### Stage 9 — QA gating · **NEEDS REVISION**
Blocked on P0-7 (enforcement) and affected by P0-3 (the `≥5 returns` gate silently loosens) and P1-12 (the spatial gate is pre-defeated).
**Silent failure:** the gate vector is computed but the tier is written from a different code path; provenance survives construction but not serialization; `tier: auto_accept` conflated with `source: pipeline_accepted`.
**Missing:** what `spatial_ok`'s "drivable check" means without an HD map — nuScenes has map masks, Dhaka will not (`comprehensive.md` §1.4 excludes HD maps). Either drop the drivable term in the pilot and say so, or implement it against the nuScenes map and mark it as a component with no production counterpart. Either is fine; silence is not.
**Test adequacy:** the boundary-value tests (just above / just below each threshold, plus combination cases) are well-conceived and should be kept as written. The provenance test is currently **vacuous** — with no split field populated, "no record claims `human_verified`" passes trivially. Replace with positive-rejection assertions (P0-7).

---

## Taxonomy Audit

The core separation is correct: run the real pipeline on nuScenes' own taxonomy, keep indigenous prompting as an isolated informational probe, and refuse to call any of it S1. That is the right call and matches `GAP_ANALYSIS.md` G3, which is explicit that S1 means something only on real imagery with a production-tier model.

Residual risks:
- **Prompt-string construction is the unguarded contamination surface** (Stage 3 above): dotted nuScenes names are not usable prompts, and the proposed test asserts the wrong invariant.
- **Containment is naming, not mechanism** (P1-10). The probe will produce boxes; boxes become screenshots; screenshots lose context.
- **Class-set mismatch downstream:** the `Annotation_pipeline.md` ε and dimension tables use `cyclist`, `traffic_cone`, `car` — not nuScenes category names. Any lookup keyed on them silently misses and falls back to a default, giving every unmatched class the same ε.
- **The 23-class count is an unverified claim.** Verify against `category.json` in your blob's metadata, and decide whether the prompt set is the 23 raw categories or the 10 detection classes — they are different experiments.

**Recommendation:** two explicit, separately-named artifacts — `taxonomy_pilot_nuscenes.yaml` (the real pilot taxonomy, with a `nuscenes_category → prompt_phrase` mapping table) and `taxonomy_probe_indigenous.yaml` (probe only, physically separate output root, `not_evidence_for: "S1"` on every record) — plus a test asserting no string crosses between them.

---

## Coordinate-Frame and Geometry Audit

Traced: `LiDAR → ego → global → camera → pixel → mask → painted points → BEV → oriented box → tracking → inflation`.

| Hop | Risk | Status in plan |
|---|---|---|
| LiDAR → ego | nuScenes `LIDAR_TOP` is ≈ −90° yaw from ego; +y is forward in sensor frame | **Unaddressed (P0-1)** |
| ego(t_lidar) → global → ego(t_cam) | two distinct ego poses per sample; omitting the hop leaves an ego-motion error | **Unaddressed (P0-2)** |
| ego → camera | `T_cam_ego` inversion direction; nuScenes gives sensor→ego, projection needs ego→sensor | Unstated; classic inverse-transform error |
| camera → pixel | `z ≤ 0` cull before divide; K applied after extrinsics; no distortion in nuScenes | Only the happy path tested |
| pixel → mask | resize/letterbox inverse; mask resolution vs image resolution | **Unaddressed (P0-4)** |
| mask → points | multi-camera overlap, no union/dedup rule | **Unaddressed (P0-5 neighbourhood)** |
| points → BEV | which frame's XY plane; ego-frame BEV vs LiDAR-frame BEV differ by 90° | **Unaddressed (P0-1)** |
| BEV → oriented box | yaw convention, `[w,l,h]` order, 90° ambiguity on square footprints | Unstated |
| box → tracking | ICP frame determines whether "velocity" is relative or absolute | Unstated |
| tracking → inflation | anchor face depends on sensor origin in the box's frame | Unstated |
| all | radians vs degrees — §2.3 says radians, A.3 says `sigma_yaw_deg` | Inherited inconsistency, uncorrected |

**"nuScenes provides calibration" does not make the geometry correct** — it makes the *inputs* correct while leaving every application decision open. The plan's single known-point projection test validates arithmetic, not direction: a transposed rotation, an inverted extrinsic, or a skipped ego hop all still map a synthetic point to a computable pixel. The two tests that would actually catch a directionally-wrong transform both use data already on disk:
1. **GT-box reprojection:** project nuScenes GT box centres/corners into each camera; assert they land on the object.
2. **Paint-inside-GT:** assert that points painted by a mask fall inside the corresponding GT 3D box at a high rate.
Add a lidar→…→pixel→(with depth)→lidar round-trip identity test as a cheap third.

---

## Temporal Audit

The plan correctly identifies that the real per-camera Δt must be stored rather than assumed — that is `comprehensive.md` I-3's requirement and the plan honours it. Everything else in the temporal chain is under-specified.

**Distinctions the plan does not draw:** `sample.timestamp` vs `sample_data.timestamp` vs the LiDAR anchor vs the camera timestamp vs the **ego-pose timestamp** — nuScenes gives each `sample_data` its own `ego_pose`, which is the fact P0-2 turns on. The anchor must be pinned to the LiDAR `sample_data.timestamp`.

**Hidden synchronisation assumptions:**
- That nuScenes' own ego poses are sufficient for accumulation and reprojection. For **accumulation** they are (each sweep has its own pose; this is the devkit's own approach). For **reprojection** they are sufficient only if both poses are used — which is exactly what the plan omits.
- That ego-motion compensation handles the accumulation window. It handles *ego* motion only; object motion smears (P0-3).
- That a hundreds-of-ms Δt threshold is a meaningful guard. It is not (P1-7).
- That 2 Hz keyframes support IoU association (P1-1).
- That timestamps are ns GPS. They are µs Unix (P1-7).

**Also:** at scene boundaries the accumulation window is truncated. `comprehensive.md` I-3 already requires the window be recorded per cloud; the pilot must honour that and no stage may assume constant point density.

---

## Model / VRAM / Checkpoint Audit

| Claim in plan | Classification | Note |
|---|---|---|
| DINOv2 ViT-S/14 ~85 MB FP32 | **known fact** (≈21 M params) | Correct |
| MobileSAM ~10 M params, ~40 MB | **known fact** | Correct |
| SAM-ViT-B "~375M params, ~360MB weights" | **incorrect** | ~91 M params, ~375 MB file — params/MB swapped (P0-8) |
| Grounding DINO Tiny "~172 MB" | **incorrect** | ~172 M params, ~690 MB FP32 checkpoint (P0-8) |
| "9× parameter difference" MobileSAM vs SAM-B | **estimated, coincidentally right** | ~91/10 ≈ 9× |
| All four inference-VRAM ranges | **unverified estimate** | Plan says so itself — keep the `verified:` flag honest |
| "~900 MB system/CUDA reserve" → 3.2 GB usable | **estimated** | Measure: CUDA context, allocator reserve, and whether the dGPU drives the display on this laptop |
| "largest serial model is the real constraint" | **insufficient as stated** | True only if teardown actually frees; `empty_cache()` does not free referenced tensors. Add an inter-stage assertion that `memory_allocated()` returns to ~0 |
| Shared DINOv2 = "0 marginal" | **contradicted by the serial policy** | P1-2 |
| `max_memory_allocated()` as the meter | **wrong instrument** | Excludes context, reserve, workspaces (P0-8) |

**Licensing** — all `[VERIFY]`, and it matters for the production choice, not the pilot: Grounding DINO code/weights are permissive (Apache-class); MobileSAM is Apache-class; SAM 2.1 weights are Apache-class; **SAM 3's licence terms are exactly what `GAP_ANALYSIS.md` §6 flags as unverified**, and SAM 3 is the model that scan recommends promoting to the unified engine — so the production path's licence is currently unknown and could prohibit dataset production. DINOv2's licence changed after initial release (NC → permissive); verify the version you pull. And **nuScenes' own licence** constrains redistribution of GT-derived artifacts such as `priors_pilot_v0.json` (P1-11).

**Publication exposure:** the two wrong figures in §5 are precisely the class of error `comprehensive.md` §7.1.1 exists to prevent, appearing inside the plan that cites §7.1.1. If any of them reached a writeup — "we ran a 172 MB open-vocabulary detector on 4 GB" — it would be a checkable, wrong claim. Run the plan's own verification checklist against the plan.

---

## Test-Plan Audit

| Stage | Strong test (keep) | Weak test (would pass on broken code) | Missing test (add) |
|---|---|---|---|
| 0 | 3-scene bucket fixture — catches the headline misclassification | none, but coverage is narrow | truncated `.pcd.bin`; dangling token; missing sweeps; allowlist/dataroot fingerprint match |
| 1 | — | Δt on synthetic timestamps (shares the unit bug); flat-ground RANSAC (a global-plane impl passes) | µs→ns conversion against a real record; sloped multi-sector ground fixture; per-filter point-count diagnostics; static-structure sharpness check on the accumulated cloud |
| 2 | OOD list ⊆ input IDs, no invented IDs | synthetic vectors through UMAP (degenerate at small n); "flags zero on all-inliers" depends on the same fragile path | flagging rule tested on precomputed labels/outlier scores, UMAP excluded from unit tests |
| 3 | threshold-filter logic on synthetic boxes | "builds the 23-class prompt set" — asserts the wrong thing (dotted names) | prompt-phrase mapping; phrase→class span mapping; output in original pixel coordinates |
| 4 | one mask per box, same order — genuinely catches the silent misassignment | — | mask returned at original image resolution; box-prompt coordinate-space round-trip |
| 5 | known point → known pixel (necessary) | it is the *only* geometry test, and cannot catch a directionally-wrong transform | GT-box reprojection; paint-inside-GT rate; behind-camera cull; near-zero depth; out-of-bounds; two-camera overlap determinism; lidar→pixel→lidar round-trip |
| 6 | noise-rejection with scattered points | square-rectangle fixture would hide `[w,l]` swaps and 90° ambiguity | non-square rectangle at 30° yaw asserted against `conventions.py`; yaw→quat→yaw round-trip; two same-class instances → two boxes; deterministic tie-break |
| 7 | fabricated similarity matrices for the matcher; min-point Kalman-fallback trigger | both assume inputs are in consistent frames and comparable timestamps | frame/time consistency assertion on association inputs; the ≥3-frame track integration test (Stage 4 above); ICP-frame test distinguishing relative from absolute velocity |
| 8 | anchor-direction test — one of the best in the plan | — | wrong-yaw case; `inflation_fraction` recorded |
| 9 | threshold boundary + combination cases | provenance test is **vacuous** with no split populated | positive rejection: `val` + `pipeline_accepted` must raise; `human_verified` without `verified_by` must raise; post-serialization revalidation |
| cross | eval-region area vs closed-form geometry — correctly identified as load-bearing | — | contract round-trip (write→read→validate); units/frame assertion on every schema; determinism (same scene twice, byte-compare); golden-scene regression fixture |

The plan's stated method — *name the silent failure first, then write the test for it* — is the right method and should not be diluted. The gap is that it was applied to about half the stages; where it was applied (Stage 0's bucket test, Stage 4's ordering test, Stage 8's direction test) the resulting tests are genuinely strong.

---

## Cross-Stage Semantic Mismatches

1. **Stage 1 → Stage 5:** cloud identity changes meaning — accumulated vs single-sweep (P0-3), and frame is undeclared (P0-1).
2. **Stage 3 → Stage 4 → Stage 5:** pixel coordinate space is undeclared and each model uses a different internal one (P0-4).
3. **Stage 3 → Stage 6:** class *labels* flow through masks into ε selection. A phrase-span bug in Stage 3 silently selects the wrong ε and the wrong prior — Stage 6 and Stage 8 both misbehave with no error.
4. **Stage 4 → Stage 5:** multiple boxes per image and overlapping cameras; no rule for which class wins a contested point.
5. **Stage 5 → Stage 6:** Stage 5 produces *class-labelled* points; Stage 6's scope (per-instance vs per-class) determines whether those labels index instances or only ε (P0-5).
6. **Stage 6 → Stage 7:** yaw convention, `[w,l,h]` order, and BEV plane orientation all cross this boundary undeclared.
7. **Stage 7 → Stage 8:** ICP velocity may be ego-relative; inflation anchoring needs the sensor origin in the box's frame.
8. **Stage 8 → Stage 9:** inflation pre-satisfies the spatial-sanity gate (P1-12), and changes the box within which `num_lidar_pts` might be recounted.
9. **Stage 1 → Stage 9:** ground removal + accumulation both change what "≥5 LiDAR returns" counts (P0-3).
10. **Stage 9 → serialization:** provenance validated at construction but not at write/read (P0-7).
11. **Stage 0 → all:** `usable_scenes.json` has no dataroot binding, so every stage can silently operate on a mismatched substrate (P0-9).

---

## Configuration Audit

**Must be configurable, currently implicit or hardcoded-by-omission:** `NUSCENES_DATAROOT`, `meta_root`, `PILOT_OUTPUT_ROOT`, work root; required-channel set; camera subset; `coverage_config` (R1/R2) and eval-region parameters; `W_acc` (count *and* duration); RANSAC sector count / distance threshold / iterations / **seed**; ground band 0.3 m; range cap 40 m; height cap 4 m; image resolution and resize policy; prompt taxonomy file; prompt chunking; per-class confidence thresholds with a default; MobileSAM input size; per-class DBSCAN ε and `min_samples`; largest-cluster tie-break; association IoU and similarity gates, matcher choice, track birth/death; Kalman noise parameters; min-point ICP guard (15); inflation trigger/blend/clamp; QA thresholds; UMAP `n_neighbors` / `min_dist` / `random_state`; HDBSCAN `min_cluster_size` / `min_samples`; keyframe sampling rate (1-in-10); global seed; VRAM ceiling and reserve; checkpoint IDs and revisions.

**"Pilot defaults" with no documented provenance** — every one of these needs a provenance note (`spec §x.y` / `measured on N frames` / `arbitrary, needs tuning`): 0.3 m, 40 m, 4 m, 5 sweeps, 1-in-10, 0.40 confidence, 15-point ICP guard, ≥5 LiDAR returns, 2× spatial multiplier, the `Annotation_pipeline.md` ε table, and the 900 MB VRAM reserve. Several were derived for different hardware, a different city, and a different sensor.

---

## Reproducibility Audit

Two people following this plan on the same dataroot **would get different results**, primarily because sector-wise RANSAC is unseeded (different ground plane → different points → different clusters → different boxes) and UMAP is stochastic. Additional divergence: package/CUDA versions, checkpoint revision drift on model hubs, DBSCAN label ordering, cuDNN autotuning, and a different `usable_scenes.json` if the blobs differ.

**Must be recorded in the final pilot artifact:** global seed; per-library seeds; `usable_scenes.json` hash + dataroot realpath + metadata fingerprint; scene partition (priors/tuning/run) and its seed; full resolved config; git commit; Python/PyTorch/CUDA/UMAP/HDBSCAN/sklearn versions; checkpoint IDs, revisions and SHA-256; `priors_pilot_v0.json` version and `source`; `W_acc` actual per cloud; image resolution and prompt configuration actually used; measured VRAM peaks per role; per-stage input/output counts. `comprehensive.md` I-7 already requires config+seed with every result — the pilot should inherit that requirement rather than defer it.

---

## Scientific-Claim Audit

| Potential reading of a pilot result | Classification |
|---|---|
| "The pipeline shape from §7.3 is implementable end to end" | **SUPPORTED BY PILOT** (once P0s are closed) |
| "Contracts I-1…I-4 are sufficient to carry data between stages" | **SUPPORTED BY PILOT** |
| "It runs on 4 GB VRAM" | **SUPPORTED BY PILOT** only with measured numbers, named resolution and prompt count |
| "Stage 0 reliably identifies usable scenes" | **ONLY PLUMBING EVIDENCE** |
| "The OOD stage discovers long-tail content" | **NOT SUPPORTED BY PILOT** — the taxonomy is known, and the method is clustering of a projection |
| "Open-vocabulary prompting separates indigenous classes" | **NOT SUPPORTED BY PILOT** — requires real Dhaka data + production model (S1) |
| "Any per-class quality / recall figure" | **REQUIRES PRODUCTION MODEL** — Tiny-tier checkpoints, deliberately |
| "Annotation quality is adequate" | **REQUIRES REAL S0** |
| "Tracking association works" | **ONLY PLUMBING EVIDENCE**, and weakened by the 2 Hz regime (P1-1) |
| "Priors/ε derivation works" | **ONLY PLUMBING EVIDENCE** — consumption side proven, S0 production side not (REQUIRES REAL S0) |
| "mAP of X on nuScenes" | **NOT SUPPORTED** as a pipeline-quality claim — leakage via GT-derived priors + tuning on the same scenes (P1-5, P1-15) |
| "Production readiness" / "Dhaka-specific performance" | **REQUIRES REAL DHAKA DATA** |
| "Model A beats model B" | **NOT SUPPORTED** — no controlled comparison exists in the pilot |

**Wording the plan needs, verbatim, in the pilot's own outputs and any writeup:** *"This demonstrates pipeline plumbing only. Label quality is not evidence of anything; models are deliberately under-tier; the substrate is nuScenes, not Dhaka."* Put it in the pilot README, in `run_manifest.json`, and in the header of any figure exported from a pilot run — not only in a planning document that a future reader may not have.

The specific overclaim risk is not vanity: `comprehensive.md` §11.6 pivot B is "the annotation-pipeline paper, validated on nuScenes," and `GAP_ANALYSIS.md` §6 positions this pipeline against VESPA's published nuScenes numbers. A pilot run producing nuScenes-shaped outputs sits directly on that path while carrying leakage the real study would not.

---

## Contradictions

**Genuine contradictions (must be resolved):**

| # | Between | Nature |
|---|---|---|
| X-1 | Pilot §3 Stage 1/5 vs `comprehensive.md` §7.3.5 + I-3 | Stage 5 consumes the accumulated cloud; spec says ground-filtered **single-sweep**, and I-3 requires both clouds be produced (P0-3) |
| X-2 | Pilot §4.2 vs §4.4 | "Serial, one role loaded at a time, cache cleared between stages" vs "loading DINOv2 twice would burn ~700 MB — the easiest way to blow the budget." Under the serial policy two copies cannot coexist (P1-2) |
| X-3 | Pilot §8 step 3 | "Before downloading any checkpoint" and "must use real `max_memory_allocated()` measurements" in the same step — you cannot measure an undownloaded model. Split into a paper check and a measured check |
| X-4 | Pilot §1 vs §3 Stage 2 | "0→9 is a real dependency chain, each stage's output is the next stage's input" — Stage 2's output feeds nothing |
| X-5 | Pilot §3 Stage 3 vs `comprehensive.md` §7.3.3 | Single global threshold vs per-class thresholds with a 0.40 default |
| X-6 | Pilot §3 Stage 6 vs `comprehensive.md` §7.2 | Uses the `Annotation_pipeline.md` ε table as a starting point; spec says nuScenes/KITTI values are "initialization only, deleted after S0", and the table's class names are not nuScenes categories |
| X-7 | Pilot time handling vs `comprehensive.md` §2.3 | µs Unix vs mandated int64 ns GPS (P1-7) |
| X-8 | *Within `comprehensive.md`* | §2.3 specifies radians; Appendix A.3 uses `sigma_yaw_deg`. Inherited; the pilot should resolve it by unit-suffixing every field |
| X-9 | Pilot §2 line 78 vs `comprehensive.md` I-5 / §7.4 | The pilot states the F1 invariant as a `source`/split rule but omits the `human_verified ⇒ verified_by` rule that its own Stage 9 test depends on (P0-7) |
| X-10 | Pilot §4.3 vs `GAP_ANALYSIS.md` §6 | Production config assumes one model per role; the scan recommends SAM 3 as a **unified** proposal+mask+track engine, which the role registry cannot express (P1-3) |

**Intentional pilot substitutions (correctly identified and justified — not contradictions):** nuScenes as substrate; ego_pose in place of PPK; nuScenes taxonomy in place of the indigenous one; nuScenes GT as S0-equivalent; MobileSAM's loss of propagation; Tiny-tier checkpoints throughout; I-5 simulated; I-6/I-7 shape-only; Stage 0 as pilot-only scaffolding.

---

## Missing Pieces

- **M-1** Stage 10 (attribute pre-fill, `comprehensive.md` §7.3.10) omitted without declaration, leaving A.1's `attribute` field unproduced. `visibility` likewise.
- **M-2** IoA-NMS across overlapping cameras (§7.3.4) omitted without comment.
- **M-3** Forward-backward track smoothing and **yaw-consistency enforcement** (§7.3.7) omitted; the latter is free geometry and is the designed 180°-flip defence.
- **M-4** Group/ignore boxes (§6.3, A.2): `GroupAnnotation` is listed in `schemas.py` with no producer or consumer. Declare out of scope.
- **M-5** ρ's definition in the pilot: is it computed over GT boxes or pipeline output? Over pipeline output it is a function of the detector, not the scene.
- **M-6** Camera subset and `coverage_config` (P1-4).
- **M-7** Scene partition between priors derivation, threshold tuning, and the pilot run (P1-5).
- **M-8** Path/config contract and dataroot validation (P0-9).
- **M-9** Failure recovery, atomicity, manifests, completion markers (P1-9).
- **M-10** Seed policy and run manifest (P1-8).
- **M-11** Kalman filter specification; track birth/death; matcher algorithm (Stage 7).
- **M-12** Inflation policy parameters beyond A.4's `{mu, sigma}` (Stage 8).
- **M-13** `spatial_ok`'s drivable-region term with no HD map (Stage 9).
- **M-14** Scale of the end-to-end run: how many scenes, which ones, and their relationship to M-7.
- **M-15** Required-channel set and `fully_present` predicate definition (P0-6).

---

## Silent Failure Risks — ranked

The pipeline runs to completion and produces plausible output in every case below.

1. **Points never leave the LiDAR frame.** Eval region selects the vehicle's right side instead of its front; every yaw is 90° off. (P0-1)
2. **Reprojection skips the ego-pose hop.** Sub-metre systematic paint offset; points bleed onto neighbouring objects; clusters biased along travel direction. (P0-2)
3. **Accumulated cloud lifted instead of single-sweep.** Moving objects elongate; yaw biased toward motion; the ≥5-return gate loosens ~5×. (P0-3)
4. **Aspect-ratio-breaking resize with a uniform inverse.** Every box stretched consistently in one axis. (P0-4)
5. **DBSCAN pooled per class.** One box per class per frame; most instances silently discarded, output looks clean. (P0-5)
6. **Phrase-span mis-mapping in Stage 3.** Correct boxes, wrong labels → wrong ε, wrong prior, wrong inflation. (Stage 3)
7. **Dotted category names used as prompts.** Prompts tokenise to nonsense; detections are near-random but non-empty; the proposed test passes. (Stage 3)
8. **µs written into an ns field.** All Δt values wrong by 1000×; the proposed test shares the error. (P1-7)
9. **Truncated `.pcd.bin` reshapes cleanly.** Fewer points, no error, sparser clusters, more inflation. (P0-6)
10. **σ_pos filled with 0.0.** Downstream reads nuScenes pose as better than PPK. (P1-6)
11. **Provenance validated at construction, not at write.** A `human_verified` record with no human reaches disk. (P0-7)
12. **Ground removal deletes most object points.** Boxes become ~90% prior via Stage 8; nothing records that they are prior-shaped. (P1-12, P1-13)
13. **Stale `usable_scenes.json` against a different dataroot.** Scenes silently included or excluded. (P0-9)
14. **Unseeded RANSAC.** Two runs, two datasets, both plausible. (P1-8)
15. **`[w,l,h]` order or yaw-quaternion sign swap.** Every box rotated 90°; all values in range. (Stage 6)
16. **Contested points in camera overlap.** Class assignment depends on iteration order. (Stage 5)
17. **Prompt chunking to fit VRAM.** Confidence scores change meaning; the tuned threshold silently no longer applies. (P1-14)
18. **Short accumulation at scene starts.** Systematically different point density and box size for the first keyframes of every scene. (Stage 1)
19. **Stage 7 crop embeddings on 2×2 patch grids.** Similarity near-degenerate; association looks like it works because IoU and appearance both fail quietly. (P1-1)
20. **Partially written stage output consumed as complete.** (P1-9)

---

## Recommended Changes Before Coding — prioritised checklist

**Close in `pipeline/common/` before any stage code (P0):**
1. Declare **ego frame canonical**; add a `frame` field to every geometric record and validate on write. Test that `LIDAR_TOP` calibration is non-identity.
2. Specify `project_lidar_to_image()` as the single four-hop implementation (`lidar → ego(t_lidar) → global → ego(t_cam) → camera`), including `z ≤ 0` culling.
3. Produce **both** clouds; fit ground on accumulated, apply to single-sweep, lift single-sweep, count `num_lidar_pts` on single-sweep pre-inflation.
4. Declare **original-resolution absolute pixels** as the 2D contract; each model adapter owns its forward/inverse transform; no square resizes of 16:9 imagery.
5. Pin DBSCAN scope to **per mask instance**; add a deterministic tie-break.
6. Rewrite Stage 0's buckets as explicit predicates: required-channel set, sweeps within `W_acc`, token-graph referential integrity, file size/parse checks, metadata version. Emit which predicate failed.
7. Add a raising `write_records()` boundary; add the `human_verified ⇒ verified_by` rule; add `allow_human_provenance: false`; test positive rejection.
8. Correct the Grounding DINO and SAM-ViT-B figures; switch the VRAM meter to `max_memory_reserved()` + `mem_get_info()` deltas at real resolution and real prompt count; define the over-ceiling hard stop.
9. Write the path contract: single resolved `Paths` object, `validate_paths()`, dataroot fingerprint in every manifest, allowlist↔dataroot binding, disjointness assertion, read-only enforcement, `data/` git-ignored, work/out outside the repo.
10. Add `time_base` and unit suffixes to every temporal/geometric field.

**Before the first full run (P1):**
11. Decide `coverage_config`, the camera subset, and `W_acc` (count vs duration).
12. Partition `usable_scenes.json` into disjoint priors / tuning / run subsets.
13. Fix the association regime for 2 Hz (predict-then-match) and record the effective Δt.
14. Drop the shared-instance VRAM claim; make preprocessing part of the role interface; assert `memory_allocated() ≈ 0` between stages.
15. Give `mask_2d` an optional temporal-state contract; allow one provider per multiple roles.
16. Add per-filter point-count diagnostics to Stage 1; mark all copied thresholds with provenance.
17. Add `inflated` / `inflation_fraction`; gate spatial sanity on pre-inflation dimensions.
18. Add seeds, `run_manifest.json`, and the determinism test.
19. Add manifests, atomic writes, `_SUCCESS` markers, per-scene failure isolation.
20. Hard-separate the indigenous probe (separate root, marker fields, cross-contamination test).
21. Set `source: "nuscenes_gt_pilot"` in the priors file; add the release-builder guard.
22. Decide the nuScenes-metrics policy.
23. Restore or explicitly waive: IoA-NMS, yaw-consistency smoothing, attribute pre-fill, group boxes.

**During implementation (P2):** split build step 3 into paper-check and measured-check; replace the `Annotation_pipeline.md` ε table with the §7.2 formula; add the GT-box reprojection and paint-inside-GT tests; resolve degrees/radians; record that undistortion is untested on this substrate; note that Stage 2 is a branch, not a chain link.

---

## Questions That Must Be Resolved (decisions only you can make)

1. **Cameras and coverage:** all six nuScenes cameras (R2-like) or `CAM_FRONT` only (R1-like)? This drives the eval region, ρ, multi-camera handling, and roughly 6× of GPU time.
2. **Accumulation window:** preserve the spec's 5-sweep *count* (0.25 s at 20 Hz) or its ~0.5 s *duration* (≈10 sweeps)?
3. **Scene budget and partition:** how many usable scenes for the end-to-end run, and do you accept splitting the allowlist into disjoint priors / tuning / run subsets?
4. **nuScenes metrics:** forbidden in the pilot, or permitted on the held-out subset with leakage disclosed?
5. **Tracking regime:** accept predict-then-match at 2 Hz, or step up to the 12 Hz camera cadence for Stage 7?
6. **Waived spec stages:** confirm attribute pre-fill, group boxes, and forward-backward smoothing are deliberately out of pilot scope (yaw-consistency enforcement is cheap enough that I would keep it).
7. **`spatial_ok` drivable term:** implement against nuScenes maps, or drop it in the pilot and record the divergence?
8. **Metadata/blob layout:** are `v1.0-trainval/` and `samples/`+`sweeps/` under one root, or will the config need separate `dataroot` and `meta_root`?

---

## Things That Are Already Sound

These survived scrutiny and should not be changed:

- **Contracts-first build order.** `pipeline/common/` before any stage, with everything importing from it, is the correct dependency structure and is also where every P0 in this review is fixable.
- **Stage 0 as a pilot-only stage,** and specifically the decision to **exclude partially-present scenes loudly rather than skip frames quietly**. The reasoning given for it is exactly right.
- **Role registry over hardcoded checkpoints.** The right abstraction; it needs extending (P1-3), not replacing.
- **Keeping the F1 provenance invariant from the first line of code** rather than retrofitting it, with the stated rationale.
- **`eval_region.py` as the single implementation of E and ρ**, with ρ defined as count-over-area — including the observation that a locally reimplemented count is where a future R1/R2 decision would silently distort a metric.
- **Honest naming of the MobileSAM propagation loss** as a capability gap rather than a quality gap, with the consequence for Stage 7 stated plainly.
- **Refusing to treat the indigenous-prompt probe as S1**, and refusing to treat the pilot's OOD stage as discovery. Both are correctly scoped; only the containment mechanism needs hardening.
- **"Weight-file size ≠ inference VRAM"** and the `verified: true/false` flag mirroring §7.1.1 — the right instinct, undermined only by two arithmetic errors in the table it governs.
- **Downgrading I-2 honestly** rather than presenting nuScenes poses as PPK.
- **The test philosophy** — name the silent failure first, then write the test for it — and the unit/integration split with integration gated behind an env var so it cannot false-pass against absent data.
- **Explicitly deferring** §3–§5 of `comprehensive.md` as having no pilot substitute, rather than inventing one.
- **Stages 1, 4, 8 and 9's specific tests** (Δt-flagging intent, mask ordering, inflation direction, threshold boundaries) are well-targeted at real failure modes.
