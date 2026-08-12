# DhakaScenes Pilot — Implementation Plan (rev 2)

**Status:** planning document. Nothing described here has been built yet.
**Revision:** rev 2, 2026-08-12. Supersedes rev 1 in full. Rewritten to close the findings in `dhakascenes-pilot-validated-comet.md` (the pre-implementation audit) and to correct rev 1's substrate premise against the data actually on disk.
**Scope:** pilot only — nuScenes **v1.0-mini** as a stand-in substrate, 3050ti laptop GPU (4 GB VRAM), identical pipeline shape to `comprehensive.md` with swapped-down model checkpoints.
**Not in scope:** real Dhaka data, real sensor rig, anything from `comprehensive.md` §3–§5 (sensor suite, capture ops, PPK trajectory). Those need real hardware and have no pilot substitute.
**Governing documents:** `comprehensive.md` (canonical v1.0 spec), `GAP_ANALYSIS.md` (literature positioning), `dhakascenes-pilot-validated-comet.md` (audit; its finding IDs P0-n / P1-n / M-n / X-n are referenced throughout). `Annotation_pipeline.md` is **superseded** by `comprehensive.md` §7 and is cited here only where this plan explicitly rejects one of its values.

**Priority, unchanged:** fidelity over quality. Same stages, same contracts, same order as the real spec — smallest checkpoint that can fill each model role on 4 GB, even if label quality is bad. The goal is proving the *plumbing*, not producing usable labels.

**What rev 2 changes, in one paragraph.** Rev 1 specified stage *responsibilities* but almost never specified the *representation* stages exchange — coordinate frame, time base, 2D pixel space, cloud identity, clustering scope, units. Every one of those gaps produces output that runs, looks plausible, and is wrong. Rev 2 fixes the representation at every boundary in `pipeline/common/` before any stage exists, corrects two arithmetic errors in the VRAM table, corrects the substrate description, replaces the flat "build order" with ten sequential phases each carrying an explicit exit gate, and resolves the eight decisions the audit left open (§11).

---

## 0. The substrate — corrected against what is actually on disk

Rev 1 said "one partial `v1.0-trainval` blob chunk." **That is wrong.** The data at `/home/mt/Zami/nuscenes/` is the **complete v1.0-mini split**, and every file its metadata references is present. Verified 2026-08-12:

| Fact | Measured value | How it was measured |
|---|---|---|
| Metadata version | `v1.0-mini/` (13 tables) | directory listing |
| Scenes | **10** | `scene.json` |
| Keyframes (`sample`) | **404** | `sample.json` |
| `sample_data` records | **31,206** | `sample_data.json` |
| Missing files on disk | **0**, across all 12 channels | every `filename` stat-ed |
| Categories | **23** (dotted hierarchical names) | `category.json` |
| Instances / annotations | 911 / **18,538** | `instance.json`, `sample_annotation.json` |
| Image resolution | **1600 × 900** (all cameras) | JPEG header + `sample_data.width/height` |
| `CAM_FRONT` intrinsics | fx = fy = 1266.417, cx = 816.267, cy = 491.507 | `calibrated_sensor.json` |
| LiDAR sweep cadence | median **49.79 ms** (≈ 20 Hz) | consecutive `LIDAR_TOP` timestamps |
| Sweeps per keyframe | **9.74** (3,935 LiDAR records / 404 keyframes) | count |
| Point record size | **20 bytes** (5 × float32); example sweep = 34,688 points | `os.path.getsize` |
| Timestamps | 16-digit **microseconds, Unix epoch** (e.g. `1542801006946864`) | inspection |
| Extras present | `maps/` + map expansion, `can_bus/` (7,832 files) | directory listing |

**Consequences of the correction — all of them make the plan *more* constrained, not less:**

**0.1 — Stage 0 is not a salvage stage; it is a verification stage.** The "partial blob" motivation is gone, but Stage 0 stays, with its purpose rewritten (§5.1, P0-6). A probe that *assumes* completeness is exactly the class of assumption this project is trying to eliminate. It must now prove completeness rather than repair incompleteness, and it must check the things a file-existence check misses: referential integrity of the token graph, sweep coverage inside the accumulation window, file-size/parse validity, and the required-channel set. `usable_scenes.json` is still the only scene list any later stage reads.

**0.2 — The scene budget is 10 scenes / 404 keyframes, and that is small enough to change several plans.** With 6 cameras this is 2,424 keyframe images. Concretely:
- Stage 2's 1-in-10 sampling over *keyframes* yields **40 samples** — HDBSCAN on 40 points is meaningless. Sampling must be over *images* (242 samples) and even that is at the edge (§5.3).
- The disjoint priors / tuning / run partition (P1-5) has to come out of 10 scenes, so each subset is 2–4 scenes. Any statistic derived from it is descriptive, never a measurement (§11, decision 3).
- The 10 scenes are: 5 Singapore / 5 Boston, **3 of them night** (`scene-1077`, `scene-1094`, `scene-1100`), one after rain. That is a usable illumination spread for plumbing purposes and worth stratifying the partition across, not a sample size for any claim.

**0.3 — Taxonomy substitution, and the prompt-string trap.** nuScenes' 23 categories are not the indigenous 6-class taxonomy of `comprehensive.md` §6.1. Two things follow. First, the pilot cannot test S1 separability (`comprehensive.md` §7.2, `GAP_ANALYSIS.md` G3) — confirmed and unchanged. Second, and new: nuScenes category names are **dotted hierarchical strings** (`vehicle.emergency.ambulance`, `human.pedestrian.adult`). Fed to Grounding DINO as text prompts they tokenise to nonsense and produce near-random but non-empty detections. A `nuscenes_category → prompt_phrase` mapping table is therefore a mandatory artifact, not a nicety, and rev 1's proposed test ("does it build the 23-class prompt set") would have passed while asserting exactly the wrong thing.

**0.4 — Two extras are available and change two decisions.** `can_bus/` carries vehicle speed, so I-2's `stationary_flag` can be honestly derived rather than left null (P1-6). The map expansion is on disk, so Stage 9's drivable-area term is *implementable* — see §11, decision 7 for why it is still defaulted off.

---

## 1. Canonical conventions — the representation contract

This section is the substance of rev 2. It is written before any directory layout or stage description because every stage below depends on it, and because nine of the audit's P0 findings are closed here and nowhere else. All of this lands in `pipeline/common/conventions.py` and `pipeline/common/schemas.py` in **Phase 2**.

### 1.1 Coordinate frames — ego is canonical (closes P0-1)

`comprehensive.md` §2.3 fixes the body frame as ISO 8855: **x forward, y left, z up**, yaw about +z measured from +x, radians.

**Measured fact that makes this load-bearing:** nuScenes `LIDAR_TOP` is **not** aligned to the ego frame. Its `calibrated_sensor` rotation is `[0.70780, −0.00649, 0.01065, −0.70631]`, i.e. a yaw of **−89.883°**. In the raw sensor frame, +x points to the vehicle's **right** and +y points **forward**. (`CAM_FRONT`'s extrinsic yaw is −89.674°, the same rotation family.) This was verified against the actual `calibrated_sensor.json` on disk, not assumed.

Rules:
1. **The ego frame is canonical for all cross-stage geometry.** Points, boxes, yaw, velocity — everything crossing a stage boundary is in ego frame unless a field says otherwise.
2. **Every geometric record carries an explicit `frame` field** ∈ `{"ego", "lidar", "camera", "global"}`, validated on write. A record with no frame is invalid, not defaulted.
3. **Stage 1 applies `T_ego_lidar` exactly once.** No later stage re-applies it. Stage 5's chain (§1.3) starts from ego-frame points and goes forward.
4. nuScenes `"global"` is a per-map arbitrary frame, **not** the published-origin local ENU of §2.3. Where the pilot uses it (only as an intermediate in the projection chain) it is named `nuscenes_global`, never `global`.

**The test that must exist:** load one real `LIDAR_TOP` `calibrated_sensor` record and assert its rotation is **not** identity, with the measured −89.9° yaw as the expected value. It fails loudly the day someone assumes sensor ≡ ego.

**What goes wrong without this — all silent:** eval region R1 (`|θ| ≤ 55°` from +x) applied to LiDAR-frame points selects the vehicle's **right-hand sector**, so every count and every ρ is about the wrong 110° of the world; Stage 6 yaw is 90° off, so Stage 8 inflates length along the width axis and boxes grow sideways into adjacent lanes; Stage 7 velocity vectors are rotated 90°. Ground removal survives (z is up in both frames) — the one filter that would have crashed does not.

### 1.2 Time base and units (closes P1-7, X-7, X-8)

nuScenes ships **microseconds, Unix epoch**. `comprehensive.md` §2.3 mandates **int64 nanoseconds, GPS time**.

1. **Every temporal record carries `time_base` ∈ `{"unix_us", "unix_ns", "gps_ns"}`.** No exceptions, including intermediate artifacts.
2. **Conversion happens in exactly one function**, in `conventions.py`. Nothing else multiplies or divides a timestamp.
3. The pilot stores `unix_ns` internally (int64) and records `time_base: "unix_ns"`; it does **not** claim GPS time, because it does not have it. The divergence from §2.3 is recorded as a declared pilot deviation, not silently absorbed.
4. **The anchor is the `LIDAR_TOP` `sample_data.timestamp`** — never `sample.timestamp`, which is a different quantity.
5. **Every geometric and temporal schema field carries a unit suffix** (`_m`, `_rad`, `_ns`, `_mps`). This also resolves the inherited `comprehensive.md` inconsistency (§2.3 says radians, Appendix A.3 says `sigma_yaw_deg`, X-8): the pilot stores `sigma_yaw_rad` and converts at the release boundary only.

**Measured Δt, camera timestamp minus LiDAR anchor, over all 404 keyframes** — this replaces rev 1's guessed "hundreds of milliseconds" threshold, which would never have fired:

| Camera | min (ms) | median (ms) | max (ms) |
|---|---|---|---|
| `CAM_FRONT_LEFT` | −48.35 | **−43.06** | −40.36 |
| `CAM_FRONT` | −40.74 | **−35.44** | −33.76 |
| `CAM_FRONT_RIGHT` | −32.67 | **−27.53** | −25.77 |
| `CAM_BACK_RIGHT` | −25.04 | **−19.95** | −18.14 |
| `CAM_BACK` | −15.59 | **−10.36** | −8.41 |
| `CAM_BACK_LEFT` | −5.74 | **−0.48** | +1.20 |

Every camera fires *before* the LiDAR anchor, and the offset is a deterministic function of where the camera sits in the LiDAR's rotation. The Δt sanity threshold is therefore **per-camera**, derived from this measured distribution (median ± a stated tolerance), recorded in config with provenance `measured on 404 v1.0-mini keyframes, 2026-08-12`. A single global threshold is either never-firing or always-firing on `CAM_BACK`.

**The test that must exist:** the µs→ns conversion is tested against a **real** `sample_data` record pair with a physically known offset — not synthetic timestamps. Rev 1's synthetic-timestamp test shares any unit error with the code it tests and passes against a 1000×-wrong implementation. This is the clearest instance in rev 1 of a test that proves nothing.

### 1.3 The projection chain — four hops, one implementation (closes P0-2)

**Measured fact:** every nuScenes `sample_data` record has its **own** `ego_pose`. Within one keyframe, LiDAR and each camera are captured up to 48 ms apart (§1.2). `project_lidar_to_image()` in `conventions.py` is the single implementation of:

```
point_ego(t_lidar) → nuscenes_global → point_ego(t_cam) → camera → pixel
```

with `T_cam_ego` obtained by **inverting** nuScenes' sensor→ego extrinsic (nuScenes gives sensor→ego; projection needs ego→sensor — a classic inverse-direction error), `K` applied after extrinsics, and **`z ≤ 0` culled before the divide** (points behind the camera otherwise project to valid-looking mirrored pixels).

Dropping the two middle hops — the natural reading of rev 1's "projects points through calibrated intrinsics/extrinsics" — omits ego motion between the two timestamps. At 50 km/h and 35 ms that is ~0.5 m. The projection still lands on the image, still hits masks, still paints points; near-range objects acquire a systematic several-pixel offset, points bleed across mask boundaries onto neighbouring objects, and clusters are biased along the direction of travel. **A synthetic-point test with one static calibration cannot detect this** — the transform is mathematically valid and directionally wrong. See §9 for the two tests that can.

Also fixed here: near-zero-depth guard, out-of-bounds handling, and the multi-camera contest rule (§1.5).

### 1.4 Cloud identity — both clouds, and which one is lifted (closes P0-3, X-1)

`comprehensive.md` I-3 mandates **both** a single-sweep and an accumulated cloud; §7.3.5 says the lift projects ground-filtered **single-sweep** points. Rev 1 produced only the accumulated cloud and lifted from it. That is a contradiction with the spec, and it is not a quality difference:

- An accumulated cloud is **ego-motion**-compensated only. Dynamic objects smear across the window. Painting a smeared cloud with a single-frame mask and clustering it inflates box length for moving vehicles along the direction of travel — plausible boxes, wrong dimensions, worse for fast objects, which reads as "hard cases are harder" rather than as a bug.
- L-shape yaw is biased toward the motion direction, corrupting exactly the quantity `comprehensive.md` §7.3.7 and §8.2.1 (APH) treat as load-bearing.
- The §7.3.9 gate "LiDAR-return ≥ 5" and §6.3's "≥ 5 points in the **single-sweep** frame" both loosen by roughly the accumulation factor (~10× on this substrate). The gate still fires, still tiers boxes, and no longer means what the spec defines.

**Rule:** produce both clouds. **Fit** the ground plane on the accumulation (spec-faithful — more points, better fit), **apply** the resulting plane to the single-sweep cloud, **lift** from single-sweep, and **count `num_lidar_pts` on single-sweep, pre-inflation**. The accumulated cloud is retained for visualisation and the 40 m+ secondary track only. Every cloud artifact records `cloud_kind: "single_sweep" | "accumulated"`, `n_sweeps_actual`, and `window_ns` — because at scene starts the window is truncated and point density (hence cluster size, hence box dimensions) differs systematically for the first keyframes of every scene.

### 1.5 The 2D contract — absolute pixels, original resolution (closes P0-4)

Images are **1600 × 900** (16:9, measured). Grounding DINO emits normalised `cxcywh` at its own resized input scale; MobileSAM takes box prompts in a 1024-longest-side transformed space and returns masks at 1024 before postprocessing. Rev 1 declared none of this, and its VRAM fallback proposed "640×640" — a **square** resize of 16:9 imagery.

Rules:
1. **Every 2D quantity crossing a stage boundary is absolute pixels in original 1600 × 900 resolution.** `xyxy`, not `cxcywh`, not normalised.
2. **Each model adapter owns its own forward and inverse transform** and returns original-resolution outputs. The transform never leaks into stage code.
3. **Square resizes of non-square imagery are forbidden.** Letterbox with recorded padding, or resize-shortest-side. A non-aspect-preserving resize with a uniform inverse stretches every box along one axis; boxes still look like boxes, SAM still returns masks, and every object's 3D extent is wrong in one axis, consistently.
4. Masks are returned at 1600 × 900 before Stage 5 indexes them, asserted, not assumed.
5. **Multi-camera contest rule:** a 3D point visible in two overlapping cameras gets its class from the camera whose **principal axis is closest to the point's bearing**, ties broken by a fixed camera-priority list in config. Rev 1 left this undefined, which makes labelling depend on camera iteration order — non-deterministic.
6. **IoA-NMS (> 0.5) across overlapping cameras** (`comprehensive.md` §7.3.4) is **restored** — rev 1 dropped it without comment (M-2). It is pure geometry, costs nothing, and its absence is what makes duplicate assignment reachable.

### 1.6 Clustering scope — per mask instance (closes P0-5)

"Class-conditional DBSCAN, keep largest cluster" admits two readings that differ in output count by an order of magnitude. The contract fixes reading (a):

- **DBSCAN runs per mask instance.** "Class-conditional" refers to *which ε is used*, nothing else. "Keep largest cluster" is the **reprojection-ghost filter** within one instance's points.
- The rejected reading (b) — pool all points of a class across the frame, cluster once, keep the largest — yields **one box per class per frame**. In a parking row or a gridlock scene (the exact regime this project exists for) adjacent cars merge under any ε large enough to hold one car, and every other instance of that class is silently discarded. The output is clean, well-formed, and missing most objects.
- **Deterministic tie-break** for equal-size clusters: lowest mean range, then lowest first-point index under a canonical sort. DBSCAN label order depends on input point order, which depends on file read order.

### 1.7 Provenance enforcement (closes P0-7, X-9)

Three fixes to rev 1's invariant handling:

1. **Keep `validate() → [errors]`** for check-then-decide, **and add a single `write_records()` boundary that raises** on any invalid record. All persistence goes through it, on **write and on read-back**. Rev 1's non-raising design made "just don't inspect the returned list" the easiest bypass of the very invariant it was protecting; serialization was the open path — records round-tripping through plain dicts are reconstructed downstream with the validator never consulted.
2. **State both directions of the invariant.** Rev 1 stated only `split ∈ {val, test} ⇒ source ≠ pipeline_accepted`. The rule the pilot actually needs is the converse: **`source ∈ {human_verified, human_created} ⇒ verified_by ≠ null ∧ verification_pass ≥ 1`**. Add a pilot-wide `allow_human_provenance: false` that makes any human-provenance record a hard error.
3. **`tier` and `source` are different fields with different rules.** Rev 1's Stage 9 prose conflated `tier: auto_accept` with `source: pipeline_accepted`. They are not the same and neither implies the other.

**Test positively.** With no split populated, "no record claims `human_verified`" is **vacuously true** and passes against a completely broken validator. The tests must construct a `val` + `pipeline_accepted` record and assert rejection, and construct `human_verified` with `verified_by = None` and assert rejection.

### 1.8 Paths, fingerprints, and the allowlist binding (closes P0-9)

Rev 1 expressed the path model as a comment in a directory tree. Rev 2 makes it a contract:

- **One `configs/paths.yaml`** resolved once into a `Paths` object: `dataroot`, `meta_root` (identical here, kept separate for the split-extraction case), `version: "v1.0-mini"`, `work_root`, `out_root`, `probe_out_root`.
- **`validate_paths()`** at startup asserts: dataroot contains `v1.0-mini/`, `samples/`, `sweeps/`; is readable; the version directory name matches the configured version (mini metadata against trainval blobs yields zero token matches, which reads as "no usable scenes"); and `commonpath` **disjointness** between dataroot and both write roots.
- **Dataroot is read-only by convention**, with a test asserting no module opens a dataroot path for writing.
- **`usable_scenes.json` records the dataroot realpath and a metadata fingerprint** (sorted SHA-256 of the 13 metadata JSONs), and **every consumer verifies the match**. Without this, an allowlist computed against one dataroot and consumed by nine stages silently excludes good scenes or admits missing ones when pointed elsewhere.
- **`data/` is git-ignored; `work_root` and `out_root` default outside the repo.** A symlink inside the project tree is copied by `shutil.copytree`, `rsync -L`, or a Docker build context; `data/work/` under the repo fills the disk with intermediate clouds.
- The dataroot fingerprint appears in **every** stage manifest.

### 1.9 Determinism, manifests, and failure semantics (closes P1-8, P1-9)

`comprehensive.md` I-7 already requires "config snapshot + seed" with every result. Rev 1 had no seed policy and at least four sources of run-to-run variation — **sector-wise RANSAC is stochastic and is the very first geometric operation**, so unseeded it produces a different ground plane, different retained points, different clusters, and different boxes on every run; UMAP is stochastic (and `random_state` forces single-threaded execution, so seeded and unseeded runs are not comparable); DBSCAN labels depend on point order; cuDNN autotuning and TF32 change numerics.

- **One global seed**, threaded into RANSAC, UMAP, and any sampling. Recorded.
- **`run_manifest.json` per stage**: seed, resolved config hash, git commit, package versions (Python / PyTorch / CUDA / UMAP / HDBSCAN / sklearn), checkpoint IDs + revisions + SHA-256, dataroot fingerprint, `usable_scenes.json` hash, scene partition + its seed, priors version and `source`, `W_acc` count and duration, actual image resolution and prompt configuration, measured VRAM peaks per role, per-stage input/output counts.
- **Atomic writes** (temp + rename), a **`_SUCCESS` marker**, and a downstream **refusal to start** when the upstream manifest is missing or its input fingerprint differs. Without this a partially written stage output is indistinguishable from a complete one.
- **Per-scene isolation:** one scene's failure lands in `failures.json` with the failing predicate; it does not abort the run.
- **Idempotent re-run** keyed on the manifest.
- **OOM is a hard stop** with stage / role / resolution logged — never a silent retry at lower resolution, which would change the output distribution mid-run.
- **Determinism test:** run one scene twice, byte-compare outputs.

### 1.10 Eval region and coverage config (closes P1-4, M-6)

`comprehensive.md` §2.3 requires `coverage_config ∈ {R1, R2}` on every release table; §3.6 defines E differently for each. Rev 1 mentioned neither and never said whether the pipeline runs on all six cameras or one. That single unmade decision determines the eval region, ρ's area normalisation, whether the multi-camera union and IoA-NMS are needed at all, what an "OOD frame" even is, and ~6× of GPU time.

**Resolved (§11, decision 1): `coverage_config: R2`, all six ring cameras.** nuScenes' 6-camera ring is R2-like and R2 is `comprehensive.md`'s default. `coverage_config` is recorded in **every** output record and E is **derived from it** in `eval_region.py`, never from a constant. A `camera_subset` config exists and a CAM_FRONT-only run is legal — but such a run must record `coverage_config: R1` and is a different experiment, not a cheaper version of the same one.

`in_region(x, y, spec)` remains the **only** place that decides membership, and ρ remains **count over area**. This survived the audit unchanged and should not be touched.

---

## 2. Directory layout

```
dhakascenes_pilot/
  pipeline/
    common/                    # contracts — Phase 2, everything depends on it
      conventions.py           # §1.1–§1.6: frames, time base, units, yaw, 2D contract, projection chain
      schemas.py               # I-1..I-5 dataclasses + validate() + write_records() (§1.7)
      eval_region.py           # in_region() + rho(), derived from coverage_config (§1.10)
      paths.py                 # Paths object + validate_paths() + fingerprints (§1.8)
      manifest.py              # run_manifest.json, atomic write, _SUCCESS, seeds (§1.9)
      model_interfaces.py      # role Protocols + registry (§7)
    stage0_data_probe/         # pilot-only substrate verification (§5.1)
    stage1_ingestion/          # keyframing, both clouds, ground removal + diagnostics (§5.2)
    stage2_ood/                # DINOv2 OOD/long-tail frame discovery — a branch, not a chain link (§5.3)
    stage3_proposals/          # 2D open-vocab proposals (§5.4)
    stage4_masks/              # mask generation + IoA-NMS (§5.5)
    stage5_lift/               # 2D→3D reprojection, four-hop (§5.6)
    stage6_cluster/            # per-instance DBSCAN + L-shape fit (§5.7)
    stage7_track/              # predict-then-match association + ICP + Kalman fallback (§5.8)
    stage8_inflate/            # prior-anchored inflation (§5.9)
    stage9_qa/                 # gate vector → tier (§5.10)
  probes/
    indigenous_prompt_probe/   # HARD-SEPARATED informational probe (§8) — never writes to out_root
  configs/
    paths.yaml                 # §1.8
    models_pilot.yaml          # 4 GB tier (§7.2)
    models_production.yaml     # full-size, same role names (§7.3)
    taxonomy_pilot_nuscenes.yaml    # nuscenes_category → prompt_phrase mapping (§0.3)
    taxonomy_probe_indigenous.yaml  # probe only, physically separate (§8)
    pipeline_pilot.yaml        # every tunable in §10, each with a provenance note
  scripts/
    probe_substrate.py         # Phase 1 standalone, no GPU
    check_vram_paper.py        # Phase 5a: arithmetic only, no download
    measure_vram.py            # Phase 5b: real forward pass, real card
    run_pilot.py               # end-to-end driver
  tests/
    unit/                      # fast, synthetic fixtures, no GPU
    integration/               # real data / real GPU, opt-in via env var
    fixtures/                  # synthetic nuScenes-shaped generator
  priors/                      # priors_pilot_v*.json — source: "nuscenes_gt_pilot" (§8)
```

`data/` is **not** in the repo (§1.8). `work_root` and `out_root` default outside it.

---

## 3. Contract spine (I-1 → I-7)

| Contract | Spec producer → consumer | Pilot stand-in | Status after rev 2 |
|---|---|---|---|
| I-1 | Capture rig → everything | nuScenes `sample_data` via the Stage 0 allowlist | Valid. The missing piece — the **required-channel set**, the analogue of "a bag missing a mandatory topic fails ingestion" — is now enumerated in config (§5.1). `session_meta.json`'s analogue is the dataroot fingerprint (§1.8). |
| I-2 | PPK/RTS trajectory → ingestion, release | nuScenes `ego_pose.json` | Downgrade now **mechanised**, not asserted (§3.1). |
| I-3 | Ingestion → annotation | keyframe pack + **both** clouds | Fixed (§1.4). Now also records `n_sweeps_actual`, `window_ns`, and `undistorted: true, method: "nuscenes_native"`. |
| I-4 | Annotation → CVAT | pre-labels + gate vector + tier, `source: "pipeline"` | Schema was fine; enforcement path is now closed (§1.7). Yaw→quaternion is an explicit, tested conversion (§3.2). |
| I-5 | CVAT → verified labels | simulated — no human annotators | Both directions of the invariant now stated and positively tested (§1.7). |
| I-6/I-7 | Release → benchmark → paper | shape only | I-7's "config snapshot + seed" is **not** deferred — it is Phase 2 work (§1.9). |

### 3.1 I-2, mechanised (closes P1-6)

nuScenes supplies translation, rotation, timestamp. It supplies **none** of I-2's quality fields. The failure mode is filling `sigma_pos_m = 0.0` to satisfy a dataclass, which every consumer then reads as *better than PPK*.

- `sigma_pos_m`, `sigma_yaw_rad`, `fix_type` → **`null`**, never zero.
- Mandatory `pose_source: "nuscenes_ego_pose"` and `quality_known: false`.
- **Consumers fail closed on `null`** — a stage that needs a σ and finds `null` errors; it does not default.
- `stationary_flag` **is** honestly derivable — from ego speed, and `can_bus/` is on disk if a better source is wanted. `fix_type` is not derivable and stays `null`.

### 3.2 I-4 field gaps

- **`rotation` is a quaternion; Stage 6 produces a scalar yaw.** That conversion is an unwitnessed point where an axis or sign error hides invisibly. It needs a **yaw → quaternion → yaw round-trip test** against `conventions.py`'s yaw definition.
- **`size` is `[w, l, h]`** (nuScenes order). KITTI is `[h, w, l]` and many L-shape implementations return `(length, width)`. A swap rotates every box 90° while all values stay plausible. The order is stated in the schema and asserted.
- **`num_lidar_pts` is counted on the single-sweep, ground-filtered, pre-inflation cloud** (§1.4). nuScenes' own field counts single-sweep *including ground* — the two are not comparable and the difference is recorded.
- **`attribute` and `visibility` have no producer.** Declared out of scope (M-1) — see §11, decision 6.

---

## 4. What the pilot deliberately does not do (declared, not silent)

Rev 1 dropped several spec components without saying so. Each is now either restored or explicitly waived:

| Spec component | Rev 2 decision | Rationale |
|---|---|---|
| Stage 10 — attribute pre-fill (§7.3.10) | **Waived** | Depends on human confirmation the pilot has no path for. Leaves I-4 `attribute` unproduced, declared. |
| IoA-NMS across cameras (§7.3.4) | **Restored** | Pure geometry, free, and its absence makes duplicate assignment reachable (§1.5). |
| SAM 2.1 mask propagation (§7.3.4) | **Waived — capability gap, not quality gap** | MobileSAM has none. Stated plainly; Stage 7 will look worse for a structural reason, not a tuning one. |
| Forward-backward track smoothing (§7.3.7) | **Waived** | Offboard refinement; not needed to prove association plumbing. |
| Yaw-consistency enforcement along tracks (§7.3.7) | **Restored** | Free geometry and the designed 180°-flip defence — directly relevant to an L-shape fitter on symmetric objects. |
| Group / ignore boxes (§6.3, A.2) | **Waived** | `GroupAnnotation` stays in `schemas.py` as shape only, with no producer or consumer. Declared, so ρ's `n_min` term is not silently misapplied. |
| `spatial_ok` drivable term (§7.3.9) | **Off by default, flagged** | See §11, decision 7. |
| Undistortion (§7.3.1) | **No-op on this substrate** | nuScenes ships rectified imagery and `camera_intrinsic` with no distortion coefficients. Recorded as `undistorted: true, method: "nuscenes_native"` so nobody later reads "we did reprojection on real calibration" as evidence the distortion path was tested. It was not. |

---

## 5. Stage specifications

### 5.1 Stage 0 — Substrate probe (pilot-only)

**Purpose (rewritten):** prove the substrate is what the config says it is, and produce a trustworthy, fingerprint-bound allowlist plus the scene partition. No model, no GPU.

Rev 1's probe checked file existence, which — on complete v1.0-mini — passes trivially and catches nothing. `fully_present` must be **executable predicates**, each emitted by name when it fails, not a bare bucket label (P0-6):

1. **`channels_complete`** — every keyframe has a `sample_data` record for every channel in the **declared required-channel set**. That set is config: `LIDAR_TOP` + the 6 `CAM_*`. **RADAR is excluded** — including it would exclude scenes for reasons the pipeline does not care about; excluding it silently would make the allowlist not mean what its name says. Declared either way.
2. **`files_resolve`** — every referenced file exists.
3. **`files_parse`** — size and parse checks. A truncated `.pcd.bin` whose length is still a multiple of **20 bytes** reshapes to `(-1, 5)` without error and yields silently fewer points; a truncated JPEG often decodes to a partial image. Both produce plausible downstream output. Check: `size % 20 == 0`, size within a sane band, JPEG header + tail marker.
4. **`sweeps_cover_window`** — every LiDAR sweep within `W_acc` of every keyframe exists. Rev 1 never mentioned sweeps; accumulation reads `sweeps/LIDAR_TOP/*.pcd.bin` and would hit a missing file mid-run, which is the precise failure Stage 0 exists to prevent.
5. **`token_graph_closed`** — `sample_data → ego_pose`, `sample_data → calibrated_sensor`, `sample → sample_annotation → instance → category`, and intact `prev`/`next` chains. A partially extracted metadata tarball otherwise surfaces as a `KeyError` inside Stage 5. Stage 8's priors depend on `sample_annotation`, which rev 1's probe never looked at.
6. **`version_matches`** — metadata directory name equals the configured version.

**Outputs:** `usable_scenes.json` (with dataroot realpath, metadata fingerprint, required-channel set, and the `W_acc` used — both change the answer), `probe_report.json` (per-scene failing predicates), and the **scene partition** (§11, decision 3).

**Hard stops:** version mismatch; zero usable scenes; dataroot missing `samples/` or the version directory.

**Keep from rev 1:** excluding partially-present scenes **loudly** rather than skipping frames **quietly**. The reasoning was right and survives.

### 5.2 Stage 1 — Ingestion

Builds I-3. Per-keyframe camera paths, LiDAR sweep paths, ego-pose tokens, and the **measured** per-camera Δt (§1.2) — not a nominal value.

- **Transform LiDAR points to ego frame once** (§1.1). Everything downstream is ego frame.
- **Produce both clouds** (§1.4). Accumulation is ego-motion-compensated using each sweep's **own** `ego_pose` (the devkit's approach, correct for accumulation).
- **`W_acc` = 0.5 s duration ≈ 10 sweeps** at this substrate's 20 Hz (§11, decision 2). Record **both** the nominal window and `n_sweeps_actual` per cloud — at scene starts the window is truncated and no stage may assume constant density. The window is a config value, never a hardcoded "5 sweeps".
- **Ground removal:** sector-wise RANSAC on the accumulation, discard |z − ground| < 0.3 m, prune > 40 m range and > 4 m height inside E. **The fitted plane is applied to the single-sweep cloud**, which is what Stage 5 lifts.
- **Per-filter point-count diagnostics are a first-class output** (P1-13): input → post-ground → post-range → post-height, per frame **and per sector**. Without them there is no way to distinguish "few objects because detection is bad" from "because filtering deleted them."
- **RANSAC sector count, distance threshold, iteration count, and seed are config** — none appear in either governing document.

**Parameter provenance warning:** 0.3 m / 40 m / 4 m were chosen for a **Livox Mid-360 on Dhaka roads** and are here applied to a **32-beam spinning LiDAR on Boston/Singapore roads**. Concretely: a 0.3 m band removes all wheel returns from every vehicle and most of a traffic cone's body, pushing small classes below the ≥ 5-point gate; sector RANSAC mis-fits on ramps, speed bumps and cambered roads; a single plane per sector cannot represent a curb. All three are config with recorded provenance `comprehensive.md §7.3.1, unvalidated on this substrate`.

**Validation worth having:** an accumulated-cloud **static-structure sharpness** check — building facades should stay thin across sweeps. Cheap, and decisive proof that ego compensation is applied in the right direction.

### 5.3 Stage 2 — OOD / long-tail discovery (a branch, not a chain link)

DINOv2 `embedding_ood` → UMAP → HDBSCAN; frames outside the dominant manifold flagged. The purpose is plumbing: the embedding → projection → clustering path runs end to end. **No taxonomy changes result** — nuScenes' taxonomy is known and there is nothing to discover.

Three corrections:

1. **This is not OOD detection.** It is clustering of a non-metric 2-D projection. UMAP preserves neither density nor global structure, so HDBSCAN membership in UMAP space has no formal relationship to outlyingness in the 384-D embedding space. If any outlier claim is wanted, decide in **embedding space** — HDBSCAN GLOSH outlier scores on the original embeddings — and keep UMAP for visualisation only.
2. **Scale.** 404 keyframes at 1-in-10 gives **40 samples**, on which HDBSCAN returns one cluster plus noise and rev 1's success criterion ("not zero, not thousands") is unfalsifiable. **Sample over images, not keyframes**: 2,424 images at 1-in-10 = **242**. State the expected sample count and check it before running.
3. **Stage 2 feeds nothing else in the pilot**, which contradicts rev 1's "0→9 is a real dependency chain." Harmless, but it means Stage 2 is a *branch*: it can be built and run out of order, and it is scheduled next to Stage 7 (Phase 9) because both are DINOv2 consumers.

UMAP `n_neighbors` / `min_dist` / `random_state` and HDBSCAN `min_cluster_size` / `min_samples` are config — Small-model embeddings will not transfer their tuning to Large.

### 5.4 Stage 3 — 2D proposals

Grounding DINO Tiny with the nuScenes prompt set.

- **Prompt strings come from `taxonomy_pilot_nuscenes.yaml`'s `nuscenes_category → prompt_phrase` mapping** (§0.3), never from raw dotted category names.
- **Phrase-span mapping is the sharpest silent failure in this stage.** Grounding DINO emits per-token logits over the concatenated prompt; recovering which *phrase* a box belongs to needs correct token-span bookkeeping, and multi-word classes make it error-prone. A span bug yields well-placed boxes with **wrong labels**, which then select the wrong DBSCAN ε, the wrong dimension prior, and the wrong inflation target — every downstream stage runs perfectly.
- **Per-class thresholds with a global default**, matching `comprehensive.md` §7.3.3. Rev 1's single tuned scalar contradicts the spec (X-5) and is indefensible anyway: Grounding DINO scores are not calibrated across phrases and longer phrases score systematically lower. A single tuned value is acceptable *as a pilot default* only if the interface accepts a per-class dict, so production needs no Stage 3 change.
- **Deduplication** of overlapping/duplicate proposals — unspecified in rev 1.
- **Output is absolute pixels at 1600 × 900** (§1.5).
- Every Stage 3 record records the **actual image resolution and prompt configuration used** (P1-14).

**Prompt chunking is forbidden unless re-tuned and re-recorded.** Chunking the 23-phrase prompt to save VRAM is the obvious mitigation and it **changes confidence semantics** — scores derive from token-level logits over the full prompt, so a 6-phrase chunk's scores are not comparable to a 23-phrase prompt's, and a threshold tuned under one regime is meaningless under the other.

### 5.5 Stage 4 — Masks

MobileSAM (~10 M params, ~40 MB), chosen over SAM-ViT-B (~91 M params, ~375 MB checkpoint) because the ~9× parameter difference matters at 4 GB. Commit to one, not "or".

- **Box prompts are converted into MobileSAM's transformed space by the adapter**, and masks come back at **1600 × 900**, asserted (§1.5).
- **IoA-NMS > 0.5 across overlapping cameras** (§4).
- **Capability gap, stated plainly:** no cross-frame propagation, so masking is independent per frame with no temporal consistency. Stage 7's tracking quality will be worse for a structural reason.

### 5.6 Stage 5 — 2D→3D lift

The four-hop chain of §1.3, on **ground-filtered single-sweep** points (§1.4), in ego frame (§1.1), indexing masks in original resolution (§1.5), with the multi-camera contest rule and per-camera frusta **unioned** for R2 coverage (`comprehensive.md` §7.3.5, absent from rev 1).

Required guards: `z ≤ 0` cull before divide; near-zero-depth; out-of-bounds; deterministic overlap resolution.

### 5.7 Stage 6 — BEV clustering + box fit

Per-instance DBSCAN (§1.6), L-shape fit → oriented box. Geometric, no learned model, same code on both tiers.

- **ε comes from `comprehensive.md` §7.2's formula — `ε ≈ 0.6 × mean footprint diagonal`, derived from the priors subset** — not from `Annotation_pipeline.md`'s hardcoded table (X-6). That table's class names (`cyclist`, `traffic_cone`, `car`) are **not nuScenes categories**, so any lookup keyed on them silently misses and gives every unmatched class the same default ε. `min_samples` is config.
- **`size` order is `[w, l, h]`**, asserted (§3.2).
- **Near-square footprints** (pedestrians, cones, barriers) have a **systematic** 90° ambiguity, not an occasional one. Policy: enforce `w ≤ l` and record an `yaw_ambiguous` flag.
- Yaw is asserted against `conventions.py`'s definition, never the fitter's own.

### 5.8 Stage 7 — Tracking

**The 2 Hz problem (P1-1).** `comprehensive.md` assumes a 10 Hz canonical rate; nuScenes keyframes are 2 Hz. A vehicle at 40 km/h moves ~5.5 m between keyframes, so **3D IoU between consecutive detections of the same object is zero for most vehicles** — the association term `3D IoU × DINOv2 cosine` collapses to appearance-only and the IoU half is never exercised. Appearance is degraded too: distant nuScenes crops are tens of pixels, and DINOv2's patch-14 tokenisation reduces a 28×28 crop to a 2×2 patch grid, so cosine similarity approaches degeneracy exactly where IoU has already failed.

**Resolution (§11, decision 5): predict-then-match.** Propagate the previous box by its estimated velocity before computing IoU. Record the **effective inter-frame Δt** in every track record. The IoU gate is a config value with provenance `derived for 2 Hz`, not a spec-copied constant. State a **minimum crop size** below which appearance similarity is not trusted.

**Specify what rev 1 left blank (M-11):** matcher algorithm (Hungarian), gate-then-score order, how the two terms combine (product, per `comprehensive.md`), gate thresholds, one-to-many / many-to-one handling, **track birth and death rules**, occlusion handling.

**ICP frame matters critically.** ICP between two **ego-frame** clusters measures motion *relative to the ego vehicle*; nuScenes mAVE is absolute. Register in the global frame or subtract ego motion explicitly, and **state which** in the record.

**Kalman fallback, specified:** constant-velocity state `[x, y, z, vx, vy, vz, yaw]`, Δt from recorded timestamps (`time_base` explicit), process/measurement noise in config, initialisation stated, trigger `< 15 points` (`comprehensive.md` §7.3.6).

**Yaw-consistency enforcement along tracks** is restored here (§4).

### 5.9 Stage 8 — Inflation

**"Inflate toward class mean" is not well-defined by A.4 alone** (M-12). A.4 gives `dims: {mu, sigma}` and nothing about how much, when, or along which axes. Add as explicit `priors_v*.json` fields or config with provenance: **trigger** (point count below threshold), **blend function** (all-the-way-to-mean vs point-count-weighted), **per-axis applicability**, **clamp**.

- **Anchoring, stated correctly:** LiDAR visibility bias means the observed face is always the **near** face. The rule is *anchor the near face and grow the far face* — not the vaguer "shift outward". Guard against growth back through the sensor-facing surface.
- Under a 90° yaw error from Stage 6, inflation grows the box **sideways into the neighbouring lane**, deterministically and invisibly. Hence §1.1.
- **Every inflated box records `inflated: bool` and `inflation_fraction`.** Without it, a run where ground removal stripped most object points produces boxes that are ~90 % prior and 10 % measurement, and nothing in the output says so.

### 5.10 Stage 9 — QA gating

Gate vector {confidence, spatial sanity, LiDAR-return count} → `auto_accept` / `flagged` / `rejected`.

- **The spatial gate runs on pre-inflation dimensions** (P1-12). `comprehensive.md` §7.3.9 defines it as "BEV box exceeds 2× class prior" and Stage 8 inflates *toward* the prior — so any box through Stage 8 passes the gate more easily, and the gate's discriminative power is lowest exactly where it is needed. This is an **inherited** spec flaw; the pilot surfaces it rather than reproducing it silently. Record both dimensions, gate on the measured one.
- **The return-count gate counts single-sweep, ground-filtered, pre-inflation points** (§1.4), so "≥ 5" means what the spec means.
- **`spatial_ok`'s drivable term is off by default** (§11, decision 7).
- **`tier` ≠ `source`** (§1.7). The provenance tests are the positive-rejection ones, not the vacuous one.

---

## 6. Priors — what the pilot can and cannot stand in for

`comprehensive.md` §7.2's S0 is 3,000–5,000 fully-manually-annotated keyframes. The pilot has no annotators and no indigenous taxonomy, so it cannot produce one. What it can do:

- Derive per-class dimension priors and DBSCAN ε from nuScenes `sample_annotation.json` — real human-quality labels already exist for this substrate — **from the `priors` scene subset only** (§11, decision 3).
- **Derive priors under the same E and range constraints the pipeline operates under.** nuScenes GT includes boxes beyond the 40 m cap and boxes with **zero** LiDAR points. If priors come from the full annotation population while the pipeline only ever sees objects inside E within 40 m, inflation pulls toward a mean the sensor never measures.
- Write `priors/priors_pilot_v0.json` in the A.4 schema so Stages 6 and 8 consume it through the same interface the real S0 output will use — proving the **consumption** side of the contract. The **production** side (real S0 annotation effort) is untouched.

**Two guards (P1-11):**
1. The file sets **`source: "nuscenes_gt_pilot"`**, not `"S0"`. `comprehensive.md` §7.2 is explicit that nuScenes/KITTI values are "initialization only, **deleted after S0**", and the pilot writes a file in the same schema, in the same directory shape, consumed through the same interface.
2. **The release builder rejects any priors file whose `source` is not `"S0"`** — five lines that stop a nuScenes-shaped prior reaching a Dhaka release.

**Licence note [VERIFY]:** the nuScenes licence (CC BY-NC-SA class) bears on redistributing GT-derived statistics. Check before `priors_pilot_v0.json` lands in any public repo.

---

## 7. Model-swap architecture

### 7.1 Roles, and where "same role" ≠ "same interface"

Four roles cover the model-dependent stages. Stages 6, 8, 9 are geometric — no role, same code on both tiers.

| Role | Used by | Pilot | Production |
|---|---|---|---|
| `embedding_ood` | Stage 2 | DINOv2 ViT-S/14 | DINOv2 ViT-L/14 |
| `proposal_2d` | Stage 3 | Grounding DINO Tiny | Grounding DINO Swin-L, **or SAM 3 text-prompt** pending `GAP_ANALYSIS.md` §6 licence check |
| `mask_2d` | Stage 4 | MobileSAM | SAM 2.1, or SAM 3 |
| `reid_embedding` | Stage 7 | DINOv2 ViT-S/14 | DINOv2 ViT-L/14 |

**Three interface fixes rev 1 needed (P1-3, X-10).** `GAP_ANALYSIS.md` §6 recommends promoting **SAM 3 to a unified proposal + mask + track engine**. Under rev 1's one-model-per-role registry, the single most likely production choice cannot be expressed, and "swap models later = edit which YAML loads" fails for exactly that case.

1. **`mask_2d` takes an optional temporal-state parameter and window size from day one.** MobileSAM's adapter ignores them. Without this, the SAM 2.1 swap — `(video, boxes, memory_state) → masks over time`, with propagation window and state re-init at block boundaries — is a Stage 4 rewrite.
2. **One provider may register against multiple roles** (composite provider).
3. **`proposal_2d` may optionally return masks**, in which case Stage 4 is a pass-through. DINO-X-class and SAM-3-class models return box + mask jointly, collapsing the Stage 3/4 boundary.

**Preprocessing belongs to the role, not the model (P1-2).** `embedding_ood` embeds a whole image (CLS token, ~518×518 input); `reid_embedding` embeds a small object crop (ideally mask-pooled patch tokens). Same weights, **different preprocessing and different output semantics**. If the registry hands back "the DINOv2 model" and Stage 7 inherits Stage 2's transform, tiny crops get upsampled ~20× and similarity is dominated by interpolation artifacts.

**The shared-instance VRAM claim is dropped (X-2).** Rev 1 said loading DINOv2 twice "burns ~700 MB" and called it the easiest way to blow the budget — while also mandating "serial, one role loaded at a time, cache cleared between stages." Both cannot be true: under the serial policy the Stage 2 instance is freed long before Stage 7, so two copies can never be resident, and the marginal cost of reloading is a few seconds of disk I/O. Registry singleton behaviour stays as a convenience; the zero-marginal-VRAM claim goes.

**Teardown must be asserted, not assumed.** `empty_cache()` does not free memory still referenced by a live Python object. Between stages, assert `torch.cuda.memory_allocated() ≈ 0`. The registry can enforce singletons; it cannot enforce ceilings.

### 7.2 `configs/models_pilot.yaml`

`total_device_mb: 4096`, a measured `system_reserve_mb`, a derived `hard_ceiling_mb`, and **a stated hard-stop behaviour when measured > ceiling** (rev 1 defined the ceiling but not the consequence). Per-role entries carry checkpoint id **+ revision + SHA-256**, VRAM estimate, and a `verified: true/false` flag mirroring `comprehensive.md` §7.1.1 — unverified is fine to *run*, never fine to *quote*.

### 7.3 `configs/models_production.yaml`

Same role names, same shape, bigger checkpoints, plus the composite-provider case from §7.1.

---

## 8. Hard separation of the indigenous-prompt probe (closes P1-10)

Rev 1 scoped the indigenous-class probe as informational and refused to call it S1 — correct. But the containment was a **naming convention**, and the realistic contamination path is concrete: Grounding DINO prompted with "cycle rickshaw" on Boston/Singapore imagery **will** return boxes, with confidences. A screenshot or JSON of those boxes is one context-loss away from reading as evidence that the model separates indigenous classes — the precise claim `GAP_ANALYSIS.md` G3 says nobody has measured.

Mechanism, not labelling:
- **Separate output root** (`probe_out_root`), separate code path (`probes/`), separate taxonomy file.
- Every probe record carries `experiment: "indigenous_prompt_probe"` and `not_evidence_for: "S1"`.
- **No probe output ever enters `out_root` or the priors.**
- **A test asserts** no indigenous prompt string appears in the pilot taxonomy array, and no probe record appears in main outputs.
- Any sentence written about it carries the explicit form: *"prompt-injection mechanism check on non-Dhaka imagery; not a separability result."*

---

## 9. Test plan

**Method, unchanged and not to be diluted:** before implementing a stage, write down (a) the one or two failure modes that would be **silent** — wrong output that still looks plausible and does not crash — and (b) a test that specifically catches that failure mode. Where rev 1 applied this it produced genuinely strong tests (Stage 0's bucket test, Stage 4's ordering test, Stage 8's direction test). The gap was that it was applied to about half the stages.

**Split:** unit tests are fast, GPU-free, synthetic-fixture. Integration tests touch the real chunk and/or the real GPU, opt-in behind an env var so they never silently false-pass against absent data.

| Stage | Keep | **Replace — would pass on broken code** | Add |
|---|---|---|---|
| 0 | 3-bucket fixture | — | truncated `.pcd.bin` (length still a multiple of 20); dangling `ego_pose` token; scene missing sweeps for its first keyframe; allowlist ↔ dataroot fingerprint mismatch |
| 1 | Δt-flagging *intent* | **Δt on synthetic timestamps** (test and code share the unit bug); **flat z=0 ground fixture** (a *global*-plane implementation passes identically, so it never tests the "sector-wise" property) | µs→ns conversion against a **real** record pair; **sloped multi-sector** ground fixture; per-filter point-count diagnostics; static-structure sharpness on the accumulated cloud; `n_sweeps_actual` recorded at scene start |
| 2 | OOD list ⊆ input IDs, no invented IDs | **synthetic vectors through UMAP** (degenerates at small n — `n_neighbors` must be < n_samples — so it either fails spuriously or exercises a bypassed path) | flagging rule on **precomputed** labels/outlier scores, UMAP excluded from unit tests entirely; expected-sample-count check |
| 3 | threshold filtering on synthetic boxes | **"builds the 23-class prompt set"** — asserts the wrong thing (dotted names) | prompt strings are natural-language phrases; **phrase→class span mapping**; output in original 1600×900 pixels |
| 4 | one mask per box, same order — catches the silent misassignment | — | mask returned at original resolution; box-prompt coordinate-space round-trip within 1 px |
| 5 | known point → known pixel (necessary, insufficient) | — | **GT-box reprojection** (project nuScenes GT box centres/corners into each camera, assert they land on the object); **paint-inside-GT rate** (mask-painted points fall inside the corresponding GT 3D box at a high rate); behind-camera cull; near-zero depth; out-of-bounds; two-camera overlap determinism; lidar→pixel→lidar round-trip |
| 6 | noise-rejection with scattered points | **square-rectangle fixture** — hides `[w,l]` swaps and 90° ambiguity | **non-square** rectangle at **30° yaw**, asserted against `conventions.py`; yaw→quat→yaw round-trip; **two same-class instances 1 m apart → two boxes**; deterministic tie-break |
| 7 | fabricated similarity matrices; min-point Kalman-fallback trigger | — | `frame` and comparable `t` assertion on association inputs; ICP-frame test distinguishing relative from absolute velocity; **the ≥3-frame track integration test** (below) |
| 8 | anchor-direction test — one of the strongest in the plan | — | deliberately-wrong-yaw case documenting the failure mode; `inflation_fraction` recorded |
| 9 | threshold boundary + combination cases | **provenance test is vacuous** with no split populated | positive rejection: `val` + `pipeline_accepted` **must raise**; `human_verified` with `verified_by=None` **must raise**; post-serialization revalidation |
| cross | eval-region area vs closed-form geometry | — | contract round-trip (write → read → validate); `frame`/`time_base`/unit assertions on every schema; **determinism** (same scene twice, byte-compare); golden-scene regression fixture; **`LIDAR_TOP` calibration is non-identity** |

**On rev 1's own open question about MobileSAM and Stage 7.** The lack of propagation does **not** by itself break Stage 7 — association operates on 3D clusters and crop embeddings, not propagated masks. The real damage is mask flicker → unstable per-frame clusters → fragmented tracks. So association-logic unit tests are **not** sufficient alone, but the missing piece is small: **run Stages 3→6 over ~10 consecutive keyframes of one usable scene and assert at least one object yields a track of ≥ 3 frames with a stable ID.** If that fails, Stage 7 has no real input and the association logic is only ever tested against fabricated matrices.

---

## 10. Configuration and provenance

**Everything below is config, and every value carries a provenance note** — `spec §x.y` / `measured on N frames, date` / `arbitrary, needs tuning`:

Paths (`dataroot`, `meta_root`, `work_root`, `out_root`, `probe_out_root`, `version`); required-channel set; `camera_subset`; `coverage_config` (R1/R2) and eval-region parameters; `W_acc` count **and** duration; RANSAC sector count / distance threshold / iterations / seed; ground band 0.3 m; range cap 40 m; height cap 4 m; image resolution and resize policy; prompt taxonomy file; prompt chunking (default: forbidden); per-class confidence thresholds + default; MobileSAM input size; per-class DBSCAN ε and `min_samples`; cluster tie-break; association IoU and similarity gates, matcher, track birth/death, minimum crop size; Kalman noise; min-point ICP guard (15); inflation trigger/blend/clamp; QA thresholds; UMAP `n_neighbors`/`min_dist`/`random_state`; HDBSCAN `min_cluster_size`/`min_samples`; sampling rate (1-in-10, over images); global seed; VRAM ceiling and reserve; checkpoint ids + revisions.

**Values inherited with no pilot validation** — each flagged in config: 0.3 m, 40 m, 4 m, 1-in-10, 0.40 confidence, 15-point ICP guard, ≥ 5 LiDAR returns, 2× spatial multiplier, and the system-reserve estimate. Several were derived for different hardware, a different city, and a different sensor. The `Annotation_pipeline.md` ε table is **rejected outright** (§5.7).

---

## 11. Decisions — resolved

The audit closed with eight decisions. Each is resolved below with a default; each is revisited at the phase gate noted. Where a default is chosen rather than forced by evidence, that is said.

1. **Cameras / coverage → `R2`, all six ring cameras.** nuScenes' ring is R2-like; R2 is `comprehensive.md`'s default. Cost is ~6× GPU time, which on 2,424 images with Tiny-tier models is acceptable. A CAM_FRONT-only run remains legal but must record `coverage_config: R1` and is a different experiment. *Locked at Phase 2.*
2. **Accumulation window → preserve duration, ~0.5 s ≈ 10 sweeps** at this substrate's measured 20 Hz. Preserving the spec's 5-sweep *count* would halve the time window and change what "accumulated" means. Record both count and duration, plus `n_sweeps_actual`. *Locked at Phase 4.*
3. **Scene budget and partition → yes, disjoint, stratified by location and illumination.** From 10 scenes: **priors** = `0061`, `0103`, `0553`, `1077`; **tuning** = `0655`, `1094`; **run** = `0757`, `0796`, `0916`, `1100`. Each subset spans both locations and contains at least one night scene. The partition and its rationale are recorded in the manifest. `GAP_ANALYSIS.md` §3 names the un-partitioned version as a known pitfall ("tuning prompts on S0 and evaluating on S0"), and rev 1 reproduced it. Cheap now, structural later. *Locked at Phase 3.*
4. **nuScenes metrics → forbidden as a quality claim.** Permitted only on the `run` subset, labelled `diagnostic_only` in the file and in every sentence about it, reported with the leakage (priors derived from the same GT population), the model tier, and the pilot's purpose in the **same sentence**. A bare mAP from this pilot would look structurally comparable to VESPA's published nuScenes detection number (`GAP_ANALYSIS.md` §6 records ~46.5 % multiclass detection and ~52.95 % AP object discovery, and explicitly warns that figures circulating in internal drafts do not match the source — re-derive from the camera-ready PDF before quoting either) while being produced by a Tiny-tier model from priors derived from the same GT. *Locked at Phase 10.*
5. **Tracking regime → predict-then-match at 2 Hz.** Stepping to 12 Hz camera cadence needs interpolated LiDAR anchors, which introduces a second synthetic quantity into the stage being tested. Record effective Δt; make the IoU gate config with `derived for 2 Hz` provenance. *Locked at Phase 9.*
6. **Waived spec components → confirmed** as tabulated in §4: attribute pre-fill, group boxes, forward-backward smoothing waived; **IoA-NMS and yaw-consistency enforcement restored** (both free geometry). *Locked at Phase 2 (schema shape) and Phase 6/9 (implementation).*
7. **`spatial_ok` drivable term → off by default, flagged.** The map expansion is on disk so it *is* implementable, but `comprehensive.md` §1.4 excludes HD maps from v1.0 — a pilot component with no production counterpart teaches nothing and risks reading as a capability the real pipeline has. If enabled it must be marked `no_production_counterpart: true`. Silence is the only unacceptable option. *Locked at Phase 10.*
8. **Metadata / blob layout → one root**, `/home/mt/Zami/nuscenes`, metadata at `v1.0-mini/`. `dataroot` and `meta_root` stay separate config fields for the split-extraction case even though they are equal here. *Locked at Phase 1.*

---

## 12. The ten phases

Sequential. Each has an entry condition, a deliverable set, an exit gate, and the audit findings it closes. **Do not start phase N+1 until phase N's exit gate passes** — the failure this ordering prevents is building against an *assumed* upstream output shape.

### Phase 1 — Substrate truth and the path contract
**Entry:** none. **No GPU. No checkpoints. No pipeline code.**
**Do:** write `configs/paths.yaml` and `pipeline/common/paths.py` (`Paths`, `validate_paths()`, metadata fingerprint). Write `scripts/probe_substrate.py` as a throwaway-quality reconnaissance script and run it: confirm the §0 table on your own machine, dump the per-camera Δt distribution, the sweep cadence, the sweeps-per-keyframe ratio, the `LIDAR_TOP` extrinsic yaw, the category list, and per-scene annotation counts. Set up the repo skeleton, `.gitignore` for `data/`, and work/out roots outside the repo.
**Exit gate:** every number in §0 reproduced from your disk; `validate_paths()` rejects a deliberately wrong dataroot; decision 8 locked.
**Closes:** P0-9, M-8, part of P1-7's measurement need.

### Phase 2 — `pipeline/common/` — the whole representation contract
**Entry:** Phase 1 gate.
**Do:** `conventions.py` (§1.1–§1.6, including `project_lidar_to_image()` as a *specified and tested* function even before Stage 5 calls it), `schemas.py` (I-1…I-5 dataclasses, `validate()`, `write_records()` raising boundary, `frame` / `time_base` / unit-suffixed fields), `eval_region.py` (`in_region`, `rho`, derived from `coverage_config`), `manifest.py` (seed threading, `run_manifest.json`, atomic write, `_SUCCESS`, upstream-fingerprint refusal).
**Tests:** eval-region area vs closed-form geometry; contract round-trip (write → read → validate); positive-rejection provenance tests; µs→ns conversion against a real record pair; `LIDAR_TOP` non-identity; yaw→quat→yaw round-trip; unit/frame assertions on every schema.
**Exit gate:** all `common/` tests green; decisions 1 and 6 locked in config; **no stage code written yet**.
**Closes:** P0-1, P0-2 (specification), P0-4, P0-7, P1-7, P1-8 (mechanism), X-7, X-8, X-9.
**Note:** this is the largest and least glamorous phase. Nine of nine P0 findings are fixable here and nowhere else. Time spent here is not overhead.

### Phase 3 — Stage 0 probe and the scene partition
**Entry:** Phase 2 gate.
**Do:** implement the six predicates of §5.1 as named, individually reportable checks; emit `usable_scenes.json` (fingerprint-bound) + `probe_report.json` + the stratified partition of decision 3.
**Tests:** three-bucket fixture; truncated `.pcd.bin`; dangling token; missing sweeps; fingerprint mismatch rejection.
**Exit gate:** run against the real dataroot; the real usable-scene count is **known and printed**, not assumed; partition written and recorded.
**Closes:** P0-6, P1-5, M-7, M-15.

### Phase 4 — Stage 1 ingestion
**Entry:** Phase 3 gate.
**Do:** keyframe pack with measured per-camera Δt and per-camera thresholds; LiDAR → ego once; **both** clouds with `n_sweeps_actual` and `window_ns`; sector RANSAC (seeded) fitted on accumulated, applied to single-sweep; per-filter, per-sector point-count diagnostics.
**Tests:** sloped multi-sector ground fixture; real-record µs→ns; static-structure sharpness; scene-start `n_sweeps_actual`; determinism (same scene twice).
**Exit gate:** diagnostics show plausible retention per filter; determinism test byte-identical; decision 2 locked.
**Closes:** P0-3 (production side), P1-13, P1-8 (in practice), the I-3 gaps.

### Phase 5 — Model registry and VRAM, split into paper then measurement
**Entry:** Phase 4 gate.
Rev 1 collapsed these into one step that asked for `max_memory_allocated()` measurements "before downloading any checkpoint" — impossible (X-3).
**5a — paper check, no downloads:** `model_interfaces.py` (role Protocols with the §7.1 extensions), `models_pilot.yaml` / `models_production.yaml`, `scripts/check_vram_paper.py`. Use the **corrected** figures (§13). Confirm the largest single role fits the ceiling on paper.
**5b — measured check, on the card:** download, run one forward pass per role at the **real** resolution and the **real** 23-phrase prompt, and measure with **`torch.cuda.max_memory_reserved()` and `torch.cuda.mem_get_info()` free-deltas** — not `max_memory_allocated()`, which excludes the CUDA context (~300 MB), reserved-but-unallocated allocator blocks, cuBLAS/cuDNN workspaces, and other processes. On a 4 GB card the gap between "allocated" and "occupied" *is* the margin being budgeted. Measure the display-server allocation first. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` given the variable-length text tensors. Flip `verified:` per role. Assert `memory_allocated() ≈ 0` between role teardowns.
**Exit gate:** measured peaks recorded in the manifest; hard-stop behaviour defined and exercised.
**Closes:** P0-8, P1-2, P1-3, X-2, X-3.

### Phase 6 — Stages 3 and 4 (proposals + masks)
**Entry:** Phase 5 gate.
**Do:** `taxonomy_pilot_nuscenes.yaml` mapping table; Grounding DINO adapter owning its transforms and phrase-span mapping; per-class threshold dict with a default, tuned on the **`tuning` subset only**; dedup; MobileSAM adapter with the optional temporal-state signature; IoA-NMS; masks asserted at 1600×900.
**Tests:** prompt strings are phrases; phrase→class span mapping; original-resolution outputs; one mask per box in order; coordinate round-trip within 1 px.
**Exit gate:** a handful of real frames eyeballed once; resolution and prompt configuration recorded in every output record; no chunking (or chunking + re-tuned threshold, recorded).
**Closes:** P0-4 (in practice), M-2, X-5, the Stage 3 silent failures.

### Phase 7 — Stage 5 lift
**Entry:** Phase 6 gate.
**Do:** the four-hop chain on single-sweep ground-filtered ego-frame points; frusta union across the six cameras; contest rule; all four guards.
**Tests:** the two decisive ones — **GT-box reprojection** and **paint-inside-GT rate** — plus behind-camera cull, near-zero depth, out-of-bounds, overlap determinism, lidar→pixel→lidar round-trip.
**Exit gate:** paint-inside-GT rate is high and *stated as a number*. This is the single strongest correctness signal available in the whole pilot, and it uses data already on disk.
**Closes:** P0-2 (in practice), P0-5 neighbourhood, the Stage 5 gaps.

### Phase 8 — Stages 6 and 8 (clustering, box fit, priors, inflation)
**Entry:** Phase 7 gate.
**Do:** derive `priors_pilot_v0.json` from the **`priors` subset**, under E and the 40 m cap, `source: "nuscenes_gt_pilot"`; ε from the §7.2 formula; per-instance DBSCAN with deterministic tie-break; L-shape fit with `[w,l,h]` order and the near-square policy; inflation with explicit trigger/blend/axes/clamp and `inflated` + `inflation_fraction`.
**Tests:** non-square rectangle at 30° yaw against `conventions.py`; two same-class instances → two boxes; noise rejection; anchor direction; wrong-yaw documentation case; release-builder priors-source guard.
**Exit gate:** box dimensions on a sample of real clusters are sane against nuScenes GT for the same objects — as a **sanity check**, explicitly not a metric.
**Closes:** P0-5, P1-11, P1-12, M-12, X-6.

### Phase 9 — Stages 7 and 2 (the two DINOv2 consumers)
**Entry:** Phase 8 gate. Both stages consume DINOv2; Stage 2 is a branch (§5.3), so it is scheduled here rather than in numeric order — which is also the honest resolution of X-4.
**Do:** Stage 7 with predict-then-match, Hungarian matching, stated gates, birth/death, minimum crop size, ICP with a declared frame, fully-specified Kalman fallback, yaw-consistency enforcement. Stage 2 with image-level sampling, GLOSH scores on raw embeddings, UMAP for visualisation only.
**Tests:** fabricated similarity matrices; min-point fallback trigger; frame/time consistency on association inputs; ICP relative-vs-absolute; **the ≥3-frame stable-track integration test**; Stage 2 flagging rule on precomputed labels; subset/no-invented-IDs.
**Exit gate:** the ≥3-frame track test passes on one real scene — otherwise Stage 7 has no real input and its logic is only ever tested against fabricated data. Decision 5 locked.
**Closes:** P1-1, M-3 (partial), M-11, the Stage 2 caveats, X-4.

### Phase 10 — Stage 9, the end-to-end run, and claim hygiene
**Entry:** Phase 9 gate.
**Do:** Stage 9 gating on **pre-inflation** dimensions and single-sweep return counts; `allow_human_provenance: false`; end-to-end run over the **`run` subset** with eyeballed output at every stage boundary; complete `run_manifest.json`; failure isolation exercised by deliberately corrupting one scene; the indigenous-prompt probe run in its separate root; the determinism re-run.
**Then, and only then:** revisit each pilot-tier shortcut — MobileSAM's missing propagation, the Grounding DINO Tiny thresholds, the ε values, the 2 Hz association regime — and decide per item whether it is a declared pilot-only limitation or something worth improving at pilot scale.
**Exit gate:** the run is reproducible from the manifest by a second person; the claim-hygiene banner (§13) is present in the README, in `run_manifest.json`, and in the header of every exported figure. Decisions 4 and 7 locked.
**Closes:** P0-7 (tests), P1-9, P1-10, P1-15, M-13, M-14, the reproducibility audit.

---

## 13. Corrected VRAM figures and the claim-hygiene banner

### 13.1 The two corrected numbers

Rev 1's §5 table contained two errors of exactly the class `comprehensive.md` §7.1.1 exists to prevent, inside the plan that cites §7.1.1:

| Model | Rev 1 said | **Correct** [VERIFY before quoting] |
|---|---|---|
| Grounding DINO Tiny | "~172 MB weight file" | **~172 M parameters**; FP32 checkpoint ≈ **690 MB** — a params→MB transcription error, and a 4× understatement of the *largest model in the budget*, the one rev 1 itself named as the binding constraint |
| SAM ViT-B | "~375 M params, ~360 MB weights" | **~91 M parameters**, ~**375 MB** checkpoint — params and MB swapped. The "9× vs MobileSAM" conclusion survives (~91/10); its stated justification did not |
| DINOv2 ViT-S/14 | ~21 M, ~85 MB | correct |
| MobileSAM | ~10 M, ~40 MB | correct |

Planning-only inference-VRAM estimates, to be **replaced by Phase 5b measurements**: DINOv2 ViT-S/14 ~500–800 MB; Grounding DINO Tiny ~1.2–1.8 GB *(low confidence — the corrected checkpoint size makes the upper end likelier)*; MobileSAM ~400–700 MB. **Weight-file size is not inference VRAM**, and `max_memory_allocated()` is not occupancy (§Phase 5b). If Grounding DINO Tiny measures above the ceiling, the first mitigation is a **letterboxed** resolution reduction with the new resolution recorded in every Stage 3 record — never a square resize (§1.5), never silent prompt chunking (§5.4).

### 13.2 The banner

This text goes in the pilot README, in `run_manifest.json`, and in the header of **any figure exported from a pilot run** — not only in a planning document a future reader may never see:

> **This demonstrates pipeline plumbing only. Label quality is not evidence of anything; models are deliberately under-tier; the substrate is nuScenes v1.0-mini, not Dhaka.**

The reason this is not ceremonial: `comprehensive.md` §11.6 pivot B is literally "the annotation-pipeline paper, validated on nuScenes," and `GAP_ANALYSIS.md` §6 positions this pipeline against VESPA's published nuScenes numbers (and against SAM 3 as the likely production engine, whose licence terms for dataset production are still on that document's verification backlog). Any number this pilot produces sits directly on that path while carrying leakage the real study would not have.

**What a completed pilot supports, and what it does not:**

| Reading | Status |
|---|---|
| "The §7.3 pipeline shape is implementable end to end" | **Supported** (once Phases 2–10 pass) |
| "Contracts I-1…I-4 carry data between stages" | **Supported** |
| "It runs on 4 GB VRAM" | **Supported only with measured numbers, named resolution, and named prompt count** |
| "Stage 0 identifies usable scenes"; "priors/ε derivation works"; "tracking association works" | **Plumbing evidence only** (tracking further weakened by the 2 Hz regime) |
| "The OOD stage discovers long-tail content" | **Not supported** — the taxonomy is known and the method is clustering of a projection |
| "Open-vocabulary prompting separates indigenous classes" | **Not supported** — requires real Dhaka data + production model (S1) |
| Any per-class quality or recall figure | **Requires production model** |
| "Annotation quality is adequate" | **Requires real S0** |
| "mAP of X on nuScenes" | **Not supported as a pipeline-quality claim** — GT-derived priors + tuning leakage |
| "Model A beats model B" | **Not supported** — no controlled comparison exists in this pilot |
| "Production readiness" / "Dhaka-specific performance" | **Requires real Dhaka data** |

---

## 14. What survived the audit unchanged

Listed so it is not lost in a rewrite this large. These were right in rev 1 and are kept verbatim in intent:

- **Contracts-first build order** — `pipeline/common/` before any stage, everything importing from it. It is the correct dependency structure and it is also where every P0 finding turned out to be fixable.
- **Stage 0 as a pilot-only stage**, and specifically **excluding partially-present scenes loudly rather than skipping frames quietly**.
- **Role registry over hardcoded checkpoints** — the right abstraction; extended in §7.1, not replaced.
- **Carrying the F1 provenance invariant from the first line of code** rather than retrofitting it, with the stated rationale.
- **`eval_region.py` as the single implementation of E and ρ**, with ρ as count-over-area, and the observation that a locally reimplemented count is where an R1/R2 decision would silently distort a metric.
- **Naming the MobileSAM propagation loss a capability gap**, not a quality gap.
- **Refusing to call the indigenous probe S1**, and refusing to call the OOD stage discovery.
- **"Weight-file size ≠ inference VRAM"** and the `verified: true/false` flag mirroring §7.1.1.
- **Downgrading I-2 honestly** rather than presenting nuScenes poses as PPK.
- **The test philosophy** — name the silent failure first, then write the test for it — and the unit/integration split gated behind an env var.
- **Explicitly deferring `comprehensive.md` §3–§5** as having no pilot substitute, rather than inventing one.
