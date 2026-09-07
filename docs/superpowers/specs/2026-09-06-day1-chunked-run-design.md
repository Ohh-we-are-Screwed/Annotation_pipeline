# Day-1 chunked annotation run — design (2026-09-06)

Approved in chat 2026-09-06 by the operator, including the CVAT purge and the
naming below.

## Goal

Run the annotation chain over the new 2026-09-04 Dhaka capture
(`Dataset/A_nusc/chunk_0000` … `chunk_0006`, 4,536 keyframes, 6 cameras +
fused `LIDAR_TOP`, raw, no annotations) **one chunk at a time**, and for each
chunk produce:

1. a nuScenes-format release under `export/day1_chunk_NNNN/` with every blob
   **hard-copied** (no symlinks), and
2. a CVAT 2D project `day1_chunk_NNNN (2D)` and 3D project
   `day1_chunk_NNNN (3D)` holding that chunk's machine pre-annotations,

after purging everything currently on the CVAT server. **Every step runs
except the VLM check (3c)** — the operator's one exclusion (2026-09-06). That
means the default chain `0 1 3 4 5 6 7 8 eval viz cvat` plus `cvat3d`, plus the
opt-in **road** arm and its `cvatroad` publish and lidarseg export.

Two things do NOT run, for substrate reasons rather than choice:

- **Stage 3b** — an opt-in arm that walks the 12 Hz sweep frames between
  keyframes. This export has exactly one image per camera per keyframe
  (685 frames / 685 samples in every chunk), so there is nothing to walk.
- **The GT halves of `eval`** — the answer-key export and `eval_2d`/`eval_3d`/
  `paint_metrics` are `soft` steps that no-op without `sample_annotation`.
  The GT-independent half of `eval` — the COCO export the CVAT publish
  uploads — runs and is `fatal`, as it should be.

Road runs in **degraded** mode: its ZED refinement keys on rings 10/11 (the
pilot_1632 convention) and this export carries ZED as ring 100+k, so the
LiDAR+camera path runs and the missing refinement is recorded as a known gap.
This is exactly how it ran on nuScenes (`_SUCCESS.degraded`), which is the
precedent that says it is safe.

## Substrate facts the design rests on

- Each chunk is its own nuScenes-like dataroot: `samples/`, `v1.0-dhaka/`,
  `export_meta.json`, one scene named after the chunk. **No `sweeps/`.**
- Cameras: `CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_LEFT, CAM_RIGHT,
  CAM_BACK` — the Dhaka rig minus the two rear corners. All 1280×720.
- `LIDAR_TOP` is already fused (Mid-360 rings 0-3 + ZED as ring 100+k); there
  is no separate ZED channel for Stage 1 to merge.
- All tables `sample_annotation/instance/category/attribute/visibility/map`
  are `[]`.

## What the calibration run found (2026-09-06, chunk_0006)

Stage 0 refused the chunk outright — `zero usable scenes` — and, evaluated
predicate by predicate, the export breaks **five** pilot-era contracts, not
the two the initial design anticipated:

| predicate / check | why it failed | resolution |
|---|---|---|
| `sweeps_cover_window` | 1–2 LiDAR records per 0.5 s window, contract expects 0.8×5; no sweeps exported | accumulation window is now a **profile property**: `dhaka6` = `w_acc_count 1, w_acc_duration_ns 0` (anchor alone); probe + Stage 1 read it |
| `channels_complete` | 7 of 101 keyframes lack a camera; ~233 of 4,536 dataset-wide (`CAM_LEFT` worst, 73 in chunk_0003); all-or-nothing per scene, and Stage 1 would `KeyError` | **data-side fixup** into `v1.0-dhaka-fixed` (pilot precedent), `scripts/fixup_a_nusc.py`, run once per chunk by the driver |
| `files_resolve` | `samples/CAM_FRONT/000100.jpg` referenced, never written (1 in the dataset) | the fixup treats a missing blob as a missing channel (`blob_exists`) |
| `files_parse` | 93 of 94 clouds outside the pilot's `[10k, 300k]` point band (fused clouds: 37,920–490,859); 925 JPEGs under the 20 kB floor (min 15,027) | parse bands are a **profile property**: `dhaka6` = points `[10k, 1M]`, JPEG `[10 kB, 4 MB]`, measured over all 4,536 / 26,799 files |
| `SubstrateManifest.validate` | floored `w_acc_duration_ns` at 1 | floor is 0; the accumulated-cloud rule already carried `(1 sweep, window 0)` |
| Stage 5 `LiftContractError` | every sensor of a sample shares the LiDAR's one `ego_pose`, so `\|ego(t_cam) − ego(t_lidar)\| == 0` everywhere and the projection chain would silently omit ego motion between capture times (§1.3) | the fixup **interpolates a per-camera ego pose** from the LiDAR trajectory at each camera's timestamp (translation linear, rotation slerp, deterministic token, idempotent) — the pilot's `fixup_export.py` rebuilt per-sensor `ego_pose` rows the same way. NOT a profile flag: silencing the guard would bake in up to ~30 cm of error at 10 m/s × 34 ms |

| Stage 6 `UpstreamRefusal` | `out_root/priors/priors_pilot_v0.json` absent — Stage 6 takes `eps_bev` and Stage 8 takes `dims.mu` from it, "no default for either (X-6)"; both refuse one not bound to the current fingerprint. Dhaka has no GT to derive it; the pilot's file was hand-**authored** (handover §6) and went with the wipe | `scripts/author_priors_dhaka.py` — the §6 recipe as code: 10 nuScenes classes transferred from the surviving nuScenes-derived file and stamped `…TRANSFERRED_not_measured_on_dhaka`; `a rickshaw` 2.70×1.15×1.75 and `an auto rickshaw` 2.65×1.30×1.75 from literature with an ASSUMED σ (10 %); `gt_derived: false`, `source_note`, `REBOUND` history; bound per chunk; re-read through the real `load_priors`. The driver runs it after the fixup. Tests: `tests/test_author_priors_dhaka.py` |

| CVAT 2D/road tasks showed the **wrong images** (operator, 01:20) | tasks are created FROM THE SHARE — `ResourceType.SHARE` with dataroot-relative paths — and the share is a docker bind of `/home/mt/Zami/nuscenes` that still held pilot_1632's frames staged 2026-08-30 under the same names (`samples/CAM_BACK/000008.jpg`: md5 `3c29…` vs the chunk's `2789…`). Every day-1 chunk repeats those names, so a flat share can hold only one | the driver stages each chunk into the share under `day1_chunk_NNNN/` as **hard links** (same filesystem; no copy, no symlink) and exports `CVAT_SHARE_PREFIX`; `cvat_setup.py --share-prefix` (env-defaulted) prefixes the frame list AND the COCO it imports (`instances.share.json`), since CVAT binds annotations to frames by name. 3D tasks were never affected (own `task.zip`). Tests: `tests/test_cvat_share_prefix.py` |

| **Arm B not run** (operator, 01:55: "did you not run the custom rickshaw and CNG class?") | the chain was `0 1 3 …` — Stage 3 alone is arm A (YOLO11x, nuScenes phrases). The rickshaw/CNG detector is the opt-in arm B `3f` (`local_yolox_build/artifacts/yolo11x-rsud20k-armb.pt`, `armb-r1280-4`) + the merge `3m` (C28/C34). Carried over from `run_nuscenes_full.sh`, which excluded arm B *because* Boston/Singapore have no rickshaws | driver default chain is now `0 1 3 3f 3m 4 5 6 7 8 road eval viz cvat cvat3d cvatroad`; Stage 4 reads the merged tree, `export_taxonomy` answers the 12-phrase Dhaka taxonomy, CVAT projects get rickshaw/auto-rickshaw labels |
| **Camera extrinsics in body convention** | `calibrated_sensor.rotation` for cameras ≈ identity (front) / 180° yaw (back): vehicle-body axes, not nuScenes' optical (z forward, x right, y down). Measured: **0.0 %** of LiDAR points inside any image → Stage 5 painted 672 of 30 M points, 4,640/4,730 instances empty, 65 boxes, road found 5–11 plane candidates per keyframe. The pilot's fixup did this same conversion ("camera rotations → optical convention") | fixup transform 4: `q_optical = q_body ⊗ (0.5, −0.5, 0.5, −0.5)` for every sensor with a 3×3 intrinsic, rows stamped `frame_convention: optical`. After: 6.6–10.6 % inside per camera. The pilot's −5.95° pitch correction was rig-specific and is NOT applied — unmeasured on this rig; recorded as a known gap to check against Stage 5 paint rates |

| **Stage 6 exhausted RAM** (02:44: 47 min at 100 % CPU, 53 GB RSS, 14 GB swap, nothing written; killed) | the fused LIDAR_TOP carries ZED depth as rings 100/101 at **8.8×** the Mid-360's density (350,595 vs 39,936 points/frame). Once projection worked, one parked car in front of a ZED painted ~38k points per keyframe (93 instances > 20k, one per frame, all "a car"), and sklearn DBSCAN's neighbour graph on such an instance is quadratic | stereo thinning is a **profile property**: `dhaka6` = `stereo_rings (100, 101)`, `stereo_stride 8`; Stage 1's `thin_stereo` keeps every 8th stereo point (file order, deterministic) at both cloud-read sites and records rings/stride/n_removed in `single_sweep_sources.stereo_thinning`. 390,531 → 83,761 points/frame; largest instance ~5k. Legacy profiles: `()` / 1. Tests: `tests/test_stage1_thin_stereo.py` |

| **CAM_LEFT / CAM_RIGHT swapped** (operator, 09:10, from the scrambled 3D task of chunk_0000) | measured by rearward image drift on moving keyframes: the `CAM_LEFT` stream drifts **+68.6 px** (faces right), `CAM_RIGHT` **−80.6 px** (faces left) — opposite to both calibrations; the "CAM_RIGHT" stream also carries 6.5× the kerb-side instances in left-hand traffic. Every projection into those two cameras landed on the wrong side of the car. Front-left/front-right: weak signal says correctly labelled; they carry ~100 instances per chunk (tiny re-encoded images) — left alone, noted | fixup transform 5: `--swap-channels CAM_LEFT CAM_RIGHT` exchanges `calibrated_sensor_token` within each sample (filename/timestamp/ego_pose stay with the row); recorded in `fixup_meta.json`. Stages 3/4 key on channel, so **every chunk reran from Stage 0** (the 09:12 launch); chunks 0000–0002's first outputs and CVAT projects were discarded |

| **Cuboids "way out of proportion"** (operator, 3D task of chunk_0000 after the swap) | measured per class on chunk_0000: heights 55–65 % of true (pedestrian 0.84 m, car 1.09, bus 1.99), small classes 1.3–1.6× too long, and **84–93 % of boxes elongated along the viewing ray**. Root cause in Stage 1: the RANSAC ground candidates are taken from z ∈ [−1.5, +1.5] (ISO 8855, z=0 at ground) but this rig's ego origin is the LiDAR ~2.3 m up — road measured at z = −2.0…−2.75 (LiDAR) / −2.5…−3.0 (ZED). The band never held the ground; planes were fit at −0.3…−0.6 through the scene, the ±0.3 m removal slab cut objects at mid-height, and the surviving road points were painted into every mask's frustum | ground band is a **profile property** `ground_z_band_m`: `dhaka6` = (−3.5, −1.0); legacy (−1.5, 1.5). `IngestConfig.ransac_candidate_z_band_m` reads it. Verified on chunk_0006: plane intercept −2.27 m median (was −0.31…−0.62), below-ground points 11.6k/kf (was 49k). Chunks must rerun from Stage 1 |
| **Rickshaw wheels detected as bicycles** (operator, 2D task, with picture) | arm A emits `a bicycle`/`a motorcycle` for the front wheel INSIDE the arm B rickshaw box; C28/C34 arbitrate by IoU, which stays ~0.15 for a contained wheel | **C36** in `merge.py`: an arm A part-class box whose own area is ≥ 60 % covered by a *surviving* arm B rickshaw / auto-rickshaw box is absorbed (containment, not IoU); runs after C28/C34; ledger `merge.suppressed_parts`; CLI `--suppress-part "a bicycle:0.6"` / `--no-suppress-parts`; default on. Tests: `tests/test_stage3_merge_parts.py` |

Operator rules recorded 2026-09-06 afternoon: **never delete CVAT projects/tasks or
exports unless told to** — reruns publish beside earlier tasks (run tags) and write
to `export/day1_chunk_NNNN$EXPORT_SUFFIX/`. Next run: chunks 0001 and 0004 with the
VLM check (`VLM_CHECK=1 VLM_USE_CHECKED=1`), which needs llama.cpp rebuilt (three
parallel builds segfaulted in unrelated files — machine, not code; serial build in
progress).

Disk: the driver prunes `stage1_ingestion/clouds` after a successful export
(`PRUNE_CLOUDS=0` keeps it) — ~1.2 GB per 93 keyframes, two copies (with
`w_acc_count=1` the accumulated cloud IS the single sweep), unread once the
box release, lidarseg layer and CVAT 3D archive are on disk.

Calibration outcome (chunk_0006, 01:15–01:17, BEFORE arm B and the optical fix): chain rc=0,
`stage8_inflate=clean`, `stage_road=degraded`; Stage 9 gate rc=0; box release
rc=0; lidarseg rc=0; export 818 regular files / 0 symlinks / 915 MB; CVAT
projects `day1_chunk_0006 (2D)` #269, `(3D)` #267, `(road)` #270.

The trajectory itself is real: chunks 0000–0005 cover 488–1,051 m each at up
to 10.5 m/s with all-distinct, `identity: False` poses. `chunk_0006` — the
calibration chunk — is the parked 40 s tail (0.3 m in 40 s), which is why its
interpolated deltas are ~1 mm; still nonzero, so the gate passes honestly.

Still degraded by design: the nuScenes-only fixed `PARTITION` reports
"unsatisfiable" → `_SUCCESS.degraded`, `rc=1`, exactly as the pilot_1632
runs (scene `chunk_0000`) always did. The wrapper carries it forward with
`--accept-degraded-upstream`.

## Changes

### 1. `pipeline/common/schemas.py` — `dhaka6` substrate profile, and three contracts moved into profiles

Add a third `SUBSTRATE_PROFILES` entry, `"dhaka6"`: the six channels above,
`image_width_px=1280`, `image_height_px=720`, **plus** the accumulation window
(`w_acc_count`, `w_acc_duration_ns`) and the Stage 0 parse bands
(`pcd_point_band`, `jpeg_byte_band`). Every profile now carries all four;
`dhaka` and `nuscenes` carry the pilot's literal values so nothing archived
changes meaning. Exported as `W_ACC_COUNT`, `W_ACC_DURATION_NS`,
`PCD_POINT_BAND`, `JPEG_BYTE_BAND`; `probe.py` and `IngestConfig` read them
instead of their former literals. Selected per run with
`DHAKASCENES_SUBSTRATE=dhaka6`.

### 1b. `scripts/fixup_a_nusc.py` — drop keyframes missing a required channel

Pure function `drop_incomplete_samples(tables, required_channels, blob_exists=None)`
+ CLI. Reads `<dataroot>/<version>/`, never writes it; creates
`<dataroot>/<version>-fixed/` (refuses if present; refuses if every sample
would go). Keeps table order, re-links `sample.prev/next` per scene, drops the
samples' `sample_data` rows, updates `scene.nbr_samples/first/last`. Required
channels default to the active profile's `REQUIRED_CHANNELS`. Blobs untouched.
Tests: `tests/test_fixup_a_nusc.py`.

### 2. `scripts/cvat_setup_3d.py` — env-overridable 3D project name

`OURS_PROJECT` becomes
`os.environ.get("CVAT_PIPELINE_3D_PROJECT", "OUR PIPELINE — machine pre-annotations (3D)")`,
mirroring how the 2D publish already reads `CVAT_PIPELINE_PROJECT`. Default
unchanged, so every existing invocation behaves identically. `GT_PROJECT` is
not touched (no answer key exists here and C13 keeps it out of any wipe).

### 3. `scripts/run_day1_chunks.sh` — the driver

For each chunk (order: `0006` first as a live calibration, then `0000`…`0005`;
overridable by argument):

1. `mkdir -p <chunk>/sweeps` — empty; satisfies `REQUIRED_BLOB_DIRS`. The
   only write into the dataroot.
2. Generate `configs/paths_day1_chunk_NNNN.yaml`: `dataroot`/`meta_root` =
   the chunk, `version: v1.0-dhaka`, `work_root=/home/mt/dhakascenes/work_day1/chunk_NNNN`,
   `out_root=/home/mt/dhakascenes/out_day1/chunk_NNNN`,
   `probe_out_root=/home/mt/dhakascenes/probe_out_day1/chunk_NNNN`. Per-chunk
   roots mean no chunk's stage tree is overwritten by the next and any chunk
   can be resumed or republished alone.
3. `DHAKASCENES_PATHS_CONFIG=<that yaml> DHAKASCENES_SUBSTRATE=dhaka6
   CVAT_PIPELINE_PROJECT="day1_chunk_NNNN (2D)"
   CVAT_PIPELINE_3D_PROJECT="day1_chunk_NNNN (3D)"
   CVAT_ROAD_PROJECT="day1_chunk_NNNN (road)"
   scripts/run_stages.sh 0 1 3 4 5 6 7 8 road eval viz cvat cvat3d cvatroad`.
   Typed order is execution order: the GPU box chain stays contiguous, road
   (consumes Stage 1 alone) follows it, then the exports and the three
   publishes. `VLM_CHECK`/`VLM_USE_CHECKED` already default to 0. The
   wrapper's `_SUCCESS.degraded` on `stage_road` is expected (see above).
4. `python -m pipeline.stage9_qa.gate --paths <yaml>` (+`--accept-degraded-upstream`
   if Stage 8 is `degraded`, as `run_nuscenes_full.sh` does) → `prelabels.jsonl`.
5. `scripts/export_release.py --prelabels <work>/stage9_qa --dataroot <chunk>
   --version v1.0-dhaka --out export/day1_chunk_NNNN/boxes --blobs copy` —
   the full nuScenes tree, every image and point cloud a regular file.
6. `python -m scripts.export_road_lidarseg --paths <yaml>
   --out export/day1_chunk_NNNN/lidarseg` (+`--accept-degraded-upstream` when
   the road marker is degraded). **Without** `--link-blobs`: the exporter then
   writes tables + per-keyframe `lidarseg/*.bin` only and neither links nor
   copies blobs — the blobs are already real files under `boxes/`.
7. Log to `/home/mt/dhakascenes/work_day1/logs/chunk_NNNN.log`; print a
   per-chunk status line; a fatal core-chain failure stops the loop (the next
   chunk is not started on a broken profile/config). A road-arm failure is
   reported, not fatal — the box release is already on disk, as in
   `run_nuscenes_full.sh`.

Before the first chunk: `scripts/cvat_purge.py --yes` once (approved).

### Out of scope

Merging chunks into one dataroot; re-enabling 3b/3c; any change to
`run_stages.sh`.

## Testing

- Unit: `dhaka6` resolves through `DHAKASCENES_SUBSTRATE` with the six
  channels and the 1280×720 pin; `dhaka`/`nuscenes` unchanged.
- Unit: `CVAT_PIPELINE_3D_PROJECT` overrides `OURS_PROJECT`; unset keeps the
  legacy name byte-identical.
- Existing 257 tests stay green (`ano_pipe` env).
- Live: `chunk_0006` (101 samples) end to end — validator accepts the chunk,
  Stage 0 keeps the scene, the export tree contains regular files only
  (`find -type l` is empty), both CVAT projects exist with one task each.

## Estimates

Measured on this 4090 from `work_nuscenes` manifests, scaled by images:
~20–25 min per 750-keyframe chunk, ~4 min for `chunk_0006`, ~2.5 h total.
