# Handoff — Dhaka pilot through the annotation pipeline

**Date:** 2026-08-30 · **Repo:** `/home/mt/Zami/Annotation_pipeline` · **Branch:** `main`
**Nothing is committed.** 8 files are modified in the working tree. See §5.

---

## 1. Read this first

The pipeline was built for **nuScenes v1.0-mini** (6 cameras, 32-beam spinning lidar,
1600×900, 2 Hz keyframes, ground truth available). It is now pointed at a **raw Dhaka
capture** (8 cameras + Livox Mid-360 + 2 ZED stereo depth passes, 1280×720, 10 Hz
keyframes, **no ground truth**). Most of the work below is closing that gap.

**Only `pilot_1632` has ever been run.** `pilot_1601` has never been processed and
cannot be without work — it lacks `CAM_BACK_LEFT`/`CAM_BACK_RIGHT` (so it fails
`REQUIRED_CHANNELS` and Stage 0 hard-stops) and has no ZED pass at all.

### Current state

| | |
|---|---|
| Working dataroot | `/home/mt/dhakascenes/data/pilot_1632` |
| Metadata version | `v1.0-dhaka-fixed` (original `v1.0-dhaka` preserved beside it) |
| Source zips | `/home/mt/Zami/Annotation_pipeline/pilot_{1601,1632}.zip` — untouched |
| Work root | `/home/mt/dhakascenes/work` |
| Helper scripts | `/home/mt/dhakascenes/fix/` |
| CVAT | `http://localhost:8081` user `mt` — live tasks are from an **older** run |

**A run is in flight** at the time of writing. See §7 before starting anything.

---

## 2. The data

`pilot_1632`: nuScenes-like export, `v1.0-dhaka`, **2303 keyframes @ 10 Hz, 230 s**,
one scene `chunk_0000`. 11 channels: `LIDAR_TOP`, 8 cameras, `ZED_FRONT`, `ZED_BACK`.
No annotations (`sample_annotation.json`, `category.json`, `instance.json` etc. are `[]`).

Rig geometry (from `calibrated_sensor`, body frame X-fwd Y-left Z-up):

| channel | yaw | nominal pitch | notes |
|---|---|---|---|
| CAM_FRONT | 0° | 7° | intrinsic measured |
| CAM_FRONT_LEFT | 40.25° | 6.5° | intrinsic **nominal** (FOV-derived) |
| CAM_FRONT_RIGHT | −44.63° | 6.5° | intrinsic **nominal** |
| CAM_LEFT | 90° | 14° | intrinsic measured |
| CAM_RIGHT | −90° | 6.5° | intrinsic measured |
| CAM_BACK_LEFT | 139.75° | 6.5° | intrinsic **nominal** |
| CAM_BACK_RIGHT | −135.37° | 6.5° | intrinsic **nominal** |
| CAM_BACK | 180° | 7° | intrinsic measured |
| ZED_FRONT / ZED_BACK | 0° / 180° | 7° | stereo depth, ≤20 m, co-located with CAM_FRONT/BACK |

Lidar is genuinely 360° (~1300 returns per 30° sector, plus a dense forward lobe
of ~5500 in 0–30°). **There is no lidar deadzone.** The coverage problem was cameras.

---

## 3. What was wrong with the export (audit findings)

1. **Camera extrinsics were in body convention, not nuScenes optical.** Standard
   nuScenes projection put **zero** points on every image. Fix = fold in
   `Ro = [[0,-1,0],[0,0,-1],[1,0,0]]` (body→optical).
2. **The nominal pitch was overstated by ~5.9°.** Measured against the ZED depth
   pass on 8 frames spanning the session: CAM_FRONT −5.88°±0.12, CAM_BACK
   −6.03°±0.15, **body-frame** (an ego-frame correction was tested and is wrong —
   CAM_BACK disambiguates: −6.0 body gives 0.31 m error, +4.0 ego gives 0.94 m).
   ~92 px vertical error at fy≈953 if uncorrected.
3. **`map.json` was `[]`** → devkit crashes with `IndexError` at `self.map[0]`.
4. **`CAM_FRONT`/`CAM_BACK` had `width: 0, height: 0`** on all 2303 rows (4606 rows).
5. **`ego_pose` was identity for every keyframe**, and one pose was shared by all 11
   `sample_data` of a sample — so hops 1 and 2 of the projection chain cancelled and
   `ego_translation_delta_m` was exactly 0.
6. `sample_data.prev`/`next` empty everywhere; ZED rows used `num_pts` not `num_points`;
   no `sweeps/` dir; cameras carry a non-nuScenes `dhaka_offset_ns`.
7. **14 corrupt JPEGs** (11 missing EOI, 3 with valid EOI but corrupt scan data).
   OpenCV reads them; **PIL refuses** — which is what crashed Stage 3.

Timing is genuinely good: 10.00 Hz, zero dropped gaps, camera↔lidar skew 4–11 ms
median / 33 ms worst. Lidar `.pcd.bin` is exactly the nuScenes 5×float32 layout.

---

## 4. Data-side fixes (all in `v1.0-dhaka-fixed`, non-destructive)

Built by `/home/mt/dhakascenes/fix/fixup_export.py`. `v1.0-dhaka` is untouched.

| fix | count |
|---|---|
| camera rotations → optical convention **+ pitch correction** | 8 |
| `width`/`height` backfilled | 4,606 |
| `num_pts` → `num_points` | 4,606 |
| `prev`/`next` chains rebuilt | 25,332 |
| `ego_pose` rows (was 2,303 identity, shared) | 25,332 |
| `map.json` stub | devkit loads |

Later, by hand (see §6 of the shell history / this doc):

- **ZED extrinsics pitch-corrected** by the same −5.95°. *This was missed initially*
  because `fixup_export.py` skips sensors without a `camera_intrinsic`. Verified:
  ZED-vs-lidar ground agreement went from **−0.89/−1.29 m to +0.01/+0.04 m**.
  Backup: `calibrated_sensor.before_zedfix.json`.
- **One keyframe dropped** (`6df7f2ca55bfdd1c6872cc426b326111`) — it alone lacked
  `CAM_RIGHT`, which fails `channels_complete` once the ring is 8 cameras.
  **2303 → 2302 keyframes.** Its 10 dangling `sample_data.prev` pointers were cleared.
  Backups: `*.before_8cam.json`.
- **14 JPEGs re-encoded** (originals in `/home/mt/dhakascenes/fix/truncated_originals/`).
  All 18,423 images now pass a strict PIL decode.

### ego_pose — recovered by lidar odometry

`/home/mt/dhakascenes/fix/lidar_odometry.py` — point-to-plane ICP, scan-to-local-map,
trimmed correspondences (Dhaka traffic rejected as outliers). Output
`ego_pose_1632.json`, interpolated per `sample_data` timestamp by the fixup.

- 2302 poses, **zero ICP failures**, 1143.6 m path / 704.0 m net over 230 s, median 17.9 km/h.
- Sweep-to-sweep alignment over 0.5–1.0 s reaches **0.14–0.25 m against a sampling
  floor of 0.11–0.15 m** — i.e. at the sensor's own resolution limit. Identity poses
  sat 2–3× above it.
- **Known limitation:** open-loop, so it drifts. ~15–20 m of unexplained z climb over
  the route (beyond what the flyover explains). Fine for accumulation, projection and
  short-lived tracks (all local); **not a global map**. Fixable with a
  ground-plane/gravity constraint or IMU — not done.

---

## 5. Code changes (UNCOMMITTED — 8 files)

```
configs/paths.yaml                  dataroot/meta_root -> pilot_1632, version v1.0-dhaka-fixed
configs/taxonomy_pilot_dhaka.yaml   pre-existing edit, not mine (dhaka.cng_autorickshaw -> dhaka.cng)
pipeline/common/schemas.py          IMAGE_WIDTH/HEIGHT_PX 1600x900 -> 1280x720
                                    RING_CAMERAS 6 -> 8 (added CAM_LEFT, CAM_RIGHT)
pipeline/common/eval_region.py      _R_MAX_M 40.0 -> 30.0
pipeline/stage0_data_probe/probe.py W_ACC_COUNT 10 -> 5  (0.5 s at 10 Hz, not nuScenes' 20 Hz)
pipeline/stage1_ingestion/ingest.py w_acc_count 10 -> 5; range_cap_m 40.0 -> 30.0
                                    + STEREO_CHANNELS and ZED fusion (see below)
scripts/export_cvat_coco.py         added --taxonomy (was hardcoded to nuScenes)
scripts/run_stages.sh               3 fixes (see below)
```

### Three genuine latent bugs found and fixed

1. **`run_stages.sh` step 1 never passed `--accept-degraded-upstream`**, with a comment
   asserting Stage 1 doesn't take it — but `ingest.load_allowlist()` refuses on a
   degraded Stage 0 marker without it (`ingest.py:789`). Never hit because Stage 0
   never degraded on nuScenes mini. The wrapper announced it would pass the flag and
   then didn't.
2. **`export_cvat_coco.py` hardcoded `configs/taxonomy_pilot_nuscenes.yaml`.** The
   merged arm A + arm B tree carries arm B's *superset* vocabulary, so the export died
   on `KeyError: 'an auto rickshaw'` **after every GPU stage had run**. C28 wired the
   superset through stages 3f/3m/4 and stopped before the exporters. Added `--taxonomy`
   (default preserves old behaviour) + an `export_taxonomy()` helper in `run_stages.sh`
   that picks arm B's taxonomy when the merged tree is in play. Same for the 3D export.
3. **The answer-key twins would have been built from nothing.** `cvat`/`cvat3d`
   unconditionally export `sample_annotation.json` as a "nuScenes HUMAN answer key".
   Dhaka's is empty → empty green twins beside the pipeline output, which reads as
   *a human agreed with everything*. Added `has_ground_truth()`; both steps now skip
   the twins and say why, and 3D drops to `--which ours`.

### ZED stereo fusion (`ingest.py`)

```python
STEREO_CHANNELS: dict[str, int] = {"ZED_FRONT": 10, "ZED_BACK": 11}
IngestConfig.fuse_stereo: bool = True
```

- ZED points merged into the **single sweep only**. The accumulation stays lidar-only,
  so the ground-plane fit is unchanged.
- Provenance rides in the **ring column** (Mid-360 emits rings 0–3, so 10/11 are free).
  Keeps the 20-byte record — no schema change — and lets anyone split the cloud apart.
- Stage 1 writes `single_sweep_sources: {n_lidar, n_stereo, n_total, ring_tags}` per
  keyframe, because the ">=5 returns" gate now counts stereo points.
- Measured effect: **median per keyframe lidar 19,968 + ZED_FRONT 40,169 +
  ZED_BACK 41,314 = 101,535 points (4.9×). Stereo is 79.6% of the fused cloud.**

### Coupling you must not break

`eval_region._R_MAX_M` and `IngestConfig.range_cap_m` **must stay equal** (both 30.0).
If they diverge, Stage 1 prunes to one radius while Stages 5/6 score against another —
points silently absent from a region the metrics still count. Check with:

```python
from pipeline.common.eval_region import R2_DEFAULT
from pipeline.stage1_ingestion.ingest import IngestConfig
assert R2_DEFAULT.r_max_m == IngestConfig().range_cap_m
```

Also note **`RING_CAMERAS` defines "the full ring" for `coverage_config: R2`**. Going
6→8 makes R2 honest (it already claimed a full annulus at 86.3% coverage) but runs
before and after are **not comparable**. Same for the 40→30 m cap: E is the denominator
of every density metric.

---

## 6. The priors file — READ THIS

`/home/mt/dhakascenes/out/priors/priors_pilot_v0.json` is **authored, not derived**.

Stage 6 refuses priors not bound to the current dataroot fingerprint, and Dhaka has no
`sample_annotation` to derive them from, so `pipeline.stage6_cluster.priors` cannot run.
On the user's explicit decision, the file was hand-authored:

- 10 nuScenes classes **transferred unchanged**, each stamped
  `nuscenes_gt_pilot_TRANSFERRED_not_measured_on_dhaka` with the original fingerprint.
- `a rickshaw` 2.70×1.15×1.75 m, `an auto rickshaw` 2.65×1.30×1.75 m (Bajaj RE class),
  both stamped `literature_ASSUMED_no_measurement`, σ explicitly assumed.
- `gt_derived: false`, a `source_note` stating **box dimensions from this run are not
  evidence about Dhaka object sizes**, and `derived_from.REBOUND` recording every
  rebinding with reason and date.
- nuScenes original preserved at `priors_pilot_v0.nuscenes-backup.json`.

**The fingerprint must be rebound whenever the metadata changes.** It has been rebound
twice already. To rebind:

```python
from pipeline.common.paths import load_paths, metadata_fingerprint
fp = metadata_fingerprint(load_paths('configs/paths.yaml'))
# write into derived_from.metadata_fingerprint, append to derived_from.REBOUND.rebound_history
```

**To replace it properly:** label some Dhaka keyframes with 3D cuboids in CVAT, then run
`python -m pipeline.stage6_cluster.priors`. The 3D CVAT task is exactly what you'd label.

---

## 7. Run in flight

Log: `/home/mt/dhakascenes/work/logs/dhaka_run_20260830_175723.log`
Command:

```bash
REID_MODEL_ID=facebook/dinov2-small scripts/run_stages.sh 1 3 3f 3m 4 5 6 7 8 viz cvat cvat3d
```

`eval` is deliberately **excluded** — it scores against ground truth Dhaka doesn't have.
`REID_MODEL_ID` is overridden because Stage 7's default (`facebook/dinov3-vits16…`) is
gated and uncached, and would silently fall back to IoU-only tracking. DINOv2-small is
cached and ungated.

**Markers for stages 4–8 on disk are from an OLDER run** and will be overwritten. Do not
read them as describing the current configuration until this run finishes.

Monitor:
```bash
tail -f <log> | grep -E "^=== STAGE|^--- STAGE|^!!!|REFUSED|ABORTED|ALL_STEPS_DONE|Traceback"
```

### Results from the PREVIOUS (6-camera, lidar-only, 40 m) run — the baseline to beat

```
Stage 3   30,428 proposals  2.20/img  5,467 empty
Stage 3f   2,505 proposals (arm B is a 2-class detector by design: rickshaw + CNG)
Stage 3m  31,353 boxes out, 1,580 arm A suppressed  <- vocabulary authority working
Stage 4   31,353 masks, 31,222 kept, 0 empty        <- clean
Stage 5   512,362 points painted, frustum union 0.434, 16.41 pts/instance
Stage 6   13,680 boxes from 31,222 instances (10,972 got ZERO lidar points)
Stage 7   6,024 births / 6,010 deaths, 14 confirmed tracks  <- heavily fragmented
Stage 8   8,347 inflated, mean fraction 0.805
```

**Why the 3D boxes looked wrong** (this was the user's complaint — diagnosed, not a bug):

```
num_lidar_pts per box:  median 10    23% <5    49% <10    68% <20
inflation_fraction:     median 0.641     42.7% of boxes >80% prior
measured -> final size: w 0.42->0.64   l 1.22->1.88   h 0.69->1.13  m
yaw ambiguous:          56%
```

A median *measured* length of 1.22 m on things labelled "a car". Boxes were mostly
prior geometry wrapped around a 10-point fragment. **Not** a frame bug — boxes are in
ego frame, centres 1.6–40.4 m, no drift. The fix is the ZED fusion (§5).

**Numbers to compare against when this run lands:** Stage 6 `inflation_fraction`
(was 0.641 median) and `num_lidar_pts` (was median 10). If they haven't moved much,
the stereo isn't reaching the objects that matter — say so, don't dress it up.

---

## 8. Known limitations — carry these into any claim

1. **Six of eight camera pitches are ASSUMED, not measured.** Only CAM_FRONT and
   CAM_BACK have a ZED depth reference. Three methods were tried for the rest and all
   failed on this data:
   - Levinson edge alignment — optima on the grid boundary, only 1.1× score gain in
     dense foliage, and it contradicted ZED truth on CAM_BACK (−0.5° vs −6.03°).
   - Cross-camera photometric NCC via calibrated neighbours — peak NCC 0.008–0.027
     (no correlation); the side cameras are motion-blurred in every frame.
   - Same restricted to stationary frames — **the rig never drops below 8.7 km/h**.

   The applied value is the mean of the two measured cameras, justified by both storing
   7.00° and both needing the same correction to 0.15° (systematic, not per-camera slop).
   **Run a target-based lidar-camera calibration before measuring anything from those six.**

2. **Intrinsics for CAM_FRONT_LEFT/RIGHT and CAM_BACK_LEFT/RIGHT are FOV-derived
   nominals** — no distortion model, principal point assumed at centre. Unchanged from
   the source export. `K = [[906.9,0,640],[0,906.9,360]]` is the giveaway.

3. **Stage 0 is permanently DEGRADED** on `PARTITION` — a hardcoded 10-scene nuScenes
   priors/tuning/run split, each subset required to span Boston+Singapore with a night
   scene. One daytime Dhaka scene cannot satisfy it. Deliberately **not** faked; the
   degradation is recorded in every downstream manifest. Downstream stages read the
   `scenes` list (which contains `chunk_0000`), not the partition.

4. **Stage 1 is DEGRADED on 76.4% sector rejection.** This is legitimate — rejected fits
   look like tilt 44.1° at inlier ratio 0.878, i.e. a *confident* fit to a wall or bus
   flank filling the sector. The guard substitutes the global plane. Do **not** loosen
   the thresholds. Note the reported `tilt_deg` on a rejected sector is the *substituted*
   plane's; the original is under `rejected_fit`.

5. **Recall, not precision, is the weak point.** Only ~44% of 2D instances became 3D
   boxes in the baseline run. The 2D masks are the trustworthy layer.

6. **CVAT 2D images are served from a share volume** (`ResourceType.SHARE`), not
   uploaded. Dhaka images were staged into the `cvat_cvat_share` docker volume. Every
   CVAT container mounts it **read-only** and there is no passwordless sudo — use a
   helper container:
   ```bash
   docker run --rm -v cvat_cvat_share:/share -v /home/mt/dhakascenes/data/pilot_1632:/src:ro alpine sh -c '...'
   ```
   nuScenes data is still in the share (filenames don't collide: `n008-…jpg` vs `000000.jpg`).
   3D needs no share — it uploads a self-contained `task.zip` (~3.9 GB for this scene).

   **Every channel the 2D export references must be staged, or the publish fails on
   missing files.** This has bitten twice: first when the share held only nuScenes, then
   again when `RING_CAMERAS` went 6→8 and `CAM_LEFT`/`CAM_RIGHT` were absent. Currently
   staged: all 8 cameras, 2303 files each (2302 for `CAM_RIGHT`). If you change
   `RING_CAMERAS` again, stage the new channels before running `cvat`.

7. The user chose **one 2302-keyframe scene** over splitting into nuScenes-sized scenes,
   knowing it makes one very large CVAT task. That was an informed decision.

---

## 9. Useful commands

```bash
# environment: THIS interpreter, always
PY=/home/mt/miniconda3/envs/ano_pipe/bin/python

# substrate check (names its failing predicates)
$PY -m pipeline.stage0_data_probe.probe --paths configs/paths.yaml

# verify calibration end-to-end (devkit + the pipeline's own projection)
$PY /home/mt/dhakascenes/fix/verify.py            # fixed export
$PY /home/mt/dhakascenes/fix/verify.py v1.0-dhaka  # original: devkit crashes

# render 8-camera lidar overlays (ground=orange, above-ground=blue)
$PY /home/mt/dhakascenes/fix/render_check.py grid 800,1500

# re-derive ego motion
$PY /home/mt/dhakascenes/fix/lidar_odometry.py \
    --dataroot /home/mt/dhakascenes/data/pilot_1632 --out ego_pose_1632.json

# rebuild the fixed metadata from scratch
$PY /home/mt/dhakascenes/fix/fixup_export.py --dataroot /home/mt/dhakascenes/data/pilot_1632 \
    --poses ego_pose_1632.json --pitch pitch_final.json
```

Weights present locally: `yolo11x.pt`, `yolov8x-oiv7.pt`, `mobile_sam.pt` in
`/home/mt/dhakascenes/cache/checkpoints/`; `facebook/sam3`, `sam3.1`, `dinov2-small`
in the HF cache. **`dinov3` is NOT cached and is gated.**

---

## 10. Suggested next steps

1. **Let the in-flight run finish** and compare Stage 6 `inflation_fraction` /
   `num_lidar_pts` against the baseline in §7. That is the test of whether the ZED
   fusion actually fixed the box quality.
2. **Target-based lidar-camera calibration** for the six assumed cameras (§8.1). This is
   the largest remaining source of geometric error.
3. **Label ~40 keyframes of 3D cuboids** in the CVAT task, then run
   `pipeline.stage6_cluster.priors` to replace the authored priors with measured ones (§6).
4. **Constrain the odometry** (ground plane / gravity, or IMU) to kill the z drift (§4).
5. Decide whether to commit the 8 modified files. They are a mix of genuine bug fixes
   (portable, should be committed) and substrate-specific repins (`paths.yaml`,
   `IMAGE_*_PX`, `W_ACC_COUNT`, `_R_MAX_M`, `RING_CAMERAS`) that make the repo
   Dhaka-only. Consider splitting: bug fixes to `main`, repins behind config.
