# Handover — day-1 chunked annotation run (2026-09-05 → 2026-09-06)

Written 2026-09-06 ~14:45 (+06) for the next session. Everything below is
uncommitted in the working tree (last commit: `8bf4757 gitignore …`).
Detailed decision log: `docs/superpowers/specs/2026-09-06-day1-chunked-run-design.md`.

## 1. What is running RIGHT NOW

tmux session `pipe`, window `day1`:

```
export VLM_CHECK=1 VLM_USE_CHECKED=1 VLM_CHECK_MODE=per_box EXPORT_SUFFIX=_vlm
CHAIN_STEPS="3c 4 5 6 7 8 road eval viz cvat cvat3d cvatroad" scripts/run_day1_chunks.sh 0001
scripts/run_day1_chunks.sh 0004
```

- `chunk_0001`: Stage 3c (VLM, per-box) **finished ~14:50**: 21,795 boxes, 5,949 VLM
  calls, **912 relabeled, 700 errors** (12 % of calls — inspect
  `work_day1/chunk_0001/stage3_checked/` and `logs/llama_server_20260906T140139.log`
  before trusting the relabels). Stage 4 OK (478 s, 17:58). **Stage 5 then CRASHED**:
  `FileNotFoundError … stage1_ingestion/clouds/…single_sweep/….pcd.bin` — the driver
  had pruned 0001's cloud cache after the 13:14 export and the resumed chain began at
  3c, so no Stage 1 ran. The fixed gate refused the export (correct); the driver moved
  on to `chunk_0004` (full chain, started 17:58). **To finish 0001** — nothing lost,
  `stage3_checked` + `stage4_masks` stand:
  `export VLM_CHECK=1 VLM_USE_CHECKED=1 VLM_CHECK_MODE=per_box EXPORT_SUFFIX=_vlm; CHAIN_STEPS="1 5 6 7 8 road eval viz cvat cvat3d cvatroad" scripts/run_day1_chunks.sh 0001`
  (Stage 1 regenerates the clouds with the current ground code). Rule for the future:
  any resume that skips Stage 1 on a pruned tree must include `1`, or run with
  `PRUNE_CLOUDS=0`.
- **Its Stage 1 clouds predate the final ground decision (§4).** Plan was: when
  `--- STAGE 4 … OK` appears in `work_day1/logs/chunk_0001.log`, kill the window
  (`tmux kill-window -t pipe:day1`, then kill the stage PIDs — see §7 hazards), and
  resume with
  `CHAIN_STEPS="1 5 6 7 8 road eval viz cvat cvat3d cvatroad" … run_day1_chunks.sh 0001`
  (keeps `stage3_checked` + `stage4_masks`; regenerates geometry), then
  `run_day1_chunks.sh 0004` whole.
- `chunk_0004` after that runs the full chain with everything current.
- **16:55 update:** the kill-and-restart is automated. `pipe:relaunch` runs
  `/home/mt/dhakascenes/tools/relaunch_0001_after_stage4.sh`: it waits for the
  `--- STAGE 4` line, kills the chain by PID (descendants of pane shell 3988056 only),
  closes `pipe:day1`, and opens `pipe:day1b` with the 0001 restart then 0004 whole
  (VLM per_box, `EXPORT_SUFFIX=_vlm`). Milestones in `work_day1/logs/relaunch_0001.log`.
  **Measured 3c throughput is ~25 boxes/min (0.4 boxes/s), not 2.7/s**: 8 s per request
  (prompt eval 65 tok/s — image encoding + `--n-cpu-moe 8`), 4 slots. 3c on 0001 had
  4,437 done at 16:51 after 170 min; expect it to finish in the evening, 0004's 3c to
  take most of the night.
- Also in `pipe`: window `viewer` = `python3 -m http.server 8099` serving
  `/home/mt/dhakascenes/viewers/` (ground-plane viewer at
  `http://100.99.150.13:8099/ground-plane-kf100.html`).

## 2. Where every chunk stands

| chunk | last run | geometry | export | CVAT projects (day1_chunk_NNNN 2D/3D/road) |
|---|---|---|---|---|
| 0000 | 09:12–11:15, swap ✓ | **old ground band** (planes at −0.5 m) — boxes too short, ray-stretched | `export/day1_chunk_0000/` (5.8 GB) | #283 / #284 / #285 |
| 0006 | 11:16–11:41, swap ✓ | old ground band | `export/day1_chunk_0006/` | present |
| 0001 | 11:41–13:14, swap ✓ | old ground band | `export/day1_chunk_0001/` (6.3 GB) | present |
| 0001 | **in progress** (VLM) | Stage 1 = new band but all rings voting; must rerun 1,5–8 | → `export/day1_chunk_0001_vlm/` | new run-tagged tasks will land beside the old |
| 0002 | 13:15, stopped mid-Stage 6 | old | none | none |
| 0003, 0005 | not run | — | — | — |
| 0004 | queued behind 0001 | will be current | → `export/day1_chunk_0004_vlm/` | — |

**Everything exported before 2026-09-06 ~15:00 has the old ground plane** and needs a
rerun from Stage 1 (Stages 3/3f/3m/4 and any 3c tree can be kept: they are image-only).
Do NOT delete the existing exports or CVAT projects (§7).

## 3. The substrate: Dataset/A_nusc and its defects

`Dataset/A_nusc/chunk_0000..0006` — 2026-09-04 session `…_A_clean30min`, 4,536
keyframes, 39 GB, nuScenes-like, raw. Six cameras at 1280×720, `LIDAR_TOP` = Mid-360
rings 0–3 (~40k pts) + two ZED depth clouds as rings 100 (rear) / 101 (front)
(~350k pts). Git-ignored. Each chunk is its own dataroot with one scene.

The exporter (outside this repo, `/mnt/ssd/dhaka_crowd_dataset` tooling) has these
defects; `scripts/fixup_a_nusc.py` writes a corrected `v1.0-dhaka-fixed` beside
`v1.0-dhaka` per chunk (non-destructive, run by the driver, `fixup_meta.json` records it):

1. ~5 % of keyframes lack a camera image (drop; `CAM_LEFT` worst).
2. 1 dangling blob reference (drop).
3. One `ego_pose` shared by all sensors of a sample → per-camera pose interpolated
   from the LiDAR trajectory (Stage 5 refuses otherwise).
4. Camera rotations in body convention → re-expressed optical
   (`q ⊗ (0.5,−0.5,0.5,−0.5)`); before this 0 % of LiDAR hit any image.
5. **CAM_LEFT / CAM_RIGHT streams swapped** → `--swap-channels CAM_LEFT CAM_RIGHT`
   (calibration tokens exchanged per sample; measured by image drift ±70–80 px).
6. No `sweeps/` (Stage 3b cannot run; empty dir created for the validator).
7. Rear ZED miscalibrated: road 0.69 m low at its camera, sinks with range (§4).
   **Not corrected yet** — it just no longer votes for the ground.

Fix these upstream in the exporter eventually.

## 4. Pipeline changes (all test-first; suite 357 green in `ano_pipe`)

`pipeline/common/schemas.py` — new profile `dhaka6`; the profile now OWNS:
`ring_cameras` (6), image pin 1280×720, accumulation window (`w_acc_count 1`,
`w_acc_duration_ns 0` — no sweeps), parse bands (`pcd [10k,1M]`, `jpeg [10kB,4MB]`),
`stereo_rings (100,101)` + `stereo_stride 8` (Stage 6 DBSCAN ate 62 GB RAM unthinned),
`ground_z_band_m (−3.5,−1.0)` (ego origin = LiDAR 2.3 m up; ISO-8855 band never
held the road → planes at −0.5 m → **the "out of proportion" cuboids**),
`ground_fit_rings (0,1,2,3,101)` + `ground_fit_range_m (3,12)` (operator decision
15:40: LiDAR + front ZED vote for the ground; rear ZED never). Select with
`DHAKASCENES_SUBSTRATE=dhaka6`. `SubstrateManifest` floor on `w_acc_duration_ns` is 0.

`pipeline/stage1_ingestion/ingest.py` — `thin_stereo`, profile-driven ground band /
candidate rings / range; provenance in `single_sweep_sources.stereo_thinning`.
`pipeline/stage0_data_probe/probe.py` — reads the profile bands/window.
`pipeline/stage1_ingestion/ingest.py` — **ground reference plane (15:00, general to every
chunk)**: the "one wedge much higher than the rest" defect. A wedge whose own fit was
rejected got a plain least-squares plane over every in-band candidate; the band reaches
1.35 m above the road and every non-road candidate is above it, so the substitute sat
+0.32 m (p10 +0.15, p90 +0.50) above the accepted wedges in 21 % of all wedge fits on
chunk_0000 (1167/5472), stripping the bottom of every object in those wedges. The
inlier-ratio gate (0.35) caused 707 of those: it measures clutter, not fit quality.
Now `fit_ground_planes()` fits a robust reference plane (same seeded RANSAC + polish,
seed index = n_sectors) and the guard rejects on tilt (>15°) or on **height disagreement
with the reference at the wedge centroid** (`reject_height_disagreement_m 0.25`);
`reject_min_inlier_ratio` defaults to 0.0 (off). Diagnostics gain
`ground_reference_plane` per keyframe; rejected fits record `height_disagreement_m`,
reason `height`. Measured on chunk_0000 (`work_scratch_v2/`, same 684 kf): substitutions
1167 → 850, substitute bias +0.32 → −0.08 m, per-kf wedge spread median 0.46 → 0.34 m
(p90 0.79 → 0.58). Rejected wedges disagree by 0.77 m median, p10 0.29 m, so 0.25 m is
not cutting into a continuum. The residual −0.08 m is the front ZED sitting ~0.2 m low
(item 4 below) dominating the reference consensus. Tests:
`tests/test_stage1_ground_reference.py` (8). **Every chunk needs Stage 1 onward rerun**
(already true for the band fix).
`pipeline/stage3_merge/merge.py` — **C36**: an arm A bicycle/motorcycle whose own area
is ≥ 60 % covered by a surviving arm B rickshaw/auto-rickshaw is absorbed as its wheel
(containment, not IoU); ledger `merge.suppressed_parts`; `--suppress-part` /
`--no-suppress-parts`; default on.
`scripts/cvat_setup.py` — `--share-prefix` (env `CVAT_SHARE_PREFIX`): the CVAT share
is a bind of `/home/mt/Zami/nuscenes` that still holds pilot_1632 frames under the
same names; each chunk is hard-linked under `day1_chunk_NNNN/` there.
`scripts/cvat_setup_3d.py` — `OURS_PROJECT` env-overridable (`CVAT_PIPELINE_3D_PROJECT`).
`scripts/author_priors_dhaka.py` — the handover-§6 authored priors as code
(nuScenes classes transferred + rickshaw/CNG literature dims, σ assumed), bound per
chunk fingerprint; box sizes are NOT evidence about Dhaka object sizes.
`scripts/fixup_a_nusc.py` — §3 transforms 1–5.
`scripts/run_day1_chunks.sh` — the driver (§5).
`scripts/run_stages.sh` — only `VLM_CHECK`/`VLM_USE_CHECKED` defaults flipped to 0.
`.env` — `HF_HOME` → `~/.cache/huggingface` (the pilot cache was wiped 2026-09-05).

Tests: `tests/test_day1_substrate_and_cvat3d_project.py`, `test_fixup_a_nusc.py`,
`test_author_priors_dhaka.py`, `test_cvat_share_prefix.py`, `test_stage1_thin_stereo.py`,
`test_stage3_merge_parts.py`.

## 5. The driver: `scripts/run_day1_chunks.sh [ids]`

Per chunk: `mkdir sweeps/` → fixup (once; reused if `v1.0-dhaka-fixed` exists — delete
it to redo) → author priors (rebinds on fingerprint change) → hard-link samples into the
CVAT share → write `configs/paths_day1_chunk_NNNN.yaml` → `run_stages.sh $CHAIN_STEPS`
(default `0 1 3 3f 3m 4 5 6 7 8 road eval viz cvat cvat3d cvatroad`; 3c inserted by
`VLM_CHECK=1`) → Stage 9 gate → `export_release --blobs copy` → `export/day1_chunk_NNNN$EXPORT_SUFFIX/{boxes,road,coco_2d,README.md}`
→ prune `stage1_ingestion/clouds`. Env knobs: `CHAIN_STEPS`, `EXPORT_SUFFIX`,
`SWAP_CHANNELS` (default `CAM_LEFT CAM_RIGHT`), `PRUNE_CLOUDS`, `DRY_RUN=1`, plus the
wrapper's `VLM_*`. The Stage 8 gate only accepts a marker newer than the chain start
(an aborted chain once exported stale trees). Per-chunk roots:
`/home/mt/dhakascenes/{work,out,probe_out}_day1/chunk_NNNN`; logs in `work_day1/logs/`.

Timings (RTX 4090): Stage 4 ≈ 0.19 s/img, Stage 6 ≈ 0.10 s/instance, Stage 7 ≈ 0.13 s/box
(both single-core) → ~1.5–2 h per 750-kf chunk; 3c per-box ≈ 2.7 boxes/s.
VLM needs `/home/mt/dhakascenes/tools/llama.cpp/build/bin/llama-server` (b10711, CUDA;
**parallel builds segfault on this machine — build with `-j 2`**), GGUFs in
`cache/checkpoints/nemotron-omni/`. `per_track` mode cannot work here (no 3b) → `per_box`.

## 6. Operator rules (memory files exist for each)

- **Never delete CVAT projects/tasks or on-disk exports unless explicitly told**
  (publishes are additive by run tag; use `EXPORT_SUFFIX`).
- **Local viewers only** — the operator is remote (Tailscale); serve HTML from
  `/home/mt/dhakascenes/viewers/` on :8099, never claude.ai Artifacts.
- Arm B (3f/3m) is always part of the Dhaka chain.
- Overnight runs in tmux `pipe`; blunt class-skips preferred.

## 7. Hazards learned the hard way

- `pkill -f "run_stages"` killed the operator's tmux **server** (its argv held the
  original launch command). Stop runs with `tmux kill-window` + kill by PID, never by
  pattern. `tmux kill-window` alone does not kill the driver's children.
- Monitors: start `tail -F` **before** launching, or an early refusal is missed.
- A reused work tree's old markers can satisfy naive "is Stage 8 done?" checks.
- The CVAT share root is shared across substrates — always publish with a prefix.
- Stage 1 writes ground-FILTERED clouds; never re-fit the ground on `stage1_ingestion/clouds` (the road is gone) — rerun Stage 1 into a scratch root instead (`configs/paths_scratch_chunk_0000_v2.yaml` → `work_scratch_v2/`).
- `chunk_0006` is parked in a crowd (one car fills the forward sector) — poor for
  calibrating a ground fitter; use `chunk_0000` (moving) — a scratch config for it exists:
  `configs/paths_scratch_chunk_0000.yaml` (work root `work_scratch/`).

## 8. Open items, in priority order

1. Finish the 0001/0004 VLM run per §1 (the Stage 1 restart of 0001 and all of 0004 pick up the ground-reference fix automatically); check `merge.suppressed_parts` counts and
   the 3D task proportions on 0001 (pedestrian ~0.7×0.8×1.7 expected now).
2. Rerun 0000/0006/0002/0003/0005 from Stage 1 with current code (`EXPORT_SUFFIX` per run).
3. Rear-ZED correction: shift ring-100 points +0.69 m (+~1° tilt about CAM_BACK) at
   Stage 1 read time, profile-declared; measured on chunk_0000 kf 40–640.
4. Front ZED is ~0.2 m low too (small); consider the same treatment.
5. Stage 6/7 parallelism (single-core; ~1.5 h/chunk together) if throughput matters.
6. Disk: 130–140 GB free; each chunk ≈ 6 GB retained work + 6–7 GB export + ~5 GB in CVAT.
7. Commit the working tree once the operator approves.
