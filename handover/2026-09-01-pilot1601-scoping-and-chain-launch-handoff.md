# Handoff — pilot_1601 runnability scoping, and the ready-to-launch 1632 comparison chain

**Date:** 2026-09-01 (night) · **Repo:** `/home/mt/Zami/Annotation_pipeline` · **Branch:** `main`
**Nothing is committed.** Continues `2026-09-01-stage4-parallel-probe-and-cvat-c30-handoff.md`
(same day, evening); its gates and §4 debts are all still open. This document covers only
the night's investigation: the operator wants the comparison chain run on **BOTH** pilot
zips, and this scopes what that takes. **No pipeline stage was run. No file outside
`handover/` and the scratchpad was written.**

---

## 1. Read this first

Three things happened tonight, all read-only:

1. **Terminology settled:** nothing "trains" on the pilot zips. They are capture data the
   pipeline annotates. The only trained artifact is arm B's
   `local_yolox_build/artifacts/yolo11x-rsud20k-armb.pt` (RSUD20K from Kaggle, run
   `r1280-4`, best epoch 36, mAP50 0.930) — consumed by step `3f`, never trained here.
2. **The 1632 comparison chain was dry-run** (`PRINT_STEPS=1`) and is ready to launch (§2).
3. **pilot_1601 was scoped** by reading its metadata out of the zip (no full extraction):
   it is blocked by four separable problems, all enumerated with evidence (§3–§4). It is
   **not** runnable by any flag; it needs a fixup pass like 1632 got, plus one code change.

| | |
|---|---|
| Pipeline stages run tonight | **none** — dry run only (`PRINT_STEPS=1` runs nothing) |
| Repo | unchanged except this file; still all uncommitted (COMMIT remains the standout risk) |
| CVAT server | untouched |
| GPU | idle throughout |
| 1601 metadata, extracted for reading | scratchpad `…/scratchpad/p1601/` (session-temp, deletable; blobs never extracted) |
| Dry-run log stub | `work/logs/run_20260901_183911.log` |

---

## 2. The 1632 run — dry-run verified, ready to launch

```bash
cd ~/Zami/Annotation_pipeline
VLM_CHECK=1 bash scripts/run_stages.sh 3f 3m 3c 4 5 6 7 8 eval viz cvat cvat3d \
  2>&1 | tee /home/mt/dhakascenes/work/logs/chain_$(date +%Y%m%dT%H%M%S).log
```

`PRINT_STEPS=1` on exactly this line resolved: steps `3f 3m 3c 4 5 6 7 8 eval viz cvat
cvat3d`, VRAM cap 22000 MiB, and two notes to expect on the real run:

- **`upstream is DEGRADED on disk`** — the chain will pass `--accept-degraded-upstream`
  (C16, recorded per-manifest). Inherited from the last full run; expected.
- **`stage4_input: stage3b_track2d`** — a snapshot of disk *now*. Once `3m`/`3c` produce
  fresh trees mid-chain, those outrank it (freshness ranking, d75c27e). Not a bug.

Reminders that still apply from the evening handoff: GPU must be idle or the injected 3c
refuses in ~0.2 s and aborts the chain (`VLM_ALLOW_SHARED_GPU=1` is the escape); budget
~7 h + ~2.6 h per-track 3c; **no `--cvat-replace`** — C30's additive default is exactly
what keeps the old CVAT tasks for side-by-side; after the publish, eyeball the task list
(C30's live-server gate is still open); this is arm B's first-ever run.

---

## 3. What pilot_1601 actually is (measured from the zip, not the 08-30 recollection)

Read via `unzip -l` and metadata-only extraction. `export_meta.json` is the authority.

- **6 cameras + LIDAR_TOP, nothing else.** `sensor.json` lists exactly: LIDAR_TOP,
  CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_LEFT, CAM_RIGHT, CAM_BACK.
  **No CAM_BACK_LEFT / CAM_BACK_RIGHT. No ZED at all.** No sweeps.
- **719 samples** (one scene, `chunk_0000` — the SAME scene name as 1632's; see §4.4),
  ~72 s @ 10 Hz. Session `pilot_1601_20260829_160128`.
- **CAM_RIGHT has 717 frames against 719 samples** (2 keyframes incomplete);
  CAM_LEFT/others 719. The zip's 720/718 file counts are the 719/717 plus directory
  entries — `export_meta.json`'s numbers are the real ones.
- **`ego_pose` is IDENTITY** — "this export carries no ego motion", exactly like raw
  1632 was before the fix pass.
- **Calibration state matches raw 1632:** FRONT_LEFT/FRONT_RIGHT intrinsics are NOMINAL
  (FOV-derived, no distortion, centre principal point); every extrinsic is the GA-01
  drawing nominal; camera rotations are in body convention (pipeline needs optical);
  latency uncorrected on most channels. Same rig, so `fix/pitch_final.json` (all 8
  channels, incl. the 6 present here) should transfer — verify, don't assume.
- Metadata tables all present; annotations empty (raw capture). `sample_data.json`
  2.4 MB, `ego_pose.json` 174 KB.

---

## 4. Why it cannot run today — four separable blockers

**4.1 — REQUIRED_CHANNELS (the hard one, needs a code decision).**
`pipeline/common/schemas.py:114`: `REQUIRED_CHANNELS = ("LIDAR_TOP",) + RING_CAMERAS`,
and `RING_CAMERAS` is pinned to all **8** Dhaka cameras. Stage 0's `channels_complete`
predicate requires every keyframe to carry every required channel; a failing predicate
makes the scene unusable (`SceneVerdict.usable = not failing`, probe.py:190), and with
the only scene excluded the probe raises `HardStop("zero usable scenes…")`
(probe.py:797-799) — rc 2, nothing written. So 1601 dies in Stage 0, every keyframe,
on the two missing rear cameras.

Fix options, undecided: (a) make the required camera set **dataset-aware** (derive from
`sensor.json`, or a `paths.yaml`/CLI override) — honest but touches the meaning of ring
coverage; (b) edit the pin for a 1601 run — fast but the schemas.py comment itself warns
that changing the ring changes what R2 counts, so runs before/after stop being
comparable. Blast radius if (a): `RING_CAMERAS`/`REQUIRED_CHANNELS` are consumed in
**9 files** — schemas.py, stage0 probe.py, stage1 ingest.py, stage2 ood.py, stage3
proposals.py, stage3b track2d.py, stage5 lift.py, scripts/render_boxes_3d.py,
scripts/collect_evidence.py. Each needs reading before any of this is coded; stage5's
camera-priority tie-break and the R2 coverage metric are the two most likely to carry
silent 8-camera assumptions.

**4.2 — the 2 keyframes with no CAM_RIGHT.** Even with a 6-camera required set,
`channels_complete` fails those 2 of 719 keyframes and excludes the scene (§4.1
mechanics). Cleanest fix: **drop those 2 samples during the fixup pass** (metadata-only,
preserves the predicate's meaning) rather than teaching the predicate tolerance.

**4.3 — identity ego_pose (the fix machinery already exists and is generic).**
`~/dhakascenes/fix/lidar_odometry.py` (scan-to-local-map point-to-plane ICP, argparse,
mover-trimming) produces a poses json; `~/dhakascenes/fix/fixup_export.py
--dataroot … --poses … --pitch …` writes a NEW `v1.0-dhaka-fixed` beside the original —
rotations body→optical with pitch fold-in, ego interpolation per sample_data timestamp,
width/height repair, prev/next chains, map stub. It is already parameterised; nothing
1632-specific is hardcoded in fixup_export.py itself (the accum_test*.py verifiers DO
hardcode 1632's dataroot). The odometry quality gates that were run for 1632
(accum tests, pole tracking, render_check) should be re-run for 1601 — 72 s of Dhaka
traffic is a harder odometry diet than 230 s was.

**4.4 — no ZED: runs, but with the known near-field quality hole.** Stage 1's stereo
fusion skips absent channels gracefully (`channel_records.get(channel) is None →
continue`, ingest.py:~690), so nothing refuses. But ZED fusion exists because the
Mid-360 alone puts a median of 10 points on a near object — the 2026-08-30 measurement
was **64% of median box volume from priors** and **56% ambiguous yaws** without it. 1601's
Stage 6 boxes will be lidar-only: expect degraded, not blocked, and say so in any
comparison. Also: both zips name their scene `chunk_0000`, so if 1601 is ever published
to CVAT its tasks are distinguishable from 1632's **only by run tag** — set
`CVAT_RUN_TAG=1601-<something>` explicitly rather than relying on manifest mtimes.

**Also required — separate roots (no code, one config file).** `run_stages.sh:175` reads
`DHAKASCENES_PATHS_CONFIG` (default `configs/paths.yaml`). 1601 needs its own
`configs/paths_1601.yaml` with its own `dataroot` AND its own `work_root`/`out_root` —
stages rewrite their trees in place, so pointing 1601 at `/home/mt/dhakascenes/work`
would **destroy the 1632 run's outputs**. The path contract already validates the
disjointness invariants.

---

## 5. Proposed 1601 sequence (nothing below has been started)

1. Decide §4.1 (a) vs (b) — this is the only step needing a design decision.
2. Extract `pilot_1601.zip` → `/home/mt/dhakascenes/data/pilot_1601`.
3. Verify image size is 1280×720 (assumed from the rig, **not verified** — the
   `IMAGE_WIDTH_PX`/`IMAGE_HEIGHT_PX` pins in schemas.py depend on it).
4. Odometry: `lidar_odometry.py` against 1601's LIDAR_TOP → poses json; re-run the
   accum/pole verifiers against it (they need their hardcoded `DR` repointed).
5. Fixup: `fixup_export.py --dataroot …/pilot_1601 --poses … --pitch fix/pitch_final.json`
   (+ drop the 2 CAM_RIGHT-less samples, a small extension) → `v1.0-dhaka-fixed`.
6. `configs/paths_1601.yaml` with disjoint work/out roots.
7. `DHAKASCENES_PATHS_CONFIG=configs/paths_1601.yaml bash scripts/run_stages.sh 0 1 …`
   — full chain from Stage 0 (nothing exists for 1601), `CVAT_RUN_TAG` set explicitly.
8. Only after 1632's chain has finished — one GPU, and 3c refuses a shared one.

Rough shape: steps 2–6 are hours of work, not days; step 1's option (a) is the only
open-ended item. The 1632 chain (§2) does not wait on any of this.

---

## 6. What was NOT determined tonight

- Whether stage5/6/7 carry silent 8-camera or ZED assumptions beyond the ones named —
  the 9-file consumer list was enumerated, not read.
- 1601 image resolution (§5.3) and whether `pitch_final.json` transfers (§3).
- Whether 72 s / 719 keyframes passes `sweeps_cover_window` and the other Stage 0
  predicates once channels are settled — unknowable without running the probe.
- Everything already owed from the afternoon/evening handoffs: C29 ratification, C30
  live-server gate, render flags, `sam3_exemplar`, and **the commit**.
