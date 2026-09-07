# Optical-convention fix + L/R swap fix, and the regenerated `v1.0-dhaka-fixed`

**Date:** 2026-09-08
**Interpreter:** `/home/mt/miniconda3/envs/ano_pipe/bin/python` (`ano_pipe`)
**Branch:** `main` (nothing pushed)
**Data:** `/media/mt/vol_2/Annotation-pipe-data/full-fused`
**Evidence:** `docs/evidence/2026-09-08-camera-lr-mapping.md` §5, §6

## Status

`v1.0-dhaka-fixed` has been regenerated and is **ready for the run**. Two blocking defects were
fixed, not one — the second was found by the numeric verification that was asked for, and it would
have silently defeated the whole point of `--swap-channels`.

| commit | what |
|---|---|
| `156d52b` | `fixup_a_nusc: recognise an exporter that already wrote optical camera rotations` |
| `5fce848` | `fixup_a_nusc: --swap-channels must swap the keyframe, not the last sweep` |
| `d85e09c` | cherry-pick of `7ba9cc0`, the evidence file (see "Repo discrepancy" below) |

Full suite: **561 passed** (`$PY -m pytest tests/ -q`), up from 544; 17 tests added.

---

## 1. The reported bug: double-applied body→optical conversion

`cameras_to_optical_convention()` guarded against double application with an exact string test:

```python
if cal.get("frame_convention") == "optical":   # scripts/fixup_a_nusc.py:253, before
    continue
```

The 2026-09-05 export declares the convention in a sentence, not the bare token:

```
"frame_convention": "camera axes are optical (x-right, y-down, z-forward); rotation maps camera-optical -> LIDAR_TOP/ego"
```

so the guard never fired and all six already-optical rotations were converted a second time,
rotating the whole camera ring by 90°. Measured, per evidence §5: median LiDAR-to-image gradient
agreement fell from r = 0.245 to −0.025 on `CAM_FRONT` and 0.168 to −0.036 on `CAM_BACK` over 202
keyframes, the raw extrinsic winning on 98 % / 95 % of them.

### Signals confirmed against the real data, not assumed

`/media/mt/vol_2/Annotation-pipe-data/full-fused/v1.0-dhaka/calibrated_sensor.json` has 7 rows
(1 LiDAR + 6 cameras). Every camera row carries **both** signals the fix looks for, and no third
unambiguous one exists — the only other per-row field is `camera_distortion_status`, which is
about intrinsics, not frames. The arithmetic claim in §5 reproduces exactly (quaternion sign
ignored, since q and −q are the same rotation):

```
CAM_RIGHT        max|rotation - rotation_body (x) q_b2o| = 1.1e-16
CAM_LEFT                                                   1.2e-16
CAM_FRONT_RIGHT                                            5.6e-17
CAM_FRONT_LEFT                                             1.1e-16
CAM_BACK                                                   0.0e+00   (sign-flipped storage)
CAM_FRONT                                                  1.1e-16   (sign-flipped storage)
```

### The new detection

`already_optical(cal)` returns a human-readable reason, or `None`:

* `frame_convention` is a string containing `"optical"` in **any case** — covers both the bare
  token this script stamps and a descriptive sentence;
* a four-element `rotation_body` that **differs** from `rotation` — an upstream exporter that kept
  the pre-conversion rotation beside the converted one. Absent, empty or identical `rotation_body`
  says nothing and is ignored.

A row with **neither** signal is converted and stamped exactly as before (the day-1 exporter's
path, pinned by test against the expected `[0.5, -0.5, 0.5, -0.5]`). Skipped rows are left byte-
identical, including the exporter's own `frame_convention` / `frame_convention_note`.

### The silent skip is now loud

`cameras_to_optical_convention()` returns `(tables, converted, already_optical)`. The CLI prints

```
  camera extrinsics: 0 converted, 6 already optical (skipped)
```

instead of the old `6 rotation(s) re-expressed body -> optical convention`, which is exactly how
this hid. `fixup_meta.json` gains `n_camera_rotations_already_optical` and
`camera_frame_convention_decisions`, a per-camera decision with its reason.

---

## 2. The bug the verification found: `--swap-channels` never touched a keyframe

`swap_camera_channels()` grouped rows as `by_sample[sample_token][channel] = row` — **one row per
(sample, channel)**. A sample owns its keyframe *and* the ~6 sweeps filed against it per side
camera, so that dict kept only the **last** row, which is a sweep. Measured on the first
regeneration:

```
KEYFRAME rows   (channel in v1.0-dhaka -> channel in fixed)
   CAM_LEFT  -> CAM_LEFT    14965        CAM_LEFT  -> CAM_RIGHT       1
   CAM_RIGHT -> CAM_RIGHT   14965        CAM_RIGHT -> CAM_LEFT        1
SWEEP rows
   CAM_LEFT  -> CAM_RIGHT   14965        CAM_RIGHT -> CAM_LEFT    14965
```

**1 keyframe pair of 14 966 swapped.** The run printed `29932 sample_data rows re-paired` and was,
for every stage that reads keyframes, a no-op — the exact failure evidence §6 exists to prevent
(`CAM_LEFT` drifts +44.6 px against a negative prediction, agreeing with its own calibration on
7.7 % of 520 pairs, p ≈ 7e-97; `CAM_RIGHT` −45.3 px against positive, 14.5 % of 531, p ≈ 5e-66).

The old fixture — one keyframe per channel per sample — could not see it. Every row of both
channels in a sample now takes the other's `calibrated_sensor_token`; the "sample holding only one
of the pair is left alone" guard, filename/timestamp/ego_pose invariance and self-inverseness are
unchanged and re-covered with sweeps present.

---

## 3. Tests (TDD, watched fail first)

17 added in `tests/test_fixup_a_nusc.py`; 14 for the convention, 3 for the swap. All 14 failed on
the unpatched script, and the swap tests reproduced the keyframe miss at unit scale before the fix.

* descriptive sentence → not converted, rotation byte-identical, counted skipped
* bare `"optical"` still skipped (no regression)
* `"...OPTICAL..."` matched case-insensitively; `"body"` is **not** a skip signal
* `rotation_body` beside `rotation` skips with no `frame_convention` at all, and stamps nothing
* absent / empty / identical `rotation_body` is not a signal — still converts
* neither signal → converted exactly as today, quaternion pinned, note stamped
* a skipped row keeps the exporter's `frame_convention`, `frame_convention_note`, `rotation_body`
* one pass over a mixed table returns `(1, 1)`; LiDAR counts as neither
* `main()` prints `camera extrinsics: 1 converted, 1 already optical (skipped)` and records both
* every row of the pair swaps, not just the last; keyframes swap when sweeps are present; still
  its own inverse with sweeps

---

## 4. Regeneration

The known-bad `v1.0-dhaka-fixed` (produced 04:49 by the double conversion, tables only, 442 MB, no
blobs, no downstream consumers) was **deleted explicitly** — the script refuses an existing
`--out-version` — and regenerated under the same name so the eight `configs/paths_*.yaml` that
name `version: v1.0-dhaka-fixed` stay valid:

```bash
rm -rf /media/mt/vol_2/Annotation-pipe-data/full-fused/v1.0-dhaka-fixed

DHAKASCENES_SUBSTRATE=dhaka6 /home/mt/miniconda3/envs/ano_pipe/bin/python scripts/fixup_a_nusc.py \
    --dataroot /media/mt/vol_2/Annotation-pipe-data/full-fused \
    --version v1.0-dhaka --out-version v1.0-dhaka-fixed \
    --sanitize-scene-names --swap-channels CAM_LEFT CAM_RIGHT
```

Output:

```
required channels from profile 'dhaka6': ['LIDAR_TOP', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_LEFT', 'CAM_RIGHT', 'CAM_BACK']
/media/mt/vol_2/Annotation-pipe-data/full-fused/v1.0-dhaka -> /media/mt/vol_2/Annotation-pipe-data/full-fused/v1.0-dhaka-fixed
  samples: 15025 -> 14966  (dropped 59)
  sample_data rows: 552448 -> 551580
  ego_pose rows: 462514 -> 552310  (+89796 per-camera poses interpolated from the LiDAR trajectory)
  camera extrinsics: 0 converted, 6 already optical (skipped)
  scene names: 11 of 11 sanitized to one path component — {'dhaka_20260905_174950/chunk_0000': 'dhaka_20260905_174950_chunk_0000', 'dhaka_20260905_174950/chunk_0001': 'dhaka_20260905_174950_chunk_0001', 'dhaka_20260905_174950/chunk_0002': 'dhaka_20260905_174950_chunk_0002', 'dhaka_20260905_174950/chunk_0003': 'dhaka_20260905_174950_chunk_0003', 'dhaka_20260905_174950/chunk_0004': 'dhaka_20260905_174950_chunk_0004', 'dhaka_20260905_174950/chunk_0005': 'dhaka_20260905_174950_chunk_0005', 'dhaka_20260905_174950/chunk_0006': 'dhaka_20260905_174950_chunk_0006', 'dhaka_20260905_174950/chunk_0007': 'dhaka_20260905_174950_chunk_0007', 'dhaka_20260905_174950/chunk_0008': 'dhaka_20260905_174950_chunk_0008', 'dhaka_20260905_174950/chunk_0009': 'dhaka_20260905_174950_chunk_0009', 'dhaka_20260905_174950/chunk_0010': 'dhaka_20260905_174950_chunk_0010'}
  channels swapped: [['CAM_LEFT', 'CAM_RIGHT']] (178731 sample_data rows re-paired)
  missing by channel among dropped: {'CAM_BACK': 45, 'CAM_FRONT': 25, 'CAM_FRONT_LEFT': 30, 'CAM_FRONT_RIGHT': 45, 'CAM_LEFT': 30, 'CAM_RIGHT': 41}
```

`178 731` rows re-paired, against `29 932` before the swap fix.

---

## 5. Numeric verification

### 5a. Camera rotations are now IDENTICAL to `v1.0-dhaka`

| camera | identical | max abs component delta vs raw | boresight yaw, new | boresight yaw, OLD bad dir | max delta, OLD bad |
|---|---|---|---|---|---|
| CAM_FRONT | ✔ | 0.0e+00 | **+3.2°** | −86.8° | 0.9986 |
| CAM_FRONT_LEFT | ✔ | 0.0e+00 | **+45.4°** | −44.6° | 0.9206 |
| CAM_LEFT | ✔ | 0.0e+00 | **+92.4°** | +2.1° | 0.7166 |
| CAM_BACK | ✔ | 0.0e+00 | **−179.3°** | +90.5° | 0.9983 |
| CAM_RIGHT | ✔ | 0.0e+00 | **−89.1°** | −179.2° | 0.7120 |
| CAM_FRONT_RIGHT | ✔ | 0.0e+00 | **−43.5°** | −133.9° | 0.9253 |

All six byte-identical; the parsed `calibrated_sensor.json` content is equal to the source's (only
JSON whitespace differs). The old-directory yaws are the 90° rotation evidence §5 measured — the
new ones are a ring: front ≈ 0, front-left ≈ +45, left ≈ +90, back ≈ 180, right ≈ −90,
front-right ≈ −45. `frame_convention` and `rotation_body` are preserved verbatim on all six.

Sample values (`CAM_FRONT`):

```
raw : [-0.492803690325,  0.533698593035, -0.509345603667,  0.461386378883]
new : [-0.492803690325,  0.533698593035, -0.509345603667,  0.461386378883]   <- identical
bad : [ 0.50581344263,   0.537230754072, -0.46491853992,   0.489271529288]   <- 90 deg off
```

### 5b. CAM_LEFT / CAM_RIGHT assignment IS exchanged, on every keyframe

```
KEYFRAME rows (file folder, channel in v1.0-dhaka -> channel in fixed): count
   ('CAM_LEFT',  'CAM_LEFT',  'CAM_RIGHT')   14966      <- 100 % of samples
   ('CAM_RIGHT', 'CAM_RIGHT', 'CAM_LEFT')    14966
SWEEP rows
   ('CAM_LEFT',  'CAM_LEFT',  'CAM_RIGHT')   74723
   ('CAM_RIGHT', 'CAM_RIGHT', 'CAM_LEFT')    74076
```

Filenames did not move: every row kept its own file, timestamp and ego_pose, and only
`calibrated_sensor_token` changed — `samples/CAM_LEFT/*.jpg`, measured to be the right-facing
stream, is now read as channel `CAM_RIGHT` with the matching calibration.

### 5c. Scene names, counts, referential integrity

```
scenes: 11        any '/' in a name: False
names : dhaka_20260905_174950_chunk_0000 .. dhaka_20260905_174950_chunk_0010
samples: 14966    == 14966 -> True      sum(scene.nbr_samples) = 14966
fixup_meta: to_optical=0  already_optical=6  swapped_rows=178731  scenes_sanitized=11  dropped=59

dangling calibrated_sensor_token : 0
dangling ego_pose_token          : 0
dangling sample_token            : 0
samples missing a required channel: 0
broken prev/next links           : 0
scene first/last consistent      : True
keyframe blobs missing on disk   : 0
```

---

## 6. Repo discrepancy worth knowing

`docs/evidence/2026-09-08-camera-lr-mapping.md` was **not** on `main` — commit `7ba9cc0` sits only
on branch `benchmark-release`, which is 2 commits *behind* `main` (the `b6e673d` merge predates
it). Both code commits cite that path, so it was cherry-picked onto `main` as `d85e09c` (docs and
figures only, no code). Drop it with `git rebase --onto 5fce848 d85e09c main` if you would rather
land it by merging `benchmark-release`; git will not mind the duplicate content either way.

## 7. Not done

Nothing was pushed. No stage was run against the regenerated substrate — the verification above is
static (table arithmetic and referential integrity), not an end-to-end projection check. Evidence
§5's projection measurement was over `v1.0-dhaka`'s rotations, which are now exactly what
`v1.0-dhaka-fixed` carries, so the 90° defect cannot recur; the L/R swap's effect on a real
projection is unverified beyond the row-level accounting in §5b.
