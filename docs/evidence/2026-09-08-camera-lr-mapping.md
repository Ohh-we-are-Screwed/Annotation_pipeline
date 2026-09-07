# CAM_LEFT / CAM_RIGHT mapping on the 2026-09-05 Dhaka capture

**Date:** 2026-09-08
**Data:** `/media/mt/vol_2/Annotation-pipe-data/full-fused`, versions `v1.0-dhaka` (exporter output)
and `v1.0-dhaka-fixed` (14 966 keyframes, 11 scenes `dhaka_20260905_174950_chunk_0000` … `_chunk_0010`,
~5 Hz, 6 cameras + fused `LIDAR_TOP`).
**Question:** `release_meta.json` lists *"Camera L/R (RealSense) channel mapping to be walk-test verified
(day1 needed a CAM_LEFT/CAM_RIGHT fixup)"*. Are the two side streams swapped again?

---

## Verdict

> ### 1. `CAM_LEFT` and `CAM_RIGHT` **are swapped**. Pass `--swap-channels CAM_LEFT CAM_RIGHT`.
> Confidence: **very high**. Four independent lines of evidence agree, the primary one over
> 605 keyframe pairs in 9 scenes with no scene dissenting (binomial p ≈ 7e-97 / 5e-66).
>
> ### 2. `CAM_FRONT_LEFT` / `CAM_FRONT_RIGHT` are **correctly** mapped. Leave them alone.
> Confidence: **high** (77.0 % / 75.8 % of pairs agree with their own calibration, p ≈ 3e-30 / 7e-16;
> sign consistent in 8/9 and 9/9 scenes). Same direction as the previous capture's decision.
>
> ### 3. **Blocker, unrelated to the swap:** `v1.0-dhaka-fixed`'s camera extrinsics are wrong.
> `scripts/fixup_a_nusc.py` applied its body→optical correction to rotations the exporter had
> **already** written in the optical convention, rotating the whole camera ring by 90°. Measured,
> not inferred (§5). **Do not run the run against `v1.0-dhaka-fixed` as it stands, and do not
> produce a new fixed version without suppressing that conversion** — otherwise every 3D box
> projects into the wrong camera regardless of what the L/R swap does.

### Command implied

```bash
DHAKASCENES_SUBSTRATE=dhaka6 python scripts/fixup_a_nusc.py \
    --dataroot /media/mt/vol_2/Annotation-pipe-data/full-fused \
    --version v1.0-dhaka \
    --out-version v1.0-dhaka-fixed2 \
    --sanitize-scene-names \
    --swap-channels CAM_LEFT CAM_RIGHT
```

Three things about that line, none of them optional:

* `--swap-channels CAM_LEFT CAM_RIGHT` is the answer to the question asked. Its effect is to exchange
  the two rows' `calibrated_sensor_token` within each sample, so `samples/CAM_LEFT/*.jpg` — measured
  here to be the **right**-facing stream — is thereafter read as channel `CAM_RIGHT`, with the
  matching calibration. Filenames on disk do not move.
* `--sanitize-scene-names` must be kept: this exporter's scene names contain a slash
  (`dhaka_20260905_174950/chunk_0000`), and a scene name is a directory component in every stage.
  The existing `v1.0-dhaka-fixed` was built with it.
* `--out-version` is needed because the script refuses to overwrite, and `v1.0-dhaka-fixed` already
  exists. Nothing is deleted here — the new version dir sits alongside the old one; blobs are shared
  and untouched.

**Caveat — that command as it stands also re-applies the bogus body→optical rotation** (see §5), so it
must not be run until `cameras_to_optical_convention()` is made to recognise this exporter's
`frame_convention` string. The guard is

```python
if cal.get("frame_convention") == "optical":   # scripts/fixup_a_nusc.py:253
    continue
```

and this exporter writes

```
"frame_convention": "camera axes are optical (x-right, y-down, z-forward); rotation maps camera-optical -> LIDAR_TOP/ego"
```

so the guard does not fire. `v1.0-dhaka-fixed/fixup_meta.json` records
`"n_camera_rotations_to_optical": 6` — all six cameras were converted, all six wrongly.
No pipeline code was changed by this investigation.

---

## 1. Method

Same idea as the 2026-09-06 diagnosis (rearward image drift on moving keyframes), with three
additions: the prediction is derived from each camera's own extrinsic rather than assumed; the
direction of travel is measured from the LiDAR instead of taken from `ego_pose` (which turned out
to be unusable, §4); and `CAM_FRONT` / `CAM_BACK` are carried through the whole measurement as
controls that must come out neutral.

For a vehicle translating forward, a static world point's mean horizontal image motion in a camera is

```
du  ≈  -f · (v · x_opt) / Z          (the mean over a left/right-symmetric image cancels the expansion term)
```

where `x_opt` is the camera's **image-right axis expressed in ego coordinates**. So the *sign* of the
dominant horizontal drift is fixed by one number per camera — `x_opt · forward` — and nothing else:

* image-right pointing **forward**  → drift **negative** (content sweeps left). A genuinely left-facing camera.
* image-right pointing **backward** → drift **positive** (content sweeps right). A genuinely right-facing camera.

A swap shows up as *both* side channels contradicting their own calibration, in opposite directions.

## 2. Sample

`ego_pose` translations (rotation-free, so unaffected by the yaw problem in §4) give per-keyframe
speed; the relative rotation angle between consecutive `ego_pose`s gives a frame-convention-free
turn rate. Of the 14 952 consecutive keyframe pairs that carry all six cameras and both LiDAR frames:

| gate | pairs |
|---|---|
| all pairs with 6 cameras + LiDAR, 0.1 s < Δt < 0.5 s | 14 952 |
| speed > 4.0 m/s **and** turn rate < 4.0 °/s (moving, near-straight) | 4 565 |
| random sample, ≤ 70 per scene, seed 20260908 | **605** in 9 scenes |

Speeds 4.01 – 11.45 m/s, median 5.40. `chunk_0004` and `chunk_0010` contribute nothing — they are
stationary or turning throughout — so 9 of 11 scenes are represented. Per scene: 70 pairs except
`chunk_0000` (45, all that qualified).

Flow: `cv2.goodFeaturesToTrack` (≤1200 corners, quality 0.01, min distance 10) +
`cv2.calcOpticalFlowPyrLK` (21×21, 4 levels), forward-backward consistency < 1.0 px, ≥ 40 surviving
tracks required. The statistic per keyframe pair is the **median** `du` over surviving tracks — median,
not mean, because Dhaka traffic fills these images with independently moving objects. Median tracks
per pair: 482 (CAM_FRONT), 493 (CAM_BACK), 192 (CAM_LEFT), 299 (CAM_RIGHT), 183 (CAM_FRONT_LEFT),
79 (CAM_FRONT_RIGHT).

## 3. Calibration prediction

From `v1.0-dhaka/calibrated_sensor.json` (the exporter's own, already-optical rotations — see §5),
with `rotation` read as camera-optical → ego and the LiDAR/ego frame's `+x` established as the
direction of travel in §4:

| channel | mount `t` (x, y, z) | optical axis in ego (yaw) | image-right in ego | `x_opt · forward` | predicted drift |
|---|---|---|---|---|---|
| CAM_FRONT       | (+0.813, +0.074, −0.733) | +3.2°   | (+0.055, −0.998, −0.010) | +0.055 | ≈ 0, expansion |
| CAM_FRONT_LEFT  | (+0.772, +0.849, −0.625) | +45.4°  | (+0.712, −0.702, −0.005) | +0.712 | **negative** |
| CAM_FRONT_RIGHT | (+0.789, −0.704, −0.632) | −43.5°  | (−0.692, −0.720, −0.040) | −0.692 | **positive** |
| CAM_LEFT        | (−0.043, **+0.709**, −0.564) | **+92.4°** | (+0.999, +0.037, −0.023) | +0.999 | **negative** |
| CAM_RIGHT       | (+0.064, **−0.701**, −0.527) | **−89.1°** | (−0.998, −0.014, −0.067) | −0.998 | **positive** |
| CAM_BACK        | (−0.865, −0.056, −0.621) | −179.3° | (−0.009, +0.999, −0.035) | −0.009 | ≈ 0, contraction |

The ring is internally consistent and the mount translations agree with the names: the calibration
filed as `CAM_LEFT` sits at ego y = **+0.709 m** (left) and looks +92.4° (left); `CAM_RIGHT` sits at
y = **−0.701 m** and looks −89.1°. Nothing is wrong with the calibration table — the question is only
which image files it is paired with.

## 4. Direction of travel: measured from the LiDAR, not from `ego_pose`

**`ego_pose.rotation` cannot be used here.** The heading it implies disagrees with the direction the
translations actually move by a roughly constant ~149° with a long tail (circular mean offset 160.9°,
concentration R = 0.92 over 5 130 fast pairs); no quaternion ordering or handedness fixes it. Whatever
that yaw is referenced to, it is not the frame the extrinsics live in. Positions are fine (the whole
capture is PPK **FLOAT**, `pct_fixed: 0.0`, σ ≈ 5–14 cm) and are used for speed only.

Instead the forward direction was measured **inside the extrinsics' own frame**. `LIDAR_TOP`'s
`calibrated_sensor` is identity rotation / zero translation, so LiDAR coordinates *are* ego
coordinates. Scan-to-scan translation between consecutive keyframe clouds, by brute-force NCC over a
non-ground BEV occupancy grid (30 m radius, 0.25 m cells, ±3 m search, sub-cell parabolic peak),
over 202 of the selected pairs:

* travel azimuth in the LiDAR/ego frame: **circular mean +1.44°, R = 0.9965**; 99.5 % of pairs within ±20° of `+x`; percentiles (1, 25, 50, 75, 99 %) = −10.2°, 0.0°, +1.5°, +2.8°, +11.1°.
* median NCC peak 0.905.

**Ego `+x` is forward.** This is independent of `ego_pose`, of GNSS and of every camera label.
(The NCC magnitude runs ~0.66× the `ego_pose` speed — clipped search window and ground self-similarity
— but only the direction is used.)

## 5. The extrinsic convention: `v1.0-dhaka` is right, `v1.0-dhaka-fixed` is wrong

This surfaced while deriving the prediction and is reported because it affects the imminent run.

**Arithmetic.** For all six cameras, to 1e-16, the exporter's `rotation` equals
`rotation_body ⊗ q_body→optical`; and `v1.0-dhaka-fixed`'s `rotation` equals
`rotation ⊗ q_body→optical`. The exporter supplied both conventions — `rotation_body` **and** the
already-converted `rotation` — and the fixup converted the converted one.

**Consequence.** In `v1.0-dhaka-fixed` the whole ring is rotated 90°: `CAM_FRONT` points at yaw
−86.8° (out of the right-hand side of the vehicle), `CAM_BACK` at +90.5°, `CAM_LEFT` at +2.1°,
`CAM_RIGHT` at −179.2°.

**Measurement.** Projecting the fused cloud into each image and scoring agreement between the
projected depth-gradient map and the image gradient map (Pearson r on a 64×36 grid, cells with ≥3
points), over 202 keyframes:

| camera | `v1.0-dhaka` extrinsic | `v1.0-dhaka-fixed` extrinsic | raw wins on |
|---|---|---|---|
| CAM_FRONT | **median r = 0.245** (193 302 pts in frame) | −0.025 (3 190 pts) | 98 % of keyframes |
| CAM_BACK  | **median r = 0.168** (175 182 pts) | −0.036 (3 165 pts) | 95 % of keyframes |

`docs/evidence/2026-09-08-fig3-lidar-extrinsics.jpg` shows the same thing at a glance: with the
`v1.0-dhaka` extrinsic the projected cloud paints a dense, correctly registered depth image of the
scene ahead (the silver car, the tree trunks, the kerb); with the `v1.0-dhaka-fixed` extrinsic only
the sparse 360° rings land in frame and they register with nothing.

**Therefore §3 and everything below use the `v1.0-dhaka` rotations.** They are the correct ones.

## 6. Measured drift

605 pairs; `n` below is the pairs on which that channel yielded ≥ 40 consistent tracks.

| channel | prediction | n | median du (px) | 10 % | 25 % | 75 % | 90 % | agrees with own calibration | binomial p |
|---|---|---|---|---|---|---|---|---|---|
| CAM_FRONT       | ≈ 0 | 598 | **−0.25** | −5.56 | −1.97 | +3.21 | +8.03 | control — median \|du\| 2.49 px | — |
| CAM_BACK        | ≈ 0 | 590 | **+0.11** | −4.93 | −1.50 | +2.26 | +5.20 | control — median \|du\| 1.93 px | — |
| CAM_FRONT_LEFT  | negative | 427 | **−40.00** | −93.53 | −68.26 | −0.12 | +44.74 | **77.0 %** ✔ | 3e-30 |
| CAM_FRONT_RIGHT | positive | 236 | **+26.79** | −18.35 | +0.00 | +127.42 | +176.98 | **75.8 %** ✔ | 7e-16 |
| CAM_LEFT        | negative | 520 | **+44.62** | +7.12 | +25.69 | +64.73 | +92.78 | **7.7 %** ✘ | 7e-97 |
| CAM_RIGHT       | positive | 531 | **−45.25** | −94.58 | −70.59 | −20.28 | +11.63 | **14.5 %** ✘ | 5e-66 |

The two controls also settle the front/back identity independently of the horizontal axis: the flow
field's divergence about its own epipole is **+7.39 px (expansion)** for `CAM_FRONT` and
**−6.84 px (contraction)** for `CAM_BACK`, consistent on 92.6 % / 93.7 % of pairs. A forward-facing
camera expands, a rear-facing one contracts. Both are what their labels claim.

**Per-scene sign of the median du** — no scene dissents on the side pair:

| channel | 0000 | 0001 | 0002 | 0003 | 0005 | 0006 | 0007 | 0008 | 0009 |
|---|---|---|---|---|---|---|---|---|---|
| CAM_FRONT_LEFT  | − | **+** | − | − | − | − | − | − | − |
| CAM_FRONT_RIGHT | + | + | + | + | + | + | + | + | + |
| **CAM_LEFT**    | + | + | + | + | + | + | + | + | + |
| **CAM_RIGHT**   | − | − | − | − | − | − | − | − | − |

Per-scene medians for the side pair range +7.8 … +80.4 px (`CAM_LEFT`) and −10.5 … −78.7 px
(`CAM_RIGHT`) — same magnitude, opposite signs, every scene, exactly the mirrored pair one expects.

`docs/evidence/2026-09-08-fig1-ring-flow.jpg` renders one such keyframe pair with the tracks drawn on;
`docs/evidence/2026-09-08-fig2-flow-distribution.png` gives the distributions.

The 2026-09-04 capture measured `CAM_LEFT` +68.6 px and `CAM_RIGHT` −80.6 px. This capture measures
**`CAM_LEFT` +44.6 px and `CAM_RIGHT` −45.3 px** — the same defect, the same sign, on a new capture.

## 7. Cross-checks

### 7a. LiDAR — what it can and cannot settle here

The fused cloud is **not usefully 360°**. Azimuth density in a representative keyframe: ~25 000–35 000
points per 10° in two dense lobes at `+x` (−30°…+40°) and `−x` (140°…−140°), and ~**900** points per
10° everywhere else, from a sparse 2-ring unit. Both side cameras look straight into that gap: the
`CAM_LEFT` extrinsic puts 4 392 points in frame and `CAM_RIGHT` 3 498, against 193 302 for `CAM_FRONT`.

So the LiDAR **cannot** adjudicate the side pair directly, and the honest report of the attempt is
that it does not: the depth-gradient/image-gradient agreement is r ≈ −0.03…+0.02 for every hypothesis
on `CAM_LEFT`, `CAM_RIGHT`, `CAM_FRONT_LEFT` and `CAM_FRONT_RIGHT`, i.e. noise. Their views are
dominated by a smooth near ground plane under heavy motion blur, which has almost no depth-edge
structure to correlate, and a mirrored ground-plane depth ramp looks much like the original anyway.

What the LiDAR **does** settle, decisively, is everything the side-camera argument rests on:
the direction of travel in the extrinsics' own frame (§4), the extrinsic convention (§5), and the
identity of `CAM_FRONT` and `CAM_BACK` (§5 table, and the controls in §6).

### 7b. Field-of-view overlap between ring cameras (calibration-free)

Adjacent cameras on a ring share a field of view; opposite ones do not. SIFT + Lowe ratio test +
MAGSAC fundamental-matrix inliers between two streams of the **same** keyframe, over 152 keyframes.
This reads no extrinsic at all, so it cannot inherit any calibration error:

| stream pair | mean inliers | frames with ≥ 25 inliers |
|---|---|---|
| CAM_FRONT_LEFT ↔ **CAM_RIGHT**  | **10.1** | **9 %** |
| CAM_FRONT_LEFT ↔ CAM_LEFT       | 2.4  | 0 % |
| CAM_FRONT_RIGHT ↔ **CAM_LEFT**  | **2.6**  | **1 %** |
| CAM_FRONT_RIGHT ↔ CAM_RIGHT     | 0.0  | 0 % |

The stream filed as `CAM_RIGHT` is the one that shares content with `CAM_FRONT_LEFT`, and the stream
filed as `CAM_LEFT` is the one that shares content with `CAM_FRONT_RIGHT`. Both point to the swap.
Where matches survive they land where the swap predicts too: at u ≈ 258 in `CAM_FRONT_LEFT` (its
left edge, the most left-facing part of its view) against u ≈ 1018 in the `CAM_RIGHT` file (its right
edge, the most forward part of a left-facing camera's view).

The absolute counts are low — the side/front-side baseline is ~0.8 m, the relative rotation ~47°, and
the side images are heavily motion-blurred — so this is corroboration, not proof on its own.

A variant that warps the side image into the front-side camera's frame through the infinite homography
and scores peak masked NCC on gradient images (±160 px search, 202 keyframes) is **split**: with
`CAM_FRONT_RIGHT` as reference it favours the swap strongly and distinctively (median NCC **0.402**
for the `CAM_LEFT` file vs 0.257 for the `CAM_RIGHT` file, swap wins on 80.2 % of keyframes), but with
`CAM_FRONT_LEFT` as reference it mildly favours as-filed (0.272 vs 0.218, swap wins on 42.1 %). The
non-swap side of that split sits at the ~0.22–0.27 level that generic blurred road texture reaches
under a ±160 px search, whereas the swap side of it stands out at 0.40; the metric is weak and is
reported as such rather than pruned.

### 7c. Roadside-vegetation co-variation (calibration-free, whole capture)

The strongest available substitute for the previous capture's kerb-side instance counts, and the
second decisive signal. A side camera and the front-side camera **on the same side of the vehicle**
watch the same kerb or median go past; their roadside-vegetation content must rise and fall together
over the drive. Score per image = fraction of lower-two-thirds pixels with excess green
(`2G − R − B > 18`), on every 4th keyframe of all 11 scenes — **3 742 keyframes**, including the
stationary and turning ones the flow measurement had to discard:

|  | CAM_FRONT_LEFT | CAM_FRONT_RIGHT |
|---|---|---|
| **CAM_LEFT**  | +0.208 | **+0.687** |
| **CAM_RIGHT** | **+0.629** | +0.077 |

The stream filed `CAM_LEFT` tracks `CAM_FRONT_RIGHT`; the stream filed `CAM_RIGHT` tracks
`CAM_FRONT_LEFT`. Each cross-pairing beats its own-name pairing by 0.42–0.55 of correlation.
Controls behave: the two genuinely opposite pairs are low — `CAM_FRONT_LEFT`↔`CAM_FRONT_RIGHT` +0.245,
`CAM_LEFT`↔`CAM_RIGHT` +0.165 — so the correlation is not a global illumination or scenery effect.
Mean vegetation fraction is near-identical between the side streams (0.136 vs 0.133), so this is
co-variation in time, not a static brightness offset.

Strictly this measures *pairing*, not absolute sides: it says the side streams are cross-paired with
the front-side streams. Anchoring it needs one of the other results — §6 and §7d both establish that
`CAM_FRONT_LEFT` / `CAM_FRONT_RIGHT` carry the names they should (§8), and §7d reads the median and
kerb off `CAM_FRONT` directly — after which the side streams sit on the opposite sides from their
names. The measurement itself is independent of every extrinsic, of `ego_pose`, and of the flow
measurement, and covers 25× more keyframes than §6.

### 7d. Scene content

The previous capture used kerb-side annotation counts. That is unavailable here —
`sample_annotation.json` is empty (0 rows); this capture is unlabelled. What is available is direct
inspection, and it is unambiguous (see the panels of
`docs/evidence/2026-09-08-fig1-ring-flow.jpg`, and §7c quantifies it). Bangladesh is left-hand traffic, so on a divided
carriageway the grassy central reservation is on the vehicle's **right** and the kerb, footpath and
pedestrians are on its **left**. `CAM_FRONT` and `CAM_BACK`, whose identities are settled in §5–6,
both show the median on the vehicle's right. And then:

* the stream filed **`CAM_LEFT`** shows the grass median in the near field — and so does `CAM_FRONT_RIGHT`;
* the stream filed **`CAM_RIGHT`** shows the road, kerb, footpath and pedestrians — and so does `CAM_FRONT_LEFT`.

## 8. CAM_FRONT_LEFT / CAM_FRONT_RIGHT

Asked for explicitly, and the answer is that they are **fine**, more clearly than the previous
capture managed. Both agree with their own calibration on ~76 % of pairs (p ≈ 3e-30 and 7e-16), the
sign of the per-scene median holds in 9/9 scenes for `CAM_FRONT_RIGHT` and 8/9 for `CAM_FRONT_LEFT`
(`chunk_0001` at +3.6 px is a near-zero median, not a reversal), and the cross-checks in §7b/§7d put
them on the sides their names claim. **No fixup for this pair.**

Two honest caveats. Their agreement rate is ~76 % rather than the side pair's 86–92 %, because these
cameras point 45° off the direction of travel: their flow fields carry as much forward expansion as
sideways translation, so the median du is a weaker discriminator and moving traffic in the near field
perturbs it more (`CAM_FRONT_RIGHT` yields ≥ 40 tracks on only 236 of 605 pairs — it mostly looks at
featureless blurred grass). And this pair is where §7b's evidence is thinnest in absolute terms. The
conclusion is nevertheless in the same direction from every measurement made, and matches the
previous capture's decision to leave them alone.

## 9. What would overturn this

Nothing short of a physical walk test, and it is not needed: a walk test would confirm which lens is
bolted to which side of the vehicle, but the defect measured here is not in the hardware — it is in
which calibration record the exporter paired each image file with. The measurement that would
overturn the verdict is a demonstration that ego `+x` is *not* the direction of travel (§4 puts that
at R = 0.9965 over 202 LiDAR scan pairs) or that the `v1.0-dhaka` extrinsics are not the correct
ones (§5 puts that at r = 0.245 vs −0.025 on 202 keyframes, 98 % of frames). Both are measured, not
assumed.

## 10. Reproduction

Throwaway analysis scripts (not committed, they are one-shot):
`build_index.py` → `select.py` → `flow.py` → `analyse.py` / `stats.py`;
`lidar_odo2.py` (§4); `project.py` / `project_score.py` (§5, §7a); `overlap.py` / `warp_overlap.py`
(§7b); `greenness.py` (§7c); `figures.py`. Written under the session scratchpad, `opencv-python-headless` 4.11.0,
numpy 1.26.4. No file under the dataroot and no pipeline source file was modified.

### Figures

* `2026-09-08-fig1-ring-flow.jpg` — all six streams at one moving keyframe (chunk_0005, keyframe 675,
  5.3 m/s, straight) with LK tracks drawn at 2× and each panel captioned with the direction its own
  calibration predicts and the direction measured. The two front-side panels and the two controls are
  green; the two side panels are red and each drifts the way the *other* one's calibration predicts.
* `2026-09-08-fig2-flow-distribution.png` — per-channel distribution of the 605 median-du values,
  and the two side channels' histograms against the sides their calibrations predict.
* `2026-09-08-fig3-lidar-extrinsics.jpg` — `LIDAR_TOP` projected into `CAM_FRONT` with the
  `v1.0-dhaka` extrinsic (registers) and with the `v1.0-dhaka-fixed` extrinsic (does not). This is
  finding 3, visually.
