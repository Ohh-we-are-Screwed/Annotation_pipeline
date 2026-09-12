# Approach A — per-mask stereo boxing on the ZED frusta: design

**Date:** 2026-09-12 · **Status:** draft for review · **Substrate:** `dhaka_20260911_141259`, `v1.0-dhaka-fixed2`, `configs/paths_zami_20260911.yaml`, `DHAKASCENES_SUBSTRATE=dhaka6`

## 0. The problem this replaces

The 3D boxes the VESPA-derived chain (Stage 5 lift → 6 cluster → 7 track → 8 inflate)
produces on Dhaka captures are unusable as pre-labels. The user confirmed all four
documented failure modes at once on the day-1 CVAT 3D tasks: boxes stretched along
the viewing ray, boxes sunk into or floating above the road, wrong yaw, and
neighbours merged or objects missing.

The recorded diagnosis (handover 2026-08-30 §7; `schemas.py` `dhaka6` profile notes,
2026-09-06) is that the recipe assumes a dense, single-sensor, accurately calibrated
cloud — a 32-beam Velodyne with ten accumulated sweeps — and here it is fed a
single 40k-point Mid-360 sweep (median 10 LiDAR points per object, 35 % of masked
instances with zero) welded to noisy stereo through tape-measured extrinsics, then
clustered as if it were one clean cloud. Feeding it better points (the 2026-09-05
fused export) did not fix it: the boxes came out 84–93 % elongated along the ray
and 55–65 % of true height. The geometry step, not the inputs, is what changes here.

## 1. Decision

**One SAM mask → one box. Geometry from the ZED's own stereo depth, in the ZED's
own frame. No clustering.** 3D boxes are produced only for masks on `CAM_FRONT`
and `CAM_BACK` (the two ZED 2i units), only inside the range where stereo depth
is trustworthy. Every other channel stays 2D-only. The release states that
coverage explicitly.

Why the ZEDs: they are the only cameras on this rig with a measured LiDAR↔camera
calibration (`rig.json`: `"calibrated": true`, source
`direct_visual_lidar_calibration`; every other camera is a tape measurement,
`calibrated: false`), and they are the only sensors with dense per-pixel depth.

Why per-mask: Stage 4 already decided what is one object. Re-deciding that with
DBSCAN on a noisy cloud is where merging and dropping came from.

## 2. Inputs that already exist (nothing new is exported)

| input | where | facts verified 2026-09-12 |
|---|---|---|
| ZED stereo depth | `ZED_WORLD` channel of the export (`--zed-world-cloud`) | 15,346/15,346 surviving samples have a row; ~115k pts/sample; ring 100 = rear ZED (serial 32957407), ring 101 = front ZED (35084019); intensity = pixel grey; pixel stride 2; `--zed-max-depth 40`; depth mode NEURAL_PLUS |
| its frame | GLOBAL, with identity `ego_pose` and identity `calibrated_sensor` | brought to ego by the inverse of the SAME sample's `LIDAR_TOP` ego_pose; verified: front ZED mean x = +14.8 m, rear −14.5 m after that transform |
| ZED↔keyframe timing | export association | median 13 ms, max 20.8 ms to the LiDAR anchor |
| ZED intrinsics | `calibrated_sensor.camera_intrinsic` for `CAM_FRONT`/`CAM_BACK` | rectified K, fx = fy = 953.16, zero distortion (`calibration.json`) |
| ZED extrinsics | `calibrated_sensor` for `CAM_FRONT`/`CAM_BACK`, optical convention | NID-calibrated (rig.json provenance) |
| masks | Stage 4 `masks/<kf>.npz` (`(N,720,160)` uint8, bit-packed along width, per channel) + `masks.jsonl` candidates | keys: `channel, proposal_index, class_name, score, n_mask_px, kept, suppressed_by, suppression_ioa, angular_footprint, track_id` |
| mask→point ownership | Stage 5 `lift.jsonl` + `points/<kf>.npz` (`instance_id` per painted point, `point_index` into the Stage 1 single-sweep cloud) | Stage 7 already reconstructs member points from exactly this |
| ground plane | Stage 1 `filter_diagnostics.json` → `keyframes[i].ground_reference_plane {a,b,d}` and `sector_planes` | per keyframe, ego frame |
| class priors | `priors_pilot_v0.json` (dims `{w,l,h}{mu,sigma}`, `eps_bev`) | **does not exist on this box** — see §3.3 |

## 3. Shared base changes (land on `main` before the branch)

### 3.1 Stage 1 ingests `ZED_WORLD` into the single sweep

`pipeline/stage1_ingestion/ingest.py`, the `fuse_stereo` block (≈ line 1322). Today
it looks for channels `ZED_FRONT`/`ZED_BACK` (rings 10/11), which no export has ever
carried, so it is dead code. Change:

- `STEREO_CHANNELS` becomes a mapping of channel → frame handling:
  `{"ZED_WORLD": "global_identity"}` alongside the existing sensor-frame entries.
- For a `global_identity` channel: read the blob, apply
  `p_ego = R_ego^T · (p_global − t_ego)` using the sample's **LIDAR_TOP** ego_pose
  (never the channel's own, which is identity by construction), keep the file's
  ring column as-is (100/101), keep intensity as-is.
- `stereo_stride` becomes an `IngestConfig` field with a CLI flag `--stereo-stride`
  (default = the profile's 8 for backward compatibility; **this branch runs with 1**).
  The 09-06 stride-8 decision existed only to keep Stage 6's DBSCAN alive; A does
  not cluster.
- A sample with no `ZED_WORLD` row is not an error: `n_stereo_pts = {}` for it and
  the keyframe proceeds LiDAR-only (24 such samples existed before the fixups; 0
  after, but the code must not assume that).
- `single_sweep_sources` in the manifest reports `n_lidar`, `n_stereo` per ring,
  and the frame handling used.

The ground-plane fit is unchanged: `ground_fit_rings (0,1,2,3,101)` already
excludes the rear ZED from voting.

### 3.2 `coverage_config: R3` — the two ZED frusta

`pipeline/common/eval_region.py`: `COVERAGE_CONFIGS` gains `"R3"`. Its admitted
azimuth set is the union of two wedges, front `[−h, +h]` and rear `[π−h, π+h]`
wrapping, with `h` = half the ZED horizontal FOV = 33.87° (from
`calibration.json` `h_fov_deg` 67.75). Expressed through the existing machinery as
R2 minus two blind wedges `(+h, π−h)` and `(π+h, 2π−h)`, so `azimuth_measure_rad`,
`in_region` and the density denominators need no new code paths — only a new
`R3_DEFAULT` and the `COVERAGE_CONFIGS` entry. `r_max_m` for R3 = `stereo_range_cap_m`
(§3.4).

Plumbing: `IngestConfig.coverage_config` accepts `"R3"`; `lift.region_for` maps
`"R3"` → `R3_DEFAULT`; Stage 9 records it. `RING_CAMERAS` is untouched — R3 is a
statement about where 3D boxes are claimed, not about which images exist.

### 3.3 Priors authored from the documented means

`scripts/author_priors_dhaka.py` gains `--from-table`: builds the file with no
template, taking `{w, l, h}` for the nuScenes-derived phrases from the table in
`docs/Annotation_pipeline.md:141` (car 1.93×4.63×1.56, truck 2.51×6.93×2.84, bus
2.96×11.19×3.44, pedestrian 0.77×0.76×1.72, bicycle 0.60×1.76×1.59, motorcycle
0.77×2.11×1.47), and for the two indigenous classes the OPERATOR's values, which
replace the script's literature dims (2.70 / 2.65 m): **`a rickshaw` l = 2.40 m,
`an auto rickshaw` l = 2.40 m** (stated by the operator 2026-09-12; w and h stay
at the script's 1.15×1.75 and 1.30×1.75 until measured), source string
`operator_stated_2026-09-12_not_measured_on_this_data`; `sigma = SIGMA_FRACTION ×
mu` (the script's existing constant), `eps_bev = 0.6 × hypot(w, l)`, and source
string `nuscenes_population_mean_LITERATURE_not_measured` on every nuScenes-derived
class. The `LITERATURE` dict in the script is edited to these values so the two
sources cannot drift apart.
`derived_from.scene_subset` is set to `"priors"` **with a `subset_note` stating no
scene was used** — the field is a P1-5 leakage guard against tuning on scored
scenes, and literature values cannot have been tuned on any scene. The four
unreachable C25 phrases and `a trailer` refuse as before.

### 3.4 Calibration spike → `stereo_range_cap_m`

A script `scripts/spike_stereo_vs_lidar.py`: over ≥ 100 keyframes of `chunk_0010`,
for every ZED point (ring 100/101) within 0.5 m (BEV) and 0.3 m (z) of a LiDAR
point after both are in the ego frame, record `(range, dz, d_range)`. Report per
1 m range bin: count, median and MAD of `d_range` and `dz`, separately for ring
100 and 101. Output `docs/evidence/2026-09-12-stereo-vs-lidar-<chunk>.md` and a
JSON.

Decision rule, written into the evidence doc: `stereo_range_cap_m` = the largest
range bin where median `|d_range|` ≤ 0.5 m AND MAD ≤ 1.0 m for **both** rings;
default 25.0 if the spike cannot be run. A systematic `dz` offset > 0.2 m on either
ring is reported as a calibration defect and applied as a per-ring z correction in
Stage 1's `global_identity` path **only if** it is constant across range bins
(median slope < 0.01 m/m); otherwise it is reported and NOT corrected. The
2026-09-12 probe already saw a road-height proxy of −4.13 m against ~−2.4 m
expected, and the 09-06 notes record the rear ZED 0.69 m low: this spike decides
whether that is an offset or range-dependent noise.

**Per-ring rigid pitch correction** *(added 2026-09-12 after final review,
controller ruling R10: this section originally described only the z
correction above; the branch also ships a second, RIGID knob for the case a
translation cannot fix — a rotation)*. When the spike finds a floor that is
misaligned AND range-dependent (a z offset is constant, so it cannot flatten
a sloped floor), it additionally fits a per-ring pitch angle. Stage 1 gains
`--stereo-pitch-correction RING:DEG:PIVOT_X_M:PIVOT_Z_M` (repeatable;
`ingest.py`'s `parse_pitch_corrections`), an `IngestConfig.stereo_pitch_correction`
field (`{ring: {deg, pivot_x_m, pivot_z_m}}`, same ring-keyed shape as
`stereo_z_correction_m`), and a matching config/provenance key. Applied in
`stereo_block_to_ego`, in ego frame, **after** the z correction: `deg` degrees,
right-handed, about the axis parallel to ego **+y** through `(pivot_x, ·,
pivot_z)` — the pivot is the camera's own optical centre in the ego frame
(from `calibrated_sensor.json`, corroborated against the rig's own extrinsic
file), **not** derived from the floor line, because camera height is
unobservable from ground points alone; y is invariant under the rotation, so
the pivot needs no y. An unnamed ring is untouched and `{}` is a no-op, like
`stereo_z_correction_m`. Covered by `tests/test_stage1_zed_world.py`
(the CLI parse, the config round-trip, the rotation itself, ordering after
the z offset, and the untouched-ring no-op).

The spike measures a *candidate* `deg` per ring, not a single trusted one:
on this export, ring 101 (CAM_FRONT)'s floor-flattening angle measured over
the window used to fit the ground plane (3–15 m) is materially different
from the angle measured over the whole span the ring reaches (§3.4
evidence doc, `docs/evidence/2026-09-12-stereo-vs-lidar-chunk_0010.md`:
roughly −12° vs −7° depending on the fitting window) — window-dependent,
which a single RIGID rotation cannot produce, so it is evidence of a
non-rigid or otherwise mismodelled defect, not a clean pitch. A per-block
rigidity check (12 blocks of 10 keyframes) additionally failed its own
spread bar even at the one candidate window whose angle numerically passed
the floor test. Decision rule: write a correction only when a single angle
is both accurate AND stable across blocks; ring 101 fails that bar, so
`stereo_pitch_correction = {}` on this export and CAM_FRONT is instead
dropped from Branch A's `active_channels` (§4) rather than corrected — the
front frustum's boxes are not produced, not silently mispitched. Ring 100
(CAM_BACK) is aligned with the LiDAR road to begin with and needs no pitch
correction. `stereo_pitch_correction` is a Stage 1 ingestion knob: Branch A
(`stereo_box.py`) records it in its manifest and never re-applies it.

## 4. Branch A — `pipeline/stage6_stereo_box/stereo_box.py`

Drop-in for Stage 6 on the ZED channels. Reads what Stage 6 reads; writes what
Stage 6 writes. Stage 7, 8, 9, release and CVAT-3D run unchanged.

### 4.1 Inputs per keyframe

- Stage 1 single-sweep cloud (ego frame, rings 0–3 LiDAR, 100/101 ZED) + ground
  reference plane `{a, b, d}`.
- Stage 5 `points/<kf>.npz` → for each instance the member point rows; from the
  cloud's ring column split them into `stereo_pts` (ring 100/101) and `lidar_pts`
  (0–3).
- Stage 5 `lift.jsonl` instance metadata (`instance_id, channel, class_name,
  proposal_index, score, n_mask_px`).
- Stage 4 `masks/<kf>.npz` for the instance's mask (needed for the footprint
  principal axis and for `n_mask_px` consistency).
- `calibrated_sensor` for the instance's channel (K, T_cam_ego), priors file.

Only instances whose `channel ∈ {CAM_FRONT, CAM_BACK}` are boxed. Instances on the
other four channels are written with `status: "out_of_r3"` and `box: null` so the
row count matches Stage 5 exactly (Stage 8 skips non-`fit` rows; Stage 9 counts
them).

### 4.2 Geometry, in order

1. **Robust depth.** Let `d_i` be the range of each stereo point from the ZED's
   optical centre (not the ego origin). `d_med = median(d_i)`,
   `mad = median(|d_i − d_med|)`. Keep `|d_i − d_med| ≤ k · mad`, `k = 3.0`, with a
   floor `mad ≥ 0.10 m` so a perfectly flat set is not trimmed to nothing. If fewer
   than `min_stereo_pts = 20` survive → `status: "too_few_stereo"`, `box: null`.
2. **LiDAR refinement.** If ≥ 5 `lidar_pts` fall inside `[d_near − 2·mad, d_near + 2·mad]`
   (same depth definition; `d_near` from step 7), `d_near` is replaced by the median
   LiDAR depth of those points and the row records `depth_source: "lidar_refined"`;
   otherwise `"stereo"`. LiDAR range is trusted over stereo whenever it exists on the
   object.
3. **Lateral and vertical extent — measured.** Project the surviving points into
   the ZED image with K. `w_meas = (u_p99 − u_p01) · d_near / fx` and
   `h_meas = (v_p99 − v_p01) · d_near / fy`, i.e. the 1st–99th percentile pixel
   spread converted to metres at the robust NEAR-FACE depth (`d_near`, step 7 —
   not `d_med`; corrected here to match the implementation). Stereo measures
   lateral and vertical extent well; it is depth extent it measures badly.
   *(revised 2026-09-12 during implementation, controller ruling R20: on
   chunk_0010, 4,247 boxes, rear ZED, `w_meas`/`h_meas` were systematically LOW
   versus the class prior independent of range — median w_meas/mu pedestrian
   0.49, rickshaw 0.73, auto rickshaw 0.87, car 0.70, motorcycle 0.60 (h/mu
   0.49 / 0.70 / 0.66 / 0.58 / 0.35). An A/B loosening the MAD trim (k_mad 6,
   mad_floor 0.5; kept/pts 0.87 → 0.99) moved pedestrian w/mu only 0.49 → 0.52,
   ruling out the trim; p1/p99 instead of p5/p95 moved it to 0.58 (rickshaw
   0.87, rickshaw h 0.90) and raised unclamped boxes from 90 to 406 of 4,247
   (double-clamped 3,543 → 2,653). The stereo points a mask owns do not reach
   the object's silhouette by construction, so p1/p99 recovers more of the
   true spread than p5/p95 did.)*
4. **Length — from the prior.** `l = prior.l.mu`. Never from the points.
5. **Width and height — measured, clamped ASYMMETRICALLY to the prior.**
   `w = mu_w` if `w_meas < mu_w − kσ_w` (rule `low_to_mu`); `w = mu_w + kσ_w` if
   `w_meas > mu_w + kσ_w` (rule `high`); otherwise `w = w_meas` (no clamp).
   Same for `h`, `k = prior_clamp_sigma = 2.0`. The row records `w_meas`,
   `h_meas`, whether each was clamped (`clamped_axes`, the existing Stage 6
   field) and the direction of each clamp (`stereo.clamp = {w, h}`, each one of
   `null | "low_to_mu" | "high"`).
   *(revised 2026-09-12 during implementation, controller ruling R20: a
   measurement below `mu − kσ` can only be explained by occlusion or a stereo
   hole at a depth edge, which SHRINKS the points a mask owns — never grows
   them — so it is evidence of a bad measurement, not a small object, and is
   clamped to the prior MEAN rather than the prior floor. Under the old
   symmetric ±2σ clamp, 83% of chunk_0010's 4,247 boxes were pinned at the
   floor, e.g. pedestrians at 0.62 × 1.38 m — too small. A measurement above
   `mu + kσ` remains clamped to `mu + kσ`: mask bleed at a depth edge can only
   grow an extent, and that growth is bounded.)*
6. **Yaw.** Project the surviving points to the ground plane (drop the component
   along the plane normal). The 2×2 covariance of that footprint supplies the
   **isotropy test only**: if the eigenvalue ratio `λ1/λ2 < 1.5` (near-round:
   pedestrians, head-on vehicles) the heading is set **along the viewing ray** and
   `yaw_ambiguous = true` with reason `"footprint_isotropic"`. Otherwise the angle
   comes from a **closeness rectangle fit** of that footprint
   (`stage6_cluster.fit_rectangle`, Zhang et al. 2017's L-shape criterion, reused
   not reimplemented), and the heading axis is the longer of the fitted
   rectangle's two extents; `yaw_ambiguous = true` with reason `"axis_only"` (the
   180° half is never decided here, matching Stage 6's `yaw_axis_only: true`).
   Stage 7's track direction resolves it downstream as it does today. The row
   records which path ran in `box.fit.yaw_source`.

   **Which object axis the longer extent is depends on how many faces are
   visible.** Let `e_major`/`e_minor` be the fitted rectangle's larger/smaller
   extent and `theta_major` the bearing of `e_major`. If
   `e_minor < single_face_minor_frac (0.35) · min(mu_w, mu_l)` only ONE face is
   visible — the strip IS a face — and its width says which one: with
   `d_w = |ln(e_major/mu_w)|` and `d_l = |ln(e_major/mu_l)|`, `d_w <= d_l` means
   the front/rear face, whose **length axis is perpendicular** to it
   (`yaw = theta_major + 90°`), otherwise the side face and `yaw = theta_major`.
   `yaw_source = "single_face_prior_match"`, and the `stereo` block records
   `single_face: {e_major_m, e_minor_m, matched: "w"|"l"}`. A class whose `w` and
   `l` are within 15% of each other (pedestrian, 0.77 / 0.76) cannot be
   disambiguated by width: it records `matched: "ambiguous"` and falls through to
   the two-face rule, which is harmless because that box is near-square. With two
   faces visible (an L) the rule is unchanged and `yaw_source =
   "l_shape_closeness"`. The isotropy test above still runs first.

   **Truncation guard.** The width match above is only valid when the whole face
   is in frame. If the instance's Stage 4 mask touches the left or right image
   border (column extent within `truncation_margin_px` (4) of 0 or `W-1`) the
   visible extent is a LOWER bound, so the single-face branch does NOT width-match:
   `yaw = 0` (ego forward — traffic runs along the road, and yaw is modulo π so the
   same value serves CAM_BACK), `yaw_source = "truncated_ego_forward"`, reason
   `"frame_truncated"` appended, counted in `n_truncated_yaw`. Every row that
   reaches this step records `stereo.frame_truncated` and
   `stereo.truncation_source` (`"mask"`, or `"points"` when the mask npz cannot be
   read and the owned points' projected u-range answers instead). The L-shape and
   isotropic branches are UNCHANGED when truncated: two visible faces still measure
   a real angle, and an isotropic footprint never reached the width match anyway.
   *(revised 2026-09-12 after the frame-truncated car case: keyframe
   895d7483e5e665fdc5d3108a2863d49d, CAM_FRONT, "a car", proposal_index 0 of
   dhaka_20260911_141259_chunk_0010. Its 2D box runs to the right image edge
   (x 1073–1280) at 3.5 m, so only 1.28 m of the car's SIDE is visible; 1.28 m is
   closer in log-ratio to mu_w (1.93) than to mu_l (4.63), so the rule declared the
   strip a rear face and laid the 4.63 m length axis perpendicular to it — yaw 61.5°
   instead of ≈0°, the box across the lane, its centre 1 m beyond the strip. Counted
   on the pre-fix output of that scene, 925 of 7,701 fitted boxes have a 2D box
   touching the left or right image border and the single-face rule fired on 455 of
   them.)*
   *(revised 2026-09-12 during implementation: PCA answered 52.6° on a 20°
   visible-surface fixture — an L-shaped footprint's centroid lies off BOTH legs
   and the resulting cross-moment rotates the principal axis toward the diagonal.)*
   *(revised 2026-09-12 after the head-on bus case: keyframe 575 of
   dhaka_20260911_141259_chunk_0010 showed a bus seen from directly behind — one
   visible face, fitted at e_major 2.48 m by e_minor 1.23 m, footprint eigenvalue
   ratio 6.0, so the isotropy gate could not catch it (a flat face is strongly
   ANISOTROPIC). "The longer visible extent is the length axis" made that 2.48 m
   strip the length axis, laid the 11.19 m prior ACROSS the road and pushed the
   centre by w/2 = 1.82 m instead of l/2 = 5.60 m. On the synthetic reproduction
   of the same view the face now yaws to within 0.8° of the viewing ray and
   pushes 5.61 m. `single_face_minor_frac` ships at the spec's 0.35; that bar is
   below the 0.415 this particular instance measures, so the rule reaches the
   class of defect (truck reprojection IoU median 0.0 → 0.175 on the same scene)
   but not yet that one box — see the sweep recorded in `configs/stereo_box.yaml`,
   and the range-cap interaction it turns on.)*
7. **Centre.** Stereo sees a *surface*, and for an oblique view (an end face plus
   part of a side) the median depth sits 0.3–0.5 m behind the nearest point, so
   the median is NOT the near face. The near face is `d_near` = the **20th
   percentile** of the surviving (MAD-trimmed) depths; the centre is that point
   pushed further along the ray by the box's **own half-extent in the ray
   direction**: with `theta` the angle between the box's length axis (yaw, step 6)
   and the BEV ray, `push = (l/2)·|cos theta| + (w/2)·|sin theta|` (Stage 8's own
   rule: hold the observed surface, grow away from the sensor). Head-on
   (`theta = 0`) that is `l/2`, the old rule exactly; side-on (`theta = 90°`) it is
   `w/2`. The ray passes through the **midpoint of the same p1/p99 percentile
   window that measured `w` and `h`** (step 3), at depth `d_near`, so the centre
   and the width are one measurement. `d_med`, `d_near`, `push_m` and `theta_deg`
   are all recorded.
   *(revised 2026-09-12 after final review: this line said "5th–95th
   percentile window" after R20 (step 3) moved the measurement window to
   p1/p99; it names the same window step 3 uses, so it is corrected to
   match, not a second decision.)*
   *(revised 2026-09-12 during implementation: the flat `l/2` push put a rickshaw
   seen side-on at x = 12.578 against a truth of 12.0, and the lateral MEDIAN — the
   old ray definition — put it at y = −0.204 against a truth of 1.0, because on an
   L-shaped visible surface the median bearing sits on whichever leg carries more
   points. With both revisions the same fixture lands at (12.171, 0.959).)*
8. **Ground snap.** `z_min` = the ground plane's z at the centre's (x, y);
   `z_max = z_min + h`. The bottom is anchored to the ground, not to the points
   (Stage 1's 0.3 m ground band removes the bottom of everything, so points cannot
   place it).
9. **Range gate.** If the **near face's** BEV range > `stereo_range_cap_m` →
   `status: "beyond_stereo_cap"`, `box: null`. The range tested is recorded as
   `stereo.range_gate_m`.
   *(revised 2026-09-12 after controller ruling R25: this gated the CENTRE's BEV
   range. The cap states where this stage's stereo evidence is trustworthy, and
   that evidence is the visible face — the centre is the face extrapolated along
   the ray by a class prior (step 7), so gating it rejects a box for having a long
   prior rather than for having untrustworthy points. It penalised exactly the
   boxes step 6's single-face rule had just got right: a bus turned to face the
   camera is pushed l/2 = 5.6 m instead of w/2 = 1.5 m, and fitted bus boxes on
   chunk_0010 fell 114 → 44 for no reason but the correct orientation.)*

### 4.3 Output row

Exactly Stage 6's `boxes.jsonl` row so downstream consumers need no change:
top-level `instance_id, channel, proposal_index, class_name, score, n_mask_px,
n_points_instance, eps_m, eps_source, min_samples, cluster_space, canonical_sort,
cluster_tie_break, cloud_kind, frame: "ego", num_lidar_pts,
num_lidar_pts_basis: "single_sweep_ground_filtered_pre_inflation",
n_points_below_gate, status, keyframe_token, t_ns, box` and
`box = {translation_m, size_wlh_m, size_order: "w,l,h", yaw_rad, rotation_wxyz,
yaw_axis_only: true, yaw_ambiguous, yaw_ambiguous_reasons, axis_swapped: false,
clamped_axes, z_min_m, z_max_m, footprint_diagonal_m, aspect_ratio_w_over_l, fit}`,
present only on `fit` rows; every other status writes `box: null`.

**`status`, all eight values** *(added 2026-09-12 after final review — this
enumeration was missing; `channel_disabled`, `no_prior`, `no_points`,
`no_ground_plane` appeared in no spec section before this, `stereo_box.py:115-121`)*:

- `fit` — a box was produced; `box` is populated and `stereo` carries the
  full block (§4.2 steps 1–9 all ran).
- `too_few_stereo` — fewer than `min_stereo_pts` survived the front-of-camera
  filter or the MAD trim (step 1). `box: null`; `stereo` is present (the
  function seeds it before either check) but stops at whichever field the
  exit point reached — `clamp` stays at its `{w: null, h: null}` seed because
  step 5 never runs.
- `beyond_stereo_cap` — the fitted centre's BEV range exceeds
  `stereo_range_cap_m` (step 9). `box: null`, but `stereo` is otherwise fully
  computed — steps 1–7 already ran and `clamp` holds real clamp directions —
  because the range gate is the last check before the ground snap.
- `out_of_r3` — the instance's channel is not `CAM_FRONT`/`CAM_BACK`, so it
  was never a stereo candidate. `box: null`, `stereo: null`.
- `channel_disabled` — the channel is a ZED channel but not in this run's
  `active_channels` (CAM_FRONT is disabled on this export — §3.4's per-ring
  pitch correction section explains why). `box: null`, `stereo: null`.
- `no_prior` — the instance's class has no measured `{w, l, h}` in the priors
  file, so length (prior-mean-only, step 4) cannot be produced. `box: null`,
  `stereo: null`.
- `no_points` — Stage 5 painted zero points to this instance. `box: null`,
  `stereo: null`.
- `no_ground_plane` — Stage 1 recorded no `ground_reference_plane` for this
  keyframe, so the box bottom (step 8) has nothing to snap to. `box: null`,
  `stereo: null`.

Only `fit`, `too_few_stereo` and `beyond_stereo_cap` reach `box_from_stereo`
and therefore carry a `stereo` block at all (`stereo_box.py:475–493`); the
other five statuses are decided beforehand and set `stereo: null` directly
(`tests/test_stereo_box.py:186` asserts `stereo is None` for those rows).
*(revised 2026-09-12 after final review: this section previously said
`{w: null, h: null}` is pre-seeded "on every row, including the non-fit
statuses" — that described only the three statuses above that reach
`box_from_stereo`; the other five never get a `stereo` dict to seed at all.)*

Additive keys (the Stage 3b trick — additive only, never renamed):
`stereo = {n_stereo_pts, n_stereo_kept, d_med_m, d_near_m, mad_m, depth_source, w_meas_m,
h_meas_m, ray_yaw_rad, footprint_eig_ratio, zed_ring, n_lidar_in_box,
n_stereo_in_box, clamp}`, `clamp = {w, h}` each `null | "low_to_mu" | "high"`
(added 2026-09-12, controller ruling R20 — the direction of the step-5 clamp,
per axis), seeded `{w: null, h: null}` when the dict is built and overwritten
by step 5 for `fit` and `beyond_stereo_cap` rows; `too_few_stereo` keeps the
seed, since it exits before step 5 runs (see the status list above). Like
`clamped_axes`, `clamp` is keyed on the PRE-swap axes: both are set before
the step-6 `w > l` swap, so after a swap the `w`/`h` names in `clamp` (and in
`clamped_axes`) refer to the measured-width/measured-height variables as
clamped, not to whichever of `size_wlh_m`'s `w`/`l` they ended up written to.

`num_lidar_pts` counts **the points Stage 5 already painted to this instance
(`rows_of_cloud`, both LiDAR rings 0–3 and stereo rings 100/101) that land
inside the final box** — not every point of the fused single sweep that
falls inside the box's geometry whether painted or not (operator decision
2026-09-12: stereo points are returns for Stage 9's "≥ 5 returns" gate). The
basis label stays `single_sweep_ground_filtered_pre_inflation`, naming the
cloud those points are drawn from, so Stage 9's refusal check is satisfied
truthfully; the manifest additionally carries `box_fit.num_lidar_pts_basis_detail:
"painted_points_inside_box_lidar_plus_stereo"` and a matching `known_gaps`
line, so a reader of the manifest alone sees the painted-only basis without
this spec. The split is recorded, not lost: `stereo.n_lidar_in_box` and
`stereo.n_stereo_in_box` are additive fields on every `fit` row, and the
manifest reports how many boxes have `n_lidar_in_box < 5` (i.e. would have
been tiered "review" under a LiDAR-only gate). `eps_m`/`min_samples` are
recorded from the priors/config for Stage 7's `reconstruct_cluster_points`
call, whose point set A does not use for its own fit.
*(revised 2026-09-12 after final review, controller ruling R22: this section
previously said, in bold, that `num_lidar_pts` "counts every point of the
fused single sweep inside the final box" — the implementation only ever sees
`cloud[rows_of_cloud, :3]`, the Stage 5 painted rows for this instance, which
`box_from_stereo` receives as its `pts_ego`/`rings` arguments and splits into
`is_st`/`is_li` (`stereo_box.py:176-179`; the call site is `stereo_box.py:493-494`),
so `inside` — and therefore `n_lidar_in_box` / `n_stereo_in_box` /
`num_lidar_pts` — is evaluated over that painted subset, never the whole
sweep. Ruling: KEEP the painted-only count. It is conservative for Stage 9's
"≥5 returns" gate (an unpainted stray return could never have inflated a
mask's count in A's favour) and it is what every archived evidence number
already describes; this section is corrected to match the code rather than
the code changed to match the old sentence.)*

### 4.4 Wrapper

`scripts/run_stages.sh` gains step `6s` (opt-in, like `3f`/`3m`): runs
`stereo_box.py`, writes `stage6_stereo_box/`, and `boxes_dir()` prefers it over
`stage6_cluster` when its marker is fresh — the same preference logic Stage 8
already applies between Stage 7 and Stage 6. `6` and `6s` are mutually exclusive in
one run and the wrapper refuses both.

## 5. Viewer — `scripts/view_boxes_3d.py` + `viewer/index.html`

Read-only. Purpose: see A's boxes on the objects, from the cloud and from both ZED
images, per keyframe. Not an annotation tool.

- `view_boxes_3d.py --boxes-dir <stage6_stereo_box> --scene <name> --out <dir>`
  writes one `kf/<index>.json` per keyframe: cloud decimated to ≤ 80k points as
  `[x,y,z,ring]` float32 rows (base64), boxes (from `boxes.jsonl`, `status: fit`
  only, plus the `stereo` block), the two ZED image paths, K and `T_cam_ego` for
  each, the ground plane, and `index.json` listing keyframes; then serves `<out>`
  with `http.server` on a stated port.
- `viewer/index.html`: Three.js (cdnjs, pinned), orbit controls, points coloured by
  ring (LiDAR grey, ring 101 warm, ring 100 cool), boxes as wireframes coloured by
  class with a heading arrow; two image panels with the same boxes projected
  through K·T_cam_ego (client-side, so the projection is checkable against the
  cloud); slider + `←/→` over keyframes; toggle for LiDAR / stereo points; hover
  shows the row's `stereo` block. Ego axes drawn; a 25 m ring shows the range cap.
- No build step, no npm: one HTML file, one script.

## 6. Run scope for the first look

`--scenes dhaka_20260911_141259_chunk_0010` (668 keyframes ≈ half a chunk, no
subsetting code). Chain: `1 3 3f 3m 4 5 6s` then the viewer. Stage 7/8 are not
required to *see* boxes and are skipped for the first look; they run unchanged in
the full chain later.

## 7. Evaluation (no GT exists)

On the same keyframes, A vs the archived unfused baseline and — when built — B:
(a) `yaw_ambiguous` rate, `clamped_axes` rate, fraction of boxes `depth_source:
lidar_refined`, `status` histogram; (b) **reprojection IoU** — each box projected
into its own camera vs the SAM mask that produced it (a strong GT-free signal, and
the viewer computes the projection already); (c) the human look in the viewer.
Written to `docs/evidence/`.

## 8. Testing

Each module ships `tests/` alongside, `pytest -q` green (suite currently 612):
- Stage 1: a synthetic `ZED_WORLD` blob at a known global pose round-trips to known
  ego coordinates; a sample without the row proceeds LiDAR-only.
- `R3`: azimuth measure = 4h; a bearing at 0°, 180° is in; at 90°, −90° is out.
- Priors: `--from-table` output loads through `load_priors`; every phrase in the
  Dhaka taxonomy is present or refused by name.
- Stereo box: a synthetic box of known pose, sampled as stereo points with
  Gaussian range noise and a background plane behind it, recovers centre within
  0.3 m, `w`/`h` within the clamp, yaw axis within 10°, `z_min` on the plane; the
  background plane is rejected by the MAD trim; a 15-point instance returns
  `too_few_stereo`; an object at 30 m with cap 25 returns `beyond_stereo_cap`.
- Viewer: the JSON writer round-trips a boxes row; the projection of a box centre
  through K·T lands inside the image for a known front-of-camera point.

## 9. Out of scope

Branch B (frustum-restricted VESPA); Stage 3b (no sweeps in this export); any 3D
box outside the two ZED frusta; re-exporting; extrinsic re-calibration upstream
of Stage 1's ingestion-time correction knobs — i.e. fixing the export's own
`calibrated_sensor`/rig extrinsics (`docs/evidence/2026-09-12-stereo-vs-lidar-chunk_0010.md`'s
"needs an UPSTREAM FIX: a re-export with corrected front-ZED extrinsics").
*(revised 2026-09-12 after final review, controller ruling R10: this line
previously read "beyond the constant per-ring z correction in §3.4", which
would have put the per-ring RIGID PITCH correction §3.4 also documents
out of scope — it is shipped in-branch, not out of scope; only the
upstream re-calibration/re-export neither knob is a substitute for stays
out of scope.)*
