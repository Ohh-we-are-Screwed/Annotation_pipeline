# DhakaScenes — Comprehensive Project Specification
### Unified sensor, annotation, benchmark, integration, and publication plan

**Status:** v1.0 unified draft — supersedes `SENSOR_SUITE_SPEC.md` (v0.2), `BENCHMARK_METRICS_SPEC.md` (v0.2), and `Annotation_pipeline.md`
**Working title:** *DhakaScenes* (placeholder — finalize before submission; check name collisions on Google Scholar / Papers-with-Code)
**Document rule:** every design decision that a reviewer could attack is stated **as a decision with a rationale**, never as an implicit assumption. Anything marked `[VERIFY]` must be checked against a primary source before the camera-ready; anything marked `[GATE]` is a go/no-go checkpoint with a defined pass criterion.

---

## 0. Publishability contract — flaw → resolution matrix

This project was audited against the failure modes that get dataset papers rejected. Each identified flaw is resolved by a binding design decision in this document. This table is the contract; if a section below contradicts it, the table wins.

| # | Flaw (as identified in review) | Severity | Binding resolution | Section |
|---|---|---|---|---|
| F1 | Auto-labels written directly into the benchmark ground truth (pipeline lineage tops out ~47–53 % mAP → test labels would be ~50 %-quality) | Fatal | **Label-tier policy:** val/test are 100 % human-verified (pipeline output is a pre-label only, every box confirmed or corrected by a human); train ships with audited pseudo-label error rates on a stratified sample | §7.4, §7.5 |
| F2 | 360° LiDAR but one ~110° forward camera → two-thirds of every sweep has no proposals; "360°" claim collapses; density metric ρ undercounts | Fatal | **Camera decision gate G1:** either (R2) hardware-triggered global-shutter camera ring giving ≥ 300° labelled coverage, or (R1) all labels, metrics, and claims formally scoped to a defined **frontal evaluation region**. No third option. ρ is redefined per-region either way | §3.1, §3.6, §8.3.1 |
| F3 | Cross-dataset domain-gap claim doubly confounded (sensor geometry + labels inherited from Western-trained foundation models) | Fatal | Sensor control: common BEV representation for all four cells. Label-provenance control: repeat the Dhaka-trained cell using the human-verified-only subset and report the delta | §8.3.5 |
| F4 | Indigenous classes (CNG, battery-rickshaw, cycle-rickshaw, tempo…) have no DBSCAN ε, no dimension priors, and unproven open-vocabulary separability | Fatal | **Seed set S0:** 3,000–5,000 fully manually annotated frames collected first. S0 bootstraps all per-class priors, calibrates QA thresholds, and runs separability study S1 with a defined pass criterion | §7.2 |
| F5 | Internal contradictions across documents (mic arrays vs. no mics; "zero proprietary" vs. API-served DINO-X Pro; "cameras" plural vs. one) | Credibility | Audio is **out of scope for v1.0** (future-work paragraph only). The model stack is restricted to **locally runnable open-weights checkpoints** (§7.1). Camera count fixed by G1. One spec (this file) is canonical | §1.4, §7.1 |
| F6 | Unverifiable/likely-hallucinated citations ("SAM 3.1 Object Multiplex" — **verified real and adopted 2026-08-19, DECISIONS C26**; Impact Scores, unverified Point-SAM/VESPA numbers — still unverified) | Credibility | Citation verification checklist (§7.1.1): every model, venue, and number is re-derived from the primary source or dropped. Unverifiable components are replaced or demoted to `[EXPERIMENTAL]` | §7.1 |
| F7 | NDS promised but mAAE incomputable (no attribute annotation planned) | Credibility | Minimal attribute vocabulary defined and annotated (§6.2) so genuine nuScenes NDS is computable. No silent metric redefinition | §6.2, §8.2.1 |
| F8 | Positioning gap vs. IDD / IDD-3D ("unstructured South Asian traffic" alone is taken) | Credibility | Explicit comparison table (§1.3); differentiation claims restricted to what is measurably ours: PPK ground truth, night share, density stratification, indigenous-3-wheeler taxonomy at 3D-box level | §1.3 |
| F9 | Dense-gridlock frames may contain visually unresolvable object clusters → annotation impossible or noisy exactly where the dataset claims value | New (self-audit) | **Group/ignore boxes** (CrowdHuman-style) with defined evaluation semantics: ignore regions never generate FP or FN | §6.3 |
| F10 | Single-antenna heading drift in stop-and-go traffic corrupts yaw ground truth in the most valuable frames | Fatal for odometry task | Dual-antenna heading (second F9P in moving-base mode, or VN-300) **required before scaled collection**, plus RTS smoothing and published per-frame yaw σ | §3.2.2, §5 |

---

## 1. Project definition

### 1.1 Paper abstract (target form)

> We present *DhakaScenes*, a multimodal driving dataset collected in Dhaka, Bangladesh — one of the densest and most heterogeneous traffic environments on Earth. The dataset provides synchronized 360° solid-state LiDAR, multi-camera imagery, and a centimetre-class post-processed kinematic (PPK) ground-truth trajectory, with 3D bounding-box and track annotations over a taxonomy that first-classes the indigenous vehicle types (auto-rickshaws, battery rickshaws, cycle rickshaws, tempos, human haulers) that dominate South Asian roads yet are absent from existing 3D benchmarks. Beyond standard detection, tracking, and odometry protocols defined for direct comparability with nuScenes/KITTI/Waymo, we introduce **density-stratified evaluation**: every headline metric is reported across measured traffic-density bins from free-flow to gridlock, converting "our data is harder" into a quantified degradation curve. A four-cell cross-dataset protocol with sensor-representation and label-provenance controls measures the generalization value of training on unstructured-traffic data. All sensor-quality figures — sync residuals, RTK/PPK fix rates, calibration residuals, inter-annotator agreement — are published as measured artifacts of the release.

### 1.2 Contribution claims (each falsifiable, each mapped to evidence)

| ID | Claim | Evidence artifact (paper) | Kill criterion |
|---|---|---|---|
| C1 | First 3D-box/track benchmark that first-classes indigenous South Asian 3-wheeler categories with per-class metrics | Taxonomy §6; per-class AP table; confusion matrix; instance counts | Fewer than ~1k instances per headline indigenous class in the release → claim weakens to "initial benchmark" |
| C2 | Density-stratified evaluation reveals systematic performance degradation invisible to whole-set averages | Degradation curves Δ(metric) across ρ-bins for ≥ 2 detectors + 1 tracker | **[GATE G5]** If the pilot curve is flat (Δ mAP < 3 pp Low→Extreme), C2 is dropped and the paper pivots (§11.6) |
| C3 | Centimetre-class PPK ground truth in a dense urban canyon, with honest per-frame uncertainty — enabling odometry/SLAM benchmarking where most datasets cannot | %-fixed statistics, σ distributions, yaw-σ moving vs. stationary | If fixed-ambiguity rate < ~40 % of frames, odometry becomes a secondary task with flagged subsets |
| C4 | Cross-dataset protocol shows data collected here transfers out better than standard data transfers in | 4-cell table + both control ablations (§8.3.5) | If controls show the gap is explained by sensor geometry or label provenance, report honestly; claim becomes descriptive not causal |
| C5 | Measured-not-asserted release quality (sync, calibration, labels) | §8.3.4 sensor-quality artifact; IAA; audited pseudo-label error | None — always publishable |

The paper survives on C1 + C2 + C5. C3 and C4 are upside.

### 1.3 Positioning against prior datasets

The comparison table the reviewer will look for. `[VERIFY]` every cell against the primary paper before submission — do not trust memory or secondary sources.

| Dataset | Region / traffic type | 3D boxes | LiDAR | GT pose quality | Night share | Indigenous 3-wheelers as classes | Density stratification |
|---|---|---|---|---|---|---|---|
| nuScenes | Boston/Singapore, structured | ✓ 1.4 M | 32-beam spinning | GNSS/INS (m-class) | ~11 % `[VERIFY]` | ✗ | ✗ |
| Waymo Open | US, structured | ✓ 12 M | proprietary spinning | GNSS/INS | minority `[VERIFY]` | ✗ | ✗ |
| KITTI | Karlsruhe, structured | ✓ | 64-beam | GNSS/INS-RTK | ✗ | ✗ | ✗ |
| Argoverse 2 | US, structured | ✓ | 2×32-beam | GNSS/INS | some | ✗ | ✗ |
| ZOD | Europe, structured | ✓ | 128-beam | RTK-class `[VERIFY]` | some | ✗ | ✗ |
| IDD (camera) | India, unstructured | ✗ (2D/seg) | ✗ | — | ✗ `[VERIFY]` | ✓ (2D only) | ✗ |
| IDD-3D `[VERIFY all]` | India, unstructured | ✓ | 64-beam class | standard GNSS/INS `[VERIFY]` | `[VERIFY]` | partial `[VERIFY]` | ✗ |
| **DhakaScenes (this work)** | Dhaka, extreme-density unstructured | ✓ | 360°×59° non-repetitive solid-state | **PPK + RTS, cm-class, per-frame σ published** | **target ≥ 25 %** | **✓ (3D boxes + tracks + attributes)** | **✓ (headline protocol)** |

**Honest differentiator statement (use verbatim in the paper):** *"Unstructured South Asian traffic is not by itself novel — IDD established it in the camera domain and IDD-3D in LiDAR. Our contribution is the combination of (i) survey-grade PPK ground truth with published per-frame uncertainty, (ii) substantial night and monsoon coverage, (iii) an indigenous-vehicle 3D taxonomy with attributes, and (iv) density-stratified evaluation as a first-class protocol."*

### 1.4 Scope — what v1.0 is and is not

**In scope:** 3D detection, 3D MOT, odometry/SLAM/localization, depth (vs. LiDAR), density/illumination-stratified evaluation, cross-dataset protocol, interaction mining (`critical` subset).

**Explicitly out of scope for v1.0** (each gets one sentence in the paper's limitations, not silence):

1. **Audio.** No microphones in the rig. All references to "spatialized audio" / MAVD-style soundscapes are removed from every document. Future work only. *(Resolves F5.)*
2. **HD maps.** None exist for Dhaka at usable quality; the forecasting benchmark (if included later) is explicitly map-free.
3. **Motion forecasting, segmentation, occupancy.** Deferred to v1.x unless annotation budget materializes; a thin benchmark is worse than none.
4. **RADAR.** Not in the rig.
5. **Camera-only 3D detection leaderboard at 360°** under config R1 (frontal camera): only frontal camera-3D is defined.

### 1.5 Naming, branding, artifacts

- Dataset name finalized after a collision check; secure a domain / GitHub org early (leaderboard credibility).
- Public artifacts at release: dataset (blurred), devkit (fork of nuscenes-devkit), evaluation code with versioned metric definitions, annotation guideline PDF, Datasheet for Datasets, priors JSON, calibration + quality metadata, paper + supplementary.

---

## 2. System architecture and integration

### 2.1 End-to-end dataflow

```
             ┌────────────────────────── COLLECTION VEHICLE ──────────────────────────┐
             │  GNSS base (F9P) ─RTCM3→ GNSS rover (F9P) ──1PPS──┐                    │
             │                          GNSS heading (F9P, moving-base)               │
             │                              │ GPS time            │                   │
             │                              ▼                     ▼                   │
             │   VN-200 INS (400 Hz IMU) ──────────────► TRIGGER/SYNC BOX (MCU)       │
             │                                            │ 10 Hz trigger + PPS       │
             │        Livox Mid-360 (PPS-synced) ◄────────┤                           │
             │        GS camera ring ◄────────────────────┤   (config R2)             │
             │        ZED 2i stereo (SW-timestamped) ◄────┘   (both configs)          │
             │                              │                                         │
             │                              ▼                                         │
             │               ROS 2 Humble → rosbag2/MCAP  (raw streams only)          │
             └──────────────────────────────┬──────────────────────────────────────────┘
                                            ▼  [I-1: MCAP session bundle + metadata]
    ┌──────────────────────┐    ┌───────────────────────────┐
    │ TRAJECTORY PIPELINE  │    │ INGESTION & KEYFRAMING     │
    │ RINEX→PPK→EKF→RTS    │───►│ keyframe index, undistort, │
    │ ego_pose + quality   │I-2 │ ego-motion-compensated     │
    └──────────────────────┘    │ accumulated clouds         │
                                └────────────┬──────────────┘
                                             ▼  [I-3: keyframes.parquet + clouds + images + poses]
                          ┌──────────────────────────────────────┐
                          │ ANNOTATION PIPELINE (open-weights)   │
                          │ proposals→masks→3D lift→track→boxes  │
                          │ + QA gates → pre-labels              │
                          └────────────┬─────────────────────────┘
                                       ▼  [I-4: prelabels.json → CVAT import]
                          ┌──────────────────────────────────────┐
                          │ HUMAN VERIFICATION (CVAT)            │
                          │ train: flagged + audit sample        │
                          │ val/test: 100 % verified, ≥5 % dual  │
                          └────────────┬─────────────────────────┘
                                       ▼  [I-5: verified labels export]
    ┌──────────────────┐  ┌────────────────────────────┐   ┌─────────────────────────┐
    │ ANONYMIZATION    │◄─┤ RELEASE BUILDER            │──►│ BENCHMARK / DEVKIT      │
    │ blur AFTER annot.│  │ relational schema, splits, │I-6│ metrics, eval server,   │
    │ + audited misses │  │ priors, quality artifacts  │   │ submission format       │
    └──────────────────┘  └────────────────────────────┘   └────────────┬────────────┘
                                                                        ▼  [I-7: results CSV]
                                                            ┌─────────────────────────┐
                                                            │ PAPER BUILD             │
                                                            │ tables/figures scripts  │
                                                            └─────────────────────────┘
```

### 2.2 Interface contracts (the integration spine)

Every arrow above is a named contract. A subsystem may change internally at will; it may not change its contract without a version bump.

| ID | Producer → Consumer | Artifact | Format / key fields | Contract rules |
|---|---|---|---|---|
| I-1 | Capture → everything | Session bundle | `session_id/` dir: MCAP bags + `session_meta.json` + calibration ID | Raw streams only; topic manifest §4.1 is exhaustive; a bag missing a mandatory topic fails ingestion |
| I-2 | Trajectory → Ingestion, Release | Ego pose + quality | `ego_pose.parquet` (t_gps_ns, x,y,z in local ENU, quaternion, σ_pos, σ_yaw, fix_type, stationary_flag), `frame_quality.parquet` | Pose is PPK+RTS output only; real-time RTK never enters this file |
| I-3 | Ingestion → Annotation | Keyframe pack | `keyframes.parquet` (kf_id, t, per-sensor sample refs, Δt per association), single-sweep `.bin` + accumulated `.bin` clouds, undistorted images | LiDAR keyframe is the anchor; accumulation window recorded per cloud |
| I-4 | Annotation → CVAT | Pre-labels | `prelabels.json` per scene: boxes (§A.1 schema) + track_id + per-gate flags + tier | Every box carries provenance `source: "pipeline"` and the QA-gate vector |
| I-5 | CVAT → Release builder | Verified labels | CVAT export → converter → §A.1 schema with `source ∈ {pipeline_accepted, human_verified, human_created}` and `verified_by`, `verification_pass` | val/test records with `source = pipeline_accepted` are a build error |
| I-6 | Release → Benchmark | Dataset release | Relational tables §10.1 + devkit API | Metric code reads only the release schema, never internal files |
| I-7 | Benchmark → Paper | Results | `results/{exp_id}/metrics.csv` + config snapshot + seed | Every paper number regenerable by one script from these CSVs |

### 2.3 Canonical conventions (used by every subsystem)

| Convention | Definition |
|---|---|
| Body frame | ISO 8855 / REP-105: origin at centre of rear axle projected to ground; x forward, y left, z up, right-handed |
| Global frame | Local ENU tangent plane anchored at a published origin (lat/lon/alt); UTM zone 46N conversion provided in devkit |
| Time | **GPS time in int64 nanoseconds** everywhere; UTC offset published once; host wall-clock recorded as secondary only |
| IDs | `session_id` = `YYYYMMDD_HHMMSS_<veh>`, scene/sample/annotation tokens = UUID4 strings (nuScenes-style); `track_id` unique within scene |
| Units | metres, radians, m/s; yaw about +z from +x |
| Box parametrization | center (x,y,z), size (w,l,h), yaw — identical to nuScenes; devkit provides KITTI/Waymo converters |
| Versioning | `calib_id` per session; `taxonomy_v`, `priors_v`, `benchmark_v` semver; any metric-definition change bumps `benchmark_v` (never silent) |
| Config flags | Every release table carries `coverage_config ∈ {R1, R2}` so tooling can enforce eval-region semantics |

---

## 3. Sensor suite (rev 1.0)

Platform: manually driven passenger car, roof-mounted rigid sensor plate. Collection area: Dhaka street network. RTK/PPK base station within 20 km of all routes.

### 3.1 Decision gate G1 — camera coverage configuration *(resolves F2)*

The single ZED 2i (~110° HFOV) cannot generate proposals for a 360° LiDAR sweep, and the annotation pipeline is vision-driven. Exactly one of the following is selected **before scaled collection**; the choice propagates into the eval region (§3.6), the taxonomy stats, the density metric (§8.3.1), and the paper's claims.

| | **R1 — Frontal-scoped (budget)** | **R2 — Ring coverage (recommended for full claims)** |
|---|---|---|
| Cameras | ZED 2i only | ZED 2i **+ 4× global-shutter machine-vision cameras** (2–3 MP, hardware trigger input, ~100–110° HFOV lenses) at 0°, ±90°, 180°, PPS-derived 10 Hz trigger |
| Labelled region | Frontal evaluation region (§3.6): azimuth ±55°, range 0–40 m | ≥ 300–360° annulus, range 0–40 m (blind wedges from mounting documented) |
| Claims permitted | "frontal-sector 3D detection/tracking in extreme density"; **no 360° detection claim** | full 360° detection/tracking claims |
| Sync status | camera software-timestamped (§3.5) | ring cameras hardware-triggered (kills the sync risk too); ZED remains SW-timestamped auxiliary |
| Cost delta | — | ~4 cameras + lenses + trigger MCU + cabling; est. low-thousands USD `[VERIFY current pricing]` |
| Risk if chosen | Reviewer discounts "360° LiDAR" as marketing; density metric is sector-only | Integration effort; more storage; more calibration surfaces |

**Default in this spec: R2.** Every subsequent section notes R1 deltas where they exist. If R1 is forced by budget, do a global find-replace of the claims *before* any text reaches the paper — a single leftover "360°" sentence is an easy reject.

### 3.2 Sensor inventory

| Role | Device | Output | Rate | Sync class |
|---|---|---|---|---|
| GNSS rover | u-blox ZED-F9P (SparkFun) | fixes + raw UBX/RINEX, **1 PPS master** | 10 Hz nav, PPS 1 Hz | master |
| GNSS heading | second ZED-F9P, **moving-base RTK** vs. rover, antenna baseline ≥ 1.0 m along vehicle x-axis | true heading independent of motion | 10 Hz | HW (GNSS time) |
| GNSS base | ZED-F9P, identical config, logged RINEX | RTCM3 (live monitor) + raw obs (PPK) | 1 Hz logging | — |
| INS | VectorNav VN-200 | 400 Hz IMU, 200 Hz INS solution | 400/200 Hz | HW (GNSS time) |
| LiDAR | Livox Mid-360 | ~200 k pts/s first return, per-point timestamps; framed at 10 Hz (~20 k pts/frame) | 10 Hz canonical + raw stream | HW (PPS) |
| Camera ring (R2) | 4× GS machine-vision | Bayer/RGB frames | 10 Hz triggered | **HW (trigger)** |
| Stereo camera | ZED 2i forward | raw stereo pairs 1280×720@60; depth generated offline | 60 fps | SW (measured offset) |

#### 3.2.1 GNSS notes (unchanged rationale, kept binding)
- 10 Hz nav rate is the stable operating point; the INS supplies high-rate pose. The paper never quotes "a GNSS rate" as the ground-truth rate.
- Per-epoch quality logged: `fix_type ∈ {NO_FIX, 2D, 3D, DGNSS, RTK_FLOAT, RTK_FIXED}`, sat count, H/PDOP, estimated accuracy, correction age. % RTK_FIXED (real-time) and % PPK-fixed (post-processed) are both headline release figures.
- **PPK is the primary trajectory path**; raw observation logging on rover, heading unit, and base is unconditional for every session.

#### 3.2.2 Heading *(resolves F10)*
- Moving-base F9P pair yields heading accuracy on the order of ~0.2–0.4° at ≥ 1 m baseline `[VERIFY against u-blox ZED-F9P moving-base integration manual; publish the measured value, not the datasheet]`, independent of vehicle motion — the stop-and-go failure mode is removed at source, not mitigated.
- Alternative accepted: swap VN-200 → VN-300 (integrated dual-antenna). Decide on procurement, not preference; both satisfy F10.
- Regardless of hardware: offline **RTS smoothing** over each session, per-frame yaw σ published, `stationary_flag` published. Yaw drift across stop durations characterized in the release even though dual-antenna makes it small — that measurement *is* the credibility artifact.

#### 3.2.3 LiDAR (Livox Mid-360) — binding consequences
- Non-repetitive rosette, 360°×59°, ~40 m @10 % reflectivity. Canonical frames 10 Hz; raw per-point-timestamp stream always logged so users can re-integrate at any window.
- Per keyframe, an **ego-motion-compensated accumulated cloud** is produced (PPK pose), window `W_acc` fixed dataset-wide after the pilot (default 5 sweeps ≈ 0.5 s; `[GATE G6]` chosen by maximizing annotator 3D-box agreement on S0, then frozen — changing it later means re-processing and possibly re-annotation).
- Positioning in the paper: deliberate low-cost solid-state-class sensor representative of deployable systems in this market; consequences for baselines in §8.5.

#### 3.2.4 ZED 2i — role clarified
- Sensor stream, not annotation source of depth truth. Depth/point cloud generated **offline** from raw stereo (best quality mode; halves capture I/O; §5.2 rationale retained).
- Night policy: fixed exposure + gain ceiling per condition chosen in pilot, WB locked, per-frame achieved timestamps recorded; achieved fps characterized per condition.
- Under R2 the ZED is an auxiliary stream (stereo/depth tasks); the triggered ring is the annotation imagery.

### 3.3 Mounting and extrinsics (unchanged, binding)
Single stiff plate; LiDAR highest and unobstructed; ring cameras with overlapping FOV seams documented; GNSS antennas on ground planes, rover/heading antennas rigid with measured baseline; INS near body origin; vibration damping; IP-rated enclosures (monsoon is a feature only if the rig survives it). All lever arms physically measured ≤ 1 cm.

### 3.4 Calibration (versioned per session; residuals published)

| Calibration | Method | Published output |
|---|---|---|
| Intrinsics (each camera) | ChArUco, full-FOV pose coverage | K, distortion, RMS reprojection (px) |
| Stereo extrinsics (ZED) | same capture; factory verified not assumed | R, t, epipolar error (px) |
| LiDAR → each camera | target-based init, targetless refinement | `T_cam_lidar`, reprojection residual (px) |
| LiDAR → body/INS | hand-eye from trajectory | `T_body_lidar`, residual (m, deg) |
| Ring camera ↔ ring camera | overlap bundle adjustment | seam consistency (px) |
| Antenna lever arms (rover, heading, base-on-vehicle n/a) | direct measurement | vectors (m) |
| Per-pair time offset | motion cross-correlation (§3.5.3) | offset (ms) + within-session stability |

Re-verified at the start of every collection day and after any disturbance; `calib_id` recorded per session; no single global calibration is ever published across months.

### 3.5 Time synchronization

**3.5.1 Architecture.** GNSS 1 PPS is the rig master. A microcontroller **trigger/sync box** (Teensy/STM32 class) phase-locks to PPS and emits: (a) 10 Hz camera trigger pulses aligned to LiDAR frame boundaries, (b) event timestamps in GPS time for every pulse. LiDAR is PPS-synchronized natively; VN-200 carries GNSS time. One clock domain: GPS time, int64 ns.

**3.5.2 Camera timestamping.** R2 ring: exposure mid-point = trigger time + exposure/2, hardware-accurate. ZED: software-synchronized; both device and host-receipt timestamps recorded; empirical offset estimated and corrected; residual sync error distribution (median / p95 / max per pair) **published** — most datasets state a target and never measure it; we measure.

**3.5.3 Offset estimation.** Dedicated slalom / stop-go segment at session start and end; cross-correlate INS angular rate vs. camera optical-flow rotation vs. LiDAR scan-to-scan registration; lag at max correlation = offset; start-vs-end drift check.

**3.5.4 Keyframe contract.** LiDAR frame is the anchor (HW-synced, 3D annotations attach to it). Each keyframe associates the nearest image per camera + interpolated INS/PPK pose; the actual Δt of every association is stored in `keyframes.parquet` and its distribution published.

**[GATE G2 — sync adequacy]** ZED-to-LiDAR p95 offset residual ≤ 10 ms measured in the pilot; ring-to-LiDAR p95 ≤ 2 ms. Fail → the affected stream is demoted from annotation use (R2 makes this non-fatal by design).

### 3.6 Evaluation region (formal definition — new, load-bearing)

All labelling, all metrics, and the density statistic are defined over a fixed region **E** in the ego frame at each keyframe:

- **R2:** E = { (r, θ): 0 < r ≤ 40 m } minus documented blind wedges from the mounting (target ≤ 15° total); vertical extent per LiDAR FOV.
- **R1:** E = { (r, θ): 0 < r ≤ 40 m, |θ| ≤ 55° } (θ from +x). 
- Objects intersecting E are annotated per the visibility rule (§6.3). Predictions strictly outside E are ignored (no FP). GT outside E does not exist (no FN). The devkit ships E as code, not prose.
- The 40 m cap follows the Mid-360's 10 %-reflectivity range; the 40 m+ bin in stratified reporting exists only for the accumulated-cloud secondary track and is clearly separated.

---

## 4. Data capture operations

### 4.1 Recording stack and topic manifest *(contract I-1)*

ROS 2 Humble, rosbag2 with **MCAP** storage (throughput, crash resilience, tooling). Raw streams only — anything derivable is regenerated offline; routes do not get re-driven.

Mandatory topic manifest (ingestion fails a session missing any):

| Topic | Type | Content |
|---|---|---|
| `/lidar/points_raw` | livox custom / PointCloud2 | per-point timestamps, first return |
| `/cam/ring{0..3}/image_raw` (R2) | Image (Bayer) | triggered frames + trigger event ids |
| `/zed/left_raw`, `/zed/right_raw` | Image | 60 fps stereo, device+host stamps |
| `/gnss/rover/ubx_raw`, `/gnss/heading/ubx_raw` | raw bytes | RINEX-convertible observations |
| `/gnss/rover/fix`, `/gnss/heading/rel` | NavSat + custom | fix, fix_type, DOPs, moving-base heading |
| `/ins/imu` (400 Hz), `/ins/solution` (200 Hz) | Imu / custom | raw + filtered |
| `/sync/trigger_events` | custom | GPS-time of every camera trigger & PPS edge |
| `/vehicle/health` | diagnostics | disk, temps, dropped-frame counters |

Base station logs RINEX independently with overlapping time coverage; a session without matching base observations is PPK-dead and flagged at ingestion, not discovered later.

### 4.2 Storage & compression policy (fixed before collection; lossy is irreversible)

| Stream | Estimate | Policy |
|---|---|---|
| ZED raw stereo 720p60 | ~100–200 GB/h | visually-lossless intra-frame codec, parameters frozen after pilot |
| Ring 4× ~2.3 MP @10 Hz (R2) | ~30–80 GB/h `[GATE G4: measure in pilot]` | lossless or visually-lossless intra-frame |
| LiDAR raw | ~10–20 GB/h | lossless |
| INS + GNSS raw | < 1 GB/h | lossless |

One recorded pilot hour is measured before capacity purchase. Released imagery codec and QP published; the annotation pipeline consumes the same codec the public receives (no hidden quality gap).

### 4.3 Session protocol & metadata
Automatic per drive: `session_id`, GPS start/end, route id, area, illumination class, weather, road types, `calib_id`, operator, vehicle, firmware/SDK versions per sensor, anomalies. Pre-drive checklist: calibration spot-check, PPS lock, disk headroom, base logging confirmed, sync-segment driven. Post-drive: sync-segment repeat, bag integrity hash, base log pulled.

### 4.4 Coverage plan and pre-registered geographic split *(cannot be repaired later)*

| Axis | Target share of released frames |
|---|---|
| Illumination | day ≤ 50 %; **night+dark ≥ 25 %** (differentiator, not token); dusk balance |
| Weather | clear / cloudy / rain; monsoon sessions explicitly planned |
| Road type | arterial, narrow residential, intersection-dense, flyover |
| Density | free-flow → gridlock; route/time-of-day chosen to fill all four ρ-bins (§8.3.1) |

**Split:** train/val/test separated by **geographic route**, drawn on a map **before collection scales**, no spatial overlap (verified programmatically against route polylines, buffer ≥ 200 m), each split spanning the full density × illumination range. Frame-level random splitting is the classic fatal flaw; it is structurally impossible here because the split is an input to route planning, not an output of sampling.

### 4.5 Pilot protocol and gates
Pilot = ≥ 2 sessions (1 day, 1 night) on non-release routes. Outputs: measured sync residuals (**G2**), real-time fixed-rate + PPK-fixed-rate (**G3:** PPK-fixed ≥ 40 % of frames on urban-canyon route, else odometry task demoted), storage rates (**G4**), exposure tables per condition, accumulated-window study input (**G6**), and the S0 seed-set frames (§7.2).

---

## 5. Ground-truth trajectory pipeline *(contract I-2)*

1. Convert rover/heading/base UBX → RINEX; PPK (e.g., RTKLIB or commercial `[VERIFY tool choice + settings published]`) against ≤ 20 km base.
2. Fuse PPK positions + moving-base heading + VN-200 IMU in an error-state EKF; then **fixed-interval RTS smoother** over the full session (propagates constraints backwards through outages — the reason PPK beats RTK exactly where RTK struggles).
3. Outputs per frame: pose in local ENU, σ_pos, σ_yaw (moving vs. stationary flagged), PPK ambiguity status, and `frame_quality` fields (§2.2 I-2).
4. Published honesty artifacts: % fixed-ambiguity frames; position σ distribution; yaw σ distribution split by stationary flag; map figure of fix quality along routes (paper figure — reviewers love it, and it pre-empts "urban canyon?" objections).
5. Frames failing quality thresholds are **flagged, not hidden**; odometry ground truth masks them; detection/tracking tolerate them (boxes are ego-relative).

---

## 6. Taxonomy and annotation policy (frozen as `taxonomy_v1.0` after S0)

The taxonomy is the hardest artifact to change (revision = re-annotation). It is drafted here, stress-tested on the S0 seed set, then frozen.

### 6.1 Classes

| Group | Classes (16) |
|---|---|
| Motorised 4-wheel+ | car, microbus/van, bus, truck, covered-van |
| Indigenous 3-wheel (the scientific point) | **cng-auto-rickshaw**, **battery-rickshaw (easy-bike)**, **tempo**, **human-hauler** |
| Human-powered | **cycle-rickshaw**, push-cart (thela), bicycle |
| 2-wheel motorised | motorcycle |
| VRU | pedestrian, animal |
| Static | traffic-cone/barrier/construction (single `static-obstacle` class in v1.0) |

Rules retained and made binding:
- Written **annotation guideline** with ≥ 6 photographic examples per class, explicitly covering the ambiguous pairs: battery-rickshaw vs. cycle-rickshaw (motor housing, wheel size, seating), tempo vs. human-hauler, covered-van vs. truck. The guideline is a released artifact.
- Published **class-mapping tables** to nuScenes, Waymo, KITTI, and IDD/IDD-3D (drives §8.3.5 and reproducibility).
- Per-class instance counts, long-tail distribution, box-count vs. range, and points-per-box distribution published (§8.3.4).

### 6.2 Attributes *(resolves F7 — makes real NDS computable)*

Minimal, annotatable, nuScenes-compatible vocabulary; exactly one per box where applicable:

| Applies to | Attribute set |
|---|---|
| All motorised vehicles + tempo/human-hauler + battery-rickshaw | `moving` / `stopped` (driver present, momentarily halted) / `parked` |
| cycle-rickshaw, bicycle, motorcycle | `with_rider` / `without_rider` (crossed with moving/stopped where ridden) |
| pedestrian | `moving` / `standing` / `sitting` |
| animal | `moving` / `static` |

mAAE is computed over this vocabulary; the definition and any deviation from nuScenes attribute semantics is stated next to the metric. Attribute annotation cost is low (one click per box, mostly auto-inferable from track velocity and human-confirmed).

### 6.3 Visibility, minimum-points, and group/ignore boxes *(resolves F9)*

- **Annotate** an object iff it intersects eval region E (§3.6) **and** (≥ 5 LiDAR points in the single-sweep frame **or** ≥ 25 % visible in any ring/ZED image). The single threshold that moves every AP number — stated, versioned, never silently changed.
- **Visibility bins** per box (nuScenes-style): 0–40 / 40–60 / 60–80 / 80–100 %.
- **Group/ignore boxes:** where individual objects are genuinely unresolvable (gridlock clumps, dense pedestrian crowds), annotators draw a single `group` box with a class-group tag (e.g., `crowd-pedestrian`, `clump-mixed-vehicle`) instead of guessing instances. Evaluation semantics (in devkit code): predictions matched to a group box are removed from scoring (no FP); group boxes are never counted as FN. This is the CrowdHuman/Waymo-NLZ pattern adapted to 3D and is what makes extreme-density annotation honest instead of noisy.
- LEVEL_1 / LEVEL_2 difficulty (Waymo convention) with point thresholds set for *this* sensor's density from S0 statistics — not copied from a 64-beam dataset.

### 6.4 Annotation quality protocol
- **Inter-annotator agreement:** ≥ 5 % of val/test frames independently double-annotated; report mean 3D IoU of matched boxes, class agreement (Cohen's κ), and missed-box rate per annotator. S0 provides the first IAA measurement and calibrates the guideline before scale-up.
- Adjudication: disagreements above threshold resolved by a senior annotator; adjudication log kept.

---

## 7. Annotation pipeline v2 — open-weights, zero-LLM, human-anchored

### 7.1 Component substitutions *(resolves F5 + F6)*

The draft pipeline mixed verifiable components with unverifiable or API-bound ones. v2 restricts the hot path to **locally runnable, open-weights, citable** models. The "fully offline / zero proprietary subscription" claim is now literally true.

| Draft component | Status | v2 decision |
|---|---|---|
| "DINO-X Pro" | API-served; contradicts offline claim; numbers unverifiable here | **Replaced** by **Grounding DINO (Swin-L)** / **MM-Grounding-DINO** open checkpoints for open-vocabulary proposals `[VERIFY exact checkpoint + license]` |
| "SAM 3.1", "Object Multiplex" | Not verifiable as of drafting | **Replaced** by **SAM 2.1** (open weights) for mask generation + video propagation with memory; multi-object handled by batched per-object state `[VERIFY version + memory-window settings]` → reinstated as the selectable `sam31_multiplex` provider by DECISIONS C26 (2026-08-19), with on-disk measurement: F6's "unverifiable" condition no longer holds |
| DINOv2 ViT-L/14 | Verified open | Kept (OOD scan §7.3.1, re-ID embeddings §7.3.6) |
| DetZero (ICCV 2023) | Verified | Kept as the design reference for offboard track-refine; numbers quoted only from the paper |
| Point-SAM | Exists; claimed numbers unverified | Demoted to `[EXPERIMENTAL]` optional refinement; excluded from headline pipeline results unless its ablation on S0 shows gain |
| SAM4D | `[VERIFY venue + code availability]` | Config-B alternative only if code exists; otherwise dropped from the document |
| VESPA (CVPR 2026) numbers | Pre-publication figures in draft | Quote only what the published paper states; otherwise cite as design inspiration without numbers |
| Otter / BLIP-2 / all LLM calls | — | Remain removed (this is the point of the zero-LLM design) |
| "Impact Scores" for datasets | Source unknown | Deleted everywhere |
| Spatialized audio outputs | No microphones exist | Deleted everywhere (§1.4) |

**7.1.1 Citation verification checklist (blocking for the paper):** for every model/number that survives into the paper — venue, year, table, and figure re-checked against the primary PDF; a `citations_verified.md` ledger records who checked what. One hallucinated citation costs more trust than the pipeline earns.

### 7.2 Seed set S0 *(resolves F4 — the bootstrap)*

- **Content:** 3,000–5,000 keyframes from pilot + early sessions, stratified across density × illumination × road type, **fully manually annotated** (two-pass: annotate, then independent review), including attributes and group boxes.
- **S0 outputs, each versioned as `priors_v1`:**
  1. **Per-class dimension priors** — mean/σ of (w, l, h) for *all 16 classes including every indigenous class* → box-inflation dictionary (nuScenes/KITTI values are initialization only, deleted after S0).
  2. **Per-class BEV DBSCAN ε** — derived from S0 footprints (e.g., ε ≈ 0.6 × mean footprint diagonal, tuned on S0).
  3. **QA-gate thresholds** — confidence cutoff, min-LiDAR-return count, spatial-sanity multipliers tuned to maximize auto-accept precision at a fixed target (§7.4).
  4. **LEVEL_1/2 point thresholds** and difficulty strata (§6.3) from measured points-per-box.
  5. **IAA baseline** and guideline revisions.
- **[GATE S1 — open-vocabulary separability]:** run the proposal model with the full taxonomy prompt array on S0. Pass criterion per indigenous class: detection recall ≥ 0.7 at the working confidence threshold **and** pairwise confusion (battery- vs. cycle-rickshaw, tempo vs. human-hauler) ≤ 30 %. **Fail →** that class pair is merged for *pipeline pre-labelling only* and split by humans during verification (taxonomy unchanged; pipeline honesty preserved); the S1 result itself is a reportable finding either way.
- S0 doubles as: pipeline development/validation split, the audit reference for §7.5, and the first IAA sample.

### 7.3 Pipeline stages (per scene; contract I-3 → I-4)

1. **Ingestion:** keyframe index; undistortion; ego-motion-compensated 5-sweep accumulated clouds (window per G6); ground removal by sector-wise RANSAC over the accumulation (drop |z−ground| < 0.3 m; prune > 40 m and > 4 m height inside E).
2. **OOD / long-tail discovery:** DINOv2 embeddings on 1-in-10 keyframes → UMAP + HDBSCAN; outlier frames routed to human review for taxonomy-gap detection (bounded human effort replaces the deleted LLM loop). Any confirmed novel label appended to the prompt array; taxonomy itself only changes pre-freeze.
3. **2D proposals:** Grounding DINO with the frozen taxonomy prompt array on every ring/ZED keyframe image; per-class thresholds from S0 (global default 0.40).
4. **Masks + short-horizon propagation:** SAM 2.1 prompted by proposal boxes; IoA-NMS (> 0.5) across overlapping cameras; propagation window ≤ 16 frames with state re-init at block boundaries (VRAM containment on the 24 GB prototype card).
5. **2D→3D lift:** project ground-filtered single-sweep points through calibrated `K, T_cam_lidar` per camera; points inside a mask inherit its class; **per-camera frusta unioned (R2) so the lift covers E**.
6. **BEV clustering:** class-conditional DBSCAN with `priors_v1` ε; keep largest connected cluster; L-shape fit → oriented box; sparse-cluster guard (< 15 pts → skip ICP, Kalman velocity fallback).
7. **Tracking + refinement (DetZero-style offboard):** associate across frames by 3D IoU × DINOv2 cosine similarity; ICP on matched clusters for velocity; forward-backward smoothing over the whole scene (offboard = use the future); yaw consistency enforcement along tracks (fights the symmetric-3-wheeler 180° flip failure).
8. **Amodal inflation:** sparse boxes inflated toward `priors_v1` class means, anchored to the LiDAR-return surface (shift away from ego).
9. **QA gating → tiering (I-4):** each box gets a gate vector {confidence, spatial-sanity vs. 2× class prior + drivable check, LiDAR-return ≥ 5} and a tier: `auto_accept`, `flagged`, `rejected`. **Nothing ships as ground truth from this stage** — see §7.4.
10. **Attribute pre-fill:** track velocity → moving/stopped/parked suggestion; rider presence from mask overlap heuristic; human-confirmed.

### 7.4 Label-tier policy *(resolves F1 — the central fix)*

| Split | Human treatment | What `source` may say | Shipped error accounting |
|---|---|---|---|
| **test** | **100 % verified**: every pre-label confirmed/corrected; missed-object sweep pass; ≥ 5 % double-annotated | `human_verified` / `human_created` only (build error otherwise) | IAA published |
| **val** | identical to test | same | IAA published |
| **train** | all `flagged`+`rejected` corrected; `auto_accept` shipped as-is **but** audited (§7.5) | any, with provenance field | audited error rates published per class × density × illumination |
| S0 | two-pass manual | `human_created` | reference standard |

CVAT integration: pre-labels imported with track interpolation; verification UI shows synchronized ring images + accumulated cloud; hotkeys for class-pair disambiguation; provenance and `verification_pass` recorded per box (contract I-5). Annotator throughput assumption: budget from S0 measurements, **not** extrapolated from KITTI (sparser non-repetitive returns annotate slower).

### 7.5 Pseudo-label audit (train split)
Stratified random sample (≥ 2,000 auto-accepted boxes across class × density × illumination cells) blind-reviewed by senior annotators. Published: box-level precision, localization error distribution vs. human boxes, class-confusion, and missed-object rate per frame. These numbers appear in the paper's dataset-quality table — turning F1's weakness into a transparency contribution ("we tell you exactly how good the machine labels are").

### 7.6 Compute & human-effort budget (resource units, not calendar)

| Item | Unit cost (measure in pilot; figures below are planning placeholders) |
|---|---|
| Pipeline inference (stages 1–9, 24 GB GPU, serial) | ~1–2 s/keyframe → ~3–6 GPU-h per 10 k keyframes |
| Pipeline inference (48 GB GPU, FP16/TRT where supported) | ~0.4–0.8 s/keyframe |
| Human verification, train flagged boxes | measure on S0; plan ~10–20 s/box |
| Human verification, val/test full pass | measure on S0; plan ~60–120 s/keyframe incl. missed-object sweep |
| S0 two-pass manual | plan ~3–5 min/keyframe |

All throughput claims in the paper are the **measured** values with the hardware named; the draft's per-module millisecond tables are internal planning only and do not appear in the paper unless re-measured on final hardware.

### 7.7 Known failure modes → mitigations (paper's supplementary table)
- Reprojection ghost points → class-conditional DBSCAN + largest-cluster rule (keep VESPA-style ablation on S0 to justify).
- Symmetric small vehicles → 180° yaw flips → track-level yaw smoothing + APH metric surfaces the residual honestly.
- Night proposal recall drop → per-illumination thresholds from S0-night; report pipeline recall by illumination.
- Occlusion chains in gridlock → group boxes (§6.3) + human sweep on val/test.
- Non-repetitive sparsity at range → 40 m eval cap; accumulated-cloud secondary track for beyond-range study.

---

## 8. Benchmark specification

### 8.1 Design principles (binding)
1. **Comparable by default** — established metric definitions adopted exactly; every deviation stated inline.
2. **Stratify rather than average away** — every headline metric also reported by density and illumination.
3. **Publish measurements, not adjectives** — sync, fix rates, calibration residuals, IAA, pseudo-label audit are release numbers.
4. **No test-set leakage** — geographic split (§4.4), enforced in code.
5. **Region-honest** — all metrics computed inside E (§3.6) with group-box semantics (§6.3); the devkit is the single source of truth.

### 8.2 Part A — established tasks

**8.2.1 3D detection (primary).** nuScenes protocol verbatim: center-distance matching D = {0.5, 1, 2, 4} m; AP with the <10 % recall/precision region excluded (omitting that exclusion inflates results — we keep the official definition); mAP over classes × thresholds; TP metrics at D = 2 m: mATE, mASE, mAOE, **mAVE reported separately for moving vs. stationary** (a stationarity-dominated dataset would otherwise zero-wash it), mAAE over §6.2 attributes; **NDS** by the official formula — now fully computable *(F7 closed)*.
Secondary: KITTI-style AP_R40 (3D + BEV), IoU 0.7 car-like / 0.5 for pedestrians and all 2-/3-wheelers, per-class thresholds for indigenous classes stated; difficulty strata redefined for this sensor from S0 (§6.3). **APH** (heading-weighted AP) reported as a first-class number — near-symmetric 3-wheelers make 180° flips common and plain AP is blind to them. LEVEL_1/LEVEL_2 per §6.3.
Stratification: range bins 0–20 / 20–40 m (40 m+ only in the accumulated-cloud side track), points-per-box bins, always per-class, plus §8.3 strata.

**8.2.2 3D multi-object tracking.** Primary: AMOTA / AMOTP (center-distance 2 m, consistent with detection). CLEAR MOT reported (MOTA, MOTP, IDS, FRAG, MT, ML, FP, FN). **HOTA with DetA/AssA decomposition as a headline result** — dense weaving traffic is an association problem; DetA answers "did you find the rickshaw", AssA answers "did you keep its ID through the gap".

**8.2.3 Odometry / SLAM / localization.** ATE (Umeyama alignment; SE(3) metric methods, Sim(3) monocular — alignment stated per table); RPE at Δ ∈ {1 s, 10 m, 100 m}; KITTI convention t_rel (%) and r_rel (deg/m) over 100–800 m subsequences; localization Recall@N, success @ (<0.5 m, <2°), **longitudinal and lateral errors reported separately**. Ground-truth honesty per §5: frames without fixed-ambiguity PPK masked and the mask published. *(C3 evidence.)*

**8.2.4 Depth.** Against LiDAR-projected depth only (ZED stereo is a sensor stream, never truth): AbsRel, SqRel, RMSE, RMSE-log, δ<1.25^i; evaluation capped at the LiDAR reliable range (stated); day/night reported separately.

**8.2.5 Segmentation / occupancy / forecasting.** v1.x only, if annotation exists (thin benchmark worse than none). Forecasting, if added, is explicitly **map-free** and framed as a legitimate open setting for non-lane-following agents.

### 8.3 Part B — proposed contributions

**8.3.1 Density-stratified evaluation (headline; C2).**
ρ(frame) = (# annotated agents whose box center lies in E within radius R = 30 m of ego) / area(E ∩ disc(R)), with E per §3.6 — the area normalization makes ρ comparable across R1/R2 and across blind-wedge layouts *(fixes the F2 interaction the draft missed)*. Group boxes contribute their annotated lower-bound count `n_min` (§A.1). Bins Low/Medium/High/Extreme by dataset quantiles, edges published. Every headline metric reported per bin + **Density Degradation Δ = metric(Low) − metric(Extreme)**.
**[GATE G5 — pilot ablation, the paper's own kill-criterion]:** run ≥ 1 strong published detector on S0 across bins; if the curve is flat, C2 is weak and the pivot (§11.6) triggers *before* the paper is written.

**8.3.2 Illumination-stratified evaluation.** Same structure over bins of **measured mean image luminance** (not subjective labels); Night Degradation Δ per task; carries weight only because §4.4 forces night ≥ 25 %.

**8.3.3 Heterogeneity / long-tail reporting.** Per-class AP for indigenous classes never hidden in a macro-average; class-balanced mAP beside plain mAP; full cross-class confusion matrix (battery- vs. cycle-rickshaw confusion is a concrete, quotable finding either way).

**8.3.4 Sensor- and label-quality artifact.** Per session: camera↔LiDAR offset median/p95/max + stability; LiDAR↔INS residual; % FIXED/FLOAT (real-time and PPK); pose σ distributions; yaw σ moving/stationary; calibration residuals; IAA; **pseudo-label audit table (§7.5)**. Cheap, credibility-defining, and genuinely useful to fusion researchers.

**8.3.5 Cross-dataset domain-gap protocol (C4) — with both confounds controlled *(resolves F3)*.**
Four cells (train × test over {nuScenes, DhakaScenes}); G(A→B) = metric(train B, test B) − metric(train A, test B); Generalization advantage = G(nu→Dhaka) − G(Dhaka→nu).
Controls (all reported):
1. identical model, init, training budget per cell;
2. matched training-set size (subsample larger; size-vs-performance curve);
3. **matched sensor representation** — both datasets rendered to a common BEV grid (resolution + encoding published); acknowledged as the largest threat and therefore controlled first;
4. matched class set via the published mapping;
5. ≥ 3 seeds, variance reported;
6. **label-provenance control** — repeat train-on-Dhaka using only the human-verified subset; the delta vs. full-train bounds pipeline-inherited bias. If controls 3 or 6 explain the gap, the paper says so plainly (a controlled negative here is still publishable; an uncontrolled positive is not).

**8.3.6 Interaction / safety mining (`critical` subset).** Mined automatically from PPK trajectory + tracks (no new annotation): interaction density (agent pairs on conflicting paths within horizon), critical-event index (min TTC below threshold; ego-jerk bound), yielding events at unsignalized conflicts. Released as a named subset with its own leaderboard tab — disproportionate value to the planning community at mining cost only.

**8.3.7 Composite headline.** Robustness Score = mean over {density × illumination} strata of the normalized primary metric, reported **with the worst-stratum value beside it** — for a safety-relevant domain the worst case is the honest summary. Never replaces the component tables.

### 8.4 Splits, leaderboard, hygiene
Geographic split (§4.4); test labels withheld behind a submission server where feasible; val public. Fixed submission JSON, published eval code, versioned metrics (`benchmark_v`), per-team submission rate limits, mandatory disclosure of training data / external pretraining / model size. A changed metric definition bumps the version; historical numbers never silently invalidated.

### 8.5 Baselines and adaptation policy

| Task | Baselines (≥ 3 per headline task, retrained + tuned — an undertrained baseline is the easiest reviewer target) |
|---|---|
| LiDAR 3D detection | PointPillars, CenterPoint, TransFusion-L (PV-RCNN optional) |
| Camera 3D detection | frontal-only under R1; BEVDet/PETR-class under R2 `[VERIFY current SOTA availability at build time]` |
| Fusion | BEVFusion or TransFusion |
| Tracking | AB3DMOT (floor), CenterPoint-tracking, one HOTA-competitive modern tracker |
| Odometry/SLAM | FAST-LIO2, KISS-ICP, LIO-SAM; ORB-SLAM3 (visual arm) |
| Depth | Monodepth2-class + one modern method |

**Non-repetitive-scan adaptation policy (binding):** range-image/ring-assuming backbones are either excluded (stated) or adapted; every adaptation documented next to its results; input variants reported where relevant (single sweep vs. 3-sweep vs. 5-sweep accumulation) — an unadapted baseline understates the method and overstates the dataset's difficulty, which reviewers now check for.
**Compute envelope (resource, not calendar):** detection baseline ≈ 40–120 GPU-h each on the stated GPUs `[measure]`; × 3 seeds × (in-domain + cross-dataset cells) dominates total compute — plan cloud burst or a small cluster; the two local GPUs alone cannot carry §8.3.5 at 3 seeds.

---

## 9. Privacy, legal, ethics

| Item | Binding requirement |
|---|---|
| Faces & plates | detected and blurred before **release**, applied **after annotation** (blurring degrades pedestrian detection; annotate on originals) |
| Blur→benchmark interaction | effect of blurring on detector scores measured and reported (val subset, blurred vs. original, model fixed) |
| Anonymization quality | miss rate from a manually audited sample, published |
| Originals | unblurred masters under access control, never released |
| Local law | Bangladeshi public-space recording + personal-data requirements verified with local counsel **before release**; summarized in the datasheet `[VERIFY — do not rely on any model's summary of Bangladeshi law]` |
| IRB / institutional ethics | clearance obtained and cited where required |
| License | decide deliberately; **recommendation: CC BY-NC-SA 4.0 for imagery + labels with a permissive devkit license**, revisiting CC BY 4.0 if counsel and institution allow — NonCommercial limits adoption (state this trade-off in the datasheet rather than pretending it away) |
| Documentation | *Datasheet for Datasets* (Gebru et al.) published with v1.0 |

---

## 10. Release engineering

### 10.1 Schema — nuScenes-devkit-compatible relational tables (+ extensions)
`log, scene, sample, sample_data, ego_pose, calibrated_sensor, sensor, instance, sample_annotation, category, attribute, visibility` — field-compatible with nuScenes so the ecosystem's tooling mostly just works; plus extensions:
- `frame_quality` (per sample: fix_type, ppk_fixed, σ_pos, σ_yaw, stationary, per-pair Δt, luminance, ρ, density_bin, illumination_bin)
- `session_quality` (per log: sync stats, calib residuals, % fixed, storage/codec info)
- `group_annotation` (group boxes: extent, class-group, `n_min`)
- `provenance` on every `sample_annotation` (source, verified_by, verification_pass, pipeline gate vector)
Raw layer: MCAP session bundles + RINEX. Priors: `priors_v*.json`. Everything versioned.

### 10.2 Directory layout
```
release/v1.0/
  tables/ (json)             raw/{session_id}/ (mcap, rinex)
  samples/{cam}/{...}.jpg    sweeps/lidar/{...}.bin
  accumulated/{...}.bin      maps/routes_split.geojson
  priors/priors_v1.json      quality/{session_quality,audits}/
  docs/{guideline.pdf, datasheet.pdf, changelog.md}
devkit/ (pip-installable; loaders, E-region code, all metrics, converters, split verifier)
eval-server/ (containerized; same metric code, pinned benchmark_v)
```

### 10.3 Submission format
One JSON per task, schema in devkit, validated client-side; results reproducible from I-7 CSVs.

---

## 11. Paper plan

### 11.1 Claim → evidence map = §1.2 table. Every experiment below exists to feed exactly one row of it.

### 11.2 Experiment list
E1 detection baselines ×3 seeds (C1, C5) · E2 density-degradation curves ≥ 2 detectors + 1 tracker (C2) · E3 tracking incl. HOTA decomposition (C1/C2) · E4 odometry suite + PPK quality figures (C3) · E5 cross-dataset 4-cell + controls 3 & 6 (C4) · E6 illumination strata (C2) · E7 confusion matrix + class-balanced mAP (C1) · E8 pseudo-label audit + IAA (C5) · E9 blur-impact study (C5) · E10 pipeline ablations on S0 (denoising, tracking, inflation — VESPA-style, with our numbers not theirs).

### 11.3 Figures & tables (build scripts live in `paper/`)
Fig 1 rig + coordinate frames · Fig 2 taxonomy montage (indigenous classes) · Fig 3 ρ histogram + degradation curves (the money figure) · Fig 4 PPK fix-quality route map · Fig 5 night/monsoon qualitative with boxes · Fig 6 confusion matrix. Tab 1 dataset comparison (§1.3) · Tab 2 detection main · Tab 3 tracking/HOTA · Tab 4 4-cell + controls · Tab 5 quality artifact (§8.3.4) · Tab 6 per-class indigenous AP.

### 11.4 Reviewer-objection pre-emption
| Anticipated objection | Pre-emption (already in design) |
|---|---|
| "Labels are machine-generated" | §7.4 tiers + §7.5 audit table; val/test 100 % human |
| "360° claim vs one camera" | G1 decision + formal E region; claims scoped in text |
| "Harder data or noisier labels?" | IAA + audit + C4 controls; degradation measured on *human-verified* val |
| "IDD-3D already did this" | §1.3 table + verbatim differentiator statement |
| "Non-repetitive LiDAR unfair to baselines" | §8.5 adaptation policy + input-variant reporting |
| "Cherry-picked density effect" | pre-registered bins by quantiles; G5 run before writing |
| "NDS not comparable" | §6.2 attributes; official formula unchanged |
| "Urban-canyon GT unreliable" | §5 per-frame σ + masks; odometry demotion rule G3 |

### 11.5 Limitations section (drafted now, kept honest)
Single city; one LiDAR class (no cross-sensor generalization within the release); 40 m labelled range; R1-if-chosen frontal scope; pseudo-label residual error on train (quantified); no audio/RADAR/HD-map; forecasting deferred; Bangladeshi legal review scope.

### 11.6 Kill criteria & pivots
- **G5 flat density curve** → pivot A: lead with C1 (taxonomy + long-tail) + C3 (PPK) and report the flat curve as a finding; pivot B: the annotation-pipeline paper (zero-LLM open-weights auto-labeling validated on nuScenes against its LLM-based ancestor, with honest component ablations) — Dhaka-independent and separately publishable.
- **S1 separability fail** → merged-pair pre-labelling (§7.2); paper reports open-vocabulary limits on indigenous classes as a finding.
- **G3 PPK fail** → odometry demoted to flagged-subset appendix; C3 dropped from claims.

---

## 12. Risk register (top items)
| Risk | L | I | Mitigation |
|---|---|---|---|
| Camera-coverage decision drifts (F2 regression) | M | Fatal | G1 is blocking; claims lint (§3.1) |
| Pseudo-labels leak into val/test | M | Fatal | I-5 build error on provenance; CI check |
| Citation errors survive to submission | M | High | §7.1.1 ledger, second-person check |
| Baseline compute underestimated | H | High | §8.5 envelope measured early; cloud burst budgeted |
| Monsoon damages rig | M | High | IP enclosures, spares for F9P/cameras |
| Annotation throughput below plan | H | Med | measured on S0 before scale; group boxes cap worst frames |
| Legal review surprises | L | High | counsel engaged before release decisions, not after |

## 13. Execution dependencies (order only — deliberately no dates)
G1 (cameras) & F10 hardware → rig integration → calibration + sync validation (G2) → pilot (G3, G4, exposure tables) → **S0 seed set** → S1 + priors_v1 + G6 window → taxonomy freeze → scaled collection (split pre-registered) ∥ pipeline v2 hardening → pre-label → verify (tiers) → audits + IAA → release build → E1–E10 → paper. The only intentionally serialized bottleneck is S0 → taxonomy freeze → scale; everything after collection parallelizes.

---

## Appendix A — schemas (abbreviated; canonical versions live in devkit)

**A.1 `sample_annotation` record**
```json
{"token":"uuid","sample_token":"uuid","instance_token":"uuid","category":"cng-auto-rickshaw",
 "translation":[x,y,z],"size":[w,l,h],"rotation":[qw,qx,qy,qz],"velocity":[vx,vy],
 "attribute":"moving","visibility":"60-80","num_lidar_pts":37,
 "provenance":{"source":"human_verified","verified_by":"ann_07","verification_pass":2,
   "gates":{"conf":0.63,"spatial_ok":true,"lidar_pts_ok":true},"tier":"flagged"}}
```
**A.2 `group_annotation`**: `{token, sample_token, class_group:"crowd-pedestrian", polygon_bev:[...], height_range:[z0,z1], n_min:6}` — eval-ignore semantics per §6.3.
**A.3 `ego_pose`**: `{token, timestamp_gps_ns, translation, rotation, sigma_pos_m, sigma_yaw_deg, ppk_fixed:true, stationary:false}`
**A.4 `priors_v1.json`**: per class `{dims:{w:{mu,sigma},l:{...},h:{...}}, eps_bev, min_pts_L1, min_pts_L2, conf_thresh, source:"S0", n_instances}`
**A.5 `session_meta.json`**: §4.3 fields + topic-manifest checksum + `calib_id`.

## Appendix B — open questions (tracked, each with an owner before scale-up)
1. R2 camera model + lens selection and mount seams (feeds G1).
2. PPK toolchain choice and settings publication (§5).
3. Eval-server hosting + submission-limit policy.
4. Final license after counsel (§9).
5. Whether Config-B (native 3D fusion) is retained at all after SAM4D availability check (§7.1).
6. Dataset name.
