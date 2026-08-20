# CONFORMANCE — rendered from conformance.yaml, do not edit

Rendered 2026-08-19 22:34 at commit `4614a9c` by `scripts/check_conformance.py`. 284 rows.

| Status | Rows | Meaning |
|---|---:|---|
| CONFORMS | 39 | demonstrated by TEST or MEASUREMENT |
| PLAUSIBLE | 135 | code appears to implement it; nothing demonstrates it |
| VIOLATES | 50 | implemented contrary to the claim |
| ABSENT | 57 | not implemented |
| UNVERIFIABLE | 2 | not checkable on this substrate |
| N/A | 1 | waived by plan §4 |

Evidence classes: ARTIFACT 35, CODE-SITE 209, MEASUREMENT 36, TEST 4

**Reading order: VIOLATES first.** A PLAUSIBLE row is an open question, not a pass — it converts to CONFORMS only when a test or measurement lands (BUILD_PROMPT.md §4.2).

## VIOLATES (50)

| id | claim | evidence | closes | phase |
|---|---|---|---|---:|
| `0-r6` | CAM_FRONT intrinsics are fx=fy=1266.417, cx=816.267, cy=491.507 | MEASUREMENT: EV cam_front_intrinsics.distinct == 2 | — | 1 |
| `1.10-r1` | coverage_config: R2 resolved; recorded in every output record; E derived from it in eval_region.py, never a constant | CODE-SITE: pipeline/common/eval_region.py:216-227,54-56; ingest.py:142; priors.py:135 | P1-4, M-6 | 2 |
| `1.8-r3` | validate_paths() asserts the version directory name matches the configured version (mini metadata against trainval blobs reads … | CODE-SITE: pipeline/common/paths.py:238-245 | P0-9 | 1 |
| `1.8-r6` | usable_scenes.json records the dataroot realpath and metadata fingerprint, and every consumer verifies the match | CODE-SITE: pipeline/stage0_data_probe/probe.py:824-827; stage1_ingestion/ingest.py:756-777; stage2_ood/ood.… | P0-9 | 3 |
| `1.9-r5` | run_manifest.json carries package versions (Python/PyTorch/CUDA/UMAP/HDBSCAN/sklearn) | ARTIFACT: stage1 manifest carries only python_version 3.10.12 and numpy_version 1.21.5 | P1-8 | 4 |
| `10-r12` | MobileSAM input size is config | CODE-SITE: adapter constant in masks.py; no config entry | — | 6 |
| `10-r14` | Cluster tie-break is config | CODE-SITE: cluster.py:435-443 hardcoded (documented) tie-break | — | 8 |
| `10-r15` | Association gates, matcher, birth/death, minimum crop size are config | CODE-SITE: track.py:181-234 dataclass defaults with provenance strings :239-296 ('derived for 2 Hz' :252-256) | M-11 | 9 |
| `10-r16` | Kalman noise + min-point ICP guard (15) are config | CODE-SITE: track.py:212-221 (Kalman noise) + :226 (icp_min_points_each_side 15) dataclass defaults | — | 9 |
| `10-r17` | Inflation trigger/blend/clamp are config | CODE-SITE: inflate.py:155-243 dataclass defaults w/ provenance | M-12 | 8 |
| `10-r19` | UMAP + HDBSCAN parameters and sampling rate (1-in-10 over images) are config | CODE-SITE: ood.py:118-125 dataclass defaults | — | 9 |
| `10-r2` | Required-channel set is config | CODE-SITE: schemas.py:93-105 — Python constant with provenance comment, not config | M-15 | 2 |
| `10-r22` | Checkpoint ids + revisions are config | CODE-SITE: proposals.py:1680-1692 / masks.py:2020-2026 build CheckpointSpec inline from dataclass defaults … | P0-8 | 5 |
| `10-r3` | camera_subset is config | CODE-SITE: dataclass default (ingest.py:142 area); recorded in stage1 manifest | — | 2 |
| `10-r4` | coverage_config + eval-region parameters (r_max, rho radius, wedge) are config | CODE-SITE: eval_region.py:54-56 module constants; three stages ignore configured values entirely (1.10-r2) | P1-4, M-6 | 2 |
| `10-r5` | W_acc count AND duration are config with provenance | CODE-SITE: ingest.py:118-119 dataclass defaults w/ provenance strings; no YAML | — | 4 |
| `10-r6` | RANSAC sector count / distance threshold / iterations / seed are config with provenance | CODE-SITE: ingest.py:123-152 dataclass defaults, three marked 'arbitrary, needs tuning' | P1-13 | 4 |
| `10-r7` | Ground band 0.3 m / range cap 40 m / height cap 4 m are config, flagged as inherited-unvalidated | CODE-SITE: ingest.py:127+ dataclass defaults with provenance 'comprehensive.md §7.3.1, unvalidated on this … | P1-13 | 4 |
| `11-r5` | Predict-then-match at 2 Hz; effective dt recorded; IoU gate config with 2 Hz provenance; locked at Phase 9 | CODE-SITE: predict-then-match + gate provenance implemented (5.8-r1/r4); effective dt NOT recorded (5.8-r2) | P1-1 | 9 |
| `12-r1` | No phase N+1 started before phase N's exit gate passed | MEASUREMENT: code for stages 2-8 exists and priors (Phase 8 deliverable) produced while zero gates from Pha… | — | 0 |
| `12-r2` | Every §0 number reproduced; validate_paths rejects a wrong dataroot; decision 8 locked | MEASUREMENT: §0 numbers reproduced twice (EV + peer session); BUT validate_paths' version check is a tautol… | P0-9, M-8 | 1 |
| `12-r3` | All common/ tests green; decisions 1+6 locked in config; no stage code written yet | MEASUREMENT: zero tests exist; ~11.5k lines of stage code exist; manifest.py absent | P0-7 | 2 |
| `12-r5` | Diagnostics plausible; determinism byte-identical; decision 2 locked | MEASUREMENT: run exists but was produced by the system interpreter (5.2-r10), is degraded by its own thresh… | P1-8 | 4 |
| `13-r5` | Banner emitted into every run_manifest.json | MEASUREMENT: stage1 run_manifest.json on disk has NO banner field; only measure_vram.py emits one (:888-892) | — | 2 |
| `14-r1` | Contracts-first build order survived (common/ before stages, everything imports from it) | CODE-SITE: 26 cross-stage imports (9-r15); manifest responsibilities in stage0/stage3; Phase 2 gate never met | — | 2 |
| `14-r3` | Role registry over hardcoded checkpoints survived | CODE-SITE: registry exists but stages hardcode checkpoint ids and never call create()/load_model_config (5.… | — | 5 |
| `14-r9` | Test philosophy (silent failure first) and env-gated integration split survived | CODE-SITE: module docstrings DO name silent failures (discipline held in code); zero tests exist (the philo… | P0-7 | 2 |
| `2-r10` | priors/ lives inside the repo (as §2 draws) and §6 writes priors/priors_pilot_v0.json | ARTIFACT: priors written to /home/mt/dhakascenes/out/priors/priors_pilot_v0.json (out_root) | P1-11 | 8 |
| `2-r11` | Repository root is a package named dhakascenes_pilot/ | ARTIFACT: root is /home/mt/Zami with pipeline/, configs/, scripts/, docs/ directly under it; no pyproject.t… | — | 0 |
| `2-r12` | data/ (the dataroot) is not in the repo | ARTIFACT: /home/mt/Zami/nuscenes — 15 GB dataroot INSIDE the repo root (git-ignored but physically present) | P0-9 | 0 |
| `2-r8` | scripts/: probe_substrate.py, check_vram_paper.py, measure_vram.py, run_pilot.py exist | ARTIFACT: scripts/ has probe_substrate.py + measure_vram.py; check_vram_paper.py and run_pilot.py absent | X-3 | 5 |
| `3-r1` | I-1 stand-in: nuScenes sample_data via the Stage 0 allowlist, with the required-channel set enumerated in config | CODE-SITE: pipeline/common/schemas.py:93-105 REQUIRED_CHANNELS | — | 3 |
| `3-r6` | I-7 config snapshot + seed is Phase 2 work, not deferred | ARTIFACT: stage1 run_manifest lacks config hash + git commit (EV artifacts.stage1_manifest.absent_1_9_fields) | P1-8 | 2 |
| `3-r9` | num_lidar_pts counted single-sweep ground-filtered pre-inflation; difference from nuScenes' own field recorded | CODE-SITE: cluster.py:900 counts POST-ghost-filter kept-cluster points (cluster_xyz) while :805 declares th… | — | 8 |
| `5.1-r5` | token_graph_closed: ego_pose, calibrated_sensor, annotation->instance->category, prev/next chains | CODE-SITE: probe.py:578-644 (checks); probe.py:248-251 (orphan bucket) | P0-6 | 3 |
| `5.1-r8` | Partially-present scenes are excluded loudly, not skipped quietly | CODE-SITE: probe.py:894,936-943 vs :929-930 | P0-6 | 3 |
| `5.2-r10` | Stage 1 artifacts were produced under the ano_pipe environment | MEASUREMENT: EV artifacts.stage1_manifest: python_version 3.10.12, numpy_version 1.21.5 (system interpreter… | P1-8 | 4 |
| `5.2-r11` | Stage 1 wrote _SUCCESS and the pipeline can proceed | MEASUREMENT: no _SUCCESS under work/stage1_ingestion/ (EV); ingest.py:985-989 suppresses it on EXIT_DEGRADE… | P1-9 | 4 |
| `5.2-r4` | Ground band 0.3 m, range 40 m and height 4 m pruning applied as stated | CODE-SITE: ingest.py:654-666 (prune_accumulated=False default at :134); :666 height predicate | P1-13 | 4 |
| `5.2-r8` | Sector-fit rejections are recorded with reasons and surviving-plane flags stay honest | CODE-SITE: ingest.py:376-397 (:385 reassigns tilt to the substituted global plane BEFORE the implausible_ti… | P1-13 | 4 |
| `5.2-r9` | Timestamp conversions go through the single conventions.py function | CODE-SITE: ingest.py:545 and :633 multiply/divide timestamps inline; probe.py:507,510 same | X-7 | 4 |
| `5.3-r5` | embedding_ood embeds the whole image (~518x518 CLS token) | CODE-SITE: ood.py:316-327 — stock AutoImageProcessor resizes shortest edge then SQUARE centre-crops; a 1600… | P1-2 | 9 |
| `5.4-r8` | Stage 3 loads its checkpoint through the model config (id+revision+SHA-256) | CODE-SITE: proposals.py:1680-1692 builds CheckpointSpec inline from the stage's own dataclass defaults; loa… | P0-8 | 5 |
| `5.5-r1` | Box prompts converted by the adapter; masks returned at 1600x900, asserted | CODE-SITE: masks.py:524-526 (box prompts carried into the 1024-longest-side space) and :543-548: the assert… | P0-4 | 6 |
| `5.5-r2` | IoA-NMS > 0.5 across overlapping cameras removes duplicate assignment | CODE-SITE: masks.py:355-405 (real, ego angular space) BUT :154 same_class_only=True, and the npz persists s… | M-2 | 6 |
| `5.8-r2` | Effective inter-frame dt recorded in every track record | CODE-SITE: track.py:1316 computes dt_s but no dt key appears in the matched, birth or passthrough rows (:14… | P1-1 | 9 |
| `5.8-r3` | Hungarian matcher, gate-then-score, product combination, birth/death rules, one-to-many handling, occlusion handling all stated | CODE-SITE: track.py:98 + :1007-1012 (Hungarian), :984-1003 (gate, then score, then product), :1104-1108 bir… | M-11 | 9 |
| `5.8-r8` | Stage 7 failure semantics: no silent behaviour change mid-run | CODE-SITE: track.py:1745-1753 -- on ModelUnavailable the driver re-runs end-to-end with appearance_enabled=… | P1-9 | 9 |
| `5.9-r5` | Stage 8 consumes yaw AFTER Stage 7's flip defence | CODE-SITE: inflate.py:962,1062 — manifest/docstring claim 'Stage 7 is not in this chain yet' (false; it exi… | — | 8 |
| `7-r6` | Registry singleton stays; the zero-marginal-VRAM claim is dropped; teardown asserted (memory_allocated ~ 0 between roles) | CODE-SITE: model_interfaces.py:1030-1043 release() only drops refs (the module is torch-free by design); th… | X-2 | 5 |

## ABSENT (57)

| id | claim | evidence | closes | phase |
|---|---|---|---|---:|
| `0-r17` | can_bus is available so I-2 stationary_flag is honestly derivable | CODE-SITE: grep: no module reads can_bus/ | P1-6 | 4 |
| `1.1-r5` | A test loads a real LIDAR_TOP calibrated_sensor record and asserts non-identity with ~-90 deg expected | CODE-SITE: tests/ does not exist | P0-1 | 2 |
| `1.10-r3` | rho records what is counted (GT vs pipeline output) via an explicit rho_source | CODE-SITE: eval_region.py:315 rho(centers_xy_m, spec, *, n_min, frame) — no rho_source parameter; Density h… | M-5 | 2 |
| `1.2-r6` | dt sanity threshold is per-camera, derived from the measured distribution, in config with measured provenance | CODE-SITE: pipeline/common/schemas.py:116-120 | P1-7 | 4 |
| `1.2-r7` | us->ns conversion tested against a real sample_data record pair, not synthetic timestamps | CODE-SITE: tests/ does not exist | X-7 | 4 |
| `1.3-r6` | The two tests that can catch a directionally-wrong-but-valid transform (GT-box reprojection, paint-inside-GT) exist | CODE-SITE: tests/ does not exist | P0-2 | 7 |
| `1.7-r5` | Positive-rejection tests exist: val+pipeline_accepted must raise; human_verified with verified_by=None must raise | CODE-SITE: tests/ does not exist | P0-7 | 2 |
| `1.9-r11` | A _SUCCESS marker is written per stage | MEASUREMENT: EV artifacts: stage0_success_marker == false, stage1 success_marker == false; no _SUCCESS anyw… | P1-9, M-9 | 2 |
| `1.9-r16` | Determinism test: run one scene twice, byte-compare outputs | CODE-SITE: tests/ does not exist | P1-8, M-10 | 4 |
| `1.9-r3` | run_manifest.json carries the resolved config hash | ARTIFACT: EV artifacts.stage1_manifest.absent_1_9_fields includes config_hash | P1-8 | 4 |
| `1.9-r4` | run_manifest.json carries the git commit | ARTIFACT: EV artifacts.stage1_manifest.absent_1_9_fields includes git_commit | P1-8 | 4 |
| `1.9-r8` | run_manifest.json carries the scene partition and its seed | ARTIFACT: EV artifacts.stage1_manifest.absent_1_9_fields includes scene_partition | P1-8, P1-5 | 4 |
| `10-r18` | QA thresholds are config | CODE-SITE: no stage9 | — | 10 |
| `10-r21` | VRAM ceiling and reserve are config | CODE-SITE: models_pilot.yaml absent; BudgetSpec schema exists unfed | P0-8 | 5 |
| `11-r7` | spatial_ok drivable term off by default, flagged no_production_counterpart if enabled; locked at Phase 10 | CODE-SITE: no stage9 to hold the flag | M-13 | 10 |
| `12-r10` | >=3-frame stable track on one real scene; decision 5 locked | ARTIFACT: stage 7 never executed (and unrunnable, 5.8-r7); stage 2 never executed | P1-1, X-4 | 9 |
| `12-r11` | Reproducible from manifest by a second person; banner in README/manifest/figures; decisions 4+7 locked | ARTIFACT: run_pilot.py absent; stage9 absent; probe absent | P1-9, P1-10, P1-15, M-13, M-14 | 10 |
| `12-r6` | 5a paper check passes; 5b measured peaks in manifest; hard stop exercised | ARTIFACT: check_vram_paper.py absent; models yamls absent; no checkpoint downloaded; no GPU manifest | P0-8, X-3 | 5 |
| `12-r7` | Real frames eyeballed; resolution+prompt in every record; no chunking | ARTIFACT: stages 3/4 never executed; no work/stage3* or stage4* outputs | — | 6 |
| `12-r8` | Paint-inside-GT rate stated as a number | ARTIFACT: stage 5 never executed; no number exists anywhere | P0-2 | 7 |
| `12-r9` | Box dimensions sane vs GT as a sanity check | ARTIFACT: stages 6/8 never executed on real lift output (no stage5 output exists) | — | 8 |
| `13-r6` | Every exported figure carries the banner in its header; a bannerless figure cannot be written | CODE-SITE: no figure-export helper exists anywhere | — | 10 |
| `2-r2` | pipeline/common/manifest.py exists (run_manifest, atomic write, _SUCCESS, seeds) | ARTIFACT: pipeline/common/manifest.py — no such file | P1-9, M-9 | 2 |
| `2-r4` | pipeline/stage9_qa/ exists | ARTIFACT: pipeline/stage9_qa — no such directory | — | 10 |
| `2-r5` | probes/indigenous_prompt_probe/ exists as a hard-separated code path | ARTIFACT: probes/ — no such directory | P1-10 | 10 |
| `2-r7` | configs/models_pilot.yaml, models_production.yaml, pipeline_pilot.yaml, taxonomy_probe_indigenous.yaml exist | ARTIFACT: configs/ holds only paths.yaml + taxonomy_pilot_nuscenes.yaml | — | 5 |
| `2-r9` | tests/unit, tests/integration, tests/fixtures exist | ARTIFACT: tests/ — no such directory anywhere in the repo | P0-7 | 2 |
| `3-r7` | stationary_flag is derived from ego speed (can_bus available if better source wanted) | CODE-SITE: grep: no stationary_flag producer in pipeline/ | P1-6 | 4 |
| `4-r7` | spatial_ok drivable term off by default, flagged | CODE-SITE: stage9_qa does not exist | M-13 | 10 |
| `5.10-r1` | Stage 9 exists: gate vector -> auto_accept/flagged/rejected | ARTIFACT: pipeline/stage9_qa — no such directory | P1-12 | 10 |
| `5.10-r2` | Spatial gate runs on pre-inflation dimensions; both dims recorded | CODE-SITE: no stage9; upstream box_measured IS recorded (inflate.py:875-876) so the input exists | P1-12 | 10 |
| `5.10-r3` | Return-count gate counts single-sweep ground-filtered pre-inflation points | CODE-SITE: no stage9; upstream num_lidar_pts has the basis defect (3-r9) | — | 10 |
| `5.2-r7` | Static-structure sharpness check on the accumulated cloud exists | CODE-SITE: no such check in ingest.py or tests/ | — | 4 |
| `5.5-r5` | Stage 4 OOM is a hard stop with stage/role/resolution logged | CODE-SITE: masks.py -- grep: no OutOfMemoryError handler anywhere in the stage (the only broad handlers are… | P1-9 | 6 |
| `7-r10` | Every manifest records the vram_cap block {value_mib, enforced: synthetic, physical_device_mib, device_name} | CODE-SITE: no vram_cap block in measure_vram.py payload (:1096-1161) or any stage manifest writer | P0-8 | 5 |
| `7-r7` | models_pilot.yaml: total_device_mb 4096, measured system_reserve_mb, derived hard_ceiling_mb, stated hard-stop, per-role checkp… | ARTIFACT: configs/models_pilot.yaml -- no such file (configs/ holds paths, taxonomy and two class maps only… | P0-8 | 5 |
| `7-r8` | models_production.yaml: same role names, bigger checkpoints, composite case | ARTIFACT: configs/models_production.yaml — no such file | — | 5 |
| `7-r9` | Every GPU process enforces the synthetic 4096 MiB cap at startup via set_per_process_memory_fraction from DHAKASCENES_VRAM_CAP_MIB | CODE-SITE: repo-wide grep: zero readers of DHAKASCENES_VRAM_CAP_MIB, zero calls to set_per_process_memory_f… | P0-8 | 5 |
| `8-r1` | Separate output root, separate code path (probes/), separate taxonomy file | ARTIFACT: probes/ and configs/taxonomy_probe_indigenous.yaml do not exist; probe_out_root exists in paths.yaml | P1-10 | 10 |
| `8-r2` | Every probe record carries experiment + not_evidence_for: S1 | CODE-SITE: no probe code exists | P1-10 | 10 |
| `8-r3` | A test asserts no indigenous prompt string in the pilot taxonomy and no probe record in main outputs | CODE-SITE: tests/ does not exist | P1-10 | 10 |
| `8-r4` | DHAKASCENES_RUN_INDIGENOUS_PROBE defaults to 0 | CODE-SITE: .env.example declares it; no code reads it | P1-10 | 10 |
| `9-r1` | 3-bucket fixture test exists | CODE-SITE: tests/unit — no such tree | — | 3 |
| `9-r10` | Fabricated similarity matrices + min-point fallback kept; frame/t assertions; ICP relative-vs-absolute; >=3-frame stable-track … | CODE-SITE: tests/ — no such tree | P1-1, M-11 | 9 |
| `9-r11` | Anchor-direction test kept; deliberately-wrong-yaw case; inflation_fraction recorded tests exist | CODE-SITE: tests/unit — no such tree | M-12 | 8 |
| `9-r12` | Threshold boundary + combination kept; NO vacuous provenance test; positive rejections + post-serialization revalidation exist | CODE-SITE: tests/unit — no such tree | P0-7 | 10 |
| `9-r13` | Eval-region closed-form area; contract round-trip; frame/time_base/unit assertions; determinism byte-compare; golden-scene fixt… | CODE-SITE: tests/ — no such tree | P1-8 | 2 |
| `9-r14` | Integration tests are opt-in behind env vars and FAIL (not skip) when the switch is on and data is absent | CODE-SITE: DHAKASCENES_RUN_INTEGRATION / _GPU_TESTS declared in .env.example, read by nothing | P1-9 | 2 |
| `9-r15` | A test asserts nothing under a stage package imports another stage package | CODE-SITE: 26 cross-stage import sites measured (grep) across stages 1-8 + scripts | — | 2 |
| `9-r16` | The >=3-frame stable-track integration test runs Stages 3->6 over ~10 consecutive keyframes of one usable scene | CODE-SITE: tests/integration — no such tree; currently impossible anyway: Stage 7 unrunnable (5.8-r7) and S… | P1-1 | 9 |
| `9-r2` | Truncated .pcd.bin (multiple of 20); dangling ego_pose; missing first-keyframe sweeps; fingerprint-mismatch tests exist | CODE-SITE: tests/unit — no such tree | P0-6 | 3 |
| `9-r4` | Real-record us->ns; sloped multi-sector ground; per-filter diagnostics assertions; static-structure sharpness; scene-start n_sw… | CODE-SITE: tests/unit — no such tree | P1-13 | 4 |
| `9-r5` | OOD subset/no-invented-IDs kept; UMAP excluded from unit tests; flagging on precomputed labels; expected-sample-count check tes… | CODE-SITE: tests/unit — no such tree | X-4 | 9 |
| `9-r6` | Threshold filtering kept; NO builds-the-23-class-prompt-set test; phrases-are-natural-language, span-mapping, original-resoluti… | CODE-SITE: tests/unit — no such tree | X-5 | 6 |
| `9-r7` | One-mask-per-box-in-order kept; original-resolution and box-prompt round-trip (1 px) tests exist | CODE-SITE: tests/unit — no such tree | — | 6 |
| `9-r8` | Known-point->known-pixel kept; GT-box reprojection; paint-inside-GT; behind-camera; near-zero depth; out-of-bounds; overlap det… | CODE-SITE: tests/ — no such tree; nuscenes-devkit 1.1.11 installed as the oracle, pyquaternion pinned and u… | P0-2 | 7 |
| `9-r9` | Noise-rejection kept; NO square-rectangle fixture; non-square 30-deg yaw; yaw round-trip; two-instances-1m; deterministic tie-b… | CODE-SITE: tests/unit — no such tree | P0-5 | 8 |

## UNVERIFIABLE (2)

| id | claim | evidence | closes | phase |
|---|---|---|---|---:|
| `13-r1` | Grounding DINO Tiny treated as ~172 M params / ~690 MB FP32 (not 172 MB) | CODE-SITE: no models_pilot.yaml or check_vram_paper.py exists to carry either figure; marked [VERIFY before… | P0-8 | 5 |
| `13-r2` | SAM ViT-B treated as ~91 M params / ~375 MB (params/MB unswapped); DINOv2-S and MobileSAM figures carried correctly | CODE-SITE: same — no artifact carries the figures yet | P0-8 | 5 |

## PLAUSIBLE (135)

| id | claim | evidence | closes | phase |
|---|---|---|---|---:|
| `0-r15` | Stage 0 is a verification stage (proves completeness), not a salvage stage | CODE-SITE: pipeline/stage0_data_probe/probe.py (predicate design) | P0-6 | 3 |
| `1.1-r1` | Ego frame is canonical for all cross-stage geometry | CODE-SITE: pipeline/common/conventions.py:76-91 | P0-1 | 2 |
| `1.1-r2` | Every geometric record carries an explicit frame field, validated on write; no frame => invalid, not defaulted | CODE-SITE: pipeline/common/schemas.py:169-175 | P0-1 | 2 |
| `1.1-r3` | Stage 1 applies T_ego_lidar exactly once; no later stage re-applies it | CODE-SITE: pipeline/stage1_ingestion/ingest.py (single transform site) | P0-1 | 4 |
| `1.1-r4` | nuScenes global frame is named nuscenes_global, never global | CODE-SITE: pipeline/common/conventions.py:76-78 | P0-1 | 2 |
| `1.1-r6` | Eval region R1 wedge is computed on ego-frame points (not LiDAR-frame) | CODE-SITE: pipeline/common/eval_region.py:272-277 | P0-1 | 7 |
| `1.10-r2` | in_region(x,y,spec) is the only place deciding membership; rho stays count-over-area | CODE-SITE: eval_region.py:261-296,315-351; consumers ingest.py:665, lift.py:693, cluster.py:907-908, priors… | M-6 | 2 |
| `1.10-r4` | A CAM_FRONT-only run is legal but must record coverage_config: R1 | CODE-SITE: eval_region.py:97-106 R1/R2 region construction; camera_subset in stage1 manifest | P1-4 | 2 |
| `1.2-r1` | Every temporal record carries time_base in {unix_us, unix_ns, gps_ns} | CODE-SITE: pipeline/common/schemas.py:178-195, conventions.py:101 | P1-7, X-7 | 2 |
| `1.2-r2` | us->ns conversion happens in exactly one function; nothing else multiplies or divides a timestamp | CODE-SITE: pipeline/common/conventions.py:122-146 | X-7 | 2 |
| `1.2-r3` | Pilot stores unix_ns int64 and does not claim GPS time; divergence from spec §2.3 is a declared deviation | CODE-SITE: pipeline/common/schemas.py:178-195 | X-7 | 2 |
| `1.2-r4` | The temporal anchor is LIDAR_TOP sample_data.timestamp, never sample.timestamp | CODE-SITE: pipeline/stage1_ingestion/ingest.py (anchor selection) | P1-7 | 4 |
| `1.2-r5` | Every geometric/temporal schema field carries a unit suffix; sigma_yaw stored as sigma_yaw_rad | CODE-SITE: pipeline/common/schemas.py:479-482,854-858 | X-8 | 2 |
| `1.3-r1` | project_lidar_to_image() is the single implementation of the four-hop chain using TWO ego poses | CODE-SITE: pipeline/common/conventions.py:318-395 | P0-2 | 2 |
| `1.3-r2` | T_cam_ego is obtained by inverting the nuScenes sensor->ego extrinsic | CODE-SITE: pipeline/common/conventions.py:393-395 | P0-2 | 2 |
| `1.3-r3` | K is applied after extrinsics, never folded into them | CODE-SITE: pipeline/common/conventions.py:404-406 | P0-2 | 2 |
| `1.3-r4` | z <= 0 culled before the divide; near-zero-depth guard present | CODE-SITE: pipeline/common/conventions.py:397-406, 287-291 | P0-2 | 2 |
| `1.3-r5` | Out-of-bounds pixels are handled | CODE-SITE: pipeline/common/conventions.py:408-413 | P0-2 | 2 |
| `1.4-r2` | Ground plane is FIT on the accumulation and APPLIED to the single-sweep cloud; the lift consumes single-sweep | CODE-SITE: pipeline/stage1_ingestion/ingest.py (fit/apply split) | P0-3, X-1 | 4 |
| `1.4-r3` | num_lidar_pts is counted on single-sweep, pre-inflation | CODE-SITE: pipeline/common/schemas.py:861,886-895 | P0-3 | 8 |
| `1.4-r4` | Every cloud artifact records cloud_kind, n_sweeps_actual, window_ns | CODE-SITE: pipeline/common/schemas.py:627-664 | X-1 | 4 |
| `1.5-r1` | Every 2D quantity crossing a stage boundary is absolute pixels xyxy at original 1600x900 | CODE-SITE: pipeline/common/schemas.py:107-111,591-598 | P0-4 | 6 |
| `1.5-r2` | Each model adapter owns its own forward and inverse transform; transforms never leak into stage code | CODE-SITE: pipeline/common/model_interfaces.py:465, stage3_proposals/proposals.py:695 | P0-4 | 6 |
| `1.5-r3` | No square resize of non-square imagery anywhere; letterbox or resize-shortest-side only | CODE-SITE: grep: no square-resize site in pipeline/ (mechanical check FP-1 in check_conformance.py) | P0-4 | 6 |
| `1.5-r4` | Masks are returned at 1600x900 before Stage 5 indexes them — asserted, not assumed | CODE-SITE: pipeline/stage4_masks/masks.py (assertion site, §5.5 rows) | P0-4 | 6 |
| `1.5-r5` | Multi-camera contest rule: class from camera whose principal axis is closest to point bearing; ties by fixed priority list in c… | CODE-SITE: pipeline/stage5_lift/lift.py (see 5.6-r4) | P0-4 | 7 |
| `1.5-r6` | IoA-NMS > 0.5 across overlapping cameras is restored | CODE-SITE: pipeline/stage4_masks/masks.py (see 5.5-r2) | M-2 | 6 |
| `1.6-r1` | DBSCAN runs per mask instance; class-conditional refers only to which epsilon is used | CODE-SITE: pipeline/stage6_cluster/cluster.py (see 5.7-r1) | P0-5 | 8 |
| `1.6-r2` | Keep-largest-cluster is the reprojection-ghost filter within one instance's points | CODE-SITE: pipeline/stage6_cluster/cluster.py (see 5.7-r2) | P0-5 | 8 |
| `1.6-r3` | Deterministic tie-break for equal-size clusters: lowest mean range, then lowest first-point index under canonical sort | CODE-SITE: pipeline/stage6_cluster/cluster.py (see 5.7-r3) | P0-5, P1-8 | 8 |
| `1.7-r1` | validate() -> [errors] kept AND a single raising write_records() boundary through which all persistence goes, validating on wri… | CODE-SITE: pipeline/common/schemas.py:969-975,1017-1090 | P0-7, X-9 | 2 |
| `1.7-r2` | Both directions of the provenance invariant are stated: val/test => not pipeline_accepted, and human source => verified_by set … | CODE-SITE: pipeline/common/schemas.py:800-818,911-917 | P0-7, X-9 | 2 |
| `1.7-r3` | Pilot-wide allow_human_provenance: false makes any human-provenance record a hard error | CODE-SITE: pipeline/common/schemas.py:123-135,806-812 | X-9 | 2 |
| `1.7-r4` | tier and source are different fields with different rules; nothing conflates them | CODE-SITE: pipeline/common/schemas.py:84-89,780-792 | X-9 | 2 |
| `1.8-r1` | One configs/paths.yaml resolved once into a Paths object with the six declared fields | CODE-SITE: pipeline/common/paths.py:87-96,143-179 | P0-9, M-8 | 1 |
| `1.8-r2` | validate_paths() asserts dataroot contains the version dir, samples/, sweeps/, and is readable | CODE-SITE: pipeline/common/paths.py:213-225 | P0-9 | 1 |
| `1.8-r4` | validate_paths() asserts commonpath disjointness between dataroot and both write roots | CODE-SITE: pipeline/common/paths.py:254-278 | P0-9 | 1 |
| `1.8-r5` | Dataroot is read-only by convention, with a test asserting no module opens a dataroot path for writing | CODE-SITE: pipeline/common/paths.py:316-327 | P0-9 | 2 |
| `1.8-r9` | The dataroot fingerprint appears in every stage manifest | CODE-SITE: stage1 manifest 'upstream' block carries metadata_fingerprint; later-stage manifest writers cite it | P0-9 | 4 |
| `1.9-r1` | One global seed threaded into RANSAC, UMAP, and any sampling; recorded | CODE-SITE: stage1 manifest seed == 20260812; ingest.py config block | P1-8, M-10 | 4 |
| `1.9-r10` | Atomic writes (temp + rename) for all artifacts | CODE-SITE: pipeline/stage0_data_probe/probe.py:739 write_json_atomic; stage3_proposals/proposals.py:752 wri… | P1-9, M-9 | 2 |
| `1.9-r12` | Downstream stages refuse to start when the upstream manifest is missing or its input fingerprint differs | CODE-SITE: stage1_ingestion/ingest.py:756-777 load_allowlist refusal; UpstreamRefusal class in proposals.py… | P1-9, M-9 | 4 |
| `1.9-r13` | Per-scene isolation: one scene's failure lands in failures.json with the failing predicate and does not abort the run | CODE-SITE: stage code failure collection (per-stage rows) | P1-9 | 10 |
| `1.9-r14` | Idempotent re-run keyed on the manifest | CODE-SITE: per-stage manifest keying (varies by stage) | P1-9 | 10 |
| `1.9-r15` | OOM is a hard stop with stage/role/resolution logged; never a silent retry at lower resolution | CODE-SITE: scripts/measure_vram.py:830-834; model_interfaces.py:1060-1093 VramBudget (on_exceed='hard_stop'… | P1-9 | 5 |
| `1.9-r9` | run_manifest.json carries priors version + source, W_acc count and duration, image resolution + prompt config, VRAM peaks, per-… | ARTIFACT: stage1 manifest carries w_acc + per-filter I/O counts; model-stage fields have no producer yet | P1-8 | 10 |
| `10-r10` | Prompt chunking default-forbidden is config | CODE-SITE: proposals.py:155 allow_prompt_chunking=False with §5.4 provenance | — | 6 |
| `10-r13` | Per-class DBSCAN epsilon and min_samples are config/derived | CODE-SITE: epsilon derived from priors (5.7-r3); min_samples dataclass default 3 (cluster.py:156) + CLI --m… | X-6 | 8 |
| `10-r20` | Global seed is config, recorded | CODE-SITE: global_seed 20260812 dataclass default, recorded in stage1 manifest | P1-8, M-10 | 2 |
| `10-r23` | Every inherited-unvalidated value (0.3/40/4/1-in-10/0.40/15/>=5/2x/reserve) is flagged in config | CODE-SITE: provenance strings on the dataclass defaults flag most; 2x spatial multiplier + reserve have no … | — | 10 |
| `10-r8` | Image resolution and resize policy are config/recorded | CODE-SITE: schemas.py:107-111 constants (measured provenance); resize_policy recorded per record (proposals… | P1-14 | 6 |
| `11-r1` | R2, all six ring cameras; CAM_FRONT-only runs record R1; locked at Phase 2 | CODE-SITE: stage1 manifest coverage_config R2 + 6-camera subset; eval_region R1/R2 machinery | P1-4 | 2 |
| `11-r2` | W_acc preserves duration (~0.5 s ~= 10 sweeps); both count and duration recorded; locked at Phase 4 | CODE-SITE: ingest.py:118-119,730-734; stage1 manifest w_acc fields | — | 4 |
| `11-r4` | nuScenes metrics forbidden as quality claims; diagnostic_only labelling; locked at Phase 10 | CODE-SITE: no metric computation exists yet anywhere in the repo (grep: no mAP/AP computation) | P1-15, M-14 | 10 |
| `11-r6` | Waived components confirmed per §4; IoA-NMS and yaw-consistency restored | CODE-SITE: see 4-r1..4-r8 | M-4 | 2 |
| `12-r4` | Probe run against real dataroot; usable count known and printed; partition written | MEASUREMENT: usable_scenes.json (10 scenes) + partition on disk; but the phase's tests do not exist, so RUN… | P0-6 | 3 |
| `13-r3` | Weight-file size is never presented as inference VRAM; max_memory_allocated is not the occupancy measure | CODE-SITE: measure_vram.py:841-846,1113-1117 (reserved-peak + free-delta; allocated explicitly not used) — … | P0-8, X-3 | 5 |
| `13-r7` | The eleven claim-table rows are enforced: supported claims quotable only with their conditions, unsupported claims nowhere asse… | CODE-SITE: README.md reproduces the claim boundary; no repo text asserts an unsupported claim (reviewed doc… | P1-15 | 10 |
| `14-r10` | comprehensive.md §3-§5 explicitly deferred, no invented substitute | CODE-SITE: no sensor-rig/capture/PPK code exists anywhere | — | 0 |
| `14-r2` | Stage 0 pilot-only with loud exclusion survived | CODE-SITE: probe.py predicates + EXIT_INCOMPLETE | — | 3 |
| `14-r4` | Provenance invariant carried from the first line of code | CODE-SITE: schemas.py Provenance machinery predates stage code per structure | — | 2 |
| `14-r5` | eval_region.py as single E/rho implementation survived | CODE-SITE: 1.10-r2 | M-6 | 2 |
| `14-r6` | MobileSAM propagation loss named a capability gap; probe not called S1; OOD not called discovery | CODE-SITE: masks.py:2217-2223 capability_gaps recorded in the run manifest (+ the propagation_note :2044-20… | — | 9 |
| `14-r7` | Weight-size != inference VRAM and verified flag survived | CODE-SITE: measure_vram.py method + CheckpointSpec.verified | — | 5 |
| `14-r8` | I-2 downgraded honestly (nuScenes poses never presented as PPK) | CODE-SITE: 3-r2 | P1-6 | 2 |
| `3-r10` | attribute and visibility have no producer, declared out of scope | CODE-SITE: schemas.py:900-903 hard-error if populated | M-1 | 2 |
| `3-r2` | I-2 downgrade is mechanised: sigma/fix_type null never zero, pose_source pinned, quality_known false, consumers fail closed | CODE-SITE: pipeline/common/schemas.py:477-545 (defaults, both-direction check, exact-0.0 rejection, require… | P1-6 | 2 |
| `3-r3` | I-3: keyframe pack + both clouds + n_sweeps_actual + window_ns + undistorted:true/nuscenes_native | CODE-SITE: schemas.py:627-664 CloudArtifact, 571-607 CameraObservation; ingest.py:711-719 | P0-3 | 4 |
| `3-r4` | I-4: pre-labels + gate vector + tier with source=pipeline; yaw->quaternion explicit and tested | CODE-SITE: schemas.py:843-895 AnnotationRecord + GateVector; conventions.py:191-196 quaternion_from_yaw_rad | — | 8 |
| `3-r5` | I-5 simulated: both directions of the invariant stated and positively tested | CODE-SITE: schemas.py:800-818,911-917 | — | 2 |
| `3-r8` | size is [w,l,h] nuScenes order, stated in the schema and asserted | CODE-SITE: schemas.py:833-836 size_wlh_m; cluster.py:591 size_order w,l,h; cluster.py:571 + :646-649 w<=l b… | — | 8 |
| `4-r1` | Stage 10 attribute pre-fill is waived, leaving I-4 attribute unproduced, declared | CODE-SITE: schemas.py:900-903 | M-1, M-4 | 2 |
| `4-r2` | IoA-NMS across cameras is restored | CODE-SITE: pipeline/stage4_masks/masks.py:355-405 ioa_nms_across_cameras; threshold 0.5 is MaskConfig.ioa_t… | M-2 | 6 |
| `4-r3` | SAM 2.1 mask propagation waived as a capability gap, stated plainly -- SUPERSEDED (C27): propagation is implemented and exercis… | CODE-SITE: SUPERSEDED by pipeline/stage3b_track2d/track2d.py (Sam3VideoTrackerAdapter :381-593; Sam31Multip… | M-3 | 6 |
| `4-r4` | Forward-backward smoothing waived | CODE-SITE: track.py — no smoothing pass exists | M-3 | 9 |
| `4-r5` | Yaw-consistency enforcement along tracks restored | CODE-SITE: track.py:741-772 disambiguate_yaw, applied pre-KF-update :1305-1310 (the filter update is :1312)… | — | 9 |
| `4-r6` | Group/ignore boxes waived; GroupAnnotation is shape-only with no producer or consumer | CODE-SITE: schemas.py:921-966 | M-4 | 2 |
| `4-r8` | Undistortion is a declared no-op: undistorted=true, method=nuscenes_native recorded | CODE-SITE: schemas.py:571-607 (hard error on any other method string) | — | 4 |
| `5.1-r1` | Six executable predicates exist, each emitted by name when it fails | CODE-SITE: probe.py:126-133 PREDICATES tuple; per-scene failing list :182-187 | P0-6, M-15 | 3 |
| `5.1-r2` | channels_complete over the declared required-channel set; RADAR excluded and the exclusion declared | CODE-SITE: schemas.py:93-105 REQUIRED_CHANNELS; probe.py:866 | M-15 | 3 |
| `5.1-r3` | files_parse: pcd size %20, sane size band, JPEG SOI+EOI markers | CODE-SITE: probe.py:374-399 | P0-6 | 3 |
| `5.1-r4` | sweeps_cover_window: every LiDAR sweep within W_acc of every keyframe exists | CODE-SITE: probe.py:496-554 | P0-6 | 3 |
| `5.1-r7` | Hard stops on version mismatch, zero usable scenes, dataroot missing samples/ or version dir | CODE-SITE: probe.py:775-783,819 raise HardStop | P0-6 | 3 |
| `5.2-r1` | LiDAR transformed to ego exactly once; both clouds ego-frame; accumulation uses each sweep's own ego_pose | CODE-SITE: ingest.py:520-537,639-646 (T_ego_lidar once per sweep + anchor); :528-537 per-sweep poses | P0-1, P0-3 | 4 |
| `5.2-r2` | W_acc is 0.5 s duration (~10 sweeps) from config; nominal + actual recorded; no hardcoded 5 | CODE-SITE: ingest.py:118-119,633-634,730-734 | — | 4 |
| `5.2-r3` | Sector RANSAC fitted on accumulation, plane applied to single-sweep, single-sweep goes forward | CODE-SITE: ingest.py:649,654-656,683; lift.py:206-208 refuses non-single-sweep | P0-3, X-1 | 4 |
| `5.2-r6` | RANSAC sector count, threshold, iterations, seed are config with provenance; seeded from global seed | CODE-SITE: ingest.py:123-152,224-232 rng = default_rng([seed, token_seed, sector]) | P1-8 | 4 |
| `5.3-r1` | Outlier decisions in embedding space (GLOSH on raw 384-D); UMAP visualisation only | CODE-SITE: ood.py:420-450 (fit_predict on raw vectors; flag_ood has no UMAP argument; umap_coords only writ… | — | 9 |
| `5.3-r2` | Sampling over images (242 expected), count stated and checked; n_neighbors < n_samples | CODE-SITE: ood.py:207-256,511,366-371 | X-4 | 9 |
| `5.3-r3` | Stage 2 is a branch feeding nothing downstream | CODE-SITE: grep: no pipeline import of stage2 outputs | X-4 | 9 |
| `5.3-r4` | UMAP/HDBSCAN parameters from config; random_state threaded from global seed | CODE-SITE: ood.py:118-125,373-377,415-419 | P1-8 | 9 |
| `5.4-1-r2` | Stage 3b's outputs land only at keyframes: the 12 Hz sweeps carry 2D identity only, and no LiDAR anchor or eval quantity is cre… | CODE-SITE: track2d.py:24-30 states the rule; :1663-1679 emits exactly one output row per Stage 3 row, :2520… | — | 6 |
| `5.4-r1` | Prompts come from the taxonomy mapping; no dotted category name can reach a prompt | CODE-SITE: proposals.py:242,548; _DOTTED_CATEGORY_RE tripwire model_interfaces.py:275, re-checked :446 | — | 6 |
| `5.4-r2` | Phrase-span mapping uses tokenizer offset_mapping with hard errors on span crossings | CODE-SITE: proposals.py:654-715 build_phrase_span_map (tokenizer offset_mapping -> phrase index; PhraseSpan… | — | 6 |
| `5.4-r3` | Per-class thresholds with a global default (dict interface), not a single scalar | CODE-SITE: proposals.py:651; thresholds.get(name, default) model_interfaces.py:306 | X-5 | 6 |
| `5.4-r4` | Deduplication of overlapping proposals with deterministic order | CODE-SITE: proposals.py:464-524 | — | 6 |
| `5.4-r5` | Output absolute pixels xyxy at 1600x900; resolution + prompt configuration recorded in every record | CODE-SITE: proposals.py:432-445,820-829 (image_size_px, model_input_size_px, resize_policy, prompt block wi… | P1-14, P0-4 | 6 |
| `5.4-r6` | Prompt chunking is forbidden by default; enabling requires re-tune provenance | CODE-SITE: proposals.py:155,280-284; model_interfaces.py:289-295; no chunking code path exists | — | 6 |
| `5.4-r7` | Aspect-preserving transform asserted per image; adapter owns forward+inverse | CODE-SITE: proposals.py:705-728 (_assert_aspect_preserved, ASPECT_TOLERANCE 0.02, _assert_unpadded); invers… | P0-4 | 6 |
| `5.5-r3` | One mask per box, same order — the misassignment tripwire | CODE-SITE: masks.py:538-542, :739-743, :1221-1225 -- the count-mismatch raise, once per adapter; assert_val… | — | 6 |
| `5.5-r4` | Adapter accepts temporal state + window and ignores them (MobileSAM) | CODE-SITE: masks.py:489-504 MobileSamAdapter.segment accepts state/window and documents ignoring them, retu… | P1-3, X-10 | 6 |
| `5.5-r7` | mask_2d provider routing is exact-match with refusal: an unrecognised model_id raises rather than defaulting to a provider nobo… | CODE-SITE: masks.py:1582-1605 infer_mask_provider -- 'facebook/sam3' matched exactly :1592-1594, prefix tab… | — | 6 |
| `5.5-r8` | The stage-4 manifest records the resolved mask_2d provider, and the checkpoint sha256 whenever the adapter supplies one | CODE-SITE: masks.py:2181 records the resolved provider; :2196-2201 prefers the adapter's stream hash (getat… | — | 6 |
| `5.5-r9` | Stage 4 carries per-box Stage 3b provenance (box_source, track_id, n_propagated_hops) onto its candidates, and refuses a presen… | CODE-SITE: masks.py:1826-1851 optional_c27_array -- absent is legal, len != n_boxes raises UpstreamRefusal … | — | 6 |
| `5.6-r1` | Lift consumes ground-filtered single-sweep ego-frame points via the single projection implementation | CODE-SITE: lift.py:772 (single_sweep only, cfg-validated :206), :429 sole call to project_lidar_to_image | P0-2, P0-3 | 7 |
| `5.6-r2` | Per-camera frusta unioned for R2 coverage | CODE-SITE: lift.py:548,579,593 (n_cameras_visible union + union_fraction) | — | 7 |
| `5.6-r3` | z<=0 cull before divide, near-zero depth, out-of-bounds, deterministic overlap — all present | CODE-SITE: conventions.py:399-406; lift.py:438,442,477,484-487 (partition assertion: counts sum to n_input) | P0-2 | 7 |
| `5.6-r4` | Contest rule: camera with principal axis closest to point bearing wins; ties to fixed priority list | CODE-SITE: lift.py:380-393,533,561 (camera-frame cosine z/|p|; strict > over sorted priority order = determ… | P0-4 | 7 |
| `5.6-r5` | Painted-point counts are single-sweep pre-inflation; in-region check meaningful | CODE-SITE: lift.py:677-693 | — | 7 |
| `5.6-r6` | A mid-run refusal cannot leave a stale _SUCCESS over partially rewritten scenes | CODE-SITE: cluster.py:1124 and lift.py:872 call clear_markers(out_dir) inside run(), before the first write… | P1-9 | 7 |
| `5.7-r1` | DBSCAN per mask instance; class picks epsilon only; keep-largest is the ghost filter | CODE-SITE: cluster.py:996 (per-instance loop in cluster_keyframe), :782 (eps by class via priors.eps_bev), … | P0-5 | 8 |
| `5.7-r2` | Deterministic DBSCAN: hand-implemented, canonical point order, §1.6 tie-break | CODE-SITE: cluster.py:357-396 (hand-written DBSCAN, ascending seed order :377), :325-353 (uniform grid retu… | P0-5, P1-8 | 8 |
| `5.7-r3` | Epsilon from the §7.2 formula (0.6 x mean footprint diagonal) via priors; no hardcoded table | CODE-SITE: priors.py:582+:145; cluster.py:782 (priors.eps_bev by class); eps_fallback stamped as eps_source… | X-6 | 8 |
| `5.7-r4` | L-shape fit emits [w,l,h] with w<=l enforced and yaw_ambiguous flagged for near-squares | CODE-SITE: cluster.py:646-649 (w<=l re-checked with a raise after the min-extent clamp), :651-660 (near_squ… | — | 8 |
| `5.7-r5` | Yaw asserted against conventions.py definition | CODE-SITE: cluster.py:689-746 _assert_geometry (re-projection + quaternion round-trip via conventions helpers) | — | 8 |
| `5.7-r6` | Stage 6 refuses priors derived from the wrong scene subset | CODE-SITE: cluster.py:1067-1074 refuses an empty priors fingerprint, :1075-1082 refuses a fingerprint misma… | P1-5 | 8 |
| `5.8-r1` | Predict-then-match: previous box propagated by velocity before IoU | CODE-SITE: track.py:930-951 predicted_local_box (Kalman propagation over dt), :1073 the predicted boxes are… | P1-1 | 9 |
| `5.8-r10` | reid_embedding uses crop semantics distinct from Stage 2's whole-image path | CODE-SITE: track.py:775-880 Dinov2ReidAdapter (a separate adapter), :834-860 embed_crops with semantics='cr… | P1-2 | 9 |
| `5.8-r4` | IoU gate is config with 'derived for 2 Hz' provenance; minimum crop size stated | CODE-SITE: track.py:185 + :252-256 (iou_gate 0.05 with the 'derived for 2 Hz' provenance), :191 + :260-261 … | P1-1 | 9 |
| `5.8-r5` | ICP with a declared frame; velocity semantics recorded; relative-vs-absolute distinguished | CODE-SITE: track.py:569-644 icp_register, frame declared :225 and recorded per row :1414 + manifest :1640-1… | M-11 | 9 |
| `5.8-r6` | Kalman fallback fully specified: CV state, dt from timestamps, config noise, stated init, <15-point trigger | CODE-SITE: track.py:163-164 (sparse-guard reason strings), :467-501 (CV state + predict over dt), :208-221 … | M-11 | 9 |
| `5.8-r7` | Stage 7 runs end-to-end on real upstream outputs | CODE-SITE: track.py:1205-1212 (birth) and :1323-1327 (update) join det['points_path'] against stage5_dir, t… | P1-1 | 9 |
| `5.8-r9` | Yaw-consistency (180-degree flip defence) enforced along tracks | CODE-SITE: track.py:741-772 disambiguate_yaw, :1303-1310 (predicted-velocity reference, applied pre-update)… | — | 9 |
| `5.9-r1` | Inflation trigger, blend, per-axis applicability, clamp are explicit with provenance | CODE-SITE: inflate.py:155-243 (trigger/trigger_max_points/blend/axes/max_growth_ratio/max_growth_m/clamp_si… | M-12 | 8 |
| `5.9-r2` | Near face anchored, far face grows, growth through the sensor guarded | CODE-SITE: inflate.py:430,449,456,612-615 (three guards raise InflationContractError) | — | 8 |
| `5.9-r3` | Every box records inflated + inflation_fraction; both pre- and post-inflation dims recorded | CODE-SITE: inflate.py:826-827,875-878 (box + box_measured side by side) | P1-12, M-12 | 8 |
| `5.9-r4` | Inflation pulls toward the priors file consumed through the A.4 interface with source recorded | CODE-SITE: inflate.py:546,763-777 (load_priors, sha256+fingerprint agreement, prior.source into every ledger) | P1-11 | 8 |
| `6-r1` | Priors derived from sample_annotation.json, priors scene subset only | CODE-SITE: priors.py:481,672,134 (subset_scene_names over PARTITION['priors']) | P1-5 | 8 |
| `6-r2` | Priors derived under the same E and 40 m constraints the pipeline operates under | CODE-SITE: priors.py:489-495 (min_lidar_pts >= 5 drop + in_region with R2/40 m) | P1-11 | 8 |
| `6-r4` | The release builder rejects any priors file whose source is not S0 | CODE-SITE: priors.py:394 assert_release_source(); release_guard field in the artifact | P1-11 | 8 |
| `7-r1` | Four role Protocols exist: embedding_ood, proposal_2d, mask_2d, reid_embedding | CODE-SITE: model_interfaces.py:698 EmbeddingOOD, :721 Proposal2D, :747 Mask2D, :824 ReidEmbedding; ROLES :121 | P1-3, X-10 | 5 |
| `7-r2` | mask_2d takes optional temporal state + window from day one | CODE-SITE: model_interfaces.py:754 | P1-3, X-10 | 5 |
| `7-r3` | One provider may register against multiple roles (composite) | CODE-SITE: model_interfaces.py:935-940 register(roles=...), :923+:926-928 ProviderEntry.roles and .composit… | X-10 | 5 |
| `7-r4` | proposal_2d may optionally return masks (Stage 4 pass-through) | CODE-SITE: model_interfaces.py:399,724; _check_masks :351 | X-10 | 5 |
| `7-r5` | Preprocessing belongs to the role: whole-image vs crop semantics distinguished and named | CODE-SITE: model_interfaces.py:621-623 (semantics + non-empty preprocessing required) | P1-2 | 5 |

## CONFORMS (39)

| id | claim | evidence | closes | phase |
|---|---|---|---|---:|
| `0-r1` | Metadata is v1.0-mini with 13 tables | MEASUREMENT: EV metadata_tables.count == 13 | — | 1 |
| `0-r10` | Timestamps are 16-digit microseconds, Unix epoch | MEASUREMENT: EV timestamps.digits_min == digits_max == 16 | P1-7 | 1 |
| `0-r11` | Per-camera dt medians (ms): FL -43.06, F -35.44, FR -27.53, BR -19.95, B -10.36, BL -0.48; every camera fires before the LiDAR … | MEASUREMENT: EV camera_dt_ms | P1-7, X-7 | 1 |
| `0-r12` | 10 scenes = 5 Singapore / 5 Boston with 3 night scenes (1077, 1094, 1100) | MEASUREMENT: EV scenes[] locations + night flags | — | 1 |
| `0-r13` | maps/ + map expansion and can_bus/ extras are present | ARTIFACT: /home/mt/Zami/nuscenes/maps, /home/mt/Zami/nuscenes/can_bus | — | 1 |
| `0-r14` | Token graph closed: no dangling ego_pose / calibrated_sensor / instance references | MEASUREMENT: EV dangling_tokens all == 0 | P0-6 | 1 |
| `0-r16` | A nuscenes_category -> prompt_phrase mapping table exists as a mandatory artifact | ARTIFACT: configs/taxonomy_pilot_nuscenes.yaml | — | 6 |
| `0-r2` | 10 scenes / 404 keyframes / 31,206 sample_data records | MEASUREMENT: EV counts.{scenes,samples,sample_data} | — | 1 |
| `0-r3` | 0 missing files across all 12 channels | MEASUREMENT: EV missing_files.count == 0, channels.count == 12 | — | 1 |
| `0-r4` | 23 categories with dotted hierarchical names; 911 instances; 18,538 annotations | MEASUREMENT: EV counts.{categories,instances,sample_annotations}, category_names_dotted | — | 1 |
| `0-r5` | Image resolution 1600x900 uniform across all keyframe camera images | MEASUREMENT: EV keyframe_images: 2424 images, dims_from_fields and dims_from_jpeg_headers both exactly {(16… | — | 1 |
| `0-r7` | LIDAR_TOP extrinsic rotation is non-identity, yaw ~= -89.883 deg | MEASUREMENT: EV lidar_top_extrinsic: is_identity=false, yaws {-90.031, -89.883} | P0-1 | 1 |
| `0-r8` | LiDAR cadence median 49.79 ms (~20 Hz); 9.74 sweeps per keyframe (3,935/404) | MEASUREMENT: EV lidar.{cadence_median_ms,sweeps_per_keyframe,sweep_records} | — | 1 |
| `0-r9` | Point record is 20 bytes (5 x float32); example sweep 34,688 points | MEASUREMENT: EV lidar.files_size_mod20_nonzero == 0 over all 3,935 files; example_points == 34688 | — | 1 |
| `1.4-r1` | Both clouds are produced: single-sweep and accumulated | MEASUREMENT: work/stage1_ingestion/clouds/ contains both kinds for all 10 scenes (1.7 GB); CloudArtifact cr… | P0-3, X-1 | 4 |
| `1.8-r7` | The recorded fingerprint matches the current dataroot (allowlist binding is live) | MEASUREMENT: EV artifacts.usable_scenes.fingerprint_matches_remeasure == true (4c5a5cfe...d7891) | P0-9 | 3 |
| `1.8-r8` | data/ is git-ignored; work_root and out_root default outside the repo | ARTIFACT: .gitignore (data/, work/, out/, probe_out/, /nuscenes/); configs/paths.yaml:36-43 (/home/mt/dhaka… | P0-9 | 1 |
| `1.9-r2` | run_manifest.json carries the seed | ARTIFACT: work/stage1_ingestion/run_manifest.json seed == 20260812 | P1-8 | 4 |
| `1.9-r7` | run_manifest.json carries the dataroot fingerprint and usable_scenes.json reference | ARTIFACT: stage1 manifest upstream.metadata_fingerprint == 4c5a5cfe... + usable_scenes_spec | P1-8 | 4 |
| `10-r1` | Paths (dataroot, meta_root, work_root, out_root, probe_out_root, version) are config with provenance | MEASUREMENT: configs/paths.yaml — all six present with rationale comments; loaded by every stage | — | 1 |
| `10-r11` | Per-class confidence thresholds + default are config | MEASUREMENT: taxonomy yaml ships thresholds:{} + default_threshold: 0.40 (read from the file) | X-5 | 6 |
| `10-r9` | Prompt taxonomy file is config | MEASUREMENT: configs/taxonomy_pilot_nuscenes.yaml — 23 keys verified byte-for-byte against category.json | — | 6 |
| `11-r3` | Disjoint stratified partition exactly as specified; recorded in the manifest; locked at Phase 3 | MEASUREMENT: EV artifacts.usable_scenes.partition == plan's exact subsets; stratification verified (both lo… | P1-5, M-7 | 3 |
| `11-r8` | One root, /home/mt/Zami/nuscenes, meta at v1.0-mini; dataroot/meta_root separate fields; locked at Phase 1 | MEASUREMENT: EV dataroot + configs/paths.yaml (dataroot == meta_root, separate keys) | — | 1 |
| `13-r4` | Banner present in README | MEASUREMENT: README.md:3-5 carries the exact banner text (created by peer session, commit 0900a7a) | — | 10 |
| `2-r1` | pipeline/common/conventions.py, schemas.py, eval_region.py, paths.py, model_interfaces.py exist | ARTIFACT: pipeline/common/ — five modules present (428/1126/361/351/1171 lines) | — | 2 |
| `2-r3` | stage0_data_probe .. stage8_inflate packages exist | ARTIFACT: pipeline/stage{0..8}* present, ~11.5k lines total | — | 2 |
| `2-r6` | configs/: paths.yaml and taxonomy_pilot_nuscenes.yaml exist | ARTIFACT: configs/paths.yaml (43 lines), configs/taxonomy_pilot_nuscenes.yaml (69 lines) | — | 1 |
| `5.1-r6` | usable_scenes.json records dataroot realpath, fingerprint, required channels, W_acc; probe_report.json has per-scene failing pr… | MEASUREMENT: EV artifacts.usable_scenes (all fields verified on the real artifact); partition matches §11 d… | P1-5, M-7 | 3 |
| `5.2-r5` | Per-filter, per-sector point-count diagnostics are a first-class output | MEASUREMENT: work/stage1_ingestion/run_manifest.json totals + per-sector ledgers on the real run; ingest.py… | P1-13 | 4 |
| `5.4-1-r1` | Stage 3b emits Stage 3's exact row schema plus additive keys only: a row with no injection and no refinement round-trips json-i… | TEST: pipeline/stage3b_track2d/track2d.py --self-test scenario 'row byte-stability: original keys survive r… | — | 6 |
| `5.4-1-r3` | A failed or anchor-less propagation window retires that camera's live tracks, so a stale seed cannot breed a duplicate identity… | TEST: track2d.py --self-test scenario 'failed window retires its tracks: no duplicate identity, no phantom … | — | 6 |
| `5.4-1-r4` | Recovered boxes are bounded from above as well as below, and the bound gates the next window's seed as well as the injection, s… | TEST: track2d.py --self-test scenarios 'oversize leak: rejected, counted, and NOT seeded' (:3159-3192) and … | — | 6 |
| `5.4-1-r5` | Stage 3b carries a degraded upstream forward in its own marker: consuming a flagged Stage 3 writes _SUCCESS.degraded with the c… | TEST: track2d.py --self-test scenario 'degraded upstream: Stage 3b's own marker is _SUCCESS.degraded and ca… | — | 6 |
| `5.5-r6` | sam31_multiplex is a selectable mask_2d provider whose checkpoint identity is pinned by sha256 and verified BEFORE the model is… | MEASUREMENT: scripts/smoke_sam31.py passes in both modes on this 4090 (C26, 2026-08-19): 32 boxes -> 32 mas… | — | 6 |
| `6-r3` | priors file sets source: nuscenes_gt_pilot, never S0 | MEASUREMENT: EV artifacts.priors.source == nuscenes_gt_pilot (real file, 23 classes) | P1-11 | 8 |
| `6-r5` | A recorded resolution of the nuScenes licence [VERIFY] exists before priors land in any public repo | ARTIFACT: DECISIONS.md C12 — nuscenes/LICENSE read; derived tables covered by the Dataset Terms; .gitignore… | — | 8 |
| `6-r6` | A.4 schema: dims {mu, sigma} per class plus eps_bev with the formula recorded | MEASUREMENT: EV artifacts.priors: 23 classes, eps_formula + eps_scale keys present; priors.py:523,581,611,705 | X-6 | 8 |
| `9-r3` | No synthetic-timestamp dt test and no flat z=0 ground fixture exist (they would pass on broken code) | MEASUREMENT: vacuously satisfied: tests/ is empty — the forbidden tests do not exist; guard stays as review… | — | 4 |

## N/A (1)

| id | claim | evidence | closes | phase |
|---|---|---|---|---:|
| `1.9-r6` | run_manifest.json carries checkpoint IDs + revisions + SHA-256 | ARTIFACT: stage 1 uses no models; no GPU-stage manifest exists yet | P1-8 | 5 |
