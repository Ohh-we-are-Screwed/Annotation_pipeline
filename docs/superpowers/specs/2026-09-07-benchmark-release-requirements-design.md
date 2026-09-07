# Benchmark release requirements — design (2026-09-07)

Approved in chat 2026-09-07 by the operator. Source requirements: the
benchmark side's "What the benchmark needs from the annotator"
(`/home/mt/dataset_benchmark/dbench/ingest/contract.md` §6,
`BENCHMARK_METRICS_SPEC.md` §1–3, §10, `configs/benchmark_v1.0.yaml`).

## Goal

Make every `export/day1_chunk_NNNN/` release satisfy the benchmark's
per-delivery checklist:

- only accepted boxes in `sample_annotation`, the rest kept beside it;
- annotation range matching `class_range` (50 / 40 / 30 m);
- the annotation rule (N points, visibility V) written down;
- `attribute.json` populated, every non-static box with an attribute where
  one can be derived;
- instances that persist through a scene, `prev`/`next` rebuilt after
  filtering;
- ≥ 5 % of keyframes double-annotated, stratified, with `annotator_pass`;
- `is_uncertain` on every row; `size` = (w, l, h) positive; unit
  quaternions; every token resolves;
- a delivery note per chunk.

## Decisions taken in chat (operator, 2026-09-07)

| # | Question | Decision |
|---|---|---|
| 1 | Range cap 30 m vs. the benchmark's 50 m | **Raise the pipeline cap to 50 m; the Stage 9 point floor decides what survives.** Delivery note reports the effective per-class range. |
| 2 | Track fragmentation | Offline **stitching + interpolation** over the finished pre-labels. (The online tracker gate change was dropped with decision 7.) |
| 3 | Attributes | **Velocity-derived `moving` / `stopped` only**, never `parked`; single-annotation instances get no attribute. |
| 4 | Double annotation | **Full loop**: stratified selection, A/B CVAT tasks, and the import-back path into I-5 records. |
| 5 | What annotators start from | **A/B tasks start empty; the single-pass review task is pre-filled** with our cuboids. |
| 6 | Anonymisation | **After annotation.** The annotated images are un-blurred; blurring is applied to the released images afterwards. |
| 7 | Architecture | **Approach 2 — freeze the pipeline; everything is export-time post-processing** over `prelabels.jsonl`. The range change is the single stage edit and forces one rerun of Stages 1–9 per chunk. |
| — | Taxonomy rules ("cycle vs battery rickshaw distinct", "no folding") | **Ignored by operator instruction.** `configs/release_category_map.yaml` stays as it is (`"a rickshaw"` → `cycle_rickshaw`). |
| — | Execution model | Implementation by Opus subagents, one per plan task; the main session reviews and corrects. |

## Substrate facts the design rests on

Measured on `export/day1_chunk_0000` (26 791 boxes, 684 keyframes) unless
stated.

- Keyframes are **2.5 Hz** (0.400 s median, no variance). A 3-keyframe gap is
  1.2 s.
- Stage 7 produced 11 882 tracks from 26 791 detections (44 % of detections
  are births); `max_misses_before_death = 2`; median instance = 1 keyframe;
  2 235 tracks ≥ 3 hits. Stage 7 velocity is `[0, 0]` on **every birth row**
  by construction, so 4 038 of 15 395 `auto_accept` boxes carry a velocity
  that means "unknown", not "stationary".
- The 30 m cap lives in two places that must agree:
  `pipeline/stage1_ingestion/ingest.py` `IngestConfig.range_cap_m = 30.0` and
  `pipeline/common/eval_region.py` `_R_MAX_M = 30.0`. Observed max box range
  30.7 m.
- Stage 9 gate: `min_lidar_returns = 5` (single-sweep, ground-filtered,
  pre-inflation), `conf_gate = 0.5`, BEV footprint ≤ 2× class prior. Tiers:
  `rejected` (physical gate failed), `flagged` (confidence below cutoff or
  prior unavailable), `auto_accept`.
- `AnnotationRecord.attribute` exists but the I-4 validator rejects any
  non-None value ("no producer"). `Provenance` carries `source`, `tier`,
  `gates`, `verified_by`, `verification_pass`; `POLICY.allow_human_provenance
  = False`.
- Detector vocabulary (`configs/taxonomy_pilot_dhaka.yaml`) is 12 phrases.
  Producible 18-class names: car, bus, truck, cng_autorickshaw,
  cycle_rickshaw, bicycle, motorcycle, pedestrian, traffic_cone, barrier,
  construction_element. **Not producible**: covered_van, microbus,
  battery_rickshaw, tempo, human_hauler, pushcart, animal. Cone, barrier and
  construction_element are in the vocabulary and had zero detections on this
  route.
- `scripts/export_cvat_3d.py` packs the **ground-filtered single sweep** of
  every keyframe as `pointcloud/NNNNNN.pcd` inside
  `<work_root>/cvat_export_3d/<scene>/task.zip`, and that archive survives
  the chain's Stage 1 cloud pruning. Frame index ↔ keyframe is implicit
  (order of `keyframes.jsonl`). Cuboids are shapes, not tracks, with no
  attributes. `scripts/cvat_setup_3d.py` imports "Datumaro 3D 1.0".
- `export_release.py` already: converts ego → global through the keyframe's
  LiDAR `ego_pose` (devkit-verified), builds `instance` / `prev` / `next`
  from whatever records it is given, has the nuScenes attribute name mapping
  (`attribute_name()`), writes `dhakascenes_*` provenance extras, refuses to
  overwrite an existing `<out>/<version>`.
- dbench reads `dbench_double_annotated` on `sample` rows and warns when
  absent; it does **not** read `annotator_pass` yet.

## Architecture

### The single pipeline edit

`range_cap_m: 30.0 → 50.0` and `_R_MAX_M = 30.0 → 50.0`, provenance comments
rewritten: *"50 m, operator decision 2026-09-07: annotate to the benchmark's
evaluation range (`class_range` 50/40/30). The Stage 9 point floor (≥ 5
returns) decides what survives; the delivery note reports the effective
per-class range. Runs before and after this line are not comparable."*
`ground_fit_range_m` (3–12 m) is untouched. Nothing else under
`pipeline/stage*` changes.

### Release post-processor

A new package `pipeline/release/` of pure functions — no `_SUCCESS`, no
`require_upstream`, no run manifest; it is not a stage — orchestrated by
`scripts/export_release.py`:

```
prelabels.jsonl (all tiers)
   │
   ├─ stitch.py      identity repair + interpolation          (§3)
   ├─ human.py       merge I-5 verified.jsonl when present    (§7)
   ├─ tiers.py       keep auto_accept + human; sidecar rest   (§5)
   │                 rebuild instance / prev / next
   ├─ attributes.py  moving / stopped from chain velocity     (§4)
   ├─ strata.py      per-keyframe density + illumination      (§6)
   ├─ double.py      stratified 5 % selection → sample flags  (§6)
   └─ note.py        DELIVERY_NOTE.md                         (§8)
   ▼
nuScenes tables + sidecars + release_meta.json + DELIVERY_NOTE.md
   │
   └─ scripts/check_release.py   the checklist               (§8)
```

**Order is fixed**: stitch on all tiers (a rejected box mid-track is still
evidence of identity) → human merge → tier filter → instance / chain rebuild
→ attributes (needs final chains) → strata / double → tables → note → check.

**`export_release.py` flags** (all recorded in `release_meta.json`):

| Flag | Default | Meaning |
|---|---|---|
| `--tiers {auto_accept,all}` | `auto_accept` | Pipeline tiers admitted to `sample_annotation`. `all` reproduces today's output. |
| `--stitch / --no-stitch` | on | §3 |
| `--attributes / --no-attributes` | on | §4 |
| `--double-fraction F` | 0.05 | §6; `0` disables |
| `--human DIR` | none | `<work_root>/stage10_human`; §7 |
| `--note / --no-note` | on | §8 |
| `--overwrite-tables` | off | Rewrite only the five annotation tables, the sidecars, `sample.json` flags, `release_meta.json` and the note inside an existing out root. Blobs and passthrough tables are never touched. |
| `--release-config PATH` | `configs/release.yaml` | Every tunable below. |

**`configs/release.yaml`** (new) holds every number with its source:

```yaml
spec: dhakascenes/release_config/v1
benchmark_source:
  path: /home/mt/dataset_benchmark/configs/benchmark_v1.0.yaml
  sha256: "…"   # the file's digest when these values were copied; the note prints it
stitch:
  max_gap_keyframes: 3        # 1.2 s at 2.5 Hz
  base_gate_m: 2.0            # benchmark tracking.dist_th
  gap_slack_m: 1.0            # per extra missing keyframe
  size_ratio_max: 2.0         # BEV area ratio P vs S
  class_agnostic: false
attributes:
  moving_speed_threshold_mps: 0.5   # benchmark velocity_split
  max_time_diff_s: 1.5              # nuScenes box_velocity()
strata:
  density_radius_m: 30.0            # benchmark stratification.density.radius_m
  density_quantiles: [0.25, 0.5, 0.75]
  illumination_channel: CAM_FRONT
  illumination_saturation_ignore_above: 250
  illumination_bin_edges: [45.0, 75.0, 95.0]
  illumination_bin_names: [dark, night, dusk, day]
double:
  fraction: 0.05
  seed: 20260812
```

## §3 Track stitching + interpolation — `pipeline/release/stitch.py`

Runs per scene over **all** I-4 records, in the global frame (each record →
global via its keyframe's LiDAR `ego_pose`, the hop `export_release` already
does).

1. **Fragments.** Group by Stage 7 `track_id` (parsed from
   `instance_token = pilot-track:<scene>:<id>`); untracked records
   (`pilot-det:*`) are singleton fragments. A fragment holds ordered rows,
   category, first/last keyframe index, global centers, BEV footprint.
   Fragment velocities come from its **own positions**: end velocity from the
   last two rows, start velocity from the first two. Stage 7's
   `velocity_mps` is never used here (it is `[0,0]` on every birth row).
2. **Candidates.** Predecessor P ending at keyframe index *k*, successor S
   starting at *k + g*, gap g ∈ [1, `max_gap_keyframes`], **same category**.
   Distance d = min over the available predictions of: P's forward
   prediction (`c_P,last + v_P,end · Δt`) vs `c_S,first`; S's back-prediction
   (`c_S,first − v_S,start · Δt`) vs `c_P,last`; if neither fragment has a
   velocity, the plain distance (stationary assumption — a single-frame car
   fragment at speed will not join, which is the honest outcome). Gate
   `base_gate_m + gap_slack_m · (g − 1)`; BEV area ratio within
   `size_ratio_max`. Cost = d / gate.
3. **Assignment.** For g = 1 … `max_gap_keyframes` in order, Hungarian
   (`scipy.optimize.linear_sum_assignment`) over eligible pairs whose ends are
   still free (each fragment has at most one successor and one predecessor);
   accept cost ≤ 1; union-find into chains. Deterministic: ties broken by
   (track_id, keyframe index).
4. **Chains → instances.** Instance token =
   `make_token("instance", scene_token, first fragment's track_id)`. Every
   row keeps `dhakascenes_track_id_pre_stitch`.
5. **Interpolation** for g ≥ 2, one row per missing keyframe: center linear
   in global, rotation slerp, size linear; transformed back to that
   keyframe's ego frame for the visibility estimator. Row fields:
   `dhakascenes_interpolated: true`, `dhakascenes_source: "pipeline"`,
   `dhakascenes_tier` = the **worse** endpoint tier (auto_accept < flagged <
   rejected), `dhakascenes_tier_basis: "inherited_from_endpoints"`,
   `dhakascenes_record_token = "<sample_token>:INTERP:<chain_id>"`.
   `num_lidar_pts` is counted inside the box from the ground-filtered single
   sweep in `<work_root>/cvat_export_3d/<scene>/task.zip` (basis
   `single_sweep_ground_filtered_pre_inflation`, same as every other row);
   if the archive is absent, from the raw `samples/LIDAR_TOP` sweep with
   `num_lidar_pts_basis: "single_sweep_raw"` on that row and the fallback
   counted in meta.
6. **Meta** (`release_meta.stitch`): n_fragments, n_chains, joins per gap,
   n_interpolated (and fallback-basis count), median and mean chain length
   before/after, fraction of rows on chains ≥ 3.

Known gap, stated in the note: a class flip along one object (pedestrian ↔
bicycle) stays two instances — `class_agnostic` is off.

## §4 Attributes — `pipeline/release/attributes.py`

Computed after chains are final and filtered, from the same quantity the
nuScenes devkit's `box_velocity()` computes: (pos_next − pos_prev) / Δt over
the annotation's chain neighbours in the global frame, one-sided at chain
ends, **undefined** for single-annotation instances or when the neighbour gap
exceeds `max_time_diff_s`. Written per row as
`dhakascenes_velocity_chain_mps` (global frame, 2-vector); the Stage 7 field
`dhakascenes_velocity_ego_mps` stays untouched.

State: `‖v‖ > moving_speed_threshold_mps` → `moving`, else `stopped`,
undefined → no attribute. Names (extends `attribute_name()`):

| Group | Classes | moving | stopped |
|---|---|---|---|
| vehicle | car, bus, truck, covered_van, microbus, cng_autorickshaw, battery_rickshaw, tempo, human_hauler, pushcart | `vehicle.moving` | `vehicle.stopped` |
| pedestrian | pedestrian | `pedestrian.moving` | `pedestrian.standing` |
| cycle | bicycle, motorcycle, cycle_rickshaw | `cycle.with_rider` | `cycle.with_rider` (nuScenes has no `cycle.moving`; a stopped cycle is assumed ridden — stated in the note) |
| none | traffic_cone, barrier, construction_element, animal | — | — |

Pushcart is placed in the vehicle group (the spec lists no attribute set for
it; not producible in this phase anyway). `parked`, `pedestrian.sitting_lying_down`
and `cycle.without_rider` are never emitted by derivation.

Human rows use the CVAT `attribute` value when set (any nuScenes name), else
the same derivation. Every row carries `dhakascenes_attribute_basis` ∈
{`chain_velocity`, `human`, `null`}. `attribute.json` lists only the names
used.

## §5 Tiers, sidecars, `is_uncertain` — `pipeline/release/tiers.py`

- `sample_annotation` = pipeline rows with `dhakascenes_tier == auto_accept`
  (interpolated rows included when they inherit it) + every human row.
- Everything else → `sample_annotation_excluded.json` beside the tables:
  same row shape plus `dhakascenes_excluded_reason` ∈ {`tier_rejected`,
  `tier_flagged`, `superseded_by_human`, `superseded_by_double_pass`} and
  the `instance_token` the row would have had. Not a nuScenes table; the
  note says so.
- `instance.json`, `prev`, `next`, `nbr_annotations`,
  `first/last_annotation_token` are rebuilt from included rows only; a gap in
  a chain is allowed (as in nuScenes).
- `is_uncertain` (bool) and `is_uncertain_reason` (string) on every row.
  Pipeline rows: `false` / `""` — no source of class uncertainty survives to
  I-4 in this phase (the VLM check is off and its verdicts do not propagate),
  which the note states. Human rows carry what the annotator set.
- `category.json` keeps all 18 names so an absent class reads as zero, not
  missing.

## §6 Strata + double-annotation selection — `strata.py`, `double.py`

- **Density** per keyframe: included boxes whose global center is within
  `density_radius_m` of the ego position at that keyframe / (π · r²) —
  dbench's ρ, computed on the rows dbench will see. Bins: this chunk's
  quartiles → Low / Medium / High / Extreme; edges recorded.
- **Illumination** per keyframe: mean BT.601 luma
  (0.299 R + 0.587 G + 0.114 B) of the CAM_FRONT keyframe JPEG, pixels with
  luma > `illumination_saturation_ignore_above` excluded, binned by
  `illumination_bin_edges` → dark / night / dusk / day. Decoded with PIL.
- **Selection** (`double.py`): target = max(⌈F · N⌉, number of non-empty
  cells); allocation proportional to cell population, ≥ 1 per non-empty
  cell (largest remainder); within a cell, `numpy.random.default_rng(seed)`
  choice without replacement over tokens sorted by timestamp. Output
  `<out>/double_annotation.json`:

  ```json
  {"spec": "dhakascenes/double_annotation/v1", "fraction": 0.05, "seed": 20260812,
   "n_keyframes": 684, "n_selected": 35,
   "density_bin_edges": [..], "illumination_bin_edges": [45, 75, 95],
   "cells": [{"density": "High", "illumination": "day", "n": 120, "selected": 6}, ...],
   "selected": [{"sample_token": "...", "density": "High", "illumination": "day"}, ...]}
  ```

  and every `sample.json` row gets `dbench_double_annotated: true|false`.
- **Frozen after first export.** When `<out>/double_annotation.json` already
  exists (the post-human re-export), it is reused verbatim and reselection is
  refused; `--reselect-double` overrides, loudly.

## §7 CVAT round trip

### `scripts/export_cvat_3d.py`

- Writes `frames.json` per scene beside `task.zip`:
  `[{"frame": 0, "name": "000001", "sample_token": "...", "channels": [...]}]`
  — the import key that today is only implicit in `keyframes.jsonl` order.
- `--frames <double_annotation.json> --blank` produces a second task set
  under `<work_root>/cvat_export_3d_double/<scene>/` containing only the
  selected keyframes (own `task.zip`, own `frames.json`, an
  `annotations_blank.json` with zero cuboids).
- Pre-filled cuboids ("ours") carry attributes `record_token` (the I-4
  record token) and `track_id`, and are emitted as CVAT **tracks** so a
  reviewer can merge/split identities. `track_id` is the post-stitch chain
  id read from `<out>/stitch_map.json` (`record_token → chain_id`, written
  by `export_release`) via `--stitch-map PATH`; without the flag it falls
  back to Stage 7's `track_id` and says so on stderr. The chain always
  passes it (§9), so reviewers see the repaired identities and merge less.
- **Verification spike before any of this is relied upon**: publish one
  small pre-filled task to CVAT 2.72, re-export "Datumaro 3D 1.0", and
  confirm (a) `track_id` and `record_token` attributes survive, (b) track
  membership survives, (c) the cuboid inverse below reproduces the input
  boxes to 1e-6. If tracks do not round-trip, cuboids stay shapes and
  identity is carried by the `track_id` attribute alone; the spec is amended
  with the finding.

### `scripts/cvat_setup_3d.py`

- `label_spec()` gains attributes: `attribute` (select; values per class
  group from §4 plus `parked`, `pedestrian.sitting_lying_down`,
  `cycle.without_rider`, and an empty default), `uncertain` (checkbox,
  default false), `uncertain_reason` (text), `record_token` (text),
  `track_id` (number).
- `--which double` publishes two projects per chunk,
  `<CVAT_PIPELINE_3D_PROJECT> — double pass A` and `… — double pass B`, both
  from the blank set; `--assignee-a`, `--assignee-b` set the CVAT assignee.
- A ledger `<work_root>/stage10_human/cvat_tasks.json` records every task
  this script creates: project name/id, task id, kind ∈ {review, double_A,
  double_B}, scene, `frames.json` path, assignee, created time. Appended,
  never rewritten.

### `scripts/import_cvat_3d.py` (new)

For each ledger task (or `--zip PATH --kind … --scene …` for offline use):

1. Export "Datumaro 3D 1.0" through `cvat_sdk`; only jobs in state
   `completed` are read by default (`--include-incomplete` overrides).
2. Invert `cuboid()`: position → `translation_m`, rotation z → yaw →
   `rotation_wxyz` via `conventions.quaternion_from_yaw_rad`, scale
   (l, w, h) → `size_wlh_m` = (w, l, h). Label → phrase → 18-class through
   `configs/release_category_map.yaml`.
3. Attributes → `attribute`, `is_uncertain`, `is_uncertain_reason`;
   `track_id` → identity; frame → `sample_token` via `frames.json`.
4. `num_lidar_pts` counted inside the box from the task.zip PCD (the cloud
   the annotator saw) → basis `single_sweep_ground_filtered_pre_inflation`.
5. Writes I-5 `AnnotationRecord`s to
   `<work_root>/stage10_human/scenes/<scene>/verified.jsonl` through
   `schemas.write_records()` with an explicit
   `ProvenancePolicy(allow_human_provenance=True)` — the only caller that
   passes it. `source = human_verified` when the row carries a
   `record_token`, else `human_created` (A/B rows and reviewer-added boxes);
   `verified_by` = CVAT assignee username (task owner if unassigned);
   `verification_pass = 1`; `annotator_pass` ∈ {A, B} on double tasks.
6. Writes `stage10_human/import_manifest.json` (tasks read, rows per source,
   per scene, unmapped labels — an unmapped label aborts, listing offenders).

**Schema edits** (all in `pipeline/common/schemas.py`, all additive, all
gated on a human `source` so I-4 pipeline records are unchanged):
`Provenance.annotator_pass: str | None = None`, validated ∈ {"A", "B"} when
set; `AnnotationRecord.attribute` accepted ∈ the nuScenes attribute names on
human sources only (pipeline records keep the "no producer" rule);
`AnnotationRecord.is_uncertain: bool | None` and
`is_uncertain_reason: str | None`, new optional fields allowed on human
sources only. `stitch_map.json` is an exporter output, not a schema change.

### `pipeline/release/human.py`

- Loads every `verified.jsonl` under `--human`. Every sample covered by a
  completed **review** task → its pipeline rows are superseded (excluded
  with `superseded_by_human`); deletions are therefore honoured (a pre-label
  the reviewer removed does not come back). Samples not covered keep their
  pipeline rows.
- Double frames — precedence is fixed so one frame never carries three row
  sets: **pass A is the GT** for a `dbench_double_annotated` frame. When A
  is imported, both the pipeline rows and any review-task rows for that
  frame go to the sidecar (`superseded_by_double_pass`); pass B rows are
  written inline with `annotator_pass: "B"`, per the contract, and are the
  agreement measurement, not GT. If only B has been imported so far, B is
  written inline flagged and the frame's GT stays whatever review/pipeline
  rows it had; the checker reports the frame as half-imported. The checker
  also warns about the inline-B double-count hazard (dbench does not read
  `annotator_pass` yet).
- Human rows carry `dhakascenes_source`, `dhakascenes_verified_by`,
  `annotator_pass` (when set), `is_uncertain*`, and `dhakascenes_tier: null`.

## §8 Delivery note + checker

### `pipeline/release/note.py` → `<out>/DELIVERY_NOTE.md`

Generated from `release_meta.json`, the Stage 9 run manifest, the import
manifest and the double-annotation file:

- pipeline git sha (`git rev-parse HEAD` at export), stage spec strings from
  the manifests, stage tree path, export time;
- human pass: none / which scenes and tasks (from the ledger), rows imported
  per source;
- **the rule actually applied**: *"a box ships iff ≥ 5 LiDAR returns
  (single-sweep, ground-filtered, pre-inflation) ∧ detector confidence ≥ 0.5
  ∧ BEV footprint ≤ 2× class prior. There are no camera-only boxes, so V does
  not apply; `visibility_token` is a camera field-of-view proxy, not an
  occlusion estimate."* (values read from the Stage 9 manifest, not typed);
- range: cap 50 m, effective per-class p99 range of included boxes;
- tiers included; sizes of the excluded sidecar by reason;
- class table split three ways: present (n instances / n boxes), in the
  detector vocabulary but zero on this route, not producible by the
  12-phrase vocabulary;
- stitching + interpolation statistics; attribute derivation and its
  threshold; `is_uncertain` provenance;
- double annotation: fraction, n frames, cell table, seed, whether A/B rows
  were imported;
- anonymisation: *after annotation — the annotated images are un-blurred;
  blurring is applied to the released images afterwards*;
- `coco_2d/` is 2D-only with phrase names; its phrase → class table.

### `scripts/check_release.py`

Loads the tables of one export root. **Hard failures (exit 2)**: non-positive
`size`; quaternion norm off by > 1e-3; dangling `instance_token`,
`category_token`, `sample_token`, `attribute_tokens`; `prev`/`next`
asymmetry or cross-instance link; `nbr_annotations` / first / last wrong; a
pipeline row whose `dhakascenes_tier` is not `auto_accept` inside
`sample_annotation`; a human row lacking `dhakascenes_verified_by`;
double-flagged frames < the fraction the note claims; a
`dbench_double_annotated` frame with `annotator_pass` rows from only one
pass when the note claims A/B were imported. **Warnings (exit 1)**: `w > l`
boxes; non-static, multi-annotation boxes without an attribute; B-pass rows
inline; zero-instance classes; interpolated fraction > 20 %. Runs at the end
of every export and standalone (`python scripts/check_release.py <out>`).

## §9 Chain + testing

- `scripts/run_day1_chunks.sh` reorders the tail so the release export
  runs **before** the 3D CVAT publish (today `cvat3d` precedes Stage 9):
  `… 8 → 9 → export_release` (writes `double_annotation.json`,
  `stitch_map.json`) `→ export_cvat_3d --stitch-map … ` (review set)
  `+ --frames … --blank` (double set) `→ cvat_setup_3d --which ours` and
  `--which double → check_release`. A new `--phase human-import` runs
  `import_cvat_3d` → `export_release --overwrite-tables --human` →
  `check_release`. All existing chunks rerun from Stage 1 for the range
  change (the chain already supports a step list).
- The README heredoc in the chain is replaced by a pointer to
  `DELIVERY_NOTE.md`.
- TDD per task: stitch (gap-1 join, gap-3 interpolation count, class
  mismatch refuses, gate refuses, determinism, tier inheritance), attributes
  (threshold, single-annotation → none, groups, human override), tier filter
  + instance rebuild (chains consistent, sidecar reasons), strata (luma on a
  synthetic image with saturated pixels; density on synthetic boxes; bin
  edges), selection (≥ 5 %, ≥ 1 per non-empty cell, seeded determinism,
  frozen reuse refused without `--reselect-double`), cuboid ↔ box inverse
  exactness, `frames.json` mapping, import produces I-5 records that validate
  under the explicit policy and fail under the default one, note fields and
  the three-way class table, each checker failure class. Then a real run on
  chunk_0000's existing `prelabels.jsonl` to eyeball stitch statistics before
  the rerun.

## Out of scope

- Any change to Stages 1–9 beyond the two range constants.
- Class-agnostic stitching (a class flip along one object stays two
  instances).
- A `parked` attribute, `sitting_lying_down`, `without_rider` from
  derivation — human-only.
- Pipeline-side `is_uncertain` from VLM verdicts (the VLM check is off in
  this phase).
- Remapping `coco_2d/` category names.
- Face / plate blurring itself.
- Changes to dbench (reading `annotator_pass`, a pass filter) — the
  operator's other repo.
