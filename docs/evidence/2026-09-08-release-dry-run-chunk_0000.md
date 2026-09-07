# Release post-processor — dry run on chunk_0000's real pre-labels (2026-09-08)

Task 18 of the benchmark-release plan. This is a **validation** run: the finished
release post-processor was run over `work_day1/chunk_0000`'s existing Stage 9
pre-labels, written to a scratch root, and measured. No code was changed and no
pipeline output was touched — blobs are symlinked, the source dataroot is
read-only for this run, and the only file this task commits is this document.

The pre-labels predate the plan's geometry fixes: they were produced under the
**old 30 m range cap** and the **old ground-fit guard**. That is expected and is
why `range.max_exported_range_m` below reads 30.9 m against a 50 m cap. Nothing
here validates the new geometry; it validates the *post-processing* — stitching,
interpolation, attributes, tiers, strata, the delivery note and the checker.

**Verdict up front:** the stitching is safe to run on all chunks at the current
`configs/release.yaml` gates. The one thing that needs a decision before the full
rerun is not a stitch gate — it is that ~49 % of the *interpolated* rows carry
fewer than 5 LiDAR points and 33 % carry zero, which contradicts the delivery
note's own stated annotation rule. See §7.

---

## 1. The run

Environment: worktree `/home/mt/Zami/Annotation_pipeline/.claude/worktrees/benchmark-release`
at `b0ca63e`, `PY=/home/mt/miniconda3/envs/ano_pipe/bin/python`.

```bash
$PY scripts/export_release.py \
  --prelabels /home/mt/dhakascenes/work_day1/chunk_0000/stage9_qa \
  --dataroot /home/mt/Zami/Annotation_pipeline/Dataset/A_nusc/chunk_0000 --version v1.0-dhaka-fixed \
  --out /home/mt/dhakascenes/work_scratch/release_dryrun_chunk_0000/boxes --blobs symlink \
  --cvat-export-3d-dir /home/mt/dhakascenes/work_day1/chunk_0000/cvat_export_3d \
  --stage-tree /home/mt/dhakascenes/work_day1/chunk_0000 --chunk-name day1_chunk_0000_dryrun
```

| | |
|---|---|
| exit code | **0** |
| wall time | **68.24 s** (`release_meta.elapsed_s` 67.827 s) |
| stderr | empty |
| output root | `/home/mt/dhakascenes/work_scratch/release_dryrun_chunk_0000/boxes` (fresh; nothing pre-existing was reused or removed) |
| on-disk size | 43 MB — `samples/`, `sweeps/`, `export_meta.json` are symlinks into the source dataroot; the 14 tables are real files |
| tables written | 14 in `v1.0-dhaka-fixed/` (the 5 annotation tables regenerated, `sample.json` rewritten with `dbench_double_annotated`, the rest copied through) |
| sidecars | `release_meta.json`, `sample_annotation_excluded.json`, `stitch_map.json`, `double_annotation.json`, `DELIVERY_NOTE.md` |

The exporter's own summary lines:

```
wrote .../boxes/v1.0-dhaka-fixed: 16141 annotations, 5485 instances, 1 scenes; visibility basis=camera_fov_corner_fraction; devkit check={'n_checked': 64, 'all_passed': True}
tiers=auto_accept; stitch={'n_records_in': 26791, 'n_fragments': 11882, 'n_chains': 8463, 'n_interpolated': 2530, 'n_interpolated_raw_basis': 2530, 'n_interpolated_no_cloud': 0, 'joins_by_gap': {'1': 1690, '2': 928, '3': 801}}; excluded=13180 {'tier_flagged': 3610, 'tier_rejected': 9570} -> .../boxes/sample_annotation_excluded.json
double_annotation=35 keyframes (selected)
release_meta.json -> .../boxes/release_meta.json
delivery note -> .../boxes/DELIVERY_NOTE.md
```

Note `exit code 0` with 732 checker warnings is deliberate: `export_release.main`
returns `2 if rc_check >= 2 else 0`, because `scripts/run_day1_chunks.sh` treats
any non-zero release rc as a dead chunk. Run standalone, `check_release.py`
exits **1** on the same tree. That is by design, documented in the exporter.

---

## 2. `release_meta.json`

### 2.1 `stitch.totals`

```json
{
  "n_records_in": 26791,
  "n_fragments": 11882,
  "n_chains": 8463,
  "joins_by_gap": { "1": 1690, "2": 928, "3": 801 },
  "n_interpolated": 2530,
  "n_interpolated_raw_basis": 2530,
  "n_interpolated_no_cloud": 0
}
```

3 419 joins total. Gates in force (`stitch.config`, echoing `configs/release.yaml`):
`max_gap_keyframes 3, base_gate_m 2.0, gap_slack_m 1.0, size_ratio_max 2.0,
class_agnostic false`.

### 2.2 Chain length, per scene (`stitch.per_scene["chunk_0000"]`)

| | before | after |
|---|---|---|
| groups | 11 882 fragments | 8 463 chains |
| rows | 26 791 | 29 321 (+2 530 interpolated) |
| **median length** | **1.0** | **1.0** |
| mean length | 2.2548 | 3.4646 (+53.7 %) |
| fraction of rows on chains ≥ 3 | 0.5849 | 0.7783 |

The median does not move. That is not a failure — it is the shape of the data:
8 172 of the 11 882 fragments are one-keyframe detections, so more than half the
*groups* are singletons before and after. What the stitch moves is the
row-weighted mass, and singleton count itself: 8 172 → 4 335 singleton groups.
§4 gives the full distribution.

### 2.3 `excluded.by_reason`

```json
{ "table": "sample_annotation_excluded.json", "n": 13180,
  "by_reason": { "tier_flagged": 3610, "tier_rejected": 9570 } }
```

These reconcile exactly with the substrate figures in the brief once the
interpolated rows are added — an interpolated row inherits the worse tier of the
two endpoints it bridges:

| tier | pre-labels (brief) | + interpolated | = post-stitch | admitted? |
|---|---|---|---|---|
| auto_accept | 15 395 | +746 | 16 141 | yes → `sample_annotation` |
| flagged | 3 144 | +466 | 3 610 | no → excluded table |
| rejected | 8 252 | +1 318 | 9 570 | no → excluded table |
| **total** | **26 791** | **+2 530** | **29 321** | |

Nothing is silently dropped; every excluded row carries
`dhakascenes_excluded_reason`.

### 2.4 `attributes`

```json
{ "enabled": true, "threshold_mps": 0.5, "n_with_attribute": 12522,
  "by_name": { "cycle.with_rider": 2285, "pedestrian.moving": 4000,
               "pedestrian.standing": 1512, "vehicle.moving": 4243,
               "vehicle.stopped": 482 },
  "by_basis": { "chain_velocity": 12522 } }
```

12 522 of 16 141 shipped rows (77.6 %) carry an attribute; every one is
`chain_velocity` (no human pass ran, so no `human` basis). `parked`,
`sitting_lying_down` and `without_rider` are never emitted, as specified.

The 3 619 rows without an attribute break down as **2 897 on singleton instances**
(velocity is undefined by definition) and **722 on multi-row instances**. The 722
are exactly the 722 checker warnings in §3, and they are fully explained — see §3.1.

### 2.5 `range`

```json
{ "cap_m": 50.0, "max_exported_range_m": 30.9,
  "effective_p99_m_by_class": {
    "bicycle": 28.41, "bus": 29.17, "car": 29.19, "cng_autorickshaw": 29.04,
    "cycle_rickshaw": 28.54, "motorcycle": 29.02, "pedestrian": 29.57, "truck": 28.95 } }
```

`cap_m` 50 m is the *new* pipeline cap read from `eval_region._R_MAX_M`;
`max_exported_range_m` 30.9 m is what these *old* pre-labels actually reached.
The 19 m difference is the whole reason the two used to share one key and now do
not: a reader of the delivery note can see at a glance that this chunk's boxes
stop at 30 m and that the 50 m figure is a cap, not a claim. Every class p99 sits
at 28–30 m, confirming the old 30 m cap end-to-end.

### 2.6 `classes`

```json
{ "present": { "bicycle": 316, "bus": 317, "car": 725, "cng_autorickshaw": 263,
               "cycle_rickshaw": 369, "motorcycle": 563, "pedestrian": 2730, "truck": 202 },
  "absent_on_route": [ "barrier", "construction_element", "traffic_cone" ],
  "not_producible": [ "animal", "battery_rickshaw", "covered_van", "human_hauler",
                      "microbus", "pushcart", "tempo" ] }
```

8 of the 18 benchmark classes are present. The three-way split does its job: it
separates "the detector can see it, this route had none" from "the 12-phrase
vocabulary cannot produce it at all", which is the distinction the checklist asks
for. Both groups still produce a `class X: zero instances` warning from the
checker (§3), which is correct behaviour and not actionable.

### 2.7 `counts` and `records`

```json
"counts": { "n_records": 26791, "n_annotations": 16141, "n_instances": 5485,
            "n_scenes": 1, "n_untracked_singletons": 0,
            "per_class": { "bicycle": 316, "bus": 317, "car": 725,
                           "cng_autorickshaw": 263, "cycle_rickshaw": 369,
                           "motorcycle": 563, "pedestrian": 2730, "truck": 202,
                           "animal": 0, "barrier": 0, "battery_rickshaw": 0,
                           "construction_element": 0, "covered_van": 0,
                           "human_hauler": 0, "microbus": 0, "pushcart": 0,
                           "tempo": 0, "traffic_cone": 0 } }
"records": { "n_loaded": 26791, "n_t_ns_corrected": 0 }
```

**`n_t_ns_corrected` is 0** — every record's `t_ns` already agreed with its
sample's timestamp to within 1 ms, so the timeline `box_velocity()` reads is the
timeline the records were written against. `n_untracked_singletons` is 0: Stage 7
gave every record a `track_id`.

Supporting fields: `devkit_cross_check = {n_checked: 64, all_passed: true}` (the
ego→global transform was verified against nuscenes-devkit on 64 sampled boxes);
`visibility.basis = camera_fov_corner_fraction` with `n_assumed_full = 0` (no
camera fell back); `num_lidar_pts_basis = ["single_sweep_ground_filtered_pre_inflation",
"single_sweep_raw"]` — the two bases are discussed in §6.

### 2.8 Double annotation

`n_selected = 35` of 684 keyframes = 5.12 %, against `fraction 0.05`
(`ceil(0.05 × 684) = 35`, so the requirement is met exactly), `seed 20260812`,
`reused: false` (first selection for this root).

| density | illumination | cell size (n) | selected |
|---|---|---|---|
| Extreme | day | 215 | 11 |
| Extreme | dusk | 1 | 1 |
| High | day | 162 | 8 |
| High | dusk | 2 | 1 |
| Low | day | 156 | 7 |
| Medium | day | 148 | 7 |
| | | **684** | **35** |

Density quartile edges came out at [0.00707, 0.00813, 0.00955] objects/m² over
the 30 m disc (≈ 20–27 objects in view), and the illumination binning put 682 of
684 keyframes in `day` and 2 in `dusk` — a single daytime drive, so the
illumination axis is nearly degenerate here. The selection still covers all six
non-empty cells, including both one-frame `dusk` cells.

---

## 3. `check_release` output

Standalone: `python scripts/check_release.py <out> --version v1.0-dhaka-fixed`
→ **exit 1**, **0 errors, 732 warnings**. Summary line:

```
check_release: 0 error(s), 732 warning(s); {'n_annotations': 16141, 'n_instances': 5485, 'n_interpolated': 746, 'present': {'cycle_rickshaw': 369, 'pedestrian': 2730, 'car': 725, 'truck': 202, 'bicycle': 316, 'motorcycle': 563, 'bus': 317, 'cng_autorickshaw': 263}}
```

**Zero errors** means every structural invariant held on the shipped tables:
positive sizes, unit quaternions, no dangling instance/sample/attribute/category
tokens, symmetric `prev`/`next` on every chain, `nbr_annotations` matching the
row count on all 5 485 instances, `first`/`last_annotation_token` correct, no
non-`auto_accept` pipeline tier inside `sample_annotation`, and the
double-annotation flags in `sample.json` agreeing with `double_annotation.json`.

The 732 warnings fall into exactly two shapes:

| n | warning |
|---|---|
| 722 | `<token>: non-static multi-annotation box without attribute` |
| 10 | `class <name>: zero instances` — animal, barrier, battery_rickshaw, construction_element, covered_van, human_hauler, microbus, pushcart, tempo, traffic_cone |

The 10 zero-instance warnings are the `absent_on_route` + `not_producible` classes
from §2.6 and are expected on a Dhaka route. The complete output is reproduced
below.

### 3.1 Why all 722 attribute warnings fire (verified, benign)

`assign_attributes` runs on the **final admitted chains only**
(`export_release.py`: `# --- 5. attributes (final chains only) ---`), after the
tier filter. So when a chain's interior rows are `flagged`/`rejected` and go to
the excluded table, the surviving shipped rows can be far apart in time, and
`chain_velocities` refuses a velocity whose window exceeds
`attributes.max_time_diff_s = 1.5 s` (3.0 s for an interior row using both
neighbours).

I measured the velocity window of all 722 rows:

```
shipped multi-row rows with no attribute: 722
  their velocity window dt (s): min=1.60  p50=2.40  max=14.40
  window span (neighbours used): Counter({1: 482, 2: 240})
  all exceed attributes.max_time_diff_s: True   <- 722 / 722
```

Every single one is over the threshold. The warning is therefore correct and the
behaviour is correct; the two just disagree about whether it is worth reporting.
Expect ~700 of these per chunk on the rerun. **No action needed** — but do not
read "732 warnings" as a red flag when the rerun prints it.

### 3.2 Full output

<details>
<summary>full <code>check_release.py</code> stdout — 733 lines, 0 errors / 732 warnings</summary>

```
WARNING 57d2b5025a894db455e32fcd3d11aa4c: non-static multi-annotation box without attribute
WARNING 55072cca8a8ca92286404d2feebb8053: non-static multi-annotation box without attribute
WARNING 225e929c2fca4603f98cc92e171e2700: non-static multi-annotation box without attribute
WARNING fe64071e126709bfd218bba6438fcecf: non-static multi-annotation box without attribute
WARNING 898ef1fda8fabb4b6c27312e530e4a71: non-static multi-annotation box without attribute
WARNING 67f272f637e600288e3128168a561144: non-static multi-annotation box without attribute
WARNING 45fba811a2470df44a7e51891215bb6c: non-static multi-annotation box without attribute
WARNING 7906462f888a9c5d6915a54f3b923aa0: non-static multi-annotation box without attribute
WARNING 74345b0db668b5d59ca2ddf89f37f819: non-static multi-annotation box without attribute
WARNING 8605719486e30fb9d9a8b28602065f84: non-static multi-annotation box without attribute
WARNING d8019ea795dc5df1ee23a5eff911296f: non-static multi-annotation box without attribute
WARNING d3365dad6d79b572e55e5a5cbc657c24: non-static multi-annotation box without attribute
WARNING 911cd68fbb304b916b8a4ca3171e748b: non-static multi-annotation box without attribute
WARNING 9fefee495e227a96a687549916769c51: non-static multi-annotation box without attribute
WARNING 76094ca3e5f5a9cf8028e45ee41bb0ec: non-static multi-annotation box without attribute
WARNING ed48003df63430977ffe8caf2e4adc46: non-static multi-annotation box without attribute
WARNING 3a7391b09236a7eba3cb0df465426f7e: non-static multi-annotation box without attribute
WARNING 4da31f9c98d5ca0133cbdc3aa319380c: non-static multi-annotation box without attribute
WARNING 0c220a50980812e6d7f04d854d68af51: non-static multi-annotation box without attribute
WARNING ca82be1b8b801d411a5820fd7254212f: non-static multi-annotation box without attribute
WARNING 489e09d41c436e0640ce7a8088656585: non-static multi-annotation box without attribute
WARNING 3f7e38ccd6f5fa0182c9135e38773741: non-static multi-annotation box without attribute
WARNING b08bdc483fd6ad1aea1e95418277f155: non-static multi-annotation box without attribute
WARNING 0861285cb86ba3f6ff16e27714a994b5: non-static multi-annotation box without attribute
WARNING 1e1a51014b32bc5018a8e33e36335bec: non-static multi-annotation box without attribute
WARNING b26b92339ff0ddb0015cca043fb7c0a0: non-static multi-annotation box without attribute
WARNING 976a0daaa4096f5f273fb67e6e009473: non-static multi-annotation box without attribute
WARNING 8dc03a28da13c03450cca9c88c6e160b: non-static multi-annotation box without attribute
WARNING 396cd9c9b5d1044210db2b7d6c549eb1: non-static multi-annotation box without attribute
WARNING 32045ad4aec318f2431f5caf826bfd7a: non-static multi-annotation box without attribute
WARNING d823ab2142039478182be9086a1cb8d0: non-static multi-annotation box without attribute
WARNING bf92d5d8ece80931803dc3ce07441fe3: non-static multi-annotation box without attribute
WARNING 53729e0c9bd1b80fec840c43c7ee4d02: non-static multi-annotation box without attribute
WARNING ee560d9673899d7a9ca14426eb27d989: non-static multi-annotation box without attribute
WARNING 62960ccebf80da8b09b0231dc0cf4e6a: non-static multi-annotation box without attribute
WARNING 642070366b845d5c73c96bc309e055f9: non-static multi-annotation box without attribute
WARNING 03b8b274291d74c10349c9fc25042839: non-static multi-annotation box without attribute
WARNING 511ef5ea74ab1797be9a7cc0d1e7fb5c: non-static multi-annotation box without attribute
WARNING d250743ae2c635291c6546ad4499e053: non-static multi-annotation box without attribute
WARNING 503b829eee947827875e0903850dff19: non-static multi-annotation box without attribute
WARNING 24385332f1244063e07d28722a0c397b: non-static multi-annotation box without attribute
WARNING 36f373dace4ca9427d0afcd5edd27164: non-static multi-annotation box without attribute
WARNING b38975b8e55284e5a67585d934f4ddde: non-static multi-annotation box without attribute
WARNING b6a844ae7c687521d7bcf0eaffd57bc6: non-static multi-annotation box without attribute
WARNING 3f0c13b661838da3ebfecb225360cb8e: non-static multi-annotation box without attribute
WARNING 77d7c843d43ddf6f9442f4de0922d060: non-static multi-annotation box without attribute
WARNING 4ea573d094797f5e6c48ba2e34b8cae3: non-static multi-annotation box without attribute
WARNING 94edaceed3b57bd8e364e75b2a95fa2b: non-static multi-annotation box without attribute
WARNING 82c1829e22e27e7a09c4ce64622ea471: non-static multi-annotation box without attribute
WARNING c4444d66f2aed4b8a84eb60dd97960df: non-static multi-annotation box without attribute
WARNING 7e23abb19752d14b88aeee5e680a3f96: non-static multi-annotation box without attribute
WARNING f81ff3d53a0ab8d2bad87a94514a2850: non-static multi-annotation box without attribute
WARNING daaffa271e3e913e826a714a1933d5cf: non-static multi-annotation box without attribute
WARNING 74dea229b45c4a0d4cd74aab7338f11a: non-static multi-annotation box without attribute
WARNING 3a5f237382d0cd1637b532a73cd11247: non-static multi-annotation box without attribute
WARNING 9dca4ee2ba9715e63457a26ab17b77bd: non-static multi-annotation box without attribute
WARNING 9f5da0d5c8020954ff639a67f259dcff: non-static multi-annotation box without attribute
WARNING 3bd9edd8c046253ce98385a35a9f3b8e: non-static multi-annotation box without attribute
WARNING be93485e9ead6a7e6295169f71d92ed8: non-static multi-annotation box without attribute
WARNING 392dbb6e0f1f9cdee10692b46bf97474: non-static multi-annotation box without attribute
WARNING c8fdce8ee8ddb7176b4d6850275a8bb0: non-static multi-annotation box without attribute
WARNING 1d6b920d674473e19150284a40359e95: non-static multi-annotation box without attribute
WARNING 536c3d45303eef7493d6d0c322345eda: non-static multi-annotation box without attribute
WARNING f26b7774e5e5e8fb8505c9d67408c8e0: non-static multi-annotation box without attribute
WARNING cd1286ba6c595a0191fd6534212f2497: non-static multi-annotation box without attribute
WARNING bc505474b613fa870b79cb17b843a271: non-static multi-annotation box without attribute
WARNING f794e4b7842b670d4658f47751fed4a5: non-static multi-annotation box without attribute
WARNING bc60000ea4a4d044a00af3c0429d4093: non-static multi-annotation box without attribute
WARNING c2039cb08d9ca27741b7c0ee6c67fa06: non-static multi-annotation box without attribute
WARNING 3320e6afe6e5ac34664da4c9ec82d61f: non-static multi-annotation box without attribute
WARNING 780d94e1924a5783188daac61f21fe20: non-static multi-annotation box without attribute
WARNING 52517fbe06796efb8925790d24f1a121: non-static multi-annotation box without attribute
WARNING 502dced7d2daecd00224fcaf9245d981: non-static multi-annotation box without attribute
WARNING b65bb1a5e7a4d2f625430ac7d6ce60d8: non-static multi-annotation box without attribute
WARNING 2dd1a23e5ee5387c73bab954727cb1a0: non-static multi-annotation box without attribute
WARNING 3ed2f5ed29548f4395dac8cd7ba9c559: non-static multi-annotation box without attribute
WARNING c5f594c11934b46c9483249975a64a95: non-static multi-annotation box without attribute
WARNING d2ad3ce30788a57d43de50f0c90ab1cd: non-static multi-annotation box without attribute
WARNING b07f196dcba104aced3e566ab5f59809: non-static multi-annotation box without attribute
WARNING 9feb40792fe05e099c11f62e18b437d2: non-static multi-annotation box without attribute
WARNING 60a146ee04c2cbe5b96a420ecdbe0d5c: non-static multi-annotation box without attribute
WARNING fd3cb51428e51d08942f39f5f25bb027: non-static multi-annotation box without attribute
WARNING 928dec1d3cae0ebc0ac6249647f96d8b: non-static multi-annotation box without attribute
WARNING 15a2a1e8fe28781d87ca7e8ae5af50df: non-static multi-annotation box without attribute
WARNING 7eb887ef74f648c672c47fccfec06dc5: non-static multi-annotation box without attribute
WARNING cd99b0ee0987c8608329e7012de8bdcd: non-static multi-annotation box without attribute
WARNING cf423c3b82c7d53b47e838eb05b42a15: non-static multi-annotation box without attribute
WARNING 2d68a9c35f4b97a89d4125b96500042a: non-static multi-annotation box without attribute
WARNING d2176c2b0b10cd788feed4efa8a838fd: non-static multi-annotation box without attribute
WARNING 184cad310d8b32cfd6b8dd5864e5fac3: non-static multi-annotation box without attribute
WARNING 99d1f95749255353f4677be46a553534: non-static multi-annotation box without attribute
WARNING 6aacaf62556c3781525525b7da716936: non-static multi-annotation box without attribute
WARNING a31bc00f9650deddfff2c6e995c4c41b: non-static multi-annotation box without attribute
WARNING c0ada28e39010a2bfc760b4189224076: non-static multi-annotation box without attribute
WARNING a2a937cb72f4a737ff695b2731c68f60: non-static multi-annotation box without attribute
WARNING 61f6d8e891ecd6be22ca038a9ff432d5: non-static multi-annotation box without attribute
WARNING a1b9ee8d294fd4f2c99f480e4a58d7f9: non-static multi-annotation box without attribute
WARNING 80b440ce62b7b66fc538671080409aae: non-static multi-annotation box without attribute
WARNING c20a1c3009a8b4f621e2aaaf6aa4c342: non-static multi-annotation box without attribute
WARNING b22fc12948d32cd964932dc4c26efef9: non-static multi-annotation box without attribute
WARNING 33a904077f3c3426760ac07253dce10d: non-static multi-annotation box without attribute
WARNING 3a40c7ca86f257e09368e0b0abda3402: non-static multi-annotation box without attribute
WARNING ed5f9e0cc03afe143b8d230fe6e5bd27: non-static multi-annotation box without attribute
WARNING cad7a12c6070c5ffcf6efa180bb683c6: non-static multi-annotation box without attribute
WARNING 8083ca5fe8298d47193048e4bf691ba9: non-static multi-annotation box without attribute
WARNING 51c552e8b70697673daca900c983ef89: non-static multi-annotation box without attribute
WARNING cc8cecdc42c443e4ab7a721f29c493f7: non-static multi-annotation box without attribute
WARNING 920d041773a044b8a3b0ca4deefefea0: non-static multi-annotation box without attribute
WARNING f14d94a23ac811dc7696d678a2e72d55: non-static multi-annotation box without attribute
WARNING 1ae9bb0edfd434cdcf1586c25292e6d0: non-static multi-annotation box without attribute
WARNING 840e85c00b747adc80131ff1974da96a: non-static multi-annotation box without attribute
WARNING 172e2dc660b82a09bdec8f096f779963: non-static multi-annotation box without attribute
WARNING 045807f71e45a149eb815e755ee40f37: non-static multi-annotation box without attribute
WARNING 8f67f886c51b8cf2c7730d1b4b5f8422: non-static multi-annotation box without attribute
WARNING ac3dd123f5d3c42f76bf699a1e1d68ee: non-static multi-annotation box without attribute
WARNING 102ab0f7b09ecd9068dac6217adf1dcb: non-static multi-annotation box without attribute
WARNING 5708a4c82b67d70bec4a990a2e199a16: non-static multi-annotation box without attribute
WARNING 53161b8edcf4b6b6599e4b97385cbfde: non-static multi-annotation box without attribute
WARNING 76a265d293413e4c78d127751d485492: non-static multi-annotation box without attribute
WARNING e376ee9957508e0d056ac9d88cee0306: non-static multi-annotation box without attribute
WARNING 94c7f55daadb76eb449fcf9bca688d7d: non-static multi-annotation box without attribute
WARNING 6e7844e543a80d073a91a5140a627321: non-static multi-annotation box without attribute
WARNING f5ce97e9a41b3790b0a83d2ce6998cc3: non-static multi-annotation box without attribute
WARNING 862b9711bbe1ab0a2099260a5b032497: non-static multi-annotation box without attribute
WARNING 8de2b1eb18c790813371c6f727085ba1: non-static multi-annotation box without attribute
WARNING 8be98b8253f52f1e2cda9706c575f4fd: non-static multi-annotation box without attribute
WARNING a53d034322c384e56a2f63de90621115: non-static multi-annotation box without attribute
WARNING 95023a0b7df0a435f33631882f748a6d: non-static multi-annotation box without attribute
WARNING f26a6e9540130230921fff8b564b80aa: non-static multi-annotation box without attribute
WARNING 595688afba0dc4c80dd2fe41cf00779c: non-static multi-annotation box without attribute
WARNING e602f71c270359e588df116c4b3273b8: non-static multi-annotation box without attribute
WARNING d86e972a9da4bdd0d678134bdfc7c4b3: non-static multi-annotation box without attribute
WARNING 00fd6d340a5433fcc4a0f3b76eb589cf: non-static multi-annotation box without attribute
WARNING ea1745f49e8ccfdd2fc656e85e88d346: non-static multi-annotation box without attribute
WARNING d78b6edb27644ad2842c6395db7c0060: non-static multi-annotation box without attribute
WARNING be4b2e9dd369ff3c6401a028fd06e03a: non-static multi-annotation box without attribute
WARNING a7b048467825287a7207903c889436d7: non-static multi-annotation box without attribute
WARNING 9391d9c8bb73a6200ea5db0444908a5a: non-static multi-annotation box without attribute
WARNING 408161c1e680faaade2fe708d1813c36: non-static multi-annotation box without attribute
WARNING b3720c8d550f6f0954d9e42327d18181: non-static multi-annotation box without attribute
WARNING 20bce5a4906e3018f3b0d75566067785: non-static multi-annotation box without attribute
WARNING 7522a189278319b9f820307c40e9faed: non-static multi-annotation box without attribute
WARNING 95f310a6039857013c6a3cda1b64c0b3: non-static multi-annotation box without attribute
WARNING 5dffe3271318df26bef7e3e502edb0b5: non-static multi-annotation box without attribute
WARNING 43e79932d826a5995c2ad19b201008b9: non-static multi-annotation box without attribute
WARNING add51d6991e7085dd993bfcaeaa419b0: non-static multi-annotation box without attribute
WARNING f83b1cd39e13c63e7b8854c8dc014ade: non-static multi-annotation box without attribute
WARNING 4e8138c327e64a08695263b282931796: non-static multi-annotation box without attribute
WARNING cfd28c4d488abc8189ffe6c7eb12aeda: non-static multi-annotation box without attribute
WARNING 1c32be7e34081b38f2a33d6f6460e982: non-static multi-annotation box without attribute
WARNING 43349bd935b286b6fc123f921848ab97: non-static multi-annotation box without attribute
WARNING 7e1517a0587556d2d4b12b6eaa71bb4e: non-static multi-annotation box without attribute
WARNING 091454ac42fb7edceb20080e1912bb43: non-static multi-annotation box without attribute
WARNING e6b5ccff92e5d0a5c2c6291a0b5e3a0b: non-static multi-annotation box without attribute
WARNING 9ad0cfec884ce6649f0063cfa1561261: non-static multi-annotation box without attribute
WARNING edd991c2c6719ee75d095199b7bac823: non-static multi-annotation box without attribute
WARNING 18d4dcf365fcad05c8f9dfa36a40c124: non-static multi-annotation box without attribute
WARNING 507de7c9dffe316df29fef5c7d8824f3: non-static multi-annotation box without attribute
WARNING 046cfea30c574413660f912273e512ea: non-static multi-annotation box without attribute
WARNING 92ec2036dcc494a31a2b45227bf2c85d: non-static multi-annotation box without attribute
WARNING 7ca10fede1a623c2c9e45a874c24fbc2: non-static multi-annotation box without attribute
WARNING 30c180b0182ff616bd9f091604ba2366: non-static multi-annotation box without attribute
WARNING 84608f3b9d282e3635a05a25f02e67e6: non-static multi-annotation box without attribute
WARNING 2565eaec48587c308a8f1f5372a9304e: non-static multi-annotation box without attribute
WARNING 3fe9bc491181c59163b71fa1742e4f9f: non-static multi-annotation box without attribute
WARNING 6056ba96ea1a26e7b84a8d5a647cabd9: non-static multi-annotation box without attribute
WARNING 7423d468be4e5ea3acfaa7d50afb6367: non-static multi-annotation box without attribute
WARNING e0e54c8c617ed9406b730e8323623f46: non-static multi-annotation box without attribute
WARNING 5de2988d42a8d6d71f6ceaf082d26e3f: non-static multi-annotation box without attribute
WARNING 5cc998dab9178d68afe048f3dba1aa6f: non-static multi-annotation box without attribute
WARNING 935b62655e042d4d21b1ed11e73038a9: non-static multi-annotation box without attribute
WARNING 36e9eff4f49a116283eabe7b3f99ae79: non-static multi-annotation box without attribute
WARNING 71b8f18626ab69f63af62e60471e288c: non-static multi-annotation box without attribute
WARNING 017735da82ad77a923892acc7d431532: non-static multi-annotation box without attribute
WARNING 42dbab3e7a34a6dcd5d13d0ff412821c: non-static multi-annotation box without attribute
WARNING 02f2845d096c005dde51fa33a1aebc0a: non-static multi-annotation box without attribute
WARNING 2692dfe4ea041074e53f282a5d1327f1: non-static multi-annotation box without attribute
WARNING a115d0a2b8410939769062ad978f5940: non-static multi-annotation box without attribute
WARNING 9c3ccfbdff3e19137d17ea6ee330eadb: non-static multi-annotation box without attribute
WARNING 5e708da2ef8a0dd7bf04d855b3dae3e0: non-static multi-annotation box without attribute
WARNING 3d2638dac843c6cc35a26b2e1a913dbe: non-static multi-annotation box without attribute
WARNING 58257027c0ea7d41182f9c7340a0913d: non-static multi-annotation box without attribute
WARNING 80122987b0c9670db91922731589448e: non-static multi-annotation box without attribute
WARNING 3af312e294a076b155501163eee9ccd2: non-static multi-annotation box without attribute
WARNING 33714a0253896cb163e0163ce2a23e31: non-static multi-annotation box without attribute
WARNING 0a27fd5732f3244a9364f315490d9b1e: non-static multi-annotation box without attribute
WARNING e0044a0796e4bb802f008adc3f5e922c: non-static multi-annotation box without attribute
WARNING 633b15de0e435af07c9bfe397cb72ac2: non-static multi-annotation box without attribute
WARNING b9488999921b8d89b73fd7a0b83097d2: non-static multi-annotation box without attribute
WARNING 4f1bf73eb70c0cd50194d46c1dda67d2: non-static multi-annotation box without attribute
WARNING f36de9e94893d5eec9117ff7223375e5: non-static multi-annotation box without attribute
WARNING d93dca0c48bdfb454cebcb4228971dbd: non-static multi-annotation box without attribute
WARNING fc990c24c805f3014f34f4986eda7e90: non-static multi-annotation box without attribute
WARNING d56a247daaae7e4cf36350ce512c8dd1: non-static multi-annotation box without attribute
WARNING 6a83bcdbc58c1710cd45dd37e3b7624e: non-static multi-annotation box without attribute
WARNING d0bf3fd9b7d2e2c7ed3fa93b5558fd53: non-static multi-annotation box without attribute
WARNING db2a6729e9e419df3c0bce284a1278f9: non-static multi-annotation box without attribute
WARNING 6c91d1059094b44deea43970f6c15a7d: non-static multi-annotation box without attribute
WARNING 659908804a08c2c1201df87bd2e5af28: non-static multi-annotation box without attribute
WARNING 85ef4fc80aaffb4ffbe0c71ef5efbb37: non-static multi-annotation box without attribute
WARNING 2016b8be10ad50965a7f349c9985984d: non-static multi-annotation box without attribute
WARNING 9bd08189672a60ad97cadd2c9a501545: non-static multi-annotation box without attribute
WARNING eb2e39819c68319aae1e47caa08b4993: non-static multi-annotation box without attribute
WARNING 46e2afc3397c73be7d129b1892f65984: non-static multi-annotation box without attribute
WARNING adc6e31d9e5ace4c1d835269dee21cd8: non-static multi-annotation box without attribute
WARNING 4ce462268e63588a69d88d507f940eac: non-static multi-annotation box without attribute
WARNING 2fca20ab10bcc14ee16007530ca8a74f: non-static multi-annotation box without attribute
WARNING dc0d1291f0125b0b06c29d21bd342b72: non-static multi-annotation box without attribute
WARNING 35c4c82546c48163118bdd6cbe030e84: non-static multi-annotation box without attribute
WARNING 81bd49586d7bf78c3544e20042e09636: non-static multi-annotation box without attribute
WARNING 509d3f7128636a3d4a303eba550250c3: non-static multi-annotation box without attribute
WARNING e7d8c248f8ff554c261eff885aa37f8b: non-static multi-annotation box without attribute
WARNING b04e7f7e33adb6b1837527c7c2ea9e69: non-static multi-annotation box without attribute
WARNING f4f78f97d7fd181119c221a3222cb94c: non-static multi-annotation box without attribute
WARNING 5fec0183d583913d55e840064036b93a: non-static multi-annotation box without attribute
WARNING 5dd143a1d24fa65fb480a81c7e3ea89f: non-static multi-annotation box without attribute
WARNING 2d2629236b17dea69f35b04f9c93aefa: non-static multi-annotation box without attribute
WARNING 773893258751eb11554a141cd176d28a: non-static multi-annotation box without attribute
WARNING 3e840bb97787c609b672a4836af0aafc: non-static multi-annotation box without attribute
WARNING d377e6d974aa3e67bc45b75d883beb32: non-static multi-annotation box without attribute
WARNING ac12685169150c2a4b5d51d489d41abe: non-static multi-annotation box without attribute
WARNING 775b98a4651ec4b71ba9ef78c8967726: non-static multi-annotation box without attribute
WARNING abc4bce2fab793eff15fca5ac0f00143: non-static multi-annotation box without attribute
WARNING 235f8d85060962c853de3121efd641c2: non-static multi-annotation box without attribute
WARNING e2ef4f88d455f26802c360f67eefa162: non-static multi-annotation box without attribute
WARNING cae0ad0112cd331889d712856dd9672b: non-static multi-annotation box without attribute
WARNING 9f15d8be940d024cd20797cd74424b26: non-static multi-annotation box without attribute
WARNING cb41f42a29f706a889309f91ca6bc2eb: non-static multi-annotation box without attribute
WARNING 16476eb6bfb706f8c050d66fe9886595: non-static multi-annotation box without attribute
WARNING b03c833204b69725b98dae467bc8bf5f: non-static multi-annotation box without attribute
WARNING 8ff8b47dcce8a977ecd50f05bc619e7b: non-static multi-annotation box without attribute
WARNING 20964ca88c876dadf321bbc3998a1819: non-static multi-annotation box without attribute
WARNING 7ac88c8b5a146a39ad2723ffe98516a4: non-static multi-annotation box without attribute
WARNING 92bb6e17e845bed31f168e867e579c3d: non-static multi-annotation box without attribute
WARNING 82b5f345969d982f115ebbddb854ad81: non-static multi-annotation box without attribute
WARNING 9539c91f08ef3719bda2c03b99283d7c: non-static multi-annotation box without attribute
WARNING 93df88939972db8d50a772a22e197118: non-static multi-annotation box without attribute
WARNING 19d87caf5728dd375da3d5fea6eed1af: non-static multi-annotation box without attribute
WARNING 1b0d8bdaf5ccf3d074ac81b9e10b1648: non-static multi-annotation box without attribute
WARNING 7178b6a3650d5c69e971c220ffd0d10c: non-static multi-annotation box without attribute
WARNING b0ec41bfa8db93cf1ebb09668cb43241: non-static multi-annotation box without attribute
WARNING 70e5ab278aa39df7a17d1e62f7b7bd52: non-static multi-annotation box without attribute
WARNING 540a1e51406a5e72556524e7155cc27c: non-static multi-annotation box without attribute
WARNING 51cf73a77a03faa726192c3cbd0b0871: non-static multi-annotation box without attribute
WARNING 20a7cc3cf7e2c4c6cc7245becd3e8d77: non-static multi-annotation box without attribute
WARNING 3670c40c906ebf584b153dcec18f0814: non-static multi-annotation box without attribute
WARNING ae8d27c26b53145e5b46d95752dd8860: non-static multi-annotation box without attribute
WARNING a533cc877f89a7d314c89014f5c4293d: non-static multi-annotation box without attribute
WARNING 6e108ec9a4cca5f7b23b9e2c8c10abcf: non-static multi-annotation box without attribute
WARNING 05eeba8a9b1151a3a6b358dea6b4019a: non-static multi-annotation box without attribute
WARNING 4562b1053b2aae86149af742347902a1: non-static multi-annotation box without attribute
WARNING da5b0cd8f3186e0b2009d47d05c485a0: non-static multi-annotation box without attribute
WARNING b931807b20d311906ec4c5d4586ef97c: non-static multi-annotation box without attribute
WARNING 8bfac0ab42bac5371c203325c1daa50c: non-static multi-annotation box without attribute
WARNING e238657ed2d3907d76d36381ff470928: non-static multi-annotation box without attribute
WARNING 46c29551a7d23792031da0173112fc76: non-static multi-annotation box without attribute
WARNING f54c3c61a2eb3193a9c4d24228567453: non-static multi-annotation box without attribute
WARNING 27f266d7c3e086ebdc7c6be6e07d82ab: non-static multi-annotation box without attribute
WARNING fa06cd89d240d843dcb3dcc9a244d208: non-static multi-annotation box without attribute
WARNING 7bb3e76b47f9dc3ef681621923dbbc80: non-static multi-annotation box without attribute
WARNING 5033723163d0fa0a35a22dd0cf9763ce: non-static multi-annotation box without attribute
WARNING 1c268f6025392570827ff76c97c441c2: non-static multi-annotation box without attribute
WARNING b0953ca1322961b41678c2b71beb129a: non-static multi-annotation box without attribute
WARNING f95fd9278bd199f029d0fbea79885469: non-static multi-annotation box without attribute
WARNING 25fa25880ba466a53f1a4f96d8f89946: non-static multi-annotation box without attribute
WARNING 5f738821356d74bd5f76d4462a848d81: non-static multi-annotation box without attribute
WARNING bd26deb12a87c0f1b253d521d5c637db: non-static multi-annotation box without attribute
WARNING 1d9b6a43cb8d8c0c402c26036976eeb1: non-static multi-annotation box without attribute
WARNING 62fb5b0949ca6b40b1a7fd8840e384a7: non-static multi-annotation box without attribute
WARNING e3e06e83fd858f483395c26d657d867d: non-static multi-annotation box without attribute
WARNING 77843097ece516da4a9618a1993c1f22: non-static multi-annotation box without attribute
WARNING 5c0cfc89055dfdacff33d3418cd21035: non-static multi-annotation box without attribute
WARNING f841afaa157415225232c015dd11914a: non-static multi-annotation box without attribute
WARNING f9ce4f356116174f1e0c29da88a9ef09: non-static multi-annotation box without attribute
WARNING 8abc1f0e38bc677d2584339b445ed77f: non-static multi-annotation box without attribute
WARNING 6476beee4c5c8ca8efd1c7a729e96df4: non-static multi-annotation box without attribute
WARNING f426e0c5a19988718178cb178bd368a9: non-static multi-annotation box without attribute
WARNING d0032effd3fbc8d8ca3186ee495e7cca: non-static multi-annotation box without attribute
WARNING 8c021c06bacfda6578cbc327cc0624c8: non-static multi-annotation box without attribute
WARNING f108d33101084c24c9c2f1139406a2b1: non-static multi-annotation box without attribute
WARNING a7cb2499b1011d2d46209ee66da44a35: non-static multi-annotation box without attribute
WARNING 2cbc5a7d9d08717d72da8854bbfc24d6: non-static multi-annotation box without attribute
WARNING a455a586917c303dedc9c6650e1fccd2: non-static multi-annotation box without attribute
WARNING d826e418e957bacf8b6b5eb76c7914ff: non-static multi-annotation box without attribute
WARNING f101b29f59b52ba4fc680f9c02141814: non-static multi-annotation box without attribute
WARNING a7f5a17997207ab784f5f63808bcdca5: non-static multi-annotation box without attribute
WARNING 1a4b596471205adfade5999aaabe310e: non-static multi-annotation box without attribute
WARNING 0813c96688aa0272a0d56085e651f660: non-static multi-annotation box without attribute
WARNING 9d481712ccbf0ad73b42132f89d1a7db: non-static multi-annotation box without attribute
WARNING 561ec588e1b1062afeb4ae3957d72096: non-static multi-annotation box without attribute
WARNING 795faf6e8ee0628eb67b02fa5b2897a8: non-static multi-annotation box without attribute
WARNING 2bbef58355e3b7877543094606925a49: non-static multi-annotation box without attribute
WARNING 342d3453d9dd27024a9ac39135458c18: non-static multi-annotation box without attribute
WARNING 1a9a62b58443f921af46473be58545e0: non-static multi-annotation box without attribute
WARNING 145f705c755194eeb7aacfa88e137786: non-static multi-annotation box without attribute
WARNING f3a90233d32540407953e98584f9d687: non-static multi-annotation box without attribute
WARNING 010dc1369965c6334cb5c8058e11d563: non-static multi-annotation box without attribute
WARNING ec7e3f0f617ea863c7d56e6de7095fd3: non-static multi-annotation box without attribute
WARNING 5567963d2017f83273b87c93b1272961: non-static multi-annotation box without attribute
WARNING 826102fbc3952105b338c58846b5f264: non-static multi-annotation box without attribute
WARNING f22e050368aa0f10e52f4c6d5cb6350f: non-static multi-annotation box without attribute
WARNING 25533061fa077a480dc72d183cca7550: non-static multi-annotation box without attribute
WARNING 7376084fe59b6253b9af5a47e1c99736: non-static multi-annotation box without attribute
WARNING 5eae8735df1d60a467f6177c195ec049: non-static multi-annotation box without attribute
WARNING 018518503a71ca7f397a94d61e39f8da: non-static multi-annotation box without attribute
WARNING 402424ce857d305545d326dfbf9ffd5a: non-static multi-annotation box without attribute
WARNING 9ba574e53bee048cdbf8e9ffdb4da742: non-static multi-annotation box without attribute
WARNING 9729bd6148a1d137fc29fd86e9d33af4: non-static multi-annotation box without attribute
WARNING a1b1935270bee3e83e3f402afc0f70fb: non-static multi-annotation box without attribute
WARNING 81c576961c9976fcc370468aab73e90d: non-static multi-annotation box without attribute
WARNING 12392a8c68d55b8eadeb09ef998a4f14: non-static multi-annotation box without attribute
WARNING a33d3608694526792821d4a07d8bc059: non-static multi-annotation box without attribute
WARNING 32dd50d8b81029be0810dbd5eda68e55: non-static multi-annotation box without attribute
WARNING 77f30164fdfaab29ab8a9ee3efb86404: non-static multi-annotation box without attribute
WARNING a34936668726fe611d09a19fe1377995: non-static multi-annotation box without attribute
WARNING d4c3573e5c49543dd5e772c3082ac4ba: non-static multi-annotation box without attribute
WARNING b1f2fe77d6257a87e947261c361ffbd1: non-static multi-annotation box without attribute
WARNING d2e3d92d1b10815bee0a713112bf88a0: non-static multi-annotation box without attribute
WARNING 43174f93a55286f5d7af515a2998066f: non-static multi-annotation box without attribute
WARNING d3d9d3e1b23f645d18810c15720d6e7f: non-static multi-annotation box without attribute
WARNING e14ce6e8b0d2a04849af3b633a92ac24: non-static multi-annotation box without attribute
WARNING 0096f2f61f7853af71902b8dc6c22fcb: non-static multi-annotation box without attribute
WARNING 9482cf273fcf0f7f9d9a4ce90738cb4d: non-static multi-annotation box without attribute
WARNING 6b1a0556bb0584554deb3f3e6b0ccd5b: non-static multi-annotation box without attribute
WARNING a29f0dc19f4c400edec63a732b841c83: non-static multi-annotation box without attribute
WARNING a0387ec6a0cd24616ec51a3c282abc9d: non-static multi-annotation box without attribute
WARNING 23c1439159ab0483d6c76aacfc774bfe: non-static multi-annotation box without attribute
WARNING 0b52472455cc00b0eb8984557ea14721: non-static multi-annotation box without attribute
WARNING 7622af70e3ac5e54eccb7a11038bc254: non-static multi-annotation box without attribute
WARNING c5289fa9c674036d527103d53f85f6da: non-static multi-annotation box without attribute
WARNING 245a6eeb5fd785622fb4557b083fb69e: non-static multi-annotation box without attribute
WARNING be08cfe62454f0dc07039e1633e4bcdf: non-static multi-annotation box without attribute
WARNING 31272c05227cc76bc4fe123600386ff4: non-static multi-annotation box without attribute
WARNING d7e40209fc5d6cc15ec1956dd631786a: non-static multi-annotation box without attribute
WARNING 4aff26b196921903bd29b8a0e8f9e77f: non-static multi-annotation box without attribute
WARNING eae8167e8056320cef362b458ac31bea: non-static multi-annotation box without attribute
WARNING 963b238ad5fc5e428385071186f5fb77: non-static multi-annotation box without attribute
WARNING 0ae6ed2def96e4e4a81b4a915058975d: non-static multi-annotation box without attribute
WARNING be3596b7f557e3bd609999d682ec3912: non-static multi-annotation box without attribute
WARNING 20bbb1e0a3ebb09bba7e1c450b0d0aa3: non-static multi-annotation box without attribute
WARNING bf72064ece375fc118705358f39fea7e: non-static multi-annotation box without attribute
WARNING e55e4e81e6d649bb1fb357c713f907cf: non-static multi-annotation box without attribute
WARNING 89f6655296929a10a2c78016d8d2c9f2: non-static multi-annotation box without attribute
WARNING 8fa92b85d1f7e297ca8495f8c8be2a34: non-static multi-annotation box without attribute
WARNING f32e655e60132812a37e7f85c88cc21f: non-static multi-annotation box without attribute
WARNING 0098c34109f55e5d1732befcb8243de1: non-static multi-annotation box without attribute
WARNING e35c98a23dc1aba2393e872eadc32a2b: non-static multi-annotation box without attribute
WARNING 542ab9a105f3779f0b74e82a9ce25fad: non-static multi-annotation box without attribute
WARNING d51ff04c5cd67d21f9bf15e1413cbb49: non-static multi-annotation box without attribute
WARNING 08d0e08aa192cd77cd48dd8e557121d4: non-static multi-annotation box without attribute
WARNING a8ea2d7212e9ac8fd604dd63acce13dc: non-static multi-annotation box without attribute
WARNING 5c0216d71a887228f015bf15902a9ce7: non-static multi-annotation box without attribute
WARNING 59bd1a7891ebc0fb44f8d4d5aba3e08c: non-static multi-annotation box without attribute
WARNING f5a48d4ce67766c6b5faabc67c5c193a: non-static multi-annotation box without attribute
WARNING 5615e0423aa23a81b37de209d913dde3: non-static multi-annotation box without attribute
WARNING 8f3d3caa55dd5657ee461e1f98374360: non-static multi-annotation box without attribute
WARNING 56b63ef06a4758687c1e5eb23b529e51: non-static multi-annotation box without attribute
WARNING 73eb24361353ef45e6c540e23e54b8ce: non-static multi-annotation box without attribute
WARNING 21e1daaca45e8224df23d57fc786cd09: non-static multi-annotation box without attribute
WARNING 5ee5a9d05281ed1b894d320cb0395c91: non-static multi-annotation box without attribute
WARNING f53b87da788d9b53f1f07bf747770ceb: non-static multi-annotation box without attribute
WARNING 431ae6990b42f76ec87c5d1d1c6c267e: non-static multi-annotation box without attribute
WARNING 4e66b6cf5c3eb151552031762beb0d35: non-static multi-annotation box without attribute
WARNING 613990f7dc61518fed26e8b61c676d66: non-static multi-annotation box without attribute
WARNING e7d400a4b4bfb2f222b9b34e11294dc9: non-static multi-annotation box without attribute
WARNING 4da89b162e0e0b9d130c99b15e04c4ce: non-static multi-annotation box without attribute
WARNING 250c17052f123d599c8e9539bef46892: non-static multi-annotation box without attribute
WARNING 340a66783aa77feb0693e837eb6a002e: non-static multi-annotation box without attribute
WARNING f4fbf64ae3084d4e5ed82c72a5242619: non-static multi-annotation box without attribute
WARNING fefe7f687515390de53a8c49fb1eb956: non-static multi-annotation box without attribute
WARNING 3366c7e7b9aacb4c891d432fcecd0f67: non-static multi-annotation box without attribute
WARNING 8bc0293c63fb7dad6c6cbb591fb07e5b: non-static multi-annotation box without attribute
WARNING a2d7016d62abd9d6be588d4758f687d4: non-static multi-annotation box without attribute
WARNING 280396c967cde9b5ac5659235268a8cd: non-static multi-annotation box without attribute
WARNING cbc7d9ca3dfd4675ab6f75f5e855647a: non-static multi-annotation box without attribute
WARNING 8234873a54a4ccbd478c8c5e8e19b59d: non-static multi-annotation box without attribute
WARNING 8f5964ea1e33a5476e6e22516221b4d9: non-static multi-annotation box without attribute
WARNING 5d2bd3e7c3a1d5626c9a7c3cee07df2e: non-static multi-annotation box without attribute
WARNING 3b865ef4031a8d4bace10ea7886f83e7: non-static multi-annotation box without attribute
WARNING 0443ea8f3e4b7e2c46a468c93a973be9: non-static multi-annotation box without attribute
WARNING 5a4858b219272868c6dedbdc3c686496: non-static multi-annotation box without attribute
WARNING b85158b50c9c27b6d253392715c2ef85: non-static multi-annotation box without attribute
WARNING 5d95346f738cc8ac36fd16d9dd915f5b: non-static multi-annotation box without attribute
WARNING 6522135ea43a809095986e716fd5eb1e: non-static multi-annotation box without attribute
WARNING 4dfc82b9ec082d072b348eb1ea53308c: non-static multi-annotation box without attribute
WARNING 4a852aad37ffcd0e491b797c585ca381: non-static multi-annotation box without attribute
WARNING 6703c96627713e4ec8d141a8e49d6dcd: non-static multi-annotation box without attribute
WARNING 0765d0f60970ec17b6174b050c6b17fd: non-static multi-annotation box without attribute
WARNING 275b11a6829c7a23cdc8fc1ad9266d57: non-static multi-annotation box without attribute
WARNING 27a04171d8e095e39b84b7d2cfd86173: non-static multi-annotation box without attribute
WARNING e4089973e23bf0d41947537194dde4c7: non-static multi-annotation box without attribute
WARNING efeddc2622f1504d0a76aaadf9a601cf: non-static multi-annotation box without attribute
WARNING 22575e72228d7fb84af5447097d11ac7: non-static multi-annotation box without attribute
WARNING 60e977cdc1ba3f8ca48f71e4ee526577: non-static multi-annotation box without attribute
WARNING b1eaf89f8752909dbfb06deb0ad3a35f: non-static multi-annotation box without attribute
WARNING a589303e66ea3f58540c5650acfd229b: non-static multi-annotation box without attribute
WARNING bbdf5edcb081a9815bceb7f6f079ad2d: non-static multi-annotation box without attribute
WARNING e2d8d300082745d6e847ffb220a32f68: non-static multi-annotation box without attribute
WARNING 4efbcd62ba16e605cddb86c8df37f211: non-static multi-annotation box without attribute
WARNING df4c5ebc1943a56650886b1033c845f8: non-static multi-annotation box without attribute
WARNING 1f0891f8eeb32caf261159b206cbf421: non-static multi-annotation box without attribute
WARNING 6943b281674d5b5d569565514f0805b0: non-static multi-annotation box without attribute
WARNING 9e957f9695d5b6cf7cf3d3ae07ae1664: non-static multi-annotation box without attribute
WARNING e2a4d20d1d95e0a834fbd1ada5d416c5: non-static multi-annotation box without attribute
WARNING eb498b6cccb1fd1bbf9a34777b07844e: non-static multi-annotation box without attribute
WARNING e054514e5e89a30283180da5e4ef3c41: non-static multi-annotation box without attribute
WARNING cf2386af6abcfc34cb9f8bac1862c074: non-static multi-annotation box without attribute
WARNING 2794743ca88a4a912dbb0cf88f8a7b0d: non-static multi-annotation box without attribute
WARNING 6e1b4a5308ffd3cdc426ecd5215efe44: non-static multi-annotation box without attribute
WARNING e988ee8898fffbbefb671a1efcfcd56c: non-static multi-annotation box without attribute
WARNING 095fb06489052fc01af3872bc9961b46: non-static multi-annotation box without attribute
WARNING 3d7e66b87825770d2ae43829ac22ef95: non-static multi-annotation box without attribute
WARNING ee2c4d50a9b0bdf5a7736f890eab9142: non-static multi-annotation box without attribute
WARNING f60a34a7f60899602c090a6f1f72a567: non-static multi-annotation box without attribute
WARNING 0001bea9895630955f2388ef9e237742: non-static multi-annotation box without attribute
WARNING 2f3f83881d0c57c97d7ec5527c9bc06e: non-static multi-annotation box without attribute
WARNING 1c5b82a3395db6e01b34ebb5a5866e1b: non-static multi-annotation box without attribute
WARNING d7ec58b402e97f7c4441ca9a897fa341: non-static multi-annotation box without attribute
WARNING 4fdc860c4104e0b51c4022c264b657d2: non-static multi-annotation box without attribute
WARNING ff22bd9e2d2dca06b7b7e507c0ccb5db: non-static multi-annotation box without attribute
WARNING 6c84bc464e1454d33ef2804c5ef79c8e: non-static multi-annotation box without attribute
WARNING dd53f9fed71855b5f3f8dd5d9f202dec: non-static multi-annotation box without attribute
WARNING 5ab5f3fe903fdf264e831c4416e2711b: non-static multi-annotation box without attribute
WARNING 823e02f68760110eead68142141277eb: non-static multi-annotation box without attribute
WARNING 0cb389a0f9ee621c2a6d9919ed911328: non-static multi-annotation box without attribute
WARNING 51df8ac0515d5ea82ccfc131497a7240: non-static multi-annotation box without attribute
WARNING 148897bcef8e6476c1a8d43133003174: non-static multi-annotation box without attribute
WARNING e1b5b8037095377f0784f9c094ac63f0: non-static multi-annotation box without attribute
WARNING 722f9fc4cc980c709c4f88c445cb531e: non-static multi-annotation box without attribute
WARNING d57c76a37924874419fddd03b4eaa694: non-static multi-annotation box without attribute
WARNING 6828774594fcdb91055f0f9fce535aa3: non-static multi-annotation box without attribute
WARNING abe2291c43561dc67d27c4a105302351: non-static multi-annotation box without attribute
WARNING bd94f969455c7a1dc781458840c88407: non-static multi-annotation box without attribute
WARNING 7b65409889befc4341333c5c1923b7d2: non-static multi-annotation box without attribute
WARNING d967a5d61f2e2e5b7a0e6aa9349d4728: non-static multi-annotation box without attribute
WARNING e87d36c373ac4be1ddbf4d1cd57f9314: non-static multi-annotation box without attribute
WARNING 455366777ca06fef15f9e42ba9093bbb: non-static multi-annotation box without attribute
WARNING 023215e7a8d0b60a521f71106da5746a: non-static multi-annotation box without attribute
WARNING 92d39517348016178f2b03bf999bdf78: non-static multi-annotation box without attribute
WARNING 1466302ab4bd78ee79d1e4bd1bf59fe1: non-static multi-annotation box without attribute
WARNING 5d422219176f9e17ecbff7cb358539c7: non-static multi-annotation box without attribute
WARNING ba3c4a7edbb39a7f0c6cea024156e4a5: non-static multi-annotation box without attribute
WARNING d88a966b3e3d891c289a49e7a19a3cca: non-static multi-annotation box without attribute
WARNING e92460445088adf3c04d9fb96393ec52: non-static multi-annotation box without attribute
WARNING bf6d19478dca75c84e8e7007807ed243: non-static multi-annotation box without attribute
WARNING d9a1a58c68a5e807e0e025d2bac8104e: non-static multi-annotation box without attribute
WARNING 91b5106539721c1aef6f9bcdc03f93f3: non-static multi-annotation box without attribute
WARNING 2daf6bbcb18e7308ba74c8e400645cc0: non-static multi-annotation box without attribute
WARNING 93835714b95f22ded3b1cb532e7174f1: non-static multi-annotation box without attribute
WARNING fdee635a34094e71f320485f39ca732b: non-static multi-annotation box without attribute
WARNING 3239261e9c2d59fc72926f6f39566efb: non-static multi-annotation box without attribute
WARNING 4b00bf6a4a0d0cd4c8fa797aa65d2e46: non-static multi-annotation box without attribute
WARNING c6729b78f5fd88a4e06574789ccd9eda: non-static multi-annotation box without attribute
WARNING f0425c007778e820d447a0df39c3fea2: non-static multi-annotation box without attribute
WARNING 7f8b9585633be6deadd5edd269488639: non-static multi-annotation box without attribute
WARNING 0706b517e7f39ccda66d7b0c01976268: non-static multi-annotation box without attribute
WARNING 2c8710e82d7d28d86aa4948b3fb83c7a: non-static multi-annotation box without attribute
WARNING 9d6b9e1a1dc77cfffea81a6d1c3a3d1c: non-static multi-annotation box without attribute
WARNING 4df0ea8b8c78440ede2f2a2368f584cb: non-static multi-annotation box without attribute
WARNING a25b8e0dd61bb94658b472702fd49b81: non-static multi-annotation box without attribute
WARNING be227d57b0d690967d12737e51625038: non-static multi-annotation box without attribute
WARNING e08fece75ceb84d2810bdc7952a75d5e: non-static multi-annotation box without attribute
WARNING c6022eb7426a8cd10f28ae55a0b95cb7: non-static multi-annotation box without attribute
WARNING 34a376bbbab4c060544b6f01e530a31f: non-static multi-annotation box without attribute
WARNING 19d48efc4dd2ece8ebe53781e409b8eb: non-static multi-annotation box without attribute
WARNING 616c8add74d6b8c37ab7ad71c47dac9b: non-static multi-annotation box without attribute
WARNING 7026c3c68803e0011785d426c7aa14b6: non-static multi-annotation box without attribute
WARNING 4e80c58a0d742f6d9ccfd2987392ba78: non-static multi-annotation box without attribute
WARNING d4b396149fb5cae831e9371aa29520f8: non-static multi-annotation box without attribute
WARNING bf5b543e2f5a65b68bdadbad552d9000: non-static multi-annotation box without attribute
WARNING b3b13699799fe50cd9e1c4618a563d5a: non-static multi-annotation box without attribute
WARNING 73727cf02937cfe1aacb52a6a3a4218c: non-static multi-annotation box without attribute
WARNING 93a37dbeb02eca5b4fe1752b98f11635: non-static multi-annotation box without attribute
WARNING 7f25f00ba2f5f8a37a4a63d5f1f6dc43: non-static multi-annotation box without attribute
WARNING c89eb2b292be14cb34c0f97e4f27c32d: non-static multi-annotation box without attribute
WARNING 34f695704db4dd13d0859d0d009887fd: non-static multi-annotation box without attribute
WARNING 2c769822f2f8dba65289dda528fcdf47: non-static multi-annotation box without attribute
WARNING b33ee385a3c24dad6a8c41a7ccacffbc: non-static multi-annotation box without attribute
WARNING 2df8c7eed1a2e1267194ba6cdc63f9be: non-static multi-annotation box without attribute
WARNING 3a851e2cd061f9f38cf506772bd6cc3c: non-static multi-annotation box without attribute
WARNING 9d1af0ae5f7ad9b4b4ea6a9bb11841fa: non-static multi-annotation box without attribute
WARNING d575c64876f9e24d5999f2e09e0e05dc: non-static multi-annotation box without attribute
WARNING 5e0cfa61b1dd1e7e0677d43cf6db79be: non-static multi-annotation box without attribute
WARNING ff0365c67467fb3ca7208ccf1fc5d7d8: non-static multi-annotation box without attribute
WARNING b2b5f0798af51206a573b14380b59874: non-static multi-annotation box without attribute
WARNING a8215ad767586be2eb573549a4a8688b: non-static multi-annotation box without attribute
WARNING 8f480a3c3be6fe9e179ad616ecbfab13: non-static multi-annotation box without attribute
WARNING e03dd2ad7074d020922db462f4d81ae7: non-static multi-annotation box without attribute
WARNING 71df268d7cd0c631a51d5c480a1d49ea: non-static multi-annotation box without attribute
WARNING cfd616a3d947fdf2447e7820c397bd52: non-static multi-annotation box without attribute
WARNING eded0ec7bbd3dd946635feaf1bee053b: non-static multi-annotation box without attribute
WARNING 348529687c58ef1187bc844bf4577e2a: non-static multi-annotation box without attribute
WARNING f046af1ce858769f9a3064a4fb78cbf7: non-static multi-annotation box without attribute
WARNING 304ad741ce64ea07b9d885600d8792a8: non-static multi-annotation box without attribute
WARNING e770b0fb58e1e574e82f684c95ba1b3b: non-static multi-annotation box without attribute
WARNING bde971530c985eb0e045e5359d90ecb9: non-static multi-annotation box without attribute
WARNING 30cd18b3dfbdb9c155479073440d9e50: non-static multi-annotation box without attribute
WARNING 120fcb709c1f722a16ea71ba2a482cf3: non-static multi-annotation box without attribute
WARNING bdebb139226350355e0164c69dc3960d: non-static multi-annotation box without attribute
WARNING e4220bf68df81456554fc98c01e8d6f7: non-static multi-annotation box without attribute
WARNING fa67a08d64f7962a32f27fbc3b823515: non-static multi-annotation box without attribute
WARNING e96f2515c514dc9dae9f0f4d441a260e: non-static multi-annotation box without attribute
WARNING caa5812bf65190989e3be7131cfac868: non-static multi-annotation box without attribute
WARNING 3aa4ec6eb5ca4aea8a5857e7d1eb83d9: non-static multi-annotation box without attribute
WARNING c922b300eb71afac94835dd124f45d06: non-static multi-annotation box without attribute
WARNING 8f49a8c51837ca859f326cf5a4807d43: non-static multi-annotation box without attribute
WARNING cf8e0e6641e1eb31a0eed63e77f0d7fc: non-static multi-annotation box without attribute
WARNING 10a41d3ce91802b304f2ffd6a5d6be81: non-static multi-annotation box without attribute
WARNING 4e5fbf9c70977cdeeba29ed969b9b7a5: non-static multi-annotation box without attribute
WARNING facb932fdf3aa8fdfda6c1cca1aa4f65: non-static multi-annotation box without attribute
WARNING fe83c42b35ca337da2d36698ad10a2c8: non-static multi-annotation box without attribute
WARNING 9734cb5432cb5a501db94a8a705bf2f6: non-static multi-annotation box without attribute
WARNING ec81c1b03de28d686e371632b8b2ae7b: non-static multi-annotation box without attribute
WARNING a4b2707ae8e99b0fb3603d46f9903693: non-static multi-annotation box without attribute
WARNING 71135a13c6151cef21317ac5da01c5e9: non-static multi-annotation box without attribute
WARNING b6781a807b60094320194cd1e2e6ef93: non-static multi-annotation box without attribute
WARNING aaee2cb2fba64abe944ba891992ef50d: non-static multi-annotation box without attribute
WARNING a5cd30266a7265c63c73cd60b517eddb: non-static multi-annotation box without attribute
WARNING cd97726155d0cb5c3f731e9771a6f74c: non-static multi-annotation box without attribute
WARNING 230891491cdc79548d05151d030f4050: non-static multi-annotation box without attribute
WARNING f89f549971e000be3c31c384ec454f1e: non-static multi-annotation box without attribute
WARNING c8b5d405e4ca87e29473de469013f23e: non-static multi-annotation box without attribute
WARNING fa7d54a778b3ec11c381be38d9fb0ee0: non-static multi-annotation box without attribute
WARNING e537230dd6665f088144a995f1fda85c: non-static multi-annotation box without attribute
WARNING 330ffcf3a009484da91ddbc4972e74f2: non-static multi-annotation box without attribute
WARNING f85c529df4cc7312a3d8396a92d02046: non-static multi-annotation box without attribute
WARNING d296c0e922650a107ededa2567e6d0c7: non-static multi-annotation box without attribute
WARNING 0ecbfe6dc9702d8ab7f7548f5a3451d7: non-static multi-annotation box without attribute
WARNING 7aca4c3ade22ad11bb451e422323d6ec: non-static multi-annotation box without attribute
WARNING 6fb15e038c8a4856a1bb3d301cf41b77: non-static multi-annotation box without attribute
WARNING 778aabb57b6e655dbfee2a0fe825557e: non-static multi-annotation box without attribute
WARNING 85ca5ec2727e105b18438d526248a8ed: non-static multi-annotation box without attribute
WARNING b73b4a359533a78f9e7af3abbc95b377: non-static multi-annotation box without attribute
WARNING 6e7d401201a43e4607634eebde28c4e3: non-static multi-annotation box without attribute
WARNING 8c9a79e75476abc90cf432f88c524450: non-static multi-annotation box without attribute
WARNING c3eda2555638ec7d17090bd210badfce: non-static multi-annotation box without attribute
WARNING abfa18ba379d7d40e128171209e77a13: non-static multi-annotation box without attribute
WARNING d1a2c783ba4612446ec8be9bbba99a05: non-static multi-annotation box without attribute
WARNING 856a23ede617afbb6c8187dc88673780: non-static multi-annotation box without attribute
WARNING 8994bc7f61e3f729093ad9304eeb649a: non-static multi-annotation box without attribute
WARNING ddae560da9d29040f07c669779b04368: non-static multi-annotation box without attribute
WARNING 477a635d2859d085529ca52b401aead3: non-static multi-annotation box without attribute
WARNING ab3bbcfd58c6e69d099891b5ed5d9bb3: non-static multi-annotation box without attribute
WARNING f3f1158cfbddfd619637088af9fd244b: non-static multi-annotation box without attribute
WARNING 99dbf4fb4750835befdb586bd587c77f: non-static multi-annotation box without attribute
WARNING e5cdff57a7bb03cd9d2144523abc5222: non-static multi-annotation box without attribute
WARNING 4b430d7ab2ab1f12c0e46b8b982f05d4: non-static multi-annotation box without attribute
WARNING 6f621abab40d582deb2adbfa8e45cff6: non-static multi-annotation box without attribute
WARNING 205a02af039ca006229e2cad830e8927: non-static multi-annotation box without attribute
WARNING e27af52c167dc0aaec69318d96f2e950: non-static multi-annotation box without attribute
WARNING 80882b7d7597bbffd9a117919d31f945: non-static multi-annotation box without attribute
WARNING 9d87d5a13291f4b56dcc5c89394100d1: non-static multi-annotation box without attribute
WARNING 906d587cc2bcd0261949c970b517013f: non-static multi-annotation box without attribute
WARNING 1fed5f8ee0fe8ab06726a29ab0459cde: non-static multi-annotation box without attribute
WARNING 9ad4efdef4d33303c73e7077618d0007: non-static multi-annotation box without attribute
WARNING d69682d34e21872924fa77032796d34f: non-static multi-annotation box without attribute
WARNING 1056391dfd89d25db2e4bc5414ac355a: non-static multi-annotation box without attribute
WARNING 8a364c5ccaede1b8d9e3ab7085bd0247: non-static multi-annotation box without attribute
WARNING d0aaf11751dccf190d948446d912cc52: non-static multi-annotation box without attribute
WARNING acc0ed5853b48ebc0c413aab3fddc4cf: non-static multi-annotation box without attribute
WARNING a4b3fee24a2d19cf3f9e92c6ae765bcc: non-static multi-annotation box without attribute
WARNING cbd14e5c4f9147771b9e81fb73b889e0: non-static multi-annotation box without attribute
WARNING dd65d83e0a5f6eaece98c46b4255f0f5: non-static multi-annotation box without attribute
WARNING c93afb285d23a0b913bd9b08b25f066c: non-static multi-annotation box without attribute
WARNING 16a827a45481f1cae2f253cb4719a124: non-static multi-annotation box without attribute
WARNING 59ea17d83838177472fb2b3f25b41a7b: non-static multi-annotation box without attribute
WARNING 258833a7ce15fd4be588a02fbfc04614: non-static multi-annotation box without attribute
WARNING fdf0c8eafce5dc320b7468a238018617: non-static multi-annotation box without attribute
WARNING 742c2c37008a8362ee583cf719249db1: non-static multi-annotation box without attribute
WARNING 0f72a2fc1377c9b5737c819453ea073a: non-static multi-annotation box without attribute
WARNING 17d271cb4d66c836b03709599e6fffae: non-static multi-annotation box without attribute
WARNING 00a0c321a2eb8a843c8d97a69ef5d0ef: non-static multi-annotation box without attribute
WARNING 4b7764acd58bbaeb8c3745ef10eb9c44: non-static multi-annotation box without attribute
WARNING e0a23aad53cf30d8b8edc0479de11794: non-static multi-annotation box without attribute
WARNING 2db96059a07742d3bbb4070dc1f73bab: non-static multi-annotation box without attribute
WARNING 073719e124e5a52bae15ed33579dbd29: non-static multi-annotation box without attribute
WARNING 8380fbafb11701cc8295d1e7d8b1332e: non-static multi-annotation box without attribute
WARNING 7af159d485798608fce05a96a0ae37f4: non-static multi-annotation box without attribute
WARNING 27c74cf27b4c7485bfa0e3f784684f5f: non-static multi-annotation box without attribute
WARNING bf67bdc9886a2200388c53c2829191d2: non-static multi-annotation box without attribute
WARNING b01a56cc66b500f1cb0ae31d21aa16d7: non-static multi-annotation box without attribute
WARNING 5433ab32c007e6e68d606d7864f73bde: non-static multi-annotation box without attribute
WARNING 1acf2489f929add6ecf3698b1583c501: non-static multi-annotation box without attribute
WARNING dae67b82f285ee2ae3769aa7b990a831: non-static multi-annotation box without attribute
WARNING 08651eec7a1c051a03dbfbd35eb606e8: non-static multi-annotation box without attribute
WARNING f528876f398d73525f302a2366f06ed9: non-static multi-annotation box without attribute
WARNING 4eea09a1f01db98343c33f043342ba58: non-static multi-annotation box without attribute
WARNING 68a16f779bdb07f8241471857f93b36c: non-static multi-annotation box without attribute
WARNING 34a52210cd93ccaad6ab21557020a713: non-static multi-annotation box without attribute
WARNING d1a7cd71b202bcc28c5c13ab5493ff39: non-static multi-annotation box without attribute
WARNING 0a0758de2ec91fb4e62f8b63c2d8ffa1: non-static multi-annotation box without attribute
WARNING 1113c6b8b166d09017e3d501f7c89dd2: non-static multi-annotation box without attribute
WARNING c1c832cf4d7f977c3134d6bdd7cd1b3e: non-static multi-annotation box without attribute
WARNING 0ba3d0213c00af5724ed46559367c18a: non-static multi-annotation box without attribute
WARNING 56aef8ba88ba3c1eda52339946e301dc: non-static multi-annotation box without attribute
WARNING 95cae2bb6f0dde9f5733f1acbfe41dea: non-static multi-annotation box without attribute
WARNING c3730952568c75f500813f3510ea11e9: non-static multi-annotation box without attribute
WARNING 7ceb353d8ddd894d43c5341c35eac40a: non-static multi-annotation box without attribute
WARNING 27dd7ca628325c3d264f74bf8acb153c: non-static multi-annotation box without attribute
WARNING 133a60aa65727ecfa46f1d394510a0ea: non-static multi-annotation box without attribute
WARNING 6220a08650f2b32ba7e45c0e22f7010c: non-static multi-annotation box without attribute
WARNING bdb9c3340b5f4fab2d8c7d331c7661d6: non-static multi-annotation box without attribute
WARNING a0c0c23e876a9ca7292833aeabcdbd1e: non-static multi-annotation box without attribute
WARNING f86dda4b43ed2c905722b1aa4084d9f1: non-static multi-annotation box without attribute
WARNING 2b6911fa04faca3e6c3f1847d41f4fa2: non-static multi-annotation box without attribute
WARNING d62a2b2d253de6047ad87374f10cdfec: non-static multi-annotation box without attribute
WARNING 02fc5a86dc1c951ba13b543d27ef9414: non-static multi-annotation box without attribute
WARNING 5f5dfabb6ff8f374186d0c3ed5a107a3: non-static multi-annotation box without attribute
WARNING f59d55e2df4369ed1ca9cfbae07af740: non-static multi-annotation box without attribute
WARNING 4c4d245c7af78d84982e09491b7f40d2: non-static multi-annotation box without attribute
WARNING 1942dce3789b8c50c68c632302897cc0: non-static multi-annotation box without attribute
WARNING e439b1c856235e828fddb84ff8c8b8ed: non-static multi-annotation box without attribute
WARNING b879e703cdec38915f7217a9f8f40f43: non-static multi-annotation box without attribute
WARNING 134e44840e1a437d3e86e245ab5ab5ac: non-static multi-annotation box without attribute
WARNING 19df2778afd911dfd800ff234811f0a2: non-static multi-annotation box without attribute
WARNING 727d4d6980ae8480e653df22c6369913: non-static multi-annotation box without attribute
WARNING fcd0f1f349cec2383495721fde121062: non-static multi-annotation box without attribute
WARNING a231e0d996b8a5a7b5c6508dc9083b98: non-static multi-annotation box without attribute
WARNING db8a5c6bb3f88537e57d8491db7c56e6: non-static multi-annotation box without attribute
WARNING 9d2d316b0ed767818f08dd8b370e7a07: non-static multi-annotation box without attribute
WARNING a959fa36a16c387a747ddcd580260114: non-static multi-annotation box without attribute
WARNING 2aa62ed4dad93d7ec24f702cccd1091e: non-static multi-annotation box without attribute
WARNING 82dd56052e3c2b84f06a3501c5497e70: non-static multi-annotation box without attribute
WARNING 07919e3c2d211d60bc0faae437b7fa9f: non-static multi-annotation box without attribute
WARNING c3aea4889129415bf76205344c6a821f: non-static multi-annotation box without attribute
WARNING 511a3b036d0e32e9203e074a5b05c7eb: non-static multi-annotation box without attribute
WARNING c543826b5e399e914715651d916521c0: non-static multi-annotation box without attribute
WARNING c52c6df59b4808004d3aa8957245145d: non-static multi-annotation box without attribute
WARNING 857975e0467e743eda26756d47185bc0: non-static multi-annotation box without attribute
WARNING 767c94b1ed318939e948ba2390cc7b47: non-static multi-annotation box without attribute
WARNING a0fa5a1cf21946d2cb70233afedbb7f1: non-static multi-annotation box without attribute
WARNING 62c99ffc23453830b688b3c72882316f: non-static multi-annotation box without attribute
WARNING 4bb27228c1c7e984cd7d9b5cb430a0fd: non-static multi-annotation box without attribute
WARNING 430d249b68e56f2fabedbeefaaca8030: non-static multi-annotation box without attribute
WARNING d244486ab66be58b1a648ad2fb1ea818: non-static multi-annotation box without attribute
WARNING 5be819f8bf336dabd776f1fa28665751: non-static multi-annotation box without attribute
WARNING 997cf66149082dabd522ece2e5bda2d2: non-static multi-annotation box without attribute
WARNING b96811b415f4269c310d2f5da8bc6700: non-static multi-annotation box without attribute
WARNING ad29ea32c947c01aae8170c6a8997ea4: non-static multi-annotation box without attribute
WARNING 0e33cc318264380cd258bf3e2931f434: non-static multi-annotation box without attribute
WARNING c85bc2cb292eb41082d72fd7a580fc03: non-static multi-annotation box without attribute
WARNING 7b781b898d7c20b9affbb7fab68cb65e: non-static multi-annotation box without attribute
WARNING 119b6306c69d3f1479e0dfc492395363: non-static multi-annotation box without attribute
WARNING 47db60af09791f1223f43bded04ad547: non-static multi-annotation box without attribute
WARNING d9f5f5574c373b4a8cdfe792ba926a62: non-static multi-annotation box without attribute
WARNING 858461e83debfbe3e9854ce1385de987: non-static multi-annotation box without attribute
WARNING 60a89aa476987d1121d13e7bd13c0f97: non-static multi-annotation box without attribute
WARNING 7987d1f83e866b7986e49dabb4fae70d: non-static multi-annotation box without attribute
WARNING dbf7dcb8cd70c2fd2bd215f4ba5b1528: non-static multi-annotation box without attribute
WARNING fa66d565b0afa0adbe4ced4387ff759f: non-static multi-annotation box without attribute
WARNING abf86e86c581d2dc344ae0538c550f64: non-static multi-annotation box without attribute
WARNING cac436f9ea0e9352f6464e12c7ae4860: non-static multi-annotation box without attribute
WARNING cd7ba76cfcd8ccfadaa79d1fdfc7ae88: non-static multi-annotation box without attribute
WARNING 21ac6e5dce26609e70ec325c567e8922: non-static multi-annotation box without attribute
WARNING 5899e31407ab71afa40ab5abc35d6595: non-static multi-annotation box without attribute
WARNING fb649f90e76800776b5db595c02db135: non-static multi-annotation box without attribute
WARNING 9f4bc218c4be9ccdb70891801555e4a4: non-static multi-annotation box without attribute
WARNING c2cfe49aecdc4f7620d53ae9f72e9cad: non-static multi-annotation box without attribute
WARNING 54355877044d5e4cb789faa87841de4d: non-static multi-annotation box without attribute
WARNING 7a5d1fca04f893e5f0a9acc652d6e4bc: non-static multi-annotation box without attribute
WARNING b858d4966bc41f59b462d2410707dbf0: non-static multi-annotation box without attribute
WARNING 3ff2d9905e2be61da958a149d1253451: non-static multi-annotation box without attribute
WARNING d36d547f178b3c33029fad94e153b229: non-static multi-annotation box without attribute
WARNING fa9d6ee40f28fd112cb304255f45659f: non-static multi-annotation box without attribute
WARNING 5e4b4be2a5c48d30f1da434ca892c55d: non-static multi-annotation box without attribute
WARNING 0fd9d472584ccf9c34d92e72f5f6f53e: non-static multi-annotation box without attribute
WARNING a0e9b7ca46e15820472bfc1d18a1ec2c: non-static multi-annotation box without attribute
WARNING 214a71cf825baa7ac96423085685becd: non-static multi-annotation box without attribute
WARNING 2c95e22dc191a1c3280d08cc4cdd2582: non-static multi-annotation box without attribute
WARNING aaebfff63c48b1ab561c2625acb91c99: non-static multi-annotation box without attribute
WARNING d65217ecf7fd2c67d2aa53628fee1e47: non-static multi-annotation box without attribute
WARNING cfa322482173d77ccee7ba3112709b3c: non-static multi-annotation box without attribute
WARNING 90819708ab595a98601ae93e096670f7: non-static multi-annotation box without attribute
WARNING 1d61d2a393d5f5aba44c0cdf39e34ebc: non-static multi-annotation box without attribute
WARNING 77c97d7966afc08ae22f7e8a74fc4a7c: non-static multi-annotation box without attribute
WARNING 1d00676485cb793783ec457ddbe6c372: non-static multi-annotation box without attribute
WARNING 52792094a3c548405ef3ef9d7d573768: non-static multi-annotation box without attribute
WARNING c630ee04d400668423462d0d7a896589: non-static multi-annotation box without attribute
WARNING 1e6b6b944283d88515176cc389cdbb35: non-static multi-annotation box without attribute
WARNING 35a91f0afab424b9a4b205afe6827df7: non-static multi-annotation box without attribute
WARNING 7c9dffef52b5a628aec46efec8b51fe0: non-static multi-annotation box without attribute
WARNING 6f5513f1b3d8e71f5947aa6e7713df84: non-static multi-annotation box without attribute
WARNING d8eb51d6d6e7bf0065029207992e976a: non-static multi-annotation box without attribute
WARNING 5c2164a2d3c289071cbb94f5a6349cd1: non-static multi-annotation box without attribute
WARNING e9ca75bbf1f1e59954a82ad0649362fb: non-static multi-annotation box without attribute
WARNING 08b66919e13fe032a7fbe1efbd09bf31: non-static multi-annotation box without attribute
WARNING 3fb27474b1657d07727905f5dfb02080: non-static multi-annotation box without attribute
WARNING ef02517f71b214eeb96bc1e0c02f4e0b: non-static multi-annotation box without attribute
WARNING 35a32e01daf66325813a532605d7c90d: non-static multi-annotation box without attribute
WARNING 7a239c9edab51fb4aa96f19707927dcf: non-static multi-annotation box without attribute
WARNING 1469e8f7d528521314b47a9d5c77e551: non-static multi-annotation box without attribute
WARNING 84f030f2ef98469987a9f9f476a4ff57: non-static multi-annotation box without attribute
WARNING 6fde326c40de059f69b72e572b2fd1bf: non-static multi-annotation box without attribute
WARNING 908fbde305662e7852c1496694c22f7c: non-static multi-annotation box without attribute
WARNING 19f249f53138907ddc297998b57a7613: non-static multi-annotation box without attribute
WARNING 45fe775bb280c335c3188b586edc619d: non-static multi-annotation box without attribute
WARNING 55fcb8d8113de28dfd4a2e055d4c9089: non-static multi-annotation box without attribute
WARNING 26c79f8080710f7238b6cef212fcfc72: non-static multi-annotation box without attribute
WARNING d5a33113498e0173da4846960515eef1: non-static multi-annotation box without attribute
WARNING 479d5414c15ab12950e1d7ba4addc47d: non-static multi-annotation box without attribute
WARNING 226bd07c008019d3686f9074bc0d006f: non-static multi-annotation box without attribute
WARNING abac119ac000a84f6fcde4d6946ba3d9: non-static multi-annotation box without attribute
WARNING a9a91692412ef8ddc9a43b0c2b2c9a1c: non-static multi-annotation box without attribute
WARNING a8a746f8a8508f05e584021d5e117d00: non-static multi-annotation box without attribute
WARNING 986dc05812ffea8962880dd52a322f28: non-static multi-annotation box without attribute
WARNING fc99679cf368dfa90077e9fbbb385fab: non-static multi-annotation box without attribute
WARNING 512e8ed404a0f29f5ea9b5832d3b61a7: non-static multi-annotation box without attribute
WARNING 6564afdca1f88eadb8eebb72d1d22986: non-static multi-annotation box without attribute
WARNING 5bbb17a9eecac3d35ff08045cf7d48db: non-static multi-annotation box without attribute
WARNING 6034f016bec0e19b00f747461c6ee689: non-static multi-annotation box without attribute
WARNING b327fd80e8cb9658eee31d74fbd2823f: non-static multi-annotation box without attribute
WARNING 9c46c28e8713d6b6b624233e9f5c61d1: non-static multi-annotation box without attribute
WARNING f6ece24f7f330c740a7bc9f5f0b4739e: non-static multi-annotation box without attribute
WARNING 3e2debfedbc3ebf6e720cb912e7613ed: non-static multi-annotation box without attribute
WARNING f2dd58ebe670aa5f2132104faa2771d5: non-static multi-annotation box without attribute
WARNING e0954e4c0605c9a2151f5436b59243f4: non-static multi-annotation box without attribute
WARNING e2e8531394f8c93f76260849ff951576: non-static multi-annotation box without attribute
WARNING 7a665d8512029f4feacd0b546bbc312a: non-static multi-annotation box without attribute
WARNING cbd025557554f5201c7a8e4d84f11f1f: non-static multi-annotation box without attribute
WARNING fa3e5ecaae907ec8ed8ab407a0dcdcfe: non-static multi-annotation box without attribute
WARNING 6823b6b419cf529e9dba9be7e67ce912: non-static multi-annotation box without attribute
WARNING 642eae8072997b89f6feb7ef5e46c949: non-static multi-annotation box without attribute
WARNING class animal: zero instances
WARNING class barrier: zero instances
WARNING class battery_rickshaw: zero instances
WARNING class construction_element: zero instances
WARNING class covered_van: zero instances
WARNING class human_hauler: zero instances
WARNING class microbus: zero instances
WARNING class pushcart: zero instances
WARNING class tempo: zero instances
WARNING class traffic_cone: zero instances
check_release: 0 error(s), 732 warning(s); {'n_annotations': 16141, 'n_instances': 5485, 'n_interpolated': 746, 'present': {'cycle_rickshaw': 369, 'pedestrian': 2730, 'car': 725, 'truck': 202, 'bicycle': 316, 'motorcycle': 563, 'bus': 317, 'cng_autorickshaw': 263}}
```

</details>

---

## 4. Before vs after stitching

Two views. The **all-tier** view is the one `stitch.per_scene` reports (it runs
before the tier filter, on all 26 791 records). The **shipped** view is what a
benchmark consumer actually loads out of `sample_annotation.json`. Both matter;
the shipped one is the one the benchmark's tracking metrics will see.

### 4.1 All tiers (the stitcher's own view)

| | BEFORE (Stage 7 fragments) | AFTER (stitched chains) |
|---|---|---|
| groups | **11 882** | **8 463** (−28.8 %) |
| rows | 26 791 | 29 321 |
| **median length** | **1.0** | **1.0** |
| mean length | 2.25 | 3.46 (+53.7 %) |
| singleton groups | 8 172 (68.8 %) | 4 335 (51.2 %) |
| groups ≥ 3 keyframes | 2 235 (18.8 %) | 3 045 (36.0 %) |
| **rows on chains ≥ 3** | **15 669 (58.5 %)** | **22 820 (77.8 %)** |
| rows on chains ≥ 5 | 11 935 (44.5 %) | 18 259 (62.3 %) |
| rows on chains ≥ 10 | 7 269 (27.1 %) | 11 032 (37.6 %) |
| rows on chains ≥ 20 | 3 350 (12.5 %) | 5 112 (17.4 %) |

### 4.2 Shipped `sample_annotation` (auto_accept only) — the number that matters

Grouping the 15 395 shipped real rows by their `dhakascenes_track_id_pre_stitch`
gives the identity the release *would* have had with `--no-stitch`; grouping the
16 141 shipped rows by `instance_token` gives what it has.

| | BEFORE (Stage 7 `track_id`) | AFTER (stitched instances) |
|---|---|---|
| instances | **7 158** | **5 485** (−23.4 %) |
| rows | 15 395 | 16 141 (+746 interpolated) |
| **median instance length** | **1.0** | **1.0** |
| mean instance length | 2.15 | 2.94 (+36.8 %) |
| singleton instances | 4 929 (68.9 %) | 2 897 (52.8 %) |
| instances ≥ 3 annotations | 1 259 (17.6 %) | 1 682 (30.7 %) |
| **rows on instances ≥ 3** | **8 526 (55.4 %)** | **11 432 (70.8 %)** |
| rows on instances ≥ 5 | 6 349 (41.2 %) | 8 573 (53.1 %) |
| rows on instances ≥ 10 | 3 701 (24.0 %) | 4 810 (29.8 %) |
| rows on instances ≥ 20 | 1 764 (11.5 %) | 2 256 (14.0 %) |

**Boxes now usable for tracking metrics.** Taking "≥ 3 keyframes" as the minimum
for an identity that a tracking metric can score (AMOTA needs a track to survive
long enough to be switched *away from*), the shipped release goes from
**8 526 → 11 432 boxes on scoreable tracks (+2 906, +34 %)**, on
**1 682 scoreable identities instead of 1 259 (+34 %)**. At the ≥ 5 threshold the
gain is **6 349 → 8 573 (+35 %)**. The number of dead-end one-frame identities —
the fragmentation this plan exists to repair — drops by **2 032 (−41 %)**.

**Read the median honestly.** It stays at 1.0 in both views, and the delivery
note prints exactly that (`median chain length before -> after, per scene:
chunk_0000 1.0 -> 1.0`). The median is dominated by the long tail of genuine
one-frame detections at range, which no gap-3 stitcher can rescue because there
is nothing to join them to. Every row-weighted statistic moves substantially.

### 4.3 What each gap tier buys

Re-cutting the shipped+excluded tables at each `max_gap_keyframes` setting:

| `max_gap_keyframes` | chains | rows | mean | rows on chains ≥ 3 |
|---|---|---|---|---|
| 0 (no stitch) | 11 882 | 26 791 | 2.25 | 15 669 / 26 791 = 58.5 % |
| 1 | 10 192 | 26 791 | 2.63 | 17 503 / 26 791 = 65.3 % |
| 2 | 9 264 | 27 719 | 2.99 | 20 009 / 27 719 = 72.2 % |
| **3 (shipped)** | **8 463** | **29 321** | **3.46** | **22 820 / 29 321 = 77.8 %** |

Every tier still pays: gap-3 joins alone are worth 5.6 points of row coverage and
801 of the 3 419 joins. There is no knee suggesting the third tier is scraping
the barrel.

---

## 5. Sanity-check of stitched chains by hand

This is the load-bearing judgement in this task: a stitch that welds two
different objects into one identity is worse than no stitch at all. I looked at
the longest chain (as the brief asks), at the *most heavily stitched* chain
(which is the harder test — the longest chain turns out to be barely stitched at
all), and at the join geometry of all 3 419 joins against a control.

#### 5.1 The longest chain in `stitch_map.json` — chain `9228` (168 real rows)

chain_id=9228  class=car  instance_token=03efe0b9222e90dffc36848e880036a7  rows=170 (real 168, interpolated 2)

```
  kf    t_s   ego_x   ego_y   rng  global_x  global_y step_m interp        tier pre_track    pts shipped
 511   0.00   -4.66    1.74  4.98    412.98     48.81   0.00      . auto_accept      9228     82     yes
 512   0.40   -4.68    1.96  5.08    413.10     49.05   0.27      . auto_accept      9228    181     yes
 513   0.80   -6.06    1.82  6.32    412.02     48.50   1.21      . auto_accept      9228   1693     yes
 514   1.20   -4.62    2.11  5.08    413.38     49.30   1.57      . auto_accept      9228   1626     yes
 515   1.60   -4.41    1.39  4.62    413.88     48.71   0.77      . auto_accept      9228   1983     yes
 516   2.00   -4.37    1.36  4.57    413.96     48.70   0.09      . auto_accept      9228   2660     yes
 517   2.40   -4.88    1.44  5.09    413.47     48.60   0.50      . auto_accept      9228   2930     yes
 518   2.80   -4.87    1.38  5.06    413.49     48.55   0.05      . auto_accept      9228   2883     yes
 519   3.20   -5.43    1.63  5.67    412.88     48.61   0.61      . auto_accept      9228   2836     yes
 520   3.60   -4.96    1.49  5.17    413.37     48.63   0.49      . auto_accept      9228   2963     yes
 521   4.00   -5.31    1.66  5.56    413.01     48.65   0.36      . auto_accept      9228   2932     yes
 522   4.40   -4.82    1.37  5.01    413.62     48.56   0.62      . auto_accept      9228   2952     yes
 523   4.80   -5.46    1.55  5.67    413.06     48.54   0.56      . auto_accept      9228   2926     yes
 524   5.20   -5.52    1.56  5.73    413.15     48.55   0.09      . auto_accept      9228   2913     yes
 525   5.60   -5.24    1.48  5.44    413.56     48.59   0.41      . auto_accept      9228   2804     yes
 526   6.00   -5.11    1.53  5.33    413.75     48.69   0.22      . auto_accept      9228   2778     yes
 527   6.40   -4.45    1.34  4.65    414.44     48.77   0.69      . auto_accept      9228   2789     yes
 528   6.80   -4.77    1.54  5.01    414.10     48.84   0.34      . auto_accept      9228   2794     yes
 529   7.20   -4.28    1.24  4.46    414.69     48.72   0.60      . auto_accept      9228   2785     yes
 530   7.60   -4.59    1.45  4.81    414.31     48.82   0.39      . auto_accept      9228   2808     yes
 531   8.00   -5.63    1.87  5.93    413.26     48.85   1.05      . auto_accept      9228   2818     yes
 532   8.40   -4.28    1.24  4.45    414.82     48.75   1.56      . auto_accept      9228   2825     yes
 533   8.80   -5.52    1.50  5.72    413.71     48.55   1.13      . auto_accept      9228   2869     yes
 534   9.20   -4.39    1.31  4.58    414.98     48.81   1.29      . auto_accept      9228   2845     yes
 535   9.60   -5.78    1.80  6.06    413.74     48.72   1.24      . auto_accept      9228   2785     yes
 536  10.00   -4.50    1.40  4.72    415.30     48.91   1.56      . auto_accept      9228   2775     yes
 537  10.40   -4.55    1.44  4.77    415.47     48.96   0.18      . auto_accept      9228   2744     yes
 538  10.80   -4.53    1.44  4.75    415.76     49.00   0.29      . auto_accept      9228   2616     yes
 539  11.20   -4.69    1.55  4.94    415.91     49.06   0.15      . auto_accept      9228   2567     yes
 540  11.60   -5.93    1.96  6.24    415.01     48.82   0.93      . auto_accept      9228   2512     yes
 541  12.00   -4.87    1.81  5.19    416.28     49.27   1.35      . auto_accept      9228   2495     yes
 542  12.40   -3.97    1.23  4.16    417.58     49.34   1.30      . auto_accept      9228   2257     yes
 543  12.80   -4.75    1.74  5.06    416.84     49.35   0.73      . auto_accept      9228   2425     yes
 544  13.20   -4.56    1.64  4.85    417.22     49.41   0.39      . auto_accept      9228   2591     yes
 545  13.60   -4.80    1.77  5.11    417.16     49.41   0.07      . auto_accept      9228   2402     yes
 546  14.00   -4.56    1.67  4.85    417.67     49.55   0.53      . auto_accept      9228   2378     yes
 547  14.40   -4.66    1.75  4.98    417.82     49.61   0.16      . auto_accept      9228   2327     yes
 548  14.80   -4.52    1.69  4.83    418.22     49.73   0.42      . auto_accept      9228   2263     yes
 549  15.20   -3.69    1.19  3.88    419.40     50.06   1.23      . auto_accept      9228   1633     yes
 550  15.60   -5.82    2.13  6.20    417.57     49.25   2.00      . auto_accept      9228   2033     yes
 551  16.00   -5.84    2.10  6.21    417.82     49.26   0.25      . auto_accept      9228   1897     yes
 552  16.40   -5.79    2.24  6.21    418.00     49.45   0.26      . auto_accept      9228   1302     yes
 553  16.80   -4.01    1.49  4.27    419.83     50.47   2.10      . auto_accept      9228   1661     yes
 554  17.20   -5.11    1.98  5.48    419.03     49.97   0.94      . auto_accept      9228   1763     yes
 555  17.60   -5.10    1.99  5.48    419.21     50.07   0.20      . auto_accept      9228   1524     yes
 556  18.00   -5.21    2.08  5.61    419.27     50.12   0.08      . auto_accept      9228   1604     yes
 557  18.40   -5.63    2.17  6.04    419.20     49.91   0.22      . auto_accept      9228   1384     yes
 558  18.80   -3.92    1.50  4.19    420.80     51.19   2.04      . auto_accept      9228   1350     yes
 559  19.20   -5.37    2.26  5.83    419.73     50.45   1.30      . auto_accept      9228   1383     yes
 560  19.60   -5.51    2.18  5.93    419.98     50.43   0.25      . auto_accept      9228   1211     yes
 561  20.00   -5.35    2.09  5.74    420.43     50.71   0.53      . auto_accept      9228   1171     yes
 562  20.40   -5.64    2.35  6.11    420.31     50.78   0.14      . auto_accept      9228   1183     yes
 563  20.80   -4.51    1.76  4.85    421.39     51.82   1.51      . auto_accept      9228   1076     yes
 564  21.20   -4.91    2.00  5.30    421.22     51.72   0.20      . auto_accept      9228   1028     yes
 565  21.60   -5.56    2.46  6.08    420.75     51.42   0.56      . auto_accept      9228   1036     yes
 566  22.00   -5.19    2.22  5.64    421.29     51.88   0.71      . auto_accept      9228    962     yes
 567  22.40   -4.54    2.01  4.97    421.81     52.64   0.93      . auto_accept      9228    858     yes
 568  22.80   -4.09    1.81  4.47    422.22     53.23   0.71      . auto_accept      9228    725     yes
 569  23.20   -5.48    2.61  6.07    421.39     52.27   1.27      .     flagged      9228    565      no
 570  23.60   -4.44    2.08  4.90    422.22     53.43   1.42      . auto_accept      9228    509     yes
 571  24.00   -5.26    2.49  5.82    421.90     52.99   0.54      . auto_accept      9228    581     yes
 572  24.40   -4.26    1.91  4.67    422.73     54.22   1.48      . auto_accept      9228    490     yes
 573  24.80   -5.69    2.76  6.32    421.91     53.15   1.34      . auto_accept      9228    562     yes
 574  25.20   -5.02    2.31  5.53    422.45     54.10   1.09      . auto_accept      9228    708     yes
 575  25.60   -4.13    1.82  4.51    423.03     55.32   1.35      . auto_accept      9228    897     yes
 576  26.00   -4.11    1.74  4.46    423.18     55.75   0.45      . auto_accept      9228   1060     yes
 577  26.40   -4.12    1.67  4.44    423.30     56.19   0.46      . auto_accept      9228   1157     yes
 578  26.80   -5.63    2.24  6.06    422.77     55.21   1.12      . auto_accept      9228   1500     yes
 579  27.20   -4.24    1.55  4.52    423.46     57.16   2.07      . auto_accept      9228   1757     yes
 580  27.60   -5.38    1.83  5.68    423.23     56.66   0.55      . auto_accept      9228   2202     yes
 581  28.00   -4.29    1.43  4.52    423.60     58.38   1.76      . auto_accept      9228   2296     yes
 582  28.40   -5.23    1.52  5.44    423.52     58.04   0.35      . auto_accept      9228   2764     yes
 583  28.80   -5.44    1.53  5.65    423.50     58.43   0.39      . auto_accept      9228   2887     yes
 584  29.20   -4.10    1.66  4.43    423.32     60.31   1.89      . auto_accept      9228   2531     yes
 585  29.60   -4.81    1.26  4.97    423.73     60.14   0.44      . auto_accept      9228   2491     yes
 586  30.00   -6.58    1.85  6.83    423.14     58.90   1.37      . auto_accept      9228   2262     yes
 587  30.40   -6.45    1.78  6.69    423.20     59.54   0.65      . auto_accept      9228   1480     yes
 588  30.80   -4.75    1.31  4.93    423.68     61.71   2.23      .     flagged      9228    470      no
 590  31.60   -5.82    0.82  5.88    424.23     61.58   0.57      . auto_accept      9228    370     yes
 591  32.00   -5.82    0.71  5.86    424.34     62.01   0.45      . auto_accept      9228   1364     yes
 592  32.40   -4.69    1.34  4.88    423.70     63.49   1.61      . auto_accept      9228   1220     yes
 593  32.80   -4.92    0.63  4.96    424.43     63.59   0.74      . auto_accept      9228   1388     yes
 594  33.20   -4.69    1.28  4.86    423.80     64.10   0.81      . auto_accept      9228   1552     yes
 595  33.60   -6.31    1.06  6.40    424.05     62.78   1.35      . auto_accept      9228   2139     yes
 596  34.00   -4.70    1.34  4.88    423.75     64.62   1.87      . auto_accept      9228   2063     yes
 597  34.40   -4.70    1.42  4.91    423.68     64.80   0.20      . auto_accept      9228   2500     yes
 598  34.80   -6.24    1.61  6.44    423.51     63.43   1.39      . auto_accept      9228   2724     yes
 599  35.20   -4.68    1.52  4.92    423.58     65.12   1.69      . auto_accept      9228   2836     yes
 600  35.60   -5.02    1.62  5.27    423.49     64.89   0.25      . auto_accept      9228   3137     yes
 601  36.00   -4.94    1.60  5.19    423.50     65.08   0.20      . auto_accept      9228   3323     yes
 602  36.40   -6.20    1.65  6.41    423.49     63.93   1.15      . auto_accept      9228   3447     yes
 603  36.80   -4.61    1.57  4.87    423.52     65.58   1.66      . auto_accept      9228   3496     yes
 604  37.20   -4.97    0.77  5.03    424.33     65.38   0.84      . auto_accept      9228   2478     yes
 605  37.60   -5.19    0.71  5.24    424.35     65.28   0.10      . auto_accept      9228   2440     yes
 606  38.00   -5.04    0.82  5.10    424.27     65.64   0.37      . auto_accept      9228   3355     yes
 607  38.40   -5.33    1.83  5.64    423.27     65.60   1.00      . auto_accept      9228   3349     yes
 608  38.80   -4.69    1.07  4.81    424.01     66.46   1.14      . auto_accept      9228   3292     yes
 609  39.20   -5.43    1.59  5.66    423.51     65.98   0.69      . auto_accept      9228   3367     yes
 610  39.60   -5.07    1.20  5.21    423.88     66.63   0.75      . auto_accept      9228   3448     yes
 611  40.00   -5.01    1.59  5.25    423.52     67.05   0.56      . auto_accept      9228   3462     yes
 612  40.40   -5.18    1.69  5.45    423.44     67.27   0.23      . auto_accept      9228   3555     yes
 613  40.80   -5.61    1.82  5.90    423.32     67.26   0.12      . auto_accept      9228   3664     yes
 614  41.20   -5.33    1.65  5.57    423.51     67.99   0.75      . auto_accept      9228   3663     yes
 615  41.60   -4.93    1.55  5.17    423.61     68.86   0.87      . auto_accept      9228   3751     yes
 616  42.00   -5.05    1.60  5.30    423.56     69.25   0.39      . auto_accept      9228   3777     yes
 617  42.40   -5.10    1.53  5.33    423.64     69.73   0.49      . auto_accept      9228   3791     yes
 618  42.80   -5.16    1.72  5.44    423.45     70.24   0.54      . auto_accept      9228   3723     yes
 619  43.20   -4.82    1.28  4.99    423.88     71.17   1.03      . auto_accept      9228   3582     yes
 620  43.60   -5.78    1.56  5.98    423.64     70.80   0.44      . auto_accept      9228   3454     yes
 621  44.00   -6.34    1.40  6.49    423.84     70.85   0.20      . auto_accept      9228   3315     yes
 622  44.40   -5.42    1.25  5.56    423.93     72.32   1.47      . auto_accept      9228   3189     yes
 623  44.80   -6.08    1.12  6.18    424.10     72.19   0.21      . auto_accept      9228   3030     yes
 624  45.20   -5.81    1.05  5.91    424.14     73.02   0.82      . auto_accept      9228   2919     yes
 625  45.60   -5.49    1.06  5.59    424.10     73.90   0.88      . auto_accept      9228   2890     yes
 626  46.00   -6.62    2.11  6.95    423.13     73.29   1.15      . auto_accept      9228   2701     yes
 627  46.40   -4.68    1.33  4.87    423.73     75.88   2.66      . auto_accept      9228   1856     yes
 628  46.80   -4.69    1.23  4.85    423.81     76.51   0.64      . auto_accept      9228   1662     yes
 629  47.20   -4.69    1.20  4.84    423.79     77.16   0.65      . auto_accept      9228   1551     yes
 630  47.60   -8.00    1.52  8.14    423.68     74.47   2.70      .     flagged      9228     25      no
 632  48.40   -6.63    1.53  6.81    423.53     77.11   2.64      . auto_accept      9228   2152     yes
 633  48.80   -4.85    0.27  4.86    424.56     79.56   2.66      . auto_accept      9228   1610     yes
 634  49.20   -4.70    0.68  4.75    424.10     80.27   0.85      . auto_accept      9228   1541     yes
 635  49.60   -4.77    0.29  4.78    424.45     80.80   0.64      . auto_accept      9228   1386     yes
 636  50.00   -5.62    0.30  5.63    424.47     80.53   0.27      . auto_accept      9228   1705     yes
 637  50.40   -4.87    0.06  4.87    424.60     81.87   1.34      . auto_accept      9228   1140     yes
 638  50.80   -6.15    0.27  6.15    424.47     81.22   0.66      . auto_accept      9228   1491     yes
 639  51.20   -6.17    0.27  6.18    424.42     81.87   0.65      . auto_accept      9228   1325     yes
 640  51.60   -6.07    0.15  6.07    424.47     82.65   0.78      . auto_accept      9228   1240     yes
 641  52.00   -4.67    0.44  4.69    424.01     84.66   2.07      . auto_accept      9228    533     yes
 642  52.40   -6.58    0.11  6.58    424.46     83.45   1.30      . auto_accept      9228   1047     yes
 643  52.80   -4.64    0.37  4.66    423.98     85.92   2.52      . auto_accept      9228    287     yes
 644  53.20   -6.37   -0.07  6.37    424.50     84.76   1.28      . auto_accept      9228   1005     yes
 645  53.60   -6.20   -0.11  6.20    424.50     85.39   0.63      . auto_accept      9228   1009     yes
 646  54.00   -4.68    0.11  4.68    424.14     87.34   1.98      . auto_accept      9228    560     yes
 647  54.40   -6.48   -0.08  6.48    424.42     86.01   1.36      . auto_accept      9228   1087     yes
 648  54.80   -6.44   -0.09  6.44    424.39     86.47   0.46      . auto_accept      9228   1126     yes
 649  55.20   -6.26   -0.14  6.26    424.39     87.08   0.61      . auto_accept      9228   1084     yes
 650  55.60   -6.13   -0.22  6.13    424.42     87.67   0.59      . auto_accept      9228   1082     yes
 651  56.00   -6.70   -0.20  6.71    424.38     87.58   0.10      . auto_accept      9228   1079     yes
 652  56.40   -4.67    0.09  4.67    423.93     90.11   2.57      . auto_accept      9228    477     yes
 653  56.80   -6.41   -0.32  6.41    424.34     88.93   1.24      . auto_accept      9228   1029     yes
 654  57.20   -6.71   -0.40  6.72    424.34     89.15   0.22      . auto_accept      9228    982     yes
 655  57.60   -4.56   -0.29  4.57    424.14     91.79   2.64      . auto_accept      9228     52     yes
 656  58.00   -4.52   -0.24  4.53    424.07     92.37   0.58      . auto_accept      9228     47     yes
 657  58.40   -4.67   -0.15  4.67    424.00     92.79   0.42      . auto_accept      9228     40     yes
 658  58.80   -4.43   -0.05  4.44    423.91     93.58   0.80      . auto_accept      9228     41     yes
 659  59.20   -6.56   -0.33  6.57    424.28     92.06   1.57      . auto_accept      9228    992     yes
 660  59.60   -4.63   -0.21  4.64    424.13     94.50   2.45      . auto_accept      9228     56     yes
 661  60.00   -4.67   -0.05  4.67    423.98     95.00   0.52      . auto_accept      9228    298     yes
 662  60.40   -4.64    0.12  4.64    423.84     95.57   0.58      . auto_accept      9228    413     yes
 663  60.80   -4.67    0.15  4.67    423.82     96.09   0.52      . auto_accept      9228    514     yes
 664  61.20   -6.49   -0.08  6.49    424.16     94.89   1.24      . auto_accept      9228   1012     yes
 665  61.60   -4.29   -2.08  4.77    426.04     97.76   3.43      . auto_accept      9228    974     yes
 666  62.00   -4.37   -1.99  4.80    425.94     98.33   0.58      . auto_accept      9228    961     yes
 667  62.40   -4.53   -1.94  4.92    425.88     98.79   0.47      . auto_accept      9228   1228     yes
 668  62.80   -4.79   -1.81  5.12    425.75     99.14   0.36      . auto_accept      9228    466     yes
 669  63.20   -4.79   -1.87  5.15    425.79     99.72   0.59      . auto_accept      9228    812     yes
 670  63.60   -4.54   -2.51  5.18    426.38    100.60   1.06      . auto_accept      9228    147     yes
 671  64.00   -4.56   -2.10  5.02    425.94    101.14   0.70      . auto_accept      9228     50     yes
 672  64.40   -4.26   -1.95  4.69    425.75    101.97   0.85      . auto_accept      9228     32     yes
 673  64.80   -4.45   -1.98  4.87    425.77    102.27   0.31      . auto_accept      9228     38     yes
 674  65.20   -6.74   -2.51  7.19    426.40    100.50   1.88      . auto_accept      9228     45     yes
 675  65.60   -6.54   -1.84  6.79    425.74    101.17   0.95      . auto_accept      9228     51     yes
 676  66.00   -7.51   -2.64  7.96    426.60    100.79   0.94      . auto_accept      9228     58     yes
 677  66.40   -6.70   -1.98  6.99    425.89    102.13   1.52      . auto_accept      9228     54     yes
 678  66.80   -6.54   -1.95  6.82    425.84    102.79   0.66      . auto_accept      9228     60     yes
 679  67.20   -6.46   -1.74  6.69    425.61    103.31   0.58      . auto_accept      9228     48     yes
 680  67.60   -7.18   -1.21  7.28    425.12    102.97   0.60    YES     flagged      None   6346      no
 681  68.00   -7.86   -0.70  7.89    424.64    102.62   0.60    YES     flagged      None    454      no
 682  68.40   -8.52   -0.18  8.52    424.15    102.27   0.60      .     flagged     11872    211      no
```


#### 5.2 Reading chain `9228` (the longest chain)

`stitch_map.json` holds 26 791 record tokens over 8 463 chains; the largest is
chain `9228` with 168 real rows. Ten largest: 168, 111, 104, 93, 86, 81, 80, 75,
74, 72.

| | |
|---|---|
| class | car |
| span | keyframes 511 → 682, 68.4 s |
| rows | 170 (168 real + 2 interpolated) |
| Stage-7 fragments merged | **2** (`9228`, `11872`) — one join, at gap 3 |
| global path | x 412.0 → 426.6, y 48.5 → 103.3; 153.6 m of path for 54.6 m of net displacement |
| step between consecutive rows | p50 0.69 m, p90 1.88 m, max 3.43 m |
| median points per box | 1 630 |
| tiers | 164 auto_accept, 6 flagged |

**This is a single vehicle and the positions are smooth.** It sits behind the ego
for the whole 68 s at `ego_x` −4 to −8 m and `ego_y` +2.5 to −2.6 m — a car
tailing the ego through stop-and-go traffic. Global `y` rises monotonically from
48.5 to 103.3 with no reversal larger than the box's own re-fit jitter, and
global `x` stays inside a 14 m band. The 153.6 m path against 54.6 m displacement
is the box centre wobbling, not the object moving: the per-step p50 of 0.69 m at
2.5 Hz is 1.7 m/s, consistent with congested traffic, and the largest step
(3.43 m at keyframe 665) is a 2 m lateral hop of the box centre while the object
stayed put. The yaw column wanders badly in places (92° → 67° → 92°, and sign
flips near ±90°) — that is the **pre-fix box-fitting noise the brief warned
about**, visible in a single unstitched Stage-7 fragment, and it is not something
the stitcher caused or could cause.

**The one join is sound.** Fragment `9228` ends at keyframe 679, global
(425.61, 103.31); fragment `11872` starts at keyframe 682, global (424.15,
102.27). That is 1.79 m over 1.2 s, against a gap-3 gate of
`2.0 + 1.0 × 2 = 4.0 m` — comfortably inside, and well inside this same object's
own frame-to-frame wobble. The two interpolated rows at keyframes 680 and 681
step 0.60 m each along a straight line between the endpoints. The joined tail is
one real keyframe; the join adds 3 rows to a 167-row chain, so even if it were
wrong the damage would be negligible. It is not wrong.

**Caveat visible in this table:** two internal holes (keyframes 589 and 631) are
*not* interpolated. They sit inside Stage-7 fragment `9228` itself, and the
interpolator only fills gaps *between* joined fragments. See §7.3.

#### 5.3 The most-stitched chain — chain `6226` (14 Stage-7 fragments merged)

chain_id=6226  class=pedestrian  instance_token=04f7a86ba57b103b77337b475e4192d1  rows=48 (real 46, interpolated 2)

```
  kf    t_s   ego_x   ego_y   rng  global_x  global_y step_m interp        tier pre_track    pts shipped
 335   0.00  -20.48   -2.15 20.59    296.32     39.35   0.00      . auto_accept      6226     13     yes
 336   0.40  -20.50   -2.29 20.63    296.33     39.23   0.12      .     flagged      6239      6      no
 337   0.80  -20.06   -2.53 20.21    296.77     39.00   0.50      . auto_accept      6239     12     yes
 338   1.20  -19.27   -2.37 19.42    297.53     39.17   0.78      .     flagged      6239     11      no
 339   1.60  -19.29   -2.44 19.44    297.51     39.11   0.06    YES     flagged      None     57      no
 340   2.00  -19.32   -2.48 19.48    297.48     39.06   0.06      . auto_accept      6292      7     yes
 341   2.40  -19.25   -2.45 19.40    297.56     39.11   0.09      . auto_accept      6292     13     yes
 342   2.80  -19.11   -2.34 19.25    297.71     39.22   0.19      .     flagged      6292     11      no
 343   3.20  -19.33   -2.35 19.48    297.49     39.21   0.22      . auto_accept      6343      9     yes
 344   3.60  -19.33   -2.41 19.47    297.50     39.13   0.09      . auto_accept      6343      9     yes
 345   4.00  -19.25   -2.38 19.40    297.55     39.16   0.06      .     flagged      6343     14      no
 346   4.40  -19.04   -2.46 19.20    297.76     39.10   0.22      . auto_accept      6343     11     yes
 347   4.80  -19.46   -2.36 19.60    297.36     39.18   0.42      . auto_accept      6389     10     yes
 348   5.20  -19.26   -2.36 19.40    297.56     39.18   0.20      .     flagged      6389      9      no
 349   5.60  -19.56   -2.40 19.71    297.26     39.14   0.30      .     flagged      6416      6      no
 350   6.00  -19.31   -2.38 19.45    297.51     39.18   0.25      .     flagged      6432      7      no
 352   6.80  -19.21   -2.45 19.37    297.59     39.11   0.11      . auto_accept      6432      8     yes
 353   7.20  -19.31   -2.43 19.46    297.49     39.12   0.10      . auto_accept      6432      9     yes
 354   7.60  -19.16   -2.30 19.29    297.65     39.26   0.21      . auto_accept      6432     10     yes
 355   8.00  -19.08   -2.39 19.23    297.74     39.17   0.12      . auto_accept      6432     11     yes
 357   8.80  -19.27   -2.32 19.41    297.54     39.24   0.21      . auto_accept      6432      8     yes
 358   9.20  -19.48   -2.40 19.63    297.33     39.14   0.24      . auto_accept      6432      9     yes
 359   9.60  -19.44   -2.48 19.60    297.37     39.07   0.08      . auto_accept      6432     10     yes
 360  10.00  -19.19   -2.48 19.35    297.63     39.06   0.26      . auto_accept      6607      9     yes
 361  10.40  -19.18   -2.51 19.35    297.63     39.04   0.02      . auto_accept      6607      7     yes
 362  10.80  -18.89   -2.44 19.05    297.92     39.12   0.31      .     flagged      6645      6      no
 363  11.20  -18.74   -2.87 18.96    298.10     38.70   0.46      . auto_accept      6654     10     yes
 365  12.00  -19.37   -2.96 19.60    297.45     38.58   0.66      . auto_accept      6654      9     yes
 368  13.20  -19.93   -3.04 20.16    296.90     38.49   0.55      .    rejected      6654      4      no
 369  13.60  -19.42   -2.85 19.63    297.42     38.69   0.55      . auto_accept      6739      8     yes
 371  14.40  -19.69   -2.88 19.90    297.15     38.65   0.27      . auto_accept      6739      7     yes
 372  14.80  -19.74   -3.07 19.98    297.09     38.46   0.19      . auto_accept      6739      8     yes
 373  15.20  -19.74   -3.08 19.98    297.10     38.45   0.02      . auto_accept      6739     10     yes
 374  15.60  -20.12   -3.08 20.36    296.71     38.45   0.39      .    rejected      6739     11      no
 375  16.00  -19.54   -3.05 19.77    297.29     38.49   0.58    YES    rejected      None      7      no
 376  16.40  -18.97   -3.03 19.21    297.86     38.52   0.58      . auto_accept      6849      9     yes
 377  16.80  -18.47   -2.96 18.70    298.36     38.62   0.51      . auto_accept      6849      8     yes
 378  17.20  -19.91   -2.95 20.13    296.93     38.59   1.44      . auto_accept      6880      9     yes
 381  18.40  -19.85   -3.20 20.10    297.00     38.34   0.26      . auto_accept      6880      8     yes
 382  18.80  -19.89   -3.20 20.15    296.95     38.32   0.05      . auto_accept      6880      8     yes
 383  19.20  -19.93   -3.03 20.16    296.90     38.49   0.18      . auto_accept      6880     12     yes
 386  20.40  -20.56   -3.06 20.79    296.39     38.44   0.52      . auto_accept      6880     11     yes
 387  20.80  -20.12   -2.95 20.34    296.94     38.56   0.57      . auto_accept      6880     11     yes
 388  21.20  -20.22   -2.87 20.43    296.98     38.64   0.09      . auto_accept      6880      7     yes
 389  21.60  -20.02   -2.69 20.20    297.31     38.86   0.40      . auto_accept      6880      8     yes
 391  22.40  -20.85   -2.93 21.05    296.73     38.59   0.64      .    rejected      6880      8      no
 392  22.80  -20.89   -2.97 21.10    296.90     38.58   0.17      . auto_accept      6880     11     yes
 393  23.20  -20.70   -3.08 20.93    297.44     38.65   0.55      . auto_accept      7115     11     yes
```

#### 5.4 Reading chain `6226` (the hardest case — 14 fragments welded together)

The longest chain barely exercises the stitcher, so I also took the chain that
merged the most fragments.

| | |
|---|---|
| class | pedestrian |
| span | keyframes 335 → 393, 23.2 s, at 19–21 m range |
| rows | 48 (46 real + 2 interpolated) |
| Stage-7 fragments merged | **14** (`6226, 6239, 6292, 6343, 6389, 6416, 6432, 6607, 6645, 6654, 6739, 6849, 6880, 7115`) |
| global position | x 296.32 → 298.36, y 38.32 → 39.35 — a **2.0 × 1.0 m box for 23 s** |
| net displacement | 1.3 m over 14.8 m of accumulated jitter |
| step between consecutive rows | p50 0.24 m, p90 0.58 m, max 1.44 m |
| median points per box | 9 |
| tiers | 35 auto_accept, 9 flagged, 4 rejected |

**This is one person standing still, and Stage 7 gave them 14 identities.** The
centre never leaves a 2 m × 1 m footprint for 23 seconds; consecutive steps are
0.24 m at the median. There is no plausible reading in which these 14 fragments
are different pedestrians — they would have to be standing on top of each other.
At 19–21 m range with 6–14 LiDAR points per box, the detector's confidence
oscillates (the tier column flickers auto_accept / flagged / rejected frame to
frame), which is exactly what breaks a per-frame-gated tracker. This is the
failure mode the plan exists to repair, and the stitcher repairs it correctly.

#### 5.5 All 3 419 joins, against a control

One chain proves nothing about the other 3 418 joins, so I measured the whole
population. The right control is **how far the box centre moves between adjacent
keyframes *inside* a single uncontested Stage-7 fragment** — the geometry's own
noise floor. If joins are no wider than that, the stitcher is not reaching
further than the tracker already does.

Centre displacement in metres, gap-1 joins vs the within-fragment control:

```
class              WITHIN-FRAGMENT (control)          ACROSS A GAP-1 JOIN
                    n    p50   p90   p95   max        n    p50   p90   p95   max
pedestrian       5984   0.41  1.19  1.50  3.22     1212   1.17  1.87  2.04  3.56
car              1647   0.95  2.48  3.12  8.74       72   1.35  2.10  2.15  3.62
motorcycle       1126   0.81  2.35  2.73  4.76       75   1.45  2.44  2.90  4.13
bus              1024   1.19  3.98  5.12 11.36       10   1.63  2.42  2.68  2.95
cng_autorickshaw  962   1.10  2.53  2.97  6.75       17   1.49  1.98  2.16  2.46
cycle_rickshaw    749   1.00  2.19  2.62  5.14       19   1.26  2.31  2.48  2.98
bicycle           471   0.58  1.38  1.67  3.65       69   1.31  1.76  1.97  2.32
truck             463   1.03  3.16  4.68  6.64        9   1.48  2.06  2.07  2.09
```

**Every class's join distribution sits inside the noise floor its own
within-fragment distribution sets.** For every vehicle class the joins are *tighter* than ordinary
frame-to-frame motion inside a track (car: join max 3.62 m vs control max
8.74 m; bus 2.95 vs 11.36). Pedestrians are the only class where the join tail
runs slightly wider (max 3.56 m vs control 3.22 m), and even there the p95 is
2.04 m against a control p95 of 1.50 m — the same order, not a different regime.
The stitcher is strictly more conservative than the tracker it is repairing.

Whole-population join geometry:

```
join centre distance (m):        min 0.01  p50 1.54  p90 3.14  p99 5.58  max 8.93
implied speed across the join:   p50 2.54  p90 4.43  p99 6.68  max 10.36 m/s
joins by gap:   1 -> 1690,  2 -> 928,  3 -> 801        (= 3419, matches release_meta)
joins by class: pedestrian 2463, car 164, motorcycle 139, bicycle 110, cycle_rickshaw 50,
                cng_autorickshaw 23, bus 20, truck 18, (chain fully excluded by tier) 432
fragments per chain: 1 x6434, 2 x1276, 3 x430, 4 x173, 5 x76, 6 x28, 7 x24,
                     8 x9, 9 x9, 10 x2, 11 x1, 14 x1
```

The "implied speed" column looks alarming for pedestrians in isolation — 375 of
3 419 joins (11 %) imply a speed above a generous per-class ceiling, the worst
being an 8.9 m/s pedestrian. **That is an artefact of the metric, not a bad
join.** Dividing a centre displacement by dt attributes all of the box's re-fit
jitter to object motion, and the control above shows the same tracks jitter up to
3.22 m between adjacent keyframes with no stitching involved at all. Judged
against the noise floor rather than against a kinematic model, no join is an
outlier. I found **no join whose two fragments are separated by more than that
object class's own frame-to-frame noise**, i.e. no evidence of two unrelated
objects being merged.

The residual risk, stated plainly: in a dense pedestrian cluster two people
standing 1.5 m apart are inside each other's 2.0 m gate, and if fragment A ends
on frame k while fragment B (the neighbour) starts on frame k+1, a swap is
geometrically possible. Three things bound it — the Hungarian assignment solves
each gap globally rather than greedily, the size-ratio gate rejects mismatched
footprints, and class-agnostic joining is off — and a swap between two adjacent
*standing* pedestrians costs a benchmark almost nothing (both boxes exist, both
are near-stationary, the identity switch is the only error). I did not find a
concrete instance of one.

---

## 6. Interpolated point counts: `n_interpolated_raw_basis`

**Yes — `n_interpolated_raw_basis = 2530`, i.e. 100 % of the interpolated rows.**
`n_interpolated_no_cloud = 0`.

Confirmed cause: `CloudSource` only opens the CVAT ground-filtered cloud when
*both* `task.zip` **and** `frames.json` exist in the scene directory. This
chunk's export predates Task 14:

```
/home/mt/dhakascenes/work_day1/chunk_0000/cvat_export_3d/chunk_0000/
  annotations_gt.json   annotations_ours.json   task.zip     <- no frames.json
```

So `self._zip` is never opened and every interpolated row falls back to the raw
single sweep from the dataroot (`BASIS_RAW`), giving the release two point-count
bases at once:

```
num_lidar_pts_basis, all rows:     {single_sweep_ground_filtered_pre_inflation: 26791,
                                    single_sweep_raw: 2530}
num_lidar_pts_basis, shipped rows: {single_sweep_ground_filtered_pre_inflation: 15395,
                                    single_sweep_raw: 746}
```

**The rerun fixes this**, because Task 14 makes the 3D export write `frames.json`
beside `task.zip`, `CloudSource` then reads the same ground-filtered cloud the
boxes were fit to, and `n_interpolated_raw_basis` should go to 0 with a single
basis string in `release_meta.num_lidar_pts_basis` and in the delivery note.
It did *not* fall back to "unavailable" for a single row, so the sweep paths in
the dataroot all resolve — the rerun has nothing else to fix here.

What the mixed basis costs today, measured: an interpolated row's point count
against the mean of its two real neighbours has p50 0.53, p90 9.01, max 274. That
is 30.9 % of interpolated rows inflated more than 2×, 18.2 % more than 5× — a raw
cloud counts the ground plane the ground-filtered cloud removed, so one
interpolated car-sized box (1.65 x 3.67 x 2.22 m) picked up 48 151 points
against neighbours in the low thousands. **This makes the interpolated rows' `num_lidar_pts` unusable for
filtering or reporting in this dry run**, and is on its own a sufficient reason
not to ship a release built the way this one was.

---

## 7. Judgement

### 7.1 Is the stitching safe to run on all chunks? **Yes, at the current gates.**

No config value in `configs/release.yaml` needs changing before the rerun.
The evidence:

- **0 checker errors.** Every structural invariant holds on the shipped tables;
  the devkit cross-check passed on all 64 sampled boxes.
- **The joins stay inside the tracker's own noise floor** (§5.5). Per class,
  join displacement is no wider than the displacement seen between adjacent
  keyframes inside an unstitched Stage-7 fragment — for every vehicle class it
  is markedly tighter, and pedestrians exceed the control only at the extreme
  tail (join max 3.56 m vs control max 3.22 m; p95 2.04 m vs 1.50 m). There is
  no class for which the gate reaches into a regime the geometry does not
  already occupy.
- **The one hand-read merge-heavy chain is unambiguously correct** (§5.4): 14
  fragments of one standing pedestrian, whose centre never leaves a 2 m × 1 m
  footprint for 23 s.
- **The gains are real and row-weighted** (§4): +34 % boxes on scoreable
  (≥ 3 keyframe) tracks in the shipped table, −41 % dead-end one-frame identities.
- **All three gap tiers still pay** (§4.3); there is no evidence that
  `max_gap_keyframes: 3` is scraping.

Gate values I would leave exactly where they are, with reasons:
`base_gate_m 2.0` (a wider gate starts admitting adjacent pedestrians;
2.0 m already sits at the pedestrian control p99 of 2.20 m),
`gap_slack_m 1.0`, `max_gap_keyframes 3` (1.2 s — beyond this, interpolating
becomes invention), `size_ratio_max 2.0`, and above all
`class_agnostic false` — 2 463 of the 3 419 joins are pedestrians, and turning
class-agnostic on would put pedestrian fragments in competition with bicycle and
motorcycle fragments at the same gate. Do not enable it.

If the rerun's `stitch.per_scene` shows a chunk with `frac_rows_on_chains_ge3_after`
below ~0.65, or a chain merging more than ~20 fragments, that chunk deserves the
same hand-read before publishing. Nothing on chunk_0000 came close.

### 7.2 What *does* need a decision before the full rerun (not a stitch gate)

**Interpolated rows are shipped with little or no LiDAR evidence, in
contradiction of the delivery note's own annotation rule.** Of the 746
interpolated rows admitted to `sample_annotation`:

```
                     points <= 0        245  (32.8 %)
                     points <= 1        286  (38.3 %)
                     points <= 5        363  (48.7 %)
  median points 7, p90 478
  by class: pedestrian 588, motorcycle 54, car 44, cycle_rickshaw 24, bicycle 19,
            bus 8, truck 7, cng_autorickshaw 2
```

against **0 of 15 395 real shipped rows below 5 points** (median 53). And these
counts are on the *more generous* raw basis (§6) — switching to the
ground-filtered cloud at the rerun will make them lower, not higher. So this is
not a basis artefact: interpolation is placing boxes at keyframes where the
sensor genuinely returned nothing, which is unsurprising, because occlusion is
often *why* Stage 7 dropped the object there in the first place.

Three consequences worth a decision:

1. `DELIVERY_NOTE.md` states "A box ships iff it has >= 5 LiDAR returns
   (single-sweep, ground-filtered, pre-inflation) AND detector confidence >= 0.5
   AND its BEV footprint is <= 2.0x class prior." 363 shipped rows do not satisfy
   the first clause. The note and the data disagree, and `check_release` does not
   test the claim.
2. 245 zero-point ground-truth boxes (1.5 % of the shipped table) can never be
   hit by a LiDAR detector. They are guaranteed false negatives and will depress
   reported recall.
3. nuScenes itself ships zero-point boxes and most protocols filter GT on
   `num_lidar_pts >= 1`, so this is defensible — *if it is stated*.

Pick one before the rerun: (a) add a minimum-point gate for interpolated rows
only, (b) leave the rows and amend the note's annotation rule to carve out
interpolated rows explicitly, or (c) leave both and add a checker warning. My
preference is (b) plus (c): the interpolated rows are what make the chains
scoreable, and dropping them re-fragments exactly what was just repaired. This is
a spec decision, so I have not made it and have changed no code.

### 7.3 Observations (reported, not fixed — this task changes no code)

- **Interpolation never fills a hole *inside* a Stage-7 fragment.**
  `stitch_scene` calls `_interpolate` only between consecutive *fragments* of a
  chain, so a keyframe missing inside one Stage-7 track stays missing. There are
  **1 834 such holes** on this chunk (1 254 of 2 keyframes, 580 of 3), worth
  2 414 unfilled rows — comparable in size to the 2 530 rows interpolation does
  fill. Both hand-read chains show them (keyframes 589 and 631 in chain `9228`;
  8 of them in chain `6226`). The exporter docstring says "a keyframe missing
  inside a joined chain is interpolated", which reads wider than what the code
  does. Given §7.2 I would *not* extend interpolation to these holes now — it
  would add ~2 400 more evidence-thin boxes — but the docstring should be
  narrowed to say "between two joined fragments".
- **The delivery note quantifies identity repair only by the median**, which is
  1.0 → 1.0 and reads as "the stitcher did nothing". `release_meta` already
  carries `mean_len_before/after` and `frac_rows_on_chains_ge3_before/after`;
  surfacing those two in the note would stop a reviewer drawing the wrong
  conclusion from the one number that does not move.
- **The note's "Extra layers" section is unconditional.** It describes `road/`
  and `coco_2d/` as siblings of `boxes/` even in a boxes-only export like this
  one, where neither directory exists. Harmless in a full release run; wrong in a
  partial one.
- **The illumination stratification axis is nearly degenerate** on this chunk:
  682 of 684 keyframes land in `day` and 2 in `dusk`, so the double-annotation
  cell table is effectively stratified by density alone. That is a property of a
  single daytime drive, not a defect, but the benchmark's illumination coverage
  claim cannot be met from chunk_0000.
- **`check_release` will always exit 1 on a chunk like this** (zero-instance
  classes + the 722 attribute warnings of §3.1). `run_day1_chunks.sh` only treats
  `rc >= 2` as fatal, so this is fine — just do not let the warning count be read
  as a problem.

### 7.4 Reproducing this

Everything above comes from the exported tree at
`/home/mt/dhakascenes/work_scratch/release_dryrun_chunk_0000/boxes` — the two
chain tables from `stitch_map.json` + `sample_annotation.json` +
`sample_annotation_excluded.json` + `ego_pose.json`, the join population from the
same tables grouped by `dhakascenes_chain_id` and
`dhakascenes_track_id_pre_stitch`. The scratch tree is left in place; it holds no
copied blobs (symlinks only, 43 MB) and nothing under `/home/mt/dhakascenes` was
deleted or modified.

---

## 8. `DELIVERY_NOTE.md`, first 40 lines

This is what the human-readable deliverable reads like (the file is 70 lines;
the remainder covers attributes, uncertainty, double annotation, anonymisation,
extra layers and the file manifest).

```markdown
# day1_chunk_0000_dryrun — delivery note

## Provenance
- export created (UTC): 2026-09-07T21:30:36Z
- nuScenes version dir: v1.0-dhaka-fixed
- pipeline git sha: b0ca63e959f3a637d43f16e78796f850bc50385e
- Stage 9 spec: dhakascenes-pilot/stage9_qa/v1
- stage tree: /home/mt/dhakascenes/work_day1/chunk_0000
- release config sha256: 4c0b6cbc79580678e810ab54c0b66a082ca05925f3aabfb8b3bbb00ce4976dce
- benchmark definition: /home/mt/dataset_benchmark/configs/benchmark_v1.0.yaml sha256 adc3cff1d0200449a42812fd2a68fe9b051bbfa52ddcea7e1e6e4154cb5a2b79

## Human pass
No human pass has run on this export: every row is a machine pre-annotation (`dhakascenes_source: pipeline`).

## Annotation rule
A box ships iff it has >= 5 LiDAR returns (single-sweep, ground-filtered, pre-inflation) AND detector confidence >= 0.5 AND its BEV footprint is <= 2.0x class prior. There are no camera-only boxes, so the visibility term V does not apply; `visibility_token` is a camera field-of-view proxy, not an occlusion estimate.

## Range
- pipeline range cap: 50 m (Stage 1 `range_cap_m` / eval region `_R_MAX_M`, the benchmark's class_range maximum)
- furthest exported box (observed, not a cap): 30.9 m in the ego BEV plane
- effective p99 range per class (included boxes): bicycle 28.41 m, bus 29.17 m, car 29.19 m, cng_autorickshaw 29.04 m, cycle_rickshaw 28.54 m, motorcycle 29.02 m, pedestrian 29.57 m, truck 28.95 m
- density radius (rho): 30 m, unchanged — `eval_region._RHO_RADIUS_M`, benchmark `stratification.density.radius_m`. It used to equal the annotation range cap; the cap is now 50 m, so rho is a density over a 30 m disc inside a 50 m region, not over the whole annotated region.

## Tiers
- admitted to sample_annotation: auto_accept
- excluded rows: 13180 in sample_annotation_excluded.json — tier_flagged 3610, tier_rejected 9570
`sample_annotation_excluded.json` is not a nuScenes table; the devkit does not load it.

## Classes
- present (instances): bicycle 316, bus 317, car 725, cng_autorickshaw 263, cycle_rickshaw 369, motorcycle 563, pedestrian 2730, truck 202
- in the detector vocabulary but absent on this route: barrier, construction_element, traffic_cone
- not producible by the 12-phrase detector vocabulary: animal, battery_rickshaw, covered_van, human_hauler, microbus, pushcart, tempo

## Identity (stitching)
- enabled: True
- fragments -> chains: 11882 -> 8463
- joins by gap (keyframes): {"1": 1690, "2": 928, "3": 801}
- interpolated rows: 2530
- median chain length before -> after, per scene: chunk_0000 1.0 -> 1.0
- max gap: 3 keyframes
```
