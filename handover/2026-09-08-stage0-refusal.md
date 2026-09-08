# Stage 0 refused the new Dhaka capture: `token_graph_closed`, and a 3.1 h probe that said nothing

**Date:** 2026-09-08
**Interpreter:** `/home/mt/miniconda3/envs/ano_pipe/bin/python` (`ano_pipe`)
**Profile:** `DHAKASCENES_SUBSTRATE=dhaka6`
**Branch:** `main` (nothing pushed)
**Data:** `/media/mt/vol_2/Annotation-pipe-data/full-fused`
**Refused run:** `/home/mt/dhakascenes/work_b/chunk_0000/logs/run_20260908_053122.log` (rc 2, 11,204 s)

## Status

**Unblocked.** `chunk_0000` is usable, the allowlist is on disk, and the probe for one scene now
takes **57 s instead of 11,204 s** (196x).

| commit | what |
|---|---|
| `ad911d1` | `fixup_a_nusc: close the sample_data chain, not just the sample chain` |
| `a28d54b` | `stage0: probe what the run asked for, and only what the substrate reads` |
| (this file) | config repoint + handover |

Full suite: **603 passed** (`$PY -m pytest tests/ -q`), up from 561; 42 tests added.

---

## 1. Root cause

`scripts/fixup_a_nusc.py` produced a metadata tree whose `sample_data` prev/next chains are broken.
`drop_incomplete_samples()` deleted the 868 `sample_data` rows belonging to the 59 dropped keyframes
and then re-linked **only the `sample` chain**. Every surviving *neighbour* of a removed row kept
pointing at it, leaving 255 distinct tokens referenced but absent from `sample_data.json` — which is
exactly what Stage 0's `token_graph_closed` predicate exists to catch:

```
sample_data_prev_dangling  4 .. 70 per scene
sample_data_next_dangling  0 .. 63 per scene
```

All eleven scenes tripped it, so `run_probe` hit `HardStop("zero usable scenes")`, which by design
writes no report and no allowlist — the evidence died with the process.

The bug survived the fixup's own 69 tests because the test fixture's `sample_data` rows all had
`prev == next == ""`. There were no chains to break.

### What it was NOT

Measured on `chunk_0000` before any change (each predicate called directly, ~3 s total):

| predicate | verdict | measurement |
|---|---|---|
| `version_matches` | ok | `v1.0-dhaka-fixed` present under `meta_root` |
| `channels_complete` | ok | 1364 keyframes, 0 incomplete, all 7 dhaka6 channels |
| `sweeps_cover_window` | **ok** | `w_acc 1 / 0 ns`; min = median = 1 record in window, 0 missing, 0 oversized gaps |
| `token_graph_closed` | **FAIL** | 70 prev + 63 next dangling |
| `files_resolve` / `files_parse` | ok (re-run) | 9548 keyframe rows checked, 0 missing, 0 unparseable |

So the leading suspect was wrong: `sweeps_cover_window` handles an anchor-only window correctly even
when the export *does* ship sweeps. With `w_acc_duration_ns: 0` the window is `[t, t]`, which always
contains the anchor row itself — 1 record against a gate of `0.8 x 1`. It passes, and it passed here.

The known-corrupt LiDAR blob was also not the cause; see §5.

## 2. The fix — data side (`ad911d1`)

`drop_incomplete_samples()` now splices the `sample_data` chain across the removed rows, walking the
**original** links past a removed run so consecutive drops close in one step and a chain that ends on
a removed row terminates at `""`.

It touches only links that pointed **into the removed set**. A link that was already dangling before
the fixup ran stays dangling — healing it here would launder a defect the script did not cause and
hide it from the one stage whose job is to find it. There is a test for that.

Nine tests added, five of them red before the change.

### The regenerated version dir

Following this script's own precedent, a **new** version dir was written; nothing was mutated or
deleted:

```
DHAKASCENES_SUBSTRATE=dhaka6 $PY scripts/fixup_a_nusc.py \
  --dataroot /media/mt/vol_2/Annotation-pipe-data/full-fused \
  --version v1.0-dhaka --out-version v1.0-dhaka-fixed2 \
  --sanitize-scene-names --swap-channels CAM_LEFT CAM_RIGHT
```

Same 59 drops, same 178,731 swapped rows, same 89,796 interpolated camera poses, same 11 sanitized
scene names. Verified against the superseded dir:

- of the 13 tables, **only `sample_data.json` differs**;
- same token set, same 551,580 rows;
- the only fields that differ are `prev` and `next`, in **276** row-fields;
- **0** dangling `sample_data` links; 0 rows whose sample is unknown.

`configs/paths_b_fused_chunk_0000.yaml` now names `v1.0-dhaka-fixed2`, with the reason in the file.
`v1.0-dhaka-fixed` is kept — it is what the refusal was measured against.

**The other ten chunks need no separate repair.** The fixup is per-root, not per-chunk, and the
dangling-link count over the whole of `v1.0-dhaka-fixed2` is zero.

## 3. The fix — cost (`a28d54b`)

Both halves were general improvements to the stage, not workarounds for this dataset.

### `--scenes`

Stage 0's `main()` accepted no scene subset, so a run invoked for one chunk probed all eleven scenes
of the single nuScenes root this capture ships as. It now takes `--scenes` on the same convention as
every other stage (`nargs="*"`, unknown name refused rather than ignored, a bare `--scenes` meaning
"all" and never "none"), and `scripts/run_stages.sh` passes the run's `SCENE_ARGS` through to it.

Because `usable_scenes.json` is the only scene list any later stage reads, **a narrower probe says so
in the artifact**. `scenes_not_probed` lands in both the report and the allowlist, and `totals` now
separates `n_scenes` (probed) from `n_scenes_in_metadata` / `n_scenes_not_probed`. One usable scene
out of eleven and one usable scene out of one are different facts and the file now distinguishes
them. The console prints `scenes NOT probed: 10 (...) — not looked at, not passed`.

### File checks scoped to the accumulation window

`check_files` stat-ed and parsed every `sample_data` row. Under a profile whose window is the anchor
keyframe alone, no stage ever opens `sweeps/` — and on this capture that is **40,288 of 49,836 rows
per scene**, 81 % of a spinning-disk walk that could not change any answer.

New predicate-independent helper:

```python
def sweeps_are_read(w_acc_count=W_ACC_COUNT, w_acc_duration_ns=W_ACC_DURATION_NS) -> bool:
    return w_acc_count > 1 or w_acc_duration_ns > 0
```

Keyframes are **always** checked. Sweeps are checked whenever the window can reach one, which is true
for `dhaka` (5 / 0.5 s) and `nuscenes` (5 / 0.5 s) — so those profiles' archived runs mean exactly
what they meant — and false for `dhaka6` (1 / 0 ns).

Honesty is kept the way this file already keeps it for RADAR. The skip is counted and declared:

```json
"sweeps_checked": false,
"n_sweep_rows_not_checked": 40288,
"scope": "required channels only (RADAR reported, not gated); keyframes only — the profile's
          accumulation window is the anchor keyframe alone, so no sweep file is read by any
          stage and none was opened here"
```

plus `config.sweep_files_checked` / `config.sweep_files_note` in the report. Rows skipped by
**window** are counted apart from channels skipped by **channel**; pooling them would make neither
number mean anything.

### And: the hard stop now names the predicate

`HardStop("zero usable scenes")` printed nothing but its own conclusion. It now lists, per scene, the
predicates that failed — so the next 3-hour refusal is not another mystery.

## 4. The re-run

```
$ DHAKASCENES_SUBSTRATE=dhaka6 DHAKASCENES_PATHS_CONFIG=configs/paths_b_fused_chunk_0000.yaml \
    $PY -m pipeline.stage0_data_probe.probe --scenes dhaka_20260905_174950_chunk_0000

metadata fingerprint : b2226aaa4223655f5dee37c2999edfe25aa1950e47d20a1952c27b09634afd5b
scenes usable        : 1/1  keyframes 1364
scenes NOT probed    : 10 (..._chunk_0001, ..._chunk_0002, ..._chunk_0003, ...)  — not looked at, not passed
file checks          : keyframe rows only (anchor-only accumulation window); sweep rows counted, not verified
  ok   dhaka_20260905_174950_chunk_0000  unknown:dhaka day   1364 kf
partition            : UNSATISFIABLE
wrote .../stage0_data_probe/usable_scenes.json
wrote .../stage0_data_probe/probe_report.json
                                              ELAPSED 57.41 s   (rc 1)
```

All five per-scene predicates pass; `files_resolve` / `files_parse` checked 9,548 keyframe rows and
declared 40,288 sweep rows unchecked.

**rc 1 is the pre-existing Dhaka condition, not a new one.** `verify_partition` encodes the §11
decision-3 partition over nuScenes scene names (`scene-0061` …), which no Dhaka capture contains, so
the partition has been reported unsatisfiable on every Dhaka run to date. `_SUCCESS.degraded` is on
disk and Stage 1 consumes it under `--accept-degraded-upstream`, exactly as the day-1 chunked runs
did. Nothing here changed that; whether to teach `verify_partition` about non-nuScenes substrates is
a separate decision and was left alone.

## 5. Still open: one content-corrupt LiDAR blob in `chunk_0008`

`samples/LIDAR_TOP/011894.pcd.bin` (the CRC failure noted on extraction) is **not** what refused
anything. It lives in `chunk_0008`, not `chunk_0000`, and it passes `files_parse`: 7,563,140 bytes,
a clean multiple of 20, 378,157 points, comfortably inside dhaka6's `(10_000, 1_000_000)` band.

It is nevertheless garbage. Read as `(-1, 5) float32` it yields `ring` values of `-9.74, -9.73, …`
where the rig writes integer ring ids `0-3` / `100` / `101`, and `y` up to 254 m / `z` up to 101 m
against a neighbouring cloud's 50 m / 11 m. The record stream is misaligned; the file will not crash
Stage 1, it will feed it a plausible-shaped cloud of nonsense for one keyframe.

**`chunk_0008` will therefore PASS the probe with a corrupt cloud in it.** That is a real exposure,
and it is not a hole this fix opened — no predicate has ever looked at file *content*.

### The general rule I would propose (not implemented here)

1. **The probe stays all-or-nothing per scene.** `usable_scenes.json` means "this scene is wholly
   sound", and §5.1 is explicit that Stage 0 repairs, substitutes and skips nothing. Teaching it to
   drop individual keyframes would make the allowlist's name a lie and move a repair into the one
   stage that is defined by not performing repairs.
2. **A per-keyframe drop belongs in `scripts/fixup_a_nusc.py`**, which already drops keyframes and
   already writes a new version dir. The natural shape is a `--drop-unparseable` pass reusing the
   probe's own `_parse_pcd_bin` / `_parse_jpeg`, recording the dropped tokens in `fixup_meta.json`.
   That keeps the split the pipeline already has: the probe *measures*, the fixup *repairs*, and the
   repair is a new immutable version dir with its provenance beside it.
3. **`files_parse` should grow one cheap structural check** for the `.pcd.bin` case, because the size
   band demonstrably cannot see this: assert the `ring` column of the first and last records is
   finite and integral. That is rig-independent (every profile's rings are integers), costs one seek
   per file, and would have flagged `011894` outright. It should be a *predicate*, not a repair.

Until (2) or (3) exists, **`chunk_0008` should be run with that keyframe's cloud treated as suspect**,
or the source re-extracted for that one file.

## 6. Verification

- `$PY -m pytest tests/ -q` → **603 passed** (was 561). 42 tests added:
  `tests/test_fixup_a_nusc.py` +9, `tests/test_stage0_probe_scope.py` +33 (new file).
- Suite also green under `DHAKASCENES_SUBSTRATE=nuscenes` for the new file.
- Under `DHAKASCENES_SUBSTRATE=dhaka6` the suite is 602 passed / 1 failed:
  `test_stage1_thin_stereo.py::test_legacy_config_lets_every_ring_vote`, which asserts
  `IngestConfig`'s *legacy* ring-gate defaults and reads them from the active profile instead.
  **Pre-existing and unrelated** — confirmed by reverting `probe.py` and `fixup_a_nusc.py` to
  `88176dd` and reproducing the same single failure.
- `bash -n scripts/run_stages.sh` clean.
