# Handoff — Stage-4 parallelism probe (negative result) and the C30 CVAT publish contract

**Date:** 2026-09-01 (evening) · **Repo:** `/home/mt/Zami/Annotation_pipeline` · **Branch:** `main`
**Nothing is committed.** Continues `2026-09-01-trackaware-3c-and-sam3text-handoff.md`
(same day, afternoon); read it first — its §5 decisions are still unratified and its
gates are still the open ones. This document covers only the evening's work.

---

## 1. Read this first

Two questions were asked and both are answered:

1. **"sam3.1 doesn't take the whole VRAM — can Stage 4 run in parallel to go faster?"**
   Measured, not guessed: yes it fits, **no it is not worth building** (§2). VRAM was
   never the bottleneck; GPU compute is. Two workers buy ~1.27×, ~12 min of a 59-min
   stage. Nothing was implemented.
2. **"A new run must not delete the old CVAT export — with a clean-slate-like flag when
   I do want everything gone."** Built, tested, recorded as **DECISIONS.md C30** (§3).
   Publishes are now additive by default (run-tagged task names); the wipe is
   `--cvat-replace`.

| | |
|---|---|
| Live repo | changed further (C30), still **all uncommitted** — the remote copy is still the only copy |
| Backup BEFORE the C30 edits | `~/Zami/ap_backup_20260901T181450.tgz` (1.6 MB; the 13:57 one still stands) |
| Test suite | **198 pass** (189 from the afternoon + 9 new naming tests), 2.5 s |
| CVAT server | **untouched today** — no publish, no wipe, no purge |
| Live work root | untouched; probe artifacts under `work/logs/smoke_partest_180006/` (kept as evidence, deletable) |
| GPU | left idle |

---

## 2. The Stage-4 parallelism probe — a measured NO

Question: the full Stage-4 run (provider `sam31_multiplex`, `facebook/sam3.1`) recorded
7,456 MiB peak on a 24,564 MiB card, so 2–3 workers fit in memory. Do they go faster?

**Method.** The 5-keyframe / 281-box smoke subset from the afternoon
(`work/logs/smoke_20260901T151325/s3_subset`), run through `masks.py` with the real
provider (`--model-id facebook/sam3.1 --revision daa63191…`, `--accept-degraded-upstream`,
explicit `--out-dir`s under `work/logs/smoke_partest_180006/`): once solo, then two
concurrent instances, with `nvidia-smi` sampled at 1 Hz. Both dual workers produced the
full 281 masks each; log timestamps separate model-load/compile from segmenting.

**Numbers.**

| | solo | 2 workers |
|---|---|---|
| wall clock | 28.0 s | 46.7 s each (fully overlapped) |
| setup (compile msg → first object) | 3.2 s | 9.2 s each (CPU contention, ~2.9×) |
| segmenting (first → last object) | 9.3 s | 14.7 s each |
| aggregate segmenting throughput | 0.54 kf/s | 0.68 kf/s = **1.27×** |
| VRAM peak | 7,291 MiB | 14,406 MiB |
| GPU util during segmenting | mean ≈ 63%, peak 98% | — (sampler mean diluted by post-run tail; unusable) |

**Reading.** The ~63% busy fraction caps the theoretical 2-worker gain at ~1/0.63 ≈ 1.59×;
contention delivers 1.27×. Projected on the full run (Stage 4 = 3,562 s ≈ 59 min):
~47 min, **saving ~12 min**. Three workers: ~22 GB — too tight for safety, and pointless
once compute is the wall. Caveat: a 5-keyframe probe, one measurement each arm —
directional, but the direction is unambiguous.

**Why it would also cost real work:** `masks.py` shards only by `--scenes` and the pilot
has exactly ONE scene (`chunk_0000`), so parallelism needs a keyframe-shard flag, per-shard
out-dirs, and a merge that restores keyframe order and emits one manifest + `_SUCCESS`.
(Keyframe sharding would at least be *correct*: cross-camera IoA-NMS is per-keyframe.)

**If Stage-4 speed ever matters, the better first move** is inside ONE process: the idle
37% is serial CPU work per frame (JPEG decode, `np.packbits`, writes) — a prefetch/
pipeline thread captures roughly the same win with no merge machinery. And keep
perspective: Stage 4 is ~14% of the 7.2 h chain; Stages 6 (8,260 s) and 7 (7,909 s) are
the wall-clock hogs.

Note for anyone re-deriving: the afternoon's `s4_ctrl` smoke ran provider `sam3_tracker`
(13.0 s); the probe deliberately used `sam31_multiplex` because that is the shipped
default and the operator's question. Do not compare the two wall-clocks across providers.

---

## 3. C30 — CVAT publishes are additive; `--cvat-replace` is the wipe

**The hazard that prompted it:** the wrapper's `cvat`/`cvat3d` arms passed `--replace`,
so the next chain run would have DELETED the previous pipeline's published tasks — the
exact tasks the operator wants to review side by side against the new run's. And merely
dropping `--replace` is a trap: names collide and `cvat_setup.py` SKIPS an existing name,
so the OLD run's pre-annotations would keep standing under names the new run appears to
own. Side-by-side required run-unique names.

**The mechanism** (full record: `DECISIONS.md` C30):

- Task names gain a **run tag**: `<scene> — OUR PIPELINE output [20260831T023812]`. The
  tag is the SOURCE manifest's mtime (`stage4_masks/run_manifest.json` for `cvat`,
  `stage8_inflate`'s for `cvat3d` — the same files the arms' freshness rules already
  anchor on), so a NEW run lands beside the old tasks and a REPUBLISH of the same run
  skips its own names. `CVAT_RUN_TAG` overrides. A publish with no source manifest is
  refused. The bracketed value above is live: it is what today's on-disk stage4 output
  would be tagged as.
- **`--cvat-replace`** (wrapper flag, parsed beside `--clean-slate`): arms
  `--replace-all-runs` on both publish arms — every pipeline-output task for the suffix
  in that project goes, tagged or legacy-untagged, then this run publishes fresh. Refused
  with exit 2 when the run has no publishing step (including via `--no-cvat`).
- **Never touched by either mode:** the human answer-key twins (untagged, separate
  project, no wipe path arms for them — C13 preserved). `--clean-slate` keeps its
  full-server purge.
- **Script-level flags** (usable standalone): `cvat_setup.py` / `cvat_setup_3d.py`
  `--run-tag` and `--replace-all-runs`; exact `--replace` now targets tagged names.
- **Known edge, pinned by a test:** the 3D OURS suffix *ends with* the 2D suffix, so in
  ONE project a 2D wipe would claim `<scene> 3D — OUR PIPELINE output`. Safe only because
  the exporters keep separate projects. **Do not merge those projects.**
- **The storage bill of side-by-side:** each kept run's `cvat3d` tasks hold their own
  ~45–80 MB point-cloud archive per scene. 2D tasks reference the share and stay cheap.
  Reclaim is `--cvat-replace` or `--clean-slate`, nothing else.

**Touched:** `scripts/cvat_setup.py` (`task_name`, `wipe_targets`, two flags, docstring),
`scripts/cvat_setup_3d.py` (same, GT hardcoded untagged/unwiped),
`scripts/run_stages.sh` (flag parse + refusal, both arms, header docs, `--help` range
125→140), `tests/test_cvat_publish_naming.py` (9 tests), `docs/DECISIONS.md` C30.

**Verified:** 198/198 pass; both refusal paths exit 2 with the message; `PRINT_STEPS=1`
baseline byte-identical; tag derivation resolves against the real manifest; both CLIs
parse; `bash -n` clean.

**C30's gate is OPEN:** no publish has run against a live CVAT server under this naming.
The first real `cvat`/`cvat3d` step is the test: expect new tagged tasks BESIDE the
legacy-named ones, twins untouched; under `--cvat-replace` expect legacy + tagged
pipeline tasks deleted, twins standing. Read the task list with your eyes.

---

## 4. What has NOT run / still owed

- **The comparison chain run itself** — the reason for all of this. Proposed shape,
  not yet launched (arm B has STILL never run; it is the actual CNG fix):

  ```bash
  VLM_CHECK=1 bash scripts/run_stages.sh 3f 3m 3c 4 5 6 7 8 eval viz cvat cvat3d
  ```

  With C30 in place this is now safe for the old export by default: the previous CVAT
  tasks survive it. (First run after it: eyeball the C30 gate, above.)
- **Everything inherited from the afternoon handoff:** C29 gate (1) — the five flagged
  decisions, unratified; gates (3)/(4) — no accuracy claim for `sam3_text` without human
  review; the `render_annotations.py` `--stage4-dir`/`--stage1-dir` flags so a human can
  actually look at smoke masks; the `sam3_exemplar` build/no-build decision.
- **COMMIT.** Now two decision records (C29, C30), two new stages' worth of code, tests,
  and four handover docs sit uncommitted, and the remote is the only copy. This remains
  the standout risk on the box.
- Deletable leftovers: `work/logs/smoke_partest_180006/` (§2 evidence),
  `~/zami_shadow/matrix.sh` + `~/zami_shadow/matrix/` (afternoon), and the now-stale
  shadow repo `~/zami_shadow/Annotation_pipeline` (live has moved past it — do not edit
  the shadow expecting it to matter).

---

## 5. Useful commands

```bash
PY=/home/mt/miniconda3/envs/ano_pipe/bin/python
cd ~/Zami/Annotation_pipeline

# the suite (198)
$PY -m pytest tests/ -q

# what the next run would do, without doing it
PRINT_STEPS=1 VLM_CHECK=1 bash scripts/run_stages.sh 3f 3m 3c 4 5 6 7 8 eval viz cvat cvat3d

# publish semantics
bash scripts/run_stages.sh cvat cvat3d                  # additive: new tasks beside old
bash scripts/run_stages.sh cvat cvat3d --cvat-replace   # wipe ALL pipeline tasks, publish fresh
bash scripts/run_stages.sh --help | sed -n '45,75p'     # the OVERWRITING contract, rewritten

# re-read the probe evidence (§2)
ls /home/mt/dhakascenes/work/logs/smoke_partest_180006/
```
