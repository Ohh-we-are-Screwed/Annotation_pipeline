# Handoff — track-aware Stage 3c, VLM toggles, SAM 3 text prompting

**Date:** 2026-09-01 (afternoon) · **Repo:** `/home/mt/Zami/Annotation_pipeline` · **Branch:** `main`
**Nothing is committed.** Continues `2026-09-01-stage3c-and-road-seg-handoff.md` (written
that morning); read it first for the substrate, Stage 3c's design and the road-seg state.
This document covers only the afternoon's work.

---

## 1. Read this first

Three things were asked for and all three are built: check each **tracked** object once
instead of once per box, a clean **on/off toggle** for the VLM check, and a Stage-4 mode
that prompts SAM with a box's **text label** instead of its box.

**None of it has touched the live repo, and none of it has run on a GPU.**

| | |
|---|---|
| Live repo `/home/mt/Zami/Annotation_pipeline` | **UNTOUCHED** — `git status` clean, HEAD `e13b0ef` |
| Shadow repo `/home/mt/zami_shadow/Annotation_pipeline` | where every change lives and every test ran |
| Live work root `/home/mt/dhakascenes/work` | **read-only throughout**; nothing written, the lock never taken |
| Backup taken before the first push | `~/Zami/ap_backup_20260901T135750.tgz` (1.1 MB — `pipeline/ scripts/ tests/ configs/ docs/ handover/`) |
| Test suite (shadow) | **184 pass**, 0 fail (60 pre-existing + 28 new Stage-3c + 96 new Stage-4) |
| GPU work done | **none** beyond the read-only spike (§3); every smoke in the plan is still owed (§4) |

**Everything below is unit-tested and measured on paper. Nothing below is a quality
claim.** The two numbers that would settle quality — did the per-track verdicts agree
with the per-box ones, and are text-prompted masks better — require the GPU passes in §4
and, for masks, human review.

---

## 2. What changed

### A — Stage 3c gains `--check-mode per_track` (`pipeline/stage3c_check/check.py`)

One VLM call per **track**, not per box. The key is `(channel, int(track_id))` and it is
**scene-local**: Stage 3b counts from 0 in every camera of every scene, so a bare integer
is not a key. A deterministic ranking picks the representative crop (detector-evidenced
before propagated → fully-inside-image before clamped → larger min-side → larger area →
file order; **never** scores, which are three incommensurable scales). The verdict is
recomputed per box from the cached *phrase*, and **sub-floor members inherit it**
(`propagated_over_small: true`) — a real behavioural divergence from `per_box`, not an
optimisation: it heals temporal label flicker and decides 6,913 boxes the per-box floor
skips. A three-candidate retry ladder re-asks on `unclear`/`error` only.

Every record now carries `verdict_source ∈ {vlm, track, none}`. `per_box` output is
**not** byte-identical to before — the exact additive delta is enumerated in
`DECISIONS.md` C29 and asserted by a test. `--max-rows`, an out-dir fence and a preflight
that refuses **before** the 24 GB server spawn are also new; `pipeline/common/rowmeta.py`
now holds `optional_c27_array`, shared by Stage 3c and Stage 4.

### B — Orchestrator toggles (`scripts/run_stages.sh`, `scripts/run_pilot.py`)

`VLM_CHECK` is tri-state: unset = today exactly; `1` removes every typed `3c` and inserts
**one immediately before the first Stage 4** (never appends — appending recreates the trap
the flag exists to fix); `0` removes a typed `3c` and says so loudly. Anything else is
refused with exit 2.

`VLM_USE_CHECKED` is a **separate** gate — consumption, not production. `0` demotes
`checked` → `merged` inside `select_stage3_dir_for_4` (one site, covering the pre-scan,
the Stage-4 arm and `export_taxonomy`). **With both unset, behaviour is today's exactly,
including today's silent stale-checked inheritance** — only an explicit
`VLM_USE_CHECKED=0` closes that.

Also: auto-demotion to `per_box` with a loud note when the 3c input is `stage3_proposals`
(the `all 3c` stale-3b cell); the missing stale-3b note; scope gates on the merged and
checked levels; `MASK_TEXT_PROMPT=1 → --text-prompt`; `PRINT_STEPS=1` dry-run; and D.2 —
the eval arm now passes `--taxonomy "$(export_taxonomy)"`, which used to `KeyError`
*after* all the GPU work on a merged run.

**New failure mode, worth knowing before the first chain run:** with `3c` injected
mid-chain, an rc-2 refusal from `check.py` aborts the whole chain before Stage 4. The
preflight makes that land in ~0.2 s instead of after a 900 s model load, and there are
exactly two escapes: `VLM_ALLOW_SHARED_GPU=1` (GPU busy) and `VLM_ALLOW_UNTRACKED=1`
(zero track coverage).

### C — Stage 4 provider `sam3_text` (`pipeline/stage4_masks/masks.py`)

A **new provider name**, not a mode of `sam3_tracker` — one provider string must never
mean two invocations. One **text-only** detector forward per *(image, distinct phrase)*;
instances are assigned to that phrase's boxes greedy-descending-IoU **one-to-one**, floor
0.5; unmatched boxes fall back per box to the tracker head's box prompt. Masks scatter
back into a pre-sized array **by original box index** (a phrase-grouped return would
permute the one-mask-per-box order undetectably). Defaults: score threshold **0.3**,
strip leading article **true**, detector dtype **bfloat16** — each set from a
measurement, each recorded with that measurement in the config provenance. `--text-prompt`
preflight-**refuses** on mobile_sam / sam2_video / sam3_tracker / sam31_multiplex (the
last is actively destructive: its text spelling calls `reset_state`).

### D — Ledger, blocker, docs

`DECISIONS.md` **C29** (the whole record — semantics, measurements, and the five flagged
decisions). `docs/conformance.yaml`: **1.5-r3 rewritten** (the square resize is real and
lives inside the vendored processor — see §3), **5.5-r1 off VIOLATES** onto CONFORMS with
a real test, and every shifted `masks.py:` / `model_interfaces.py:` anchor re-anchored;
`docs/CONFORMANCE.md` re-rendered. D.1: `int(None)` on a merged-over-3b tree crashed Stage
4 — fixed, with a regression test.

---

## 3. The measured numbers

**Track savings — measured read-only against the LIVE 3b tree with the shipped planner,
not estimated.** 18,416 rows / 46,722 boxes / 5,795 tracks, coverage 46,722/46,722 =
1.0000, scanned in 0.2 s. At the shipped 32 px floor: **31,378 per-box calls → 4,203
per-track = 7.47×** (at 24 px: 37,180 → 4,851 = 7.66×). Members per track: median 3, p90
18, **max 242**. 6,913 sub-floor boxes gain a propagated verdict; 8,431 stay skipped;
6,913 + 8,431 = 15,344 = exactly today's per-box skipped count.

**Wrapper matrix.** `{'', all, 'all 3c', '3f 3m 4', 4, 3c, '5 6 7'}` × `VLM_CHECK`
{unset, 0, 1} × `VLM_USE_CHECKED` {unset, 0}, over two synthetic work roots =
**84 cells, 84 match** expectations written before the run.

**SAM 3 text spike** (10 Dhaka CAM_FRONT frames, 66 stage-3 boxes, 29 (image, phrase)
groups — read-only, idle 4090). Four headline findings:

1. **Text-only matches 61% of boxes** at threshold 0.4 and **71%** at 0.3. The rest take
   the box-prompt fallback. That is the shipped mechanism's real yield.
2. **The attribution finding.** Supplying each phrase's own boxes lifts matching to
   **97%** — but a *nonsense* string ("a purple hovercraft") with the same boxes scores
   **94%**, and no text at all scores 97%. With boxes present the text's entire
   measurable contribution is **+2 boxes of 66 over a nonsense string**: the geometry
   encoder is doing the work. So the exemplar path was **deliberately not built**. It is
   recorded as a named, unbuilt variant `sam3_exemplar` (in the manifest and in C29), to
   be built only under its own provider name with that control in its own provenance.
3. **"a bus" returns ZERO instances on 10 of 10 frames** at every threshold ≥ 0.3,
   despite 4 boxes being present. Every bus box would silently take the fallback — correct
   behaviour, invisible unless counted, which is why `text_prompt.per_phrase` exists.
4. **C13 is settled, and not the way the plan expected.** The tracker path does not merely
   square-resize internally — it is the *same* image-processor class with the *same*
   geometry as the detector: 1008×1008, `default_to_square: true`, mask 288×288, no
   padding, and `pixel_values [1,3,1008,1008]` measured from a real 1280×720 frame on
   **both**. There is no mismatch between the two paths and **no letterboxing work item**.
   Reading (a) is taken for both and flagged (§5). Consequence to carry: 720 → 1008 → 288
   decides the mask boundary at **~2.5 px of native vertical resolution**.

Also measured: both heads resident = **3,775 MiB** peak (bf16 detector + fp32 tracker;
fp32/fp32 = 5,629 MiB), **83 ms** per detector forward in bf16 (2.7× fp32, no accuracy
loss: 41/66 vs 40/66).

**Caveat on every text number above:** 10 frames, one scene, one camera, daytime, 66
boxes, on a tree carrying the **nuScenes** taxonomy — so `a rickshaw` / `an auto rickshaw`
were measured for instances-returned only, with no arm-B box to align against. Tight-box
IoU against the stage-3 box measures **alignment**, not mask quality, and rewards sloppy
masks. Directional only.

---

## 4. What has NOT run

**Every GPU pass in the plan's Verification section is still owed.** Unit tests and a
read-only tree scan are the whole of the evidence.

- **V3 — the 3c smoke A/B.** `per_box` and `per_track` over `stage3b_track2d`, each with
  its **own** `--out-dir` under `$WORK_ROOT/logs/smoke_<ts>/` (never omit `--out-dir`:
  `masks.py` defaults it to the live `stage4_masks` tree and clears markers there), and
  `--taxonomy configs/taxonomy_pilot_nuscenes.yaml` (the 3b tree's caption is nuScenes;
  the default taxonomy refuses). Assert `n_vlm_calls` drops; the
  sub-floor-of-a-checked-track index set must equal the `propagated_over_small` set
  **exactly** — predicted cardinality **6,913**. Above-floor disagreements are *expected*
  and are a number to read, not a gate.
- **V4 — the Stage-4 text smoke** plus its box-prompt control, and ~20 side-by-side
  renders for the operator. No accuracy claim exists without that human review.
- **V1 at scale / V6.** The suite passes; no chain has been run end to end.

Two of Workstream B's leftovers on the box, re-runnable and deletable:
`~/zami_shadow/matrix.sh` and `~/zami_shadow/matrix/` (~200 KB of synthetic work roots).

---

## 5. Decisions taken on your behalf — flagged for ratification

Full reasoning and the numbers behind each are in `DECISIONS.md` C29 gate (1).

1. **The wrapper defaults `VLM_CHECK_MODE=per_track`** while `check.py`'s own CLI defaults
   to `per_box`. Standalone behaviour is preserved; the wrapper opts into the 7.47×.
2. **`VLM_CHECK=0` removes a typed `3c`** — a flag overriding an explicit argument. It is
   announced loudly, but it is still an override.
3. **The C13 reading (a)**, taken for the tracker path as well as `sam3_text`. Rejecting it
   changes no code — it puts 1.5-r3 back on VIOLATES and opens a letterboxing item against
   **today's default provider**, not just the new one.
4. **`text_prompt_strip_article = true`**, on a +3-of-66 effect from one scene.
5. **`text_detector_dtype = bfloat16`**, on a 41/66-vs-40/66 accuracy control.

Standing risk, not a decision to ratify but a thing to know: one representative decides
its whole track, and the retry ladder fires on `unclear`/`error` — **never** on
confident-and-wrong. One bad verdict can rewrite up to **242** frames. `n_members` rides
on every propagated verdict and `largest_relabel_track` names the widest actual relabel in
the run, so the blast radius is readable from the manifest rather than re-derived.

---

## 6. Useful commands

```bash
PY=/home/mt/miniconda3/envs/ano_pipe/bin/python
SHADOW=~/zami_shadow/Annotation_pipeline

# the suite (184)
cd $SHADOW && $PY -m pytest tests/ -q

# what the wrapper would run, without running it
PRINT_STEPS=1 VLM_CHECK=1 bash scripts/run_stages.sh all

# produce a checked tree and consume it (the 7.47x path needs a track-bearing input)
VLM_CHECK=1 VLM_CHECK_MODE=per_track bash scripts/run_stages.sh 3b 3f 3m 3c 4

# produce it but do NOT consume it this run
VLM_CHECK=1 VLM_USE_CHECKED=0 bash scripts/run_stages.sh 4

# text-prompted masks
MASK_TEXT_PROMPT=1 bash scripts/run_stages.sh 4

# re-validate + re-render the ledger (GIT_DIR only so the stamp names the real commit)
cd $SHADOW && GIT_DIR=/home/mt/Zami/Annotation_pipeline/.git $PY scripts/check_conformance.py
```

---

## 7. Suggested next steps

1. **Ratify or reject the five decisions in §5.** Item 3 is the only one that reaches
   outside this change.
2. **Run the 3c smoke A/B (§4).** It is bounded by `--max-rows` and it produces the first
   per-call 3c throughput number that has ever existed.
3. **Run the Stage-4 text smoke and look at the renders.** 39% of boxes falling back is
   the expected shape, not a bug; "a bus" at 100% fallback is the thing to look at.
4. **Then decide whether `sam3_exemplar` is wanted** (§3 finding 2). It nearly doubles the
   yield at +4 ms, and it is honest only under its own name.
5. **Promote the shadow to live** once 2–4 pass — or **commit the shadow**, which is
   probably overdue: `pipeline/stage3c_check/`, `pipeline/common/rowmeta.py`, `tests/` and
   `handover/` are all still untracked, and the remote copy is the only copy.

---

## 8. Addendum (same day, later): the smokes RAN — results and one fix

Steps 2, 3 and 5 of §7 are DONE. Everything below is measured, on this box, artifacts under
`work/logs/smoke_20260901T151325/` (kept deliberately — they are the evidence; 1.4 MB).

### 3c smoke A/B (40 rows / 281 boxes of the live 3b tree, chunk_0000)

| | per_box | per_track |
|---|---|---|
| VLM calls | 190 | **67** (2.8× fewer) |
| wall clock | 421.5 s | **144.5 s** (2.92×) |
| sec/call (FIRST-EVER 3c throughput number) | 2.218 | 2.157 |
| relabeled | 17 | 16 |
| errors | 0 | 0 |
| track_coverage | — | 1.0 (86 tracks, 62 checked) |

Structural row-diff: PASS (only `vlm_check` + the three label arrays differ).
`largest_relabel_track`: CAM_FRONT_LEFT track 1, "a car" → "a motorcycle", 5 members.
Extrapolated full-tree 3c cost at 2.2 s/call: per_box ≈ 19 h, per_track ≈ 2.6 h.

### Stage-4 text smoke (same 40-row subset)

First attempt REFUSED (rc 2) on a real defect: transformers' own
`post_process_instance_segmentation` skips its `target_sizes` resize when ZERO instances
survive the threshold (`if len(masks) > 0`, image_processing_sam3.py:919-927), returning
(0, 288, 288); the adapter's resolution guard refused that legitimate empty case. Fixed
(18 lines in `Sam3TextAdapter._detect_text`: empty stack ⇒ no instances ⇒ box fallback;
the guard is unchanged for every non-empty stack) + 5 new CPU tests that exercise the REAL
transformers post-processor (`TestRealPostProcessInstanceSegmentation`) — two of them fail
against the pre-fix code. Suite is now **189 passed** (live repo).

Re-run (rc 0, `_SUCCESS`): 281 masks, all asserted 1280×720; **224 text_matched (79.7%)
/ 57 box_fallback**; 100 detector forwards; n_text_duplicate_rejected 0;
n_cross_phrase_mask_overlap 1 (recorded, not investigated). Per phrase (matched/boxes):
bicycle 9/9, car 54/74, motorcycle 5/7, pedestrian 142/167, truck 13/16, **bus 1/8** —
the predicted systematic text-path failure class, now visible in
`text_prompt.per_phrase` instead of invisible. Control (box-prompt) arm: 281 masks,
13.0 s, byte-level manifest shape as before.

Renders (§7.3's "look at the renders") were SKIPPED: `render_annotations.py` exposes no
per-stage-dir overrides, so it cannot read a smoke tree without touching the live paths.
A follow-up could add `--stage4-dir`/`--stage1-dir` flags.

Note for a separate look: with HF_HOME unset the adapter resolved the facebook/sam3
snapshot from `~/.cache/huggingface/hub`, not the project cache — pre-existing
environment-dependence, not introduced here.
