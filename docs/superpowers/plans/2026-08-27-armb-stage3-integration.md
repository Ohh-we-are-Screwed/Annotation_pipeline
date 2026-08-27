# Arm B Stage-3 Integration (two-detector merge) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the RSUD20K-fine-tuned YOLO11x (arm B) into the Stage 3 proposal path as a second detector, and build the vocabulary-authority merge that combines it with frozen arm A into `stage3_merged/`, which Stage 4 consumes unchanged.

**Architecture:** Arm B runs through the *existing* `proposals.py` `Yolo11Adapter` — the adapter is already vocabulary-generic (the OIV7 map proves it); it only needs a new class-map YAML and a superset taxonomy whose caption keeps arm A's caption as a byte prefix. A new `pipeline/stage3_merge/merge.py` pairs the two output trees row-by-row, arbitrates overlaps by a class-pair table (never by score), moves suppressed arm A boxes into an auditable per-row ledger, and emits Stage 3's exact schema plus additive keys (the Stage 3b / C27 trick, used a second time).

**Tech Stack:** ultralytics 8.4.120 · torch 2.5.1+cu124 · Python `/home/mt/miniconda3/envs/ano_pipe/bin/python` · pytest 8.3.3 · RTX 4090

**Spec:** `docs/RUNNING.md` § "Stage 3 arm A / arm B — the two-detector proposal design (2026-08-26, IN BUILD)". Also grounded in `docs/GAP_ANALYSIS.md` §3 (G3), `docs/comprehensive.md` §7.2 (gate S1), README §7.3 (C25), `docs/DECISIONS.md` C21/C25/C27.

## Global Constraints

- Interpreter is ALWAYS `/home/mt/miniconda3/envs/ano_pipe/bin/python` (C15). Below, `PY` means exactly that path.
- Run tests as: `cd /home/mt/Zami/Annotation_pipeline && $PY -m pytest tests/<file> -v` (tests insert repo root into `sys.path` themselves — copy the `tests/test_export_release.py` header pattern).
- **Arm A is frozen.** Nothing may modify `configs/taxonomy_pilot_nuscenes.yaml`, `configs/coco_to_phrase_nuscenes.yaml`, `pipeline/stage3_proposals/proposals.py` behavior for existing runs, or anything under `Results/`. New class spaces go in NEW files.
- **Additive schema only** (C27): merged rows carry every Stage 3 (and, when present, Stage 3b) key with values copied — never recomputed, never re-rounded. New information rides in new keys.
- Arbitration is by **vocabulary authority, never by score** (spec table). Suppressed boxes are retained in a per-row ledger, never silently dropped.
- The phrase spellings are fixed by the spec: `a rickshaw` and `an auto rickshaw` (NOT "a cng" — SigLIP/S1 read phrases as natural language).
- Arm B artifact and its labels inherit RSUD20K's **CC BY-NC 4.0** licence; the DECISIONS entry in Task 9 records this (owed per RUNNING.md).
- No new dependencies. No network at test time. GPU only in Task 7.
- Prerequisite (manual, outside this plan): training run `r1280-4` was cut by a power failure at epoch 45/80; resume with
  `/home/mt/miniconda3/envs/ano_pipe/bin/yolo train resume model=local_yolox_build/runs/r1280-4/weights/last.pt`.
  Tasks 1–6, 8, 9 do not depend on it. Tasks 4 and 7 run against whatever `weights/best.pt` exists (currently epoch 36, mAP50-95 0.740) and are simply re-run after training finishes — Task 4's provenance JSON records exactly which checkpoint shipped, so a stale artifact is detectable, not silent.

## File Structure

| File | Responsibility |
|---|---|
| Create `configs/taxonomy_pilot_dhaka.yaml` | v3 class space: the 10 nuScenes-benchmark phrases (verbatim, same order) + 2 Dhaka phrases appended. Caption-prefix property is what lets merged rows keep arm A spans untouched. |
| Create `configs/rsud20k_to_phrase_dhaka.yaml` | arm B's model.names → phrase bridge: 2 mapped, 11 excluded. Mirrors `coco_to_phrase_nuscenes.yaml` / `oiv7_to_phrase_nuscenes.yaml`. |
| Modify `configs/release_category_map.yaml` | add the 2 new phrase aliases → dbench classes. |
| Create `local_yolox_build/scripts/export_armb.py` | copy a run's `best.pt` → `artifacts/yolo11x-rsud20k-armb.pt` + provenance JSON. |
| Modify `local_yolox_build/scripts/predict_armb.py:52` | default `--weights` → the new artifact name. |
| Create `pipeline/stage3_merge/__init__.py`, `pipeline/stage3_merge/merge.py` | the merge: arbitration, row surgery, manifest, three-state marker, CLI. |
| Modify `scripts/run_stages.sh` | opt-in steps `3f` (arm B proposals) and `3m` (merge); `select_stage3_dir_for_4` prefers `stage3_merged`. |
| Create `tests/test_armb_configs.py`, `tests/test_stage3_merge.py`, `tests/test_export_armb.py` | unit tests. |
| Modify `docs/RUNNING.md`, `docs/DECISIONS.md` (C28), `docs/CVAT_GUIDE.md` note | status flip IN BUILD → built; licence + merge decision record. |

Interfaces locked across tasks (later tasks import these exact names):

- `configs/taxonomy_pilot_dhaka.yaml` phrases, in caption order: the 10 v2 phrases, then `"a rickshaw"`, then `"an auto rickshaw"`.
- `pipeline/stage3_merge/merge.py` exports: `STAGE_SPEC = "dhakascenes-pilot/stage3_merge/v1"`, `KEEP_BOTH`, `SUPPRESS_ARM_A`, `ARBITRATION: dict[str,str]`, `ARM_B_PHRASES = ("a rickshaw", "an auto rickshaw")`, `MergeContractError(RuntimeError)`, `merge_rows(row_a, row_b, *, caption, taxonomy, iou_threshold) -> dict`, `run(...)`, `main(argv) -> int`.
- Artifact path: `local_yolox_build/artifacts/yolo11x-rsud20k-armb.pt` (basename MUST start with `yolo` — `infer_proposal_provider()` at `proposals.py:1455` routes on that prefix).

---

### Task 1: Superset taxonomy `configs/taxonomy_pilot_dhaka.yaml`

**Files:**
- Create: `configs/taxonomy_pilot_dhaka.yaml`
- Test: `tests/test_armb_configs.py`

**Interfaces:**
- Consumes: `load_taxonomy`, `build_caption` from `pipeline/stage3_proposals/proposals.py` (verified importable headless).
- Produces: the v3 class space every later task keys on; the caption-prefix invariant Task 5 asserts at merge time.

Why a NEW file and not an edit: `class_space.py:_from_manifest` re-derives unreachability from the taxonomy *file on disk* for older manifests, and `eval_3d.py` defaults to the nuScenes file — editing it in place would silently change the class space of archived arm A evaluations. The frozen-arm-A constraint therefore forces a superset **sibling**.

Why the prefix property holds: `build_caption` joins phrases with `". "` and appends a final `"."` (`proposals.py:594-624`). v2's caption ends `"…a construction vehicle. a trailer."`; appending two phrases yields `"…a trailer. a rickshaw. an auto rickshaw."` — the v2 caption, trailing period included, is a byte prefix, so every arm A `phrase_char_spans` entry is valid under the v3 caption unchanged.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_armb_configs.py`:

```python
"""Tests for the arm B class-space configs (Tasks 1-3 of the 2026-08-27 plan).

Run: /home/mt/miniconda3/envs/ano_pipe/bin/python -m pytest tests/test_armb_configs.py -v
"""

from __future__ import annotations

import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pipeline.stage3_proposals.proposals import (  # noqa: E402
    build_caption,
    load_class_map,
    load_taxonomy,
)

NUSCENES_TAXONOMY = os.path.join(ROOT, "configs", "taxonomy_pilot_nuscenes.yaml")
DHAKA_TAXONOMY = os.path.join(ROOT, "configs", "taxonomy_pilot_dhaka.yaml")
RSUD20K_MAP = os.path.join(ROOT, "configs", "rsud20k_to_phrase_dhaka.yaml")

# The 13 class names baked into the arm B checkpoint's model.names, copied from
# local_yolox_build/configs/rsud20k_yolo11x.yaml `names:` (index order).
RSUD20K_NAMES = (
    "person", "rickshaw", "rickshaw van", "cng", "truck", "pickup truck",
    "car", "motorcycle", "bicycle", "bus", "micro bus", "covered van",
    "human hauler",
)


class TestDhakaTaxonomy:
    def test_loads_and_has_twelve_phrases(self):
        t = load_taxonomy(DHAKA_TAXONOMY)
        assert len(t.phrases) == 12

    def test_first_ten_phrases_are_v2_verbatim_in_order(self):
        v2 = load_taxonomy(NUSCENES_TAXONOMY)
        v3 = load_taxonomy(DHAKA_TAXONOMY)
        assert v3.phrases[: len(v2.phrases)] == v2.phrases

    def test_appended_phrases_and_order(self):
        v3 = load_taxonomy(DHAKA_TAXONOMY)
        assert v3.phrases[-2:] == ("a rickshaw", "an auto rickshaw")

    def test_caption_prefix_property(self):
        v2 = load_taxonomy(NUSCENES_TAXONOMY)
        v3 = load_taxonomy(DHAKA_TAXONOMY)
        cap2, cap3 = build_caption(v2.phrases), build_caption(v3.phrases)
        assert cap3.text.startswith(cap2.text)
        # arm A spans must be valid verbatim under the v3 caption
        assert cap3.phrase_char_spans[: len(v2.phrases)] == cap2.phrase_char_spans

    def test_dhaka_categories_map_to_new_phrases(self):
        v3 = load_taxonomy(DHAKA_TAXONOMY)
        p2c = v3.phrase_to_categories
        assert p2c["a rickshaw"] == ("dhaka.cycle_rickshaw",)
        assert p2c["an auto rickshaw"] == ("dhaka.cng_autorickshaw",)

    def test_v2_exclusions_survive(self):
        v2 = load_taxonomy(NUSCENES_TAXONOMY)
        v3 = load_taxonomy(DHAKA_TAXONOMY)
        assert set(v2.excluded_categories) <= set(v3.excluded_categories)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `$PY -m pytest tests/test_armb_configs.py -v`
Expected: FAIL / ERROR — `taxonomy file not found: .../taxonomy_pilot_dhaka.yaml` (an `UpstreamRefusal`).

- [ ] **Step 3: Write the taxonomy file**

Create `configs/taxonomy_pilot_dhaka.yaml`. The `prompt_phrase` block is the v2 block **copied verbatim, same order**, plus two appended rows; `excluded_categories` copied verbatim. Full content:

```yaml
# DhakaScenes — v3 class space: the pilot's nuScenes-benchmark 10 phrases plus
# the two arm B phrases (2026-08-27, Stage 3 two-arm design; docs/RUNNING.md
# "Stage 3 arm A / arm B").
#
# THIS FILE IS A SUPERSET SIBLING of taxonomy_pilot_nuscenes.yaml, NOT its
# replacement. Arm A and every archived Results/ evaluation stay bound to v2;
# editing v2 in place would re-derive the class space of frozen runs
# (class_space.py falls back to the taxonomy file on disk for older manifests).
#
# INVARIANT the merge depends on (asserted in pipeline/stage3_merge/merge.py
# and tests/test_armb_configs.py): the first 10 phrases are v2's, verbatim, in
# v2's order, so build_caption() of this file yields a caption of which v2's
# caption is a byte PREFIX — every arm A phrase_char_spans entry is then valid
# under this caption unchanged.
#
# The two appended category keys are PSEUDO-CATEGORIES: no nuScenes category
# exists for a rickshaw or a CNG. The `dhaka.` prefix keeps them out of any
# nuScenes namespace and makes rows carrying them self-describing in the
# `nuscenes_categories` record field (which nothing downstream reads — X-6).
#
# Phrase spellings are the spec's: "a rickshaw", "an auto rickshaw" — NOT
# "a cng". Two consumers read phrases as natural language (review-tool SigLIP,
# gate S1 text arms); `cng` is Bangladeshi usage a text encoder has not seen.
#
# provenance: authored 2026-08-27 from taxonomy_pilot_nuscenes.yaml v2
#             (C21 class space) + docs/RUNNING.md arm B design (2026-08-26).

spec: dhakascenes-pilot/taxonomy_pilot_dhaka/v3
version: 3
n_categories: 17   # 15 nuScenes categories + 2 dhaka pseudo-categories
n_phrases: 12      # the 10-phrase v2 space + a rickshaw + an auto rickshaw

prompt_phrase:
  # --- v2 block, verbatim, same order (PREFIX INVARIANT — do not reorder) ---
  vehicle.car: "a car"
  human.pedestrian.adult: "a pedestrian"
  human.pedestrian.child: "a pedestrian"
  human.pedestrian.construction_worker: "a pedestrian"
  human.pedestrian.police_officer: "a pedestrian"
  human.pedestrian.personal_mobility: "a pedestrian"
  movable_object.barrier: "a road barrier"
  movable_object.trafficcone: "a traffic cone"
  vehicle.truck: "a truck"
  vehicle.motorcycle: "a motorcycle"
  vehicle.bus.bendy: "a bus"
  vehicle.bus.rigid: "a bus"
  vehicle.bicycle: "a bicycle"
  vehicle.construction: "a construction vehicle"
  vehicle.trailer: "a trailer"
  # --- arm B additions (dhaka pseudo-categories; no nuScenes source) --------
  dhaka.cycle_rickshaw: "a rickshaw"
  dhaka.cng_autorickshaw: "an auto rickshaw"

excluded_categories:
  movable_object.debris: "no clean everyday phrase; benchmark ignores it"
  movable_object.pushable_pullable: "973 false positives as 'a trash bin' against 82 GT boxes"
  static_object.bicycle_rack: "680 false positives against 54 GT boxes; static furniture"
  animal: "0 instances"
  human.pedestrian.stroller: "0 instances"
  human.pedestrian.wheelchair: "0 instances"
  vehicle.emergency.ambulance: "0 instances"
  vehicle.emergency.police: "0 instances; was the most-predicted class at 0.0% class-correct"

# Per-class thresholds keyed by PHRASE. The two arm B phrases fall back to
# default_threshold like everything else: UNTUNED, same standing caveat as the
# v2 file. Arm B's precision is load-bearing for the merge (an arm B false
# positive deletes an arm A label), so tuning these two on the `tuning` scene
# subset is owed BEFORE any label-quality claim — recorded here, not hidden.
thresholds: {}
default_threshold: 0.40
```

- [ ] **Step 4: Run the taxonomy tests, verify they pass**

Run: `$PY -m pytest tests/test_armb_configs.py::TestDhakaTaxonomy -v`
Expected: 6 passed. (The `TestRsud20kMap` class arrives in Task 2; if you wrote it already, those still fail — that is Task 2's business.)

- [ ] **Step 5: Commit**

```bash
git add configs/taxonomy_pilot_dhaka.yaml tests/test_armb_configs.py
git commit -m "Add v3 Dhaka taxonomy: v2 class space + arm B phrases, caption-prefix invariant"
```

---

### Task 2: Arm B class map `configs/rsud20k_to_phrase_dhaka.yaml`

**Files:**
- Create: `configs/rsud20k_to_phrase_dhaka.yaml`
- Test: `tests/test_armb_configs.py` (append a class)

**Interfaces:**
- Consumes: `load_class_map(path, taxonomy)` (`proposals.py:524`) — the YAML top-level key MUST be `coco_to_phrase` / `excluded_coco_classes` (the loader's key names are fixed; the OIV7 map already reuses them for a non-COCO vocabulary, so this is the established pattern, not a hack).
- Produces: the file `proposals.py --class-map` takes for the arm B run; `ClassMap.assert_covers` will hold it against the checkpoint's 13 `model.names` at load.

The ship-list semantics: mapping ONLY `rickshaw` and `cng` and excluding the other 11 gives the arm B ship filter for free — `Yolo11Adapter.propose()` passes `classes=sorted(self._phrase_index_of_class_id)` (only mapped ids) into ultralytics NMS (`proposals.py:1341-1345`), which is exactly `predict_armb.py`'s filter, applied at the same layer, so scaffolding classes can never consume a `max_det` slot.

- [ ] **Step 1: Append the failing tests**

Append to `tests/test_armb_configs.py`:

```python
class TestRsud20kMap:
    def test_map_loads_against_dhaka_taxonomy(self):
        t = load_taxonomy(DHAKA_TAXONOMY)
        m = load_class_map(RSUD20K_MAP, t)
        assert m.mapping == {"rickshaw": "a rickshaw", "cng": "an auto rickshaw"}

    def test_covers_the_checkpoint_names_exactly(self):
        t = load_taxonomy(DHAKA_TAXONOMY)
        m = load_class_map(RSUD20K_MAP, t)
        m.assert_covers(RSUD20K_NAMES)  # raises UpstreamRefusal on any drift

    def test_unreachable_is_the_whole_v2_space(self):
        v2 = load_taxonomy(NUSCENES_TAXONOMY)
        v3 = load_taxonomy(DHAKA_TAXONOMY)
        m = load_class_map(RSUD20K_MAP, v3)
        assert m.unreachable_phrases(v3) == v2.phrases

    def test_rejected_against_v2_taxonomy(self):
        import pytest
        from pipeline.common.manifest import UpstreamRefusal
        v2 = load_taxonomy(NUSCENES_TAXONOMY)
        with pytest.raises(UpstreamRefusal):
            load_class_map(RSUD20K_MAP, v2)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_armb_configs.py::TestRsud20kMap -v`
Expected: 4 errors — `class map not found`.

- [ ] **Step 3: Write the map file**

Create `configs/rsud20k_to_phrase_dhaka.yaml`:

```yaml
# DhakaScenes — RSUD20K class -> prompt phrase (Stage 3 arm B, YOLO11 provider).
#
# The bridge from the arm B checkpoint's own model.names (13 RSUD20K classes,
# baked at fine-tune time from local_yolox_build/configs/rsud20k_yolo11x.yaml)
# into the v3 phrase class space (configs/taxonomy_pilot_dhaka.yaml). Same
# contract as coco_to_phrase_nuscenes.yaml, enforced by the same loader
# (proposals.py:load_class_map + ClassMap.assert_covers): mapped + excluded
# must equal model.names EXACTLY, so a wrong checkpoint refuses instead of
# relabelling every box. The top-level keys are the loader's fixed names
# (`coco_to_phrase` / `excluded_coco_classes`); the OIV7 map set the precedent
# of reusing them for a non-COCO vocabulary.
#
# ARM B SHIPS 2 OF ITS 5 TRAINED CLASSES. person / car / motorcycle were
# trained as decision-boundary scaffolding (car<->cng, person<->rickshaw) and
# are deliberately EXCLUDED here: arm A already covers them from COCO, and
# emitting them from arm B would put the two arms in competition on arm A's
# own turf, where the vocabulary-authority merge rule does not apply
# (docs/RUNNING.md "Arm B trains 5 classes and ships 2"). Exclusion at this
# layer IS the ship filter: Yolo11Adapter passes only mapped ids as
# `classes=` into ultralytics NMS, the same mechanism predict_armb.py uses.
# The remaining 8 were never trained (train-time classes=[0,1,3,6,7]) and
# their head rows are untrained noise.
#
# provenance: authored 2026-08-27. names read from the fine-tune data config
#             local_yolox_build/configs/rsud20k_yolo11x.yaml (class 6 renamed
#             `private car` -> `car` there for COCO warm-start, so the
#             checkpoint's model.names says `car`).

spec: dhakascenes-pilot/rsud20k_to_phrase_dhaka/v1
version: 1
n_source_classes: 13
n_mapped: 2

coco_to_phrase:
  rickshaw: "a rickshaw"
  cng: "an auto rickshaw"

excluded_coco_classes:
  # trained scaffolding (see header):
  person: "trained for the person<->rickshaw boundary; arm A's turf at inference"
  car: "trained for the car<->cng boundary; arm A's turf at inference"
  motorcycle: "trained as nearest COCO geometry to a three-wheeler; arm A's turf"
  # untrained (outside train-time classes=[0,1,3,6,7]; head rows are noise):
  rickshaw van: "untrained (filtered at fine-tune time)"
  truck: "untrained (filtered at fine-tune time)"
  pickup truck: "untrained (filtered at fine-tune time)"
  bicycle: "untrained (filtered at fine-tune time)"
  bus: "untrained (filtered at fine-tune time)"
  micro bus: "untrained (filtered at fine-tune time)"
  covered van: "untrained (filtered at fine-tune time)"
  human hauler: "untrained (filtered at fine-tune time)"
```

- [ ] **Step 4: Run the tests, verify pass**

Run: `$PY -m pytest tests/test_armb_configs.py -v`
Expected: all 10 pass.

- [ ] **Step 5: Commit**

```bash
git add configs/rsud20k_to_phrase_dhaka.yaml tests/test_armb_configs.py
git commit -m "Add RSUD20K->phrase class map: arm B ships 2, excludes 11 by name"
```

---

### Task 3: Release-map aliases for the two new phrases

**Files:**
- Modify: `configs/release_category_map.yaml` (the `map:` block, next to the existing rickshaw aliases at lines ~63-75)
- Test: `tests/test_armb_configs.py` (append a class)

**Interfaces:**
- Consumes: the release map's `map:` alias table (exact-match after lowercase + whitespace collapse, per its header).
- Produces: `"a rickshaw" -> cycle_rickshaw`, `"an auto rickshaw" -> cng_autorickshaw` so `scripts/export_release.py` can resolve merged-pipeline categories instead of erroring on unknown strings.

Note the mapping judgment, stated rather than hidden: RSUD20K's `rickshaw` class does not distinguish cycle from battery rickshaws; the release taxonomy does. `"a rickshaw" -> cycle_rickshaw` follows the existing `"a cycle rickshaw"` alias precedent and the RSUD20K paper's usage; the conflation is recorded in the YAML comment so a reviewer of battery-rickshaw counts knows where to look.

- [ ] **Step 1: Append the failing test**

```python
class TestReleaseMapAliases:
    def test_arm_b_phrases_resolve(self):
        with open(os.path.join(ROOT, "configs", "release_category_map.yaml"), "rb") as fh:
            doc = yaml.safe_load(fh)
        assert doc["map"]["a rickshaw"] == "cycle_rickshaw"
        assert doc["map"]["an auto rickshaw"] == "cng_autorickshaw"
        # alias values must stay inside the declared 18-class space
        assert set(doc["map"].values()) <= set(doc["classes"])
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_armb_configs.py::TestReleaseMapAliases -v`
Expected: FAIL — `KeyError: 'a rickshaw'`.

- [ ] **Step 3: Edit the release map**

In `configs/release_category_map.yaml`, inside the `map:` block beside the existing `"a cycle rickshaw": cycle_rickshaw` and `"a cng auto-rickshaw": cng_autorickshaw` entries, add:

```yaml
  # arm B phrases (taxonomy_pilot_dhaka.yaml, 2026-08-27). RSUD20K `rickshaw`
  # does not separate cycle from battery rickshaws; mapping to cycle_rickshaw
  # follows the paper's usage — the conflation is recorded here on purpose.
  "a rickshaw": cycle_rickshaw
  "an auto rickshaw": cng_autorickshaw
```

- [ ] **Step 4: Run full config tests + the existing export-release suite (regression)**

Run: `$PY -m pytest tests/test_armb_configs.py tests/test_export_release.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add configs/release_category_map.yaml tests/test_armb_configs.py
git commit -m "Release map: alias the two arm B phrases into the dbench class space"
```

---

### Task 4: Arm B artifact export (`export_armb.py`)

**Files:**
- Create: `local_yolox_build/scripts/export_armb.py`
- Modify: `local_yolox_build/scripts/predict_armb.py` (the `--weights` default, line 52)
- Test: `tests/test_export_armb.py`

**Interfaces:**
- Consumes: a run dir (`runs/r1280-4`) holding `weights/best.pt`, `args.yaml`, `results.csv`; `ship_indices`/`SHIP_NAMES` from `predict_armb.py`.
- Produces: `local_yolox_build/artifacts/yolo11x-rsud20k-armb.pt` (basename starts with `yolo` — provider inference requirement) and `local_yolox_build/artifacts/armb_provenance.json` with keys `{"artifact", "sha256", "source_run", "checkpoint_epoch", "best_fitness", "classes_trained", "ship_names", "best_row", "exported_at"}`. Task 7 and the wrapper use the artifact path; the DECISIONS entry quotes the sha.
- Pure function for tests: `provenance_from_run(run_dir: str) -> dict` — everything derivable WITHOUT torch (args.yaml classes / model / data; the results.csv row with max fitness `0.1*mAP50 + 0.9*mAP50-95`).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_export_armb.py`:

```python
"""Tests for local_yolox_build/scripts/export_armb.py (torch-free parts).

Run: /home/mt/miniconda3/envs/ano_pipe/bin/python -m pytest tests/test_export_armb.py -v
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "local_yolox_build", "scripts"))

from export_armb import provenance_from_run  # noqa: E402

CSV = """epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
1,100.0,1.0,1.0,1.0,0.8,0.7,0.90,0.60,1.0,0.7,1.1,0.005,0.005,0.005
2,200.0,0.9,0.9,0.9,0.9,0.8,0.93,0.74,0.9,0.6,1.0,0.006,0.006,0.006
3,300.0,0.8,0.8,0.8,0.9,0.8,0.92,0.73,0.8,0.5,1.0,0.007,0.007,0.007
"""

ARGS = """task: detect
model: /somewhere/yolo11x.pt
data: /somewhere/rsud20k_yolo11x.yaml
epochs: 80
classes:
- 0
- 1
- 3
- 6
- 7
"""


def _fake_run(tmp_path):
    run = tmp_path / "r-test"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"not a real checkpoint")
    (run / "results.csv").write_text(CSV)
    (run / "args.yaml").write_text(ARGS)
    return str(run)


def test_best_row_is_max_fitness(tmp_path):
    prov = provenance_from_run(_fake_run(tmp_path))
    # fitness = 0.1*mAP50 + 0.9*mAP50-95 -> epoch 2 (0.759) beats epoch 3 (0.749)
    assert prov["best_row"]["epoch"] == 2
    assert abs(prov["best_fitness"] - (0.1 * 0.93 + 0.9 * 0.74)) < 1e-9
    assert prov["classes_trained"] == [0, 1, 3, 6, 7]
    assert prov["source_run"].endswith("r-test")
    assert prov["ship_names"] == ["rickshaw", "cng"]


def test_refuses_run_without_best_pt(tmp_path):
    import pytest
    run = tmp_path / "empty"
    (run / "weights").mkdir(parents=True)
    (run / "results.csv").write_text(CSV)
    (run / "args.yaml").write_text(ARGS)
    with pytest.raises(SystemExit):
        provenance_from_run(str(run))
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_export_armb.py -v`
Expected: import error — `No module named 'export_armb'`.

- [ ] **Step 3: Write the exporter**

Create `local_yolox_build/scripts/export_armb.py`:

```python
"""Export a fine-tune run's best.pt as THE arm B artifact, with provenance.

The artifact name starts with `yolo` on purpose: Stage 3's provider inference
(pipeline/stage3_proposals/proposals.py:infer_proposal_provider) routes a
weights basename starting with `yolo` to the ultralytics adapter. Rename it
and Stage 3 will try to treat the file as a caption-provider hub id.

Usage:
    python scripts/export_armb.py --run runs/r1280-4
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

BUILD = Path("/home/mt/Zami/Annotation_pipeline/local_yolox_build")
ARTIFACT_NAME = "yolo11x-rsud20k-armb.pt"

# Same ship list as inference; imported where torch is available, duplicated
# here so provenance_from_run stays importable without ultralytics installed.
SHIP_NAMES = ("rickshaw", "cng")


def provenance_from_run(run_dir: str) -> dict:
    """Everything about the run that does NOT need torch: args, best epoch."""
    run = Path(run_dir)
    best = run / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"{best} not found: nothing to export")
    args = yaml.safe_load((run / "args.yaml").read_text())
    with open(run / "results.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{run/'results.csv'}: no completed epochs")

    def fitness(row: dict) -> float:
        return 0.1 * float(row["metrics/mAP50(B)"]) + 0.9 * float(row["metrics/mAP50-95(B)"])

    best_row = max(rows, key=fitness)
    return {
        "source_run": str(run.resolve()),
        "classes_trained": list(args.get("classes") or []),
        "ship_names": list(SHIP_NAMES),
        "epochs_completed": int(rows[-1]["epoch"]),
        "epochs_budget": int(args.get("epochs", 0)),
        "best_fitness": fitness(best_row),
        "best_row": {
            "epoch": int(best_row["epoch"]),
            "mAP50": float(best_row["metrics/mAP50(B)"]),
            "mAP50_95": float(best_row["metrics/mAP50-95(B)"]),
        },
        "data_config": str(args.get("data", "")),
        "base_model": str(args.get("model", "")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(BUILD / "runs" / "r1280-4"))
    ap.add_argument("--out", default=str(BUILD / "artifacts" / ARTIFACT_NAME))
    args = ap.parse_args()

    prov = provenance_from_run(args.run)

    # Torch-side checks: the checkpoint must name every shipped class, and its
    # recorded best_fitness must agree with results.csv (a mismatch means the
    # csv and the weights are from different runs).
    import torch
    from predict_armb import ship_indices  # sibling module; refuses on missing names

    src = Path(args.run) / "weights" / "best.pt"
    ck = torch.load(src, map_location="cpu", weights_only=False)
    names = {int(k): str(v) for k, v in ck["model"].names.items()}
    prov["ship_indices"] = ship_indices(names)
    prov["checkpoint_epoch"] = int(ck.get("epoch", -1))
    ck_fitness = ck.get("best_fitness")
    if ck_fitness is not None and abs(float(ck_fitness) - prov["best_fitness"]) > 1e-3:
        raise SystemExit(
            f"checkpoint best_fitness {float(ck_fitness):.5f} != results.csv max "
            f"{prov['best_fitness']:.5f}: weights and csv are not from the same run"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    prov.update({
        "artifact": str(out),
        "sha256": sha,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY-NC 4.0 (inherited from RSUD20K; see DECISIONS C28)",
    })
    prov_path = out.parent / "armb_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    print(f"exported {out}\nsha256   {sha}\nprovenance {prov_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Point `predict_armb.py` at the artifact name**

In `local_yolox_build/scripts/predict_armb.py` line 52, change:

```python
    ap.add_argument("--weights", default=str(BUILD / "artifacts" / "arm_b.pt"))
```

to:

```python
    ap.add_argument("--weights", default=str(BUILD / "artifacts" / "yolo11x-rsud20k-armb.pt"))
```

- [ ] **Step 5: Run tests, verify pass**

Run: `$PY -m pytest tests/test_export_armb.py -v`
Expected: 2 passed.

- [ ] **Step 6: Run the real export (uses current best.pt, epoch 36; re-run after training resumes/finishes)**

Run: `cd local_yolox_build && $PY scripts/export_armb.py --run runs/r1280-4`
Expected: prints artifact path + sha256; `artifacts/armb_provenance.json` exists; `best_row.epoch` matches the checkpoint (36 today, later after the resume).

- [ ] **Step 7: Commit**

```bash
git add local_yolox_build/scripts/export_armb.py local_yolox_build/scripts/predict_armb.py tests/test_export_armb.py
git commit -m "Add arm B artifact exporter with provenance; artifact name feeds provider inference"
```

(`artifacts/*.pt` is git-ignored; the provenance JSON is small — add it too if present: `git add -f local_yolox_build/artifacts/armb_provenance.json` is NOT needed, it is not ignored.)

---

### Task 5: Merge core — arbitration + row surgery (pure, no I/O)

**Files:**
- Create: `pipeline/stage3_merge/__init__.py` (empty), `pipeline/stage3_merge/merge.py` (core half)
- Test: `tests/test_stage3_merge.py`

**Interfaces:**
- Consumes: `build_caption`, `load_taxonomy`, `pairwise_iou` from `pipeline.stage3_proposals.proposals`; serialized Stage 3 / Stage 3b rows (plain dicts).
- Produces: `merge_rows(row_a, row_b, *, caption, taxonomy, iou_threshold) -> dict` — the merged row; plus module constants `ARBITRATION`, `ARM_B_PHRASES`, `KEEP_BOTH`, `SUPPRESS_ARM_A`, `STAGE3B_EXTENSIONS`, `MergeContractError`. Task 6 wraps this in the driver.

Design decisions locked here (from the spec, with the two calls the spec leaves open resolved as follows):

1. **Suppressed arm A boxes leave the main arrays** and move to `row["merge"]["suppressed_arm_a"]`. Rationale: Stage 4 masks *every* box in the arrays in order, and Stage 5 lifts every mask — a suppressed-but-in-array box would propagate its wrong label through the whole chain. The in-record ledger is Stage 3's own `dedup` precedent; §8.4's reviewer sees what the contest removed because the ledger carries the full box, score, class, winner index and IoU.
2. **Overlap trigger is IoU > `iou_threshold` (default 0.5)** against the best-overlapping arm B box. The class-pair *table* decides what happens; IoU only decides *whether* the pair is a contest. The threshold is config, recorded per-row and in the manifest, flagged unvalidated.
3. Phrases in neither table row (`a road barrier`, `a traffic cone`, `a construction vehicle`, `a trailer`, `a car`… — i.e. anything not listed) default to **keep both**, counted in `n_overlap_out_of_table`, so an unforeseen contest is visible, not resolved silently.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_stage3_merge.py`:

```python
"""Tests for pipeline/stage3_merge/merge.py.

Run: /home/mt/miniconda3/envs/ano_pipe/bin/python -m pytest tests/test_stage3_merge.py -v
"""

from __future__ import annotations

import copy
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pipeline.stage3_proposals.proposals import build_caption, load_taxonomy  # noqa: E402
from pipeline.stage3_merge.merge import (  # noqa: E402
    ARM_B_PHRASES,
    MergeContractError,
    merge_rows,
)

DHAKA = os.path.join(ROOT, "configs", "taxonomy_pilot_dhaka.yaml")


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy(DHAKA)


@pytest.fixture(scope="module")
def caption(taxonomy):
    return build_caption(taxonomy.phrases)


def _span(caption, phrase):
    return list(caption.phrase_char_spans[caption.phrases.index(phrase)])


def _row(caption, names, boxes, **over):
    row = {
        "spec": "dhakascenes-pilot/stage3_proposals/v1",
        "keyframe_token": "kf0", "scene_token": "sc0", "t_ns": 1, "time_base": "utc",
        "coverage_config": "full", "channel": "CAM_FRONT",
        "sample_data_token": "sd0", "calibrated_sensor_token": "cs0",
        "ego_pose_token": "ep0", "dt_ns": 0, "image_path": "img/0.jpg",
        "image_size_px": [1600, 900], "model_input_size_px": [1600, 928],
        "resize_policy": "letterbox",
        "checkpoint": {"model_id": "m", "revision": "r", "sha256": "s"},
        "prompt": {"caption_sha256": "old", "taxonomy_sha256": "old", "span_map": None},
        "n_proposals": len(boxes),
        "score_aggregation": "yolo_class_confidence",
        "dedup": {"n_in": len(boxes), "n_out": len(boxes)},
        "boxes_xyxy_px": [list(b) for b in boxes],
        "scores": [0.9] * len(boxes),
        "class_names": list(names),
        "nuscenes_categories": [["x"] for _ in names],
        "phrase_char_spans": [_span(caption, n) for n in names],
        "seed": 0,
    }
    row.update(over)
    return row


BOX = [100.0, 100.0, 200.0, 200.0]        # the contested region
BOX_FAR = [500.0, 500.0, 600.0, 600.0]    # elsewhere


class TestArbitration:
    def test_cng_suppresses_car(self, caption, taxonomy):
        a = _row(caption, ["a car"], [BOX])
        b = _row(caption, ["an auto rickshaw"], [[102.0, 101.0, 199.0, 198.0]])
        m = merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        assert m["class_names"] == ["an auto rickshaw"]
        assert m["proposal_arm"] == ["arm_b"]
        assert m["n_proposals"] == 1
        led = m["merge"]
        assert led["n_suppressed_arm_a"] == 1
        (s,) = led["suppressed_arm_a"]
        assert s["class_name"] == "a car"
        assert s["suppressed_by"] == 0          # index of the cng box in MERGED arrays
        assert s["box_xyxy_px"] == BOX

    def test_pedestrian_keeps_both(self, caption, taxonomy):
        a = _row(caption, ["a pedestrian"], [BOX])
        b = _row(caption, ["a rickshaw"], [[110.0, 90.0, 210.0, 205.0]])
        m = merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        assert m["class_names"] == ["a pedestrian", "a rickshaw"]
        assert m["proposal_arm"] == ["arm_a", "arm_b"]
        assert m["merge"]["n_suppressed_arm_a"] == 0
        assert m["merge"]["n_kept_both"] == 1

    def test_below_iou_threshold_is_no_contest(self, caption, taxonomy):
        a = _row(caption, ["a car"], [BOX])
        b = _row(caption, ["an auto rickshaw"], [BOX_FAR])
        m = merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        assert m["class_names"] == ["a car", "an auto rickshaw"]
        assert m["merge"]["n_suppressed_arm_a"] == 0

    def test_out_of_table_overlap_keeps_both_and_counts(self, caption, taxonomy):
        a = _row(caption, ["a road barrier"], [BOX])
        b = _row(caption, ["a rickshaw"], [[101.0, 101.0, 201.0, 201.0]])
        m = merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        assert m["merge"]["n_overlap_out_of_table"] == 1
        assert len(m["class_names"]) == 2

    def test_scores_never_arbitrate(self, caption, taxonomy):
        # arm A very confident, arm B weak: authority still wins (C21's lesson)
        a = _row(caption, ["a car"], [BOX], scores=[0.99])
        b = _row(caption, ["an auto rickshaw"], [[102.0, 101.0, 199.0, 198.0]],
                 scores=[0.31])
        m = merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        assert m["class_names"] == ["an auto rickshaw"]


class TestSchema:
    def test_empty_arm_b_roundtrips_original_keys(self, caption, taxonomy):
        a = _row(caption, ["a car", "a bus"], [BOX, BOX_FAR])
        b = _row(caption, [], [])
        before = copy.deepcopy(a)
        m = merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        for key, value in before.items():
            if key == "prompt":
                continue  # caption/taxonomy shas are rewritten by design
            assert json.dumps(m[key], sort_keys=True) == json.dumps(value, sort_keys=True), key
        assert m["prompt"]["caption_sha256"] == caption.sha256
        assert m["merge"]["n_arm_b_in"] == 0

    def test_input_rows_not_mutated(self, caption, taxonomy):
        a = _row(caption, ["a car"], [BOX])
        b = _row(caption, ["an auto rickshaw"], [[102.0, 101.0, 199.0, 198.0]])
        a2, b2 = copy.deepcopy(a), copy.deepcopy(b)
        merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        assert a == a2 and b == b2

    def test_arm_b_spans_and_categories(self, caption, taxonomy):
        a = _row(caption, [], [])
        b = _row(caption, ["a rickshaw", "an auto rickshaw"], [BOX, BOX_FAR])
        m = merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        assert m["phrase_char_spans"] == [_span(caption, "a rickshaw"),
                                         _span(caption, "an auto rickshaw")]
        assert m["nuscenes_categories"] == [["dhaka.cycle_rickshaw"],
                                            ["dhaka.cng_autorickshaw"]]

    def test_stage3b_parallel_arrays_extended(self, caption, taxonomy):
        a = _row(caption, ["a car", "a pedestrian"], [BOX, BOX_FAR],
                 track_ids=[7, 8], box_sources=["yolo", "recovered"],
                 n_propagated_hops=[0, 2], refined=[False, True],
                 boxes_xyxy_px_original=[None, [1.0, 2.0, 3.0, 4.0]])
        b = _row(caption, ["an auto rickshaw"], [[102.0, 101.0, 199.0, 198.0]])
        m = merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        # car suppressed -> its entries leave EVERY parallel array
        assert m["class_names"] == ["a pedestrian", "an auto rickshaw"]
        assert m["track_ids"] == [8, None]
        assert m["box_sources"] == ["recovered", "arm_b"]
        assert m["n_propagated_hops"] == [0, 0]
        assert m["refined"] == [True, False]
        assert m["boxes_xyxy_px_original"] == [[1.0, 2.0, 3.0, 4.0], None]

    def test_refuses_leaked_arm_b_class(self, caption, taxonomy):
        a = _row(caption, [], [])
        b = _row(caption, ["a car"], [BOX])
        with pytest.raises(MergeContractError):
            merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)

    def test_refuses_frame_identity_mismatch(self, caption, taxonomy):
        a = _row(caption, [], [])
        b = _row(caption, [], [], sample_data_token="OTHER")
        with pytest.raises(MergeContractError):
            merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_stage3_merge.py -v`
Expected: collection error — `No module named 'pipeline.stage3_merge'`.

- [ ] **Step 3: Write the core**

Create empty `pipeline/stage3_merge/__init__.py`, then `pipeline/stage3_merge/merge.py`:

```python
"""Stage 3 merge — arm A (COCO YOLO11x) + arm B (RSUD20K fine-tune) -> stage3_merged/.

Design: docs/RUNNING.md "Stage 3 arm A / arm B" (2026-08-26) and DECISIONS C28.

Emits Stage 3's exact schema in Stage 3's row order plus additive keys (the
Stage 3b trick, C27, used a second time), so Stage 4 consumes the output
unchanged via --stage3-dir. Arbitration is a CLASS-PAIR TABLE, never a score
contest: arm A is confident on its wrong answers (`car` 0.85 beats `cng`
0.55), and a score contest would rebuild C21's failure in a new mechanism.

A suppressed arm A box LEAVES the parallel arrays and moves, whole, into the
row's `merge.suppressed_arm_a` ledger. In the arrays it would ride through
Stage 4 (one mask per box, in order) and Stage 5 (one lift per mask) under a
label the merge just ruled impossible; in the ledger it is retained, auditable
and inert — Stage 3's own `dedup` ledger precedent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pipeline.common.manifest import (  # noqa: E402
    UpstreamRefusal,
    clear_markers,
    require_upstream,
    write_json_atomic,
    write_jsonl_atomic,
    write_marker,
)
from pipeline.stage3_proposals.proposals import (  # noqa: E402
    build_caption,
    load_taxonomy,
    pairwise_iou,
)

STAGE = "stage3_merge"
STAGE_SPEC = "dhakascenes-pilot/stage3_merge/v1"

KEEP_BOTH = "keep_both"
SUPPRESS_ARM_A = "suppress_arm_a"

# The vocabulary-authority table (docs/RUNNING.md, verbatim). Keyed by arm A
# PHRASE. Absent phrase = keep both, counted as out-of-table so an unforeseen
# contest is visible rather than silently resolved.
ARBITRATION: dict[str, str] = {
    "a car": SUPPRESS_ARM_A,          # COCO has no word for the object
    "a truck": SUPPRESS_ARM_A,
    "a bus": SUPPRESS_ARM_A,
    "a motorcycle": SUPPRESS_ARM_A,   # three-wheeler forced onto a two-wheeler label
    "a bicycle": SUPPRESS_ARM_A,
    "a pedestrian": KEEP_BOTH,        # the puller/rider is a SEPARATE object
}

# What arm B is allowed to put in a row. Anything else is a leaked scaffolding
# class: the ship filter failed, and merging would poison arm A's turf.
ARM_B_PHRASES = ("a rickshaw", "an auto rickshaw")

# Stage 3b's per-box parallel arrays (track2d.py:rewrite_row), extended — never
# rebuilt — when the arm A tree is a stage3b_track2d tree. Values are the fill
# for an appended arm B box: not tracked, provenance arm_b, zero hops.
STAGE3B_EXTENSIONS: dict[str, object] = {
    "track_ids": None,
    "box_sources": "arm_b",
    "n_propagated_hops": 0,
    "refined": False,
    "boxes_xyxy_px_original": None,
}

# Row keys that must agree between the two arms for the rows to describe the
# same camera of the same keyframe of the same substrate.
FRAME_IDENTITY_KEYS = (
    "keyframe_token", "scene_token", "channel", "sample_data_token",
    "image_path", "image_size_px",
)


class MergeContractError(RuntimeError):
    """A pair of rows (or trees) that cannot honestly be merged."""


def merge_rows(row_a: dict, row_b: dict, *, caption, taxonomy, iou_threshold: float) -> dict:
    """One arm A row + its arm B counterpart -> one merged row.

    Copies, never recomputes: every surviving arm A value rides through
    byte-identical (the C27 rule). Arm B boxes are appended after the surviving
    arm A boxes; `suppressed_by` indices in the ledger point into the MERGED
    arrays.
    """
    for key in FRAME_IDENTITY_KEYS:
        if row_a.get(key) != row_b.get(key):
            raise MergeContractError(
                f"row identity mismatch on {key!r}: arm A {row_a.get(key)!r} vs "
                f"arm B {row_b.get(key)!r} — these rows do not describe the same frame"
            )
    leaked = sorted({n for n in row_b["class_names"] if n not in ARM_B_PHRASES})
    if leaked:
        raise MergeContractError(
            f"arm B row {row_b['sample_data_token']} emits {leaked}: the ship filter "
            f"leaked a non-shipped class; refusing to arbitrate on arm A's own turf"
        )

    boxes_a = [list(b) for b in row_a["boxes_xyxy_px"]]
    boxes_b = [list(b) for b in row_b["boxes_xyxy_px"]]
    n_a, n_b = len(boxes_a), len(boxes_b)

    n_kept_both = 0
    n_out_of_table = 0
    suppressed: dict[int, tuple[int, float]] = {}  # arm A index -> (arm B index, IoU)
    if n_a and n_b:
        iou = pairwise_iou(
            np.asarray(boxes_a, dtype=np.float32), np.asarray(boxes_b, dtype=np.float32)
        )
        for i in range(n_a):
            j = int(np.argmax(iou[i]))
            best = float(iou[i, j])
            if best <= iou_threshold:
                continue
            action = ARBITRATION.get(row_a["class_names"][i])
            if action == SUPPRESS_ARM_A:
                suppressed[i] = (j, best)
            elif action == KEEP_BOTH:
                n_kept_both += 1
            else:
                n_out_of_table += 1

    keep_a = [i for i in range(n_a) if i not in suppressed]

    def take(seq, idxs):
        return [seq[i] for i in idxs]

    span_of = {p: list(s) for p, s in zip(caption.phrases, caption.phrase_char_spans)}
    p2c = taxonomy.phrase_to_categories

    merged = dict(row_a)  # shallow: every list we touch is rebuilt below
    merged["boxes_xyxy_px"] = take(boxes_a, keep_a) + boxes_b
    merged["scores"] = take(list(row_a["scores"]), keep_a) + list(row_b["scores"])
    merged["class_names"] = take(list(row_a["class_names"]), keep_a) + list(row_b["class_names"])
    merged["nuscenes_categories"] = (
        take([list(c) for c in row_a["nuscenes_categories"]], keep_a)
        + [list(p2c[n]) for n in row_b["class_names"]]
    )
    merged["phrase_char_spans"] = (
        take([list(s) for s in row_a["phrase_char_spans"]], keep_a)
        + [span_of[n] for n in row_b["class_names"]]
    )
    merged["n_proposals"] = len(merged["boxes_xyxy_px"])
    merged["proposal_arm"] = ["arm_a"] * len(keep_a) + ["arm_b"] * n_b
    for key, fill in STAGE3B_EXTENSIONS.items():
        if key in row_a:
            merged[key] = take(list(row_a[key]), keep_a) + [fill] * n_b

    # The class space widened: the prompt block must name the caption these
    # rows are actually scored against. Arm A spans stay valid because the v2
    # caption is a byte prefix of the v3 caption (asserted by the driver).
    merged["prompt"] = {
        **row_a["prompt"],
        "caption_sha256": caption.sha256,
        "taxonomy_sha256": taxonomy.sha256,
    }
    merged["merge"] = {
        "spec": STAGE_SPEC,
        "iou_threshold": float(iou_threshold),
        "n_arm_a_in": n_a,
        "n_arm_b_in": n_b,
        "n_suppressed_arm_a": len(suppressed),
        "n_kept_both": n_kept_both,
        "n_overlap_out_of_table": n_out_of_table,
        "suppressed_arm_a": [
            {
                "index_in_arm_a": i,
                "box_xyxy_px": boxes_a[i],
                "score": row_a["scores"][i],
                "class_name": row_a["class_names"][i],
                "suppressed_by": len(keep_a) + j,   # index in MERGED arrays
                "iou": round(best, 4),
            }
            for i, (j, best) in sorted(suppressed.items())
        ],
    }
    return merged
```

- [ ] **Step 4: Run the core tests, verify pass**

Run: `$PY -m pytest tests/test_stage3_merge.py -v`
Expected: 11 passed. (Driver tests arrive in Task 6.)

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage3_merge/ tests/test_stage3_merge.py
git commit -m "Stage 3 merge core: vocabulary-authority arbitration, ledgered suppression, additive schema"
```

---

### Task 6: Merge driver — trees, manifest, marker, CLI

**Files:**
- Modify: `pipeline/stage3_merge/merge.py` (append the driver half)
- Test: `tests/test_stage3_merge.py` (append `TestDriver`)

**Interfaces:**
- Consumes: two Stage-3-shaped trees (`scenes/<name>/proposals.jsonl` + `run_manifest.json` + marker) — arm A may be `stage3_proposals` OR `stage3b_track2d` output; arm B is always plain `stage3_proposals` output.
- Produces: `stage3_merged/scenes/<name>/proposals.jsonl`, `run_manifest.json` whose `prompt.caption_sha256`, `upstream.metadata_fingerprint`, `upstream.fingerprint_spec` and `class_map` blocks satisfy Stage 4's reader (`masks.py:2186-2196`) and C25's (`class_space.py:_from_manifest`: `provider`, `class_map.phrases_in_use`, `class_map.unreachable_phrases`); a three-state marker via `write_marker`.
- Exit codes: 0 clean, 1 degraded (either upstream degraded, accepted), 2 refusal — the wrapper's contract.

- [ ] **Step 1: Append the failing driver tests**

Append to `tests/test_stage3_merge.py`:

```python
from pipeline.common.manifest import write_json_atomic, write_jsonl_atomic, write_marker  # noqa: E402
from pipeline.stage3_merge import merge as m3  # noqa: E402

FP = "fp-test-0001"


def _tree(root, rows_by_scene, *, caption_text, phrases_in_use, degraded=False):
    os.makedirs(root, exist_ok=True)
    write_json_atomic(os.path.join(root, "run_manifest.json"), {
        "spec": "dhakascenes-pilot/stage3_proposals/v1",
        "provider": "yolo11",
        "prompt": {"caption": caption_text, "caption_sha256": "irrelevant"},
        "upstream": {"metadata_fingerprint": FP, "fingerprint_spec": "spec/v1"},
        "checkpoint": {"model_id": "m", "revision": "r", "sha256": "s"},
        "class_map": {"path": "p", "sha256": "s", "phrases_in_use": list(phrases_in_use),
                      "unreachable_phrases": []},
    })
    for scene, rows in rows_by_scene.items():
        write_jsonl_atomic(os.path.join(root, "scenes", scene, "proposals.jsonl"), rows)
    write_marker(root, FP, degraded=degraded,
                 causes=("scene-x: flagged",) if degraded else ())


class TestDriver:
    def _dirs(self, tmp_path, caption, *, b_caption_text=None, degraded_a=False):
        a_rows = [_row(caption, ["a car"], [BOX]),
                  _row(caption, ["a pedestrian"], [BOX_FAR], keyframe_token="kf1",
                       sample_data_token="sd1")]
        b_rows = [_row(caption, ["an auto rickshaw"], [[102.0, 101.0, 199.0, 198.0]]),
                  _row(caption, [], [], keyframe_token="kf1", sample_data_token="sd1")]
        a_dir, b_dir = str(tmp_path / "a"), str(tmp_path / "b")
        # arm A ran under the v2 caption: a byte prefix of the v3 caption
        v2_text = caption.text[: caption.text.index(" a rickshaw.")]
        _tree(a_dir, {"scene-0001": a_rows}, caption_text=v2_text,
              phrases_in_use=["a car", "a pedestrian"], degraded=degraded_a)
        _tree(b_dir, {"scene-0001": b_rows},
              caption_text=b_caption_text or caption.text,
              phrases_in_use=list(ARM_B_PHRASES))
        return a_dir, b_dir, str(tmp_path / "out")

    def test_clean_merge_writes_rows_manifest_marker(self, tmp_path, caption, taxonomy):
        a, b, out = self._dirs(tmp_path, caption)
        rc = m3.main(["--arm-a-dir", a, "--arm-b-dir", b, "--out-dir", out,
                      "--taxonomy", DHAKA])
        assert rc == 0
        assert os.path.isfile(os.path.join(out, "_SUCCESS"))
        with open(os.path.join(out, "scenes", "scene-0001", "proposals.jsonl")) as fh:
            rows = [json.loads(line) for line in fh]
        assert [r["merge"]["n_suppressed_arm_a"] for r in rows] == [1, 0]
        with open(os.path.join(out, "run_manifest.json")) as fh:
            man = json.load(fh)
        assert man["spec"] == m3.STAGE_SPEC
        assert man["prompt"]["caption_sha256"] == caption.sha256
        assert man["upstream"]["metadata_fingerprint"] == FP
        assert set(ARM_B_PHRASES) <= set(man["class_map"]["phrases_in_use"])
        assert "a car" in man["class_map"]["phrases_in_use"]
        # C25: the two arm B phrases are now reachable; barrier/cone etc. are not
        assert "a rickshaw" not in man["class_map"]["unreachable_phrases"]
        assert "a road barrier" in man["class_map"]["unreachable_phrases"]

    def test_degraded_upstream_needs_flag_and_degrades_output(self, tmp_path, caption, taxonomy):
        a, b, out = self._dirs(tmp_path, caption, degraded_a=True)
        assert m3.main(["--arm-a-dir", a, "--arm-b-dir", b, "--out-dir", out,
                        "--taxonomy", DHAKA]) == 2
        rc = m3.main(["--arm-a-dir", a, "--arm-b-dir", b, "--out-dir", out,
                      "--taxonomy", DHAKA, "--accept-degraded-upstream"])
        assert rc == 1
        assert os.path.isfile(os.path.join(out, "_SUCCESS.degraded"))

    def test_refuses_non_prefix_arm_a_caption(self, tmp_path, caption, taxonomy):
        a, b, out = self._dirs(tmp_path, caption)
        with open(os.path.join(a, "run_manifest.json")) as fh:
            man = json.load(fh)
        man["prompt"]["caption"] = "a completely different caption."
        write_json_atomic(os.path.join(a, "run_manifest.json"), man)
        assert m3.main(["--arm-a-dir", a, "--arm-b-dir", b, "--out-dir", out,
                        "--taxonomy", DHAKA]) == 2

    def test_refuses_scene_set_mismatch(self, tmp_path, caption, taxonomy):
        a, b, out = self._dirs(tmp_path, caption)
        os.rename(os.path.join(b, "scenes", "scene-0001"),
                  os.path.join(b, "scenes", "scene-0002"))
        assert m3.main(["--arm-a-dir", a, "--arm-b-dir", b, "--out-dir", out,
                        "--taxonomy", DHAKA]) == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `$PY -m pytest tests/test_stage3_merge.py::TestDriver -v`
Expected: FAIL — `module 'pipeline.stage3_merge.merge' has no attribute 'main'`.

- [ ] **Step 3: Append the driver to `merge.py`**

```python
def _read_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run(
    arm_a_dir: str,
    arm_b_dir: str,
    out_dir: str,
    taxonomy_path: str,
    *,
    iou_threshold: float = 0.5,
    accept_degraded: bool = False,
) -> int:
    taxonomy = load_taxonomy(taxonomy_path)
    caption = build_caption(taxonomy.phrases)

    man_a, marker_a = require_upstream(
        arm_a_dir, stage_name="Stage 3 (arm A)",
        module_hint="pipeline.stage3_proposals.proposals",
        accept_degraded=accept_degraded,
    )
    # Cross-bound to arm A's substrate fingerprint: the two trees must descend
    # from the same Stage 1 output, or the row pairing compares different worlds.
    man_b, marker_b = require_upstream(
        arm_b_dir, stage_name="Stage 3 (arm B)",
        module_hint="pipeline.stage3_proposals.proposals",
        current_fingerprint=marker_a.fingerprint,
        accept_degraded=accept_degraded,
    )

    cap_a = str(man_a["prompt"]["caption"])
    if not caption.text.startswith(cap_a):
        raise UpstreamRefusal(
            f"arm A's caption is not a prefix of {taxonomy_path}'s caption: arm A spans "
            "would be invalid under the merged class space. The v3 taxonomy must extend "
            "the arm A taxonomy by APPENDING phrases only"
        )
    cap_b = str(man_b["prompt"]["caption"])
    if cap_b != caption.text:
        raise UpstreamRefusal(
            f"arm B ran under a different caption than {taxonomy_path}: re-run arm B "
            "with --taxonomy pointing at the same file the merge uses"
        )
    b_in_use = tuple((man_b.get("class_map") or {}).get("phrases_in_use") or ())
    stray = sorted(set(b_in_use) - set(ARM_B_PHRASES))
    if stray:
        raise UpstreamRefusal(
            f"arm B's class map puts {stray} in use; arm B may only ship {ARM_B_PHRASES}"
        )

    root_a = os.path.join(arm_a_dir, "scenes")
    root_b = os.path.join(arm_b_dir, "scenes")
    scenes_a = sorted(os.listdir(root_a)) if os.path.isdir(root_a) else []
    scenes_b = sorted(os.listdir(root_b)) if os.path.isdir(root_b) else []
    if scenes_a != scenes_b or not scenes_a:
        raise UpstreamRefusal(
            f"scene sets differ (arm A {scenes_a} vs arm B {scenes_b}): the merge "
            "pairs rows frame-by-frame and cannot invent an absent arm"
        )

    clear_markers(out_dir)
    totals = {"n_rows": 0, "n_arm_a_in": 0, "n_arm_b_in": 0, "n_suppressed_arm_a": 0,
              "n_kept_both": 0, "n_overlap_out_of_table": 0, "n_out": 0}
    per_scene: dict[str, dict] = {}
    for scene in scenes_a:
        rows_a = _read_rows(os.path.join(root_a, scene, "proposals.jsonl"))
        rows_b = _read_rows(os.path.join(root_b, scene, "proposals.jsonl"))
        index_b = {(r["keyframe_token"], r["channel"]): r for r in rows_b}
        if len(index_b) != len(rows_b):
            raise MergeContractError(f"{scene}: duplicate (keyframe, channel) rows in arm B")
        merged_rows = []
        seen = set()
        for row_a in rows_a:  # arm A's row order IS the output row order (C27)
            key = (row_a["keyframe_token"], row_a["channel"])
            if key not in index_b:
                raise MergeContractError(f"{scene}: arm B has no row for {key}")
            seen.add(key)
            merged = merge_rows(row_a, index_b[key], caption=caption,
                                taxonomy=taxonomy, iou_threshold=iou_threshold)
            led = merged["merge"]
            totals["n_rows"] += 1
            for k in ("n_arm_a_in", "n_arm_b_in", "n_suppressed_arm_a",
                      "n_kept_both", "n_overlap_out_of_table"):
                totals[k] += led[k]
            totals["n_out"] += merged["n_proposals"]
            merged_rows.append(merged)
        extra = set(index_b) - seen
        if extra:
            raise MergeContractError(f"{scene}: arm B rows with no arm A counterpart: {sorted(extra)}")
        write_jsonl_atomic(os.path.join(out_dir, "scenes", scene, "proposals.jsonl"), merged_rows)
        per_scene[scene] = {"n_rows": len(merged_rows)}

    in_use_a = tuple((man_a.get("class_map") or {}).get("phrases_in_use") or ())
    union = [p for p in caption.phrases if p in set(in_use_a) | set(b_in_use)]
    manifest = {
        "spec": STAGE_SPEC,
        "stage": STAGE,
        "provider": "yolo11_two_arm_merge",
        "score_semantics": "yolo_class_confidence (both arms; scores are never compared across arms)",
        "upstream": {
            "metadata_fingerprint": marker_a.fingerprint,
            "fingerprint_spec": man_a["upstream"]["fingerprint_spec"],
            "arm_a": {"dir": os.path.realpath(arm_a_dir), "spec": man_a.get("spec"),
                      "checkpoint": man_a.get("checkpoint"), "degraded": marker_a.degraded,
                      "degraded_causes": list(marker_a.causes)},
            "arm_b": {"dir": os.path.realpath(arm_b_dir), "spec": man_b.get("spec"),
                      "checkpoint": man_b.get("checkpoint"), "degraded": marker_b.degraded,
                      "degraded_causes": list(marker_b.causes)},
            "accepted_degraded_upstream": accept_degraded,
        },
        "taxonomy": taxonomy.as_dict(),
        "prompt": {
            "caption": caption.text,
            "caption_sha256": caption.sha256,
            "caption_is_input": False,
            "phrases": list(caption.phrases),
        },
        "class_map": {
            # C25 reads this block: the merged run's reachable set is the UNION.
            "path": f"merge({(man_a.get('class_map') or {}).get('path')}, "
                    f"{(man_b.get('class_map') or {}).get('path')})",
            "sha256": "",
            "n_mapped": len(union),
            "phrases_in_use": union,
            "unreachable_phrases": [p for p in caption.phrases if p not in set(union)],
            "arm_a_class_map": man_a.get("class_map"),
            "arm_b_class_map": man_b.get("class_map"),
        },
        "arbitration": {
            "table": dict(ARBITRATION),
            "iou_threshold": float(iou_threshold),
            "provenance": "docs/RUNNING.md two-arm design 2026-08-26; DECISIONS C28. "
                          "Vocabulary authority, never score (C21). iou_threshold UNVALIDATED.",
        },
        "totals": totals,
        "scenes": per_scene,
    }
    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)

    degraded = marker_a.degraded or marker_b.degraded
    causes = tuple(f"arm_a: {c}" for c in marker_a.causes) + tuple(
        f"arm_b: {c}" for c in marker_b.causes)
    write_marker(out_dir, marker_a.fingerprint, degraded=degraded, causes=causes)
    print(f"stage3_merge: {totals['n_rows']} rows, {totals['n_out']} boxes out, "
          f"{totals['n_suppressed_arm_a']} arm A suppressed, "
          f"{totals['n_kept_both']} kept-both, "
          f"{totals['n_overlap_out_of_table']} out-of-table overlaps")
    return 1 if degraded else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a-dir", required=True,
                        help="stage3_proposals or stage3b_track2d tree (arm A)")
    parser.add_argument("--arm-b-dir", required=True,
                        help="arm B stage3 tree (stage3_finetuned)")
    parser.add_argument("--out-dir", required=True, help="stage3_merged tree to write")
    parser.add_argument("--taxonomy", default="configs/taxonomy_pilot_dhaka.yaml")
    parser.add_argument("--iou-threshold", type=float, default=0.5,
                        help="overlap that makes a pair a contest (recorded; unvalidated)")
    parser.add_argument("--accept-degraded-upstream", action="store_true")
    args = parser.parse_args(argv)
    try:
        return run(
            args.arm_a_dir, args.arm_b_dir, args.out_dir, args.taxonomy,
            iou_threshold=args.iou_threshold,
            accept_degraded=args.accept_degraded_upstream,
        )
    except (UpstreamRefusal, MergeContractError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run the full merge suite, verify pass**

Run: `$PY -m pytest tests/test_stage3_merge.py -v`
Expected: 15 passed.

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage3_merge/merge.py tests/test_stage3_merge.py
git commit -m "Stage 3 merge driver: paired trees, union class map for C25, three-state marker"
```

---

### Task 7: GPU integration trial — arm B proposals + merge on one scene

**Files:** none created in the repo (outputs under `/tmp/`); this task validates Tasks 1-6 against the real substrate and checkpoint.

**Interfaces:**
- Consumes: `local_yolox_build/artifacts/yolo11x-rsud20k-armb.pt` (Task 4), the two configs (Tasks 1-2), the existing `work_root` Stage 1 output.
- Produces: evidence. Follow RUNNING.md's trial-run pattern (scratch dirs, real tree untouched).

- [ ] **Step 1: Arm B Stage 3 trial on scene-0061**

```bash
cd /home/mt/Zami/Annotation_pipeline
PY=/home/mt/miniconda3/envs/ano_pipe/bin/python
export DHAKASCENES_VRAM_CAP_MIB=22000
$PY pipeline/stage3_proposals/proposals.py --scenes scene-0061 \
    --out-dir /tmp/trial_stage3_armb \
    --model-id local_yolox_build/artifacts/yolo11x-rsud20k-armb.pt \
    --revision "$(python3 -c "import json;print(json.load(open('local_yolox_build/artifacts/armb_provenance.json'))['sha256'][:12])")" \
    --taxonomy configs/taxonomy_pilot_dhaka.yaml \
    --class-map configs/rsud20k_to_phrase_dhaka.yaml \
    --accept-degraded-upstream
```

Expected: clean exit; `/tmp/trial_stage3_armb/run_manifest.json` has `class_map.n_mapped: 2`, `class_map.unreachable_phrases` = the 10 v2 phrases, `provider: yolo11`. Every row's `class_names` ⊆ {`a rickshaw`, `an auto rickshaw`}. Box counts will be small/zero — nuScenes Boston streets have few rickshaws; **that is plumbing evidence, not model evidence** (standing banner). Verify with:

```bash
$PY - <<'EOF'
import json
names = set()
for line in open('/tmp/trial_stage3_armb/scenes/scene-0061/proposals.jsonl'):
    names.update(json.loads(line)["class_names"])
print("classes emitted:", sorted(names) or "(none — acceptable on this substrate)")
assert names <= {"a rickshaw", "an auto rickshaw"}, names
EOF
```

- [ ] **Step 2: Arm A trial + merge trial**

```bash
$PY pipeline/stage3_proposals/proposals.py --scenes scene-0061 \
    --out-dir /tmp/trial_stage3_arma --taxonomy configs/taxonomy_pilot_dhaka.yaml \
    --model-id "${YOLO11_CHECKPOINT:-/home/mt/dhakascenes/cache/checkpoints/yolo11x.pt}" \
    --revision v8.3.0 --accept-degraded-upstream
$PY pipeline/stage3_merge/merge.py \
    --arm-a-dir /tmp/trial_stage3_arma --arm-b-dir /tmp/trial_stage3_armb \
    --out-dir /tmp/trial_stage3_merged --taxonomy configs/taxonomy_pilot_dhaka.yaml
```

(The arm A trial runs under the v3 taxonomy — legal for a scratch run because a closed-vocabulary provider never reads the caption; the byte-prefix rule also permits merging the REAL v2-captioned `stage3_proposals` tree.)
Expected: `stage3_merge: … rows` summary; `_SUCCESS` present; merged row counts = arm A trial's row count.

- [ ] **Step 3: Stage 4 smoke over the merged tree**

```bash
$PY pipeline/stage4_masks/masks.py --scenes scene-0061 \
    --stage3-dir /tmp/trial_stage3_merged --out-dir /tmp/trial_stage4_merged \
    --revision 3c879f39826c281e95690f02c7821c4de09afae7 --accept-degraded-upstream
```

Expected: Stage 4 consumes the merged tree with no code change (the C27 claim, now demonstrated); masks.jsonl rows carry one mask per merged box.

- [ ] **Step 4: Record the evidence**

Save the three manifests' totals into the Task 9 doc update (numbers go in RUNNING.md's status paragraph). Clean up `/tmp/trial_*` only after Task 9 quotes them.

---

### Task 8: Wrapper wiring — opt-in `3f` and `3m` steps

**Files:**
- Modify: `scripts/run_stages.sh`

**Interfaces:**
- Consumes: the artifact path (Task 4), the configs (Tasks 1-2), `pipeline/stage3_merge/merge.py` (Task 6).
- Produces: `scripts/run_stages.sh 3 3b 3f 3m 4 …` runs the full two-arm chain; without `3f`/`3m` nothing changes (arm A remains the default pipeline — the merge stays opt-in until its thresholds are tuned, mirroring how `3b` shipped under C27).

- [ ] **Step 1: Add env defaults**

Next to the `PROPOSAL_MODEL_ID` block (`run_stages.sh:126-137`), add:

```bash
# Arm B (opt-in steps 3f/3m; docs/RUNNING.md two-arm design, DECISIONS C28).
ARMB_MODEL_ID="${ARMB_MODEL_ID:-local_yolox_build/artifacts/yolo11x-rsud20k-armb.pt}"
ARMB_REVISION="${ARMB_REVISION:-armb-r1280-4}"
ARMB_TAXONOMY="${ARMB_TAXONOMY:-configs/taxonomy_pilot_dhaka.yaml}"
ARMB_CLASS_MAP="${ARMB_CLASS_MAP:-configs/rsud20k_to_phrase_dhaka.yaml}"
```

- [ ] **Step 2: Register the steps**

- `run_stages.sh:220`: `OPT_IN_STEPS=(3b)` → `OPT_IN_STEPS=(3b 3f 3m)`
- In the argument `case` (`run_stages.sh:~252`): `0|1|3|3b|4|…` → `0|1|3|3b|3f|3m|4|…`
- In the `--clean-slate` removal list (`run_stages.sh:~336`) and the final marker-summary loop (`run_stages.sh:776`): add `stage3_finetuned stage3_merged` after `stage3b_track2d`.

- [ ] **Step 3: Add the step bodies**

After the `3b)` case in the steps loop:

```bash
    3f) acc
        # Arm B proposals (opt-in). Same driver as Stage 3, different weights,
        # class map and (superset) taxonomy; its own tree, arm A untouched.
        run_step "STAGE 3f (proposal_2d arm B: $ARMB_MODEL_ID)" "$WORK_ROOT/stage3_finetuned" \
          "$PY" pipeline/stage3_proposals/proposals.py \
            --model-id "$ARMB_MODEL_ID" --revision "$ARMB_REVISION" \
            --taxonomy "$ARMB_TAXONOMY" --class-map "$ARMB_CLASS_MAP" \
            --out-dir "$WORK_ROOT/stage3_finetuned" \
            ${ACC[@]+"${ACC[@]}"} ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;

    3m) # The merge (opt-in). Arm A input is whatever Stage 4 would have read
        # (stage3b_track2d when fresh and covering, else stage3_proposals).
        select_stage3_dir_for_4
        acc
        run_step "STAGE 3m (merge: $(basename "$STAGE3_DIR_FOR_4") + stage3_finetuned)" "$WORK_ROOT/stage3_merged" \
          "$PY" pipeline/stage3_merge/merge.py \
            --arm-a-dir "$STAGE3_DIR_FOR_4" \
            --arm-b-dir "$WORK_ROOT/stage3_finetuned" \
            --out-dir "$WORK_ROOT/stage3_merged" \
            --taxonomy "$ARMB_TAXONOMY" \
            ${ACC[@]+"${ACC[@]}"} || break
        ;;
```

- [ ] **Step 4: Teach `select_stage3_dir_for_4` about the merged tree**

At the end of `select_stage3_dir_for_4()` (`run_stages.sh:431-459`), after the existing `STAGE3_DIR_FOR_4="$s3b"` resolution, append (before the final degraded check so the C16 arming still sees the chosen dir):

```bash
  # stage3_merged outranks both arms when it is complete and NEWER than the
  # arm A dir just chosen — a stale merge over a fresh arm A would resurrect
  # boxes the newer run no longer proposes. Same freshness rule as s3b.
  local s3m="$WORK_ROOT/stage3_merged"
  if [ "$(marker_state "$s3m")" != none ] \
     && [ "$s3m/run_manifest.json" -nt "$STAGE3_DIR_FOR_4/run_manifest.json" ]; then
    STAGE3_DIR_FOR_4="$s3m"
  fi
```

- [ ] **Step 5: Syntax check + behavior spot-checks**

```bash
bash -n scripts/run_stages.sh
scripts/run_stages.sh --help | head -30          # help still renders
scripts/run_stages.sh bogus-step 2>&1 | grep 3f  # error message lists the new opt-ins
```

Expected: `bash -n` silent; error message names `3f 3m` among opt-in steps.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_stages.sh
git commit -m "run_stages: opt-in 3f/3m steps; Stage 4 prefers a fresh stage3_merged"
```

---

### Task 9: Documentation + decision record

**Files:**
- Modify: `docs/RUNNING.md` (the "Stage 3 arm A / arm B" section), `docs/DECISIONS.md` (append C28), `docs/CVAT_GUIDE.md` (one note), `README.md` (stage table row)

**Interfaces:** consumes Task 7's measured numbers and Task 4's artifact sha256.

- [ ] **Step 1: RUNNING.md status flip**

In the section header, change `(2026-08-26, IN BUILD)` → `(2026-08-26; built 2026-08-27)`, and replace the paragraph beginning `**Status: arm B is being trained; the merge step does not exist yet.**` with:

```markdown
**Status: built, opt-in.** `scripts/run_stages.sh 3 3b 3f 3m 4 …` runs the
two-arm chain: `3f` writes `stage3_finetuned/` (the same `proposals.py`
driver, arm B weights + `configs/rsud20k_to_phrase_dhaka.yaml` +
`configs/taxonomy_pilot_dhaka.yaml`), `3m` writes `stage3_merged/`
(`pipeline/stage3_merge/merge.py`), and Stage 4 reads the merged tree when it
is fresh (`select_stage3_dir_for_4`). Without `3f`/`3m` nothing changes: arm A
alone remains the default until the two arm B thresholds are tuned (they
currently ride the 0.40 default — the same standing caveat as every other
per-class threshold). The `rsud20k_to_phrase_dhaka.yaml` layer sketched below
now exists; the trial numbers on this substrate are <TASK-7 NUMBERS HERE>.
```

Also update the design-sketch line `` `rsud20k_to_phrase_dhaka.yaml` ← NOT WRITTEN YET `` → `` ← configs/rsud20k_to_phrase_dhaka.yaml `` and the arm B table row `Status … in build` → `built; artifact local_yolox_build/artifacts/yolo11x-rsud20k-armb.pt (sha256 in artifacts/armb_provenance.json)`.

- [ ] **Step 2: DECISIONS.md C28**

Append, following the house format of C23-C27 (heading, date, decision, mechanism, cost):

```markdown
### C28 — Stage 3 grows a second arm: RSUD20K fine-tune, vocabulary-authority merge (2026-08-27)

**Decision.** Arm B (`yolo11x-rsud20k-armb.pt`, YOLO11x fine-tuned on RSUD20K,
5 classes trained / 2 shipped) runs through the unmodified Stage 3 driver under
a superset taxonomy (`configs/taxonomy_pilot_dhaka.yaml`, v3: the C21 10-phrase
space + `a rickshaw` + `an auto rickshaw`, appended so the v2 caption stays a
byte prefix and arm A spans survive verbatim). A new opt-in merge
(`pipeline/stage3_merge/merge.py`) pairs the two trees row-by-row and
arbitrates overlaps by a class-pair table — car/truck/bus/motorcycle/bicycle
lose to an overlapping arm B claim because COCO has no word for the object;
pedestrian keeps both per the rider convention; scores never arbitrate (C21).
Suppressed arm A boxes move whole into the row's `merge.suppressed_arm_a`
ledger: retained and auditable, but out of the arrays, because Stage 4 masks
every array box in order and Stage 5 lifts every mask — an in-array
"suppressed" box would propagate a label the merge just ruled impossible.

**Costs, recorded.** (1) Arm B and every label it produces inherit RSUD20K's
CC BY-NC 4.0 — research/non-commercial only; this constrains any DhakaScenes
release that includes arm-B-descended labels. (2) The arm B thresholds and the
merge's iou_threshold=0.5 are UNVALIDATED. (3) Arm B's training labels are
80.7% machine-generated (third-generation chain: human → YOLOv6-M6 → YOLO11x);
"measured against human ground truth" remains a wrong sentence for this arm.
(4) On the nuScenes pilot substrate arm B mostly proves plumbing: rickshaw
counts near zero are expected, not evidence.
```

- [ ] **Step 3: CVAT note + README row**

In `docs/CVAT_GUIDE.md`, add one line to the C27 label-schema warning: publishing a merged run's export adds two labels (`a rickshaw`, `an auto rickshaw`); a pre-existing project cannot accept them — delete and republish into a fresh project, as with attribute additions. In `README.md`'s stage table (§ the pipeline map), extend the Stage 3 row with `+ 3f/3m (arm B + merge, opt-in, C28)`.

- [ ] **Step 4: Full test sweep + commit**

```bash
$PY -m pytest tests/ -v
git add docs/RUNNING.md docs/DECISIONS.md docs/CVAT_GUIDE.md README.md
git commit -m "C28: two-arm Stage 3 documented; licence inheritance recorded"
```

---

## Self-Review (performed while writing)

- **Spec coverage:** two-arm table → Tasks 4/7; three-layer spelling diagram → Tasks 1/3 (`rsud20k_to_phrase_dhaka.yaml` is Task 2); merge dirs + Stage-3b-trick schema → Tasks 5/6; arbitration table incl. person rule → Task 5; suppression retention → Task 5 decision 1; "cannot settle, must not guess" → contested pairs live in the ledger and the fallback stays merged-pair pre-labelling (no Stage 3 guessing); provenance/licence paragraphs → Task 9; C25 reachability under the union → Task 6 manifest + driver test. Gap deliberately left: per-phrase threshold tuning (S1's business, flagged in three places, not silently absorbed).
- **Placeholder scan:** one intentional `<TASK-7 NUMBERS HERE>` in Task 9 — filled from Task 7's measured output at execution time, stated as such.
- **Type consistency:** `merge_rows` signature, `ARBITRATION`/`ARM_B_PHRASES`/`STAGE_SPEC` names, artifact filename, and the 12-phrase order are used identically in Tasks 1, 2, 5, 6, 7, 8; driver exit codes match the wrapper's three-state table.
