# Approach A — Per-Mask Stereo Boxing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce 3D boxes for every SAM mask on the two ZED cameras from the ZED's own stereo depth (one mask → one box, no clustering), run it on `chunk_0010`, and show the result in a read-only 3D viewer.

**Architecture:** Stage 1 learns to ingest the export's `ZED_WORLD` channel (global-frame stereo points) into the ego-frame single sweep as rings 100/101. A new drop-in stage `stage6_stereo_box` reads Stage 5's mask→point ownership and emits Stage 6's exact `boxes.jsonl` row, so Stages 7/8/9 run unchanged. A new coverage config `R3` names the two ZED frusta. A Three.js viewer renders cloud + boxes + both ZED images with projected boxes.

**Tech Stack:** Python 3.10 (`/home/saif/miniconda3/envs/ano_pipe/bin/python`), numpy 1.26, scipy 1.14 (cKDTree), PyYAML, pytest; Three.js 0.160 via jsdelivr importmap (viewer is served locally, no CSP allowlist applies).

**Spec:** `docs/superpowers/specs/2026-09-12-stereo-box-a-design.md` — read it first; every task below argues from it.

## Global Constraints

- Interpreter: ALWAYS `/home/saif/miniconda3/envs/ano_pipe/bin/python` with `PYTHONNOUSERSITE=1`. Alias in every shell below: `PY=/home/saif/miniconda3/envs/ano_pipe/bin/python`.
- Run every command from the repo root `/home/saif/pipeline/Annotation_pipeline` (the branch worktree in Task 0 replaces this path — use the worktree path the orchestrator gives you).
- Tests run with the DEFAULT substrate: `PYTHONNOUSERSITE=1 $PY -m pytest tests -q -p no:cacheprovider`. Baseline: **616 passed, 1 failed** (`tests/test_export_release.py::test_dbench_ingest_validate_passes` — environmental, ignore it; it must stay the ONLY failure).
- Pipeline runs use `.env` (already written): `DHAKASCENES_PATHS_CONFIG=configs/paths_zami_20260911.yaml`, `DHAKASCENES_SUBSTRATE=dhaka6`. Load it with `set -a && . ./.env && set +a`.
- Data: dataroot `/home/saif/dhaka-export-pipeline-20260911/export/dhaka_20260911_141259/full`, version `v1.0-dhaka-fixed2`; work root `/mnt/hdd/dhakascenes/work_zami/20260911_zed`. Stage 0 has already run there (11/11 usable, DEGRADED on the hardcoded partition — expected).
- Scene for every run in this plan: `--scenes dhaka_20260911_141259_chunk_0010` (668 keyframes).
- Never write into the dataroot. Never run `--clean-slate`. Never publish to CVAT (`--no-cvat` always).
- Every new stage records provenance in its `run_manifest.json` and writes `_SUCCESS` / `_SUCCESS.degraded` via `pipeline.common.manifest` (three-state exit codes 0/1/2).
- Frames: ego frame = LiDAR frame (`calibrated_sensor` translation 0,0,0). Yaw about +z from +x. Box size order `[w, l, h]` with `w <= l` always (swap + `axis_swapped` if not).
- Commit after every task with the message given; end commit messages with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

## File Structure

| file | responsibility |
|---|---|
| `pipeline/stage1_ingestion/ingest.py` (modify) | ingest `ZED_WORLD` (global-frame) into the single sweep; `--stereo-stride`, `--coverage-config`, `--stereo-z-correction` CLI knobs |
| `pipeline/common/eval_region.py` (modify) | `R3` coverage config: full circle minus the two side wedges |
| `pipeline/common/schemas.py` (modify) | `R3` camera-subset consistency rule |
| `pipeline/stage5_lift/lift.py` (modify, 3 lines) | `region_for("R3")` |
| `scripts/author_priors_dhaka.py` (modify) | `--from-table` mode; operator rickshaw/CNG length 2.40 m |
| `scripts/spike_stereo_vs_lidar.py` (create) | ZED-vs-LiDAR agreement by range → range cap + z offset evidence |
| `configs/stereo_box.yaml` (create) | the stereo-box tunables with provenance (range cap, k_mad, min points) |
| `pipeline/stage6_stereo_box/__init__.py`, `stereo_box.py` (create) | the new stage |
| `scripts/run_stages.sh` (modify) | step `6s`, `boxes_dir()` preference, env knobs for Stage 1 |
| `scripts/view_boxes_3d.py` (create) + `viewer/index.html` (create) | read-only viewer: exporter + page |
| `scripts/eval_stereo_box.py` (create) | GT-free metrics incl. reprojection IoU → evidence doc |
| `tests/test_stage1_zed_world.py`, `tests/test_eval_region_r3.py`, `tests/test_author_priors_table.py`, `tests/test_stereo_box.py`, `tests/test_view_boxes_3d.py` (create) | tests |

---

### Task 0: Base commit and branch (orchestrator, not an agent)

**Files:** none new.

- [ ] **Step 1: Commit the provisioning fixes on `main`**

```bash
cd /home/saif/pipeline/Annotation_pipeline
git add requirements-torch.txt requirements.txt pipeline/common/schemas.py configs/paths_zami_20260911.yaml docs/superpowers/specs/2026-09-12-stereo-box-a-design.md docs/superpowers/plans/2026-09-12-stereo-box-a.md
git commit -m "Blackwell box provisioning: cu128 torch, ftfy/opencv pins, drop dt_ns==0 assertion; paths for the ZED-bearing 2026-09-11 export; approach A spec + plan

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

- [ ] **Step 2: Branch in a worktree**

```bash
git worktree add -b stereo-box-a /home/saif/pipeline/wt-stereo-box-a main
cp /home/saif/pipeline/Annotation_pipeline/.env /home/saif/pipeline/wt-stereo-box-a/.env
cd /home/saif/pipeline/wt-stereo-box-a && git lfs pull
```

Every task below runs in `/home/saif/pipeline/wt-stereo-box-a`.

---

### Task 1: Priors from the documented table

**Files:**
- Modify: `scripts/author_priors_dhaka.py`
- Test: `tests/test_author_priors_table.py`

**Interfaces:**
- Consumes: `pipeline.stage6_cluster.priors.load_priors(path) -> Priors`; `Priors.get(class_name) -> ClassPrior | None`; `ClassPrior.mu(axis) / .sigma(axis)`.
- Produces: `TABLE: dict[str, dict]` (phrase → `{category, w, l, h}`), `author_from_table(phrases, *, fingerprint, binding, authored_on) -> dict`, CLI flag `--from-table`. Output file `<out_root>/priors/priors_pilot_v0.json` loadable by `load_priors`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_author_priors_table.py
"""author_priors_dhaka.py --from-table: priors with no template (2026-09-12).

The nuScenes-derived template lived on the old box and is gone. The documented
population means (docs/Annotation_pipeline.md:141) plus the operator's rickshaw
and CNG lengths (2.40 m, stated 2026-09-12) are enough for Stage 6's epsilon and
Stage 8's dims. Unreachable phrases get a dims=None block, like the template's
trailer did, so load_priors accepts the file and Stage 6 refuses by name.
"""
from __future__ import annotations
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.stage6_cluster.priors import load_priors  # noqa: E402
from scripts.author_priors_dhaka import (  # noqa: E402
    LITERATURE, TABLE, TABLE_SOURCE, OPERATOR_SOURCE, author_from_table, main,
)

FP = "ab12" * 16
BINDING = {"dataroot_realpath": "/data/x", "version": "v1.0-dhaka-fixed2",
           "fingerprint_spec": "sha256-of-sorted-name-digest-manifest/v1"}
PHRASES = ["a car", "a pedestrian", "a road barrier", "a traffic cone", "a truck",
           "a motorcycle", "a bus", "a bicycle", "a construction vehicle", "a trailer",
           "a rickshaw", "an auto rickshaw"]


def test_operator_lengths_replace_literature():
    assert LITERATURE["a rickshaw"]["l"] == 2.40
    assert LITERATURE["an auto rickshaw"]["l"] == 2.40
    assert LITERATURE["a rickshaw"]["source"] == OPERATOR_SOURCE


def test_table_payload_loads_and_covers_every_phrase(tmp_path):
    payload = author_from_table(PHRASES, fingerprint=FP, binding=BINDING, authored_on="2026-09-12")
    out = tmp_path / "priors_pilot_v0.json"
    out.write_text(json.dumps(payload))
    priors = load_priors(str(out))
    assert priors.metadata_fingerprint == FP
    car = priors.get("a car")
    assert car is not None and abs(car.mu("l") - 4.63) < 1e-9 and abs(car.mu("w") - 1.93) < 1e-9
    rick = priors.get("a rickshaw")
    assert abs(rick.mu("l") - 2.40) < 1e-9
    assert payload["classes"]["a car"]["source"] == TABLE_SOURCE
    assert payload["classes"]["a rickshaw"]["source"] == OPERATOR_SOURCE
    # unreachable phrases are present with no dims, never silently absent
    for ph in ("a road barrier", "a traffic cone", "a construction vehicle", "a trailer"):
        assert ph in payload["classes"] and payload["classes"][ph]["dims"] is None
    eps, src = priors.eps_bev("a car", fallback_m=9.9)
    assert abs(eps - 0.6 * (1.93 ** 2 + 4.63 ** 2) ** 0.5) < 1e-6 and not src.startswith("config_fallback")
    assert payload["derived_from"]["scene_subset"] == "priors"
    assert "no scene" in payload["derived_from"]["subset_note"]


def test_cli_from_table_writes_file(tmp_path):
    out = tmp_path / "p.json"
    rc = main(["--from-table", "--out", str(out), "--fingerprint", FP,
               "--dataroot", "/data/x", "--version", "v1.0-dhaka-fixed2"])
    assert rc == 0 and out.is_file()
    load_priors(str(out))
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `PYTHONNOUSERSITE=1 $PY -m pytest tests/test_author_priors_table.py -q -p no:cacheprovider`
Expected: FAIL — `ImportError: cannot import name 'TABLE'`.

- [ ] **Step 3: Implement**

In `scripts/author_priors_dhaka.py`, after `SIGMA_FRACTION`:

```python
OPERATOR_SOURCE = "operator_stated_2026-09-12_not_measured_on_this_data"
TABLE_SOURCE = "nuscenes_population_mean_LITERATURE_not_measured"
UNREACHABLE_SOURCE = "no_prior_unreachable_under_arm_a_and_arm_b"

# Operator values (2026-09-12) REPLACE the 2.70 / 2.65 literature lengths. l x w x h.
LITERATURE: dict[str, dict] = {
    "a rickshaw": {"category": "dhaka.cycle_rickshaw", "l": 2.40, "w": 1.15, "h": 1.75,
                   "note": "cycle rickshaw; length stated by the operator 2026-09-12, w/h typical build",
                   "source": OPERATOR_SOURCE},
    "an auto rickshaw": {"category": "dhaka.cng", "l": 2.40, "w": 1.30, "h": 1.75,
                         "note": "CNG auto rickshaw; length stated by the operator 2026-09-12, w/h Bajaj RE class",
                         "source": OPERATOR_SOURCE},
}

# docs/Annotation_pipeline.md:141 — nuScenes/KITTI population means, W x L x H.
TABLE: dict[str, dict] = {
    "a car":        {"category": "vehicle.car",              "w": 1.93, "l": 4.63,  "h": 1.56},
    "a truck":      {"category": "vehicle.truck",            "w": 2.51, "l": 6.93,  "h": 2.84},
    "a bus":        {"category": "vehicle.bus.rigid",        "w": 2.96, "l": 11.19, "h": 3.44},
    "a pedestrian": {"category": "human.pedestrian.adult",   "w": 0.77, "l": 0.76,  "h": 1.72},
    "a bicycle":    {"category": "vehicle.bicycle",          "w": 0.60, "l": 1.76,  "h": 1.59},
    "a motorcycle": {"category": "vehicle.motorcycle",       "w": 0.77, "l": 2.11,  "h": 1.47},
}
TABLE_PATH = "docs/Annotation_pipeline.md:141"
```

Update `_literature_block` to use the per-entry source: `"source": lit.get("source", ASSUMED_SOURCE)`.

Add:

```python
def _table_block(phrase: str, eps_scale: float) -> dict:
    t = TABLE[phrase]
    dims = {ax: {"mu": float(t[ax]), "sigma": round(SIGMA_FRACTION * t[ax], 4)} for ax in ("w", "l", "h")}
    return {
        "category": t["category"], "categories": [t["category"]], "dims": dims,
        "eps_bev": eps_scale * math.hypot(t["w"], t["l"]), "n_instances": 0, "conf_thresh": None,
        "gaps": [], "source": TABLE_SOURCE,
        "sigma_assumption": f"{SIGMA_FRACTION:.0%} of mu per axis, ASSUMED (no Dhaka measurement)",
        "dims_note": f"population mean from {TABLE_PATH}",
    }


def _unreachable_block(phrase: str) -> dict:
    return {"category": None, "categories": [], "dims": None, "eps_bev": None, "n_instances": 0,
            "conf_thresh": None, "gaps": ["no_source", "unreachable_under_arm_a_and_arm_b"],
            "source": UNREACHABLE_SOURCE}


def author_from_table(phrases: list[str], *, fingerprint: str, binding: dict, authored_on: str,
                      eps_scale: float = 0.6) -> dict:
    """The priors file with NO template: table means + operator/literature indigenous dims."""
    classes: dict[str, dict] = {}
    for phrase in phrases:
        if phrase in TABLE:
            classes[phrase] = _table_block(phrase, eps_scale)
        elif phrase in LITERATURE:
            classes[phrase] = _literature_block(phrase, eps_scale)
        else:
            classes[phrase] = _unreachable_block(phrase)
    return {
        "spec": PRIORS_SPEC, "name": PRIORS_NAME,
        "source": "authored_dhaka:table_means+operator_indigenous",
        "gt_derived": False, "source_note": SOURCE_NOTE, "eps_scale": eps_scale,
        "eps_formula": "eps_bev = eps_scale * sqrt(w^2 + l^2) on the class mean",
        "classes": classes,
        "classes_without_instances": list(phrases),
        "derived_from": {
            "metadata_fingerprint": fingerprint,
            "fingerprint_spec": binding.get("fingerprint_spec"),
            "dataroot_realpath": binding.get("dataroot_realpath"),
            "version": binding.get("version"),
            # P1-5 guard field: literature/table values were tuned on NO scene.
            "scene_subset": "priors",
            "subset_note": "no scene was used: every value is a published population mean or an operator statement",
            "scenes": [], "authored_on": authored_on, "authored_by": "scripts/author_priors_dhaka.py --from-table",
            "transferred_from": {"table": TABLE_PATH, "operator": OPERATOR_SOURCE},
            "REBOUND": {"note": "the fingerprint must be rebound whenever the metadata tables change",
                        "rebound_history": []},
        },
        "release_guard": None,
    }
```

In `main`: add `parser.add_argument("--from-table", action="store_true", help="no template: table means + operator dims")`. Where the template is opened, branch: if `args.from_table`, `payload = author_from_table(phrases, fingerprint=fingerprint, binding=binding, authored_on=authored_on)`; else the existing path. Keep everything else (fingerprint computation from `--paths`, the `load_priors` re-read, the `--out` default) as is. If `load_priors` rejects any key in the payload above, fix the payload to what the loader requires — the loader is the contract, this plan's dict is the starting point.

- [ ] **Step 4: Update the one old assertion this change is MEANT to break, then run tests**

`tests/test_author_priors_dhaka.py:90` asserts the rickshaw block's `source == ASSUMED_SOURCE`.
The rickshaw/CNG source is now the operator statement, on purpose. Change that line to
`assert r["source"] == OPERATOR_SOURCE and r["n_instances"] == 0` and add `OPERATOR_SOURCE`
to that file's import list. Do NOT weaken any other assertion in that file.

Run: `PYTHONNOUSERSITE=1 $PY -m pytest tests/test_author_priors_table.py tests/test_author_priors_dhaka.py -q -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Author the real file for this run**

```bash
set -a && . ./.env && set +a
PYTHONNOUSERSITE=1 $PY scripts/author_priors_dhaka.py --from-table --paths configs/paths_zami_20260911.yaml
ls -la /mnt/hdd/dhakascenes/out_zami/20260911_zed/priors/priors_pilot_v0.json
```

- [ ] **Step 6: Commit**

```bash
git add scripts/author_priors_dhaka.py tests/test_author_priors_table.py
git commit -m "priors: --from-table authors the Dhaka priors without the lost template; rickshaw/CNG length 2.40 m (operator)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Coverage config `R3` — the two ZED frusta

**Files:**
- Modify: `pipeline/common/eval_region.py`, `pipeline/common/schemas.py:636-648`, `pipeline/stage5_lift/lift.py:246-252`
- Test: `tests/test_eval_region_r3.py`

**Interfaces:**
- Produces: `eval_region.R3_HALF_WIDTH_RAD`, `eval_region.R3_DEFAULT: RegionSpec`, `eval_region.STEREO_RANGE_CAP_DEFAULT_M = 25.0`, `"R3"` in `COVERAGE_CONFIGS`; `lift.region_for("R3") -> R3_DEFAULT`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_eval_region_r3.py
"""coverage_config R3 = the two ZED frusta (2026-09-12): front +-h, rear pi+-h."""
from __future__ import annotations
import math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.common import eval_region as er  # noqa: E402
from pipeline.common.eval_region import in_region, region_spec_from_config  # noqa: E402


def test_r3_is_a_known_config_with_two_wedges():
    assert "R3" in er.COVERAGE_CONFIGS
    spec = er.R3_DEFAULT
    assert spec.coverage_config == "R3"
    assert abs(spec.azimuth_measure_rad - 4 * er.R3_HALF_WIDTH_RAD) < 1e-9
    assert spec.r_max_m == er.STEREO_RANGE_CAP_DEFAULT_M == 25.0


def test_r3_membership():
    spec = er.R3_DEFAULT
    r = 10.0
    for deg, inside in ((0, True), (180, True), (25, True), (-25, True), (155, True), (-155, True),
                        (90, False), (-90, False), (60, False), (120, False)):
        th = math.radians(deg)
        got = bool(in_region([r * math.cos(th)], [r * math.sin(th)], spec, frame="ego")[0])
        assert got is inside, f"{deg} deg: expected {inside}, got {got}"
    assert not in_region([30.0], [0.0], spec, frame="ego")[0]  # beyond the cap


def test_r3_from_config_dict():
    spec = region_spec_from_config({"coverage_config": "R3", "r_max_m": 18.0})
    assert spec.coverage_config == "R3" and spec.r_max_m == 18.0
    assert abs(spec.azimuth_measure_rad - 4 * er.R3_HALF_WIDTH_RAD) < 1e-9
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `PYTHONNOUSERSITE=1 $PY -m pytest tests/test_eval_region_r3.py -q -p no:cacheprovider`
Expected: FAIL — `AttributeError: module ... has no attribute 'R3_DEFAULT'`.

- [ ] **Step 3: Implement in `eval_region.py`**

```python
COVERAGE_CONFIGS: tuple[str, ...] = ("R1", "R2", "R3")

# R3 (2026-09-12): the two ZED 2i frusta. h = half the rectified horizontal FOV
# (calibration.json h_fov_deg 67.748); the cap is the spike's default until
# docs/evidence/2026-09-12-stereo-vs-lidar-*.md says otherwise.
R3_HALF_WIDTH_RAD = math.radians(67.748 / 2.0)
STEREO_RANGE_CAP_DEFAULT_M = 25.0
_R3_BLIND_WEDGES = (
    (R3_HALF_WIDTH_RAD, math.pi - R3_HALF_WIDTH_RAD),        # left side
    (-math.pi + R3_HALF_WIDTH_RAD, -R3_HALF_WIDTH_RAD),      # right side
)
```

In `_admitted_azimuth`: `if spec.coverage_config in ("R2", "R3"): base = [(-math.pi, math.pi)]`.
In `RegionSpec.__post_init__`, after the R1 check: `if self.coverage_config == "R3" and not self.blind_wedges_rad: raise ValueError("R3 requires the two side blind wedges; use R3_DEFAULT or region_spec_from_config")`.
After `R2_DEFAULT`:

```python
R3_DEFAULT = RegionSpec(
    coverage_config="R3", r_max_m=STEREO_RANGE_CAP_DEFAULT_M, azimuth_half_width_rad=None,
    blind_wedges_rad=_R3_BLIND_WEDGES,
    provenance="2026-09-12 approach A: the two ZED frusta; cap = STEREO_RANGE_CAP_DEFAULT_M until measured",
)
```

In `region_spec_from_config`: when `coverage == "R3"` and the dict gives no `blind_wedges_rad`, use `_R3_BLIND_WEDGES`; when it gives no `r_max_m`, use `STEREO_RANGE_CAP_DEFAULT_M`. Update the error text "(R1 or R2)" → "(R1, R2 or R3)".

`schemas.py` (the block at 636-648): add after the R1 rule:

```python
            if self.coverage_config == "R3":
                need = {"CAM_FRONT", "CAM_BACK"}
                if not need <= set(self.camera_subset):
                    _err(e, p, "coverage_config=R3 claims the two ZED frusta and needs CAM_FRONT and CAM_BACK in camera_subset")
```

`lift.py` `region_for`: add `if coverage_config == "R3": return R3_DEFAULT` (import it) and update the error text to "not R1, R2 or R3".

- [ ] **Step 4: Run tests**

Run: `PYTHONNOUSERSITE=1 $PY -m pytest tests/test_eval_region_r3.py tests/test_range_cap_50m.py -q -p no:cacheprovider && PYTHONNOUSERSITE=1 $PY -m pytest tests -q -p no:cacheprovider | tail -2`
Expected: new tests PASS; suite = 616 + 3 passed, the same 1 environmental failure.

- [ ] **Step 5: Commit**

```bash
git add pipeline/common/eval_region.py pipeline/common/schemas.py pipeline/stage5_lift/lift.py tests/test_eval_region_r3.py
git commit -m "eval_region: coverage_config R3 = the two ZED frusta, cap 25 m default

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Stage 1 ingests `ZED_WORLD`

**Files:**
- Modify: `pipeline/stage1_ingestion/ingest.py` (STEREO_CHANNELS ≈ line 149; IngestConfig ≈ 160-300; the fuse block ≈ 1318-1340; `main` argparse ≈ 1750+)
- Test: `tests/test_stage1_zed_world.py`

**Interfaces:**
- Consumes: `pipeline.common.conventions.Transform.from_nuscenes(record, source_frame=EGO, parent_frame=NUSCENES_GLOBAL).inverse_matrix()`, `apply_transform(T, xyz)`, `thin_stereo(cloud, rings, stride)`, `sub.by_token("ego_pose.json")`, `sub.blob(record)`.
- Produces: `STEREO_CHANNELS: dict[str, dict]` with entries `{"frame": "sensor"|"global_identity", "ring": int|None}`; pure function `stereo_block_to_ego(raw: np.ndarray, *, frame: str, ring: int | None, t_sensor_to_ego: np.ndarray | None, t_global_to_ego: np.ndarray | None, z_correction_m: dict[int, float]) -> np.ndarray` returning `(N, 5)` float64 `[x, y, z, intensity, ring]` in ego; CLI flags `--stereo-stride INT`, `--coverage-config {R1,R2,R3}`, `--stereo-z-correction RING:METRES` (repeatable).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stage1_zed_world.py
"""Stage 1 ZED_WORLD ingestion (2026-09-12): a GLOBAL-frame stereo channel with
identity ego_pose/calibrated_sensor is brought into ego with the inverse of the
LiDAR ego_pose of the SAME sample; ring tags 100/101 ride through untouched."""
from __future__ import annotations
import math, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.common.conventions import EGO, NUSCENES_GLOBAL, Transform  # noqa: E402
from pipeline.stage1_ingestion import ingest  # noqa: E402


def _ego_pose(yaw_deg, t):
    h = math.radians(yaw_deg) / 2
    return {"translation": list(t), "rotation": [math.cos(h), 0.0, 0.0, math.sin(h)]}


def test_stereo_channels_declare_zed_world_as_global_identity():
    assert ingest.STEREO_CHANNELS["ZED_WORLD"] == {"frame": "global_identity", "ring": None}
    assert ingest.STEREO_CHANNELS["ZED_FRONT"] == {"frame": "sensor", "ring": 10}


def test_global_identity_block_lands_in_ego():
    pose = _ego_pose(90.0, (100.0, 50.0, 2.0))           # ego at (100,50,2), facing +y
    T = Transform.from_nuscenes(pose, source_frame=EGO, parent_frame=NUSCENES_GLOBAL)
    # a point 10 m AHEAD of the ego in ego coords is (10,0,0) -> global (100, 60, 2)
    raw = np.array([[100.0, 60.0, 2.0, 200.0, 101.0]], dtype=np.float32)
    out = ingest.stereo_block_to_ego(raw, frame="global_identity", ring=None, t_sensor_to_ego=None,
                                     t_global_to_ego=T.inverse_matrix(), z_correction_m={})
    assert out.shape == (1, 5)
    assert np.allclose(out[0, :3], [10.0, 0.0, 0.0], atol=1e-6)
    assert out[0, 3] == 200.0 and out[0, 4] == 101.0      # intensity + file ring untouched


def test_sensor_frame_block_uses_fixed_ring_and_sensor_transform():
    Ts = np.eye(4); Ts[0, 3] = 1.0                          # sensor 1 m ahead of ego origin
    raw = np.array([[0.0, 0.0, 0.0, 7.0, 0.0]], dtype=np.float32)
    out = ingest.stereo_block_to_ego(raw, frame="sensor", ring=10, t_sensor_to_ego=Ts,
                                     t_global_to_ego=None, z_correction_m={})
    assert np.allclose(out[0, :3], [1.0, 0.0, 0.0]) and out[0, 4] == 10.0


def test_z_correction_applies_per_ring_only():
    T = np.eye(4)
    raw = np.array([[0, 0, 0, 1, 100.0], [0, 0, 0, 1, 101.0]], dtype=np.float32)
    out = ingest.stereo_block_to_ego(raw, frame="global_identity", ring=None, t_sensor_to_ego=None,
                                     t_global_to_ego=T, z_correction_m={100: 0.69})
    assert abs(out[0, 2] - 0.69) < 1e-9 and out[1, 2] == 0.0


def test_cli_parses_new_knobs():
    p = ingest.build_parser()
    a = p.parse_args(["--stereo-stride", "1", "--coverage-config", "R3",
                      "--stereo-z-correction", "100:0.69", "--stereo-z-correction", "101:-0.1"])
    assert a.stereo_stride == 1 and a.coverage_config == "R3"
    assert ingest.parse_z_corrections(a.stereo_z_correction) == {100: 0.69, 101: -0.1}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `PYTHONNOUSERSITE=1 $PY -m pytest tests/test_stage1_zed_world.py -q -p no:cacheprovider`
Expected: FAIL — `AttributeError: module ... has no attribute 'stereo_block_to_ego'` (and STEREO_CHANNELS shape mismatch).

- [ ] **Step 3: Implement**

Replace the `STEREO_CHANNELS` definition (≈ line 149):

```python
# Stereo channels Stage 1 may merge into the single sweep. Two frame conventions:
#   sensor           — a normal nuScenes channel: points in the sensor frame, a
#                      real calibrated_sensor; the ring tag is ASSIGNED here.
#   global_identity  — the exporter's --zed-world-cloud: BOTH ZEDs' depth in the
#                      GLOBAL frame with identity ego_pose/calibrated_sensor, so
#                      the normal chain must NOT be applied; ego = inv(LIDAR_TOP
#                      ego_pose of the same sample). The file's own ring column
#                      (100 = rear ZED, 101 = front ZED) is KEPT.
STEREO_CHANNELS: dict[str, dict] = {
    "ZED_FRONT": {"frame": "sensor", "ring": 10},
    "ZED_BACK": {"frame": "sensor", "ring": 11},
    "ZED_WORLD": {"frame": "global_identity", "ring": None},
}
```

Add to `IngestConfig`: `stereo_z_correction_m: dict = field(default_factory=dict)` with provenance entry `"stereo_z_correction_m": "per-ring constant z offset in metres applied to stereo points at ingestion; EMPTY unless scripts/spike_stereo_vs_lidar.py found a range-constant offset (its evidence doc names the value)"`. Update the `fuse_stereo` and `stereo_stride` provenance strings to mention `ZED_WORLD` / `global_identity` and that stride 1 is the approach-A setting. `as_dict` must serialise the dict with string keys: `out["stereo_z_correction_m"] = {str(k): v for k, v in self.stereo_z_correction_m.items()}`.

Add the pure function near `thin_stereo`:

```python
def stereo_block_to_ego(raw: np.ndarray, *, frame: str, ring: int | None,
                        t_sensor_to_ego: np.ndarray | None, t_global_to_ego: np.ndarray | None,
                        z_correction_m: dict[int, float]) -> np.ndarray:
    """One stereo blob -> (N, 5) float64 [x, y, z, intensity, ring] in the EGO frame."""
    if frame == "sensor":
        xyz = apply_transform(t_sensor_to_ego, raw[:, :3].astype(np.float64))
        rings = np.full(xyz.shape[0], float(ring))
    elif frame == "global_identity":
        xyz = apply_transform(t_global_to_ego, raw[:, :3].astype(np.float64))
        rings = raw[:, 4].astype(np.float64)
    else:
        raise ValueError(f"unknown stereo frame handling {frame!r}")
    for r, dz in z_correction_m.items():
        xyz[rings == float(r), 2] += float(dz)
    return np.column_stack([xyz, raw[:, 3:4].astype(np.float64), rings])
```

Rewrite the fuse block (≈ 1322-1340):

```python
    n_stereo_pts: dict[str, int] = {}
    stereo_frame_handling: dict[str, str] = {}
    if cfg.fuse_stereo:
        t_global_to_ego = Transform.from_nuscenes(
            sub.by_token("ego_pose.json")[anchor["ego_pose_token"]],
            source_frame=EGO, parent_frame=NUSCENES_GLOBAL,
        ).inverse_matrix()
        for channel, how in STEREO_CHANNELS.items():
            record = channel_records.get(channel)
            if record is None:
                continue
            stereo_raw, _ = thin_stereo(read_pcd_bin(sub.blob(record)), cfg.stereo_rings, cfg.stereo_stride)
            t_sensor_to_ego = None
            if how["frame"] == "sensor":
                t_sensor_to_ego = Transform.from_nuscenes(
                    sub.by_token("calibrated_sensor.json")[record["calibrated_sensor_token"]],
                    source_frame=LIDAR, parent_frame=EGO,
                ).matrix()
            block = stereo_block_to_ego(
                stereo_raw, frame=how["frame"], ring=how["ring"], t_sensor_to_ego=t_sensor_to_ego,
                t_global_to_ego=t_global_to_ego, z_correction_m=cfg.stereo_z_correction_m,
            )
            single = np.vstack([single, block])
            stereo_frame_handling[channel] = how["frame"]
            for r in np.unique(block[:, 4]).astype(int):
                n_stereo_pts[f"{channel}:ring{r}"] = int(np.count_nonzero(block[:, 4] == r))
```

Add `stereo_frame_handling` to the `single_sweep_sources` dict written per keyframe (≈ line 1436). Confirm `Transform`, `NUSCENES_GLOBAL`, `LIDAR`, `EGO`, `apply_transform` are imported from `pipeline.common.conventions` (add any missing).

CLI: factor the parser into `def build_parser() -> argparse.ArgumentParser` (called by `main`) and add:

```python
    parser.add_argument("--stereo-stride", type=int, default=None,
                        help="keep every k-th stereo point (profile default 8; approach A uses 1)")
    parser.add_argument("--coverage-config", default=None, choices=("R1", "R2", "R3"),
                        help="eval region E recorded on every keyframe (default R2)")
    parser.add_argument("--stereo-z-correction", action="append", default=[],
                        metavar="RING:METRES", help="constant z offset for one stereo ring; repeatable")
```

```python
def parse_z_corrections(items: list[str]) -> dict[int, float]:
    out: dict[int, float] = {}
    for item in items:
        ring, _, metres = item.partition(":")
        out[int(ring)] = float(metres)
    return out
```

Apply them when building `IngestConfig` in `main`: `stereo_stride=args.stereo_stride if args.stereo_stride is not None else STEREO_STRIDE`, `coverage_config=args.coverage_config or "R2"`, `stereo_z_correction_m=parse_z_corrections(args.stereo_z_correction)`.

- [ ] **Step 4: Run tests**

Run: `PYTHONNOUSERSITE=1 $PY -m pytest tests/test_stage1_zed_world.py tests/test_stage1_thin_stereo.py tests/test_stage1_ground_reference.py tests/test_stage1_seed_determinism.py tests/test_stage1_heartbeat.py -q -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Run Stage 1 for real on chunk_0010 with full stereo density and R3**

```bash
set -a && . ./.env && set +a
PYTHONNOUSERSITE=1 $PY -m pipeline.stage1_ingestion.ingest --accept-degraded-upstream \
  --scenes dhaka_20260911_141259_chunk_0010 --stereo-stride 1 --coverage-config R3 2>&1 | tail -15
```

Expected: `--- ... DEGRADED` or `OK` (never REFUSED), and in the printed scene line the single-sweep input count is roughly `40k + 115k` per keyframe. Verify:

```bash
$PY -c "
import json,glob
r=json.loads(open(glob.glob('/mnt/hdd/dhakascenes/work_zami/20260911_zed/stage1_ingestion/scenes/*/keyframes.jsonl')[0]).readline())
print(r['coverage_config'], r['single_sweep_cloud']['n_points'])"
```
Expected: `R3` and n_points well above 40,000 (the fused sweep, after ground/range filtering).

- [ ] **Step 6: Commit**

```bash
git add pipeline/stage1_ingestion/ingest.py tests/test_stage1_zed_world.py
git commit -m "stage1: ingest the export's ZED_WORLD channel (global-frame stereo) into the ego single sweep; --stereo-stride/--coverage-config/--stereo-z-correction knobs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Calibration spike → range cap and z offset

**Files:**
- Create: `scripts/spike_stereo_vs_lidar.py`, `configs/stereo_box.yaml`
- Evidence: `docs/evidence/2026-09-12-stereo-vs-lidar-chunk_0010.md` + `.json`

**Interfaces:**
- Consumes: Stage 1 single-sweep clouds from Task 3 (`<work_root>/stage1_ingestion/scenes/<scene>/keyframes.jsonl` → `single_sweep_cloud.path`), `read_pcd_bin`, `scipy.spatial.cKDTree`.
- Produces: `configs/stereo_box.yaml` with the measured `stereo_range_cap_m` and `stereo_z_correction_m`.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python3
"""ZED stereo vs LiDAR agreement by range, on Stage 1's fused single sweep.

For every stereo point (ring 100/101) the nearest LiDAR point (rings 0-3) within
0.6 m is its reference. Per 1 m LiDAR-range bin and per ring: count, median and
MAD of d_range = |p_zed| - |p_lidar| and of dz = z_zed - z_lidar (ego frame).
Decision rule (spec §3.4): cap = largest bin with median |d_range| <= 0.5 m and
MAD <= 1.0 m for BOTH rings, bins with >= 200 pairs only. A per-ring dz that is
constant across 3-15 m (|slope| < 0.01 m/m) and > 0.2 m is reported as a
constant offset to correct; otherwise reported and not corrected.
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.stage1_ingestion.ingest import read_pcd_bin  # noqa: E402

STEREO = (100, 101)


def pairs_for_cloud(cloud: np.ndarray, max_pair_m: float = 0.6):
    lidar = cloud[cloud[:, 4] < 10]
    if len(lidar) < 100:
        return None
    tree = cKDTree(lidar[:, :3])
    out = {}
    for ring in STEREO:
        z = cloud[cloud[:, 4] == ring]
        if not len(z):
            continue
        d, j = tree.query(z[:, :3], distance_upper_bound=max_pair_m)
        ok = np.isfinite(d)
        if not ok.any():
            continue
        ref = lidar[j[ok]]
        out[ring] = np.column_stack([
            np.linalg.norm(ref[:, :3], axis=1),                                     # lidar range
            np.linalg.norm(z[ok, :3], axis=1) - np.linalg.norm(ref[:, :3], axis=1),  # d_range
            z[ok, 2] - ref[:, 2],                                                    # dz
        ])
    return out


def mad(x):
    return float(np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n-keyframes", type=int, default=120)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    a = ap.parse_args(argv)
    paths = load_paths(a.paths)
    kf_path = os.path.join(paths.work_root, "stage1_ingestion", "scenes", a.scene, "keyframes.jsonl")
    rows = [json.loads(l) for l in open(kf_path) if l.strip()]
    step = max(1, len(rows) // a.n_keyframes)
    acc = {r: [] for r in STEREO}
    used = 0
    for row in rows[::step][: a.n_keyframes]:
        pr = pairs_for_cloud(read_pcd_bin(row["single_sweep_cloud"]["path"]))
        if not pr:
            continue
        used += 1
        for r, arr in pr.items():
            acc[r].append(arr)
    bins = {}
    for r in STEREO:
        arr = np.vstack(acc[r]) if acc[r] else np.zeros((0, 3))
        per = []
        for lo in range(0, 40):
            sel = arr[(arr[:, 0] >= lo) & (arr[:, 0] < lo + 1)]
            per.append({"range_m": [lo, lo + 1], "n": int(len(sel)),
                        "d_range_median": float(np.median(sel[:, 1])) if len(sel) else None,
                        "d_range_mad": mad(sel[:, 1]) if len(sel) else None,
                        "dz_median": float(np.median(sel[:, 2])) if len(sel) else None,
                        "dz_mad": mad(sel[:, 2]) if len(sel) else None})
        bins[r] = per
    # decision rule
    cap = 0
    for lo in range(0, 40):
        ok = all(bins[r][lo]["n"] >= 200 and abs(bins[r][lo]["d_range_median"]) <= 0.5
                 and bins[r][lo]["d_range_mad"] <= 1.0 for r in STEREO)
        if ok:
            cap = lo + 1
        elif cap:
            break
    zcorr, zreport = {}, {}
    for r in STEREO:
        arr = np.vstack(acc[r]) if acc[r] else np.zeros((0, 3))
        sel = arr[(arr[:, 0] >= 3) & (arr[:, 0] < 15)]
        if len(sel) < 500:
            zreport[r] = "insufficient pairs"; continue
        med = float(np.median(sel[:, 2])); slope = float(np.polyfit(sel[:, 0], sel[:, 2], 1)[0])
        zreport[r] = {"dz_median_3_15m": med, "dz_slope_m_per_m": slope}
        if abs(med) > 0.2 and abs(slope) < 0.01:
            zcorr[r] = round(-med, 3)   # correction = minus the offset
    result = {"scene": a.scene, "keyframes_used": used, "bins": bins, "stereo_range_cap_m": cap or None,
              "stereo_z_correction_m": zcorr, "dz_report": zreport}
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(result, open(a.out_json, "w"), indent=1)
    with open(a.out_md, "w") as f:
        f.write(f"# ZED stereo vs LiDAR — {a.scene}\n\nKeyframes used: {used}. Pairs within 0.6 m.\n\n")
        f.write(f"**stereo_range_cap_m = {cap or 'UNDETERMINED (default 25.0)'}**  \n")
        f.write(f"**stereo_z_correction_m = {zcorr or 'none'}**  (dz report: {zreport})\n\n")
        for r in STEREO:
            f.write(f"## ring {r}\n\n| range | n | d_range med | d_range MAD | dz med | dz MAD |\n|---|---|---|---|---|---|\n")
            for b in bins[r]:
                if b["n"]:
                    f.write(f"| {b['range_m'][0]}-{b['range_m'][1]} | {b['n']} | {b['d_range_median']:.2f} | {b['d_range_mad']:.2f} | {b['dz_median']:.2f} | {b['dz_mad']:.2f} |\n")
            f.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k != "bins"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run it**

```bash
set -a && . ./.env && set +a
PYTHONNOUSERSITE=1 $PY scripts/spike_stereo_vs_lidar.py --scene dhaka_20260911_141259_chunk_0010 \
  --out-md docs/evidence/2026-09-12-stereo-vs-lidar-chunk_0010.md \
  --out-json docs/evidence/2026-09-12-stereo-vs-lidar-chunk_0010.json
```

Expected: a printed JSON with `stereo_range_cap_m` (an integer, or null) and `stereo_z_correction_m`.

- [ ] **Step 3: Write `configs/stereo_box.yaml` from the result**

```yaml
# Approach A tunables (spec §3.4, §4.2). Values with provenance; the evidence doc
# named below is the measurement behind the two calibration-derived ones.
stereo_range_cap_m: 25.0          # REPLACE with the spike's value if not null; provenance:
                                  # docs/evidence/2026-09-12-stereo-vs-lidar-chunk_0010.md
stereo_z_correction_m: {}         # REPLACE with the spike's per-ring dict if non-empty, e.g. {100: 0.69}
k_mad: 3.0                        # spec §4.2 step 1
mad_floor_m: 0.10                 # spec §4.2 step 1
min_stereo_pts: 20                # spec §4.2 step 1
lidar_refine_min_pts: 5           # spec §4.2 step 2
eig_ratio_isotropic: 1.5          # spec §4.2 step 6
percentile_lo: 5                  # spec §4.2 step 3
percentile_hi: 95
near_face_percentile: 20          # spec §4.2 step 7: the near face is p20 of the trimmed depths
prior_clamp_sigma: 2.0            # spec §4.2 step 5
min_samples: 5                    # recorded for Stage 7's reconstruct_cluster_points; A does not cluster
```

If the spike returned a z correction, ALSO re-run Task 3 Step 5 with `--stereo-z-correction RING:METRES` for each entry so the Stage 1 tree used downstream carries it. If `stereo_range_cap_m` differs from 25.0, set `STEREO_RANGE_CAP_DEFAULT_M` in `eval_region.py` to the same value (one number, two readers — keep them equal) and rerun `tests/test_eval_region_r3.py` after updating its `== 25.0` assertion to the new value.

- [ ] **Step 4: Commit**

```bash
git add scripts/spike_stereo_vs_lidar.py configs/stereo_box.yaml docs/evidence/2026-09-12-stereo-vs-lidar-chunk_0010.md docs/evidence/2026-09-12-stereo-vs-lidar-chunk_0010.json pipeline/common/eval_region.py tests/test_eval_region_r3.py
git commit -m "spike: ZED stereo vs LiDAR agreement by range on chunk_0010 -> stereo_range_cap_m and per-ring z offset; configs/stereo_box.yaml

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: `stage6_stereo_box` — the stage

**Files:**
- Create: `pipeline/stage6_stereo_box/__init__.py` (empty), `pipeline/stage6_stereo_box/stereo_box.py`
- Test: `tests/test_stereo_box.py`

**Interfaces:**
- Consumes: `pipeline.stage1_ingestion.ingest.read_pcd_bin(path) -> (N,5)`; Stage 5 `lift.jsonl` rows (`keyframe_token, t_ns, coverage_config, cloud_path, points_path, mask_path, instances[{instance_id, channel, class_name, proposal_index, score, n_mask_px}]`); Stage 5 `points/<kf>.npz` (`point_index`, `instance_id`); `pipeline.stage5_lift.lift.MaskFile(path).mask(channel, index)`; Stage 1 `filter_diagnostics.json` (`keyframes[i].keyframe_token`, `.ground_reference_plane{a,b,d}`), Stage 1 `keyframes.jsonl` (`cameras[ch].calibrated_sensor_token`); `pipeline.stage0_data_probe.probe.Substrate.load(paths).by_token("calibrated_sensor.json")`; `pipeline.common.conventions` (`Transform`, `apply_transform`, `quaternion_from_yaw_rad`, `EGO`, `CAMERA`); `pipeline.common.manifest` (`require_upstream`, `write_jsonl_atomic`, `write_json_atomic`, `clear_markers`, `write_marker`); `pipeline.stage6_cluster.priors.load_priors`; `pipeline.stage6_cluster.cluster.STATUS_FIT`.
- Produces: pure function `box_from_stereo(pts_ego: np.ndarray, rings: np.ndarray, *, K: np.ndarray, T_ego_cam: np.ndarray, prior: dict, ground_abd: tuple, cfg: dict) -> tuple[dict | None, str, dict]` returning `(box_dict_or_None, status, stereo_block)`; CLI `python -m pipeline.stage6_stereo_box.stereo_box --paths P [--stage5-dir D] [--stage1-dir D] [--priors F] [--out-dir D] [--config configs/stereo_box.yaml] [--scenes ...] [--accept-degraded-upstream]`; output `<out_dir>/scenes/<scene>/boxes.jsonl`, `run_manifest.json`, marker.

- [ ] **Step 1: Write the failing tests (synthetic scene, known answer)**

```python
# tests/test_stereo_box.py
"""stage6_stereo_box: one mask -> one box from stereo points, no clustering.
Synthetic: a box of known pose sampled as noisy stereo points, with a background
plane behind it; the MAD trim must reject the plane, the near face must sit at
the robust depth, the bottom on the ground plane, the yaw axis along the box."""
from __future__ import annotations
import math, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.stage6_stereo_box.stereo_box import box_from_stereo, DEFAULT_CFG  # noqa: E402

# front ZED: optical frame x-right y-down z-forward, mounted 0.8 m ahead, 0.7 m below ego origin
K = np.array([[953.16, 0, 656.28], [0, 953.16, 375.74], [0, 0, 1.0]])
R_opt_to_ego = np.array([[0, 0, 1.0], [-1.0, 0, 0], [0, -1.0, 0]])   # optical -> body (x fwd, y left, z up)
T_EGO_CAM = np.eye(4); T_EGO_CAM[:3, :3] = R_opt_to_ego; T_EGO_CAM[:3, 3] = [0.8, 0.0, -0.7]
GROUND = (0.0, 0.0, -2.4)   # z = -2.4 everywhere
PRIOR = {"w": (1.15, 0.115), "l": (2.40, 0.24), "h": (1.75, 0.175)}   # (mu, sigma)


def _rickshaw(center=(12.0, 1.0), yaw=math.radians(20), n=800, seed=0):
    """VISIBLE surface of a 1.15 x 2.40 x 1.75 box standing on the ground, ego frame: the end face
    nearest the camera (60 % of points) plus the side face turned toward it (40 %) — what a stereo
    camera actually sees — with 15 cm range noise along the camera ray and a wall 6 m behind."""
    rng = np.random.default_rng(seed)
    w, l, h = 1.15, 2.40, 1.75
    n_end = int(0.6 * n); n_side = n - n_end
    u = np.concatenate([np.full(n_end, -l / 2), rng.uniform(-l / 2, l / 2, n_side)])
    v = np.concatenate([rng.uniform(-w / 2, w / 2, n_end), np.full(n_side, w / 2)])
    z = rng.uniform(0, h, n)
    xy = np.column_stack([u, v]) @ np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
    pts = np.column_stack([xy[:, 0] + center[0], xy[:, 1] + center[1], z + GROUND[2]])
    ray = pts - T_EGO_CAM[:3, 3]; ray /= np.linalg.norm(ray, axis=1, keepdims=True)
    pts += ray * rng.normal(0, 0.15, (n, 1))                       # range noise 15 cm
    wall = np.column_stack([np.full(120, center[0] + 6.0), rng.uniform(-2, 4, 120), rng.uniform(GROUND[2], GROUND[2] + 3, 120)])
    allp = np.vstack([pts, wall]); rings = np.full(len(allp), 101.0)
    return allp, rings


def test_recovers_pose_and_rejects_wall():
    pts, rings = _rickshaw()
    box, status, st = box_from_stereo(pts, rings, K=K, T_ego_cam=T_EGO_CAM, prior=PRIOR, ground_abd=GROUND, cfg=DEFAULT_CFG)
    assert status == "fit" and box is not None
    assert st["n_stereo_kept"] < st["n_stereo_pts"]                 # the wall was trimmed
    tx, ty, tz = box["translation_m"]
    # 0.4 m: 15 cm range noise plus the p20-vs-true-near-face residual on an oblique view
    assert abs(tx - 12.0) < 0.4 and abs(ty - 1.0) < 0.4, (tx, ty)
    w, l, h = box["size_wlh_m"]
    assert l == 2.40 and 0.9 <= w <= 1.4 and 1.4 <= h <= 2.1
    assert abs(box["z_min_m"] - GROUND[2]) < 1e-9 and abs(box["z_max_m"] - (GROUND[2] + h)) < 1e-9
    yaw = box["yaw_rad"] % math.pi
    # 15 deg: the principal axis of an L-shaped (end + side) footprint is biased toward the long leg
    d = abs(yaw - math.radians(20)) % math.pi
    assert min(d, math.pi - d) < math.radians(15), yaw
    assert box["yaw_axis_only"] is True and box["yaw_ambiguous"] is True
    assert box["size_order"] == "w,l,h" and w <= l


def test_too_few_points_and_beyond_cap():
    pts, rings = _rickshaw(n=10)
    box, status, _ = box_from_stereo(pts[:10], rings[:10], K=K, T_ego_cam=T_EGO_CAM, prior=PRIOR, ground_abd=GROUND, cfg=DEFAULT_CFG)
    assert box is None and status == "too_few_stereo"
    pts, rings = _rickshaw(center=(30.0, 0.0))
    box, status, _ = box_from_stereo(pts, rings, K=K, T_ego_cam=T_EGO_CAM, prior=PRIOR, ground_abd=GROUND, cfg={**DEFAULT_CFG, "stereo_range_cap_m": 25.0})
    assert box is None and status == "beyond_stereo_cap"


def test_lidar_refines_depth_when_present():
    pts, rings = _rickshaw()
    # add 8 LiDAR points on the near face, 0.4 m closer than the (biased) stereo median would say
    near = pts[:8].copy(); near[:, 0] -= 0.4
    allp = np.vstack([pts, near]); allr = np.concatenate([rings, np.zeros(8)])
    _, status, st = box_from_stereo(allp, allr, K=K, T_ego_cam=T_EGO_CAM, prior=PRIOR, ground_abd=GROUND, cfg=DEFAULT_CFG)
    assert status == "fit" and st["depth_source"] == "lidar_refined" and st["n_lidar_in_box"] >= 1


def test_pedestrian_prior_keeps_w_le_l_by_swapping():
    ped = {"w": (0.77, 0.077), "l": (0.76, 0.076), "h": (1.72, 0.172)}
    rng = np.random.default_rng(1)
    pts = np.column_stack([rng.normal(8.0, 0.15, 300), rng.normal(0.0, 0.3, 300), rng.uniform(GROUND[2], GROUND[2] + 1.7, 300)])
    box, status, _ = box_from_stereo(pts, np.full(300, 101.0), K=K, T_ego_cam=T_EGO_CAM, prior=ped, ground_abd=GROUND, cfg=DEFAULT_CFG)
    assert status == "fit"
    w, l, _ = box["size_wlh_m"]
    assert w <= l
```

- [ ] **Step 2: Run to confirm failure**

Run: `PYTHONNOUSERSITE=1 $PY -m pytest tests/test_stereo_box.py -q -p no:cacheprovider`
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline.stage6_stereo_box'`.

- [ ] **Step 3: Implement the geometry (pure) and the stage**

`pipeline/stage6_stereo_box/stereo_box.py`:

```python
#!/usr/bin/env python3
"""Stage 6s — per-mask stereo boxing on the ZED frusta (approach A, 2026-09-12).

One Stage 4 mask -> one box. Geometry from the ZED's stereo points in the ZED's
own optical frame; no clustering. Spec: docs/superpowers/specs/2026-09-12-stereo-box-a-design.md §4.
Emits Stage 6's exact boxes.jsonl row (+ additive `stereo` block) so 7/8/9 run unchanged.
"""
from __future__ import annotations
import argparse, json, math, os, sys, time
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pipeline.common.conventions import CAMERA, EGO, Transform, apply_transform, quaternion_from_yaw_rad  # noqa: E402
from pipeline.common.manifest import (UpstreamRefusal, clear_markers, require_upstream,  # noqa: E402
                                      write_json_atomic, write_jsonl_atomic, write_marker)
from pipeline.common.paths import PathValidationError, load_paths, metadata_fingerprint  # noqa: E402
from pipeline.stage0_data_probe.probe import Substrate  # noqa: E402
from pipeline.stage1_ingestion.ingest import read_pcd_bin  # noqa: E402
from pipeline.stage6_cluster.cluster import STATUS_FIT  # noqa: E402
from pipeline.stage6_cluster.priors import PRIORS_NAME, load_priors  # noqa: E402

STAGE = "stage6_stereo_box"
STAGE_SPEC = "dhakascenes-pilot/stage6_stereo_box/v1"
ZED_CHANNELS = ("CAM_FRONT", "CAM_BACK")
STEREO_RINGS = (100.0, 101.0)
EXIT_OK, EXIT_DEGRADED, EXIT_REFUSED = 0, 1, 2
DEFAULT_CFG = {
    "stereo_range_cap_m": 25.0, "stereo_z_correction_m": {}, "k_mad": 3.0, "mad_floor_m": 0.10,
    "min_stereo_pts": 20, "lidar_refine_min_pts": 5, "eig_ratio_isotropic": 1.5,
    "percentile_lo": 5, "percentile_hi": 95, "prior_clamp_sigma": 2.0, "min_samples": 5,
    "near_face_percentile": 20,
}


def _plane_z(abd, x, y):
    a, b, d = abd
    return a * x + b * y + d


def box_from_stereo(pts_ego, rings, *, K, T_ego_cam, prior, ground_abd, cfg):
    """(box | None, status, stereo_block). Spec §4.2 steps 1-9, in that order."""
    T_cam_ego = np.linalg.inv(T_ego_cam)
    cam = apply_transform(T_cam_ego, np.asarray(pts_ego, dtype=np.float64))   # optical: x right, y down, z fwd
    rings = np.asarray(rings)
    is_st = np.isin(rings, STEREO_RINGS); is_li = ~is_st
    st = cam[is_st]; li = cam[is_li]
    stereo = {"n_stereo_pts": int(len(st)), "n_stereo_kept": 0, "d_med_m": None, "d_near_m": None, "mad_m": None,
              "depth_source": None, "w_meas_m": None, "h_meas_m": None, "ray_yaw_rad": None,
              "footprint_eig_ratio": None, "zed_ring": int(rings[is_st][0]) if is_st.any() else None,
              "n_lidar_in_box": 0, "n_stereo_in_box": 0}
    front = st[st[:, 2] > 0.1]
    if len(front) < cfg["min_stereo_pts"]:
        return None, "too_few_stereo", stereo
    # 1. robust depth: MAD trim about the median, then the NEAR FACE is the 20th
    #    percentile of what survives (spec §4.2 step 7: stereo sees a surface, and
    #    the median of an oblique surface sits behind the nearest point).
    d = front[:, 2]; d_med = float(np.median(d)); mad = max(float(np.median(np.abs(d - d_med))), cfg["mad_floor_m"])
    keep = np.abs(d - d_med) <= cfg["k_mad"] * mad
    kept = front[keep]
    stereo.update(n_stereo_kept=int(len(kept)), d_med_m=round(d_med, 4), mad_m=round(mad, 4))
    if len(kept) < cfg["min_stereo_pts"]:
        return None, "too_few_stereo", stereo
    d_near = float(np.percentile(kept[:, 2], cfg["near_face_percentile"]))
    # 2. LiDAR refinement, about the near face
    depth_source = "stereo"
    if len(li):
        li_front = li[li[:, 2] > 0.1]
        band = li_front[np.abs(li_front[:, 2] - d_near) <= 2.0 * mad]
        if len(band) >= cfg["lidar_refine_min_pts"]:
            d_near = float(np.median(band[:, 2])); depth_source = "lidar_refined"
    stereo["depth_source"] = depth_source; stereo["d_near_m"] = round(d_near, 4)
    # 3. measured lateral / vertical extent (pixels at d_med -> metres)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = fx * kept[:, 0] / kept[:, 2] + cx; v = fy * kept[:, 1] / kept[:, 2] + cy
    lo, hi = cfg["percentile_lo"], cfg["percentile_hi"]
    w_meas = float((np.percentile(u, hi) - np.percentile(u, lo)) * d_near / fx)
    h_meas = float((np.percentile(v, hi) - np.percentile(v, lo)) * d_near / fy)
    stereo.update(w_meas_m=round(w_meas, 4), h_meas_m=round(h_meas, 4))
    # 4./5. length from prior; w,h measured & clamped
    (mu_w, s_w), (mu_l, s_l), (mu_h, s_h) = prior["w"], prior["l"], prior["h"]
    k = cfg["prior_clamp_sigma"]; clamped = []
    w = min(max(w_meas, mu_w - k * s_w), mu_w + k * s_w); clamped += ["w"] if w != w_meas else []
    h = min(max(h_meas, mu_h - k * s_h), mu_h + k * s_h); clamped += ["h"] if h != h_meas else []
    l = float(mu_l)
    # 7. centre: near face on the ray through the lateral median at depth d_near, pushed l/2 along it
    p_near = np.array([np.median(kept[:, 0] * d_near / kept[:, 2]), np.median(kept[:, 1] * d_near / kept[:, 2]), d_near])
    ray = p_near / np.linalg.norm(p_near)
    c_cam = p_near + ray * (l / 2.0)
    c_ego = apply_transform(T_ego_cam, c_cam[None, :])[0]
    # 9. range gate (BEV)
    if math.hypot(c_ego[0], c_ego[1]) > cfg["stereo_range_cap_m"]:
        return None, "beyond_stereo_cap", stereo
    # 6. yaw: principal axis of the ground-projected footprint (ego BEV)
    kept_ego = apply_transform(T_ego_cam, kept)
    xy = kept_ego[:, :2] - kept_ego[:, :2].mean(axis=0)
    cov = xy.T @ xy / max(1, len(xy) - 1)
    evals, evecs = np.linalg.eigh(cov)
    ratio = float(evals[1] / max(evals[0], 1e-9)); stereo["footprint_eig_ratio"] = round(ratio, 3)
    ray_yaw = math.atan2(c_ego[1], c_ego[0]); stereo["ray_yaw_rad"] = round(ray_yaw, 6)
    reasons = ["axis_only"]
    if ratio < cfg["eig_ratio_isotropic"]:
        yaw = ray_yaw; reasons.insert(0, "footprint_isotropic")
    else:
        v1 = evecs[:, 1]; yaw = math.atan2(v1[1], v1[0])
    yaw = yaw % math.pi                                  # axis only: [0, pi)
    axis_swapped = False
    if w > l:                                            # keep the [w, l, h] invariant
        w, l = l, w; yaw = (yaw + math.pi / 2) % math.pi; axis_swapped = True
    # 8. ground snap
    z_min = float(_plane_z(ground_abd, c_ego[0], c_ego[1])); z_max = z_min + h
    center = [float(c_ego[0]), float(c_ego[1]), z_min + h / 2.0]
    # points inside the final box (all rings), for num_lidar_pts and the split
    rel = np.asarray(pts_ego, dtype=np.float64) - np.array(center)
    cy_, sy_ = math.cos(-yaw), math.sin(-yaw)
    bx = rel[:, 0] * cy_ - rel[:, 1] * sy_; by = rel[:, 0] * sy_ + rel[:, 1] * cy_
    inside = (np.abs(bx) <= l / 2) & (np.abs(by) <= w / 2) & (rel[:, 2] >= -h / 2) & (rel[:, 2] <= h / 2)
    stereo["n_lidar_in_box"] = int(np.count_nonzero(inside & is_li)); stereo["n_stereo_in_box"] = int(np.count_nonzero(inside & is_st))
    box = {
        "translation_m": [round(c, 4) for c in center], "size_wlh_m": [round(w, 4), round(l, 4), round(h, 4)],
        "size_order": "w,l,h", "yaw_rad": round(yaw, 6), "rotation_wxyz": [round(q, 9) for q in quaternion_from_yaw_rad(yaw)],
        "yaw_axis_only": True, "yaw_ambiguous": True, "yaw_ambiguous_reasons": reasons, "axis_swapped": axis_swapped,
        "clamped_axes": clamped, "z_min_m": round(z_min, 4), "z_max_m": round(z_max, 4),
        "footprint_diagonal_m": round(math.hypot(w, l), 4), "aspect_ratio_w_over_l": round(w / l, 4),
        "fit": {"method": "per_mask_stereo", "k_mad": cfg["k_mad"], "length_source": "prior_mu",
                "depth_source": depth_source, "anchor": "near_face_at_robust_depth", "bottom": "ground_plane"},
    }
    return box, STATUS_FIT, stereo
```

Then the driver (same file): `load_cfg(path) -> dict` (yaml over `DEFAULT_CFG`, int-key the z-correction dict); `load_ground_planes(stage1_dir, scene) -> dict[token, (a,b,d)]` from `filter_diagnostics.json`; `load_calibs(paths, stage1_dir, scene) -> dict[channel, (K, T_ego_cam)]` — read the first `keyframes.jsonl` row's `cameras[ch].calibrated_sensor_token`, then `Substrate.load(paths).by_token("calibrated_sensor.json")[tok]` → `K = np.array(rec["camera_intrinsic"])`, `T_ego_cam = Transform.from_nuscenes(rec, source_frame=CAMERA, parent_frame=EGO).matrix()`; `prior_for(priors, class_name) -> dict | None` from `priors.get(class_name)` → `{"w": (p.mu("w"), p.sigma("w")), ...}` or None when `dims` is None.

Per keyframe (`box_keyframe(lift_row, stage5_dir, calibs, ground, priors, cfg) -> list[dict]`): `cloud = read_pcd_bin(lift_row["cloud_path"])`; `npz = np.load(os.path.join(stage5_dir, lift_row["points_path"]))`; for each instance: rows = `npz["point_index"][npz["instance_id"] == iid]`; base row = Stage 6's keys exactly:

```python
base = {"instance_id": inst["instance_id"], "channel": inst["channel"], "proposal_index": inst["proposal_index"],
        "class_name": inst["class_name"], "score": inst["score"], "n_mask_px": inst["n_mask_px"],
        "n_points_instance": int(len(rows)), "eps_m": round(eps, 5), "eps_source": eps_src,
        "min_samples": cfg["min_samples"], "cluster_space": "none:per_mask_stereo", "canonical_sort": "n/a",
        "cluster_tie_break": "n/a", "cloud_kind": "single_sweep", "frame": EGO,
        "num_lidar_pts_basis": "single_sweep_ground_filtered_pre_inflation", "num_lidar_pts": 0,
        "n_points_below_gate": len(rows) < cfg["min_samples"], "keyframe_token": lift_row["keyframe_token"],
        "t_ns": lift_row["t_ns"], "spec": STAGE_SPEC}
```

Status logic: channel not in `ZED_CHANNELS` → `status "out_of_r3", box None`; no prior → `"no_prior"`; `len(rows) == 0` → `"no_points"`; else `box, status, stereo = box_from_stereo(cloud[rows, :3], cloud[rows, 4], K=K, T_ego_cam=T, prior=prior, ground_abd=ground[token], cfg=cfg)`; `num_lidar_pts = stereo["n_lidar_in_box"] + stereo["n_stereo_in_box"]` when fit. Every row gets `"stereo": stereo` and `"box": box`, `"status": status`, `"cluster": None`.

`run(...)`: mirror Stage 6's manifest (`spec, stage, seed, config, upstream{metadata_fingerprint, fingerprint_spec, stage5_spec, stage5_degraded, stage5_degraded_causes, accepted_degraded_upstream, priors}, paths, frame, cloud_kind, box_fit{method: "per_mask_stereo", ...}, known_gaps[...], numpy_version, python_version, elapsed_s, scenes, totals`); totals: `n_instances, n_fit, n_out_of_r3, n_too_few_stereo, n_beyond_stereo_cap, n_no_points, n_no_prior, n_lidar_refined, n_clamped_w, n_clamped_h, n_isotropic_yaw, n_boxes_lidar_lt5` (boxes with `n_lidar_in_box < 5`). DEGRADED when a scene has `n_fit == 0` or `n_too_few_stereo > n_fit`; causes name the scene and the count. Upstream: `require_upstream(stage5_dir, stage_name="Stage 5", module_hint="pipeline.stage5_lift.lift", current_fingerprint=metadata_fingerprint(paths), accept_degraded=...)`, check `stage5["frame"] == EGO`, `load_priors` + fingerprint equality (copy Stage 6's two checks). `main`: argparse as in Interfaces, `clear_markers(out_dir)` first, write `scenes/<scene>/boxes.jsonl`, `run_manifest.json`, `write_marker(out_dir, fingerprint, degraded=..., causes=...)`, return exit code; `PathValidationError`/`UpstreamRefusal` → print and `EXIT_REFUSED`.

- [ ] **Step 4: Run tests**

Run: `PYTHONNOUSERSITE=1 $PY -m pytest tests/test_stereo_box.py -q -p no:cacheprovider`
Expected: 4 PASS. Tune nothing in the test; if a tolerance fails, fix the geometry — the synthetic answer is known.

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage6_stereo_box tests/test_stereo_box.py
git commit -m "stage6_stereo_box: per-mask boxes from ZED stereo on the two ZED frusta (approach A)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Wrapper step `6s`

**Files:**
- Modify: `scripts/run_stages.sh` (lines 389-390 step lists, 415 token case, ≈ 631 clean-slate dir list, ≈ 1022 `boxes_dir()`, the `1)` stage block ≈ 1062, add a `6s)` block after `6)`, ≈ 1482 summary dir list)

- [ ] **Step 1: Edit**

- `OPT_IN_STEPS=(3b 3f 3m 3c 6s cvatroad)`; token case: add `6s` to the alternation on line 415.
- After parsing STEPS: `if printf '%s\n' "${STEPS[@]}" | grep -qx 6 && printf '%s\n' "${STEPS[@]}" | grep -qx 6s; then echo "!!! steps 6 and 6s both requested: pick one box producer" >&2; exit 2; fi`.
- Env knobs near `MASK_MODEL_ID`: `STEREO_STRIDE="${STEREO_STRIDE:-}"`, `COVERAGE_CONFIG="${COVERAGE_CONFIG:-}"`, `STEREO_Z_CORR="${STEREO_Z_CORR:-}"` (space-separated `RING:M` items).
- In the `1)` block add to the ingest command: `${STEREO_STRIDE:+--stereo-stride "$STEREO_STRIDE"} ${COVERAGE_CONFIG:+--coverage-config "$COVERAGE_CONFIG"}` and `for zc in $STEREO_Z_CORR; do ING_ARGS+=(--stereo-z-correction "$zc"); done` passed as `${ING_ARGS[@]+"${ING_ARGS[@]}"}`.
- New block:

```bash
    6s) acc
        run_step "STAGE 6s (per-mask stereo boxes on the ZED frusta)" "$WORK_ROOT/stage6_stereo_box" \
          "$PY" -m pipeline.stage6_stereo_box.stereo_box --config configs/stereo_box.yaml \
            ${ACC[@]+"${ACC[@]}"} ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;
```

- `boxes_dir()`:

```bash
boxes_dir() {
  if [ "$(marker_state "$WORK_ROOT/stage7_track")" != none ]; then echo "$WORK_ROOT/stage7_track"; return; fi
  local s6="$WORK_ROOT/stage6_cluster" s6s="$WORK_ROOT/stage6_stereo_box"
  if [ "$(marker_state "$s6s")" != none ] && { [ "$(marker_state "$s6")" = none ] || [ "$s6s/run_manifest.json" -nt "$s6/run_manifest.json" ]; }; then
    echo "$s6s"; return
  fi
  echo "$s6"
}
```

- Add `stage6_stereo_box` to the dir lists at ≈ 631 and ≈ 1482 (next to `stage6_cluster`).

- [ ] **Step 2: Verify**

Run: `bash -n scripts/run_stages.sh && scripts/run_stages.sh --help | grep -c 6s`
Expected: syntax OK; the help mentions `6s` (add it to the header's opt-in list so `--help` documents it).

- [ ] **Step 3: Commit**

```bash
git add scripts/run_stages.sh
git commit -m "run_stages: opt-in step 6s (stereo boxes), boxes_dir prefers it, Stage 1 stereo/coverage env knobs

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 7: The viewer

**Files:**
- Create: `scripts/view_boxes_3d.py`, `viewer/index.html`
- Test: `tests/test_view_boxes_3d.py`

**Interfaces:**
- Consumes: `boxes.jsonl` rows from Task 5, Stage 1 `keyframes.jsonl` (`single_sweep_cloud.path`, `cameras[ch].path`, `.calibrated_sensor_token`), `Substrate.load(paths).by_token("calibrated_sensor.json")`, `read_pcd_bin`, `Transform`.
- Produces: `export_keyframe(kf_row, boxes_rows, calibs, out_dir, max_points) -> dict` (writes `kf/<i>.json`, returns the index entry); `--serve PORT` runs `http.server` on `out_dir`.

- [ ] **Step 1: Failing test**

```python
# tests/test_view_boxes_3d.py
from __future__ import annotations
import base64, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.view_boxes_3d import decimate, encode_cloud, project_corners  # noqa: E402


def test_decimate_and_encode_roundtrip():
    cloud = np.column_stack([np.arange(1000.0), np.zeros(1000), np.zeros(1000), np.ones(1000), np.full(1000, 101.0)]).astype(np.float32)
    d = decimate(cloud, 100)
    assert d.shape[0] <= 100 and d.shape[1] == 5
    enc = encode_cloud(d)
    xyz = np.frombuffer(base64.b64decode(enc["xyz_b64"]), dtype=np.float32).reshape(-1, 3)
    ring = np.frombuffer(base64.b64decode(enc["ring_b64"]), dtype=np.uint8)
    assert xyz.shape[0] == ring.shape[0] == enc["n"] and set(ring.tolist()) == {101}


def test_project_corners_in_front_of_camera_land_in_image():
    K = np.array([[953.16, 0, 656.28], [0, 953.16, 375.74], [0, 0, 1.0]])
    T_cam_ego = np.array([[0, -1.0, 0, 0], [0, 0, -1.0, -0.7], [1.0, 0, 0, -0.8], [0, 0, 0, 1.0]])  # ego -> optical, cam 0.8 ahead
    uv, vis = project_corners([12.0, 0.0, -1.5], [1.15, 2.4, 1.75], 0.3, K, T_cam_ego, (1280, 720))
    assert uv.shape == (8, 2) and vis.all()
    assert (uv[:, 0] > 0).all() and (uv[:, 0] < 1280).all() and (uv[:, 1] > 0).all() and (uv[:, 1] < 720).all()
```

- [ ] **Step 2: Run to confirm failure** — `ModuleNotFoundError: No module named 'scripts.view_boxes_3d'`.

- [ ] **Step 3: Implement `scripts/view_boxes_3d.py`**

```python
#!/usr/bin/env python3
"""Read-only 3D viewer exporter for stage6_stereo_box output (approach A, 2026-09-12).

Writes <out>/index.json + <out>/kf/<i>.json (decimated ego cloud as base64 float32,
boxes, the two ZED images' K and T_cam_ego, ground plane) and copies the two JPEGs
per keyframe to <out>/img/. `--serve PORT` serves <out> with http.server; open
http://localhost:PORT/ (viewer/index.html is copied to <out>/index.html).
"""
from __future__ import annotations
import argparse, base64, json, math, os, shutil, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.common.conventions import CAMERA, EGO, Transform  # noqa: E402
from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.stage0_data_probe.probe import Substrate  # noqa: E402
from pipeline.stage1_ingestion.ingest import read_pcd_bin  # noqa: E402
from scripts.render_boxes_3d import box_corners_ego  # noqa: E402

ZED = ("CAM_FRONT", "CAM_BACK")


def decimate(cloud: np.ndarray, max_points: int) -> np.ndarray:
    if len(cloud) <= max_points:
        return cloud
    idx = np.linspace(0, len(cloud) - 1, max_points).astype(int)   # deterministic, file order
    return cloud[idx]


def encode_cloud(cloud: np.ndarray) -> dict:
    xyz = np.ascontiguousarray(cloud[:, :3].astype(np.float32))
    ring = np.ascontiguousarray(np.clip(cloud[:, 4], 0, 255).astype(np.uint8))
    return {"n": int(len(cloud)), "xyz_b64": base64.b64encode(xyz.tobytes()).decode(),
            "ring_b64": base64.b64encode(ring.tobytes()).decode()}


def project_corners(center, size_wlh, yaw, K, T_cam_ego, image_size):
    """(8,2) pixel corners + (8,) visibility (z>0 and inside the image)."""
    h = yaw / 2.0
    corners = box_corners_ego(center, size_wlh, [math.cos(h), 0.0, 0.0, math.sin(h)])
    hom = np.column_stack([corners, np.ones(8)]) @ np.asarray(T_cam_ego).T
    z = hom[:, 2]; ok = z > 0.05
    uv = np.zeros((8, 2))
    uv[ok, 0] = K[0, 0] * hom[ok, 0] / z[ok] + K[0, 2]; uv[ok, 1] = K[1, 1] * hom[ok, 1] / z[ok] + K[1, 2]
    W, H = image_size
    vis = ok & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv, vis


def load_calibs(paths, kf_row):
    sub = Substrate.load(paths); cs = sub.by_token("calibrated_sensor.json")
    out = {}
    for ch in ZED:
        rec = cs[kf_row["cameras"][ch]["calibrated_sensor_token"]]
        T_ego_cam = Transform.from_nuscenes(rec, source_frame=CAMERA, parent_frame=EGO).matrix()
        out[ch] = {"K": np.array(rec["camera_intrinsic"]), "T_cam_ego": np.linalg.inv(T_ego_cam)}
    return out


def export_keyframe(i, kf_row, boxes_rows, calibs, ground, dataroot, out_dir, max_points) -> dict:
    cloud = decimate(read_pcd_bin(kf_row["single_sweep_cloud"]["path"]), max_points)
    cams = {}
    for ch in ZED:
        src = os.path.join(dataroot, kf_row["cameras"][ch]["path"]) if not os.path.isabs(kf_row["cameras"][ch]["path"]) else kf_row["cameras"][ch]["path"]
        dst = os.path.join(out_dir, "img", f"{i:05d}_{ch}.jpg"); os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            try: os.link(src, dst)
            except OSError: shutil.copy2(src, dst)
        cams[ch] = {"image": f"img/{i:05d}_{ch}.jpg", "K": calibs[ch]["K"].tolist(), "T_cam_ego": calibs[ch]["T_cam_ego"].tolist()}
    boxes = [{"instance_id": r["instance_id"], "channel": r["channel"], "class_name": r["class_name"], "score": r["score"],
              "status": r["status"], "stereo": r.get("stereo"), "box": r["box"]} for r in boxes_rows]
    payload = {"index": i, "token": kf_row["keyframe_token"], "t_ns": kf_row["t_ns"], "cloud": encode_cloud(cloud),
               "boxes": boxes, "cameras": cams, "ground": ground}
    os.makedirs(os.path.join(out_dir, "kf"), exist_ok=True)
    with open(os.path.join(out_dir, "kf", f"{i:05d}.json"), "w") as f:
        json.dump(payload, f)
    return {"index": i, "token": kf_row["keyframe_token"], "n_boxes": sum(1 for b in boxes if b["box"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    ap.add_argument("--boxes-dir", default=None, help="default <work_root>/stage6_stereo_box")
    ap.add_argument("--stage1-dir", default=None)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-points", type=int, default=80000)
    ap.add_argument("--serve", type=int, default=0, help="port; 0 = export only")
    a = ap.parse_args(argv)
    paths = load_paths(a.paths)
    boxes_dir = a.boxes_dir or os.path.join(paths.work_root, "stage6_stereo_box")
    stage1 = a.stage1_dir or os.path.join(paths.work_root, "stage1_ingestion")
    kfs = [json.loads(l) for l in open(os.path.join(stage1, "scenes", a.scene, "keyframes.jsonl")) if l.strip()]
    diag = json.load(open(os.path.join(stage1, "scenes", a.scene, "filter_diagnostics.json")))["keyframes"]
    ground = {d["keyframe_token"]: {k: d["ground_reference_plane"][k] for k in ("a", "b", "d")} for d in diag}
    by_kf: dict[str, list] = {}
    for l in open(os.path.join(boxes_dir, "scenes", a.scene, "boxes.jsonl")):
        if l.strip():
            r = json.loads(l); by_kf.setdefault(r["keyframe_token"], []).append(r)
    calibs = load_calibs(paths, kfs[0])
    os.makedirs(a.out, exist_ok=True)
    index = [export_keyframe(i, kf, by_kf.get(kf["keyframe_token"], []), calibs, ground[kf["keyframe_token"]],
                             paths.dataroot, a.out, a.max_points) for i, kf in enumerate(kfs)]
    json.dump({"scene": a.scene, "keyframes": index, "range_cap_m": 25.0}, open(os.path.join(a.out, "index.json"), "w"))
    shutil.copy2(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "viewer", "index.html"),
                 os.path.join(a.out, "index.html"))
    print(f"exported {len(index)} keyframes to {a.out}")
    if a.serve:
        os.chdir(a.out)
        print(f"serving http://localhost:{a.serve}/  (Ctrl-C to stop)")
        ThreadingHTTPServer(("0.0.0.0", a.serve), SimpleHTTPRequestHandler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Read `range_cap_m` from `configs/stereo_box.yaml` instead of the literal 25.0 (import yaml, same key).

- [ ] **Step 4: Implement `viewer/index.html`** — one file, Three.js 0.160 via importmap from `https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js` and `.../examples/jsm/controls/OrbitControls.js`. Required behaviour (write it fully; keep it under ~300 lines):
  - Load `index.json`; slider + `←`/`→` keys select a keyframe; fetch `kf/<i>.json`.
  - Points: `THREE.Points` from the decoded Float32Array; per-vertex colour by ring: rings 0–3 grey `#9aa0a6`, 101 (front ZED) warm `#ff9f43`, 100 (rear ZED) cool `#54a0ff`; checkboxes toggle LiDAR / stereo visibility (rebuild geometry from filtered arrays).
  - Boxes: for each row with `box`, 8 corners from (translation, size_wlh, yaw) — same formula as `box_corners_ego` (corners at ±l/2 along the heading axis, ±w/2 across, z from z_min to z_max) — drawn as `THREE.LineSegments` (12 edges) coloured by class (fixed palette keyed by `class_name`), plus a short `THREE.ArrowHelper` from the centre along +heading; `status != "fit"` rows are not drawn but counted in the side panel.
  - Ego axes (`THREE.AxesHelper(2)`), a ring of radius `range_cap_m` at ground height, a light grid.
  - Two `<canvas>` panels (CAM_FRONT, CAM_BACK): draw the JPEG, then for each box project the 8 corners with `T_cam_ego` and `K` (skip corners with z ≤ 0.05), draw the 12 edges in the class colour and the class label at the top-left visible corner.
  - Side panel: keyframe token, counts by status, and a clickable list of boxes (class, channel, `stereo.depth_source`, `d_med_m`, `n_stereo_kept`, `n_lidar_in_box`, `clamped_axes`); clicking a row highlights its wireframe (thicker/white) in 3D and on the image panels.
  - Camera: `OrbitControls`, default view from behind-above the ego looking forward; `R` resets.

- [ ] **Step 5: Run tests**

Run: `PYTHONNOUSERSITE=1 $PY -m pytest tests/test_view_boxes_3d.py -q -p no:cacheprovider`
Expected: 2 PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/view_boxes_3d.py viewer/index.html tests/test_view_boxes_3d.py
git commit -m "viewer: read-only Three.js viewer for stereo boxes (cloud + wireframes + both ZED images with projected boxes)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 8: Run the chain on chunk_0010 and export the viewer

**Files:** none new (evidence in Task 9).

- [ ] **Step 1: Run the chain in tmux so the operator can watch**

```bash
tmux kill-session -t pipe 2>/dev/null; tmux new-session -d -s pipe -x 200 -y 50 -c /home/saif/pipeline/wt-stereo-box-a
tmux set-option -t pipe history-limit 200000
tmux send-keys -t pipe 'export PYTHONNOUSERSITE=1 PY=/home/saif/miniconda3/envs/ano_pipe/bin/python STEREO_STRIDE=1 COVERAGE_CONFIG=R3 CHUNK=dhaka_20260911_141259_chunk_0010' C-m
tmux send-keys -t pipe 'time scripts/run_stages.sh 1 3 3f 3m 4 5 6s --scenes "$CHUNK" --no-cvat' C-m
```

If Task 4 produced a z correction, prepend `STEREO_Z_CORR="100:0.69"` (the actual values) to the export line. Stage 1 here re-runs with the wrapper's knobs (it replaces the Task 3 tree — same inputs, same result).

Wait for `=== ALL_STEPS_DONE` (Stage 4 ≈ 30 min on 668 keyframes; the rest minutes). Every stage must end `OK` or `DEGRADED`; a `REFUSED`/`CRASHED` stops the plan — report it with the log path `/mnt/hdd/dhakascenes/work_zami/20260911_zed/logs/run_*.log`.

- [ ] **Step 2: Export and serve the viewer**

```bash
set -a && . ./.env && set +a
PYTHONNOUSERSITE=1 $PY scripts/view_boxes_3d.py --scene dhaka_20260911_141259_chunk_0010 \
  --out /mnt/hdd/dhakascenes/viewer_zami/chunk_0010 --serve 8765
```

Report the URL `http://localhost:8765/` and the per-stage timing table the wrapper prints.

---

### Task 9: GT-free evaluation → evidence doc

**Files:**
- Create: `scripts/eval_stereo_box.py`, `docs/evidence/2026-09-12-stereo-box-a-chunk_0010.md`

**Interfaces:**
- Consumes: `boxes.jsonl` (Task 5), Stage 5 `lift.jsonl` (`mask_path`), `MaskFile`, `project_corners` (Task 7), `cv2.fillConvexPoly` (opencv-python-headless is pinned).
- Produces: per-scene metrics JSON + markdown.

- [ ] **Step 1: Write the script**

For every `fit` row: project the 8 corners into its own camera (`project_corners`), take the convex hull (`cv2.convexHull`) of the visible corners, rasterise it with `cv2.fillConvexPoly` onto a 720×1280 uint8, load the mask via `MaskFile(mask_path).mask(channel, proposal_index)`, IoU = `|hull & mask| / |hull | mask|`. Report: status histogram; `yaw_ambiguous_reasons` histogram; `clamped_axes` rate per axis; `depth_source` split; median/p10/p90 of reprojection IoU per class and overall; fraction with `n_lidar_in_box < 5`; median `d_med_m`; runtime from `run_manifest.json.elapsed_s`. Write JSON + a markdown table. Name the caveat at the top: *no ground truth exists on this substrate; these are consistency signals, not accuracy.*

- [ ] **Step 2: Run it**

```bash
PYTHONNOUSERSITE=1 $PY scripts/eval_stereo_box.py --scene dhaka_20260911_141259_chunk_0010 \
  --out-md docs/evidence/2026-09-12-stereo-box-a-chunk_0010.md --out-json docs/evidence/2026-09-12-stereo-box-a-chunk_0010.json
```

- [ ] **Step 3: Full suite, then commit**

Run: `PYTHONNOUSERSITE=1 $PY -m pytest tests -q -p no:cacheprovider | tail -2`
Expected: 616 + (3 + 5 + 4 + 2 + 3) new tests passed, the one environmental failure only.

```bash
git add scripts/eval_stereo_box.py docs/evidence/2026-09-12-stereo-box-a-chunk_0010.md docs/evidence/2026-09-12-stereo-box-a-chunk_0010.json
git commit -m "eval: GT-free metrics for stereo boxes on chunk_0010 (status, yaw, clamps, reprojection IoU)

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

## Self-review notes (done while writing)

- Spec coverage: §3.1 → Task 3; §3.2 → Task 2; §3.3 → Task 1; §3.4 → Task 4; §4 → Tasks 5–6; §5 → Task 7; §6 → Task 8; §7 → Task 9; §8 → each task's tests.
- Type consistency: `box_from_stereo(pts_ego, rings, *, K, T_ego_cam, prior, ground_abd, cfg)` is called with the same keyword names in Task 5's tests and driver; `project_corners(center, size_wlh, yaw, K, T_cam_ego, image_size)` is shared by Tasks 7 and 9; `stereo_block_to_ego(raw, *, frame, ring, t_sensor_to_ego, t_global_to_ego, z_correction_m)` matches its test; `STEREO_CHANNELS` entries are `{"frame", "ring"}` everywhere.
- One deliberate deviation from the spec text: the viewer shows the `stereo` block in a clickable side list rather than on hover (same information, simpler, keyboard-friendly).
