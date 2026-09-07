# Benchmark Release Requirements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every `export/day1_chunk_NNNN/` release satisfy the benchmark's per-delivery checklist (accepted-only tables, 50 m range, attributes, persistent instances, stratified 5 % double annotation with a CVAT round trip, delivery note, checker) without changing Stages 1–9 beyond the range cap.

**Architecture:** One two-constant edit raises the pipeline range cap to 50 m. Everything else is export-time post-processing in a new pure-function package `pipeline/release/` that `scripts/export_release.py` orchestrates in a fixed order (stitch → human merge → tier filter → chains → attributes → strata/double → tables → note → check). The CVAT round trip adds `frames.json` + attributes to the 3D export, an A/B publish, and a new importer that writes I-5 records.

**Tech Stack:** Python 3.10 (`/home/mt/miniconda3/envs/ano_pipe/bin/python`), numpy 1.26, scipy 1.14 (`linear_sum_assignment`), PIL 10.4, PyYAML, cvat_sdk 2.73, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-07-benchmark-release-requirements-design.md`

## Global Constraints

- Interpreter for every command: `PY=/home/mt/miniconda3/envs/ano_pipe/bin/python`. System `python3` cannot import the suite.
- Run the suite as `$PY -m pytest tests/ -q` from the repo root; it must stay green after every task (365 tests at start).
- No edits under `pipeline/stage*` except Task 0's two constants. No edits to `configs/release_category_map.yaml`.
- Every tunable lives in `configs/release.yaml` with a `source` comment; no literal thresholds in `pipeline/release/*.py`.
- New per-row output fields use the `dhakascenes_` prefix except the contract-named ones: `annotator_pass`, `is_uncertain`, `is_uncertain_reason`, and `dbench_double_annotated` on `sample` rows.
- Sizes are `[w, l, h]`; quaternions `[w, x, y, z]`; records crossing a stage boundary are ego frame; the exporter's tables are global frame.
- Tests are pytest files under `tests/`, `from __future__ import annotations`, `ROOT` sys.path insertion as in `tests/test_stage3_merge_parts.py`; synthetic inputs only — never read `/home/mt/dhakascenes` or `export/` in unit tests.
- Commit after each task with a message body naming the spec section; end with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`. Do not push.
- The user's rules from memory apply: never delete CVAT tasks or exports; fixes are general, never per-chunk.

## File Structure

| Path | Responsibility |
|---|---|
| `pipeline/stage1_ingestion/ingest.py:183,228`, `pipeline/common/eval_region.py:48-60` | Task 0: range constants + provenance |
| `configs/release.yaml` (new) | every release tunable with source |
| `pipeline/release/__init__.py` (new) | package marker |
| `pipeline/release/config.py` (new) | `ReleaseConfig` loader / validator |
| `pipeline/release/geometry.py` (new) | tokens, quaternion helpers, ego↔global, corners, slerp, points-in-box — moved out of `export_release.py` |
| `pipeline/release/frames.py` (new) | `SceneFrames`: ordered keyframes, timestamps, poses; `CloudSource`: PCD-from-task.zip / raw fallback |
| `pipeline/release/stitch.py` (new) | fragments → chains, interpolation |
| `pipeline/release/attributes.py` (new) | chain velocity, attribute state, nuScenes names |
| `pipeline/release/tiers.py` (new) | included / excluded partition with reasons |
| `pipeline/release/human.py` (new) | I-5 load, precedence, superseding |
| `pipeline/release/strata.py` (new) | density, luma, bins |
| `pipeline/release/double.py` (new) | stratified selection, frozen reuse |
| `pipeline/release/note.py` (new) | `DELIVERY_NOTE.md` |
| `pipeline/common/schemas.py` | Task 7: `annotator_pass`, `is_uncertain*`, human-only `attribute` |
| `scripts/export_release.py` | orchestration, flags, tables, sidecars, `stitch_map.json`, meta |
| `scripts/check_release.py` (new) | checklist validator |
| `scripts/export_cvat_3d.py` | `frames.json`, `--stitch-map`, `--frames/--blank`, cuboid attributes/tracks |
| `scripts/cvat_setup_3d.py` | label attributes, `--which double`, ledger |
| `scripts/import_cvat_3d.py` (new) | CVAT → I-5 `verified.jsonl` |
| `scripts/run_day1_chunks.sh` | reordered tail, `--phase human-import` |
| `tests/test_release_*.py`, `tests/test_check_release.py`, `tests/test_import_cvat_3d.py`, `tests/test_export_cvat_3d_frames.py` | per-task tests |

Row shape shared by every `pipeline/release` module — an I-4/I-5 record as a dict, exactly as `load_prelabels()` returns it (`token`, `sample_token`, `instance_token`, `category`, `frame`, `t_ns`, `translation_m`, `size_wlh_m`, `rotation_wxyz`, `num_lidar_pts`, `num_lidar_pts_basis`, `provenance{source,tier,gates,verified_by,verification_pass[,annotator_pass]}`, `velocity_mps`, `track_id`, `attribute`, `visibility`, optional `is_uncertain`, `is_uncertain_reason`), plus the keys each module documents it adds. Modules mutate and return the same dicts.

---

### Task 0: Range cap to 50 m

**Files:**
- Modify: `pipeline/stage1_ingestion/ingest.py:183` and `:228`
- Modify: `pipeline/common/eval_region.py:48-60`
- Test: `tests/test_range_cap_50m.py`

**Interfaces:**
- Produces: `IngestConfig().range_cap_m == 50.0`, `eval_region._R_MAX_M == 50.0`, `RegionSpec("R2").r_max_m == 50.0`.

- [ ] **Step 1: Write the failing test**

```python
"""Operator decision 2026-09-07: the pipeline annotates to the benchmark's
evaluation range (class_range 50/40/30 m); the Stage 9 point floor decides
what survives. Both caps must agree or Stage 1 prunes to one radius while
Stage 5/6 score against another."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common import eval_region  # noqa: E402
from pipeline.stage1_ingestion.ingest import IngestConfig  # noqa: E402


def test_stage1_cap_is_50m():
    assert IngestConfig().range_cap_m == 50.0


def test_eval_region_cap_is_50m():
    assert eval_region._R_MAX_M == 50.0
    assert eval_region.region_spec_from_config({"coverage_config": "R2"}).r_max_m == 50.0


def test_the_two_caps_agree():
    assert IngestConfig().range_cap_m == eval_region._R_MAX_M


def test_provenance_names_the_decision():
    assert "2026-09-07" in IngestConfig().provenance["range_cap_m"]
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_range_cap_50m.py -v`
Expected: 3 failures (`30.0 != 50.0`, provenance string).

- [ ] **Step 3: Edit the two constants and their provenance**

`pipeline/stage1_ingestion/ingest.py` line 183: `range_cap_m: float = 50.0`.
Line 228 provenance value becomes:

```python
            "range_cap_m": "50 m, operator decision 2026-09-07: annotate to the benchmark's evaluation range (class_range 50/40/30 m). The Stage 9 point floor (>= 5 returns) decides what survives; the delivery note reports the effective per-class range. Must match eval_region._R_MAX_M or Stage 1 prunes to one radius while Stage 5/6 score against another. Was 30 m (human-directed 2026-08-30); runs before and after are not comparable.",
```

`pipeline/common/eval_region.py`: replace the comment block above `_R_MAX_M` (lines 48–57) with:

```python
# --- inherited values, flagged (§10) ---------------------------------------
# 50 m range cap: operator decision 2026-09-07 — annotate to the benchmark's
#   evaluation range (benchmark_v1.0.yaml class_range 50/40/30 m). The Stage 9
#   point floor (>= 5 single-sweep returns) decides which far boxes survive and
#   the release's delivery note reports the effective per-class range. History:
#   spec §3.6 gave 40 m (Mid-360 10 %-reflectivity range); cut to 30 m on
#   2026-08-30 because the ZED densification caps at 20 m and boxes beyond
#   ~30 m were lidar-sparse. Runs before and after each change are NOT
#   comparable: E is the denominator of every density metric.
# 30 m rho radius: spec §8.3.1, verbatim.
# 55 deg R1 half-width: spec §3.6, verbatim.
_R_MAX_M = 50.0
```

- [ ] **Step 4: Run the whole suite; fix any test that encoded 30 m**

Run: `$PY -m pytest tests/ -q`
Expected: green. If a test asserts `30.0` for `range_cap_m` / `r_max_m` (search `grep -rn "30\.0" tests/ | grep -iE "range|r_max"`), update its expected value to `50.0` and its docstring to cite the 2026-09-07 decision — do not weaken the assertion.

- [ ] **Step 5: Commit**

```bash
git add pipeline/stage1_ingestion/ingest.py pipeline/common/eval_region.py tests/test_range_cap_50m.py
git commit -m "Range cap 30 m -> 50 m in Stage 1 and eval_region (spec §1)"
```

---

### Task 1: `configs/release.yaml` + `ReleaseConfig`

**Files:**
- Create: `configs/release.yaml`
- Create: `pipeline/release/__init__.py` (empty)
- Create: `pipeline/release/config.py`
- Test: `tests/test_release_config.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass(frozen=True)
  class ReleaseConfig:
      path: str; sha256: str; benchmark_source: dict
      stitch: StitchConfig; attributes: AttributeConfig; strata: StrataConfig; double: DoubleConfig
  @dataclass(frozen=True) class StitchConfig: max_gap_keyframes: int; base_gate_m: float; gap_slack_m: float; size_ratio_max: float; class_agnostic: bool
  @dataclass(frozen=True) class AttributeConfig: moving_speed_threshold_mps: float; max_time_diff_s: float
  @dataclass(frozen=True) class StrataConfig: density_radius_m: float; density_quantiles: list[float]; illumination_channel: str; illumination_saturation_ignore_above: int; illumination_bin_edges: list[float]; illumination_bin_names: list[str]; density_bin_names: list[str]
  @dataclass(frozen=True) class DoubleConfig: fraction: float; seed: int
  def load_release_config(path: str) -> ReleaseConfig   # raises ReleaseConfigError listing every problem
  ```

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.release.config import ReleaseConfigError, load_release_config  # noqa: E402

DEFAULT = os.path.join(ROOT, "configs", "release.yaml")


def test_default_config_loads_with_benchmark_values():
    cfg = load_release_config(DEFAULT)
    assert cfg.stitch.max_gap_keyframes == 3
    assert cfg.stitch.base_gate_m == 2.0
    assert cfg.attributes.moving_speed_threshold_mps == 0.5
    assert cfg.strata.illumination_bin_edges == [45.0, 75.0, 95.0]
    assert cfg.strata.illumination_bin_names == ["dark", "night", "dusk", "day"]
    assert cfg.strata.density_bin_names == ["Low", "Medium", "High", "Extreme"]
    assert cfg.double.fraction == 0.05 and cfg.double.seed == 20260812
    assert len(cfg.sha256) == 64
    assert cfg.benchmark_source["path"].endswith("benchmark_v1.0.yaml")


def test_bad_values_are_all_reported(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(
        "spec: dhakascenes/release_config/v1\n"
        "benchmark_source: {path: x, sha256: y}\n"
        "stitch: {max_gap_keyframes: 0, base_gate_m: -1, gap_slack_m: 1, size_ratio_max: 0.5, class_agnostic: false}\n"
        "attributes: {moving_speed_threshold_mps: 0.5, max_time_diff_s: 1.5}\n"
        "strata: {density_radius_m: 30, density_quantiles: [0.25, 0.5, 0.75], illumination_channel: CAM_FRONT,\n"
        "  illumination_saturation_ignore_above: 250, illumination_bin_edges: [45, 75], illumination_bin_names: [a, b, c, d],\n"
        "  density_bin_names: [Low, Medium, High, Extreme]}\n"
        "double: {fraction: 1.5, seed: 1}\n")
    with pytest.raises(ReleaseConfigError) as exc:
        load_release_config(str(p))
    msg = str(exc.value)
    for needle in ("max_gap_keyframes", "base_gate_m", "size_ratio_max", "illumination_bin_edges", "fraction"):
        assert needle in msg


def test_unknown_key_is_an_error(tmp_path):
    p = tmp_path / "r.yaml"
    p.write_text(open(DEFAULT).read() + "\nextra_key: 1\n")
    with pytest.raises(ReleaseConfigError):
        load_release_config(str(p))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_release_config.py -v`
Expected: `ModuleNotFoundError: pipeline.release`.

- [ ] **Step 3: Write the config file**

`configs/release.yaml` (compute the sha with `sha256sum /home/mt/dataset_benchmark/configs/benchmark_v1.0.yaml` and paste it):

```yaml
# DhakaScenes release post-processing — every tunable scripts/export_release.py
# and pipeline/release/* read, with its source. Values copied from the benchmark
# definition are pinned here (the benchmark file's digest is recorded below and
# printed in DELIVERY_NOTE.md) so a release is reproducible from this repo alone.
# Spec: docs/superpowers/specs/2026-09-07-benchmark-release-requirements-design.md
spec: dhakascenes/release_config/v1
benchmark_source:
  path: /home/mt/dataset_benchmark/configs/benchmark_v1.0.yaml
  sha256: "<paste sha256sum output here>"
stitch:
  max_gap_keyframes: 3        # 1.2 s at the capture's 2.5 Hz keyframes (measured 2026-09-07)
  base_gate_m: 2.0            # benchmark tracking.dist_th (center distance)
  gap_slack_m: 1.0            # extra metres of gate per missing keyframe beyond the first
  size_ratio_max: 2.0         # BEV footprint area ratio a join tolerates
  class_agnostic: false       # a class flip along one object stays two instances (spec §3)
attributes:
  moving_speed_threshold_mps: 0.5   # benchmark detection_nuscenes.velocity_split.moving_speed_threshold_mps
  max_time_diff_s: 1.5              # nuScenes devkit box_velocity() default
strata:
  density_radius_m: 30.0            # benchmark stratification.density.radius_m
  density_quantiles: [0.25, 0.5, 0.75]   # benchmark stratification.density.quantiles
  density_bin_names: [Low, Medium, High, Extreme]
  illumination_channel: CAM_FRONT   # benchmark stratification.illumination.source_channel
  illumination_saturation_ignore_above: 250
  illumination_bin_edges: [45.0, 75.0, 95.0]   # benchmark stratification.illumination.bin_edges (PROVISIONAL there too)
  illumination_bin_names: [dark, night, dusk, day]
double:
  fraction: 0.05              # contract §6: at least 5 % of frames
  seed: 20260812              # the pipeline's global seed (configs/pipeline_pilot.yaml)
```

- [ ] **Step 4: Write the loader**

`pipeline/release/config.py`:

```python
"""Release post-processing config: one loader, every problem reported at once."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, fields

import yaml

SPEC = "dhakascenes/release_config/v1"


class ReleaseConfigError(ValueError):
    pass


@dataclass(frozen=True)
class StitchConfig:
    max_gap_keyframes: int
    base_gate_m: float
    gap_slack_m: float
    size_ratio_max: float
    class_agnostic: bool


@dataclass(frozen=True)
class AttributeConfig:
    moving_speed_threshold_mps: float
    max_time_diff_s: float


@dataclass(frozen=True)
class StrataConfig:
    density_radius_m: float
    density_quantiles: list
    density_bin_names: list
    illumination_channel: str
    illumination_saturation_ignore_above: int
    illumination_bin_edges: list
    illumination_bin_names: list


@dataclass(frozen=True)
class DoubleConfig:
    fraction: float
    seed: int


@dataclass(frozen=True)
class ReleaseConfig:
    path: str
    sha256: str
    benchmark_source: dict
    stitch: StitchConfig
    attributes: AttributeConfig
    strata: StrataConfig
    double: DoubleConfig

    def as_dict(self) -> dict:
        return {
            "path": self.path, "sha256": self.sha256, "benchmark_source": dict(self.benchmark_source),
            "stitch": vars(self.stitch), "attributes": vars(self.attributes),
            "strata": vars(self.strata), "double": vars(self.double),
        }


_SECTIONS = {"stitch": StitchConfig, "attributes": AttributeConfig,
             "strata": StrataConfig, "double": DoubleConfig}


def _build(section: str, cls, raw: dict, errors: list[str]):
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    missing = sorted(known - set(raw))
    if unknown:
        errors.append(f"{section}: unknown key(s) {unknown}")
    if missing:
        errors.append(f"{section}: missing key(s) {missing}")
    if unknown or missing:
        return None
    return cls(**raw)


def _check(cfg: ReleaseConfig, errors: list[str]) -> None:
    s, a, t, d = cfg.stitch, cfg.attributes, cfg.strata, cfg.double
    if s.max_gap_keyframes < 1:
        errors.append(f"stitch.max_gap_keyframes={s.max_gap_keyframes} must be >= 1")
    if s.base_gate_m <= 0:
        errors.append(f"stitch.base_gate_m={s.base_gate_m} must be > 0")
    if s.gap_slack_m < 0:
        errors.append(f"stitch.gap_slack_m={s.gap_slack_m} must be >= 0")
    if s.size_ratio_max < 1.0:
        errors.append(f"stitch.size_ratio_max={s.size_ratio_max} must be >= 1")
    if a.moving_speed_threshold_mps <= 0:
        errors.append("attributes.moving_speed_threshold_mps must be > 0")
    if a.max_time_diff_s <= 0:
        errors.append("attributes.max_time_diff_s must be > 0")
    if t.density_radius_m <= 0:
        errors.append("strata.density_radius_m must be > 0")
    if len(t.density_quantiles) + 1 != len(t.density_bin_names):
        errors.append("strata.density_quantiles must have one fewer entry than density_bin_names")
    if len(t.illumination_bin_edges) + 1 != len(t.illumination_bin_names):
        errors.append("strata.illumination_bin_edges must have one fewer entry than illumination_bin_names")
    if list(t.illumination_bin_edges) != sorted(t.illumination_bin_edges):
        errors.append("strata.illumination_bin_edges must be ascending")
    if not 0.0 <= d.fraction <= 1.0:
        errors.append(f"double.fraction={d.fraction} must be in [0, 1]")


def load_release_config(path: str) -> ReleaseConfig:
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    raw = yaml.safe_load(text) or {}
    errors: list[str] = []
    if raw.get("spec") != SPEC:
        errors.append(f"spec={raw.get('spec')!r}, expected {SPEC!r}")
    top_known = {"spec", "benchmark_source", *_SECTIONS}
    unknown = sorted(set(raw) - top_known)
    if unknown:
        errors.append(f"unknown top-level key(s) {unknown}")
    src = raw.get("benchmark_source") or {}
    if not isinstance(src, dict) or "path" not in src or "sha256" not in src:
        errors.append("benchmark_source needs path and sha256")
    built = {}
    for name, cls in _SECTIONS.items():
        sec = raw.get(name)
        if not isinstance(sec, dict):
            errors.append(f"{name}: section missing")
            continue
        built[name] = _build(name, cls, sec, errors)
    if errors or any(v is None for v in built.values()):
        raise ReleaseConfigError(f"{path}:\n  - " + "\n  - ".join(errors))
    cfg = ReleaseConfig(path=path, sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        benchmark_source=dict(src), **built)
    _check(cfg, errors)
    if errors:
        raise ReleaseConfigError(f"{path}:\n  - " + "\n  - ".join(errors))
    return cfg
```

- [ ] **Step 5: Run tests, then the suite**

Run: `$PY -m pytest tests/test_release_config.py -v && $PY -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add configs/release.yaml pipeline/release/__init__.py pipeline/release/config.py tests/test_release_config.py
git commit -m "Release config: configs/release.yaml + ReleaseConfig loader (spec §2)"
```

---

### Task 2: `pipeline/release/geometry.py` — move the exporter's geometry helpers

**Files:**
- Create: `pipeline/release/geometry.py`
- Modify: `scripts/export_release.py:138-142, 315-363` (delete the moved bodies, import them)
- Test: `tests/test_release_geometry.py`

**Interfaces:**
- Produces (all previously in `export_release.py`, same names and behaviour): `make_token(*parts) -> str`, `quat_multiply(a, b)`, `normalise_quat(q)`, `box_ego_to_global(t, q, pose) -> (list, list)`, `box_global_to_ego(t, q, pose) -> (list, list)`, `box_corners_ego(t, size_wlh, q) -> np.ndarray(8,3)`.
- New: `slerp(q0, q1, s: float) -> np.ndarray` (shortest arc, unit output), `yaw_of(q_wxyz) -> float`, `points_in_box(points_xyz: np.ndarray, translation, size_wlh, rotation_wxyz) -> int` (points strictly inside the oriented box; `points_xyz` is (N, ≥3), uses the first three columns).
- `ExportError` stays in `export_release.py`; `geometry.py` raises `ValueError` for a degenerate quaternion and the exporter wraps it.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.conventions import EGO, NUSCENES_GLOBAL, Transform, quaternion_from_yaw_rad  # noqa: E402
from pipeline.release.geometry import (  # noqa: E402
    box_corners_ego, box_ego_to_global, box_global_to_ego, make_token, normalise_quat,
    points_in_box, slerp, yaw_of,
)


def _pose(x, y, yaw):
    return Transform.from_nuscenes({"translation": [x, y, 0.0], "rotation": list(quaternion_from_yaw_rad(yaw))},
                                   source_frame=EGO, parent_frame=NUSCENES_GLOBAL)


def test_ego_global_round_trip():
    pose = _pose(100.0, -20.0, 0.7)
    t, q = [5.0, 1.0, -1.5], list(quaternion_from_yaw_rad(0.3))
    tg, qg = box_ego_to_global(t, q, pose)
    te, qe = box_global_to_ego(tg, qg, pose)
    assert np.allclose(te, t, atol=1e-9)
    assert np.allclose(qe, normalise_quat(np.asarray(q)), atol=1e-9)


def test_make_token_is_deterministic_32_hex():
    a, b = make_token("instance", "s", "1"), make_token("instance", "s", "1")
    assert a == b and len(a) == 32 and int(a, 16) >= 0
    assert make_token("instance", "s", "2") != a


def test_slerp_endpoints_and_shortest_arc():
    q0 = np.asarray(quaternion_from_yaw_rad(0.0))
    q1 = np.asarray(quaternion_from_yaw_rad(math.radians(170)))
    assert np.allclose(slerp(q0, q1, 0.0), q0)
    assert abs(yaw_of(slerp(q0, q1, 1.0)) - math.radians(170)) < 1e-9
    mid = yaw_of(slerp(q0, q1, 0.5))
    assert abs(mid - math.radians(85)) < 1e-9
    # 350 deg is -10 deg: the short way round passes through -5, not 175
    q2 = np.asarray(quaternion_from_yaw_rad(math.radians(-10)))
    assert abs(yaw_of(slerp(q0, q2, 0.5)) - math.radians(-5)) < 1e-9


def test_points_in_box_counts_only_inside_oriented_box():
    yaw = math.radians(90)
    q = list(quaternion_from_yaw_rad(yaw))
    # width 1 (across heading), length 4 (along heading, now along +y), height 2
    pts = np.array([[0.0, 1.9, 0.0], [0.0, 2.1, 0.0], [0.4, 0.0, 0.9], [0.6, 0.0, 0.0], [0.0, 0.0, 1.1]])
    assert points_in_box(pts, [0.0, 0.0, 0.0], [1.0, 4.0, 2.0], q) == 2


def test_corners_match_size_order():
    c = box_corners_ego([0, 0, 0], [1.0, 4.0, 2.0], list(quaternion_from_yaw_rad(0.0)))
    assert c[:, 0].max() - c[:, 0].min() == pytest.approx(4.0)   # length along heading (+x)
    assert c[:, 1].max() - c[:, 1].min() == pytest.approx(1.0)   # width across
    assert c[:, 2].max() - c[:, 2].min() == pytest.approx(2.0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_release_geometry.py -v`
Expected: `ModuleNotFoundError: pipeline.release.geometry`.

- [ ] **Step 3: Create `geometry.py`**

Move `make_token`, `quat_multiply`, `normalise_quat`, `box_ego_to_global`, `box_global_to_ego`, `box_corners_ego` verbatim from `scripts/export_release.py` (replace `raise ExportError(...)` in `normalise_quat` with `raise ValueError(...)`), then add:

```python
def yaw_of(q_wxyz) -> float:
    return yaw_rad_from_quaternion([float(v) for v in q_wxyz])


def slerp(q0, q1, s: float) -> np.ndarray:
    """Shortest-arc spherical interpolation of unit [w,x,y,z] quaternions."""
    a = normalise_quat(np.asarray(q0, dtype=np.float64))
    b = normalise_quat(np.asarray(q1, dtype=np.float64))
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b, dot = -b, -dot
    if dot > 1.0 - 1e-9:
        return normalise_quat(a + s * (b - a))
    theta = math.acos(min(1.0, dot))
    sin_t = math.sin(theta)
    return normalise_quat((math.sin((1.0 - s) * theta) / sin_t) * a + (math.sin(s * theta) / sin_t) * b)


def points_in_box(points_xyz: np.ndarray, translation, size_wlh, rotation_wxyz) -> int:
    """Count points strictly inside the oriented box (size [w, l, h]: l along heading)."""
    if points_xyz.size == 0:
        return 0
    R = quaternion_to_rotation_matrix([float(v) for v in rotation_wxyz])
    local = (np.asarray(points_xyz[:, :3], dtype=np.float64) - np.asarray(translation, dtype=np.float64)) @ R
    w, l, h = (float(v) for v in size_wlh)
    inside = (np.abs(local[:, 0]) < l / 2) & (np.abs(local[:, 1]) < w / 2) & (np.abs(local[:, 2]) < h / 2)
    return int(inside.sum())
```

Imports at the top of `geometry.py`: `hashlib`, `math`, `numpy as np`, and from `pipeline.common.conventions`: `Transform`, `apply_transform`, `quaternion_to_rotation_matrix`, `yaw_rad_from_quaternion`. `normalise_quat` sign convention (w ≥ 0) is kept.

- [ ] **Step 4: Point `export_release.py` at the module**

Delete the moved function bodies from `scripts/export_release.py` and add after the `conventions` import block:

```python
from pipeline.release.geometry import (  # noqa: E402
    box_corners_ego, box_ego_to_global, box_global_to_ego, make_token, normalise_quat, quat_multiply,
)
```

Keep `sha256_of`, `verify_with_devkit`, `ExportError` where they are.

- [ ] **Step 5: Run the new test and the whole suite**

Run: `$PY -m pytest tests/test_release_geometry.py -v && $PY -m pytest tests/ -q`
Expected: PASS; any existing exporter test still passes.

- [ ] **Step 6: Commit**

```bash
git add pipeline/release/geometry.py scripts/export_release.py tests/test_release_geometry.py
git commit -m "pipeline/release/geometry: exporter geometry helpers + slerp, points_in_box (spec §2)"
```

---

### Task 3: `pipeline/release/frames.py` — scene keyframes and cloud source

**Files:**
- Create: `pipeline/release/frames.py`
- Test: `tests/test_release_frames.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass
  class SceneFrames:
      scene_token: str
      scene_name: str
      tokens: list[str]                 # sample tokens in timestamp order
      timestamps_ns: list[int]          # same order (sample.timestamp is µs in nuScenes; stored here as ns)
      poses: dict[str, Transform]       # sample_token -> LiDAR ego_pose (ego -> global)
      index: dict[str, int]             # sample_token -> position in `tokens`
      def dt_s(self, i: int, j: int) -> float          # (timestamps[j] - timestamps[i]) / 1e9
  def scene_frames_from_root(root, scene_token: str) -> SceneFrames
      # `root` is export_release.SourceRoot (duck-typed: .sample, .scene, .lidar_ego_pose(), .tables["sample"])
  class CloudSource:
      def __init__(self, dataroot: str, cvat_export_3d_dir: str | None, frames: SceneFrames, root)
      def points(self, sample_token: str) -> tuple[np.ndarray | None, str]
          # (xyz (N,3) float64 or None, basis) — basis "single_sweep_ground_filtered_pre_inflation" when
          # read from <cvat_export_3d_dir>/<scene_name>/task.zip pointcloud/<NNNNNN>.pcd via frames.json,
          # "single_sweep_raw" when read from the dataroot LIDAR_TOP .pcd.bin, (None, "unavailable") otherwise
      def close(self) -> None
  def read_pcd_v07_binary(data: bytes) -> np.ndarray        # (N, 4) x y z intensity, the layout export_cvat_3d.write_pcd writes
  ```
- Consumes: `pipeline.stage1_ingestion.ingest.read_pcd_bin`, `Transform`.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import io
import json
import os
import sys
import zipfile

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.conventions import EGO, NUSCENES_GLOBAL, Transform, quaternion_from_yaw_rad  # noqa: E402
from pipeline.release.frames import CloudSource, SceneFrames, read_pcd_v07_binary, scene_frames_from_root  # noqa: E402


class FakeRoot:
    """The slice of export_release.SourceRoot the module reads."""

    def __init__(self):
        self.scene = {"sc": {"token": "sc", "name": "chunk_test"}}
        self.tables = {"sample": [
            {"token": "s2", "scene_token": "sc", "timestamp": 2_400_000},
            {"token": "s1", "scene_token": "sc", "timestamp": 2_000_000},
            {"token": "s3", "scene_token": "sc", "timestamp": 2_800_000},
            {"token": "other", "scene_token": "sc2", "timestamp": 1},
        ]}
        self.sample = {r["token"]: r for r in self.tables["sample"]}

    def lidar_ego_pose(self, tok):
        x = {"s1": 0.0, "s2": 4.0, "s3": 8.0}[tok]
        return Transform.from_nuscenes({"translation": [x, 0, 0], "rotation": list(quaternion_from_yaw_rad(0))},
                                       source_frame=EGO, parent_frame=NUSCENES_GLOBAL)


def test_scene_frames_are_time_ordered_and_scene_scoped():
    fr = scene_frames_from_root(FakeRoot(), "sc")
    assert fr.tokens == ["s1", "s2", "s3"]
    assert fr.timestamps_ns == [2_000_000_000, 2_400_000_000, 2_800_000_000]
    assert fr.index["s3"] == 2
    assert fr.dt_s(0, 2) == pytest.approx(0.8)
    assert fr.poses["s2"].translation_m[0] == 4.0


def _pcd_bytes(points):
    n = len(points)
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\n"
              "SIZE 4 4 4 4\nTYPE F F F F\nCOUNT 1 1 1 1\n"
              f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    return header.encode() + np.asarray(points, np.float32).tobytes()


def test_read_pcd_v07_binary():
    arr = read_pcd_v07_binary(_pcd_bytes([[1, 2, 3, 0.5], [4, 5, 6, 0.1]]))
    assert arr.shape == (2, 4) and arr[1, 2] == 6.0


def test_cloud_source_prefers_task_zip_then_raw(tmp_path):
    fr = SceneFrames(scene_token="sc", scene_name="chunk_test", tokens=["s1", "s2"],
                     timestamps_ns=[0, 400_000_000], poses={}, index={"s1": 0, "s2": 1})
    export_dir = tmp_path / "cvat_export_3d" / "chunk_test"
    export_dir.mkdir(parents=True)
    with zipfile.ZipFile(export_dir / "task.zip", "w") as zf:
        zf.writestr("pointcloud/000001.pcd", _pcd_bytes([[1, 1, 1, 0], [2, 2, 2, 0], [3, 3, 3, 0]]))
    (export_dir / "frames.json").write_text(json.dumps(
        [{"frame": 0, "name": "000001", "sample_token": "s1", "channels": []}]))
    dataroot = tmp_path / "root"
    (dataroot / "samples" / "LIDAR_TOP").mkdir(parents=True)
    np.zeros((7, 5), np.float32).tofile(dataroot / "samples" / "LIDAR_TOP" / "s2.pcd.bin")

    class Root:
        def lidar_sd(self, tok):
            return {"filename": f"samples/LIDAR_TOP/{tok}.pcd.bin"}

    src = CloudSource(str(dataroot), str(tmp_path / "cvat_export_3d"), fr, Root())
    pts, basis = src.points("s1")
    assert pts.shape == (3, 3) and basis == "single_sweep_ground_filtered_pre_inflation"
    pts, basis = src.points("s2")
    assert pts.shape == (7, 3) and basis == "single_sweep_raw"
    src.close()
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_release_frames.py -v`
Expected: import error.

- [ ] **Step 3: Implement `frames.py`**

```python
"""Per-scene keyframe order, poses, and the cloud each keyframe's boxes were fit to."""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import dataclass, field

import numpy as np

from pipeline.common.conventions import Transform
from pipeline.stage1_ingestion.ingest import read_pcd_bin

BASIS_GROUND_FILTERED = "single_sweep_ground_filtered_pre_inflation"
BASIS_RAW = "single_sweep_raw"
BASIS_UNAVAILABLE = "unavailable"
US_TO_NS = 1_000


@dataclass
class SceneFrames:
    scene_token: str
    scene_name: str
    tokens: list
    timestamps_ns: list
    poses: dict
    index: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.index:
            self.index = {t: i for i, t in enumerate(self.tokens)}

    def dt_s(self, i: int, j: int) -> float:
        return (self.timestamps_ns[j] - self.timestamps_ns[i]) / 1e9


def scene_frames_from_root(root, scene_token: str) -> SceneFrames:
    rows = sorted((r for r in root.tables["sample"] if r["scene_token"] == scene_token),
                  key=lambda r: (r["timestamp"], r["token"]))
    tokens = [r["token"] for r in rows]
    return SceneFrames(
        scene_token=scene_token,
        scene_name=root.scene[scene_token]["name"],
        tokens=tokens,
        timestamps_ns=[int(r["timestamp"]) * US_TO_NS for r in rows],
        poses={t: root.lidar_ego_pose(t) for t in tokens},
    )


def read_pcd_v07_binary(data: bytes) -> np.ndarray:
    head_end = data.index(b"DATA binary\n") + len(b"DATA binary\n")
    header = data[:head_end].decode("ascii", "replace")
    fields = next(l for l in header.splitlines() if l.startswith("FIELDS")).split()[1:]
    n = int(next(l for l in header.splitlines() if l.startswith("POINTS")).split()[1])
    return np.frombuffer(data[head_end:], dtype=np.float32, count=n * len(fields)).reshape(n, len(fields))


class CloudSource:
    def __init__(self, dataroot: str, cvat_export_3d_dir: str | None, frames: SceneFrames, root):
        self.dataroot = dataroot
        self.frames = frames
        self.root = root
        self._zip = None
        self._name_of: dict = {}
        if cvat_export_3d_dir:
            scene_dir = os.path.join(cvat_export_3d_dir, frames.scene_name)
            zpath, fpath = os.path.join(scene_dir, "task.zip"), os.path.join(scene_dir, "frames.json")
            if os.path.isfile(zpath) and os.path.isfile(fpath):
                with open(fpath, "r", encoding="utf-8") as fh:
                    self._name_of = {r["sample_token"]: r["name"] for r in json.load(fh)}
                self._zip = zipfile.ZipFile(zpath)

    def points(self, sample_token: str):
        name = self._name_of.get(sample_token)
        if self._zip is not None and name is not None:
            try:
                arr = read_pcd_v07_binary(self._zip.read(f"pointcloud/{name}.pcd"))
                return arr[:, :3].astype(np.float64), BASIS_GROUND_FILTERED
            except KeyError:
                pass
        try:
            sd = self.root.lidar_sd(sample_token)
            path = os.path.join(self.dataroot, sd["filename"])
            if os.path.isfile(path):
                return read_pcd_bin(path)[:, :3].astype(np.float64), BASIS_RAW
        except Exception:  # noqa: BLE001 — a missing sweep is reported as unavailable, not raised
            pass
        return None, BASIS_UNAVAILABLE

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
            self._zip = None
```

- [ ] **Step 4: Run tests + suite, commit**

Run: `$PY -m pytest tests/test_release_frames.py -v && $PY -m pytest tests/ -q`

```bash
git add pipeline/release/frames.py tests/test_release_frames.py
git commit -m "pipeline/release/frames: SceneFrames + CloudSource (task.zip PCD, raw fallback) (spec §3)"
```

---

### Task 4: `pipeline/release/stitch.py` — fragments → chains, interpolation

**Files:**
- Create: `pipeline/release/stitch.py`
- Test: `tests/test_release_stitch.py`

**Interfaces:**
- Consumes: `SceneFrames`, `CloudSource` (Task 3), `StitchConfig` (Task 1), geometry (Task 2).
- Produces:
  ```python
  @dataclass
  class StitchStats:  # all ints/floats, JSON-able via vars()
      n_records_in: int; n_fragments: int; n_chains: int; joins_by_gap: dict[int, int]
      n_interpolated: int; n_interpolated_raw_basis: int; n_interpolated_no_cloud: int
      median_len_before: float; median_len_after: float; mean_len_before: float; mean_len_after: float
      frac_rows_on_chains_ge3_before: float; frac_rows_on_chains_ge3_after: float
  def stitch_scene(records: list[dict], frames: SceneFrames, clouds: CloudSource | None,
                   cfg: StitchConfig) -> tuple[list[dict], StitchStats]
  ```
  Returned rows are the input dicts (mutated) plus interpolated dicts, all with these keys set:
  `instance_token = f"chain:{scene_token}:{chain_id}"`, `stitch_chain_id` (str), `stitch_track_id_pre` (str|None),
  `stitch_interpolated` (bool), `stitch_tier_basis` ("gate" | "inherited_from_endpoints").
  Interpolated rows additionally are complete I-4-shaped dicts: `token = f"{sample_token}:INTERP:{chain_id}"`,
  `frame = "ego"`, `t_ns`, `time_base = "unix_ns"`, `translation_m`/`rotation_wxyz` in that keyframe's ego frame,
  `size_wlh_m`, `num_lidar_pts`, `num_lidar_pts_basis`, `velocity_mps = None`, `track_id = None`, `attribute = None`,
  `visibility = None`, `category`, `coverage_config` (copied from P), `provenance = {"source": "pipeline",
  "tier": <worse endpoint tier>, "gates": None, "verified_by": None, "verification_pass": 0}`.
- Also produces `chain_id_of_track(records) -> dict[str, str]` used by Task 10 to write `stitch_map.json`
  (record token → chain id) — implemented as `{r["token"]: r["stitch_chain_id"] for r in rows}` in the exporter; nothing extra here.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.conventions import EGO, NUSCENES_GLOBAL, Transform, quaternion_from_yaw_rad  # noqa: E402
from pipeline.release.config import StitchConfig  # noqa: E402
from pipeline.release.frames import SceneFrames  # noqa: E402
from pipeline.release.stitch import stitch_scene  # noqa: E402

CFG = StitchConfig(max_gap_keyframes=3, base_gate_m=2.0, gap_slack_m=1.0, size_ratio_max=2.0, class_agnostic=False)
DT_NS = 400_000_000


def _frames(n=8, ego_speed=0.0):
    toks = [f"s{i}" for i in range(n)]
    poses = {t: Transform.from_nuscenes({"translation": [ego_speed * 0.4 * i, 0.0, 0.0],
                                         "rotation": list(quaternion_from_yaw_rad(0.0))},
                                        source_frame=EGO, parent_frame=NUSCENES_GLOBAL)
             for i, t in enumerate(toks)}
    return SceneFrames("sc", "chunk_t", toks, [i * DT_NS for i in range(n)], poses)


def _rec(i, track, x, y=0.0, cat="a car", tier="auto_accept", size=(1.8, 4.5, 1.6), yaw=0.0):
    return {
        "token": f"s{i}:CAM_FRONT:{track}", "sample_token": f"s{i}",
        "instance_token": f"pilot-track:sc:{track}", "category": cat, "frame": "ego",
        "t_ns": i * DT_NS, "time_base": "unix_ns",
        "translation_m": [x, y, 0.0], "size_wlh_m": list(size),
        "rotation_wxyz": list(quaternion_from_yaw_rad(yaw)), "num_lidar_pts": 20,
        "num_lidar_pts_basis": "single_sweep_ground_filtered_pre_inflation",
        "provenance": {"source": "pipeline", "tier": tier, "gates": {"conf": 0.9}, "verified_by": None,
                       "verification_pass": 0},
        "coverage_config": "R2", "velocity_mps": [0.0, 0.0], "track_id": str(track), "split": None,
        "attribute": None, "visibility": None,
    }


def _chains(rows):
    out = {}
    for r in rows:
        out.setdefault(r["stitch_chain_id"], []).append(r["sample_token"])
    return {k: sorted(v, key=lambda t: int(t[1:])) for k, v in out.items()}


def test_gap1_join_of_a_moving_car():
    # track 1 at 10 m/s for frames 0-2, track 2 continues at frames 3-5
    rows = [_rec(i, 1, 10.0 + 4.0 * i) for i in range(3)] + [_rec(i, 2, 10.0 + 4.0 * i) for i in range(3, 6)]
    out, st = stitch_scene(rows, _frames(), None, CFG)
    ch = _chains(out)
    assert len(ch) == 1 and st.n_chains == 1 and st.joins_by_gap == {1: 1}
    assert st.n_interpolated == 0
    assert all(r["instance_token"] == "chain:sc:1" for r in out)
    assert {r["stitch_track_id_pre"] for r in out} == {"1", "2"}


def test_gap3_join_interpolates_two_rows_and_inherits_worse_tier():
    rows = [_rec(i, 1, 1.0 * i, tier="auto_accept") for i in range(3)]          # frames 0,1,2 (1 m/s)
    rows += [_rec(i, 2, 1.0 * i, tier="flagged") for i in range(5, 8)]          # frames 5,6,7
    out, st = stitch_scene(rows, _frames(), None, CFG)
    assert st.joins_by_gap == {3: 1} and st.n_interpolated == 2
    interp = sorted((r for r in out if r["stitch_interpolated"]), key=lambda r: r["t_ns"])
    assert [r["sample_token"] for r in interp] == ["s3", "s4"]
    assert interp[0]["translation_m"][0] == pytest.approx(3.0) and interp[1]["translation_m"][0] == pytest.approx(4.0)
    assert interp[0]["provenance"]["tier"] == "flagged" and interp[0]["stitch_tier_basis"] == "inherited_from_endpoints"
    assert interp[0]["token"] == "s3:INTERP:1" and interp[0]["num_lidar_pts_basis"] == "unavailable"
    assert interp[0]["num_lidar_pts"] == 0
    assert interp[0]["track_id"] is None and interp[0]["velocity_mps"] is None


def test_class_mismatch_and_gate_refuse():
    rows = [_rec(0, 1, 0.0), _rec(1, 2, 0.0, cat="a bus")]                 # class differs
    rows += [_rec(3, 3, 50.0), _rec(4, 4, 60.0)]                           # 10 m apart, no velocity
    out, st = stitch_scene(rows, _frames(), None, CFG)
    assert st.n_chains == 4 and st.joins_by_gap == {}


def test_back_prediction_joins_single_frame_predecessor():
    # a single-frame fragment (no velocity) followed by a moving fragment whose back-prediction lands on it
    rows = [_rec(0, 1, 0.0)] + [_rec(i, 2, 4.0 * i) for i in range(1, 4)]   # 10 m/s
    out, st = stitch_scene(rows, _frames(), None, CFG)
    assert st.n_chains == 1


def test_stationary_assumption_refuses_fast_single_frames():
    rows = [_rec(0, 1, 0.0), _rec(1, 2, 4.0)]     # 4 m apart, neither has a velocity
    out, st = stitch_scene(rows, _frames(), None, CFG)
    assert st.n_chains == 2


def test_interpolation_is_in_each_keyframes_ego_frame():
    # ego drives +x at 5 m/s; object is stationary at global x=20
    fr = _frames(ego_speed=5.0)
    rows = [_rec(0, 1, 20.0), _rec(3, 2, 20.0 - 5.0 * 0.4 * 3)]
    out, st = stitch_scene(rows, fr, None, CFG)
    assert st.n_chains == 1 and st.n_interpolated == 2
    i1 = next(r for r in out if r["sample_token"] == "s1")
    assert i1["translation_m"][0] == pytest.approx(20.0 - 2.0)


def test_deterministic_and_idempotent_on_singletons():
    rows = [_rec(i, i, 3.0 * i, y=float(i)) for i in range(4)]     # 3.16 m apart each step, stationary assumption
    a, sa = stitch_scene([dict(r) for r in rows], _frames(), None, CFG)
    b, sb = stitch_scene([dict(r) for r in rows], _frames(), None, CFG)
    assert [r["stitch_chain_id"] for r in a] == [r["stitch_chain_id"] for r in b]
    assert sa.n_chains == 4
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_release_stitch.py -v`
Expected: import error.

- [ ] **Step 3: Implement `stitch.py`**

```python
"""Offline track stitching + interpolation over one scene's I-4 records (spec §3)."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, median

import numpy as np
from scipy.optimize import linear_sum_assignment

from pipeline.release.config import StitchConfig
from pipeline.release.frames import BASIS_UNAVAILABLE, CloudSource, SceneFrames
from pipeline.release.geometry import box_ego_to_global, box_global_to_ego, points_in_box, slerp

TIER_ORDER = {"auto_accept": 0, "flagged": 1, "rejected": 2}


@dataclass
class StitchStats:
    n_records_in: int = 0
    n_fragments: int = 0
    n_chains: int = 0
    joins_by_gap: dict = field(default_factory=dict)
    n_interpolated: int = 0
    n_interpolated_raw_basis: int = 0
    n_interpolated_no_cloud: int = 0
    median_len_before: float = 0.0
    median_len_after: float = 0.0
    mean_len_before: float = 0.0
    mean_len_after: float = 0.0
    frac_rows_on_chains_ge3_before: float = 0.0
    frac_rows_on_chains_ge3_after: float = 0.0


@dataclass
class Fragment:
    fid: str                      # Stage 7 track_id, or the record token for an untracked singleton
    category: str
    rows: list                    # ordered by keyframe index
    kidx: list                    # keyframe indices
    centers: np.ndarray           # (n, 3) global
    quats: list                   # global [w,x,y,z] per row
    sizes: np.ndarray             # (n, 3) w,l,h

    @property
    def first(self) -> int:
        return self.kidx[0]

    @property
    def last(self) -> int:
        return self.kidx[-1]

    def bev_area(self) -> float:
        return float(self.sizes[:, 0].mean() * self.sizes[:, 1].mean())

    def end_velocity(self, frames: SceneFrames):
        if len(self.rows) < 2:
            return None
        dt = frames.dt_s(self.kidx[-2], self.kidx[-1])
        return (self.centers[-1] - self.centers[-2]) / dt if dt > 0 else None

    def start_velocity(self, frames: SceneFrames):
        if len(self.rows) < 2:
            return None
        dt = frames.dt_s(self.kidx[0], self.kidx[1])
        return (self.centers[1] - self.centers[0]) / dt if dt > 0 else None


def _fragment_id(r: dict) -> str:
    return str(r["track_id"]) if r.get("track_id") is not None else f"det:{r['token']}"


def _build_fragments(records: list, frames: SceneFrames) -> list:
    groups: dict = defaultdict(list)
    for r in records:
        groups[_fragment_id(r)].append(r)
    out = []
    for fid in sorted(groups, key=lambda k: (len(k), k)):
        rows = sorted(groups[fid], key=lambda r: frames.index[r["sample_token"]])
        cats = {r["category"] for r in rows}
        if len(cats) != 1:
            raise ValueError(f"fragment {fid!r} spans categories {sorted(cats)}; Stage 7 gates per class")
        centers, quats = [], []
        for r in rows:
            t, q = box_ego_to_global(r["translation_m"], r["rotation_wxyz"], frames.poses[r["sample_token"]])
            centers.append(t)
            quats.append(q)
        out.append(Fragment(fid=fid, category=rows[0]["category"], rows=rows,
                            kidx=[frames.index[r["sample_token"]] for r in rows],
                            centers=np.asarray(centers, dtype=np.float64), quats=quats,
                            sizes=np.asarray([r["size_wlh_m"] for r in rows], dtype=np.float64)))
    return out


def _distance(p: Fragment, s: Fragment, frames: SceneFrames):
    dt = frames.dt_s(p.last, s.first)
    cands = []
    vp, vs = p.end_velocity(frames), s.start_velocity(frames)
    if vp is not None:
        cands.append(np.linalg.norm(p.centers[-1] + vp * dt - s.centers[0]))
    if vs is not None:
        cands.append(np.linalg.norm(s.centers[0] - vs * dt - p.centers[-1]))
    if not cands:
        cands.append(np.linalg.norm(s.centers[0] - p.centers[-1]))
    return float(min(cands))


def _assign(frags: list, frames: SceneFrames, cfg: StitchConfig):
    """Returns next_of: fragment index -> successor index, plus joins_by_gap."""
    next_of: dict = {}
    prev_of: dict = {}
    joins: dict = {}
    for gap in range(1, cfg.max_gap_keyframes + 1):
        gate = cfg.base_gate_m + cfg.gap_slack_m * (gap - 1)
        preds = [i for i, f in enumerate(frags) if i not in next_of]
        succs = [j for j, f in enumerate(frags) if j not in prev_of]
        pairs = []
        for i in preds:
            p = frags[i]
            for j in succs:
                s = frags[j]
                if i == j or s.first - p.last != gap:
                    continue
                if not cfg.class_agnostic and s.category != p.category:
                    continue
                ratio = p.bev_area() / max(s.bev_area(), 1e-9)
                if ratio > cfg.size_ratio_max or ratio < 1.0 / cfg.size_ratio_max:
                    continue
                d = _distance(p, s, frames)
                if d <= gate:
                    pairs.append((i, j, d / gate))
        if not pairs:
            continue
        rows_i = sorted({i for i, _, _ in pairs})
        cols_j = sorted({j for _, j, _ in pairs})
        cost = np.full((len(rows_i), len(cols_j)), 10.0)
        ri, cj = {i: a for a, i in enumerate(rows_i)}, {j: b for b, j in enumerate(cols_j)}
        for i, j, c in pairs:
            cost[ri[i], cj[j]] = c
        for a, b in zip(*linear_sum_assignment(cost)):
            if cost[a, b] <= 1.0:
                i, j = rows_i[a], cols_j[b]
                next_of[i] = j
                prev_of[j] = i
                joins[gap] = joins.get(gap, 0) + 1
    return next_of, joins


def _worse_tier(a: str, b: str) -> str:
    return a if TIER_ORDER.get(a, 9) >= TIER_ORDER.get(b, 9) else b


def _interpolate(p: Fragment, s: Fragment, chain_id: str, frames: SceneFrames,
                 clouds: CloudSource | None, stats: StitchStats) -> list:
    rows = []
    k0, k1 = p.last, s.first
    c0, c1 = p.centers[-1], s.centers[0]
    q0, q1 = np.asarray(p.quats[-1]), np.asarray(s.quats[0])
    z0, z1 = p.sizes[-1], s.sizes[0]
    tier = _worse_tier(p.rows[-1]["provenance"]["tier"], s.rows[0]["provenance"]["tier"])
    for k in range(k0 + 1, k1):
        f = (k - k0) / (k1 - k0)
        tok = frames.tokens[k]
        cg = (1 - f) * c0 + f * c1
        qg = slerp(q0, q1, f)
        size = ((1 - f) * z0 + f * z1).tolist()
        te, qe = box_global_to_ego(cg.tolist(), qg.tolist(), frames.poses[tok])
        n_pts, basis = 0, BASIS_UNAVAILABLE
        if clouds is not None:
            pts, basis = clouds.points(tok)
            if pts is not None:
                n_pts = points_in_box(pts, te, size, qe)
                if basis == "single_sweep_raw":
                    stats.n_interpolated_raw_basis += 1
            else:
                stats.n_interpolated_no_cloud += 1
        else:
            stats.n_interpolated_no_cloud += 1
        src = p.rows[-1]
        rows.append({
            "token": f"{tok}:INTERP:{chain_id}", "sample_token": tok,
            "instance_token": f"chain:{frames.scene_token}:{chain_id}", "category": p.category,
            "frame": "ego", "t_ns": int(frames.timestamps_ns[k]), "time_base": "unix_ns",
            "translation_m": te, "size_wlh_m": size, "rotation_wxyz": qe,
            "num_lidar_pts": int(n_pts), "num_lidar_pts_basis": basis,
            "provenance": {"source": "pipeline", "tier": tier, "gates": None,
                           "verified_by": None, "verification_pass": 0},
            "coverage_config": src.get("coverage_config"), "velocity_mps": None, "track_id": None,
            "split": None, "attribute": None, "visibility": None,
            "stitch_chain_id": chain_id, "stitch_track_id_pre": None,
            "stitch_interpolated": True, "stitch_tier_basis": "inherited_from_endpoints",
        })
        stats.n_interpolated += 1
    return rows


def _length_stats(groups: dict, n_rows: int):
    lens = [len(v) for v in groups.values()]
    ge3 = sum(len(v) for v in groups.values() if len(v) >= 3)
    return (float(median(lens)) if lens else 0.0, float(mean(lens)) if lens else 0.0,
            ge3 / n_rows if n_rows else 0.0)


def stitch_scene(records: list, frames: SceneFrames, clouds: CloudSource | None,
                 cfg: StitchConfig) -> tuple:
    stats = StitchStats(n_records_in=len(records))
    frags = _build_fragments(records, frames)
    stats.n_fragments = len(frags)
    before = {f.fid: f.rows for f in frags}
    stats.median_len_before, stats.mean_len_before, stats.frac_rows_on_chains_ge3_before = \
        _length_stats(before, len(records))

    next_of, joins = _assign(frags, frames, cfg)
    stats.joins_by_gap = dict(sorted(joins.items()))
    has_prev = set(next_of.values())
    out: list = []
    chain_rows: dict = {}
    for i, f in enumerate(frags):
        if i in has_prev:
            continue
        chain_id = f.fid
        j = i
        members = []
        while True:
            members.append(j)
            if j not in next_of:
                break
            j = next_of[j]
        rows_here = []
        for a, b in zip(members, members[1:]):
            rows_here.extend(frags[a].rows)
            if frags[b].first - frags[a].last >= 2:
                rows_here.extend(_interpolate(frags[a], frags[b], chain_id, frames, clouds, stats))
        rows_here.extend(frags[members[-1]].rows)
        for r in rows_here:
            if not r.get("stitch_interpolated"):
                r["stitch_track_id_pre"] = str(r["track_id"]) if r.get("track_id") is not None else None
                r["stitch_interpolated"] = False
                r["stitch_tier_basis"] = "gate"
            r["stitch_chain_id"] = chain_id
            r["instance_token"] = f"chain:{frames.scene_token}:{chain_id}"
        chain_rows[chain_id] = rows_here
        out.extend(rows_here)
    stats.n_chains = len(chain_rows)
    stats.median_len_after, stats.mean_len_after, stats.frac_rows_on_chains_ge3_after = \
        _length_stats(chain_rows, len(out))
    out.sort(key=lambda r: (frames.index[r["sample_token"]], r["stitch_chain_id"], r["token"]))
    return out, stats
```

Note for the implementer: `test_gap3_join_interpolates_two_rows_and_inherits_worse_tier` passes `clouds=None`, so interpolated rows get `num_lidar_pts 0` / basis `unavailable` there; the exporter always passes a `CloudSource`. The fragment-id sort `(len(k), k)` makes `"1" < "2" < "10"` and keeps chain ids stable.

- [ ] **Step 4: Run tests + suite, commit**

Run: `$PY -m pytest tests/test_release_stitch.py -v && $PY -m pytest tests/ -q`

```bash
git add pipeline/release/stitch.py tests/test_release_stitch.py
git commit -m "pipeline/release/stitch: offline track stitching + interpolation (spec §3)"
```

---

### Task 5: `pipeline/release/attributes.py` — chain velocity and attribute state

**Files:**
- Create: `pipeline/release/attributes.py`
- Test: `tests/test_release_attributes.py`

**Interfaces:**
- Consumes: `AttributeConfig` (Task 1); rows grouped by final `instance_token`; a `positions` mapping `row token → global center (3,)` and `t_ns` on each row.
- Produces:
  ```python
  VEHICLE_CLASSES = ("car","bus","truck","covered_van","microbus","cng_autorickshaw","battery_rickshaw","tempo","human_hauler","pushcart")
  PEDESTRIAN_CLASSES = ("pedestrian",)
  CYCLE_CLASSES = ("bicycle","motorcycle","cycle_rickshaw")
  NO_ATTRIBUTE_CLASSES = ("traffic_cone","barrier","construction_element","animal")
  ATTRIBUTE_NAMES = ("vehicle.moving","vehicle.stopped","vehicle.parked","pedestrian.moving","pedestrian.standing",
                     "pedestrian.sitting_lying_down","cycle.with_rider","cycle.without_rider")
  def chain_velocities(rows: list[dict], global_center_of: dict[str, np.ndarray], cfg: AttributeConfig) -> dict[str, np.ndarray | None]
      # token -> [vx, vy] global, nuScenes box_velocity semantics (next-prev / dt, one-sided at ends, None if single or gap > max_time_diff_s)
  def attribute_state(speed_mps: float | None, cfg: AttributeConfig) -> str | None   # "moving" | "stopped" | None
  def attribute_name_for(dbench_class: str, state: str | None) -> str | None
  def assign_attributes(rows: list[dict], class_of: dict[str, str], global_center_of: dict[str, np.ndarray], cfg: AttributeConfig) -> None
      # sets on every row: velocity_chain_mps ([vx,vy] | None), attr_state, attr_name (nuScenes name | None),
      # attribute_basis ("human" if row["attribute"] already holds a nuScenes name, "chain_velocity" if derived, None)
  ```

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.release.attributes import (  # noqa: E402
    assign_attributes, attribute_name_for, attribute_state, chain_velocities,
)
from pipeline.release.config import AttributeConfig  # noqa: E402

CFG = AttributeConfig(moving_speed_threshold_mps=0.5, max_time_diff_s=1.5)


def _row(tok, inst, t_s, attribute=None):
    return {"token": tok, "instance_token": inst, "t_ns": int(t_s * 1e9), "attribute": attribute}


def test_chain_velocity_matches_devkit_semantics():
    rows = [_row("a", "i", 0.0), _row("b", "i", 0.4), _row("c", "i", 0.8)]
    pos = {"a": np.array([0.0, 0, 0]), "b": np.array([2.0, 0, 0]), "c": np.array([4.0, 0, 0])}
    v = chain_velocities(rows, pos, CFG)
    assert np.allclose(v["b"], [5.0, 0.0])          # central difference
    assert np.allclose(v["a"], [5.0, 0.0]) and np.allclose(v["c"], [5.0, 0.0])   # one-sided at the ends


def test_single_annotation_and_big_gap_are_undefined():
    rows = [_row("a", "i", 0.0)]
    assert chain_velocities(rows, {"a": np.zeros(3)}, CFG) == {"a": None}
    rows = [_row("a", "j", 0.0), _row("b", "j", 2.0)]
    v = chain_velocities(rows, {"a": np.zeros(3), "b": np.array([3.0, 0, 0])}, CFG)
    assert v == {"a": None, "b": None}


def test_state_threshold():
    assert attribute_state(None, CFG) is None
    assert attribute_state(0.49, CFG) == "stopped"
    assert attribute_state(0.51, CFG) == "moving"


@pytest.mark.parametrize("cls,state,name", [
    ("car", "moving", "vehicle.moving"), ("cng_autorickshaw", "stopped", "vehicle.stopped"),
    ("pushcart", "moving", "vehicle.moving"),
    ("pedestrian", "moving", "pedestrian.moving"), ("pedestrian", "stopped", "pedestrian.standing"),
    ("bicycle", "moving", "cycle.with_rider"), ("cycle_rickshaw", "stopped", "cycle.with_rider"),
    ("traffic_cone", "moving", None), ("animal", "stopped", None), ("car", None, None),
])
def test_names(cls, state, name):
    assert attribute_name_for(cls, state) == name


def test_assign_attributes_sets_basis_and_respects_human_value():
    rows = [_row("a", "i", 0.0), _row("b", "i", 0.4), _row("h", "k", 0.0, attribute="vehicle.parked")]
    pos = {"a": np.zeros(3), "b": np.array([0.1, 0, 0]), "h": np.zeros(3)}
    assign_attributes(rows, {"i": "car", "k": "car"}, pos, CFG)
    a, b, h = rows
    assert a["attr_name"] == "vehicle.stopped" and a["attribute_basis"] == "chain_velocity"
    assert b["velocity_chain_mps"] == pytest.approx([0.25, 0.0])
    assert h["attr_name"] == "vehicle.parked" and h["attribute_basis"] == "human"
    assert h["velocity_chain_mps"] is None


def test_unknown_human_attribute_name_raises():
    rows = [_row("h", "k", 0.0, attribute="vehicle.flying")]
    with pytest.raises(ValueError):
        assign_attributes(rows, {"k": "car"}, {"h": np.zeros(3)}, CFG)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_release_attributes.py -v`

- [ ] **Step 3: Implement `attributes.py`**

```python
"""Velocity-derived attributes (spec §4). Never emits parked / sitting / without_rider."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from pipeline.release.config import AttributeConfig

VEHICLE_CLASSES = ("car", "bus", "truck", "covered_van", "microbus", "cng_autorickshaw",
                   "battery_rickshaw", "tempo", "human_hauler", "pushcart")
PEDESTRIAN_CLASSES = ("pedestrian",)
CYCLE_CLASSES = ("bicycle", "motorcycle", "cycle_rickshaw")
NO_ATTRIBUTE_CLASSES = ("traffic_cone", "barrier", "construction_element", "animal")
ATTRIBUTE_NAMES = ("vehicle.moving", "vehicle.stopped", "vehicle.parked", "pedestrian.moving",
                   "pedestrian.standing", "pedestrian.sitting_lying_down", "cycle.with_rider",
                   "cycle.without_rider")
BASIS_HUMAN = "human"
BASIS_CHAIN = "chain_velocity"


def chain_velocities(rows: list, global_center_of: dict, cfg: AttributeConfig) -> dict:
    by_inst: dict = defaultdict(list)
    for r in rows:
        by_inst[r["instance_token"]].append(r)
    out: dict = {}
    max_dt = cfg.max_time_diff_s
    for members in by_inst.values():
        members.sort(key=lambda r: r["t_ns"])
        n = len(members)
        for i, r in enumerate(members):
            if n == 1:
                out[r["token"]] = None
                continue
            j0, j1 = max(i - 1, 0), min(i + 1, n - 1)
            dt = (members[j1]["t_ns"] - members[j0]["t_ns"]) / 1e9
            if dt <= 0 or dt > max_dt * (2 if (j1 - j0 == 2) else 1):
                out[r["token"]] = None
                continue
            d = global_center_of[members[j1]["token"]][:2] - global_center_of[members[j0]["token"]][:2]
            out[r["token"]] = np.asarray(d, dtype=np.float64) / dt
    return out


def attribute_state(speed_mps, cfg: AttributeConfig):
    if speed_mps is None:
        return None
    return "moving" if speed_mps > cfg.moving_speed_threshold_mps else "stopped"


def attribute_name_for(dbench_class: str, state):
    if state is None or dbench_class in NO_ATTRIBUTE_CLASSES:
        return None
    if dbench_class in PEDESTRIAN_CLASSES:
        return "pedestrian.moving" if state == "moving" else "pedestrian.standing"
    if dbench_class in CYCLE_CLASSES:
        return "cycle.with_rider"
    if dbench_class in VEHICLE_CLASSES:
        return f"vehicle.{state}"
    raise ValueError(f"unknown dbench class {dbench_class!r}")


def assign_attributes(rows: list, class_of: dict, global_center_of: dict, cfg: AttributeConfig) -> None:
    vel = chain_velocities(rows, global_center_of, cfg)
    for r in rows:
        v = vel.get(r["token"])
        r["velocity_chain_mps"] = None if v is None else [float(v[0]), float(v[1])]
        human = r.get("attribute")
        if human:
            if human not in ATTRIBUTE_NAMES:
                raise ValueError(f"record {r['token']}: attribute {human!r} is not a nuScenes attribute name")
            r["attr_state"], r["attr_name"], r["attribute_basis"] = None, human, BASIS_HUMAN
            r["velocity_chain_mps"] = None if v is None else r["velocity_chain_mps"]
            continue
        speed = None if v is None else float(np.hypot(v[0], v[1]))
        state = attribute_state(speed, cfg)
        name = attribute_name_for(class_of[r["instance_token"]], state)
        r["attr_state"] = state
        r["attr_name"] = name
        r["attribute_basis"] = BASIS_CHAIN if name else None
```

Note: the `max_dt * 2` for a central difference mirrors the devkit, which bounds each one-sided gap by `max_time_diff`; with 2.5 Hz keyframes and `max_gap_keyframes = 3` every stitched chain passes.

- [ ] **Step 4: Run tests + suite, commit**

```bash
git add pipeline/release/attributes.py tests/test_release_attributes.py
git commit -m "pipeline/release/attributes: chain velocity -> moving/stopped, nuScenes names (spec §4)"
```

---

### Task 6: `pipeline/release/tiers.py` — included / excluded partition

**Files:**
- Create: `pipeline/release/tiers.py`
- Test: `tests/test_release_tiers.py`

**Interfaces:**
- Produces:
  ```python
  REASON_REJECTED = "tier_rejected"; REASON_FLAGGED = "tier_flagged"
  REASON_HUMAN = "superseded_by_human"; REASON_DOUBLE = "superseded_by_double_pass"
  ADMIT_AUTO = "auto_accept"; ADMIT_ALL = "all"
  def partition(rows: list[dict], admit: str, superseded: dict[str, str]) -> tuple[list[dict], list[dict]]
      # superseded: row token -> reason (from human.py). Returns (included, excluded); every excluded row gets
      # `excluded_reason`. Human rows (provenance.source in HUMAN_SOURCES) are always included unless superseded.
      # admit == "all" admits every pipeline tier (today's behaviour); "auto_accept" admits only that tier.
  ```

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.release.tiers import (  # noqa: E402
    ADMIT_ALL, ADMIT_AUTO, REASON_DOUBLE, REASON_FLAGGED, REASON_HUMAN, REASON_REJECTED, partition,
)


def _r(tok, tier=None, source="pipeline"):
    return {"token": tok, "provenance": {"source": source, "tier": tier}}


def test_auto_accept_only_by_default_with_reasons():
    rows = [_r("a", "auto_accept"), _r("b", "flagged"), _r("c", "rejected"),
            _r("h", None, "human_verified")]
    inc, exc = partition(rows, ADMIT_AUTO, {})
    assert [r["token"] for r in inc] == ["a", "h"]
    assert {r["token"]: r["excluded_reason"] for r in exc} == {"b": REASON_FLAGGED, "c": REASON_REJECTED}


def test_superseded_wins_over_tier_and_applies_to_humans():
    rows = [_r("a", "auto_accept"), _r("h", None, "human_verified")]
    inc, exc = partition(rows, ADMIT_AUTO, {"a": REASON_HUMAN, "h": REASON_DOUBLE})
    assert inc == [] and {r["token"]: r["excluded_reason"] for r in exc} == {"a": REASON_HUMAN, "h": REASON_DOUBLE}


def test_admit_all_reproduces_legacy():
    rows = [_r("a", "auto_accept"), _r("b", "flagged"), _r("c", "rejected")]
    inc, exc = partition(rows, ADMIT_ALL, {})
    assert len(inc) == 3 and exc == []


def test_unknown_admit_raises():
    with pytest.raises(ValueError):
        partition([], "some", {})
```

- [ ] **Step 2: Run it to verify it fails**, then **Step 3: implement**

```python
"""Which rows ship in sample_annotation and why the rest do not (spec §5)."""

from __future__ import annotations

from pipeline.common.schemas import HUMAN_SOURCES

REASON_REJECTED = "tier_rejected"
REASON_FLAGGED = "tier_flagged"
REASON_HUMAN = "superseded_by_human"
REASON_DOUBLE = "superseded_by_double_pass"
ADMIT_AUTO = "auto_accept"
ADMIT_ALL = "all"
ADMIT_MODES = (ADMIT_AUTO, ADMIT_ALL)


def partition(rows: list, admit: str, superseded: dict) -> tuple:
    if admit not in ADMIT_MODES:
        raise ValueError(f"admit={admit!r} is not one of {ADMIT_MODES}")
    included, excluded = [], []
    for r in rows:
        prov = r.get("provenance") or {}
        reason = superseded.get(r["token"])
        if reason is None and prov.get("source") not in HUMAN_SOURCES and admit == ADMIT_AUTO:
            tier = prov.get("tier")
            if tier == "rejected":
                reason = REASON_REJECTED
            elif tier == "flagged":
                reason = REASON_FLAGGED
            elif tier != ADMIT_AUTO:
                reason = f"tier_{tier}"
        if reason is None:
            included.append(r)
        else:
            r["excluded_reason"] = reason
            excluded.append(r)
    return included, excluded
```

- [ ] **Step 4: Run tests + suite, commit**

```bash
git add pipeline/release/tiers.py tests/test_release_tiers.py
git commit -m "pipeline/release/tiers: accepted-only partition with excluded reasons (spec §5)"
```

---

### Task 7: Schema edits — `annotator_pass`, `is_uncertain*`, human-only `attribute`

**Files:**
- Modify: `pipeline/common/schemas.py` (`Provenance` ~line 973, `AnnotationRecord` ~line 1016, `__all__`/constants near line 98)
- Test: `tests/test_schema_human_fields.py`

**Interfaces:**
- Produces:
  - `Provenance.annotator_pass: str | None = None`; valid values `ANNOTATOR_PASSES = ("A", "B")`; allowed only when `source in HUMAN_SOURCES`.
  - `AnnotationRecord.is_uncertain: bool | None = None`, `AnnotationRecord.is_uncertain_reason: str | None = None`; allowed only on human sources; `is_uncertain_reason` requires `is_uncertain is True`.
  - `AnnotationRecord.attribute`: on human sources must be one of `ATTRIBUTE_NAMES_NUSCENES` (same eight names as Task 5's `ATTRIBUTE_NAMES`, defined here so schemas has no import from `pipeline.release`); on pipeline sources the existing "no producer" error stays.
- `_NESTED_TYPES` / `_build` need no change (new fields are plain scalars). Check `_NESTED_TYPES["AnnotationRecord"]["provenance"]` still maps to `Provenance`.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.schemas import (  # noqa: E402
    AnnotationRecord, GateVector, Provenance, ProvenancePolicy, Record, SchemaValidationError,
    read_records, write_records,
)

HUMAN = ProvenancePolicy(allow_human_provenance=True)


def _rec(**over):
    base = dict(
        token="s1:CAM_FRONT:0", sample_token="s1", instance_token="i", category="a car", frame="ego",
        t_ns=1_700_000_000_000_000_000, time_base="unix_ns", translation_m=[1.0, 2.0, 0.0],
        size_wlh_m=[1.8, 4.5, 1.6], rotation_wxyz=[1.0, 0.0, 0.0, 0.0], num_lidar_pts=10,
        provenance=Provenance(source="human_created", tier="auto_accept",
                              gates=GateVector(conf=1.0, lidar_pts_ok=True, spatial_ok=True, spatial_ok_source="human"),
                              verified_by="ann_a", verification_pass=1, annotator_pass="A"),
        coverage_config="R2",
    )
    base.update(over)
    return AnnotationRecord(**base)


def test_human_record_with_new_fields_round_trips(tmp_path):
    rec = _rec(attribute="vehicle.parked", is_uncertain=True, is_uncertain_reason="could be a microbus")
    assert rec.validate(policy=HUMAN) == []
    p = tmp_path / "v.jsonl"
    write_records(p, [rec], policy=HUMAN)
    back = read_records(p, policy=HUMAN)[0]
    assert back.provenance.annotator_pass == "A" and back.is_uncertain is True
    assert back.attribute == "vehicle.parked"


def test_default_policy_still_refuses_human(tmp_path):
    with pytest.raises(SchemaValidationError):
        write_records(tmp_path / "v.jsonl", [_rec()])


def test_annotator_pass_values_and_source_gate():
    bad = _rec(provenance=Provenance(source="human_created", tier="auto_accept",
                                     gates=GateVector(conf=1.0, lidar_pts_ok=True, spatial_ok=True, spatial_ok_source="h"),
                                     verified_by="x", verification_pass=1, annotator_pass="C"))
    assert any("annotator_pass" in e for e in bad.validate(policy=HUMAN))
    pipe = _rec(provenance=Provenance(source="pipeline", tier="auto_accept",
                                      gates=GateVector(conf=1.0, lidar_pts_ok=True, spatial_ok=True, spatial_ok_source="g"),
                                      annotator_pass="A"))
    assert any("annotator_pass" in e for e in pipe.validate())


def test_pipeline_records_keep_the_no_producer_rules():
    pipe = _rec(provenance=Provenance(source="pipeline", tier="auto_accept",
                                      gates=GateVector(conf=1.0, lidar_pts_ok=True, spatial_ok=True, spatial_ok_source="g")),
                attribute="vehicle.moving", is_uncertain=False)
    errs = pipe.validate()
    assert any("attribute" in e for e in errs) and any("is_uncertain" in e for e in errs)


def test_human_attribute_must_be_a_nuscenes_name_and_reason_needs_flag():
    assert any("attribute" in e for e in _rec(attribute="vehicle.flying").validate(policy=HUMAN))
    assert any("is_uncertain_reason" in e for e in _rec(is_uncertain_reason="why").validate(policy=HUMAN))
```

- [ ] **Step 2: Run it to verify it fails**

Run: `$PY -m pytest tests/test_schema_human_fields.py -v`
Expected: `TypeError: unexpected keyword argument 'annotator_pass'`.

- [ ] **Step 3: Edit `schemas.py`**

Near the other vocabularies (after `SPLITS`):

```python
ANNOTATOR_PASSES: tuple[str, ...] = ("A", "B")
ATTRIBUTE_NAMES_NUSCENES: tuple[str, ...] = (
    "vehicle.moving", "vehicle.stopped", "vehicle.parked", "pedestrian.moving", "pedestrian.standing",
    "pedestrian.sitting_lying_down", "cycle.with_rider", "cycle.without_rider",
)
```

`Provenance`: add field `annotator_pass: str | None = None` after `verification_pass`, and in `validate()`:

```python
        _check_str(e, p, "annotator_pass", self.annotator_pass, allowed=ANNOTATOR_PASSES, allow_none=True)
        ...
        if self.source in HUMAN_SOURCES:
            (existing checks)
        else:
            (existing checks)
            if self.annotator_pass is not None:
                _err(e, p, f"source={self.source!r} must not carry annotator_pass (double annotation is a human pass)")
```

`AnnotationRecord`: add fields after `visibility`:

```python
    # Human-pass only (spec 2026-09-07 §7): the contract's per-row uncertainty flag.
    is_uncertain: bool | None = None
    is_uncertain_reason: str | None = None
```

and replace the two "no producer" lines in `validate()` with:

```python
        human = isinstance(self.provenance, Provenance) and self.provenance.source in HUMAN_SOURCES
        _check_bool(e, p, "is_uncertain", self.is_uncertain, allow_none=True)
        _check_str(e, p, "is_uncertain_reason", self.is_uncertain_reason, allow_none=True)
        if human:
            if self.attribute is not None and self.attribute not in ATTRIBUTE_NAMES_NUSCENES:
                _err(e, p, f"attribute={self.attribute!r} is not a nuScenes attribute name")
            if self.is_uncertain_reason is not None and self.is_uncertain is not True:
                _err(e, p, "is_uncertain_reason requires is_uncertain=true")
        else:
            if self.attribute is not None:
                _err(e, p, "attribute has no producer in the pipeline (Stage 10 waived, §4); human passes only")
            if self.is_uncertain is not None or self.is_uncertain_reason is not None:
                _err(e, p, "is_uncertain is set by a human pass only")
        if self.visibility is not None:
            _err(e, p, "visibility has no producer in the pilot (§4)")
```

Add the two names to `__all__` if the module exports a list.

- [ ] **Step 4: Run the new test and the whole suite**

Run: `$PY -m pytest tests/test_schema_human_fields.py -v && $PY -m pytest tests/ -q`
Expected: PASS — existing schema tests (`tests/test_schemas*.py`) still pass because defaults are `None`.

- [ ] **Step 5: Commit**

```bash
git add pipeline/common/schemas.py tests/test_schema_human_fields.py
git commit -m "schemas: annotator_pass, is_uncertain*, human-only attribute names (spec §7)"
```

---

### Task 8: `pipeline/release/human.py` — load I-5 rows, precedence, superseding

**Files:**
- Create: `pipeline/release/human.py`
- Test: `tests/test_release_human.py`

**Interfaces:**
- Consumes: `<human_dir>/scenes/<scene>/verified.jsonl` (I-5 records, Task 7 schema) and `<human_dir>/scenes/<scene>/coverage.json` written by Task 16:
  ```json
  {"spec": "dhakascenes/human_coverage/v1", "scene": "chunk_0000",
   "review": ["<sample_token>", ...], "double_A": [...], "double_B": [...]}
  ```
  (a task's frames count as covered once its jobs are completed, even if the annotator left a frame empty).
- Produces:
  ```python
  KIND_REVIEW = "review"; KIND_A = "double_A"; KIND_B = "double_B"
  @dataclass
  class HumanMerge:
      rows: list[dict]               # human rows to add; each has `human_kind` and `annotator_pass` (top level, copied from provenance)
      superseded: dict[str, str]     # token -> tiers.REASON_HUMAN | tiers.REASON_DOUBLE, over pipeline AND human rows
      coverage: dict[str, set[str]]  # kind -> sample tokens
      half_imported: list[str]       # samples with B rows and no A coverage
      stats: dict
  def load_human(human_dir: str) -> tuple[list[dict], dict[str, set[str]]]
  def merge_human(pipeline_rows: list[dict], human_rows: list[dict], coverage: dict[str, set[str]]) -> HumanMerge
  ```
  Precedence per sample (spec §7): `double_A` covered → every pipeline row and every review row on that sample is superseded with `REASON_DOUBLE`; A rows and B rows included. Else `review` covered → pipeline rows superseded with `REASON_HUMAN`; review rows included. Else pipeline rows stay; any B rows are included and the sample is listed in `half_imported`.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.schemas import AnnotationRecord, GateVector, Provenance, ProvenancePolicy, write_records  # noqa: E402
from pipeline.release.human import KIND_A, KIND_B, KIND_REVIEW, load_human, merge_human  # noqa: E402
from pipeline.release.tiers import REASON_DOUBLE, REASON_HUMAN  # noqa: E402

HUMAN = ProvenancePolicy(allow_human_provenance=True)


def _human(tok, sample, annotator_pass=None, source="human_created"):
    return AnnotationRecord(
        token=tok, sample_token=sample, instance_token=f"h:{tok}", category="a car", frame="ego",
        t_ns=1_700_000_000_000_000_000, time_base="unix_ns", translation_m=[1.0, 0.0, 0.0],
        size_wlh_m=[1.8, 4.5, 1.6], rotation_wxyz=[1.0, 0.0, 0.0, 0.0], num_lidar_pts=3,
        provenance=Provenance(source=source, tier="auto_accept",
                              gates=GateVector(conf=1.0, lidar_pts_ok=True, spatial_ok=True, spatial_ok_source="h"),
                              verified_by="ann", verification_pass=1, annotator_pass=annotator_pass),
        coverage_config="R2")


def _pipe(tok, sample):
    return {"token": tok, "sample_token": sample, "provenance": {"source": "pipeline", "tier": "auto_accept"}}


def test_load_reads_every_scene(tmp_path):
    d = tmp_path / "scenes" / "chunk_x"
    d.mkdir(parents=True)
    write_records(d / "verified.jsonl", [_human("r1", "s1"), _human("a1", "s2", "A")], policy=HUMAN)
    (d / "coverage.json").write_text(json.dumps({"spec": "dhakascenes/human_coverage/v1", "scene": "chunk_x",
                                                 "review": ["s1", "s9"], "double_A": ["s2"], "double_B": []}))
    rows, cov = load_human(str(tmp_path))
    assert {r["token"] for r in rows} == {"r1", "a1"}
    assert cov[KIND_REVIEW] == {"s1", "s9"} and cov[KIND_A] == {"s2"} and cov[KIND_B] == set()
    kinds = {r["token"]: r["human_kind"] for r in rows}
    assert kinds == {"r1": KIND_REVIEW, "a1": KIND_A}
    assert next(r for r in rows if r["token"] == "a1")["annotator_pass"] == "A"


def test_precedence_rules():
    pipeline = [_pipe("p1", "s1"), _pipe("p2", "s2"), _pipe("p3", "s3"), _pipe("p4", "s4")]
    human = [
        {**_human("r1", "s1").to_dict(), "human_kind": KIND_REVIEW, "annotator_pass": None},
        {**_human("r2", "s2").to_dict(), "human_kind": KIND_REVIEW, "annotator_pass": None},
        {**_human("a2", "s2", "A").to_dict(), "human_kind": KIND_A, "annotator_pass": "A"},
        {**_human("b2", "s2", "B").to_dict(), "human_kind": KIND_B, "annotator_pass": "B"},
        {**_human("b3", "s3", "B").to_dict(), "human_kind": KIND_B, "annotator_pass": "B"},
    ]
    cov = {KIND_REVIEW: {"s1", "s2", "s9"}, KIND_A: {"s2"}, KIND_B: {"s2", "s3"}}
    m = merge_human(pipeline, human, cov)
    assert m.superseded == {"p1": REASON_HUMAN, "p2": REASON_DOUBLE, "r2": REASON_DOUBLE}
    assert {r["token"] for r in m.rows} == {"r1", "r2", "a2", "b2", "b3"}   # r2 is returned; tiers.partition drops it
    assert m.half_imported == ["s3"]
    assert m.stats["n_samples_review"] == 3 and m.stats["n_samples_double_A"] == 1


def test_review_deletion_is_honoured():
    # s1 is review-covered but the reviewer removed every box: pipeline rows go, nothing comes back
    m = merge_human([_pipe("p1", "s1")], [], {KIND_REVIEW: {"s1"}, KIND_A: set(), KIND_B: set()})
    assert m.superseded == {"p1": REASON_HUMAN} and m.rows == []
```

- [ ] **Step 2: Run it to verify it fails**, then **Step 3: implement**

```python
"""I-5 human rows: load, classify, and decide what they supersede (spec §7)."""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

from pipeline.common.schemas import AnnotationRecord, ProvenancePolicy, read_records
from pipeline.release.tiers import REASON_DOUBLE, REASON_HUMAN

KIND_REVIEW = "review"
KIND_A = "double_A"
KIND_B = "double_B"
KINDS = (KIND_REVIEW, KIND_A, KIND_B)
COVERAGE_SPEC = "dhakascenes/human_coverage/v1"
HUMAN_POLICY = ProvenancePolicy(allow_human_provenance=True)


@dataclass
class HumanMerge:
    rows: list
    superseded: dict
    coverage: dict
    half_imported: list
    stats: dict = field(default_factory=dict)


def _kind_of(row: dict) -> str:
    ap = (row.get("provenance") or {}).get("annotator_pass")
    return {None: KIND_REVIEW, "A": KIND_A, "B": KIND_B}[ap]


def load_human(human_dir: str):
    rows: list = []
    coverage = {k: set() for k in KINDS}
    for vpath in sorted(glob.glob(os.path.join(human_dir, "scenes", "*", "verified.jsonl"))):
        for rec in read_records(vpath, policy=HUMAN_POLICY, expect_type=AnnotationRecord):
            d = rec.to_dict()
            d["human_kind"] = _kind_of(d)
            d["annotator_pass"] = d["provenance"].get("annotator_pass")
            d["__source_file__"] = vpath
            rows.append(d)
        cpath = os.path.join(os.path.dirname(vpath), "coverage.json")
        if os.path.isfile(cpath):
            with open(cpath, "r", encoding="utf-8") as fh:
                cov = json.load(fh)
            if cov.get("spec") != COVERAGE_SPEC:
                raise ValueError(f"{cpath}: spec={cov.get('spec')!r}, expected {COVERAGE_SPEC!r}")
            for k in KINDS:
                coverage[k].update(cov.get(k) or [])
    return rows, coverage


def merge_human(pipeline_rows: list, human_rows: list, coverage: dict) -> HumanMerge:
    cov = {k: set(coverage.get(k) or ()) for k in KINDS}
    superseded: dict = {}
    for r in pipeline_rows:
        s = r["sample_token"]
        if s in cov[KIND_A]:
            superseded[r["token"]] = REASON_DOUBLE
        elif s in cov[KIND_REVIEW]:
            superseded[r["token"]] = REASON_HUMAN
    for r in human_rows:
        if r["human_kind"] == KIND_REVIEW and r["sample_token"] in cov[KIND_A]:
            superseded[r["token"]] = REASON_DOUBLE
    b_samples = {r["sample_token"] for r in human_rows if r["human_kind"] == KIND_B}
    half = sorted(b_samples - cov[KIND_A])
    stats = {f"n_samples_{k}": len(cov[k]) for k in KINDS}
    stats.update({f"n_rows_{k}": sum(1 for r in human_rows if r["human_kind"] == k) for k in KINDS})
    stats["n_superseded_by_human"] = sum(1 for v in superseded.values() if v == REASON_HUMAN)
    stats["n_superseded_by_double_pass"] = sum(1 for v in superseded.values() if v == REASON_DOUBLE)
    stats["n_half_imported_samples"] = len(half)
    return HumanMerge(rows=list(human_rows), superseded=superseded, coverage=cov, half_imported=half, stats=stats)
```

- [ ] **Step 4: Run tests + suite, commit**

```bash
git add pipeline/release/human.py tests/test_release_human.py
git commit -m "pipeline/release/human: I-5 load, coverage, precedence (spec §7)"
```

---

### Task 9: `strata.py` + `double.py` — density, illumination, stratified selection

**Files:**
- Create: `pipeline/release/strata.py`, `pipeline/release/double.py`
- Test: `tests/test_release_strata.py`, `tests/test_release_double.py`

**Interfaces:**
- Consumes: `StrataConfig`, `DoubleConfig` (Task 1); `SceneFrames` (Task 3); included rows with global centers.
- Produces:
  ```python
  # strata.py
  def density_per_keyframe(frames: SceneFrames, global_centers_by_sample: dict[str, list[np.ndarray]], cfg: StrataConfig) -> dict[str, float]
      # rho = count of centers within cfg.density_radius_m (BEV) of the keyframe's ego position / (pi r^2); every token in frames.tokens gets a value (0.0 if no boxes)
  def luma_of_image(path: str, saturation_ignore_above: int) -> float | None   # mean BT.601 luma over unsaturated pixels; None if unreadable or all saturated
  def illumination_per_keyframe(frames: SceneFrames, image_path_of: dict[str, str | None], cfg: StrataConfig) -> dict[str, float | None]
  def quantile_edges(values: list[float], quantiles: list[float]) -> list[float]      # numpy quantile, linear
  def bin_of(value: float | None, edges: list[float], names: list[str]) -> str | None  # right-open bins; None -> None
  @dataclass class Strata: density: dict[str,float]; illumination: dict[str,float|None]; density_edges: list[float]; density_bin: dict[str,str]; illumination_bin: dict[str,str|None]
  def compute_strata(frames, global_centers_by_sample, image_path_of, cfg) -> Strata
  # double.py
  DOUBLE_SPEC = "dhakascenes/double_annotation/v1"
  def select_double(strata: Strata, frames: SceneFrames, cfg: DoubleConfig, strata_cfg: StrataConfig) -> dict
      # returns the double_annotation.json document (spec §6); keyframes whose illumination is None go to a cell named "unknown"
  def load_or_select(path: str, strata, frames, cfg, strata_cfg, reselect: bool) -> tuple[dict, bool]
      # (document, reused). Existing file + reselect=False -> reused verbatim. Existing + reselect=True -> reselected, old file kept as <path>.superseded-<utc>.json. Missing -> selected and written.
  ```
  Selection: N = number of keyframes (all scenes, one document per export); target = max(ceil(F·N), n_non_empty_cells); allocation by largest remainder of `n_cell · target / N`, min 1 per non-empty cell, never more than the cell holds; within a cell `numpy.random.default_rng(seed).choice(sorted_tokens_by_timestamp, k, replace=False)`; cells iterated in sorted (density, illumination) name order so the RNG stream is reproducible.

- [ ] **Step 1: Write the failing tests**

`tests/test_release_strata.py`:

```python
from __future__ import annotations

import math
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.conventions import EGO, NUSCENES_GLOBAL, Transform, quaternion_from_yaw_rad  # noqa: E402
from pipeline.release.config import StrataConfig  # noqa: E402
from pipeline.release.frames import SceneFrames  # noqa: E402
from pipeline.release.strata import bin_of, compute_strata, density_per_keyframe, luma_of_image, quantile_edges  # noqa: E402

CFG = StrataConfig(density_radius_m=30.0, density_quantiles=[0.25, 0.5, 0.75],
                   density_bin_names=["Low", "Medium", "High", "Extreme"], illumination_channel="CAM_FRONT",
                   illumination_saturation_ignore_above=250, illumination_bin_edges=[45.0, 75.0, 95.0],
                   illumination_bin_names=["dark", "night", "dusk", "day"])


def _frames(n=4):
    toks = [f"s{i}" for i in range(n)]
    poses = {t: Transform.from_nuscenes({"translation": [10.0 * i, 0.0, 0.0], "rotation": list(quaternion_from_yaw_rad(0))},
                                        source_frame=EGO, parent_frame=NUSCENES_GLOBAL) for i, t in enumerate(toks)}
    return SceneFrames("sc", "chunk_t", toks, [i * 400_000_000 for i in range(n)], poses)


def test_density_counts_within_radius_of_ego():
    fr = _frames()
    centers = {"s0": [np.array([5.0, 0, 0]), np.array([29.0, 0, 0]), np.array([31.0, 0, 0])], "s1": []}
    rho = density_per_keyframe(fr, centers, CFG)
    assert rho["s0"] == 2 / (math.pi * 900) and rho["s1"] == 0.0 and rho["s3"] == 0.0


def test_luma_ignores_saturated_pixels(tmp_path):
    img = np.zeros((4, 4, 3), np.uint8)
    img[:2] = 255                        # saturated half is ignored
    img[2:] = (100, 100, 100)
    p = tmp_path / "f.jpg"
    Image.fromarray(img).save(p, quality=100)
    assert abs(luma_of_image(str(p), 250) - 100.0) < 3.0
    assert luma_of_image(str(tmp_path / "missing.jpg"), 250) is None


def test_edges_and_bins():
    assert quantile_edges([1, 2, 3, 4, 5], [0.5]) == [3.0]
    assert bin_of(44.9, [45, 75, 95], ["dark", "night", "dusk", "day"]) == "dark"
    assert bin_of(45.0, [45, 75, 95], ["dark", "night", "dusk", "day"]) == "night"
    assert bin_of(200.0, [45, 75, 95], ["dark", "night", "dusk", "day"]) == "day"
    assert bin_of(None, [45], ["a", "b"]) is None


def test_compute_strata_end_to_end(tmp_path):
    fr = _frames()
    paths = {}
    for i, t in enumerate(fr.tokens):
        p = tmp_path / f"{t}.jpg"
        Image.fromarray(np.full((2, 2, 3), 30 * i + 20, np.uint8)).save(p)
        paths[t] = str(p)
    centers = {t: [np.array([10.0 * i + 1.0, 0, 0])] * i for i, t in enumerate(fr.tokens)}
    st = compute_strata(fr, centers, paths, CFG)
    assert len(st.density_edges) == 3 and set(st.density_bin.values()) <= set(CFG.density_bin_names)
    assert st.illumination_bin["s0"] == "dark" and st.illumination_bin["s3"] == "day"
```

`tests/test_release_double.py`:

```python
from __future__ import annotations

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.release.config import DoubleConfig, StrataConfig  # noqa: E402
from pipeline.release.double import DOUBLE_SPEC, load_or_select, select_double  # noqa: E402
from pipeline.release.frames import SceneFrames  # noqa: E402
from pipeline.release.strata import Strata  # noqa: E402

SCFG = StrataConfig(density_radius_m=30.0, density_quantiles=[0.25, 0.5, 0.75],
                    density_bin_names=["Low", "Medium", "High", "Extreme"], illumination_channel="CAM_FRONT",
                    illumination_saturation_ignore_above=250, illumination_bin_edges=[45.0, 75.0, 95.0],
                    illumination_bin_names=["dark", "night", "dusk", "day"])
DCFG = DoubleConfig(fraction=0.05, seed=20260812)


def _strata(n=200):
    toks = [f"s{i:04d}" for i in range(n)]
    dens = {t: (i % 4) for i, t in enumerate(toks)}
    dbin = {t: ["Low", "Medium", "High", "Extreme"][i % 4] for i, t in enumerate(toks)}
    ibin = {t: ("night" if i < 3 else "day") for i, t in enumerate(toks)}      # a rare cell: 3 night frames
    frames = SceneFrames("sc", "chunk_t", toks, [i * 400_000_000 for i in range(n)], {})
    return Strata(density=dens, illumination={t: 50.0 for t in toks}, density_edges=[0.5, 1.5, 2.5],
                  density_bin=dbin, illumination_bin=ibin), frames


def test_selection_meets_fraction_and_covers_every_cell():
    st, fr = _strata()
    doc = select_double(st, fr, DCFG, SCFG)
    assert doc["spec"] == DOUBLE_SPEC and doc["n_keyframes"] == 200
    assert doc["n_selected"] >= math.ceil(0.05 * 200)
    cells = {(c["density"], c["illumination"]): c for c in doc["cells"]}
    for c in cells.values():
        if c["n"] > 0:
            assert c["selected"] >= 1
    assert len({s["sample_token"] for s in doc["selected"]}) == doc["n_selected"]


def test_deterministic():
    st, fr = _strata()
    a, b = select_double(st, fr, DCFG, SCFG), select_double(st, fr, DCFG, SCFG)
    assert a["selected"] == b["selected"]


def test_frozen_reuse_and_reselect(tmp_path):
    st, fr = _strata()
    p = tmp_path / "double_annotation.json"
    doc, reused = load_or_select(str(p), st, fr, DCFG, SCFG, reselect=False)
    assert not reused and p.is_file()
    doc2, reused2 = load_or_select(str(p), st, fr, DCFG, SCFG, reselect=False)
    assert reused2 and doc2["selected"] == doc["selected"]
    doc3, reused3 = load_or_select(str(p), st, fr, DoubleConfig(fraction=0.10, seed=1), SCFG, reselect=True)
    assert not reused3 and doc3["fraction"] == 0.10
    assert any(f.name.startswith("double_annotation.json.superseded-") for f in tmp_path.iterdir())
```

- [ ] **Step 2: Run them to verify they fail**, then **Step 3: implement `strata.py`**

```python
"""Per-keyframe density and illumination, the benchmark's stratification axes (spec §6)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from pipeline.release.config import StrataConfig
from pipeline.release.frames import SceneFrames


@dataclass
class Strata:
    density: dict
    illumination: dict
    density_edges: list
    density_bin: dict
    illumination_bin: dict


def density_per_keyframe(frames: SceneFrames, global_centers_by_sample: dict, cfg: StrataConfig) -> dict:
    area = math.pi * cfg.density_radius_m ** 2
    out = {}
    for tok in frames.tokens:
        ego = np.asarray(frames.poses[tok].translation_m[:2], dtype=np.float64)
        n = 0
        for c in global_centers_by_sample.get(tok, ()):
            if np.linalg.norm(np.asarray(c[:2], dtype=np.float64) - ego) <= cfg.density_radius_m:
                n += 1
        out[tok] = n / area
    return out


def luma_of_image(path: str, saturation_ignore_above: int):
    try:
        with Image.open(path) as im:
            rgb = np.asarray(im.convert("RGB"), dtype=np.float64)
    except Exception:  # noqa: BLE001 — unreadable image is "unknown", not fatal
        return None
    luma = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    keep = luma <= saturation_ignore_above
    if not keep.any():
        return None
    return float(luma[keep].mean())


def illumination_per_keyframe(frames: SceneFrames, image_path_of: dict, cfg: StrataConfig) -> dict:
    return {tok: (luma_of_image(image_path_of[tok], cfg.illumination_saturation_ignore_above)
                  if image_path_of.get(tok) else None) for tok in frames.tokens}


def quantile_edges(values: list, quantiles: list) -> list:
    if not values:
        return [0.0 for _ in quantiles]
    return [float(v) for v in np.quantile(np.asarray(values, dtype=np.float64), quantiles)]


def bin_of(value, edges: list, names: list):
    if value is None:
        return None
    return names[int(np.searchsorted(np.asarray(edges, dtype=np.float64), value, side="right"))]


def compute_strata(frames: SceneFrames, global_centers_by_sample: dict, image_path_of: dict,
                   cfg: StrataConfig) -> Strata:
    dens = density_per_keyframe(frames, global_centers_by_sample, cfg)
    illum = illumination_per_keyframe(frames, image_path_of, cfg)
    edges = quantile_edges(list(dens.values()), cfg.density_quantiles)
    return Strata(
        density=dens, illumination=illum, density_edges=edges,
        density_bin={t: bin_of(v, edges, cfg.density_bin_names) for t, v in dens.items()},
        illumination_bin={t: bin_of(v, cfg.illumination_bin_edges, cfg.illumination_bin_names)
                          for t, v in illum.items()},
    )
```

`double.py`:

```python
"""Stratified double-annotation frame selection, frozen after the first export (spec §6)."""

from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict

import numpy as np

from pipeline.release.config import DoubleConfig, StrataConfig
from pipeline.release.frames import SceneFrames
from pipeline.release.strata import Strata

DOUBLE_SPEC = "dhakascenes/double_annotation/v1"
UNKNOWN = "unknown"


def select_double(strata: Strata, frames: SceneFrames, cfg: DoubleConfig, strata_cfg: StrataConfig) -> dict:
    cells: dict = defaultdict(list)
    for tok in frames.tokens:   # already timestamp-ordered
        cells[(strata.density_bin[tok], strata.illumination_bin[tok] or UNKNOWN)].append(tok)
    n = len(frames.tokens)
    non_empty = sorted(k for k, v in cells.items() if v)
    target = max(math.ceil(cfg.fraction * n), len(non_empty)) if n else 0
    target = min(target, n)
    raw = {k: len(cells[k]) * target / n for k in non_empty} if n else {}
    alloc = {k: max(1, int(math.floor(raw[k]))) for k in non_empty}
    alloc = {k: min(v, len(cells[k])) for k, v in alloc.items()}
    remaining = target - sum(alloc.values())
    for k in sorted(non_empty, key=lambda k: (raw[k] - math.floor(raw[k])), reverse=True):
        if remaining <= 0:
            break
        room = len(cells[k]) - alloc[k]
        take = min(room, remaining)
        alloc[k] += take
        remaining -= take
    rng = np.random.default_rng(cfg.seed)
    selected = []
    cell_rows = []
    for k in non_empty:
        toks = cells[k]
        pick = sorted(rng.choice(np.asarray(toks, dtype=object), size=alloc[k], replace=False).tolist(),
                      key=frames.index.__getitem__)
        selected.extend({"sample_token": t, "density": k[0], "illumination": k[1]} for t in pick)
        cell_rows.append({"density": k[0], "illumination": k[1], "n": len(toks), "selected": alloc[k]})
    selected.sort(key=lambda s: frames.index[s["sample_token"]])
    return {
        "spec": DOUBLE_SPEC, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fraction": cfg.fraction, "seed": cfg.seed, "n_keyframes": n, "n_selected": len(selected),
        "density_bin_edges": list(strata.density_edges), "density_bin_names": list(strata_cfg.density_bin_names),
        "illumination_bin_edges": list(strata_cfg.illumination_bin_edges),
        "illumination_bin_names": list(strata_cfg.illumination_bin_names),
        "method": "proportional allocation by density x illumination cell, >= 1 per non-empty cell, "
                  "largest remainder, seeded choice without replacement within a cell",
        "cells": cell_rows, "selected": selected,
    }


def load_or_select(path: str, strata: Strata, frames: SceneFrames, cfg: DoubleConfig,
                   strata_cfg: StrataConfig, reselect: bool):
    if os.path.isfile(path) and not reselect:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        if doc.get("spec") != DOUBLE_SPEC:
            raise ValueError(f"{path}: spec={doc.get('spec')!r}, expected {DOUBLE_SPEC!r}")
        return doc, True
    if os.path.isfile(path):
        os.replace(path, f"{path}.superseded-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    doc = select_double(strata, frames, cfg, strata_cfg)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, path)
    return doc, False
```

`SceneFrames` in the exporter spans every scene of the export (Task 10 builds one `SceneFrames` per scene and a concatenated "all keyframes" view for selection: tokens from all scenes in scene order, `index` over the concatenation).

- [ ] **Step 4: Run tests + suite, commit**

```bash
git add pipeline/release/strata.py pipeline/release/double.py tests/test_release_strata.py tests/test_release_double.py
git commit -m "pipeline/release: strata (density, luma) + stratified double-annotation selection (spec §6)"
```

---

### Task 10: `scripts/export_release.py` — orchestrate the post-processor

**Files:**
- Modify: `scripts/export_release.py` (`export_release()` at line 502, `main()` at line 752, module docstring)
- Modify: `tests/test_export_release.py` (fixture records no longer carry `attribute`; attribute assertion updated)
- Test: `tests/test_export_release_pipeline.py` (new)

**Interfaces:**
- Consumes every `pipeline/release` module from Tasks 1–9.
- Produces:
  ```python
  DEFAULT_RELEASE_CONFIG = os.path.join(here, "configs", "release.yaml")
  EXCLUDED_TABLE = "sample_annotation_excluded.json"     # at <out>/, beside release_meta.json — NOT inside <out>/<version>/
  STITCH_MAP = "stitch_map.json"; DOUBLE_FILE = "double_annotation.json"   # both at <out>/
  def export_release(prelabels, dataroot, version, out, mapper_path, human_verified_scenes_path=None,
                     blobs="symlink", run_manifest_path=None, pipeline_version=None, *,
                     release_config_path=DEFAULT_RELEASE_CONFIG, tiers="auto_accept", stitch=True,
                     attributes=True, double_fraction=None, human_dir=None, overwrite_tables=False,
                     cvat_export_3d_dir=None, reselect_double=False) -> ExportResult
  ```
  `ExportResult` gains `double: dict | None` and `excluded: list[dict]`. `release_meta.json` gains the keys listed in step 4. Per-annotation output fields added: `dhakascenes_tier_basis`, `dhakascenes_interpolated`, `dhakascenes_chain_id`, `dhakascenes_track_id_pre_stitch`, `dhakascenes_velocity_chain_mps`, `dhakascenes_attribute_basis`, `dhakascenes_verified_by`, `annotator_pass` (only when set), `is_uncertain` (bool), `is_uncertain_reason` (str, `""` when none). `sample.json` rows gain `dbench_double_annotated`.

- [ ] **Step 1: Write the failing test**

`tests/test_export_release_pipeline.py` — reuse the synthetic root builder from `tests/test_export_release.py` (`build_dataroot`, `_record`, `VERSION`) via import; extend it to 4 samples so a gap can be interpolated. If `build_dataroot` hard-codes 2 samples, add an optional `n_samples=2` parameter to it (keep the default so the existing test is unchanged). When `n_samples > 2`: sample 0 keeps `EGO_POSES[0]`, sample 1 keeps `EGO_POSES[1]`, every sample `i >= 2` uses `EGO_POSES[1]` translated by `+5.0 * (i - 1)` m in x (same rotation), and **every** sample `i` gets timestamp `T0_US + 400_000 * i` — so from sample 1 onward the ego translates a constant 5 m per 0.4 s and a box fixed in the ego frame moves at a constant 12.5 m/s in global, which is what the stitch gate and the velocity-derived attribute need.

```python
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from scripts import export_release as er  # noqa: E402
from tests.test_export_release import VERSION, _record, build_dataroot  # noqa: E402

MAPPER = os.path.join(ROOT, "configs", "release_category_map.yaml")


def _write(path, recs):
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")


def _prelabels(s):
    # car track 7 at samples 1,2 then track 8 at sample 4 (gap of 2 -> one interpolated row at sample 3);
    # the ego translates 5 m per keyframe from sample 1 on, so a box fixed in the ego frame moves at a
    # constant 12.5 m/s in global and the forward prediction lands exactly on track 8.
    # A rejected pedestrian and a flagged bus at sample 0 exercise the sidecar.
    recs = [
        _record("k1:CAM_FRONT:0", s[1], "a car", [10.0, 0.5, 0.8], [1.9, 4.5, 1.6], 0.0, "7", 120),
        _record("k2:CAM_FRONT:0", s[2], "a car", [10.0, 0.5, 0.8], [1.9, 4.5, 1.6], 0.0, "7", 110),
        _record("k4:CAM_FRONT:0", s[4], "a car", [10.0, 0.5, 0.8], [1.9, 4.5, 1.6], 0.0, "8", 100),
        _record("k0:CAM_FRONT:3", s[0], "a pedestrian", [-8.0, 1.0, 0.9], [0.6, 0.7, 1.7], 2.0, None, 9),
        _record("k0:CAM_FRONT:4", s[0], "a bus", [20.0, 3.0, 1.5], [2.5, 11.0, 3.2], 0.0, "9", 40),
    ]
    recs[3]["provenance"]["tier"] = "rejected"
    recs[4]["provenance"]["tier"] = "flagged"
    return recs


@pytest.fixture
def exported(tmp_path):
    src = str(tmp_path / "src")
    info = build_dataroot(src, n_samples=5)
    pre = str(tmp_path / "prelabels.jsonl")
    _write(pre, _prelabels(info["sample_tokens"]))
    out = str(tmp_path / "release")
    res = er.export_release(pre, src, VERSION, out, MAPPER, double_fraction=0.5)
    return {"src": src, "out": out, "res": res, **info}


def _load(out, name):
    with open(os.path.join(out, VERSION, f"{name}.json")) as fh:
        return json.load(fh)


def test_accepted_only_with_sidecar(exported):
    anns = _load(exported["out"], "sample_annotation")
    assert all(a["dhakascenes_tier"] == "auto_accept" for a in anns)
    exc = json.load(open(os.path.join(exported["out"], er.EXCLUDED_TABLE)))
    assert sorted(e["dhakascenes_excluded_reason"] for e in exc) == ["tier_flagged", "tier_rejected"]
    assert all(len(e["instance_token"]) == 32 for e in exc)


def test_stitched_chain_and_interpolated_row(exported):
    anns = _load(exported["out"], "sample_annotation")
    inst = _load(exported["out"], "instance")
    car = [a for a in anns if not a["dhakascenes_record_token"].endswith(":3")]
    assert len(car) == 4 and len({a["instance_token"] for a in car}) == 1
    assert len(inst) == 1 and inst[0]["nbr_annotations"] == 4
    interp = [a for a in car if a["dhakascenes_interpolated"]]
    assert len(interp) == 1 and interp[0]["sample_token"] == exported["sample_tokens"][3]
    assert interp[0]["dhakascenes_record_token"].split(":")[1] == "INTERP"
    assert {a["dhakascenes_track_id_pre_stitch"] for a in car} == {"7", "8", None}
    ordered = sorted(car, key=lambda a: a["sample_token"] and exported["sample_tokens"].index(a["sample_token"]))
    for a, b in zip(ordered, ordered[1:]):
        assert a["next"] == b["token"] and b["prev"] == a["token"]
    stitch_map = json.load(open(os.path.join(exported["out"], er.STITCH_MAP)))
    assert stitch_map["k4:CAM_FRONT:0"] == "7" and "k0:CAM_FRONT:3" in stitch_map


def test_attributes_from_chain_velocity(exported):
    anns = _load(exported["out"], "sample_annotation")
    attrs = {a["token"]: a["name"] for a in _load(exported["out"], "attribute")}
    for a in anns:
        assert [attrs[t] for t in a["attribute_tokens"]] == ["vehicle.moving"]   # 12.5 m/s in global (see _prelabels)
        assert a["dhakascenes_attribute_basis"] == "chain_velocity"
        assert a["is_uncertain"] is False and a["is_uncertain_reason"] == ""


def test_double_selection_flags_and_meta(exported):
    samples = _load(exported["out"], "sample")
    flagged = [s["token"] for s in samples if s["dbench_double_annotated"]]
    doc = json.load(open(os.path.join(exported["out"], er.DOUBLE_FILE)))
    assert sorted(flagged) == sorted(s["sample_token"] for s in doc["selected"])
    assert doc["n_selected"] >= 2
    meta = json.load(open(os.path.join(exported["out"], "release_meta.json")))
    assert meta["tiers_admitted"] == "auto_accept"
    assert meta["stitch"]["totals"]["n_interpolated"] == 1
    assert meta["excluded"]["by_reason"] == {"tier_flagged": 1, "tier_rejected": 1}
    assert meta["classes"]["present"]["car"] == 1
    assert "battery_rickshaw" in meta["classes"]["not_producible"]
    assert meta["release_config"]["sha256"]


def test_overwrite_tables_reuses_selection_and_keeps_blobs(exported, tmp_path):
    out = exported["out"]
    before = json.load(open(os.path.join(out, er.DOUBLE_FILE)))["selected"]
    pre = str(tmp_path / "prelabels.jsonl")   # same records, re-export in place
    er.export_release(pre, exported["src"], VERSION, out, MAPPER, overwrite_tables=True, double_fraction=0.5)
    after = json.load(open(os.path.join(out, er.DOUBLE_FILE)))["selected"]
    assert after == before
    assert os.path.exists(os.path.join(out, "samples"))


def test_tiers_all_reproduces_legacy(tmp_path):
    src = str(tmp_path / "src")
    info = build_dataroot(src, n_samples=5)
    pre = str(tmp_path / "prelabels.jsonl")
    _write(pre, _prelabels(info["sample_tokens"]))
    res = er.export_release(pre, src, VERSION, str(tmp_path / "rel"), MAPPER, tiers="all", stitch=False,
                            attributes=False, double_fraction=0.0)
    anns = res.tables["sample_annotation"]
    assert len(anns) == 5 and res.excluded == []
    assert all(a["attribute_tokens"] == [] for a in anns)
    assert len(res.tables["instance"]) == 4          # 7, 8, 9, and the untracked pedestrian
```

Update `tests/test_export_release.py`: remove the `"moving"` argument from the first `_record(...)` call in `write_prelabels` (pipeline records carry no attribute), and change `test_visibility_and_attributes` to:

```python
    assert [attrs[t] for t in anns["k0:CAM_FRONT:0"]["attribute_tokens"]] == ["vehicle.moving"]
    assert [attrs[t] for t in anns["k1:CAM_FRONT:0"]["attribute_tokens"]] == ["vehicle.moving"]
    assert anns["k1:CAM_FRONT:3"]["attribute_tokens"] == []
```

(the two-sample car chain moves ~11 m in global between samples; the pedestrian is a singleton). `test_tokens_chains_and_instances` is unchanged.

- [ ] **Step 2: Run to verify failures**

Run: `$PY -m pytest tests/test_export_release_pipeline.py tests/test_export_release.py -v`
Expected: TypeError on the new keyword arguments; attribute assertion fails.

- [ ] **Step 3: Rewrite `export_release()`**

Replace the body from `mapper = CategoryMapper.load(mapper_path)` through the `instances.append(...)` loop with the pipeline below; keep the surrounding pieces (`SourceRoot`, refusal checks, table writing, meta) and extend them as shown.

```python
from dataclasses import replace as dc_replace
from pipeline.release.attributes import assign_attributes
from pipeline.release.config import load_release_config
from pipeline.release.double import load_or_select
from pipeline.release.frames import CloudSource, SceneFrames, scene_frames_from_root
from pipeline.release.human import load_human, merge_human
from pipeline.release.stitch import stitch_scene
from pipeline.release.strata import compute_strata
from pipeline.release.tiers import ADMIT_ALL, ADMIT_AUTO, ADMIT_MODES, partition

DEFAULT_RELEASE_CONFIG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "release.yaml")
EXCLUDED_TABLE = "sample_annotation_excluded.json"
STITCH_MAP = "stitch_map.json"
DOUBLE_FILE = "double_annotation.json"
TAXONOMY_DHAKA = os.path.join(os.path.dirname(DEFAULT_RELEASE_CONFIG), "taxonomy_pilot_dhaka.yaml")
T_TOLERANCE_NS = 1_000_000   # a record's t_ns is expected to agree with its sample's timestamp to 1 ms; the sample table wins


def _legacy_identity(rows: list, frames: SceneFrames) -> None:
    """--no-stitch: today's identity (Stage 7 track_id, else singleton), same keys stitch.py sets."""
    for r in rows:
        chain = str(r["track_id"]) if r.get("track_id") is not None else f"det:{r['token']}"
        r.update(instance_token=f"chain:{frames.scene_token}:{chain}", stitch_chain_id=chain,
                 stitch_track_id_pre=str(r["track_id"]) if r.get("track_id") is not None else None,
                 stitch_interpolated=False, stitch_tier_basis="gate")


def _producible_classes(mapper: CategoryMapper) -> set:
    with open(TAXONOMY_DHAKA, "r", encoding="utf-8") as fh:
        phrases = set((yaml.safe_load(fh) or {}).get("prompt_phrase", {}).values())
    return {mapper.mapping[CategoryMapper.normalise(p)] for p in phrases if CategoryMapper.normalise(p) in mapper.mapping}
```

Inside `export_release()`:

```python
    if tiers not in ADMIT_MODES:
        raise ExportError(f"--tiers {tiers!r} is not one of {ADMIT_MODES}")
    out_tables = os.path.join(out, version)
    if os.path.isdir(out_tables) and os.listdir(out_tables) and not overwrite_tables:
        raise ExportError(f"{out}/{version} already exists and is not empty; refusing to overwrite "
                          "(--overwrite-tables rewrites the annotation tables in place)")
    if overwrite_tables and not os.path.isfile(os.path.join(out_tables, "sample.json")):
        raise ExportError(f"--overwrite-tables needs an existing export at {out_tables}")
    cfg = load_release_config(release_config_path)
    if double_fraction is not None:
        cfg = dc_replace(cfg, double=dc_replace(cfg.double, fraction=float(double_fraction)))

    mapper = CategoryMapper.load(mapper_path)
    records, source_files = load_prelabels(prelabels)
    if not records:
        raise ExportError("no records to export")
    unknown = sorted({r["sample_token"] for r in records if r["sample_token"] not in src.sample})
    if unknown:
        raise ExportError(f"{len(unknown)} sample_token(s) not in {src.table_dir}/sample.json: {unknown[:5]}")

    # --- scenes, keyframe order, record timestamps ---------------------------
    by_scene: dict = defaultdict(list)
    for r in records:
        by_scene[src.sample[r["sample_token"]]["scene_token"]].append(r)
    frames_of = {st: scene_frames_from_root(src, st) for st in by_scene}
    n_t_ns_corrected = 0
    for st, rows in by_scene.items():
        fr = frames_of[st]
        for r in rows:
            t_sample = fr.timestamps_ns[fr.index[r["sample_token"]]]
            if abs(int(r["t_ns"]) - t_sample) > T_TOLERANCE_NS:
                n_t_ns_corrected += 1      # the sample table is the timeline nuScenes' box_velocity() reads; recorded in meta
            r["t_ns"] = t_sample
    if n_t_ns_corrected:
        print(f"export_release: {n_t_ns_corrected} record(s) carried a t_ns that disagrees with sample.timestamp "
              f"by > 1 ms; the sample timestamp was used", file=sys.stderr)

    # --- 1. stitch (all tiers) ------------------------------------------------
    stitch_meta: dict = {"enabled": bool(stitch), "per_scene": {}, "totals": {}}
    rows_all: list = []
    for st in sorted(by_scene):
        fr = frames_of[st]
        if stitch:
            clouds = CloudSource(src.dataroot, cvat_export_3d_dir, fr, src)
            try:
                rows, stats = stitch_scene(by_scene[st], fr, clouds, cfg.stitch)
            finally:
                clouds.close()
            stitch_meta["per_scene"][fr.scene_name] = vars(stats)
        else:
            rows = list(by_scene[st])
            _legacy_identity(rows, fr)
        rows_all.extend(rows)
    if stitch:
        keys = ("n_records_in", "n_fragments", "n_chains", "n_interpolated", "n_interpolated_raw_basis",
                "n_interpolated_no_cloud")
        stitch_meta["totals"] = {k: sum(v[k] for v in stitch_meta["per_scene"].values()) for k in keys}
        stitch_meta["totals"]["joins_by_gap"] = {}
        for v in stitch_meta["per_scene"].values():
            for g, n in v["joins_by_gap"].items():
                stitch_meta["totals"]["joins_by_gap"][str(g)] = stitch_meta["totals"]["joins_by_gap"].get(str(g), 0) + n
        stitch_meta["config"] = vars(cfg.stitch)

    # --- 2. human merge ------------------------------------------------------
    superseded: dict = {}
    human_meta: dict = {"enabled": bool(human_dir), "dir": os.path.abspath(human_dir) if human_dir else None}
    if human_dir:
        human_rows, coverage = load_human(human_dir)
        for r in human_rows:
            if r["sample_token"] not in src.sample:
                raise ExportError(f"human record {r['token']}: sample {r['sample_token']} not in this dataroot")
            st = src.sample[r["sample_token"]]["scene_token"]
            r["t_ns"] = frames_of.setdefault(st, scene_frames_from_root(src, st)).timestamps_ns[
                frames_of[st].index[r["sample_token"]]]
            chain = r["instance_token"]
            r.update(instance_token=f"chain:{st}:{chain}", stitch_chain_id=chain, stitch_track_id_pre=None,
                     stitch_interpolated=False, stitch_tier_basis="human")
        merge = merge_human(rows_all, human_rows, coverage)
        rows_all.extend(merge.rows)
        superseded = merge.superseded
        human_meta.update(stats=merge.stats, half_imported_samples=merge.half_imported,
                          coverage={k: len(v) for k, v in merge.coverage.items()})
    category_of = mapper.resolve(r["category"] for r in rows_all)

    # --- 3. tier filter --------------------------------------------------------
    included, excluded = partition(rows_all, tiers, superseded)
    if not included:
        raise ExportError("no rows admitted to sample_annotation")

    # --- 4. instances (by chain), global geometry -----------------------------
    def instance_token_of(r):
        return make_token("instance", src.sample[r["sample_token"]]["scene_token"], r["stitch_chain_id"])

    global_center_of: dict = {}
    global_quat_of: dict = {}
    for r in included + excluded:
        pose = src.lidar_ego_pose(r["sample_token"])
        t_g, q_g = box_ego_to_global(r["translation_m"], r["rotation_wxyz"], pose)
        global_center_of[r["token"]] = np.asarray(t_g)
        global_quat_of[r["token"]] = q_g
    by_instance: dict = defaultdict(list)
    instance_category: dict = {}
    for r in included:
        inst = instance_token_of(r)
        cat = category_of[r["category"]]
        if inst in instance_category and instance_category[inst] != cat:
            raise ExportError(f"instance {inst} changes class {instance_category[inst]} -> {cat}")
        instance_category[inst] = cat
        r["__instance__"] = inst
        by_instance[inst].append(r)

    # --- 5. attributes (final chains only) -----------------------------------
    for r in included:
        r["instance_token"] = r["__instance__"]     # attributes.py groups by instance_token
    if attributes:
        assign_attributes(included, instance_category, global_center_of, cfg.attributes)
    else:
        for r in included:
            r.update(velocity_chain_mps=None, attr_state=None, attr_name=None, attribute_basis=None)
```

Then the per-instance loop stays as today (sorted instances, rows sorted by timestamp, `ann_tokens = [make_token("annotation", inst, r["token"]) ...]`, devkit check, visibility) but builds the annotation row with a shared helper used for both tables:

```python
    def annotation_row(r, inst, tok, prev, nxt, vis_token, frac, t_g, q_g):
        prov = r.get("provenance") or {}
        attr_list = []
        name = r.get("attr_name")
        if name:
            attribute_tokens.setdefault(name, make_token("attribute", name))
            attr_list = [attribute_tokens[name]]
        row = {
            "token": tok, "sample_token": r["sample_token"], "instance_token": inst,
            "visibility_token": vis_token, "attribute_tokens": attr_list,
            "translation": t_g, "size": [float(v) for v in r["size_wlh_m"]], "rotation": q_g,
            "prev": prev, "next": nxt, "num_lidar_pts": int(r["num_lidar_pts"]), "num_radar_pts": 0,
            "dhakascenes_record_token": r["token"], "dhakascenes_source": prov.get("source", "pipeline"),
            "dhakascenes_tier": prov.get("tier") if prov.get("source") not in HUMAN_SOURCES else None,
            "dhakascenes_tier_basis": r.get("stitch_tier_basis"),
            "dhakascenes_interpolated": bool(r.get("stitch_interpolated")),
            "dhakascenes_chain_id": r.get("stitch_chain_id"),
            "dhakascenes_track_id_pre_stitch": r.get("stitch_track_id_pre"),
            "num_lidar_pts_basis": r.get("num_lidar_pts_basis"),
            "visibility_basis": "camera_fov_corner_fraction" if frac is not None else "assumed_full",
            "dhakascenes_velocity_chain_mps": r.get("velocity_chain_mps"),
            "dhakascenes_attribute_basis": r.get("attribute_basis"),
            "dhakascenes_verified_by": prov.get("verified_by"),
            "is_uncertain": bool(r.get("is_uncertain")) if r.get("is_uncertain") is not None else False,
            "is_uncertain_reason": r.get("is_uncertain_reason") or "",
        }
        if r.get("velocity_mps") is not None:
            row["dhakascenes_velocity_ego_mps"] = [float(v) for v in r["velocity_mps"]]
        if prov.get("annotator_pass"):
            row["annotator_pass"] = prov["annotator_pass"]
        return row
```

Excluded rows are written with the same helper (`prev`/`next` empty, `inst = instance_token_of(r)`, visibility computed the same way) plus `"dhakascenes_excluded_reason": r["excluded_reason"]`, to `<out>/sample_annotation_excluded.json`. Remove the old `if source in HUMAN_SOURCES and prov.get("verification_pass"): ann["annotator_pass"] = "A"` line.

- [ ] **Step 4: Strata, double selection, sample flags, stitch map, meta**

After the tables are built (before writing):

```python
    # --- 6. strata + double annotation ---------------------------------------
    centers_by_sample: dict = defaultdict(list)
    for r in included:
        centers_by_sample[r["sample_token"]].append(global_center_of[r["token"]])
    all_tokens, all_ts, all_poses = [], [], {}
    scene_names = []
    for st in sorted(frames_of, key=lambda s: frames_of[s].scene_name):
        fr = frames_of[st]
        all_tokens.extend(fr.tokens); all_ts.extend(fr.timestamps_ns); all_poses.update(fr.poses)
        scene_names.append(fr.scene_name)
    all_frames = SceneFrames("*", "+".join(scene_names), all_tokens, all_ts, all_poses)
    image_path_of = {}
    for tok in all_tokens:
        sd = src.sd_by_sample.get(tok, {}).get(cfg.strata.illumination_channel)
        image_path_of[tok] = os.path.join(src.dataroot, sd["filename"]) if sd else None
    strata = compute_strata(all_frames, centers_by_sample, image_path_of, cfg.strata)
    double_doc = None
    if cfg.double.fraction > 0:
        os.makedirs(out, exist_ok=True)
        double_doc, double_reused = load_or_select(os.path.join(out, DOUBLE_FILE), strata, all_frames,
                                                   cfg.double, cfg.strata, reselect_double)
    double_tokens = {s["sample_token"] for s in (double_doc or {}).get("selected", [])}
    sample_rows = [dict(r, dbench_double_annotated=(r["token"] in double_tokens)) for r in src.tables["sample"]]
    stitch_map = {r["token"]: r["stitch_chain_id"] for r in rows_all
                  if not r.get("stitch_interpolated") and (r.get("provenance") or {}).get("source") not in HUMAN_SOURCES}
```

Writing: in non-overwrite mode copy blobs and passthrough tables as today, **then** overwrite `sample.json` with `sample_rows`; in overwrite mode skip the blob/passthrough copy and write only the five annotation tables, `sample.json` (from `sample_rows`), the sidecar, `stitch_map.json`, `release_meta.json`. Write the sidecar and stitch map with `json.dump(..., indent=1)` at `<out>/`.

`release_meta.json` additions (keep every existing key):

```python
        "release_config": cfg.as_dict(),
        "tiers_admitted": tiers,
        "stitch": stitch_meta,
        "human": human_meta,
        "excluded": {"table": EXCLUDED_TABLE, "n": len(excluded),
                     "by_reason": dict(sorted(Counter(r["excluded_reason"] for r in excluded).items()))},
        "attributes": {"enabled": attributes, "threshold_mps": cfg.attributes.moving_speed_threshold_mps,
                       "n_with_attribute": ..., "by_name": Counter(names), "by_basis": Counter(bases)},
        "strata": {"density_bin_edges": strata.density_edges, "density_bin_names": cfg.strata.density_bin_names,
                   "illumination_bin_edges": cfg.strata.illumination_bin_edges,
                   "illumination_bin_names": cfg.strata.illumination_bin_names,
                   "n_illumination_unknown": sum(1 for v in strata.illumination.values() if v is None),
                   "per_keyframe": {tok: {"density": strata.density[tok], "density_bin": strata.density_bin[tok],
                                          "luma": strata.illumination[tok], "illumination_bin": strata.illumination_bin[tok]}
                                    for tok in all_tokens}},
        "double_annotation": None if double_doc is None else {"file": DOUBLE_FILE, "reused": double_reused,
                                                             "n_selected": double_doc["n_selected"], "cells": double_doc["cells"]},
        "range": {"cap_m": <max BEV range over included+excluded rows in ego frame, rounded 1 dp>,
                  "effective_p99_m_by_class": {cls: p99 of hypot(x,y) of included ego-frame boxes of that class}},
        "classes": {"present": {cls: n_instances}, "absent_on_route": sorted(producible - present),
                    "not_producible": sorted(set(mapper.classes) - producible), "producible_by_vocabulary": sorted(producible)},
        "git_sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=here).stdout.strip() or None,
        "records": {"n_loaded": len(records), "n_t_ns_corrected": n_t_ns_corrected},
```

where `producible = _producible_classes(mapper)` and `present` counts instances per class from `instance_category`.

- [ ] **Step 5: `main()` flags**

```python
    ap.add_argument("--release-config", default=DEFAULT_RELEASE_CONFIG)
    ap.add_argument("--tiers", choices=ADMIT_MODES, default=ADMIT_AUTO,
                    help="pipeline tiers admitted to sample_annotation; 'all' reproduces the pre-2026-09-07 output")
    ap.add_argument("--stitch", dest="stitch", action="store_true", default=True)
    ap.add_argument("--no-stitch", dest="stitch", action="store_false")
    ap.add_argument("--attributes", dest="attributes", action="store_true", default=True)
    ap.add_argument("--no-attributes", dest="attributes", action="store_false")
    ap.add_argument("--double-fraction", type=float, default=None, help="override configs/release.yaml double.fraction; 0 disables")
    ap.add_argument("--reselect-double", action="store_true", help="discard an existing double_annotation.json (kept as .superseded-*)")
    ap.add_argument("--human", default=None, help="<work_root>/stage10_human from scripts/import_cvat_3d.py")
    ap.add_argument("--cvat-export-3d-dir", default=None, help="<work_root>/cvat_export_3d: task.zip clouds for interpolated point counts")
    ap.add_argument("--overwrite-tables", action="store_true", help="rewrite annotation tables/sidecars/meta in an existing export; blobs untouched")
```

pass them through to `export_release(...)`; print the stitch totals and excluded counts in the summary line. Update the module docstring's "Attributes" and "Identity" paragraphs to describe the new behaviour (chain velocity; stitching) and add a "Tiers" paragraph.

- [ ] **Step 6: Run the two exporter test files, then the whole suite**

Run: `$PY -m pytest tests/test_export_release_pipeline.py tests/test_export_release.py -v && $PY -m pytest tests/ -q`
Expected: PASS, including `test_dbench_ingest_validate_passes` (it runs dbench's validator on the output; the new fields are extras it ignores and `dbench_double_annotated` is now present).

- [ ] **Step 7: Commit**

```bash
git add scripts/export_release.py tests/test_export_release.py tests/test_export_release_pipeline.py
git commit -m "export_release: stitch -> human -> tiers -> attributes -> strata/double pipeline, sidecars, stitch_map (spec §2, §5, §6)"
```

---

### Task 11: `pipeline/release/note.py` — `DELIVERY_NOTE.md`

**Files:**
- Create: `pipeline/release/note.py`
- Modify: `scripts/export_release.py` `main()` (`--note/--no-note`, default on; calls `write_note` after export)
- Test: `tests/test_release_note.py`

**Interfaces:**
- Consumes: `release_meta.json` (Task 10 shape), optional Stage 9 `run_manifest.json` (`config.min_lidar_returns`, `config.conf_gate`, `config.spatial_multiplier`, `spec`), optional `<human_dir>/import_manifest.json` (Task 16), optional `double_annotation.json`.
- Produces:
  ```python
  def render_note(meta: dict, stage9_manifest: dict | None, import_manifest: dict | None, double_doc: dict | None,
                  stage_tree: str | None, chunk_name: str) -> str
  def write_note(out_root: str, *, stage9_manifest_path: str | None, import_manifest_path: str | None,
                 stage_tree: str | None, chunk_name: str) -> str     # writes <out_root>/DELIVERY_NOTE.md, returns path
  ```
  Sections, in this order, each a `## ` heading: Provenance; Human pass; Annotation rule; Range; Tiers; Classes; Identity (stitching); Attributes; Uncertainty; Double annotation; Anonymisation; Extra layers; Files. Numbers come from `meta`; nothing typed by hand except the fixed sentences quoted in the spec §8.

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.release.note import render_note, write_note  # noqa: E402


def _meta():
    return {
        "version": "v1.0-dhaka-fixed", "created_utc": "2026-09-08T00:00:00Z", "git_sha": "abc123",
        "pipeline_version": "dhakascenes-pilot/stage9_qa/v1", "tiers_admitted": "auto_accept",
        "counts": {"n_annotations": 10, "n_instances": 4, "n_scenes": 1, "per_class": {"car": 3, "pedestrian": 1}},
        "excluded": {"table": "sample_annotation_excluded.json", "n": 6, "by_reason": {"tier_rejected": 4, "tier_flagged": 2}},
        "stitch": {"enabled": True, "totals": {"n_fragments": 9, "n_chains": 4, "n_interpolated": 1, "joins_by_gap": {"1": 4, "3": 1}},
                   "per_scene": {"chunk_0000": {"median_len_before": 1.0, "median_len_after": 2.5}}, "config": {"max_gap_keyframes": 3}},
        "attributes": {"enabled": True, "threshold_mps": 0.5, "n_with_attribute": 8, "by_name": {"vehicle.moving": 8}, "by_basis": {"chain_velocity": 8}},
        "range": {"cap_m": 50.0, "effective_p99_m_by_class": {"car": 41.2, "pedestrian": 22.0}},
        "classes": {"present": {"car": 3, "pedestrian": 1}, "absent_on_route": ["barrier", "traffic_cone", "construction_element"],
                    "not_producible": ["animal", "battery_rickshaw"], "producible_by_vocabulary": ["car", "pedestrian", "barrier"]},
        "human": {"enabled": False},
        "double_annotation": {"file": "double_annotation.json", "reused": False, "n_selected": 35,
                              "cells": [{"density": "High", "illumination": "day", "n": 300, "selected": 20}]},
        "strata": {"density_bin_edges": [0.001, 0.002, 0.003], "illumination_bin_edges": [45, 75, 95], "n_illumination_unknown": 0},
        "release_config": {"sha256": "cfgsha", "benchmark_source": {"path": "/x/benchmark_v1.0.yaml", "sha256": "bsha"}},
        "visibility": {"basis": "camera_fov_corner_fraction"},
        "num_lidar_pts_basis": ["single_sweep_ground_filtered_pre_inflation"],
        "mapper": {"used": {"a car": "car", "a pedestrian": "pedestrian"}},
    }


S9 = {"spec": "dhakascenes-pilot/stage9_qa/v1", "config": {"min_lidar_returns": 5, "conf_gate": 0.5, "spatial_multiplier": 2.0}}


def test_note_states_the_rule_range_classes_and_anonymisation():
    text = render_note(_meta(), S9, None, {"fraction": 0.05, "n_keyframes": 684, "seed": 20260812}, "/work/chunk_0000", "day1_chunk_0000")
    assert ">= 5 LiDAR returns" in text and "confidence >= 0.5" in text and "2.0x class prior" in text
    assert "V does not apply" in text
    assert "50" in text and "41.2" in text
    assert "not producible" in text.lower() and "battery_rickshaw" in text
    assert "absent on this route" in text.lower() and "traffic_cone" in text
    assert "after annotation" in text and "un-blurred" in text
    assert "35" in text and "20260812" in text
    assert "no human pass" in text.lower()
    assert "abc123" in text and "bsha" in text


def test_note_reports_human_pass_when_present():
    meta = _meta()
    meta["human"] = {"enabled": True, "stats": {"n_rows_review": 12, "n_rows_double_A": 3, "n_rows_double_B": 4,
                                                 "n_samples_review": 5, "n_samples_double_A": 1, "n_samples_double_B": 1},
                     "half_imported_samples": ["s9"]}
    imp = {"tasks": [{"task_id": 41, "kind": "review", "scene": "chunk_0000", "assignee": "ann_a"}]}
    text = render_note(meta, S9, imp, None, None, "day1_chunk_0000")
    assert "ann_a" in text and "task 41" in text and "s9" in text


def test_write_note_writes_file(tmp_path):
    (tmp_path / "release_meta.json").write_text(json.dumps(_meta()))
    p = write_note(str(tmp_path), stage9_manifest_path=None, import_manifest_path=None, stage_tree=None, chunk_name="x")
    assert os.path.isfile(p) and open(p).read().startswith("# ")
```

- [ ] **Step 2: Run to verify failure**, then **Step 3: implement `note.py`**

```python
"""DELIVERY_NOTE.md — the per-chunk text block the benchmark requires (spec §8)."""

from __future__ import annotations

import json
import os

RULE_SENTENCE = ("A box ships iff it has >= {n} LiDAR returns (single-sweep, ground-filtered, pre-inflation) "
                 "AND detector confidence >= {c} AND its BEV footprint is <= {m}x class prior. There are no "
                 "camera-only boxes, so the visibility term V does not apply; `visibility_token` is a camera "
                 "field-of-view proxy, not an occlusion estimate.")
ANON_SENTENCE = ("After annotation. The annotated images are un-blurred; face and plate blurring is applied "
                 "to the released images afterwards, so any box drawn from image evidence saw the original pixels.")


def _kv(rows):
    return "\n".join(f"- {k}: {v}" for k, v in rows)


def render_note(meta, stage9_manifest, import_manifest, double_doc, stage_tree, chunk_name) -> str:
    s9 = (stage9_manifest or {}).get("config", {})
    rule = RULE_SENTENCE.format(n=s9.get("min_lidar_returns", "?"), c=s9.get("conf_gate", "?"),
                                m=s9.get("spatial_multiplier", "?"))
    cls = meta.get("classes", {})
    rng = meta.get("range", {})
    st = meta.get("stitch", {})
    hu = meta.get("human", {})
    at = meta.get("attributes", {})
    dbl = meta.get("double_annotation")
    ex = meta.get("excluded", {})
    rc = meta.get("release_config", {})
    lines = [f"# {chunk_name} — delivery note", ""]
    lines += ["## Provenance", _kv([
        ("export created (UTC)", meta.get("created_utc")), ("nuScenes version dir", meta.get("version")),
        ("pipeline git sha", meta.get("git_sha")), ("Stage 9 spec", (stage9_manifest or {}).get("spec") or meta.get("pipeline_version")),
        ("stage tree", stage_tree or "(not recorded)"),
        ("release config sha256", rc.get("sha256")),
        ("benchmark definition", f"{rc.get('benchmark_source', {}).get('path')} sha256 {rc.get('benchmark_source', {}).get('sha256')}"),
    ]), ""]
    if hu.get("enabled"):
        stt = hu.get("stats", {})
        task_lines = [f"task {t.get('task_id')} ({t.get('kind')}, {t.get('scene')}, assignee {t.get('assignee')})"
                      for t in (import_manifest or {}).get("tasks", [])]
        lines += ["## Human pass", _kv([
            ("review rows / samples", f"{stt.get('n_rows_review', 0)} / {stt.get('n_samples_review', 0)}"),
            ("double pass A rows / samples", f"{stt.get('n_rows_double_A', 0)} / {stt.get('n_samples_double_A', 0)}"),
            ("double pass B rows / samples", f"{stt.get('n_rows_double_B', 0)} / {stt.get('n_samples_double_B', 0)}"),
            ("half-imported double frames (B without A)", ", ".join(hu.get("half_imported_samples", [])) or "none"),
            ("CVAT tasks imported", "; ".join(task_lines) or "(no import manifest)"),
        ]), ""]
    else:
        lines += ["## Human pass", "No human pass has run on this export: every row is a machine pre-annotation "
                  "(`dhakascenes_source: pipeline`).", ""]
    lines += ["## Annotation rule", rule, "", "## Range", _kv([
        ("pipeline range cap", f"{rng.get('cap_m')} m (Stage 1 / eval region)"),
        ("effective p99 range per class (included boxes)", ", ".join(f"{k} {v} m" for k, v in sorted((rng.get('effective_p99_m_by_class') or {}).items())) or "n/a"),
    ]), ""]
    lines += ["## Tiers", _kv([
        ("admitted to sample_annotation", meta.get("tiers_admitted")),
        ("excluded rows", f"{ex.get('n', 0)} in {ex.get('table')} — " + ", ".join(f"{k} {v}" for k, v in sorted((ex.get('by_reason') or {}).items()))),
    ]), "`sample_annotation_excluded.json` is not a nuScenes table; the devkit does not load it.", ""]
    present = cls.get("present", {})
    lines += ["## Classes", _kv([
        ("present (instances)", ", ".join(f"{k} {v}" for k, v in sorted(present.items())) or "none"),
        ("in the detector vocabulary but absent on this route", ", ".join(cls.get("absent_on_route", [])) or "none"),
        ("not producible by the 12-phrase detector vocabulary", ", ".join(cls.get("not_producible", [])) or "none"),
    ]), ""]
    tot = st.get("totals", {})
    lines += ["## Identity (stitching)", _kv([
        ("enabled", st.get("enabled")), ("fragments -> chains", f"{tot.get('n_fragments')} -> {tot.get('n_chains')}"),
        ("joins by gap (keyframes)", json.dumps(tot.get("joins_by_gap", {}))), ("interpolated rows", tot.get("n_interpolated")),
        ("median chain length before -> after, per scene", "; ".join(f"{k} {v.get('median_len_before')} -> {v.get('median_len_after')}" for k, v in (st.get("per_scene") or {}).items()) or "n/a"),
        ("max gap", f"{(st.get('config') or {}).get('max_gap_keyframes')} keyframes"),
    ]), "A class flip along one object stays two instances (class-agnostic stitching is off).", ""]
    lines += ["## Attributes", _kv([
        ("derivation", f"chain velocity (nuScenes box_velocity semantics); moving if > {at.get('threshold_mps')} m/s else stopped; "
                       "single-annotation instances get none; static classes and animals never; parked / sitting / without_rider are human-only"),
        ("rows with an attribute", at.get("n_with_attribute")), ("by name", json.dumps(at.get("by_name", {}))), ("by basis", json.dumps(at.get("by_basis", {}))),
    ]), ""]
    lines += ["## Uncertainty", "`is_uncertain` is false on every pipeline row: no class-uncertainty signal survives to the "
              "pre-labels in this phase (the VLM check is off). Human rows carry what the annotator set.", ""]
    if dbl:
        cells = "; ".join(f"{c['density']}/{c['illumination']} {c['selected']}/{c['n']}" for c in dbl.get("cells", []))
        lines += ["## Double annotation", _kv([
            ("frames selected", f"{dbl.get('n_selected')} of {(double_doc or {}).get('n_keyframes', '?')} "
                                f"(fraction {(double_doc or {}).get('fraction', '?')}, seed {(double_doc or {}).get('seed', '?')})"),
            ("stratification", f"density quartiles {meta.get('strata', {}).get('density_bin_edges')} x illumination luma edges {meta.get('strata', {}).get('illumination_bin_edges')}"),
            ("cells (selected/n)", cells), ("selection reused from an earlier export", dbl.get("reused")),
            ("A/B rows imported", "yes" if hu.get("enabled") and (hu.get("stats") or {}).get("n_rows_double_A") else "no"),
        ]), ""]
    else:
        lines += ["## Double annotation", "Disabled for this export (double fraction 0).", ""]
    lines += ["## Anonymisation", ANON_SENTENCE, ""]
    used = meta.get("mapper", {}).get("used", {})
    lines += ["## Extra layers", "- road/: driveable-surface lidarseg (nuScenes-lidarseg tables + .bin), reads against boxes/samples/.",
              "- coco_2d/: 2D-only auxiliary layer with the detector's phrase names; not consumed by the 3D benchmark. Phrase -> class: "
              + ", ".join(f"{k} -> {v}" for k, v in sorted(used.items())), ""]
    lines += ["## Files", "- boxes/<version>/: the 13 nuScenes tables; sample.json carries `dbench_double_annotated`.",
              "- boxes/sample_annotation_excluded.json, boxes/stitch_map.json, boxes/double_annotation.json, boxes/release_meta.json.",
              f"- num_lidar_pts basis: {', '.join(meta.get('num_lidar_pts_basis', []))}; visibility basis: {meta.get('visibility', {}).get('basis')}.", ""]
    return "\n".join(lines)


def write_note(out_root: str, *, stage9_manifest_path, import_manifest_path, stage_tree, chunk_name) -> str:
    with open(os.path.join(out_root, "release_meta.json"), "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    s9 = json.load(open(stage9_manifest_path)) if stage9_manifest_path and os.path.isfile(stage9_manifest_path) else None
    imp = json.load(open(import_manifest_path)) if import_manifest_path and os.path.isfile(import_manifest_path) else None
    dpath = os.path.join(out_root, "double_annotation.json")
    dbl = json.load(open(dpath)) if os.path.isfile(dpath) else None
    path = os.path.join(out_root, "DELIVERY_NOTE.md")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(render_note(meta, s9, imp, dbl, stage_tree, chunk_name))
    os.replace(tmp, path)
    return path
```

In `export_release.main()`: add `--note/--no-note` (default on), `--stage-tree` (str, default None), `--chunk-name` (default: basename of `--out`'s parent), `--import-manifest` (default `<--human>/import_manifest.json` when `--human` given); after a successful export call `write_note(out, stage9_manifest_path=<the run_manifest path the exporter resolved>, ...)` and print the path.

- [ ] **Step 4: Run tests + suite, commit**

```bash
git add pipeline/release/note.py scripts/export_release.py tests/test_release_note.py
git commit -m "pipeline/release/note: generated DELIVERY_NOTE.md (spec §8)"
```

---

### Task 12: `scripts/check_release.py` — the checklist

**Files:**
- Create: `scripts/check_release.py`
- Modify: `scripts/export_release.py` `main()` — run the checker at the end and return its exit code if worse
- Test: `tests/test_check_release.py`

**Interfaces:**
- Produces:
  ```python
  @dataclass class Report: errors: list[str]; warnings: list[str]; info: dict
  def check_release(out_root: str, version: str | None = None) -> Report   # version auto-detected: the single subdir with sample.json
  def main(argv) -> int     # 0 clean, 1 warnings only, 2 errors; prints every line
  ```
  Errors (spec §8): non-positive size; quaternion norm off by > 1e-3; dangling `instance_token` / `category_token` / `sample_token` / `attribute_tokens`; `prev`/`next` asymmetric or pointing to another instance; `nbr_annotations`/first/last inconsistent; pipeline row with `dhakascenes_tier != "auto_accept"` inside `sample_annotation` unless `release_meta.tiers_admitted == "all"`; human row (`dhakascenes_source` in human sources) without `dhakascenes_verified_by`; fewer `dbench_double_annotated` samples than `ceil(fraction × n_samples)` when `double_annotation.json` exists; a double-flagged sample with rows from only one pass when `release_meta.human.stats.n_rows_double_A > 0`. Warnings: `size[0] > size[1]`; non-static multi-annotation instance rows without attribute; any `annotator_pass == "B"` rows inline; zero-instance classes; interpolated fraction > 20 %; `DELIVERY_NOTE.md` missing.

- [ ] **Step 1: Write the failing test** — build a minimal valid export in `tmp_path` with a helper, then mutate one thing per test:

```python
from __future__ import annotations

import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.check_release import check_release  # noqa: E402

V = "v1.0-x"


def _export(tmp_path, anns=None, samples=None, meta=None, instances=None, note=True, double=None):
    d = tmp_path / V
    d.mkdir(exist_ok=True)
    cat = [{"token": "c1", "name": "car"}, {"token": "c2", "name": "pedestrian"}]
    inst = instances if instances is not None else [
        {"token": "i1", "category_token": "c1", "nbr_annotations": 2, "first_annotation_token": "a1", "last_annotation_token": "a2"}]
    base_anns = [
        {"token": "a1", "sample_token": "s1", "instance_token": "i1", "visibility_token": "4", "attribute_tokens": ["t1"],
         "translation": [1, 2, 0], "size": [1.8, 4.5, 1.6], "rotation": [1, 0, 0, 0], "prev": "", "next": "a2",
         "num_lidar_pts": 5, "num_radar_pts": 0, "dhakascenes_source": "pipeline", "dhakascenes_tier": "auto_accept",
         "dhakascenes_interpolated": False},
        {"token": "a2", "sample_token": "s2", "instance_token": "i1", "visibility_token": "4", "attribute_tokens": ["t1"],
         "translation": [2, 2, 0], "size": [1.8, 4.5, 1.6], "rotation": [1, 0, 0, 0], "prev": "a1", "next": "",
         "num_lidar_pts": 5, "num_radar_pts": 0, "dhakascenes_source": "pipeline", "dhakascenes_tier": "auto_accept",
         "dhakascenes_interpolated": False},
    ]
    tables = {
        "sample_annotation": anns if anns is not None else base_anns, "instance": inst, "category": cat,
        "attribute": [{"token": "t1", "name": "vehicle.moving"}], "visibility": [{"token": "4", "level": "v80-100"}],
        "sample": samples if samples is not None else [{"token": "s1", "dbench_double_annotated": True}, {"token": "s2", "dbench_double_annotated": False}],
    }
    for k, v in tables.items():
        (d / f"{k}.json").write_text(json.dumps(v))
    (tmp_path / "release_meta.json").write_text(json.dumps(meta or {"tiers_admitted": "auto_accept", "human": {"enabled": False}}))
    if double is not None:
        (tmp_path / "double_annotation.json").write_text(json.dumps(double))
    if note:
        (tmp_path / "DELIVERY_NOTE.md").write_text("# x")
    return str(tmp_path)


def test_clean_export_has_no_errors(tmp_path):
    rep = check_release(_export(tmp_path))
    assert rep.errors == []


def test_bad_size_quaternion_and_dangling(tmp_path):
    root = _export(tmp_path)
    a = json.load(open(os.path.join(root, V, "sample_annotation.json")))
    a[0]["size"] = [1.8, -1.0, 1.6]; a[0]["rotation"] = [1, 1, 0, 0]; a[1]["instance_token"] = "nope"
    rep = check_release(_export(tmp_path, anns=a))
    assert any("size" in e for e in rep.errors) and any("quaternion" in e for e in rep.errors)
    assert any("instance_token" in e for e in rep.errors)


def test_chain_and_count_errors(tmp_path):
    root = _export(tmp_path)
    a = json.load(open(os.path.join(root, V, "sample_annotation.json")))
    a[1]["prev"] = ""
    rep = check_release(_export(tmp_path, anns=a))
    assert any("prev/next" in e for e in rep.errors)
    inst = [{"token": "i1", "category_token": "c1", "nbr_annotations": 3, "first_annotation_token": "a1", "last_annotation_token": "a2"}]
    rep = check_release(_export(tmp_path, instances=inst))
    assert any("nbr_annotations" in e for e in rep.errors)


def test_tier_and_human_rules(tmp_path):
    root = _export(tmp_path)
    a = json.load(open(os.path.join(root, V, "sample_annotation.json")))
    a[0]["dhakascenes_tier"] = "flagged"
    a[1].update(dhakascenes_source="human_created", dhakascenes_tier=None)
    rep = check_release(_export(tmp_path, anns=a))
    assert any("tier" in e for e in rep.errors) and any("verified_by" in e for e in rep.errors)


def test_double_fraction_and_half_import(tmp_path):
    root = _export(tmp_path, double={"fraction": 0.5, "n_keyframes": 2, "n_selected": 1, "selected": [{"sample_token": "s1"}]})
    assert check_release(root).errors == []
    root = _export(tmp_path, samples=[{"token": "s1", "dbench_double_annotated": False}, {"token": "s2", "dbench_double_annotated": False}],
                   double={"fraction": 0.5, "n_keyframes": 2, "n_selected": 1, "selected": [{"sample_token": "s1"}]})
    assert any("double" in e.lower() for e in check_release(root).errors)


def test_warnings(tmp_path):
    root = _export(tmp_path, note=False)
    a = json.load(open(os.path.join(root, V, "sample_annotation.json")))
    a[0]["size"] = [4.5, 1.8, 1.6]; a[1]["attribute_tokens"] = []; a[1]["annotator_pass"] = "B"
    rep = check_release(_export(tmp_path, anns=a, note=False))
    joined = "\n".join(rep.warnings)
    assert "w > l" in joined and "without attribute" in joined and "pass B" in joined and "DELIVERY_NOTE" in joined
    assert "pedestrian" in joined     # zero-instance class
```

- [ ] **Step 2: Run to verify failure**, then **Step 3: implement `check_release.py`**

```python
#!/usr/bin/env python3
"""Release checklist (spec §8): errors exit 2, warnings exit 1, clean exit 0.

    python scripts/check_release.py export/day1_chunk_0000/boxes [--version v1.0-dhaka-fixed]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.schemas import HUMAN_SOURCES  # noqa: E402

STATIC = ("traffic_cone", "barrier", "construction_element", "animal")
INTERP_WARN_FRACTION = 0.20


@dataclass
class Report:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


def _load(d, name):
    with open(os.path.join(d, f"{name}.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def _detect_version(out_root):
    cands = [e for e in os.listdir(out_root) if os.path.isfile(os.path.join(out_root, e, "sample.json"))]
    if len(cands) != 1:
        raise SystemExit(f"cannot detect the version dir under {out_root}: {cands}")
    return cands[0]


def check_release(out_root: str, version=None) -> Report:
    rep = Report()
    version = version or _detect_version(out_root)
    d = os.path.join(out_root, version)
    anns, inst, cats = _load(d, "sample_annotation"), _load(d, "instance"), _load(d, "category")
    attrs, samples = _load(d, "attribute"), _load(d, "sample")
    meta_p = os.path.join(out_root, "release_meta.json")
    meta = json.load(open(meta_p)) if os.path.isfile(meta_p) else {}
    admitted = meta.get("tiers_admitted", "auto_accept")
    inst_by = {i["token"]: i for i in inst}
    cat_by = {c["token"]: c["name"] for c in cats}
    attr_by = {a["token"]: a["name"] for a in attrs}
    sample_by = {s["token"]: s for s in samples}
    ann_by = {a["token"]: a for a in anns}
    by_inst = defaultdict(list)
    for a in anns:
        t = a["token"]
        if any(float(v) <= 0 for v in a["size"]):
            rep.errors.append(f"{t}: non-positive size {a['size']}")
        elif float(a["size"][0]) > float(a["size"][1]):
            rep.warnings.append(f"{t}: w > l size {a['size']}")
        n = math.sqrt(sum(float(v) ** 2 for v in a["rotation"]))
        if abs(n - 1.0) > 1e-3:
            rep.errors.append(f"{t}: quaternion norm {n:.4f}")
        if a["instance_token"] not in inst_by:
            rep.errors.append(f"{t}: dangling instance_token {a['instance_token']}")
        if a["sample_token"] not in sample_by:
            rep.errors.append(f"{t}: dangling sample_token {a['sample_token']}")
        for at in a.get("attribute_tokens", []):
            if at not in attr_by:
                rep.errors.append(f"{t}: dangling attribute token {at}")
        for k in ("prev", "next"):
            other = a.get(k)
            if other:
                o = ann_by.get(other)
                back = "next" if k == "prev" else "prev"
                if o is None or o.get(back) != t or o["instance_token"] != a["instance_token"]:
                    rep.errors.append(f"{t}: prev/next asymmetry on {k}={other}")
        src = a.get("dhakascenes_source", "pipeline")
        if src in HUMAN_SOURCES:
            if not a.get("dhakascenes_verified_by"):
                rep.errors.append(f"{t}: human row without verified_by")
        elif admitted != "all" and a.get("dhakascenes_tier") != "auto_accept":
            rep.errors.append(f"{t}: pipeline tier {a.get('dhakascenes_tier')!r} inside sample_annotation")
        if a.get("annotator_pass") == "B":
            rep.warnings.append(f"{t}: pass B row inline (a harness without a pass filter double-counts this frame)")
        by_inst[a["instance_token"]].append(a)
    for i in inst:
        if i["category_token"] not in cat_by:
            rep.errors.append(f"instance {i['token']}: dangling category_token")
        rows = sorted(by_inst.get(i["token"], []), key=lambda a: sample_by.get(a["sample_token"], {}).get("timestamp", 0))
        if i["nbr_annotations"] != len(rows):
            rep.errors.append(f"instance {i['token']}: nbr_annotations {i['nbr_annotations']} != {len(rows)} rows")
        if rows:
            heads = [a for a in rows if not a.get("prev")]
            tails = [a for a in rows if not a.get("next")]
            if len(heads) != 1 or len(tails) != 1 or heads[0]["token"] != i["first_annotation_token"] or tails[0]["token"] != i["last_annotation_token"]:
                rep.errors.append(f"instance {i['token']}: first/last/chain inconsistent")
            cls = cat_by.get(i["category_token"])
            if cls not in STATIC and len(rows) >= 2:
                for a in rows:
                    if not a.get("attribute_tokens"):
                        rep.warnings.append(f"{a['token']}: non-static multi-annotation box without attribute")
    present = Counter(cat_by.get(i["category_token"]) for i in inst)
    for name in sorted(cat_by.values()):
        if present[name] == 0:
            rep.warnings.append(f"class {name}: zero instances")
    n_interp = sum(1 for a in anns if a.get("dhakascenes_interpolated"))
    if anns and n_interp / len(anns) > INTERP_WARN_FRACTION:
        rep.warnings.append(f"interpolated rows are {n_interp / len(anns):.0%} of sample_annotation")
    dpath = os.path.join(out_root, "double_annotation.json")
    if os.path.isfile(dpath):
        doc = json.load(open(dpath))
        flagged = {s["token"] for s in samples if s.get("dbench_double_annotated")}
        need = math.ceil(float(doc.get("fraction", 0)) * len(samples))
        if len(flagged) < need:
            rep.errors.append(f"double annotation: {len(flagged)} flagged samples < required {need}")
        if set(s["sample_token"] for s in doc.get("selected", [])) != flagged:
            rep.errors.append("double annotation: sample.json flags disagree with double_annotation.json")
        if (meta.get("human") or {}).get("stats", {}).get("n_rows_double_A", 0) > 0:
            passes = defaultdict(set)
            for a in anns:
                if a.get("annotator_pass"):
                    passes[a["sample_token"]].add(a["annotator_pass"])
            for s in sorted(flagged):
                if passes.get(s) and passes[s] != {"A", "B"}:
                    rep.errors.append(f"double annotation: sample {s} has rows from pass {sorted(passes[s])} only")
    if not os.path.isfile(os.path.join(out_root, "DELIVERY_NOTE.md")):
        rep.warnings.append("DELIVERY_NOTE.md missing")
    rep.info = {"n_annotations": len(anns), "n_instances": len(inst), "n_interpolated": n_interp, "present": dict(present)}
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_root")
    ap.add_argument("--version", default=None)
    args = ap.parse_args(argv)
    rep = check_release(args.out_root, args.version)
    for e in rep.errors:
        print(f"ERROR   {e}")
    for w in rep.warnings:
        print(f"WARNING {w}")
    print(f"check_release: {len(rep.errors)} error(s), {len(rep.warnings)} warning(s); {rep.info}")
    return 2 if rep.errors else (1 if rep.warnings else 0)


if __name__ == "__main__":
    raise SystemExit(main())
```

In `export_release.main()`: after the note, run `check_release(out, version)`, print its lines, and return `max(0, rc_check)` (an export whose checker errors exits 2).

- [ ] **Step 4: Run tests + suite, commit**

```bash
git add scripts/check_release.py scripts/export_release.py tests/test_check_release.py
git commit -m "check_release: the benchmark checklist as a validator (spec §8)"
```

---

### Task 13: CVAT track round-trip spike (live server)

**Files:**
- Create: `scripts/spike_cvat_3d_roundtrip.py` (throwaway; kept for the record)
- Output: `docs/evidence/2026-09-08-cvat-3d-roundtrip.md`

**Interfaces:** none produced; the finding fixes Task 14/16's encoding. Needs `.env` (`CVAT_HOST`, `CVAT_USER`, `CVAT_PASSWORD`) — source it (`set -a; . ./.env; set +a`).

- [ ] **Step 1: Build one tiny 3D task locally**

Write `scripts/spike_cvat_3d_roundtrip.py` that: (a) creates a 3-frame `task.zip` in a temp dir with three 200-point random PCDs (use `export_cvat_3d.write_pcd`) and no related images; (b) writes a Datumaro 3D document with two labels (`a car`, `a pedestrian`), each label declaring attributes `record_token` (text), `track_id` (number), `attribute` (select: `vehicle.moving,vehicle.stopped,vehicle.parked,pedestrian.moving,pedestrian.standing,pedestrian.sitting_lying_down,cycle.with_rider,cycle.without_rider,`), `uncertain` (checkbox), `uncertain_reason` (text), and cuboids: one car present on frames 0–2 with attributes `{"track_id": 7, "record_token": "k0:CAM_FRONT:0", "keyframe": true}`, one pedestrian on frame 1 only with `{"record_token": "k1:CAM_FRONT:3"}`, positions/yaw/scale distinct per cuboid; (c) creates a project `SPIKE — 3D roundtrip (delete me)` with `cvat_setup_3d.label_spec`-style labels **plus** the attribute specs as CVAT expects (`{"name": ..., "input_type": "text"|"number"|"select"|"checkbox", "mutable": True, "values": [...]}`), a task from the zip, imports the JSON as "Datumaro 3D 1.0"; (d) exports the task back with `task.export_dataset("Datumaro 3D 1.0", filename, include_images=False)`, unzips, and prints the `annotations` of every item with their `attributes`; (e) compares position/rotation/scale of each cuboid to the input to 1e-4 and reports which attributes survived; (f) deletes the spike project only when `--cleanup` is passed (the operator's rule: nothing on CVAT is deleted without an explicit ask — this project is the spike's own, created seconds earlier; pass `--cleanup` in the same run so nothing is left behind).

- [ ] **Step 2: Run it**

Run: `set -a; . ./.env; set +a; $PY scripts/spike_cvat_3d_roundtrip.py --cleanup`
Record in `docs/evidence/2026-09-08-cvat-3d-roundtrip.md`: CVAT server version (`client.api_client.server_version` or the `/api/server/about` JSON), whether `track_id` came back on the car's three cuboids (tracks round-trip), whether `record_token` / `attribute` / `uncertain` / `uncertain_reason` survived, whether `rotation` and `scale` came back byte-equal or re-ordered, and any label-attribute declaration CVAT rejected.

- [ ] **Step 3: Decide and write the decision into the evidence file**

- Tracks round-trip → Task 14 emits cuboids with `track_id` (tracks) and Task 16 reads identity from `track_id`.
- Tracks do not round-trip → Task 14 still writes `track_id` as a plain attribute (identity survives as data), Task 16 reads it the same way; the spec's fallback applies. Either way Task 16's parser is identical; only the evidence differs.
- If `scale` came back re-ordered, Task 16's inverse follows the observed order, and the evidence file says so.

- [ ] **Step 4: Commit**

```bash
git add scripts/spike_cvat_3d_roundtrip.py docs/evidence/2026-09-08-cvat-3d-roundtrip.md
git commit -m "Spike: CVAT 2.7x Datumaro 3D round trip — tracks and attributes (spec §7)"
```

---

### Task 14: `scripts/export_cvat_3d.py` — `frames.json`, stitch map, blank double set, cuboid attributes

**Files:**
- Modify: `scripts/export_cvat_3d.py` (`cuboid()` line 88, `datumaro_document()` line 105, `main()` line 124)
- Test: `tests/test_export_cvat_3d_frames.py`

**Interfaces:**
- Produces:
  ```python
  CUBOID_ATTRIBUTES = ("record_token", "track_id", "attribute", "uncertain", "uncertain_reason")   # names declared on every label
  def cuboid(index, label_id, center_xyz, yaw_rad, extent_lwh, *, record_token=None, track_id=None) -> dict
      # attributes: {"occluded": False, "record_token": str|"" , "track_id": int|None (omitted when None), "uncertain": False, "uncertain_reason": "", "attribute": ""}
  def datumaro_document(labels, items) -> dict     # each label declares CUBOID_ATTRIBUTES in categories.label.labels[i].attributes
  def frames_manifest(keyframes: list[dict], start_index: int = 0) -> list[dict]
      # [{"frame": i, "name": f"{i+1:06d}", "sample_token": kf["keyframe_token"], "channels": sorted(kf["cameras"])}]
  def load_stitch_map(path) -> dict[str, str]
  ```
  CLI: `--stitch-map PATH` (record token → chain id; when given, `track_id` on a cuboid is the chain id, else Stage 7's `row["track_id"]`; chain ids that are not integers are hashed to a stable positive int: `int(hashlib.md5(cid.encode()).hexdigest()[:8], 16)` and the mapping `chain_id ↔ int` written to `<scene_dir>/track_ids.json`), `--frames DOUBLE_JSON --blank --out-subdir cvat_export_3d_double` (only the selected keyframes of each scene; every item has zero annotations; `annotations_blank.json` written instead of `annotations_ours.json`; `frames.json` still maps every packed frame to its sample token).
  Every run writes `<scene_dir>/frames.json`.

- [ ] **Step 1: Write the failing test** (pure functions only — no CVAT, no clouds)

```python
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.export_cvat_3d import (  # noqa: E402
    CUBOID_ATTRIBUTES, cuboid, datumaro_document, frames_manifest, load_stitch_map, stable_track_int,
)


def test_cuboid_carries_declared_attributes():
    c = cuboid(3, 1, [1, 2, 3], 0.5, (4.5, 1.8, 1.6), record_token="k0:CAM_FRONT:0", track_id=7)
    assert c["attributes"]["record_token"] == "k0:CAM_FRONT:0" and c["attributes"]["track_id"] == 7
    assert c["attributes"]["uncertain"] is False and c["attributes"]["attribute"] == ""
    assert c["scale"] == [4.5, 1.8, 1.6] and c["rotation"] == [0.0, 0.0, 0.5]
    c2 = cuboid(4, 1, [0, 0, 0], 0.0, (1, 1, 1))
    assert "track_id" not in c2["attributes"] and c2["attributes"]["record_token"] == ""


def test_document_declares_attributes_on_every_label():
    doc = datumaro_document(["a car", "a pedestrian"], [])
    for lab in doc["categories"]["label"]["labels"]:
        assert lab["attributes"] == list(CUBOID_ATTRIBUTES)


def test_frames_manifest_and_stitch_map(tmp_path):
    kfs = [{"keyframe_token": "s1", "cameras": {"CAM_FRONT": {}, "CAM_BACK": {}}}, {"keyframe_token": "s2", "cameras": {"CAM_FRONT": {}}}]
    fm = frames_manifest(kfs)
    assert fm == [{"frame": 0, "name": "000001", "sample_token": "s1", "channels": ["CAM_BACK", "CAM_FRONT"]},
                  {"frame": 1, "name": "000002", "sample_token": "s2", "channels": ["CAM_FRONT"]}]
    p = tmp_path / "stitch_map.json"
    p.write_text(json.dumps({"k0:CAM_FRONT:0": "7", "k1:CAM_FRONT:2": "det:k1:CAM_FRONT:2"}))
    m = load_stitch_map(str(p))
    assert stable_track_int(m["k0:CAM_FRONT:0"]) == 7
    assert stable_track_int(m["k1:CAM_FRONT:2"]) > 0 and stable_track_int(m["k1:CAM_FRONT:2"]) == stable_track_int("det:k1:CAM_FRONT:2")
```

- [ ] **Step 2: Run to verify failure**, then **Step 3: implement**

Add near the top:

```python
import hashlib
CUBOID_ATTRIBUTES = ("record_token", "track_id", "attribute", "uncertain", "uncertain_reason")


def stable_track_int(chain_id: str) -> int:
    s = str(chain_id)
    if s.isdigit():
        return int(s)
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) | 1


def load_stitch_map(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return {str(k): str(v) for k, v in json.load(fh).items()}


def frames_manifest(keyframes: list, start_index: int = 0) -> list:
    return [{"frame": start_index + i, "name": f"{start_index + i + 1:06d}",
             "sample_token": kf["keyframe_token"], "channels": sorted(kf["cameras"])} for i, kf in enumerate(keyframes)]
```

`cuboid()` gains keyword-only `record_token=None, track_id=None` and builds
`attributes = {"occluded": False, "record_token": record_token or "", "attribute": "", "uncertain": False, "uncertain_reason": ""}` plus `"track_id": int(track_id)` when not None. `datumaro_document()` writes `"attributes": list(CUBOID_ATTRIBUTES)` per label.

`main()`: add `--stitch-map`, `--frames`, `--blank`, `--out-subdir` (default `cvat_export_3d`); `out_root = os.path.join(paths.work_root, args.out_subdir)`. When `--frames` is given, load the JSON, take `selected = {s["sample_token"]}` and filter `keyframes` to those (keep scene order; skip scenes with none). Build cuboids with `record_token=f"{row['keyframe_token']}:{row['channel']}:{row['proposal_index']}"` (the Stage 9 token format) and `track_id = stable_track_int(stitch_map.get(record_token, row.get("track_id")))` when either exists. With `--blank`, skip the cuboid loop and write `annotations_blank.json`. Always write `frames.json` (`frames_manifest(keyframes)`) and, when a stitch map was used, `track_ids.json` (`{int: chain_id}`) into `scene_dir`. Print the "publish with" hint with the matching `--which`.

- [ ] **Step 4: Run tests + suite, commit**

```bash
git add scripts/export_cvat_3d.py tests/test_export_cvat_3d_frames.py
git commit -m "export_cvat_3d: frames.json, cuboid attributes, stitch-map track ids, blank double set (spec §7)"
```

---

### Task 15: `scripts/cvat_setup_3d.py` — label attributes, `--which double`, ledger

**Files:**
- Modify: `scripts/cvat_setup_3d.py` (`label_spec()` line 66, `main()` line 74)
- Test: `tests/test_cvat_setup_3d_ledger.py`

**Interfaces:**
- Produces:
  ```python
  ATTRIBUTE_VALUES = ["", "vehicle.moving", "vehicle.stopped", "vehicle.parked", "pedestrian.moving", "pedestrian.standing",
                      "pedestrian.sitting_lying_down", "cycle.with_rider", "cycle.without_rider"]
  def cuboid_attribute_specs() -> list[dict]
      # [{"name": "record_token", "input_type": "text", "mutable": False, "default_value": "", "values": [""]},
      #  {"name": "track_id", "input_type": "number", "mutable": True, "default_value": "0", "values": ["0", "999999999", "1"]},
      #  {"name": "attribute", "input_type": "select", "mutable": True, "default_value": "", "values": ATTRIBUTE_VALUES},
      #  {"name": "uncertain", "input_type": "checkbox", "mutable": True, "default_value": "false", "values": ["false", "true"]},
      #  {"name": "uncertain_reason", "input_type": "text", "mutable": True, "default_value": "", "values": [""]}]
  def label_spec(names, color=None) -> list[dict]      # now includes "attributes": cuboid_attribute_specs()
  DOUBLE_SUFFIX_A = "3D — double pass A"; DOUBLE_SUFFIX_B = "3D — double pass B"
  def double_project_names(base: str) -> tuple[str, str]     # (f"{base} — double pass A", f"{base} — double pass B")
  def ledger_append(path, entry: dict) -> None       # entry: {"project": name, "project_id": int, "task_id": int, "kind": "review"|"double_A"|"double_B", "scene": str, "frames_json": path, "assignee": str|None, "created_utc": str, "run_tag": str}
  def ledger_load(path) -> list[dict]
  ```
  CLI: `--which {both,ours,gt,double}`; `--assignee-a`, `--assignee-b` (CVAT usernames; the task is assigned via `task.update({"assignee_id": <id>})` after lookup in `client.users.list()`); `--ledger` (default `<work_root>/stage10_human/cvat_tasks.json`). `double` reads `<work_root>/cvat_export_3d_double/<scene>/{task.zip, annotations_blank.json, frames.json}`, creates the two projects (both from `label_spec(labels)`), one task per scene per project named `task_name(scene, DOUBLE_SUFFIX_A|B, run_tag)`, imports `annotations_blank.json` (so labels/attributes bind), records each in the ledger. `ours` also records its tasks in the ledger with kind `review`. Existing tasks are skipped (never replaced) unless `--replace` — `double` ignores `--replace-all-runs` entirely: A/B tasks are human work and are never wiped by this script.

- [ ] **Step 1: Write the failing test** (pure helpers)

```python
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.cvat_setup_3d import (  # noqa: E402
    ATTRIBUTE_VALUES, cuboid_attribute_specs, double_project_names, label_spec, ledger_append, ledger_load,
)


def test_label_spec_declares_the_five_attributes():
    spec = label_spec(["a car"])
    names = [a["name"] for a in spec[0]["attributes"]]
    assert names == ["record_token", "track_id", "attribute", "uncertain", "uncertain_reason"]
    assert spec[0]["type"] == "cuboid"
    sel = next(a for a in spec[0]["attributes"] if a["name"] == "attribute")
    assert sel["input_type"] == "select" and sel["values"] == ATTRIBUTE_VALUES and sel["values"][0] == ""


def test_double_project_names():
    assert double_project_names("day1_chunk_0000 (3D)") == ("day1_chunk_0000 (3D) — double pass A", "day1_chunk_0000 (3D) — double pass B")


def test_ledger_appends_and_loads(tmp_path):
    p = tmp_path / "stage10_human" / "cvat_tasks.json"
    ledger_append(str(p), {"project": "x", "project_id": 1, "task_id": 10, "kind": "double_A", "scene": "chunk_0", "frames_json": "/f.json", "assignee": None, "created_utc": "t", "run_tag": ""})
    ledger_append(str(p), {"project": "y", "project_id": 2, "task_id": 11, "kind": "double_B", "scene": "chunk_0", "frames_json": "/f.json", "assignee": "b", "created_utc": "t", "run_tag": ""})
    rows = ledger_load(str(p))
    assert [r["task_id"] for r in rows] == [10, 11] and rows[1]["assignee"] == "b"
```

- [ ] **Step 2: Run to verify failure**, then **Step 3: implement** — helpers as specified; in `main()` extend `variants` with, for `--which double`, two entries `("blank", name_a, DOUBLE_SUFFIX_A, None, args.replace, False, args.run_tag, "double_A", args.assignee_a)` and the B twin, and read `export_root = <work_root>/cvat_export_3d_double` for those; after every successful `create_from_data` + import, `ledger_append(args.ledger, {...})` with `frames_json = os.path.join(export_root, scene, "frames.json")` and `assignee` resolved by username (warn and leave unassigned if the user does not exist). For `ours`, kind is `review`. Refuse `--which double` when the double export dir has no scene (`run scripts/export_cvat_3d.py --frames ... --blank first`).

- [ ] **Step 4: Run tests + suite, commit**

```bash
git add scripts/cvat_setup_3d.py tests/test_cvat_setup_3d_ledger.py
git commit -m "cvat_setup_3d: label attributes, double pass A/B projects, task ledger (spec §7)"
```

---

### Task 16: `scripts/import_cvat_3d.py` — CVAT → I-5 `verified.jsonl`

**Files:**
- Create: `scripts/import_cvat_3d.py`
- Test: `tests/test_import_cvat_3d.py`

**Interfaces:**
- Consumes: the ledger (Task 15), `frames.json` + `task.zip` (Task 14), the Datumaro 3D 1.0 export CVAT produces (a zip holding `annotations/default.json` — confirm the inner path from Task 13's evidence and read whichever `*.json` under `annotations/` exists), `configs/release_category_map.yaml`, `pipeline.common.schemas` (Task 7).
- Produces:
  ```python
  @dataclass class ImportedBox: sample_token: str; frame: int; label: str; translation_m: list; size_wlh_m: list; rotation_wxyz: list;
                                record_token: str | None; track_id: int | None; attribute: str | None; uncertain: bool; uncertain_reason: str
  def parse_datumaro_3d(doc: dict, frames: list[dict]) -> list[ImportedBox]
      # inverse of export_cvat_3d.cuboid(): position -> translation_m; rotation[2] -> yaw -> quaternion_from_yaw_rad;
      # scale (l, w, h) -> size_wlh_m [w, l, h]; item "attr.frame" (or the item id "NNNNNN") -> frames[i]["sample_token"]
  def records_from_boxes(boxes, *, scene_token, kind, verified_by, timestamps_ns: dict[str, int], point_counter, mapper_classes: set) -> list[AnnotationRecord]
      # kind: "review" | "double_A" | "double_B"; source = human_verified if record_token else human_created;
      # instance_token = f"human:{kind}:{track_id}" if track_id is not None else f"human:{kind}:{record_token or f'{sample_token}:{frame}:{i}'}";
      # token = f"{sample_token}:HUMAN_{kind}:{i}"; provenance gates = GateVector(conf=1.0, lidar_pts_ok=True, spatial_ok=True, spatial_ok_source="human");
      # verification_pass=1; annotator_pass = "A"/"B" for double kinds; num_lidar_pts = point_counter(sample_token, translation, size, rotation)
  def write_scene(human_dir, scene_name, records, coverage: dict[str, list[str]]) -> tuple[str, str]   # verified.jsonl (write_records, HUMAN policy), coverage.json (spec "dhakascenes/human_coverage/v1")
  def main(argv) -> int
  ```
  CLI: `--paths` (work_root → default ledger `<work_root>/stage10_human/cvat_tasks.json` and output dir `<work_root>/stage10_human`), `--ledger`, `--tasks 41 42` (subset), `--zip PATH --kind KIND --scene NAME --frames-json PATH --verified-by NAME` (offline, one task, no server), `--include-incomplete` (default: only jobs whose `state == "completed"`; a task with any other job is skipped with a message), `--host/--user/--password` as in `cvat_setup_3d`. Labels not in the mapper abort listing every offender. `import_manifest.json` written at the end: `{"spec": "dhakascenes/import_cvat_3d/v1", "created_utc", "tasks": [{"task_id", "kind", "scene", "assignee", "n_boxes", "n_frames_covered", "skipped_reason"}], "rows_by_source": {...}, "rows_by_scene": {...}}`.

- [ ] **Step 1: Write the failing test** (offline; the Datumaro document is built by hand from the exporter's own `cuboid()`)

```python
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.schemas import AnnotationRecord, ProvenancePolicy, read_records  # noqa: E402
from scripts.export_cvat_3d import cuboid, datumaro_document, item_skeleton  # noqa: E402
from scripts.import_cvat_3d import parse_datumaro_3d, records_from_boxes, write_scene  # noqa: E402

HUMAN = ProvenancePolicy(allow_human_provenance=True)
FRAMES = [{"frame": 0, "name": "000001", "sample_token": "s1", "channels": []},
          {"frame": 1, "name": "000002", "sample_token": "s2", "channels": []}]


def _doc():
    labels = ["a car", "a pedestrian"]
    i0, i1 = item_skeleton(0, []), item_skeleton(1, [])
    c = cuboid(1, 0, [10.0, 2.0, 0.5], 0.3, (4.5, 1.8, 1.6), record_token="k0:CAM_FRONT:0", track_id=7)
    c["attributes"].update(attribute="vehicle.parked", uncertain=True, uncertain_reason="could be a microbus")
    i0["annotations"].append(c)
    i1["annotations"].append(cuboid(2, 0, [11.0, 2.0, 0.5], 0.3, (4.5, 1.8, 1.6), track_id=7))
    i1["annotations"].append(cuboid(3, 1, [-3.0, 1.0, 0.9], 1.0, (0.7, 0.6, 1.7)))
    return datumaro_document(labels, [i0, i1])


def test_parse_inverts_the_exporter():
    boxes = parse_datumaro_3d(_doc(), FRAMES)
    assert [b.sample_token for b in boxes] == ["s1", "s2", "s2"]
    car = boxes[0]
    assert car.size_wlh_m == [1.8, 4.5, 1.6] and car.translation_m == [10.0, 2.0, 0.5]
    assert abs(2 * math.atan2(car.rotation_wxyz[3], car.rotation_wxyz[0]) - 0.3) < 1e-6
    assert car.record_token == "k0:CAM_FRONT:0" and car.track_id == 7 and car.attribute == "vehicle.parked"
    assert car.uncertain is True and car.uncertain_reason == "could be a microbus"
    ped = boxes[2]
    assert ped.record_token is None and ped.track_id is None and ped.attribute is None and ped.uncertain is False


def test_records_sources_identity_and_policy(tmp_path):
    boxes = parse_datumaro_3d(_doc(), FRAMES)
    recs = records_from_boxes(boxes, scene_token="sc", kind="review", verified_by="ann_a",
                              timestamps_ns={"s1": 1_700_000_000_000_000_000, "s2": 1_700_000_000_400_000_000},
                              point_counter=lambda tok, t, s, q: 12, mapper_classes={"a car", "a pedestrian"})
    assert [r.provenance.source for r in recs] == ["human_verified", "human_created", "human_created"]
    assert recs[0].instance_token == recs[1].instance_token == "human:review:7"
    assert recs[0].attribute == "vehicle.parked" and recs[0].is_uncertain is True
    assert recs[0].num_lidar_pts == 12 and recs[0].provenance.verification_pass == 1
    assert all(r.provenance.annotator_pass is None for r in recs)
    dbl = records_from_boxes(boxes, scene_token="sc", kind="double_B", verified_by="ann_b",
                             timestamps_ns={"s1": 1, "s2": 2}, point_counter=lambda *a: 0, mapper_classes={"a car", "a pedestrian"})
    assert all(r.provenance.annotator_pass == "B" for r in dbl)
    v, c = write_scene(str(tmp_path), "chunk_x", recs, {"review": ["s1", "s2"], "double_A": [], "double_B": []})
    back = read_records(v, policy=HUMAN, expect_type=AnnotationRecord)
    assert len(back) == 3 and json.load(open(c))["review"] == ["s1", "s2"]


def test_unmapped_label_aborts():
    boxes = parse_datumaro_3d(_doc(), FRAMES)
    try:
        records_from_boxes(boxes, scene_token="sc", kind="review", verified_by="x", timestamps_ns={"s1": 1, "s2": 2},
                           point_counter=lambda *a: 0, mapper_classes={"a car"})
    except ValueError as exc:
        assert "a pedestrian" in str(exc)
    else:
        raise AssertionError("expected ValueError for an unmapped label")
```

- [ ] **Step 2: Run to verify failure**, then **Step 3: implement** — parsing and record building as specified (yaw → `quaternion_from_yaw_rad`, `frame = EGO`, `time_base = "unix_ns"`, `coverage_config = "R2"` read from the ledger's scene if recorded else `"R2"`, `num_lidar_pts_basis = BASIS_GROUND_FILTERED` from `pipeline.release.frames`), `write_scene` via `write_records(path, records, policy=HUMAN_POLICY, expect_type=AnnotationRecord, allow_empty=True)`. Server path in `main()`: for each ledger task (filtered by `--tasks`), `task = client.tasks.retrieve(task_id)`; skip unless every `task.get_jobs()` has `state == "completed"` (or `--include-incomplete`); `task.export_dataset("Datumaro 3D 1.0", tmpzip, include_images=False)`; read the JSON inside; `frames = json.load(open(entry["frames_json"]))`; `verified_by = entry["assignee"] or task.owner.username`; point counter = `CloudSource`-style read of `task.zip` next to `frames.json` (reuse `pipeline.release.frames.read_pcd_v07_binary` + `geometry.points_in_box`); coverage for the task's kind = every `sample_token` in `frames.json`; scene timestamps from the dataroot `sample.json` (via `load_paths` → `SourceRoot`-like read of `<dataroot>/<version>/sample.json`). Records of all tasks of one scene are concatenated into one `verified.jsonl`; coverage lists are unioned per kind.

- [ ] **Step 4: Run tests + suite, commit**

```bash
git add scripts/import_cvat_3d.py tests/test_import_cvat_3d.py
git commit -m "import_cvat_3d: CVAT Datumaro 3D -> I-5 verified.jsonl + coverage (spec §7)"
```

---

### Task 17: `scripts/run_day1_chunks.sh` — reorder the tail, double publish, `--phase human-import`

**Files:**
- Modify: `scripts/run_day1_chunks.sh`

**Interfaces:** shell only. `CHAIN_STEPS` default loses `cvat3d` (the 3D publish now runs after the release export); a new function `publish_3d` and a new `--phase` argument.

- [ ] **Step 1: Edit the defaults and the per-chunk flow**

- `CHAIN_STEPS` default becomes `"0 1 3 3f 3m 4 5 6 7 8 road eval viz cvat cvatroad"` (comment: `cvat3d` moved after the release export so the review task shows stitched identities and the double set exists).
- In `run_chunk`, step 3 becomes:

```bash
  say "  [$id] 3/5 export_release --blobs copy -> $out/boxes"
  mkdir -p "$out"
  "$PY" scripts/export_release.py \
    --prelabels "$work/stage9_qa" --dataroot "$chunk" --version "$FIXED_VERSION" \
    --out "$out/boxes" --mapper configs/release_category_map.yaml --blobs copy \
    --cvat-export-3d-dir "$work/cvat_export_3d" --stage-tree "$work" --chunk-name "$name"
  local rel_rc=$?
  echo "export_release rc=$rel_rc  (0 clean, 1 checker warnings, 2 refused/errors)"
```

  and a new step 4 (before the extras, which become 5/5):

```bash
  # 4. 3D CVAT: the pre-filled review task (stitched identities from stitch_map.json)
  #    and the two blank double-annotation tasks (frames from double_annotation.json).
  if [ $rel_rc -le 1 ] && [ -n "${CVAT_PASSWORD:-}" ]; then
    say "  [$id] 4/5 cvat3d: review (pre-filled) + double pass A/B (blank)"
    local tag; tag=$(date +%Y%m%dT%H%M%S -r "$work/stage8_inflate/run_manifest.json")
    "$PY" scripts/export_cvat_3d.py --taxonomy configs/taxonomy_pilot_dhaka.yaml \
      --stitch-map "$out/boxes/stitch_map.json" \
    && "$PY" -m scripts.cvat_setup_3d --which ours --run-tag "$tag" \
    && "$PY" scripts/export_cvat_3d.py --taxonomy configs/taxonomy_pilot_dhaka.yaml \
      --frames "$out/boxes/double_annotation.json" --blank --out-subdir cvat_export_3d_double --skip-archive-if-present \
    && "$PY" -m scripts.cvat_setup_3d --which double --run-tag "$tag" \
      ${CVAT_ASSIGNEE_A:+--assignee-a "$CVAT_ASSIGNEE_A"} ${CVAT_ASSIGNEE_B:+--assignee-b "$CVAT_ASSIGNEE_B"}
    echo "cvat3d rc=$?"
  else
    echo "cvat3d  : skipped (export rc=$rel_rc or no CVAT_PASSWORD)"
  fi
```

  (`--skip-archive-if-present` is `--skip-archive` when the double `task.zip` already exists; implement as a tiny wrapper in the script: test for the zip and choose the flag.) The existing `--taxonomy "$(export_taxonomy)"` from `run_stages.sh` resolves to the dhaka taxonomy in this chain; use the explicit file here.
- Replace the README heredoc with:

```bash
  cat > "$out/README.md" <<EOF
# $name — machine pre-annotations

See boxes/DELIVERY_NOTE.md for the delivery note the benchmark requires (rule, range,
tiers, classes, identity, attributes, double annotation, anonymisation). Layout:
- boxes/    nuScenes $FIXED_VERSION release (13 tables + samples/ as real files) plus
            sample_annotation_excluded.json, stitch_map.json, double_annotation.json, release_meta.json.
- road/     EXTRA: driveable-surface lidarseg layer. Absent if the road stage wrote no marker.
- coco_2d/  EXTRA: 2D-only COCO layer with the detector's phrase names.
Stage tree: $work
EOF
```

- Add the phase switch at the top of the driver: `PHASE=${PHASE:-run}`; `scripts/run_day1_chunks.sh --phase human-import 0000` sets `PHASE=human-import` and, per chunk, runs:

```bash
human_import_chunk() {  # chunk id -> rc
  local id=$1 name=${DAY}_chunk_$1 work=$ROOT_WORK/chunk_$1 chunk=$DATASET/chunk_$1
  local out=$EXPORT/${DAY}_chunk_$1${EXPORT_SUFFIX:-} cfg; cfg=$(write_paths_config "$id")
  export DHAKASCENES_PATHS_CONFIG=$cfg
  say "HUMAN IMPORT $id"
  "$PY" scripts/import_cvat_3d.py --paths "$cfg" || { echo "!!! [$id] import failed"; return 2; }
  "$PY" scripts/export_release.py \
    --prelabels "$work/stage9_qa" --dataroot "$chunk" --version "$FIXED_VERSION" \
    --out "$out/boxes" --mapper configs/release_category_map.yaml --blobs copy \
    --cvat-export-3d-dir "$work/cvat_export_3d" --stage-tree "$work" --chunk-name "$name" \
    --human "$work/stage10_human" --overwrite-tables
  local rc=$?
  echo "re-export rc=$rc"
  return $rc
}
```

  Parse `--phase X` out of `"$@"` before `CHUNKS=("$@")`; in the driver loop call `human_import_chunk` instead of `run_chunk` when `PHASE=human-import`.

- [ ] **Step 2: Syntax-check and dry-run**

Run: `bash -n scripts/run_day1_chunks.sh && DRY_RUN=1 scripts/run_day1_chunks.sh 0006 | tail -20`
Expected: no syntax error; the dry run prints the plan without `cvat3d` in the chain steps.

- [ ] **Step 3: Commit**

```bash
git add scripts/run_day1_chunks.sh
git commit -m "run_day1_chunks: release export before 3D publish, double A/B publish, --phase human-import (spec §9)"
```

---

### Task 18: Real-data dry run on chunk_0000's existing pre-labels

**Files:** none committed except `docs/evidence/2026-09-08-release-dry-run-chunk_0000.md`.

- [ ] **Step 1: Export to a scratch root with symlinked blobs**

```bash
$PY scripts/export_release.py --prelabels /home/mt/dhakascenes/work_day1/chunk_0000/stage9_qa \
  --dataroot /home/mt/Zami/Annotation_pipeline/Dataset/A_nusc/chunk_0000 --version v1.0-dhaka-fixed \
  --out /home/mt/dhakascenes/work_scratch/release_dryrun_chunk_0000/boxes --blobs symlink \
  --cvat-export-3d-dir /home/mt/dhakascenes/work_day1/chunk_0000/cvat_export_3d \
  --stage-tree /home/mt/dhakascenes/work_day1/chunk_0000 --chunk-name day1_chunk_0000_dryrun
```

- [ ] **Step 2: Record** in the evidence file: exit code; `release_meta.stitch.totals` (fragments → chains, joins by gap, interpolated, raw-basis count); median chain length before → after; `excluded.by_reason`; attribute counts by name; `range.effective_p99_m_by_class`; the double-annotation cell table; the checker's error/warning lines; wall time. Compare against the pre-change numbers in the spec's "Substrate facts" (11 882 tracks, median 1). If `n_interpolated_raw_basis > 0`, note that the task.zip for this chunk predates `frames.json` (expected — it was exported before Task 14) and that the rerun will fix it.
- [ ] **Step 3: Sanity-check one stitched chain visually** — pick the longest chain from `stitch_map.json`, print its rows (sample index, ego x/y, interpolated flag) with a 15-line Python snippet, and confirm the positions are monotone/smooth. Paste the table into the evidence file.
- [ ] **Step 4: Commit the evidence**

```bash
git add docs/evidence/2026-09-08-release-dry-run-chunk_0000.md
git commit -m "Evidence: release post-processor dry run on chunk_0000 pre-labels"
```

---

### Task 19: Launch the 50 m rerun of every chunk

**Files:** none. Operational.

- [ ] **Step 1: Check nothing is running** — `tmux list-windows -a`, `nvidia-smi --query-gpu=memory.used --format=csv`, `pgrep -af run_stages.sh`. If a chain is still running (window `day1b` from 2026-09-06), do **not** start; write the state into the handover and stop here.
- [ ] **Step 2: Launch** in a new tmux window (never `send-keys` into a pane with pending input; per memory, create the window fresh):

```bash
tmux new-window -d -t pipe -n day1c 'scripts/run_day1_chunks.sh 0006 0000 0001 0002 0003 0004 0005 2>&1 | tee -a /home/mt/dhakascenes/work_day1/logs/day1c_rerun_50m.log'
```

  (The chain reruns from Stage 0 on the existing work roots; `export/day1_chunk_NNNN/boxes/<version>` already exists for 0000/0001/0006 and the chain refuses to overwrite it — set `EXPORT_SUFFIX=_r50` so the reruns land beside the earlier exports: `EXPORT_SUFFIX=_r50 scripts/run_day1_chunks.sh ...`. Nothing already exported is deleted.)
- [ ] **Step 3: Write the handover** — `handover/2026-09-08-benchmark-release-handoff.md`: what changed, how to run the export alone, how to run `--phase human-import` after annotators finish, where the ledger is, which CVAT projects are the A/B ones, and the tmux window/log of the rerun. Commit it.

---

## Self-review

- **Spec coverage:** §1 → Task 0; §2 → Tasks 1, 2, 10; §3 → Tasks 3, 4; §4 → Task 5; §5 → Tasks 6, 10; §6 → Task 9 (+10); §7 → Tasks 7, 8, 13, 14, 15, 16 (+10 human merge); §8 → Tasks 11, 12; §9 → Tasks 17, 18, 19. Out-of-scope items untouched.
- **Type consistency:** `stitch_chain_id` / `stitch_track_id_pre` / `stitch_interpolated` / `stitch_tier_basis` (Task 4) are the keys Tasks 8 and 10 read; `attr_name` / `attribute_basis` / `velocity_chain_mps` (Task 5) are what Task 10's `annotation_row` reads; `excluded_reason` (Task 6) becomes `dhakascenes_excluded_reason` in Task 10; `human_kind` / `annotator_pass` (Task 8) match Task 16's `kind` values `review|double_A|double_B`; `frames.json` shape (Task 14) is what Tasks 3 and 16 read; `coverage.json` (Task 16) is what Task 8 reads.
- **Placeholders:** the only deliberately unknown value is the benchmark file's sha in `configs/release.yaml`, filled at Task 1 step 3 from `sha256sum`.
