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
