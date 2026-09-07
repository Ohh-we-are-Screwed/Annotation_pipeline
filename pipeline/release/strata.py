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
