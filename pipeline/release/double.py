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
