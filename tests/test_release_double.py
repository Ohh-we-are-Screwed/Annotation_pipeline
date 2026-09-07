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
