"""Stage 1 `_sector_rng` — the per-sector RANSAC seed must not move between processes.

The seed is derived from the keyframe token so a wedge's ground plane does not
depend on how many keyframes were processed before it. A token of 16+ hex
characters is read as hex; anything shorter used to fall back to the builtin
`hash()`, which CPython salts per process (PYTHONHASHSEED), so the same short
token seeded a different stream in every interpreter — the ground fit of a
short-token keyframe was irreproducible, and
`tests/test_stage1_ground_reference.py` flaked for roughly 1 in 6 hash seeds.
The fallback is a sha256 digest instead (fixed 2026-09-08).

Production nuScenes/day-1 tokens are 32 hex characters and take the hex branch,
which is unchanged — the second test pins that byte for byte.
"""

from __future__ import annotations

import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.stage1_ingestion.ingest import IngestConfig, _sector_rng  # noqa: E402

_SNIPPET = (
    "from pipeline.stage1_ingestion.ingest import IngestConfig, _sector_rng; "
    "print(int(_sector_rng(IngestConfig(), 'kf', 0).integers(1 << 30)))"
)


def test_short_token_seed_is_stable_across_processes():
    """A short token draws the same first value in this process and in fresh ones."""
    d0 = int(_sector_rng(IngestConfig(), "kf", 0).integers(1 << 30))
    for s in ("12", "37"):
        proc = subprocess.run(
            [sys.executable, "-c", _SNIPPET],
            env={**os.environ, "PYTHONHASHSEED": s, "PYTHONPATH": ROOT},
            capture_output=True,
            text=True,
            check=True,
        )
        assert int(proc.stdout.strip()) == d0, f"PYTHONHASHSEED={s} moved the seed"


def test_long_token_path_unchanged():
    """A real 32-hex token still seeds on int(token[:16], 16) — byte for byte."""
    cfg = IngestConfig()
    token = "0123456789abcdef0123456789abcdef"
    expected = np.random.default_rng(
        [cfg.global_seed, int("0123456789abcdef", 16), 3]).integers(1 << 30)
    assert _sector_rng(cfg, token, 3).integers(1 << 30) == expected
