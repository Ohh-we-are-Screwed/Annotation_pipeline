"""Tests for scripts/export_road_lidarseg.py.

stage_road is built elsewhere; these tests synthesize its work tree with the
PRODUCTION marker/atomic-write helpers (so the exporter's upstream gate is
exercised against real markers, not hand-rolled ones) plus a tiny fake
nuScenes dataroot, then drive the exporter's main() end to end. No GPU, no
network, no reading the real dataroot: every directory is passed explicitly.

Run: /home/mt/miniconda3/envs/ano_pipe/bin/python -m pytest tests/test_export_road_lidarseg.py -v
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pipeline.common.manifest import (  # noqa: E402
    write_json_atomic,
    write_jsonl_atomic,
    write_marker,
)
from scripts import export_road_lidarseg as erl  # noqa: E402

VERSION = "v1.0-dhaka"
RAW_BASIS = "raw_lidar_top_file_order"
FINGERPRINT = "f" * 64
SCENE = "scene-0001"
SCENE_TOKEN = "e" * 32
ROAD_INDEX = 24

# One scene, two keyframes: (keyframe_token, lidar sd token, n_points_raw, road indices).
KEYFRAMES = [
    ("kf000000" * 4, "sdL00000" * 4, 10, [1, 3, 4, 7]),
    ("kf111111" * 4, "sdL11111" * 4, 6, [0, 5]),
]
N_ROAD_TOTAL = sum(len(k[3]) for k in KEYFRAMES)


# ---------------------------------------------------------------------------
# Synthetic trees
# ---------------------------------------------------------------------------


def build_dataroot(root: str) -> str:
    """A tiny nuScenes-shaped dataroot: <version>/ tables + empty blob dirs.

    category.json carries the REAL field shape of a v1.0 row (token, name,
    description) so the exporter's "mirror the source shape, add index" path
    is what runs, not a fallback.
    """
    tdir = os.path.join(root, VERSION)
    os.makedirs(tdir)
    for sub in ("samples", "sweeps"):
        os.makedirs(os.path.join(root, sub, "LIDAR_TOP"))
    tables = {
        "category.json": [
            {"token": "c" * 32, "name": "vehicle.car",
             "description": "Vehicle designed primarily for personal use."},
        ],
        "sensor.json": [
            {"token": "s" * 32, "channel": "LIDAR_TOP", "modality": "lidar"},
        ],
        "scene.json": [
            {"token": SCENE_TOKEN, "name": SCENE, "description": ""},
        ],
    }
    for name, rows in tables.items():
        with open(os.path.join(tdir, name), "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
    return tdir


def build_stage_road(
    root: str,
    *,
    manifest: bool = True,
    marker: bool = True,
    degraded: bool = False,
    basis: str = RAW_BASIS,
    npz_token: str | None = None,
) -> str:
    """The stage_road work tree the (concurrently built) stage will produce."""
    os.makedirs(root, exist_ok=True)
    if manifest:
        write_json_atomic(
            os.path.join(root, "run_manifest.json"),
            {"spec": "dhakascenes-pilot/stage_road/v1", "n_keyframes": len(KEYFRAMES)},
        )
    points_dir = os.path.join(root, "scenes", SCENE, "points")
    os.makedirs(points_dir, exist_ok=True)
    rows = []
    for kf_tok, sd_tok, n_raw, road_idx in KEYFRAMES:
        rows.append({
            "keyframe_token": kf_tok,
            "scene_token": SCENE_TOKEN,
            "lidar_sample_data_token": sd_tok,
            "n_points_raw": n_raw,
        })
        seen = sorted(set(road_idx) | {0, n_raw - 1})
        np.savez(
            os.path.join(points_dir, f"{kf_tok}.npz"),
            road_point_index=np.asarray(road_idx, dtype=np.int32),
            n_cameras_road=np.ones(len(road_idx), dtype=np.int8),
            seen_point_index=np.asarray(seen, dtype=np.int32),
            __n_points_raw__=np.asarray([n_raw], dtype=np.int32),
            __lidar_sample_data_token__=npz_token if npz_token is not None else sd_tok,
            __frame__="ego",
            __basis__=basis,
        )
    write_jsonl_atomic(os.path.join(root, "scenes", SCENE, "road.jsonl"), rows)
    if marker:
        causes = (f"{SCENE}: seen fraction below threshold",) if degraded else ()
        write_marker(root, FINGERPRINT, degraded=degraded, causes=causes)
    return root


@pytest.fixture
def roots(tmp_path):
    dataroot = str(tmp_path / "dataroot")
    build_dataroot(dataroot)
    return {
        "dataroot": dataroot,
        "stage": str(tmp_path / "work" / "stage_road"),
        "out": str(tmp_path / "lidarseg_release"),
    }


def run_main(roots, *extra: str) -> int:
    return erl.main([
        "--stage-road-dir", roots["stage"],
        "--dataroot", roots["dataroot"],
        "--out", roots["out"],
        "--version", VERSION,
        *extra,
    ])


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_bins_category_and_index(self, roots, capsys):
        build_stage_road(roots["stage"])
        assert run_main(roots) == 0
        out = roots["out"]

        # Bins: byte-exact uint8 over the RAW point order, road index at the
        # listed positions and 0 (noise) everywhere else; no .tmp survivors.
        for _, sd_tok, n_raw, road_idx in KEYFRAMES:
            bin_path = os.path.join(out, "lidarseg", VERSION, f"{sd_tok}_lidarseg.bin")
            labels = np.fromfile(bin_path, dtype=np.uint8)
            expected = np.zeros(n_raw, dtype=np.uint8)
            expected[np.asarray(road_idx)] = ROAD_INDEX
            assert labels.shape == (n_raw,)
            assert np.array_equal(labels, expected)
            assert not os.path.exists(bin_path + ".tmp")

        # category.json: the FULL canonical 32-row table with CONTIGUOUS
        # indices — the devkit's stats/render APIs assume index == position,
        # so a sparse {0, 24} table would crash them. driveable_surface stays
        # at the official 24.
        with open(os.path.join(out, VERSION, "category.json"), encoding="utf-8") as fh:
            cats = json.load(fh)
        assert len(cats) == 32
        assert [c["index"] for c in cats] == list(range(32))
        assert cats[0]["name"] == "noise"
        assert cats[ROAD_INDEX]["name"] == "flat.driveable_surface"
        from nuscenes.utils.color_map import get_colormap
        assert [c["name"] for c in cats] == list(get_colormap().keys())
        for c in cats:
            assert {"token", "name", "description", "index"} <= set(c)
            assert isinstance(c["index"], int)
            assert len(c["token"]) == 32

        # lidarseg.json: one row per keyframe binding token -> bin file.
        with open(os.path.join(out, VERSION, "lidarseg.json"), encoding="utf-8") as fh:
            index_rows = json.load(fh)
        assert index_rows == [
            {
                "token": sd_tok,
                "sample_data_token": sd_tok,
                "filename": f"lidarseg/{VERSION}/{sd_tok}_lidarseg.bin",
            }
            for _, sd_tok, _, _ in KEYFRAMES
        ]

        # Every other table rides along byte-for-byte.
        for name in ("sensor.json", "scene.json"):
            with open(os.path.join(roots["dataroot"], VERSION, name), "rb") as fh:
                src_bytes = fh.read()
            with open(os.path.join(out, VERSION, name), "rb") as fh:
                assert fh.read() == src_bytes

        # Blobs are NOT linked by default.
        assert not os.path.lexists(os.path.join(out, "samples"))
        assert not os.path.lexists(os.path.join(out, "sweeps"))

        # One-line summary: n keyframes, n road points, out path.
        out_text = capsys.readouterr().out
        assert f"{len(KEYFRAMES)} keyframes" in out_text
        assert f"{N_ROAD_TOTAL} road points" in out_text
        assert os.path.realpath(out) in out_text

    def test_link_blobs_makes_relative_symlinks(self, roots):
        build_stage_road(roots["stage"])
        assert run_main(roots, "--link-blobs") == 0
        for sub in ("samples", "sweeps"):
            link = os.path.join(roots["out"], sub)
            assert os.path.islink(link)
            assert not os.path.isabs(os.readlink(link))
            assert os.path.realpath(link) == os.path.realpath(
                os.path.join(roots["dataroot"], sub)
            )

    def test_version_inferred_from_the_single_v1_dir(self, roots):
        build_stage_road(roots["stage"])
        rc = erl.main([
            "--stage-road-dir", roots["stage"],
            "--dataroot", roots["dataroot"],
            "--out", roots["out"],
        ])
        assert rc == 0
        assert os.path.isdir(os.path.join(roots["out"], VERSION))

    def test_degraded_marker_with_flag_exports(self, roots):
        # On this substrate the road marker WILL be degraded; the opt-in must
        # actually produce the export, not merely not-crash.
        build_stage_road(roots["stage"], degraded=True)
        assert run_main(roots, "--accept-degraded-upstream") == 0
        for _, sd_tok, n_raw, _ in KEYFRAMES:
            bin_path = os.path.join(roots["out"], "lidarseg", VERSION, f"{sd_tok}_lidarseg.bin")
            assert os.path.getsize(bin_path) == n_raw


# ---------------------------------------------------------------------------
# Refusals: exit 2, message to stderr, nothing written
# ---------------------------------------------------------------------------


class TestRefusals:
    def _assert_untouched(self, out: str) -> None:
        assert not os.path.exists(os.path.join(out, "lidarseg"))
        assert not os.path.exists(os.path.join(out, VERSION))

    def test_missing_marker(self, roots, capsys):
        build_stage_road(roots["stage"], marker=False)
        assert run_main(roots) == 2
        assert "marker" in capsys.readouterr().err
        assert not os.path.exists(roots["out"])

    def test_missing_manifest(self, roots, capsys):
        build_stage_road(roots["stage"], manifest=False)
        assert run_main(roots) == 2
        assert "run_manifest.json" in capsys.readouterr().err
        assert not os.path.exists(roots["out"])

    def test_degraded_marker_requires_the_flag(self, roots, capsys):
        build_stage_road(roots["stage"], degraded=True)
        assert run_main(roots) == 2
        assert "--accept-degraded-upstream" in capsys.readouterr().err
        assert not os.path.exists(roots["out"])

    def test_nonempty_out_version_dir(self, roots, capsys):
        build_stage_road(roots["stage"])
        vdir = os.path.join(roots["out"], VERSION)
        os.makedirs(vdir)
        with open(os.path.join(vdir, "already_here.json"), "w", encoding="utf-8") as fh:
            fh.write("{}")
        assert run_main(roots) == 2
        assert "not empty" in capsys.readouterr().err
        assert os.listdir(vdir) == ["already_here.json"]
        assert not os.path.exists(os.path.join(roots["out"], "lidarseg"))

    def test_out_is_the_dataroot(self, roots, capsys):
        build_stage_road(roots["stage"])
        rc = erl.main([
            "--stage-road-dir", roots["stage"],
            "--dataroot", roots["dataroot"],
            "--out", roots["dataroot"],
            "--version", VERSION,
        ])
        assert rc == 2
        assert "dataroot" in capsys.readouterr().err
        assert not os.path.exists(os.path.join(roots["dataroot"], "lidarseg"))
        assert sorted(os.listdir(os.path.join(roots["dataroot"], VERSION))) == [
            "category.json", "scene.json", "sensor.json",
        ]

    def test_basis_mismatch_names_the_basis(self, roots, capsys):
        # Labels are positional over the RAW blob; any other basis silently
        # shears every label, so the exporter must refuse before writing.
        build_stage_road(roots["stage"], basis="ego_ground_removed")
        assert run_main(roots) == 2
        assert RAW_BASIS in capsys.readouterr().err
        self._assert_untouched(roots["out"])

    def test_npz_token_mismatch(self, roots, capsys):
        build_stage_road(roots["stage"], npz_token="x" * 32)
        assert run_main(roots) == 2
        assert "__lidar_sample_data_token__" in capsys.readouterr().err
        self._assert_untouched(roots["out"])
