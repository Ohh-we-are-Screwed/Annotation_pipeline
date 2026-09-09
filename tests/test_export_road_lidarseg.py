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
LOG_TOKEN = "l" * 32
SENSOR_TOKEN = "s" * 32
CALIB_TOKEN = "c" * 32
T0_US = 1_700_000_000_000_000  # plausible microseconds since epoch
ROAD_INDEX = 24

# One scene, two keyframes: (keyframe_token, lidar sd token, n_points_raw, road indices).
KEYFRAMES = [
    ("kf000000" * 4, "sdL00000" * 4, 10, [1, 3, 4, 7]),
    ("kf111111" * 4, "sdL11111" * 4, 6, [0, 5]),
]
N_ROAD_TOTAL = sum(len(k[3]) for k in KEYFRAMES)
# A SECOND scene, always in the dataroot and optionally in stage_road, so
# --scenes has something to leave out of both the tables and the labels.
OTHER_SCENE = "scene-0002"
OTHER_SCENE_TOKEN = "d" * 32
OTHER_KEYFRAME = ("kf222222" * 4, "sdL22222" * 4, 4, [2])


# ---------------------------------------------------------------------------
# Synthetic trees
# ---------------------------------------------------------------------------


def build_dataroot(root: str) -> str:
    """A tiny nuScenes-shaped dataroot: all 13 <version>/ tables + blob dirs.

    Two scenes, and every table the exporter's SourceRoot demands — it now
    builds the eight it scopes rather than copying them, and refuses a root
    missing any. map.json is `[]`, as the real capture's is, so the synthetic
    map row is what the exporter has to write. category.json carries the REAL
    field shape of a v1.0 row (token, name, description) so the exporter's
    "mirror the source shape, add index" path is what runs, not a fallback.
    """
    tdir = os.path.join(root, VERSION)
    os.makedirs(tdir)
    for sub in ("samples", "sweeps"):
        os.makedirs(os.path.join(root, sub, "LIDAR_TOP"))
    samples, sample_data, ego_pose = [], [], []
    for scene_token, keyframes in ((SCENE_TOKEN, KEYFRAMES), (OTHER_SCENE_TOKEN, [OTHER_KEYFRAME])):
        for i, (kf_tok, sd_tok, _n_raw, _road_idx) in enumerate(keyframes):
            ts = T0_US + i * 500_000
            pose_tok = f"pose{len(ego_pose):028d}"
            ego_pose.append({"token": pose_tok, "timestamp": ts, "translation": [0.0, 0.0, 0.0],
                             "rotation": [1.0, 0.0, 0.0, 0.0]})
            samples.append({"token": kf_tok, "timestamp": ts, "scene_token": scene_token,
                            "prev": keyframes[i - 1][0] if i else "", "next": ""})
            if i:
                samples[-2]["next"] = kf_tok
            sample_data.append({"token": sd_tok, "sample_token": kf_tok, "ego_pose_token": pose_tok,
                                "calibrated_sensor_token": CALIB_TOKEN, "fileformat": "pcd",
                                "filename": f"samples/LIDAR_TOP/{kf_tok}.pcd.bin", "timestamp": ts,
                                "is_key_frame": True, "height": 0, "width": 0, "prev": "", "next": ""})
    tables = {
        "category.json": [
            {"token": "a" * 32, "name": "vehicle.car",
             "description": "Vehicle designed primarily for personal use."},
        ],
        "attribute.json": [], "visibility.json": [], "instance.json": [], "sample_annotation.json": [],
        "sensor.json": [
            {"token": SENSOR_TOKEN, "channel": "LIDAR_TOP", "modality": "lidar"},
        ],
        "calibrated_sensor.json": [
            {"token": CALIB_TOKEN, "sensor_token": SENSOR_TOKEN, "translation": [0.0, 0.0, 0.0],
             "rotation": [1.0, 0.0, 0.0, 0.0], "camera_intrinsic": []},
        ],
        "log.json": [
            {"token": LOG_TOKEN, "logfile": "synthetic", "vehicle": "test",
             "date_captured": "2026-09-09", "location": "dhaka"},
        ],
        "scene.json": [
            {"token": SCENE_TOKEN, "log_token": LOG_TOKEN, "name": SCENE, "description": "",
             "nbr_samples": len(KEYFRAMES), "first_sample_token": KEYFRAMES[0][0],
             "last_sample_token": KEYFRAMES[-1][0]},
            {"token": OTHER_SCENE_TOKEN, "log_token": LOG_TOKEN, "name": OTHER_SCENE,
             "description": "", "nbr_samples": 1, "first_sample_token": OTHER_KEYFRAME[0],
             "last_sample_token": OTHER_KEYFRAME[0]},
        ],
        "sample.json": samples, "sample_data.json": sample_data, "ego_pose.json": ego_pose,
        "map.json": [],
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
    extra_scene: bool = False,
) -> str:
    """The stage_road work tree the (concurrently built) stage will produce.

    `extra_scene` also covers OTHER_SCENE, the way a stage_road run over the
    whole capture covers scenes a later --scenes export leaves out.
    """
    os.makedirs(root, exist_ok=True)
    if manifest:
        write_json_atomic(
            os.path.join(root, "run_manifest.json"),
            {"spec": "dhakascenes-pilot/stage_road/v1", "n_keyframes": len(KEYFRAMES)},
        )
    covered = [(SCENE, SCENE_TOKEN, KEYFRAMES)]
    if extra_scene:
        covered.append((OTHER_SCENE, OTHER_SCENE_TOKEN, [OTHER_KEYFRAME]))
    for scene_name, scene_token, keyframes in covered:
        points_dir = os.path.join(root, "scenes", scene_name, "points")
        os.makedirs(points_dir, exist_ok=True)
        rows = []
        for kf_tok, sd_tok, n_raw, road_idx in keyframes:
            rows.append({
                "keyframe_token": kf_tok,
                "scene_token": scene_token,
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
        write_jsonl_atomic(os.path.join(root, "scenes", scene_name, "road.jsonl"), rows)
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

        # The five annotation tables ride along byte-for-byte; the eight
        # SourceRoot owns are rewritten from it, so they are equal as TABLES
        # (same rows, no scoping asked for) rather than as bytes.
        for name in ("attribute.json", "instance.json", "sample_annotation.json"):
            with open(os.path.join(roots["dataroot"], VERSION, name), "rb") as fh:
                src_bytes = fh.read()
            with open(os.path.join(out, VERSION, name), "rb") as fh:
                assert fh.read() == src_bytes
        for name in ("sensor.json", "scene.json", "sample.json", "sample_data.json"):
            with open(os.path.join(roots["dataroot"], VERSION, name), encoding="utf-8") as fh:
                src_rows = json.load(fh)
            with open(os.path.join(out, VERSION, name), encoding="utf-8") as fh:
                assert json.load(fh) == src_rows

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
# Scoping and the map row: what made road/ a valid root of its own
# ---------------------------------------------------------------------------


class TestScopedRoot:
    """road/ used to ship the WHOLE capture's tables around one chunk's labels,
    and an empty map.json the devkit refuses to load. Both are now fixed here."""

    def _table(self, out: str, name: str):
        with open(os.path.join(out, VERSION, f"{name}.json"), encoding="utf-8") as fh:
            return json.load(fh)

    def test_scenes_scopes_the_tables_and_the_labels(self, roots, capsys):
        # stage_road covered BOTH scenes; this export asks for one.
        build_stage_road(roots["stage"], extra_scene=True)
        assert run_main(roots, "--scenes", SCENE) == 0
        out = roots["out"]
        assert [s["name"] for s in self._table(out, "scene")] == [SCENE]
        assert [s["token"] for s in self._table(out, "sample")] == [k[0] for k in KEYFRAMES]
        assert [s["token"] for s in self._table(out, "sample_data")] == [k[1] for k in KEYFRAMES]
        assert len(self._table(out, "ego_pose")) == len(KEYFRAMES)
        # The labels narrow with the tables: no bin and no lidarseg row for a
        # keyframe whose sample_data row this root no longer has.
        assert [r["sample_data_token"] for r in self._table(out, "lidarseg")] == \
            [k[1] for k in KEYFRAMES]
        assert sorted(os.listdir(os.path.join(out, "lidarseg", VERSION))) == \
            sorted(f"{k[1]}_lidarseg.bin" for k in KEYFRAMES)
        assert f"{len(KEYFRAMES)} keyframes" in capsys.readouterr().out

    def test_map_row_is_synthesised_for_a_capture_without_one(self, roots):
        build_stage_road(roots["stage"])
        result = erl.export_road_lidarseg(
            roots["stage"], roots["dataroot"], VERSION, roots["out"],
            erl._guard_paths(roots["dataroot"], VERSION, roots["out"]))
        assert result["map_synthesised"] is True
        # One row binding every exported log, with no mask to name: the devkit
        # dereferences self.map[0] and raises IndexError on an empty table.
        rows = self._table(roots["out"], "map")
        assert len(rows) == 1 and rows[0]["log_tokens"] == [LOG_TOKEN]
        assert rows[0]["filename"] == "" and len(rows[0]["token"]) == 32
        with open(os.path.join(roots["dataroot"], VERSION, "map.json"), encoding="utf-8") as fh:
            assert json.load(fh) == []

    def test_devkit_loads_the_scoped_road_root(self, roots):
        # The point of the whole layer: road/ is a root the stock devkit opens,
        # with no blobs linked in and no arguments beyond version and dataroot.
        nuscenes = pytest.importorskip("nuscenes.nuscenes")
        build_stage_road(roots["stage"], extra_scene=True)
        assert run_main(roots, "--scenes", SCENE) == 0
        nusc = nuscenes.NuScenes(version=VERSION, dataroot=roots["out"], verbose=False)
        assert len(nusc.lidarseg) == len(KEYFRAMES)
        assert len(nusc.sample) == len(KEYFRAMES)
        assert len(nusc.category) == 32
        assert nusc.log[0]["map_token"] == nusc.map[0]["token"]


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
        tables = os.path.join(roots["dataroot"], VERSION)
        before = sorted(os.listdir(tables))
        rc = erl.main([
            "--stage-road-dir", roots["stage"],
            "--dataroot", roots["dataroot"],
            "--out", roots["dataroot"],
            "--version", VERSION,
        ])
        assert rc == 2
        assert "dataroot" in capsys.readouterr().err
        assert not os.path.exists(os.path.join(roots["dataroot"], "lidarseg"))
        assert sorted(os.listdir(tables)) == before

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
