"""export_lidarseg.py: do the labels land on the RIGHT raw rows? (2026-09-13)

Everything else this exporter does is bookkeeping; the one thing that can be
silently wrong is the index mapping, because a lidarseg `.bin` is positional
and a sheared one looks exactly like a correct one. So the fixture builds the
fused cloud the way Stage 1 builds it — the SAME `read_pcd_bin` / `thin_stereo`
/ `Transform` / `stereo_block_to_ego`, the same order, then a boolean mask that
preserves order — and the tests assert the label of a painted point comes back
out on the row of the RAW blob it started on, in both channels.

Two keyframes, a 6-point LIDAR_TOP blob and a 5-point ZED_WORLD blob, a
non-identity ego_pose (so a missing global->ego hop cannot pass), three
instances of which one carries a phrase the release has no class for.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.conventions import (  # noqa: E402
    EGO,
    LIDAR,
    NUSCENES_GLOBAL,
    Transform,
    apply_transform,
)
from pipeline.common.manifest import write_marker  # noqa: E402
from pipeline.stage1_ingestion.ingest import (  # noqa: E402
    read_pcd_bin,
    stereo_block_to_ego,
    thin_stereo,
)
from scripts import export_lidarseg as ex  # noqa: E402

VERSION = "v1.0-dhaka-fixed2"
SCENE = "s"
TOKENS = ("kf0", "kf1")
# The ego_pose the LIDAR_TOP row carries. ZED_WORLD points are in the GLOBAL
# frame with an identity pose of their own, so this translation is the entire
# hop: get it wrong and no stereo row matches.
EGO_POSE = {"token": "ep-lidar", "timestamp": 1, "translation": [10.0, -3.0, 0.5],
            "rotation": [0.0, 0.0, 0.0, 1.0]}     # 180 deg about z, not identity
IDENTITY = {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]}
STAGE1_CONFIG = {"stereo_rings": [100, 101], "stereo_stride": 1, "fuse_stereo": True,
                 "stereo_z_correction_m": {}, "stereo_pitch_correction": {}}
# Rows of the FUSED cloud Stage 1's ground/range/height filter threw away. It is
# a boolean mask, so order survives and the kept cloud is a subsequence.
DROPPED = (1, 4, 7)
CATEGORIES = [{"token": "cat-car", "name": "car", "description": "a car"},
              {"token": "cat-ped", "name": "pedestrian", "description": ""}]
# instance_id -> phrase. 2 is the phrase this release has no class for.
PHRASES = ["a car", "a pedestrian", "a flying saucer"]


def _lidar_blob(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    cloud = rng.uniform(-20, 20, size=(6, 5)).astype(np.float32)
    cloud[:, 4] = np.array([0, 1, 2, 3, 0, 1], np.float32)      # Livox rings
    return cloud


def _zed_blob(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 100)
    cloud = rng.uniform(-20, 20, size=(5, 5)).astype(np.float32)
    cloud[:, 4] = np.array([100, 101, 100, 101, 101], np.float32)
    return cloud


def _fuse(lidar: np.ndarray, zed: np.ndarray) -> np.ndarray:
    """Stage 1's single sweep, unfiltered — ingest.py's own functions, its order."""
    raw, _ = thin_stereo(lidar, STAGE1_CONFIG["stereo_rings"], STAGE1_CONFIG["stereo_stride"])
    t = Transform.from_nuscenes({**IDENTITY}, source_frame=LIDAR, parent_frame=EGO)
    single = np.column_stack([apply_transform(t.matrix(), raw[:, :3].astype(np.float64)),
                              raw[:, 3:5].astype(np.float64)])
    zraw, _ = thin_stereo(zed, STAGE1_CONFIG["stereo_rings"], STAGE1_CONFIG["stereo_stride"])
    block = stereo_block_to_ego(
        zraw, frame="global_identity", ring=None, t_sensor_to_ego=None,
        t_global_to_ego=Transform.from_nuscenes(
            EGO_POSE, source_frame=EGO, parent_frame=NUSCENES_GLOBAL).inverse_matrix(),
        z_correction_m={}, pitch_correction={})
    return np.vstack([single, block]).astype(np.float32)


class Fixture:
    def __init__(self, tmp_path):
        self.work = tmp_path / "work"
        self.export = tmp_path / "chunk_99" / "boxes"
        self.tables = self.export / VERSION
        self.stage1 = self.work / "stage1_ingestion"
        self.stage5 = self.work / "stage5_lift"
        self.kept: dict[str, np.ndarray] = {}
        self.blobs: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def run(self, argv=()):
        return ex.main(["--paths", "unused", "--export-dir", str(self.export),
                        "--stage1-dir", str(self.stage1), "--stage5-dir", str(self.stage5),
                        "--scene", SCENE, *argv])

    def meta(self):
        return json.loads((self.export / "lidarseg" / "lidarseg_meta.json").read_text())

    def bin_of(self, token, channel):
        sd = f"sd-{channel}-{token}"
        return np.fromfile(self.export / "lidarseg" / VERSION / f"{sd}_lidarseg.bin", np.uint8)


@pytest.fixture()
def fx(tmp_path, monkeypatch):
    f = Fixture(tmp_path)
    (f.tables).mkdir(parents=True)
    (f.export / "samples" / "LIDAR_TOP").mkdir(parents=True)
    (f.export / "samples" / "ZED_WORLD").mkdir(parents=True)
    (f.stage1 / "scenes" / SCENE).mkdir(parents=True)
    (f.stage1 / "clouds" / SCENE).mkdir(parents=True)
    (f.stage5 / "scenes" / SCENE / "points").mkdir(parents=True)

    sample_data, keyframes, lift = [], [], []
    for i, token in enumerate(TOKENS, start=1):
        lidar, zed = _lidar_blob(i), _zed_blob(i)
        f.blobs[token] = (lidar, zed)
        for channel, blob in (("LIDAR_TOP", lidar), ("ZED_WORLD", zed)):
            name = f"samples/{channel}/{i:06d}.pcd.bin"
            blob.tofile(f.export / name)
            sample_data.append({
                "token": f"sd-{channel}-{token}", "sample_token": token,
                "filename": name, "fileformat": "pcd", "width": 0, "height": 0,
                "timestamp": 1_700_000_000_000_000 + i, "is_key_frame": True,
                "num_points": int(blob.shape[0]),
                "calibrated_sensor_token": f"cs-{channel}",
                "ego_pose_token": EGO_POSE["token"] if channel == "LIDAR_TOP" else "ep-zed",
            })
        fused = _fuse(lidar, zed)
        keep = np.ones(fused.shape[0], bool)
        keep[list(DROPPED)] = False
        kept = fused[keep]
        f.kept[token] = kept
        cloud_path = f.stage1 / "clouds" / SCENE / f"{token}.pcd.bin"
        kept.tofile(cloud_path)
        keyframes.append({"keyframe_token": token, "scene_token": "sc",
                          "lidar_sample_data_token": f"sd-LIDAR_TOP-{token}",
                          "single_sweep_cloud": {"path": str(cloud_path),
                                                 "n_points": int(kept.shape[0])}})
        # Paint one point per instance, spread over both channels: rows 0 and 2
        # are lidar, the rest stereo (the fused cloud is 6 lidar + 5 stereo, less
        # the three dropped rows).
        point_index = np.array([0, 2, 5, 7], np.int32)
        instance_id = np.array([0, 1, 0, 2], np.int32)
        np.savez(f.stage5 / "scenes" / SCENE / "points" / f"{token}.npz",
                 point_index=point_index, instance_id=instance_id,
                 channel_index=np.zeros(4, np.int8))
        lift.append({"keyframe_token": token, "scene_token": "sc",
                     "n_points_cloud": int(kept.shape[0]),
                     "points_path": f"scenes/{SCENE}/points/{token}.npz",
                     "cloud_path": str(cloud_path),
                     "instances": [{"instance_id": j, "class_name": phrase, "channel": "CAM_FRONT",
                                    "proposal_index": j} for j, phrase in enumerate(PHRASES)]})

    (f.tables / "sample_data.json").write_text(json.dumps(sample_data))
    (f.tables / "category.json").write_text(json.dumps(CATEGORIES))
    (f.tables / "calibrated_sensor.json").write_text(json.dumps(
        [{"token": f"cs-{c}", **IDENTITY} for c in ("LIDAR_TOP", "ZED_WORLD")]))
    (f.tables / "ego_pose.json").write_text(json.dumps(
        [EGO_POSE, {"token": "ep-zed", "timestamp": 1, **IDENTITY}]))
    (f.export / "release_meta.json").write_text(json.dumps(
        {"version": VERSION,
         "mapper": {"used": {"a car": "car", "a pedestrian": "pedestrian"}}}))
    (f.export / "DELIVERY_NOTE.md").write_text(
        "# chunk_99\n\n## Annotation rule\n\nOne cuboid per shipped detection.\n")
    (f.stage1 / "scenes" / SCENE / "keyframes.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in keyframes))
    (f.stage1 / "scenes" / SCENE / "filter_diagnostics.json").write_text(
        json.dumps({"scene": SCENE, "config": STAGE1_CONFIG}))
    (f.stage5 / "scenes" / SCENE / "lift.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in lift))
    (f.stage5 / "run_manifest.json").write_text(json.dumps({"stage": "stage5_lift"}))
    write_marker(str(f.stage5), "fingerprint", degraded=False)
    monkeypatch.setattr(ex, "load_paths", lambda _p: type("P", (), {"work_root": str(f.work)})())
    return f


# ---------------------------------------------------------------------------
# the index mapping — the only thing that can be silently wrong
# ---------------------------------------------------------------------------

def _expected(fx, token):
    """{(channel, raw row): label} worked out independently of the exporter.

    The fused cloud is rebuilt here, the painted fused rows are found by
    matching the kept rows back into it, and the block boundary (6 lidar rows,
    then stereo) turns a fused row into a raw row.
    """
    lidar, zed = fx.blobs[token]
    fused = _fuse(lidar, zed)
    kept = fx.kept[token]
    keys = {row.tobytes(): i for i, row in enumerate(fused)}
    label_of = {0: 1, 1: 2, 2: 0}          # car, pedestrian, unmapped phrase
    out = {}
    for point_index, instance_id in zip((0, 2, 5, 7), (0, 1, 0, 2)):
        fused_row = keys[kept[point_index].tobytes()]
        channel, raw_row = (("LIDAR_TOP", fused_row) if fused_row < lidar.shape[0]
                            else ("ZED_WORLD", fused_row - lidar.shape[0]))
        out[(channel, raw_row)] = label_of[instance_id]
    return out


def test_labels_land_on_the_right_raw_rows_in_both_channels(fx):
    assert fx.run() == 0
    for token in TOKENS:
        expected = _expected(fx, token)
        assert {c for c, _ in expected} == {"LIDAR_TOP", "ZED_WORLD"}, "fixture paints both"
        for channel, n in (("LIDAR_TOP", 6), ("ZED_WORLD", 5)):
            labels = fx.bin_of(token, channel)
            assert labels.shape == (n,), "one label per RAW point, not per kept point"
            for row in range(n):
                assert int(labels[row]) == expected.get((channel, row), 0)


def test_every_unpainted_row_is_zero(fx):
    fx.run()
    painted = sum(len(_expected(fx, t)) for t in TOKENS)
    labelled = sum(int((fx.bin_of(t, c) > 0).sum()) for t in TOKENS
                   for c in ("LIDAR_TOP", "ZED_WORLD"))
    # 4 painted points per keyframe, one of them an unmapped phrase that stays 0
    assert painted == 8 and labelled == 6
    assert fx.meta()["counts"]["points_labelled"] == 6


def test_the_basis_names_the_raw_file_order_of_each_channel(fx):
    fx.run()
    assert fx.meta()["basis"] == {"LIDAR_TOP": "raw_lidar_top_file_order",
                                  "ZED_WORLD": "raw_zed_world_file_order"}


# ---------------------------------------------------------------------------
# the tables
# ---------------------------------------------------------------------------

def test_category_gains_index_and_a_noise_row_and_a_rerun_is_idempotent(fx):
    fx.run()
    once = json.loads((fx.tables / "category.json").read_text())
    assert [r["name"] for r in once] == ["noise", "car", "pedestrian"]
    assert [r["index"] for r in once] == [0, 1, 2]
    assert "NOT PAINTED BY ANY OBJECT MASK" in once[0]["description"]
    # the release's own rows keep every byte they had, plus the index
    for before, after in zip(CATEGORIES, once[1:]):
        assert {k: v for k, v in after.items() if k != "index"} == before
    assert json.loads((fx.tables / ex.CATEGORY_BACKUP).read_text()) == CATEGORIES
    fx.run()
    assert json.loads((fx.tables / "category.json").read_text()) == once
    assert json.loads((fx.tables / ex.CATEGORY_BACKUP).read_text()) == CATEGORIES


def test_lidarseg_json_matches_the_bins_on_disk(fx):
    fx.run()
    rows = json.loads((fx.tables / "lidarseg.json").read_text())
    files = sorted(p for p in os.listdir(fx.export / "lidarseg" / VERSION)
                   if p.endswith(".bin"))
    # the devkit asserts exactly this equality at load time
    assert len(rows) == len(files) == 4
    tokens = {r["token"] for r in json.loads((fx.tables / "sample_data.json").read_text())}
    for row in rows:
        assert row["token"] == row["sample_data_token"] and row["token"] in tokens
        assert os.path.isfile(fx.export / row["filename"])
        assert not os.path.islink(fx.export / row["filename"])
        assert row["filename"] == f"lidarseg/{VERSION}/{row['token']}_lidarseg.bin"


def test_an_unmapped_phrase_is_counted_and_labels_nothing(fx):
    fx.run()
    meta = fx.meta()
    assert meta["unmapped_phrases"] == {"a flying saucer": 2}    # once per keyframe
    assert meta["counts"]["points_unmapped_phrase"] == 2         # one painted point each
    assert set(meta["points_by_class_and_channel"]) == {"car", "pedestrian"}


def test_the_delivery_note_section_survives_a_rerun(fx):
    before = (fx.export / "DELIVERY_NOTE.md").read_text()
    fx.run()
    once = (fx.export / "DELIVERY_NOTE.md").read_text()
    fx.run()
    twice = (fx.export / "DELIVERY_NOTE.md").read_text()
    assert once == twice
    assert once.count(ex.NOTE_HEADING) == 1
    for line in before.splitlines():
        assert line in once
    assert "0 means UNLABELLED, not noise" in once
    assert "TWO bins per keyframe" in once


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------

def _break_one_keyframe(fx):
    """Move a painted point of kf0 off the reconstruction by one float."""
    path = fx.stage1 / "clouds" / SCENE / "kf0.pcd.bin"
    kept = read_pcd_bin(str(path))
    kept[0, 0] = np.float32(kept[0, 0] + 1.0)     # row 0 is painted
    kept.tofile(path)


def test_a_keyframe_with_an_unmatched_painted_point_is_refused_and_counted(fx, monkeypatch):
    monkeypatch.setattr(ex, "MAX_REFUSED_FRACTION", 0.9)   # let the other keyframe through
    _break_one_keyframe(fx)
    assert fx.run() == 0
    meta = fx.meta()
    assert meta["counts"]["keyframes_refused"] == 1
    assert meta["counts"]["bins_written"] == 2             # kf1 only, both channels
    assert "did not match the rebuilt fused cloud" in meta["refused_keyframes"][0]
    assert not os.path.exists(fx.export / "lidarseg" / VERSION / "sd-LIDAR_TOP-kf0_lidarseg.bin")


def test_too_many_refused_keyframes_refuses_the_scene_and_writes_nothing(fx, capsys):
    _break_one_keyframe(fx)
    assert fx.run() == 2                                   # 1 of 2 keyframes, way over 1%
    assert "export_lidarseg" in capsys.readouterr().err
    assert not os.path.exists(fx.export / "lidarseg")
    assert not os.path.exists(fx.tables / "lidarseg.json")
    assert json.loads((fx.tables / "category.json").read_text()) == CATEGORIES
    assert ex.NOTE_HEADING not in (fx.export / "DELIVERY_NOTE.md").read_text()


def test_a_stage5_tree_with_no_marker_is_refused(fx, capsys):
    os.unlink(fx.stage5 / "_SUCCESS")
    assert fx.run() == 2
    assert "completion marker" in capsys.readouterr().err
    assert not os.path.exists(fx.export / "lidarseg")


def test_a_kept_cloud_that_is_not_the_painted_one_is_refused(fx, monkeypatch):
    """`n_points_cloud` is lift.jsonl's claim about the cloud it painted; if the
    cloud on disk has a different length it is a different cloud."""
    monkeypatch.setattr(ex, "MAX_REFUSED_FRACTION", 0.9)
    path = fx.stage5 / "scenes" / SCENE / "lift.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["n_points_cloud"] += 1
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert fx.run() == 0
    assert "n_points_cloud" in fx.meta()["refused_keyframes"][0]
