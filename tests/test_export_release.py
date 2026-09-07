"""Tests for scripts/export_release.py on a tiny synthetic nuScenes root.

Synthetic root: 1 scene, 2 samples (`build_dataroot(root, n_samples=N)` builds
more), LIDAR_TOP + CAM_FRONT, sample 0 with an identity ego pose and sample 1
with a non-identity one. 3 pre-labels: one track seen in both samples, one
untracked detection in sample 1 (behind the camera, so its visibility differs).
The dbench validator is run as a subprocess on the output and must report 0
errors. `tests/test_export_release_pipeline.py` reuses this builder for the
post-processing pipeline (stitch / tiers / attributes / double annotation).

Run (cwd anywhere; the test uses tmp_path):
    /home/mt/miniconda3/envs/ano_pipe/bin/python -m pytest tests/test_export_release.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from scripts import export_release as er  # noqa: E402
from pipeline.common.conventions import EGO  # noqa: E402

DBENCH_PY = "/home/mt/dataset_benchmark/.venv/bin/python"
DBENCH_ROOT = "/home/mt/dataset_benchmark"
VERSION = "v1.0-dhaka"
T0_US = 1_700_000_000_000_000  # plausible microseconds since epoch


def _quat_from_yaw(yaw: float) -> list[float]:
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _quat_from_matrix(R: np.ndarray) -> list[float]:
    from pyquaternion import Quaternion
    return [float(v) for v in Quaternion(matrix=R).elements]


# camera -> ego: nuScenes camera frame (x right, y down, z forward) facing +x of ego.
CAM_R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
CAM_K = [[800.0, 0.0, 800.0], [0.0, 800.0, 450.0], [0.0, 0.0, 1.0]]
EGO_POSES = [
    {"translation": [0.0, 0.0, 0.0], "rotation": [1.0, 0.0, 0.0, 0.0]},
    {"translation": [12.5, -3.0, 0.2], "rotation": _quat_from_yaw(0.7)},
]


def _ego_pose_of(i: int) -> dict:
    """Sample 0/1 keep EGO_POSES; every later sample is EGO_POSES[1] + 5 m of x per keyframe."""
    if i < 2:
        return dict(EGO_POSES[i])
    tx, ty, tz = EGO_POSES[1]["translation"]
    return {"translation": [tx + 5.0 * (i - 1), ty, tz], "rotation": list(EGO_POSES[1]["rotation"])}


def build_dataroot(root: str, n_samples: int = 2) -> dict:
    """`n_samples` > 2 also tightens the keyframe period to 0.4 s, so the constant

    5 m/keyframe ego translation is a constant 12.5 m/s for a box fixed in the ego
    frame — what the stitch gate and the velocity-derived attribute need. The
    2-sample default is unchanged.
    """
    step_us = 400_000 if n_samples > 2 else 500_000
    tdir = os.path.join(root, VERSION)
    os.makedirs(tdir)
    log_tok, scene_tok = "log0" * 8, "scn0" * 8
    sensors = [{"token": "senL" * 8, "channel": "LIDAR_TOP", "modality": "lidar"},
               {"token": "senC" * 8, "channel": "CAM_FRONT", "modality": "camera"}]
    calib = [
        {"token": "calL" * 8, "sensor_token": "senL" * 8, "translation": [0, 0, 0],
         "rotation": [1, 0, 0, 0], "camera_intrinsic": []},
        {"token": "calC" * 8, "sensor_token": "senC" * 8, "translation": [1.5, 0.0, 1.4],
         "rotation": _quat_from_matrix(CAM_R), "camera_intrinsic": CAM_K},
    ]
    samples, sds, poses = [], [], []
    sample_tokens = [f"samp{i}000" * 4 for i in range(n_samples)]
    for i, stok in enumerate(sample_tokens):
        ts = T0_US + i * step_us
        samples.append({"token": stok, "timestamp": ts, "scene_token": scene_tok,
                        "prev": sample_tokens[i - 1] if i else "", "next": "",
                        })
        if i:
            samples[i - 1]["next"] = stok
        ptok = f"pose{i}000" * 4
        poses.append({"token": ptok, "timestamp": ts, **_ego_pose_of(i)})
        for ch, cal, fmt, w, h in (("LIDAR_TOP", "calL" * 8, "pcd", 0, 0),
                                   ("CAM_FRONT", "calC" * 8, "jpg", 1600, 900)):
            rel = os.path.join("samples", ch, f"{i:06d}.{'pcd.bin' if fmt == 'pcd' else 'jpg'}")
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            with open(os.path.join(root, rel), "wb") as fh:
                fh.write(b"\0" * 20)
            sds.append({"token": f"sd{ch[:3]}{i}" + "0" * 24, "sample_token": stok,
                        "ego_pose_token": ptok, "calibrated_sensor_token": cal,
                        "filename": rel, "fileformat": fmt, "timestamp": ts,
                        "is_key_frame": True, "height": h, "width": w, "prev": "", "next": ""})
    tables = {
        "log": [{"token": log_tok, "logfile": "synthetic", "vehicle": "test",
                 "date_captured": "2026-08-23", "location": "dhaka"}],
        "scene": [{"token": scene_tok, "log_token": log_tok, "name": "synthetic-0001",
                   "description": "", "nbr_samples": n_samples, "first_sample_token": sample_tokens[0],
                   "last_sample_token": sample_tokens[-1]}],
        "sample": samples, "sample_data": sds, "ego_pose": poses,
        "calibrated_sensor": calib, "sensor": sensors, "map": [],
        "category": [], "attribute": [], "visibility": [], "instance": [], "sample_annotation": [],
    }
    for name, rows in tables.items():
        with open(os.path.join(tdir, f"{name}.json"), "w") as fh:
            json.dump(rows, fh)
    return {"scene_token": scene_tok, "sample_tokens": sample_tokens}


def _record(token, sample_token, category, t, size, yaw, track_id, n_pts, attribute=None):
    return {
        "__contract__": "I-4", "__schema_version__": "dhakascenes-pilot/schemas/v1",
        "token": token, "sample_token": sample_token,
        "instance_token": f"pilot-track:x:{track_id}" if track_id else f"pilot-det:{token}",
        "category": category, "frame": EGO, "t_ns": T0_US * 1000, "time_base": "unix_ns",
        "translation_m": t, "size_wlh_m": size, "rotation_wxyz": _quat_from_yaw(yaw),
        "num_lidar_pts": n_pts, "track_id": track_id, "velocity_mps": [1.0, 0.0],
        "num_lidar_pts_basis": "single_sweep_ground_filtered_pre_inflation",
        "provenance": {"source": "pipeline", "tier": "auto_accept",
                       "gates": {"conf": 0.9, "lidar_pts_ok": True}},
        "coverage_config": "R1", "attribute": attribute,
    }


def write_prelabels(path: str, s: list[str]) -> list[dict]:
    recs = [
        _record("k0:CAM_FRONT:0", s[0], "a car", [10.0, 0.5, 0.8], [1.9, 4.5, 1.6], 0.1, "7", 120),
        _record("k1:CAM_FRONT:0", s[1], "a car", [10.5, 0.4, 0.8], [1.9, 4.5, 1.6], 0.12, "7", 110),
        _record("k1:CAM_FRONT:3", s[1], "a pedestrian", [-8.0, 1.0, 0.9], [0.6, 0.7, 1.7], 2.0, None, 9),
    ]
    with open(path, "w") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    return recs


@pytest.fixture
def exported(tmp_path):
    src = str(tmp_path / "src")
    info = build_dataroot(src)
    pre = str(tmp_path / "prelabels.jsonl")
    recs = write_prelabels(pre, info["sample_tokens"])
    hv = tmp_path / "hv.txt"
    hv.write_text("synthetic-0001\n")
    out = str(tmp_path / "release")
    res = er.export_release(pre, src, VERSION, out,
                            os.path.join(ROOT, "configs", "release_category_map.yaml"),
                            human_verified_scenes_path=str(hv))
    return {"src": src, "out": out, "res": res, "recs": recs, **info}


def _load(out, name):
    with open(os.path.join(out, VERSION, f"{name}.json")) as fh:
        return json.load(fh)


def test_round_trip_geometry(exported):
    anns = {a["dhakascenes_record_token"]: a for a in _load(exported["out"], "sample_annotation")}
    src = er.SourceRoot(exported["src"], VERSION)
    for r in exported["recs"]:
        a = anns[r["token"]]
        pose = src.lidar_ego_pose(r["sample_token"])
        t_e, q_e = er.box_global_to_ego(a["translation"], a["rotation"], pose)
        assert np.allclose(t_e, r["translation_m"], atol=1e-9)
        assert np.allclose(q_e, er.normalise_quat(np.asarray(r["rotation_wxyz"])), atol=1e-9)
        assert a["size"] == r["size_wlh_m"]
        assert abs(np.linalg.norm(a["rotation"]) - 1.0) < 1e-9
    # non-identity pose actually moved the second box
    a1 = anns["k1:CAM_FRONT:0"]
    assert not np.allclose(a1["translation"], [10.5, 0.4, 0.8])
    # explicit hand computation for the non-identity pose
    yaw, (tx, ty, tz) = 0.7, EGO_POSES[1]["translation"]
    exp = [tx + 10.5 * math.cos(yaw) - 0.4 * math.sin(yaw),
           ty + 10.5 * math.sin(yaw) + 0.4 * math.cos(yaw), tz + 0.8]
    assert np.allclose(a1["translation"], exp, atol=1e-9)
    assert exported["res"].meta["devkit_cross_check"]["all_passed"] is True
    assert exported["res"].meta["devkit_cross_check"]["n_checked"] == 3


def test_tokens_chains_and_instances(exported):
    out = exported["out"]
    anns = _load(out, "sample_annotation")
    inst = {i["token"]: i for i in _load(out, "instance")}
    cats = {c["token"]: c["name"] for c in _load(out, "category")}
    assert len(anns) == 3 and len(inst) == 2 and len(cats) == 18
    by_tok = {a["token"]: a for a in anns}
    tracked = [a for a in anns if a["dhakascenes_record_token"].endswith(":0")]
    tracked.sort(key=lambda a: a["dhakascenes_record_token"])
    a0, a1 = tracked
    assert a0["instance_token"] == a1["instance_token"]
    assert a0["prev"] == "" and a0["next"] == a1["token"]
    assert a1["prev"] == a0["token"] and a1["next"] == ""
    i = inst[a0["instance_token"]]
    assert i["nbr_annotations"] == 2
    assert i["first_annotation_token"] == a0["token"] and i["last_annotation_token"] == a1["token"]
    assert cats[i["category_token"]] == "car"
    single = by_tok[[a["token"] for a in anns if a["dhakascenes_record_token"].endswith(":3")][0]]
    assert single["prev"] == single["next"] == ""
    assert cats[inst[single["instance_token"]]["category_token"]] == "pedestrian"
    assert inst[single["instance_token"]]["nbr_annotations"] == 1
    for a in anns:
        assert a["num_radar_pts"] == 0 and isinstance(a["num_lidar_pts"], int)
        assert len(a["token"]) == 32


def test_visibility_and_attributes(exported):
    out = exported["out"]
    anns = {a["dhakascenes_record_token"]: a for a in _load(out, "sample_annotation")}
    attrs = {a["token"]: a["name"] for a in _load(out, "attribute")}
    assert anns["k0:CAM_FRONT:0"]["visibility_token"] == "4"      # 10 m ahead of the camera
    assert anns["k1:CAM_FRONT:3"]["visibility_token"] == "1"      # behind the camera
    assert all(a["visibility_basis"] == "camera_fov_corner_fraction" for a in anns.values())
    assert [attrs[t] for t in anns["k0:CAM_FRONT:0"]["attribute_tokens"]] == ["vehicle.moving"]
    assert [attrs[t] for t in anns["k1:CAM_FRONT:0"]["attribute_tokens"]] == ["vehicle.moving"]
    assert anns["k1:CAM_FRONT:3"]["attribute_tokens"] == []
    levels = {v["token"]: v["level"] for v in _load(out, "visibility")}
    assert levels == {"1": "v0-40", "2": "v40-60", "3": "v60-80", "4": "v80-100"}


def test_release_meta_and_source_untouched(exported):
    meta = json.load(open(os.path.join(exported["out"], "release_meta.json")))
    assert meta["scenes"]["synthetic-0001"]["human_verified"] is True
    assert meta["counts"]["n_annotations"] == 3 and meta["counts"]["n_instances"] == 2
    assert meta["mapper"]["sha256"] and meta["visibility"]["basis"] == "camera_fov_corner_fraction"
    assert meta["num_lidar_pts_basis"] == ["single_sweep_ground_filtered_pre_inflation"]
    # source root's annotation tables are still empty
    for name in ("sample_annotation", "instance", "category"):
        assert json.load(open(os.path.join(exported["src"], VERSION, f"{name}.json"))) == []
    # blobs reachable through the symlink
    assert os.path.isfile(os.path.join(exported["out"], "samples", "CAM_FRONT", "000000.jpg"))


def test_unmapped_category_errors(tmp_path):
    src = str(tmp_path / "src")
    info = build_dataroot(src)
    pre = tmp_path / "p.jsonl"
    r = _record("k0:CAM_FRONT:9", info["sample_tokens"][0], "a flying saucer", [5, 0, 0], [1, 1, 1], 0, None, 3)
    r2 = _record("k0:CAM_FRONT:8", info["sample_tokens"][0], "static-obstacle", [5, 0, 0], [1, 1, 1], 0, None, 3)
    pre.write_text(json.dumps(r) + "\n" + json.dumps(r2) + "\n")
    with pytest.raises(er.ExportError) as ei:
        er.export_release(str(pre), src, VERSION, str(tmp_path / "o"),
                          os.path.join(ROOT, "configs", "release_category_map.yaml"))
    assert "a flying saucer" in str(ei.value) and "static-obstacle" in str(ei.value)


def test_visibility_fallback_without_intrinsics(tmp_path):
    src = str(tmp_path / "src")
    info = build_dataroot(src)
    p = os.path.join(src, VERSION, "calibrated_sensor.json")
    rows = json.load(open(p))
    for row in rows:
        row["camera_intrinsic"] = []
    json.dump(rows, open(p, "w"))
    pre = str(tmp_path / "p.jsonl")
    write_prelabels(pre, info["sample_tokens"])
    res = er.export_release(pre, src, VERSION, str(tmp_path / "o"),
                            os.path.join(ROOT, "configs", "release_category_map.yaml"))
    assert res.meta["visibility"]["basis"] == "assumed_full"
    assert res.meta["visibility"]["fallback_reason"]
    assert all(a["visibility_token"] == "4" for a in res.tables["sample_annotation"])


@pytest.mark.skipif(not os.path.exists(DBENCH_PY), reason="dbench venv not present")
def test_dbench_ingest_validate_passes(exported, tmp_path):
    proc = subprocess.run(
        [DBENCH_PY, "-m", "dbench.cli", "--artifacts", str(tmp_path / "artifacts"),
         "ingest", "validate", "--root", exported["out"], "--version", VERSION],
        cwd=DBENCH_ROOT, capture_output=True, text=True,
    )
    print(proc.stdout, proc.stderr)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.startswith("VALID")
    report = json.load(open(tmp_path / "artifacts" / "quality" / "dataset_validation.json"))
    assert report["n_errors"] == 0 and report["stats"]["n_annotations"] == 3
