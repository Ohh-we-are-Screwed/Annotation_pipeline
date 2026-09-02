"""Tests for scripts/export_road_coco.py — road masks to COCO RLE for CVAT (C33).

Run: /home/mt/miniconda3/envs/ano_pipe/bin/python -m pytest tests/test_export_road_coco.py -v

The load-bearing assertions: the segmentation is RLE with iscrowd=1 (CVAT
2.72's COCO importer then creates a native MASK shape), and a road with a
vehicle-shaped hole round-trips WITH the hole — the polygon path measured
+29.4% area over-claim on occluders, which is why it is refused here.
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

from pipeline.common.manifest import write_jsonl_atomic, write_marker  # noqa: E402
from pipeline.common.schemas import IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX  # noqa: E402
from pipeline.stage_road.road import write_road_masks_npz  # noqa: E402
from scripts.export_road_coco import main  # noqa: E402

W, H = IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX


def _road_mask_with_hole():
    m = np.zeros((H, W), bool)
    m[400:700, 100:1200] = True     # the road band
    m[500:600, 400:600] = False     # a vehicle standing on it
    return m


def _trees(tmp_path):
    stage1 = tmp_path / "stage1_ingestion" / "scenes" / "chunk_0000"
    stage1.mkdir(parents=True)
    write_jsonl_atomic(str(stage1 / "keyframes.jsonl"), [{
        "keyframe_token": "kf0",
        "cameras": {
            "CAM_FRONT": {"path": "samples/CAM_FRONT/0.jpg"},
            "CAM_BACK": {"path": "samples/CAM_BACK/0.jpg"},
        },
    }])
    road = tmp_path / "stage_road"
    scene = road / "scenes" / "chunk_0000"
    (scene / "masks").mkdir(parents=True)
    write_road_masks_npz(str(scene / "masks" / "kf0.npz"), {
        "CAM_FRONT": _road_mask_with_hole(),
        "CAM_BACK": np.zeros((H, W), bool),   # prompt fired nothing here
    })
    write_jsonl_atomic(str(scene / "road.jsonl"), [{
        "keyframe_token": "kf0", "scene_token": "sc0",
        "mask_path": "scenes/chunk_0000/masks/kf0.npz",
        "cameras": {"CAM_FRONT": {"fired": True, "n_regions": 2, "coverage": 0.2,
                                  "score": 0.83},
                    "CAM_BACK": {"fired": False, "n_regions": 0, "coverage": 0.0,
                                 "score": None}},
    }])
    with open(road / "run_manifest.json", "w") as fh:
        json.dump({"spec": "dhakascenes-pilot/stage_road/v1", "stage": "stage_road"}, fh)
    write_marker(str(road), "fp0", degraded=True, causes=("upstream: sadness",))
    return str(tmp_path / "stage1_ingestion"), str(road), str(tmp_path / "cvat_export_road")


class TestExport:
    def test_rle_with_iscrowd_1_and_the_hole_survives(self, tmp_path):
        from pycocotools import mask as pm
        stage1, road, out = _trees(tmp_path)
        rc = main(["--stage-road-dir", road, "--stage1-dir", stage1,
                   "--out-root", out, "--accept-degraded-upstream"])
        assert rc == 0
        doc = json.load(open(os.path.join(out, "chunk_0000", "instances.json")))
        assert [c["name"] for c in doc["categories"]] == ["road surface"]
        assert len(doc["images"]) == 2      # every camera listed, road or not
        assert len(doc["annotations"]) == 1  # only the fired camera annotates
        ann = doc["annotations"][0]
        assert ann["iscrowd"] == 1
        seg = ann["segmentation"]
        assert isinstance(seg, dict) and seg["size"] == [H, W]
        decoded = pm.decode({"size": seg["size"],
                             "counts": seg["counts"].encode("ascii")}).astype(bool)
        assert np.array_equal(decoded, _road_mask_with_hole())
        assert ann["area"] == int(_road_mask_with_hole().sum())
        # "score" is special-cased (popped into shape confidence) by CVAT's
        # importer; the SAM3 score must ride under an un-special name, and the
        # export must declare it so a created project's schema holds it
        assert ann["attributes"] == {"sam3_score": 0.83}
        assert doc["info"]["cvat_label_attributes"][0]["name"] == "sam3_score"
        front = next(i for i in doc["images"] if i["file_name"] == "samples/CAM_FRONT/0.jpg")
        assert ann["image_id"] == front["id"]

    def test_refuses_degraded_without_flag(self, tmp_path):
        stage1, road, out = _trees(tmp_path)
        rc = main(["--stage-road-dir", road, "--stage1-dir", stage1, "--out-root", out])
        assert rc == 2
        assert not os.path.exists(out)

    def test_refuses_missing_marker(self, tmp_path):
        stage1, road, out = _trees(tmp_path)
        for name in os.listdir(road):
            if name.startswith("_SUCCESS"):
                os.remove(os.path.join(road, name))
        rc = main(["--stage-road-dir", road, "--stage1-dir", stage1,
                   "--out-root", out, "--accept-degraded-upstream"])
        assert rc == 2
