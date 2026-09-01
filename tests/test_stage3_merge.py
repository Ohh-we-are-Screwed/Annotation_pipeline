"""Tests for pipeline/stage3_merge/merge.py.

Run: /home/mt/miniconda3/envs/ano_pipe/bin/python -m pytest tests/test_stage3_merge.py -v
"""

from __future__ import annotations

import copy
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pipeline.stage3_proposals.proposals import build_caption, load_taxonomy  # noqa: E402
from pipeline.stage3_merge.merge import (  # noqa: E402
    ARM_B_PHRASES,
    MergeContractError,
    merge_rows,
)

DHAKA = os.path.join(ROOT, "configs", "taxonomy_pilot_dhaka.yaml")


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy(DHAKA)


@pytest.fixture(scope="module")
def caption(taxonomy):
    return build_caption(taxonomy.phrases)


def _span(caption, phrase):
    return list(caption.phrase_char_spans[caption.phrases.index(phrase)])


def _row(caption, names, boxes, **over):
    row = {
        "spec": "dhakascenes-pilot/stage3_proposals/v1",
        "keyframe_token": "kf0", "scene_token": "sc0", "t_ns": 1, "time_base": "utc",
        "coverage_config": "full", "channel": "CAM_FRONT",
        "sample_data_token": "sd0", "calibrated_sensor_token": "cs0",
        "ego_pose_token": "ep0", "dt_ns": 0, "image_path": "img/0.jpg",
        "image_size_px": [1600, 900], "model_input_size_px": [1600, 928],
        "resize_policy": "letterbox",
        "checkpoint": {"model_id": "m", "revision": "r", "sha256": "s"},
        "prompt": {"caption_sha256": "old", "taxonomy_sha256": "old", "span_map": None},
        "n_proposals": len(boxes),
        "score_aggregation": "yolo_class_confidence",
        "dedup": {"n_in": len(boxes), "n_out": len(boxes)},
        "boxes_xyxy_px": [list(b) for b in boxes],
        "scores": [0.9] * len(boxes),
        "class_names": list(names),
        "nuscenes_categories": [["x"] for _ in names],
        "phrase_char_spans": [_span(caption, n) for n in names],
        "seed": 0,
    }
    row.update(over)
    return row


BOX = [100.0, 100.0, 200.0, 200.0]        # the contested region
BOX_FAR = [500.0, 500.0, 600.0, 600.0]    # elsewhere


class TestArbitration:
    def test_cng_suppresses_car(self, caption, taxonomy):
        a = _row(caption, ["a car"], [BOX])
        b = _row(caption, ["an auto rickshaw"], [[102.0, 101.0, 199.0, 198.0]])
        m = merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        assert m["class_names"] == ["an auto rickshaw"]
        assert m["proposal_arm"] == ["arm_b"]
        assert m["n_proposals"] == 1
        led = m["merge"]
        assert led["n_suppressed_arm_a"] == 1
        (s,) = led["suppressed_arm_a"]
        assert s["class_name"] == "a car"
        assert s["suppressed_by"] == 0          # index of the cng box in MERGED arrays
        assert s["box_xyxy_px"] == BOX

    def test_pedestrian_keeps_both(self, caption, taxonomy):
        a = _row(caption, ["a pedestrian"], [BOX])
        b = _row(caption, ["a rickshaw"], [[110.0, 90.0, 210.0, 205.0]])
        m = merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        assert m["class_names"] == ["a pedestrian", "a rickshaw"]
        assert m["proposal_arm"] == ["arm_a", "arm_b"]
        assert m["merge"]["n_suppressed_arm_a"] == 0
        assert m["merge"]["n_kept_both"] == 1

    def test_below_iou_threshold_is_no_contest(self, caption, taxonomy):
        a = _row(caption, ["a car"], [BOX])
        b = _row(caption, ["an auto rickshaw"], [BOX_FAR])
        m = merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        assert m["class_names"] == ["a car", "an auto rickshaw"]
        assert m["merge"]["n_suppressed_arm_a"] == 0

    def test_out_of_table_overlap_keeps_both_and_counts(self, caption, taxonomy):
        a = _row(caption, ["a road barrier"], [BOX])
        b = _row(caption, ["a rickshaw"], [[101.0, 101.0, 201.0, 201.0]])
        m = merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        assert m["merge"]["n_overlap_out_of_table"] == 1
        assert len(m["class_names"]) == 2

    def test_scores_never_arbitrate(self, caption, taxonomy):
        # arm A very confident, arm B weak: authority still wins (C21's lesson)
        a = _row(caption, ["a car"], [BOX], scores=[0.99])
        b = _row(caption, ["an auto rickshaw"], [[102.0, 101.0, 199.0, 198.0]],
                 scores=[0.31])
        m = merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        assert m["class_names"] == ["an auto rickshaw"]


class TestSchema:
    def test_empty_arm_b_roundtrips_original_keys(self, caption, taxonomy):
        a = _row(caption, ["a car", "a bus"], [BOX, BOX_FAR])
        b = _row(caption, [], [])
        before = copy.deepcopy(a)
        m = merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        for key, value in before.items():
            if key == "prompt":
                continue  # caption/taxonomy shas are rewritten by design
            assert json.dumps(m[key], sort_keys=True) == json.dumps(value, sort_keys=True), key
        assert m["prompt"]["caption_sha256"] == caption.sha256
        assert m["merge"]["n_arm_b_in"] == 0

    def test_input_rows_not_mutated(self, caption, taxonomy):
        a = _row(caption, ["a car"], [BOX])
        b = _row(caption, ["an auto rickshaw"], [[102.0, 101.0, 199.0, 198.0]])
        a2, b2 = copy.deepcopy(a), copy.deepcopy(b)
        merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        assert a == a2 and b == b2

    def test_arm_b_spans_and_categories(self, caption, taxonomy):
        a = _row(caption, [], [])
        b = _row(caption, ["a rickshaw", "an auto rickshaw"], [BOX, BOX_FAR])
        m = merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        assert m["phrase_char_spans"] == [_span(caption, "a rickshaw"),
                                         _span(caption, "an auto rickshaw")]
        assert m["nuscenes_categories"] == [["dhaka.cycle_rickshaw"],
                                            ["dhaka.cng"]]

    def test_stage3b_parallel_arrays_extended(self, caption, taxonomy):
        a = _row(caption, ["a car", "a pedestrian"], [BOX, BOX_FAR],
                 track_ids=[7, 8], box_sources=["yolo", "recovered"],
                 n_propagated_hops=[0, 2], refined=[False, True],
                 boxes_xyxy_px_original=[None, [1.0, 2.0, 3.0, 4.0]])
        b = _row(caption, ["an auto rickshaw"], [[102.0, 101.0, 199.0, 198.0]])
        m = merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)
        # car suppressed -> its entries leave EVERY parallel array
        assert m["class_names"] == ["a pedestrian", "an auto rickshaw"]
        assert m["track_ids"] == [8, None]
        assert m["box_sources"] == ["recovered", "arm_b"]
        # the SURVIVOR is the pedestrian (hops 2); the arm B box fills with 0
        assert m["n_propagated_hops"] == [2, 0]
        assert m["refined"] == [True, False]
        assert m["boxes_xyxy_px_original"] == [[1.0, 2.0, 3.0, 4.0], None]

    def test_refuses_leaked_arm_b_class(self, caption, taxonomy):
        a = _row(caption, [], [])
        b = _row(caption, ["a car"], [BOX])
        with pytest.raises(MergeContractError):
            merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)

    def test_refuses_frame_identity_mismatch(self, caption, taxonomy):
        a = _row(caption, [], [])
        b = _row(caption, [], [], sample_data_token="OTHER")
        with pytest.raises(MergeContractError):
            merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5)


from pipeline.common.manifest import write_json_atomic, write_jsonl_atomic, write_marker  # noqa: E402
from pipeline.stage3_merge import merge as m3  # noqa: E402

FP = "fp-test-0001"


def _tree(root, rows_by_scene, *, caption_text, phrases_in_use, degraded=False,
          image_size_px=(1600, 900)):
    os.makedirs(root, exist_ok=True)
    write_json_atomic(os.path.join(root, "run_manifest.json"), {
        "spec": "dhakascenes-pilot/stage3_proposals/v1",
        "provider": "yolo11",
        "prompt": {"caption": caption_text, "caption_sha256": "irrelevant"},
        "upstream": {"metadata_fingerprint": FP, "fingerprint_spec": "spec/v1"},
        "checkpoint": {"model_id": "m", "revision": "r", "sha256": "s"},
        "image_size_px": list(image_size_px),
        "class_map": {"path": "p", "sha256": "s", "phrases_in_use": list(phrases_in_use),
                      "unreachable_phrases": []},
    })
    for scene, rows in rows_by_scene.items():
        write_jsonl_atomic(os.path.join(root, "scenes", scene, "proposals.jsonl"), rows)
    write_marker(root, FP, degraded=degraded,
                 causes=("scene-x: flagged",) if degraded else ())


class TestDriver:
    def _dirs(self, tmp_path, caption, *, b_caption_text=None, degraded_a=False,
              b_image_size_px=(1600, 900)):
        a_rows = [_row(caption, ["a car"], [BOX]),
                  _row(caption, ["a pedestrian"], [BOX_FAR], keyframe_token="kf1",
                       sample_data_token="sd1")]
        b_rows = [_row(caption, ["an auto rickshaw"], [[102.0, 101.0, 199.0, 198.0]]),
                  _row(caption, [], [], keyframe_token="kf1", sample_data_token="sd1")]
        a_dir, b_dir = str(tmp_path / "a"), str(tmp_path / "b")
        # arm A ran under the v2 caption: a byte prefix of the v3 caption
        v2_text = caption.text[: caption.text.index(" a rickshaw.")]
        _tree(a_dir, {"scene-0001": a_rows}, caption_text=v2_text,
              phrases_in_use=["a car", "a pedestrian"], degraded=degraded_a)
        _tree(b_dir, {"scene-0001": b_rows},
              caption_text=b_caption_text or caption.text,
              phrases_in_use=list(ARM_B_PHRASES),
              image_size_px=b_image_size_px)
        return a_dir, b_dir, str(tmp_path / "out")

    def test_clean_merge_writes_rows_manifest_marker(self, tmp_path, caption, taxonomy):
        a, b, out = self._dirs(tmp_path, caption)
        rc = m3.main(["--arm-a-dir", a, "--arm-b-dir", b, "--out-dir", out,
                      "--taxonomy", DHAKA])
        assert rc == 0
        assert os.path.isfile(os.path.join(out, "_SUCCESS"))
        with open(os.path.join(out, "scenes", "scene-0001", "proposals.jsonl")) as fh:
            rows = [json.loads(line) for line in fh]
        assert [r["merge"]["n_suppressed_arm_a"] for r in rows] == [1, 0]
        with open(os.path.join(out, "run_manifest.json")) as fh:
            man = json.load(fh)
        assert man["spec"] == m3.STAGE_SPEC
        assert man["prompt"]["caption_sha256"] == caption.sha256
        assert man["upstream"]["metadata_fingerprint"] == FP
        assert set(ARM_B_PHRASES) <= set(man["class_map"]["phrases_in_use"])
        assert "a car" in man["class_map"]["phrases_in_use"]
        # C25: the two arm B phrases are now reachable; barrier/cone etc. are not
        assert "a rickshaw" not in man["class_map"]["unreachable_phrases"]
        assert "a road barrier" in man["class_map"]["unreachable_phrases"]
        # Stage 4 refuses an upstream manifest without this key (masks.py:1811)
        assert man["image_size_px"] == [1600, 900]

    def test_degraded_upstream_needs_flag_and_degrades_output(self, tmp_path, caption, taxonomy):
        a, b, out = self._dirs(tmp_path, caption, degraded_a=True)
        assert m3.main(["--arm-a-dir", a, "--arm-b-dir", b, "--out-dir", out,
                        "--taxonomy", DHAKA]) == 2
        rc = m3.main(["--arm-a-dir", a, "--arm-b-dir", b, "--out-dir", out,
                      "--taxonomy", DHAKA, "--accept-degraded-upstream"])
        assert rc == 1
        assert os.path.isfile(os.path.join(out, "_SUCCESS.degraded"))

    def test_refuses_non_prefix_arm_a_caption(self, tmp_path, caption, taxonomy):
        a, b, out = self._dirs(tmp_path, caption)
        with open(os.path.join(a, "run_manifest.json")) as fh:
            man = json.load(fh)
        man["prompt"]["caption"] = "a completely different caption."
        write_json_atomic(os.path.join(a, "run_manifest.json"), man)
        assert m3.main(["--arm-a-dir", a, "--arm-b-dir", b, "--out-dir", out,
                        "--taxonomy", DHAKA]) == 2

    def test_refuses_scene_set_mismatch(self, tmp_path, caption, taxonomy):
        a, b, out = self._dirs(tmp_path, caption)
        os.rename(os.path.join(b, "scenes", "scene-0001"),
                  os.path.join(b, "scenes", "scene-0002"))
        assert m3.main(["--arm-a-dir", a, "--arm-b-dir", b, "--out-dir", out,
                        "--taxonomy", DHAKA]) == 2

    def test_refuses_arms_at_different_resolutions(self, tmp_path, caption, taxonomy):
        # Boxes from two resolutions are not comparable, and Stage 4 would
        # consume the merged tree under whichever number the manifest carried.
        a, b, out = self._dirs(tmp_path, caption, b_image_size_px=(1280, 720))
        assert m3.main(["--arm-a-dir", a, "--arm-b-dir", b, "--out-dir", out,
                        "--taxonomy", DHAKA, "--accept-degraded-upstream"]) == 2
