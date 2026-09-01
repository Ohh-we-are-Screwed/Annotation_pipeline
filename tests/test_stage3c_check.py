"""Tests for pipeline/stage3c_check/check.py.

Run: /home/mt/miniconda3/envs/ano_pipe/bin/python -m pytest tests/test_stage3c_check.py -v
"""

from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pipeline.common.manifest import UpstreamRefusal, read_marker, write_marker  # noqa: E402
from pipeline.stage3_proposals.proposals import build_caption, load_taxonomy  # noqa: E402
from pipeline.stage3c_check.check import (  # noqa: E402
    VERDICT_CONFIRMED,
    VERDICT_ERROR,
    VERDICT_RELABELED,
    VERDICT_SKIPPED_SMALL,
    VERDICT_UNCLEAR,
    apply_verdicts,
    crop_box,
    parse_vlm_reply,
    run,
    verdict_for,
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
        "image_size_px": [200, 100], "model_input_size_px": [1600, 928],
        "resize_policy": "letterbox",
        "checkpoint": {"model_id": "m", "revision": "r", "sha256": "s"},
        "prompt": {"caption_sha256": caption.sha256, "taxonomy_sha256": "t"},
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


class TestCropBox:
    def test_margin_expands_around_center_and_returns_ints(self):
        got = crop_box([100.0, 100.0, 200.0, 200.0], (1000, 1000), margin=1.5)
        assert got == [75, 75, 225, 225]
        assert all(isinstance(v, int) for v in got)

    def test_clamps_to_image_bounds(self):
        got = crop_box([-10.0, 5.0, 50.0, 400.0], (100, 300), margin=1.0)
        assert got == [0, 5, 50, 300]


class TestParseVlmReply:
    ALLOWED = ("a car", "a truck", "an auto rickshaw")

    def test_plain_json(self):
        assert parse_vlm_reply('{"label": "an auto rickshaw"}', self.ALLOWED) == "an auto rickshaw"

    def test_think_block_and_prose_around_json(self):
        text = '<think>hmm, three wheels</think>\nThe answer: {"label": "a truck"} done.'
        assert parse_vlm_reply(text, self.ALLOWED) == "a truck"

    def test_unclear_is_returned(self):
        assert parse_vlm_reply('{"label": "unclear"}', self.ALLOWED) == "unclear"

    def test_unknown_label_is_none(self):
        assert parse_vlm_reply('{"label": "a spaceship"}', self.ALLOWED) is None

    def test_garbage_is_none(self):
        assert parse_vlm_reply("no json here", self.ALLOWED) is None


class TestVerdictFor:
    ALLOWED = ("a car", "a truck", "an auto rickshaw")

    def test_same_phrase_confirms(self):
        assert verdict_for("a car", "a car", self.ALLOWED) == VERDICT_CONFIRMED

    def test_different_allowed_phrase_relabels(self):
        assert verdict_for("a car", "an auto rickshaw", self.ALLOWED) == VERDICT_RELABELED

    def test_unclear_keeps_label(self):
        assert verdict_for("a car", "unclear", self.ALLOWED) == VERDICT_UNCLEAR

    def test_unparseable_is_error(self):
        assert verdict_for("a car", None, self.ALLOWED) == VERDICT_ERROR


class TestApplyVerdicts:
    def test_relabel_rewrites_the_three_arrays_at_that_index_only(self, caption, taxonomy):
        row = _row(caption, ["a car", "a truck"], [[10, 10, 80, 80], [100, 20, 190, 90]])
        verdicts = [
            {"action": VERDICT_RELABELED, "vlm_phrase": "an auto rickshaw"},
            {"action": VERDICT_CONFIRMED, "vlm_phrase": "a truck"},
        ]
        out = apply_verdicts(row, verdicts, caption=caption, taxonomy=taxonomy)
        assert out["class_names"] == ["an auto rickshaw", "a truck"]
        assert out["nuscenes_categories"][0] == list(taxonomy.phrase_to_categories["an auto rickshaw"])
        assert out["phrase_char_spans"][0] == _span(caption, "an auto rickshaw")
        # index 1 untouched
        assert out["nuscenes_categories"][1] == row["nuscenes_categories"][1]
        assert out["phrase_char_spans"][1] == _span(caption, "a truck")
        # geometry, scores, order, count all unchanged
        assert out["boxes_xyxy_px"] == row["boxes_xyxy_px"]
        assert out["scores"] == row["scores"]
        assert out["n_proposals"] == 2
        # audit trail
        v = out["vlm_check"]["verdicts"]
        assert v[0]["action"] == VERDICT_RELABELED
        assert v[0]["original_class_name"] == "a car"
        assert v[1]["action"] == VERDICT_CONFIRMED
        # input row not mutated
        assert row["class_names"] == ["a car", "a truck"]

    def test_confirmed_row_arrays_unchanged(self, caption, taxonomy):
        row = _row(caption, ["a car"], [[10, 10, 80, 80]])
        out = apply_verdicts(
            row, [{"action": VERDICT_CONFIRMED, "vlm_phrase": "a car"}],
            caption=caption, taxonomy=taxonomy)
        assert out["class_names"] == row["class_names"]
        assert out["nuscenes_categories"] == row["nuscenes_categories"]
        assert out["phrase_char_spans"] == row["phrase_char_spans"]

    def test_stage3b_extension_arrays_ride_through(self, caption, taxonomy):
        row = _row(caption, ["a car"], [[10, 10, 80, 80]],
                   track_ids=[7], box_sources=["tracked"], n_propagated_hops=[2],
                   refined=[True], boxes_xyxy_px_original=[[9, 9, 79, 79]])
        out = apply_verdicts(
            row, [{"action": VERDICT_RELABELED, "vlm_phrase": "an auto rickshaw"}],
            caption=caption, taxonomy=taxonomy)
        assert out["track_ids"] == [7]
        assert out["box_sources"] == ["tracked"]
        assert out["boxes_xyxy_px_original"] == [[9, 9, 79, 79]]


class FakeVLM:
    """Scripted stand-in for the Nemotron server; records what it was asked."""

    def __init__(self, mapping=None, reply=None):
        self.mapping = mapping or {}
        self.reply = reply
        self.calls = []

    def __call__(self, crop, original_phrase, allowed_phrases):
        self.calls.append((crop.size, original_phrase))
        if self.reply is not None:
            return self.reply
        return json.dumps({"label": self.mapping.get(original_phrase, original_phrase)})


def _tree(tmp_path, caption, rows, *, degraded=False, caption_text=None):
    """A minimal upstream stage-3-shaped tree that require_upstream accepts."""
    d = tmp_path / "stage3_merged"
    scene = d / "scenes" / "chunk_0000"
    scene.mkdir(parents=True)
    with open(scene / "proposals.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    manifest = {
        "spec": "dhakascenes-pilot/stage3_merge/v1",
        "stage": "stage3_merge",
        "checkpoint": {"model_id": "m", "revision": "r", "sha256": "s"},
        "image_size_px": [200, 100],
        "prompt": {"caption": caption_text if caption_text is not None else caption.text,
                   "caption_sha256": caption.sha256},
        "class_map": {"phrases_in_use": ["a car", "a truck", "a pedestrian"]},
        "upstream": {"fingerprint_spec": "fspec/v1"},
    }
    with open(d / "run_manifest.json", "w") as fh:
        json.dump(manifest, fh)
    write_marker(str(d), "fp0", degraded=degraded,
                 causes=("chunk_0000: upstream sadness",) if degraded else ())
    return str(d)


def _dataroot(tmp_path):
    from PIL import Image
    root = tmp_path / "dataroot"
    (root / "img").mkdir(parents=True)
    Image.new("RGB", (200, 100), (90, 90, 90)).save(root / "img" / "0.jpg")
    return str(root)


class TestRun:
    def test_relabels_audits_and_carries_contract_keys(self, tmp_path, caption, taxonomy):
        rows = [_row(caption,
                     ["a car", "a truck", "a pedestrian"],
                     [[10, 10, 80, 80], [100, 20, 190, 90], [0, 0, 8, 8]])]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        vlm = FakeVLM(mapping={"a car": "an auto rickshaw"})
        rc = run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=vlm, min_side_px=24)
        assert rc == 0

        got = [json.loads(l) for l in open(os.path.join(out, "scenes", "chunk_0000", "proposals.jsonl"))]
        assert got[0]["class_names"] == ["an auto rickshaw", "a truck", "a pedestrian"]
        verdicts = got[0]["vlm_check"]["verdicts"]
        assert [v["action"] for v in verdicts] == [
            VERDICT_RELABELED, VERDICT_CONFIRMED, VERDICT_SKIPPED_SMALL]
        # the tiny box never reached the model
        assert len(vlm.calls) == 2

        man = json.load(open(os.path.join(out, "run_manifest.json")))
        assert man["image_size_px"] == [200, 100]          # Stage 4 refuses without it
        assert man["prompt"]["caption"] == caption.text    # class space unchanged
        assert man["totals"]["n_relabeled"] == 1
        assert man["totals"]["n_confirmed"] == 1
        assert man["totals"]["n_skipped_small"] == 1
        assert man["totals"]["confusion"] == {"a car -> an auto rickshaw": 1}
        assert "an auto rickshaw" in man["class_map"]["phrases_in_use"]

        marker = read_marker(out)
        assert marker is not None and not marker.degraded
        assert marker.fingerprint == "fp0"

    def test_refuses_caption_mismatch(self, tmp_path, caption, taxonomy):
        rows = [_row(caption, ["a car"], [[10, 10, 80, 80]])]
        src = _tree(tmp_path, caption, rows, caption_text="a car. a truck.")
        with pytest.raises(UpstreamRefusal):
            run(src, str(tmp_path / "out"), DHAKA,
                dataroot=_dataroot(tmp_path), vlm=FakeVLM())

    def test_refuses_degraded_upstream_without_flag(self, tmp_path, caption):
        rows = [_row(caption, ["a car"], [[10, 10, 80, 80]])]
        src = _tree(tmp_path, caption, rows, degraded=True)
        with pytest.raises(UpstreamRefusal):
            run(src, str(tmp_path / "out"), DHAKA,
                dataroot=_dataroot(tmp_path), vlm=FakeVLM())

    def test_accepted_degraded_upstream_propagates_causes(self, tmp_path, caption):
        rows = [_row(caption, ["a car"], [[10, 10, 80, 80]])]
        src = _tree(tmp_path, caption, rows, degraded=True)
        out = str(tmp_path / "stage3_checked")
        rc = run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=FakeVLM(),
                 accept_degraded=True)
        assert rc == 1
        marker = read_marker(out)
        assert marker.degraded
        assert any("upstream sadness" in c for c in marker.causes)

    def test_garbage_reply_keeps_label_and_degrades(self, tmp_path, caption):
        rows = [_row(caption, ["a car"], [[10, 10, 80, 80]])]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        rc = run(src, out, DHAKA, dataroot=_dataroot(tmp_path),
                 vlm=FakeVLM(reply="I refuse to answer in JSON"))
        assert rc == 1
        got = [json.loads(l) for l in open(os.path.join(out, "scenes", "chunk_0000", "proposals.jsonl"))]
        assert got[0]["class_names"] == ["a car"]
        assert got[0]["vlm_check"]["verdicts"][0]["action"] == VERDICT_ERROR
        marker = read_marker(out)
        assert marker.degraded
        assert any("vlm_errors" in c for c in marker.causes)


class TestParseVlmReplyArticles:
    ALLOWED = ("a car", "a truck", "an auto rickshaw")

    def test_missing_article_an(self):
        assert parse_vlm_reply('{"label": "auto rickshaw"}', self.ALLOWED) == "an auto rickshaw"

    def test_missing_article_a(self):
        assert parse_vlm_reply('{"label": "truck"}', self.ALLOWED) == "a truck"
