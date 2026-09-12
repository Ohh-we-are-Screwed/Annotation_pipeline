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
    SOURCE_NONE,
    SOURCE_TRACK,
    SOURCE_VLM,
    VERDICT_CONFIRMED,
    VERDICT_ERROR,
    VERDICT_RELABELED,
    VERDICT_SKIPPED_SMALL,
    VERDICT_UNCLEAR,
    apply_verdicts,
    crop_box,
    parse_vlm_reply,
    plan_track_checks,
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


def _frame(caption, names, boxes, *, n, **over):
    """One row of a sequence, with the per-keyframe tokens actually distinct."""
    over.setdefault("keyframe_token", f"kf{n}")
    over.setdefault("sample_data_token", f"sd{n}")
    return _row(caption, names, boxes, **over)


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

    def test_parenthetical_gloss_is_stripped(self):
        text = '{"label": "a rickshaw (cycle rickshaw: pedal-driven passenger three-wheeler)"}'
        assert parse_vlm_reply(text, self.ALLOWED + ("a rickshaw",)) == "a rickshaw"

    def test_parenthetical_gloss_with_trailing_period(self):
        assert parse_vlm_reply(
            '{"label": "an auto rickshaw (CNG)."}', self.ALLOWED
        ) == "an auto rickshaw"

    def test_plain_label_unaffected_by_gloss_stripping(self):
        assert parse_vlm_reply('{"label": "a car"}', self.ALLOWED) == "a car"

    def test_unclear_unaffected_by_gloss_stripping(self):
        assert parse_vlm_reply('{"label": "unclear"}', self.ALLOWED) == "unclear"

    def test_reasoning_text_around_gloss_json_still_parses(self):
        text = ('<think>looks like a rickshaw</think>\n'
                'Answer: {"label": "a rickshaw (cycle rickshaw)"} done.')
        assert parse_vlm_reply(text, self.ALLOWED + ("a rickshaw",)) == "a rickshaw"

    def test_stray_close_paren_does_not_crash(self):
        # Malformed gloss with no opening paren: stripping is a no-op, and an
        # unknown label still resolves to None rather than raising.
        assert parse_vlm_reply('{"label": "a bus )"}', self.ALLOWED) is None


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


def _crop_id(crop):
    """FakeVLM's record of what it was shown: PIL size, or ('jpeg', n_bytes)."""
    return crop.size if hasattr(crop, "size") else ("jpeg", len(crop))


class FakeVLM:
    """Scripted stand-in for the Nemotron server; records what it was asked."""

    def __init__(self, mapping=None, reply=None):
        self.mapping = mapping or {}
        self.reply = reply
        self.calls = []

    def __call__(self, crop, original_phrase, allowed_phrases):
        self.calls.append((_crop_id(crop), original_phrase))
        if self.reply is not None:
            return self.reply
        return json.dumps({"label": self.mapping.get(original_phrase, original_phrase)})


class SequenceVLM:
    """Replies in call order (the last reply repeats); records crop kind."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, crop, original_phrase, allowed_phrases):
        self.calls.append(("bytes" if isinstance(crop, (bytes, bytearray)) else "pil",
                           original_phrase))
        return self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]


class BoomVLM:
    """Every ask raises — the audit must record the reason, not just the count."""

    def __init__(self):
        self.calls = []

    def __call__(self, crop, original_phrase, allowed_phrases):
        self.calls.append(original_phrase)
        raise RuntimeError("connection refused")


def _tree(tmp_path, caption, rows, *, degraded=False, caption_text=None, name="stage3_merged"):
    """A minimal upstream stage-3-shaped tree that require_upstream accepts.

    `rows` is a list (one scene, chunk_0000) or a {scene_name: rows} mapping.
    """
    d = tmp_path / name
    by_scene = rows if isinstance(rows, dict) else {"chunk_0000": rows}
    for scene_name, scene_rows in by_scene.items():
        scene = d / "scenes" / scene_name
        scene.mkdir(parents=True, exist_ok=True)
        with open(scene / "proposals.jsonl", "w") as fh:
            for r in scene_rows:
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


def _out_rows(out_dir, scene="chunk_0000"):
    path = os.path.join(out_dir, "scenes", scene, "proposals.jsonl")
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _manifest(out_dir):
    with open(os.path.join(out_dir, "run_manifest.json")) as fh:
        return json.load(fh)


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


# ---------------------------------------------------------------------------
# Track-aware mode. Fixtures use the REAL Stage 3b vocabulary: box_sources is
# "yolo" or "recovered" (never "tracked"), hops is 0 for a detection.
# ---------------------------------------------------------------------------


class TestPlanTrackChecks:
    """The pure planner: grouping, ranking, and the three untracked cases."""

    def test_ranks_detector_evidenced_untruncated_largest(self, caption):
        rows = [
            # tier 0 (yolo, hops 0) but the margin-expanded crop would clamp at x<0
            _frame(caption, ["a car"], [[0, 0, 70, 70]], n=0,
                   track_ids=[3], box_sources=["yolo"], n_propagated_hops=[0]),
            # biggest min side, but a propagated recovered box
            _frame(caption, ["a car"], [[10, 10, 90, 90]], n=1,
                   track_ids=[3], box_sources=["recovered"], n_propagated_hops=[1]),
            # smaller than both, but detector-evidenced AND fully inside
            _frame(caption, ["a car"], [[100, 20, 164, 84]], n=2,
                   track_ids=[3], box_sources=["yolo"], n_propagated_hops=[0]),
        ]
        plan = plan_track_checks(rows, min_side_px=24.0)
        key = ("CAM_FRONT", 3)
        assert plan.members[key] == [(0, 0), (1, 0), (2, 0)]
        assert plan.representatives[key] == [(2, 0), (0, 0), (1, 0)]
        assert plan.untracked == []
        assert plan.all_below_floor == set()

    def test_absent_track_ids_is_entirely_untracked(self, caption):
        plan = plan_track_checks([_row(caption, ["a car"], [[10, 10, 80, 80]])], min_side_px=24.0)
        assert plan.members == {}
        assert plan.untracked == [(0, 0)]

    def test_none_entry_falls_back_per_box(self, caption):
        rows = [_row(caption, ["a car", "a truck"], [[10, 10, 80, 80], [100, 20, 190, 90]],
                     track_ids=[5, None])]
        plan = plan_track_checks(rows, min_side_px=24.0)
        assert plan.members == {("CAM_FRONT", 5): [(0, 0)]}
        assert plan.untracked == [(0, 1)]

    def test_short_track_ids_is_an_upstream_refusal(self, caption):
        rows = [_row(caption, ["a car", "a truck"], [[10, 10, 80, 80], [100, 20, 190, 90]],
                     track_ids=[5])]
        with pytest.raises(UpstreamRefusal):
            plan_track_checks(rows, min_side_px=24.0)

    def test_track_with_no_member_above_the_floor(self, caption):
        rows = [_frame(caption, ["a car"], [[0, 0, 8, 8]], n=n, track_ids=[2]) for n in (0, 1)]
        plan = plan_track_checks(rows, min_side_px=24.0)
        assert plan.representatives == {}
        assert plan.all_below_floor == {("CAM_FRONT", 2)}
        assert len(plan.members[("CAM_FRONT", 2)]) == 2

    def test_scene_local_key_separates_channels(self, caption):
        rows = [
            _frame(caption, ["a car"], [[10, 10, 80, 80]], n=0, track_ids=[0]),
            _frame(caption, ["a car"], [[10, 10, 80, 80]], n=1, track_ids=[0],
                   channel="CAM_BACK"),
        ]
        plan = plan_track_checks(rows, min_side_px=24.0)
        assert set(plan.members) == {("CAM_FRONT", 0), ("CAM_BACK", 0)}


class TestPerTrackRun:
    def test_one_call_per_track_propagates_with_provenance(self, tmp_path, caption):
        rows = [
            _frame(caption, ["a car"], [[10, 10, 80, 80]], n=0, track_ids=[0]),
            _frame(caption, ["a car"], [[20, 15, 90, 85]], n=1, track_ids=[0]),
        ]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        vlm = FakeVLM(mapping={"a car": "an auto rickshaw"})
        rc = run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=vlm,
                 min_side_px=24, check_mode="per_track")
        assert rc == 0
        assert len(vlm.calls) == 1                     # 2 boxes, 1 ask

        got = _out_rows(out)
        assert [r["class_names"] for r in got] == [["an auto rickshaw"]] * 2
        first = got[0]["vlm_check"]["verdicts"][0]
        second = got[1]["vlm_check"]["verdicts"][0]
        assert first["verdict_source"] == SOURCE_VLM
        assert second["verdict_source"] == SOURCE_TRACK
        assert second["track_id"] == 0
        assert second["checked_at"] == {"keyframe_token": "kf0", "box_index": 0}
        assert second["n_members"] == 2
        assert got[0]["vlm_check"]["n_fresh_checks"] == 1
        assert got[1]["vlm_check"]["n_from_track_verdict"] == 1

        totals = _manifest(out)["totals"]
        assert totals["check_mode"] == "per_track"
        assert totals["n_vlm_calls"] == 1
        assert totals["n_tracks"] == 1
        assert totals["n_tracks_checked"] == 1
        assert totals["n_boxes_from_track_verdict"] == 1
        assert totals["n_boxes_untracked"] == 0
        assert totals["track_coverage"] == 1.0
        # per-box confusion is track-length-weighted; per-track counts decisions
        assert totals["confusion"] == {"a car -> an auto rickshaw": 2}
        assert totals["confusion_by_track"] == {"a car -> an auto rickshaw": 1}
        assert totals["max_relabel_track_members"] == 2
        assert totals["largest_relabel_track"]["track_id"] == 0
        vlm_block = _manifest(out)["vlm"]
        assert vlm_block["check_mode"] == "per_track"
        assert "scene-local" in vlm_block["track_key_definition"]
        assert "min side" in vlm_block["representative_ranking_rule"].lower()
        assert vlm_block["track_retry_candidates"] == 3

    def test_sub_floor_member_inherits_the_track_verdict(self, tmp_path, caption):
        rows = [
            _frame(caption, ["a car"], [[10, 10, 80, 80]], n=0, track_ids=[0]),
            _frame(caption, ["a car"], [[0, 0, 8, 8]], n=1, track_ids=[0]),
        ]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "per_track")
        vlm = FakeVLM(mapping={"a car": "an auto rickshaw"})
        run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=vlm,
            min_side_px=24, check_mode="per_track")
        assert len(vlm.calls) == 1
        got = _out_rows(out)
        small = got[1]["vlm_check"]["verdicts"][0]
        assert small["action"] == VERDICT_RELABELED
        assert small["propagated_over_small"] is True
        assert small["verdict_source"] == SOURCE_TRACK
        assert got[1]["class_names"] == ["an auto rickshaw"]

    def test_per_box_control_leaves_the_sub_floor_frame_flickering(self, tmp_path, caption):
        """The documented divergence: per_box is why one object carries two labels."""
        rows = [
            _frame(caption, ["a car"], [[10, 10, 80, 80]], n=0, track_ids=[0]),
            _frame(caption, ["a car"], [[0, 0, 8, 8]], n=1, track_ids=[0]),
        ]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "per_box")
        vlm = FakeVLM(mapping={"a car": "an auto rickshaw"})
        run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=vlm, min_side_px=24)
        got = _out_rows(out)
        assert got[0]["class_names"] == ["an auto rickshaw"]
        assert got[1]["class_names"] == ["a car"]          # same object, other label
        assert got[1]["vlm_check"]["verdicts"][0]["action"] == VERDICT_SKIPPED_SMALL

    def test_track_entirely_below_the_floor_costs_no_calls(self, tmp_path, caption):
        rows = [_frame(caption, ["a car"], [[0, 0, 8, 8]], n=n, track_ids=[0]) for n in (0, 1)]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        vlm = FakeVLM(mapping={"a car": "an auto rickshaw"})
        run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=vlm,
            min_side_px=24, check_mode="per_track")
        assert vlm.calls == []
        got = _out_rows(out)
        for r in got:
            rec = r["vlm_check"]["verdicts"][0]
            assert rec["action"] == VERDICT_SKIPPED_SMALL
            assert rec["verdict_source"] == SOURCE_NONE
            assert r["class_names"] == ["a car"]
        totals = _manifest(out)["totals"]
        assert totals["n_tracks"] == 1 and totals["n_tracks_checked"] == 0
        assert totals["n_vlm_calls"] == 0

    def test_absent_track_ids_is_identical_to_per_box(self, tmp_path, caption):
        rows = [
            _frame(caption, ["a car", "a pedestrian"], [[10, 10, 80, 80], [0, 0, 8, 8]], n=0),
            _frame(caption, ["a truck"], [[100, 20, 190, 90]], n=1),
        ]
        src = _tree(tmp_path, caption, rows)
        dataroot = _dataroot(tmp_path)
        box_out = str(tmp_path / "per_box")
        track_out = str(tmp_path / "per_track")
        v1 = FakeVLM(mapping={"a car": "an auto rickshaw"})
        v2 = FakeVLM(mapping={"a car": "an auto rickshaw"})
        run(src, box_out, DHAKA, dataroot=dataroot, vlm=v1, min_side_px=24)
        # allow_untracked is mandatory here: a tree with zero track ids is exactly
        # what the coverage preflight refuses (see the next test).
        run(src, track_out, DHAKA, dataroot=dataroot, vlm=v2, min_side_px=24,
            check_mode="per_track", allow_untracked=True)
        assert len(v1.calls) == len(v2.calls) == 2
        assert _out_rows(box_out) == _out_rows(track_out)
        assert _manifest(track_out)["totals"]["track_coverage"] == 0.0

    def test_per_track_refuses_a_tree_with_no_track_ids(self, tmp_path, caption):
        rows = [_row(caption, ["a car"], [[10, 10, 80, 80]])]
        src = _tree(tmp_path, caption, rows)
        with pytest.raises(UpstreamRefusal) as exc:
            run(src, str(tmp_path / "out"), DHAKA, dataroot=_dataroot(tmp_path),
                vlm=FakeVLM(), min_side_px=24, check_mode="per_track")
        assert "STALE" in str(exc.value)
        assert "--allow-untracked" in str(exc.value)

    def test_none_track_ids_are_asked_per_box(self, tmp_path, caption):
        rows = [
            _frame(caption, ["a car", "a truck"], [[10, 10, 80, 80], [100, 20, 190, 90]],
                   n=0, track_ids=[7, None]),
            _frame(caption, ["a car", "a truck"], [[12, 12, 82, 82], [100, 20, 190, 90]],
                   n=1, track_ids=[7, None]),
        ]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        vlm = FakeVLM(mapping={"a car": "an auto rickshaw"})
        run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=vlm,
            min_side_px=24, check_mode="per_track")
        assert len(vlm.calls) == 3          # one for track 7, one per arm-B box
        got = _out_rows(out)
        assert got[1]["vlm_check"]["verdicts"][0]["verdict_source"] == SOURCE_TRACK
        arm_b = got[1]["vlm_check"]["verdicts"][1]
        assert arm_b["verdict_source"] == SOURCE_VLM
        assert "track_id" not in arm_b
        totals = _manifest(out)["totals"]
        assert totals["n_boxes_untracked"] == 2
        assert totals["track_coverage"] == 0.5

    def test_same_id_on_two_channels_is_two_calls(self, tmp_path, caption):
        rows = [
            _frame(caption, ["a car"], [[10, 10, 80, 80]], n=0, track_ids=[0]),
            _frame(caption, ["a car"], [[10, 10, 80, 80]], n=1, track_ids=[0],
                   channel="CAM_BACK"),
        ]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        vlm = FakeVLM(mapping={"a car": "an auto rickshaw"})
        run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=vlm,
            min_side_px=24, check_mode="per_track")
        assert len(vlm.calls) == 2
        assert _manifest(out)["totals"]["n_tracks"] == 2

    def test_cache_is_scene_local(self, tmp_path, caption):
        rows = {
            "chunk_0000": [_frame(caption, ["a car"], [[10, 10, 80, 80]], n=0, track_ids=[0])],
            "chunk_0001": [_frame(caption, ["a car"], [[10, 10, 80, 80]], n=1, track_ids=[0])],
        }
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        vlm = FakeVLM(mapping={"a car": "an auto rickshaw"})
        run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=vlm,
            min_side_px=24, check_mode="per_track")
        assert len(vlm.calls) == 2          # id 0 in two scenes is two objects
        assert _manifest(out)["totals"]["n_tracks"] == 2

    def test_representative_is_the_ranked_crop(self, tmp_path, caption):
        rows = [
            _frame(caption, ["a car"], [[0, 0, 70, 70]], n=0,
                   track_ids=[3], box_sources=["yolo"], n_propagated_hops=[0]),
            _frame(caption, ["a car"], [[10, 10, 90, 90]], n=1,
                   track_ids=[3], box_sources=["recovered"], n_propagated_hops=[1]),
            _frame(caption, ["a car"], [[100, 20, 164, 84]], n=2,
                   track_ids=[3], box_sources=["yolo"], n_propagated_hops=[0]),
        ]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        vlm = FakeVLM(mapping={"a car": "an auto rickshaw"})
        run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=vlm,
            min_side_px=24, check_mode="per_track", track_retry_candidates=1)
        assert vlm.calls == [((74, 74), "a car")]   # the row-2 crop, not the biggest
        for r in _out_rows(out):
            assert r["class_names"] == ["an auto rickshaw"]
        assert _out_rows(out)[2]["vlm_check"]["verdicts"][0]["verdict_source"] == SOURCE_VLM

    def test_retry_ladder_reasks_a_held_jpeg_candidate(self, tmp_path, caption):
        rows = [_frame(caption, ["a car"], [[10 + n, 10, 80 + n, 80]], n=n, track_ids=[0])
                for n in (0, 1, 2)]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        vlm = SequenceVLM(['{"label": "unclear"}', '{"label": "an auto rickshaw"}'])
        rc = run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=vlm,
                 min_side_px=24, check_mode="per_track", track_retry_candidates=3)
        assert rc == 0
        assert [c[0] for c in vlm.calls] == ["pil", "bytes"]   # #1 live, #2 pre-encoded
        for r in _out_rows(out):
            assert r["class_names"] == ["an auto rickshaw"]
        totals = _manifest(out)["totals"]
        assert totals["n_track_retries"] == 1
        assert totals["n_vlm_calls"] == 2
        assert _manifest(out)["vlm"]["track_retry_held_max_crops"] == 2

    def test_a_failed_representative_amplifies_and_is_counted_per_track(self, tmp_path, caption):
        rows = [_frame(caption, ["a car"], [[10, 10, 80, 80]], n=n, track_ids=[0])
                for n in (0, 1, 2)]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        vlm = BoomVLM()
        rc = run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=vlm,
                 min_side_px=24, check_mode="per_track", track_retry_candidates=1)
        assert rc == 1
        recs = [r["vlm_check"]["verdicts"][0] for r in _out_rows(out)]
        assert [r["action"] for r in recs] == [VERDICT_ERROR] * 3
        assert all(r["error"] == "connection refused" for r in recs)
        totals = _manifest(out)["totals"]
        assert totals["n_errors"] == 3 and totals["n_error_tracks"] == 1
        marker = read_marker(out)
        assert any("amplified to 3 boxes" in c for c in marker.causes)

    def test_short_track_ids_refuses_before_any_call(self, tmp_path, caption):
        rows = [_row(caption, ["a car", "a truck"], [[10, 10, 80, 80], [100, 20, 190, 90]],
                     track_ids=[5])]
        src = _tree(tmp_path, caption, rows)
        vlm = FakeVLM()
        with pytest.raises(UpstreamRefusal):
            run(src, str(tmp_path / "out"), DHAKA, dataroot=_dataroot(tmp_path),
                vlm=vlm, min_side_px=24, check_mode="per_track")
        assert vlm.calls == []

    def test_equivalent_to_per_box_when_every_track_is_fully_checkable(self, tmp_path, caption):
        rows = [
            _frame(caption, ["a car"], [[10, 10, 80, 80]], n=0, track_ids=[0]),
            _frame(caption, ["a car"], [[10, 10, 80, 80]], n=1, track_ids=[0]),
            _frame(caption, ["a truck"], [[100, 20, 190, 90]], n=2, track_ids=[1]),
        ]
        src = _tree(tmp_path, caption, rows)
        dataroot = _dataroot(tmp_path)
        box_out = str(tmp_path / "per_box")
        track_out = str(tmp_path / "per_track")
        mapping = {"a car": "an auto rickshaw"}
        run(src, box_out, DHAKA, dataroot=dataroot, vlm=FakeVLM(mapping=mapping), min_side_px=24)
        run(src, track_out, DHAKA, dataroot=dataroot, vlm=FakeVLM(mapping=mapping),
            min_side_px=24, check_mode="per_track")
        for a, b in zip(_out_rows(box_out), _out_rows(track_out)):
            a = {k: v for k, v in a.items() if k != "vlm_check"}
            b = {k: v for k, v in b.items() if k != "vlm_check"}
            assert a == b
        assert _manifest(box_out)["totals"]["n_vlm_calls"] == 3
        assert _manifest(track_out)["totals"]["n_vlm_calls"] == 2


class TestPerBoxAdditiveKeyDelta:
    """The EXACT row/record delta a per_box run gains. min_side_px is explicit:
    the 24 -> 32 default realignment would otherwise move the baseline."""

    BASE_BLOCK_KEYS = {"spec", "verdicts", "n_relabeled", "n_confirmed", "n_unclear",
                       "n_skipped_small", "n_skipped_class", "n_errors"}
    PER_TRACK_TOTALS = ("n_tracks", "n_tracks_checked", "n_boxes_from_track_verdict",
                        "n_boxes_untracked", "track_coverage", "n_track_retries",
                        "n_error_tracks", "confusion_by_track", "max_relabel_track_members",
                        "largest_relabel_track")

    def test_records_gain_only_verdict_source(self, tmp_path, caption):
        rows = [_row(caption, ["a car", "a truck", "a pedestrian"],
                     [[10, 10, 80, 80], [100, 20, 190, 90], [0, 0, 8, 8]])]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        run(src, out, DHAKA, dataroot=_dataroot(tmp_path),
            vlm=FakeVLM(mapping={"a car": "an auto rickshaw"}), min_side_px=32.0)

        block = _out_rows(out)[0]["vlm_check"]
        assert set(block) == self.BASE_BLOCK_KEYS | {"n_fresh_checks"}
        assert block["n_fresh_checks"] == 2
        relabeled, confirmed, skipped = block["verdicts"]
        assert set(relabeled) == {"action", "vlm_phrase", "verdict_source",
                                  "original_class_name", "original_nuscenes_categories",
                                  "original_phrase_char_spans"}
        assert set(confirmed) == {"action", "vlm_phrase", "verdict_source"}
        assert set(skipped) == {"action", "vlm_phrase", "verdict_source"}
        assert [r["verdict_source"] for r in block["verdicts"]] == [
            SOURCE_VLM, SOURCE_VLM, SOURCE_NONE]

    def test_manifest_gains_only_the_named_keys(self, tmp_path, caption):
        rows = [_row(caption, ["a car", "a truck", "a pedestrian"],
                     [[10, 10, 80, 80], [100, 20, 190, 90], [0, 0, 8, 8]])]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        run(src, out, DHAKA, dataroot=_dataroot(tmp_path),
            vlm=FakeVLM(mapping={"a car": "an auto rickshaw"}), min_side_px=32.0)
        man = _manifest(out)

        assert man["totals"]["n_vlm_calls"] == 2
        assert man["totals"]["check_mode"] == "per_box"
        for key in self.PER_TRACK_TOTALS:
            assert key not in man["totals"]
        assert man["vlm"]["check_mode"] == "per_box"
        for key in ("track_key_definition", "representative_ranking_rule",
                    "track_retry_candidates", "track_accounting_notes"):
            assert key not in man["vlm"]
        # the out-dir fence block, and no partial flag on a complete run
        assert man["scenes_written"] == ["chunk_0000"]
        assert man["scenes_in_out_dir"] == ["chunk_0000"]
        assert man["foreign_scene_dirs"] == []
        assert man["config_matched_previous"] is None
        assert "partial" not in man
        # the llama-server log now lives under work_root/logs/, not in out_dir
        assert "llama_server.log" not in os.listdir(out)

    def test_the_error_string_reaches_the_audit_record(self, tmp_path, caption):
        rows = [_row(caption, ["a car"], [[10, 10, 80, 80]])]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        rc = run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=BoomVLM(), min_side_px=32.0)
        assert rc == 1
        rec = _out_rows(out)[0]["vlm_check"]["verdicts"][0]
        assert rec["action"] == VERDICT_ERROR
        assert rec["error"] == "connection refused"
        assert rec["verdict_source"] == SOURCE_VLM


class TestOutDirFenceAndMaxRows:
    def _two_scene_tree(self, tmp_path, caption):
        return _tree(tmp_path, caption, {
            "chunk_0000": [_frame(caption, ["a car"], [[10, 10, 80, 80]], n=0)],
            "chunk_0001": [_frame(caption, ["a truck"], [[100, 20, 190, 90]], n=1)],
        })

    def test_subset_rerun_under_another_config_is_refused(self, tmp_path, caption):
        src = self._two_scene_tree(tmp_path, caption)
        dataroot = _dataroot(tmp_path)
        out = str(tmp_path / "stage3_checked")
        run(src, out, DHAKA, dataroot=dataroot, vlm=FakeVLM(), min_side_px=24,
            scenes=["chunk_0000"])
        with pytest.raises(UpstreamRefusal) as exc:
            run(src, out, DHAKA, dataroot=dataroot, vlm=FakeVLM(), min_side_px=48,
                scenes=["chunk_0001"])
        assert "chunk_0000" in str(exc.value)

    def test_subset_rerun_under_the_identical_config_is_allowed(self, tmp_path, caption):
        src = self._two_scene_tree(tmp_path, caption)
        dataroot = _dataroot(tmp_path)
        out = str(tmp_path / "stage3_checked")
        run(src, out, DHAKA, dataroot=dataroot, vlm=FakeVLM(), min_side_px=24,
            scenes=["chunk_0000"])
        run(src, out, DHAKA, dataroot=dataroot, vlm=FakeVLM(), min_side_px=24,
            scenes=["chunk_0001"])
        man = _manifest(out)
        assert man["scenes_written"] == ["chunk_0001"]
        assert man["scenes_in_out_dir"] == ["chunk_0000", "chunk_0001"]
        assert man["foreign_scene_dirs"] == ["chunk_0000"]
        assert man["config_matched_previous"] is True

    def test_unknown_scene_name_is_refused(self, tmp_path, caption):
        src = self._two_scene_tree(tmp_path, caption)
        with pytest.raises(UpstreamRefusal):
            run(src, str(tmp_path / "out"), DHAKA, dataroot=_dataroot(tmp_path),
                vlm=FakeVLM(), min_side_px=24, scenes=["chunk_9999"])

    def test_max_rows_truncates_and_writes_no_marker(self, tmp_path, caption):
        rows = [_frame(caption, ["a car"], [[10, 10, 80, 80]], n=n) for n in (0, 1, 2)]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        vlm = FakeVLM()
        rc = run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=vlm,
                 min_side_px=24, max_rows=2)
        assert rc == 0
        assert len(_out_rows(out)) == 2
        assert len(vlm.calls) == 2
        man = _manifest(out)
        assert man["partial"] is True and man["max_rows"] == 2
        # marker_state == none: no selector can ever adopt a partial tree
        assert read_marker(out) is None

    def test_max_rows_refuses_a_non_empty_out_dir(self, tmp_path, caption):
        rows = [_frame(caption, ["a car"], [[10, 10, 80, 80]], n=n) for n in (0, 1)]
        src = _tree(tmp_path, caption, rows)
        dataroot = _dataroot(tmp_path)
        out = str(tmp_path / "stage3_checked")
        run(src, out, DHAKA, dataroot=dataroot, vlm=FakeVLM(), min_side_px=24)
        with pytest.raises(UpstreamRefusal) as exc:
            run(src, out, DHAKA, dataroot=dataroot, vlm=FakeVLM(), min_side_px=24, max_rows=1)
        assert "--max-rows" in str(exc.value)


class TestSkipClasses:
    """--skip-class (C31): boxes of a skipped class are never asked about.

    The VLM systematically flips rider crops against its own prompt rule
    (pedestrian -> motorcycle, 2706 boxes in the first pilot_1632 run, zero in
    reverse), so whole classes can be excluded from checking. A skipped box
    keeps its label and gets an honest `skipped_class` verdict: no call, no
    propagation, verdict_source none.
    """

    def test_per_track_skipped_class_costs_no_calls_and_keeps_labels(self, tmp_path, caption):
        rows = [
            _frame(caption, ["a pedestrian", "a car"],
                   [[10, 10, 80, 80], [100, 20, 190, 90]], n=0, track_ids=[0, 1]),
            _frame(caption, ["a pedestrian", "a car"],
                   [[12, 10, 82, 80], [102, 20, 192, 90]], n=1, track_ids=[0, 1]),
        ]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        vlm = FakeVLM(mapping={"a car": "an auto rickshaw"})
        rc = run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=vlm,
                 min_side_px=24, check_mode="per_track",
                 skip_classes=("a pedestrian",))
        assert rc == 0
        # one ask for the car track; the pedestrian track is never asked about
        assert [orig for (_, orig) in vlm.calls] == ["a car"]

        got = _out_rows(out)
        assert [r["class_names"] for r in got] == [["a pedestrian", "an auto rickshaw"]] * 2
        ped = got[0]["vlm_check"]["verdicts"][0]
        assert ped["action"] == "skipped_class"
        assert ped["vlm_phrase"] is None
        assert ped["verdict_source"] == SOURCE_NONE
        assert ped["track_id"] == 0
        assert got[0]["vlm_check"]["n_skipped_class"] == 1

        man = _manifest(out)
        assert man["vlm"]["skip_classes"] == ["a pedestrian"]
        totals = man["totals"]
        assert totals["n_skipped_class"] == 2
        assert totals["n_checked"] == 2          # the two car boxes only
        assert totals["n_tracks"] == 2
        assert totals["n_tracks_checked"] == 1   # the pedestrian track never got a verdict

    def test_per_box_skipped_class_costs_no_call(self, tmp_path, caption):
        rows = [_row(caption, ["a motorcycle", "a car"],
                     [[10, 10, 80, 80], [100, 20, 190, 90]])]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        vlm = FakeVLM(mapping={"a car": "a truck"})
        rc = run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=vlm,
                 min_side_px=24, skip_classes=("a motorcycle",))
        assert rc == 0
        assert [orig for (_, orig) in vlm.calls] == ["a car"]
        got = _out_rows(out)
        assert got[0]["class_names"] == ["a motorcycle", "a truck"]
        assert got[0]["vlm_check"]["verdicts"][0]["action"] == "skipped_class"

    def test_skip_class_beats_the_size_floor(self, tmp_path, caption):
        # a sub-floor box of a skipped class reads skipped_class, not
        # skipped_small: the class policy, not the crop size, is why no model
        # ever saw it
        rows = [_row(caption, ["a pedestrian"], [[0, 0, 8, 8]])]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=FakeVLM(),
            min_side_px=24, skip_classes=("a pedestrian",))
        blk = _out_rows(out)[0]["vlm_check"]
        assert blk["verdicts"][0]["action"] == "skipped_class"
        assert blk["n_skipped_class"] == 1
        assert blk["n_skipped_small"] == 0

    def test_unknown_skip_class_is_refused_before_any_write(self, tmp_path, caption):
        rows = [_row(caption, ["a car"], [[10, 10, 80, 80]])]
        src = _tree(tmp_path, caption, rows)
        out = str(tmp_path / "stage3_checked")
        with pytest.raises(RuntimeError) as exc:
            run(src, out, DHAKA, dataroot=_dataroot(tmp_path), vlm=FakeVLM(),
                min_side_px=24, skip_classes=("a spaceship",))
        assert "a spaceship" in str(exc.value)
        assert not os.path.exists(os.path.join(out, "scenes"))

    def test_apply_verdicts_counts_skipped_class(self, caption, taxonomy):
        row = _row(caption, ["a pedestrian"], [[10, 10, 80, 80]])
        checked = apply_verdicts(
            row,
            [{"action": "skipped_class", "vlm_phrase": None,
              "verdict_source": SOURCE_NONE}],
            caption=caption, taxonomy=taxonomy)
        blk = checked["vlm_check"]
        assert blk["n_skipped_class"] == 1
        assert blk["n_confirmed"] == 0
        assert checked["class_names"] == ["a pedestrian"]
