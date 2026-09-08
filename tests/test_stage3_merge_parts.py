"""C36 — rickshaw PARTS are not bicycles (2026-09-06).

Operator, from the chunk_0000 2D task: a rickshaw's front wheel comes back
from arm A as `a bicycle` (or `a motorcycle`) sitting INSIDE the arm B
rickshaw box, and survives the merge because C28/C34 arbitrate by IoU — a
wheel-sized box inside a rickshaw-sized box has an IoU near 0.15 however
completely it is contained. The rule the operator asked for: "if any bicycle
or motorcycle is over 60 % of a rickshaw, discard it" — read as the fraction
of the PART box's own area covered by a rickshaw / auto-rickshaw box
(containment), the quantity that is actually large in the picture.

Runs AFTER the C28/C34 passes, against arm B boxes that survived them, so a
confident protected bicycle that vetoed its rickshaw is never then absorbed
by it.
"""

from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.stage3_merge.merge import (  # noqa: E402
    PART_SUPPRESSION,
    merge_rows,
    parse_part_args,
)
from pipeline.stage3_proposals.proposals import build_caption, load_taxonomy  # noqa: E402

DHAKA = os.path.join(ROOT, "configs", "taxonomy_pilot_dhaka.yaml")


@pytest.fixture(scope="module")
def taxonomy():
    return load_taxonomy(DHAKA)


@pytest.fixture(scope="module")
def caption(taxonomy):
    return build_caption(taxonomy.phrases)


def _span(caption, phrase):
    return list(caption.phrase_char_spans[caption.phrases.index(phrase)])


def _row(caption, names, boxes, scores=None, **over):
    row = {
        "spec": "dhakascenes-pilot/stage3_proposals/v1",
        "keyframe_token": "kf0", "scene_token": "sc0", "t_ns": 1, "time_base": "utc",
        "coverage_config": "full", "channel": "CAM_FRONT",
        "sample_data_token": "sd0", "calibrated_sensor_token": "cs0",
        "ego_pose_token": "ep0", "dt_ns": 0, "image_path": "img/0.jpg",
        "image_size_px": [1280, 720], "model_input_size_px": [1280, 736],
        "resize_policy": "letterbox",
        "checkpoint": {"model_id": "m", "revision": "r", "sha256": "s"},
        "prompt": {"caption_sha256": "old", "taxonomy_sha256": "old", "span_map": None},
        "n_proposals": len(boxes),
        "score_aggregation": "yolo_class_confidence",
        "dedup": {"n_in": len(boxes), "n_out": len(boxes)},
        "boxes_xyxy_px": [list(b) for b in boxes],
        "scores": list(scores) if scores else [0.9] * len(boxes),
        "class_names": list(names),
        "nuscenes_categories": [["x"] for _ in names],
        "phrase_char_spans": [_span(caption, n) for n in names],
        "seed": 0,
    }
    row.update(over)
    return row


RICKSHAW = [100.0, 100.0, 300.0, 400.0]          # 200 x 300
WHEEL = [180.0, 300.0, 240.0, 390.0]             # 60 x 90, fully inside RICKSHAW  (IoU ~ 0.09)
HALF_OUT = [250.0, 300.0, 350.0, 390.0]          # 100 x 90, 50 % of its area inside RICKSHAW
FAR = [800.0, 100.0, 900.0, 200.0]


def _merge(caption, taxonomy, a, b, **kw):
    return merge_rows(a, b, caption=caption, taxonomy=taxonomy, iou_threshold=0.5, **kw)


class TestPartSuppression:
    def test_wheel_bicycle_inside_a_rickshaw_is_discarded(self, caption, taxonomy):
        a = _row(caption, ["a bicycle"], [WHEEL], scores=[0.3])
        b = _row(caption, ["a rickshaw"], [RICKSHAW])
        m = _merge(caption, taxonomy, a, b)
        assert m["class_names"] == ["a rickshaw"]
        led = m["merge"]["suppressed_parts"]
        assert len(led) == 1 and m["merge"]["n_suppressed_parts"] == 1
        assert led[0]["class_name"] == "a bicycle" and led[0]["box_xyxy_px"] == WHEEL
        assert led[0]["absorbed_by"] == 0 and led[0]["absorbed_class_name"] == "a rickshaw"
        assert led[0]["overlap"] == pytest.approx(1.0)

    def test_motorcycle_inside_an_auto_rickshaw_is_discarded(self, caption, taxonomy):
        a = _row(caption, ["a motorcycle"], [WHEEL], scores=[0.3])
        b = _row(caption, ["an auto rickshaw"], [RICKSHAW])
        m = _merge(caption, taxonomy, a, b)
        assert m["class_names"] == ["an auto rickshaw"]

    def test_half_overlap_is_below_the_floor_and_kept(self, caption, taxonomy):
        a = _row(caption, ["a bicycle"], [HALF_OUT], scores=[0.3])
        b = _row(caption, ["a rickshaw"], [RICKSHAW])
        m = _merge(caption, taxonomy, a, b)
        assert sorted(m["class_names"]) == ["a bicycle", "a rickshaw"]
        assert m["merge"]["n_suppressed_parts"] == 0

    def test_a_pedestrian_inside_a_rickshaw_is_not_a_part(self, caption, taxonomy):
        # the puller / passenger is a separate object (C28 KEEP_BOTH stands)
        a = _row(caption, ["a pedestrian"], [WHEEL])
        b = _row(caption, ["a rickshaw"], [RICKSHAW])
        m = _merge(caption, taxonomy, a, b)
        assert sorted(m["class_names"]) == ["a pedestrian", "a rickshaw"]

    def test_disabled_with_an_empty_table(self, caption, taxonomy):
        a = _row(caption, ["a bicycle"], [WHEEL], scores=[0.3])
        b = _row(caption, ["a rickshaw"], [RICKSHAW])
        m = _merge(caption, taxonomy, a, b, part_floor={})
        assert sorted(m["class_names"]) == ["a bicycle", "a rickshaw"]

    def test_a_confident_bicycle_that_vetoed_its_rickshaw_is_not_absorbed(self, caption, taxonomy):
        # C34: a bicycle at >= 0.40 contesting a rickshaw by IoU > 0.5 removes
        # the rickshaw. Nothing is left to absorb it — it must survive.
        same = [100.0, 100.0, 300.0, 400.0]
        a = _row(caption, ["a bicycle"], [same], scores=[0.9])
        b = _row(caption, ["a rickshaw"], [same])
        m = _merge(caption, taxonomy, a, b)
        assert m["class_names"] == ["a bicycle"]
        assert m["merge"]["n_suppressed_parts"] == 0 and m["merge"]["n_suppressed_arm_b"] == 1

    def test_ledger_indices_point_into_the_merged_arrays(self, caption, taxonomy):
        a = _row(caption, ["a car", "a bicycle"], [FAR, WHEEL], scores=[0.9, 0.3])
        b = _row(caption, ["a rickshaw"], [RICKSHAW])
        m = _merge(caption, taxonomy, a, b)
        assert m["class_names"] == ["a car", "a rickshaw"]
        (led,) = m["merge"]["suppressed_parts"]
        assert m["class_names"][led["absorbed_by"]] == "a rickshaw"
        assert led["index_in_arm_a"] == 1
        assert m["n_proposals"] == 2 and len(m["boxes_xyxy_px"]) == 2 and len(m["scores"]) == 2

    def test_default_table_is_the_operators_rule(self):
        assert PART_SUPPRESSION == {"a bicycle": 0.6, "a motorcycle": 0.6}


class TestParsePartArgs:
    def test_parses_phrase_and_floor(self):
        assert parse_part_args(["a bicycle:0.7", "a motorcycle:0.5"]) == {"a bicycle": 0.7, "a motorcycle": 0.5}

    def test_refuses_a_phrase_that_is_not_an_arm_a_part(self):
        with pytest.raises(ValueError, match="a rickshaw"):
            parse_part_args(["a rickshaw:0.6"])

    def test_refuses_a_floor_outside_unit_interval(self):
        with pytest.raises(ValueError):
            parse_part_args(["a bicycle:1.5"])


# The real chunk_0000 state that crashed Stage 3m on 2026-09-08 (scene
# dhaka_20260905_174950_chunk_0000, CAM_BACK, keyframe
# a2cc8778cb09e6152d6ac302e1c88d8e, arm A index 15), boxes verbatim: a
# confident bicycle VETOES the rickshaw it overlaps (IoU 0.536 > 0.5, C34) and
# is then ABSORBED as a part of a DIFFERENT, larger rickshaw that never
# contested it (containment 0.977, IoU 0.076 — far below the threshold, C36).
BIKE_15 = [301.2969970703125, 411.9620056152344, 382.3269958496094, 500.531005859375]
RICKSHAW_TIGHT = [293.5889892578125, 356.52301025390625, 386.1289978027344, 501.0769958496094]
RICKSHAW_BIG = [303.1830139160156, 372.1709899902344, 566.75, 720.0]


class TestProtectorRemovedAsPart:
    """C34 x C36: the protecting arm A box is itself absorbed as a part.

    `vetoed` names the arm A box that removed each arm B box; the ledger then
    reports where that box sits in the MERGED arrays. C36 runs afterwards and
    can take the protector out of those arrays, so the ledger must say so
    rather than index a box that is not there.
    """

    def test_absorbed_protector_does_not_crash_and_is_recorded(self, caption, taxonomy):
        a = _row(caption, ["a bicycle"], [BIKE_15], scores=[0.517])
        b = _row(caption, ["a rickshaw", "a rickshaw"],
                 [RICKSHAW_TIGHT, RICKSHAW_BIG], scores=[0.853, 0.90])
        m = _merge(caption, taxonomy, a, b)
        # The boxes are the arbitration's business and are unchanged: the tight
        # rickshaw is vetoed, the bicycle is absorbed, the big rickshaw stands.
        assert m["class_names"] == ["a rickshaw"]
        assert m["boxes_xyxy_px"] == [RICKSHAW_BIG]
        led = m["merge"]
        assert led["n_protected_arm_a"] == 1
        assert led["n_suppressed_arm_b"] == 1
        assert led["n_suppressed_parts"] == 1
        assert led["n_protected_arm_a_removed"] == 1
        (v,) = led["suppressed_arm_b"]
        assert v["index_in_arm_b"] == 0 and v["box_xyxy_px"] == RICKSHAW_TIGHT
        assert v["protected_index_in_arm_a"] == 0        # the ORIGINAL arm A index
        assert v["protected_by"] is None                 # not in the merged arrays
        assert v["protected_survived"] is False
        assert v["protected_removed_by"] == {
            "reason": "absorbed_as_part",
            "absorbed_by": 0,                            # index in MERGED arrays
            "absorbed_class_name": "a rickshaw",
            "overlap": pytest.approx(0.977, abs=1e-3),
        }
        assert m["class_names"][v["protected_removed_by"]["absorbed_by"]] == "a rickshaw"

    def test_surviving_protector_still_points_into_the_merged_arrays(self, caption, taxonomy):
        # arm A index 1 protects; arm A index 0 is a wheel absorbed by the
        # rickshaw, so the protector's merged index (0) and its arm A index (1)
        # differ — the ledger must carry both, and say the protector survived.
        bike_far = [800.0, 100.0, 900.0, 200.0]
        a = _row(caption, ["a bicycle", "a bicycle"], [WHEEL, bike_far], scores=[0.30, 0.90])
        b = _row(caption, ["a rickshaw", "a rickshaw"],
                 [RICKSHAW, [802.0, 101.0, 899.0, 198.0]])
        m = _merge(caption, taxonomy, a, b)
        assert m["class_names"] == ["a bicycle", "a rickshaw"]
        led = m["merge"]
        assert led["n_protected_arm_a"] == 1 and led["n_protected_arm_a_removed"] == 0
        (v,) = led["suppressed_arm_b"]
        assert v["index_in_arm_b"] == 1
        assert v["protected_index_in_arm_a"] == 1
        assert v["protected_by"] == 0
        assert m["class_names"][v["protected_by"]] == "a bicycle"
        assert v["protected_survived"] is True
        assert v["protected_removed_by"] is None

    def test_a_protector_is_never_suppressed_by_the_c28_table(self, caption, taxonomy):
        # The other half of the invariant the fix relies on: absorption is the
        # ONLY way a protector leaves the arrays, because the C28 pass skips
        # every protected box. A rickshaw straddling both bicycles is vetoed by
        # the confident one and suppresses nothing.
        a = _row(caption, ["a bicycle"], [[100.0, 100.0, 200.0, 200.0]], scores=[0.90])
        b = _row(caption, ["a rickshaw"], [[102.0, 101.0, 199.0, 198.0]])
        m = _merge(caption, taxonomy, a, b)
        assert m["merge"]["suppressed_arm_a"] == []
        assert m["merge"]["n_protected_arm_a_removed"] == 0
