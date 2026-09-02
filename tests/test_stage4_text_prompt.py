"""Tests for Stage 4's text-prompt provider `sam3_text` (C29) and the D.1 fix.

Run: /home/mt/miniconda3/envs/ano_pipe/bin/python -m pytest tests/test_stage4_text_prompt.py -v

NO GPU, no weights, no model imports at module scope. `masks.py` imports torch
nowhere at module level — every adapter's `load()` imports it locally — so the
module is safe to import here; what is never done is CONSTRUCT-AND-LOAD. The one
adapter that is constructed (`Sam3TextAdapter`) is constructed only: its
`__init__` sets attributes and imports nothing, and its heads are replaced with
fakes before any call. `TransformersSamAdapter` is reached through
`__new__` for the wrong-rank test, which never runs `__init__` either.

What each group is for:
  * `select_text_masks` owns the bridge between an instance-detection output
    (cardinality decided by a score threshold) and this stage's
    one-mask-per-box-in-order contract. A permuted or duplicated assignment is
    undetectable everywhere downstream, so it is tested as a pure function.
  * the scatter-back test is the one that would catch a phrase-grouped return
    being handed back as if it were box-ordered.
  * the D.1 test pins a crash: `int(None)` on a merged-over-3b tree.
  * the wrong-rank test moves conformance 5.5-r1 off VIOLATES.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pipeline.common.conventions import CAMERA, EGO  # noqa: E402
from pipeline.common.manifest import UpstreamRefusal  # noqa: E402
from pipeline.common.model_interfaces import (  # noqa: E402
    MASK_2D,
    CheckpointSpec,
    Mask2D,
    MaskResult,
    PER_FRAME_WINDOW,
    ROLE_METHODS,
    TemporalWindow,
)
from pipeline.stage4_masks import masks as masks_module  # noqa: E402
from pipeline.stage4_masks.masks import (  # noqa: E402
    BOX_FALLBACK,
    TEXT_MATCHED,
    MaskConfig,
    MaskContractError,
    Sam3TextAdapter,
    TransformersSamAdapter,
    _MASK_ADAPTERS,
    _PROVIDER_PROVENANCE,
    box_iou,
    candidate_rows,
    check_detector_dtype_name,
    count_cross_phrase_mask_overlaps,
    count_text_duplicate_rejections,
    main,
    mask_shape_desc,
    preflight_mask_adapter,
    process_keyframe,
    refuse_text_prompt_provider,
    required_class_names,
    resolve_text_prompt,
    run,
    select_text_masks,
    strip_leading_article,
    text_prompt_manifest_block,
)

W, H = 1280, 720

# The five MaskConfig fields C29 adds, with the values the spike decided. Named
# once so the wiring test and the default test cannot drift apart.
NEW_CONFIG_FIELDS = {
    "text_prompt": False,
    "text_prompt_strip_article": True,
    "text_match_min_iou": 0.5,
    "text_score_threshold": 0.3,
    "text_detector_dtype": "bfloat16",
}


def rect(x1: int, y1: int, x2: int, y2: int, *, width: int, height: int) -> np.ndarray:
    """A bool mask that is True exactly inside [x1,x2) x [y1,y2)."""
    mask = np.zeros((height, width), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


# ---------------------------------------------------------------------------
# select_text_masks — the assignment rule
# ---------------------------------------------------------------------------


class TestSelectTextMasks:
    def test_each_box_takes_its_highest_iou_instance(self):
        boxes = [(0, 0, 10, 10), (100, 100, 110, 110)]
        instances = [(100, 100, 110, 110), (0, 0, 10, 10)]
        assert select_text_masks(instances, boxes, 0.5) == [1, 0]

    def test_below_the_floor_is_not_a_match(self):
        boxes = [(0, 0, 10, 10)]
        # IoU 0.25: half the width, half the height overlapping.
        instances = [(5, 5, 15, 15)]
        assert select_text_masks(instances, boxes, 0.5) == [None]
        # ... and the SAME pair matches once the floor drops below it.
        assert select_text_masks(instances, boxes, 0.1) == [0]

    def test_the_floor_is_inclusive(self):
        boxes = [(0, 0, 10, 10)]
        instances = [(0, 0, 10, 10)]
        assert select_text_masks(instances, boxes, 1.0) == [0]

    def test_two_boxes_one_instance_is_assigned_once(self):
        # The case a per-box argmax gets wrong: both boxes want instance 0, and
        # handing it to both would duplicate a mask. In-channel pairs are exempt
        # from IoA-NMS, so both duplicates would survive to Stage 5.
        boxes = [(0, 0, 10, 10), (1, 1, 11, 11)]
        instances = [(0, 0, 10, 10)]
        assignment = select_text_masks(instances, boxes, 0.5)
        assert assignment == [0, None]
        assert sum(1 for a in assignment if a == 0) == 1

    def test_the_better_iou_wins_the_contested_instance(self):
        # Order of the boxes must not decide it — the IoU must.
        boxes = [(2, 2, 12, 12), (0, 0, 10, 10)]
        instances = [(0, 0, 10, 10)]
        assert select_text_masks(instances, boxes, 0.5) == [None, 0]

    def test_tie_break_is_deterministic_and_repeatable(self):
        # Every pair scores IoU 1.0, so only the tie-break decides: lower box
        # index first, then lower instance index.
        boxes = [(0, 0, 10, 10), (0, 0, 10, 10)]
        instances = [(0, 0, 10, 10), (0, 0, 10, 10)]
        first = select_text_masks(instances, boxes, 0.5)
        assert first == [0, 1]
        assert all(select_text_masks(instances, boxes, 0.5) == first for _ in range(5))

    def test_one_box_two_equal_instances_takes_the_lower_index(self):
        boxes = [(0, 0, 10, 10)]
        instances = [(0, 0, 10, 10), (0, 0, 10, 10)]
        assert select_text_masks(instances, boxes, 0.5) == [0]

    def test_empty_instance_set_falls_back_everywhere(self):
        boxes = [(0, 0, 10, 10), (20, 20, 30, 30)]
        assert select_text_masks([], boxes, 0.5) == [None, None]

    def test_empty_box_set_returns_empty(self):
        assert select_text_masks([(0, 0, 10, 10)], [], 0.5) == []

    def test_none_instances_are_skipped(self):
        # An instance whose mask had no pixels has no tight box.
        boxes = [(0, 0, 10, 10)]
        assert select_text_masks([None, (0, 0, 10, 10)], boxes, 0.5) == [1]

    def test_disjoint_boxes_never_match(self):
        boxes = [(0, 0, 10, 10)]
        assert select_text_masks([(500, 500, 510, 510)], boxes, 0.5) == [None]

    def test_box_iou_is_symmetric_and_bounded(self):
        a, b = (0, 0, 10, 10), (5, 0, 15, 10)
        assert box_iou(a, b) == pytest.approx(box_iou(b, a))
        assert box_iou(a, a) == pytest.approx(1.0)
        assert box_iou(a, (100, 100, 110, 110)) == 0.0

    def test_duplicate_rejections_are_counted_only_when_the_instance_was_taken(self):
        boxes = [(0, 0, 10, 10), (1, 1, 11, 11)]
        instances = [(0, 0, 10, 10)]
        assignment = select_text_masks(instances, boxes, 0.5)
        assert count_text_duplicate_rejections(instances, boxes, 0.5, assignment) == 1
        # A box that simply had nothing above the floor is NOT a duplicate
        # rejection — that distinction is the whole point of the counter.
        lonely = [(0, 0, 10, 10), (500, 500, 510, 510)]
        lonely_assignment = select_text_masks(instances, lonely, 0.5)
        assert lonely_assignment == [0, None]
        assert count_text_duplicate_rejections(instances, lonely, 0.5, lonely_assignment) == 0


# ---------------------------------------------------------------------------
# strip-article / prompt resolution
# ---------------------------------------------------------------------------


class TestStripArticle:
    @pytest.mark.parametrize(
        "phrase,expected",
        [
            ("a car", "car"),
            ("an auto rickshaw", "auto rickshaw"),
            ("the road", "road"),
            ("a pedestrian", "pedestrian"),
            ("rickshaw", "rickshaw"),
            ("A Car", "Car"),
            ("THE ROAD", "ROAD"),
        ],
    )
    def test_the_three_articles_are_stripped_case_insensitively(self, phrase, expected):
        assert strip_leading_article(phrase) == expected

    def test_a_bare_noun_keeps_every_token(self):
        # proposals.py's bare-noun fallback, imitated.
        assert strip_leading_article("bus") == "bus"

    def test_a_phrase_that_is_only_an_article_is_returned_verbatim(self):
        assert strip_leading_article("a ") == "a"
        assert strip_leading_article("the ") == "the"

    def test_an_internal_article_is_not_touched(self):
        assert strip_leading_article("man in a hat") == "man in a hat"

    def test_a_word_starting_with_a_is_not_an_article(self):
        assert strip_leading_article("ambulance") == "ambulance"

    def test_resolve_honours_the_toggle_in_both_directions(self):
        assert resolve_text_prompt("a car", strip_article=True) == "car"
        assert resolve_text_prompt("a car", strip_article=False) == "a car"

    def test_resolve_never_reaches_the_model_empty(self):
        # The guard that matters: Sam3Processor substitutes the literal string
        # "visual" for a missing text and silently runs the exemplar path.
        assert resolve_text_prompt("a ", strip_article=True) == "a"


# ---------------------------------------------------------------------------
# The adapter: scatter-back order, fallback, counters
# ---------------------------------------------------------------------------


class FakeTracker:
    """Stands in for the loaded TransformersSamAdapter that owns the box path."""

    def __init__(self, width: int, height: int) -> None:
        self.calls: list[list[list[float]]] = []
        self._width, self._height = width, height

    def _segment_single(self, image, boxes) -> np.ndarray:
        self.calls.append([[float(v) for v in b] for b in np.asarray(boxes)])
        out = np.zeros((len(boxes), self._height, self._width), dtype=bool)
        for i, box in enumerate(np.asarray(boxes)):
            x1, y1, x2, y2 = (int(v) for v in box)
            out[i, y1:y2, x1:x2] = True
            # A marker no text-matched mask carries, so the two sources are
            # distinguishable by pixels and not only by the reported kind.
            out[i, 0, 0] = True
        return out


def text_adapter(cfg: MaskConfig) -> tuple[Sam3TextAdapter, FakeTracker]:
    """A constructed-but-never-loaded adapter with both heads faked."""
    spec = CheckpointSpec(
        role=MASK_2D, provider="sam3_text", model_id="facebook/sam3", revision="deadbeef"
    )
    adapter = Sam3TextAdapter(spec, cfg)
    tracker = FakeTracker(cfg.image_width_px, cfg.image_height_px)
    adapter._detector = object()
    adapter._processor = object()
    adapter._tracker = tracker
    return adapter, tracker


class TestScatterBack:
    """mask[i] belongs to box[i], across phrase groups and a fallback."""

    # Small frame: MaskConfig owns the resolution the adapter asserts, so the
    # geometry stays legible and the arrays stay small. The order contract under
    # test is resolution-independent.
    CFG = MaskConfig(text_prompt=True, image_width_px=200, image_height_px=100)

    # index -> (phrase, box). Two distinct phrases, interleaved, so a
    # phrase-grouped return would be visibly permuted; box 1 matches nothing.
    LAYOUT = [
        ("a car", (10, 10, 30, 30)),
        ("a bus", (50, 10, 70, 30)),      # -> box_fallback
        ("a car", (90, 10, 110, 30)),
        ("a bus", (130, 10, 150, 30)),
        ("a car", (170, 50, 190, 70)),
    ]

    def _adapter(self):
        adapter, tracker = text_adapter(self.CFG)

        def fake_detect(image, prompt, channel):
            # Returned deliberately OUT of box order, and only for boxes whose
            # phrase resolves to this prompt. "bus" returns nothing for box 1 —
            # the spike's measured failure class, reproduced.
            if prompt == "car":
                order = [4, 0, 2]
            elif prompt == "bus":
                order = [3]
            else:  # pragma: no cover - the layout has two phrases
                order = []
            masks, boxes = [], []
            for index in order:
                x1, y1, x2, y2 = self.LAYOUT[index][1]
                masks.append(rect(x1, y1, x2, y2,
                                  width=self.CFG.image_width_px,
                                  height=self.CFG.image_height_px))
                boxes.append((float(x1), float(y1), float(x2), float(y2)))
            adapter.text_counts["n_detector_forwards"] += 1
            return masks, boxes

        adapter._detect_text = fake_detect
        return adapter, tracker

    def _segment(self):
        adapter, tracker = self._adapter()
        image = np.zeros((self.CFG.image_height_px, self.CFG.image_width_px, 3), dtype=np.uint8)
        boxes = np.asarray([b for _n, b in self.LAYOUT], dtype=np.float32)
        result = adapter.segment(
            [image], boxes, channel="CAM_FRONT", class_names=[n for n, _b in self.LAYOUT]
        )
        return adapter, tracker, result

    def test_every_mask_lands_on_its_own_box(self):
        _adapter, _tracker, result = self._segment()
        assert result.masks.shape == (5, self.CFG.image_height_px, self.CFG.image_width_px)
        for index, (_phrase, (x1, y1, x2, y2)) in enumerate(self.LAYOUT):
            mask = result.masks[index]
            expected = rect(x1, y1, x2, y2,
                            width=self.CFG.image_width_px, height=self.CFG.image_height_px)
            if index == 1:
                # The fallback mask carries the marker pixel as well.
                expected = expected.copy()
                expected[0, 0] = True
            assert np.array_equal(mask, expected), f"mask {index} is not box {index}'s"

    def test_the_prompt_kind_is_recorded_per_box(self):
        adapter, _tracker, _result = self._segment()
        assert adapter.last_mask_prompt == [
            TEXT_MATCHED, BOX_FALLBACK, TEXT_MATCHED, TEXT_MATCHED, TEXT_MATCHED
        ]

    def test_only_the_unmatched_box_reaches_the_box_prompt(self):
        _adapter, tracker, _result = self._segment()
        assert tracker.calls == [[[50.0, 10.0, 70.0, 30.0]]]

    def test_counts_add_up_and_are_recorded_per_phrase(self):
        adapter, _tracker, _result = self._segment()
        counts = adapter.text_counts
        assert counts["n_text_matched"] == 4
        assert counts["n_box_fallback"] == 1
        assert counts["n_detector_forwards"] == 2  # one per DISTINCT phrase
        per_phrase = counts["per_phrase"]
        assert set(per_phrase) == {"car", "bus"}
        assert per_phrase["car"]["n_text_matched"] == 3
        assert per_phrase["car"]["n_box_fallback"] == 0
        # The number that must be visible per phrase: a class that NEVER
        # text-matches is 100% fallback and says so.
        assert per_phrase["bus"]["n_text_matched"] == 1
        assert per_phrase["bus"]["n_box_fallback"] == 1
        assert per_phrase["bus"]["class_names"] == ["a bus"]

    def test_the_resolved_prompt_of_every_class_is_recorded(self):
        adapter, _tracker, _result = self._segment()
        assert adapter.text_prompts == {"a car": "car", "a bus": "bus"}

    def test_no_cross_phrase_overlap_on_disjoint_geometry(self):
        adapter, _tracker, _result = self._segment()
        assert adapter.text_counts["n_cross_phrase_mask_overlap"] == 0

    def test_an_all_fallback_frame_makes_the_identical_box_prompt_call(self):
        # The strong form of "byte-identical fallback": when the detector matches
        # nothing, the tracker head is handed exactly the caller's box array, in
        # exactly the caller's order — the call a sam3_tracker run would make.
        adapter, tracker = self._adapter()
        adapter._detect_text = lambda image, prompt, channel: ([], [])
        image = np.zeros((self.CFG.image_height_px, self.CFG.image_width_px, 3), dtype=np.uint8)
        boxes = np.asarray([b for _n, b in self.LAYOUT], dtype=np.float32)
        adapter.segment(
            [image], boxes, channel="CAM_FRONT", class_names=[n for n, _b in self.LAYOUT]
        )
        assert tracker.calls == [[[float(v) for v in b] for _n, b in self.LAYOUT]]
        assert adapter.last_mask_prompt == [BOX_FALLBACK] * 5
        assert adapter.text_counts["n_box_fallback"] == 5
        assert adapter.text_counts["n_text_matched"] == 0

    def test_a_short_class_names_array_is_refused_by_the_adapter(self):
        adapter, _tracker = self._adapter()
        image = np.zeros((self.CFG.image_height_px, self.CFG.image_width_px, 3), dtype=np.uint8)
        boxes = np.asarray([b for _n, b in self.LAYOUT], dtype=np.float32)
        with pytest.raises(MaskContractError, match="class_names"):
            adapter.segment([image], boxes, channel="CAM_FRONT", class_names=["a car"])

    def test_no_class_names_at_all_is_refused(self):
        adapter, _tracker = self._adapter()
        image = np.zeros((self.CFG.image_height_px, self.CFG.image_width_px, 3), dtype=np.uint8)
        boxes = np.asarray([b for _n, b in self.LAYOUT], dtype=np.float32)
        with pytest.raises(MaskContractError, match="class_names"):
            adapter.segment([image], boxes, channel="CAM_FRONT")

    def test_zero_boxes_returns_the_empty_contract(self):
        adapter, _tracker = self._adapter()
        image = np.zeros((self.CFG.image_height_px, self.CFG.image_width_px, 3), dtype=np.uint8)
        result = adapter.segment(
            [image], np.zeros((0, 4), dtype=np.float32), channel="CAM_FRONT", class_names=[]
        )
        assert result.masks.shape == (0, self.CFG.image_height_px, self.CFG.image_width_px)
        assert result.state is None and result.propagated is False

    def test_an_empty_prompt_never_reaches_the_processor(self):
        # `text=None`/"" makes Sam3Processor substitute the literal string
        # "visual" and run the exemplar path, attributing masks to a phrase no
        # model saw. The real _detect_text is used here, and refuses first.
        adapter, _tracker = text_adapter(self.CFG)
        image = np.zeros((self.CFG.image_height_px, self.CFG.image_width_px, 3), dtype=np.uint8)
        with pytest.raises(MaskContractError, match="visual"):
            adapter.segment(
                [image],
                np.asarray([[1, 1, 5, 5]], dtype=np.float32),
                channel="CAM_FRONT",
                class_names=["   "],
            )

    def test_a_fallback_that_returns_the_wrong_count_is_refused(self):
        adapter, tracker = self._adapter()
        tracker._segment_single = lambda image, boxes: np.zeros(
            (0, self.CFG.image_height_px, self.CFG.image_width_px), dtype=bool
        )
        image = np.zeros((self.CFG.image_height_px, self.CFG.image_width_px, 3), dtype=np.uint8)
        boxes = np.asarray([b for _n, b in self.LAYOUT], dtype=np.float32)
        with pytest.raises(MaskContractError, match="fallback"):
            adapter.segment(
                [image], boxes, channel="CAM_FRONT", class_names=[n for n, _b in self.LAYOUT]
            )

    def test_a_wrong_sized_frame_is_refused(self):
        adapter, _tracker = self._adapter()
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        with pytest.raises(MaskContractError, match="Stage 4 runs at"):
            adapter.segment(
                [image], np.asarray([[1, 1, 5, 5]], dtype=np.float32),
                channel="CAM_FRONT", class_names=["a car"],
            )

    def test_an_unloaded_adapter_raises_rather_than_returning_nothing(self):
        spec = CheckpointSpec(role=MASK_2D, provider="sam3_text", model_id="facebook/sam3",
                              revision="deadbeef")
        adapter = Sam3TextAdapter(spec, self.CFG)
        with pytest.raises(RuntimeError, match="not loaded"):
            adapter.segment([np.zeros((100, 200, 3), np.uint8)],
                            np.zeros((0, 4), np.float32), class_names=[])


class TestCrossPhraseOverlap:
    def test_overlapping_masks_from_different_phrases_are_counted(self):
        masks = np.stack([rect(0, 0, 10, 10, width=20, height=20),
                          rect(0, 0, 10, 10, width=20, height=20)])
        assert count_cross_phrase_mask_overlaps(masks, ["car", "rickshaw"],
                                                [TEXT_MATCHED, TEXT_MATCHED], 0.8) == 1

    def test_the_same_phrase_is_never_counted(self):
        # One-to-one already holds inside a phrase group; counting it here would
        # report the rule as if it had failed.
        masks = np.stack([rect(0, 0, 10, 10, width=20, height=20),
                          rect(0, 0, 10, 10, width=20, height=20)])
        assert count_cross_phrase_mask_overlaps(masks, ["car", "car"],
                                                [TEXT_MATCHED, TEXT_MATCHED], 0.8) == 0

    def test_box_fallback_masks_are_never_counted(self):
        masks = np.stack([rect(0, 0, 10, 10, width=20, height=20),
                          rect(0, 0, 10, 10, width=20, height=20)])
        assert count_cross_phrase_mask_overlaps(masks, ["car", "rickshaw"],
                                                [TEXT_MATCHED, BOX_FALLBACK], 0.8) == 0

    def test_disjoint_masks_are_not_an_overlap(self):
        masks = np.stack([rect(0, 0, 10, 10, width=40, height=20),
                          rect(20, 0, 30, 10, width=40, height=20)])
        assert count_cross_phrase_mask_overlaps(masks, ["car", "rickshaw"],
                                                [TEXT_MATCHED, TEXT_MATCHED], 0.8) == 0


# ---------------------------------------------------------------------------
# The wrong-rank branch (conformance 5.5-r1)
# ---------------------------------------------------------------------------


class TestWrongRankIsAContractError:
    """A 2-D masks array must refuse cleanly, not raise IndexError.

    The assertion's CONDITION was always short-circuit-safe; its MESSAGE was
    not — it interpolated `shape[2]` after the `ndim != 3` test had already
    passed, so a wrong-rank result raised IndexError, escaped main()'s
    `except MaskContractError` and became an uncaught traceback that
    run_stages.sh reads as DEGRADED rather than as the exit-2 refusal it is.
    """

    def _adapter(self):
        # __new__, never __init__: nothing is imported, nothing is loaded.
        adapter = TransformersSamAdapter.__new__(TransformersSamAdapter)
        adapter._cfg = MaskConfig(image_width_px=W, image_height_px=H)
        adapter._model = object()
        adapter._processor = object()
        adapter._torch = None
        return adapter

    def test_a_two_dimensional_mask_array_raises_maskcontracterror(self):
        adapter = self._adapter()
        # shape[0] == n_boxes, so the COUNT assert passes and control reaches
        # the resolution assert — which is the branch under test.
        adapter._segment_single = lambda image, boxes: np.zeros((2, H), dtype=bool)
        with pytest.raises(MaskContractError) as excinfo:
            adapter.segment(
                [np.zeros((H, W, 3), dtype=np.uint8)],
                np.asarray([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32),
                channel="CAM_FRONT",
            )
        assert "shape (2, 720)" in str(excinfo.value)

    def test_a_four_dimensional_mask_array_also_refuses_cleanly(self):
        adapter = self._adapter()
        adapter._segment_single = lambda image, boxes: np.zeros((1, 1, H, W), dtype=bool)
        with pytest.raises(MaskContractError):
            adapter.segment(
                [np.zeros((H, W, 3), dtype=np.uint8)],
                np.asarray([[0, 0, 10, 10]], dtype=np.float32),
                channel="CAM_FRONT",
            )

    def test_a_well_formed_stack_still_reports_width_by_height(self):
        assert mask_shape_desc(np.zeros((3, H, W), dtype=bool)) == f"{W}x{H}"

    def test_a_wrong_rank_stack_reports_its_shape_instead_of_raising(self):
        assert mask_shape_desc(np.zeros((2, 3), dtype=bool)) == "an array of shape (2, 3)"
        assert mask_shape_desc(np.zeros((5,), dtype=bool)) == "an array of shape (5,)"


# ---------------------------------------------------------------------------
# The driver: class_names gating, mask_prompt, and the D.1 regression
# ---------------------------------------------------------------------------


class FakeSubstrate:
    def __init__(self, token: str) -> None:
        self._token = token

    def by_token(self, name: str) -> dict:
        assert name == "calibrated_sensor.json"
        return {
            self._token: {
                "translation": [1.0, 0.0, 1.5],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "camera_intrinsic": [[1000.0, 0.0, 640.0], [0.0, 1000.0, 360.0], [0.0, 0.0, 1.0]],
            }
        }


class BoxOnlyAdapter:
    """A provider from before C29: no `supports_text_prompt`, no `class_names` kwarg."""

    supports_temporal = False

    def __init__(self) -> None:
        self.calls = 0

    def segment(self, images, boxes_xyxy_px, *, state=None, window=PER_FRAME_WINDOW, channel=""):
        self.calls += 1
        boxes = np.asarray(boxes_xyxy_px).reshape(-1, 4)
        masks = np.zeros((len(boxes), H, W), dtype=bool)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = (int(v) for v in box)
            masks[i, y1:y2, x1:x2] = True
        return MaskResult(masks=masks, state=None, window=window, propagated=False)


class TextAdapterSpy(BoxOnlyAdapter):
    """A C29 provider: declares the capability and records what it was handed."""

    supports_text_prompt = True

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[list[str]] = []
        self.last_mask_prompt: list[str] = []

    def segment(self, images, boxes_xyxy_px, *, state=None, window=PER_FRAME_WINDOW,
                channel="", class_names=None):
        self.seen.append(list(class_names or []))
        result = super().segment(images, boxes_xyxy_px, state=state, window=window, channel=channel)
        self.last_mask_prompt = [
            TEXT_MATCHED if index % 2 == 0 else BOX_FALLBACK
            for index in range(int(result.masks.shape[0]))
        ]
        return result


@pytest.fixture()
def keyframe(tmp_path):
    """One camera row on disk: an image, a calibrated sensor, two boxes."""
    from PIL import Image

    dataroot = tmp_path / "data"
    (dataroot / "samples" / "CAM_FRONT").mkdir(parents=True)
    rel = os.path.join("samples", "CAM_FRONT", "000001.jpg")
    Image.new("RGB", (W, H), (12, 34, 56)).save(dataroot / rel)
    row = {
        "sample_data_token": "sd0",
        "keyframe_token": "kf0",
        "scene_token": "sc0",
        "channel": "CAM_FRONT",
        "image_path": rel,
        "calibrated_sensor_token": "cs0",
        "boxes_xyxy_px": [[100.0, 100.0, 300.0, 300.0], [500.0, 100.0, 700.0, 300.0]],
        "class_names": ["a car", "a bus"],
        "scores": [0.9, 0.8],
    }
    return str(dataroot), row


def run_keyframe(dataroot, row, adapter, cfg=None):
    return process_keyframe(
        [row], adapter, FakeSubstrate("cs0"), cfg or MaskConfig(), dataroot,
        ("CAM_FRONT", "CAM_BACK"),
    )


class TestDriverTextGating:
    def test_a_pre_c29_adapter_is_never_passed_class_names(self, keyframe):
        # It has no such keyword: passing it would be a TypeError on every
        # keyframe, which is why the driver checks the capability first.
        dataroot, row = keyframe
        adapter = BoxOnlyAdapter()
        _masks, candidates, _ledger = run_keyframe(dataroot, row, adapter)
        assert adapter.calls == 1
        assert [c.mask_prompt for c in candidates] == [None, None]

    def test_a_text_adapter_receives_the_rows_phrases_in_box_order(self, keyframe):
        dataroot, row = keyframe
        adapter = TextAdapterSpy()
        _masks, candidates, _ledger = run_keyframe(dataroot, row, adapter)
        assert adapter.seen == [["a car", "a bus"]]
        assert [c.mask_prompt for c in candidates] == [TEXT_MATCHED, BOX_FALLBACK]

    def test_mask_prompt_is_emitted_only_when_it_was_measured(self, keyframe):
        dataroot, row = keyframe
        _masks, box_only, _ledger = run_keyframe(dataroot, row, BoxOnlyAdapter())
        _masks, texted, _ledger = run_keyframe(dataroot, row, TextAdapterSpy())
        plain_rows = candidate_rows("kf0", "sc0", box_only)
        text_rows = candidate_rows("kf0", "sc0", texted)
        assert all("mask_prompt" not in r for r in plain_rows)
        assert [r["mask_prompt"] for r in text_rows] == [TEXT_MATCHED, BOX_FALLBACK]
        # ... and the key is APPENDED: nothing else about the record moved.
        assert list(text_rows[0])[:-1] == list(plain_rows[0])

    def test_absent_class_names_refuse_only_under_a_text_adapter(self, keyframe):
        dataroot, row = keyframe
        row = dict(row)
        row.pop("class_names")
        with pytest.raises(UpstreamRefusal, match="class_names is absent"):
            run_keyframe(dataroot, row, TextAdapterSpy())

    def test_short_class_names_refuse_only_under_a_text_adapter(self, keyframe):
        dataroot, row = keyframe
        row = dict(row, class_names=["a car"])
        with pytest.raises(UpstreamRefusal, match="class_names has 1 entries"):
            run_keyframe(dataroot, row, TextAdapterSpy())
        # The existing providers' refusal behaviour is UNCHANGED: a box-prompted
        # run indexes class_names per candidate exactly as it always did, and a
        # short array is only ever an IndexError there, not a new refusal.
        with pytest.raises(IndexError):
            run_keyframe(dataroot, row, BoxOnlyAdapter())

    def test_required_class_names_is_its_own_predicate(self):
        # NOT optional_c27_array: absent is a REFUSAL here, not a legal [].
        assert required_class_names({"class_names": ["a car"]}, 1) == ["a car"]
        with pytest.raises(UpstreamRefusal):
            required_class_names({}, 1)
        with pytest.raises(UpstreamRefusal):
            required_class_names({"class_names": ["a car", "a bus"]}, 1)


class TestD1NullTrackIds:
    """`int(None)` on a merged-over-3b tree (masks.py D.1).

    stage3_merge writes `track_ids: None` for every arm-B box, and Stage 3b
    itself writes None for an untracked one. The array is PRESENT and the right
    length, so optional_c27_array passes it through — and the row builder then
    called int() on it.
    """

    def test_a_null_track_id_does_not_crash_the_row_builder(self, keyframe):
        dataroot, row = keyframe
        row = dict(row, track_ids=[None, None], box_sources=["arm_b", "arm_b"],
                   n_propagated_hops=[0, 0])
        _masks, candidates, _ledger = run_keyframe(dataroot, row, BoxOnlyAdapter())
        assert [c.track_id for c in candidates] == [None, None]
        assert [r["track_id"] for r in candidate_rows("kf0", "sc0", candidates)] == [None, None]

    def test_a_mixed_array_keeps_the_integers_and_the_nulls(self, keyframe):
        dataroot, row = keyframe
        row = dict(row, track_ids=[37, None], box_sources=["yolo", "arm_b"],
                   n_propagated_hops=[2, 0])
        _masks, candidates, _ledger = run_keyframe(dataroot, row, BoxOnlyAdapter())
        assert [c.track_id for c in candidates] == [37, None]
        assert [c.box_source for c in candidates] == ["yolo", "arm_b"]
        assert [c.n_propagated_hops for c in candidates] == [2, 0]

    def test_a_short_track_ids_array_is_still_refused(self, keyframe):
        # D.1 relaxes None, never the length contract (C27).
        dataroot, row = keyframe
        row = dict(row, track_ids=[1])
        with pytest.raises(UpstreamRefusal, match="track_ids has 1 entries"):
            run_keyframe(dataroot, row, BoxOnlyAdapter())

    def test_optional_c27_array_is_imported_from_rowmeta_not_redefined(self):
        # A.1: one definition, in pipeline/common/rowmeta.py.
        from pipeline.common import rowmeta
        from pipeline.stage4_masks import masks

        assert masks.optional_c27_array is rowmeta.optional_c27_array
        source = inspect.getsource(masks)
        assert "def optional_c27_array" not in source


# ---------------------------------------------------------------------------
# Config, provenance and the six registration points
# ---------------------------------------------------------------------------


class TestConfigAndProvenance:
    def test_every_new_field_carries_a_provenance_entry(self):
        cfg = MaskConfig()
        for name in NEW_CONFIG_FIELDS:
            assert name in cfg.as_dict(), f"{name} is not in the recorded config"
            assert name in cfg.provenance, f"{name} has no provenance entry"
            assert len(cfg.provenance[name].strip()) > 40, f"{name}'s provenance is a stub"

    def test_no_config_field_is_missing_from_provenance(self):
        # The house rule, checked over the whole dataclass rather than only over
        # the fields this change added.
        cfg = MaskConfig()
        documented = set(cfg.provenance)
        undocumented = {
            name for name in cfg.as_dict()
            if name != "provenance" and name not in documented
        }
        # Fields that predate the rule stay as they are; the point of the test is
        # that C29 added none to that set.
        assert undocumented & set(NEW_CONFIG_FIELDS) == set()

    def test_the_defaults_are_the_measured_ones(self):
        cfg = MaskConfig()
        for name, expected in NEW_CONFIG_FIELDS.items():
            assert getattr(cfg, name) == expected

    def test_the_provenance_quotes_the_strip_article_measurement(self):
        text = MaskConfig().provenance["text_prompt_strip_article"]
        assert "+3" in text and "66" in text
        assert "no phrase regressed" in text.lower()
        assert "gloss" in text.lower()

    def test_the_provenance_quotes_the_threshold_sweep_and_the_dtype_measurement(self):
        cfg = MaskConfig()
        assert "0.3" in cfg.provenance["text_score_threshold"]
        assert "0.4" in cfg.provenance["text_score_threshold"]
        assert "bfloat16" in cfg.provenance["text_detector_dtype"]
        assert "3775" in cfg.provenance["text_detector_dtype"]

    def test_text_prompting_is_off_by_default(self):
        assert MaskConfig().text_prompt is False

    def test_the_dtype_name_is_validated_without_torch(self):
        assert check_detector_dtype_name("bfloat16") == "bfloat16"
        assert check_detector_dtype_name("float32") == "float32"
        with pytest.raises(UpstreamRefusal, match="text_detector_dtype"):
            check_detector_dtype_name("fp16")


class TestRegistration:
    def test_the_adapter_table_and_the_provenance_table_agree(self):
        assert _MASK_ADAPTERS["sam3_text"] is Sam3TextAdapter
        # run() indexes _PROVIDER_PROVENANCE unconditionally: a miss is a KeyError
        # after clear_markers.
        assert set(_MASK_ADAPTERS) <= set(_PROVIDER_PROVENANCE)

    def test_the_provenance_string_carries_the_measured_peak(self):
        text = _PROVIDER_PROVENANCE["sam3_text"]
        assert "3775 MiB peak" in text
        assert "max_memory_allocated" in text
        assert "5629 MiB" in text  # the fp32/fp32 comparison, not hidden

    def test_the_registry_knows_the_provider(self):
        from pipeline.common.model_interfaces import provider_names, roles_of

        assert "sam3_text" in provider_names(MASK_2D)
        assert roles_of("sam3_text") == (MASK_2D,)

    def test_the_cli_offers_the_provider_and_the_flag(self, capsys):
        with pytest.raises(SystemExit):
            main(["--help"])
        out = capsys.readouterr().out
        assert "--text-prompt" in out
        assert "sam3_text" in out

    def test_an_unknown_provider_is_still_rejected_by_argparse(self, capsys):
        with pytest.raises(SystemExit):
            main(["--provider", "sam3_exemplar"])
        assert "invalid choice" in capsys.readouterr().err

    def test_the_adapter_declares_the_capability(self):
        adapter = Sam3TextAdapter.__new__(Sam3TextAdapter)
        assert adapter.supports_text_prompt is True
        # Honest about what it does NOT do: it never opens a video session.
        assert adapter.supports_temporal is False

    def test_supports_text_prompt_is_not_in_role_methods(self):
        # Adding it there would fail conformance for all four shipped adapters
        # and for measure_vram.py's harness adapter on the day it was added.
        assert "supports_text_prompt" not in ROLE_METHODS[MASK_2D]
        assert "supports_text_prompt" in dir(Mask2D)

    def test_the_protocol_signature_carries_class_names(self):
        parameters = inspect.signature(Mask2D.segment).parameters
        assert "class_names" in parameters
        assert parameters["class_names"].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters["class_names"].default is None


class TestTextPromptRefusals:
    @pytest.mark.parametrize("provider", ["mobile_sam", "sam2_video", "sam31_multiplex"])
    def test_a_provider_that_cannot_honour_text_refuses(self, provider):
        with pytest.raises(UpstreamRefusal, match="--text-prompt"):
            refuse_text_prompt_provider(provider, MaskConfig(text_prompt=True))

    def test_the_sam31_refusal_names_the_destructive_reset(self):
        with pytest.raises(UpstreamRefusal) as excinfo:
            refuse_text_prompt_provider("sam31_multiplex", MaskConfig(text_prompt=True))
        assert "reset_state" in str(excinfo.value)

    def test_the_tracker_head_is_refused_too(self):
        # sam3_tracker is the DEFAULT provider and the likeliest mistake.
        with pytest.raises(UpstreamRefusal, match="detector head"):
            refuse_text_prompt_provider("sam3_tracker", MaskConfig(text_prompt=True))

    def test_sam3_text_is_the_one_provider_that_passes(self):
        refuse_text_prompt_provider("sam3_text", MaskConfig(text_prompt=True))

    def test_nothing_is_refused_when_the_flag_is_off(self):
        for provider in ("mobile_sam", "sam2_video", "sam3_tracker", "sam31_multiplex"):
            refuse_text_prompt_provider(provider, MaskConfig())

    def test_every_preflight_branch_assigns_not_covered(self):
        # `preflight_mask_adapter` returns a dict naming `not_covered`; a branch
        # that forgot to set it raises UnboundLocalError AT THE RETURN, i.e.
        # after the expensive checks and with a useless message.
        tree = ast.parse(inspect.getsource(preflight_mask_adapter))
        function = tree.body[0]
        chains = [node for node in function.body if isinstance(node, ast.If)]
        assert chains, "the provider branch chain moved"
        branches, node = [], chains[-1]
        while True:
            branches.append(node.body)
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                node = node.orelse[0]
                continue
            assert node.orelse, "the branch chain has no else: an unknown provider would fall through"
            branches.append(node.orelse)
            break
        assert len(branches) >= 4, "expected mobile_sam / sam31 / sam3_text / else"
        for body in branches:
            assigned = [
                target.id
                for statement in ast.walk(ast.Module(body=body, type_ignores=[]))
                if isinstance(statement, ast.Assign)
                for target in statement.targets
                if isinstance(target, ast.Name)
            ]
            assert "not_covered" in assigned

    def test_preflight_refuses_a_text_prompt_before_anything_expensive(self):
        pytest.importorskip("torch")
        with pytest.raises(UpstreamRefusal, match="--text-prompt"):
            preflight_mask_adapter(object(), "mobile_sam", MaskConfig(text_prompt=True))


class Stopped(Exception):
    """Sentinel: run() reached the preflight, i.e. every free refusal passed."""


class TestRunProviderResolution:
    """run()'s pre-marker block: the inference override and the attribution guard.

    Every assertion here is reached BEFORE `Substrate.load`, before
    `clear_markers` and before any adapter is loaded — which is the point. The
    preflight is stubbed out so nothing touches the hub, the cache or the card.
    """

    @pytest.fixture(autouse=True)
    def stub_preflight(self, monkeypatch):
        self.seen: dict = {}

        def fake_preflight(adapter, provider, cfg):
            self.seen = {"provider": provider, "text_prompt": cfg.text_prompt,
                         "adapter": type(adapter).__name__}
            raise Stopped()

        monkeypatch.setattr(masks_module, "preflight_mask_adapter", fake_preflight)

    def _run(self, cfg):
        # paths / upstream / marker are never touched before the preflight.
        return run(None, {}, None, cfg, "/nonexistent", "/nonexistent", None)

    def test_the_default_inference_is_untouched(self):
        with pytest.raises(Stopped):
            self._run(MaskConfig(revision="r"))
        assert self.seen["provider"] == "sam3_tracker"
        assert self.seen["text_prompt"] is False

    def test_text_prompt_switches_the_inferred_provider(self):
        with pytest.raises(Stopped):
            self._run(MaskConfig(revision="r", text_prompt=True))
        assert self.seen["provider"] == "sam3_text"
        assert self.seen["adapter"] == "Sam3TextAdapter"

    def test_an_explicit_provider_is_never_silently_replaced(self):
        # --text-prompt overrides the INFERENCE, not an operator's own choice:
        # mobile_sam stays mobile_sam and is then refused, loudly.
        with pytest.raises(UpstreamRefusal, match="MobileSAM has no text encoder"):
            self._run(MaskConfig(revision="r", text_prompt=True, provider="mobile_sam",
                                 model_id="mobile_sam_vit_t", checkpoint_path="/nonexistent"))

    def test_text_prompt_on_another_checkpoint_does_not_switch_the_weights(self):
        # No --provider, but a model_id that is not facebook/sam3: the inferred
        # provider stands and the flag refuses, rather than quietly loading a
        # different model than the operator asked for.
        with pytest.raises(UpstreamRefusal, match="SAM 2 has no text conditioning"):
            self._run(MaskConfig(revision="r", text_prompt=True, model_id="facebook/sam2.1-hiera-large"))

    def test_provider_sam3_text_corrects_the_recorded_config(self, capsys):
        # --provider sam3_text without --text-prompt means the same thing; the
        # config is corrected so the manifest cannot say text_prompt=false for a
        # run whose every mask came from a phrase.
        with pytest.raises(Stopped):
            self._run(MaskConfig(revision="r", provider="sam3_text"))
        assert self.seen["text_prompt"] is True
        assert "recording config.text_prompt = true" in capsys.readouterr().err

    def test_sam3_text_may_only_be_recorded_against_facebook_sam3(self):
        with pytest.raises(UpstreamRefusal, match="would attribute SAM 3 text-prompted masks"):
            self._run(MaskConfig(revision="r", provider="sam3_text", model_id="facebook/sam2"))

    def test_a_bad_detector_dtype_refuses_before_the_marker_comes_down(self):
        with pytest.raises(UpstreamRefusal, match="text_detector_dtype"):
            self._run(MaskConfig(revision="r", text_prompt=True, text_detector_dtype="fp16"))

    def test_the_existing_attribution_guards_still_fire(self):
        with pytest.raises(UpstreamRefusal, match="MobileSAM masks"):
            self._run(MaskConfig(revision="r", provider="mobile_sam", model_id="facebook/sam3"))


# ---------------------------------------------------------------------------
# The manifest block
# ---------------------------------------------------------------------------


class TestManifestBlock:
    def test_a_box_prompted_run_records_that_no_phrase_reached_a_model(self):
        block = text_prompt_manifest_block(MaskConfig(), "sam3_tracker", object())
        assert block["enabled"] is False
        assert "sam3_tracker" in block["note"]

    def test_an_enabled_block_carries_the_strings_their_hash_and_the_counts(self):
        cfg = MaskConfig(text_prompt=True)
        adapter, _tracker = text_adapter(cfg)
        adapter.text_prompts = {"a car": "car", "a bus": "bus"}
        adapter.text_counts["n_text_matched"] = 41
        adapter.text_counts["n_box_fallback"] = 25
        adapter.text_counts["per_phrase"] = {"bus": {"n_text_matched": 0, "n_box_fallback": 4}}
        adapter.processor_build = "hand_built"
        block = text_prompt_manifest_block(cfg, "sam3_text", adapter)

        assert block["enabled"] is True
        assert block["resolved_prompts"] == {"a car": "car", "a bus": "bus"}
        assert len(block["resolved_prompts_sha256"]) == 64
        assert block["counts"]["n_text_matched"] == 41
        assert block["counts"]["n_box_fallback"] == 25
        # The counts block must not swallow the per-phrase table, and vice versa.
        assert "per_phrase" not in block["counts"]
        assert block["per_phrase"]["bus"]["n_box_fallback"] == 4
        assert block["strip_article"] is True
        assert block["match_min_iou"] == 0.5
        assert block["score_threshold"] == 0.3
        assert block["detector_dtype"] == "bfloat16"
        assert block["tracker_dtype"] == "float32"
        assert block["processor_build"] == "hand_built"
        # The four outcome counters the plan names are all present.
        for key in ("n_text_matched", "n_box_fallback", "n_text_duplicate_rejected",
                    "n_cross_phrase_mask_overlap"):
            assert key in block["counts"]

    def test_the_block_records_the_c13_reading_and_the_unbuilt_variant(self):
        cfg = MaskConfig(text_prompt=True)
        adapter, _tracker = text_adapter(cfg)
        block = text_prompt_manifest_block(cfg, "sam3_text", adapter)
        assert "reading (a)" in block["c13_reading"]
        assert "1008" in block["c13_reading"]
        assert "ratification" in block["c13_reading"]
        assert "sam3_exemplar" in block["future_variant"]
        assert "NOT BUILT" in block["future_variant"]
        # Stage 4 is the FIRST consumer in this chain for which the caption is
        # an input, and the manifest says so in its own block.
        assert block["caption_is_input"] is True

    def test_the_hash_follows_the_strings(self):
        cfg = MaskConfig(text_prompt=True)
        first, _tracker = text_adapter(cfg)
        first.text_prompts = {"a car": "car"}
        second, _tracker2 = text_adapter(cfg)
        second.text_prompts = {"a car": "a car"}
        assert (
            text_prompt_manifest_block(cfg, "sam3_text", first)["resolved_prompts_sha256"]
            != text_prompt_manifest_block(cfg, "sam3_text", second)["resolved_prompts_sha256"]
        )

    def test_the_selection_rule_is_recorded_in_words(self):
        cfg = MaskConfig(text_prompt=True)
        adapter, _tracker = text_adapter(cfg)
        block = text_prompt_manifest_block(cfg, "sam3_text", adapter)
        assert "one-to-one" in block["selection_rule"]
        assert "PRE-SIZED" in block["selection_rule"]
        assert "input_boxes are never passed" in block["mechanism"].replace("NEVER", "never")


def test_the_conventions_import_is_used():
    # CAMERA / EGO are the frames process_keyframe builds its transform between;
    # importing them here keeps this file honest about what it exercised.
    assert CAMERA != EGO
    assert TemporalWindow(frames=1).frames == 1


# ---------------------------------------------------------------------------
# The REAL post-processor, on CPU (regression: the 2026-09-01 smoke refusal)
# ---------------------------------------------------------------------------


class FakeSam3Outputs:
    """Exactly the four fields `post_process_instance_segmentation` reads.

    Not a mock of the post-processor: a stand-in for `Sam3ImageSegmentationOutput`
    that the REAL post-processor is run over. `pred_masks` is at the decoder's
    own mask resolution (288x288 on the real model, `RAW` here), which is the
    whole point — that resolution is what leaks out when nothing matches.
    """

    def __init__(self, pred_logits, pred_boxes, pred_masks, presence_logits) -> None:
        self.pred_logits = pred_logits          # (batch, num_queries)
        self.pred_boxes = pred_boxes            # (batch, num_queries, 4), xyxy, normalised
        self.pred_masks = pred_masks            # (batch, num_queries, raw, raw) logits
        self.presence_logits = presence_logits  # (batch, 1)


class RealPostProcessProcessor:
    """A processor whose post-processing is the REAL `Sam3ImageProcessor` method.

    Only the encode is faked (it just has to produce the tensors `_detect_text`
    moves to the device); `post_process_instance_segmentation` delegates exactly
    the way `Sam3Processor` does — same argument order, same object.
    """

    def __init__(self, torch_module, image_processor, drop_target_sizes: bool = False) -> None:
        self._torch = torch_module
        self.image_processor = image_processor
        self._drop_target_sizes = drop_target_sizes
        self.encode_calls: list[dict] = []
        self.post_calls: list[dict] = []

    def __call__(self, images=None, text=None, return_tensors=None, **kwargs):
        torch = self._torch
        self.encode_calls.append({"text": text, "return_tensors": return_tensors})
        image = np.asarray(images)
        return {
            "pixel_values": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
            "input_ids": torch.zeros((1, 4), dtype=torch.int64),
            "attention_mask": torch.ones((1, 4), dtype=torch.int64),
            # Sam3Processor emits this and the adapter forwards it; it rides
            # along here for the same reason it does there.
            "original_sizes": torch.tensor([[image.shape[0], image.shape[1]]], dtype=torch.int64),
        }

    def post_process_instance_segmentation(
        self, outputs, threshold=0.3, mask_threshold=0.5, target_sizes=None
    ):
        self.post_calls.append({"threshold": threshold, "mask_threshold": mask_threshold,
                                "target_sizes": target_sizes})
        if self._drop_target_sizes:
            # Only used to prove the resolution guard is still armed: the real
            # method with no inverse applied returns raw-resolution masks.
            target_sizes = None
        return self.image_processor.post_process_instance_segmentation(
            outputs, threshold, mask_threshold, target_sizes
        )


class TestRealPostProcessInstanceSegmentation:
    """`_detect_text` over the REAL transformers post-processor. CPU, no weights.

    THIS IS THE TEST THAT WAS MISSING. Every other adapter test in this file
    replaces `_detect_text` or hands it a processor whose post-processing is a
    fake that always returns frame-sized masks, and that is exactly why the
    defect shipped: the real method guards its resize with `if len(masks) > 0`
    (image_processing_sam3.py:921), so a phrase that scores NOTHING above the
    threshold comes back at the decoder's raw mask geometry — (0, 288, 288),
    never (0, H, W). Zero instances is the designed, measured-normal outcome
    (24% of groups; "a bus" on 10 of 10 frames), and the adapter refused the
    whole run on it: the 2026-09-01 GPU smoke died on its first frame,
    CAM_BACK_RIGHT/000001 + "pedestrian" (max score 0.1709 < 0.3).
    """

    CFG = MaskConfig(text_prompt=True, image_width_px=200, image_height_px=100)
    RAW = 8            # stands in for the real decoder's 288x288 mask geometry
    N_QUERIES = 3

    def _torch(self):
        return pytest.importorskip("torch")

    def _image_processor(self):
        pytest.importorskip("transformers")
        # Imported HERE, not at module scope: this file's contract is that
        # importing it costs nothing. The class is CPU-only and pure — it
        # downloads nothing and touches no weights.
        from transformers import Sam3ImageProcessor

        return Sam3ImageProcessor()

    def _outputs(self, kept: list[int]):
        """Fake detector outputs whose queries in `kept` beat the score threshold."""
        torch = self._torch()
        logits = torch.full((1, self.N_QUERIES), -5.0)   # sigmoid ~= 0.007
        for index in kept:
            logits[0, index] = 5.0                       # sigmoid ~= 0.993
        boxes = torch.tensor([[[0.1, 0.1, 0.5, 0.5]] * self.N_QUERIES], dtype=torch.float32)
        masks = torch.full((1, self.N_QUERIES, self.RAW, self.RAW), -10.0)
        masks[:, :, 1:6, 1:6] = 10.0                     # a solid blob, sigmoid ~= 1
        presence = torch.full((1, 1), 10.0)
        return FakeSam3Outputs(logits, boxes, masks, presence)

    def _adapter(self, kept: list[int], *, drop_target_sizes: bool = False):
        torch = self._torch()
        adapter, tracker = text_adapter(self.CFG)
        adapter._torch = torch
        adapter._device = "cpu"
        adapter._detector_dtype = torch.float32
        adapter._processor = RealPostProcessProcessor(
            torch, self._image_processor(), drop_target_sizes=drop_target_sizes
        )
        outputs = self._outputs(kept)
        adapter._detector = lambda **kwargs: outputs
        return adapter, tracker

    def _frame(self):
        return np.zeros((self.CFG.image_height_px, self.CFG.image_width_px, 3), dtype=np.uint8)

    def test_the_real_post_processor_returns_raw_geometry_when_nothing_matches(self):
        # The upstream fact the adapter has to live with, asserted directly so a
        # transformers change that fixes it is visible here rather than silently
        # making the branch below dead.
        image_processor = self._image_processor()
        result = image_processor.post_process_instance_segmentation(
            self._outputs([]),
            self.CFG.text_score_threshold,
            Sam3TextAdapter._TEXT_MASK_THRESHOLD,
            [(self.CFG.image_height_px, self.CFG.image_width_px)],
        )[0]
        assert tuple(result["masks"].shape) == (0, self.RAW, self.RAW)
        assert tuple(result["masks"].shape[1:]) != (
            self.CFG.image_height_px, self.CFG.image_width_px
        )

    def test_a_zero_instance_phrase_is_no_instances_not_a_refusal(self):
        # Pre-fix this raised MaskContractError("... 8x8 instance masks ...").
        adapter, _tracker = self._adapter(kept=[])
        masks, boxes = adapter._detect_text(self._frame(), "pedestrian", "CAM_BACK_RIGHT")
        assert masks == [] and boxes == []
        assert adapter.text_counts["n_detector_forwards"] == 1
        assert adapter.text_counts["n_instances_returned"] == 0
        assert adapter.text_counts["n_empty_instances_dropped"] == 0
        assert adapter._processor.post_calls[0]["target_sizes"] == [
            (self.CFG.image_height_px, self.CFG.image_width_px)
        ]

    def test_a_matched_phrase_still_comes_back_at_frame_resolution(self):
        adapter, _tracker = self._adapter(kept=[0, 2])
        masks, boxes = adapter._detect_text(self._frame(), "car", "CAM_BACK_RIGHT")
        assert len(masks) == 2 and len(boxes) == 2
        for mask in masks:
            assert mask.shape == (self.CFG.image_height_px, self.CFG.image_width_px)
            assert mask.dtype == np.bool_ and mask.any()
        assert adapter.text_counts["n_instances_returned"] == 2

    def test_the_resolution_guard_still_refuses_a_wrong_sized_non_empty_stack(self):
        # The guard is not weakened, only made blind to the EMPTY case: a
        # non-empty stack that never took the inverse still refuses the run.
        adapter, _tracker = self._adapter(kept=[0, 2], drop_target_sizes=True)
        with pytest.raises(MaskContractError, match="instance masks"):
            adapter._detect_text(self._frame(), "car", "CAM_BACK_RIGHT")

    def test_a_frame_whose_phrase_matches_nothing_falls_back_box_by_box(self):
        # End to end through segment(), which is where the smoke died: every box
        # of a zero-instance phrase takes the box-prompt fallback, in box order.
        adapter, tracker = self._adapter(kept=[])
        boxes = np.asarray([[10, 10, 30, 30], [50, 10, 70, 30]], dtype=np.float32)
        result = adapter.segment(
            [self._frame()], boxes, channel="CAM_BACK_RIGHT",
            class_names=["a pedestrian", "a pedestrian"],
        )
        assert result.masks.shape == (2, self.CFG.image_height_px, self.CFG.image_width_px)
        assert adapter.last_mask_prompt == [BOX_FALLBACK, BOX_FALLBACK]
        assert adapter.text_counts["n_box_fallback"] == 2
        assert adapter.text_counts["n_text_matched"] == 0
        assert tracker.calls == [[[10.0, 10.0, 30.0, 30.0], [50.0, 10.0, 70.0, 30.0]]]
