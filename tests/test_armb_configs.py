"""Tests for the arm B class-space configs (Tasks 1-3 of the 2026-08-27 plan).

Run: /home/mt/miniconda3/envs/ano_pipe/bin/python -m pytest tests/test_armb_configs.py -v
"""

from __future__ import annotations

import os
import sys

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pipeline.stage3_proposals.proposals import (  # noqa: E402
    build_caption,
    load_class_map,
    load_taxonomy,
)

NUSCENES_TAXONOMY = os.path.join(ROOT, "configs", "taxonomy_pilot_nuscenes.yaml")
DHAKA_TAXONOMY = os.path.join(ROOT, "configs", "taxonomy_pilot_dhaka.yaml")
RSUD20K_MAP = os.path.join(ROOT, "configs", "rsud20k_to_phrase_dhaka.yaml")

# The 13 class names baked into the arm B checkpoint's model.names, copied from
# local_yolox_build/configs/rsud20k_yolo11x.yaml `names:` (index order).
RSUD20K_NAMES = (
    "person", "rickshaw", "rickshaw van", "cng", "truck", "pickup truck",
    "car", "motorcycle", "bicycle", "bus", "micro bus", "covered van",
    "human hauler",
)


class TestDhakaTaxonomy:
    def test_loads_and_has_twelve_phrases(self):
        t = load_taxonomy(DHAKA_TAXONOMY)
        assert len(t.phrases) == 12

    def test_first_ten_phrases_are_v2_verbatim_in_order(self):
        v2 = load_taxonomy(NUSCENES_TAXONOMY)
        v3 = load_taxonomy(DHAKA_TAXONOMY)
        assert v3.phrases[: len(v2.phrases)] == v2.phrases

    def test_appended_phrases_and_order(self):
        v3 = load_taxonomy(DHAKA_TAXONOMY)
        assert v3.phrases[-2:] == ("a rickshaw", "an auto rickshaw")

    def test_caption_prefix_property(self):
        v2 = load_taxonomy(NUSCENES_TAXONOMY)
        v3 = load_taxonomy(DHAKA_TAXONOMY)
        cap2, cap3 = build_caption(v2.phrases), build_caption(v3.phrases)
        assert cap3.text.startswith(cap2.text)
        # arm A spans must be valid verbatim under the v3 caption
        assert cap3.phrase_char_spans[: len(v2.phrases)] == cap2.phrase_char_spans

    def test_dhaka_categories_map_to_new_phrases(self):
        v3 = load_taxonomy(DHAKA_TAXONOMY)
        p2c = v3.phrase_to_categories
        assert p2c["a rickshaw"] == ("dhaka.cycle_rickshaw",)
        assert p2c["an auto rickshaw"] == ("dhaka.cng_autorickshaw",)

    def test_v2_exclusions_survive(self):
        v2 = load_taxonomy(NUSCENES_TAXONOMY)
        v3 = load_taxonomy(DHAKA_TAXONOMY)
        assert set(v2.excluded_categories) <= set(v3.excluded_categories)


class TestRsud20kMap:
    def test_map_loads_against_dhaka_taxonomy(self):
        t = load_taxonomy(DHAKA_TAXONOMY)
        m = load_class_map(RSUD20K_MAP, t)
        assert m.mapping == {"rickshaw": "a rickshaw", "cng": "an auto rickshaw"}

    def test_covers_the_checkpoint_names_exactly(self):
        t = load_taxonomy(DHAKA_TAXONOMY)
        m = load_class_map(RSUD20K_MAP, t)
        m.assert_covers(RSUD20K_NAMES)  # raises UpstreamRefusal on any drift

    def test_unreachable_is_the_whole_v2_space(self):
        v2 = load_taxonomy(NUSCENES_TAXONOMY)
        v3 = load_taxonomy(DHAKA_TAXONOMY)
        m = load_class_map(RSUD20K_MAP, v3)
        assert m.unreachable_phrases(v3) == v2.phrases

    def test_rejected_against_v2_taxonomy(self):
        import pytest
        from pipeline.common.manifest import UpstreamRefusal
        v2 = load_taxonomy(NUSCENES_TAXONOMY)
        with pytest.raises(UpstreamRefusal):
            load_class_map(RSUD20K_MAP, v2)


class TestReleaseMapAliases:
    def test_arm_b_phrases_resolve(self):
        with open(os.path.join(ROOT, "configs", "release_category_map.yaml"), "rb") as fh:
            doc = yaml.safe_load(fh)
        assert doc["map"]["a rickshaw"] == "cycle_rickshaw"
        assert doc["map"]["an auto rickshaw"] == "cng_autorickshaw"
        # alias values must stay inside the declared 18-class space
        assert set(doc["map"].values()) <= set(doc["classes"])
