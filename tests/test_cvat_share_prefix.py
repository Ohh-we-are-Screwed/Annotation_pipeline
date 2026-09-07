"""cvat_setup.py --share-prefix (2026-09-06).

The 2D and road publishes create tasks FROM THE SHARE: `resources` are the
COCO file_names (`samples/CAM_BACK/000008.jpg`) resolved against whatever the
CVAT server has mounted at /home/django/share. That bind held pilot_1632's
images staged on 2026-08-30, so chunk_0006's first publish showed the pilot's
frames under the new annotations — and every day-1 chunk carries the same
relative names, so a flat share cannot hold two of them. Each chunk is now
staged under its own prefix and the publish references it, in the frame list
AND in the COCO it imports (CVAT binds annotations to frames by name).
"""

from __future__ import annotations

import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.cvat_setup import prefix_share  # noqa: E402

DOC = {
    "images": [{"id": 1, "file_name": "samples/CAM_BACK/000008.jpg", "width": 1280, "height": 720},
               {"id": 2, "file_name": "samples/CAM_FRONT/000008.jpg", "width": 1280, "height": 720}],
    "annotations": [{"id": 1, "image_id": 1, "category_id": 3, "bbox": [1, 2, 3, 4]}],
    "categories": [{"id": 3, "name": "a car"}],
}


def test_empty_prefix_is_the_identity():
    resources, doc = prefix_share(DOC, "")
    assert resources == ["samples/CAM_BACK/000008.jpg", "samples/CAM_FRONT/000008.jpg"]
    assert doc == DOC


def test_prefix_lands_in_resources_and_file_names_alike():
    resources, doc = prefix_share(DOC, "day1_chunk_0006/")
    assert resources == ["day1_chunk_0006/samples/CAM_BACK/000008.jpg", "day1_chunk_0006/samples/CAM_FRONT/000008.jpg"]
    assert [i["file_name"] for i in doc["images"]] == resources


def test_missing_trailing_slash_is_normalised():
    resources, _ = prefix_share(DOC, "day1_chunk_0006")
    assert resources[0] == "day1_chunk_0006/samples/CAM_BACK/000008.jpg"


def test_everything_else_is_untouched_and_input_not_mutated():
    before = copy.deepcopy(DOC)
    _, doc = prefix_share(DOC, "p/")
    assert doc["annotations"] == DOC["annotations"] and doc["categories"] == DOC["categories"]
    assert [i["id"] for i in doc["images"]] == [1, 2]
    assert DOC == before


def test_env_default_reaches_argparse(monkeypatch):
    # The wrapper never mentions the prefix; the driver exports CVAT_SHARE_PREFIX
    # and cvat_setup reads it the way it reads CVAT_HOST.
    monkeypatch.setenv("CVAT_SHARE_PREFIX", "day1_chunk_0006/")
    import importlib
    import scripts.cvat_setup as m
    importlib.reload(m)
    parser = m.build_parser()
    assert parser.parse_args([]).share_prefix == "day1_chunk_0006/"
    monkeypatch.delenv("CVAT_SHARE_PREFIX")
    importlib.reload(m)
    assert m.build_parser().parse_args([]).share_prefix == ""
