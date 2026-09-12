"""The tail (stages 7-9) on stage6_stereo_box boxes.

stage6_stereo_box emits Stage 6's row shape, so stages 7-9 consume it unchanged
-- except in the three places that named `stage6_cluster` for reasons that have
nothing to do with clustering: Stage 7's input directory, the module a refusal
tells the operator to run, and the provenance of the boxes that ship. The first
real stereo run refused at Stage 7 with

    STAGE 7: REFUSING TO START: <work_root>/stage6_cluster/run_manifest.json
    not found; run `python3 -m pipeline.stage6_cluster.cluster` first

with a complete stage6_stereo_box tree sitting beside it, because the wrapper
picked the producer for Stage 8 and the release (`boxes_dir`) but not for Stage
7. One helper now answers it for both, and these tests pin that they agree.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.common.manifest import (  # noqa: E402
    UpstreamRefusal,
    boxes_module_hint,
    boxes_source,
    require_upstream,
)

WRAPPER = os.path.join(ROOT, "scripts", "run_stages.sh")


def _wrapper_text() -> str:
    with open(WRAPPER, "r", encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------


def test_wrapper_is_syntactically_valid():
    assert subprocess.run(["bash", "-n", WRAPPER], capture_output=True).returncode == 0


def test_stage7_and_the_box_consumers_pick_the_producer_the_same_way():
    text = _wrapper_text()
    step7 = text[text.index("    7)  acc"):text.index("    8)  acc")]
    assert '--stage6-dir "$(boxes_dir_for_stage6)"' in step7, step7
    # ...and boxes_dir() (Stage 8 + the release) delegates rather than
    # re-implementing the choice, so the two can never disagree.
    boxes_dir = text[text.index("boxes_dir() {"):]
    boxes_dir = boxes_dir[:boxes_dir.index("\n}")]
    assert "boxes_dir_for_stage6" in boxes_dir
    assert "stage6_stereo_box" not in boxes_dir, "the stage6 choice lives in ONE helper"


@pytest.mark.parametrize(
    "stereo_marker,cluster_marker,stereo_newer,expected",
    [
        (True, False, False, "stage6_stereo_box"),   # only stereo ran
        (False, True, False, "stage6_cluster"),      # only cluster ran
        (True, True, True, "stage6_stereo_box"),     # both, stereo is fresher
        (True, True, False, "stage6_cluster"),       # both, stereo is STALE
        (False, False, False, "stage6_cluster"),     # neither: the default refusal
    ],
)
def test_boxes_dir_for_stage6_freshness_rule(tmp_path, stereo_marker, cluster_marker, stereo_newer, expected):
    """The helper itself, run by bash with the wrapper's own marker_state."""
    for name, marked, mtime in (
        ("stage6_cluster", cluster_marker, 1_700_000_100),
        ("stage6_stereo_box", stereo_marker, 1_700_000_200 if stereo_newer else 1_700_000_000),
    ):
        d = tmp_path / name
        d.mkdir()
        (d / "run_manifest.json").write_text("{}")
        os.utime(d / "run_manifest.json", (mtime, mtime))
        if marked:
            (d / "_SUCCESS").write_text("fingerprint\n")

    text = _wrapper_text()
    helper = text[text.index("boxes_dir_for_stage6() {"):]
    helper = helper[:helper.index("\n}") + 2]
    marker_state = text[text.index("marker_state() {"):]
    marker_state = marker_state[:marker_state.index("\n}") + 2]
    script = f'WORK_ROOT="{tmp_path}"\n{marker_state}\n{helper}\nboxes_dir_for_stage6\n'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True)
    assert os.path.basename(out.stdout.strip()) == expected


# ---------------------------------------------------------------------------
# The stages
# ---------------------------------------------------------------------------


def _stage6_tree(tmp_path, stage: str, fingerprint: str = "f" * 8):
    d = tmp_path / stage
    d.mkdir()
    (d / "run_manifest.json").write_text(json.dumps({
        "spec": f"dhakascenes-pilot/{stage}/v1",
        "stage": stage,
        "upstream": {"metadata_fingerprint": fingerprint, "fingerprint_spec": "sha256/v1"},
    }))
    (d / "_SUCCESS").write_text(fingerprint + "\n")
    return str(d)


def test_stage7_accepts_a_stereo_box_tree(tmp_path):
    """The gate is about completion and substrate, never about which producer."""
    d = _stage6_tree(tmp_path, "stage6_stereo_box")
    manifest, marker = require_upstream(
        d, stage_name="Stage 6", module_hint=boxes_module_hint(d), current_fingerprint="f" * 8,
    )
    assert manifest["spec"] == "dhakascenes-pilot/stage6_stereo_box/v1"
    assert marker.state == "clean"


def test_a_missing_tree_names_the_producer_that_was_asked_for(tmp_path):
    """"run stage6_cluster first" is wrong advice on a stereo run."""
    with pytest.raises(UpstreamRefusal) as exc:
        require_upstream(
            str(tmp_path / "stage6_stereo_box"), stage_name="Stage 6",
            module_hint=boxes_module_hint(str(tmp_path / "stage6_stereo_box")),
        )
    assert "pipeline.stage6_stereo_box.stereo_box" in str(exc.value)
    assert boxes_module_hint("/w/stage6_cluster") == "pipeline.stage6_cluster.cluster"
    assert boxes_module_hint("/w/stage7_track/") == "pipeline.stage7_track.track"


def test_boxes_source_survives_the_whole_tail():
    """Stage 9's manifest has to name the Stage 6 producer, three stages later."""
    s6 = {"stage": "stage6_stereo_box", "spec": "dhakascenes-pilot/stage6_stereo_box/v1"}
    s7 = {"stage": "stage7_track", "boxes_source": boxes_source(s6)}
    s8 = {"stage": "stage8_inflate", "boxes_source": boxes_source(s7)}
    assert boxes_source(s8) == "stage6_stereo_box"
    assert boxes_source({"stage": "stage6_cluster"}) == "stage6_cluster"
    # A pre-change manifest says "unknown" rather than naming the stage that
    # merely passed the boxes on.
    assert boxes_source({"stage": "stage7_track"}) == "unknown"


def test_every_tail_stage_records_the_producer():
    for rel in ("pipeline/stage7_track/track.py", "pipeline/stage8_inflate/inflate.py",
                "pipeline/stage9_qa/gate.py"):
        with open(os.path.join(ROOT, rel), "r", encoding="utf-8") as fh:
            text = fh.read()
        assert re.search(r'"boxes_source": boxes_source\(', text), rel
