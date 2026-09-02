"""C30 — the CVAT publish naming contract.

Task names are the third provenance level: "<scene> <suffix> [<run tag>]".
These tests pin the two pure pieces the publish scripts share — task_name()
and the --replace-all-runs matcher wipe_targets() — because every deletion
either script ever performs is decided by them, against a live server with
no undo. The server-bound halves (cvat_setup.main, cvat_setup_3d.main) are
exercised only by a real publish.
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.cvat_setup import (  # noqa: E402
    category_disagreements,
    task_name,
    undeclared_labels,
    wipe_targets,
)
from scripts.cvat_setup_3d import GT_SUFFIX as GT_SUFFIX_3D  # noqa: E402
from scripts.cvat_setup_3d import OURS_SUFFIX as OURS_SUFFIX_3D  # noqa: E402

# The wrapper's defaults, verbatim (run_stages.sh reads them from env; these
# literals are the contract the server already holds tasks under).
SUFFIX = "— OUR PIPELINE output"
GT_SUFFIX = "— nuScenes HUMAN answer key"


class TestTaskName:
    def test_untagged_is_the_legacy_format_exactly(self):
        # Every task created before C30 carries this byte layout; an untagged
        # call must keep producing it or republishes stop being idempotent.
        assert task_name("chunk_0000", SUFFIX) == "chunk_0000 — OUR PIPELINE output"

    def test_tagged_appends_bracketed_tag(self):
        assert (task_name("chunk_0000", SUFFIX, "20260901T151325")
                == "chunk_0000 — OUR PIPELINE output [20260901T151325]")

    def test_same_run_same_name(self):
        # The tag derives from the source manifest, so a republish of the same
        # run must regenerate the identical name (and be skipped, not doubled).
        a = task_name("chunk_0000", SUFFIX, "20260901T151325")
        b = task_name("chunk_0000", SUFFIX, "20260901T151325")
        assert a == b


class TestReplaceTargetsThisRunOnly:
    """--replace deletes exactly the names THIS publish will create."""

    def test_other_runs_and_legacy_names_survive(self):
        existing = [
            "chunk_0000 — OUR PIPELINE output",                      # legacy, untagged
            "chunk_0000 — OUR PIPELINE output [20260830T090000]",    # an older run
            "chunk_0000 — OUR PIPELINE output [20260901T151325]",    # THIS run
            "chunk_0000 — nuScenes HUMAN answer key",                # the twin
        ]
        targets = {task_name(s, SUFFIX, "20260901T151325") for s in ["chunk_0000"]}
        doomed = [n for n in existing if n in targets]
        assert doomed == ["chunk_0000 — OUR PIPELINE output [20260901T151325]"]


class TestWipeTargetsAllRuns:
    """--replace-all-runs deletes every run's output for the suffix — and
    nothing published under any other suffix."""

    def test_matches_every_tag_and_the_legacy_untagged(self):
        existing = [
            "chunk_0000 — OUR PIPELINE output",
            "chunk_0000 — OUR PIPELINE output [20260830T090000]",
            "chunk_0000 — OUR PIPELINE output [20260901T151325]",
            "scene-0061 — OUR PIPELINE output [20260830T090000]",
        ]
        assert wipe_targets(existing, SUFFIX) == sorted(existing)

    def test_never_matches_the_answer_key(self):
        existing = [
            "chunk_0000 — nuScenes HUMAN answer key",
            "chunk_0000 3D — nuScenes HUMAN answer key",
        ]
        assert wipe_targets(existing, SUFFIX) == []
        assert wipe_targets(existing, OURS_SUFFIX_3D) == []

    def test_3d_suffixes_do_not_cross(self):
        ours_3d = ["chunk_0000 " + OURS_SUFFIX_3D]
        gt_3d = ["chunk_0000 " + GT_SUFFIX_3D]
        assert wipe_targets(ours_3d, OURS_SUFFIX_3D) == ours_3d
        assert wipe_targets(gt_3d, OURS_SUFFIX_3D) == []
        assert wipe_targets(ours_3d, GT_SUFFIX_3D) == []

    def test_trailing_junk_is_not_a_pipeline_name(self):
        existing = ["chunk_0000 — OUR PIPELINE output [20260830T090000] copy"]
        assert wipe_targets(existing, SUFFIX) == []

    def test_the_shared_suffix_hazard_is_real_and_scoped_by_project(self):
        # The 3D OURS name ends with the 2D suffix ("<scene> 3D — OUR PIPELINE
        # output"), so inside ONE project a 2D wipe WOULD claim it — that is
        # why exact-name --replace exists and why the wrapper keeps 2D and 3D
        # in separate projects. This test pins the behaviour so a change to
        # the matcher is made knowing it, not by accident.
        assert OURS_SUFFIX_3D.endswith(SUFFIX)
        lookalike = ["chunk_0000 " + OURS_SUFFIX_3D]
        assert wipe_targets(lookalike, SUFFIX) == lookalike


class TestUndeclaredLabels:
    """A publish into an existing project must refuse BEFORE creating tasks
    when the export carries categories the project's labels do not declare.
    CVAT rejects such an import only server-side, AFTER the task and its
    frames already stand — the first C28-superset publish (2026-09-02) left
    exactly such a broken half-task under the run's own name."""

    PROJECT = ["a car", "a truck", "a pedestrian"]

    def test_missing_categories_are_named_sorted(self):
        got = undeclared_labels(self.PROJECT, ["a car", "an auto rickshaw", "a rickshaw"])
        assert got == ["a rickshaw", "an auto rickshaw"]

    def test_full_coverage_is_empty(self):
        assert undeclared_labels(self.PROJECT, ["a car", "a truck"]) == []

    def test_extra_project_labels_are_not_flagged(self):
        # A project MAY declare more than this export uses (the C28 superset
        # project against an arm-A-only export): that is not a mismatch.
        assert undeclared_labels(self.PROJECT + ["a bus"], ["a car"]) == []


class TestCategoryDisagreements:
    """Every scene of one publish must carry the SAME category set. The export
    root accumulates scene dirs across runs and taxonomy eras, and the label
    universe is seeded from the alphabetically-first scene only — so a stale
    10-phrase-era scene either recreates a short-labelled project (import
    crash) or rides silently into a 12-phrase run's tasks."""

    V2 = ["a car", "a truck", "a pedestrian"]
    V3 = V2 + ["a rickshaw", "an auto rickshaw"]

    def test_identical_scenes_are_clean(self):
        assert category_disagreements([("a", self.V3), ("b", list(self.V3))]) == {}

    def test_stale_scene_is_named_with_the_delta(self):
        got = category_disagreements([("a", self.V3), ("stale", self.V2)])
        assert got == {"stale": {"missing": ["a rickshaw", "an auto rickshaw"],
                                 "extra": []}}

    def test_first_scene_is_the_reference_even_when_it_is_the_stale_one(self):
        got = category_disagreements([("stale", self.V2), ("b", self.V3)])
        assert got == {"b": {"missing": [],
                             "extra": ["a rickshaw", "an auto rickshaw"]}}

    def test_order_differences_are_not_disagreements(self):
        assert category_disagreements([("a", self.V2), ("b", sorted(self.V2, reverse=True))]) == {}
