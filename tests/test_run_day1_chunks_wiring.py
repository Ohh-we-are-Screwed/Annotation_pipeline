"""The producer/consumer contracts scripts/run_day1_chunks.sh has to honour.

There is no shell test harness in this repo, so the driver's data dependencies
are pinned by reading the script. Both rules here are final-review findings:

* **C1** — the driver runs `export_release` BEFORE `export_cvat_3d`, so the
  CVAT archive `--cvat-export-3d-dir` names does not exist yet on a fresh work
  root. `--stage1-dir` is what makes an interpolated point count ground-filtered
  anyway, and the Stage 1 cloud prune must stay AFTER the export that reads it.
"""

from __future__ import annotations

import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "run_day1_chunks.sh")


def _text() -> str:
    with open(SCRIPT, "r", encoding="utf-8") as fh:
        return fh.read()


def test_script_is_syntactically_valid():
    assert subprocess.run(["bash", "-n", SCRIPT], capture_output=True).returncode == 0


def test_every_export_release_call_passes_stage1_dir():
    text = _text()
    calls = [m for m in re.findall(r"scripts/export_release\.py(?:.|\n)*?\n\n", text)]
    assert calls, "no export_release.py invocation found"
    for call in calls:
        assert '--stage1-dir "$work/stage1_ingestion"' in call, call


def test_the_cloud_prune_runs_after_the_export_that_reads_the_clouds():
    text = _text()
    export_at = text.index("scripts/export_release.py")
    prune_at = text.index('rm -rf "$work/stage1_ingestion/clouds"')
    assert export_at < prune_at
    # ... and only on a clean export, so a failed chunk keeps its clouds.
    prune_guard = text[text.index('if [ "${PRUNE_CLOUDS:-1}" = 1 ]'):prune_at]
    assert "$rel_rc -eq 0" in prune_guard
