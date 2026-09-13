"""Contracts for scripts/run_all_chunks.py — the 38-chunk parallel batch runner.

Nothing here runs a pipeline stage or touches the SSD: the four units that can
silently ruin a multi-day batch are pinned instead — the wrapper-stdout parser
(a misread status line is a chunk reported as finished when it refused), the
chunk map (a wrong global number sends a chunk's output to another chunk's
work root), config generation (a shared work_root would make the wrapper's
per-root flock serialise everything), and the status write + resume decision
(a torn status.json, or a re-run of a chunk already released).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "run_all_chunks", ROOT / "scripts" / "run_all_chunks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rac = _load()


# --- the wrapper's stdout ---------------------------------------------------
#
# Captured verbatim from scripts/run_stages.sh (the `=== $label`, `--- $label:`
# and `!!! $label:` forms of run_step, plus the two terminators). The parser
# reads these live to drive current_stage; anything it cannot classify must
# leave the record untouched rather than guess.
SAMPLE = """\
log: /mnt/hdd/dhakascenes/batch_20260912/11/work/logs/run_20260912_120000.log
steps: 0 1 3 3f 3m 4 5 6s 7 8 release   scenes: dhaka_20260911_141259_chunk_0010
exports: /media/ssd/exports/chunk_11

=== STAGE 0 (data probe: scene allowlist)  12:00:00
--- STAGE 0 (data probe: scene allowlist): OK  (31s)

=== STAGE 1 (ingestion: keyframe index + ground-filtered clouds)  12:00:31
--- STAGE 1 (ingestion: keyframe index + ground-filtered clouds): OK  (59s)

=== STAGE 3 (proposal_2d: Ultralytics/YOLO11)  12:01:30
--- STAGE 3 (proposal_2d: Ultralytics/YOLO11): OK  (106s)

=== STAGE 3f (proposal_2d arm B: local/armb)  12:03:16
--- STAGE 3f (proposal_2d arm B: local/armb): OK  (89s)

=== STAGE 3m (merge: stage3_proposals + stage3_finetuned)  12:04:45
--- STAGE 3m (merge: stage3_proposals + stage3_finetuned): OK  (4s)

=== STAGE 4 (mask_2d: facebook/sam3.1; proposals from stage3_merged)  12:04:49
--- STAGE 4 (mask_2d: facebook/sam3.1; proposals from stage3_merged): OK  (1502s)

=== STAGE 5 (2D->3D lift)  12:29:51
--- STAGE 5 (2D->3D lift): DEGRADED  (110s) — output is COMPLETE and quality-flagged;
    causes are in the _SUCCESS.degraded marker; downstream stages now get --accept-degraded-upstream

=== STAGE 6s (per-mask stereo boxes on the ZED frusta)  12:31:41
--- STAGE 6s (per-mask stereo boxes on the ZED frusta): OK  (54s)

=== STAGE 7 (track; reid local/reid)  12:32:35
--- STAGE 7 (track; reid local/reid): OK  (12s)

=== STAGE 8 (inflate against priors; boxes from stage7_track)  12:32:47
--- STAGE 8 (inflate against priors; boxes from stage7_track): OK  (8s)

=== STAGE 9 (QA gate for nuScenes release)  12:32:55
--- STAGE 9 (QA gate for nuScenes release): OK  (6s)

=== EXPORT nuScenes release (/media/ssd/exports/chunk_11/boxes)  12:33:01
--- EXPORT nuScenes release (/media/ssd/exports/chunk_11/boxes): OK  (243s)
=== ALL_STEPS_DONE 12:37:04
"""


def _feed(text, record=None):
    record = record if record is not None else rac.new_record(n=11)
    for line in text.splitlines():
        rac.parse_wrapper_line(line, record)
    return record


def test_parser_reads_every_stage_state_and_the_terminator():
    record = _feed(SAMPLE)
    assert record["log"].endswith("/logs/run_20260912_120000.log")
    assert record["stage_states"]["1"] == {"state": "ok", "seconds": 59}
    assert record["stage_states"]["4"] == {"state": "ok", "seconds": 1502}
    assert record["stage_states"]["3f"]["seconds"] == 89
    assert record["stage_states"]["6s"]["state"] == "ok"
    assert record["stage_states"]["5"]["state"] == "degraded"
    assert record["stage_states"]["9"]["state"] == "ok"
    assert record["stage_states"]["release"] == {"state": "ok", "seconds": 243}
    assert record["run_result"] == "done"
    # Nothing is left "running" once the run has ended.
    assert record["current_stage"] is None


def test_parser_tracks_the_running_stage_before_it_finishes():
    record = _feed(SAMPLE[:SAMPLE.index("--- STAGE 4")])
    assert record["current_stage"] == "4"
    assert record["stage_states"]["4"]["state"] == "running"
    assert "5" not in record["stage_states"]


def test_parser_classifies_refused_crashed_and_incomplete():
    record = _feed("""\
=== STAGE 4 (mask_2d: facebook/sam3.1)  12:00:00
!!! STAGE 4 (mask_2d: facebook/sam3.1): REFUSED rc=2 — the stage wrote NOTHING and left no marker.
ABORTED at: STAGE 4 (mask_2d: facebook/sam3.1)  (rc>=2 = REFUSED: nothing written, no marker)
=== RUN_INCOMPLETE 12:00:09
""")
    assert record["stage_states"]["4"]["state"] == "refused"
    assert record["run_result"] == "incomplete"

    record = _feed("!!! STAGE 5 (2D->3D lift): exited 1 with NO degraded marker in /w/stage5_lift.")
    assert record["stage_states"]["5"]["state"] == "crashed"

    record = _feed("!!! STAGE 5 (2D->3D lift): exited 0 but its marker is 'none', not 'clean'.")
    assert record["stage_states"]["5"]["state"] == "crashed"

    record = _feed("!!! EXPORT nuScenes release (/x/boxes): FAILED rc=1 — aborting the chain.")
    assert record["stage_states"]["release"]["state"] == "crashed"


def test_parser_ignores_noise():
    record = rac.new_record(n=1)
    before = json.dumps(record, sort_keys=True)
    for line in ("", "Traceback (most recent call last):", "--- not a stage: OK  (3s)",
                 "  File \"/x.py\", line 1, in <module>", "=== something else"):
        rac.parse_wrapper_line(line, record)
    assert json.dumps(record, sort_keys=True) == before


# --- the chunk map ----------------------------------------------------------

# The 13 tables pipeline/common/paths.py demands, so load_paths() accepts these
# fixtures and metadata_fingerprint() has something real to hash -- which is
# what author_priors_dhaka.py binds a priors file to.
METADATA_TABLES = ("attribute.json", "calibrated_sensor.json", "category.json",
                   "ego_pose.json", "instance.json", "log.json", "map.json",
                   "sample.json", "sample_annotation.json", "sample_data.json",
                   "scene.json", "sensor.json", "visibility.json")


def _mini_export(tmp_path, layout, salt=""):
    """layout: {session: ([chunk suffixes], [keyframes...], versions_present)}"""
    root = tmp_path / "export"
    for session, (names, counts, versions) in layout.items():
        for version in versions:
            vdir = root / session / "full" / version
            vdir.mkdir(parents=True)
            scenes = [{"token": f"{session}-{i}", "name": f"{session}_chunk_{name}",
                       "nbr_samples": counts[i]} for i, name in enumerate(names)]
            samples = [{"token": f"s{i}-{k}", "scene_token": f"{session}-{i}"}
                       for i, _ in enumerate(names) for k in range(counts[i])]
            for table in METADATA_TABLES:
                (vdir / table).write_text("[]")
            (vdir / "scene.json").write_text(json.dumps(scenes))
            (vdir / "sample.json").write_text(json.dumps(samples))
            if salt:                       # a different fingerprint, same shape
                (vdir / "log.json").write_text(json.dumps([{"salt": salt}]))
        for blobs in ("samples", "sweeps"):
            (root / session / "full" / blobs).mkdir(parents=True, exist_ok=True)
    return root


def _repo(tmp_path, wrapper=None):
    """A throwaway repo with a stub run_stages.sh but the REAL priors author.

    The author is not stubbed anywhere in this file: it is the thing that
    decides whether Stage 6s will accept the chunk, and a stub would only
    prove that the runner can call a stub.
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "run_stages.sh").write_text(wrapper if wrapper is not None else STUB)
    (repo / "configs").mkdir()
    (repo / "configs" / "paths_zami_20260911.yaml").write_text(yaml.safe_dump({
        "dataroot": "/replaced", "meta_root": "/replaced", "version": "v1.0-dhaka-fixed2",
        "work_root": str(tmp_path / "tmpl_work"), "out_root": str(tmp_path / "tmpl_out"),
        "probe_out_root": str(tmp_path / "tmpl_probe")}))
    (repo / "scripts" / "author_priors_dhaka.py").symlink_to(
        ROOT / "scripts" / "author_priors_dhaka.py")
    (repo / "configs" / "taxonomy_pilot_dhaka.yaml").symlink_to(
        ROOT / "configs" / "taxonomy_pilot_dhaka.yaml")
    (repo / "pipeline").symlink_to(ROOT / "pipeline")
    return repo


def test_chunk_map_numbers_chunks_globally_across_sessions(tmp_path):
    root = _mini_export(tmp_path, {
        "sess_a": (["0000", "0001", "0002"], [10, 20, 5], ["v1.0-dhaka", "v1.0-dhaka-fixed2"]),
        "sess_b": (["0000", "0001"], [7, 8], ["v1.0-dhaka", "v1.0-dhaka-fixed2"]),
    })
    chunks = rac.build_chunk_map(root, ["sess_a", "sess_b"])
    assert [c["n"] for c in chunks] == [1, 2, 3, 4, 5]
    assert [c["scene"] for c in chunks] == [
        "sess_a_chunk_0000", "sess_a_chunk_0001", "sess_a_chunk_0002",
        "sess_b_chunk_0000", "sess_b_chunk_0001"]
    assert [c["keyframes"] for c in chunks] == [10, 20, 5, 7, 8]
    assert chunks[3]["session"] == "sess_b"
    assert chunks[0]["dataroot"] == str(root / "sess_a" / "full")
    assert all(c["blocked"] is None for c in chunks)


def test_chunk_map_blocks_a_session_whose_fixed_version_is_missing(tmp_path):
    root = _mini_export(tmp_path, {
        "sess_a": (["0000"], [10], ["v1.0-dhaka", "v1.0-dhaka-fixed2"]),
        "sess_b": (["0000", "0001"], [7, 8], ["v1.0-dhaka"]),
    })
    chunks = rac.build_chunk_map(root, ["sess_a", "sess_b"])
    # The fallback is for LISTING only: the blocked chunks still get numbers,
    # scenes and keyframe counts so the dashboard shows the whole batch.
    assert [c["n"] for c in chunks] == [1, 2, 3]
    assert chunks[0]["blocked"] is None
    assert chunks[1]["blocked"] == "version missing"
    assert chunks[2]["keyframes"] == 8


def test_chunk_map_refuses_an_export_with_no_readable_version(tmp_path):
    root = _mini_export(tmp_path, {"sess_a": (["0000"], [1], [])})
    with pytest.raises(SystemExit):
        rac.build_chunk_map(root, ["sess_a"])


# --- generated per-chunk configs -------------------------------------------

def test_config_isolates_every_write_root_and_keeps_the_read_side(tmp_path):
    template = tmp_path / "paths.yaml"
    template.write_text(yaml.safe_dump({
        "dataroot": "/data/old/full", "meta_root": "/data/old/full",
        "version": "v1.0-dhaka-fixed2", "work_root": "/mnt/hdd/work",
        "out_root": "/mnt/hdd/out", "probe_out_root": "/mnt/hdd/probe"}))
    chunk = {"n": 7, "session": "sess_a", "scene": "sess_a_chunk_0006",
             "keyframes": 10, "dataroot": "/data/sess_a/full", "blocked": None}
    path = rac.write_chunk_config(template, chunk, tmp_path / "cfg", "/mnt/hdd/batch")
    assert path == tmp_path / "cfg" / "chunk_07.yaml"
    config = yaml.safe_load(path.read_text())
    assert config["dataroot"] == "/data/sess_a/full"
    assert config["meta_root"] == "/data/sess_a/full"
    assert config["version"] == "v1.0-dhaka-fixed2"
    assert config["work_root"] == "/mnt/hdd/batch/07/work"
    assert config["out_root"] == "/mnt/hdd/batch/07/out"
    assert config["probe_out_root"] == "/mnt/hdd/batch/07/probe"


def test_configs_never_share_a_work_root(tmp_path):
    template = tmp_path / "paths.yaml"
    template.write_text(yaml.safe_dump({
        "dataroot": "/d", "meta_root": "/d", "version": "v",
        "work_root": "/w", "out_root": "/o", "probe_out_root": "/p"}))
    roots = set()
    for n in (1, 2, 38):
        chunk = {"n": n, "session": "s", "scene": f"s_chunk_{n:04d}",
                 "keyframes": 1, "dataroot": "/d", "blocked": None}
        path = rac.write_chunk_config(template, chunk, tmp_path / "cfg", "/mnt/hdd/batch")
        roots.add(yaml.safe_load(path.read_text())["work_root"])
    assert len(roots) == 3


# --- status.json ------------------------------------------------------------

def test_status_write_is_atomic_and_leaves_no_temp_behind(tmp_path):
    path = tmp_path / "status.json"
    rac.write_json_atomic(path, {"chunks": [], "progress": "0/38"})
    rac.write_json_atomic(path, {"chunks": [{"n": 1}], "progress": "1/38"})
    assert json.loads(path.read_text())["progress"] == "1/38"
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]


def test_status_write_never_truncates_the_previous_file_on_failure(tmp_path):
    path = tmp_path / "status.json"
    rac.write_json_atomic(path, {"progress": "1/38"})

    class Unserialisable:
        pass

    with pytest.raises(TypeError):
        rac.write_json_atomic(path, {"progress": Unserialisable()})
    assert json.loads(path.read_text())["progress"] == "1/38"
    assert [p.name for p in tmp_path.iterdir()] == ["status.json"]


def test_resume_skips_only_a_released_and_done_chunk(tmp_path):
    exports = tmp_path / "exports"
    boxes = exports / "chunk_03" / "boxes"
    boxes.mkdir(parents=True)
    (boxes / "release_meta.json").write_text("{}")

    assert rac.should_skip({"n": 3, "state": "done"}, exports) is True
    # Released but the runner never recorded the chunk as finished: re-run and
    # let the wrapper's own markers decide what work is left.
    assert rac.should_skip({"n": 3, "state": "failed"}, exports) is False
    assert rac.should_skip({"n": 3, "state": "degraded"}, exports) is True
    # Recorded done, but no release on the SSD: the export is the deliverable.
    assert rac.should_skip({"n": 4, "state": "done"}, exports) is False
    assert rac.should_skip({"n": 3, "state": "running"}, exports) is False


def test_resume_reads_release_meta_from_the_exports_dir_and_suffix(tmp_path):
    ssd_exports = tmp_path / "ssd_exports"           # status/events live here
    release_root = tmp_path / "other_exports"         # --exports-dir
    boxes = release_root / "chunk_03_vlm" / "boxes"
    boxes.mkdir(parents=True)
    (boxes / "release_meta.json").write_text("{}")

    # Looking under the status dir (no suffix) must NOT find it.
    assert rac.should_skip({"n": 3, "state": "done"}, ssd_exports) is False
    assert rac.should_skip({"n": 3, "state": "done"}, release_root, "_vlm") is True
    assert rac.should_skip({"n": 3, "state": "done"}, release_root, "") is False


def test_export_stats_counts_files_bytes_and_symlinks(tmp_path):
    boxes = tmp_path / "chunk_01" / "boxes"
    boxes.mkdir(parents=True)
    (boxes / "release_meta.json").write_text("x" * 10)
    (boxes / "samples").mkdir()
    (boxes / "samples" / "a.bin").write_text("y" * 5)
    os.symlink(boxes / "samples" / "a.bin", boxes / "samples" / "link.bin")
    stats = rac.export_stats(tmp_path / "chunk_01")
    assert stats["files"] == 3
    assert stats["bytes"] == 15
    assert stats["symlinks"] == 1


# --- CLI plumbing -----------------------------------------------------------

def test_chunk_selection_accepts_ranges_and_lists():
    assert rac.parse_chunks("1-5") == [1, 2, 3, 4, 5]
    assert rac.parse_chunks("1,5,7") == [1, 5, 7]
    assert rac.parse_chunks("1-3,38") == [1, 2, 3, 38]
    with pytest.raises(ValueError):
        rac.parse_chunks("5-1")


def test_dotenv_parses_plain_assignments_and_secrets_are_masked(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\n\nHF_TOKEN=hf_supersecret\nexport OMP_NUM_THREADS=4\n"
                        "CVAT_PASSWORD=hunter2\nHF_HOME=/mnt/hdd/hf\n"
                        "TOKENIZERS_PARALLELISM=false\nBROKEN LINE\n")
    env = rac.load_dotenv(env_file)
    assert env == {"HF_TOKEN": "hf_supersecret", "OMP_NUM_THREADS": "4",
                   "CVAT_PASSWORD": "hunter2", "HF_HOME": "/mnt/hdd/hf",
                   "TOKENIZERS_PARALLELISM": "false"}
    masked = rac.mask_env(env)
    assert masked["HF_TOKEN"] == "***"
    assert masked["CVAT_PASSWORD"] == "***"
    assert masked["HF_HOME"] == "/mnt/hdd/hf"
    # A setting that merely spells "token" is not a secret; masking it would
    # make the --dry-run printout lie about the environment a chunk gets.
    assert masked["TOKENIZERS_PARALLELISM"] == "false"
    assert "hf_supersecret" not in json.dumps(masked)


# --- the queue end to end ---------------------------------------------------
#
# The riskiest code here is not a regex: it is a thread pool reading a
# subprocess's stdout while rewriting a shared status file. This runs the real
# Batch against a STUB wrapper that prints the real wrapper's lines, so the
# plumbing (env, cwd, live parse, markers, export verification, resume) is
# exercised without a GPU.

STUB = """#!/bin/bash
set -e
scene="${@: -2:1}"
work="$(grep '^work_root:' "$DHAKASCENES_PATHS_CONFIG" | cut -d' ' -f2)"
mkdir -p "$work/logs" "$work/stage1_ingestion" "$work/stage5_lift"
echo "$@" > "$work/argv.txt"
touch "$work/stage1_ingestion/_SUCCESS"
echo '{"causes": ["ego_motion_unavailable"]}' > "$work/stage5_lift/_SUCCESS.degraded"
echo "log: $work/logs/run_stub.log"
echo "=== STAGE 1 (ingestion: keyframe index + ground-filtered clouds)  00:00:00"
echo "--- STAGE 1 (ingestion: keyframe index + ground-filtered clouds): OK  (7s)"
echo "=== STAGE 5 (2D->3D lift)  00:00:07"
echo "--- STAGE 5 (2D->3D lift): DEGRADED  (3s) — output is COMPLETE and quality-flagged;"
if [ "$STUB_FAIL" = "$scene" ]; then
  echo "!!! EXPORT nuScenes release ($EXPORT_ROOT/$EXPORT_NAME/boxes): FAILED rc=1 — aborting the chain." >&2
  echo "=== RUN_INCOMPLETE 00:00:09"
  exit 1
fi
mkdir -p "$EXPORT_ROOT/$EXPORT_NAME/boxes"
echo '{"stub": true}' > "$EXPORT_ROOT/$EXPORT_NAME/boxes/release_meta.json"
echo "=== EXPORT nuScenes release ($EXPORT_ROOT/$EXPORT_NAME/boxes)  00:00:09"
echo "--- EXPORT nuScenes release ($EXPORT_ROOT/$EXPORT_NAME/boxes): OK  (11s)"
echo "=== ALL_STEPS_DONE 00:00:20"
"""


@pytest.fixture()
def stub_batch(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / ".env").write_text("HF_TOKEN=hf_supersecret\n")
    export_root = _mini_export(tmp_path, {
        "sess_a": (["0000", "0001"], [30, 10], ["v1.0-dhaka-fixed2"])})
    monkeypatch.setattr(rac, "SESSIONS", ("sess_a",))
    return repo, export_root, tmp_path / "ssd"


def _argv(repo, export_root, ssd, tmp_path, *extra):
    return ["--repo", str(repo), "--export-root", str(export_root), "--ssd", str(ssd),
            "--batch-root", str(tmp_path / "batch"), "--workers", "2",
            "--stagger", "0", "--steps", "1 5 release", "--chunks", "1-2", *extra]


def test_batch_runs_both_chunks_and_records_markers_and_exports(stub_batch, tmp_path):
    repo, export_root, ssd = stub_batch
    assert rac.main(_argv(repo, export_root, ssd, tmp_path)) == 0

    status = json.loads((ssd / "exports" / "status.json").read_text())
    assert status["progress"] == "2/2"
    assert status["stages"] == ["1", "5", "9", "release"]
    assert [c["state"] for c in status["chunks"]] == ["degraded", "degraded"]  # Stage 5 marker
    first = status["chunks"][0]
    assert first["stage_states"]["1"] == {"state": "ok", "seconds": 7}
    assert first["stage_states"]["5"]["causes"] == ["ego_motion_unavailable"]
    assert first["stage_states"]["release"]["state"] == "ok"
    assert first["export"] == {"files": 1, "bytes": 15, "symlinks": 0}
    assert first["log"].endswith("run_stub.log")
    assert first["worker"] in (0, 1) and first["finished"]
    events = [json.loads(l) for l in (ssd / "exports" / "events.jsonl").read_text().splitlines()]
    # Both chunks ran; their event ORDER is not asserted here because two
    # workers write events.jsonl concurrently (see the longest-first test).
    assert sorted(e["n"] for e in events if e["event"] == "chunk_start") == [1, 2]
    # manifest.json is the static map, and no secret reached the SSD.
    manifest = json.loads((ssd / "exports" / "manifest.json").read_text())
    assert [c["scene"] for c in manifest["chunks"]] == ["sess_a_chunk_0000", "sess_a_chunk_0001"]
    assert "hf_supersecret" not in (ssd / "exports" / "status.json").read_text()


def test_the_queue_dispatches_the_longest_chunk_first(tmp_path, monkeypatch):
    """The tail of a batch is only as short as its longest remaining chunk, so
    the big ones go first. One worker, so the dispatch order IS the event order.
    Chunk 2 is the long one here, i.e. the opposite of numeric order."""
    repo = _repo(tmp_path)
    export_root = _mini_export(tmp_path, {
        "sess_a": (["0000", "0001"], [10, 30], ["v1.0-dhaka-fixed2"])})
    monkeypatch.setattr(rac, "SESSIONS", ("sess_a",))
    ssd = tmp_path / "ssd"
    assert rac.main(["--repo", str(repo), "--export-root", str(export_root), "--ssd", str(ssd),
                     "--batch-root", str(tmp_path / "batch"), "--workers", "1",
                     "--stagger", "0", "--steps", "1 release", "--chunks", "1-2"]) == 0
    events = [json.loads(l) for l in (ssd / "exports" / "events.jsonl").read_text().splitlines()]
    assert [e["n"] for e in events if e["event"] == "chunk_start"] == [2, 1]


def test_a_failed_chunk_is_recorded_and_the_queue_continues(stub_batch, tmp_path, monkeypatch):
    repo, export_root, ssd = stub_batch
    monkeypatch.setenv("STUB_FAIL", "sess_a_chunk_0000")
    assert rac.main(_argv(repo, export_root, ssd, tmp_path)) == 1
    chunks = json.loads((ssd / "exports" / "status.json").read_text())["chunks"]
    assert chunks[0]["state"] == "failed"
    assert chunks[0]["stage_states"]["release"]["state"] == "crashed"
    assert any("FAILED rc=1" in line for line in chunks[0]["error_tail"])
    assert chunks[1]["state"] == "degraded"          # the queue carried on


def test_a_second_run_resumes_past_the_released_chunks(stub_batch, tmp_path, capsys):
    repo, export_root, ssd = stub_batch
    rac.main(_argv(repo, export_root, ssd, tmp_path))
    capsys.readouterr()
    rac.main(_argv(repo, export_root, ssd, tmp_path))
    out = capsys.readouterr().out
    assert "resume: chunk 01 already released" in out
    assert "resume: chunk 02 already released" in out
    assert "0 chunks queued" in out


def test_blocked_chunks_are_never_dispatched(tmp_path, monkeypatch):
    repo = _repo(tmp_path, wrapper="#!/bin/bash\nexit 9\n")
    export_root = _mini_export(tmp_path, {"sess_a": (["0000"], [5], ["v1.0-dhaka"])})
    monkeypatch.setattr(rac, "SESSIONS", ("sess_a",))
    ssd = tmp_path / "ssd"
    assert rac.main(["--repo", str(repo), "--export-root", str(export_root), "--ssd", str(ssd),
                     "--batch-root", str(tmp_path / "batch"), "--workers", "1",
                     "--stagger", "0", "--chunks", "1"]) == 0
    chunk = json.loads((ssd / "exports" / "status.json").read_text())["chunks"][0]
    assert chunk["state"] == "blocked" and chunk["blocked"] == "version missing"
    assert chunk["started"] is None


# --- step-level resume ------------------------------------------------------
#
# scripts/run_stages.sh runs every step it is given, unconditionally
# (its case arm at :1095-1305 has no marker check). So resuming a chunk that
# died at Stage 4 means the RUNNER must shorten the step list, or Stages 0-3
# are recomputed for hours.

def _markers(work, *stages, degraded=()):
    for stage in stages:
        d = Path(work) / rac.STAGE_DIRS[stage]
        d.mkdir(parents=True, exist_ok=True)
        (d / "_SUCCESS").write_text("")
    for stage in degraded:
        d = Path(work) / rac.STAGE_DIRS[stage]
        d.mkdir(parents=True, exist_ok=True)
        (d / "_SUCCESS.degraded").write_text('{"causes": ["x"]}')
    return work


STEPS = "0 1 3 3f 3m 4 5 6s 7 8 release".split()


def test_steps_to_run_starts_at_the_first_missing_marker(tmp_path):
    work = _markers(tmp_path / "work", "0", "1", "3")
    assert rac.steps_to_run(STEPS, work, False) == (
        ["3f", "3m", "4", "5", "6s", "7", "8", "release"], ["0", "1", "3"])


def test_steps_to_run_treats_a_degraded_marker_as_done(tmp_path):
    work = _markers(tmp_path / "work", "0", "1", degraded=["3"])
    run, skipped = rac.steps_to_run(STEPS, work, False)
    assert skipped == ["0", "1", "3"]
    assert run[0] == "3f"


def test_steps_to_run_never_skips_past_a_hole(tmp_path):
    # Stage 4 has a marker but Stage 3f does not: restart at 3f and redo 4.
    work = _markers(tmp_path / "work", "0", "1", "3", "4")
    run, skipped = rac.steps_to_run(STEPS, work, False)
    assert skipped == ["0", "1", "3"]
    assert "4" in run


def test_steps_to_run_on_a_fresh_work_root_runs_everything(tmp_path):
    assert rac.steps_to_run(STEPS, tmp_path / "nothing", False) == (STEPS, [])


def test_steps_to_run_uses_the_release_artefact_for_the_release_step(tmp_path):
    work = _markers(tmp_path / "work", *[s for s in STEPS if s != "release"])
    assert rac.steps_to_run(STEPS, work, False) == (["release"], STEPS[:-1])
    assert rac.steps_to_run(STEPS, work, True) == ([], STEPS)


def test_the_road_step_is_in_the_default_chain_and_resumes_on_stage_road(tmp_path):
    """`road` joined the default chain (it feeds the release's road/ layer and
    the lidarseg road labels), so a resume must skip it on ITS marker rather
    than pay for SAM 3 over every keyframe again."""
    steps = rac.DEFAULT_STEPS.split()
    assert steps.index("road") == steps.index("release") - 1, "road before release"
    work = _markers(tmp_path / "work", *[s for s in steps if s != "release"])
    assert (Path(work) / "stage_road" / "_SUCCESS").exists()
    assert rac.steps_to_run(steps, work, False) == (["release"], steps[:-1])


def test_a_stage_road_without_a_marker_reruns_road_even_after_release(tmp_path):
    steps = rac.DEFAULT_STEPS.split()
    work = _markers(tmp_path / "work", *[s for s in steps if s not in ("road", "release")])
    assert rac.steps_to_run(steps, work, True) == (["road", "release"], steps[:-2])


def test_the_overlay_lets_the_road_step_share_the_gpu(tmp_path):
    """SAM 3 for road wants ~6.5 GB and a batch with several chunks in flight
    never has an idle GPU, so the runner opts in to the sharing the wrapper
    refuses by default."""
    from types import SimpleNamespace
    fake = SimpleNamespace(args=SimpleNamespace(py="/py"), configs={14: tmp_path / "c14.yaml"},
                           release_root=tmp_path / "exports", export_suffix="")
    assert rac.Batch.overlay(fake, {"n": 14})["ROAD_ALLOW_SHARED_GPU"] == "1"


def test_batch_shortens_the_wrapper_invocation_for_a_resumed_chunk(stub_batch, tmp_path):
    repo, export_root, ssd = stub_batch
    work = tmp_path / "batch" / "01" / "work"
    _markers(work, "0", "1", "3")
    (work / "argv.txt").parent.mkdir(parents=True, exist_ok=True)
    argv = _argv(repo, export_root, ssd, tmp_path, "--chunks", "1")
    argv[argv.index("--steps") + 1] = " ".join(STEPS)
    assert rac.main(argv) == 0
    assert (work / "argv.txt").read_text().split() == [
        "3f", "3m", "4", "5", "6s", "7", "8", "release",
        "--scenes", "sess_a_chunk_0000", "--no-cvat"]
    chunk = json.loads((ssd / "exports" / "status.json").read_text())["chunks"][0]
    assert chunk["steps_skipped_by_marker"] == ["0", "1", "3"]
    assert chunk["steps_run"][0] == "3f"
    # The skipped stages are not shown as pending: their markers are read in.
    assert chunk["stage_states"]["0"]["state"] == "ok"
    events = [json.loads(l) for l in (ssd / "exports" / "events.jsonl").read_text().splitlines()]
    assert [e for e in events if e["event"] == "chunk_start"][0]["steps_run"][0] == "3f"


def test_no_step_resume_forces_the_full_list(stub_batch, tmp_path):
    repo, export_root, ssd = stub_batch
    work = tmp_path / "batch" / "01" / "work"
    _markers(work, "0", "1", "3")
    argv = _argv(repo, export_root, ssd, tmp_path, "--chunks", "1", "--no-step-resume")
    argv[argv.index("--steps") + 1] = " ".join(STEPS)
    assert rac.main(argv) == 0
    assert (work / "argv.txt").read_text().split()[:4] == ["0", "1", "3", "3f"]


def test_a_fully_marked_chunk_is_finished_without_running_the_wrapper(stub_batch, tmp_path):
    repo, export_root, ssd = stub_batch
    work = _markers(tmp_path / "batch" / "01" / "work", *[s for s in STEPS if s != "release"])
    boxes = ssd / "exports" / "chunk_01" / "boxes"
    boxes.mkdir(parents=True)
    (boxes / "release_meta.json").write_text("{}")
    argv = _argv(repo, export_root, ssd, tmp_path, "--chunks", "1", "--no-resume")
    argv[argv.index("--steps") + 1] = " ".join(STEPS)
    assert rac.main(argv) == 0
    assert not (work / "argv.txt").exists()          # the wrapper never ran
    chunk = json.loads((ssd / "exports" / "status.json").read_text())["chunks"][0]
    assert chunk["state"] == "done" and chunk["steps_run"] == []


# --- concurrency ------------------------------------------------------------

class _Guarded(dict):
    """A dict that records every mutation made while `held()` is false."""

    def __init__(self, source, held, violations):
        super().__init__(source)
        self.held, self.violations = held, violations

    def _check(self, what):
        if not self.held():
            self.violations.append(what)

    def __setitem__(self, key, value):
        self._check(key)
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):       # dict.update skips __setitem__
        self._check(sorted(kwargs) or args)
        super().update(*args, **kwargs)


class _WatchedLock:
    """threading.RLock that knows whether this thread is inside it."""

    def __init__(self):
        self._lock = threading.RLock()
        self.depth = 0

    def __enter__(self):
        self._lock.acquire()
        self.depth += 1

    def __exit__(self, *exc):
        self.depth -= 1
        self._lock.release()


def test_every_record_mutation_happens_under_the_status_lock(tmp_path):
    """flush() serialises the records with json.dumps, which ITERATES them. A
    stage key inserted by another worker mid-iteration is a RuntimeError that
    would mark a chunk failed while its wrapper runs on, detached and holding
    the flock that the re-dispatch then blocks on. So: no mutation outside the
    lock, including the ones dict.update() makes without __setitem__."""
    work = tmp_path / "work"
    _markers(work, "1")
    config = tmp_path / "chunk_01.yaml"
    config.write_text(yaml.safe_dump({"work_root": str(work)}))
    args = argparse.Namespace(ssd=str(tmp_path), workers=1, steps="1 release",
                              repo=str(tmp_path), stagger=0, py="python",
                              dotenv={}, step_resume=True)
    batch = rac.Batch(args, [{"n": 1, "session": "s", "scene": "s_chunk_0000",
                              "keyframes": 1, "blocked": None}])
    batch.exports.mkdir(parents=True, exist_ok=True)
    batch.configs[1] = config
    batch.lock = _WatchedLock()
    violations: list = []
    held = lambda: batch.lock.depth > 0
    record = _Guarded(batch.records[1], held, violations)
    record["stage_states"] = _Guarded({}, held, violations)
    violations.clear()                       # the two lines above are the setup
    batch.records[1] = record
    batch.command = lambda r: ["bash", "-c", (
        'echo "log: /x/run.log";'
        'echo "=== STAGE 1 (ingestion)  00:00:00";'
        'echo "--- STAGE 1 (ingestion): OK  (7s)";'
        'echo "=== ALL_STEPS_DONE 00:00:08"')]

    rc, tail = batch.pump(record)
    batch.finish(record, rc, tail)

    assert violations == []
    assert record["stage_states"]["1"] == {"state": "ok", "seconds": 7}
    assert record["rc"] == 0


def test_a_runner_side_failure_kills_the_wrapper_it_can_no_longer_follow(tmp_path, monkeypatch):
    """run_chunk marks the chunk failed on any exception. A wrapper left alive
    after that holds the flock on a work root the record says is idle, so the
    next --resume would block on it forever."""
    args = argparse.Namespace(ssd=str(tmp_path), workers=1, steps="1", repo=str(tmp_path),
                              stagger=0, py="python", dotenv={}, step_resume=True)
    batch = rac.Batch(args, [{"n": 1, "session": "s", "scene": "s_chunk_0000",
                              "keyframes": 1, "blocked": None}])
    batch.exports.mkdir(parents=True, exist_ok=True)
    batch.configs[1] = tmp_path / "chunk_01.yaml"
    pidfile = tmp_path / "pgid"
    batch.command = lambda r: ["bash", "-c", f"echo $$ > {pidfile}; echo hello; sleep 120"]
    monkeypatch.setattr(rac, "parse_wrapper_line",
                        lambda line, record: (_ for _ in ()).throw(OSError("SSD gone")))
    started = time.monotonic()
    with pytest.raises(OSError):
        batch.pump(batch.records[1])
    assert time.monotonic() - started < 30          # not waiting the `sleep` out
    # start_new_session makes the wrapper its own process group; signal 0 says
    # whether anything at all is left in it, `sleep` included.
    pgid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.killpg(pgid, 0)


# --- the environment a chunk actually gets ----------------------------------
#
# The real .env sets PYTHONNOUSERSITE, DHAKASCENES_SUBSTRATE and
# DHAKASCENES_PATHS_CONFIG, all of which the per-chunk overlay sets too. Built
# as dict(base, **dotenv, **overlay) that is a TypeError -- two ** expansions
# sharing a key -- and it killed the first chunk of the first launch. The dry
# run had used dict(dotenv, **overlay) instead, where a positional mapping
# makes the collision legal, so it could not have caught it. One code path now.

COLLIDING_DOTENV = {"PYTHONNOUSERSITE": "1", "DHAKASCENES_PATHS_CONFIG": "x",
                    "DHAKASCENES_SUBSTRATE": "dhaka", "OMP_NUM_THREADS": "8",
                    "HF_TOKEN": "hf_supersecret"}


def test_the_overlay_wins_over_dotenv_without_a_keyword_collision(tmp_path):
    args = argparse.Namespace(ssd=str(tmp_path), workers=1, steps="1", repo=str(tmp_path),
                              stagger=0, py="/env/bin/python", dotenv=dict(COLLIDING_DOTENV),
                              step_resume=True)
    batch = rac.Batch(args, [{"n": 7, "session": "s", "scene": "s_chunk_0006",
                              "keyframes": 1, "blocked": None}])
    batch.configs[7] = tmp_path / "chunk_07.yaml"
    env = batch.env_for(batch.records[7])            # must not raise
    # the overlay wins...
    assert env["DHAKASCENES_PATHS_CONFIG"] == str(tmp_path / "chunk_07.yaml")
    assert env["DHAKASCENES_SUBSTRATE"] == "dhaka6"
    assert env["PYTHONNOUSERSITE"] == "1"
    assert env["EXPORT_NAME"] == "chunk_07"
    assert env["PY"] == "/env/bin/python"
    # ...over .env, which still supplies everything the overlay does not set...
    assert env["OMP_NUM_THREADS"] == "8"
    assert env["HF_TOKEN"] == "hf_supersecret"
    # ...and the process environment is still underneath both.
    assert env["PATH"] == os.environ["PATH"]


def test_overlay_applies_exports_dir_and_export_suffix(tmp_path):
    args = argparse.Namespace(ssd=str(tmp_path / "ssd"), workers=1, steps="1",
                              repo=str(tmp_path), stagger=0, py="/env/bin/python",
                              dotenv={}, step_resume=True,
                              exports_dir=str(tmp_path / "custom_exports"),
                              export_suffix="_vlm")
    batch = rac.Batch(args, [{"n": 14, "session": "s", "scene": "s_chunk_0013",
                              "keyframes": 1, "blocked": None}])
    batch.configs[14] = tmp_path / "chunk_14.yaml"
    env = batch.env_for(batch.records[14])
    assert env["EXPORT_NAME"] == "chunk_14_vlm"
    assert env["EXPORT_ROOT"] == str(tmp_path / "custom_exports")
    # status/events still live under --ssd, unaffected by --exports-dir
    assert batch.exports == tmp_path / "ssd" / "exports"


def test_overlay_defaults_are_unchanged_without_the_new_flags(tmp_path):
    args = argparse.Namespace(ssd=str(tmp_path), workers=1, steps="1", repo=str(tmp_path),
                              stagger=0, py="/env/bin/python", dotenv={}, step_resume=True)
    batch = rac.Batch(args, [{"n": 7, "session": "s", "scene": "s_chunk_0006",
                              "keyframes": 1, "blocked": None}])
    batch.configs[7] = tmp_path / "chunk_07.yaml"
    env = batch.env_for(batch.records[7])
    assert env["EXPORT_NAME"] == "chunk_07"
    assert env["EXPORT_ROOT"] == str(tmp_path / "exports")


def test_dry_run_builds_the_env_through_the_same_path(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "run_stages.sh").write_text(STUB)
    (repo / "configs").mkdir()
    (repo / "configs" / "paths_zami_20260911.yaml").write_text(yaml.safe_dump({
        "dataroot": "/r", "meta_root": "/r", "version": "v1.0-dhaka-fixed2",
        "work_root": "/w", "out_root": "/o", "probe_out_root": "/p"}))
    (repo / ".env").write_text(
        "".join(f"{k}={v}\n" for k, v in COLLIDING_DOTENV.items()))
    export_root = _mini_export(tmp_path, {"sess_a": (["0000"], [5], ["v1.0-dhaka-fixed2"])})
    monkeypatch.setattr(rac, "SESSIONS", ("sess_a",))
    assert rac.main(["--repo", str(repo), "--export-root", str(export_root),
                     "--ssd", str(tmp_path / "ssd"), "--batch-root", str(tmp_path / "batch"),
                     "--dry-run", "--chunks", "1"]) == 0
    env_line = [l for l in capsys.readouterr().out.splitlines() if "env:" in l][0]
    assert "DHAKASCENES_SUBSTRATE=dhaka6" in env_line      # the overlay's value, not .env's
    assert "DHAKASCENES_SUBSTRATE=dhaka " not in env_line
    assert "OMP_NUM_THREADS=8" in env_line                 # .env-only keys are still shown
    assert "HF_TOKEN=***" in env_line and "hf_supersecret" not in env_line


def test_dry_run_prints_the_resolved_export_root_and_name_with_suffix(tmp_path, monkeypatch, capsys):
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "run_stages.sh").write_text(STUB)
    (repo / "configs").mkdir()
    (repo / "configs" / "paths_zami_20260911.yaml").write_text(yaml.safe_dump({
        "dataroot": "/r", "meta_root": "/r", "version": "v1.0-dhaka-fixed2",
        "work_root": "/w", "out_root": "/o", "probe_out_root": "/p"}))
    export_root = _mini_export(tmp_path, {"sess_a": (["0000"], [5], ["v1.0-dhaka-fixed2"])})
    monkeypatch.setattr(rac, "SESSIONS", ("sess_a",))
    exports_dir = tmp_path / "custom_exports"
    assert rac.main(["--repo", str(repo), "--export-root", str(export_root),
                     "--ssd", str(tmp_path / "ssd"), "--batch-root", str(tmp_path / "batch"),
                     "--exports-dir", str(exports_dir), "--export-suffix", "_vlm",
                     "--dry-run", "--chunks", "1"]) == 0
    env_line = [l for l in capsys.readouterr().out.splitlines() if "env:" in l][0]
    assert f"EXPORT_ROOT={exports_dir}" in env_line
    assert "EXPORT_NAME=chunk_01_vlm" in env_line


def test_batch_with_exports_dir_and_suffix_releases_and_records_globals(stub_batch, tmp_path):
    repo, export_root, ssd = stub_batch
    exports_dir = tmp_path / "custom_exports"
    assert rac.main(_argv(repo, export_root, ssd, tmp_path,
                          "--exports-dir", str(exports_dir), "--export-suffix", "_vlm")) == 0

    # release landed at <exports-dir>/chunk_NN_vlm/boxes, not under --ssd
    assert (exports_dir / "chunk_01_vlm" / "boxes" / "release_meta.json").exists()
    assert not (ssd / "exports" / "chunk_01_vlm").exists()

    # status/events/manifest still live under --ssd/exports
    status = json.loads((ssd / "exports" / "status.json").read_text())
    assert status["exports_dir"] == str(exports_dir)
    assert status["export_suffix"] == "_vlm"
    manifest = json.loads((ssd / "exports" / "manifest.json").read_text())
    assert manifest["exports_dir"] == str(exports_dir)
    assert manifest["export_suffix"] == "_vlm"
    first = status["chunks"][0]
    assert first["export"] == {"files": 1, "bytes": 15, "symlinks": 0}

# --- class priors -----------------------------------------------------------
#
# Stage 6s and Stage 8 read <out_root>/priors/priors_pilot_v0.json, REFUSE
# without it, and REFUSE one whose derived_from.metadata_fingerprint is not
# this dataroot's. Nothing in run_stages.sh writes it ("an INPUT to Stage 6,
# not an output of it"), every chunk here has a fresh out_root, and the four
# sessions have four fingerprints — so a file copied from one session is
# refused by the other three, three quarters of an hour into the chunk.
#
# These tests run the REAL scripts/author_priors_dhaka.py against synthetic
# datarootsthat pipeline.common.paths accepts. Stubbing it would only prove the
# runner can call a stub; what matters is that what lands on disk is what the
# pipeline's own loader and fingerprint check will accept.

sys.path.insert(0, str(ROOT))
from pipeline.common.paths import load_paths, metadata_fingerprint  # noqa: E402
from pipeline.stage6_cluster.priors import load_priors  # noqa: E402


def _priors_setup(tmp_path, salt=""):
    repo = _repo(tmp_path)
    export_root = _mini_export(tmp_path, {
        "sess_a": (["0000"], [10], ["v1.0-dhaka-fixed2"])}, salt=salt)
    chunk = {"n": 1, "session": "sess_a", "scene": "sess_a_chunk_0000", "keyframes": 10,
             "dataroot": str(export_root / "sess_a" / "full"), "blocked": None}
    config = rac.write_chunk_config(repo / "configs" / "paths_zami_20260911.yaml",
                                    chunk, tmp_path / "cfg", str(tmp_path / "batch"))
    return repo, config


def test_priors_are_authored_bound_to_this_chunks_own_dataroot(tmp_path):
    repo, config = _priors_setup(tmp_path)
    state, fingerprint = rac.provision_priors(config, repo, sys.executable)
    assert state == "authored"
    target = rac.priors_path(config)
    assert target.is_file()
    # the fingerprint the runner recorded, the one in the file, and the one the
    # pipeline computes for this dataroot are the same number
    assert fingerprint == metadata_fingerprint(load_paths(config))
    assert rac.priors_fingerprint(target) == fingerprint
    # and the pipeline's own reader accepts what was written
    priors = load_priors(str(target))
    assert priors.metadata_fingerprint == fingerprint
    assert priors.get("a car") is not None


def test_a_second_pass_leaves_a_correctly_bound_file_byte_identical(tmp_path):
    repo, config = _priors_setup(tmp_path)
    rac.provision_priors(config, repo, sys.executable)
    target = rac.priors_path(config)
    before = target.read_bytes()
    state, fingerprint = rac.provision_priors(config, repo, sys.executable)
    assert state == "present"
    assert target.read_bytes() == before
    assert fingerprint == metadata_fingerprint(load_paths(config))


def test_priors_bound_to_another_dataroot_are_rebound(tmp_path):
    """What the copy-one-file-everywhere approach produces, and what Stage 6s
    refuses: the file is there, but it carries another session's fingerprint."""
    other_repo, other_config = _priors_setup(tmp_path / "other", salt="another session")
    rac.provision_priors(other_config, other_repo, sys.executable)
    stale = rac.priors_path(other_config).read_text()

    repo, config = _priors_setup(tmp_path)
    target = rac.priors_path(config)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(stale)                         # the mis-bound copy
    mine = metadata_fingerprint(load_paths(config))
    assert rac.priors_fingerprint(target) != mine    # Stage 6s would refuse this

    state, fingerprint = rac.provision_priors(config, repo, sys.executable)
    assert state == "rebound"
    assert fingerprint == mine and rac.priors_fingerprint(target) == mine
    history = json.loads(target.read_text())["derived_from"]["REBOUND"]["rebound_history"]
    assert history[-1]["to"] == mine                 # the rebind is recorded, not silent
    assert load_priors(str(target)).classes.keys() == load_priors(
        str(rac.priors_path(other_config))).classes.keys()   # same values, new binding


def test_a_chunk_whose_priors_cannot_be_authored_is_blocked_not_dispatched(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "scripts" / "author_priors_dhaka.py").unlink()
    (repo / "scripts" / "author_priors_dhaka.py").write_text(
        "import sys; print('no taxonomy', file=sys.stderr); sys.exit(2)\n")
    export_root = _mini_export(tmp_path, {"sess_a": (["0000"], [10], ["v1.0-dhaka-fixed2"])})
    monkeypatch.setattr(rac, "SESSIONS", ("sess_a",))
    ssd = tmp_path / "ssd"
    assert rac.main(["--repo", str(repo), "--export-root", str(export_root), "--ssd", str(ssd),
                     "--batch-root", str(tmp_path / "batch"), "--workers", "1", "--stagger", "0",
                     "--steps", "1 release", "--chunks", "1", "--py", sys.executable]) == 0
    chunk = json.loads((ssd / "exports" / "status.json").read_text())["chunks"][0]
    assert chunk["state"] == "blocked" and chunk["blocked"] == "priors"
    assert chunk["started"] is None                  # the wrapper never ran
    assert "rc=2" in chunk["error_tail"][0]


def test_the_batch_authors_priors_for_every_chunk_before_dispatch(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    export_root = _mini_export(tmp_path, {
        "sess_a": (["0000", "0001"], [10, 20], ["v1.0-dhaka-fixed2"])})
    monkeypatch.setattr(rac, "SESSIONS", ("sess_a",))
    ssd = tmp_path / "ssd"
    assert rac.main(["--repo", str(repo), "--export-root", str(export_root), "--ssd", str(ssd),
                     "--batch-root", str(tmp_path / "batch"), "--workers", "1", "--stagger", "0",
                     "--steps", "1 release", "--chunks", "1-2", "--py", sys.executable]) == 0
    status = json.loads((ssd / "exports" / "status.json").read_text())
    assert [c["priors_provisioned"] for c in status["chunks"]] == ["authored", "authored"]
    for n, chunk in enumerate(status["chunks"], start=1):
        target = tmp_path / "batch" / f"{n:02d}" / "out" / "priors" / "priors_pilot_v0.json"
        assert rac.priors_fingerprint(target) == chunk["priors_fingerprint"]
    manifest = json.loads((ssd / "exports" / "manifest.json").read_text())
    assert [c["priors"] for c in manifest["chunks"]] == ["authored", "authored"]
    assert manifest["chunks"][0]["priors_fingerprint"] == status["chunks"][0]["priors_fingerprint"]


def test_dry_run_prints_the_author_command_and_writes_no_priors(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    export_root = _mini_export(tmp_path, {"sess_a": (["0000"], [10], ["v1.0-dhaka-fixed2"])})
    monkeypatch.setattr(rac, "SESSIONS", ("sess_a",))
    assert rac.main(["--repo", str(repo), "--export-root", str(export_root),
                     "--ssd", str(tmp_path / "ssd"), "--batch-root", str(tmp_path / "batch"),
                     "--dry-run", "--chunks", "1", "--py", "/env/bin/python"]) == 0
    out = capsys.readouterr().out
    assert "scripts/author_priors_dhaka.py --from-table --paths" in out
    assert not (tmp_path / "batch").exists()
