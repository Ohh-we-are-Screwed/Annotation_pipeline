"""Contracts for scripts/batch_status.py — the read-only batch dashboard.

The server is started in-process on port 0 against a synthetic status.json, so
the tests never need the SSD, the batch, or the fixed port 8766.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "batch_status", ROOT / "scripts" / "batch_status.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bs = _load()

STATUS = {
    "started": "2026-09-12T12:00:00+06:00",
    "workers": 4,
    "git_sha": "deadbeef",
    "host": "blackwell",
    "progress": "1/2",
    "stages": ["0", "1", "3", "3f", "3m", "4", "5", "6s", "7", "8", "9", "release"],
    "chunks": [
        {"n": 1, "session": "dhaka_20260911_141259", "scene": "dhaka_20260911_141259_chunk_0000",
         "keyframes": 1328, "state": "done", "started": "2026-09-12T12:00:00+06:00",
         "finished": "2026-09-12T13:00:00+06:00", "worker": 0, "current_stage": None,
         "stage_states": {"1": {"state": "ok", "seconds": 59},
                          "5": {"state": "degraded", "seconds": 110,
                                "causes": ["ego_motion_unavailable"]}},
         "log": "/mnt/hdd/dhakascenes/batch_20260912/01/work/logs/run_1.log",
         "export": {"files": 12, "bytes": 345, "symlinks": 0}, "error_tail": []},
        {"n": 2, "session": "dhaka_20260911_151029", "scene": "dhaka_20260911_151029_chunk_0000",
         "keyframes": 1376, "state": "blocked", "started": None, "finished": None,
         "worker": None, "current_stage": None, "stage_states": {},
         "log": None, "export": None, "error_tail": ["boom"], "blocked": "version missing"},
    ],
}


@pytest.fixture()
def server(tmp_path):
    exports = tmp_path / "exports"
    exports.mkdir()
    (exports / "status.json").write_text(json.dumps(STATUS))
    httpd, thread = bs.serve(exports, port=0, background=True)
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    thread.join(timeout=5)


def _get(base, path):
    with urllib.request.urlopen(base + path, timeout=5) as response:
        return response.status, response.read().decode()


def test_index_is_a_self_contained_page(server):
    status, body = _get(server, "/")
    assert status == 200
    assert "<table id=\"grid\">" in body           # the grid the JS fills in
    assert "<title>DhakaScenes batch</title>" in body
    assert "function render(live)" in body and "fetch(\"/live.json\")" in body
    # Self-contained: no external asset may be fetched from the page.
    for forbidden in ("http://", "https://", "cdn"):
        assert forbidden not in body.replace("http://127.0.0.1", "")
    # Every stage column the runner declares must be rendered by the page's JS.
    assert "release" in body


def test_live_endpoint_carries_the_status_and_the_ssd_free_bytes(server):
    status, body = _get(server, "/live.json")
    assert status == 200
    live = json.loads(body)
    assert live["status"]["progress"] == "1/2"
    assert live["status"]["chunks"][1]["blocked"] == "version missing"
    assert isinstance(live["ssd"]["free"], int) and live["ssd"]["free"] > 0
    assert live["ssd"]["total"] >= live["ssd"]["free"]


def test_status_endpoint_serves_the_runner_file_verbatim(server):
    status, body = _get(server, "/status.json")
    assert status == 200
    assert json.loads(body) == STATUS


def test_missing_status_file_is_not_a_crash(tmp_path):
    exports = tmp_path / "exports"
    exports.mkdir()
    httpd, thread = bs.serve(exports, port=0, background=True)
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        assert _get(base, "/")[0] == 200
        live = json.loads(_get(base, "/live.json")[1])
        assert live["status"]["chunks"] == []
    finally:
        httpd.shutdown()
        thread.join(timeout=5)


def test_unknown_paths_are_refused(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(server, "/../../etc/passwd")
    assert excinfo.value.code == 404


def test_stage_columns_and_states_are_all_stylable(server):
    """A state the runner can write with no CSS class renders as a blank cell."""
    status, body = _get(server, "/")
    for state in ("pending", "running", "ok", "degraded", "refused", "crashed",
                  "blocked", "failed", "done", "interrupted"):
        assert f".s-{state}{{" in body.replace(" ", "") or f".s-{state} {{" in body
