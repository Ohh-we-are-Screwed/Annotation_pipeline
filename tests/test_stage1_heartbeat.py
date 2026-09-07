"""Stage 1 progress heartbeat + per-keyframe watchdog (2026-09-08).

On 2026-09-06 a Stage 1 run sat at 100 % CPU for THIRTY HOURS with 2 of 744
clouds written and printed nothing after its banner; SIGINT produced no
traceback, so the cause is still unknown and the night was spent. These are the
two mechanisms that make the next occurrence cost minutes: a rate-limited
progress line (so silence is itself a signal) and a `faulthandler` timer armed
per keyframe (so a hang dumps where it is stuck, even inside a C extension
holding the GIL, and the run walks past it as a degraded keyframe).

The helpers are tested DIRECTLY: a unit test never runs an ingestion.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pipeline.stage1_ingestion.ingest import (  # noqa: E402
    FILTERS,
    IngestConfig,
    KeyframeWatchdog,
    ProgressHeartbeat,
    aggregate,
    summarise_scene,
)


# ---------------------------------------------------------------------------
# 1. The config fields exist, with the documented defaults and provenance
# ---------------------------------------------------------------------------


def test_config_carries_the_two_new_fields_with_their_documented_defaults():
    cfg = IngestConfig()
    assert cfg.heartbeat_seconds == 30.0
    assert cfg.keyframe_timeout_s == 300.0


def test_both_new_fields_carry_a_provenance_string_like_every_other_field():
    cfg = IngestConfig()
    for name in ("heartbeat_seconds", "keyframe_timeout_s"):
        assert name in cfg.provenance, f"{name} has no provenance entry"
        assert len(cfg.provenance[name]) > 40, f"{name} provenance is not a real note"
    # ... and both survive into the manifest's config block.
    as_dict = cfg.as_dict()
    assert as_dict["heartbeat_seconds"] == 30.0
    assert as_dict["keyframe_timeout_s"] == 300.0


# ---------------------------------------------------------------------------
# 2. The watchdog
# ---------------------------------------------------------------------------


def _watchdog(timeout_s: float):
    """A watchdog whose faulthandler dump goes to a temp file, not the console.

    faulthandler writes through a raw fd, so the sink has to be a real file.
    """
    sink = tempfile.TemporaryFile()
    return KeyframeWatchdog(timeout_s, stream=sink), sink


def test_a_slow_keyframe_trips_the_watchdog_and_is_recorded():
    dog, sink = _watchdog(0.05)
    try:
        with dog.keyframe(scene="scene-0001", token="deadbeef", index=3):
            time.sleep(0.30)
        assert dog.n_firings == 1
        firing = dog.firings[0]
        assert firing["keyframe_token"] == "deadbeef"
        assert firing["scene"] == "scene-0001"
        assert firing["index"] == 3
        assert firing["elapsed_s"] >= 0.05
        assert dog.fired_tokens == ["deadbeef"]
        # The point of the exercise: a stack was dumped where the hang was.
        sink.seek(0)
        assert b"Timeout" in sink.read()
    finally:
        sink.close()


def test_a_fast_keyframe_does_not_trip_it():
    dog, sink = _watchdog(30.0)
    try:
        with dog.keyframe(scene="scene-0001", token="cafe", index=0):
            pass
        assert dog.n_firings == 0 and dog.firings == []
    finally:
        sink.close()


def test_zero_timeout_disables_the_watchdog_entirely():
    dog, sink = _watchdog(0.0)
    try:
        assert dog.enabled is False
        with dog.keyframe(scene="scene-0001", token="deadbeef", index=0):
            time.sleep(0.05)
        assert dog.n_firings == 0
        sink.seek(0)
        assert sink.read() == b""  # nothing armed, so nothing dumped
    finally:
        sink.close()


def test_the_watchdog_does_not_swallow_the_keyframes_own_exception():
    dog, sink = _watchdog(30.0)
    try:
        raised = False
        try:
            with dog.keyframe(scene="s", token="t", index=0):
                raise ValueError("boom")
        except ValueError:
            raised = True
        assert raised
    finally:
        sink.close()


def test_the_watchdog_reports_itself_for_the_run_manifest():
    dog, sink = _watchdog(0.05)
    try:
        with dog.keyframe(scene="scene-0001", token="aaa", index=0):
            time.sleep(0.10)
        block = dog.as_dict()
        assert block["keyframe_timeout_s"] == 0.05
        assert block["enabled"] is True
        assert block["n_firings"] == 1
        assert block["keyframe_tokens"] == ["aaa"]
    finally:
        sink.close()


# ---------------------------------------------------------------------------
# 3. The heartbeat rate-limits
# ---------------------------------------------------------------------------


class _Sink:
    def __init__(self):
        self.lines: list[str] = []
        self.flushed = 0

    def write(self, text):
        if text.strip():
            self.lines.append(text)

    def flush(self):
        self.flushed += 1


def _tick(hb, sink, index=1, **kw):
    kw.setdefault("scene", "scene-0001")
    kw.setdefault("token", "0123456789abcdef")
    kw.setdefault("total", 744)
    kw.setdefault("keyframe_elapsed_s", 1.5)
    return hb.tick(index=index, **kw)


def test_two_quick_calls_produce_one_line():
    sink = _Sink()
    hb = ProgressHeartbeat(30.0, total=744, stream=sink)
    assert _tick(hb, sink, index=1) is True     # first keyframe always speaks
    assert _tick(hb, sink, index=2) is False    # inside the interval: silent
    assert len(sink.lines) == 1


def test_a_line_carries_scene_index_total_token_and_an_eta():
    sink = _Sink()
    hb = ProgressHeartbeat(30.0, total=744, stream=sink)
    _tick(hb, sink, index=7, keyframe_elapsed_s=2.0)
    line = sink.lines[0]
    for fragment in ("scene-0001", "7/744", "0123456789ab", "eta"):
        assert fragment in line, f"{fragment!r} missing from {line!r}"
    assert sink.flushed >= 1, "the line was not flushed — buffering is the failure mode"


def test_the_interval_lets_a_later_call_through():
    sink = _Sink()
    hb = ProgressHeartbeat(0.0, total=10, stream=sink)   # 0 = every keyframe
    _tick(hb, sink, index=1)
    _tick(hb, sink, index=2)
    assert len(sink.lines) == 2


def test_force_overrides_the_rate_limit():
    sink = _Sink()
    hb = ProgressHeartbeat(1e6, total=10, stream=sink)
    _tick(hb, sink, index=1)
    assert _tick(hb, sink, index=2, force=True) is True
    assert len(sink.lines) == 2


def test_the_scene_banner_is_unconditional():
    sink = _Sink()
    hb = ProgressHeartbeat(1e6, total=10, stream=sink)
    hb.scene_start("scene-0001", 55)
    hb.scene_start("scene-0002", 61)
    assert len(sink.lines) == 2
    assert "scene-0002" in sink.lines[1] and "61" in sink.lines[1]


def test_the_mean_is_taken_over_the_recent_window_only():
    sink = _Sink()
    hb = ProgressHeartbeat(0.0, total=100, stream=sink, window=2)
    _tick(hb, sink, index=1, keyframe_elapsed_s=100.0)
    _tick(hb, sink, index=2, keyframe_elapsed_s=1.0)
    _tick(hb, sink, index=3, keyframe_elapsed_s=1.0)
    assert hb.mean_seconds_per_keyframe == 1.0   # the 100 s outlier has aged out


# ---------------------------------------------------------------------------
# 4. A firing reaches the run manifest and degrades the run
# ---------------------------------------------------------------------------


def _diag(token: str, *, watchdog: dict | None) -> dict:
    zero = {f: 0 for f in FILTERS}
    ledger = {"survivors": dict(zero), "per_sector": [{"sector": s, **zero} for s in range(IngestConfig().n_sectors)]}
    diag = {
        "keyframe_token": token,
        "ego_compensation_check": {"ok": True, "max_residual_m": 0.0},
        "sector_planes": [],
        "accumulation": {"truncated": False},
        "ledgers": [dict(ledger), dict(ledger)],
    }
    if watchdog is not None:
        diag["watchdog"] = watchdog
    return diag


def test_a_watchdog_firing_makes_the_scene_degraded_and_is_counted():
    cfg = IngestConfig()
    fired = _diag("aaa", watchdog={"fired": True, "elapsed_s": 900.0, "timeout_s": 300.0})
    summary = summarise_scene("scene-0001", [fired], cfg)
    assert summary["n_keyframes_watchdog_timeout"] == 1
    assert summary["watchdog_keyframe_tokens"] == ["aaa"]
    assert summary["degraded"] is True
    assert aggregate([summary])["n_keyframes_watchdog_timeout"] == 1


def test_a_clean_keyframe_is_not_degraded_by_the_watchdog():
    cfg = IngestConfig()
    ok = _diag("bbb", watchdog={"fired": False, "elapsed_s": 1.0, "timeout_s": 300.0})
    summary = summarise_scene("scene-0001", [ok], cfg)
    assert summary["n_keyframes_watchdog_timeout"] == 0
    assert summary["degraded"] is False


def test_diagnostics_without_a_watchdog_block_still_summarise():
    """Older diagnostics files predate the watchdog; reading one must not raise."""
    cfg = IngestConfig()
    summary = summarise_scene("scene-0001", [_diag("ccc", watchdog=None)], cfg)
    assert summary["n_keyframes_watchdog_timeout"] == 0
    assert summary["degraded"] is False
