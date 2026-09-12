#!/usr/bin/env python
"""Run the annotation pipeline over every chunk of the 2026-09-11 Dhaka exports.

WHAT THIS IS. A queue in front of `scripts/run_stages.sh`, nothing more. It
does not know what a stage does, it never writes into a work root, and it
never decides that work can be skipped: the wrapper's own markers do that. It
generates one paths config per chunk so each chunk gets ISOLATED work / out /
probe roots — which is also what makes parallelism safe, because the wrapper
takes a flock per work root and two chunks sharing one would serialise (or,
worse, overwrite each other's stage trees).

THE FOUR EXPORTS are four separate nuScenes roots holding 38 scenes between
them. Chunks are numbered globally 1..38 in session order so one number names
one scene for the whole batch; `manifest.json` on the SSD records the mapping.

WHAT IT WRITES.
  <repo>/configs/batch_20260912/chunk_NN.yaml   generated, git-ignored
  <ssd>/exports/manifest.json                   the static chunk map
  <ssd>/exports/status.json                     rewritten atomically per event
  <ssd>/exports/events.jsonl                    append-only event log
  <dataroot>/sweeps/                            created if missing — the ONLY
                                                write this program makes into
                                                an export (pipeline/common/
                                                paths.py demands the directory)
Everything else on the SSD is written by the release export itself, under
<ssd>/exports/chunk_NN/, with RELEASE_BLOBS=copy so the blobs are real files.

RESUME is the default: a chunk whose release wrote boxes/release_meta.json AND
whose recorded state is done/degraded is skipped. Anything else is handed back
to the wrapper, which resumes from its own _SUCCESS markers.

SIGINT/SIGTERM stops dispatching and lets running wrappers finish — they are
started in their own session so a Ctrl-C in this terminal does not reach them,
and each holds the lock on its work root until it is done.

  $ PYTHONNOUSERSITE=1 python scripts/run_all_chunks.py --workers 4 --chunks 1-38
  $ python scripts/run_all_chunks.py --dry-run --chunks 1-38     # nothing runs
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

import yaml

REPO = Path(__file__).resolve().parents[1]

EXPORT_ROOT = "/home/saif/dhaka-export-pipeline-20260911/export"
SSD = "/media/saif/f1b1e65c-6762-4561-b5b1-e7bcb0679ac4"
BATCH_ROOT = "/mnt/hdd/dhakascenes/batch_20260912"
CONFIG_SUBDIR = "configs/batch_20260912"
TEMPLATE = "configs/paths_zami_20260911.yaml"
INTERPRETER = "/home/saif/miniconda3/envs/ano_pipe/bin/python"

# Session order fixes the global chunk numbering: 1-11, 12-13, 14-27, 28-38.
SESSIONS = ("dhaka_20260911_141259", "dhaka_20260911_151029",
            "dhaka_20260911_154512", "dhaka_20260911_170051")
VERSION = "v1.0-dhaka-fixed2"      # the version the pipeline reads
FALLBACK_VERSION = "v1.0-dhaka"    # for LISTING a session whose fixup is unfinished

DEFAULT_STEPS = "0 1 3 3f 3m 4 5 6s 7 8 release"

# Stage token -> the directory the wrapper cross-examines for markers.
STAGE_DIRS = {
    "0": "stage0_data_probe", "1": "stage1_ingestion", "3": "stage3_proposals",
    "3b": "stage3b_track2d", "3f": "stage3_finetuned", "3m": "stage3_merged",
    "3c": "stage3_checked", "4": "stage4_masks", "5": "stage5_lift",
    "6": "stage6_cluster", "6s": "stage6_stereo_box", "7": "stage7_track",
    "8": "stage8_inflate", "9": "stage9_qa", "road": "stage_road",
}

SECRET = re.compile(r"(^|_)TOKEN($|_)|PASSWORD|SECRET|CREDENTIAL|API_?KEY", re.I)


# ---------------------------------------------------------------------------
# Small pure helpers (each one is pinned by tests/test_run_all_chunks.py)
# ---------------------------------------------------------------------------

def now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def parse_chunks(spec: str) -> list[int]:
    """'1-38' or '1,5,7' or '1-3,38' -> sorted unique global chunk numbers."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"descending chunk range: {part}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return sorted(out)


def load_dotenv(path) -> dict[str, str]:
    """KEY=value lines, `export ` and surrounding quotes tolerated."""
    env: dict[str, str] = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return env
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        key = key.strip()
        if not key.isidentifier():
            continue
        env[key] = value.strip().strip('"').strip("'")
    return env


def mask_env(env: dict[str, str]) -> dict[str, str]:
    """Never let a token reach a log, a status file or a --dry-run printout."""
    return {k: ("***" if SECRET.search(k) else v) for k, v in env.items()}


def write_json_atomic(path, obj) -> None:
    """Serialise first, then rename: a reader never sees a half-written file."""
    path = Path(path)
    text = json.dumps(obj, indent=1, sort_keys=False)  # raises before any write
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def new_record(n, session=None, scene=None, keyframes=0, blocked=None) -> dict:
    return {"n": n, "session": session, "scene": scene, "keyframes": keyframes,
            "state": "blocked" if blocked else "pending", "blocked": blocked,
            "started": None, "finished": None, "worker": None,
            "stage_states": {}, "current_stage": None, "log": None,
            "export": None, "error_tail": [], "run_result": None}


# --- the wrapper's stdout ---------------------------------------------------
#
# run_stages.sh prints `=== <label>` when a step starts, `--- <label>: OK|
# DEGRADED  (Ns)` when it ends well and `!!! <label>: ...` when it does not.
# Labels carry colons inside their parentheses, so the verdict is anchored to
# the END of the line, not to the first colon.
_LABEL = r"(?:STAGE (?P<stage>\S+)|EXPORT (?P<release>nuScenes) release)"
_START = re.compile(r"^=== " + _LABEL + r"\b")
_ENDED = re.compile(r"^--- " + _LABEL + r".*: (?P<verdict>OK|DEGRADED)\s+\((?P<secs>\d+)s\)")
_ANGRY = re.compile(r"^!!! " + _LABEL + r".*: (?P<verdict>REFUSED|FAILED|exited)\b")
_LOG = re.compile(r"^log: (\S+)")

_VERDICTS = {"OK": "ok", "DEGRADED": "degraded", "REFUSED": "refused",
             # `FAILED` is a fatal non-stage step; `exited N ...` is the
             # wrapper catching a stage that lied about its exit code. Both
             # mean the same thing to a queue: this chunk produced nothing.
             "FAILED": "crashed", "exited": "crashed"}


def parse_wrapper_line(line: str, record: dict) -> bool:
    """Fold one line of wrapper stdout into `record`. True if it changed it."""
    if line.startswith("=== ALL_STEPS_DONE"):
        record["run_result"], record["current_stage"] = "done", None
        return True
    if line.startswith("=== RUN_INCOMPLETE"):
        record["run_result"], record["current_stage"] = "incomplete", None
        return True
    match = _LOG.match(line)
    if match:
        record["log"] = match.group(1)
        return True
    for pattern in (_START, _ENDED, _ANGRY):
        match = pattern.match(line)
        if not match:
            continue
        stage = "release" if match.group("release") else match.group("stage")
        if pattern is _START:
            record["stage_states"][stage] = {"state": "running"}
            record["current_stage"] = stage
            return True
        state = _VERDICTS[match.group("verdict")]
        entry = {"state": state}
        if pattern is _ENDED:
            entry["seconds"] = int(match.group("secs"))
        record["stage_states"][stage] = entry
        if record["current_stage"] == stage:
            record["current_stage"] = None
        return True
    return False


# --- the chunk map ----------------------------------------------------------

def _scene_keyframes(version_dir: Path) -> list[tuple[str, int]]:
    scenes = json.loads((version_dir / "scene.json").read_text())
    counts: collections.Counter = collections.Counter()
    sample = version_dir / "sample.json"
    if sample.exists():
        counts.update(s["scene_token"] for s in json.loads(sample.read_text()))
    return [(s["name"], counts.get(s["token"], s.get("nbr_samples", 0)))
            for s in sorted(scenes, key=lambda s: s["name"])]


def build_chunk_map(export_root, sessions=SESSIONS, version=VERSION,
                    fallback=FALLBACK_VERSION) -> list[dict]:
    """Global chunk numbers over the sessions, in the order given.

    A session whose `version` directory has no scene.json is listed from
    `fallback` and every one of its chunks is marked blocked — the fixup that
    produces `version` may still be running, and running the pipeline against
    the unfixed tables would annotate a substrate nothing else matches.
    """
    export_root = Path(export_root)
    chunks: list[dict] = []
    for session in sessions:
        dataroot = export_root / session / "full"
        blocked = None
        vdir = dataroot / version
        if not (vdir / "scene.json").exists():
            blocked = "version missing"
            vdir = dataroot / fallback
            if not (vdir / "scene.json").exists():
                sys.exit(f"no readable version for {session}: neither "
                         f"{version} nor {fallback} under {dataroot}")
        for name, keyframes in _scene_keyframes(vdir):
            chunks.append({"n": len(chunks) + 1, "session": session, "scene": name,
                           "keyframes": keyframes, "dataroot": str(dataroot),
                           "blocked": blocked})
    return chunks


def write_chunk_config(template, chunk, config_dir, batch_root) -> Path:
    """One paths.yaml per chunk: this session's dataroot, isolated write roots."""
    config = yaml.safe_load(Path(template).read_text())
    nn = f"{chunk['n']:02d}"
    config.update(dataroot=chunk["dataroot"], meta_root=chunk["dataroot"],
                  work_root=f"{batch_root}/{nn}/work",
                  out_root=f"{batch_root}/{nn}/out",
                  probe_out_root=f"{batch_root}/{nn}/probe")
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"chunk_{nn}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    return path


# --- the deliverable on the SSD --------------------------------------------

def export_stats(path) -> dict:
    """files / bytes / symlinks under an export. RELEASE_BLOBS=copy must give 0
    symlinks: a link into /mnt/hdd would leave the SSD delivery unreadable off
    this machine."""
    files = total = links = 0
    for root, _dirs, names in os.walk(path):
        for name in names:
            full = os.path.join(root, name)
            files += 1
            if os.path.islink(full):
                links += 1
            else:
                total += os.lstat(full).st_size
    return {"files": files, "bytes": total, "symlinks": links}


def should_skip(record, exports_root) -> bool:
    """Resume: a chunk counts as finished only if BOTH the runner said so and
    the release artefact is on the SSD."""
    if record.get("state") not in ("done", "degraded"):
        return False
    return (Path(exports_root) / f"chunk_{record['n']:02d}" / "boxes"
            / "release_meta.json").exists()


def read_markers(work_root, record) -> None:
    """After a chunk ends, believe the markers on disk over the stdout parse."""
    for stage, directory in STAGE_DIRS.items():
        stage_dir = Path(work_root) / directory
        degraded = stage_dir / "_SUCCESS.degraded"
        entry = record["stage_states"].get(stage)
        if degraded.exists():
            entry = dict(entry or {}, state="degraded")
            try:
                entry["causes"] = json.loads(degraded.read_text()).get("causes")
            except (OSError, ValueError):
                entry["causes"] = ["unreadable _SUCCESS.degraded"]
            record["stage_states"][stage] = entry
        elif (stage_dir / "_SUCCESS").exists() and entry is None:
            record["stage_states"][stage] = {"state": "ok"}


# ---------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------

class Batch:
    def __init__(self, args, chunks):
        self.args = args
        self.exports = Path(args.ssd) / "exports"
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.next_start = 0.0
        self.records = {c["n"]: new_record(**{k: c[k] for k in
                        ("n", "session", "scene", "keyframes", "blocked")})
                        for c in chunks}
        self.configs = {}
        self.worker_ids: queue.Queue = queue.Queue()
        for i in range(args.workers):
            self.worker_ids.put(i)
        self.stages = []
        for step in args.steps.split():
            self.stages.extend(["9", "release"] if step == "release" else [step])
        self.global_block = {
            "started": now(), "workers": args.workers, "steps": args.steps,
            "git_sha": git_sha(args.repo), "host": socket.gethostname(),
            "stages": self.stages, "progress": "0/0",
        }

    # -- state ---------------------------------------------------------------

    def flush(self) -> None:
        done = sum(1 for r in self.records.values()
                   if r["state"] in ("done", "degraded"))
        self.global_block["progress"] = f"{done}/{len(self.records)}"
        write_json_atomic(self.exports / "status.json",
                          dict(self.global_block, updated=now(),
                               chunks=[self.records[n] for n in sorted(self.records)]))

    def emit(self, event, record=None, **extra) -> None:
        line = {"t": now(), "event": event}
        if record is not None:
            line.update(n=record["n"], scene=record["scene"], state=record["state"],
                        current_stage=record["current_stage"])
        line.update(extra)
        with (self.exports / "events.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line) + "\n")

    # -- one chunk -----------------------------------------------------------

    def command(self, record):
        return ["bash", "scripts/run_stages.sh", *self.args.steps.split(),
                "--scenes", record["scene"], "--no-cvat"]

    def overlay(self, record):
        nn = f"{record['n']:02d}"
        return {"PYTHONNOUSERSITE": "1", "PYTHONUNBUFFERED": "1",
                "PY": self.args.py,
                "DHAKASCENES_PATHS_CONFIG": str(self.configs[record["n"]]),
                "DHAKASCENES_SUBSTRATE": "dhaka6", "STEREO_STRIDE": "1",
                "COVERAGE_CONFIG": "R3", "EXPORT_ROOT": str(self.exports),
                "EXPORT_NAME": f"chunk_{nn}", "RELEASE_BLOBS": "copy"}

    def stagger(self) -> None:
        """Keep 90 s between wrapper starts: Stage 1 writes ground-filtered
        clouds to the same HDD for every chunk, and four of them opening at
        once is where that disk falls over. --stagger is the knob."""
        with self.lock:
            wait = max(0.0, self.next_start - time.monotonic())
            self.next_start = time.monotonic() + wait + self.args.stagger
        while wait > 0 and not self.stop.is_set():
            step = min(1.0, wait)
            time.sleep(step)
            wait -= step

    def run_chunk(self, n) -> None:
        record = self.records[n]
        if self.stop.is_set() or record["blocked"]:
            return
        worker = self.worker_ids.get()
        try:
            self.stagger()
            if self.stop.is_set():
                return
            with self.lock:
                record.update(state="running", started=now(), finished=None,
                              worker=worker, error_tail=[], run_result=None,
                              stage_states={}, current_stage=None)
                self.flush()
            self.emit("chunk_start", record)
            print(f"[{now()}] w{worker} chunk {n:02d} {record['scene']} "
                  f"({record['keyframes']} kf) starting", flush=True)
            rc, tail = self.pump(record)
            self.finish(record, rc, tail)
        except Exception as exc:                      # a worker must not take the queue down
            with self.lock:
                record.update(state="failed", finished=now(),
                              error_tail=(record["error_tail"] + [repr(exc)])[-40:])
                self.flush()
            self.emit("chunk_error", record, error=repr(exc))
            print(f"[{now()}] chunk {n:02d} FAILED in the runner: {exc!r}", flush=True)
        finally:
            self.worker_ids.put(worker)

    def pump(self, record):
        """Run the wrapper, folding its stdout into the record as it arrives."""
        env = dict(os.environ, **self.args.dotenv, **self.overlay(record))
        tail: collections.deque = collections.deque(maxlen=40)
        # start_new_session: a Ctrl-C in this terminal must not reach a wrapper
        # mid-stage — it holds the flock on its work root and owns the cleanup.
        proc = subprocess.Popen(self.command(record), cwd=self.args.repo, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, start_new_session=True)
        last = 0.0
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            changed = parse_wrapper_line(line, record)
            if changed or time.monotonic() - last > 5:
                with self.lock:
                    record["error_tail"] = list(tail)
                    self.flush()
                last = time.monotonic()
            if changed and record["current_stage"]:
                self.emit("stage", record)
        proc.stdout.close()
        return proc.wait(), tail

    def finish(self, record, rc, tail) -> None:
        config = yaml.safe_load(Path(self.configs[record["n"]]).read_text())
        read_markers(config["work_root"], record)
        export = self.exports / f"chunk_{record['n']:02d}"
        record["export"] = export_stats(export) if export.exists() else None
        degraded = any(s.get("state") == "degraded" for s in record["stage_states"].values())
        if record["run_result"] == "done":
            record["state"] = "degraded" if degraded else "done"
        elif self.stop.is_set():
            record["state"] = "interrupted"
        else:
            record["state"] = "failed"
        if record["state"] in ("done", "degraded"):
            released = (export / "boxes" / "release_meta.json").exists()
            if not released:
                record["state"] = "failed"
                tail.append("release wrote no boxes/release_meta.json")
            elif record["export"]["symlinks"]:
                record["state"] = "degraded"
                entry = record["stage_states"].setdefault("release", {"state": "ok"})
                entry["causes"] = (entry.get("causes") or []) + [
                    f"{record['export']['symlinks']} symlinks in the export "
                    "(RELEASE_BLOBS=copy should have left none)"]
        record.update(finished=now(), current_stage=None, error_tail=list(tail)[-40:], rc=rc)
        with self.lock:
            self.flush()
        self.emit("chunk_end", record, rc=rc, export=record["export"])
        print(f"[{now()}] chunk {record['n']:02d} {record['state'].upper()} "
              f"(rc={rc}, {record['scene']})", flush=True)

    # -- the whole batch -----------------------------------------------------

    def run(self, wanted) -> int:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(self.args.workers) as pool:
            list(pool.map(self.run_chunk, wanted))
        with self.lock:
            self.flush()
        states = collections.Counter(self.records[n]["state"] for n in wanted)
        print(f"[{now()}] batch finished: " +
              ", ".join(f"{v} {k}" for k, v in sorted(states.items())), flush=True)
        return 0 if not (states["failed"] or states["interrupted"]) else 1


def git_sha(repo) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunks", default="1-38", help="1-38 or 1,5,7")
    parser.add_argument("--steps", default=DEFAULT_STEPS)
    parser.add_argument("--dry-run", action="store_true",
                        help="write the configs, print every command, run nothing")
    parser.add_argument("--ssd", default=SSD)
    parser.add_argument("--repo", default=str(REPO), help="repository run_stages.sh lives in")
    parser.add_argument("--export-root", default=EXPORT_ROOT)
    parser.add_argument("--batch-root", default=BATCH_ROOT)
    parser.add_argument("--template", default=None, help=f"default: <repo>/{TEMPLATE}")
    parser.add_argument("--py", default=INTERPRETER)
    parser.add_argument("--stagger", type=float, default=90.0,
                        help="seconds between wrapper starts (default 90)")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="re-run chunks already recorded done (the wrapper still "
                             "resumes from its own markers)")
    args = parser.parse_args(argv)
    args.repo = Path(args.repo).resolve()
    args.template = Path(args.template) if args.template else args.repo / TEMPLATE
    args.dotenv = load_dotenv(args.repo / ".env")

    chunks = build_chunk_map(args.export_root, SESSIONS)
    by_n = {c["n"]: c for c in chunks}
    wanted = [n for n in parse_chunks(args.chunks) if n in by_n]
    if len(wanted) != len(parse_chunks(args.chunks)):
        parser.error(f"chunk numbers outside 1..{len(chunks)}")

    config_dir = args.repo / CONFIG_SUBDIR
    batch = Batch(args, chunks)
    for n in wanted:
        batch.configs[n] = write_chunk_config(args.template, by_n[n], config_dir, args.batch_root)
        if not args.dry_run:
            # The one write this program makes into an export: paths.py refuses
            # a dataroot without sweeps/, and this capture has no sweep frames.
            (Path(by_n[n]["dataroot"]) / "sweeps").mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        for n in wanted:
            record, chunk = batch.records[n], by_n[n]
            sweeps = Path(chunk["dataroot"]) / "sweeps"
            print(f"\n=== chunk {n:02d}  {record['scene']}  {record['keyframes']} kf"
                  + (f"  BLOCKED: {chunk['blocked']}" if chunk["blocked"] else ""))
            print(f"    config: {batch.configs[n]}"
                  + ("" if sweeps.is_dir() else f"   (would mkdir {sweeps})"))
            print(f"    cwd:    {args.repo}")
            print("    env:    " + " ".join(
                f"{k}={v}" for k, v in sorted(mask_env(dict(args.dotenv,
                                                            **batch.overlay(record))).items())))
            print("    cmd:    " + " ".join(batch.command(record)))
        print(f"\n{len(wanted)} chunks; "
              f"{sum(1 for n in wanted if by_n[n]['blocked'])} blocked; "
              f"{sum(batch.records[n]['keyframes'] for n in wanted)} keyframes; "
              f"nothing written to {batch.exports}")
        return 0

    batch.exports.mkdir(parents=True, exist_ok=True)
    write_json_atomic(batch.exports / "manifest.json",
                      {"written": now(), "version": VERSION, "template": str(args.template),
                       "batch_root": args.batch_root,
                       "chunks": [dict(c, config=str(batch.configs.get(c["n"], "")))
                                  for c in chunks]})
    previous = batch.exports / "status.json"
    if args.resume and previous.exists():
        old = {r["n"]: r for r in json.loads(previous.read_text()).get("chunks", [])}
        for n in list(wanted):
            if n in old and should_skip(old[n], batch.exports):
                batch.records[n] = old[n]
                wanted.remove(n)
                print(f"resume: chunk {n:02d} already released — skipping")
    blocked = [n for n in wanted if by_n[n]["blocked"]]
    for n in blocked:
        wanted.remove(n)
        print(f"blocked: chunk {n:02d} {by_n[n]['scene']} — {by_n[n]['blocked']}")
    batch.flush()

    def halt(signum, _frame):
        batch.stop.set()
        print(f"\n[{now()}] signal {signum}: no new chunks will start; "
              "running wrappers keep going (they hold their own locks).", flush=True)
    signal.signal(signal.SIGINT, halt)
    signal.signal(signal.SIGTERM, halt)

    print(f"{len(wanted)} chunks queued, {args.workers} workers, "
          f"{sum(batch.records[n]['keyframes'] for n in wanted)} keyframes; "
          f"status: {batch.exports / 'status.json'}", flush=True)
    # Longest first: the tail of a batch is only as short as its longest chunk.
    wanted.sort(key=lambda n: -batch.records[n]["keyframes"])
    return batch.run(wanted)


if __name__ == "__main__":
    sys.exit(main())
