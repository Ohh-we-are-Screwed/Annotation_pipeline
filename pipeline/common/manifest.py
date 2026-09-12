"""Markers, atomic writers, and the upstream gate (§1.9; register C3, C16).

This module is the sole owner of three things that previously leaked into stage
code and, in leaking, produced the exact failure the register records:

  1. **The stage markers.** C16, verified on disk: Stage 1 ran to completion —
     404/404 keyframes written, validated, atomic — and then withheld `_SUCCESS`
     because two scenes crossed a quality threshold whose own provenance says
     "arbitrary, needs tuning". Every downstream stage hard-refuses on the
     missing marker, so a *quality* signal presented as an *incompleteness*
     signal and the whole pipeline read as broken. §1.9's marker exists to make
     a partially written output distinguishable from a complete one; §1.9's
     per-scene isolation says two degraded scenes must not block the other
     eight. The resolution is three states, not a looser threshold:

         _SUCCESS           complete, clean
         _SUCCESS.degraded  complete, quality-flagged, carries the causes
         (absent)           incomplete — refuse, unconditionally

     A degraded upstream is consumable ONLY under an explicit
     `accept_degraded_upstream` config recorded in the consumer's own manifest.
     The default remains refusal: the change makes acceptance possible and
     auditable, not automatic.

  2. **The atomic writers.** `write_json_atomic` lived in stage0, `write_jsonl_atomic`
     in stage3, and seven stage packages imported them from each other (C3) —
     which is how Stage 5 came to depend on three sibling stages it never calls.
     Both writers now live here; the stage-local names survive as re-exports so
     existing importers keep working while they migrate.

  3. **The upstream gate.** `require_upstream()` is the one implementation of
     "may I consume this stage's output?": manifest present, marker present,
     fingerprint matching, degraded-acceptance policy applied. Before it existed
     every stage answered a slightly different subset of those questions.

One more mechanism, for a defect found in the C18 sweep: `clear_markers()` runs
BEFORE a stage writes its first byte. Without it, a mid-run refusal leaves the
PREVIOUS run's `_SUCCESS` standing over a partially rewritten scenes/ tree —
precisely the state the marker exists to make impossible.

Phase 2 constraints: no GPU, no models, stdlib only.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, Sequence

__all__ = [
    "MARKER_CLEAN",
    "MARKER_DEGRADED",
    "MARKER_SPEC",
    "UpstreamRefusal",
    "Marker",
    "write_json_atomic",
    "write_jsonl_atomic",
    "clear_markers",
    "write_marker",
    "read_marker",
    "require_upstream",
]

MARKER_CLEAN = "_SUCCESS"
MARKER_DEGRADED = "_SUCCESS.degraded"

# Versioned like every other artifact: a consumer that meets a marker written
# under a different spec refuses rather than guessing at its semantics.
MARKER_SPEC = "dhakascenes-pilot/marker/v1"


class UpstreamRefusal(RuntimeError):
    """An upstream artifact is missing, incomplete, foreign, or unaccepted.

    Canonical home (C3). Stage modules alias their local `UpstreamRefusal` to
    this class so an `except UpstreamRefusal` written against any stage keeps
    catching what the gate raises.
    """


# ---------------------------------------------------------------------------
# Atomic writers (migrated: probe.py / proposals.py re-export these)
# ---------------------------------------------------------------------------


def write_json_atomic(path: str, payload: dict) -> str:
    """Write, fsync, read back, compare, then rename.

    A failed read-back never lands, so there is no half-written artifact for a
    later stage to mistake for a complete one (§1.9).
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        with open(tmp, "r", encoding="utf-8") as fh:
            if json.load(fh) != payload:
                raise RuntimeError(f"{path}: payload does not survive the JSON round trip")
        os.replace(tmp, path)
        tmp = ""
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
    return path


def write_jsonl_atomic(path: str, rows: Iterable[dict]) -> str:
    """The same discipline for jsonl documents."""
    payload = list(rows)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for row in payload:
                fh.write(json.dumps(row, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        with open(tmp, "r", encoding="utf-8") as fh:
            restored = [json.loads(line) for line in fh if line.strip()]
        if restored != payload:
            raise RuntimeError(f"{path}: payload does not survive the JSONL round trip")
        os.replace(tmp, path)
        tmp = ""
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
    return path


# ---------------------------------------------------------------------------
# The three-state marker (C16)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Marker:
    """A stage's completion marker, as read back from disk."""

    state: str  # "clean" | "degraded"
    fingerprint: str
    causes: tuple[str, ...] = ()
    path: str = ""

    @property
    def degraded(self) -> bool:
        return self.state == "degraded"


def clear_markers(out_dir: str) -> None:
    """Remove BOTH markers before the stage writes anything else.

    Called at the top of every stage's run, so that a refusal or crash midway
    through rewriting scenes/ leaves NO marker — never the previous run's
    `_SUCCESS` standing over a tree that is half old bytes, half new.
    """
    for name in (MARKER_CLEAN, MARKER_DEGRADED):
        path = os.path.join(out_dir, name)
        if os.path.exists(path):
            os.unlink(path)


def write_marker(
    out_dir: str,
    fingerprint: str,
    *,
    degraded: bool,
    causes: Sequence[str] = (),
) -> str:
    """Write exactly one marker, removing the other.

    Clean: `_SUCCESS`, first line the metadata fingerprint — byte-compatible
    with every consumer written before this module existed, including the peer
    stages that only test for its presence.

    Degraded: `_SUCCESS.degraded`, a JSON document naming the causes, because
    C16's whole point is that "degraded" without WHY is indistinguishable from
    "broken" to the next reader.
    """
    if not fingerprint or not isinstance(fingerprint, str):
        raise ValueError("write_marker needs the metadata fingerprint; a marker that does not "
                         "bind the output to its substrate marks nothing")
    if degraded and not causes:
        raise ValueError("a degraded marker with no causes is a quality flag that cannot be "
                         "audited; pass the failing scenes / predicates")
    os.makedirs(out_dir, exist_ok=True)
    clear_markers(out_dir)
    if degraded:
        path = os.path.join(out_dir, MARKER_DEGRADED)
        write_json_atomic(
            path,
            {
                "spec": MARKER_SPEC,
                "state": "degraded",
                "metadata_fingerprint": fingerprint,
                "causes": list(causes),
            },
        )
    else:
        path = os.path.join(out_dir, MARKER_CLEAN)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(fingerprint + "\n")
    return path


def read_marker(stage_dir: str) -> Marker | None:
    """Read whichever marker is present; None means incomplete.

    Both present is a contract violation, not a tie to break: it can only mean
    two writers that do not share this module, and trusting either would decide
    the C16 question by filesystem accident.
    """
    clean_path = os.path.join(stage_dir, MARKER_CLEAN)
    degraded_path = os.path.join(stage_dir, MARKER_DEGRADED)
    has_clean, has_degraded = os.path.isfile(clean_path), os.path.isfile(degraded_path)
    if has_clean and has_degraded:
        raise UpstreamRefusal(
            f"{stage_dir} carries BOTH {MARKER_CLEAN} and {MARKER_DEGRADED}; the markers are "
            "mutually exclusive by construction and this directory was written by something "
            "other than pipeline.common.manifest"
        )
    if has_clean:
        with open(clean_path, "r", encoding="utf-8") as fh:
            fingerprint = fh.readline().strip()
        return Marker(state="clean", fingerprint=fingerprint, path=clean_path)
    if has_degraded:
        with open(degraded_path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("spec") != MARKER_SPEC:
            raise UpstreamRefusal(
                f"{degraded_path}: marker spec {payload.get('spec')!r} != {MARKER_SPEC!r}"
            )
        return Marker(
            state="degraded",
            fingerprint=str(payload.get("metadata_fingerprint", "")),
            causes=tuple(payload.get("causes", ())),
            path=degraded_path,
        )
    return None


# ---------------------------------------------------------------------------
# The upstream gate
# ---------------------------------------------------------------------------


def require_upstream(
    stage_dir: str,
    *,
    stage_name: str,
    module_hint: str,
    current_fingerprint: str | None = None,
    accept_degraded: bool = False,
) -> tuple[dict, Marker]:
    """The one answer to "may I consume this stage's output?".

    Checks, in order, each with its own refusal:
      1. `run_manifest.json` exists — the stage ran at all.
      2. A marker exists — the run COMPLETED. Absence is §1.9's partial-write
         state and is never acceptable, flag or no flag.
      3. If the marker is degraded, `accept_degraded` must be True — C16's
         explicit, recorded opt-in. The refusal names the causes and the flag,
         so "stage N refuses" reads as "stage N-1 degraded", not "stage N broken".
      4. If `current_fingerprint` is given, the marker must carry the same one —
         the output in front of us descends from the substrate in front of us.

    Returns the upstream manifest and the marker; the caller records
    `marker.degraded` and `marker.causes` in its own manifest, because a run
    built on accepted degradation must say so in its provenance (§1.7).
    """
    manifest_path = os.path.join(stage_dir, "run_manifest.json")
    if not os.path.isfile(manifest_path):
        raise UpstreamRefusal(f"{manifest_path} not found; run `python3 -m {module_hint}` first")

    marker = read_marker(stage_dir)
    if marker is None:
        raise UpstreamRefusal(
            f"{stage_dir} has no completion marker: {stage_name} did not finish, and a partially "
            "written stage output is otherwise indistinguishable from a complete one (§1.9)"
        )
    if marker.degraded and not accept_degraded:
        causes = "; ".join(marker.causes) or "unspecified"
        raise UpstreamRefusal(
            f"{stage_name} completed DEGRADED ({causes}). Its output is complete and consumable, "
            "but consuming it is a recorded decision, not a default: re-run with "
            "--accept-degraded-upstream (C16) or resolve the degradation upstream"
        )
    if current_fingerprint is not None and marker.fingerprint != current_fingerprint:
        raise UpstreamRefusal(
            f"metadata fingerprint mismatch: {stage_name} completed against {marker.fingerprint}, "
            f"this dataroot is {current_fingerprint}. The output describes a different substrate"
        )

    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    return manifest, marker


# ---------------------------------------------------------------------------
# Which Stage 6 box producer the chain ran on
# ---------------------------------------------------------------------------

# stage6_cluster (DBSCAN on the painted cloud) and stage6_stereo_box (per-mask
# stereo boxes) write the same row shape, so stages 7-9 consume either without
# knowing which. Their refusals and their provenance still have to name the one
# in front of them: "run pipeline.stage6_cluster.cluster first" is wrong advice
# on a stereo tree, and a Stage 9 manifest that does not say which producer the
# release descends from cannot be audited against its evidence.
_BOXES_MODULE_HINT = {
    "stage6_stereo_box": "pipeline.stage6_stereo_box.stereo_box",
    "stage7_track": "pipeline.stage7_track.track",
}


def boxes_module_hint(boxes_dir: str) -> str:
    """The module to tell the operator to run, for THIS box directory."""
    return _BOXES_MODULE_HINT.get(
        os.path.basename(os.path.normpath(boxes_dir)), "pipeline.stage6_cluster.cluster"
    )


def boxes_source(upstream_manifest: dict) -> str:
    """The Stage 6 box producer this chain's boxes came from.

    Read from the upstream manifest, not from a directory name: the producer is
    what wrote the rows. Stages 7 and 8 record the key themselves and it is read
    back here first, so a stage reading Stage 7's or Stage 8's manifest still
    gets "stage6_stereo_box", not the name of the stage it read it from. A
    manifest that predates the key answers "unknown" rather than naming the
    stage that merely passed the boxes on.
    """
    stage = str(upstream_manifest.get("stage") or "")
    return str(upstream_manifest.get("boxes_source") or (stage if stage.startswith("stage6_") else "unknown"))


def num_lidar_pts_basis_detail(upstream_manifest: dict) -> str | None:
    """WHAT the producer counted into `num_lidar_pts`, or None if it never said.

    Distinct from `num_lidar_pts_basis` (WHEN it counted: single-sweep,
    ground-filtered, pre-inflation), which every row carries and Stage 9 gates
    on. stage6_stereo_box counts LiDAR *plus* the ZED stereo points its mask
    painted, and says so in `box_fit`; carried down the chain the same way
    `boxes_source` is, so DELIVERY_NOTE.md can name it three stages later.
    """
    detail = upstream_manifest.get("num_lidar_pts_basis_detail")
    if detail is None:
        detail = (upstream_manifest.get("box_fit") or {}).get("num_lidar_pts_basis_detail")
    return str(detail) if detail else None
