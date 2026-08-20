#!/usr/bin/env python3
"""Frame sequencing over the 12 Hz camera sweeps between keyframes (Stage 3b, C27).

Single purpose: walk the nuScenes `sample_data` prev/next chain of one (scene, camera
channel) pair and cut it into inclusive keyframe-to-keyframe windows. Metadata only —
no image is opened, no model loaded, no cloud read.

C27 adds Stage 3b, which propagates Stage 3's YOLO detections through the sweep frames
between two 2 Hz keyframes so a detection keeps its 2D identity across the gap. This
module is that stage's frame-sequencing foundation and nothing more: it decides WHICH
frames exist and in WHICH order. **12 Hz sweep frames are connective tissue for 2D
identity only; no LiDAR or eval quantity exists off-keyframe (pilot_plan §11 decision 5
is not reopened.)** A sweep frame never becomes a track record, a lifted box or a scored
row — which is why nothing here returns anything but frame references.

Channel derivation: `substrate.channel(record)`, the calibrated_sensor -> sensor ->
channel lookup Stage 0 builds as `channel_of_calibrated_sensor` (probe.py). The filename
prefix ("sweeps/CAM_FRONT/...") carries the same information, but it is a string
convention over blob paths, whereas the calibrated_sensor edge is a table Stage 0's
predicates actually gate on: a channel resolved here is a channel Stage 0 verified.

The substrate is duck-typed (`Any`) — only `sample_data_by_scene`, `sample_data_by_token`
and `channel()` are read, so a test double is three members, not a loaded dataset. There
is deliberately no `pipeline` import in this module body; the __main__ self-test is the
single exception.

    python3 pipeline/common/sequences.py [--paths configs/paths.yaml]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

__all__ = ["SequenceError", "FrameRef", "Window", "camera_frame_chain", "keyframe_windows"]


class SequenceError(RuntimeError):
    """A sample_data chain violated a structural invariant Stage 0 was supposed to guarantee."""


@dataclass(frozen=True)
class FrameRef:
    """One camera frame, keyframe or sweep. Frozen: a chain is never edited in place."""
    sample_data_token: str
    filename: str  # dataroot-relative, e.g. "sweeps/CAM_FRONT/x.jpg"
    timestamp_us: int  # microseconds, verbatim from sample_data.timestamp
    is_key_frame: bool
    sample_token: str


@dataclass(frozen=True)
class Window:
    """One keyframe-to-keyframe span, both endpoints included."""
    start: FrameRef  # keyframe kf_i   (frames[0])
    frames: tuple[FrameRef, ...]  # inclusive [kf_i .. kf_{i+1}], time-ordered
    end: FrameRef  # keyframe kf_{i+1} (frames[-1])
    subsampled: bool  # True when the interior was thinned to fit max_frames


def _violated(scene_token: str, channel: str, predicate: str, detail: str) -> SequenceError:
    """Every failure names the scene, the camera and the predicate — never just "bad chain"."""
    return SequenceError(f"{predicate}: scene {scene_token} channel {channel}: {detail}")


def camera_frame_chain(substrate: Any, scene_token: str, channel: str) -> list[FrameRef]:
    """Every frame of one camera in one scene, in prev/next chain order.

    `substrate` is duck-typed; the members read are `sample_data_by_scene` (scene token ->
    list of raw sample_data dicts), `sample_data_by_token` (token -> record) and
    `channel(record) -> str | None`. Raises SequenceError naming one of: head_exists,
    head_is_unique, chain_is_acyclic, chain_covers_every_record,
    timestamps_strictly_increasing, keyframes_all_on_chain.
    """
    collected: dict[str, dict] = {}
    for record in substrate.sample_data_by_scene.get(scene_token, ()):
        if substrate.channel(record) == channel:
            collected[record["token"]] = record

    # The head is the record nothing in this set precedes. A `prev` may point at another
    # scene's or another channel's record; outside the set is where this chain starts.
    heads = [r for r in collected.values() if not r.get("prev") or r["prev"] not in collected]
    if not heads:
        raise _violated(scene_token, channel, "head_exists", f"no record has a prev outside the "
                        f"set among {len(collected)} collected (empty channel, or a closed loop)")
    if len(heads) > 1:
        raise _violated(scene_token, channel, "head_is_unique", f"{len(heads)} heads: "
                        f"{sorted(r['token'] for r in heads)[:5]} — these frames form more than "
                        "one chain, so no single frame order exists")

    # Walked, not sorted by timestamp: a timestamp sort succeeds silently on a chain with a
    # break in it and returns a plausible frame list that steps straight over the
    # discontinuity. The links are the substrate's own statement about adjacency, and
    # adjacency is the whole premise of propagation — so they are walked, and the walk is
    # then asserted to have covered everything (Stage 0's token_graph_closed, re-checked
    # here at the point of use).
    walked: list[dict] = []
    seen: set[str] = set()
    token = heads[0]["token"]
    while token:
        if token in seen:
            raise _violated(scene_token, channel, "chain_is_acyclic",
                            f"next revisits {token} after {len(walked)} frames")
        seen.add(token)
        record = collected[token]
        walked.append(record)
        # Resolve through the substrate's global index, then require membership: a `next`
        # leaving the collected set ENDS this camera's chain rather than failing it — that is
        # how the scene's last frame terminates, and how a dangling link stays contained.
        nxt = substrate.sample_data_by_token.get(record.get("next") or "")
        token = nxt["token"] if nxt is not None and nxt["token"] in collected else ""

    if len(walked) != len(collected):
        orphans = sorted(set(collected) - seen)
        raise _violated(scene_token, channel, "chain_covers_every_record",
                        f"walked {len(walked)} of {len(collected)}; {len(orphans)} unreachable, "
                        f"first {orphans[:5]}")

    for earlier, later in zip(walked, walked[1:]):
        if later["timestamp"] <= earlier["timestamp"]:
            raise _violated(scene_token, channel, "timestamps_strictly_increasing",
                            f"{earlier['token']} at {earlier['timestamp']} us is followed by "
                            f"{later['token']} at {later['timestamp']} us")

    # Implied by chain_covers_every_record, asserted separately so a keyframe falling off the
    # chain reads as the keyframe problem it is: a lost propagation ANCHOR, not a lost sweep.
    n_kf_chain = sum(1 for r in walked if r["is_key_frame"])
    n_kf_collected = sum(1 for r in collected.values() if r["is_key_frame"])
    if n_kf_chain != n_kf_collected:
        raise _violated(scene_token, channel, "keyframes_all_on_chain",
                        f"{n_kf_chain} keyframes on the chain, {n_kf_collected} collected")

    # `timestamp` is microseconds in the table and stays microseconds here; the ns conversion
    # belongs to whichever stage emits a record, not to the sequencer.
    return [FrameRef(r["token"], r["filename"], int(r["timestamp"]), bool(r["is_key_frame"]),
                     r["sample_token"]) for r in walked]


def _round_half_up(numerator: int, denominator: int) -> int:
    """Integer round-half-up (numerator >= 0, denominator > 0).

    Not `round(a / b)`: float division carries binary-representation error and Python's
    `round` is banker's rounding, so the same k could select a different frame for a
    different pair of values. Which sweeps get propagated has to be reproducible.
    """
    return (2 * numerator + denominator) // (2 * denominator)


def keyframe_windows(
    chain: list[FrameRef], keyframe_sd_tokens: Sequence[str], max_frames: int
) -> tuple[list[Window], list[str]]:
    """Cut `chain` into windows between consecutive keyframe anchors.

    `keyframe_sd_tokens` is the time-ordered list of the camera's KEYFRAME sample_data tokens
    as Stage 3's rows record them. Tokens absent from the chain are skipped and returned as
    `missing` (in order, no duplicates); a window only ever spans two CONSECUTIVE FOUND
    anchors, so a dropped Stage 3 row widens one window rather than silently deleting every
    frame beyond it. Raises ValueError for max_frames < 2 (a window without both endpoints is
    not a window), SequenceError if two found anchors are not in increasing chain order.
    """
    if max_frames < 2:
        raise ValueError(f"max_frames must be >= 2 to keep both endpoints, got {max_frames}")

    index_of = {frame.sample_data_token: i for i, frame in enumerate(chain)}
    anchors: list[int] = []
    missing: list[str] = []
    seen_missing: set[str] = set()
    for token in keyframe_sd_tokens:
        position = index_of.get(token)
        if position is not None:
            anchors.append(position)
        elif token not in seen_missing:
            seen_missing.add(token)
            missing.append(token)

    windows: list[Window] = []
    for a, b in zip(anchors, anchors[1:]):
        if a >= b:
            raise SequenceError(
                f"keyframe anchors out of order: chain index {a} is followed by {b} "
                f"({chain[a].sample_data_token} -> {chain[b].sample_data_token}); "
                "keyframe_sd_tokens must be time-ordered"
            )
        span = b - a
        subsampled = span + 1 > max_frames
        if not subsampled:
            indices = list(range(a, b + 1))
        else:
            # The endpoints ARE the anchors and are never dropped; the interior is thinned to
            # an even stride so the kept frames stay spread over the whole gap instead of
            # clustering at one end and leaving the other half unpropagated.
            interior: list[int] = []
            taken = {a, b}
            for k in range(1, max_frames - 1):
                index = a + _round_half_up(k * span, max_frames - 1)
                if index in taken:  # dedup, and never re-emit an endpoint
                    continue
                taken.add(index)
                interior.append(index)
            indices = [a] + interior + [b]  # interior ascends: the stride is monotonic in k
        windows.append(Window(chain[a], tuple(chain[i] for i in indices), chain[b], subsampled))

    return windows, missing


if __name__ == "__main__":
    # The ONLY place in this module that may import from pipeline/: everything above stays
    # duck-typed so Stage 3b's sequencing can be tested without a substrate at all.
    import argparse
    import os
    import sys
    from collections import Counter

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from pipeline.common.paths import PathValidationError, load_paths  # noqa: E402
    from pipeline.stage0_data_probe.probe import Substrate  # noqa: E402

    CHANNELS = ("CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
                "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
    MAX_FRAMES = 16

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # Same .env contract as every other entry point (C17): a declared key has a reader.
    parser.add_argument("--paths",
                        default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    args = parser.parse_args()

    try:
        paths = load_paths(args.paths)
    except PathValidationError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    substrate = Substrate.load(paths)
    scenes = substrate.tables["scene.json"]
    totals: Counter = Counter()
    total_lens: Counter = Counter()
    failed = False

    for scene in scenes:
        for channel in CHANNELS:
            try:
                chain = camera_frame_chain(substrate, scene["token"], channel)
                # Anchors here are the chain's own keyframes; Stage 3b passes Stage 3's rows.
                anchors = [f.sample_data_token for f in chain if f.is_key_frame]
                windows, missing = keyframe_windows(chain, anchors, MAX_FRAMES)
            except SequenceError as exc:
                failed = True
                print(f"FAIL {scene['name']} {channel}: {exc}", file=sys.stderr)
                continue
            lens = Counter(len(w.frames) for w in windows)
            row = Counter({"chain": len(chain), "kf": len(anchors), "win": len(windows),
                           "subsampled": sum(1 for w in windows if w.subsampled),
                           "missing": len(missing)})
            print(f"{scene['name']:<11} {scene['token'][:8]} {channel:<16} chain={row['chain']:4d} "
                  f"kf={row['kf']:3d} win={row['win']:3d} lens={dict(sorted(lens.items()))} "
                  f"subsampled={row['subsampled']} missing={row['missing']}")
            totals.update(row)
            total_lens.update(lens)

    print(f"TOTALS scenes={len(scenes)} channels={len(CHANNELS)} max_frames={MAX_FRAMES} "
          f"chain={totals['chain']} kf={totals['kf']} win={totals['win']} "
          f"lens={dict(sorted(total_lens.items()))} subsampled={totals['subsampled']} "
          f"missing={totals['missing']}")
    sys.exit(1 if failed else 0)
