#!/usr/bin/env python3
"""Drop keyframes whose required-channel blobs do not PARSE — into a NEW version dir.

Stage 0's `files_parse` predicate is all-or-nothing per SCENE, and each chunk is
one scene, so a single corrupt blob refuses the whole chunk. On
dhaka_20260905_174950 that is what happened: twelve bad files — eleven
CAM_FRONT_LEFT JPEGs written 2026-09-06 22:28-23:36 and one LiDAR cloud
truncated during the 2026-09-08 fused-cloud build — refused six of eleven
chunks, about 8,900 keyframes:

    chunk_0001  2   chunk_0004  5   chunk_0006  2
    chunk_0002  1   chunk_0005  1   chunk_0007  1

This is `fixup_a_nusc.py`'s repair, one predicate over: that script drops a
keyframe whose required channel is MISSING, this one also drops a keyframe whose
required channel is PRESENT BUT UNPARSEABLE. The two share the same machinery —
`drop_incomplete_samples` does the chain splicing here as it does there — and the
parse test is imported from the probe rather than restated, so the rule this
drops under is by construction the rule Stage 0 will gate under.

Non-destructive by construction: <dataroot>/<version>/ is read and never
written; <dataroot>/<out-version>/ is created and refused if it already exists.
Blobs are untouched — the dropped samples' files simply go unreferenced. Nothing
is repaired: a corrupt frame stays corrupt on disk, it just stops being named by
a table. That is the same bargain fixup_a_nusc.py makes, for the same reason —
the alternative is a silent skip at Stage 5.

    DHAKASCENES_SUBSTRATE=dhaka6 python scripts/fixup_unparseable.py \\
        --dataroot /home/mt/dhakascenes/data/dhaka_20260905_174950 \\
        --version v1.0-dhaka-fixed2 --out-version v1.0-dhaka-fixed3
    # -> /home/mt/dhakascenes/data/dhaka_20260905_174950/v1.0-dhaka-fixed3/

Run --dry-run first: it prints every offending file and the samples it would
drop, and writes nothing.

Scope. Only KEYFRAME rows of REQUIRED channels are checked, matching
`check_files`' own scope under a profile whose accumulation window is the anchor
keyframe alone (dhaka6): a sweep no stage opens cannot refuse a scene, and
stat-ing 550k of them to prove it costs an hour. `--include-sweeps` widens the
scan for a profile that does read them; it does not change what is dropped,
since a sample is only ever dropped for its own keyframe rows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The repair machinery is fixup_a_nusc's; only the predicate differs. Importing
# it rather than restating it means the chain splicing that took full-fused's
# 868-removed-rows bug to find is fixed in one place.
from scripts.fixup_a_nusc import (  # noqa: E402
    TABLES,
    _channel_of_calibrated_sensor,
    _read_tables,
    drop_incomplete_samples,
)


def parse_reason(dataroot: str, filename: str, fileformat: str | None) -> str | None:
    """Why this blob fails `files_parse`, or None if it passes.

    The two format tests and their bands come from the probe, so this cannot
    drift from the gate it exists to satisfy: a fixup that dropped under looser
    thresholds than Stage 0 enforces would write a version dir that Stage 0
    still refuses, and one that dropped under tighter thresholds would throw
    away frames the pipeline would have accepted.
    """
    from pipeline.stage0_data_probe.probe import _parse_jpeg, _parse_pcd_bin  # noqa: E402

    path = os.path.join(dataroot, filename)
    try:
        size = os.path.getsize(path)
    except OSError:
        return "missing"
    fmt = (fileformat or "").lower()
    if fmt == "pcd" or filename.endswith(".pcd.bin"):
        return _parse_pcd_bin(path, size)
    if fmt in ("jpg", "jpeg") or filename.endswith((".jpg", ".jpeg")):
        return _parse_jpeg(path, size)
    return f"unhandled fileformat {fileformat!r}"


def scan(
    tables: dict[str, list],
    dataroot: str,
    required_channels,
    include_sweeps: bool = False,
    workers: int = 16,
) -> dict[str, str]:
    """filename -> reason, for every required-channel row that does not parse.

    Threaded because the check is one stat plus at most two short reads per
    file and the corpus is ~100k files on a spinning disk; the work is entirely
    I/O wait, and single-threaded this is the difference between two minutes and
    half an hour.
    """
    cs_channel = _channel_of_calibrated_sensor(tables)
    required = set(required_channels)
    rows = [
        r
        for r in tables["sample_data"]
        if (include_sweeps or r.get("is_key_frame"))
        and cs_channel.get(r["calibrated_sensor_token"]) in required
    ]
    bad: dict[str, str] = {}

    def check(row):
        return row["filename"], parse_reason(dataroot, row["filename"], row.get("fileformat"))

    with ThreadPoolExecutor(workers) as pool:
        for filename, reason in pool.map(check, rows, chunksize=64):
            if reason is not None:
                bad[filename] = reason
    return bad


def _scene_of_sample(tables: dict[str, list]) -> dict[str, str]:
    scene_name = {s["token"]: s["name"] for s in tables["scene"]}
    return {s["token"]: scene_name.get(s["scene_token"], s["scene_token"]) for s in tables["sample"]}


def verify(fixed: dict[str, list]) -> list[str]:
    """Referential integrity of the written tables. Empty list means clean.

    Checked because the whole point of this script is to leave Stage 0 with
    nothing to refuse, and `token_graph_closed` is the predicate that would
    catch a botched splice — after the fact, an hour into a run, on a machine
    that has already given the chunk its GPU.
    """
    errors: list[str] = []
    sample_tokens = {s["token"] for s in fixed["sample"]}
    sd_tokens = {r["token"] for r in fixed["sample_data"]}
    ego_tokens = {e["token"] for e in fixed["ego_pose"]}
    cal_tokens = {c["token"] for c in fixed["calibrated_sensor"]}

    for row in fixed["sample_data"]:
        if row["sample_token"] not in sample_tokens:
            errors.append(f"sample_data {row['token']} -> missing sample {row['sample_token']}")
        if row["ego_pose_token"] not in ego_tokens:
            errors.append(f"sample_data {row['token']} -> missing ego_pose {row['ego_pose_token']}")
        if row["calibrated_sensor_token"] not in cal_tokens:
            errors.append(f"sample_data {row['token']} -> missing calibrated_sensor")
        for link in ("prev", "next"):
            tok = row.get(link) or ""
            if tok and tok not in sd_tokens:
                errors.append(f"sample_data {row['token']}.{link} -> dangling {tok}")
    for s in fixed["sample"]:
        for link in ("prev", "next"):
            tok = s.get(link) or ""
            if tok and tok not in sample_tokens:
                errors.append(f"sample {s['token']}.{link} -> dangling {tok}")
    by_scene: dict[str, int] = Counter(s["scene_token"] for s in fixed["sample"])
    for scene in fixed["scene"]:
        n = by_scene.get(scene["token"], 0)
        if scene["nbr_samples"] != n:
            errors.append(f"scene {scene['name']}: nbr_samples {scene['nbr_samples']} != {n} rows")
        for endpoint in ("first_sample_token", "last_sample_token"):
            tok = scene.get(endpoint) or ""
            if n and tok not in sample_tokens:
                errors.append(f"scene {scene['name']}.{endpoint} -> dangling {tok}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataroot", required=True, help="root holding <version>/ and samples/")
    parser.add_argument("--version", default="v1.0-dhaka-fixed2", help="source version dir (read-only)")
    parser.add_argument("--out-version", default=None, help="new version dir; default <version>-parsed")
    parser.add_argument("--required-channels", nargs="+", default=None,
                        help="channels a keyframe must carry and parse; default: the active "
                             "substrate profile's REQUIRED_CHANNELS (DHAKASCENES_SUBSTRATE)")
    parser.add_argument("--include-sweeps", action="store_true",
                        help="also scan non-keyframe rows. Reports them; does not drop on them, "
                             "since a sample is dropped only for its own keyframe rows")
    parser.add_argument("--workers", type=int, default=16, help="scan threads (I/O bound)")
    parser.add_argument("--dry-run", action="store_true", help="report and write nothing")
    args = parser.parse_args(argv)

    out_version = args.out_version or f"{args.version}-parsed"
    src_dir = os.path.join(args.dataroot, args.version)
    out_dir = os.path.join(args.dataroot, out_version)
    if not args.dry_run and os.path.exists(out_dir):
        print(f"refusing: {out_dir} already exists (delete it to redo the fixup)", file=sys.stderr)
        return 2
    if args.required_channels:
        required = tuple(args.required_channels)
    else:
        from pipeline.common.schemas import REQUIRED_CHANNELS, SUBSTRATE  # noqa: E402 — env-resolved
        required = REQUIRED_CHANNELS
        print(f"required channels from profile {SUBSTRATE!r}: {list(required)}")

    try:
        tables = _read_tables(src_dir)
    except FileNotFoundError as exc:
        print(f"refusing: missing table {exc}", file=sys.stderr)
        return 2

    print(f"scanning {args.version} ...", flush=True)
    bad = scan(tables, args.dataroot, required, args.include_sweeps, args.workers)
    if not bad:
        print("every required-channel blob parses; nothing to drop")
        return 0

    scene_of = _scene_of_sample(tables)
    sd_by_file = {r["filename"]: r for r in tables["sample_data"]}
    offending = sorted(
        (
            {
                "filename": f,
                "reason": reason,
                "scene": scene_of.get(sd_by_file[f]["sample_token"], "?"),
                "sample_token": sd_by_file[f]["sample_token"],
                "is_key_frame": bool(sd_by_file[f].get("is_key_frame")),
            }
            for f, reason in bad.items()
        ),
        key=lambda d: (d["scene"], d["filename"]),
    )
    print(f"\n{len(offending)} unparseable blob(s):")
    for row in offending:
        kf = "" if row["is_key_frame"] else "  [sweep — reported, not dropped on]"
        print(f"  {row['scene']}  {row['filename']}  -> {row['reason']}{kf}")

    # A row whose blob does not parse is a row the pipeline cannot use, so it is
    # made to look absent to the SAME drop that handles a genuinely absent one.
    # `drop_incomplete_samples` then decides which samples go, splices both
    # chains, and fixes the scene endpoints.
    def blob_ok(filename: str) -> bool:
        return filename not in bad and os.path.isfile(os.path.join(args.dataroot, filename))

    fixed, dropped = drop_incomplete_samples(tables, required, blob_ok)
    n_before = len(tables["sample"])
    if n_before and not fixed["sample"]:
        print(f"refusing: every one of the {n_before} samples would be dropped; nothing written",
              file=sys.stderr)
        return 2

    dropped_set = set(dropped)
    per_scene = Counter(scene_of.get(t, "?") for t in dropped_set)
    by_channel = Counter()
    cs_channel = _channel_of_calibrated_sensor(tables)
    for f in bad:
        row = sd_by_file[f]
        if row.get("is_key_frame"):
            by_channel[cs_channel.get(row["calibrated_sensor_token"], "?")] += 1

    kept_by_scene: dict[str, int] = Counter(
        scene_of[s["token"]] for s in fixed["sample"] if s["token"] in scene_of
    )
    print(f"\ndropping {len(dropped)} of {n_before} sample(s):")
    for scene in sorted(set(per_scene) | set(kept_by_scene)):
        d = per_scene.get(scene, 0)
        if d:
            print(f"  {scene}: -{d}  ({kept_by_scene.get(scene, 0)} keyframes remain)")
    print(f"unparseable keyframe blobs by channel: {dict(by_channel)}")
    print(f"sample_data rows: {len(tables['sample_data'])} -> {len(fixed['sample_data'])}")

    errors = verify(fixed)
    if errors:
        print(f"\nrefusing: {len(errors)} integrity error(s) after the drop; nothing written",
              file=sys.stderr)
        for e in errors[:20]:
            print(f"  {e}", file=sys.stderr)
        return 2
    print("integrity: sample_data -> sample/ego_pose/calibrated_sensor closed, "
          "prev/next spliced, scene counts and endpoints consistent")

    if args.dry_run:
        print(f"\ndry run: would write {out_dir}")
        return 0

    os.makedirs(out_dir)
    for name in TABLES:
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(fixed[name], fh)
    with open(os.path.join(out_dir, "fixup_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "tool": "scripts/fixup_unparseable.py",
            "source_version": args.version,
            "predicate": "files_parse (pipeline.stage0_data_probe.probe._parse_jpeg / _parse_pcd_bin)",
            "required_channels": list(required),
            "scanned_sweeps": bool(args.include_sweeps),
            "unparseable_blobs": offending,
            "unparseable_keyframe_blobs_by_channel": dict(by_channel),
            "dropped_samples": sorted(dropped_set),
            "dropped_samples_by_scene": dict(per_scene),
            "n_samples_before": n_before,
            "n_samples_after": len(fixed["sample"]),
            "n_sample_data_before": len(tables["sample_data"]),
            "n_sample_data_after": len(fixed["sample_data"]),
            "note": "Blobs are untouched; the dropped samples' files are simply no longer "
                    "referenced by any table. Nothing was repaired.",
        }, fh, indent=2)
    print(f"\nwrote {out_dir}")
    print(f"  {len(fixed['sample'])} samples, {len(fixed['sample_data'])} sample_data rows")
    print(f"  fixup_meta.json records all {len(offending)} offending blob(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
