#!/usr/bin/env python3
"""Export stage_road's per-point road labels as a nuScenes-lidarseg root.

The road stage (<work_root>/stage_road) records, per keyframe, WHICH raw
LIDAR_TOP points are road. This exporter turns that into the devkit's native
lidarseg format so `NuScenes(version, dataroot=<out>)` loads it with zero
custom code: one uint8 .bin per keyframe positional over the RAW point order,
the FULL canonical 32-row category.json (contiguous indices 0..31 — the
devkit's stats/render APIs assume index == list position, so a sparse
two-row table would crash them; `flat.driveable_surface` keeps its official
24), and a lidarseg.json table binding each .bin to its lidar
sample_data token. The 30 categories this layer never assigns have zero
points.

The output is a SEPARATE, self-contained root: every metadata table json is
copied byte-for-byte from <dataroot>/<version>/, category.json is then
replaced (the devkit asserts an `index` field on every row), and the
samples/ + sweeps/ blobs are only symlinked in under --link-blobs. The
dataroot itself is never written (§1.8).

Refusals (exit 2, message to stderr, nothing written): stage_road missing
its manifest or completion marker, or degraded without
--accept-degraded-upstream (C16); an --out inside (or equal to) the
dataroot; a non-empty <out>/<version>; a points .npz whose __basis__ is not
"raw_lidar_top_file_order" or whose __lidar_sample_data_token__ disagrees
with its road.jsonl row — a lidarseg bin is positional, so labels computed
on any other basis would shear silently, never loudly.

    python -m scripts.export_road_lidarseg --out /path/to/lidarseg_root
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.manifest import (  # noqa: E402
    UpstreamRefusal,
    require_upstream,
    write_json_atomic,
)
from pipeline.common.paths import (  # noqa: E402
    Paths,
    PathValidationError,
    assert_dataroot_read_only,
    load_paths,
)

RAW_BASIS = "raw_lidar_top_file_order"
ROAD_INDEX = 24
ROAD_NAME = "flat.driveable_surface"
ROAD_DESCRIPTION = ("All paved or unpaved surfaces that a car can drive on with no concern "
                    "of traffic rules.")
NOISE_INDEX = 0
NOISE_NAME = "noise"
NOISE_DESCRIPTION = ("Any lidar return that does not correspond to a physical object, such as "
                     "dust, vapor, noise, fog, raindrops, smoke and reflections.")

# The npz arrays the exporter consumes; anything else stage_road writes
# (n_cameras_road, seen_point_index, __frame__) rides along unread.
REQUIRED_NPZ_FIELDS = (
    "road_point_index",
    "__n_points_raw__",
    "__lidar_sample_data_token__",
    "__basis__",
)
REQUIRED_ROW_FIELDS = ("keyframe_token", "scene_token", "lidar_sample_data_token", "n_points_raw")


class ExportRefusal(RuntimeError):
    """A precondition of the export is violated; nothing has been written."""


# ---------------------------------------------------------------------------
# Read side: stage_road rows and their point files
# ---------------------------------------------------------------------------


def load_road_rows(stage_road_dir: str) -> list[tuple[dict, str]]:
    """Every road.jsonl row across scenes, each paired with its .npz path."""
    scenes_root = os.path.join(stage_road_dir, "scenes")
    if not os.path.isdir(scenes_root):
        raise ExportRefusal(f"{scenes_root} not found: stage_road wrote no scenes")
    rows: list[tuple[dict, str]] = []
    for scene in sorted(os.listdir(scenes_root)):
        scene_dir = os.path.join(scenes_root, scene)
        if not os.path.isdir(scene_dir):
            continue
        jsonl = os.path.join(scene_dir, "road.jsonl")
        if not os.path.isfile(jsonl):
            raise ExportRefusal(f"{jsonl} missing: scene {scene} is present without road rows")
        with open(jsonl, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = [k for k in REQUIRED_ROW_FIELDS if k not in row]
                if missing:
                    raise ExportRefusal(f"{jsonl}:{lineno}: row is missing {missing}")
                npz = os.path.join(scene_dir, "points", f"{row['keyframe_token']}.npz")
                rows.append((row, npz))
    if not rows:
        raise ExportRefusal(f"{scenes_root}: no road.jsonl rows to export")
    return rows


def validate_points(row: dict, npz_path: str) -> None:
    """Refuse unless the .npz labels the exact raw blob the row names.

    Runs over EVERY keyframe before the first byte is written, so a refusal
    mid-tree cannot leave a partial lidarseg root behind.
    """
    if not os.path.isfile(npz_path):
        raise ExportRefusal(f"{npz_path} missing for keyframe {row['keyframe_token']}")
    with np.load(npz_path) as z:
        missing = [f for f in REQUIRED_NPZ_FIELDS if f not in z.files]
        if missing:
            raise ExportRefusal(f"{npz_path}: missing arrays {missing}")
        basis = str(z["__basis__"])
        if basis != RAW_BASIS:
            raise ExportRefusal(
                f"{npz_path}: __basis__={basis!r}; a lidarseg bin is positional over the raw "
                f"LIDAR_TOP blob, so only basis {RAW_BASIS!r} is exportable"
            )
        token = str(z["__lidar_sample_data_token__"])
        if token != row["lidar_sample_data_token"]:
            raise ExportRefusal(
                f"{npz_path}: __lidar_sample_data_token__ {token!r} != road.jsonl row's "
                f"{row['lidar_sample_data_token']!r}"
            )
        n_raw = int(np.asarray(z["__n_points_raw__"]).reshape(-1)[0])
        if n_raw != int(row["n_points_raw"]):
            raise ExportRefusal(
                f"{npz_path}: __n_points_raw__={n_raw} != road.jsonl n_points_raw="
                f"{row['n_points_raw']}; the label array would be sized against the wrong blob"
            )
        idx = np.asarray(z["road_point_index"])
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= n_raw):
            raise ExportRefusal(f"{npz_path}: road_point_index out of range [0, {n_raw})")


# ---------------------------------------------------------------------------
# Write side
# ---------------------------------------------------------------------------


def canonical_lidarseg_names() -> list[str]:
    """The devkit's 32 lidarseg categories, in canonical index order.

    Read from the INSTALLED devkit's colormap rather than retyped: its dict
    insertion order is the canonical index order (noise=0 ...
    flat.driveable_surface=24 ... vehicle.ego=31), and the devkit's own
    stats/render APIs merge that colormap against a release's categories BY
    NAME, KeyErroring on anything else.
    """
    from nuscenes.utils.color_map import get_colormap
    return list(get_colormap().keys())


def category_rows(src_category_path: str) -> list[dict]:
    """The FULL canonical 32-row table, contiguous indices 0..31.

    A sparse two-row table ({noise: 0, road: 24}) loads, but every devkit
    lidarseg stats/render API assumes category index == list position, so
    sparse indices crash them. Writing the whole canonical table keeps
    flat.driveable_surface at its official 24 AND keeps the devkit's tools
    working; the 30 categories this layer never assigns simply have zero
    points.
    """
    template = ["token", "name", "description"]
    if os.path.isfile(src_category_path):
        with open(src_category_path, "r", encoding="utf-8") as fh:
            src_rows = json.load(fh)
        if src_rows:
            template = list(src_rows[0].keys())
    descriptions = {
        NOISE_NAME: NOISE_DESCRIPTION,
        ROAD_NAME: ROAD_DESCRIPTION,
    }
    rows = []
    for index, name in enumerate(canonical_lidarseg_names()):
        row = {key: "" for key in template}
        row.update({
            "token": hashlib.sha256(f"dhakascenes/lidarseg/category/{name}".encode()).hexdigest()[:32],
            "name": name,
            "description": descriptions.get(
                name, "nuScenes-lidarseg canonical category; this road-only layer assigns it no points"),
            "index": index,  # asserted by the devkit on every category row
        })
        rows.append(row)
    return rows


def write_labels_bin(row: dict, npz_path: str, bin_path: str) -> int:
    """uint8 labels over the raw point order; atomic via .tmp + os.replace."""
    with np.load(npz_path) as z:
        idx = np.asarray(z["road_point_index"])
    labels = np.zeros(int(row["n_points_raw"]), dtype=np.uint8)
    labels[idx] = ROAD_INDEX
    tmp = bin_path + ".tmp"
    labels.tofile(tmp)
    os.replace(tmp, bin_path)
    return int(idx.size)


def export_road_lidarseg(
    stage_road_dir: str,
    dataroot: str,
    version: str,
    out: str,
    guard_paths: Paths,
    *,
    accept_degraded: bool = False,
    link_blobs: bool = False,
) -> dict:
    """The whole export. Raises ExportRefusal/UpstreamRefusal/PathValidationError
    with nothing written; returns {"n_keyframes", "n_road_points", "out"}."""
    # 1. May we consume stage_road at all? (manifest, marker, degraded policy)
    _manifest, marker = require_upstream(
        stage_road_dir,
        stage_name="stage_road",
        module_hint="pipeline.stage_road.road",
        accept_degraded=accept_degraded,
    )

    # 2. Write-side gates, all before the first byte lands.
    dataroot = os.path.realpath(os.path.expanduser(dataroot))
    resolved_out = os.path.realpath(os.path.expanduser(out))
    if resolved_out == dataroot:
        raise ExportRefusal(
            f"--out {resolved_out} IS the dataroot; the lidarseg root must be a separate tree (§1.8)"
        )
    out = assert_dataroot_read_only(guard_paths, out)
    src_tables = os.path.join(dataroot, version)
    if not os.path.isdir(src_tables):
        raise ExportRefusal(f"no {version}/ metadata directory under {dataroot}")
    out_tables = os.path.join(out, version)
    if os.path.isdir(out_tables) and os.listdir(out_tables):
        raise ExportRefusal(f"{out_tables} already exists and is not empty; refusing to overwrite")

    # 3. Validate EVERY keyframe's points before writing anything.
    rows = load_road_rows(stage_road_dir)
    seen_tokens: set[str] = set()
    for row, npz_path in rows:
        validate_points(row, npz_path)
        token = row["lidar_sample_data_token"]
        if token in seen_tokens:
            # Two bins would collapse onto one file and the devkit's
            # records-vs-files count assertion would fail at load time.
            raise ExportRefusal(f"duplicate lidar_sample_data_token {token!r} across road.jsonl rows")
        seen_tokens.add(token)

    # 4. Metadata tables: byte-for-byte, then replace category.json.
    os.makedirs(out_tables, exist_ok=True)
    for name in sorted(os.listdir(src_tables)):
        src = os.path.join(src_tables, name)
        if name.endswith(".json") and os.path.isfile(src):
            shutil.copyfile(src, os.path.join(out_tables, name))
    write_json_atomic(
        os.path.join(out_tables, "category.json"),
        category_rows(os.path.join(src_tables, "category.json")),
    )

    # 5. One bin per keyframe, then the lidarseg table LAST: the devkit
    # asserts len(records) == len(bin files), so the table only ever names a
    # complete set.
    lidarseg_dir = os.path.join(out, "lidarseg", version)
    os.makedirs(lidarseg_dir, exist_ok=True)
    index_rows, n_road_points = [], 0
    for row, npz_path in rows:
        token = row["lidar_sample_data_token"]
        bin_path = os.path.join(lidarseg_dir, f"{token}_lidarseg.bin")
        n_road_points += write_labels_bin(row, npz_path, bin_path)
        index_rows.append({
            "token": token,
            "sample_data_token": token,
            "filename": f"lidarseg/{version}/{token}_lidarseg.bin",
        })
    write_json_atomic(os.path.join(out_tables, "lidarseg.json"), index_rows)

    # 6. Blobs: opt-in relative symlinks; never copied, never required.
    if link_blobs:
        for sub in ("samples", "sweeps"):
            src = os.path.join(dataroot, sub)
            dst = os.path.join(out, sub)
            if os.path.isdir(src) and not os.path.lexists(dst):
                os.symlink(os.path.relpath(src, out), dst)

    return {
        "n_keyframes": len(rows),
        "n_road_points": n_road_points,
        "out": out,
        "upstream_marker": marker.state,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _infer_version(dataroot: str) -> str:
    """The version directory the dataroot holds, when it holds exactly one."""
    if not os.path.isdir(dataroot):
        raise ExportRefusal(f"dataroot does not exist: {dataroot}")
    found = sorted(e.name for e in os.scandir(dataroot) if e.is_dir() and e.name.startswith("v1.0"))
    if len(found) != 1:
        raise ExportRefusal(
            f"cannot infer --version: {dataroot} holds {found or 'no v1.0* directories'}; "
            "pass --version explicitly"
        )
    return found[0]


def _guard_paths(dataroot: str, version: str, out: str) -> Paths:
    """A minimal Paths for assert_dataroot_read_only when --dataroot bypasses
    paths.yaml (tests; foreign roots). Only the read side is live; the write
    roots point at the out tree, which is the only place this exporter writes."""
    return Paths(
        dataroot=dataroot,
        meta_root=dataroot,
        version=version,
        work_root=out,
        out_root=out,
        probe_out_root=out,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--stage-road-dir", default=None,
                        help="stage_road work tree (default <work_root>/stage_road from --paths)")
    parser.add_argument("--dataroot", default=None,
                        help="nuScenes root holding <version>/ tables, read-only "
                             "(default: dataroot from --paths)")
    parser.add_argument("--out", required=True,
                        help="new lidarseg root to write; must not be (or sit inside) the dataroot")
    parser.add_argument("--version", default=None,
                        help="version directory name (default: the configured version, or the "
                             "single v1.0* directory an explicit --dataroot holds)")
    parser.add_argument("--accept-degraded-upstream", action="store_true",
                        help="consume a _SUCCESS.degraded stage_road (C16: an explicit, recorded "
                             "decision, never a default)")
    parser.add_argument("--link-blobs", action="store_true",
                        help="symlink samples/ and sweeps/ from the dataroot into --out")
    args = parser.parse_args(argv)

    try:
        # Explicit --stage-road-dir + --dataroot bypasses paths.yaml entirely,
        # so tests (and exports off a foreign root) never touch the configured
        # substrate; anything defaulted resolves through the path contract.
        cfg = None if (args.stage_road_dir and args.dataroot) else load_paths(args.paths)
        dataroot = (os.path.realpath(os.path.expanduser(args.dataroot))
                    if args.dataroot else cfg.dataroot)
        stage_road_dir = (os.path.realpath(args.stage_road_dir)
                          if args.stage_road_dir else os.path.join(cfg.work_root, "stage_road"))
        if args.version:
            version = args.version
        elif cfg is not None:
            version = cfg.version
        else:
            version = _infer_version(dataroot)
        guard = cfg if (cfg is not None and not args.dataroot) else _guard_paths(dataroot, version, args.out)
        result = export_road_lidarseg(
            stage_road_dir, dataroot, version, args.out, guard,
            accept_degraded=args.accept_degraded_upstream,
            link_blobs=args.link_blobs,
        )
    except (ExportRefusal, UpstreamRefusal, PathValidationError) as exc:
        print(f"export_road_lidarseg: {exc}", file=sys.stderr)
        return 2

    print(f"wrote {result['n_keyframes']} keyframes, {result['n_road_points']} road points "
          f"-> {result['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
