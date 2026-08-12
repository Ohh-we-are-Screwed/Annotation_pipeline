"""Path contract and metadata fingerprint for the DhakaScenes Pilot (§1.8).

One `configs/paths.yaml` is resolved exactly once into a frozen `Paths` object.
`validate_paths()` is called at process start, before any stage touches disk.
`metadata_fingerprint()` binds every downstream artifact to the exact bytes of
the 13 metadata tables it was computed from.

Phase 1 constraints: no GPU, no models, stdlib + PyYAML only.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, asdict
from pathlib import Path as _FsPath

import yaml

__all__ = [
    "METADATA_TABLES",
    "FINGERPRINT_SPEC",
    "Paths",
    "PathValidationError",
    "load_paths",
    "validate_paths",
    "metadata_fingerprint",
    "metadata_file_digests",
    "assert_dataroot_read_only",
]

# The 13 v1.0-mini metadata tables. Frozen: a 14th file appearing in the
# version directory, or one of these missing, is a substrate change and must
# invalidate every fingerprint computed under the old set.
METADATA_TABLES: tuple[str, ...] = (
    "attribute.json",
    "calibrated_sensor.json",
    "category.json",
    "ego_pose.json",
    "instance.json",
    "log.json",
    "map.json",
    "sample.json",
    "sample_annotation.json",
    "sample_data.json",
    "scene.json",
    "sensor.json",
    "visibility.json",
)

# Required top-level blob subdirectories under dataroot.
REQUIRED_BLOB_DIRS: tuple[str, ...] = ("samples", "sweeps")

# The plan (§1.8) specifies "sorted SHA-256 of the 13 metadata JSONs" but does
# not fix the hash-input construction. It is pinned here, versioned, and
# recorded alongside every digest so a future change is detectable rather than
# silent:
#
#   1. For each of METADATA_TABLES, SHA-256 the file's raw bytes -> hex.
#   2. Emit one line per table, sorted by filename (byte order, C locale):
#          "{filename}  {sha256_hex}\n"
#   3. SHA-256 the UTF-8 encoding of that concatenated manifest -> hex.
#
# Filenames are inside the hash, so a rename or a table swap changes the
# fingerprint. Sorting is on the fixed table tuple, not on a directory listing,
# so filesystem ordering cannot perturb the result.
FINGERPRINT_SPEC = "sha256-of-sorted-name-digest-manifest/v1"


class PathValidationError(RuntimeError):
    """Raised by validate_paths() when the substrate contract is violated.

    `errors` holds every failed predicate, not just the first — a misconfigured
    root usually breaks several at once and reporting one at a time turns
    setup into a guessing game.
    """

    def __init__(self, errors: list[str], config_path: str | None = None) -> None:
        self.errors = list(errors)
        self.config_path = config_path
        head = f"path contract violated ({len(self.errors)} error(s))"
        if config_path:
            head += f" for {config_path}"
        super().__init__(head + ":\n" + "\n".join(f"  - {e}" for e in self.errors))


@dataclass(frozen=True)
class Paths:
    """Resolved, absolute, symlink-free path set. Constructed only by load_paths()."""

    dataroot: str
    meta_root: str
    version: str
    work_root: str
    out_root: str
    probe_out_root: str

    # --- derived read locations (never written) ---

    @property
    def version_dir(self) -> str:
        """Directory holding the 13 metadata tables."""
        return os.path.join(self.meta_root, self.version)

    @property
    def samples_dir(self) -> str:
        return os.path.join(self.dataroot, "samples")

    @property
    def sweeps_dir(self) -> str:
        return os.path.join(self.dataroot, "sweeps")

    def table(self, name: str) -> str:
        """Absolute path to one metadata table, e.g. table("sample_data.json")."""
        if name not in METADATA_TABLES:
            raise KeyError(f"{name!r} is not one of the {len(METADATA_TABLES)} metadata tables")
        return os.path.join(self.version_dir, name)

    @property
    def write_roots(self) -> tuple[str, ...]:
        return (self.work_root, self.out_root, self.probe_out_root)

    def as_dict(self) -> dict[str, str]:
        """Serialisable form for run_manifest.json / usable_scenes.json."""
        return asdict(self)


def _resolve(raw: str, field: str) -> str:
    """Expand ~ and ${VAR}, then make absolute and symlink-free.

    realpath (not abspath) because §1.8 records the dataroot *realpath*, and
    because commonpath disjointness is meaningless if one root reaches the
    other through a symlink.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise PathValidationError([f"paths.{field} must be a non-empty string, got {raw!r}"])
    expanded = os.path.expandvars(os.path.expanduser(raw.strip()))
    if "$" in expanded:
        raise PathValidationError([f"paths.{field}: unresolved environment variable in {expanded!r}"])
    return os.path.realpath(expanded)


def load_paths(config_path: str | os.PathLike = "configs/paths.yaml", *, validate: bool = True) -> Paths:
    """Read paths.yaml once and resolve it into a Paths object.

    Every field is required; there are no defaults. An omitted work_root that
    silently falls back to cwd is exactly the failure mode §1.8 exists to
    prevent.
    """
    config_path = os.fspath(config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise PathValidationError([f"expected a YAML mapping, got {type(raw).__name__}"], config_path)

    required = ("dataroot", "meta_root", "version", "work_root", "out_root", "probe_out_root")
    missing = [k for k in required if k not in raw]
    unknown = [k for k in raw if k not in required]
    problems = [f"missing required key: {k}" for k in missing]
    problems += [f"unknown key (tunables belong in pipeline_pilot.yaml): {k}" for k in unknown]
    if problems:
        raise PathValidationError(problems, config_path)

    version = raw["version"]
    if not isinstance(version, str) or not version.strip():
        raise PathValidationError([f"paths.version must be a non-empty string, got {version!r}"], config_path)

    paths = Paths(
        dataroot=_resolve(raw["dataroot"], "dataroot"),
        meta_root=_resolve(raw["meta_root"], "meta_root"),
        version=version.strip(),
        work_root=_resolve(raw["work_root"], "work_root"),
        out_root=_resolve(raw["out_root"], "out_root"),
        probe_out_root=_resolve(raw["probe_out_root"], "probe_out_root"),
    )
    if validate:
        validate_paths(paths, config_path=config_path)
    return paths


def _contains(outer: str, inner: str) -> bool:
    """True if `outer` is `inner` or an ancestor of it."""
    try:
        return os.path.commonpath([outer, inner]) == outer
    except ValueError:  # different drives / mixed absolute-relative
        return False


def validate_paths(
    paths: Paths,
    *,
    config_path: str | None = None,
    raise_on_error: bool = True,
) -> list[str]:
    """Assert the substrate contract. Returns the error list; raises by default.

    Checked, per §1.8:
      - dataroot exists, is a directory, is readable, contains samples/ + sweeps/
      - the version directory exists under meta_root and its NAME equals the
        configured version verbatim
      - all 13 metadata tables are present and readable
      - commonpath disjointness: no write root contains or is contained by
        dataroot (nor by meta_root, which may be split out later)

    Deliberately NOT checked: write-root existence or writability. Stages
    create their own roots under a manifest-guarded atomic-write discipline;
    demanding they pre-exist would make a fresh checkout fail validation.
    """
    errors: list[str] = []

    # --- dataroot structure and readability ---
    if not os.path.exists(paths.dataroot):
        errors.append(f"dataroot does not exist: {paths.dataroot}")
    elif not os.path.isdir(paths.dataroot):
        errors.append(f"dataroot is not a directory: {paths.dataroot}")
    elif not os.access(paths.dataroot, os.R_OK | os.X_OK):
        errors.append(f"dataroot is not readable: {paths.dataroot}")
    else:
        for sub in REQUIRED_BLOB_DIRS:
            d = os.path.join(paths.dataroot, sub)
            if not os.path.isdir(d):
                errors.append(f"dataroot missing required directory {sub}/: {d}")
            elif not os.access(d, os.R_OK | os.X_OK):
                errors.append(f"dataroot/{sub}/ is not readable: {d}")

    # --- version directory: existence AND exact name match ---
    version_dir = paths.version_dir
    if not os.path.isdir(version_dir):
        siblings = []
        if os.path.isdir(paths.meta_root):
            siblings = sorted(
                e.name for e in os.scandir(paths.meta_root) if e.is_dir() and e.name.startswith("v1.0")
            )
        hint = f" (found version directories: {siblings})" if siblings else ""
        errors.append(f"version directory {paths.version!r} not found under meta_root: {version_dir}{hint}")
    else:
        on_disk = os.path.basename(version_dir.rstrip(os.sep))
        if on_disk != paths.version:
            # Mini metadata against trainval blobs produces zero token matches,
            # which surfaces downstream as "no usable scenes" rather than as a
            # config error. Catch it here instead.
            errors.append(
                f"version directory name {on_disk!r} does not match configured version {paths.version!r}"
            )
        else:
            for table in METADATA_TABLES:
                t = os.path.join(version_dir, table)
                if not os.path.isfile(t):
                    errors.append(f"missing metadata table: {t}")
                elif not os.access(t, os.R_OK):
                    errors.append(f"metadata table is not readable: {t}")

    # --- commonpath disjointness between read roots and every write root ---
    read_roots = {"dataroot": paths.dataroot, "meta_root": paths.meta_root}
    write_roots = {
        "work_root": paths.work_root,
        "out_root": paths.out_root,
        "probe_out_root": paths.probe_out_root,
    }
    for r_name, r in read_roots.items():
        for w_name, w in write_roots.items():
            if r == w:
                errors.append(f"{w_name} is identical to {r_name}: {w}")
            elif _contains(r, w):
                errors.append(f"{w_name} is inside {r_name} (dataroot is read-only): {w} under {r}")
            elif _contains(w, r):
                errors.append(f"{r_name} is inside {w_name}; a write root must never contain the substrate: {r} under {w}")

    # --- write roots must not nest inside one another (probe hard separation, §8) ---
    for a_name, a in write_roots.items():
        for b_name, b in write_roots.items():
            if a_name >= b_name:
                continue
            if a == b:
                errors.append(f"{a_name} and {b_name} are the same directory: {a}")
            elif _contains(a, b) or _contains(b, a):
                errors.append(f"{a_name} and {b_name} are nested; probe and pipeline outputs must stay separate: {a} / {b}")

    if errors and raise_on_error:
        raise PathValidationError(errors, config_path)
    return errors


def metadata_file_digests(paths: Paths, *, chunk_size: int = 1 << 20) -> dict[str, str]:
    """SHA-256 hex of each of the 13 metadata tables, keyed by filename.

    Exposed separately from the fingerprint so a mismatch can be localised to a
    single table instead of reported as one opaque hash difference.
    """
    digests: dict[str, str] = {}
    for name in sorted(METADATA_TABLES):
        h = hashlib.sha256()
        with open(paths.table(name), "rb") as fh:
            for block in iter(lambda: fh.read(chunk_size), b""):
                h.update(block)
        digests[name] = h.hexdigest()
    return digests


def metadata_fingerprint(paths: Paths, *, with_digests: bool = False):
    """Sorted SHA-256 fingerprint of the 13 metadata JSONs (§1.8).

    Construction is pinned by FINGERPRINT_SPEC; see the constant for the exact
    manifest format. Recorded in usable_scenes.json and in every stage
    manifest; every consumer verifies the match before reading an allowlist.

    Returns the hex digest, or (hex_digest, per_file_digests) if with_digests.
    """
    digests = metadata_file_digests(paths)
    manifest = "".join(f"{name}  {digests[name]}\n" for name in sorted(digests))
    fingerprint = hashlib.sha256(manifest.encode("utf-8")).hexdigest()
    return (fingerprint, digests) if with_digests else fingerprint


def assert_dataroot_read_only(paths: Paths, target: str | os.PathLike) -> str:
    """Guard for write sites: refuse any path resolving inside dataroot/meta_root.

    Dataroot is read-only by convention (§1.8); this is the convention made
    executable. Route every open(..., "w") through it, and assert in
    tests/unit that no module writes to a dataroot path.
    """
    resolved = os.path.realpath(os.path.expanduser(os.fspath(target)))
    for name, root in (("dataroot", paths.dataroot), ("meta_root", paths.meta_root)):
        if _contains(root, resolved):
            raise PathValidationError([f"refusing to write inside read-only {name}: {resolved}"])
    return resolved


if __name__ == "__main__":  # resolve + fingerprint the configured substrate
    import json
    import sys

    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/paths.yaml"
    try:
        p = load_paths(cfg)
    except PathValidationError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(2)
    fp, per_file = metadata_fingerprint(p, with_digests=True)
    print(json.dumps(
        {
            "config": os.path.realpath(cfg),
            "paths": p.as_dict(),
            "version_dir": p.version_dir,
            "fingerprint_spec": FINGERPRINT_SPEC,
            "metadata_fingerprint": fp,
            "metadata_file_sha256": per_file,
        },
        indent=2,
    ))
