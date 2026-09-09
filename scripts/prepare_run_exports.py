"""Keep CVAT exports in one delivery folder, with compatible work-tree links."""

import argparse
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common.paths import load_paths

EXPORT_DIRS = ("cvat_export", "cvat_export_gt", "cvat_export_3d",
               "cvat_export_3d_double", "cvat_export_road")


def prepare(work_root, destination):
    work_root, destination = Path(work_root), Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Check every collision before moving anything. Never merge two runs.
    for name in EXPORT_DIRS:
        source, target = work_root / name, destination / name
        if source.is_symlink():
            if source.resolve() != target:
                raise ValueError(f"{source} already points elsewhere: {source.resolve()}")
        elif source.exists() and target.exists():
            raise ValueError(f"both {source} and {target} exist; refusing to overwrite")
    for name in EXPORT_DIRS:
        source, target = work_root / name, destination / name
        if source.is_symlink():
            target.mkdir(exist_ok=True)
            continue
        if source.exists():
            shutil.move(str(source), str(target))
        else:
            target.mkdir(exist_ok=True)
        source.symlink_to(target, target_is_directory=True)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--export-root", default=str(Path(__file__).resolve().parents[1] / "export"))
    parser.add_argument("--export-name", default=None)
    args = parser.parse_args()
    paths = load_paths(args.paths)
    name = args.export_name or f"{Path(paths.dataroot).name}_{Path(paths.work_root).name}"
    if Path(name).name != name or name in (".", ".."):
        parser.error("export-name must be a single directory name")
    print(prepare(paths.work_root, Path(args.export_root) / name))


if __name__ == "__main__":
    main()
