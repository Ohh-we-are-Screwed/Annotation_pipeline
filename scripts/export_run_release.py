"""Export this work tree's Stage 9 annotations as a scoped nuScenes release."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.common.paths import load_paths
from scripts.export_release import main as export_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--blobs", choices=("hardlink", "copy", "symlink"), default="hardlink")
    parser.add_argument("--scenes", nargs="+", default=None)
    args = parser.parse_args()
    paths = load_paths(args.paths)
    work = Path(paths.work_root)
    scenes = args.scenes or sorted(p.parent.name for p in (work / "stage9_qa/scenes").glob("*/prelabels.jsonl"))
    if not scenes:
        parser.error("no Stage 9 scenes to export")
    return export_main([
        "--prelabels", str(work / "stage9_qa"), "--dataroot", paths.dataroot,
        "--version", paths.version, "--out", args.out, "--blobs", args.blobs,
        "--stage1-dir", str(work / "stage1_ingestion"),
        "--cvat-export-3d-dir", str(work / "cvat_export_3d"),
        "--stage-tree", str(work), "--chunk-name", Path(args.out).parent.name,
        "--scenes", *scenes,
    ])


if __name__ == "__main__":
    sys.exit(main())
