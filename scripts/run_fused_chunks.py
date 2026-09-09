"""Sequential fused-capture chunks, nuScenes release, then CVAT publication.

Run with the ano_pipe Python in tmux. Defaults to chunks 0001 through 0010,
using chunk 0000's paths, priors, and model pins. Records failed chunks and continues with the next chunk.
"""

import argparse
import atexit
import errno
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from pipeline.common.paths import load_paths, metadata_fingerprint
from scripts.prepare_run_exports import prepare


def stage_images(work, dataroot, share, prefix):
    for export in (work / "cvat_export").glob("*/instances.json"):
        for image in json.loads(export.read_text())["images"]:
            relative = Path(image["file_name"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe image path: {relative}")
            source, target = dataroot / relative, share / prefix / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if not os.path.samefile(source, target) and source.read_bytes() != target.read_bytes():
                    raise ValueError(f"CVAT share contains a different image: {target}")
                continue
            try:
                os.link(source.resolve(), target)
            except OSError as exc:
                if exc.errno != errno.EXDEV:
                    raise
                shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", nargs="*", default=[f"{i:04d}" for i in range(1, 11)])
    parser.add_argument("--base-paths", default="configs/paths_b_fused_chunk_0000.yaml")
    parser.add_argument("--share-root", default="/home/mt/Zami/nuscenes")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO)
    base = load_paths(args.base_paths)
    queue_root = Path(base.work_root).parent
    queue_root.mkdir(parents=True, exist_ok=True)
    lock = (queue_root / ".fused_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scenes = {s["name"] for s in json.loads(Path(base.table("scene.json")).read_text())}
    priors = Path(base.out_root) / "priors/priors_pilot_v0.json"
    assert json.loads(priors.read_text())["derived_from"]["metadata_fingerprint"] == metadata_fingerprint(base)
    env_base = os.environ.copy()
    for stage, prefix in (("stage3_proposals", "PROPOSAL"), ("stage3b_track2d", "TRACK2D"),
                          ("stage3_finetuned", "ARMB"), ("stage4_masks", "MASK")):
        checkpoint = json.loads((Path(base.work_root) / stage / "run_manifest.json").read_text())["checkpoint"]
        env_base[prefix + "_MODEL_ID"] = checkpoint["model_id"]
        env_base[prefix + "_REVISION"] = checkpoint["revision"]
    # This capture has six cameras and fused clouds, as recorded by chunk 0000.
    # The generic default is an eight-camera/raw-LiDAR substrate.
    env_base.update(PY=sys.executable, DHAKASCENES_SUBSTRATE="dhaka6",
                    VLM_CHECK="0", VLM_USE_CHECKED="0", PYTHONUNBUFFERED="1")
    jobs = []
    for chunk in args.chunks:
        if len(chunk) != 4 or not chunk.isdigit() or chunk == "0000":
            parser.error("expected remaining chunk IDs 0001 through 0010")
        scene = f"{Path(base.dataroot).name}_chunk_{chunk}"
        if scene not in scenes:
            parser.error(f"scene missing: {scene}")
        work = queue_root / f"chunk_{chunk}"
        config = base.as_dict()
        config.update(work_root=str(work),
                      out_root=str(Path(base.out_root).parent / f"chunk_{chunk}"),
                      probe_out_root=str(Path(base.probe_out_root).parent / f"chunk_{chunk}"))
        work.mkdir(parents=True, exist_ok=True)
        cfg = work / "paths.yaml"
        cfg.write_text(yaml.safe_dump(config))
        target_priors = Path(config["out_root"]) / "priors/priors_pilot_v0.json"
        target_priors.parent.mkdir(parents=True, exist_ok=True)
        if target_priors.exists():
            assert target_priors.read_bytes() == priors.read_bytes(), "different existing priors"
        else:
            shutil.copy2(priors, target_priors)
        load_paths(cfg)
        prepare(work, Path(env_base.get("EXPORT_ROOT", REPO / "export")) / scene)
        jobs.append((chunk, scene, work, cfg))
    print("Queued chunks: " + " ".join(j[0] for j in jobs), flush=True)
    if args.prepare_only:
        return
    # Bash reads a long-running script incrementally. Keep its bytes stable
    # while repository edits may continue during a multi-day queue.
    fd, runner = tempfile.mkstemp(prefix=".run_stages_", suffix=".sh", dir=REPO / "scripts")
    os.close(fd)
    shutil.copyfile(REPO / "scripts/run_stages.sh", runner)
    atexit.register(lambda: Path(runner).unlink(missing_ok=True))
    for chunk, scene, work, cfg in jobs:
        release_meta = Path(env_base.get("EXPORT_ROOT", REPO / "export")) / scene / "boxes/release_meta.json"
        if (work / ".fused_queue_complete").exists() and release_meta.exists():
            print(f"Skipping completed chunk {chunk}", flush=True)
            continue
        env = dict(env_base, DHAKASCENES_PATHS_CONFIG=str(cfg), EXPORT_NAME=scene,
                   CVAT_SHARE_PREFIX=scene + "/", CVAT_HOST="http://localhost:8081",
                   CVAT_PIPELINE_PROJECT=f"{scene} (2D)",
                   CVAT_PIPELINE_3D_PROJECT=f"{scene} (3D)", CVAT_ROAD_PROJECT=f"{scene} (road)")
        print(f"Starting chunk {chunk}; exports: {REPO / 'export' / scene}", flush=True)
        try:
            if not (work / ".fused_annotations_complete").exists():
                subprocess.run(["bash", runner, "0", "1", "3", "3b", "3f", "3m",
                                "4", "5", "6", "7", "8", "road", "eval", "viz", "--scenes", scene],
                               env=env, check=True)
                (work / ".fused_annotations_complete").touch()
            subprocess.run(["bash", runner, "release", "--scenes", scene],
                           env=env, check=True)
            stage_images(work, Path(base.dataroot), Path(args.share_root), scene)
            subprocess.run(["bash", runner, "cvat", "cvat3d", "cvatroad", "--scenes", scene],
                           env=env, check=True)
            (work / ".fused_queue_complete").touch()
            print(f"CHUNK {chunk} COMPLETE AND PUBLISHED", flush=True)
        except (subprocess.CalledProcessError, OSError, ValueError) as exc:
            (work / ".fused_queue_failed").write_text(str(exc) + "\n")
            print(f"CHUNK {chunk} FAILED: {exc}; continuing with the next chunk", flush=True)
            continue
        (work / ".fused_queue_failed").unlink(missing_ok=True)



if __name__ == "__main__":
    main()
