#!/usr/bin/env python3
"""Publish the 3D point-cloud tasks to CVAT: our cuboids, and the human twins.

The 3D counterpart of scripts/cvat_setup.py, and it keeps that script's two
provenance levels (docs/CVAT_GUIDE.md, register C13):

  * separate PROJECTS, so the task lists never interleave and the answer key
    can be painted one uniform green;
  * task-name SUFFIXES within each project, which --replace keys off.

Reads <work_root>/cvat_export_3d/<scene>/{task.zip, annotations_*.json} from
scripts/export_cvat_3d.py and imports the JSON as "Datumaro 3D 1.0".

Each task uploads its own copy of the archive — CVAT has no way to share frame
data between two tasks, and the 2D pair already works this way. A scene is
~45 MB, so ours + twin for ten scenes is ~900 MB of server storage.

    python -m scripts.cvat_setup_3d                      # both projects, all scenes
    python -m scripts.cvat_setup_3d --which ours         # ours only
    python -m scripts.cvat_setup_3d --scenes scene-0061 --replace

--run-tag stamps OUR task names with the run that produced the cuboids
(C30, same contract as cvat_setup.py) so successive runs land side by side;
the answer-key twins stay untagged — they are the dataset's own and do not
change between runs. --replace deletes and recreates ONLY the task names this
run would create; --replace-all-runs deletes OUR tasks from every run (any
tag, and the untagged legacy names) before publishing. Neither ever touches
the answer key (C13) — and note each kept run holds its own copy of the
point-cloud archives, ~45-80 MB per scene of server storage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cvat_sdk import make_client  # noqa: E402
from cvat_sdk.core.proxies.tasks import ResourceType  # noqa: E402

from pipeline.common.paths import load_paths  # noqa: E402
from scripts.cvat_setup import (  # noqa: E402  (one naming contract, C30)
    category_disagreements,
    task_name,
    undeclared_labels,
    wipe_targets,
)

OURS_PROJECT = "OUR PIPELINE — machine pre-annotations (3D)"
GT_PROJECT = "nuScenes GT — HUMAN answer key (3D)"
OURS_SUFFIX = "3D — OUR PIPELINE output"
GT_SUFFIX = "3D — nuScenes HUMAN answer key"
GT_LABEL_COLOR = "#2ecc71"
FORMAT = "Datumaro 3D 1.0"


def label_spec(names: list[str], color: str | None = None) -> list[dict]:
    """3D labels are cuboid-typed; a 2D-typed label cannot hold a cuboid shape."""
    return [
        {"name": name, "type": "cuboid", "attributes": [], **({"color": color} if color else {})}
        for name in names
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--host", default=os.environ.get("CVAT_HOST", "http://localhost:8081"))
    parser.add_argument("--user", default=os.environ.get("CVAT_USER", "mt"))
    parser.add_argument("--password", default=os.environ.get("CVAT_PASSWORD", ""))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--which", choices=("both", "ours", "gt"), default="both")
    parser.add_argument("--run-tag", default="",
                        help="stamp OUR task names with the run that produced the cuboids "
                             "(C30); the answer-key twins are never tagged")
    parser.add_argument("--replace", action="store_true",
                        help="delete and recreate the tasks this run would create (ours only)")
    parser.add_argument("--replace-all-runs", action="store_true",
                        help="delete OUR tasks from EVERY run — any tag, and the untagged legacy "
                             "names — before publishing (ours only; the answer key is never "
                             "touched). The wrapper's --cvat-replace arms this")
    parser.add_argument("--reimport", action="store_true",
                        help="existing task: replace its annotations without rebuilding frames")
    args = parser.parse_args(argv)

    if not args.password:
        print("set CVAT_PASSWORD or pass --password", file=sys.stderr)
        return 2

    paths = load_paths(args.paths)
    export_root = os.path.join(paths.work_root, "cvat_export_3d")
    names = sorted(
        n for n in os.listdir(export_root)
        if os.path.isfile(os.path.join(export_root, n, "task.zip"))
    )
    if args.scenes:
        names = [n for n in names if n in args.scenes]
    if not names:
        print(f"nothing to publish under {export_root} — run scripts/export_cvat_3d.py first",
              file=sys.stderr)
        return 2

    variants = []
    if args.which in ("both", "ours"):
        variants.append(("ours", OURS_PROJECT, OURS_SUFFIX, None,
                         args.replace, args.replace_all_runs, args.run_tag))
    if args.which in ("both", "gt"):
        # Never --replace, never --replace-all-runs, never tagged: the answer
        # key is the dataset's own and does not change between runs, so a
        # rebuild can only destroy review work (C13).
        variants.append(("gt", GT_PROJECT, GT_SUFFIX, GT_LABEL_COLOR, False, False, ""))

    with make_client(host=args.host, credentials=(args.user, args.password)) as client:
        for kind, project_name, suffix, color, replace, replace_all_runs, run_tag in variants:
            print(f"\n=== {project_name}")
            project = next((p for p in client.projects.list() if p.name == project_name), None)
            per_scene = []
            for scene in names:
                with open(os.path.join(export_root, scene, f"annotations_{kind}.json")) as fh:
                    per_scene.append(
                        (scene,
                         [l["name"] for l in json.load(fh)["categories"]["label"]["labels"]]))
            disagree = category_disagreements(per_scene)
            if disagree:
                print(f"!!! the scenes under {export_root} do not agree on their "
                      f"{kind} labels — some are a different taxonomy era's leftovers:",
                      file=sys.stderr)
                for scene, delta in disagree.items():
                    print(f"!!!   {scene}: missing {delta['missing'] or '-'}, "
                          f"extra {delta['extra'] or '-'} vs {names[0]}", file=sys.stderr)
                print("!!! Re-export every scene from the current run, then publish.",
                      file=sys.stderr)
                return 2
            labels = per_scene[0][1]
            if project is None:
                project = client.projects.create(
                    {"name": project_name, "labels": label_spec(labels, color)}
                )
                print(f"created project #{project.id} with {len(labels)} cuboid labels"
                      + (f", all {color}" if color else ""))
            else:
                print(f"project #{project.id} exists")
                # Same refusal as cvat_setup.main: a label the project does not
                # declare fails the Datumaro import only AFTER the task stands.
                missing = undeclared_labels(
                    (label.name for label in project.get_labels()), labels)
                if missing:
                    print(f"!!! project #{project.id} {project_name!r} does not declare "
                          f"label(s) this export carries: {', '.join(missing)}", file=sys.stderr)
                    print("!!! Append them to the project (CVAT UI label editor) and re-run, "
                          "or delete the project to rebuild it from this export (loses "
                          "CVAT-side edits).", file=sys.stderr)
                    return 2

            existing = {t.name: t for t in project.get_tasks()}
            if replace_all_runs:
                targets = set(wipe_targets(existing, suffix))
            elif replace:
                targets = {task_name(scene, suffix, run_tag) for scene in names}
            else:
                targets = set()
            for name, task in list(existing.items()):
                if name in targets:
                    task_id = task.id
                    task.remove()
                    existing.pop(name)
                    print(f"  deleted task #{task_id}  {name}")

            for scene in names:
                name = task_name(scene, suffix, run_tag)
                annotations = os.path.join(export_root, scene, f"annotations_{kind}.json")
                n_cuboids = sum(len(i["annotations"]) for i in json.load(open(annotations))["items"])
                if name in existing:
                    if not args.reimport:
                        print(f"  {scene}: task #{existing[name].id} exists, skipped "
                              "(--reimport to replace annotations, --replace to rebuild)")
                        continue
                    existing[name].import_annotations(format_name=FORMAT, filename=annotations)
                    print(f"  {scene}: task #{existing[name].id} annotations REPLACED "
                          f"({n_cuboids} cuboids)")
                    continue
                task = client.tasks.create_from_data(
                    spec={"name": name, "project_id": project.id,
                          "sorting_method": "lexicographical"},
                    resource_type=ResourceType.LOCAL,
                    resources=[os.path.join(export_root, scene, "task.zip")],
                    data_params={"image_quality": 90},
                )
                task.fetch()
                if task.dimension != "3d":
                    # A task CVAT read as 2D would take no cuboid: the archive
                    # did not carry the pointcloud/ layout it expects.
                    print(f"  {scene}: task #{task.id} came back dimension={task.dimension}, not 3d",
                          file=sys.stderr)
                    return 2
                task.import_annotations(format_name=FORMAT, filename=annotations)
                print(f"  {scene}: task #{task.id}  {task.size} frames, {n_cuboids} cuboids imported")

    print(f"\nopen {args.host} -> Projects")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
