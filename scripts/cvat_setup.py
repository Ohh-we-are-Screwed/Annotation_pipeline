#!/usr/bin/env python3
"""Create CVAT tasks for every scene and import the Stage 3+4 pre-annotations.

Runs against a local CVAT (docker compose, share = nuScenes dataroot) using
cvat-sdk. The target project (--project) holds the taxonomy phrases as labels,
read from the export's own `categories` (with `score` and `suppressed`
attributes so the COCO `attributes` survive import); one task per scene is
created FROM THE SHARE — no image bytes are copied — and that scene's
`instances.json` (scripts/export_cvat_coco.py) is imported as COCO 1.0.

PROVENANCE IS THE PROJECT, not just the task name. Machine output and the
human answer key live in SEPARATE projects — the wrapper publishes the
pipeline into one and the GT twins into another, with --label-color painting
every answer-key label one uniform green. Same-named tasks in one flat list,
distinguishable only by suffix, proved genuinely confusing to review.

Idempotent-ish: scenes whose task name already exists in the project are
skipped, so a partial run can simply be re-run.

    CVAT_PASSWORD=... python scripts/cvat_setup.py --user mt [--scenes scene-0061]
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

DEFAULT_PROJECT = "DhakaScenes pilot — stages 3+4 (2D review)"


def label_spec(phrases: list[str], color: str | None = None) -> list[dict]:
    return [
        {
            "name": phrase,
            **({"color": color} if color else {}),
            "attributes": [
                {
                    "name": "score",
                    "input_type": "number",
                    "mutable": False,
                    "default_value": "0",
                    "values": ["0", "1", "0.01"],
                },
                {
                    "name": "suppressed",
                    "input_type": "checkbox",
                    "mutable": False,
                    "default_value": "false",
                    "values": ["false"],
                },
            ],
        }
        for phrase in phrases
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--host", default=os.environ.get("CVAT_HOST", "http://localhost:8081"))
    parser.add_argument("--user", default=os.environ.get("CVAT_USER", "mt"))
    parser.add_argument("--password", default=os.environ.get("CVAT_PASSWORD", ""))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--export-dir", default="cvat_export",
                        help="subdirectory of work_root holding <scene>/instances.json")
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help="CVAT project to publish into; created if missing. Machine output "
                             "and the human answer key belong in DIFFERENT projects")
    parser.add_argument("--label-color", default=None, metavar="HEX",
                        help="paint EVERY label this one color on project creation, e.g. "
                             "'#2ecc71' for the answer key — one visual voice for human truth. "
                             "Omit to let CVAT assign distinct per-class colors. No effect on "
                             "a project that already exists")
    parser.add_argument("--task-suffix", default="(pre-annotated)",
                        help="task name suffix, e.g. '(nuScenes GT)' for the GT twins")
    parser.add_argument("--reimport", action="store_true",
                        help="re-upload annotations into EXISTING tasks (replaces any manual edits)")
    parser.add_argument("--replace", action="store_true",
                        help="DELETE every task in the project whose name ends with --task-suffix, "
                             "then create it fresh from the current export. This is how a new "
                             "pipeline run replaces its own previous output; tasks carrying any "
                             "other suffix (the nuScenes answer keys) are never touched. Deletes "
                             "CVAT-side annotation edits along with the task.")
    args = parser.parse_args(argv)
    if not args.password:
        print("set CVAT_PASSWORD or pass --password", file=sys.stderr)
        return 2

    paths = load_paths(args.paths)
    export_root = os.path.join(paths.work_root, args.export_dir)
    names = sorted(
        n for n in os.listdir(export_root)
        if os.path.isfile(os.path.join(export_root, n, "instances.json"))
    )
    if args.scenes:
        names = [n for n in names if n in args.scenes]
    if not names:
        print(f"nothing to import under {export_root}", file=sys.stderr)
        return 2

    with make_client(host=args.host, credentials=(args.user, args.password)) as client:
        # --- project, created once ------------------------------------------
        project = next(
            (p for p in client.projects.list() if p.name == args.project), None
        )
        if project is None:
            with open(os.path.join(export_root, names[0], "instances.json")) as fh:
                phrases = [c["name"] for c in json.load(fh)["categories"]]
            project = client.projects.create(
                {"name": args.project, "labels": label_spec(phrases, args.label_color)}
            )
            print(f"created project #{project.id} {args.project!r} with {len(phrases)} labels"
                  + (f", all {args.label_color}" if args.label_color else ""))
        else:
            print(f"project #{project.id} {args.project!r} exists")

        existing = {t.name: t for t in project.get_tasks()}

        # --replace: this run owns EXACTLY the task names it is about to create,
        # and nothing else. Matching is by exact name, not by id (ids shift on
        # every republish) and not by suffix alone — the 3D task
        # "<scene> 3D — OUR PIPELINE output" shares the suffix but is built by a
        # different exporter, so a suffix match would delete something this
        # script cannot recreate.
        if args.replace:
            targets = {f"{scene} {args.task_suffix}" for scene in names}
            doomed = [t for name, t in existing.items() if name in targets]
            for task in doomed:
                task_id, task_name = task.id, task.name
                task.remove()
                print(f"  deleted task #{task_id}  {task_name}")
            if not doomed:
                print("  nothing to delete: no task carries a name this run will create")
            existing = {n: t for n, t in existing.items() if n not in targets}

        for scene in names:
            task_name = f"{scene} {args.task_suffix}"
            coco_path = os.path.join(export_root, scene, "instances.json")
            if task_name in existing:
                if not args.reimport:
                    print(f"  {scene}: task exists, skipped (--reimport to replace annotations)")
                    continue
                task = existing[task_name]
                task.import_annotations(format_name="COCO 1.0", filename=coco_path)
                print(f"  {scene}: task #{task.id} annotations REPLACED from {coco_path}")
                continue
            with open(coco_path) as fh:
                doc = json.load(fh)
            share_files = [img["file_name"] for img in doc["images"]]
            task = client.tasks.create_from_data(
                spec={
                    "name": task_name,
                    "project_id": project.id,
                    # keyframe/channel order is the sort of the dataroot paths;
                    # lexicographical keeps it stable and matches the COCO ids.
                    "sorting_method": "lexicographical",
                },
                resource_type=ResourceType.SHARE,
                resources=share_files,
                data_params={"image_quality": 90},
            )
            task.import_annotations(format_name="COCO 1.0", filename=coco_path)
            print(f"  {scene}: task #{task.id}  {len(share_files)} images, "
                  f"{len(doc['annotations'])} pre-annotations imported")

    print(f"\nopen {args.host} -> Projects -> {args.project!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
