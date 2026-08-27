#!/usr/bin/env python3
"""Create CVAT tasks for every scene and import the Stage 3+4 pre-annotations.

Runs against a local CVAT (docker compose, share = nuScenes dataroot) using
cvat-sdk. The target project (--project) holds the taxonomy phrases as labels,
read from the export's own `categories` (each carrying every attribute in
LABEL_ATTRIBUTES, so the COCO `attributes` survive import); one task per scene
is created FROM THE SHARE — no image bytes are copied — and that scene's
`instances.json` (scripts/export_cvat_coco.py) is imported as COCO 1.0.

A project's label schema is written ONCE, at creation. An attribute the schema
does not name is dropped by the importer without an error, so a project older
than an attribute quietly loses it on every publish; an existing project is
therefore checked against what the export actually carries and REFUSED when it
is short, rather than publishing provenance that is not there
(--accept-missing-attributes to publish anyway).

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


# The COCO `attributes` every label must declare for CVAT to keep them. CVAT
# DROPS an attribute the project's label schema does not name -- silently, with
# a successful import -- so this list and scripts/export_cvat_coco.py's
# `attributes` dict are one contract in two files.
#
# All five are `mutable: False`: they are properties of the box as the pipeline
# produced it, fixed for the life of the object, not per-frame annotations a
# reviewer tracks. A reviewer correcting geometry must not be able to make
# `source` say the detector saw a box it never saw.
LABEL_ATTRIBUTES: list[dict] = [
    {
        "name": "score",
        "input_type": "number",
        "mutable": False,
        "default_value": "0",
        "values": ["0", "1", "0.01"],  # CVAT reads a number's values as min;max;step
    },
    {
        "name": "suppressed",
        "input_type": "checkbox",
        "mutable": False,
        "default_value": "false",
        "values": ["false"],
    },
    # --- Stage 3b provenance (C27) -----------------------------------------
    # A closed set, so `select` and not free text: the reviewer sees which
    # boxes no detector ever fired on, and cannot invent an unlisted answer.
    # "human" is the checker gate's value (review_fix_sam31 frames mode): an
    # object a person drew because the pipeline missed it. mutable: False still
    # holds — provenance is set at creation, never edited into something else.
    {
        "name": "source",
        "input_type": "select",
        "mutable": False,
        "default_value": "yolo",
        "values": ["yolo", "recovered", "human"],
    },
    # An IDENTIFIER, not a quantity -- unique per (scene, channel) only, and
    # null on any box that never belonged to a 3b track. `text` is the only
    # input type that carries both without pretending the number means
    # something when compared or that its absence is a zero.
    {
        "name": "track_id",
        "input_type": "text",
        "mutable": False,
        "default_value": "",
        "values": [""],
    },
    # Propagation distance from the last real detection: 0 on every detected
    # box, higher the further a recovered box stands from evidence. The max is a
    # UI spinner bound picked well above anything reachable (a scene is ~40
    # keyframes, ~480 sweep frames), not a contract.
    {
        "name": "hops",
        "input_type": "number",
        "mutable": False,
        "default_value": "0",
        "values": ["0", "1000", "1"],
    },
]


def label_spec(phrases: list[str], color: str | None = None,
               extra_attributes: list[dict] | None = None) -> list[dict]:
    return [
        {
            "name": phrase,
            **({"color": color} if color else {}),
            "attributes": [dict(attr) for attr in (*LABEL_ATTRIBUTES, *(extra_attributes or []))],
        }
        for phrase in phrases
    ]


def undeclared_attributes(project, needed: set[str]) -> list[str]:
    """Attributes the export carries that some label of `project` does not declare.

    CVAT fixes a project's label schema when the project is created, and this
    script has never updated an existing one -- so a project created before an
    attribute existed keeps dropping it on every import, run after run, while
    reporting success. The check is per label because CVAT's schema is per label.
    """
    missing: set[str] = set()
    for label in project.get_labels():
        declared = {attr.name for attr in (label.attributes or [])}
        missing |= needed - declared
    return sorted(missing)


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
    parser.add_argument("--accept-missing-attributes", action="store_true",
                        help="publish into an EXISTING project whose labels do not declare every "
                             "attribute this export carries, knowing CVAT will drop the "
                             "undeclared ones on import. Without it such a project is refused, "
                             "because an import that silently loses provenance looks exactly like "
                             "one that kept it")
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

    # The taxonomy and the attribute set this run is about to import, read from
    # the export itself: the first scene names the labels, and the first scene
    # that HAS annotations names the attributes (a scene with no annotations
    # carries none, and its silence is not evidence that the run has none).
    phrases: list[str] = []
    needed: set[str] = set()
    extra_attributes: list[dict] = []
    for scene in names:
        with open(os.path.join(export_root, scene, "instances.json")) as fh:
            doc = json.load(fh)
        if not phrases:
            phrases = [c["name"] for c in doc["categories"]]
            # An export may declare attributes beyond LABEL_ATTRIBUTES (the
            # review export of scripts/review_fix_sam31.py does: review /
            # iou_before / iou_after). They join the schema of a project
            # CREATED by this run; an existing project is still checked
            # against what the export carries, exactly as before.
            known = {a["name"] for a in LABEL_ATTRIBUTES}
            extra_attributes = [a for a in (doc.get("info") or {}).get("cvat_label_attributes", [])
                                if a["name"] not in known]
        if doc["annotations"]:
            needed = {key for ann in doc["annotations"] for key in (ann.get("attributes") or {})}
            break

    with make_client(host=args.host, credentials=(args.user, args.password)) as client:
        # --- project, created once ------------------------------------------
        project = next(
            (p for p in client.projects.list() if p.name == args.project), None
        )
        if project is None:
            project = client.projects.create(
                {"name": args.project,
                 "labels": label_spec(phrases, args.label_color, extra_attributes)}
            )
            print(f"created project #{project.id} {args.project!r} with {len(phrases)} labels"
                  + (f", all {args.label_color}" if args.label_color else ""))
        else:
            print(f"project #{project.id} {args.project!r} exists")
            # A project's label schema is fixed at creation and this script does
            # not rewrite it: a labels PATCH is how CVAT DELETES labels, and
            # deleting a label deletes every annotation drawn with it. Losing
            # review work to repair a display attribute is the wrong trade, so
            # the mismatch is reported and the operator decides.
            undeclared = undeclared_attributes(project, needed)
            if undeclared:
                print(f"!!! project #{project.id} {args.project!r} was created before this export's "
                      f"attributes existed and does not declare: {', '.join(undeclared)}",
                      file=sys.stderr)
                print("!!! CVAT drops an undeclared attribute on import WITHOUT failing, so those "
                      "values would be absent from the review tasks with nothing to show for it.",
                      file=sys.stderr)
                if not args.accept_missing_attributes:
                    print(f"!!! Delete project {args.project!r} in the CVAT UI (Projects -> the "
                          "project's ... menu -> Delete) and re-run this publish: it recreates the "
                          "project with the full schema and rebuilds every task from the current "
                          "export. Only THIS project goes — the other one, and every 3D task, is "
                          "untouched. Its tasks are regenerated from work_root, but CVAT-side "
                          "annotation edits inside them are lost, exactly as --replace loses them.",
                          file=sys.stderr)
                    print("!!! To publish without these attributes instead, pass "
                          "--accept-missing-attributes.", file=sys.stderr)
                    return 2
                print("!!! --accept-missing-attributes: publishing anyway; the attributes above "
                      "will NOT appear in CVAT.", file=sys.stderr)

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
