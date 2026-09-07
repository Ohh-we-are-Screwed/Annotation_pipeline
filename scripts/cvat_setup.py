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

RUNS ARE THE THIRD PROVENANCE LEVEL (C30). --run-tag stamps every task name
with the run that produced its annotations — "<scene> <suffix> [<tag>]" — so
two runs' tasks stand side by side in one project and neither is mistaken for
the other. Without a tag the name is "<scene> <suffix>" exactly as before.
Deleting is opt-in either way: --replace removes only the names THIS publish
is about to create (same suffix, same tag); --replace-all-runs removes every
task in the project that this naming scheme could have produced for the
suffix, tagged or legacy-untagged, from any run — the wrapper's --cvat-replace
is what arms it.

Idempotent-ish: scenes whose task name already exists in the project are
skipped, so a partial run can simply be re-run — and since the tag is derived
from the SOURCE manifest, republishing the same run is a no-op rather than a
duplicate.

    CVAT_PASSWORD=... python scripts/cvat_setup.py --user mt [--scenes scene-0061]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
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


def task_name(scene: str, suffix: str, run_tag: str = "") -> str:
    """The one place a task name is built. '<scene> <suffix>' with no tag —
    byte-identical to every task this script ever created — or
    '<scene> <suffix> [<tag>]' when the publish names its run (C30)."""
    return f"{scene} {suffix} [{run_tag}]" if run_tag else f"{scene} {suffix}"


def wipe_targets(existing_names, suffix: str) -> list[str]:
    """Every name in `existing_names` that task_name() could have produced for
    `suffix` — any scene, any tag, or no tag at all. This is the
    --replace-all-runs matcher: it deliberately reaches across runs, which is
    exactly what --replace refuses to do.

    Scope is the CALLER'S project listing. Suffixes are not unique across
    exporters (the 3D "<scene> 3D — OUR PIPELINE output" ends with the 2D
    suffix), so this matcher is safe ONLY because machine 2D, machine 3D and
    the answer keys live in separate projects — the same separation the rest
    of this file leans on. Do not point it at a mixed project.
    """
    pat = re.compile(r"^.+ " + re.escape(suffix) + r"( \[[^\]]+\])?$")
    return sorted(n for n in existing_names if pat.match(n))


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


def category_disagreements(per_scene_categories) -> dict:
    """Scenes whose category SET differs from the first scene's.

    The export root accumulates scene dirs across runs (exporters overwrite
    only what they export), while the publish seeds its label universe from
    the alphabetically-first scene alone — so a scene left over from another
    taxonomy era either creates a short-labelled project (the import then
    crashes) or slides its old-taxonomy boxes silently into this run's tasks.
    Order differences are not disagreements: COCO ids are per-file and the
    importer maps by name.

    Takes (scene, categories) pairs in publish order; returns
    {scene: {"missing": [...], "extra": [...]}} versus the first scene,
    empty when every scene agrees.
    """
    pairs = list(per_scene_categories)
    if not pairs:
        return {}
    reference = set(pairs[0][1])
    out = {}
    for scene, cats in pairs[1:]:
        got = set(cats)
        if got != reference:
            out[scene] = {"missing": sorted(reference - got),
                          "extra": sorted(got - reference)}
    return out


def undeclared_labels(project_label_names, export_categories) -> list[str]:
    """Categories the export carries that the project's labels do not declare.

    CVAT refuses such an import only server-side, AFTER the task and its
    frames already stand — the first C28-superset publish (2026-09-02) died
    mid-import on 'an auto rickshaw' and left a broken zero-annotation task
    under the run's own name. Checked here instead, before the first task.
    A project declaring MORE labels than the export uses is not a mismatch.
    """
    return sorted(set(export_categories) - set(project_label_names))


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


def prefix_share(doc: dict, prefix: str) -> tuple[list[str], dict]:
    """(share resources, COCO doc to import) for a task whose images sit under
    `prefix` inside the CVAT share.

    The share is ONE mount for every substrate that ever published (2026-09-06:
    it still held pilot_1632's frames, and chunk_0006's task showed them under
    the new annotations because both name `samples/CAM_BACK/000008.jpg`). Each
    dataset is therefore staged under its own prefix, and the prefix must reach
    BOTH the frame list and the COCO file_names — CVAT binds annotations to
    frames by name. Empty prefix: byte-identical to the pre-prefix behaviour.
    """
    prefix = prefix.strip()
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    out = copy.deepcopy(doc)
    for img in out["images"]:
        img["file_name"] = prefix + img["file_name"]
    return [img["file_name"] for img in out["images"]], out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--share-prefix", default=os.environ.get("CVAT_SHARE_PREFIX", ""),
                        help="directory INSIDE the CVAT share under which this dataset's samples/ is "
                             "staged, e.g. 'day1_chunk_0006/'. Prefixed onto every frame path and "
                             "every COCO file_name. Default: $CVAT_SHARE_PREFIX, else none (legacy: "
                             "the share root IS the dataroot)")
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
    parser.add_argument("--run-tag", default="",
                        help="stamp task names with the run that produced the annotations: "
                             "'<scene> <suffix> [<tag>]'. Different runs then land side by side "
                             "instead of colliding into a skip, and republishing the SAME run "
                             "skips exactly as before. Empty (the default) keeps the untagged "
                             "legacy names. The wrapper derives it from the source stage's "
                             "run_manifest mtime (C30)")
    parser.add_argument("--reimport", action="store_true",
                        help="re-upload annotations into EXISTING tasks (replaces any manual edits)")
    parser.add_argument("--replace", action="store_true",
                        help="DELETE the exact task names THIS publish is about to create "
                             "(same scenes, same suffix, same --run-tag), then create them fresh "
                             "from the current export. Other runs' tags, other suffixes (the "
                             "nuScenes answer keys) and untagged legacy names are never touched. "
                             "Deletes CVAT-side annotation edits along with the task.")
    parser.add_argument("--replace-all-runs", action="store_true",
                        help="DELETE every task in the project that this exporter's naming could "
                             "have produced for --task-suffix — every scene, every run tag, and "
                             "the untagged legacy names — then publish fresh. This is the "
                             "wrapper's --cvat-replace: the clean-slate for the pipeline's own "
                             "output tasks, in this one project, with the answer-key project "
                             "untouched. Deletes CVAT-side annotation edits along with the tasks.")
    parser.add_argument("--accept-missing-attributes", action="store_true",
                        help="publish into an EXISTING project whose labels do not declare every "
                             "attribute this export carries, knowing CVAT will drop the "
                             "undeclared ones on import. Without it such a project is refused, "
                             "because an import that silently loses provenance looks exactly like "
                             "one that kept it")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    per_scene_categories: list[tuple[str, list[str]]] = []
    for scene in names:
        with open(os.path.join(export_root, scene, "instances.json")) as fh:
            doc = json.load(fh)
        per_scene_categories.append((scene, [c["name"] for c in doc["categories"]]))
        if not phrases:
            phrases = per_scene_categories[0][1]
            # An export may declare attributes beyond LABEL_ATTRIBUTES (the
            # review export of scripts/review_fix_sam31.py does: review /
            # iou_before / iou_after). They join the schema of a project
            # CREATED by this run; an existing project is still checked
            # against what the export carries, exactly as before.
            known = {a["name"] for a in LABEL_ATTRIBUTES}
            extra_attributes = [a for a in (doc.get("info") or {}).get("cvat_label_attributes", [])
                                if a["name"] not in known]
        if not needed and doc["annotations"]:
            needed = {key for ann in doc["annotations"] for key in (ann.get("attributes") or {})}
    disagree = category_disagreements(per_scene_categories)
    if disagree:
        print(f"!!! the scenes under {export_root} do not agree on their categories — "
              "some are a different taxonomy era's leftovers:", file=sys.stderr)
        for scene, delta in disagree.items():
            print(f"!!!   {scene}: missing {delta['missing'] or '-'}, "
                  f"extra {delta['extra'] or '-'} vs {per_scene_categories[0][0]}",
              file=sys.stderr)
        print("!!! Re-export every scene from the current run (delete the stale scene "
              "dirs or rerun the export step), then publish.", file=sys.stderr)
        return 2

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
            # A project's label schema is fixed at creation and this script
            # does not rewrite it — the mismatch is reported and the operator
            # decides. (Verified on server 2.72.1: a partial_update carrying
            # ONLY new label entries APPENDS them, keeping existing labels,
            # ids and annotations — that is the operator's least destructive
            # repair. Deletion happens only through an explicit per-label
            # delete, never as a side effect of appending.)
            missing_labels = undeclared_labels(
                (label.name for label in project.get_labels()), phrases)
            if missing_labels:
                print(f"!!! project #{project.id} {args.project!r} does not declare label(s) "
                      f"this export carries: {', '.join(missing_labels)}", file=sys.stderr)
                print("!!! CVAT would refuse the import only AFTER the task and its frames are "
                      "created, leaving a broken half-task standing under this publish's name.",
                      file=sys.stderr)
                print("!!! Least destructive fix: append the missing labels to the project "
                      "(CVAT UI -> the project's label editor), mirroring an existing label's "
                      "attributes, then re-run this publish. Or publish into a fresh --project; "
                      "or delete the project to rebuild it from this export (loses CVAT-side "
                      "edits).", file=sys.stderr)
                return 2
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

        # Two deletion modes, one loop (C30). --replace: this publish owns
        # EXACTLY the task names it is about to create — same scenes, same
        # suffix, same tag — and nothing else. Matching is by exact name, not
        # by id (ids shift on every republish) and not by suffix alone — the 3D
        # task "<scene> 3D — OUR PIPELINE output" shares the suffix but is
        # built by a different exporter, so a suffix match would delete
        # something this script cannot recreate. --replace-all-runs is that
        # suffix match, on purpose: every run's output for this suffix in THIS
        # project goes, which is safe only because the exporters keep separate
        # projects (see wipe_targets).
        if args.replace_all_runs:
            targets = set(wipe_targets(existing, args.task_suffix))
        elif args.replace:
            targets = {task_name(scene, args.task_suffix, args.run_tag) for scene in names}
        else:
            targets = set()
        if args.replace_all_runs or args.replace:
            doomed = [t for name, t in existing.items() if name in targets]
            for task in doomed:
                doomed_id, doomed_name = task.id, task.name
                task.remove()
                print(f"  deleted task #{doomed_id}  {doomed_name}")
            if not doomed:
                print("  nothing to delete: no task carries a name this "
                      + ("suffix could have produced" if args.replace_all_runs else "run will create"))
            existing = {n: t for n, t in existing.items() if n not in targets}

        for scene in names:
            name = task_name(scene, args.task_suffix, args.run_tag)
            coco_path = os.path.join(export_root, scene, "instances.json")
            with open(coco_path) as fh:
                doc = json.load(fh)
            share_files, doc = prefix_share(doc, args.share_prefix)
            upload_path = coco_path
            if args.share_prefix:
                # The file CVAT imports must carry the prefixed names too; the
                # exporter's instances.json is left as written (its names are
                # dataroot-relative, which is what the export/ bundle wants).
                upload_path = os.path.join(export_root, scene, "instances.share.json")
                with open(upload_path, "w") as fh:
                    json.dump(doc, fh)
            if name in existing:
                if not args.reimport:
                    print(f"  {scene}: task exists, skipped (--reimport to replace annotations)")
                    continue
                task = existing[name]
                task.import_annotations(format_name="COCO 1.0", filename=upload_path)
                print(f"  {scene}: task #{task.id} annotations REPLACED from {upload_path}")
                continue
            task = client.tasks.create_from_data(
                spec={
                    "name": name,
                    "project_id": project.id,
                    # keyframe/channel order is the sort of the dataroot paths;
                    # lexicographical keeps it stable and matches the COCO ids.
                    "sorting_method": "lexicographical",
                },
                resource_type=ResourceType.SHARE,
                resources=share_files,
                data_params={"image_quality": 90},
            )
            task.import_annotations(format_name="COCO 1.0", filename=upload_path)
            print(f"  {scene}: task #{task.id}  {len(share_files)} images, "
                  f"{len(doc['annotations'])} pre-annotations imported")

    print(f"\nopen {args.host} -> Projects -> {args.project!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
