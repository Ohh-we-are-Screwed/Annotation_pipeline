#!/usr/bin/env python3
"""Publish the 3D point-cloud tasks to CVAT: our cuboids, the human twins, the A/B pass.

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
    python -m scripts.cvat_setup_3d --which double --assignee-a ann_a --assignee-b ann_b

--run-tag stamps OUR task names with the run that produced the cuboids
(C30, same contract as cvat_setup.py) so successive runs land side by side;
the answer-key twins stay untagged — they are the dataset's own and do not
change between runs. --replace deletes and recreates ONLY the task names this
run would create; --replace-all-runs deletes OUR tasks from every run (any
tag, and the untagged legacy names) before publishing. Neither ever touches
the answer key (C13) — and note each kept run holds its own copy of the
point-cloud archives, ~45-80 MB per scene of server storage.

THE FIVE CUBOID ATTRIBUTES (spec §7). Every label declares record_token,
track_id, attribute, uncertain and uncertain_reason, because a CVAT project's
label schema is written ONCE, at creation, and an attribute the schema does not
name is dropped by the importer WITHOUT AN ERROR — the provenance would look
published and simply not be there. The declaration shape is the one CVAT 2.72
was measured to accept, docs/evidence/2026-09-08-cvat-3d-roundtrip.md §5.

--which double PUBLISHES THE DOUBLE-ANNOTATION PASS: two projects, "<base>
— double pass A" and "<base> — double pass B", holding the same blank tasks
(<work_root>/cvat_export_3d_double, written by
`export_cvat_3d.py --frames <double_annotation.json> --blank --out-subdir
cvat_export_3d_double`). Two annotators fill them independently and the
agreement between the passes is what the release reports, so those tasks hold
HUMAN WORK and this script never deletes them: --replace and --reimport are
REFUSED for `double` and --replace-all-runs is ignored; an existing task is
skipped, never rebuilt. Delete one by hand in the UI if that is really meant.

EVERY TASK THIS SCRIPT CREATES IS RECORDED IN A LEDGER, by default
<work_root>/stage10_human/cvat_tasks.json — the index scripts/import_cvat_3d.py
reads to find the tasks, their kind (review / double_A / double_B), the scene
they belong to and the frames.json that maps a CVAT frame back to its sample
token. A published task missing from the ledger is human work the importer will
never read, so an existing task the ledger does not know is recorded too.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cvat_sdk import make_client  # noqa: E402
from cvat_sdk.core.proxies.tasks import ResourceType  # noqa: E402

from pipeline.common.paths import load_paths  # noqa: E402
from scripts.cvat_setup import (  # noqa: E402  (one naming contract, C30)
    category_disagreements,
    task_name,
    undeclared_attributes,
    undeclared_labels,
    wipe_targets,
)

# Env-overridable like the 2D publish's CVAT_PIPELINE_PROJECT (run_stages.sh),
# so one run can land its cuboids in a project of its own — the day-1 chunked
# run publishes each chunk to "day1_chunk_NNNN (3D)". `or`, not a default arg:
# an exported-but-empty variable must fall back, not create a project named "".
# GT_PROJECT deliberately has no knob (C13): the answer key is never renamed
# by the pipeline's own output settings.
OURS_PROJECT = os.environ.get("CVAT_PIPELINE_3D_PROJECT") or "OUR PIPELINE — machine pre-annotations (3D)"
GT_PROJECT = "nuScenes GT — HUMAN answer key (3D)"
OURS_SUFFIX = "3D — OUR PIPELINE output"
GT_SUFFIX = "3D — nuScenes HUMAN answer key"
DOUBLE_SUFFIX_A = "3D — double pass A"
DOUBLE_SUFFIX_B = "3D — double pass B"
GT_LABEL_COLOR = "#2ecc71"
FORMAT = "Datumaro 3D 1.0"
DOUBLE_SUBDIR = "cvat_export_3d_double"
LEDGER_NAME = os.path.join("stage10_human", "cvat_tasks.json")

# The nuScenes attribute vocabulary, empty option FIRST so the dropdown opens on
# "not answered" — which is also `default_value`, and a select's default has to
# be one of its own values. It is the same vocabulary the release schema uses
# (pipeline/common/schemas.py), so a reviewer's answer needs no translation.
ATTRIBUTE_VALUES = ["", "vehicle.moving", "vehicle.stopped", "vehicle.parked", "pedestrian.moving", "pedestrian.standing",
                    "pedestrian.sitting_lying_down", "cycle.with_rider", "cycle.without_rider"]


def cuboid_attribute_specs() -> list[dict]:
    """The five attributes every cuboid label declares, in CVAT's own JSON.

    Shape measured against CVAT 2.72, not read off a doc page
    (docs/evidence/2026-09-08-cvat-3d-roundtrip.md §5): `default_value` is
    always a STRING (including "false" and "0"), and a number's `values` are
    (min, max, step). Fresh dicts on every call — the caller hands one list per
    label to the project create, and a shared dict would let an edit to one
    label's schema reach the others.

    `track_id` is a CVAT-internal attribute name. It is accepted and stored and
    the reviewer sees it, but the Datumaro export overwrites it with CVAT's own
    per-task track index — identity comes back on `record_token`, which is why
    that one is here too (§2 of the same evidence).
    """
    return [
        {"name": "record_token", "input_type": "text", "mutable": False, "default_value": "", "values": [""]},
        {"name": "track_id", "input_type": "number", "mutable": True, "default_value": "0",
         "values": ["0", "999999999", "1"]},
        {"name": "attribute", "input_type": "select", "mutable": True, "default_value": "",
         "values": list(ATTRIBUTE_VALUES)},
        {"name": "uncertain", "input_type": "checkbox", "mutable": True, "default_value": "false",
         "values": ["false", "true"]},
        {"name": "uncertain_reason", "input_type": "text", "mutable": True, "default_value": "", "values": [""]},
    ]


def label_spec(names: list[str], color: str | None = None) -> list[dict]:
    """3D labels are cuboid-typed; a 2D-typed label cannot hold a cuboid shape."""
    return [
        {"name": name, "type": "cuboid", "attributes": cuboid_attribute_specs(),
         **({"color": color} if color else {})}
        for name in names
    ]


def double_project_names(base: str) -> tuple[str, str]:
    """The A/B project pair for a base project name.

    Derived from OURS_PROJECT (so the day-1 chunked run's per-chunk project
    knob carries over: "day1_chunk_0000 (3D)" -> "... — double pass A") and
    SEPARATE from it, because pass A must not see pass B's boxes or ours.
    """
    return (f"{base} — double pass A", f"{base} — double pass B")


def ledger_load(path: str) -> list[dict]:
    """The task ledger as a list of rows, empty when it does not exist yet."""
    if not os.path.isfile(path):
        return []
    with open(path) as fh:
        rows = json.load(fh)
    if not isinstance(rows, list):
        raise ValueError(f"{path} is not a task ledger (expected a JSON array of rows)")
    return rows


def _ledger_write(path: str, rows: list[dict]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(rows, fh, indent=1)
    os.replace(tmp, path)


def ledger_append(path: str, entry: dict) -> None:
    """Record one published task, in publish order.

    Rows are keyed by `task_id` — a CVAT task id is unique server-wide — so
    re-recording a task UPDATES its row in place instead of doubling it, and a
    ledger read twice by the importer cannot import the same task twice.
    """
    rows = ledger_load(path)
    for i, row in enumerate(rows):
        if row.get("task_id") == entry.get("task_id"):
            rows[i] = dict(entry)
            break
    else:
        rows.append(dict(entry))
    _ledger_write(path, rows)


def ledger_drop_tasks(path: str, task_ids) -> int:
    """Forget the rows of tasks that no longer exist (--replace deleted them).

    Returns how many rows went. A row pointing at a deleted task id is not
    harmless: the importer retrieves every ledger task by id and would fail on
    the 404 rather than importing the run that replaced it.
    """
    doomed = set(task_ids)
    rows = ledger_load(path)
    kept = [row for row in rows if row.get("task_id") not in doomed]
    if len(kept) != len(rows):
        _ledger_write(path, kept)
    return len(rows) - len(kept)


def find_user_id(users, username: str | None):
    """CVAT's numeric id for `username`, or None when there is no such user.

    None means "leave the task unassigned" — assigning A's work to whoever
    happens to be first in the user list would corrupt the A/B split silently.
    """
    if not username:
        return None
    for user in users:
        if getattr(user, "username", None) == username:
            return user.id
    return None


def task_assignee(task) -> str | None:
    """The username a task is already assigned to on the server, if any."""
    return getattr(getattr(task, "assignee", None), "username", None)


def ledger_row(project: str, project_id: int, task_id: int, kind: str, scene: str,
               frames_json: str, assignee: str | None, run_tag: str) -> dict:
    """One ledger row. `created_utc` is when the row was written, not when the
    task was created — the importer keys off task_id, and the timestamp is only
    there so a stale ledger can be told apart from a fresh one by eye."""
    return {
        "project": project,
        "project_id": project_id,
        "task_id": task_id,
        "kind": kind,
        "scene": scene,
        "frames_json": frames_json,
        "assignee": assignee,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_tag": run_tag,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--host", default=os.environ.get("CVAT_HOST", "http://localhost:8081"))
    parser.add_argument("--user", default=os.environ.get("CVAT_USER", "mt"))
    parser.add_argument("--password", default=os.environ.get("CVAT_PASSWORD", ""))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--which", choices=("both", "ours", "gt", "double"), default="both")
    parser.add_argument("--run-tag", default="",
                        help="stamp OUR task names with the run that produced the cuboids "
                             "(C30); the answer-key twins are never tagged")
    parser.add_argument("--replace", action="store_true",
                        help="delete and recreate the tasks this run would create (ours only; "
                             "REFUSED for --which double, whose tasks hold human work)")
    parser.add_argument("--replace-all-runs", action="store_true",
                        help="delete OUR tasks from EVERY run — any tag, and the untagged legacy "
                             "names — before publishing (ours only; the answer key and the A/B "
                             "passes are never touched). The wrapper's --cvat-replace arms this")
    parser.add_argument("--reimport", action="store_true",
                        help="existing task: replace its annotations without rebuilding frames "
                             "(ours only; REFUSED for --which double)")
    parser.add_argument("--assignee-a", default=None,
                        help="CVAT username to assign every pass-A task to (--which double)")
    parser.add_argument("--assignee-b", default=None,
                        help="CVAT username to assign every pass-B task to (--which double)")
    parser.add_argument("--ledger", default=None,
                        help="task ledger the importer reads; default <work_root>/" + LEDGER_NAME)
    args = parser.parse_args(argv)

    if not args.password:
        print("set CVAT_PASSWORD or pass --password", file=sys.stderr)
        return 2

    double = args.which == "double"
    if double and (args.replace or args.reimport):
        # Not a wipe with a warning: the A/B tasks ARE the second and third
        # opinions the release's agreement number is computed from, and both
        # --replace (delete + rebuild) and --reimport (replace annotations)
        # would destroy an annotator's work with no way back.
        flag = "--replace" if args.replace else "--reimport"
        print(f"!!! {flag} with --which double would destroy the A/B annotators' own work.",
              file=sys.stderr)
        print("!!! Existing A/B tasks are skipped, never rebuilt. Delete one by hand in the "
              "CVAT UI if that is really what you mean.", file=sys.stderr)
        return 2
    if double and args.replace_all_runs:
        print("note: --replace-all-runs does not apply to the A/B passes and is ignored")

    paths = load_paths(args.paths)
    ledger_path = args.ledger or os.path.join(paths.work_root, LEDGER_NAME)
    export_root = os.path.join(paths.work_root, DOUBLE_SUBDIR if double else "cvat_export_3d")
    names = sorted(
        n for n in os.listdir(export_root)
        if os.path.isfile(os.path.join(export_root, n, "task.zip"))
    ) if os.path.isdir(export_root) else []
    if args.scenes:
        names = [n for n in names if n in args.scenes]
    if not names:
        if double:
            print(f"nothing to publish under {export_root} — run scripts/export_cvat_3d.py "
                  f"--frames <double_annotation.json> --blank --out-subdir {DOUBLE_SUBDIR} first",
                  file=sys.stderr)
        else:
            print(f"nothing to publish under {export_root} — run scripts/export_cvat_3d.py first",
                  file=sys.stderr)
        return 2

    # (annotation-file kind, project, task suffix, label colour, replace,
    #  replace-all-runs, run tag, ledger kind, assignee username)
    variants = []
    if args.which in ("both", "ours"):
        variants.append(("ours", OURS_PROJECT, OURS_SUFFIX, None,
                         args.replace, args.replace_all_runs, args.run_tag, "review", None))
    if args.which in ("both", "gt"):
        # Never --replace, never --replace-all-runs, never tagged: the answer
        # key is the dataset's own and does not change between runs, so a
        # rebuild can only destroy review work (C13). It is not ledgered
        # either — the importer reads human ANSWERS, and the key is not one.
        variants.append(("gt", GT_PROJECT, GT_SUFFIX, GT_LABEL_COLOR, False, False, "", None, None))
    if double:
        name_a, name_b = double_project_names(OURS_PROJECT)
        # replace is False by the refusal above; replace-all-runs never reaches
        # the A/B projects. Both passes read the same blank export.
        variants.append(("blank", name_a, DOUBLE_SUFFIX_A, None,
                         args.replace, False, args.run_tag, "double_A", args.assignee_a))
        variants.append(("blank", name_b, DOUBLE_SUFFIX_B, None,
                         args.replace, False, args.run_tag, "double_B", args.assignee_b))

    ledgered = 0
    with make_client(host=args.host, credentials=(args.user, args.password)) as client:
        users = None  # listed once, only if a task is actually to be assigned
        for kind, project_name, suffix, color, replace, replace_all_runs, run_tag, ledger_kind, assignee in variants:
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
                      + (f", all {color}" if color else "")
                      + f", {len(cuboid_attribute_specs())} attributes each")
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
                # A project older than these attributes keeps dropping them,
                # run after run, while reporting success. Checked for the two
                # kinds that carry answers; the answer key holds no attribute
                # values, so its schema being short costs nothing.
                if ledger_kind is not None:
                    short = undeclared_attributes(
                        project, {attr["name"] for attr in cuboid_attribute_specs()})
                    if short and ledger_kind != "review":
                        # An A/B annotator cannot answer a field the UI does
                        # not show, so a short A/B project makes the whole pass
                        # worthless — refuse before creating a single task.
                        print(f"!!! project #{project.id} {project_name!r} does not declare "
                              f"cuboid attribute(s) the A/B pass needs: {', '.join(short)}",
                              file=sys.stderr)
                        print("!!! The annotators would have no field to answer in. Delete the "
                              "empty project and re-run so it is created with the full schema.",
                              file=sys.stderr)
                        return 2
                    if short:
                        print(f"  ! project #{project.id} does not declare attribute(s) "
                              f"{', '.join(short)} — CVAT will DROP them on every import; "
                              "provenance will be missing from what comes back", file=sys.stderr)

            existing = {t.name: t for t in project.get_tasks()}
            if replace_all_runs:
                targets = set(wipe_targets(existing, suffix))
            elif replace:
                targets = {task_name(scene, suffix, run_tag) for scene in names}
            else:
                targets = set()
            deleted = []
            for name, task in list(existing.items()):
                if name in targets:
                    task_id = task.id
                    task.remove()
                    existing.pop(name)
                    deleted.append(task_id)
                    print(f"  deleted task #{task_id}  {name}")
            if deleted and ledger_kind is not None:
                dropped = ledger_drop_tasks(ledger_path, deleted)
                if dropped:
                    print(f"  dropped {dropped} stale row(s) from {ledger_path}")

            known = {row.get("task_id") for row in ledger_load(ledger_path)}
            for scene in names:
                name = task_name(scene, suffix, run_tag)
                annotations = os.path.join(export_root, scene, f"annotations_{kind}.json")
                # Absolute: the importer is run from wherever, and this path is
                # how it finds the frame -> sample-token map (and the task.zip
                # beside it) for a task it only knows by id.
                frames_json = os.path.abspath(os.path.join(export_root, scene, "frames.json"))
                n_cuboids = sum(len(i["annotations"]) for i in json.load(open(annotations))["items"])
                if name in existing:
                    task = existing[name]
                    if ledger_kind is not None and task.id not in known:
                        # A published task the ledger does not know is human
                        # work the importer will never read. Record it as the
                        # SERVER has it — its own assignee, not this run's.
                        ledger_append(ledger_path, ledger_row(
                            project_name, project.id, task.id, ledger_kind, scene,
                            frames_json, task_assignee(task), run_tag))
                        known.add(task.id)
                        ledgered += 1
                        print(f"  {scene}: task #{task.id} was missing from the ledger, recorded")
                    if not args.reimport:
                        print(f"  {scene}: task #{task.id} exists, skipped "
                              + ("(A/B tasks are never rebuilt by this script)" if double else
                                 "(--reimport to replace annotations, --replace to rebuild)"))
                        continue
                    task.import_annotations(format_name=FORMAT, filename=annotations)
                    print(f"  {scene}: task #{task.id} annotations REPLACED "
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
                # A blank set still imports: that is what binds the project's
                # labels (and their attributes) to the task the annotator opens.
                task.import_annotations(format_name=FORMAT, filename=annotations)
                assigned = None
                if assignee:
                    if users is None:
                        users = list(client.users.list())
                    user_id = find_user_id(users, assignee)
                    if user_id is None:
                        print(f"  ! CVAT has no user {assignee!r} — task #{task.id} left "
                              "unassigned; assign it in the UI", file=sys.stderr)
                    else:
                        task.update({"assignee_id": user_id})
                        assigned = assignee
                if ledger_kind is not None:
                    ledger_append(ledger_path, ledger_row(
                        project_name, project.id, task.id, ledger_kind, scene,
                        frames_json, assigned, run_tag))
                    known.add(task.id)
                    ledgered += 1
                print(f"  {scene}: task #{task.id}  {task.size} frames, {n_cuboids} cuboids imported"
                      + (f", assigned to {assigned}" if assigned else ""))

    print(f"\nopen {args.host} -> Projects")
    if ledgered:
        print(f"{ledgered} task(s) recorded in {ledger_path} "
              "(scripts/import_cvat_3d.py reads it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
