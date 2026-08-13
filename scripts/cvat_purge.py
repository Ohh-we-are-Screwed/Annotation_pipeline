#!/usr/bin/env python3
"""Delete EVERY task and project on the CVAT server. Clean slate, no filter.

This is the `--clean-slate` half of scripts/run_stages.sh, split out because it
is the one irreversible step in the chain and deserves to be readable on its
own. It does not consult task names, provenance, or the export directory: it
removes what is there.

WHAT THIS DESTROYS, stated plainly because CVAT has no undo:

  * the pipeline's own pre-annotated tasks — regenerable, that is the point;
  * the `— nuScenes HUMAN answer key` twins — regenerable from the dataset
    metadata via scripts/export_gt_coco.py + scripts/cvat_setup.py;
  * the 3D tasks and every cuboid in them — **NOT regenerable**. No committed
    script imports 3D cuboids (scripts/export_cvat_3d.py builds the point-cloud
    archive and says so in its own docstring); whatever produced them was
    ad-hoc and is gone. Deleting them is a one-way door.
  * anything a human drew or corrected inside any task (register C13).

Refuses to run without --yes, because a mistyped host is otherwise a silent
catastrophe against whichever CVAT answered.

    CVAT_PASSWORD=... python scripts/cvat_purge.py --host http://localhost:8081 --yes
"""

from __future__ import annotations

import argparse
import os
import sys

from cvat_sdk import make_client  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=os.environ.get("CVAT_HOST", "http://localhost:8081"))
    parser.add_argument("--user", default=os.environ.get("CVAT_USER", "mt"))
    parser.add_argument("--password", default=os.environ.get("CVAT_PASSWORD", ""))
    parser.add_argument("--yes", action="store_true", help="required; there is no undo")
    parser.add_argument(
        "--keep",
        nargs="*",
        default=None,
        metavar="SUBSTRING",
        help="spare any task whose name contains one of these substrings",
    )
    args = parser.parse_args(argv)

    if not args.password:
        print("set CVAT_PASSWORD or pass --password", file=sys.stderr)
        return 2
    if not args.yes:
        print("refusing to purge without --yes (this deletes every task and project)", file=sys.stderr)
        return 2

    with make_client(host=args.host, credentials=(args.user, args.password)) as client:
        tasks = list(client.tasks.list())
        spared, removed = [], 0
        print(f"{args.host}: {len(tasks)} task(s) present")
        for task in tasks:
            name = task.name
            if args.keep and any(k in name for k in args.keep):
                spared.append(name)
                print(f"  kept    #{task.id}  {name}")
                continue
            task_id = task.id
            task.remove()
            removed += 1
            print(f"  deleted #{task_id}  {name}")

        # Projects go last: deleting a project cascades to its tasks, and doing
        # it first would race the per-task deletes above into 404s.
        projects = list(client.projects.list())
        for project in projects:
            # A project holding a spared task must survive with it.
            if spared and any(t.name in spared for t in project.get_tasks()):
                print(f"  kept    project #{project.id}  {project.name} (holds a spared task)")
                continue
            project_id, project_name = project.id, project.name
            project.remove()
            print(f"  deleted project #{project_id}  {project_name}")

    print(f"purged {removed} task(s); {len(spared)} spared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
