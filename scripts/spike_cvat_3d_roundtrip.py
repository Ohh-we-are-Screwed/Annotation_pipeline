#!/usr/bin/env python3
"""SPIKE (throwaway, kept for the record): does a CVAT "Datumaro 3D 1.0" round
trip preserve cuboid TRACKS and custom label ATTRIBUTES?

Tasks 14 (the CVAT export) and 16 (the importer that reads the reviewed task
back) encode object identity and the reviewer's answers in exactly those two
places, so the encoding has to be chosen from measurement, not from the format
docs. This script builds the smallest task that can answer the question —

    3 frames, 200 random points each, no related images
    labels `a car` / `a pedestrian`, each declaring the five attributes
        record_token (text), track_id (number), attribute (select),
        uncertain (checkbox), uncertain_reason (text)
    one car on all three frames as a TRACK (track_id + keyframe)
    one pedestrian on frame 1 only as a plain SHAPE (the control)
    one pedestrian on frames 0-1 as a SHORT TRACK, to see whether a track that
        ends before the last frame leaves a phantom cuboid on frame 2

— publishes it, imports the Datumaro JSON, exports the task back, and diffs
what came out against what went in: track membership, every attribute's value
AND its JSON type, position/rotation/scale to 1e-4, and the inner path of the
JSON inside the export zip (Task 16 has to find it).

    set -a; . ./.env; set +a
    python scripts/spike_cvat_3d_roundtrip.py
    # or, from a worktree with no .env of its own:
    python scripts/spike_cvat_3d_roundtrip.py --env-file /path/to/.env

Nothing is deleted. The operator's standing rule is that this script does not
get to remove anything from the CVAT server, so --cleanup is accepted and
refused: the spike project is left in place and its id printed, for the
operator to delete by hand if they want it gone. Re-running reuses the project
of the same name and adds another task to it.

Findings are written to docs/evidence/2026-09-08-cvat-3d-roundtrip.md by hand
from this script's stdout plus the raw export it leaves in --out-dir.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cvat_sdk import make_client  # noqa: E402
from cvat_sdk.core.proxies.tasks import ResourceType  # noqa: E402

from scripts.export_cvat_3d import cuboid, datumaro_document, item_skeleton, write_pcd  # noqa: E402

PROJECT_NAME = "SPIKE — 3D roundtrip (delete me)"
FORMAT = "Datumaro 3D 1.0"
LABELS = ["a car", "a pedestrian"]
SELECT_VALUES = [
    "vehicle.moving",
    "vehicle.stopped",
    "vehicle.parked",
    "pedestrian.moving",
    "pedestrian.standing",
    "pedestrian.sitting_lying_down",
    "cycle.with_rider",
    "cycle.without_rider",
    "",
]

# The five attributes Task 14 wants to carry. `track_id` is declared as a
# number deliberately: CVAT reserves that name internally (bindings.py's
# CVAT_INTERNAL_ATTRIBUTES), and whether a label may still declare it is one of
# the things this spike is here to find out.
ATTRIBUTE_SPECS = [
    {"name": "record_token", "input_type": "text", "mutable": False, "default_value": "", "values": []},
    {"name": "track_id", "input_type": "number", "mutable": False,
     "default_value": "0", "values": ["0", "100000", "1"]},
    {"name": "attribute", "input_type": "select", "mutable": True,
     "default_value": "", "values": SELECT_VALUES},
    {"name": "uncertain", "input_type": "checkbox", "mutable": True, "default_value": "false", "values": []},
    {"name": "uncertain_reason", "input_type": "text", "mutable": True, "default_value": "", "values": []},
]


def label_spec(attributes: list[dict]) -> list[dict]:
    return [{"name": name, "type": "cuboid", "attributes": attributes} for name in LABELS]


# --- what goes in -----------------------------------------------------------
# Three cuboids for the car (one per frame, same track), one for the pedestrian.
# Positions, yaw and scale are all distinct so a re-ordering of any triple is
# visible rather than accidentally symmetric.
CAR_ATTRS = {
    "track_id": 7,
    "record_token": "k0:CAM_FRONT:0",
    "keyframe": True,
    "attribute": "vehicle.moving",
    "uncertain": True,
    "uncertain_reason": "occluded by bus",
}
PED_ATTRS = {
    "record_token": "k1:CAM_FRONT:3",
    "attribute": "pedestrian.standing",
    "uncertain": False,
    "uncertain_reason": "",
}
SHORT_ATTRS = {
    "track_id": 9,
    "record_token": "k2:CAM_FRONT:9",
    "keyframe": True,
    "attribute": "pedestrian.moving",
    "uncertain": False,
    "uncertain_reason": "",
}
CAR_POSE = [
    ([10.1234, 2.5678, 0.9012], 0.3456, [4.2345, 1.8123, 1.6789]),
    ([12.4321, 2.8765, 0.9345], 0.4567, [4.2345, 1.8123, 1.6789]),
    ([14.9876, 3.1234, 0.9678], 0.5678, [4.2345, 1.8123, 1.6789]),
]
PED_POSE = ([5.5555, -3.3333, 0.7777], 1.2345, [0.7123, 0.6234, 1.7345])
SHORT_POSE = [
    ([-6.1111, 8.2222, 0.6666], 2.3456, [0.8123, 0.5234, 1.8345]),
    ([-6.9999, 9.4444, 0.6543], 2.4567, [0.8123, 0.5234, 1.8345]),
]


def build_input(tmp: str) -> tuple[str, str, dict]:
    """Write task.zip + annotations.json; return (zip, json, expected-by-key)."""
    staging = os.path.join(tmp, "_staging")
    os.makedirs(os.path.join(staging, "pointcloud"))
    rng = np.random.default_rng(20260908)
    for index in range(3):
        points = np.empty((200, 4), dtype="float32")
        points[:, :3] = rng.uniform(-20.0, 20.0, size=(200, 3))
        points[:, 3] = rng.uniform(0.0, 1.0, size=200)
        write_pcd(os.path.join(staging, "pointcloud", f"{index + 1:06d}.pcd"), points)

    zip_path = os.path.join(tmp, "task.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for root, _, files in os.walk(staging):
            for name in sorted(files):
                full = os.path.join(root, name)
                archive.write(full, os.path.relpath(full, staging))

    # Objects are keyed by (frame, record_token) on both sides: record_token is
    # unique per object and is the one attribute we can afford to assume back,
    # so an EXTRA exported cuboid (an interpolated or extrapolated track shape)
    # shows up as a key with no input counterpart rather than overwriting one.
    expected: dict[tuple[int, str], dict] = {}
    items = []
    for index in range(3):
        item = item_skeleton(index, [])
        item["related_images"] = []
        position, yaw, scale = CAR_POSE[index]
        car = cuboid(index, 0, position, yaw, scale)
        car["attributes"] = {"occluded": False, **CAR_ATTRS}
        item["annotations"].append(car)
        expected[(index, CAR_ATTRS["record_token"])] = car
        if index < 2:
            position, yaw, scale = SHORT_POSE[index]
            short = cuboid(200 + index, 1, position, yaw, scale)
            short["attributes"] = {"occluded": False, **SHORT_ATTRS}
            item["annotations"].append(short)
            expected[(index, SHORT_ATTRS["record_token"])] = short
        if index == 1:
            position, yaw, scale = PED_POSE
            ped = cuboid(100, 1, position, yaw, scale)
            ped["attributes"] = {"occluded": False, **PED_ATTRS}
            item["annotations"].append(ped)
            expected[(index, PED_ATTRS["record_token"])] = ped
        items.append(item)

    document = datumaro_document(LABELS, items)
    # Datumaro declares a label's attribute names alongside the label; CVAT
    # matches them to the project's declarations by name on import.
    for label in document["categories"]["label"]["labels"]:
        label["attributes"] = [spec["name"] for spec in ATTRIBUTE_SPECS]
    json_path = os.path.join(tmp, "annotations.json")
    with open(json_path, "w") as fh:
        json.dump(document, fh, indent=2)
    return zip_path, json_path, expected


# --- what comes out ---------------------------------------------------------
def unzip_export(zip_path: str, dest: str) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        archive.extractall(dest)
    return names


def close_enough(a, b, tol=1e-4) -> bool:
    return len(a) == len(b) and all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def describe(value) -> str:
    return f"{value!r} ({type(value).__name__})"


def load_env_file(path: str) -> None:
    """KEY=VALUE lines into os.environ, for a worktree with no .env of its own."""
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env-file", default=None)
    known, argv = pre.parse_known_args(argv)
    if known.env_file:
        load_env_file(known.env_file)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-file", default=None,
                        help="read CVAT_HOST/CVAT_USER/CVAT_PASSWORD from this file first")
    parser.add_argument("--host", default=os.environ.get("CVAT_HOST", "http://localhost:8081"))
    parser.add_argument("--user", default=os.environ.get("CVAT_USER", "mt"))
    parser.add_argument("--password", default=os.environ.get("CVAT_PASSWORD", ""))
    parser.add_argument("--project-name", default=PROJECT_NAME)
    parser.add_argument("--out-dir", default=None,
                        help="keep the exported zip and its JSON here (default: a temp dir "
                             "that survives the run)")
    parser.add_argument("--cleanup", action="store_true",
                        help="REFUSED — see the module docstring; nothing on CVAT is deleted "
                             "by this script")
    args = parser.parse_args(argv)

    if not args.password:
        print("set CVAT_PASSWORD (set -a; . ./.env; set +a)", file=sys.stderr)
        return 2
    if args.cleanup:
        print("!!! --cleanup refused: the operator's standing rule is that nothing on the CVAT\n"
              "!!! server is deleted without an explicit per-item ask. The spike project is left\n"
              "!!! in place; its id is printed below so it can be removed by hand.\n")

    out_dir = args.out_dir or tempfile.mkdtemp(prefix="cvat3d-spike-")
    os.makedirs(out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="cvat3d-spike-in-")
    zip_path, json_path, expected = build_input(tmp)
    shutil.copy(json_path, os.path.join(out_dir, "input.json"))
    print(f"input:  {json_path}  ({os.path.getsize(zip_path)} B task.zip)")
    print(f"output: {out_dir}\n")

    with make_client(host=args.host, credentials=(args.user, args.password)) as client:
        about = client.api_client.server_api.retrieve_about()[0]
        print(f"[1] CVAT server: {about.version}  (SDK {client.api_client.configuration.host})")

        # --- label attribute declarations ---------------------------------
        # Try the full five; on refusal, report the error and retry without the
        # attribute CVAT complained about, so the run still produces evidence.
        project = next((p for p in client.projects.list() if p.name == args.project_name), None)
        rejected: list[str] = []
        if project is None:
            attempts = [
                ("all five", ATTRIBUTE_SPECS),
                ("without track_id", [s for s in ATTRIBUTE_SPECS if s["name"] != "track_id"]),
            ]
            for what, specs in attempts:
                try:
                    project = client.projects.create(
                        {"name": args.project_name, "labels": label_spec(specs)})
                    print(f"[5] label attributes accepted: {what} "
                          f"({', '.join(s['name'] for s in specs)})")
                    break
                except Exception as exc:  # noqa: BLE001 — the error text IS the finding
                    rejected.append(f"{what}: {type(exc).__name__}: {exc}")
                    print(f"[5] label attributes REJECTED ({what}): {exc}")
            if project is None:
                print("could not create the spike project at all", file=sys.stderr)
                return 2
        else:
            print(f"[5] reusing project #{project.id} {args.project_name!r}")
        print(f"    project #{project.id}")
        for label in project.get_labels():
            declared = [
                {"name": a.name, "input_type": str(a.input_type), "mutable": a.mutable,
                 "default_value": a.default_value, "values": list(a.values)}
                for a in label.attributes
            ]
            print(f"    label {label.name!r} type={label.type} attributes={json.dumps(declared)}")

        # --- the task ------------------------------------------------------
        name = f"spike-3d-roundtrip-{os.getpid()}"
        task = client.tasks.create_from_data(
            spec={"name": name, "project_id": project.id, "sorting_method": "lexicographical"},
            resource_type=ResourceType.LOCAL,
            resources=[zip_path],
            data_params={"image_quality": 90},
        )
        task.fetch()
        print(f"    task #{task.id} {name!r}  dimension={task.dimension} frames={task.size}")
        if task.dimension != "3d":
            print("task is not 3d — the archive layout was not recognised", file=sys.stderr)
            return 2
        task.import_annotations(format_name=FORMAT, filename=json_path)
        print("    annotations imported\n")

        # --- how the SERVER stored it (tracks vs shapes) -------------------
        stored = client.api_client.tasks_api.retrieve_annotations(task.id)[1].json()
        stored = json.loads(stored) if isinstance(stored, str) else stored
        with open(os.path.join(out_dir, "server_annotations.json"), "w") as fh:
            json.dump(stored, fh, indent=2)
        print(f"[2] server-side: {len(stored.get('tracks', []))} track(s), "
              f"{len(stored.get('shapes', []))} shape(s), {len(stored.get('tags', []))} tag(s)")
        for track in stored.get("tracks", []):
            frames = [s["frame"] for s in track["shapes"]]
            print(f"    track id={track['id']} label_id={track['label_id']} frames={frames} "
                  f"track-level attributes={track.get('attributes')} "
                  f"shape attributes[0]={track['shapes'][0].get('attributes')}")
        for shape in stored.get("shapes", []):
            print(f"    shape id={shape['id']} label_id={shape['label_id']} frame={shape['frame']} "
                  f"attributes={shape.get('attributes')}")
        print()

        # --- the export ----------------------------------------------------
        export_zip = os.path.join(out_dir, "export.zip")
        task.export_dataset(FORMAT, export_zip, include_images=False)

    names = unzip_export(export_zip, os.path.join(out_dir, "export"))
    inner = [n for n in names if n.endswith(".json")]
    print(f"[6] export zip members: {names}")
    print(f"    annotation JSON inner path: {inner}")
    with open(os.path.join(out_dir, "export", inner[0])) as fh:
        exported = json.load(fh)
    print(f"    categories: {json.dumps(exported['categories'], indent=2)[:800]}\n")

    print("[2/3/4] exported items")
    got: dict[tuple[int, str], dict] = {}
    for item in exported["items"]:
        print(f"  item {item['id']!r} frame_attr={item.get('attr')} "
              f"point_cloud={item.get('point_cloud')}")
        for ann in item["annotations"]:
            print(f"    {json.dumps(ann, sort_keys=True)}")
            frame = int(item["id"]) - 1
            got[(frame, ann["attributes"].get("record_token", "<no record_token>"))] = ann
    print()

    # --- the four comparisons -------------------------------------------
    print("[2] track membership")
    for key in sorted(got):
        ann = got[key]
        print(f"    frame={key[0]} {key[1]!r}: label_id={ann['label_id']} id={ann.get('id')} "
              f"track_id={describe(ann['attributes'].get('track_id', '<absent>'))} "
              f"keyframe={describe(ann['attributes'].get('keyframe', '<absent>'))}")
    extra = sorted(set(got) - set(expected))
    missing = sorted(set(expected) - set(got))
    print(f"    cuboids in: {len(expected)}  out: {len(got)}  "
          f"extra (interpolated/extrapolated): {extra}  missing: {missing}")

    print("\n[3] attribute survival (name: in -> out)")
    for key in sorted(got):
        source = expected.get(key)
        if source is None:
            print(f"    frame={key[0]} {key[1]!r}: no input counterpart — "
                  f"{json.dumps(got[key], sort_keys=True)}")
            continue
        for attr in [s["name"] for s in ATTRIBUTE_SPECS]:
            went_in = source["attributes"].get(attr, "<not set>")
            came_out = got[key]["attributes"].get(attr, "<ABSENT>")
            verdict = "ok " if str(went_in) == str(came_out) else "DIFF"
            print(f"    {verdict} frame={key[0]} {key[1]}.{attr}: "
                  f"{describe(went_in)} -> {describe(came_out)}")

    print("\n[4] geometry")
    for key in sorted(got):
        source = expected.get(key)
        if source is None:
            continue
        for field in ("position", "rotation", "scale"):
            a, b = source[field], got[key][field]
            worst = max(abs(float(x) - float(y)) for x, y in zip(a, b)) if len(a) == len(b) else None
            reordered = (not close_enough(a, b)) and sorted(
                round(float(v), 2) for v in a) == sorted(round(float(v), 2) for v in b)
            verdict = ("ok@1e-4  " if close_enough(a, b, 1e-4)
                       else "ok@1e-2  " if close_enough(a, b, 5e-3 + 1e-9)
                       else "REORDERED" if reordered else "DIFF     ")
            print(f"    {verdict} frame={key[0]} {key[1]}.{field}: {a} -> {b}  "
                  f"(worst |delta| = {worst})")

    if rejected:
        print("\n[5] rejected declarations")
        for line in rejected:
            print(f"    {line}")

    print(f"\nkept: {out_dir}")
    print("NOT deleted (operator rule): the spike project and its task remain on the server.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
