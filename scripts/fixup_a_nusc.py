#!/usr/bin/env python3
"""Drop keyframes that lack a required channel — into a NEW version dir.

The 2026-09-04 export (Dataset/A_nusc) has ~5 % of keyframes missing at least
one camera image (CAM_LEFT worst: 73 of chunk_0003's 750). Stage 0's
channels_complete predicate is all-or-nothing per SCENE and each chunk is one
scene, so a single such keyframe refuses 750 of them; had it passed, Stage 1
would KeyError on the missing channel (ingest.py builds CameraObservation over
the fixed ring). The pilot's own answer was a data-side fixup into a sibling
version dir — v1.0-dhaka-fixed, 2026-08-30: "drop the 2 CAM_RIGHT-less
samples". This is that, generalised.

Non-destructive by construction: <dataroot>/<version>/ is read and never
written; <dataroot>/<out-version>/ is created and refused if it already exists.
Blobs are untouched — the dropped samples' files simply go unreferenced.

    DHAKASCENES_SUBSTRATE=dhaka6 python scripts/fixup_a_nusc.py \\
        --dataroot Dataset/A_nusc/chunk_0006 --version v1.0-dhaka
    # -> Dataset/A_nusc/chunk_0006/v1.0-dhaka-fixed/

"Required" defaults to the active substrate profile's REQUIRED_CHANNELS
(LIDAR_TOP + the ring), so the profile that Stage 0 will gate under is the
profile this fixup drops under.

--sanitize-scene-names (off by default) is the same shape of repair for a
scene NAME that is legal in JSON but not as a directory component: every stage
writes scenes/<scene name>/, so full-fused's "dhaka_20260905_174950/chunk_0000"
nests a level deeper than Stage 9's listdir and the release export's
scenes/*/ glob can see.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The 13 nuScenes v1.0 tables. Every one is copied through; three are edited.
TABLES: tuple[str, ...] = (
    "attribute",
    "calibrated_sensor",
    "category",
    "ego_pose",
    "instance",
    "log",
    "map",
    "sample",
    "sample_annotation",
    "sample_data",
    "scene",
    "sensor",
    "visibility",
)
EDITED = ("sample", "sample_data", "scene")


def _channel_of_calibrated_sensor(tables: dict[str, list]) -> dict[str, str]:
    sensor_channel = {s["token"]: s["channel"] for s in tables["sensor"]}
    return {c["token"]: sensor_channel[c["sensor_token"]] for c in tables["calibrated_sensor"]}


def channels_per_sample(tables: dict[str, list], blob_exists=None) -> dict[str, set[str]]:
    """sample token -> the channels that have a KEYFRAME sample_data row for it.

    With `blob_exists(filename) -> bool`, a row whose file is not there does
    not count: chunk_0006 names samples/CAM_FRONT/000100.jpg, which the
    exporter never wrote, and Stage 0's files_resolve fails the scene on it.
    Default None trusts the tables (pure, no I/O).
    """
    cs_channel = _channel_of_calibrated_sensor(tables)
    have: dict[str, set[str]] = defaultdict(set)
    for row in tables["sample_data"]:
        if not row.get("is_key_frame"):
            continue
        if blob_exists is not None and not blob_exists(row["filename"]):
            continue
        have[row["sample_token"]].add(cs_channel[row["calibrated_sensor_token"]])
    return have


def drop_incomplete_samples(
    tables: dict[str, list], required_channels, blob_exists=None
) -> tuple[dict[str, list], list[str]]:
    """Return (new tables, sorted dropped sample tokens). `tables` is not mutated.

    A sample is kept iff every channel in `required_channels` has a keyframe
    sample_data row for it (whose blob exists, when `blob_exists` is given).
    Extra channels (RADAR) count for nothing either way. Kept samples keep
    their table order; prev/next are re-linked per scene, and scene.json's
    nbr_samples / first / last follow. The sample_data prev/next chains are
    spliced across the removed rows too, so no surviving link names a token
    that is no longer in the table.
    """
    if not required_channels:
        raise ValueError("required_channels must name at least one channel")
    required = set(required_channels)

    have = channels_per_sample(tables, blob_exists)
    dropped = sorted(s["token"] for s in tables["sample"] if not required <= have.get(s["token"], set()))
    out = {name: copy.deepcopy(rows) for name, rows in tables.items()}
    if not dropped:
        return out, []
    gone = set(dropped)

    out["sample"] = [s for s in out["sample"] if s["token"] not in gone]

    # The sample_data chain has to close as well as the sample chain
    # (2026-09-08). Removing a sample's rows leaves every surviving NEIGHBOUR of
    # a removed row still naming it, and a prev/next that names a token no
    # longer in the table is exactly what Stage 0's token_graph_closed refuses:
    # on full-fused, 868 rows went and 255 of their tokens stayed referenced, so
    # all 11 scenes were excluded. The splice walks the ORIGINAL links past the
    # removed run, which handles consecutive drops in one step and terminates at
    # "" when the chain runs off its end.
    #
    # Only links that pointed INTO the removed set are touched. A link that was
    # already dangling before this fixup ran stays dangling: repairing it here
    # would launder a defect this script did not cause, and the probe is the
    # thing that should be telling us about it.
    removed_sd = {r["token"] for r in out["sample_data"] if r["sample_token"] in gone}
    original_sd = {r["token"]: r for r in tables["sample_data"]}
    out["sample_data"] = [r for r in out["sample_data"] if r["sample_token"] not in gone]

    def _splice(token: str, link: str) -> str:
        seen: set[str] = set()
        while token in removed_sd and token not in seen:
            seen.add(token)
            token = original_sd[token].get(link) or ""
        return token

    for row in out["sample_data"]:
        for link in ("prev", "next"):
            if (row.get(link) or "") in removed_sd:
                row[link] = _splice(row[link], link)

    # Re-link per scene in kept order; endpoints and counts follow.
    by_scene: dict[str, list[dict]] = defaultdict(list)
    for s in out["sample"]:
        by_scene[s["scene_token"]].append(s)
    for chain in by_scene.values():
        for i, s in enumerate(chain):
            s["prev"] = chain[i - 1]["token"] if i > 0 else ""
            s["next"] = chain[i + 1]["token"] if i + 1 < len(chain) else ""
    for scene in out["scene"]:
        chain = by_scene.get(scene["token"], [])
        scene["nbr_samples"] = len(chain)
        scene["first_sample_token"] = chain[0]["token"] if chain else ""
        scene["last_sample_token"] = chain[-1]["token"] if chain else ""
    return out, dropped


def _slerp(q0, q1, alpha: float):
    """Spherical interpolation of unit quaternions (w, x, y, z), shortest arc.
    alpha outside [0, 1] extrapolates along the same arc — a camera at a scene
    edge sits at most ~35 ms outside the LiDAR trajectory."""
    import numpy as np

    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:  # nearly parallel: lerp, then renormalise
        q = q0 + alpha * (q1 - q0)
        return (q / np.linalg.norm(q)).tolist()
    theta = math.acos(max(-1.0, min(1.0, dot)))
    s = math.sin(theta)
    q = (math.sin((1.0 - alpha) * theta) / s) * q0 + (math.sin(alpha * theta) / s) * q1
    return (q / np.linalg.norm(q)).tolist()


def interpolate_camera_ego_poses(tables: dict[str, list]) -> tuple[dict[str, list], int]:
    """Give every camera keyframe row its OWN ego_pose at its OWN timestamp.

    The exporter hands all sensors of a sample the LiDAR's one ego_pose, so
    |ego(t_cam) - ego(t_lidar)| == 0 everywhere and Stage 5 refuses the scene
    (the chain would silently omit ego motion between capture times, §1.3).
    Per scene, the LiDAR keyframe poses form the trajectory; each camera row
    gets translation interpolated linearly and rotation slerped at its
    timestamp, in a new ego_pose row with a deterministic token. LiDAR rows
    and the original ego_pose rows are untouched; a camera row that already
    carries a pose distinct from its LiDAR's is left alone (idempotent). A
    scene with fewer than two LiDAR poses cannot be interpolated and is left
    shared. Returns (new tables, number of ego_pose rows added).
    """
    import numpy as np

    out = {name: copy.deepcopy(rows) for name, rows in tables.items()}
    cs_channel = _channel_of_calibrated_sensor(out)
    poses = {e["token"]: e for e in out["ego_pose"]}
    scene_of_sample = {s["token"]: s["scene_token"] for s in out["sample"]}

    lidar_by_sample: dict[str, dict] = {}
    cams_by_scene: dict[str, list[dict]] = defaultdict(list)
    for row in out["sample_data"]:
        if not row.get("is_key_frame"):
            continue
        if cs_channel[row["calibrated_sensor_token"]] == "LIDAR_TOP":
            lidar_by_sample[row["sample_token"]] = row
        else:
            cams_by_scene[scene_of_sample[row["sample_token"]]].append(row)

    added = 0
    for scene_token, cam_rows in cams_by_scene.items():
        traj = sorted(
            (r for s, r in lidar_by_sample.items() if scene_of_sample.get(s) == scene_token),
            key=lambda r: r["timestamp"],
        )
        if len(traj) < 2:
            continue
        times = np.array([r["timestamp"] for r in traj], dtype=np.float64)
        trans = np.array([poses[r["ego_pose_token"]]["translation"] for r in traj], dtype=np.float64)
        rots = [poses[r["ego_pose_token"]]["rotation"] for r in traj]
        for row in cam_rows:
            lidar = lidar_by_sample.get(row["sample_token"])
            if lidar is None or row["ego_pose_token"] != lidar["ego_pose_token"]:
                continue  # no LiDAR anchor, or already per-sensor
            t = float(row["timestamp"])
            i = int(np.clip(np.searchsorted(times, t) - 1, 0, len(times) - 2))
            alpha = (t - times[i]) / (times[i + 1] - times[i])
            translation = (trans[i] + alpha * (trans[i + 1] - trans[i])).tolist()
            rotation = _slerp(rots[i], rots[i + 1], float(alpha))
            token = hashlib.md5(f"{row['token']}:ego_pose@camera".encode()).hexdigest()
            out["ego_pose"].append({
                "token": token,
                "timestamp": row["timestamp"],
                "translation": [float(v) for v in translation],
                "rotation": [float(v) for v in rotation],
                "identity": False,
            })
            row["ego_pose_token"] = token
            added += 1
    return out, added


# Rotation taking OPTICAL camera axes (x right, y down, z forward) into the
# vehicle BODY axes (x forward, y left, z up): columns are the optical axes
# expressed in body coordinates. As a (w, x, y, z) quaternion this is the
# classic nuScenes CAM_FRONT rotation.
BODY_TO_OPTICAL_WXYZ = (0.5, -0.5, 0.5, -0.5)


def _qmul(a, b):
    """Hamilton product (w, x, y, z): rotation a followed-by-composition with b, R_a @ R_b."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def already_optical(cal: dict) -> str | None:
    """Why this camera row's `rotation` is ALREADY optical, or None if it is not.

    An exporter can hand us either convention, and converting an optical
    rotation a second time rotates the camera ring by 90 deg — measured on the
    2026-09-05 capture, where it put CAM_FRONT at yaw -86.8 deg and CAM_BACK at
    +90.5, so every 3D box would project into the wrong camera
    (docs/evidence/2026-09-08-camera-lr-mapping.md §5). The detection is
    therefore by SIGNAL, not by one exact string:

    * `frame_convention` is a string containing the word "optical", in any
      case. That covers the bare "optical" this script stamps AND a
      descriptive sentence — full-fused/v1.0-dhaka writes "camera axes are
      optical (x-right, y-down, z-forward); rotation maps camera-optical ->
      LIDAR_TOP/ego", which an `== "optical"` test missed.
    * a four-element `rotation_body` that differs from `rotation`. An exporter
      that kept the pre-conversion rotation beside the converted one has
      already done this conversion; on that capture `rotation` equals
      `rotation_body` ⊗ q_body->optical to 1e-16 for all six cameras. An
      absent, empty or identical `rotation_body` says nothing and is ignored.

    Those are the only two signals full-fused/v1.0-dhaka carries; its other
    per-row field, `camera_distortion_status`, is about intrinsics, not frames.
    Returns a short human-readable reason (for the log), or None.
    """
    convention = cal.get("frame_convention")
    if isinstance(convention, str) and "optical" in convention.lower():
        return f"frame_convention says {convention!r}"
    body = cal.get("rotation_body")
    rotation = cal.get("rotation")
    if isinstance(body, (list, tuple)) and len(body) == 4 and isinstance(rotation, (list, tuple)):
        if len(rotation) == 4 and any(
            abs(float(a) - float(b)) > 1e-12 for a, b in zip(body, rotation)
        ):
            return "rotation_body holds the pre-conversion rotation beside rotation"
    return None


def cameras_to_optical_convention(tables: dict[str, list]) -> tuple[dict[str, list], int, int]:
    """Re-express every camera's `calibrated_sensor.rotation` in the optical convention.

    The day-1 exporter writes camera rotations in the vehicle body convention
    (CAM_FRONT ~ identity); every projection in the pipeline assumes nuScenes'
    optical convention. `rotation` is sensor->ego, so the fix is
    R_ego<-optical = R_ego<-body @ R_body<-optical, i.e. q_body * q_b2o.
    Translations are untouched (same origin). A sensor without a 3x3
    `camera_intrinsic` is not a camera and is left alone.

    A row `already_optical()` recognises is left EXACTLY as the exporter wrote
    it — rotation and its own `frame_convention` / `frame_convention_note`
    included — and counted separately: applying this twice rotates the ring by
    90 deg, and a silent skip is how that hid. Rows this function does convert
    are stamped `frame_convention: "optical"`, so a second pass is a no-op.
    Returns (new tables, rows converted, rows skipped as already optical).
    """
    out = {name: copy.deepcopy(rows) for name, rows in tables.items()}
    converted = 0
    skipped = 0
    for cal in out["calibrated_sensor"]:
        intrinsic = cal.get("camera_intrinsic")
        if not (isinstance(intrinsic, list) and len(intrinsic) == 3):
            continue
        if already_optical(cal) is not None:
            skipped += 1
            continue
        q = _qmul(tuple(float(v) for v in cal["rotation"]), BODY_TO_OPTICAL_WXYZ)
        norm = math.sqrt(sum(v * v for v in q))
        cal["rotation"] = [v / norm for v in q]
        cal["frame_convention"] = "optical"
        cal["frame_convention_note"] = "body -> optical applied by scripts/fixup_a_nusc.py (exporter wrote body-frame camera rotations)"
        converted += 1
    return out, converted, skipped


def swap_camera_channels(tables: dict[str, list], channel_a: str, channel_b: str) -> tuple[dict[str, list], int]:
    """Exchange the calibration of two camera channels, sample by sample.

    The exporter filed the right-facing stream as CAM_LEFT and the left-facing
    one as CAM_RIGHT (measured 2026-09-06 by rearward image drift on chunk_0000:
    CAM_LEFT +68.6 px, CAM_RIGHT -80.6 px — the opposite of what their
    calibrations say; the "CAM_RIGHT" stream also carries 6.5x the kerb-side
    instances of "CAM_LEFT" in left-hand traffic). Within each sample EVERY row
    of the two channels exchanges `calibrated_sensor_token`, so each image row
    is paired with the calibration — and, through the sensor table, the channel
    name — that matches its content. filename, timestamp and ego_pose stay with
    the row: they belong to the file. A sample holding only one of the pair is
    left alone. Its own inverse. Returns (new tables, rows changed).

    Every row, not one per channel: a sample owns its keyframe AND the sweeps
    filed against it (~6 per side camera on full-fused/v1.0-dhaka). Keeping a
    single row per (sample, channel) kept the LAST one, which is a sweep, and
    left the keyframe — the only row any stage reads — pointing at the
    calibration this swap exists to correct. Measured 2026-09-08 on the
    regenerated v1.0-dhaka-fixed: 1 keyframe pair of 14 966 swapped, 14 965
    sweep pairs swapped in their place.
    """
    cs_channel = _channel_of_calibrated_sensor(tables)
    known = set(cs_channel.values())
    for ch in (channel_a, channel_b):
        if ch not in known:
            raise ValueError(f"channel {ch!r} has no calibrated_sensor row; known: {sorted(known)}")
    out = {name: copy.deepcopy(rows) for name, rows in tables.items()}
    by_sample: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in out["sample_data"]:
        ch = cs_channel[row["calibrated_sensor_token"]]
        if ch in (channel_a, channel_b):
            by_sample[row["sample_token"]][ch].append(row)
    changed = 0
    for pair in by_sample.values():
        if channel_a in pair and channel_b in pair:
            token_a = pair[channel_a][0]["calibrated_sensor_token"]
            token_b = pair[channel_b][0]["calibrated_sensor_token"]
            for row in pair[channel_a]:
                row["calibrated_sensor_token"] = token_b
            for row in pair[channel_b]:
                row["calibrated_sensor_token"] = token_a
            changed += len(pair[channel_a]) + len(pair[channel_b])
    return out, changed


# A scene name becomes a DIRECTORY component: every stage writes its per-scene
# output to os.path.join(out_dir, "scenes", scene_name).
UNSAFE_IN_A_PATH_COMPONENT = re.compile(r"[^A-Za-z0-9._-]")


def sanitize_scene_names(tables: dict[str, list]) -> tuple[dict[str, list], dict[str, str]]:
    """Make every scene `name` usable as one directory component.

    The 2026-09-05 capture (full-fused, v1.0-dhaka) names its 11 scenes
    "dhaka_20260905_174950/chunk_0000".. — with a slash. Every stage builds its
    per-scene output as os.path.join(out_dir, "scenes", scene_name), so the
    slash silently NESTS one level deeper than the layout everything downstream
    assumes; Stage 9 enumerates scenes with a single-level os.listdir and the
    release export globs scenes/*/prelabels.jsonl, so both find NOTHING and the
    run "succeeds" with an empty release. Each character outside
    [A-Za-z0-9._-] becomes "_". Only the human-readable `name` moves — tokens,
    which are what the rest of the substrate points at, are untouched, and so
    is every other table (only scene.json carries the name as a string).

    Two scenes that would end up sharing a name are a hard error naming the
    offenders: merging them would quietly fold one scene's frames into
    another's directory. Names the sanitizer does not touch are not its
    business, so a duplicate the source already had passes through. Returns
    (new tables, {old name: new name} for the names that changed); `tables` is
    not mutated, and a second pass changes nothing.
    """
    out = {name: copy.deepcopy(rows) for name, rows in tables.items()}
    renamed: dict[str, str] = {}
    claimed: dict[str, list[str]] = defaultdict(list)
    for scene in out["scene"]:
        old = scene["name"]
        new = UNSAFE_IN_A_PATH_COMPONENT.sub("_", old)
        claimed[new].append(old)
        if new != old:
            renamed[old] = new
            scene["name"] = new
    collisions = {
        new: sorted(olds) for new, olds in claimed.items()
        if len(olds) > 1 and any(o in renamed for o in olds)
    }
    if collisions:
        detail = "; ".join(f"{new!r} <- {olds}" for new, olds in sorted(collisions.items()))
        raise ValueError(
            f"sanitizing scene names would merge distinct scenes into one scenes/ directory: {detail}"
        )
    return out, renamed


def _read_tables(version_dir: str) -> dict[str, list]:
    tables = {}
    for name in TABLES:
        path = os.path.join(version_dir, f"{name}.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, encoding="utf-8") as fh:
            tables[name] = json.load(fh)
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataroot", required=True, help="chunk root holding <version>/ and samples/")
    parser.add_argument("--version", default="v1.0-dhaka", help="source version dir name (read-only)")
    parser.add_argument("--out-version", default=None, help="new version dir name; default <version>-fixed")
    parser.add_argument("--required-channels", nargs="+", default=None,
                        help="channels a keyframe must carry; default: the active substrate "
                             "profile's REQUIRED_CHANNELS (DHAKASCENES_SUBSTRATE)")
    parser.add_argument("--swap-channels", nargs=2, action="append", default=[], metavar=("A", "B"),
                        help="exchange the calibration (hence the channel) of these two camera streams "
                             "in every sample — for an exporter that filed them under each other's "
                             "name. Repeatable. Recorded in <out-version>/fixup_meta.json")
    parser.add_argument("--sanitize-scene-names", action="store_true",
                        help="replace every character outside [A-Za-z0-9._-] in each scene's `name` "
                             "with '_' — a scene name is a directory component, and a slash in one "
                             "(full-fused's dhaka_20260905_174950/chunk_0000) nests every stage's "
                             "scenes/<name>/ output where Stage 9 and the release export cannot see "
                             "it. Off by default. Recorded in <out-version>/fixup_meta.json")
    args = parser.parse_args(argv)

    out_version = args.out_version or f"{args.version}-fixed"
    src_dir = os.path.join(args.dataroot, args.version)
    out_dir = os.path.join(args.dataroot, out_version)
    if os.path.exists(out_dir):
        print(f"refusing: {out_dir} already exists (delete it to redo the fixup)", file=sys.stderr)
        return 2
    if args.required_channels:
        required = tuple(args.required_channels)
    else:
        from pipeline.common.schemas import REQUIRED_CHANNELS, SUBSTRATE  # noqa: E402 — env-resolved
        required = REQUIRED_CHANNELS
        print(f"required channels from profile {SUBSTRATE!r}: {list(required)}")

    try:
        tables = _read_tables(src_dir)
    except FileNotFoundError as exc:
        print(f"refusing: missing table {exc}", file=sys.stderr)
        return 2

    def blob_exists(filename: str) -> bool:
        return os.path.isfile(os.path.join(args.dataroot, filename))

    have = channels_per_sample(tables, blob_exists)
    fixed, dropped = drop_incomplete_samples(tables, required, blob_exists)
    n_before = len(tables["sample"])
    if n_before and not fixed["sample"]:
        print(f"refusing: every one of the {n_before} samples would be dropped; nothing written", file=sys.stderr)
        return 2

    missing_by_channel = Counter(
        ch for tok in dropped for ch in required if ch not in have.get(tok, set())
    )
    n_swapped = 0
    for a, b in args.swap_channels:
        fixed, n = swap_camera_channels(fixed, a, b)
        n_swapped += n
    fixed, n_cam_poses = interpolate_camera_ego_poses(fixed)
    fixed, n_optical, n_already_optical = cameras_to_optical_convention(fixed)
    # Per camera, in words: converted here, or left alone and why. A silent
    # skip is how the double conversion hid (evidence 2026-09-08 §5).
    _channel = _channel_of_calibrated_sensor(fixed)
    camera_decisions = {
        _channel.get(cal["token"], cal["token"]):
            f"already optical, skipped — {reason}" if (reason := already_optical(cal))
            else "converted body -> optical by this script"
        for cal in fixed["calibrated_sensor"]
        if isinstance(cal.get("camera_intrinsic"), list) and len(cal["camera_intrinsic"]) == 3
    }
    renamed_scenes: dict[str, str] = {}
    if args.sanitize_scene_names:
        try:
            fixed, renamed_scenes = sanitize_scene_names(fixed)
        except ValueError as exc:
            print(f"refusing: {exc}", file=sys.stderr)
            return 2
    os.makedirs(out_dir)
    for name in TABLES:
        with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
            json.dump(fixed[name], fh)
    # What this fixup did, beside the 13 tables (not one of them; the
    # fingerprint hashes the tables only).
    with open(os.path.join(out_dir, "fixup_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "source_version": args.version,
            "required_channels": list(required),
            "dropped_samples": dropped,
            "swapped_channels": [list(p) for p in args.swap_channels],
            "n_sample_data_rows_swapped": n_swapped,
            "n_camera_ego_poses_interpolated": n_cam_poses,
            "n_camera_rotations_to_optical": n_optical,
            "n_camera_rotations_already_optical": n_already_optical,
            "camera_frame_convention_decisions": camera_decisions,
            "scene_names_sanitized": renamed_scenes,
            "n_scene_names_sanitized": len(renamed_scenes),
            "tool": "scripts/fixup_a_nusc.py",
        }, fh, indent=2)
    print(f"{src_dir} -> {out_dir}")
    print(f"  samples: {n_before} -> {len(fixed['sample'])}  (dropped {len(dropped)})")
    print(f"  sample_data rows: {len(tables['sample_data'])} -> {len(fixed['sample_data'])}")
    print(f"  ego_pose rows: {len(tables['ego_pose'])} -> {len(fixed['ego_pose'])}  "
          f"(+{n_cam_poses} per-camera poses interpolated from the LiDAR trajectory)")
    print(f"  camera extrinsics: {n_optical} converted, {n_already_optical} already optical (skipped)")
    if args.sanitize_scene_names:
        print(f"  scene names: {len(renamed_scenes)} of {len(fixed['scene'])} sanitized to one "
              f"path component{' — ' + str(renamed_scenes) if renamed_scenes else ''}")
    if args.swap_channels:
        print(f"  channels swapped: {args.swap_channels} ({n_swapped} sample_data rows re-paired)")
    if dropped:
        print(f"  missing by channel among dropped: {dict(sorted(missing_by_channel.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
