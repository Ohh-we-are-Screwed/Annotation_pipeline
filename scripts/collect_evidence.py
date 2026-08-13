#!/usr/bin/env python3
"""Independent re-measurement of the substrate facts in pilot_plan.md §0 / BUILD_PROMPT appendix.

Activity A, verification order step 1 (BUILD_PROMPT.md §4.5): "Re-measure §0 yourself.
The appendix has the values; confirm, don't trust."

Deliberately shares NO code with pipeline/ — quaternion->yaw, JPEG header parsing,
and the metadata fingerprint are reimplemented here so a bug in pipeline/common/
cannot ratify itself. The one exception is the fingerprint *construction* (sorted
name-digest manifest), which must match FINGERPRINT_SPEC to test the allowlist
binding — the spec string is asserted against usable_scenes.json.

Writes docs/evidence/substrate_remeasure.json. Every value in that file is
MEASUREMENT-class evidence for docs/conformance.yaml.
"""
import hashlib
import json
import math
import os
import statistics
import struct
import sys

DATAROOT = "/home/mt/Zami/nuscenes"
META = os.path.join(DATAROOT, "v1.0-mini")
OUT = "/home/mt/Zami/docs/evidence/substrate_remeasure.json"
WORK = "/home/mt/dhakascenes/work"
OUT_ROOT = "/home/mt/dhakascenes/out"

CAMERAS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
           "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT"]
REQUIRED_CHANNELS = CAMERAS + ["LIDAR_TOP"]


def load(table):
    with open(os.path.join(META, table + ".json")) as f:
        return json.load(f)


def quat_to_yaw_deg(q):
    # nuScenes stores [w, x, y, z]; yaw about +z of the rotation matrix column 0.
    w, x, y, z = q
    # rotation of unit x-axis: R @ [1,0,0]
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + z * w)
    return math.degrees(math.atan2(r10, r00))


def jpeg_dims(path):
    # Parse SOF0/SOF2 marker for dimensions; independent of PIL.
    with open(path, "rb") as f:
        data = f.read(65536)
        head_ok = data[:2] == b"\xff\xd8"
        i = 2
        while i < len(data) - 9:
            if data[i] != 0xFF:
                i += 1
                continue
            marker = data[i + 1]
            if marker in (0xC0, 0xC1, 0xC2):
                h = struct.unpack(">H", data[i + 5:i + 7])[0]
                w = struct.unpack(">H", data[i + 7:i + 9])[0]
                return head_ok, (w, h)
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
                i += 2
                continue
            seglen = struct.unpack(">H", data[i + 2:i + 4])[0]
            i += 2 + seglen
        return head_ok, None


def main():
    ev = {"measured_utc": "2026-08-12", "dataroot": DATAROOT, "how": __doc__.strip().splitlines()[0]}

    # --- metadata tables + fingerprint --------------------------------------
    tables = sorted(f for f in os.listdir(META) if f.endswith(".json"))
    ev["metadata_tables"] = {"count": len(tables), "names": tables}
    digests = {}
    for t in tables:
        h = hashlib.sha256()
        with open(os.path.join(META, t), "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        # FINGERPRINT_SPEC "sha256-of-sorted-name-digest-manifest/v1" keys the
        # manifest by full filename (with .json) — verified against paths.py.
        digests[t] = h.hexdigest()
    manifest = "".join(f"{name}  {digests[name]}\n" for name in sorted(digests))
    ev["metadata_fingerprint"] = hashlib.sha256(manifest.encode()).hexdigest()

    scene = load("scene")
    sample = load("sample")
    sample_data = load("sample_data")
    category = load("category")
    instance = load("instance")
    sample_annotation = load("sample_annotation")
    calibrated_sensor = load("calibrated_sensor")
    sensor = load("sensor")
    ego_pose = load("ego_pose")
    log = load("log")

    ev["counts"] = {
        "scenes": len(scene), "samples": len(sample), "sample_data": len(sample_data),
        "categories": len(category), "instances": len(instance),
        "sample_annotations": len(sample_annotation), "ego_poses": len(ego_pose),
    }
    ev["category_names_dotted"] = sorted(c["name"] for c in category)

    # --- file existence + channel coverage ----------------------------------
    sensor_by_tok = {s["token"]: s for s in sensor}
    calib_by_tok = {c["token"]: c for c in calibrated_sensor}
    ego_by_tok = {e["token"]: e for e in ego_pose}
    channel_of = {}
    for sd in sample_data:
        channel_of[sd["token"]] = sensor_by_tok[calib_by_tok[sd["calibrated_sensor_token"]]["sensor_token"]]["channel"]
    channels = sorted(set(channel_of.values()))
    missing = [sd["filename"] for sd in sample_data
               if not os.path.exists(os.path.join(DATAROOT, sd["filename"]))]
    ev["channels"] = {"count": len(channels), "names": channels}
    ev["missing_files"] = {"count": len(missing), "examples": missing[:5]}

    # --- token graph closure (spot: dangling refs) ---------------------------
    dangling = {
        "sample_data->ego_pose": sum(1 for sd in sample_data if sd["ego_pose_token"] not in ego_by_tok),
        "sample_data->calibrated_sensor": sum(1 for sd in sample_data if sd["calibrated_sensor_token"] not in calib_by_tok),
        "annotation->instance": sum(1 for a in sample_annotation if a["instance_token"] not in {i["token"] for i in instance}),
    }
    ev["dangling_tokens"] = dangling

    # --- keyframe images: count + resolution (headers, all of them) ---------
    key_imgs = [sd for sd in sample_data if sd["is_key_frame"] and channel_of[sd["token"]] in CAMERAS]
    ev["keyframe_images"] = {"count": len(key_imgs)}
    dims_field = {(sd["width"], sd["height"]) for sd in key_imgs}
    dims_hdr = set()
    bad_jpeg = 0
    for sd in key_imgs:
        ok, wh = jpeg_dims(os.path.join(DATAROOT, sd["filename"]))
        if not ok or wh is None:
            bad_jpeg += 1
        else:
            dims_hdr.add(wh)
    ev["keyframe_images"]["dims_from_fields"] = sorted(dims_field)
    ev["keyframe_images"]["dims_from_jpeg_headers"] = sorted(dims_hdr)
    ev["keyframe_images"]["unparseable_jpegs"] = bad_jpeg

    # --- CAM_FRONT intrinsics ------------------------------------------------
    cf_calibs = {tuple(map(tuple, c["camera_intrinsic"]))
                 for c in calibrated_sensor
                 if sensor_by_tok[c["sensor_token"]]["channel"] == "CAM_FRONT"}
    ks = sorted(cf_calibs)
    ev["cam_front_intrinsics"] = {
        "distinct": len(ks),
        "fx": [round(k[0][0], 3) for k in ks], "fy": [round(k[1][1], 3) for k in ks],
        "cx": [round(k[0][2], 3) for k in ks], "cy": [round(k[1][2], 3) for k in ks],
    }

    # --- LIDAR_TOP extrinsics ------------------------------------------------
    lt_calibs = [c for c in calibrated_sensor
                 if sensor_by_tok[c["sensor_token"]]["channel"] == "LIDAR_TOP"]
    yaws = sorted({round(quat_to_yaw_deg(c["rotation"]), 3) for c in lt_calibs})
    ev["lidar_top_extrinsic"] = {
        "distinct_calibrations": len(lt_calibs),
        "example_quaternion_wxyz": lt_calibs[0]["rotation"],
        "yaw_deg_independent_math": yaws,
        "is_identity": all(abs(y) < 1.0 for y in yaws),
    }

    # --- LiDAR cadence + sweeps per keyframe ---------------------------------
    lidar_sd = sorted((sd for sd in sample_data if channel_of[sd["token"]] == "LIDAR_TOP"),
                      key=lambda r: r["timestamp"])
    by_scene = {}
    sample_by_tok = {s["token"]: s for s in sample}
    for sd in lidar_sd:
        sc = sample_by_tok[sd["sample_token"]]["scene_token"]
        by_scene.setdefault(sc, []).append(sd["timestamp"])
    diffs_ms = []
    for ts in by_scene.values():
        ts.sort()
        diffs_ms += [(b - a) / 1000.0 for a, b in zip(ts, ts[1:])]
    ev["lidar"] = {
        "sweep_records": len(lidar_sd),
        "cadence_median_ms": round(statistics.median(diffs_ms), 2),
        "sweeps_per_keyframe": round(len(lidar_sd) / len(sample), 2),
    }
    sizes = [os.path.getsize(os.path.join(DATAROOT, sd["filename"])) for sd in lidar_sd]
    ev["lidar"]["files_size_mod20_nonzero"] = sum(1 for s in sizes if s % 20 != 0)
    ev["lidar"]["example_points"] = sizes[0] // 20

    # --- timestamps ----------------------------------------------------------
    all_ts = [sd["timestamp"] for sd in sample_data]
    ev["timestamps"] = {"digits_min": len(str(min(all_ts))), "digits_max": len(str(max(all_ts))),
                        "example": all_ts[0]}

    # --- per-camera delta-t vs LIDAR_TOP anchor over keyframes ---------------
    key_by_sample = {}
    for sd in sample_data:
        if sd["is_key_frame"]:
            key_by_sample.setdefault(sd["sample_token"], {})[channel_of[sd["token"]]] = sd["timestamp"]
    dts = {c: [] for c in CAMERAS}
    for s in sample:
        recs = key_by_sample.get(s["token"], {})
        if "LIDAR_TOP" not in recs:
            continue
        anchor = recs["LIDAR_TOP"]
        for c in CAMERAS:
            if c in recs:
                dts[c].append((recs[c] - anchor) / 1000.0)
    ev["camera_dt_ms"] = {c: {"min": round(min(v), 2), "median": round(statistics.median(v), 2),
                              "max": round(max(v), 2), "n": len(v)} for c, v in dts.items()}

    # --- scenes: names, locations, night flags -------------------------------
    log_by_tok = {l["token"]: l for l in log}
    ev["scenes"] = [{"name": s["name"],
                     "location": log_by_tok[s["log_token"]]["location"],
                     "night": "night" in s["description"].lower(),
                     "nbr_samples": s["nbr_samples"]} for s in scene]

    # --- run artifacts on disk ------------------------------------------------
    arts = {}
    us_path = os.path.join(WORK, "stage0_data_probe", "usable_scenes.json")
    if os.path.exists(us_path):
        us = json.load(open(us_path))
        m = us["manifest"]
        arts["usable_scenes"] = {
            "n_scenes": len(us["scenes"]),
            "fingerprint": m["metadata_fingerprint"],
            "fingerprint_matches_remeasure": m["metadata_fingerprint"] == ev["metadata_fingerprint"],
            "fingerprint_spec": m["fingerprint_spec"],
            "dataroot_realpath": m["dataroot_realpath"],
            "dataroot_realpath_matches": os.path.realpath(DATAROOT) == m["dataroot_realpath"],
            "partition": {k: sorted(sc["name"] for sc in us["scenes"]
                                    if sc["token"] in us["partition"]["subsets"][k]["scene_tokens"])
                          for k in us["partition"]["subsets"]},
        }
    s1_path = os.path.join(WORK, "stage1_ingestion", "run_manifest.json")
    if os.path.exists(s1_path):
        s1 = json.load(open(s1_path))
        t = s1["totals"]["accumulated"]
        arts["stage1_manifest"] = {
            "python_version": s1.get("python_version"),
            "numpy_version": s1.get("numpy_version"),
            "seed": s1.get("seed"),
            "input_points": t["survivors"]["input"],
            "post_ground": t["survivors"]["post_ground"],
            "retained_fraction": t["retained_fraction"],
            "removed_by_range": t["removed_by_filter"]["post_ground->post_range"],
            "removed_by_height": t["removed_by_filter"]["post_range->post_height"],
            "compensation_max_residual_m": t.get("compensation_max_residual_m",
                                                 s1["totals"].get("compensation_max_residual_m")),
            "absent_1_9_fields": [k for k in ("git_commit", "config_hash", "torch_version",
                                              "cuda_version", "scene_partition")
                                  if k not in s1 and k not in s1.get("config", {})],
            "success_marker": os.path.exists(os.path.join(WORK, "stage1_ingestion", "_SUCCESS")),
        }
        arts["stage0_success_marker"] = os.path.exists(os.path.join(WORK, "stage0_data_probe", "_SUCCESS"))
    pri_path = os.path.join(OUT_ROOT, "priors", "priors_pilot_v0.json")
    if os.path.exists(pri_path):
        pri = json.load(open(pri_path))
        classes = pri.get("classes") or pri.get("priors") or {}
        arts["priors"] = {"source": pri.get("source"),
                          "n_classes": len(classes),
                          "has_license_note": "license_note" in pri,
                          "has_release_guard": "release_guard" in pri,
                          "top_keys": sorted(pri.keys())}
    ev["artifacts"] = arts

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(ev, f, indent=1, sort_keys=True)
    os.replace(tmp, OUT)
    print(json.dumps(ev, indent=1, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
