#!/usr/bin/env python3
"""Phase 1 substrate verification for the DhakaScenes Pilot.

Re-derives every claim in the §0 substrate table from the bytes on this
machine's disk and fails loudly on any divergence. This is the Phase 1 exit
gate: no pipeline code is written until this script is green.

Hard constraints:
  * NO GPU, NO ML models, NO nuscenes-devkit. The devkit is the thing under
    test here — verifying the substrate with the library that assumes the
    substrate is circular. Metadata is parsed as plain JSON, JPEG dimensions
    come from the SOF marker, point clouds from struct arithmetic.
  * Dataroot is opened read-only. Nothing here writes into the substrate.
  * stdlib + numpy only.

Frame convention (ISO 8855, §1.1): x forward, y left, z up, yaw about +z from
+x, radians. Time (§1.2): nuScenes ships microseconds Unix epoch; the pilot
stores int64 unix_ns internally. This script reports the raw µs it reads and
the ns it would store, tagged with time_base, and converts nowhere else.

Usage:
    python3 scripts/probe_substrate.py [--config configs/paths.yaml] [--json OUT]

Exit codes:
    0  every contract check passed
    1  at least one contract check failed
    2  path/config contract violated (substrate not addressable at all)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.paths import (  # noqa: E402
    METADATA_TABLES,
    FINGERPRINT_SPEC,
    Paths,
    PathValidationError,
    load_paths,
    metadata_fingerprint,
)

# ---------------------------------------------------------------------------
# The §0 contract. Every value below is a claim under test, not a computation.
# Changing one of these is a substrate change and must be a deliberate edit.
# ---------------------------------------------------------------------------

EXPECT_SCENES = 10
EXPECT_KEYFRAMES = 404
EXPECT_SAMPLE_DATA = 31_206
EXPECT_CATEGORIES = 23
EXPECT_INSTANCES = 911
EXPECT_ANNOTATIONS = 18_538

EXPECT_IMAGE_W = 1600
EXPECT_IMAGE_H = 900
EXPECT_N_CAMERAS = 6
EXPECT_CAMERAS = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)

# CAM_FRONT intrinsics, §0. Tolerance is 1e-3 because the plan quotes 3dp.
EXPECT_CAM_FRONT_FX = 1266.417
EXPECT_CAM_FRONT_FY = 1266.417
EXPECT_CAM_FRONT_CX = 816.267
EXPECT_CAM_FRONT_CY = 491.507
INTRINSIC_TOL = 1e-3

# LiDAR cadence, §0: median 49.79 ms ≈ 20 Hz. The band is deliberately wide
# enough to admit real jitter and narrow enough to catch a 10 Hz or 2 Hz mixup.
EXPECT_LIDAR_DT_MS = 49.79
LIDAR_DT_BAND_MS = (45.0, 55.0)
EXPECT_LIDAR_HZ_BAND = (18.0, 22.0)
EXPECT_KEYFRAME_DT_MS_BAND = (450.0, 550.0)  # keyframes are ~2 Hz

POINT_RECORD_BYTES = 20  # 5 × float32: x, y, z, intensity, ring
POINT_FIELDS = ("x", "y", "z", "intensity", "ring")

EXPECT_TIMESTAMP_DIGITS = 16  # microseconds since Unix epoch

LIDAR_CHANNEL = "LIDAR_TOP"


# ---------------------------------------------------------------------------
# Check accounting
# ---------------------------------------------------------------------------


@dataclass
class Check:
    name: str
    ok: bool
    expected: Any
    measured: Any
    detail: str = ""

    def line(self) -> str:
        mark = "PASS" if self.ok else "FAIL"
        s = f"  [{mark}] {self.name:<44} expected={self.expected!s:<28} measured={self.measured!s}"
        if self.detail:
            s += f"\n         {self.detail}"
        return s


@dataclass
class Probe:
    checks: list[Check] = field(default_factory=list)
    recon: dict[str, Any] = field(default_factory=dict)

    def check(self, name, ok, expected, measured, detail="") -> bool:
        self.checks.append(Check(name, bool(ok), expected, measured, detail))
        return bool(ok)

    def eq(self, name, expected, measured, detail="") -> bool:
        return self.check(name, expected == measured, expected, measured, detail)

    def within(self, name, lo, hi, measured, detail="") -> bool:
        ok = measured is not None and lo <= measured <= hi
        return self.check(name, ok, f"[{lo}, {hi}]", measured, detail)

    def close(self, name, expected, measured, tol, detail="") -> bool:
        ok = measured is not None and abs(measured - expected) <= tol
        return self.check(name, ok, f"{expected} ±{tol}", measured, detail)

    @property
    def failed(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]


# ---------------------------------------------------------------------------
# Substrate readers — no devkit, no image library
# ---------------------------------------------------------------------------


def load_table(paths: Paths, name: str) -> list[dict]:
    with open(paths.table(name), "rb") as fh:
        return json.load(fh)


_SOF_MARKERS = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def jpeg_dimensions(path: str) -> tuple[int, int]:
    """(width, height) from the JPEG SOF marker. No PIL: read the header itself."""
    with open(path, "rb") as fh:
        if fh.read(2) != b"\xff\xd8":
            raise ValueError(f"not a JPEG (bad SOI): {path}")
        while True:
            b = fh.read(1)
            if not b:
                raise ValueError(f"no SOF marker before EOF: {path}")
            if b != b"\xff":
                continue
            marker = fh.read(1)
            while marker == b"\xff":  # fill bytes
                marker = fh.read(1)
            if not marker:
                raise ValueError(f"truncated marker: {path}")
            m = marker[0]
            if m in (0xD8, 0x01) or 0xD0 <= m <= 0xD7:
                continue  # standalone, no payload
            seg = fh.read(2)
            if len(seg) != 2:
                raise ValueError(f"truncated segment length: {path}")
            (length,) = struct.unpack(">H", seg)
            if m in _SOF_MARKERS:
                payload = fh.read(5)
                _precision, height, width = struct.unpack(">BHH", payload)
                return int(width), int(height)
            fh.seek(length - 2, os.SEEK_CUR)


def quaternion_yaw_rad(q: list[float]) -> float:
    """Yaw about +z from +x, ISO 8855, from a nuScenes [w, x, y, z] quaternion."""
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def digit_count(n: int) -> int:
    return len(str(abs(int(n))))


def describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    vs = sorted(values)
    return {
        "n": len(vs),
        "min": round(vs[0], 4),
        "p05": round(vs[max(0, int(0.05 * (len(vs) - 1)))], 4),
        "median": round(statistics.median(vs), 4),
        "mean": round(statistics.fmean(vs), 4),
        "p95": round(vs[int(0.95 * (len(vs) - 1))], 4),
        "max": round(vs[-1], 4),
        "stdev": round(statistics.pstdev(vs), 4) if len(vs) > 1 else 0.0,
    }


# ---------------------------------------------------------------------------
# Probe sections
# ---------------------------------------------------------------------------


def probe_metadata(p: Probe, paths: Paths, tables: dict[str, list[dict]]) -> None:
    print("\n== A. metadata tables ==")
    p.eq("metadata table count", len(METADATA_TABLES), len(METADATA_TABLES),
         detail=f"version_dir={paths.version_dir}")

    on_disk = sorted(f for f in os.listdir(paths.version_dir) if f.endswith(".json"))
    p.eq("metadata .json files on disk", sorted(METADATA_TABLES), on_disk)

    p.eq("scenes (scene.json)", EXPECT_SCENES, len(tables["scene.json"]))
    p.eq("keyframes (sample.json)", EXPECT_KEYFRAMES, len(tables["sample.json"]))
    p.eq("sample_data records", EXPECT_SAMPLE_DATA, len(tables["sample_data.json"]))
    p.eq("categories", EXPECT_CATEGORIES, len(tables["category.json"]))
    p.eq("instances", EXPECT_INSTANCES, len(tables["instance.json"]))
    p.eq("annotations", EXPECT_ANNOTATIONS, len(tables["sample_annotation.json"]))

    # nbr_samples must agree with the samples actually present per scene.
    by_scene = Counter(s["scene_token"] for s in tables["sample.json"])
    mismatched = [
        {"name": sc["name"], "nbr_samples": sc["nbr_samples"], "present": by_scene.get(sc["token"], 0)}
        for sc in tables["scene.json"]
        if sc["nbr_samples"] != by_scene.get(sc["token"], 0)
    ]
    p.eq("scene.nbr_samples vs samples present", 0, len(mismatched),
         detail=json.dumps(mismatched) if mismatched else "")

    fp, per_file = metadata_fingerprint(paths, with_digests=True)
    p.recon["metadata_fingerprint"] = fp
    p.recon["fingerprint_spec"] = FINGERPRINT_SPEC
    p.recon["metadata_file_sha256"] = per_file
    print(f"  metadata fingerprint ({FINGERPRINT_SPEC}): {fp}")


def build_channel_index(tables: dict[str, list[dict]]) -> tuple[dict[str, str], dict[str, dict]]:
    """calibrated_sensor_token -> channel, and sensor_token -> sensor record."""
    sensors = {s["token"]: s for s in tables["sensor.json"]}
    cs_channel = {
        cs["token"]: sensors[cs["sensor_token"]]["channel"] for cs in tables["calibrated_sensor.json"]
    }
    return cs_channel, sensors


def probe_cameras(p: Probe, paths: Paths, tables: dict[str, list[dict]], cs_channel: dict[str, str]) -> None:
    print("\n== B. cameras and image resolution ==")
    channels = {s["channel"] for s in tables["sensor.json"]}
    cams = sorted(c for c in channels if c.startswith("CAM_"))
    p.eq("camera channel count", EXPECT_N_CAMERAS, len(cams))
    p.eq("camera channel names", sorted(EXPECT_CAMERAS), cams)

    dims_by_cam: dict[str, Counter] = defaultdict(Counter)
    for sd in tables["sample_data.json"]:
        ch = cs_channel[sd["calibrated_sensor_token"]]
        if ch.startswith("CAM_"):
            dims_by_cam[ch][(sd["width"], sd["height"])] += 1

    def fmt(counter: Counter) -> dict[str, int]:
        return {f"{w}x{h}": n for (w, h), n in sorted(counter.items())}

    bad = {c: fmt(v) for c, v in dims_by_cam.items() if set(v) != {(EXPECT_IMAGE_W, EXPECT_IMAGE_H)}}
    p.eq(
        f"sample_data w×h == {EXPECT_IMAGE_W}×{EXPECT_IMAGE_H} (all 6 cams)",
        {},
        bad,
        detail=json.dumps({c: fmt(v) for c, v in sorted(dims_by_cam.items())}),
    )

    # The JSON says 1600×900; confirm the pixels agree. Metadata claiming a
    # resolution the encoder did not write would silently corrupt every
    # projection downstream.
    header_checked = []
    for cam in sorted(dims_by_cam):
        files = [
            sd["filename"] for sd in tables["sample_data.json"]
            if cs_channel[sd["calibrated_sensor_token"]] == cam and sd["is_key_frame"]
        ][:2]
        for rel in files:
            w, h = jpeg_dimensions(os.path.join(paths.dataroot, rel))
            header_checked.append({"channel": cam, "file": rel, "width": w, "height": h})
    off = [r for r in header_checked if (r["width"], r["height"]) != (EXPECT_IMAGE_W, EXPECT_IMAGE_H)]
    p.eq(
        f"JPEG SOF header == {EXPECT_IMAGE_W}×{EXPECT_IMAGE_H} ({len(header_checked)} files)",
        0, len(off), detail=json.dumps(off) if off else "",
    )
    p.recon["jpeg_header_samples"] = header_checked


def probe_intrinsics(p: Probe, tables: dict[str, list[dict]], sensors: dict[str, dict]) -> None:
    print("\n== C. CAM_FRONT intrinsics ==")
    variants: dict[tuple, int] = Counter()
    for cs in tables["calibrated_sensor.json"]:
        if sensors[cs["sensor_token"]]["channel"] != "CAM_FRONT":
            continue
        K = cs["camera_intrinsic"]
        variants[tuple(round(v, 6) for row in K for v in row)] += 1

    p.recon["cam_front_intrinsic_variants"] = [
        {"K": [list(k[0:3]), list(k[3:6]), list(k[6:9])], "n_calibrations": n}
        for k, n in variants.items()
    ]
    print(f"  distinct CAM_FRONT calibrations: {len(variants)}")
    for k, n in variants.items():
        print(f"    n={n}  fx={k[0]:.3f} fy={k[4]:.3f} cx={k[2]:.3f} cy={k[5]:.3f}")

    p.check("CAM_FRONT intrinsics present", len(variants) > 0, ">=1", len(variants))

    def matches(k) -> bool:
        return (
            abs(k[0] - EXPECT_CAM_FRONT_FX) <= INTRINSIC_TOL
            and abs(k[4] - EXPECT_CAM_FRONT_FY) <= INTRINSIC_TOL
            and abs(k[2] - EXPECT_CAM_FRONT_CX) <= INTRINSIC_TOL
            and abs(k[5] - EXPECT_CAM_FRONT_CY) <= INTRINSIC_TOL
        )

    hit = [k for k in variants if matches(k)]
    p.check(
        "CAM_FRONT K matches §0 (fx,fy,cx,cy)",
        len(hit) > 0,
        f"fx=fy={EXPECT_CAM_FRONT_FX}, cx={EXPECT_CAM_FRONT_CX}, cy={EXPECT_CAM_FRONT_CY}",
        [f"fx={k[0]:.3f} fy={k[4]:.3f} cx={k[2]:.3f} cy={k[5]:.3f}" for k in variants],
    )

    # Structural invariants that must hold for every camera calibration, not
    # just CAM_FRONT: upper-triangular K, zero skew, unit bottom row.
    struct_bad = []
    for cs in tables["calibrated_sensor.json"]:
        ch = sensors[cs["sensor_token"]]["channel"]
        if not ch.startswith("CAM_"):
            continue
        K = cs["camera_intrinsic"]
        if (
            len(K) != 3 or any(len(r) != 3 for r in K)
            or K[1][0] != 0 or K[2][0] != 0 or K[2][1] != 0 or K[2][2] != 1
            or K[0][1] != 0
        ):
            struct_bad.append({"channel": ch, "token": cs["token"], "K": K})
    p.eq("camera K upper-triangular, zero skew, K[2][2]==1", 0, len(struct_bad),
         detail=json.dumps(struct_bad[:3]) if struct_bad else "")

    # Non-camera sensors must carry an empty intrinsic — a populated one means
    # the channel join is wrong.
    noncam_bad = [
        sensors[cs["sensor_token"]]["channel"]
        for cs in tables["calibrated_sensor.json"]
        if not sensors[cs["sensor_token"]]["channel"].startswith("CAM_") and cs["camera_intrinsic"]
    ]
    p.eq("non-camera sensors have empty camera_intrinsic", 0, len(noncam_bad),
         detail=str(sorted(set(noncam_bad))) if noncam_bad else "")


def probe_timing(p: Probe, tables: dict[str, list[dict]], cs_channel: dict[str, str]) -> None:
    print("\n== D. timestamps and cadence ==")
    sample_scene = {s["token"]: s["scene_token"] for s in tables["sample.json"]}

    # --- 16-digit microsecond timestamps ---
    sd_digits = Counter(digit_count(sd["timestamp"]) for sd in tables["sample_data.json"])
    sm_digits = Counter(digit_count(s["timestamp"]) for s in tables["sample.json"])
    p.eq("sample_data.timestamp digit counts", {EXPECT_TIMESTAMP_DIGITS: len(tables["sample_data.json"])},
         dict(sd_digits))
    p.eq("sample.timestamp digit counts", {EXPECT_TIMESTAMP_DIGITS: len(tables["sample.json"])},
         dict(sm_digits))

    all_ts = [sd["timestamp"] for sd in tables["sample_data.json"]]
    t_min, t_max = min(all_ts), max(all_ts)
    p.check("timestamps are positive int64", all(isinstance(t, int) and 0 < t < 2**63 for t in all_ts),
            "int64 > 0", f"[{t_min}, {t_max}]")
    p.recon["timestamps"] = {
        "time_base_on_disk": "unix_us",
        "time_base_internal": "unix_ns",
        "min_us": t_min,
        "max_us": t_max,
        "min_ns": t_min * 1000,
        "max_ns": t_max * 1000,
        "span_s": round((t_max - t_min) / 1e6, 3),
        "digits": dict(sd_digits),
    }
    print(f"  span {t_min} .. {t_max} µs  ({p.recon['timestamps']['span_s']} s of wall time)")

    # --- per-channel Δt, grouped by scene so cross-scene gaps never enter ---
    per_channel: dict[str, list[float]] = defaultdict(list)
    per_channel_kf: dict[str, list[float]] = defaultdict(list)
    counts: Counter = Counter()
    kf_counts: Counter = Counter()
    grouped: dict[tuple[str, str], list[tuple[int, bool]]] = defaultdict(list)

    for sd in tables["sample_data.json"]:
        ch = cs_channel[sd["calibrated_sensor_token"]]
        counts[ch] += 1
        if sd["is_key_frame"]:
            kf_counts[ch] += 1
        scene = sample_scene.get(sd["sample_token"])
        grouped[(ch, scene)].append((sd["timestamp"], bool(sd["is_key_frame"])))

    for (ch, _scene), rows in grouped.items():
        rows.sort()
        ts = [t for t, _ in rows]
        per_channel[ch].extend((b - a) / 1000.0 for a, b in zip(ts, ts[1:]))
        kts = [t for t, k in rows if k]
        kts.sort()
        per_channel_kf[ch].extend((b - a) / 1000.0 for a, b in zip(kts, kts[1:]))

    p.recon["records_per_channel"] = dict(sorted(counts.items()))
    p.recon["keyframes_per_channel"] = dict(sorted(kf_counts.items()))
    p.recon["dt_ms_all_records"] = {ch: describe(v) for ch, v in sorted(per_channel.items())}
    p.recon["dt_ms_keyframes_only"] = {ch: describe(v) for ch, v in sorted(per_channel_kf.items())}

    print(f"  {'channel':<20} {'records':>8} {'keyfr':>6} {'dt_med_ms':>10} {'Hz':>7}")
    for ch in sorted(counts):
        d = p.recon["dt_ms_all_records"].get(ch, {})
        med = d.get("median")
        hz = round(1000.0 / med, 2) if med else float("nan")
        print(f"  {ch:<20} {counts[ch]:>8} {kf_counts[ch]:>6} {med if med else '-':>10} {hz:>7}")

    # --- the §0 LiDAR claims ---
    lidar_dt = per_channel[LIDAR_CHANNEL]
    lidar_med = statistics.median(lidar_dt) if lidar_dt else None
    p.within(f"{LIDAR_CHANNEL} median Δt (ms)", *LIDAR_DT_BAND_MS, lidar_med,
             detail=f"§0 claims {EXPECT_LIDAR_DT_MS} ms; full distribution in dt_ms_all_records")
    p.within(f"{LIDAR_CHANNEL} implied rate (Hz)", *EXPECT_LIDAR_HZ_BAND,
             round(1000.0 / lidar_med, 3) if lidar_med else None)
    kf_med = statistics.median(per_channel_kf[LIDAR_CHANNEL]) if per_channel_kf[LIDAR_CHANNEL] else None
    p.within(f"{LIDAR_CHANNEL} keyframe median Δt (ms)", *EXPECT_KEYFRAME_DT_MS_BAND, kf_med,
             detail="keyframes are the ~2 Hz annotation cadence, not the sweep cadence")

    p.eq(f"{LIDAR_CHANNEL} keyframe count == samples", EXPECT_KEYFRAMES, kf_counts[LIDAR_CHANNEL])
    ratio = round(counts[LIDAR_CHANNEL] / EXPECT_KEYFRAMES, 2)
    p.recon["sweeps_per_keyframe"] = {
        "lidar_records": counts[LIDAR_CHANNEL],
        "keyframes": EXPECT_KEYFRAMES,
        "ratio": ratio,
    }
    print(f"  sweeps per keyframe: {counts[LIDAR_CHANNEL]} / {EXPECT_KEYFRAMES} = {ratio}")

    # --- LIDAR_TOP extrinsic yaw (ISO 8855, about +z) ---
    yaws = []
    for cs in tables["calibrated_sensor.json"]:
        if cs_channel.get(cs["token"]) == LIDAR_CHANNEL:
            yaws.append({
                "token": cs["token"],
                "translation_m": cs["translation"],
                "yaw_rad": round(quaternion_yaw_rad(cs["rotation"]), 6),
                "yaw_deg": round(math.degrees(quaternion_yaw_rad(cs["rotation"])), 4),
            })
    p.recon["lidar_top_extrinsics"] = yaws
    print(f"  {LIDAR_CHANNEL} extrinsic yaw: " + ", ".join(f"{y['yaw_deg']}°" for y in yaws))


def probe_point_records(p: Probe, paths: Paths, tables: dict[str, list[dict]],
                        cs_channel: dict[str, str], n_files: int = 5) -> None:
    print("\n== E. LiDAR point records ==")
    lidar_files = [
        sd["filename"] for sd in tables["sample_data.json"]
        if cs_channel[sd["calibrated_sensor_token"]] == LIDAR_CHANNEL
    ]
    p.check(f"{LIDAR_CHANNEL} files referenced", len(lidar_files) > 0, ">0", len(lidar_files))

    ext_bad = sorted({os.path.splitext(f)[1] for f in lidar_files} - {".bin"})
    p.eq(f"{LIDAR_CHANNEL} filenames end .pcd.bin", 0, len(ext_bad), detail=str(ext_bad))

    # Every file, not a sample: a single non-conforming cloud breaks the
    # struct-array read in stage 1 with an off-by-one shear, not an exception.
    remainders = Counter()
    sizes = []
    for rel in lidar_files:
        n = os.path.getsize(os.path.join(paths.dataroot, rel))
        sizes.append(n)
        remainders[n % POINT_RECORD_BYTES] += 1
    p.eq(f"filesize % {POINT_RECORD_BYTES} == 0 (all {len(lidar_files)} clouds)",
         {0: len(lidar_files)}, dict(remainders),
         detail=f"{POINT_RECORD_BYTES} bytes = 5 × float32 {POINT_FIELDS}")

    pts = [n // POINT_RECORD_BYTES for n in sizes]
    p.recon["lidar_point_counts"] = describe([float(x) for x in pts])
    print(f"  {len(sizes)} clouds, {min(pts)}–{max(pts)} points "
          f"(median {int(statistics.median(pts))}), all {POINT_RECORD_BYTES}-byte records")

    # Decode a handful and confirm the 5-column layout is real, not just a
    # size coincidence: ring must be a small non-negative integer, xyz finite.
    decoded = []
    for rel in lidar_files[:n_files]:
        full = os.path.join(paths.dataroot, rel)
        arr = np.fromfile(full, dtype=np.float32)
        ok_reshape = arr.size % len(POINT_FIELDS) == 0
        pc = arr.reshape(-1, len(POINT_FIELDS)) if ok_reshape else np.empty((0, 5), np.float32)
        ring = pc[:, 4]
        rec = {
            "file": rel,
            "bytes": os.path.getsize(full),
            "points": int(pc.shape[0]),
            "xyz_finite": bool(np.isfinite(pc[:, :3]).all()),
            "ring_integral": bool(np.all(ring == np.floor(ring))),
            "ring_min": float(ring.min()) if pc.size else None,
            "ring_max": float(ring.max()) if pc.size else None,
            "intensity_min": float(pc[:, 3].min()) if pc.size else None,
            "intensity_max": float(pc[:, 3].max()) if pc.size else None,
            "range_m_max": float(np.linalg.norm(pc[:, :3], axis=1).max()) if pc.size else None,
        }
        decoded.append(rec)
    p.recon["decoded_clouds"] = decoded
    p.eq("decoded clouds: xyz finite", n_files, sum(d["xyz_finite"] for d in decoded))
    p.eq("decoded clouds: ring column integral", n_files, sum(d["ring_integral"] for d in decoded))
    p.check("decoded clouds: ring in [0, 63]",
            all(0 <= d["ring_min"] and d["ring_max"] <= 63 for d in decoded),
            "[0, 63]", [(d["ring_min"], d["ring_max"]) for d in decoded])

    # RADAR is a different container entirely — ASCII PCD header, 18 fields.
    # Recorded so no one later assumes the 20-byte layout generalises.
    radar = next(
        (sd["filename"] for sd in tables["sample_data.json"]
         if cs_channel[sd["calibrated_sensor_token"]].startswith("RADAR_")), None)
    if radar:
        with open(os.path.join(paths.dataroot, radar), "rb") as fh:
            head = fh.read(512).split(b"DATA", 1)[0].decode("ascii", "replace")
        fields = next((l.split()[1:] for l in head.splitlines() if l.startswith("FIELDS")), [])
        p.recon["radar_container"] = {"file": radar, "format": "PCD (ASCII header)", "fields": fields}
        print(f"  RADAR container: PCD with {len(fields)} fields — NOT the 20-byte layout")


def probe_files_present(p: Probe, paths: Paths, tables: dict[str, list[dict]]) -> None:
    print("\n== F. referenced blobs present on disk ==")
    missing = [
        sd["filename"] for sd in tables["sample_data.json"]
        if not os.path.isfile(os.path.join(paths.dataroot, sd["filename"]))
    ]
    p.eq("referenced files missing on disk", 0, len(missing),
         detail=json.dumps(missing[:5]) if missing else "")
    roots = Counter(f["filename"].split("/", 1)[0] for f in tables["sample_data.json"])
    p.recon["blob_roots"] = dict(roots)
    print(f"  {len(tables['sample_data.json'])} referenced blobs, {len(missing)} missing, roots={dict(roots)}")


def probe_recon_extras(p: Probe, tables: dict[str, list[dict]]) -> None:
    """Phase 1 reconnaissance dump: category list and per-scene annotation counts."""
    print("\n== G. reconnaissance ==")
    cats = sorted(c["name"] for c in tables["category.json"])
    p.recon["categories"] = cats
    print(f"  {len(cats)} categories: {cats[0]} … {cats[-1]}")

    sample_scene = {s["token"]: s["scene_token"] for s in tables["sample.json"]}
    ann_by_scene = Counter(sample_scene[a["sample_token"]] for a in tables["sample_annotation.json"])
    cat_of_instance = {i["token"]: i["category_token"] for i in tables["instance.json"]}
    cat_name = {c["token"]: c["name"] for c in tables["category.json"]}
    cat_hist = Counter(cat_name[cat_of_instance[a["instance_token"]]]
                       for a in tables["sample_annotation.json"])
    p.recon["annotations_per_category"] = dict(cat_hist.most_common())

    per_scene = []
    print(f"  {'scene':<12} {'samples':>8} {'anns':>7} {'anns/frame':>11}  description")
    for sc in sorted(tables["scene.json"], key=lambda s: s["name"]):
        n_ann = ann_by_scene.get(sc["token"], 0)
        row = {
            "name": sc["name"],
            "token": sc["token"],
            "nbr_samples": sc["nbr_samples"],
            "annotations": n_ann,
            "anns_per_frame": round(n_ann / sc["nbr_samples"], 2) if sc["nbr_samples"] else 0.0,
            "description": sc["description"],
        }
        per_scene.append(row)
        print(f"  {row['name']:<12} {row['nbr_samples']:>8} {n_ann:>7} {row['anns_per_frame']:>11}"
              f"  {row['description'][:44]}")
    p.recon["per_scene"] = per_scene


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/paths.yaml")
    ap.add_argument("--json", metavar="OUT", help="write the full probe record here (never under dataroot)")
    args = ap.parse_args()

    try:
        paths = load_paths(args.config)
    except (PathValidationError, OSError) as exc:
        print(f"substrate not addressable: {exc}", file=sys.stderr)
        return 2

    print("DhakaScenes Pilot — Phase 1 substrate probe")
    print(f"  dataroot   {paths.dataroot}")
    print(f"  version    {paths.version}  ({paths.version_dir})")
    print("  frame      ISO 8855 ego-centric: x fwd, y left, z up, yaw about +z")
    print("  time       on disk unix_us -> internal int64 unix_ns")
    print("  compute    CPU only, no models, no devkit")

    tables = {name: load_table(paths, name) for name in METADATA_TABLES}
    cs_channel, sensors = build_channel_index(tables)

    p = Probe()
    p.recon["dataroot_realpath"] = paths.dataroot
    p.recon["paths"] = paths.as_dict()

    probe_metadata(p, paths, tables)
    probe_cameras(p, paths, tables, cs_channel)
    probe_intrinsics(p, tables, sensors)
    probe_timing(p, tables, cs_channel)
    probe_point_records(p, paths, tables, cs_channel)
    probe_files_present(p, paths, tables)
    probe_recon_extras(p, tables)

    print("\n== §0 contract ==")
    for c in p.checks:
        print(c.line())

    failed = p.failed
    print(f"\n{len(p.checks) - len(failed)}/{len(p.checks)} checks passed")

    if args.json:
        from pipeline.common.paths import assert_dataroot_read_only
        out = assert_dataroot_read_only(paths, args.json)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "checks": [vars(c) for c in p.checks],
                    "n_failed": len(failed),
                    "recon": p.recon,
                },
                fh, indent=2, sort_keys=True, default=str,
            )
        print(f"probe record -> {out}")

    if failed:
        print("\nSUBSTRATE DOES NOT MATCH §0 — Phase 1 gate is closed:", file=sys.stderr)
        for c in failed:
            print(f"  {c.name}: expected {c.expected}, measured {c.measured}", file=sys.stderr)
        return 1

    print("substrate verified: every §0 claim reproduced from this disk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
