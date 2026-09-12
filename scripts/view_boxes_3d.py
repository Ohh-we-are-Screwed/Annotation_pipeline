#!/usr/bin/env python3
"""Read-only 3D viewer exporter for stage6_stereo_box output (approach A, 2026-09-12).

Writes <out>/index.json + <out>/kf/<i>.json (decimated ego cloud as base64 float32,
boxes, the two ZED images' K and T_cam_ego, ground plane) and copies the two JPEGs
per keyframe to <out>/img/. `--serve PORT` serves <out> with http.server; open
http://localhost:PORT/ (viewer/index.html is copied to <out>/index.html).

    python -m scripts.view_boxes_3d --scene chunk_0010 --out /tmp/view --serve 8000
"""
from __future__ import annotations
import argparse, base64, json, math, os, shutil, sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import numpy as np
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.common.conventions import CAMERA, EGO, Transform  # noqa: E402
from pipeline.common.eval_region import STEREO_RANGE_CAP_DEFAULT_M  # noqa: E402
from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.stage0_data_probe.probe import Substrate  # noqa: E402
from pipeline.stage1_ingestion.ingest import pitch_rotate_xz, read_pcd_bin  # noqa: E402
from scripts.render_boxes_3d import box_corners_ego  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZED = ("CAM_FRONT", "CAM_BACK")
# Stage 5/6's own cap, drawn as a ring so "why is there no box out there" is
# answered on screen. The file is the spike's (Task 4); 25.0 until it exists.
STEREO_BOX_CONFIG = os.path.join(ROOT, "configs", "stereo_box.yaml")


def decimate(cloud: np.ndarray, max_points: int) -> np.ndarray:
    if len(cloud) <= max_points:
        return cloud
    idx = np.linspace(0, len(cloud) - 1, max_points).astype(int)   # deterministic, file order
    return cloud[idx]


def encode_cloud(cloud: np.ndarray) -> dict:
    xyz = np.ascontiguousarray(cloud[:, :3].astype(np.float32))
    ring = np.ascontiguousarray(np.clip(cloud[:, 4], 0, 255).astype(np.uint8))
    return {"n": int(len(cloud)), "xyz_b64": base64.b64encode(xyz.tobytes()).decode(),
            "ring_b64": base64.b64encode(ring.tobytes()).decode()}


def project_corners(center, size_wlh, yaw, K, T_cam_ego, image_size):
    """(8,2) pixel corners + (8,) visibility (z>0 and inside the image)."""
    h = yaw / 2.0
    corners = box_corners_ego(center, size_wlh, [math.cos(h), 0.0, 0.0, math.sin(h)])
    hom = np.column_stack([corners, np.ones(8)]) @ np.asarray(T_cam_ego).T
    z = hom[:, 2]; ok = z > 0.05
    uv = np.zeros((8, 2))
    uv[ok, 0] = K[0, 0] * hom[ok, 0] / z[ok] + K[0, 2]; uv[ok, 1] = K[1, 1] * hom[ok, 1] / z[ok] + K[1, 2]
    W, H = image_size
    vis = ok & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return uv, vis


def range_cap_m(config_path: str = STEREO_BOX_CONFIG) -> float:
    """`stereo_range_cap_m` from configs/stereo_box.yaml, or the default if unwritten."""
    if not os.path.exists(config_path):
        return STEREO_RANGE_CAP_DEFAULT_M
    with open(config_path) as f:
        return float((yaml.safe_load(f) or {}).get("stereo_range_cap_m", STEREO_RANGE_CAP_DEFAULT_M))


def plane_abd(plane):
    """{a, b, d} of a Stage 1 ground_reference_plane; None stays None.

    Stage 1 writes null when RANSAC found no plane (ingest.py:1451). The viewer
    draws no ground for such a keyframe rather than failing the whole scene.
    """
    return None if plane is None else {k: plane[k] for k in ("a", "b", "d")}


def pose_corrections(config_path: str = STEREO_BOX_CONFIG) -> dict:
    """`camera_pose_pitch_correction` from configs/stereo_box.yaml, or {}.

    IMAGE-SPACE CONSUMERS ONLY (this exporter and scripts/eval_stereo_box.py).
    pipeline/ must NOT read it: Stage 5 lifts masks through the EXPORT pose, and
    points and masks stay mutually consistent only while it does.
    """
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as f:
        return (yaml.safe_load(f) or {}).get("camera_pose_pitch_correction") or {}


def pitch_correction_matrix(deg: float, pivot_x: float, pivot_z: float) -> np.ndarray:
    """4x4 rigid rotation about the axis parallel to ego +y through (pivot_x, ., pivot_z).

    Built FROM `pipeline.stage1_ingestion.ingest.pitch_rotate_xz` — the images of the
    ego basis vectors and of the origin are its columns — so the viewer's convention
    and sign cannot drift from Stage 1's recorded `--stereo-pitch-correction` flag.
    """
    ox, oz = pitch_rotate_xz(0.0, 0.0, deg, pivot_x, pivot_z)
    xx, xz = pitch_rotate_xz(1.0, 0.0, deg, pivot_x, pivot_z)
    zx, zz = pitch_rotate_xz(0.0, 1.0, deg, pivot_x, pivot_z)
    M = np.eye(4)
    M[0, 0], M[2, 0] = xx - ox, xz - oz
    M[0, 2], M[2, 2] = zx - ox, zz - oz
    M[0, 3], M[2, 3] = ox, oz
    return M


def load_calibs(paths, kf_row, pose_corrections: dict | None = None):
    """K and ego->camera per ZED channel, from the export's calibrated_sensor records.

    `pose_corrections` is `{channel: {deg, pivot_x_m, pivot_z_m}}` (see the function of
    that name). For a named channel the camera POSE is rotated in the ego frame,
    `T_ego_cam_corr = R_pivot @ T_ego_cam_export`, before inverting — the export's
    CAM_FRONT pose is pitched down by ~9.1 deg, so the uncorrected projection puts a
    box's wireframe ~150 px ABOVE the object it was built from. Rotating the POSE (not
    the box) pitches the camera back up and the wireframe lands on the object.
    """
    sub = Substrate.load(paths); cs = sub.by_token("calibrated_sensor.json")
    out = {}
    for ch in ZED:
        rec = cs[kf_row["cameras"][ch]["calibrated_sensor_token"]]
        T_ego_cam = Transform.from_nuscenes(rec, source_frame=CAMERA, parent_frame=EGO).matrix()
        corr = (pose_corrections or {}).get(ch)
        if corr:
            T_ego_cam = pitch_correction_matrix(
                float(corr["deg"]), float(corr["pivot_x_m"]), float(corr["pivot_z_m"])) @ T_ego_cam
        out[ch] = {"K": np.array(rec["camera_intrinsic"]), "T_cam_ego": np.linalg.inv(T_ego_cam)}
    return out


def export_keyframe(i, kf_row, boxes_rows, calibs, ground, dataroot, out_dir, max_points) -> dict:
    cloud = decimate(read_pcd_bin(kf_row["single_sweep_cloud"]["path"]), max_points)
    cams = {}
    for ch in ZED:
        src = os.path.join(dataroot, kf_row["cameras"][ch]["path"]) if not os.path.isabs(kf_row["cameras"][ch]["path"]) else kf_row["cameras"][ch]["path"]
        # Keyed by token, not index: re-using one --out for a second scene must not
        # leave the old scene's photos under the new scene's clouds and boxes.
        rel = f"img/{kf_row['keyframe_token']}_{ch}.jpg"
        dst = os.path.join(out_dir, rel); os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.lexists(dst):
            os.unlink(dst)
        try: os.link(src, dst)
        except OSError: shutil.copy2(src, dst)
        cams[ch] = {"image": rel, "K": calibs[ch]["K"].tolist(), "T_cam_ego": calibs[ch]["T_cam_ego"].tolist()}
    boxes = [{"instance_id": r["instance_id"], "channel": r["channel"], "class_name": r["class_name"], "score": r["score"],
              "status": r["status"], "stereo": r.get("stereo"), "box": r["box"]} for r in boxes_rows]
    payload = {"index": i, "token": kf_row["keyframe_token"], "t_ns": kf_row["t_ns"], "cloud": encode_cloud(cloud),
               "boxes": boxes, "cameras": cams, "ground": ground}
    os.makedirs(os.path.join(out_dir, "kf"), exist_ok=True)
    with open(os.path.join(out_dir, "kf", f"{i:05d}.json"), "w") as f:
        json.dump(payload, f)
    return {"index": i, "token": kf_row["keyframe_token"], "n_boxes": sum(1 for b in boxes if b["box"])}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    ap.add_argument("--config", default=STEREO_BOX_CONFIG, help="the stage config the range cap and the "
                    "image-space camera pose correction are read from")
    ap.add_argument("--boxes-dir", default=None, help="default <work_root>/stage6_stereo_box")
    ap.add_argument("--stage1-dir", default=None)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-points", type=int, default=80000)
    ap.add_argument("--serve", type=int, default=0, help="port; 0 = export only")
    a = ap.parse_args(argv)
    paths = load_paths(a.paths)
    boxes_dir = a.boxes_dir or os.path.join(paths.work_root, "stage6_stereo_box")
    stage1 = a.stage1_dir or os.path.join(paths.work_root, "stage1_ingestion")
    kfs = [json.loads(l) for l in open(os.path.join(stage1, "scenes", a.scene, "keyframes.jsonl")) if l.strip()]
    diag = json.load(open(os.path.join(stage1, "scenes", a.scene, "filter_diagnostics.json")))["keyframes"]
    ground = {d["keyframe_token"]: plane_abd(d["ground_reference_plane"]) for d in diag}
    by_kf: dict[str, list] = {}
    for l in open(os.path.join(boxes_dir, "scenes", a.scene, "boxes.jsonl")):
        if l.strip():
            r = json.loads(l); by_kf.setdefault(r["keyframe_token"], []).append(r)
    corrections = pose_corrections(a.config)
    calibs = load_calibs(paths, kfs[0], corrections)
    os.makedirs(a.out, exist_ok=True)
    index = [export_keyframe(i, kf, by_kf.get(kf["keyframe_token"], []), calibs, ground[kf["keyframe_token"]],
                             paths.dataroot, a.out, a.max_points) for i, kf in enumerate(kfs)]
    json.dump({"scene": a.scene, "keyframes": index, "range_cap_m": range_cap_m(a.config),
               "camera_pose_corrections": corrections},
              open(os.path.join(a.out, "index.json"), "w"))
    shutil.copy2(os.path.join(ROOT, "viewer", "index.html"), os.path.join(a.out, "index.html"))
    print(f"exported {len(index)} keyframes to {a.out}")
    if a.serve:
        os.chdir(a.out)
        print(f"serving http://localhost:{a.serve}/  (Ctrl-C to stop)")
        ThreadingHTTPServer(("127.0.0.1", a.serve), SimpleHTTPRequestHandler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
