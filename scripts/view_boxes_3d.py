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
from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.stage0_data_probe.probe import Substrate  # noqa: E402
from pipeline.stage1_ingestion.ingest import read_pcd_bin  # noqa: E402
from scripts.render_boxes_3d import box_corners_ego  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZED = ("CAM_FRONT", "CAM_BACK")
# Stage 5/6's own cap, drawn as a ring so "why is there no box out there" is
# answered on screen. The file is the spike's (Task 4); 25.0 until it exists.
STEREO_BOX_CONFIG = os.path.join(ROOT, "configs", "stereo_box.yaml")
DEFAULT_RANGE_CAP_M = 25.0


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
        return DEFAULT_RANGE_CAP_M
    with open(config_path) as f:
        return float((yaml.safe_load(f) or {}).get("stereo_range_cap_m", DEFAULT_RANGE_CAP_M))


def load_calibs(paths, kf_row):
    sub = Substrate.load(paths); cs = sub.by_token("calibrated_sensor.json")
    out = {}
    for ch in ZED:
        rec = cs[kf_row["cameras"][ch]["calibrated_sensor_token"]]
        T_ego_cam = Transform.from_nuscenes(rec, source_frame=CAMERA, parent_frame=EGO).matrix()
        out[ch] = {"K": np.array(rec["camera_intrinsic"]), "T_cam_ego": np.linalg.inv(T_ego_cam)}
    return out


def export_keyframe(i, kf_row, boxes_rows, calibs, ground, dataroot, out_dir, max_points) -> dict:
    cloud = decimate(read_pcd_bin(kf_row["single_sweep_cloud"]["path"]), max_points)
    cams = {}
    for ch in ZED:
        src = os.path.join(dataroot, kf_row["cameras"][ch]["path"]) if not os.path.isabs(kf_row["cameras"][ch]["path"]) else kf_row["cameras"][ch]["path"]
        dst = os.path.join(out_dir, "img", f"{i:05d}_{ch}.jpg"); os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            try: os.link(src, dst)
            except OSError: shutil.copy2(src, dst)
        cams[ch] = {"image": f"img/{i:05d}_{ch}.jpg", "K": calibs[ch]["K"].tolist(), "T_cam_ego": calibs[ch]["T_cam_ego"].tolist()}
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
    ground = {d["keyframe_token"]: {k: d["ground_reference_plane"][k] for k in ("a", "b", "d")} for d in diag}
    by_kf: dict[str, list] = {}
    for l in open(os.path.join(boxes_dir, "scenes", a.scene, "boxes.jsonl")):
        if l.strip():
            r = json.loads(l); by_kf.setdefault(r["keyframe_token"], []).append(r)
    calibs = load_calibs(paths, kfs[0])
    os.makedirs(a.out, exist_ok=True)
    index = [export_keyframe(i, kf, by_kf.get(kf["keyframe_token"], []), calibs, ground[kf["keyframe_token"]],
                             paths.dataroot, a.out, a.max_points) for i, kf in enumerate(kfs)]
    json.dump({"scene": a.scene, "keyframes": index, "range_cap_m": range_cap_m()},
              open(os.path.join(a.out, "index.json"), "w"))
    shutil.copy2(os.path.join(ROOT, "viewer", "index.html"), os.path.join(a.out, "index.html"))
    print(f"exported {len(index)} keyframes to {a.out}")
    if a.serve:
        os.chdir(a.out)
        print(f"serving http://localhost:{a.serve}/  (Ctrl-C to stop)")
        ThreadingHTTPServer(("0.0.0.0", a.serve), SimpleHTTPRequestHandler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
