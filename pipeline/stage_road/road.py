"""Stage road — the road-surface layer (C33). Consumes Stage 1 ALONE.

One class ("road surface"), one SAM 3 text prompt per (keyframe, camera), the
union of the returned regions as that camera's road mask, and a plane-gated
paint onto the RAW LIDAR_TOP cloud as the 3D product. Opt-in step `road` in
run_stages.sh; the CVAT publish is `cvatroad` via scripts/export_road_coco.py.

Design decisions this file is the record of (C33; full trail in
handover/2026-09-01-stage3c-and-road-seg-handoff.md §5-§6 and the
`road-seg-design` workflow output):

- **The prompt is "paved road", never "road".** Measured on 60 images across
  all 8 cameras: plain "road" returns ZERO instances on 11 of 60 images, and
  an empty mask downstream reads as "no road in this frame" when it means
  "the prompt missed" — the silent-wrongness shape this pipeline refuses
  elsewhere. "paved road" fired on 60/60.
- **Union, not instance identity.** SAM 3's text path is instance grounding:
  a divided road returns each carriageway separately. The road product is a
  surface, so a camera's mask is the union of every returned region and the
  row records how many regions were unioned.
- **No cross-camera contest.** Stage 4's IoA-NMS exists to delete duplicate
  views of one OBJECT; a road genuinely is the same surface in every camera,
  so a point is a candidate if ANY camera's mask covers it and
  `n_cameras_road` counts how many agreed. Running the road through Stage 4's
  contest would delete it in all but one camera.
- **The plane gate is the occlusion policy.** The 2D->3D paint claims every
  point along the ray through a road pixel — road 30 m out, the underside of
  a bus, a wall past the road edge. The gate fits a plane to the stage's OWN
  candidates (anchor = median z of candidates within `anchor_radius_m`, then
  a least-squares fit on the band around the anchor) and keeps only points
  within `band_m` of it. It never consults Stage 1's fitted plane, so it is
  immune to the Stage 1 ground-frame bug.
- **The cloud is the RAW LIDAR_TOP blob**, never Stage 1's filtered
  single-sweep cloud: 47.9% of that cloud is road band only because Stage 1's
  ground filter is broken, and a corrected filter deletes precisely the road.
  This stage applies T_ego_lidar exactly once, itself.
- **`__basis__ = "raw_lidar_top_file_order"`** rides in every points file:
  nothing else in this repo indexes the raw blob (Stage 5's point_index
  addresses the fused/filtered 82k-row derivative), and a consumer that
  guesses the wrong basis produces a plausible, fully populated, wrong answer.
- **Square-resize note (C13):** SAM 3's processor works at 1008x1008 and the
  masks are mapped back to 1280x720 by its own post-processing, exactly as
  Stage 4's SAM 3 path already does; the C13 prohibition is read as binding
  on OUR resize choices, with the model-internal geometry recorded here
  rather than hidden.

Run: python3 -m pipeline.stage_road.road --out-dir <work>/stage_road
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pipeline.common.conventions import (  # noqa: E402
    CAMERA,
    EGO,
    LIDAR,
    NUSCENES_GLOBAL,
    Transform,
    apply_transform,
    project_lidar_to_image,
)
from pipeline.common.manifest import (  # noqa: E402
    UpstreamRefusal,
    clear_markers,
    require_upstream,
    write_json_atomic,
    write_jsonl_atomic,
    write_marker,
)
from pipeline.common.schemas import IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX  # noqa: E402
from pipeline.stage1_ingestion.ingest import _least_squares_plane, read_pcd_bin  # noqa: E402

STAGE = "stage_road"
STAGE_SPEC = "dhakascenes-pilot/stage_road/v1"
PROVIDER = "sam3_image_text"

# Load-bearing constant, measured — see the module docstring.
ROAD_PROMPT = "paved road"

# The paint reuses Stage 5's two depth guards verbatim (lift.py cfg defaults);
# stage-imports-stage is the house pattern (lift.py imports probe, check.py
# imports proposals).
from pipeline.common.conventions import MIN_DEPTH_M  # noqa: E402
from pipeline.stage5_lift.lift import BEHIND_CAMERA_EPS_M  # noqa: E402

DEFAULT_BAND_M = 0.30           # ingest.py ground_band_m — inherited, untuned here
DEFAULT_ANCHOR_RADIUS_M = 20.0
DEFAULT_MIN_ANCHOR_POINTS = 200
DEFAULT_MAX_TILT_DEG = 10.0
DEFAULT_CONFIDENCE = 0.40       # the spike's single measured threshold pair

INDEX_BASIS = "raw_lidar_top_file_order"

# --- v2: the sidewalk class and the ZED refinement (C33 v2) -----------------
SIDEWALK_PROMPT = "sidewalk"    # 44/48 spike images fired; see C33 v2
SIDEWALK_INDEX = 26             # nuScenes-lidarseg flat.sidewalk
# ZED provenance rides in the fused cloud's ring column (Stage 1 fusion);
# each ring has a native co-located camera. ZED is the only ground-referenced
# calibrated signal on this rig (extrinsics validated against lidar ground
# agreement), and only these two cameras have it.
ZED_RING_FOR_CAMERA = {"CAM_FRONT": 10, "CAM_BACK": 11}
CARVE_RAISE_M = 0.08            # a Dhaka curb is 0.10-0.20 m; ZED noise ~±0.05
CARVE_RADIUS_PX = 7             # pixel reach of one raised ZED point's carve
# Sidewalk 3D band vs the ROAD plane: above the carve threshold, below half a
# metre (anything higher is wall/vegetation, not walkable surface). Untuned.
SIDEWALK_BAND_M = (0.02, 0.45)


# ---------------------------------------------------------------------------
# Pure pieces
# ---------------------------------------------------------------------------

def union_masks(masks) -> np.ndarray | None:
    """OR of a text forward's instance masks; None when nothing fired.

    None is not an empty mask: the spike showed an absent firing means "the
    prompt missed", and the row records it as such rather than as road-free.
    """
    out = None
    for m in masks:
        m = np.asarray(m, dtype=bool)
        out = m if out is None else (out | m)
    return out


def plane_gate(points_ego: np.ndarray, candidate_index: np.ndarray, *,
               band_m: float = DEFAULT_BAND_M,
               anchor_radius_m: float = DEFAULT_ANCHOR_RADIUS_M,
               min_anchor_points: int = DEFAULT_MIN_ANCHOR_POINTS,
               max_tilt_deg: float = DEFAULT_MAX_TILT_DEG) -> dict:
    """The occlusion policy: keep candidates within `band_m` of the road plane.

    Anchored on the MEDIAN z of the candidates within `anchor_radius_m` (not a
    global percentile: a flyover keyframe measured at d = -6.7 m would drag a
    percentile anchor off the road), fitted with Stage 1's own least-squares
    plane on the |z - anchor| <= 0.5 band. Refuses — kept empty, reason named —
    on a thin anchor or a fit steeper than `max_tilt_deg`.
    """
    cand = np.asarray(candidate_index, dtype=np.int64)
    empty = np.zeros(0, dtype=np.int64)
    if cand.size == 0:
        return {"kept_index": empty, "plane": None, "tilt_deg": None,
                "n_rejected_off_plane": 0, "refusal": "no candidates"}
    pts = points_ego[cand]
    near = pts[np.hypot(pts[:, 0], pts[:, 1]) <= anchor_radius_m]
    if near.shape[0] < min_anchor_points:
        return {"kept_index": empty, "plane": None, "tilt_deg": None,
                "n_rejected_off_plane": 0,
                "refusal": f"anchor has {near.shape[0]} points within "
                           f"{anchor_radius_m} m, needs {min_anchor_points}"}
    anchor_z = float(np.median(near[:, 2]))
    band = pts[np.abs(pts[:, 2] - anchor_z) <= 0.5]
    plane = _least_squares_plane(band[:, :3]) if band.shape[0] >= 3 else None
    if plane is None:
        return {"kept_index": empty, "plane": None, "tilt_deg": None,
                "n_rejected_off_plane": 0, "refusal": "plane fit failed"}
    a, b, c = plane
    tilt = math.degrees(math.atan(math.hypot(a, b)))
    if tilt > max_tilt_deg:
        return {"kept_index": empty, "plane": [a, b, c], "tilt_deg": tilt,
                "n_rejected_off_plane": 0,
                "refusal": f"fitted tilt {tilt:.1f} deg exceeds {max_tilt_deg} deg"}
    dist = np.abs(pts[:, 2] - (a * pts[:, 0] + b * pts[:, 1] + c))
    keep = dist <= band_m
    return {"kept_index": cand[keep], "plane": [a, b, c], "tilt_deg": tilt,
            "n_rejected_off_plane": int((~keep).sum()), "refusal": None}


def carve_raised(mask: np.ndarray, raised_vu: np.ndarray,
                 radius_px: int = CARVE_RADIUS_PX) -> tuple:
    """Remove mask pixels near ZED points that sit ABOVE the road plane.

    The refinement can only REMOVE road, never invent it: pixels with no
    elevated ZED support (beyond ZED range, occluded, on-plane) stand as SAM
    produced them. Returns (carved mask, n pixels carved)."""
    import cv2
    mask = np.asarray(mask, dtype=bool)
    raised_vu = np.asarray(raised_vu, dtype=np.int64).reshape(-1, 2)
    if raised_vu.shape[0] == 0:
        return mask.copy(), 0
    stamp = np.zeros(mask.shape, dtype=np.uint8)
    stamp[raised_vu[:, 0], raised_vu[:, 1]] = 1
    k = 2 * int(radius_px) + 1
    grown = cv2.dilate(stamp, np.ones((k, k), np.uint8)).astype(bool)
    return mask & ~grown, int((mask & grown).sum())


def resolve_sidewalk(road_mask: np.ndarray, sidewalk_mask: np.ndarray) -> np.ndarray:
    """The overlap rule: ROAD keeps contested pixels; sidewalk takes the rest.

    Measured on the spike (CAM_RIGHT): the sidewalk prompt can over-claim the
    ENTIRE road, so subtracting sidewalk from road would delete real road.
    Where ZED exists the carve has already arbitrated by height — pixels it
    released from the road mask fall through to the sidewalk claim here."""
    return np.asarray(sidewalk_mask, dtype=bool) & ~np.asarray(road_mask, dtype=bool)


def paint_cameras(points_ego: np.ndarray, *, observations, road_mask_by_channel: dict,
                  lidar_ego_pose: dict, ego_pose_table: dict, calibrated_table: dict):
    """Project the cloud into each camera and test its road mask. Union, no
    contest: (seen_index, candidate_index, n_cameras_road per candidate)."""
    ego_pose_at_lidar = Transform.from_nuscenes(
        lidar_ego_pose, source_frame=EGO, parent_frame=NUSCENES_GLOBAL)
    n = points_ego.shape[0]
    seen = np.zeros(n, dtype=bool)
    votes = np.zeros(n, dtype=np.int16)
    for obs in observations:
        mask = road_mask_by_channel.get(obs.channel)
        ego_pose_at_camera = Transform.from_nuscenes(
            ego_pose_table[obs.ego_pose_token], source_frame=EGO, parent_frame=NUSCENES_GLOBAL)
        calibrated = calibrated_table[obs.calibrated_sensor_token]
        extrinsic = Transform.from_nuscenes(calibrated, source_frame=CAMERA, parent_frame=EGO)
        intrinsic = np.asarray(calibrated["camera_intrinsic"], dtype=np.float64)
        projection = project_lidar_to_image(
            points_ego, ego_pose_at_lidar, ego_pose_at_camera, extrinsic, intrinsic,
            (obs.width_px, obs.height_px), min_depth_m=BEHIND_CAMERA_EPS_M)
        visible = (projection.depth_m > MIN_DEPTH_M) & projection.in_image
        idx = projection.source_index[visible]
        seen[idx] = True
        if mask is None:
            continue
        uv = projection.uv_px[visible]
        u = np.floor(uv[:, 0]).astype(np.int64)
        v = np.floor(uv[:, 1]).astype(np.int64)
        votes[idx[mask[v, u]]] += 1
    candidate = np.flatnonzero(votes > 0)
    return (np.flatnonzero(seen).astype(np.int64), candidate.astype(np.int64),
            votes[candidate].astype(np.int8))


def compose_marker(upstream_causes, plane_refusals: dict) -> tuple[bool, tuple]:
    """3b/3m/3c's protocol, NOT 4/5's: an accepted degraded upstream and every
    plane refusal ride into THIS stage's marker instead of being laundered."""
    causes = tuple(f"upstream: {c}" for c in upstream_causes)
    for scene, count in sorted(plane_refusals.items()):
        if count:
            causes += (f"{scene}: {count} keyframe(s) with no fittable road plane; "
                       "their points files carry an empty road_point_index",)
    return (bool(causes), causes)


# ---------------------------------------------------------------------------
# On-disk formats
# ---------------------------------------------------------------------------

def _savez_atomic(path: str, payload: dict) -> None:
    # ".tmp.npz", not ".tmp": np.savez_compressed APPENDS ".npz" to any other
    # name and the rename would look for a file that was never created
    # (masks.py's writer, same reasoning).
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    if not os.path.isfile(tmp):
        raise RuntimeError(f"{tmp}: np.savez_compressed did not write the name it was given")
    os.replace(tmp, path)


def write_road_masks_npz(path: str, per_channel: dict) -> None:
    """Byte-for-byte Stage 4's layout — ONE mask per channel — so Stage 5's
    MaskFile reads it unchanged."""
    payload = {
        "__width_px__": np.array([IMAGE_WIDTH_PX], dtype=np.int32),
        "__height_px__": np.array([IMAGE_HEIGHT_PX], dtype=np.int32),
        "__bit_packed__": np.array([1], dtype=np.int8),
    }
    for channel, mask in per_channel.items():
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX):
            raise ValueError(f"{channel}: mask is {mask.shape}, contract is "
                             f"{(IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX)}")
        payload[channel] = np.packbits(mask[None].astype(np.uint8), axis=-1)
    _savez_atomic(path, payload)


def write_points_npz(path: str, *, road_point_index, n_cameras_road, seen_point_index,
                     n_points_raw: int, lidar_sample_data_token: str) -> None:
    _savez_atomic(path, {
        "road_point_index": np.asarray(road_point_index, dtype=np.int32),
        "n_cameras_road": np.asarray(n_cameras_road, dtype=np.int8),
        "seen_point_index": np.asarray(seen_point_index, dtype=np.int32),
        "__n_points_raw__": np.array([n_points_raw], dtype=np.int32),
        "__lidar_sample_data_token__": np.array(lidar_sample_data_token),
        "__frame__": np.array("ego"),
        "__basis__": np.array(INDEX_BASIS),
    })


# ---------------------------------------------------------------------------
# The production segmenter (GPU; unit tests inject a fake instead)
# ---------------------------------------------------------------------------

class Sam3TextSegmenter:
    """transformers' Sam3 image model driven by one text prompt per image.

    The spike's route (handover 2026-09-01 §6), with its two recorded gotchas:
    the cached facebook/sam3 snapshot ships only the VIDEO processor config, so
    the image processor + tokenizer pair is built by hand; and the repo is
    gated, so HF_TOKEN must reach from ~/.cache/huggingface/token explicitly.
    """

    def __init__(self, *, model_id: str = "facebook/sam3",
                 confidence_threshold: float = DEFAULT_CONFIDENCE,
                 mask_threshold: float = 0.5, prompt: str = ROAD_PROMPT,
                 device: str = "cuda"):
        import torch
        from transformers import AutoTokenizer, Sam3Model
        from transformers.models.sam3 import Sam3ImageProcessor, Sam3Processor

        token_path = os.path.expanduser("~/.cache/huggingface/token")
        if "HF_TOKEN" not in os.environ and os.path.isfile(token_path):
            with open(token_path) as fh:
                os.environ["HF_TOKEN"] = fh.read().strip()

        self.prompt = prompt
        self.confidence_threshold = float(confidence_threshold)
        self.mask_threshold = float(mask_threshold)
        self.device = device
        self._torch = torch
        image_processor = Sam3ImageProcessor()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.processor = Sam3Processor(image_processor=image_processor, tokenizer=tokenizer)
        self.model = Sam3Model.from_pretrained(model_id, dtype=torch.float16).to(device).eval()
        self.checkpoint = {"model_id": model_id,
                           "revision": getattr(self.model.config, "_commit_hash", None),
                           "dtype": "float16"}

    def __call__(self, image):
        torch = self._torch
        inputs = self.processor(images=image, text=self.prompt,
                                return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model(**inputs)
        h, w = image.height, image.width
        res = self.processor.post_process_instance_segmentation(
            out, threshold=self.confidence_threshold,
            mask_threshold=self.mask_threshold, target_sizes=[(h, w)])[0]
        masks = [m.cpu().numpy().astype(bool) for m in res["masks"]]
        scores = [float(s) for s in res["scores"]]
        return masks, scores


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _read_keyframe_rows(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class _CameraView:
    """Duck-typed camera fields for paint_cameras, from a keyframe row dict."""

    def __init__(self, channel: str, blob: dict):
        self.channel = channel
        self.path = blob["path"]
        self.ego_pose_token = blob["ego_pose_token"]
        self.calibrated_sensor_token = blob["calibrated_sensor_token"]
        self.width_px = int(blob.get("width_px", IMAGE_WIDTH_PX))
        self.height_px = int(blob.get("height_px", IMAGE_HEIGHT_PX))


def _load_table(dataroot: str, version: str, name: str) -> dict:
    with open(os.path.join(dataroot, version, name), "r", encoding="utf-8") as fh:
        return {row["token"]: row for row in json.load(fh)}


def _detect_version(dataroot: str) -> str:
    """Fallback for direct run() callers only; main() passes paths.version —
    the real dataroot holds two version dirs and a scan there is ambiguous."""
    versions = sorted(n for n in os.listdir(dataroot)
                      if n.startswith("v") and os.path.isdir(os.path.join(dataroot, n)))
    if len(versions) != 1:
        raise UpstreamRefusal(f"{dataroot}: expected exactly one version dir, found {versions}; "
                              "pass version= (paths.yaml's `version`)")
    return versions[0]


def run(stage1_dir: str, out_dir: str, *, dataroot: str, segmenter,
        version: str | None = None,
        current_fingerprint: str | None = None, accept_degraded: bool = False,
        scenes=None, max_keyframes: int | None = None,
        band_m: float = DEFAULT_BAND_M,
        anchor_radius_m: float = DEFAULT_ANCHOR_RADIUS_M,
        min_anchor_points: int = DEFAULT_MIN_ANCHOR_POINTS,
        max_tilt_deg: float = DEFAULT_MAX_TILT_DEG,
        vram_block: dict | None = None, seed: int = 0) -> int:
    from PIL import Image  # deferred, like 3c

    man1, marker1 = require_upstream(
        stage1_dir, stage_name="Stage 1", module_hint="pipeline.stage1_ingestion.ingest",
        current_fingerprint=current_fingerprint, accept_degraded=accept_degraded)

    if max_keyframes is not None:
        root = os.path.join(out_dir, "scenes")
        if os.path.isdir(root) and any(
                os.path.isdir(os.path.join(root, n)) for n in os.listdir(root)):
            raise UpstreamRefusal(
                f"--max-keyframes refuses to write into {root}, which already holds scene "
                "directories. A bounded run truncates every scene it touches, so pointed at "
                "a COMPLETE stage_road tree it would silently replace it with the first "
                f"{max_keyframes} keyframes under the previous run's marker. "
                "--max-keyframes is a timing probe: give it its own --out-dir "
                "(check.py's --max-rows guard, ported).")

    version = version or _detect_version(dataroot)
    ego_pose_table = _load_table(dataroot, version, "ego_pose.json")
    calibrated_table = _load_table(dataroot, version, "calibrated_sensor.json")

    scenes_root = os.path.join(stage1_dir, "scenes")
    scene_names = sorted(n for n in os.listdir(scenes_root)
                         if os.path.isfile(os.path.join(scenes_root, n, "keyframes.jsonl")))
    if scenes:
        missing = [s for s in scenes if s not in set(scene_names)]
        if missing:
            raise UpstreamRefusal(f"--scenes names {missing}; {scenes_root} holds {scene_names}")
        scene_names = [s for s in scene_names if s in set(scenes)]
    if not scene_names:
        raise UpstreamRefusal(f"{scenes_root} has no scenes with keyframes.jsonl")

    # Every refusal above leaves a previous run's marker intact (rc 2 means
    # nothing written); past this point the old marker must GO before the
    # first byte, or a mid-run crash leaves it standing over a mixed tree
    # (manifest.py's clear_markers contract, followed by every peer stage).
    clear_markers(out_dir)

    totals = {"n_keyframes": 0, "n_images": 0, "n_images_with_road": 0,
              "n_keyframes_plane_ok": 0, "n_road_points": 0, "n_seen_points": 0,
              "n_points_raw": 0}
    channels_seen: set = set()
    plane_refusals: dict = {}
    per_scene: dict = {}
    budget = max_keyframes

    for scene in scene_names:
        t0 = time.monotonic()
        rows = _read_keyframe_rows(os.path.join(scenes_root, scene, "keyframes.jsonl"))
        if budget is not None:
            rows = rows[:budget]
            budget -= len(rows)
        scene_dir = os.path.join(out_dir, "scenes", scene)
        os.makedirs(os.path.join(scene_dir, "masks"), exist_ok=True)
        os.makedirs(os.path.join(scene_dir, "points"), exist_ok=True)
        out_rows = []
        s_counts = {"n_keyframes": 0, "n_road_points": 0, "n_plane_refused": 0}

        for kf in rows:
            token = kf["keyframe_token"]
            cameras = {ch: _CameraView(ch, blob) for ch, blob in kf["cameras"].items()}
            channels_seen.update(cameras)

            per_channel_masks: dict = {}
            cam_rows: dict = {}
            for ch in sorted(cameras):
                obs = cameras[ch]
                with Image.open(os.path.join(dataroot, obs.path)) as im:
                    image = im.convert("RGB")
                masks, scores = segmenter(image)
                union = union_masks(masks)
                fired = union is not None
                per_channel_masks[ch] = union if fired else np.zeros(
                    (obs.height_px, obs.width_px), dtype=bool)
                cam_rows[ch] = {
                    "fired": fired,
                    "n_regions": len(masks),
                    "coverage": float(union.mean()) if fired else 0.0,
                    "score": max(scores) if scores else None,
                }
                totals["n_images"] += 1
                totals["n_images_with_road"] += int(fired)

            mask_path = os.path.join("scenes", scene, "masks", f"{token}.npz")
            write_road_masks_npz(os.path.join(out_dir, mask_path), per_channel_masks)

            raw = read_pcd_bin(os.path.join(dataroot, kf["lidar_path"]))
            n_raw = raw.shape[0]
            t_ego_lidar = Transform.from_nuscenes(
                calibrated_table[kf["lidar_calibrated_sensor_token"]],
                source_frame=LIDAR, parent_frame=EGO)
            points_ego = apply_transform(t_ego_lidar.matrix(), raw[:, :3].astype(np.float64))

            seen, candidate, votes = paint_cameras(
                points_ego, observations=list(cameras.values()),
                road_mask_by_channel=per_channel_masks,
                lidar_ego_pose=ego_pose_table[kf["lidar_ego_pose_token"]],
                ego_pose_table=ego_pose_table, calibrated_table=calibrated_table)

            gate = plane_gate(points_ego, candidate, band_m=band_m,
                              anchor_radius_m=anchor_radius_m,
                              min_anchor_points=min_anchor_points,
                              max_tilt_deg=max_tilt_deg)
            kept = gate["kept_index"]
            kept_votes = votes[np.isin(candidate, kept)] if kept.size else np.zeros(0, np.int8)
            if gate["refusal"] is not None:
                s_counts["n_plane_refused"] += 1
            else:
                totals["n_keyframes_plane_ok"] += 1

            points_path = os.path.join("scenes", scene, "points", f"{token}.npz")
            write_points_npz(os.path.join(out_dir, points_path),
                             road_point_index=kept, n_cameras_road=kept_votes,
                             seen_point_index=seen, n_points_raw=n_raw,
                             lidar_sample_data_token=kf["lidar_sample_data_token"])

            out_rows.append({
                "keyframe_token": token, "scene_token": kf["scene_token"],
                "t_ns": kf["t_ns"], "lidar_sample_data_token": kf["lidar_sample_data_token"],
                "n_points_raw": int(n_raw), "cameras": cam_rows,
                "plane": {k: gate[k] for k in ("plane", "tilt_deg", "n_rejected_off_plane", "refusal")},
                "n_road_points": int(kept.size), "n_seen_points": int(seen.size),
                "mask_path": mask_path, "points_path": points_path,
            })
            s_counts["n_keyframes"] += 1
            s_counts["n_road_points"] += int(kept.size)
            totals["n_keyframes"] += 1
            totals["n_road_points"] += int(kept.size)
            totals["n_seen_points"] += int(seen.size)
            totals["n_points_raw"] += int(n_raw)

        write_jsonl_atomic(os.path.join(scene_dir, "road.jsonl"), out_rows)
        if s_counts["n_plane_refused"]:
            plane_refusals[scene] = s_counts["n_plane_refused"]
        s_counts["seconds"] = round(time.monotonic() - t0, 1)
        per_scene[scene] = s_counts
        print(f"{STAGE}: {scene}: {s_counts['n_keyframes']} keyframes, "
              f"{s_counts['n_road_points']} road points, "
              f"{s_counts['n_plane_refused']} plane refusals, {s_counts['seconds']}s",
              flush=True)
        if budget is not None and budget <= 0:
            break

    manifest = {
        "spec": STAGE_SPEC, "stage": STAGE, "provider": PROVIDER,
        "score_semantics": ("SAM 3 grounding score; the row's `score` is the MAX over the "
                            "regions unioned into that camera's road mask, not a calibrated "
                            "road probability"),
        "upstream": {"metadata_fingerprint": marker1.fingerprint,
                     "stage1": {"dir": os.path.realpath(stage1_dir), "spec": man1.get("spec"),
                                "degraded": marker1.degraded,
                                "degraded_causes": list(marker1.causes)},
                     "accepted_degraded_upstream": accept_degraded},
        "image_size_px": [IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX],
        "prompt": ROAD_PROMPT,
        "prompt_rationale": ("'road' returned zero instances on 11/60 spike images; an empty "
                             "mask downstream reads as 'no road here' when it means 'the "
                             "prompt missed'"),
        "checkpoint": getattr(segmenter, "checkpoint",
                              {"provider": type(segmenter).__name__}),
        "confidence_threshold": getattr(segmenter, "confidence_threshold", None),
        "cameras": sorted(channels_seen),
        "index_basis": INDEX_BASIS,
        "cloud_source": ("dataroot samples/LIDAR_TOP/*.pcd.bin (RAW); Stage 1's filtered "
                         "single_sweep is deliberately NOT consumed — 47.9% of it is road "
                         "band only because the Stage 1 ground filter is broken"),
        "plane_gate": {"anchor": f"median z of candidates within {anchor_radius_m} m",
                       "fit": "pipeline.stage1_ingestion.ingest._least_squares_plane on "
                              "|z-anchor| <= 0.5",
                       "band_m": band_m, "band_m_home": "ingest.py ground_band_m (inherited, untuned here)",
                       "min_anchor_points": min_anchor_points, "max_tilt_deg": max_tilt_deg},
        "vram": vram_block,
        "global_seed": seed,
        "totals": totals,
        "scenes": per_scene,
        "known_gaps": [
            "labels only LIDAR_TOP; ZED_FRONT/ZED_BACK carry no road label",
            "0 conflates 'seen and not road' with 'never observed'",
            "6 of 8 camera pitches are assumed, not measured",
            f"prompt validated on 60 daytime images of one scene ({ROAD_PROMPT!r})",
        ],
    }
    if max_keyframes is not None:
        manifest["partial"] = True
        manifest["max_keyframes"] = int(max_keyframes)
    write_json_atomic(os.path.join(out_dir, "run_manifest.json"), manifest)

    degraded, causes = compose_marker(marker1.causes, plane_refusals)
    if max_keyframes is None:
        write_marker(out_dir, marker1.fingerprint, degraded=degraded, causes=causes)
    else:
        print(f"{STAGE}: --max-keyframes {max_keyframes}: PARTIAL tree, no marker written",
              flush=True)
    print(f"{STAGE}: {totals['n_keyframes']} keyframes, {totals['n_images']} images, "
          f"{totals['n_images_with_road']} with road, {totals['n_road_points']} road points, "
          f"{totals['n_seen_points']} seen points", flush=True)
    return 1 if degraded else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage1-dir", default=None,
                        help="default: <work_root>/stage1_ingestion")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--paths",
                        default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--scenes", nargs="+", default=None)
    parser.add_argument("--max-keyframes", type=int, default=None,
                        help="timing probe: PARTIAL tree, no marker written")
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--band-m", type=float, default=DEFAULT_BAND_M)
    parser.add_argument("--accept-degraded-upstream", action="store_true")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    from pipeline.common.paths import assert_dataroot_read_only, load_paths, metadata_fingerprint
    paths = load_paths(args.paths)
    assert_dataroot_read_only(paths, args.out_dir)
    stage1_dir = args.stage1_dir or os.path.join(paths.work_root, "stage1_ingestion")

    try:
        # Refusals before the model load, 3c's ordering: a stale tree or a
        # degraded upstream must not cost a checkpoint load first.
        require_upstream(stage1_dir, stage_name="Stage 1",
                         module_hint="pipeline.stage1_ingestion.ingest",
                         current_fingerprint=metadata_fingerprint(paths),
                         accept_degraded=args.accept_degraded_upstream)
        if not args.allow_shared_gpu:
            from pipeline.stage3c_check.check import assert_gpu_exclusive
            assert_gpu_exclusive()
        import torch
        from pipeline.common.model_interfaces import apply_vram_cap
        vram_block = apply_vram_cap(torch, args.device)
        segmenter = Sam3TextSegmenter(confidence_threshold=args.confidence_threshold,
                                      device=args.device)
        return run(stage1_dir, args.out_dir, dataroot=paths.dataroot, segmenter=segmenter,
                   version=paths.version,
                   current_fingerprint=metadata_fingerprint(paths),
                   accept_degraded=args.accept_degraded_upstream,
                   scenes=args.scenes, max_keyframes=args.max_keyframes,
                   band_m=args.band_m, vram_block=vram_block)
    except (UpstreamRefusal, RuntimeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
