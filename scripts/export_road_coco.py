"""Export the road-surface masks to COCO 1.0 with RLE segmentation, for CVAT.

One category ("road surface"), one annotation per (keyframe, camera) whose
prompt fired, `iscrowd: 1` + RLE — NOT the instance exporter's polygon path:
`mask_to_polygons` uses cv2.RETR_EXTERNAL, which has no hole semantics, and
COCO's polygon list cannot express a hole at all, so a road with vehicles
standing on it would export with every occluder painted road (measured
+29.4% area over-claim on a 15-occluder synthetic). CVAT 2.72's COCO 1.0
importer turns `iscrowd: 1` + RLE into a native MASK shape.

Publish with the machinery that already exists — its own project, per C33:

    python scripts/cvat_setup.py --export-dir cvat_export_road \\
        --project "DhakaScenes road surface" --task-suffix "(road, machine)" \\
        --run-tag <stage_road manifest mtime>

Run: python -m scripts.export_road_coco --accept-degraded-upstream
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from pipeline.common.manifest import UpstreamRefusal, require_upstream, write_json_atomic  # noqa: E402
from pipeline.common.schemas import IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX  # noqa: E402
from pipeline.stage5_lift.lift import MaskFile  # noqa: E402

CATEGORY = "road surface"


def rle_segmentation(mask: np.ndarray) -> dict:
    """pycocotools RLE as JSON-serialisable {size, counts}."""
    from pycocotools import mask as pm
    encoded = pm.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(s) for s in encoded["size"]],
            "counts": encoded["counts"].decode("ascii")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--stage-road-dir", default=None, help="default: <work_root>/stage_road")
    parser.add_argument("--stage1-dir", default=None, help="default: <work_root>/stage1_ingestion")
    parser.add_argument("--out-root", default=None, help="default: <work_root>/cvat_export_road")
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--accept-degraded-upstream", action="store_true")
    args = parser.parse_args(argv)

    if args.stage_road_dir and args.stage1_dir and args.out_root:
        stage_road, stage1, out_root = args.stage_road_dir, args.stage1_dir, args.out_root
    else:
        from pipeline.common.paths import load_paths
        paths = load_paths(args.paths)
        stage_road = args.stage_road_dir or os.path.join(paths.work_root, "stage_road")
        stage1 = args.stage1_dir or os.path.join(paths.work_root, "stage1_ingestion")
        out_root = args.out_root or os.path.join(paths.work_root, "cvat_export_road")

    try:
        require_upstream(stage_road, stage_name="stage_road",
                         module_hint="pipeline.stage_road.road",
                         accept_degraded=args.accept_degraded_upstream)
    except UpstreamRefusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2

    categories = [{"id": 1, "name": CATEGORY, "supercategory": ""}]
    scene_root = os.path.join(stage_road, "scenes")
    names = sorted(n for n in os.listdir(scene_root)
                   if os.path.isfile(os.path.join(scene_root, n, "road.jsonl")))
    if args.scenes:
        names = [n for n in names if n in args.scenes]

    for scene in names:
        with open(os.path.join(scene_root, scene, "road.jsonl")) as fh:
            road_rows = [json.loads(line) for line in fh if line.strip()]
        with open(os.path.join(stage1, "scenes", scene, "keyframes.jsonl")) as fh:
            keyframes = {r["keyframe_token"]: r for r in map(json.loads, fh) if r}

        images, annotations = [], []
        image_id, ann_id = 0, 0
        for row in road_rows:
            keyframe = keyframes[row["keyframe_token"]]
            masks = MaskFile(os.path.join(stage_road, row["mask_path"]))
            try:
                for channel, cam in sorted(keyframe["cameras"].items()):
                    image_id += 1
                    images.append({"id": image_id, "file_name": cam["path"],
                                   "width": IMAGE_WIDTH_PX, "height": IMAGE_HEIGHT_PX})
                    cam_row = (row.get("cameras") or {}).get(channel) or {}
                    if not cam_row.get("fired"):
                        continue
                    mask = masks.mask(channel, 0)
                    n_px = int(mask.sum())
                    if n_px == 0:
                        continue
                    ys, xs = np.nonzero(mask)
                    x1, y1 = int(xs.min()), int(ys.min())
                    ann_id += 1
                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [x1, y1, int(xs.max()) - x1 + 1, int(ys.max()) - y1 + 1],
                        "area": n_px,
                        # 1, not 0: CVAT's COCO importer maps iscrowd=1 + RLE to
                        # a native MASK shape; 0 forces polygons, which cannot
                        # hold the vehicle-shaped holes.
                        "iscrowd": 1,
                        "segmentation": rle_segmentation(mask),
                        # NOT "score": CVAT's COCO importer pops that exact key
                        # into the shape's confidence field before attributes
                        # are built, so a declared "score" attribute would
                        # forever display its default. And no "source": the
                        # instance schema's select is closed over
                        # yolo/recovered/human — the road project's provenance
                        # is the project itself, single-producer.
                        "attributes": {"sam3_score": cam_row.get("score")},
                    })
            finally:
                masks.close()

        doc = {
            # cvat_setup.py reads info.cvat_label_attributes into the schema of
            # a project it CREATES, beyond its LABEL_ATTRIBUTES — the road
            # score rides under a name the importer does not special-case.
            "info": {"description": f"DhakaScenes road surface, {scene} (machine, C33)",
                     "cvat_label_attributes": [{
                         "name": "sam3_score",
                         "input_type": "number",
                         "mutable": False,
                         "default_value": "0",
                         "values": ["0", "1", "0.01"],
                     }]},
            "licenses": [],
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }
        out_path = os.path.join(out_root, scene, "instances.json")
        write_json_atomic(out_path, doc)
        print(f"  {scene}: {len(images)} images, {len(annotations)} road masks -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
