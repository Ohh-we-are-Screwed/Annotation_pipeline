#!/usr/bin/env python3
"""Export Stage 3 boxes + Stage 4 masks as COCO 1.0 for CVAT pre-annotation.

One COCO JSON per scene under <work_root>/cvat_export/<scene>/instances.json.
`file_name` entries are the dataroot-relative jpg paths (e.g.
`samples/CAM_FRONT/n015-...jpg`), which is what CVAT sees when the nuScenes
dataroot is mounted as its share — create the task from those share files and
the import matches by name.

Encodes the KEPT masks (Stage 4 contest survivors) as polygon segmentation via
cv2 contours; suppressed duplicates are exported as bbox-only annotations with
`attributes.suppressed = true`, so the reviewer sees what the contest removed
without the polygons doubling up. Every annotation additionally carries its
box's Stage 3b provenance — `attributes.source` ("yolo" or "recovered"),
`track_id`, `hops` — so a box no detector ever saw is visible AS a recovered
box in CVAT; on a run without Stage 3b those read "yolo" / null / 0.

Diagnostic/export only: no markers, nothing downstream reads this.

    python -m scripts.export_cvat_coco [--scenes scene-0061]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.manifest import write_json_atomic  # noqa: E402
from pipeline.common.paths import load_paths  # noqa: E402
from pipeline.common.schemas import IMAGE_HEIGHT_PX, IMAGE_WIDTH_PX  # noqa: E402
from pipeline.stage5_lift.lift import MaskFile  # noqa: E402


def mask_to_polygons(mask: np.ndarray, min_area_px: float = 20.0) -> list[list[float]]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area_px:
            continue
        contour = cv2.approxPolyDP(contour, epsilon=1.0, closed=True)
        if len(contour) < 3:
            continue
        polygons.append([float(v) for v in contour.reshape(-1)])
    return polygons


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--taxonomy", default="configs/taxonomy_pilot_nuscenes.yaml",
                        help="taxonomy whose phrases become the COCO categories. A MERGED "
                             "(arm A + arm B) tree carries arm B's superset vocabulary and "
                             "must be exported against it, or every dhaka.* phrase KeyErrors "
                             "here (C28 did not reach the CVAT exporters).")
    args = parser.parse_args(argv)

    paths = load_paths(args.paths)
    stage1 = os.path.join(paths.work_root, "stage1_ingestion")
    stage4 = os.path.join(paths.work_root, "stage4_masks")
    out_root = os.path.join(paths.work_root, "cvat_export")

    with open(args.taxonomy) as fh:
        import yaml
        phrases = sorted(set(yaml.safe_load(fh)["prompt_phrase"].values()))
    category_id = {phrase: i + 1 for i, phrase in enumerate(phrases)}
    categories = [{"id": i, "name": p, "supercategory": ""} for p, i in category_id.items()]

    scene_root = os.path.join(stage4, "scenes")
    names = sorted(n for n in os.listdir(scene_root) if os.path.isdir(os.path.join(scene_root, n)))
    if args.scenes:
        names = [n for n in names if n in args.scenes]

    for scene in names:
        with open(os.path.join(stage4, "scenes", scene, "masks.jsonl")) as fh:
            mask_rows = [json.loads(line) for line in fh if line.strip()]
        with open(os.path.join(stage1, "scenes", scene, "keyframes.jsonl")) as fh:
            keyframes = {r["keyframe_token"]: r for r in map(json.loads, fh) if r}

        images, annotations = [], []
        image_id, ann_id = 0, 0
        for mask_row in mask_rows:
            keyframe = keyframes[mask_row["keyframe_token"]]
            masks = MaskFile(os.path.join(stage4, mask_row["mask_path"]))
            try:
                for channel, cam in sorted(keyframe["cameras"].items()):
                    image_id += 1
                    images.append(
                        {
                            "id": image_id,
                            "file_name": cam["path"],
                            "width": IMAGE_WIDTH_PX,
                            "height": IMAGE_HEIGHT_PX,
                        }
                    )
                    for c in mask_row["candidates"]:
                        if c["channel"] != channel:
                            continue
                        x1, y1, x2, y2 = c["proposal_box_xyxy_px"]
                        # The Stage 3 proposal BOX, always, as a rectangle: the
                        # detector's claim. For kept objects the Stage 4 MASK
                        # follows as a second shape — box vs mask disagreement
                        # is precisely what a human verifier should see.
                        ann_id += 1
                        annotations.append({
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": category_id[c["class_name"]],
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "area": float((x2 - x1) * (y2 - y1)),
                            "iscrowd": 0,
                            "segmentation": [],
                            "attributes": {
                                "score": c["score"],
                                "suppressed": not c["kept"],
                                # Stage 3b provenance, per box. Defaulted rather
                                # than required: a pre-3b masks.jsonl is still a
                                # valid export, and its boxes really are the
                                # detector's own, at zero propagation hops.
                                "source": c.get("box_source", "yolo"),
                                "track_id": c.get("track_id"),
                                "hops": c.get("n_propagated_hops", 0),
                            },
                        })
                        if c["kept"]:
                            mask = masks.mask(channel, c["proposal_index"])
                            ann_id += 1
                            annotations.append({
                                "id": ann_id,
                                "image_id": image_id,
                                "category_id": category_id[c["class_name"]],
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "area": float(c["n_mask_px"]),
                                "iscrowd": 0,
                                "segmentation": mask_to_polygons(mask),
                                "attributes": {
                                    "score": c["score"],
                                    "suppressed": False,
                                    "source": c.get("box_source", "yolo"),
                                    "track_id": c.get("track_id"),
                                    "hops": c.get("n_propagated_hops", 0),
                                },
                            })
            finally:
                masks.close()

        doc = {
            "info": {"description": f"DhakaScenes pilot stages 3+4, {scene} (pre-annotation)"},
            "licenses": [],
            "categories": categories,
            "images": images,
            "annotations": annotations,
        }
        out_path = os.path.join(out_root, scene, "instances.json")
        write_json_atomic(out_path, doc)
        print(f"  {scene}: {len(images)} images, {len(annotations)} annotations -> {out_path}")

    print(f"\nCVAT import: create a task per scene from the share files listed in "
          f"images[].file_name, then Actions > Upload annotations > COCO 1.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
