"""Arm B inference: detect ONLY the classes arm B ships.

Arm B trains on 5 classes but ships 2. The other three -- person, car,
motorcycle -- exist to teach the boundaries the model must not cross
(car<->cng and person<->rickshaw are the confusions arm B exists to resolve,
and motorcycle is the nearest COCO geometry to a three-wheeler). They are
scaffolding: never emitted, because arm A already covers them from COCO and
emitting them here would put the two arms in direct competition on arm A's own
turf, where the vocabulary-authority merge rule does not apply.

The filter is `classes=[...]` on predict(). This module is the single place the
ship list is computed, so the Stage 3 arm B adapter can import it rather than
hardcoding indices that move whenever the class subset changes.

Usage:
    python scripts/predict_armb.py --weights artifacts/arm_b.pt --source <img|dir>
    python scripts/predict_armb.py --weights artifacts/arm_b.pt --self-test
"""
from __future__ import annotations

import argparse
from pathlib import Path

BUILD = Path("/home/mt/Zami/Annotation_pipeline/local_yolox_build")

# The classes arm B is allowed to emit. Names, not indices -- indices shift with
# the class subset, names do not.
SHIP_NAMES = ("rickshaw", "cng")


def ship_indices(names: dict[int, str]) -> list[int]:
    """Map SHIP_NAMES to this checkpoint's own class indices.

    Refuses rather than silently shipping the wrong classes: a checkpoint whose
    `model.names` lacks a shipped name is the wrong checkpoint, and a filter
    built from stale indices would emit `car` boxes labelled `rickshaw`.
    """
    by_name = {v: k for k, v in names.items()}
    missing = [n for n in SHIP_NAMES if n not in by_name]
    if missing:
        raise SystemExit(
            f"checkpoint cannot ship arm B: {missing} absent from model.names={names}"
        )
    return sorted(by_name[n] for n in SHIP_NAMES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(BUILD / "artifacts" / "arm_b.pt"))
    ap.add_argument("--source", help="image, directory, or video")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", action="store_true", help="write annotated images")
    ap.add_argument("--self-test", action="store_true",
                    help="run on a few dataset images and assert nothing else leaks")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    keep = ship_indices(model.names)
    print(f"checkpoint classes : {model.names}")
    print(f"arm B ships        : {keep}  ({', '.join(model.names[i] for i in keep)})")
    print(f"suppressed at infer: {[model.names[i] for i in sorted(model.names) if i not in keep]}")

    if args.self_test:
        src = BUILD / "datasets" / "rsud20k" / "images" / "test"
        images = sorted(src.glob("*.jpg"))[:25]
        if not images:
            raise SystemExit(f"no test images under {src}")
        seen: set[int] = set()
        n_boxes = 0
        for r in model.predict([str(p) for p in images], classes=keep, imgsz=args.imgsz,
                               conf=args.conf, verbose=False):
            ids = [int(c) for c in r.boxes.cls]
            seen.update(ids)
            n_boxes += len(ids)
        leaked = seen - set(keep)
        print(f"\nself-test: {len(images)} images, {n_boxes} boxes, class ids present {sorted(seen)}")
        if leaked:
            raise SystemExit(f"FAIL - filter leaked non-shipped classes: "
                             f"{[model.names[i] for i in sorted(leaked)]}")
        print("OK - only arm B classes were emitted")
        return

    if not args.source:
        raise SystemExit("pass --source, or --self-test")

    results = model.predict(args.source, classes=keep, imgsz=args.imgsz,
                            conf=args.conf, save=args.save, verbose=False)
    total = sum(len(r.boxes) for r in results)
    print(f"\n{len(results)} image(s), {total} boxes emitted (arm B classes only)")


if __name__ == "__main__":
    main()
