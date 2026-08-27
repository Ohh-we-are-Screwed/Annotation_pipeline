"""Evaluate a trained arm-B checkpoint on the RSUD20K test split.

Prints per-class AP50-95 beside the published YOLOv6-L / YOLOv8-L figures from
arXiv:2401.07322 Table 2, so the run is positioned against the literature rather
than reported in isolation. The two rows that matter for arm B are `rickshaw`
and `cng` (the paper's "auto rickshaw").

With --export, copies the checkpoint to artifacts/arm_b.pt and writes
artifacts/arm_b_provenance.json carrying every field `CheckpointSpec` requires.
`verified` stays false: only scripts/measure_vram.py may flip it, against a
measured peak on the target card (README section 6.2).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

BUILD = Path("/home/mt/Zami/Annotation_pipeline/local_yolox_build")
DATA = BUILD / "configs" / "rsud20k_yolo11x.yaml"

# arXiv:2401.07322 Table 2, mAP (%). Only the rows extracted from the paper.
PAPER = {
    "rickshaw": {"YOLOv6-L": 86.6, "YOLOv8-L": 88.3},
    "cng": {"YOLOv6-L": 88.4, "YOLOv8-L": 88.5},
    "rickshaw van": {"YOLOv6-L": 54.0, "YOLOv8-L": 47.4},
    "human hauler": {"YOLOv6-L": 79.0, "YOLOv8-L": 67.1},
}
PAPER_OVERALL = {"YOLOv6-L": 73.7, "YOLOv8-L": 70.4}
ARM_B_NAMES = ("rickshaw", "cng")

# CAVEAT ON THE PAPER COLUMNS. Table 2's numbers come from a 13-class model.
# Arm B trains a 5-class subset (via the train-time `classes=` filter), so the
# OVERALL row is not comparable at all (fewer classes to confuse => mechanically
# higher mAP). The PER-CLASS rickshaw and cng figures are the closest thing to a
# fair comparison and even they are only indicative. Treat the paper column as a
# reference point, not a target that can be "beaten".
#
# The evaluation MUST apply the same class filter the run trained with --
# otherwise the 8 untrained classes score ~0 and poison the summary -- so the
# filter is read from the run's own args.yaml (the run's recorded configuration,
# in the spirit of C25: the run's manifest, never the config on disk).


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "-C", "/home/mt/Zami/Annotation_pipeline", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--imgsz", type=int, required=True, help="imgsz the checkpoint was trained at")
    ap.add_argument("--split", default="test", choices=("test", "val"))
    ap.add_argument("--data", default=str(DATA), help=f"dataset config (default: {DATA.name})")
    ap.add_argument("--export", action="store_true", help="copy to artifacts/ and write provenance")
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights).resolve()
    model = YOLO(str(weights))
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = BUILD / data_path

    # The class filter comes from the run's own recorded configuration.
    run_args_path = weights.parent.parent / "args.yaml"
    trained_classes = None
    if run_args_path.exists():
        import yaml
        trained_classes = (yaml.safe_load(run_args_path.read_text()) or {}).get("classes")
    if trained_classes is not None:
        trained_classes = [int(c) for c in trained_classes]
        print(f"class filter from {run_args_path.name}: {trained_classes} "
              f"({', '.join(model.names[c] for c in trained_classes)})")

    metrics = model.val(
        data=str(data_path), split=args.split, imgsz=args.imgsz,
        device=0, plots=True, verbose=False, max_det=1000,
        classes=trained_classes,
    )

    # BOTH IoU regimes are reported, because the paper does not say which it used.
    # arXiv:2401.07322 sec 4.1 says only "We employ mean average precision (mAP),
    # a standard metric in object detection" -- no IoU threshold, in Table 2,
    # Table 3 or Fig. 6. Per-class values in the high 80s make AP@0.5 far more
    # likely than AP@0.5:0.95, but "likely" is not a basis for a success claim.
    # If this run clears the paper on BOTH columns the ambiguity is moot; if it
    # lands between them, the comparison must be reported as unresolved.
    names = model.names
    ap50: dict[str, float] = {}
    ap5095: dict[str, float] = {}
    for pos, cid in enumerate(metrics.box.ap_class_index):
        nm = names[int(cid)]
        ap50[nm] = round(float(metrics.box.ap50[pos]) * 100, 1)
        ap5095[nm] = round(float(metrics.box.ap[pos]) * 100, 1)

    print(f"\nper-class AP on the RSUD20K {args.split} split (%):")
    print(f"  {'class':<15}{'AP50':>8}{'AP50-95':>9}{'YOLOv6-L':>10}{'YOLOv8-L':>10}   (paper Table 2, IoU unstated)")
    for name in sorted(ap50, key=lambda n: -ap50[n]):
        ref = PAPER.get(name, {})
        v6 = f"{ref['YOLOv6-L']:>10.1f}" if "YOLOv6-L" in ref else f"{'-':>10}"
        v8 = f"{ref['YOLOv8-L']:>10.1f}" if "YOLOv8-L" in ref else f"{'-':>10}"
        star = "   <== arm B" if name in ARM_B_NAMES else ""
        print(f"  {name:<15}{ap50[name]:>8.1f}{ap5095[name]:>9.1f}{v6}{v8}{star}")
    m50 = round(float(metrics.box.map50) * 100, 1)
    m5095 = round(float(metrics.box.map) * 100, 1)
    print(f"  {'ALL':<15}{m50:>8.1f}{m5095:>9.1f}"
          f"{PAPER_OVERALL['YOLOv6-L']:>10.1f}{PAPER_OVERALL['YOLOv8-L']:>10.1f}")

    print("\narm B success criterion vs paper YOLOv6-L (verdict on the WORSE of the two IoU regimes):")
    for name in ARM_B_NAMES:
        if name not in ap50:
            print(f"  {name:<12} MISSING from metrics")
            continue
        want = PAPER[name]["YOLOv6-L"]
        lo = min(ap50[name], ap5095[name])
        if lo >= want:
            verdict = "PASS (clears on both IoU regimes)"
        elif max(ap50[name], ap5095[name]) >= want:
            verdict = "UNRESOLVED (clears on AP50 only; paper's IoU unstated)"
        else:
            verdict = "BELOW on both"
        print(f"  {name:<12} AP50={ap50[name]:>5.1f}  AP50-95={ap5095[name]:>5.1f}  vs {want:>5.1f}   {verdict}")

    if not args.export:
        return

    art = BUILD / "artifacts"
    art.mkdir(exist_ok=True)
    dest = art / "arm_b.pt"
    dest.write_bytes(weights.read_bytes())
    digest = sha256_of(dest)

    train_args_path = weights.parent.parent / "args.yaml"
    train_args = train_args_path.read_text() if train_args_path.exists() else ""
    eff_classes = trained_classes if trained_classes is not None else sorted(names)

    provenance = {
        "role": "proposal_2d",
        "provider": "yolo11x_rsud20k_armb",
        "model_id": "dhakascenes/yolo11x-rsud20k-armb",
        "revision": weights.parent.parent.name,
        "sha256": digest,
        "verified": False,
        "vram_measured_mb": None,
        "provenance": (
            "YOLO11x fine-tuned from COCO weights on RSUD20K (arXiv:2401.07322); "
            f"trained on {len(eff_classes)} of {len(names)} classes "
            f"({', '.join(names[i] for i in eff_classes)}) via {data_path.name}"
            f"{' with train-time classes filter' if trained_classes is not None else ''}; "
            f"arm B ships {', '.join(f'{i} ({names[i]})' for i in eff_classes if names[i] in ARM_B_NAMES)}. "
            "Dataset licence CC BY-NC 4.0 -- research / non-commercial only. "
            f"Repo rev {git_rev()}."
        ),
        "options": {
            "base_weights": "yolo11x.pt",
            "dataset": "RSUD20K",
            "dataset_license": "CC BY-NC 4.0",
            # Label provenance, recorded rather than absorbed. `images/train`
            # ships as the union of human and machine labels, inseparable by
            # filename, index range or box density -- so this fraction is a
            # property of the checkpoint, not an optional annotation.
            "label_provenance": {
                "note": (
                    "Fractions below are properties of the RSUD20K corpus as shipped. "
                    "Box counts refer to the FULL 13-class annotation; a run trained "
                    "with a class filter (classes_trained, from the run's args.yaml) "
                    "saw only the label rows of those classes."
                ),
                "training_images_total": 18681,
                "training_images_human": 3985,
                "training_images_machine": 14696,
                "machine_image_fraction": 0.787,
                "training_instances_total": 118810,
                "training_instances_human": 22884,
                "training_instances_machine": 95926,
                "machine_instance_fraction": 0.807,
                "pseudo_label_generator": "YOLOv6-M6 trained on the 3,985 human-labelled RSUD20K images",
                "pseudo_label_filter": "null-prediction images removed; no confidence threshold documented",
                "pseudo_review": "none -- the paper reports no quality metric for the pseudo split",
                "label_generation_depth": "3rd generation (human -> YOLOv6-M6 -> YOLO11x)",
                "eval_split_provenance": "val/test are model-seeded and human-REFINED (48s -> 8s per image), not human-authored",
                "separable_from_download": False,
                "separable_via_manifest": True,
                "manifest": "artifacts/split-names.csv",
                "manifest_sha256": "78b9ed93ef21e595e417c853f11b53c3d39a88626e520ac4d0c9d676b89bfb4a",
                "reconciliation": "artifacts/label_provenance.json (scripts/label_provenance.py)",
            },
            "classes_trained": eff_classes,
            "classes_shipped": [i for i, n in names.items() if n in ARM_B_NAMES],
            "class_names": {int(k): v for k, v in names.items()},
            "data_config": data_path.name,
            "imgsz": args.imgsz,
            "eval_split": args.split,
            f"{args.split}_map50": m50,
            f"{args.split}_map50_95": m5095,
            f"{args.split}_per_class_ap50": ap50,
            f"{args.split}_per_class_ap50_95": ap5095,
            "paper_metric_iou_threshold": "UNSTATED in arXiv:2401.07322 sec 4.1; comparisons report both regimes",
            "ultralytics_train_args": train_args,
        },
    }
    (art / "arm_b_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nexported:   {dest}")
    print(f"sha256:     {digest}")
    print(f"provenance: {art / 'arm_b_provenance.json'}")


if __name__ == "__main__":
    main()
