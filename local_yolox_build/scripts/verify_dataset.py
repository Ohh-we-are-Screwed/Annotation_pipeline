"""Verify the RSUD20K download: structure, label validity, and split composition.

This script DISCOVERS whatever split directories exist rather than assuming a
fixed set. The published figures for this dataset are easy to misread: 18,762
is the PRE-ANNOTATION sampled frame pool, while the released training split is
18,681 = 3,985 human-labelled + 14,696 YOLOv6-M6 pseudo-labelled, already
merged into one `images/train` directory by the Kaggle release. So ~78.7% of
training images carry machine labels and they are NOT separable by filename,
index range, or box density -- a label-provenance fact this project records
rather than absorbs.

So: report what is actually on disk, per split, with per-class instance counts.
Structural problems (malformed labels, out-of-range class ids, unnormalised
coordinates, orphan label files) FAIL. Count mismatches against published
figures are REPORTED, not failed, because the download may legitimately be a
refined revision -- but whatever this prints is what you cite from here on.

Exit codes: 0 = structurally valid, 1 = structural problems found.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

BUILD = Path("/home/mt/Zami/Annotation_pipeline/local_yolox_build")
ROOT = BUILD / "datasets" / "rsud20k"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Index order is the dataset's own and must not be re-sorted. `cng` is our
# rename of index 3; RSUD20K itself spells it "auto rickshaw".
CLASSES = [
    "person", "rickshaw", "rickshaw van", "cng", "truck",
    "pickup truck", "car", "motorcycle", "bicycle", "bus",
    "micro bus", "covered van", "human hauler",
]
ARM_B = {1, 3}  # rickshaw, cng

# What the sources claim, for comparison only.
#
# The 18,762 figure that circulates for "train" is the PRE-ANNOTATION sampled
# frame pool (4,000 drawn for manual labelling + 14,762 sent to pseudo-labelling),
# NOT a released split. Attrition of 15 (no target objects) and 66 (null
# predictions) gives the released 18,681. Never pair 18,762 with an instance
# count -- 118,810 instances belong to the 18,681 that shipped.
PUBLISHED = {
    "train": "18,681 img / 118,810 inst  = 3,985 human + 14,696 MACHINE-labelled, merged",
    "val": "1,004 img / 7,385 inst  (model-seeded, human-refined)",
    "test": "649 img / 3,805 inst  (model-seeded, human-refined)",
    "pseudo": "14,696 img (MACHINE-LABELLED) -- merged into train in the Kaggle release",
}


def scan_split(name: str, img_dir: Path, lbl_dir: Path, problems: list[str]) -> dict:
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    stems = {p.stem for p in images}
    counts: Counter[int] = Counter()
    n_inst = 0
    n_negative = 0
    n_missing_label = 0

    for lbl in lbl_dir.glob("*.txt"):
        if lbl.stem not in stems:
            problems.append(f"{name}: orphan label with no image: {lbl.name}")

    for img in images:
        lbl = lbl_dir / f"{img.stem}.txt"
        if not lbl.exists():
            n_missing_label += 1
            continue
        rows = [ln.strip() for ln in lbl.read_text().splitlines() if ln.strip()]
        if not rows:
            n_negative += 1
            continue
        for lineno, line in enumerate(rows, 1):
            parts = line.split()
            if len(parts) != 5:
                problems.append(f"{lbl}:{lineno}: expected 5 fields, got {len(parts)}")
                continue
            try:
                cid = int(float(parts[0]))
                xywh = [float(v) for v in parts[1:]]
            except ValueError:
                problems.append(f"{lbl}:{lineno}: non-numeric field in {line!r}")
                continue
            if not 0 <= cid < len(CLASSES):
                problems.append(f"{lbl}:{lineno}: class id {cid} outside 0..{len(CLASSES) - 1}")
                continue
            if any(not 0.0 <= v <= 1.0 for v in xywh):
                problems.append(f"{lbl}:{lineno}: coords not normalised: {xywh}")
                continue
            counts[cid] += 1
            n_inst += 1

    return {
        "images": len(images),
        "instances": n_inst,
        "negatives": n_negative,
        "missing_label_files": n_missing_label,
        "per_class": {CLASSES[c]: n for c, n in sorted(counts.items())},
        "_counts": counts,
    }


def main() -> int:
    if not ROOT.is_dir():
        print(f"FAIL - dataset root does not exist: {ROOT}")
        print("       run scripts/fetch_dataset.sh first")
        return 1

    img_root, lbl_root = ROOT / "images", ROOT / "labels"
    if not img_root.is_dir() or not lbl_root.is_dir():
        print(f"FAIL - expected {img_root} and {lbl_root}")
        print(f"       actual top level: {sorted(p.name for p in ROOT.iterdir())}")
        return 1

    splits = sorted(p.name for p in img_root.iterdir() if p.is_dir())
    print(f"discovered image splits: {splits}")

    problems: list[str] = []
    report: dict[str, dict] = {}
    grand: Counter[int] = Counter()

    for split in splits:
        lbl_dir = lbl_root / split
        if not lbl_dir.is_dir():
            print(f"\n[{split}] images present but NO labels/{split} directory - skipping")
            continue
        info = scan_split(split, img_root / split, lbl_dir, problems)
        grand.update(info.pop("_counts"))
        report[split] = info

        print(f"\n[{split}]  images={info['images']:,}  instances={info['instances']:,}")
        if split in PUBLISHED:
            print(f"         published: {PUBLISHED[split]}")
        if info["negatives"]:
            print(f"         empty label files (negatives): {info['negatives']:,}")
        if info["missing_label_files"]:
            print(f"         images with no label file at all: {info['missing_label_files']:,}")

    width = max(len(c) for c in CLASSES)
    total = sum(grand.values())
    print(f"\nper-class instances across all discovered splits ({total:,} total):")
    for cid, name in enumerate(CLASSES):
        marker = "   <== arm B" if cid in ARM_B else ""
        share = (grand[cid] / total * 100) if total else 0.0
        print(f"  {cid:>2}  {name:<{width}}  {grand[cid]:>8,}  {share:>5.1f}%{marker}")

    out = BUILD / "artifacts" / "dataset_report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"splits": report, "totals": {CLASSES[c]: n for c, n in sorted(grand.items())}}, indent=2) + "\n")
    print(f"\nwrote {out}")

    if "pseudo" in report:
        print(
            "\nNOTE: a `pseudo` split is present. Those labels are MACHINE-GENERATED.\n"
            "      Training on them is a label-provenance decision, not a default.\n"
            "      See docs/RUNNING.md 'Stage 3 arm B' and the plan's Task 4."
        )

    if problems:
        print(f"\nFAIL - {len(problems)} structural problem(s):")
        for p in problems[:40]:
            print(f"  {p}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        return 1

    print("\nOK - structure and labels valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
