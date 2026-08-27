"""Reconcile the on-disk RSUD20K training split against upstream per-image provenance.

The Kaggle release merges the 3,985 human-labelled and 14,696 YOLOv6-M6
pseudo-labelled images into one `images/train` directory, all named `trainN.jpg`.
From the download alone they are indistinguishable -- no filename marker, no
index boundary, no box-density signal (mean boxes 6.32 vs 6.39 either side of
index 3,985).

The upstream repository, however, ships `csv/split-names.csv`: a Filename->Split
manifest for all 20,334 images that separates `train` (human) from `pseudo`
(machine). That file is the audit trail which turns the machine-label fraction
from an asserted number into a verifiable one, so this project keeps a hashed
copy in artifacts/ and reconciles against it rather than trusting the fraction.

Reports per-class instance counts split by label provenance -- which matters
for arm B specifically, since it answers "how many of my rickshaw and cng boxes
were drawn by a person?"

Exit codes: 0 = reconciled, 1 = mismatch between manifest and disk.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

BUILD = Path("/home/mt/Zami/Annotation_pipeline/local_yolox_build")
ROOT = BUILD / "datasets" / "rsud20k"
MANIFEST = BUILD / "artifacts" / "split-names.csv"
MANIFEST_SHA256 = "78b9ed93ef21e595e417c853f11b53c3d39a88626e520ac4d0c9d676b89bfb4a"

CLASSES = [
    "person", "rickshaw", "rickshaw van", "cng", "truck",
    "pickup truck", "car", "motorcycle", "bicycle", "bus",
    "micro bus", "covered van", "human hauler",
]
ARM_B = {1, 3}
# Manifest split name -> label provenance.
PROVENANCE = {"train": "human", "pseudo": "machine", "val": "human", "test": "human"}


def sha256_of(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not MANIFEST.exists():
        print(f"FAIL - missing {MANIFEST}")
        print("       curl -sSL -o artifacts/split-names.csv \\")
        print("         https://raw.githubusercontent.com/hasibzunair/RSUD20K/main/csv/split-names.csv")
        return 1

    got = sha256_of(MANIFEST)
    if got != MANIFEST_SHA256:
        print(f"FAIL - manifest sha256 changed\n  expected {MANIFEST_SHA256}\n  got      {got}")
        print("       upstream revised the manifest; re-verify before trusting this reconciliation")
        return 1
    print(f"manifest sha256 OK ({got[:16]}...)")

    rows = list(csv.DictReader(MANIFEST.open(newline="")))
    stem_split = {r["Filename"].strip(): r["Split"].strip() for r in rows}
    print(f"manifest rows: {len(rows):,}  splits: {dict(Counter(stem_split.values()))}")

    train_img = ROOT / "images" / "train"
    train_lbl = ROOT / "labels" / "train"
    if not train_img.is_dir():
        print(f"FAIL - {train_img} does not exist; run scripts/fetch_dataset.sh")
        return 1

    counts: dict[str, Counter[int]] = {"human": Counter(), "machine": Counter()}
    images: Counter[str] = Counter()
    unmapped: list[str] = []

    for img in sorted(train_img.iterdir()):
        if img.suffix.lower() != ".jpg":
            continue
        split = stem_split.get(img.stem)
        if split is None:
            unmapped.append(img.stem)
            continue
        prov = PROVENANCE[split]
        images[prov] += 1
        lbl = train_lbl / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        for line in lbl.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            counts[prov][int(float(line.split()[0]))] += 1

    n_h, n_m = images["human"], images["machine"]
    total_img = n_h + n_m
    print(f"\non-disk images/train: {total_img:,}")
    print(f"  human-labelled  : {n_h:,}")
    print(f"  machine-labelled: {n_m:,}   ({n_m / total_img * 100:.1f}% of training images)")
    if unmapped:
        print(f"  NOT IN MANIFEST : {len(unmapped):,}  e.g. {unmapped[:5]}")

    inst_h, inst_m = sum(counts["human"].values()), sum(counts["machine"].values())
    inst_total = inst_h + inst_m
    print(f"\ninstances: {inst_total:,} total  |  human {inst_h:,}  machine {inst_m:,}"
          f"  ({inst_m / inst_total * 100:.1f}% machine)")
    print(f"boxes/image: human {inst_h / max(n_h, 1):.2f}  machine {inst_m / max(n_m, 1):.2f}")

    width = max(len(c) for c in CLASSES)
    print(f"\nper-class training instances by label provenance:")
    print(f"  {'':>2}  {'class':<{width}}  {'human':>8} {'machine':>9} {'total':>9}  {'%mach':>6}")
    for cid, name in enumerate(CLASSES):
        h, m = counts["human"][cid], counts["machine"][cid]
        t = h + m
        pct = (m / t * 100) if t else 0.0
        marker = "  <== arm B" if cid in ARM_B else ""
        print(f"  {cid:>2}  {name:<{width}}  {h:>8,} {m:>9,} {t:>9,}  {pct:>5.1f}%{marker}")

    out = BUILD / "artifacts" / "label_provenance.json"
    out.write_text(json.dumps({
        "manifest_sha256": MANIFEST_SHA256,
        "manifest_source": "https://raw.githubusercontent.com/hasibzunair/RSUD20K/main/csv/split-names.csv",
        "train_images": {"human": n_h, "machine": n_m, "total": total_img},
        "train_instances": {"human": inst_h, "machine": inst_m, "total": inst_total},
        "machine_image_fraction": round(n_m / total_img, 4) if total_img else None,
        "machine_instance_fraction": round(inst_m / inst_total, 4) if inst_total else None,
        "per_class": {
            CLASSES[c]: {"human": counts["human"][c], "machine": counts["machine"][c]}
            for c in range(len(CLASSES))
        },
        "pseudo_label_generator": "YOLOv6-M6 trained on the 3,985 human-labelled images",
        "pseudo_review": "none -- the paper reports no quality metric for the pseudo split",
    }, indent=2) + "\n")
    print(f"\nwrote {out}")

    if unmapped:
        print(f"\nFAIL - {len(unmapped)} on-disk images absent from the upstream manifest")
        return 1
    print("\nOK - every on-disk training image is provenance-tagged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
