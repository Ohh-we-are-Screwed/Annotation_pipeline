# RSUD20K YOLO11x Fine-Tune (Stage 3 arm B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a fine-tuned YOLO11x checkpoint that detects `rickshaw` and `cng` (RSUD20K's "auto rickshaw") at the best achievable accuracy, to serve as arm B of the two-arm Stage 3 proposal design.

**Architecture:** Train on **all 13 RSUD20K classes**, ship a **2-class filtered subset** at inference. Training on the full label set is strictly better than training on 2 classes: the model learns the `private car` vs `cng` and `person` vs `rickshaw` decision boundaries explicitly — exactly the confusions that motivated arm B — instead of collapsing all 11 other classes into undifferentiated background. It also requires **zero label remapping**, since RSUD20K already ships YOLO-format `images/{split}` + `labels/{split}` directories. Two training runs at different input resolutions are run and the winner picked by validation mAP; compute is not a constraint.

**Tech Stack:** ultralytics 8.4.120 · torch 2.5.1+cu124 · RTX 4090 (24 GB) · Python `/home/mt/miniconda3/envs/ano_pipe/bin/python` · Kaggle CLI

**Spec:** No written spec file. This plan implements the two-arm Stage 3 design agreed in conversation on 2026-08-26, grounded in `docs/GAP_ANALYSIS.md` §3 (G3), `docs/comprehensive.md` §7.2 (gate S1), and README §7.3 (decision C25). Evidence basis: RSUD20K (arXiv:2401.07322) Table 2 supervised baselines and Table 3 zero-shot-annotator results.

---

## EXECUTION STATUS AND CORRECTIONS (2026-08-26, after Tasks 1–3)

Tasks 1–3 are done and `r1280` is training. Four things in the plan below were
**wrong** and are corrected here; the original text is left intact so the record
of what changed is auditable.

1. **`multi_scale=True` is a critical bug, not a setting.** In ultralytics
   8.4.x `multi_scale` is a *fraction* (`cfg/default.yaml:40`) in
   `CFG_FRACTION_KEYS`, validated to `[0.0, 1.0]`. Python bools subclass int, so
   `True` passes validation **silently** as `1.0` — the top of the legal range —
   and `trainer.py:655` computes `max_imgsz = ceil(imgsz * (1 + 1.0) / stride) *
   stride`, training at up to **2× imgsz** (2560 px at imgsz=1280). Now `0.0`,
   with a fraction-validated CLI flag that rejects a bool.

2. **`batch=8`/`batch=4` was far too small.** Measured capacity on this 4090 at
   ~85% of 24 GB: imgsz 1280 → 18, 1600 → 11, 1920 → 7. `nbs=64` gradient
   accumulation fixes gradient noise, LR scaling and weight decay, but
   **BatchNorm still normalises over the micro-batch**. Now 16 / 8 / 6.

3. **`cache='disk'` would have written ~119 GB** of uncompressed `.npy` beside
   the images (pre-flight demands ~178 GB free against 288 GB), and still
   re-runs `cv2.resize` per load. Now `False`.

4. **"train 18,762 images / 118,810 instances" is a wrong pairing.** 18,762 is
   the *pre-annotation sampled frame pool* (4,000 drawn for manual annotation +
   14,762 sent to pseudo-labelling). The released training split is **18,681** =
   3,985 human + 14,696 machine, after 81 images were dropped (15 with no target
   objects, 66 with null predictions). The 118,810 instances belong to the
   18,681. Verified against disk and against the upstream `split-names.csv`.

Also changed, as improvements rather than corrections:

- **Class index 6 renamed `private car` → `car`.** `cls_remap: True` is an
  8.4.x default that warm-starts pretrained head rows by class-name match; this
  takes the number of COCO-matching classes from 5 to 6.
- **`optimizer` is named explicitly** (`SGD`). At `epochs>=300`, `'auto'`
  resolves to MuSGD and silently ignores any `lr0`/`momentum` passed.
- **`cls_pw=0.5`** added for the 69× class imbalance — then **restricted to
  unfiltered runs only** (review pass 2): `set_class_weights()` clamps the
  filtered classes' zero counts to 1, so with `classes=[0,1,3,6,7]` the
  mean-normalisation hands the TRAINED classes weights of 0.009–0.015 and the
  phantoms 1.62 — a silent ~100× cut to classification gradient
  (`loss.py:439` multiplies the whole per-class BCE). `train.py` now defaults
  cls_pw to 0.0 whenever a class filter is active and REFUSES an explicit
  `--cls-pw >0` + filter combination. Within the 5-class subset the imbalance
  is only 2.6×, so nothing of value is lost.
- **`deterministic=False`** for accuracy runs; it only forces slower kernels and
  does not guarantee bit-reproducibility anyway (ultralytics sets
  `warn_only=True`). `seed=0` retained.
- **New: `scripts/label_provenance.py` + `artifacts/split-names.csv`.** Task 1
  concluded the human/machine split was unrecoverable; that is true of the
  *download* but false of the *upstream repo*, which ships a per-image manifest.
  Reconciled: 22,884 human / 95,926 machine instances, 80.7% machine.

**Class-subset decision (user-directed, then simplified).** Arm B trains 5 of
the 13 classes — person, rickshaw, cng, car, motorcycle (85.0% of boxes; the
confusable classes stay labelled so the model learns `car`, not "not-rickshaw")
— and ships rickshaw + cng only. A derived 5-class dataset tree was built for
this and then **deleted**: ultralytics' train-time `classes=` argument does the
same filtering in memory (verified in 8.4.120 source and by count — val 6,374/
7,385 boxes, 6 negatives, all 1,004 images kept, ids unmapped). `train.py`
defaults to `--classes 0 1 3 6 7`; `evaluate.py` reads the filter back from the
run's own `args.yaml`; `predict_armb.py` resolves the ship filter by NAME.

The `--include-pseudo` flag described in Task 3 was **removed**: there is no
separable `pseudo` directory in the Kaggle release, so `images/train` is already
the union and the flag had nothing to switch on.

---

## Global Constraints

- **Build directory:** `/home/mt/Zami/Annotation_pipeline/local_yolox_build` (name is historical; the model is YOLO11x, not YOLOX).
- **Interpreter:** `/home/mt/miniconda3/envs/ano_pipe/bin/python`, always invoked with `PYTHONNOUSERSITE=1`.
- **Dataset license: CC BY-NC 4.0 — research and non-commercial use only.** This is now confirmed, not assumed. Any model trained on RSUD20K inherits this restriction, and so do labels that model produces. Before DhakaScenes ships publicly, this must be recorded as a decision entry in `docs/DECISIONS.md`. **It does not block training or research use.**
- **Class index order is fixed by the dataset** and must never be re-sorted: `0 person, 1 rickshaw, 2 rickshaw van, 3 auto rickshaw, 4 truck, 5 pickup truck, 6 private car, 7 motorcycle, 8 bicycle, 9 bus, 10 micro bus, 11 covered van, 12 human hauler`.
- **The `cng` rename lives only in `data.yaml` `names:`.** Label `.txt` files contain integer indices and are never edited. The internal pipeline phrase stays `an auto rickshaw`; `cng_autorickshaw` is the release-layer name in `configs/release_category_map.yaml`.
- **Determinism:** every training run sets `seed=0, deterministic=True`. Every artifact that leaves this build is sha256-hashed.
- **Never modify the downloaded dataset in place.** `datasets/rsud20k/` is a pristine third-party artifact.

---

## Prerequisite (blocking, human action required)

**Kaggle API credentials are not present on this machine** (`~/.kaggle/kaggle.json` missing, `kaggle` CLI not installed). Nothing in Task 1 can run until this is done, and it cannot be automated:

1. Go to https://www.kaggle.com/settings/account → **API** → **Create New Token**. A `kaggle.json` downloads.
2. Install it:
```bash
mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json && chmod 600 ~/.kaggle/kaggle.json
```
3. Accept the dataset's terms once, by visiting https://www.kaggle.com/datasets/hasibzunair/rsud20k-bangladesh-road-scene-understanding and clicking Download in the browser (you can cancel the browser download; the click is what registers acceptance for the API).

---

## File Structure

```
local_yolox_build/
├── datasets/
│   └── rsud20k/                    # PRISTINE download — never edited
│       ├── images/{train,val,test}/
│       └── labels/{train,val,test}/
├── configs/
│   └── rsud20k_yolo11x.yaml        # ultralytics data config; the ONLY place "cng" appears
├── scripts/
│   ├── fetch_dataset.sh            # Kaggle download + unpack
│   ├── verify_dataset.py           # integrity + label validity + counts vs published figures
│   ├── train.py                    # parameterized training entrypoint
│   └── evaluate.py                 # test-split eval + per-class table vs paper baselines
├── runs/                           # ultralytics output (gitignored)
│   ├── r1280/
│   └── r1600/
├── artifacts/
│   ├── arm_b.pt                    # the winning checkpoint, copied out
│   └── arm_b_provenance.json       # CheckpointSpec fields + training recipe + hashes
└── .gitignore
```

---

### Task 1: Fetch and verify the dataset

**Files:**
- Create: `local_yolox_build/.gitignore`
- Create: `local_yolox_build/scripts/fetch_dataset.sh`
- Create: `local_yolox_build/scripts/verify_dataset.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `datasets/rsud20k/{images,labels}/{train,val,test}/` populated; `verify_dataset.py` exits 0 and prints a per-class instance table later tasks rely on for sanity.

- [ ] **Step 1: Create the gitignore**

```bash
cd /home/mt/Zami/Annotation_pipeline/local_yolox_build
cat > .gitignore <<'EOF'
datasets/
runs/
artifacts/*.pt
*.zip
EOF
```

- [ ] **Step 2: Install the Kaggle CLI**

```bash
PYTHONNOUSERSITE=1 /home/mt/miniconda3/envs/ano_pipe/bin/python -m pip install kaggle
```
Expected: `Successfully installed kaggle-<version>`

- [ ] **Step 3: Write the fetch script**

```bash
mkdir -p /home/mt/Zami/Annotation_pipeline/local_yolox_build/scripts
cat > /home/mt/Zami/Annotation_pipeline/local_yolox_build/scripts/fetch_dataset.sh <<'EOF'
#!/usr/bin/env bash
# Download RSUD20K from Kaggle into datasets/rsud20k/. Idempotent: skips if already present.
set -euo pipefail
BUILD_DIR="/home/mt/Zami/Annotation_pipeline/local_yolox_build"
DEST="${BUILD_DIR}/datasets/rsud20k"
SLUG="hasibzunair/rsud20k-bangladesh-road-scene-understanding"
PY="/home/mt/miniconda3/envs/ano_pipe/bin/python"

if [ -d "${DEST}/images/train" ]; then
    echo "already present at ${DEST}; nothing to do"
    exit 0
fi

mkdir -p "${DEST}"
echo "downloading ${SLUG} ..."
PYTHONNOUSERSITE=1 "${PY}" -m kaggle datasets download -d "${SLUG}" -p "${DEST}" --unzip
echo "download complete"

# The archive may nest everything one level down; flatten so images/ and labels/ sit at DEST root.
if [ ! -d "${DEST}/images" ]; then
    inner=$(find "${DEST}" -maxdepth 2 -type d -name images | head -1)
    if [ -n "${inner}" ]; then
        parent=$(dirname "${inner}")
        echo "flattening from ${parent}"
        mv "${parent}"/* "${DEST}/"
    fi
fi
find "${DEST}" -maxdepth 1 -type d | sort
EOF
chmod +x /home/mt/Zami/Annotation_pipeline/local_yolox_build/scripts/fetch_dataset.sh
```

- [ ] **Step 4: Run the fetch**

Run: `/home/mt/Zami/Annotation_pipeline/local_yolox_build/scripts/fetch_dataset.sh`
Expected: download completes (~several GB), and the final `find` lists `images` and `labels` directories.

If it fails with `401 Unauthorized`, the Prerequisite above was not completed. If it fails with `403 Forbidden`, the dataset terms have not been accepted in the browser.

- [ ] **Step 5: Write the verification script**

This is the integrity gate. It asserts the download matches the figures published in the paper, so a partial or re-versioned download fails loudly instead of silently training on less data.

```bash
cat > /home/mt/Zami/Annotation_pipeline/local_yolox_build/scripts/verify_dataset.py <<'PYEOF'
"""Verify the RSUD20K download: structure, label validity, and counts vs the paper.

Published figures (arXiv:2401.07322, Table 1 / Sec. 3):
    train 18,762 images / 118,810 instances
    val    1,004 images /   7,385 instances
    test     649 images /   3,805 instances
A mismatch is not automatically fatal (the Kaggle release may be a refined
revision), but it is REPORTED, because training on a silently different corpus
than the one you cite is the failure this script exists to prevent.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/home/mt/Zami/Annotation_pipeline/local_yolox_build/datasets/rsud20k")
SPLITS = ("train", "val", "test")
IMG_EXT = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
CLASSES = [
    "person", "rickshaw", "rickshaw van", "auto rickshaw", "truck",
    "pickup truck", "private car", "motorcycle", "bicycle", "bus",
    "micro bus", "covered van", "human hauler",
]
PUBLISHED = {"train": (18762, 118810), "val": (1004, 7385), "test": (649, 3805)}


def main() -> int:
    problems: list[str] = []
    grand = Counter()

    for split in SPLITS:
        img_dir, lbl_dir = ROOT / "images" / split, ROOT / "labels" / split
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            problems.append(f"{split}: missing {img_dir if not img_dir.is_dir() else lbl_dir}")
            continue

        images = sorted(p for p in img_dir.iterdir() if p.suffix in IMG_EXT)
        counts, n_inst, orphan_labels, missing_labels = Counter(), 0, 0, 0

        stems = {p.stem for p in images}
        for lbl in lbl_dir.glob("*.txt"):
            if lbl.stem not in stems:
                orphan_labels += 1

        for img in images:
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                missing_labels += 1
                continue
            for lineno, line in enumerate(lbl.read_text().splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    problems.append(f"{lbl}:{lineno}: expected 5 fields, got {len(parts)}")
                    continue
                cid = int(float(parts[0]))
                if not 0 <= cid < len(CLASSES):
                    problems.append(f"{lbl}:{lineno}: class id {cid} outside 0..{len(CLASSES)-1}")
                    continue
                xywh = [float(v) for v in parts[1:]]
                if any(not 0.0 <= v <= 1.0 for v in xywh):
                    problems.append(f"{lbl}:{lineno}: coords not normalised: {xywh}")
                    continue
                counts[cid] += 1
                n_inst += 1

        grand.update(counts)
        exp_i, exp_n = PUBLISHED[split]
        flag_i = "" if len(images) == exp_i else f"  <-- published {exp_i}"
        flag_n = "" if n_inst == exp_n else f"  <-- published {exp_n}"
        print(f"\n[{split}] images={len(images)}{flag_i}  instances={n_inst}{flag_n}")
        if missing_labels:
            print(f"  images with NO label file (treated as negatives): {missing_labels}")
        if orphan_labels:
            problems.append(f"{split}: {orphan_labels} label files with no matching image")

    print("\nper-class instances (all splits):")
    width = max(len(c) for c in CLASSES)
    for cid, name in enumerate(CLASSES):
        marker = "  <== arm B" if cid in (1, 3) else ""
        print(f"  {cid:>2}  {name:<{width}}  {grand[cid]:>7,}{marker}")

    if problems:
        print(f"\nFAIL — {len(problems)} problem(s):")
        for p in problems[:40]:
            print(f"  {p}")
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        return 1
    print("\nOK — structure and labels valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
PYEOF
```

- [ ] **Step 6: Run verification**

Run:
```bash
PYTHONNOUSERSITE=1 /home/mt/miniconda3/envs/ano_pipe/bin/python \
  /home/mt/Zami/Annotation_pipeline/local_yolox_build/scripts/verify_dataset.py
```
Expected: `OK — structure and labels valid`, and a per-class table where `rickshaw` and `auto rickshaw` both show substantial counts (they are among the most frequent classes in this dataset).

If image/instance counts differ from the published figures, **stop and record the actual numbers** — they are what you cite from here on, not the paper's.

- [ ] **Step 7: Commit**

```bash
cd /home/mt/Zami/Annotation_pipeline
git add local_yolox_build/.gitignore local_yolox_build/scripts/fetch_dataset.sh local_yolox_build/scripts/verify_dataset.py
git commit -m "feat(arm-b): RSUD20K fetch and integrity verification"
```

---

### Task 2: Author the data config

**Files:**
- Create: `local_yolox_build/configs/rsud20k_yolo11x.yaml`

**Interfaces:**
- Consumes: `datasets/rsud20k/` from Task 1.
- Produces: a data-config path consumed by `train.py` and `evaluate.py` in Tasks 3–5. Class index 1 is `rickshaw`, index 3 is `cng`.

- [ ] **Step 1: Write the config**

The only file in this build where `cng` appears. All 13 classes are declared because all 13 are trained; the filtering to two happens at inference.

```bash
mkdir -p /home/mt/Zami/Annotation_pipeline/local_yolox_build/configs
cat > /home/mt/Zami/Annotation_pipeline/local_yolox_build/configs/rsud20k_yolo11x.yaml <<'EOF'
# RSUD20K -> YOLO11x arm B training config.
#
# ALL 13 CLASSES ARE TRAINED. Arm B ships only indices 1 (rickshaw) and
# 3 (cng) at inference, but training on the full label set is what teaches the
# model the `private car` vs `cng` and `person` vs `rickshaw` boundaries --
# precisely the confusions arm B exists to resolve. Training 2-class would make
# every car and every pedestrian undifferentiated background.
#
# INDEX ORDER IS THE DATASET'S AND MUST NOT BE RE-SORTED: the .txt label files
# carry integers, and re-ordering these names silently relabels all 130K boxes.
#
# `cng` replaces RSUD20K's own "auto rickshaw" spelling at index 3. This is a
# display string baked into the checkpoint's `model.names`; the model never
# reads it. The pipeline-internal phrase remains "an auto rickshaw" (SigLIP
# suggestion prompts and S1 text arms need a term a text encoder knows), and
# the release name is `cng_autorickshaw` in configs/release_category_map.yaml.
#
# license: dataset is CC BY-NC 4.0 -- research / non-commercial only.
# provenance: authored 2026-08-26 for the two-arm Stage 3 design.

path: /home/mt/Zami/Annotation_pipeline/local_yolox_build/datasets/rsud20k
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: rickshaw
  2: rickshaw van
  3: cng
  4: truck
  5: pickup truck
  6: private car
  7: motorcycle
  8: bicycle
  9: bus
  10: micro bus
  11: covered van
  12: human hauler
EOF
```

- [ ] **Step 2: Verify ultralytics parses it and finds the data**

Run:
```bash
PYTHONNOUSERSITE=1 /home/mt/miniconda3/envs/ano_pipe/bin/python -c "
from ultralytics.data.utils import check_det_dataset
d = check_det_dataset('/home/mt/Zami/Annotation_pipeline/local_yolox_build/configs/rsud20k_yolo11x.yaml')
print('nc:', d['nc'])
print('names[1]:', d['names'][1], '| names[3]:', d['names'][3])
print('train:', d['train'])
print('val:', d['val'])
"
```
Expected:
```
nc: 13
names[1]: rickshaw | names[3]: cng
```
plus resolved absolute paths for train and val.

- [ ] **Step 3: Commit**

```bash
cd /home/mt/Zami/Annotation_pipeline
git add local_yolox_build/configs/rsud20k_yolo11x.yaml
git commit -m "feat(arm-b): RSUD20K data config, auto rickshaw renamed cng at index 3"
```

---

### Task 3: Write the training entrypoint

**Files:**
- Create: `local_yolox_build/scripts/train.py`

**Interfaces:**
- Consumes: `configs/rsud20k_yolo11x.yaml` from Task 2.
- Produces: CLI `python scripts/train.py --imgsz <int> --name <str>`, writing to `runs/<name>/weights/best.pt`. Tasks 4 and 5 read that path.

- [ ] **Step 1: Write the script**

Hyperparameters chosen for accuracy, not speed, per the stated goal. Rationale is in the docstring so the choices are auditable rather than folklore.

```bash
cat > /home/mt/Zami/Annotation_pipeline/local_yolox_build/scripts/train.py <<'PYEOF'
"""Fine-tune YOLO11x on RSUD20K. Accuracy-first; compute is not a constraint.

Choices and why:

  pretrained yolo11x.pt  COCO init. RSUD20K's own baselines (YOLOv6-L 73.7 mAP,
                         YOLOv8-L 70.4) are the numbers to beat; YOLO11x is the
                         same lineage, one generation on, and the largest tier.

  imgsz                  Swept, not guessed. RSUD20K is 1920x1080 and Stage 3
                         feeds 1600x900 -- identical 16:9 aspect. imgsz=1600
                         reproduces deployment object scale 1:1; imgsz=1280 is
                         the well-trodden setting with healthier BatchNorm
                         statistics at a larger batch. Run both, pick by val
                         mAP50-95 (Task 4).

  epochs=300 patience=60 Long schedule with early stop. 18.7K images fine-tuning
                         13 classes converges well before 300; patience decides.

  cos_lr=True            Cosine decay outperforms the default linear ramp-down
                         on long schedules.

  close_mosaic=20        Mosaic is a strong regulariser but distorts object
                         statistics; disabling it for the final 20 epochs lets
                         the model settle on real image composition. Standard
                         practice and consistently worth ~0.5-1 mAP.

  mixup=0.15             Helps large-capacity models on cluttered scenes.
                         copy_paste is NOT set: it needs segmentation masks and
                         RSUD20K is bbox-only.

  multi_scale=True       Scale robustness matters because the deployment rig is
                         not the dashcam this data came from.

  seed=0 deterministic   The pipeline treats determinism as a contract; a
                         training run that cannot be reproduced cannot be cited.

  cache='disk'           18.7K x 1920x1080 will not fit in RAM. Disk cache still
                         removes the JPEG decode from the hot loop.
"""
from __future__ import annotations

import argparse
from pathlib import Path

BUILD = Path("/home/mt/Zami/Annotation_pipeline/local_yolox_build")
DATA = BUILD / "configs" / "rsud20k_yolo11x.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, required=True, help="1280 or 1600")
    ap.add_argument("--batch", type=int, default=None, help="default: 8 at 1280, 4 at 1600")
    ap.add_argument("--name", type=str, required=True, help="run name under runs/")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    batch = args.batch if args.batch is not None else (8 if args.imgsz <= 1280 else 4)

    from ultralytics import YOLO

    model = YOLO("yolo11x.pt")  # downloads COCO weights on first use
    model.train(
        data=str(DATA),
        project=str(BUILD / "runs"),
        name=args.name,
        exist_ok=False,
        # --- schedule ---
        epochs=args.epochs,
        patience=60,
        cos_lr=True,
        close_mosaic=20,
        # --- capacity / input ---
        imgsz=args.imgsz,
        batch=batch,
        multi_scale=True,
        # --- augmentation ---
        mixup=0.15,
        # --- reproducibility ---
        seed=0,
        deterministic=True,
        # --- runtime ---
        device=0,
        workers=8,
        cache="disk",
        amp=True,
        val=True,
        plots=True,
    )
    print(f"\nbest weights: {BUILD / 'runs' / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
PYEOF
```

- [ ] **Step 2: Smoke-test the wiring with a 1-epoch run**

Proves the data pipeline, GPU, and checkpoint download all work before committing to a long run.

Run:
```bash
cd /home/mt/Zami/Annotation_pipeline/local_yolox_build
PYTHONNOUSERSITE=1 /home/mt/miniconda3/envs/ano_pipe/bin/python scripts/train.py \
  --imgsz 1280 --epochs 1 --name smoke
```
Expected: one epoch completes, a val pass runs and prints a per-class table including `rickshaw` and `cng`, and `runs/smoke/weights/best.pt` exists. mAP after one epoch will be poor — that is fine; this step tests plumbing only.

- [ ] **Step 3: Delete the smoke run**

```bash
rm -rf /home/mt/Zami/Annotation_pipeline/local_yolox_build/runs/smoke
```

- [ ] **Step 4: Commit**

```bash
cd /home/mt/Zami/Annotation_pipeline
git add local_yolox_build/scripts/train.py
git commit -m "feat(arm-b): YOLO11x training entrypoint, accuracy-first hyperparameters"
```

---

### Task 4: Run both resolutions and pick the winner

**Files:**
- Modify: none (runs only)
- Create: `local_yolox_build/runs/r1280/`, `local_yolox_build/runs/r1600/` (generated)

**Interfaces:**
- Consumes: `scripts/train.py` from Task 3.
- Produces: `runs/r1280/weights/best.pt` and `runs/r1600/weights/best.pt`; the winner's path feeds Task 5.

- [ ] **Step 1: Launch the 1280 run**

Long-running. Run it in the background and let it complete.

```bash
cd /home/mt/Zami/Annotation_pipeline/local_yolox_build
PYTHONNOUSERSITE=1 /home/mt/miniconda3/envs/ano_pipe/bin/python scripts/train.py \
  --imgsz 1280 --name r1280 2>&1 | tee runs_r1280.log
```
Expected: training proceeds; `runs/r1280/weights/best.pt` written at completion.

- [ ] **Step 2: Launch the 1600 run**

```bash
cd /home/mt/Zami/Annotation_pipeline/local_yolox_build
PYTHONNOUSERSITE=1 /home/mt/miniconda3/envs/ano_pipe/bin/python scripts/train.py \
  --imgsz 1600 --name r1600 2>&1 | tee runs_r1600.log
```
Expected: same, at `runs/r1600/weights/best.pt`. If this OOMs, re-run with `--batch 2`.

- [ ] **Step 3: Compare on the validation split**

```bash
cd /home/mt/Zami/Annotation_pipeline/local_yolox_build
for r in r1280 r1600; do
  echo "=== $r ==="
  PYTHONNOUSERSITE=1 /home/mt/miniconda3/envs/ano_pipe/bin/python -c "
import csv, sys
rows = list(csv.DictReader(open('runs/$r/results.csv')))
key = [k for k in rows[0] if 'mAP50-95' in k][0]
best = max(rows, key=lambda r: float(r[key]))
print('  epochs run :', len(rows))
print('  best mAP50-95:', round(float(best[key]), 4), 'at epoch', best[[k for k in best if 'epoch' in k][0]].strip())
"
done
```
Expected: two numbers. **The larger `mAP50-95` wins.** Record both — the comparison is a reportable result, not just a selection step.

- [ ] **Step 4: Record the decision**

Append the winner and both numbers to the plan's execution notes or a scratch file; Task 5 needs the winning run name.

---

### Task 5: Evaluate on the test split and export arm B

**Files:**
- Create: `local_yolox_build/scripts/evaluate.py`
- Create: `local_yolox_build/artifacts/arm_b.pt` (generated)
- Create: `local_yolox_build/artifacts/arm_b_provenance.json` (generated)

**Interfaces:**
- Consumes: the winning `runs/<name>/weights/best.pt` from Task 4.
- Produces: `artifacts/arm_b.pt` (the checkpoint Stage 3 arm B loads) and `artifacts/arm_b_provenance.json` carrying the fields `CheckpointSpec` requires — `model_id`, `revision`, `sha256`, `provenance`, plus the training recipe.

- [ ] **Step 1: Write the evaluation script**

```bash
cat > /home/mt/Zami/Annotation_pipeline/local_yolox_build/scripts/evaluate.py <<'PYEOF'
"""Evaluate a trained arm-B checkpoint on the RSUD20K test split.

Prints per-class AP50-95 next to the published YOLOv6-L / YOLOv8-L figures from
arXiv:2401.07322 Table 2, so the run is positioned against the literature rather
than reported in isolation. The two rows that matter for arm B are `rickshaw`
and `cng` (the paper's "auto rickshaw").
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BUILD = Path("/home/mt/Zami/Annotation_pipeline/local_yolox_build")
DATA = BUILD / "configs" / "rsud20k_yolo11x.yaml"

# arXiv:2401.07322 Table 2, mAP (%). None = not extracted from the paper.
PAPER = {
    "rickshaw": {"YOLOv6-L": 86.6, "YOLOv8-L": 88.3},
    "cng": {"YOLOv6-L": 88.4, "YOLOv8-L": 88.5},
    "rickshaw van": {"YOLOv6-L": 54.0, "YOLOv8-L": 47.4},
    "human hauler": {"YOLOv6-L": 79.0, "YOLOv8-L": 67.1},
}
PAPER_OVERALL = {"YOLOv6-L": 73.7, "YOLOv8-L": 70.4}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path to best.pt")
    ap.add_argument("--imgsz", type=int, required=True, help="imgsz the winner was trained at")
    ap.add_argument("--export", action="store_true", help="copy to artifacts/ and write provenance")
    args = ap.parse_args()

    from ultralytics import YOLO

    weights = Path(args.weights)
    model = YOLO(str(weights))
    metrics = model.val(data=str(DATA), split="test", imgsz=args.imgsz, device=0, plots=True)

    names = model.names
    per_class = {}
    for idx, cid in enumerate(metrics.box.ap_class_index):
        per_class[names[int(cid)]] = round(float(metrics.box.maps[int(cid)]) * 100, 1)

    print("\nper-class AP50-95 on the RSUD20K test split (%):")
    print(f"  {'class':<15}{'this run':>10}{'YOLOv6-L':>10}{'YOLOv8-L':>10}   (paper Table 2)")
    for name, ap_val in sorted(per_class.items(), key=lambda kv: -kv[1]):
        ref = PAPER.get(name, {})
        v6 = f"{ref['YOLOv6-L']:>10.1f}" if "YOLOv6-L" in ref else f"{'-':>10}"
        v8 = f"{ref['YOLOv8-L']:>10.1f}" if "YOLOv8-L" in ref else f"{'-':>10}"
        star = "  <== arm B" if name in ("rickshaw", "cng") else ""
        print(f"  {name:<15}{ap_val:>10.1f}{v6}{v8}{star}")
    overall = round(float(metrics.box.map) * 100, 1)
    print(f"  {'ALL':<15}{overall:>10.1f}{PAPER_OVERALL['YOLOv6-L']:>10.1f}{PAPER_OVERALL['YOLOv8-L']:>10.1f}")

    if not args.export:
        return

    art = BUILD / "artifacts"
    art.mkdir(exist_ok=True)
    dest = art / "arm_b.pt"
    dest.write_bytes(weights.read_bytes())
    digest = sha256_of(dest)

    provenance = {
        "role": "proposal_2d",
        "provider": "yolo11x_rsud20k_armb",
        "model_id": "dhakascenes/yolo11x-rsud20k-armb",
        "revision": weights.parent.parent.name,
        "sha256": digest,
        "verified": False,
        "vram_measured_mb": None,
        "provenance": (
            "YOLO11x fine-tuned from COCO weights on RSUD20K (arXiv:2401.07322), "
            "all 13 classes trained, indices 1 (rickshaw) and 3 (cng) shipped. "
            "Dataset licence CC BY-NC 4.0, research / non-commercial only."
        ),
        "options": {
            "base_weights": "yolo11x.pt",
            "dataset": "RSUD20K",
            "dataset_license": "CC BY-NC 4.0",
            "classes_trained": 13,
            "classes_shipped": [1, 3],
            "imgsz": args.imgsz,
            "seed": 0,
            "deterministic": True,
            "test_map50_95": overall,
            "test_per_class_ap": per_class,
        },
    }
    (art / "arm_b_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nexported: {dest}")
    print(f"sha256:   {digest}")
    print(f"provenance: {art / 'arm_b_provenance.json'}")


if __name__ == "__main__":
    main()
PYEOF
```

- [ ] **Step 2: Evaluate and export the winner**

Substitute the winning run name and its imgsz from Task 4 Step 3.

Run:
```bash
cd /home/mt/Zami/Annotation_pipeline/local_yolox_build
PYTHONNOUSERSITE=1 /home/mt/miniconda3/envs/ano_pipe/bin/python scripts/evaluate.py \
  --weights runs/<WINNER>/weights/best.pt --imgsz <WINNER_IMGSZ> --export
```
Expected: the comparison table prints, `artifacts/arm_b.pt` and `artifacts/arm_b_provenance.json` are written, and a sha256 is echoed.

**Success criterion:** `rickshaw` and `cng` AP50-95 at or above the paper's YOLOv6-L figures (86.6 and 88.4). YOLO11x is a later, larger model than either paper baseline, so falling materially short of them means something is wrong with the recipe — investigate before shipping.

- [ ] **Step 3: Verify the inference-time class filter works**

Arm B ships two classes; confirm the filter selects the right indices on a real test image.

Run:
```bash
cd /home/mt/Zami/Annotation_pipeline/local_yolox_build
PYTHONNOUSERSITE=1 /home/mt/miniconda3/envs/ano_pipe/bin/python -c "
from pathlib import Path
from ultralytics import YOLO
m = YOLO('artifacts/arm_b.pt')
print('model.names[1] =', m.names[1], '| model.names[3] =', m.names[3])
img = sorted(Path('datasets/rsud20k/images/test').glob('*'))[0]
r = m.predict(str(img), classes=[1, 3], imgsz=1280, verbose=False)[0]
ids = [int(c) for c in r.boxes.cls]
print('detections:', len(ids), '| class ids present:', sorted(set(ids)))
assert set(ids) <= {1, 3}, f'filter leaked non-arm-B classes: {set(ids)}'
print('OK - filter restricts output to arm B classes')
"
```
Expected: `model.names[1] = rickshaw | model.names[3] = cng`, then `OK - filter restricts output to arm B classes`.

- [ ] **Step 4: Commit**

```bash
cd /home/mt/Zami/Annotation_pipeline
git add local_yolox_build/scripts/evaluate.py local_yolox_build/artifacts/arm_b_provenance.json
git commit -m "feat(arm-b): test-split evaluation vs RSUD20K paper baselines, checkpoint export with provenance"
```

---

## What this plan deliberately does NOT do

Out of scope here; each is its own piece of work once arm B exists:

- **The merge step and class-pair table.** Arm A ∥ arm B arbitration, `stage3_merged/`, the `person`-is-never-suppressed rule.
- **The Stage 3 adapter registration.** `register("yolo11x_rsud20k_armb", PROPOSAL_2D, Yolo11Adapter)` and `rsud20k_to_phrase_dhaka.yaml`.
- **`measure_vram.py` on arm B**, which is what flips `verified` to true and makes the number quotable.
- **The `DECISIONS.md` entry** for the CC BY-NC 4.0 licence constraint on the DhakaScenes release.

---

## Self-Review

**Spec coverage.** Two-arm design → Tasks 3–5 produce arm B, arm A untouched. `cng` rename → Task 2, single location, verified in Task 2 Step 2 and Task 5 Step 3. All-13-train / 2-class-ship → Task 2 config plus Task 5 Step 3 filter check. Provenance discipline → Task 5 Step 2 writes every `CheckpointSpec` field. Determinism → `seed=0, deterministic=True` in `train.py`. Licence → Global Constraints plus `arm_b_provenance.json`. Downstream integration is explicitly deferred above rather than left as a gap.

**Placeholders.** Two intentional substitutions in Task 5 Step 2 (`<WINNER>`, `<WINNER_IMGSZ>`) — values produced by Task 4 Step 3, which is the only way they can be known. No other placeholders.

**Type consistency.** `--imgsz` / `--name` / `--batch` in `train.py` match their invocations in Task 4. `--weights` / `--imgsz` / `--export` in `evaluate.py` match Task 5. Class indices 1 and 3 are consistent across the config, the provenance `classes_shipped`, and the filter assertion. `sha256_of` is defined and used once.
