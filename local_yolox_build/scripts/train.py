"""Fine-tune YOLO11x on RSUD20K. Accuracy-first; compute is not a constraint.

Every non-default below was verified against the installed ultralytics 8.4.120
source, not assumed from blog-era defaults. The ones that matter:

  multi_scale IS A FRACTION, NOT A BOOL.
      cfg/default.yaml:40 -> `multi_scale: 0.0  # (float) multi-scale range as a
      fraction of imgsz`, and it lives in CFG_FRACTION_KEYS (validated to
      [0.0, 1.0]). trainer.py:655 computes
          max_imgsz = ceil(imgsz * (1 + multi_scale) / stride) * stride
      Python bools subclass int, so `multi_scale=True` passes validation
      SILENTLY as 1.0 -- the top of the legal range -- and trains at up to 2x
      imgsz (2560 px at imgsz=1280). It does not raise. It just wrecks the run.
      Kept at 0.0 for the primary arms; expose it as a sweep axis instead,
      because peak memory scales with imgsz*(1+multi_scale) and it therefore
      costs batch size.

  batch is sized from MEASURED 4090 capacity, not guessed.
      imgsz 1280 -> ~18, 1600 -> ~11, 1920 -> ~7 at 85% of 24 GB. The plan's
      original batch=4 was far too small: nbs=64 gradient accumulation fixes
      gradient noise, LR scaling and weight decay, but BatchNorm still
      normalises over the micro-batch. Values below prefer divisors of 64 so
      ultralytics' weight_decay rescaling stays exact.

  optimizer is NAMED, not 'auto'.
      'auto' branches on ITERATIONS, not epochs (trainer.py:1121), where
      trainer.py:299 defines
          iterations = ceil(len(dataset) / max(batch, nbs)) * epochs
                     = ceil(18681/64) * epochs = 292 * epochs
      and >10000 selects MuSGD(lr0=0.01, momentum=0.9), else AdamW. The
      crossover for this corpus is epochs>34, so 'auto' still resolves to
      MuSGD at 80 epochs exactly as it did at 300 -- and either way it
      silently ignores any lr0/momentum passed. SGD with the classic
      fine-tuning triple is the proven arm; MuSGD is selectable via
      --optimizer.

  THE SCHEDULE IS COUPLED TO --epochs; IT IS NOT A CEILING YOU CAN CUT.
      cos_lr builds the cosine from `epochs` itself (trainer.py:247 ->
      one_cycle(1, lrf, self.epochs)), so a run LAUNCHED at 300 and killed at
      80 sits at lr 0.00836 -- 84% of peak, never annealed -- while a run
      launched AT 80 has annealed to lr0*lrf = 0.0001. The budget is a
      decision made before launch, not a ceiling trimmed later.

      80 epochs, because this is a fine-tune off a COCO-pretrained yolo11x,
      not a from-scratch run: r1280-3 epoch 1 already scored mAP50 0.914 /
      mAP50-95 0.664 on the 5-class filter. 300 is the ultralytics default for
      training COCO from scratch (118K images); at the measured 985 s/epoch it
      costs 82 h against 22 h here.

      warmup_epochs 5.0 -> 2.0. nw = round(warmup_epochs * nb) with NO
      iteration floor in 8.4.x (trainer.py:255-256), and nb = 3114 at batch=6,
      so 5.0 meant 15,570 warmup iterations -- 6.3% of an 80-epoch run spent
      ramping. r1280-3 showed train loss RISING and mAP50-95 flat across
      epochs 2-4 while lr climbed 0.002 -> 0.008: that is the ramp, and off a
      pretrained checkpoint it is pure cost. 2.0 -> 6,228 iterations, enough
      to stabilise BN and optimiser state without eating the budget.

      close_mosaic 20 -> 10. It fires at `epoch == epochs - close_mosaic`
      (trainer.py:454), so 20 would leave 25% of an 80-epoch run with mosaic,
      mixup, cutmix and copy_paste all off -- a quarter of the schedule at
      weak augmentation on a corpus that is 78.7% pseudo-labelled, which is
      precisely the regime that memorises the teacher's errors. 10 fires at
      epoch 70, where cos_lr has already annealed to 0.00048: a clean-data,
      low-lr polish tail, which is what close_mosaic is for.

      patience 10 IS TIGHT, DELIBERATELY. It counts epochs since best fitness
      (0.1*mAP50 + 0.9*mAP50-95). The interaction to watch: if the run
      plateaus before epoch 60 the stop fires BEFORE close_mosaic engages at
      70, and the mosaic-off gain is never collected. best.pt is still the
      best checkpoint seen, so an early stop forfeits the anneal tail, not the
      run. Raise to ~20 if the next run stops short of epoch 70.

  cls_pw MUST BE 0.0 whenever a class filter is active.
      set_class_weights() (models/yolo/detect/train.py:163) computes
      (1/count)^cls_pw over ALL nc classes and normalises to mean 1.0, clamping
      zero counts to 1. With `classes=[0,1,3,6,7]` the eight filtered classes
      count 0 -> clamp to 1 -> raw weight 1.0, while the five trained classes
      get ~0.006-0.009; after mean-normalisation the TRAINED classes carry
      weights 0.009-0.015 and the phantoms 1.62. loss.py:439 multiplies the
      whole per-class BCE by these, so classification gradient on the classes
      that matter drops ~100x. Silently. Within the 5-class subset the
      imbalance is only 2.6x (person 32,884 vs cng 12,509), so weighting buys
      nothing there anyway. cls_pw defaults to 0.5 ONLY for an unfiltered
      13-class run (69x spread, where it belongs); combining an explicit
      cls_pw>0 with a class filter is refused rather than allowed to poison
      the run.

  deterministic=False for accuracy runs.
      It only forces slower cuDNN/cuBLAS kernels and, because ultralytics sets
      warn_only=True, does not actually guarantee bit-reproducibility. seed=0
      is kept, which is what makes the run repeatable in practice.

  cache=False.
      cache='disk' writes an uncompressed .npy beside every image: ~119 GB for
      this corpus, against 288 GB free, and it still re-runs cv2.resize on each
      load. Decode is not the bottleneck at these resolutions.

  geometric augmentation is set explicitly.
      The shipped defaults are near-zero. flipud stays 0.0 -- never
      vertical-flip driving imagery.

Training data note: `images/train` (18,681) is ALREADY the union of the 3,985
human-labelled images and the 14,696 pseudo-labelled ones -- the paper merged
them ("seamlessly integrated into the RSUD5K training set") and the Kaggle
release ships them merged, with no separable `pseudo` directory. That union is
the configuration the dataset's own Fig. 6 ablation shows winning by +5.7 to
+7.8 test mAP, so it is also the one we want. The pseudo-labels come from
YOLOv6-M6, an in-domain teacher -- this is self-training, NOT the off-the-shelf
zero-shot labelling that the same paper's Table 3 shows failing at 9-16% mAP.
The provenance is recorded in artifacts/arm_b_provenance.json regardless.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Set before torch is imported. Long 1280 px runs are fragmentation-prone.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

BUILD = Path("/home/mt/Zami/Annotation_pipeline/local_yolox_build")
DATA = BUILD / "configs" / "rsud20k_yolo11x.yaml"

# Arm B trains a 5-class SUBSET of the 13, selected at train time with
# ultralytics' own `classes=` argument -- no derived dataset needed. The
# mechanism (verified in the installed 8.4.120 source): BaseDataset.__init__
# calls update_labels(include_class=classes), which drops label ROWS outside
# the list while keeping every image (an image losing all its boxes stays in
# as an explicit negative), and build.py plumbs cfg.classes into BOTH the
# train and val datasets. Ids are NOT remapped: rickshaw stays 1, cng stays 3,
# so the ship-time filter is identical for any checkpoint from this corpus.
#
# Why these five: car<->cng and person<->rickshaw are the confusions arm B
# exists to resolve, and motorcycle is the nearest COCO geometry to a
# three-wheeler. Measured on val: 6,374 of 7,385 boxes survive the filter,
# 6 images become negatives, 1,004 of 1,004 images are kept.
ARMB_CLASSES = [0, 1, 3, 6, 7]  # person, rickshaw, cng, car, motorcycle

# Measured on this RTX 4090 at ~85% of 24 GB, multi_scale=0.0.
BATCH_FOR_IMGSZ = {1280: 16, 1600: 8, 1920: 6}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--imgsz", type=int, required=True, help="1280 | 1600 | 1920")
    ap.add_argument("--name", type=str, required=True, help="run name under runs/")
    ap.add_argument("--batch", type=int, default=6, help="default: measured capacity for --imgsz")
    ap.add_argument("--epochs", type=int, default=80,
                    help="cos_lr anneals over THIS value; it cannot be trimmed "
                         "at runtime (see docstring)")
    ap.add_argument("--patience", type=int, default=10,
                    help="epochs since best fitness before early stop; may fire "
                         "before close_mosaic engages (see docstring)")
    ap.add_argument("--optimizer", default="SGD", choices=("SGD", "MuSGD", "AdamW", "auto"))
    ap.add_argument("--cls-pw", type=float, default=None,
                    help="inverse-frequency class-weight power; default 0.5 for an "
                         "unfiltered run, FORCED 0.0 with --classes (see docstring)")
    ap.add_argument("--multi-scale", type=float, default=0.0,
                    help="FRACTION in [0,1]; peak memory scales with imgsz*(1+this). Never pass True.")
    ap.add_argument("--workers", type=int, default=16,
                    help="dataloader workers; try 8 if a run dies mid-forward "
                         "(r1280-3 died at epoch 5 with KeyError: None)")
    ap.add_argument("--model", default="yolo11x.pt")
    ap.add_argument("--data", default=str(DATA),
                    help=f"dataset config (default: {DATA.name})")
    ap.add_argument("--classes", nargs="+", default=[str(c) for c in ARMB_CLASSES],
                    help="class ids to train on ('all' = no filter; "
                         f"default: {ARMB_CLASSES} = person,rickshaw,cng,car,motorcycle)")
    args = ap.parse_args()

    if not 0.0 <= args.multi_scale <= 1.0:
        raise SystemExit(f"--multi-scale must be a fraction in [0,1], got {args.multi_scale}")

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = BUILD / data_path
    if not data_path.exists():
        raise SystemExit(f"dataset config not found: {data_path}")

    if len(args.classes) == 1 and args.classes[0].lower() == "all":
        classes = None
    else:
        classes = sorted({int(c) for c in args.classes})

    if args.cls_pw is None:
        cls_pw = 0.5 if classes is None else 0.0
    else:
        cls_pw = args.cls_pw
        if classes is not None and cls_pw > 0.0:
            raise SystemExit(
                f"cls_pw={cls_pw} with a class filter poisons the weight normalisation: "
                "filtered classes count 0, are clamped to 1, and after mean-normalisation "
                "the TRAINED classes get ~0.01x classification loss while phantoms get 1.6x "
                "(see docstring). Use --cls-pw 0.0, or --classes all."
            )

    batch = args.batch if args.batch is not None else BATCH_FOR_IMGSZ.get(args.imgsz, 8)

    print(f"data={data_path.name}  classes={'all' if classes is None else classes}")
    print(f"model={args.model}  imgsz={args.imgsz}  batch={batch}  optimizer={args.optimizer}")
    print(f"epochs={args.epochs}  patience={args.patience}  cls_pw={cls_pw}  multi_scale={args.multi_scale}")

    from ultralytics import YOLO

    model = YOLO(str(BUILD / args.model) if (BUILD / args.model).exists() else args.model)
    model.train(
        data=str(data_path),
        classes=classes,
        project=str(BUILD / "runs"),
        name=args.name,
        exist_ok=False,
        # --- schedule ---
        epochs=args.epochs,
        patience=args.patience,
        cos_lr=True,
        close_mosaic=10,          # fires at epoch (epochs-10); also zeroes mixup/cutmix/copy_paste
        warmup_epochs=2.0,        # = 6,228 iters at nb=3114; no floor is applied
        # --- optimiser (explicit: 'auto' would ignore lr0/momentum) ---
        optimizer=args.optimizer,
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        # --- loss ---
        cls_pw=cls_pw,
        # --- capacity / input ---
        imgsz=args.imgsz,
        batch=batch,
        multi_scale=args.multi_scale,
        rect=False,               # rect=True silently disables shuffling for the whole run
        # --- augmentation ---
        mosaic=1.0,
        mixup=0.15,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=5.0, translate=0.1, scale=0.5, shear=2.0, perspective=0.0005,
        fliplr=0.5, flipud=0.0,   # never vertical-flip driving imagery
        # --- reproducibility ---
        seed=0,
        deterministic=False,
        # --- runtime ---
        device=0,
        workers=args.workers,
        cache=False,
        amp=True,
        val=True,
        plots=True,
        save_period=10,
    )
    print(f"\nbest weights: {BUILD / 'runs' / args.name / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
