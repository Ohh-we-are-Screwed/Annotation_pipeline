"""Export a fine-tune run's best.pt as THE arm B artifact, with provenance.

The artifact name starts with `yolo` on purpose: Stage 3's provider inference
(pipeline/stage3_proposals/proposals.py:infer_proposal_provider) routes a
weights basename starting with `yolo` to the ultralytics adapter. Rename it
and Stage 3 will try to treat the file as a caption-provider hub id.

Usage:
    python scripts/export_armb.py --run runs/r1280-4
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml

BUILD = Path("/home/mt/Zami/Annotation_pipeline/local_yolox_build")
ARTIFACT_NAME = "yolo11x-rsud20k-armb.pt"

# Same ship list as inference; imported where torch is available, duplicated
# here so provenance_from_run stays importable without ultralytics installed.
SHIP_NAMES = ("rickshaw", "cng")


def fitness(row: dict) -> float:
    """The fitness ultralytics 8.4.120 actually ranks checkpoints by.

    `DetMetrics.fitness` (utils/metrics.py:1007-1010) weights
    [P, R, mAP50, mAP50-95] as [0, 0, 0, 1]: fitness IS mAP50-95, alone. The
    older 0.1*mAP50 + 0.9*mAP50-95 blend belongs to a previous generation of
    the library and would name a different best epoch than the one the run's
    own best.pt holds -- which is why main() cross-checks the two.
    """
    return float(row["metrics/mAP50-95(B)"])


def checkpoint_names(ck: dict) -> dict[int, str]:
    """The class names of whichever module the checkpoint actually carries.

    A FINISHED run's best.pt has been through `strip_optimizer`, which moves
    the EMA weights into `model` and drops `ema`. A run stopped mid-flight --
    a crash, a power cut -- has `model: None` and the weights still in `ema`.
    Reading only `ck["model"]` therefore works on every completed run and dies
    on exactly the checkpoints an interrupted run leaves behind.
    """
    holder = ck.get("model") or ck.get("ema")
    names = getattr(holder, "names", None)
    if not names:
        raise SystemExit(
            "checkpoint carries neither `model` nor `ema` with class names; "
            "without them a class id means nothing and the ship filter cannot be built"
        )
    return {int(k): str(v) for k, v in names.items()}


def provenance_from_run(run_dir: str) -> dict:
    """Everything about the run that does NOT need torch: args, best epoch."""
    run = Path(run_dir)
    best = run / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"{best} not found: nothing to export")
    args = yaml.safe_load((run / "args.yaml").read_text())
    with open(run / "results.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit(f"{run/'results.csv'}: no completed epochs")

    best_row = max(rows, key=fitness)
    return {
        "source_run": str(run.resolve()),
        "classes_trained": list(args.get("classes") or []),
        "ship_names": list(SHIP_NAMES),
        "epochs_completed": int(rows[-1]["epoch"]),
        "epochs_budget": int(args.get("epochs", 0)),
        "best_fitness": fitness(best_row),
        "best_row": {
            "epoch": int(best_row["epoch"]),
            "mAP50": float(best_row["metrics/mAP50(B)"]),
            "mAP50_95": float(best_row["metrics/mAP50-95(B)"]),
        },
        "data_config": str(args.get("data", "")),
        "base_model": str(args.get("model", "")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(BUILD / "runs" / "r1280-4"))
    ap.add_argument("--out", default=str(BUILD / "artifacts" / ARTIFACT_NAME))
    args = ap.parse_args()

    prov = provenance_from_run(args.run)

    # Torch-side checks: the checkpoint must name every shipped class, and its
    # recorded best_fitness must agree with results.csv (a mismatch means the
    # csv and the weights are from different runs).
    import torch
    from predict_armb import ship_indices  # sibling module; refuses on missing names

    src = Path(args.run) / "weights" / "best.pt"
    ck = torch.load(src, map_location="cpu", weights_only=False)
    names = checkpoint_names(ck)
    prov["ship_indices"] = ship_indices(names)
    prov["checkpoint_epoch"] = int(ck.get("epoch", -1))
    ck_fitness = ck.get("best_fitness")
    if ck_fitness is not None and abs(float(ck_fitness) - prov["best_fitness"]) > 1e-3:
        raise SystemExit(
            f"checkpoint best_fitness {float(ck_fitness):.5f} != results.csv max "
            f"{prov['best_fitness']:.5f}: weights and csv are not from the same run"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    prov.update({
        "artifact": str(out),
        "sha256": sha,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY-NC 4.0 (inherited from RSUD20K; see DECISIONS C28)",
    })
    prov_path = out.parent / "armb_provenance.json"
    prov_path.write_text(json.dumps(prov, indent=2) + "\n")
    print(f"exported {out}\nsha256   {sha}\nprovenance {prov_path}")


if __name__ == "__main__":
    main()
