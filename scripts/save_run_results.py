#!/usr/bin/env python3
"""Copy one run's NUMERICAL metrics out of work_root into Results/<label>/.

work_root holds exactly one run at a time: the next `--clean-slate` erases it.
A comparison across detector/re-ID pairings therefore has to be lifted out of
the tree before the next arm overwrites it, and lifted out WITH the identity of
the models that produced it -- a metrics file alone cannot say which checkpoint
ran, and a directory name is not evidence.

`run_config.json` is therefore built from the stage manifests, never from the
command line that invoked this script: it records what the run actually loaded
(model ids, hub revisions, weight sha256s, the class map and its reachable
phrases), so a number and the configuration that produced it stay readable
together.

    python3 scripts/save_run_results.py --label yolo11x_dinov3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.common.paths import load_paths  # noqa: E402

SPEC = "dhakascenes-pilot/run_results/v1"
# What Stage 4's own manifest records as its upstream when Stage 3b sat between
# them. It is the only evidence in the tree that the recovered boxes reached the
# masks; the presence of a stage3b_track2d directory is not.
TRACK2D_SPEC = "dhakascenes-pilot/stage3b_track2d/v1"
METRIC_FILES = ("detect2d_metrics.json", "detect3d_metrics.json", "paint_metrics.json")
RESULTS_ROOT = "/home/mt/Zami/Results"


def _read(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_config(work_root: str) -> dict:
    """The identity of the models this run actually loaded, from their manifests."""
    s3 = _read(os.path.join(work_root, "stage3_proposals", "run_manifest.json")) or {}
    # Stage 3b is OPT-IN, so its absence is a fact about the run: `None`, never
    # an empty block that reads like a stage which ran and did nothing.
    s3b = _read(os.path.join(work_root, "stage3b_track2d", "run_manifest.json"))
    s4 = _read(os.path.join(work_root, "stage4_masks", "run_manifest.json")) or {}
    s7 = _read(os.path.join(work_root, "stage7_track", "run_manifest.json")) or {}
    class_map = s3.get("class_map") or {}
    s3b_config = (s3b or {}).get("config") or {}
    appearance = s7.get("appearance") or {}
    # Stage 7 records the re-ID checkpoint in its CONFIG block (the adapter's
    # CheckpointSpec is not serialised separately), so that is where its
    # identity is read from -- not from the label this script was invoked with.
    s7_config = s7.get("config") or {}

    return {
        "spec": SPEC,
        "proposal_2d": {
            "provider": s3.get("provider"),
            **(s3.get("checkpoint") or {}),
            "class_map_path": class_map.get("path"),
            "class_map_sha256": class_map.get("sha256"),
            "n_source_classes": class_map.get("n_source_classes"),
            "n_mapped": class_map.get("n_mapped"),
            "reachable_phrases": class_map.get("phrases_in_use"),
            "unreachable_phrases": class_map.get("unreachable_phrases"),
            "elapsed_s": s3.get("elapsed_s"),
        },
        "track2d": None if s3b is None else {
            "provider": s3b.get("provider"),
            "checkpoint": s3b.get("checkpoint"),
            "detect_on_sweeps": s3b_config.get("detect_on_sweeps"),
            "refine_matched_boxes": s3b_config.get("refine_matched_boxes"),
            "match_iou": s3b_config.get("match_iou"),
            "miss_tolerance_keyframes": s3b_config.get("miss_tolerance_keyframes"),
            "recovered_score_decay": s3b_config.get("recovered_score_decay"),
            "sweep_birth_min_hits": s3b_config.get("sweep_birth_min_hits"),
            "totals": s3b.get("totals"),
            "elapsed_s": s3b.get("elapsed_s"),
            # A stage3b tree on disk is NOT evidence that its boxes were used:
            # run_stages.sh points Stage 4 at Stage 3 whenever the 3b tree is
            # stale, so a leftover directory would otherwise attribute a
            # baseline number to the recovery pass. Stage 4's own upstream
            # block is the record of which tree it actually read.
            "consumed_by_stage4":
                ((s4.get("upstream") or {}).get("stage3_spec") == TRACK2D_SPEC),
        },
        "mask_2d": {"provider": s4.get("provider"), **(s4.get("checkpoint") or {}),
                    "elapsed_s": s4.get("elapsed_s")},
        "reid_embedding": {
            "model_id": s7_config.get("reid_model_id"),
            "revision": s7_config.get("reid_revision"),
            "min_crop_px": s7_config.get("min_crop_px"),
            "enabled": appearance.get("enabled"),
            "unavailable_reason": appearance.get("unavailable_reason"),
            "elapsed_s": s7.get("elapsed_s"),
        },
        "stage7_totals": s7.get("totals"),
        "scenes": [s.get("scene") or s.get("name") for s in (s3.get("scenes") or [])],
        "work_root": work_root,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--label", required=True, help="subdirectory under Results/")
    ap.add_argument("--paths", default=os.environ.get("DHAKASCENES_PATHS_CONFIG", "configs/paths.yaml"))
    ap.add_argument("--results-root", default=RESULTS_ROOT)
    ap.add_argument("--note", default="", help="one line recorded in run_config.json")
    args = ap.parse_args(argv)

    work_root = load_paths(args.paths).work_root
    metrics_dir = os.path.join(work_root, "metrics")
    out_dir = os.path.join(args.results_root, args.label)
    os.makedirs(out_dir, exist_ok=True)

    copied, missing = [], []
    for name in METRIC_FILES:
        src = os.path.join(metrics_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, name))
            copied.append(name)
        else:
            missing.append(name)

    config = run_config(work_root)
    if args.note:
        config["note"] = args.note
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print(f"wrote {out_dir}")
    print(f"  metrics : {', '.join(copied) if copied else '(none)'}")
    if missing:
        print(f"  MISSING : {', '.join(missing)} — that eval sub-step did not run")
    p = config["proposal_2d"]
    r = config["reid_embedding"]
    print(f"  proposal: {p.get('model_id')}  ({p.get('n_mapped')}/{p.get('n_source_classes')} classes mapped)")
    print(f"  reid    : {r.get('model_id')} @ {str(r.get('revision'))[:12]}  enabled={r.get('enabled')}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
