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
# What Stage 4's own manifest records as its upstream when Stage 3b sat DIRECTLY
# between them. It is the only evidence in the tree that the recovered boxes
# reached the masks; the presence of a stage3b_track2d directory is not.
TRACK2D_SPEC = "dhakascenes-pilot/stage3b_track2d/v1"
# Which work_root directory each Stage-3-family spec is written to. Used only to
# find the START of the upstream chain when Stage 4's own manifest predates the
# `stage3_dir` key; every later hop follows a recorded path, never a guess.
STAGE3_DIR_BY_SPEC = {
    "dhakascenes-pilot/stage3_proposals/v1": "stage3_proposals",
    "dhakascenes-pilot/stage3b_track2d/v1": "stage3b_track2d",
    "dhakascenes-pilot/stage3_merge/v1": "stage3_merged",
    "dhakascenes-pilot/stage3c_check/v1": "stage3_checked",
}
METRIC_FILES = ("detect2d_metrics.json", "detect3d_metrics.json", "paint_metrics.json")
RESULTS_ROOT = "/home/mt/Zami/Results"


def _read(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _upstream_dirs(manifest: dict) -> list[str]:
    """The input directories a Stage-3-family manifest records, in a fixed order.

    Three shapes exist and all three are read, because a chain walk that knows
    only one of them silently stops at the first hop it cannot parse:
      - `upstream.input.dir`   -- stage3c_check (one input)
      - `upstream.arm_a.dir` / `upstream.arm_b.dir` -- stage3_merge (two)
      - neither                -- stage3b_track2d and stage3_proposals record
        their upstream by SPEC, not by path, and are chain ENDS anyway.
    """
    upstream = manifest.get("upstream") or {}
    dirs: list[str] = []
    for key in ("input", "arm_a", "arm_b"):
        block = upstream.get(key)
        if isinstance(block, dict) and block.get("dir"):
            dirs.append(str(block["dir"]))
    return dirs


def stage3_chain(work_root: str, s4: dict, *, max_hops: int = 8) -> list[str]:
    """Every Stage-3-family spec Stage 4's masks actually rest on, nearest first.

    `consumed_by_stage4` used to be one string compare -- Stage 4's
    `upstream.stage3_spec == TRACK2D_SPEC` -- which was correct only while
    Stage 4 read the 3b tree DIRECTLY. It goes false the moment anything sits
    between them: 3m merges the 3b tree with arm B, or 3c relabels it, and Stage
    4 then reads a stage3_merged / stage3_checked tree whose rows still carry
    3b's track ids and recovered boxes. The saved table would report "boxes
    reached Stage 4: no" for a run in which they plainly did -- the same class of
    silent misattribution the row exists to prevent. So the chain is WALKED,
    through each manifest's own recorded input path, and every spec on it counts.

    Cycles and runaway chains are bounded rather than trusted: a `seen` set plus
    `max_hops`, because these are paths read off disk and a hand-edited manifest
    is not a hypothetical failure.
    """
    upstream = s4.get("upstream") or {}
    start = upstream.get("stage3_dir") or STAGE3_DIR_BY_SPEC.get(upstream.get("stage3_spec") or "")
    if not start:
        return []
    pending = [start]
    seen: set[str] = set()
    specs: list[str] = []
    for _hop in range(max_hops):
        if not pending:
            break
        current, pending = pending[0], pending[1:]
        # Recorded dirs are realpaths on the box that ran. Fall back to the same
        # basename under THIS work_root so a moved or copied tree still walks.
        candidates = [current, os.path.join(work_root, os.path.basename(str(current).rstrip("/")))]
        manifest = None
        for path in candidates:
            if path in seen:
                manifest = None
                break
            manifest = _read(os.path.join(path, "run_manifest.json"))
            if manifest is not None:
                seen.add(path)
                break
        if manifest is None:
            continue
        spec = manifest.get("spec")
        if spec and spec not in specs:
            specs.append(spec)
        pending += _upstream_dirs(manifest)
    return specs


def run_config(work_root: str) -> dict:
    """The identity of the models this run actually loaded, from their manifests."""
    s3 = _read(os.path.join(work_root, "stage3_proposals", "run_manifest.json")) or {}
    # Stage 3b is OPT-IN, so its absence is a fact about the run: `None`, never
    # an empty block that reads like a stage which ran and did nothing.
    s3b = _read(os.path.join(work_root, "stage3b_track2d", "run_manifest.json"))
    # Stage 3c is opt-in for the same reason and gets the same treatment. Until
    # now it was invisible in every archived Results/ table: a run that spent a
    # VLM pass relabelling its proposals and a run that did not were the same
    # two columns.
    s3c = _read(os.path.join(work_root, "stage3_checked", "run_manifest.json"))
    s4 = _read(os.path.join(work_root, "stage4_masks", "run_manifest.json")) or {}
    s7 = _read(os.path.join(work_root, "stage7_track", "run_manifest.json")) or {}
    class_map = s3.get("class_map") or {}
    s3b_config = (s3b or {}).get("config") or {}
    s3c_totals = (s3c or {}).get("totals") or {}
    s3c_vlm = (s3c or {}).get("vlm") or {}
    s4_config = s4.get("config") or {}
    s4_text = s4.get("text_prompt") or {}
    s4_text_counts = s4_text.get("counts") or {}
    appearance = s7.get("appearance") or {}
    chain = stage3_chain(work_root, s4)
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
            # block is the record of which tree it actually read -- WALKED, not
            # compared once, because 3m and 3c legitimately sit between them and
            # a single spec compare reports "no" for a chain in which the boxes
            # plainly reached the masks (see stage3_chain).
            "consumed_by_stage4": TRACK2D_SPEC in chain,
            # The walk itself, so the yes/no above is checkable rather than trusted.
            "stage4_input_chain": chain,
        },
        "vlm_check": None if s3c is None else {
            "check_mode": s3c_totals.get("check_mode") or s3c_vlm.get("check_mode"),
            "provider": s3c.get("provider"),
            "n_vlm_calls": s3c_totals.get("n_vlm_calls"),
            "n_checked": s3c_totals.get("n_checked"),
            "n_relabeled": s3c_totals.get("n_relabeled"),
            "n_boxes": s3c_totals.get("n_boxes"),
            # per_track only; None under per_box, which is a fact about the run
            # and not a gap in the record.
            "track_coverage": s3c_totals.get("track_coverage"),
            "n_boxes_from_track_verdict": s3c_totals.get("n_boxes_from_track_verdict"),
            "elapsed_s": s3c.get("elapsed_s"),
            # Same reasoning as track2d's: a stage3_checked tree on disk is not
            # evidence that its relabels reached the masks.
            "consumed_by_stage4": (s3c.get("spec") or "") in chain,
        },
        "mask_2d": {"provider": s4.get("provider"), **(s4.get("checkpoint") or {}),
                    # C29: without these the text arm and the box arm are two
                    # identical configurations wearing two different metric sets.
                    "text_prompt": s4_config.get("text_prompt"),
                    "text_prompt_strip_article": s4_config.get("text_prompt_strip_article"),
                    "text_match_min_iou": s4_config.get("text_match_min_iou"),
                    "text_score_threshold": s4_config.get("text_score_threshold"),
                    "text_detector_dtype": s4_config.get("text_detector_dtype"),
                    "n_text_matched": s4_text_counts.get("n_text_matched"),
                    "n_box_fallback": s4_text_counts.get("n_box_fallback"),
                    "n_text_duplicate_rejected": s4_text_counts.get("n_text_duplicate_rejected"),
                    "n_cross_phrase_mask_overlap":
                        s4_text_counts.get("n_cross_phrase_mask_overlap"),
                    "resolved_prompts_sha256": s4_text.get("resolved_prompts_sha256"),
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
    m = config["mask_2d"]
    v = config.get("vlm_check")
    print(f"  proposal: {p.get('model_id')}  ({p.get('n_mapped')}/{p.get('n_source_classes')} classes mapped)")
    print(f"  mask_2d : {m.get('provider')}  text_prompt={m.get('text_prompt')}  "
          f"text_matched={m.get('n_text_matched')}  box_fallback={m.get('n_box_fallback')}")
    if v is None:
        print("  vlm     : (Stage 3c did not run — labels are the detector's)")
    else:
        print(f"  vlm     : {v.get('check_mode')}  {v.get('n_vlm_calls')} calls  "
              f"{v.get('n_relabeled')} relabeled  consumed_by_stage4={v.get('consumed_by_stage4')}")
    print(f"  reid    : {r.get('model_id')} @ {str(r.get('revision'))[:12]}  enabled={r.get('enabled')}")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
