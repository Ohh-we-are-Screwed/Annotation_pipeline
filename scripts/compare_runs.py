#!/usr/bin/env python3
"""Tabulate the saved runs under Results/ side by side.

Reads only what `save_run_results.py` wrote -- the metrics files and the
`run_config.json` built from each run's own manifests -- so the table can never
describe a configuration the run did not have.

A cell that is missing a metric prints `--`, never 0.0: a zero is a measurement
and an absence is not, and averaging the two is how a comparison comes to
understate an arm that simply did not finish.

    python3 scripts/compare_runs.py                    # every run in Results/
    python3 scripts/compare_runs.py --runs a b         # a chosen order
    python3 scripts/compare_runs.py --markdown         # GitHub table
"""

from __future__ import annotations

import argparse
import json
import os

RESULTS_ROOT = "/home/mt/Zami/Results"

# (section, label, file, dotted path, format)
ROWS: list[tuple[str, str, str, str, str]] = [
    ("3D boxes (vs human 3D answer key)", "precision — localization", "detect3d_metrics.json", "precision_localization", "pct"),
    ("3D boxes (vs human 3D answer key)", "precision — class-aware", "detect3d_metrics.json", "precision_class_aware", "pct"),
    ("3D boxes (vs human 3D answer key)", "GT recall (reachable classes)", "detect3d_metrics.json", "gt_recall", "pct"),
    ("3D boxes (vs human 3D answer key)", "ATE (m, lower better)", "detect3d_metrics.json", "matched.ate_m_mean", "f3"),
    ("3D boxes (vs human 3D answer key)", "ASE (lower better)", "detect3d_metrics.json", "matched.ase_mean", "f3"),
    ("3D boxes (vs human 3D answer key)", "AOE (rad, lower better)", "detect3d_metrics.json", "matched.aoe_rad_mean_trusted_yaw", "f3"),
    ("3D boxes (vs human 3D answer key)", "boxes shipped", "detect3d_metrics.json", "n_pred_boxes", "int"),
    ("3D boxes (vs human 3D answer key)", "matched to a GT box", "detect3d_metrics.json", "matched.n", "int"),
    ("2D proposals (vs human 2D answer key)", "precision — localization", "detect2d_metrics.json", "precision_localization", "pct"),
    ("2D proposals (vs human 2D answer key)", "precision — class-aware", "detect2d_metrics.json", "precision_class_aware", "pct"),
    ("2D proposals (vs human 2D answer key)", "GT recall (all GT)", "detect2d_metrics.json", "gt_recall_all", "pct"),
    ("2D proposals (vs human 2D answer key)", "GT recall (>=32px)", "detect2d_metrics.json", "gt_recall_visible", "pct"),
    ("2D proposals (vs human 2D answer key)", "predictions counted", "detect2d_metrics.json", "n_predictions", "int"),
    # Stage 3b is the A/B ARM ITSELF: without these rows the base / +3b /
    # +3b+refine / +3b+sam3.1 columns are four identical configurations wearing
    # four different metric sets, and nothing in the table says which is which.
    # `--` here means the run has no track2d block at all -- Stage 3b did not
    # run, and the CONFIGURATION section above says so in words.
    ("Stage 3b (2D track recovery)", "boxes reached Stage 4", "run_config.json", "track2d.consumed_by_stage4", "yesno"),
    ("Stage 3b (2D track recovery)", "provider", "run_config.json", "track2d.provider", "text"),
    ("Stage 3b (2D track recovery)", "checkpoint", "run_config.json", "track2d.checkpoint.model_id", "base"),
    ("Stage 3b (2D track recovery)", "checkpoint revision", "run_config.json", "track2d.checkpoint.revision", "sha"),
    ("Stage 3b (2D track recovery)", "refine_matched_boxes (A/B)", "run_config.json", "track2d.refine_matched_boxes", "yesno"),
    ("Stage 3b (2D track recovery)", "detect_on_sweeps (A/B)", "run_config.json", "track2d.detect_on_sweeps", "yesno"),
    ("Stage 3b (2D track recovery)", "boxes recovered", "run_config.json", "track2d.totals.n_recovered", "int"),
    ("Stage 3b (2D track recovery)", "boxes refined", "run_config.json", "track2d.totals.n_refined", "int"),
    ("Stage 3b (2D track recovery)", "mid-gap births", "run_config.json", "track2d.totals.n_midgap_births", "int"),
    ("Stage 3b (2D track recovery)", "2D tracks", "run_config.json", "track2d.totals.n_tracks", "int"),
    # Stage 3c and the Stage 4 text arm are A/B ARMS for the same reason Stage 3b
    # is, and neither had a row here: a VLM-relabelled run and a raw one, or a
    # text-prompted run and a box-prompted one, were identical columns wearing
    # different metric sets. `--` means the block is absent -- 3c did not run, or
    # the run predates C29 -- and the CONFIGURATION section above says so in words.
    ("Stage 3c (VLM label check)", "verdicts reached Stage 4", "run_config.json", "vlm_check.consumed_by_stage4", "yesno"),
    ("Stage 3c (VLM label check)", "check mode (A/B)", "run_config.json", "vlm_check.check_mode", "text"),
    ("Stage 3c (VLM label check)", "VLM calls", "run_config.json", "vlm_check.n_vlm_calls", "int"),
    ("Stage 3c (VLM label check)", "boxes checked", "run_config.json", "vlm_check.n_checked", "int"),
    ("Stage 3c (VLM label check)", "boxes relabeled", "run_config.json", "vlm_check.n_relabeled", "int"),
    ("Stage 3c (VLM label check)", "track coverage (per_track)", "run_config.json", "vlm_check.track_coverage", "pct"),
    ("Stage 3c (VLM label check)", "boxes from a track verdict", "run_config.json", "vlm_check.n_boxes_from_track_verdict", "int"),
    ("Stage 4 (masks)", "provider", "run_config.json", "mask_2d.provider", "text"),
    ("Stage 4 (masks)", "text prompting (A/B)", "run_config.json", "mask_2d.text_prompt", "yesno"),
    ("Stage 4 (masks)", "strip leading article", "run_config.json", "mask_2d.text_prompt_strip_article", "yesno"),
    ("Stage 4 (masks)", "text match floor (IoU)", "run_config.json", "mask_2d.text_match_min_iou", "f2"),
    ("Stage 4 (masks)", "detector score threshold", "run_config.json", "mask_2d.text_score_threshold", "f2"),
    ("Stage 4 (masks)", "detector dtype", "run_config.json", "mask_2d.text_detector_dtype", "text"),
    ("Stage 4 (masks)", "prompt strings sha", "run_config.json", "mask_2d.resolved_prompts_sha256", "sha"),
    ("Stage 4 (masks)", "masks from a text match", "run_config.json", "mask_2d.n_text_matched", "int"),
    ("Stage 4 (masks)", "masks from the box fallback", "run_config.json", "mask_2d.n_box_fallback", "int"),
    ("Stage 4 (masks)", "duplicate instances rejected", "run_config.json", "mask_2d.n_text_duplicate_rejected", "int"),
    ("Stage 4 (masks)", "cross-phrase mask overlaps", "run_config.json", "mask_2d.n_cross_phrase_mask_overlap", "int"),
    ("Paint / lift geometry", "GT coverage rate", "paint_metrics.json", "gt_coverage.rate", "pct"),
    ("Paint / lift geometry", "painted points inside a GT box", "paint_metrics.json", "paint_inside_gt_rate", "pct"),
    ("Paint / lift geometry", "enrichment over base rate", "paint_metrics.json", "enrichment", "f2"),
    ("Paint / lift geometry", "points painted", "paint_metrics.json", "totals.n_points_painted", "int"),
    ("Tracking (Stage 7)", "tracks total", "run_config.json", "stage7_totals.n_tracks_total", "int"),
    ("Tracking (Stage 7)", "tracks >=3 hits", "run_config.json", "stage7_totals.n_tracks_stable", "int"),
    ("Tracking (Stage 7)", "matched pairs", "run_config.json", "stage7_totals.n_matched", "int"),
    ("Tracking (Stage 7)", "appearance-trusted pairs", "run_config.json", "stage7_totals.n_appearance_trusted_pairs", "int"),
    ("Tracking (Stage 7)", "yaw flips applied", "run_config.json", "stage7_totals.n_yaw_flips", "int"),
    ("Runtime", "stage 3 proposals (s)", "run_config.json", "proposal_2d.elapsed_s", "f1"),
    ("Runtime", "stage 3b track2d (s)", "run_config.json", "track2d.elapsed_s", "f1"),
    ("Runtime", "stage 3c vlm check (s)", "run_config.json", "vlm_check.elapsed_s", "f1"),
    ("Runtime", "stage 4 masks (s)", "run_config.json", "mask_2d.elapsed_s", "f1"),
    ("Runtime", "stage 7 track (s)", "run_config.json", "reid_embedding.elapsed_s", "f1"),
]

PER_CLASS_METRIC = "precision_localization"


def dig(doc: dict | None, path: str):
    cur = doc
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def fmt(value, kind: str) -> str:
    if value is None:
        return "--"
    if kind == "pct":
        return f"{100.0 * float(value):.1f}%"
    if kind == "int":
        return f"{int(value):,}"
    # A recorded False is a measurement like any other and prints as one: only
    # an ABSENT key is `--`. `int` would render both as 0 and lose the
    # difference, which is the whole point of the flag rows.
    if kind == "yesno":
        return "yes" if value else "no"
    if kind == "text":
        return str(value)
    if kind == "sha":
        return str(value)[:12]
    if kind == "base":
        return os.path.basename(str(value))
    if kind.startswith("f"):
        return f"{float(value):.{int(kind[1:])}f}"
    return str(value)


def load_run(root: str, label: str) -> dict:
    docs = {}
    for name in ("detect2d_metrics.json", "detect3d_metrics.json",
                 "paint_metrics.json", "run_config.json"):
        path = os.path.join(root, label, name)
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as fh:
                docs[name] = json.load(fh)
    return docs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-root", default=RESULTS_ROOT)
    ap.add_argument("--runs", nargs="*", default=None, help="labels, in the order to print")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args(argv)

    root = args.results_root
    labels = args.runs or sorted(
        d for d in os.listdir(root)
        if os.path.isfile(os.path.join(root, d, "run_config.json"))
    )
    if not labels:
        print(f"no saved runs under {root}")
        return 1
    runs = {label: load_run(root, label) for label in labels}

    # --- the configuration each column actually ran -------------------------
    print("=" * 100)
    print("CONFIGURATION (read from each run's own stage manifests)")
    print("=" * 100)
    for label in labels:
        cfg = runs[label].get("run_config.json") or {}
        p, r = cfg.get("proposal_2d") or {}, cfg.get("reid_embedding") or {}
        reach = p.get("reachable_phrases") or []
        unreach = p.get("unreachable_phrases") or []
        print(f"\n  {label}")
        print(f"    proposal_2d    {os.path.basename(str(p.get('model_id')))}  "
              f"sha {str(p.get('sha256'))[:12]}  "
              f"{p.get('n_mapped')}/{p.get('n_source_classes')} source classes mapped")
        print(f"    class map      {os.path.basename(str(p.get('class_map_path')))}  "
              f"sha {str(p.get('class_map_sha256'))[:12]}")
        print(f"    reachable      {len(reach)} phrases: {', '.join(reach)}")
        print(f"    UNREACHABLE    {len(unreach)} phrases: {', '.join(unreach) or '(none)'}")
        # `track2d` is None when Stage 3b did not run, and that is a FACT about
        # the arm, not a gap in the record -- said in words here so the reader
        # never has to read `--` in the table below as "unknown".
        t = cfg.get("track2d")
        if t is None:
            print("    track2d        (Stage 3b did not run — Stage 4 read the raw Stage 3 boxes)")
        else:
            ck = t.get("checkpoint") or {}
            print(f"    track2d        {t.get('provider')}  "
                  f"{os.path.basename(str(ck.get('model_id')))} @ {str(ck.get('revision'))[:12]}  "
                  f"refine={t.get('refine_matched_boxes')} sweeps={t.get('detect_on_sweeps')}  "
                  f"consumed_by_stage4={t.get('consumed_by_stage4')}")
        # Same treatment for 3c and for Stage 4's prompt mode: `None` / absent is
        # a FACT about the arm, said in words, so the `--` cells below are never
        # read as "unknown".
        v = cfg.get("vlm_check")
        if v is None:
            print("    vlm_check      (Stage 3c did not run — Stage 4 read the detector's labels)")
        else:
            print(f"    vlm_check      {v.get('check_mode')}  {v.get('n_vlm_calls')} calls  "
                  f"{v.get('n_relabeled')} relabeled  "
                  f"consumed_by_stage4={v.get('consumed_by_stage4')}")
        m = cfg.get("mask_2d") or {}
        if m.get("text_prompt"):
            print(f"    mask_2d        {m.get('provider')}  TEXT-PROMPTED  "
                  f"strip_article={m.get('text_prompt_strip_article')} "
                  f"floor={m.get('text_match_min_iou')} score>={m.get('text_score_threshold')} "
                  f"{m.get('text_detector_dtype')}  "
                  f"prompts sha {str(m.get('resolved_prompts_sha256'))[:12]}")
        else:
            print(f"    mask_2d        {m.get('provider')}  box-prompted "
                  f"(text_prompt={m.get('text_prompt')})")
        print(f"    reid_embedding {r.get('model_id')} @ {str(r.get('revision'))[:12]}  "
              f"enabled={r.get('enabled')}")

    width = max(34, *(len(x) for x in labels)) if labels else 34
    colw = max(14, *(len(x) for x in labels))

    def line(cells: list[str]) -> str:
        if args.markdown:
            return "| " + " | ".join(cells) + " |"
        return cells[0].ljust(width) + "".join(c.rjust(colw + 2) for c in cells[1:])

    print()
    print("=" * 100)
    print("METRICS")
    print("=" * 100)
    print()
    print(line(["metric", *labels]))
    if args.markdown:
        print("|" + "|".join(["---"] * (len(labels) + 1)) + "|")

    section = None
    for sec, label_text, fname, path, kind in ROWS:
        if sec != section:
            section = sec
            print(line([f"**{sec}**" if args.markdown else f"-- {sec} " + "-" * 6,
                        *[""] * len(labels)]))
        cells = [label_text]
        for run_label in labels:
            cells.append(fmt(dig(runs[run_label].get(fname), path), kind))
        print(line(cells))

    # --- per-class 3D localization precision --------------------------------
    classes: list[str] = []
    for run_label in labels:
        per = dig(runs[run_label].get("detect3d_metrics.json"), "per_class") or {}
        for name in per:
            if name not in classes:
                classes.append(name)
    if classes:
        print()
        print(line([f"**3D per-class {PER_CLASS_METRIC} (boxes)**" if args.markdown
                    else f"-- 3D per-class {PER_CLASS_METRIC} " + "-" * 6, *[""] * len(labels)]))
        for name in sorted(classes):
            cells = [f"  {name}"]
            for run_label in labels:
                per = dig(runs[run_label].get("detect3d_metrics.json"), f"per_class.{name}")
                cells.append("--" if not per else
                             f"{100.0 * per[PER_CLASS_METRIC]:.1f}% ({per['boxes']:,})")
            print(line(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
