#!/usr/bin/env bash
# The Stage 3b (12 Hz identity propagation, C27) A/B: four cells, each a CLEAN
# SLATE, all four on the same yolo11x proposal checkpoint so the only thing that
# moves between them is Stage 3b itself.
#
#   1 base    stages 0-8 with NO 3b            — the baseline every other cell is read against
#   2 fill    the same, plus 3b                — recovery only: missed boxes come back, detected ones are untouched
#   3 refine  cell 2 + TRACK2D_REFINE=1        — 3b may also REPLACE a detected box with its propagated mask's box
#   4 sam31   cell 2 on facebook/sam3.1        — SAM 3.1 as Stage 3b's TRACKER, the same pass
#   5 m2sam31 cell 1 with mask_2d on sam3.1     — the actual C26 Gate (2) arm: SAM 3.1 as the
#                                                 SEGMENTER, read against cell 1, no 3b involved
#
# Cells 4 and 5 answer different questions and neither substitutes for the other:
# 4 varies the tracker that PROPOSES recovered boxes, 5 varies the model that
# SEGMENTS every box. C26 promotes sam3.1 to the mask_2d default only on 5.
#
# Every cell rebuilds stages 0-8 from scratch, for the reason run_matrix.sh
# states: a comparison whose arms were produced by different amounts of
# recomputation invites the question of whether the difference IS the reuse.
# Here it does double duty — --clean-slate is also what removes a previous
# cell's stage3b_track2d, and a leftover 3b tree is exactly what the baseline
# cell must not be able to pick up (run_stages.sh: stage3_dir_for_4).
#
# CVAT IS NOT PUBLISHED BY ANY CELL, deliberately. run_stages.sh's `cvat` step is
# export AND publish: it rebuilds cvat_export if stale and then runs cvat_setup
# --replace, which DELETES and recreates every "— OUR PIPELINE output" task on
# the server, destroying whatever a human had done inside them. Running that
# four times over would flood the one CVAT instance and leave it describing cell
# 4 regardless of which cell anyone wanted to look at. The two halves are
# separable, so this script takes the export half only — `eval` already runs
# scripts/export_cvat_coco.py per cell, and the Stage 3b provenance the reviewer
# needs (attributes.source / track_id / hops) is in that COCO file. Publishing
# stays a separate, explicit, human act against ONE cell's tree, e.g.
#
#   scripts/run_matrix_track2d.sh 2 && scripts/run_stages.sh cvat
#
# (cell 2's tree is the one still on disk after that; running the whole matrix
# and then publishing would publish cell 4.)
#
#   scripts/run_matrix_track2d.sh          # all four cells
#   scripts/run_matrix_track2d.sh 2 3      # only cells 2 and 3 (1-indexed)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CKPT=/home/mt/dhakascenes/cache/checkpoints
YOLO11X="$CKPT/yolo11x.pt"
SAM31=facebook/sam3.1
SAM31_REV=daa63191845a41281374e725f4c9e51c7a824460

BASE_STEPS="0 1 3 4 5 6 7 8 eval"
FILL_STEPS="0 1 3 3b 4 5 6 7 8 eval"

# label | steps | extra environment for the cell | note recorded in run_config.json
CELLS=(
  "yolo11x_3b_base|$BASE_STEPS||baseline: no Stage 3b, Stage 4 reads stage3_proposals directly"
  "yolo11x_3b_fill|$FILL_STEPS||Stage 3b recovery only (sweep detection on, boxes not refined)"
  "yolo11x_3b_refine|$FILL_STEPS|TRACK2D_REFINE=1|Stage 3b with --refine-boxes: detected boxes may be replaced by their propagated mask box"
  "yolo11x_3b_sam31|$FILL_STEPS|TRACK2D_MODEL_ID=$SAM31 TRACK2D_REVISION=$SAM31_REV|Stage 3b tracker on facebook/sam3.1 multiplex; mask_2d stays on facebook/sam3"
  "yolo11x_mask2d_sam31|$BASE_STEPS|MASK_MODEL_ID=$SAM31 MASK_REVISION=$SAM31_REV|C26 gate (2): mask_2d on facebook/sam3.1, no Stage 3b — read against cell 1, the only pair that differs by the segmenter alone"
)

WANT=("$@")
[ ${#WANT[@]} -eq 0 ] && WANT=($(seq 1 ${#CELLS[@]}))

FAILED=()
SAVED=()

for i in "${WANT[@]}"; do
  IFS='|' read -r LABEL STEPS EXTRA NOTE <<< "${CELLS[$((i-1))]}"
  echo
  echo "############################################################"
  echo "### CELL $i/${#CELLS[@]}  $LABEL"
  echo "###   proposal_2d     $YOLO11X"
  echo "###   steps           $STEPS"
  echo "###   track2d env     ${EXTRA:-<defaults: same checkpoint as mask_2d, no refine>}"
  echo "###   started         $(date -Is)"
  echo "############################################################"

  # The class map is selected by run_stages.sh from the checkpoint basename, so
  # it can never be paired with the wrong vocabulary from here. $EXTRA and
  # $STEPS are UNQUOTED on purpose: both are word lists, not single words.
  # shellcheck disable=SC2086
  env $EXTRA PROPOSAL_MODEL_ID="$YOLO11X" FORCE_GT_EXPORT=1 \
    scripts/run_stages.sh $STEPS --clean-slate --no-cvat
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! CELL $i ($LABEL) FAILED rc=$rc — its metrics are NOT saved"
    FAILED+=("$LABEL(rc=$rc)")
    continue
  fi

  # run_config.json records track2d.consumed_by_stage4 from Stage 4's OWN
  # upstream block, so a cell that ran 3b and did not reach Stage 4 with it
  # cannot be read as if it had.
  "${PY:-/home/mt/miniconda3/envs/ano_pipe/bin/python}" scripts/save_run_results.py \
    --label "$LABEL" \
    --note "clean-slate cell $i of the Stage 3b A/B (scripts/run_matrix_track2d.sh) — $NOTE"
  SAVED+=("$LABEL")
  echo "### CELL $i DONE  $(date -Is)"
done

# The comparison table is part of the artifact, not an afterthought: C26 gate (2)
# and C27 both ask for these four cells "saved under Results/ via
# save_run_results + compare_runs". Only the cells that actually saved are
# listed, so a failed arm prints as absent rather than as a row of `--`.
if [ ${#SAVED[@]} -gt 0 ]; then
  echo
  "${PY:-/home/mt/miniconda3/envs/ano_pipe/bin/python}" scripts/compare_runs.py \
    --runs "${SAVED[@]}" --markdown
fi

echo
echo "############################################################"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "### MATRIX COMPLETE — all requested cells saved under Results/"
else
  echo "### MATRIX INCOMPLETE — failed: ${FAILED[*]}"
fi
echo "### CVAT was NOT published; run 'scripts/run_stages.sh cvat' against the"
echo "### tree of the ONE cell you want reviewed (the last cell run is on disk)."
echo "############################################################"
[ ${#FAILED[@]} -eq 0 ]
