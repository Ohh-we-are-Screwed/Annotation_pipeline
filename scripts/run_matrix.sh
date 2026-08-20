#!/usr/bin/env bash
# The proposal_2d x reid_embedding comparison: four cells, each a CLEAN SLATE.
#
# Every cell rebuilds stages 0-8 from scratch. Stages 0-6 do not depend on the
# re-ID checkpoint, so two of the four cells could have reused the other two's
# trees -- and that is exactly what is NOT done here (human-directed
# 2026-08-14). A comparison whose arms were produced by different amounts of
# recomputation invites the question of whether the difference IS the reuse; a
# clean slate per cell costs ~12 minutes and removes the question.
#
# CVAT is untouched: the step list omits `cvat`/`cvat3d`, which is also what
# stops --clean-slate from purging the review server four times over.
#
#   scripts/run_matrix.sh            # all four cells
#   scripts/run_matrix.sh 3 4        # only cells 3 and 4 (1-indexed)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

CKPT=/home/mt/dhakascenes/cache/checkpoints
DINOV2=facebook/dinov2-small
DINOV3=facebook/dinov3-vits16-pretrain-lvd1689m

# label | proposal checkpoint | reid checkpoint
CELLS=(
  "yolo11x_dinov2|$CKPT/yolo11x.pt|$DINOV2"
  "yolo11x_dinov3|$CKPT/yolo11x.pt|$DINOV3"
  "yolov8x-oiv7_dinov2|$CKPT/yolov8x-oiv7.pt|$DINOV2"
  "yolov8x-oiv7_dinov3|$CKPT/yolov8x-oiv7.pt|$DINOV3"
)

WANT=("$@")
[ ${#WANT[@]} -eq 0 ] && WANT=($(seq 1 ${#CELLS[@]}))

STEPS=(0 1 3 4 5 6 7 8 eval)
FAILED=()

for i in "${WANT[@]}"; do
  IFS='|' read -r LABEL PROPOSAL REID <<< "${CELLS[$((i-1))]}"
  echo
  echo "############################################################"
  echo "### CELL $i/${#CELLS[@]}  $LABEL"
  echo "###   proposal_2d     $PROPOSAL"
  echo "###   reid_embedding  $REID"
  echo "###   started         $(date -Is)"
  echo "############################################################"

  # The class map is selected by run_stages.sh from the checkpoint basename, so
  # it can never be paired with the wrong vocabulary from here.
  PROPOSAL_MODEL_ID="$PROPOSAL" REID_MODEL_ID="$REID" FORCE_GT_EXPORT=1 \
    scripts/run_stages.sh "${STEPS[@]}" --clean-slate --no-cvat
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!! CELL $i ($LABEL) FAILED rc=$rc — its metrics are NOT saved"
    FAILED+=("$LABEL(rc=$rc)")
    continue
  fi

  "${PY:-/home/mt/miniconda3/envs/ano_pipe/bin/python}" scripts/save_run_results.py \
    --label "$LABEL" \
    --note "clean-slate cell $i of the proposal_2d x reid_embedding matrix (scripts/run_matrix.sh)"
  echo "### CELL $i DONE  $(date -Is)"
done

echo
echo "############################################################"
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "### MATRIX COMPLETE — all requested cells saved under Results/"
else
  echo "### MATRIX INCOMPLETE — failed: ${FAILED[*]}"
fi
echo "############################################################"
[ ${#FAILED[@]} -eq 0 ]
