#!/usr/bin/env bash
# DhakaScenes pilot — one-shot run: stages 0-8, then eval + viz, then publish
# the 2D review tasks to CVAT.
#
# This is still a convenience wrapper around the per-stage CLIs, not the
# Phase-10 orchestrator (scripts/run_pilot.py, still unwritten): every stage
# writes its own run_manifest.json and _SUCCESS / _SUCCESS.degraded marker, and
# the wrapper adds nothing to provenance. See docs/RUNNING.md.
#
# SELF-MODERATION (the reason this rewrite exists)
# ------------------------------------------------
# Every stage CLI has a THREE-state exit code, and the previous wrapper
# collapsed it into two by treating any non-zero as failure:
#
#   0  OK        — everything under contract
#   1  DEGRADED  — the stage RAN TO COMPLETION and wrote everything; some
#                  scenes are quality-flagged (e.g. images with zero
#                  proposals). The marker is `_SUCCESS.degraded` and it names
#                  the causes. This is a complete output, not a failure.
#   2  REFUSED   — upstream contract broken or a checkpoint is unavailable.
#                  NOTHING was written. There is no marker.
#
# So: rc 1 is recorded and the chain CONTINUES; rc 2 aborts the chain and
# suppresses the publish, because publishing a half-built tree for human review
# is worse than not publishing.
#
# But the exit code alone is NOT sufficient evidence, and trusting it burned us
# once already: python also exits 1 on an uncaught traceback, so a stage that
# died on `import pipeline` was announced as "output is COMPLETE and
# quality-flagged" and the chain marched on. rc 1 is therefore cross-examined
# against the marker the stage actually wrote — no _SUCCESS.degraded on disk
# means it crashed, not degraded, and the run stops. The marker is the record;
# the exit code is a claim about it.
#
# The C16 consequence, stated because it is a real decision and not an
# accident: once any upstream stage is degraded, this script passes
# `--accept-degraded-upstream` to every stage after it. That flag is designed
# as an explicit human opt-in, and here a human opted in ONCE, for the whole
# chain, by running this script. It stays recorded in each consumer's own
# run_manifest.json (`upstream.accepted_degraded_upstream: true`), so the
# provenance is intact — but no per-stage prompt stands between a degraded
# Stage 1 and a published CVAT task. That is the trade this script makes.
# The flag is NOT passed blanket-on: it is passed only after a degraded marker
# or a degraded exit has actually been observed in this run (see `acc`).
#
# OVERWRITING
# -----------
# Every stage clears its own markers and rewrites its own output tree in place;
# the CVAT publish DELETES and recreates the pipeline's own tasks. Nothing is
# versioned and nothing is kept. Human edits made inside an
# "— OUR PIPELINE output" CVAT task are destroyed by the publish. The
# "— nuScenes HUMAN answer key" tasks are never touched by an ordinary run
# (register C13) — only --clean-slate removes them.
#
# --clean-slate goes further: it deletes every stage tree, export, metric and
# render under work_root, and every task and project on the CVAT server, before
# the first step runs. It spares work_root/logs (the audit trail), the priors
# under out_root (a Stage 6 INPUT that nothing here regenerates), and the
# read-only dataroot. CVAT has no undo; 3D cuboids in particular are not
# rebuildable by any committed script.
#
# Usage:
#   scripts/run_stages.sh                 # 0 1 3 4 5 6 7 8, eval, viz, cvat
#   scripts/run_stages.sh --clean-slate   # the same, from an empty work tree and an empty CVAT
#   scripts/run_stages.sh 3 4 5           # a subset, in the order given
#   scripts/run_stages.sh 4 5 6 7 8 eval cvat   # resume after a completed stage 3
#   scripts/run_stages.sh cvat            # republish to CVAT from what is on disk
#   scripts/run_stages.sh --no-cvat       # full chain, no publish
#   scripts/run_stages.sh --scenes scene-0061   # one scene, end to end
#
#   MASK_MODEL_ID=facebook/sam2.1-hiera-large MASK_REVISION=<sha> scripts/run_stages.sh 4
#
# Steps: 0 1 3 4 5 6 7 8 | eval (COCO export + metrics) | viz (PNG renders) | cvat
# (Stage 2 is the peer-owned OOD branch and is not part of this chain.)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Per-scene status lines must appear in a redirected log AS the run progresses,
# not in one flush at process exit (python block-buffers stdout to files).
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Environment (.env is the machine-local contract; configs/ is the experiment)
# ---------------------------------------------------------------------------

# Caller-set variables WIN over .env: an explicit `FOO=bar scripts/run_stages.sh`
# must not be silently overwritten by the file. Only unset keys are filled in.
load_env_file() {
  local file="$1" line key value
  [ -f "$file" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in ''|'#'*) continue ;; esac
    [[ "$line" == *=* ]] || continue
    key="${line%%=*}"; value="${line#*=}"
    key="${key//[[:space:]]/}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    [ -n "${!key-}" ] && continue
    export "$key=$value"
  done < "$file"
}
load_env_file "$REPO/.env"

# C15 was caused by the wrong interpreter. This one, always.
PY="${PY:-/home/mt/miniconda3/envs/ano_pipe/bin/python}"
[ -x "$PY" ] || { echo "interpreter not found: $PY" >&2; exit 2; }

PATHS_CONFIG="${DHAKASCENES_PATHS_CONFIG:-configs/paths.yaml}"

# proposal_2d (Stage 3) — default since 2026-08-14: YOLO11x (ultralytics), whose
# boxes Stage 4 hands to SAM 3. ultralytics ships weights as a FILE, so the id is
# a path and the "revision" is the release tag that file was downloaded from;
# Stage 3 additionally hashes the bytes into the manifest. CheckpointSpec refuses
# an empty revision either way (§7.2).
#   The previous open-vocabulary default stays one env var away:
#   PROPOSAL_MODEL_ID=iSEE-Laboratory/llmdet_large \
#   PROPOSAL_REVISION=bec37f296f05b22f6c6b39bc05a6c611239f4e31 scripts/run_stages.sh 3
PROPOSAL_MODEL_ID="${PROPOSAL_MODEL_ID:-${YOLO11_CHECKPOINT:-/home/mt/dhakascenes/cache/checkpoints/yolo11x.pt}}"
PROPOSAL_REVISION="${PROPOSAL_REVISION:-v8.3.0}"

# ultralytics writes a settings.json at import time; without this it lands in
# $HOME/.config and prints a warning on every stage invocation.
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/home/mt/dhakascenes/cache/ultralytics}"
mkdir -p "$YOLO_CONFIG_DIR"

# mask_2d (Stage 4) — C19 default: SAM 3 tracker (licence granted 2026-08-13);
# SAM 2.1-L is the ungated alternate: facebook/sam2.1-hiera-large @ 665f8e2ad61cf5f53d65644ff27c8ee525124610
MASK_MODEL_ID="${MASK_MODEL_ID:-facebook/sam3}"
MASK_REVISION="${MASK_REVISION:-3c879f39826c281e95690f02c7821c4de09afae7}"

# reid_embedding (Stage 7) — facebook/dinov2-small. Mandatory: track.py:1449
# refuses (rc 2) with an unpinned id rather than tracking the default branch.
REID_REVISION="${REID_REVISION:-ed25f3a31f01632728cabb09d1542f84ab7b0056}"

# CVAT review server. Write credentials: the publish deletes tasks.
CVAT_HOST="${CVAT_HOST:-http://localhost:8081}"
CVAT_USER="${CVAT_USER:-mt}"
# Provenance is carried at TWO levels so machine output and human truth can
# never be mistaken for each other (docs/CVAT_GUIDE.md):
#   - separate PROJECTS: the task lists never interleave, and the answer-key
#     project paints every label one uniform green (#2ecc71) while the
#     pipeline project keeps distinct per-class colors;
#   - task-name SUFFIXES within each project, which --replace keys off.
CVAT_PIPELINE_PROJECT="${CVAT_PIPELINE_PROJECT:-OUR PIPELINE — machine pre-annotations (2D)}"
CVAT_GT_PROJECT="${CVAT_GT_PROJECT:-nuScenes GT — HUMAN answer key (2D)}"
CVAT_GT_LABEL_COLOR="${CVAT_GT_LABEL_COLOR:-#2ecc71}"
CVAT_PIPELINE_SUFFIX="${CVAT_PIPELINE_SUFFIX:-— OUR PIPELINE output}"
CVAT_GT_SUFFIX="${CVAT_GT_SUFFIX:-— nuScenes HUMAN answer key}"

echo "DHAKASCENES_VRAM_CAP_MIB=${DHAKASCENES_VRAM_CAP_MIB:-<unset: physical card is the ceiling, fit claims verified:false (C1)>}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

ALL_STEPS=(0 1 3 4 5 6 7 8 eval viz cvat)
STEPS=()
SCENES=()
WANT_CVAT=1
CLEAN_SLATE=0

while [ $# -gt 0 ]; do
  case "$1" in
    0|1|3|4|5|6|7|8|eval|viz|cvat) STEPS+=("$1") ;;
    all) STEPS+=("${ALL_STEPS[@]}") ;;
    --clean-slate) CLEAN_SLATE=1 ;;
    --no-cvat) WANT_CVAT=0 ;;
    --scenes)
      shift
      while [ $# -gt 0 ] && [[ "$1" != --* ]]; do SCENES+=("$1"); shift; done
      continue
      ;;
    -h|--help) sed -n '2,60p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument '$1' (steps: ${ALL_STEPS[*]} all; flags: --clean-slate --no-cvat --scenes)" >&2; exit 2 ;;
  esac
  shift
done
[ ${#STEPS[@]} -eq 0 ] && STEPS=("${ALL_STEPS[@]}")
if [ "$WANT_CVAT" = 0 ]; then
  filtered=()
  for s in "${STEPS[@]}"; do [ "$s" = cvat ] || filtered+=("$s"); done
  STEPS=(${filtered[@]+"${filtered[@]}"})
fi

# --scenes rides into every stage and every export that accepts it.
SCENE_ARGS=()
[ ${#SCENES[@]} -gt 0 ] && SCENE_ARGS=(--scenes "${SCENES[@]}")

# ---------------------------------------------------------------------------
# Paths, lock, log
# ---------------------------------------------------------------------------

# work_root comes from the path contract (§1.8), never from string concatenation
# off cwd. This also fails early and loudly if configs/paths.yaml is broken.
WORK_ROOT="$("$PY" -c 'import sys; sys.path.insert(0, "."); from pipeline.common.paths import load_paths; print(load_paths(sys.argv[1]).work_root)' "$PATHS_CONFIG")" || {
  echo "could not resolve work_root from $PATHS_CONFIG" >&2; exit 2; }

mkdir -p "$WORK_ROOT/logs"

# One run at a time. Two concurrent runs would interleave writes into the same
# stage trees and the markers would describe neither of them.
#
# `9<>` and not `9>`: opening with `>` TRUNCATES, which would erase the current
# holder's identity line before we are entitled to read it — and the whole point
# of that line is to be readable by the run that just got refused.
LOCK="$WORK_ROOT/.run_stages.lock"
exec 9<>"$LOCK"
if ! flock -n 9; then
  echo "another run_stages.sh holds $LOCK — refusing to overwrite its outputs" >&2
  echo "  holder: $(cat "$LOCK" 2>/dev/null || echo '<no identity line>')" >&2
  echo "  live processes:" >&2
  pgrep -af 'run_stages\.sh|pipeline/stage[0-9]' | sed 's/^/    /' >&2 || echo "    (none — see below)" >&2
  echo "  If no process is listed, a stray child still holds the fd; find it with:" >&2
  echo "    lsof $LOCK" >&2
  exit 2
fi
# Now that the lock is ours, say who we are. A refused run prints this back.
printf 'pid %s  started %s  steps: %s\n' "$$" "$(date -Is)" "${STEPS[*]}" > "$LOCK"

LOG="$WORK_ROOT/logs/run_$(date +%Y%m%d_%H%M%S).log"
# `exec 9>&-` INSIDE the process substitution: a child inherits every open fd,
# and an inherited fd holds the flock open independently of this shell. Without
# it, `tee` keeps the lock alive after this script dies and the next run is
# refused by a ghost. Same reason every step below runs with `9>&-`.
exec > >(exec 9>&-; tee -a "$LOG") 2>&1
echo "log: $LOG"
echo "steps: ${STEPS[*]}${SCENES[0]+   scenes: ${SCENES[*]}}"

# ---------------------------------------------------------------------------
# Clean slate (--clean-slate)
# ---------------------------------------------------------------------------
#
# Deliberately AFTER the lock: wiping a tree that another run is mid-write on
# is the one way to produce an output nobody can reason about afterwards.
#
# What it removes and what it spares:
#   removed  every stage tree under work_root (0-8), the COCO exports, the
#            metrics, the renders — everything this pipeline generates;
#   removed  every task and project on the CVAT server, if `cvat` is a step;
#   SPARED   work_root/logs — the audit trail of previous runs, which is the
#            only record of what the wiped outputs used to say;
#   SPARED   out_root/priors — an INPUT to Stage 6, not an output of it. No
#            step in this chain regenerates it, so wiping it would break the
#            run it is supposed to clean up for;
#   SPARED   the nuScenes dataroot, which is read-only by contract (§1.8).
if [ "$CLEAN_SLATE" = 1 ]; then
  echo
  echo "=== CLEAN SLATE  $(date +%H:%M:%S)"
  for d in stage0_data_probe stage1_ingestion stage2_ood \
           stage3_proposals stage4_masks stage5_lift \
           stage6_cluster stage7_track stage8_inflate \
           cvat_export cvat_export_gt cvat_export_3d \
           metrics viz viz_boxes; do
    if [ -e "$WORK_ROOT/$d" ]; then
      rm -rf "${WORK_ROOT:?}/$d"
      echo "  removed  $WORK_ROOT/$d"
    fi
  done
  echo "  spared   $WORK_ROOT/logs (audit trail), $(dirname "$WORK_ROOT")/out/priors (Stage 6 input)"

  # The CVAT purge only runs when this run intends to republish. Emptying the
  # review server and then not refilling it is never what anyone wants.
  if [[ " ${STEPS[*]} " == *" cvat "* ]]; then
    if [ -z "${CVAT_PASSWORD:-}" ]; then
      echo "!!! CVAT_PASSWORD unset — skipping the CVAT purge; the server keeps its old tasks" >&2
    else
      run_step_early_rc=0
      "$PY" scripts/cvat_purge.py --host "$CVAT_HOST" --user "$CVAT_USER" --yes 9>&- || run_step_early_rc=$?
      [ "$run_step_early_rc" -ne 0 ] && {
        echo "!!! CVAT purge failed rc=$run_step_early_rc — refusing to continue into a publish" >&2
        exit 2
      }
    fi
  else
    echo "  CVAT untouched (no 'cvat' step in this run)"
  fi
fi

# ---------------------------------------------------------------------------
# Step bookkeeping and the three-state exit contract
# ---------------------------------------------------------------------------

STEP_NAMES=()
STEP_STATUS=()
STEP_SECS=()
DEGRADED_SEEN=0
ABORTED=""

marker_state() {  # stage dir -> clean | degraded | none
  if   [ -e "$1/_SUCCESS" ];          then echo clean
  elif [ -e "$1/_SUCCESS.degraded" ]; then echo degraded
  else echo none
  fi
}

# Stage 1 is upstream of everything here and was already flagged before this
# script ever ran (C16: 2 of 10 scenes over the sector-rejection threshold), so
# the opt-in usually starts armed. On a clean substrate it starts disarmed and
# only arms if something in THIS run degrades.
for upstream_dir in "$WORK_ROOT/stage1_ingestion" "$WORK_ROOT/stage0_data_probe"; do
  [ "$(marker_state "$upstream_dir")" = degraded ] && DEGRADED_SEEN=1
done
[ "$DEGRADED_SEEN" = 1 ] && echo "upstream is DEGRADED on disk — stages will run with --accept-degraded-upstream (C16, recorded in each manifest)"

# Expands to `--accept-degraded-upstream` once anything upstream is flagged,
# and to nothing at all before that.
ACC=()
acc() { if [ "$DEGRADED_SEEN" = 1 ]; then ACC=(--accept-degraded-upstream); else ACC=(); fi; }

# run_step <label> <severity> <command...>
#   soft        any rc is recorded and the chain continues (a diagnostic — no
#               markers, nothing downstream consumes it)
#   fatal       rc 0 continues, anything else aborts (an export or a publish:
#               no three-state contract, it either worked or it did not)
#   <stage dir> a pipeline stage: the full three-state contract, VERIFIED
#               against the marker the stage actually wrote
#
# THE EXIT CODE IS NOT TRUSTED ON ITS OWN. Python exits 1 on an uncaught
# traceback, and a stage CLI exits 1 to mean DEGRADED — the same number for
# "complete, quality-flagged" and for "crashed on import". Believing the code
# alone made this wrapper announce a stack trace as "output is COMPLETE" and
# carry on, which is exactly the silent-wrong-output failure the whole project
# is built to refuse. So rc 1 is only DEGRADED if the degraded marker is on
# disk; a stage that claims rc 1 and left no marker CRASHED, and the chain stops.
run_step() {
  local label="$1" severity="$2"; shift 2
  local start=$SECONDS rc=0
  echo
  echo "=== $label  $(date +%H:%M:%S)"
  # `9>&-` closes the lock fd in the child. A stage that outlives this shell
  # (killed wrapper, orphaned python) would otherwise keep the flock held and
  # the next run would be refused by a process that is no longer running one.
  "$@" 9>&-
  rc=$?
  local elapsed=$((SECONDS - start))
  STEP_NAMES+=("$label"); STEP_SECS+=("$elapsed")

  if [ "$severity" = soft ]; then
    if [ $rc -eq 0 ]; then STEP_STATUS+=("ok")
    else STEP_STATUS+=("warn rc=$rc"); echo "--- $label: rc=$rc (diagnostic, no marker, chain continues)"; fi
    return 0
  fi

  if [ "$severity" = fatal ]; then
    if [ $rc -eq 0 ]; then
      STEP_STATUS+=("ok"); echo "--- $label: OK  (${elapsed}s)"; return 0
    fi
    STEP_STATUS+=("FAILED rc=$rc"); ABORTED="$label"
    echo "!!! $label: FAILED rc=$rc — aborting the chain." >&2
    return "$rc"
  fi

  # severity is a stage output directory: cross-examine the exit code.
  local state; state="$(marker_state "$severity")"
  case $rc in
    0)
      if [ "$state" != clean ]; then
        STEP_STATUS+=("BROKEN rc=0 marker=$state"); ABORTED="$label"
        echo "!!! $label: exited 0 but its marker is '$state', not 'clean'." >&2
        echo "!!! A stage that reports success without writing _SUCCESS has not produced" >&2
        echo "!!! something downstream may consume. Aborting rather than guessing." >&2
        return 2
      fi
      STEP_STATUS+=("ok"); echo "--- $label: OK  (${elapsed}s)"
      ;;
    1)
      if [ "$state" != degraded ]; then
        STEP_STATUS+=("CRASHED rc=1 no marker"); ABORTED="$label"
        echo "!!! $label: exited 1 with NO degraded marker in $severity." >&2
        echo "!!! rc 1 means DEGRADED only when the stage wrote _SUCCESS.degraded to say so." >&2
        echo "!!! With no marker this is a crash (python exits 1 on an uncaught exception)," >&2
        echo "!!! not a quality flag. Read the traceback above; the chain stops here." >&2
        return 2
      fi
      STEP_STATUS+=("degraded")
      DEGRADED_SEEN=1
      echo "--- $label: DEGRADED  (${elapsed}s) — output is COMPLETE and quality-flagged;"
      echo "    causes are in the _SUCCESS.degraded marker; downstream stages now get --accept-degraded-upstream"
      ;;
    *)
      STEP_STATUS+=("REFUSED rc=$rc")
      ABORTED="$label"
      echo "!!! $label: REFUSED rc=$rc — the stage wrote NOTHING and left no marker." >&2
      echo "!!! aborting the chain; the CVAT publish is suppressed so no half-built tree reaches review." >&2
      return "$rc"
      ;;
  esac
  return 0
}

# Stage 8 consumes "whatever produced the boxes". Prefer Stage 7's tracked
# boxes; fall back to Stage 6 when Stage 7 has no marker (e.g. a subset run
# that skipped it) rather than refusing on a missing directory.
boxes_dir() {
  if [ "$(marker_state "$WORK_ROOT/stage7_track")" != none ]; then
    echo "$WORK_ROOT/stage7_track"
  else
    echo "$WORK_ROOT/stage6_cluster"
  fi
}

# ---------------------------------------------------------------------------
# The steps
# ---------------------------------------------------------------------------

for s in "${STEPS[@]}"; do
  [ -n "$ABORTED" ] && break
  case "$s" in

    # Stages 0 and 1 take no --accept-degraded-upstream: Stage 0 has no
    # upstream at all, and Stage 1's upstream is Stage 0's allowlist, which it
    # reads directly. Both use the same three-state exit contract, so a Stage 0
    # that excludes a scene (rc 1) or a Stage 1 that falls back on a sector
    # (rc 1) flags the run and keeps going, exactly like the later stages.
    # `-m`, not a file path: unlike stages 3-8 these two have no sys.path
    # bootstrap of their own, so running them by path dies on `import pipeline`
    # before argparse ever sees a flag. Their own docstrings document the
    # module form; `cd "$REPO"` above is what puts the package on sys.path.
    0)  run_step "STAGE 0 (data probe: scene allowlist)" "$WORK_ROOT/stage0_data_probe" \
          "$PY" -m pipeline.stage0_data_probe.probe || break
        ;;

    1)  run_step "STAGE 1 (ingestion: keyframe index + ground-filtered clouds)" "$WORK_ROOT/stage1_ingestion" \
          "$PY" -m pipeline.stage1_ingestion.ingest \
            ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;

    3)  acc
        run_step "STAGE 3 (proposal_2d: $PROPOSAL_MODEL_ID)" "$WORK_ROOT/stage3_proposals" \
          "$PY" pipeline/stage3_proposals/proposals.py \
            --model-id "$PROPOSAL_MODEL_ID" --revision "$PROPOSAL_REVISION" \
            ${ACC[@]+"${ACC[@]}"} ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;

    4)  acc
        run_step "STAGE 4 (mask_2d: $MASK_MODEL_ID)" "$WORK_ROOT/stage4_masks" \
          "$PY" pipeline/stage4_masks/masks.py \
            --model-id "$MASK_MODEL_ID" --revision "$MASK_REVISION" \
            ${ACC[@]+"${ACC[@]}"} ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;

    5)  acc
        run_step "STAGE 5 (2D->3D lift)" "$WORK_ROOT/stage5_lift" \
          "$PY" pipeline/stage5_lift/lift.py \
            ${ACC[@]+"${ACC[@]}"} ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;

    6)  acc
        run_step "STAGE 6 (cluster -> 3D boxes)" "$WORK_ROOT/stage6_cluster" \
          "$PY" pipeline/stage6_cluster/cluster.py \
            ${ACC[@]+"${ACC[@]}"} ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;

    7)  acc
        run_step "STAGE 7 (track; reid facebook/dinov2-small)" "$WORK_ROOT/stage7_track" \
          "$PY" pipeline/stage7_track/track.py \
            --reid-revision "$REID_REVISION" \
            ${ACC[@]+"${ACC[@]}"} ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;

    8)  acc
        run_step "STAGE 8 (inflate against priors; boxes from $(basename "$(boxes_dir)"))" "$WORK_ROOT/stage8_inflate" \
          "$PY" pipeline/stage8_inflate/inflate.py \
            --boxes-dir "$(boxes_dir)" \
            ${ACC[@]+"${ACC[@]}"} ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;

    eval)
        # The COCO export is what the CVAT publish uploads, so it is FATAL even
        # though it writes no marker: publishing stale pre-annotations against a
        # fresh pipeline run is exactly the silent-wrong-output failure this
        # project is built to refuse.
        run_step "EXPORT cvat_export (Stage 3 boxes + Stage 4 masks -> COCO)" fatal \
          "$PY" scripts/export_cvat_coco.py ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break

        # The GT twins are the dataset's own human labels projected 3D->2D. They
        # do not depend on any pipeline output, so they are regenerated only when
        # missing (FORCE_GT_EXPORT=1 to rebuild anyway).
        if [ "${FORCE_GT_EXPORT:-0}" = 1 ] || ! compgen -G "$WORK_ROOT/cvat_export_gt/*/instances.json" > /dev/null; then
          run_step "EXPORT cvat_export_gt (nuScenes human answer key -> COCO)" soft \
            "$PY" scripts/export_gt_coco.py ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"}
        else
          echo; echo "=== EXPORT cvat_export_gt: present, skipped (FORCE_GT_EXPORT=1 to rebuild)"
        fi

        run_step "METRICS paint (Phase-7 gate number)" soft \
          "$PY" scripts/paint_metrics.py ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"}
        run_step "METRICS eval_2d (our boxes vs human answer key)" soft \
          "$PY" scripts/eval_2d.py ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"}
        run_step "METRICS eval_3d (our 3D boxes vs human 3D boxes)" soft \
          "$PY" scripts/eval_3d.py ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"}
        ;;

    viz)
        run_step "VIZ render_annotations (stages 3-5 on the images + BEV)" soft \
          "$PY" scripts/render_annotations.py ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"}
        run_step "VIZ render_boxes_bev (our 3D boxes vs human, BEV)" soft \
          "$PY" scripts/render_boxes_bev.py ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"}
        ;;

    cvat)
        if [ -z "${CVAT_PASSWORD:-}" ]; then
          echo "!!! CVAT_PASSWORD is unset (.env SECRET block) — cannot publish" >&2
          STEP_NAMES+=("CVAT publish"); STEP_STATUS+=("skipped: no CVAT_PASSWORD"); STEP_SECS+=(0)
          continue
        fi
        # The pre-annotations must be at least as new as the Stage 4 output they
        # claim to describe; a subset run of `cvat` alone would otherwise upload
        # a previous run's boxes under this run's name.
        if [ ! -e "$WORK_ROOT/cvat_export" ] || \
           [ "$WORK_ROOT/stage4_masks/run_manifest.json" -nt "$WORK_ROOT/cvat_export" ]; then
          run_step "EXPORT cvat_export (stale or missing — rebuilding before publish)" fatal \
            "$PY" scripts/export_cvat_coco.py ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        fi
        # --replace DELETES each "<scene> — OUR PIPELINE output" task and
        # recreates it. The answer-key twins live in a DIFFERENT project and
        # carry a different name, so they survive untouched (C13).
        run_step "CVAT publish ($CVAT_HOST -> '$CVAT_PIPELINE_PROJECT')" fatal \
          "$PY" scripts/cvat_setup.py \
            --host "$CVAT_HOST" --user "$CVAT_USER" \
            --project "$CVAT_PIPELINE_PROJECT" \
            --task-suffix "$CVAT_PIPELINE_SUFFIX" --replace \
            ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break

        # The answer-key twins. Deliberately NOT --replace: these are the
        # dataset's own human labels, they do not change between runs, and
        # cvat_setup refuses --replace on this suffix anyway (C13). Existing
        # twins are skipped, so this is a no-op unless they are missing —
        # which is exactly the case after --clean-slate.
        if [ ! -e "$WORK_ROOT/cvat_export_gt" ]; then
          run_step "EXPORT cvat_export_gt (needed for the answer-key twins)" fatal \
            "$PY" scripts/export_gt_coco.py ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        fi
        run_step "CVAT publish twins ($CVAT_HOST -> '$CVAT_GT_PROJECT')" fatal \
          "$PY" scripts/cvat_setup.py \
            --host "$CVAT_HOST" --user "$CVAT_USER" \
            --project "$CVAT_GT_PROJECT" --label-color "$CVAT_GT_LABEL_COLOR" \
            --export-dir cvat_export_gt --task-suffix "$CVAT_GT_SUFFIX" \
            ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;
  esac
done

# ---------------------------------------------------------------------------
# Summary — the manifests are the record, but a human needs one screen
# ---------------------------------------------------------------------------

echo
echo "==================== SUMMARY $(date +%H:%M:%S) ===================="
for i in "${!STEP_NAMES[@]}"; do
  printf '  %-9s %5ss  %s\n' "${STEP_STATUS[$i]}" "${STEP_SECS[$i]}" "${STEP_NAMES[$i]}"
done

echo
echo "  markers on disk:"
for d in stage0_data_probe stage1_ingestion stage3_proposals stage4_masks stage5_lift stage6_cluster stage7_track stage8_inflate; do
  [ -d "$WORK_ROOT/$d" ] || continue
  printf '    %-18s %s\n' "$d" "$(marker_state "$WORK_ROOT/$d")"
done

if [ -f "$WORK_ROOT/metrics/paint_metrics.json" ]; then
  echo
  echo "  metrics: $WORK_ROOT/metrics/  (paint / detect2d / detect3d)"
fi
echo "  log    : $LOG"

if [ -n "$ABORTED" ]; then
  echo
  echo "ABORTED at: $ABORTED  (rc>=2 = REFUSED: nothing written, no marker)"
  echo "=== RUN_INCOMPLETE $(date +%H:%M:%S)"
  exit 1
fi

if [ "$DEGRADED_SEEN" = 1 ]; then
  echo
  echo "  DEGRADED: the run is COMPLETE and quality-flagged. Every consumer recorded"
  echo "  accepted_degraded_upstream: true in its own run_manifest.json (C16)."
fi
echo "  review : $CVAT_HOST -> Projects -> 'DhakaScenes pilot — stages 3+4 (2D review)'"
echo "=== ALL_STEPS_DONE $(date +%H:%M:%S)"
