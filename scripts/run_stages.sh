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
# rebuildable by any committed script. Ordinary publishes are the additive
# exception (C30): task names carry the run that produced them — "<scene>
# <suffix> [<tag>]" — so a publish lands BESIDE the previous runs' tasks
# instead of deleting them, and a republish of the SAME run skips its own
# names. The bill for side-by-side review is
# server storage that only --cvat-replace or --clean-slate ever reclaims —
# cvat3d tasks in particular hold ~45-80 MB of point-cloud archive per scene
# PER KEPT RUN. --cvat-replace deletes every "— OUR PIPELINE output" task from
# the pipeline projects — every scene, every run tag, and the legacy untagged
# names — before publishing fresh; the "— nuScenes HUMAN answer
# key" tasks are never touched by an ordinary run OR by --cvat-replace
# (wipe_targets never matches their suffix; C13).
#
# Usage:
#   scripts/run_stages.sh                 # 0 1 3 4 5 6 7 8, eval, viz, cvat
#   scripts/run_stages.sh --clean-slate   # the same, from an empty work tree and an empty CVAT
#   scripts/run_stages.sh 3 4 5           # a subset, in the order given
#   scripts/run_stages.sh 4 5 6 7 8 eval cvat   # resume after a completed stage 3
#   scripts/run_stages.sh cvat            # republish to CVAT from what is on disk
#   scripts/run_stages.sh --no-cvat       # full chain, no publish (2D OR 3D)
#   scripts/run_stages.sh cvat cvat3d --cvat-replace  # wipe every previous run's
#                                         # pipeline-output tasks, then publish
#                                         # (answer keys untouched; C30)
#   scripts/run_stages.sh --scenes scene-0061   # one scene, end to end
#   scripts/run_stages.sh 0 1 3 3b 4 5 6 7 8 eval   # with the Stage 3b pass
#   VLM_CHECK=1 scripts/run_stages.sh 0 1 3 3b 4 5 6 7 8 eval  # ...and the VLM check, in front of Stage 4
#   PRINT_STEPS=1 scripts/run_stages.sh all 3c      # what WOULD run, on which tree; runs nothing
#
#   MASK_MODEL_ID=facebook/sam2.1-hiera-large MASK_REVISION=<sha> scripts/run_stages.sh 4
#
# Steps: 0 1 3 4 5 6 7 8 | eval (COCO export + metrics) | viz (PNG renders)
#      | cvat (2D review tasks) | cvat3d (point-cloud cuboid tasks)
# Both cvat steps PUBLISH to the review server and both are suppressed by
# --no-cvat, which is also what decides whether --clean-slate purges the server.
#
# Plus SIX opt-in arms: accepted as arguments and ordered by hand, deliberately
# absent from the default list and from `all`, because an opt-in arm in the
# default chain would silently change what "the pipeline" means and a baseline
# run has to stay the one nobody had to ask for.
#   3b  12 Hz identity propagation (C27) -> stage3b_track2d. Stage 4 prefers it
#       when it is fresh and covers this run's scenes; it is also the ONLY
#       source of the track ids a per_track 3c keys on.
#   3f  arm B proposals (C28) -> stage3_finetuned: the fine-tuned checkpoint,
#       its own class map, and a SUPERSET taxonomy.
#   3m  the merge (C28): arm A + arm B -> stage3_merged, which then outranks
#       both. Takes no --scenes — it pairs whole trees or refuses.
#   3c  the VLM label check -> stage3_checked, which outranks everything.
#   road  the road-surface layer (C33) -> stage_road: SAM 3 "paved road" masks
#         per camera + plane-gated LiDAR point labels. Opt-in; consumes Stage 1
#         alone. Publish with `cvatroad` (own project, RLE masks — polygons
#         cannot hold the vehicle-shaped holes).
#       Steps run in the order TYPED, so `all 3c` checks the labels only after
#       Stage 4, the export and the publish have consumed the unchecked ones.
#       VLM_CHECK=1 is the fix.
#
# Environment (each knob is declared, with the why of its default, below):
#   VLM_CHECK=1          run ONE 3c immediately before the first Stage 4 —
#                        wherever the token was typed, and even if it was not.
#                        =0 removes a typed 3c and says so. Unset (the default)
#                        leaves the step list exactly as typed.
#   VLM_USE_CHECKED=0    do not let a stage3_checked tree feed Stage 4 or the
#                        exporters this run. Omitting the step never did that:
#                        a PREVIOUS run's checked tree captures Stage 4 with
#                        nothing typed, and this is what closes that.
#   VLM_CHECK_MODE=      per_track (the default: ONE VLM call per 3b track id,
#     per_track|per_box  propagated to every box of the track) or per_box (one
#                        call per box, check.py's own default). per_track needs
#                        track ids, so over a plain stage3_proposals tree the 3c
#                        arm demotes to per_box out loud rather than letting
#                        check.py refuse in the middle of a chain.
#   VLM_TRACK_RETRIES=N  ranked crops tried when a track's answer comes back
#                        unclear or errored (default 3).
#   VLM_SKIP_CLASSES     comma-separated caption phrases whose boxes 3c never
#                        VLM-checks (default "a pedestrian,a motorcycle,a bicycle"
#                        — C31: the checker systematically flipped rider crops
#                        pedestrian -> motorcycle against its own prompt rule,
#                        2706 boxes in the first pilot_1632 run, 0 in reverse;
#                        C31 amendment with C34: it also turned 180 bicycles
#                        into rickshaws, 151 by track propagation).
#                        Set to "" to check every class.
#   VLM_ALLOW_UNTRACKED=1  check boxes carrying no track id per-box instead of
#                        refusing the run.
#   VLM_ALLOW_SHARED_GPU=1  let 3c share the GPU instead of refusing: an
#                        injected mid-chain 3c otherwise dies on a stray CUDA
#                        process and takes the whole chain with it.
#   MASK_TEXT_PROMPT=1   Stage 4 prompts SAM 3 with each box's class phrase as
#                        well as the box (provider sam3_text — a different model
#                        path, not a tweak of the default one).
#   CVAT_RUN_TAG=<tag>   override the run tag the publishes stamp into task
#                        names; default is the SOURCE manifest's mtime (as
#                        %Y%m%dT%H%M%S — stage4_masks for cvat, stage8_inflate
#                        for cvat3d). Same run, same tag, no duplicates (C30).
#   PRINT_STEPS=1        print the resolved step list and the tree Stage 4 would
#                        be handed, then exit having run nothing. Refuses to run
#                        together with --clean-slate.
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

# The class map is a property of the CHECKPOINT'S VOCABULARY, not of the run:
# yolo11x predicts COCO-80, yolov8x-oiv7 predicts Open Images V7's 601, and
# `ClassMap.assert_covers` refuses either map against the other checkpoint. It
# is selected here so the pairing cannot be got wrong by forgetting a flag.
#   PROPOSAL_MODEL_ID=/home/mt/dhakascenes/cache/checkpoints/yolov8x-oiv7.pt \
#     scripts/run_stages.sh --clean-slate
case "$(basename "$PROPOSAL_MODEL_ID")" in
  *oiv7*) PROPOSAL_CLASS_MAP="${PROPOSAL_CLASS_MAP:-configs/oiv7_to_phrase_nuscenes.yaml}" ;;
  *)      PROPOSAL_CLASS_MAP="${PROPOSAL_CLASS_MAP:-configs/coco_to_phrase_nuscenes.yaml}" ;;
esac

# Arm B (opt-in steps 3f/3m; docs/RUNNING.md two-arm design, DECISIONS C28).
# The artifact basename MUST start with `yolo`: infer_proposal_provider() routes
# on that prefix and would otherwise treat the file as a hub id. Its taxonomy is
# a SUPERSET of arm A's (v2's phrases first, verbatim), which is what keeps arm
# A's caption a byte prefix and its phrase spans valid in the merged rows.
ARMB_MODEL_ID="${ARMB_MODEL_ID:-local_yolox_build/artifacts/yolo11x-rsud20k-armb.pt}"
ARMB_REVISION="${ARMB_REVISION:-armb-r1280-4}"
ARMB_TAXONOMY="${ARMB_TAXONOMY:-configs/taxonomy_pilot_dhaka.yaml}"
ARMB_CLASS_MAP="${ARMB_CLASS_MAP:-configs/rsud20k_to_phrase_dhaka.yaml}"

# Stage 3c (opt-in) — VLM label check: Nemotron 3 Nano Omni through a local
# llama.cpp llama-server. The model owns the WHOLE GPU while 3c runs (check.py
# refuses a busy GPU), so 3c never shares a chain position with another GPU
# stage; the wrapper's sequential chain already guarantees that.
VLM_GGUF="${VLM_GGUF:-/home/mt/dhakascenes/cache/checkpoints/nemotron-omni/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf}"
VLM_MMPROJ="${VLM_MMPROJ:-/home/mt/dhakascenes/cache/checkpoints/nemotron-omni/mmproj-BF16.gguf}"
LLAMA_SERVER_BIN="${LLAMA_SERVER_BIN:-/home/mt/dhakascenes/tools/llama.cpp/build/bin/llama-server}"
VLM_PARALLEL="${VLM_PARALLEL:-4}"
VLM_N_CPU_MOE="${VLM_N_CPU_MOE:-8}"   # MoE layers whose experts sit in system RAM (the 24 GB fit knob)

# The check is opt-in TWICE OVER, because "do not run 3c" and "do not believe a
# 3c that already ran" are different requests and collapsing them into one flag
# gets one of the two wrong:
#   VLM_CHECK       the PRODUCTION gate. Empty — the default — is exactly
#                   today's behavior: 3c runs if and only if its token was
#                   typed, in the position it was typed. 1 puts ONE 3c
#                   immediately before the first Stage 4 (STEPS is executed in
#                   typed order and never sorted, so the habitual `all 3c`
#                   otherwise checks the labels only after Stage 4, the COCO
#                   export and the CVAT publish have consumed the unchecked
#                   ones). 0 is the kill switch: a typed 3c is removed, loudly.
#   VLM_USE_CHECKED the CONSUMPTION gate (default 1). 0 stops Stage 4 and the
#                   exporters from reading a stage3_checked tree a PREVIOUS run
#                   left standing — which omitting the step never did, and
#                   which is the hole this pair exists to close. Applied inside
#                   select_stage3_dir_for_4 so the pre-scan, the Stage 4 arm
#                   and export_taxonomy cannot disagree about it.
VLM_CHECK="${VLM_CHECK:-}"
VLM_USE_CHECKED="${VLM_USE_CHECKED:-1}"
# per_track asks the VLM ONCE per 3b track id and propagates that verdict to
# every box of the track: 46,722 boxes / 5,795 tracks on the live Dhaka tree —
# ~7.5x fewer calls, and the sub-floor boxes get a verdict for the first time.
# It needs track ids, so it only pays over a 3b tree (or a merge of one); the 3c
# arm demotes to per_box, out loud, when the tree in hand is stage3_proposals,
# rather than meeting check.py's coverage refusal mid-chain. per_box is
# check.py's own default and the A/B control.
VLM_CHECK_MODE="${VLM_CHECK_MODE:-per_track}"
VLM_TRACK_RETRIES="${VLM_TRACK_RETRIES:-3}"   # ranked crops tried when a track's answer is unclear/error
# C31: classes 3c never asks the VLM about (comma-separated caption phrases;
# "" checks everything). The checker's one-way pedestrian -> motorcycle flips
# violated its own rider rule at scale, so both rider-adjacent classes sit out.
VLM_SKIP_CLASSES="${VLM_SKIP_CLASSES-a pedestrian,a motorcycle,a bicycle}"
# Both off: a refusal that names its cause beats a silently different run.
# 1 checks boxes that carry no track id per-box instead of refusing
# (--allow-untracked); 1 lets 3c share the GPU with another process instead of
# refusing (--allow-shared-gpu) — an injected mid-chain 3c makes a stray CUDA
# process abort the whole run, so this is the escape hatch for that case.
VLM_ALLOW_UNTRACKED="${VLM_ALLOW_UNTRACKED:-0}"
VLM_ALLOW_SHARED_GPU="${VLM_ALLOW_SHARED_GPU:-0}"

# ultralytics writes a settings.json at import time; without this it lands in
# $HOME/.config and prints a warning on every stage invocation.
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/home/mt/dhakascenes/cache/ultralytics}"
mkdir -p "$YOLO_CONFIG_DIR"

# mask_2d (Stage 4) — C19 default: SAM 3 tracker (licence granted 2026-08-13);
# SAM 2.1-L is the ungated alternate: facebook/sam2.1-hiera-large @ 665f8e2ad61cf5f53d65644ff27c8ee525124610
MASK_MODEL_ID="${MASK_MODEL_ID:-facebook/sam3}"
MASK_REVISION="${MASK_REVISION:-3c879f39826c281e95690f02c7821c4de09afae7}"
# Off by default, like every A/B knob here: 1 hands each box's class phrase
# ("a car") to SAM 3 alongside the box, which is a DIFFERENT provider
# (sam3_text) loading a different head — not a tweak to the default one. The
# default run therefore keeps meaning what it meant, and every archived
# Results/ number still describes the same invocation.
MASK_TEXT_PROMPT="${MASK_TEXT_PROMPT:-0}"

# track2d (Stage 3b, C27) — the SAME CHECKPOINT FAMILY as mask_2d, defaulted off
# it so there is ONE place to bump the SAM pin; Stage 3b loads only the video
# tracker classes, Stage 4 only the image ones, so the two are never resident
# together. Independently overridable when the tiers must differ, e.g. mask on
# sam3 + track on sam3.1 (C26):
#   TRACK2D_MODEL_ID=facebook/sam3.1 TRACK2D_REVISION=daa63191845a41281374e725f4c9e51c7a824460 \
#     scripts/run_stages.sh 3 3b 4
TRACK2D_MODEL_ID="${TRACK2D_MODEL_ID:-$MASK_MODEL_ID}"
TRACK2D_REVISION="${TRACK2D_REVISION:-$MASK_REVISION}"
# The two A/B knobs, off by default so `3b` alone is the plain recovery pass:
# 1 replaces a detected box with its propagated mask's tight box where the two
# agree (--refine-boxes); 1 drops Phase A entirely — no detector on the sweep
# frames and no mid-gap births (--no-sweep-detection).
TRACK2D_REFINE="${TRACK2D_REFINE:-0}"
TRACK2D_NO_SWEEPS="${TRACK2D_NO_SWEEPS:-0}"

# reid_embedding (Stage 7) — DINOv3 since 2026-08-14 (human-directed, after the
# 4-cell comparison saved under Results/). It measured IDENTICAL to
# facebook/dinov2-small on every accuracy metric; the switch is a choice, not a
# gain, and Results/COMPARISON.md is the evidence. DINOv3 is a GATED hub repo:
# Stage 7 needs an HF_TOKEN whose account has access, or it silently falls back
# to IoU-only tracking.
# Mandatory pin: track.py:1449 refuses (rc 2) with an unpinned id rather than
# tracking the default branch.
#
# The revision is PINNED PER CHECKPOINT: a sha belongs to one repo, so pairing
# DINOv3's id with DINOv2's sha is a hub 404 at best and the wrong weights at
# worst. Setting REID_MODEL_ID alone therefore picks the matching sha from the
# table below; setting REID_REVISION explicitly still wins.
#   REID_MODEL_ID=facebook/dinov2-small scripts/run_stages.sh 7 8 eval
REID_MODEL_ID="${REID_MODEL_ID:-facebook/dinov3-vits16-pretrain-lvd1689m}"
case "$REID_MODEL_ID" in
  facebook/dinov2-small)                    REID_PINNED=ed25f3a31f01632728cabb09d1542f84ab7b0056 ;;
  facebook/dinov3-vits16-pretrain-lvd1689m) REID_PINNED=114c1379950215c8b35dfcd4e90a5c251dde0d32 ;;
  facebook/dinov3-vitb16-pretrain-lvd1689m) REID_PINNED=5931719e67bbdb9737e363e781fb0c67687896bc ;;
  *)                                        REID_PINNED="" ;;
esac
REID_REVISION="${REID_REVISION:-$REID_PINNED}"
if [ -z "$REID_REVISION" ]; then
  echo "REID_MODEL_ID=$REID_MODEL_ID has no pinned revision in scripts/run_stages.sh —" >&2
  echo "  pass REID_REVISION=<hub commit sha> explicitly, or add it to the table there" >&2
  exit 2
fi

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
# C33: the road layer publishes into its OWN project — adding a surface label
# to the shared instance project would renumber its categories and cannot be
# reviewed under the same task names.
CVAT_ROAD_PROJECT="${CVAT_ROAD_PROJECT:-DhakaScenes road surface}"
CVAT_ROAD_SUFFIX="${CVAT_ROAD_SUFFIX:-(road, machine)}"
CVAT_GT_PROJECT="${CVAT_GT_PROJECT:-nuScenes GT — HUMAN answer key (2D)}"
CVAT_GT_LABEL_COLOR="${CVAT_GT_LABEL_COLOR:-#2ecc71}"
CVAT_PIPELINE_SUFFIX="${CVAT_PIPELINE_SUFFIX:-— OUR PIPELINE output}"
CVAT_GT_SUFFIX="${CVAT_GT_SUFFIX:-— nuScenes HUMAN answer key}"

echo "DHAKASCENES_VRAM_CAP_MIB=${DHAKASCENES_VRAM_CAP_MIB:-<unset: physical card is the ceiling, fit claims verified:false (C1)>}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

# ALL_STEPS is BOTH the default step list and what `all` expands to — it is not
# what validates an argument (the case below does that). 3b is therefore
# accepted by name but never runs unless it was asked for: an opt-in A/B arm in
# the default chain would silently change what "the pipeline" means.
ALL_STEPS=(0 1 3 4 5 6 7 8 eval viz cvat cvat3d)
OPT_IN_STEPS=(3b 3f 3m 3c road cvatroad)

# Which steps PUBLISH to the CVAT server. ONE definition, read by both the
# --no-cvat filter and the --clean-slate purge condition, because the two
# disagreed once and the disagreement is a one-way door: `cvat3d` was added to
# ALL_STEPS and to the purge condition while --no-cvat still stripped only the
# exact word `cvat`, so `--clean-slate --no-cvat` — documented above as "full
# chain, no publish" — deleted every task and project on the review server,
# including the 3D cuboid work no committed script can rebuild, and then
# republished 3D. The predicate is a PREFIX so a future cvat<N> cannot slip
# past either reader by being forgotten in one list.
is_cvat_step() { case "$1" in cvat*) return 0 ;; *) return 1 ;; esac; }
CVAT_STEPS=()
# opt-in publish steps (cvatroad) are publishes too: the refusal and the
# clean-slate message must name them or they misstate the publish set
for s in "${ALL_STEPS[@]}" "${OPT_IN_STEPS[@]}"; do is_cvat_step "$s" && CVAT_STEPS+=("$s"); done

STEPS=()
SCENES=()
WANT_CVAT=1
CVAT_REPLACE=0
CLEAN_SLATE=0

while [ $# -gt 0 ]; do
  case "$1" in
    0|1|3|3b|3f|3m|3c|4|5|6|7|8|road|eval|viz|cvat|cvat3d|cvatroad) STEPS+=("$1") ;;
    all) STEPS+=("${ALL_STEPS[@]}") ;;
    --clean-slate) CLEAN_SLATE=1 ;;
    --cvat-replace) CVAT_REPLACE=1 ;;
    --no-cvat) WANT_CVAT=0 ;;
    --scenes)
      shift
      while [ $# -gt 0 ] && [[ "$1" != --* ]]; do SCENES+=("$1"); shift; done
      continue
      ;;
    # Through line 153, the last line of the header — the line ABOVE
    # `set -uo pipefail`, and the number to re-check whenever a header line is
    # added or removed. The range stopped at 60 once and so cut off Usage and
    # the step list — the two things --help is for, and the only place the
    # reader is told what --no-cvat covers. It stopped at 81 until the opt-in
    # arms and the VLM/mask env knobs were documented.
    -h|--help) sed -n '2,153p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument '$1' (steps: ${ALL_STEPS[*]} all; opt-in: ${OPT_IN_STEPS[*]}; flags: --clean-slate --cvat-replace --no-cvat --scenes)" >&2; exit 2 ;;
  esac
  shift
done
[ ${#STEPS[@]} -eq 0 ] && STEPS=("${ALL_STEPS[@]}")
# --no-cvat means NO PUBLISH AT ALL: every step in CVAT_STEPS goes, not the one
# whose name happens to be spelled `cvat`.
if [ "$WANT_CVAT" = 0 ]; then
  filtered=()
  for s in "${STEPS[@]}"; do is_cvat_step "$s" || filtered+=("$s"); done
  STEPS=(${filtered[@]+"${filtered[@]}"})
fi

# --cvat-replace acts only inside a publish (the wipe is what the fresh tasks
# replace): a wipe flag on a run that publishes nothing would empty the review
# server and refill none of it — the exact loss the purge-arming predicate
# lesson. Refuse, naming the contradiction. Checked AFTER the --no-cvat filter
# so `--cvat-replace --no-cvat` lands here too.
if [ "$CVAT_REPLACE" = 1 ]; then
  has_publish=0
  for s in ${STEPS[@]+"${STEPS[@]}"}; do is_cvat_step "$s" && has_publish=1; done
  if [ "$has_publish" = 0 ]; then
    echo "--cvat-replace: this run has no publishing step (${CVAT_STEPS[*]}), so there is" >&2
    echo "  no publish for the wipe to make room for. Add cvat/cvat3d, or drop --no-cvat," >&2
    echo "  or drop the flag. (To empty the server without republishing: scripts/cvat_purge.py --yes)" >&2
    exit 2
  fi
fi

# PRINT_STEPS=1 is a DRY RUN: it prints the resolved step list and the tree
# Stage 4 would be handed, and exits before the first stage. The print itself
# lives further down (it has to wait for the pre-scan to resolve the tree), but
# the refusal below cannot: --clean-slate deletes every stage tree long before
# that point, and "describe this run without running it" must never be the
# command that empties the work root and the CVAT server first.
if [ "${PRINT_STEPS-}" = 1 ] && [ "$CLEAN_SLATE" = 1 ]; then
  echo "PRINT_STEPS=1 with --clean-slate: refusing." >&2
  echo "  PRINT_STEPS describes a run without performing it; --clean-slate would delete every" >&2
  echo "  stage tree (and purge CVAT) before the description could be printed. Drop one of them." >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# VLM_CHECK — WHERE 3c runs, not merely whether
# ---------------------------------------------------------------------------
#
# STEPS is executed in the order it was typed and is never sorted, so the
# habitual `all 3c` runs the check AFTER Stage 4, the COCO export and both CVAT
# publishes have already consumed the UNCHECKED tree: the relabels land in the
# NEXT run and in nothing this one produced. So VLM_CHECK=1 does not append a
# step — appending reproduces the exact trap the flag exists to fix — it puts
# one 3c immediately before the first Stage 4 and says so when that moved a
# token the operator typed somewhere else.
#
# This is deliberately its own site: after the `-eq 0` default fill (which must
# not see a list this rewrote) and after the --no-cvat filter (so the positions
# counted here are the ones that will run), but before the lock identity line
# and the banner, both of which must report the list that actually runs.
steps_contain() {  # <token> — is it in STEPS?
  local want="$1" s
  for s in ${STEPS[@]+"${STEPS[@]}"}; do [ "$s" = "$want" ] && return 0; done
  return 1
}

case "$VLM_CHECK" in
  '') : ;;   # unset: the typed token, and only the typed token, decides. Today's behavior.

  0)  # The kill switch wins over an explicitly typed token, and is announced —
      # the reverse (silently honouring the token) would make the flag a lie.
      if steps_contain 3c; then
        vlm_rebuilt=()
        for s in ${STEPS[@]+"${STEPS[@]}"}; do [ "$s" = 3c ] || vlm_rebuilt+=("$s"); done
        STEPS=(${vlm_rebuilt[@]+"${vlm_rebuilt[@]}"})
        echo "!!! VLM_CHECK=0 — the 3c step you typed was REMOVED from this run." >&2
        echo "!!!   Nothing will write stage3_checked. Whether an EXISTING stage3_checked tree" >&2
        echo "!!!   still feeds Stage 4 is a separate question: VLM_USE_CHECKED=0 answers it." >&2
      fi
      ;;

  1)  vlm_typed_pos=0
      vlm_i=0
      for s in ${STEPS[@]+"${STEPS[@]}"}; do
        vlm_i=$((vlm_i + 1))
        if [ "$s" = 3c ] && [ "$vlm_typed_pos" = 0 ]; then vlm_typed_pos=$vlm_i; fi
      done
      if steps_contain 4; then
        vlm_rebuilt=()
        vlm_inserted=0
        for s in ${STEPS[@]+"${STEPS[@]}"}; do
          [ "$s" = 3c ] && continue                        # every typed 3c goes...
          if [ "$s" = 4 ] && [ "$vlm_inserted" = 0 ]; then  # ...and exactly one comes back,
            vlm_rebuilt+=(3c); vlm_inserted=1               # before the FIRST Stage 4
          fi
          vlm_rebuilt+=("$s")
        done
        vlm_before="${STEPS[*]}"
        STEPS=(${vlm_rebuilt[@]+"${vlm_rebuilt[@]}"})
        if [ "$vlm_typed_pos" != 0 ] && [ "${STEPS[*]}" != "$vlm_before" ]; then
          echo "note: VLM_CHECK=1 moved 3c before Stage 4 (typed position $vlm_typed_pos)"
        fi
      else
        # No Stage 4 to protect, so there is no position to inject at, and a 3c
        # appended to a chain with no consumer would only look like one.
        if [ "$vlm_typed_pos" != 0 ]; then
          echo "note: 3c runs as typed; no stage-4 step to inject before"
        else
          echo "note: no stage-4 step in this run; 3c not injected"
        fi
      fi
      # Legal, and odd enough to say out loud: the check runs, writes its tree,
      # and this run's Stage 4 is under orders not to read it.
      if steps_contain 3c && [ "$VLM_USE_CHECKED" = 0 ]; then
        echo "note: 3c will run but its output will NOT be consumed by Stage 4 this run"
        echo "note:   (VLM_USE_CHECKED=0) — stage3_checked is written and left for a later run."
      fi
      ;;

  *)  echo "VLM_CHECK='$VLM_CHECK' is not a value this script knows." >&2
      echo "  <unset/empty> = the step list decides (today's behavior);  0 = remove a typed 3c;" >&2
      echo "  1 = run one 3c immediately before the first Stage 4. Refusing rather than guessing." >&2
      exit 2
      ;;
esac

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

# Same contract, for the has_ground_truth probe below: a raw capture carries an
# empty sample_annotation.json and must not get answer-key twins.
DATAROOT_META="$("$PY" -c 'import sys; sys.path.insert(0, "."); from pipeline.common.paths import load_paths; p=load_paths(sys.argv[1]); print(p.version_dir)' "$PATHS_CONFIG")" || {
  echo "could not resolve the metadata version dir from $PATHS_CONFIG" >&2; exit 2; }

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
  # Every generated tree, including the ones no step in this chain rebuilds:
  # a stage9_qa/ or pilot_run/ left standing describes the PREVIOUS run's
  # boxes while sitting beside this one's, which is worse than absent.
  for d in stage0_data_probe stage1_ingestion stage2_ood \
           stage3_proposals stage3b_track2d stage3_finetuned stage3_merged \
           stage3_checked \
           stage4_masks stage5_lift \
           stage6_cluster stage7_track stage8_inflate stage9_qa \
           stage_road cvat_export_road \
           cvat_export cvat_export_gt cvat_export_3d \
           metrics viz viz_boxes viz_3d pilot_run; do
    if [ -e "$WORK_ROOT/$d" ]; then
      rm -rf "${WORK_ROOT:?}/$d"
      echo "  removed  $WORK_ROOT/$d"
    fi
  done
  echo "  spared   $WORK_ROOT/logs (audit trail), $(dirname "$WORK_ROOT")/out/priors (Stage 6 input)"

  # The CVAT purge only runs when this run intends to republish. Emptying the
  # review server and then not refilling it is never what anyone wants — and
  # --no-cvat has already removed every publishing step from STEPS, so asking
  # STEPS (through the same predicate the filter used) is what keeps the two
  # answers identical.
  PURGE_CVAT=0
  for s in ${STEPS[@]+"${STEPS[@]}"}; do is_cvat_step "$s" && PURGE_CVAT=1; done
  if [ "$PURGE_CVAT" = 1 ]; then
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
    echo "  CVAT untouched (no publishing step — ${CVAT_STEPS[*]} — in this run)"
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

# The scene set a stage tree actually holds. Stage 4 enumerates its input with
# os.listdir(<stage3_dir>/scenes), so this is not a detail: it is the definition
# of how much of the run that tree can carry.
# A plain glob and not `find -printf`: if -printf were ever unavailable the find
# would fail, both trees would read as empty, `missing` would be empty, and the
# scope gate would silently go back to adopting whatever is there — the exact
# failure it exists to prevent. An unmatched `*/` is not a directory, so an empty
# scenes/ prints nothing.
scene_names() {
  [ -d "$1/scenes" ] || return 0
  ( cd "$1/scenes" && for d in */; do [ -d "$d" ] && printf '%s\n' "${d%/}"; done ) | sort
}

# scope_shortfall <candidate tree> <tree it would displace>
#   The scenes the candidate would DROP if it were adopted, space-separated and
#   empty when it drops none. ONE implementation, used by all three preference
#   steps below (3b over Stage 3, the merge over the arm, the check over the
#   merge), because three hand-written copies of a `comm` pipeline is three
#   chances to get the direction of the set difference wrong.
#
# The set the candidate is measured against is what THIS run would otherwise
# hand Stage 4 — the --scenes selection when there is one (a stage tree keeps
# the scene dirs earlier runs wrote, so under --scenes a narrower tree narrows
# nothing), and everything the displaced tree holds when there is not. Scenes
# neither tree has are Stage 4's refusal to make, not this guard's.
scope_shortfall() {
  local cand="$1" base="$2" want="" missing=""
  if [ ${#SCENES[@]} -gt 0 ]; then
    want="$(comm -12 <(printf '%s\n' "${SCENES[@]}" | sort -u) <(scene_names "$base"))"
  else
    want="$(scene_names "$base")"
  fi
  missing="$(comm -23 <(printf '%s\n' "$want" | sed '/^$/d') <(scene_names "$cand") | tr '\n' ' ')"
  printf '%s' "${missing% }"
}

# Stage 4 consumes "whatever produced the proposals". Stage 3b writes Stage 3's
# own schema into its own tree, so prefer it when it exists — but ONLY when it
# describes THIS Stage 3 output, in FULL.
#
# Three gates, all of them load-bearing:
#   marker     a tree with no completion marker is a partial write (§1.9);
#   freshness  a leftover stage3b_track2d older than the current stage3_proposals
#              must never capture a baseline run. Marker presence alone is not
#              enough, because --clean-slate is what usually removes the tree and
#              a resumed subset run never touches it; the manifest mtime is what
#              says which of the two was written last;
#   scope      a 3b tree built with `--scenes scene-0061` holds one scene, and
#              adopting it would silently narrow an entire full-chain run to that
#              scene with nothing said — Stage 4 enumerates os.listdir() and asks
#              no questions. Narrowing the run is not a decision this wrapper may
#              make unannounced, and neither is quietly dropping a recovery pass
#              the operator did ask for, so a short tree falls back to
#              stage3_proposals and says which scenes it was short of.
#              The set it is measured against is what THIS run will hand Stage 4:
#              the --scenes selection when there is one (a stage tree keeps the
#              scene dirs earlier runs wrote, so under --scenes the 3b tree is
#              legitimately shorter than stage3_proposals and narrows nothing),
#              and everything stage3_proposals holds when there is not. Scenes
#              neither tree has are Stage 4's refusal to make, not this guard's.
#
# The answer is computed by ONE function into ONE variable, called both by the
# C16 pre-scan below and by the Stage 4 arm, so the pre-scan cannot be looking at
# stage3_proposals while Stage 4 reads stage3b_track2d. It also arms C16 off the
# tree it selected: a degraded upstream that Stage 4 will actually consume has to
# arm --accept-degraded-upstream exactly as a degraded Stage 1 does, or Stage 4
# meets require_upstream with an empty ACC and REFUSES the run.
STAGE3_DIR_FOR_4=""
STAGE3_NOTE_SHOWN=""
# One dedupe latch per note, not one shared latch: the function is asked up to
# four times a run, and two different trees can be short of the SAME scene — a
# shared latch would print one of the two and silently swallow the other.
STAGE3M_NOTE_SHOWN=""
STAGE3C_NOTE_SHOWN=""
STAGE3B_STALE_NOTE_SHOWN=""
VLM_USE_CHECKED_NOTE_SHOWN=""
# The optional argument names the highest DERIVED tree the caller may be
# handed: `arms` (3/3b only — what the 3m merge may consume as arm A: handing
# it a fresh stage3_merged or stage3_checked would re-append arm B boxes into
# a tree that already carries them), `merged` (what 3c may consume: everything
# except its own previous output), or the default `checked` (everything —
# Stage 4 and the exporters).
select_stage3_dir_for_4() {
  local level="${1:-checked}"
  # The CONSUMPTION gate, applied HERE and nowhere else. This function is what
  # the C16 pre-scan, the Stage 4 arm and export_taxonomy all ask, so demoting
  # the level in one place is what makes VLM_USE_CHECKED=0 mean the same thing
  # to all three — a gate bolted onto the Stage 4 arm alone would leave the
  # exporters reading the tree Stage 4 was told to ignore.
  # Deliberately NOT wired to VLM_CHECK: not RUNNING 3c and disowning a 3c that
  # already ran are different requests (C29). With both vars unset this is a
  # no-op and a stale stage3_checked still inherits the run, exactly as today.
  if [ "${VLM_USE_CHECKED-1}" = 0 ] && [ "$level" = checked ]; then
    level=merged
    if [ -z "$VLM_USE_CHECKED_NOTE_SHOWN" ]; then
      VLM_USE_CHECKED_NOTE_SHOWN=1
      echo "note: VLM_USE_CHECKED=0 — a stage3_checked tree will NOT be consumed by this run." >&2
      echo "note:   Stage 4 and the exporters read the merged/arm tree instead. A 3c in THIS run" >&2
      echo "note:   still writes stage3_checked (VLM_CHECK=0 is what stops it running at all)." >&2
    fi
  fi
  local s3="$WORK_ROOT/stage3_proposals" s3b="$WORK_ROOT/stage3b_track2d" missing=""
  STAGE3_DIR_FOR_4="$s3"
  if [ "$(marker_state "$s3b")" != none ]; then
    if [ "$s3b/run_manifest.json" -nt "$s3/run_manifest.json" ]; then
      missing="$(scope_shortfall "$s3b" "$STAGE3_DIR_FOR_4")"
      if [ -n "$missing" ]; then
        if [ "$STAGE3_NOTE_SHOWN" != "$missing" ]; then
          STAGE3_NOTE_SHOWN="$missing"
          echo "note: $s3b is FRESHER than stage3_proposals but does not cover every scene" >&2
          echo "note:   this run will hand Stage 4. Missing from the 3b tree: $missing" >&2
          echo "note:   Stage 4 will read stage3_proposals instead — the run is NOT narrowed to the" >&2
          echo "note:   3b subset, and the recovered boxes are NOT in it either." >&2
          echo "note:   Re-run 3b over those scenes, or give this run the same --scenes, to use them." >&2
        fi
      else
        STAGE3_DIR_FOR_4="$s3b"
      fi
    elif [ -z "$STAGE3B_STALE_NOTE_SHOWN" ]; then
      # A 3b tree that exists but is OLDER than stage3_proposals is the silent
      # half of the freshness rule, and it is the one that costs money: arm A
      # falls back to plain Stage 3, whose rows carry no track_ids, so a
      # per_track 3c has nothing to key on and collapses to one call per box.
      STAGE3B_STALE_NOTE_SHOWN=1
      echo "note: stage3b_track2d is STALE — arm A will be plain Stage 3; a per_track 3c will save nothing" >&2
      echo "note:   (its manifest is older than stage3_proposals', so it describes a PREVIOUS Stage 3.)" >&2
      echo "note:   Put 3b in this chain — e.g. '3 3b 3c 4' — to give 3c and Stage 4 track ids again." >&2
    fi
  fi
  # stage3_merged outranks both arms when it is complete and NEWER than the
  # arm A dir just chosen — a stale merge over a fresh arm A would resurrect
  # boxes the newer run no longer proposes. Same freshness rule as s3b.
  if [ "$level" != arms ]; then
    local s3m="$WORK_ROOT/stage3_merged"
    if [ "$(marker_state "$s3m")" != none ] && \
       [ "$s3m/run_manifest.json" -nt "$STAGE3_DIR_FOR_4/run_manifest.json" ]; then
      # The same scope gate the 3b branch applies, and for the same reason —
      # but measured against the tree the merge would DISPLACE, not against
      # stage3_proposals: that displaced tree is what this run would otherwise
      # hand Stage 4, and it is the only set a narrower merge could silently
      # cut down.
      missing="$(scope_shortfall "$s3m" "$STAGE3_DIR_FOR_4")"
      if [ -n "$missing" ]; then
        if [ "$STAGE3M_NOTE_SHOWN" != "$missing" ]; then
          STAGE3M_NOTE_SHOWN="$missing"
          echo "note: $s3m is FRESHER than $(basename "$STAGE3_DIR_FOR_4") but does not cover every" >&2
          echo "note:   scene this run will hand Stage 4. Missing from the merged tree: $missing" >&2
          echo "note:   Stage 4 will read $(basename "$STAGE3_DIR_FOR_4") instead — arm B's boxes are NOT in this run." >&2
          echo "note:   Re-run 3f/3m over those scenes, or give this run the same --scenes, to use them." >&2
        fi
      else
        STAGE3_DIR_FOR_4="$s3m"
      fi
    fi
  fi
  # stage3_checked (opt-in 3c) outranks everything above by the same freshness
  # rule: the checker consumed whichever tree was current when it ran, and a
  # stale check standing over a fresher merge would resurrect labels the
  # checker never saw.
  if [ "$level" != arms ] && [ "$level" != merged ]; then
    local s3c="$WORK_ROOT/stage3_checked"
    if [ "$(marker_state "$s3c")" != none ] && \
       [ "$s3c/run_manifest.json" -nt "$STAGE3_DIR_FOR_4/run_manifest.json" ]; then
      # Scope gate, measured against the tree the check would displace. This is
      # not hypothetical any more: 3c takes --scenes, so a one-scene check now
      # exists on disk and must never silently narrow a later full run to it.
      missing="$(scope_shortfall "$s3c" "$STAGE3_DIR_FOR_4")"
      if [ -n "$missing" ]; then
        if [ "$STAGE3C_NOTE_SHOWN" != "$missing" ]; then
          STAGE3C_NOTE_SHOWN="$missing"
          echo "note: $s3c is FRESHER than $(basename "$STAGE3_DIR_FOR_4") but does not cover every" >&2
          echo "note:   scene this run will hand Stage 4. Missing from the checked tree: $missing" >&2
          echo "note:   Stage 4 will read $(basename "$STAGE3_DIR_FOR_4") instead — the VLM's relabels are NOT in this run." >&2
          echo "note:   Re-run 3c over those scenes, or give this run the same --scenes, to use them." >&2
        fi
      else
        STAGE3_DIR_FOR_4="$s3c"
      fi
    fi
  fi
  [ "$(marker_state "$STAGE3_DIR_FOR_4")" = degraded ] && DEGRADED_SEEN=1
  return 0
}

# Stage 1 is upstream of everything here and was already flagged before this
# script ever ran (C16: 2 of 10 scenes over the sector-rejection threshold), so
# the opt-in usually starts armed. On a clean substrate it starts disarmed and
# only arms if something in THIS run degrades.
for upstream_dir in "$WORK_ROOT/stage1_ingestion" "$WORK_ROOT/stage0_data_probe"; do
  [ "$(marker_state "$upstream_dir")" = degraded ] && DEGRADED_SEEN=1
done
# ...and the same question asked of the tree Stage 4 will actually read, because
# a resume that starts at stage 4 has no earlier step in which to observe it.
select_stage3_dir_for_4
[ "$DEGRADED_SEEN" = 1 ] && echo "upstream is DEGRADED on disk — stages will run with --accept-degraded-upstream (C16, recorded in each manifest)"

# The dry run ends HERE, and here specifically: both answers are now resolved —
# the step list (VLM_CHECK surgery included) and the tree Stage 4 would be
# handed — and no stage has run, nothing has been written but the lock and the
# log. Everything above only read markers and manifest mtimes.
# Two lines, in a fixed shape, so a matrix harness can parse them; the notes
# each answer came with are above, for the human.
if [ "${PRINT_STEPS-}" = 1 ]; then
  echo "steps: ${STEPS[*]}"
  echo "stage4_input: $(basename "$STAGE3_DIR_FOR_4")"
  exit 0
fi

# The CVAT exporters build their COCO category table from a taxonomy FILE, and
# a merged (arm A + arm B) tree carries arm B's SUPERSET vocabulary. Exporting
# it against arm A's taxonomy dies on `KeyError: 'an auto rickshaw'` after every
# stage has already run. C28 wired the superset through stages 3f/3m/4 and
# stopped there; this is the same question asked of the publish.
taxonomy_for_tree() {
  case "$(basename "$1")" in
    stage3_merged|stage3_finetuned) echo "$ARMB_TAXONOMY" ;;
    stage3_checked)
      # The checked tree carries the vocabulary of the tree it checked (3c
      # never widens the class space); its manifest records that tree. An
      # absent/unreadable manifest falls through to arm A's taxonomy.
      taxonomy_for_tree "$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["upstream"]["input"]["dir"])' \
          "$WORK_ROOT/stage3_checked/run_manifest.json" 2>/dev/null || echo stage3_proposals)" ;;
    *) echo "configs/taxonomy_pilot_nuscenes.yaml" ;;
  esac
}
export_taxonomy() { taxonomy_for_tree "$STAGE3_DIR_FOR_4"; }

# The answer-key twins are the DATASET'S OWN human labels, projected. A
# substrate with an empty sample_annotation.json (any raw capture, e.g. the
# dhaka pilots) has none to project, and export_gt_coco would build empty twins
# that read as "the pipeline found everything the humans did".
has_ground_truth() {
  "$PY" - "$DATAROOT_META" <<'PYGT' 2>/dev/null
import json, os, sys
try:
    with open(os.path.join(sys.argv[1], "sample_annotation.json")) as fh:
        sys.exit(0 if json.load(fh) else 1)
except Exception:
    sys.exit(1)
PYGT
}

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

    # Stage 0 takes no --accept-degraded-upstream: it has no upstream at all.
    # Stage 1 DOES take it. Its upstream is Stage 0's allowlist, and
    # ingest.load_allowlist() refuses outright on a degraded Stage 0 marker
    # unless the flag is passed (ingest.py:789). This was previously commented
    # as "stages 0 and 1 take no flag", which held only because Stage 0 never
    # degraded on v1.0-mini; the first substrate whose partition is
    # unsatisfiable (one scene, one location, no night) refused at Stage 1 with
    # the wrapper having just announced it would pass the flag.
    # Both use the same three-state exit contract, so a Stage 0 that excludes a
    # scene (rc 1) or a Stage 1 that falls back on a sector (rc 1) flags the
    # run and keeps going, exactly like the later stages.
    # `-m`, not a file path: unlike stages 3-8 these two have no sys.path
    # bootstrap of their own, so running them by path dies on `import pipeline`
    # before argparse ever sees a flag. Their own docstrings document the
    # module form; `cd "$REPO"` above is what puts the package on sys.path.
    0)  run_step "STAGE 0 (data probe: scene allowlist)" "$WORK_ROOT/stage0_data_probe" \
          "$PY" -m pipeline.stage0_data_probe.probe || break
        ;;

    1)  acc
        run_step "STAGE 1 (ingestion: keyframe index + ground-filtered clouds)" "$WORK_ROOT/stage1_ingestion" \
          "$PY" -m pipeline.stage1_ingestion.ingest \
            ${ACC[@]+"${ACC[@]}"} ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;

    3)  acc
        run_step "STAGE 3 (proposal_2d: $PROPOSAL_MODEL_ID)" "$WORK_ROOT/stage3_proposals" \
          "$PY" pipeline/stage3_proposals/proposals.py \
            --model-id "$PROPOSAL_MODEL_ID" --revision "$PROPOSAL_REVISION" \
            --class-map "$PROPOSAL_CLASS_MAP" \
            ${ACC[@]+"${ACC[@]}"} ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;

    3b) acc
        # Opt-in (C27). It rewrites Stage 3's proposals — same schema, same row
        # order, plus the recovered boxes and their provenance — into its OWN
        # tree, and changes nothing under stage3_proposals. Stage 4 is then
        # pointed at whichever of the two is current; see select_stage3_dir_for_4.
        TRACK2D_ARGS=()
        [ "$TRACK2D_REFINE" = 1 ]    && TRACK2D_ARGS+=(--refine-boxes)
        [ "$TRACK2D_NO_SWEEPS" = 1 ] && TRACK2D_ARGS+=(--no-sweep-detection)
        run_step "STAGE 3b (track2d: $TRACK2D_MODEL_ID)" "$WORK_ROOT/stage3b_track2d" \
          "$PY" pipeline/stage3b_track2d/track2d.py \
            --model-id "$TRACK2D_MODEL_ID" --revision "$TRACK2D_REVISION" \
            ${TRACK2D_ARGS[@]+"${TRACK2D_ARGS[@]}"} \
            ${ACC[@]+"${ACC[@]}"} ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;

    3f) acc
        # Arm B proposals (opt-in, C28). The SAME driver as Stage 3 — different
        # weights, class map and (superset) taxonomy — into its OWN tree, so
        # arm A stays frozen and every archived Results/ number stands.
        run_step "STAGE 3f (proposal_2d arm B: $ARMB_MODEL_ID)" "$WORK_ROOT/stage3_finetuned" \
          "$PY" pipeline/stage3_proposals/proposals.py \
            --model-id "$ARMB_MODEL_ID" --revision "$ARMB_REVISION" \
            --taxonomy "$ARMB_TAXONOMY" --class-map "$ARMB_CLASS_MAP" \
            --out-dir "$WORK_ROOT/stage3_finetuned" \
            ${ACC[@]+"${ACC[@]}"} ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;

    3m) # The merge (opt-in, C28). Arm A input is whatever Stage 4 would have
        # read (stage3b_track2d when fresh and covering, else stage3_proposals);
        # the merge writes stage3_merged, which select_stage3_dir_for_4 then
        # prefers on the next ask. No --scenes: the merge pairs whole trees and
        # refuses a scene-set mismatch rather than merging a subset silently.
        # `arms`: a fresh stage3_merged/stage3_checked must never become arm A —
        # both already carry arm B boxes, and merging them again doubles every one.
        select_stage3_dir_for_4 arms
        acc
        run_step "STAGE 3m (merge: $(basename "$STAGE3_DIR_FOR_4") + stage3_finetuned)" "$WORK_ROOT/stage3_merged" \
          "$PY" pipeline/stage3_merge/merge.py \
            --arm-a-dir "$STAGE3_DIR_FOR_4" \
            --arm-b-dir "$WORK_ROOT/stage3_finetuned" \
            --out-dir "$WORK_ROOT/stage3_merged" \
            --taxonomy "$ARMB_TAXONOMY" \
            ${ACC[@]+"${ACC[@]}"} || break
        ;;

    3c) # The VLM label check (opt-in). Consumes the ONE tree Stage 4 would
        # otherwise read — stage3_merged when 3f/3m ran — and writes
        # stage3_checked, which select_stage3_dir_for_4 then prefers over
        # everything. check.py starts and stops its own llama-server.
        # `merged`: 3c may read the merge but never its own previous output.
        select_stage3_dir_for_4 merged
        acc
        # per_track keys on the track ids 3b wrote, and ONLY a 3b tree (or a
        # merge of one) carries them. stage3_proposals never does — which is
        # exactly what an `all`-style chain hands us, because step 3 has just
        # rewritten the proposals and staled the 3b tree. check.py would refuse
        # that (coverage 0, rc 2) and abort the chain mid-run, so the mode is
        # demoted HERE, with the arithmetic said out loud, and the refusal is
        # left to mean what it should: a track-bearing tree that unexpectedly
        # lost its ids.
        CHECK_MODE="$VLM_CHECK_MODE"
        if [ "$CHECK_MODE" != per_box ] && [ "$(basename "$STAGE3_DIR_FOR_4")" = stage3_proposals ]; then
          CHECK_MODE=per_box
          echo "note: 3c input has no track ids — running per_box: every above-floor box is one VLM call."
          echo "note:   ($(basename "$STAGE3_DIR_FOR_4") carries no track_ids; a per_track run would refuse.)"
          echo "note:   Run 3b in this chain — e.g. '3 3b 3c 4' — for the ~7.5x per-track saving."
        fi
        CHECK_ARGS=(--check-mode "$CHECK_MODE")
        # Only under per_track: it is the retry ladder's depth and means nothing
        # per-box, and a flag that means nothing is a flag someone will read as
        # if it did.
        [ "$CHECK_MODE" = per_track ] && CHECK_ARGS+=(--track-retry-candidates "$VLM_TRACK_RETRIES")
        [ "$VLM_ALLOW_UNTRACKED" = 1 ]  && CHECK_ARGS+=(--allow-untracked)
        [ "$VLM_ALLOW_SHARED_GPU" = 1 ] && CHECK_ARGS+=(--allow-shared-gpu)
        if [ -n "$VLM_SKIP_CLASSES" ]; then
          # C31: comma-separated phrases -> one --skip-class each
          IFS=',' read -r -a SKIP_PHRASES <<< "$VLM_SKIP_CLASSES"
          for phrase in "${SKIP_PHRASES[@]}"; do
            CHECK_ARGS+=(--skip-class "$phrase")
          done
        fi
        run_step "STAGE 3c (vlm check: $(basename "$VLM_GGUF" .gguf) on $(basename "$STAGE3_DIR_FOR_4"); $CHECK_MODE)" "$WORK_ROOT/stage3_checked" \
          "$PY" pipeline/stage3c_check/check.py \
            --stage3-dir "$STAGE3_DIR_FOR_4" \
            --out-dir "$WORK_ROOT/stage3_checked" \
            --taxonomy "$(taxonomy_for_tree "$STAGE3_DIR_FOR_4")" \
            --paths "$PATHS_CONFIG" \
            --gguf "$VLM_GGUF" --mmproj "$VLM_MMPROJ" \
            --llama-server-bin "$LLAMA_SERVER_BIN" \
            --parallel "$VLM_PARALLEL" --n-cpu-moe "$VLM_N_CPU_MOE" \
            ${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"} \
            ${ACC[@]+"${ACC[@]}"} ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;

    4)  # Re-asked here and not reused from the pre-scan: step 3b may have
        # written its tree since, and `acc` must see what that tree's marker
        # says before Stage 4 is handed it.
        select_stage3_dir_for_4
        acc
        MASK_ARGS=()
        [ "$MASK_TEXT_PROMPT" = 1 ] && MASK_ARGS+=(--text-prompt)
        run_step "STAGE 4 (mask_2d: $MASK_MODEL_ID; proposals from $(basename "$STAGE3_DIR_FOR_4"))" "$WORK_ROOT/stage4_masks" \
          "$PY" pipeline/stage4_masks/masks.py \
            --model-id "$MASK_MODEL_ID" --revision "$MASK_REVISION" \
            --stage3-dir "$STAGE3_DIR_FOR_4" \
            ${MASK_ARGS[@]+"${MASK_ARGS[@]}"} \
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
        run_step "STAGE 7 (track; reid $REID_MODEL_ID)" "$WORK_ROOT/stage7_track" \
          "$PY" pipeline/stage7_track/track.py \
            --reid-model-id "$REID_MODEL_ID" --reid-revision "$REID_REVISION" \
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
        # --taxonomy, exactly as the `cvat` arm's rebuild passes it (they
        # disagreed until now): export_cvat_coco.py defaults to arm A's
        # nuScenes taxonomy and indexes its category table with no fallback, so
        # one 'an auto rickshaw' from a merged — or checked-merged — tree is a
        # KeyError at severity `fatal`, i.e. the chain aborts AFTER every GPU
        # stage has already run. export_taxonomy answers for the tree this run
        # actually selected.
        run_step "EXPORT cvat_export (Stage 3 boxes + Stage 4 masks -> COCO)" fatal \
          "$PY" scripts/export_cvat_coco.py --taxonomy "$(export_taxonomy)" \
            ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break

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
            "$PY" scripts/export_cvat_coco.py --taxonomy "$(export_taxonomy)" \
              ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        fi
        # The task names carry the run (C30): the tag is the stage4 manifest's
        # mtime, so a NEW run publishes beside the previous runs' tasks and a
        # republish of the SAME run skips its own names. No manifest, no tag —
        # and no tag means publishing pre-annotations that nothing ties to a
        # run, which is the stale-under-a-fresh-name failure again; refuse.
        # --cvat-replace arms --replace-all-runs: every pipeline-output task in
        # this project goes first, whatever run made it. The answer-key twins
        # live in a DIFFERENT project and carry a different name, so they
        # survive both modes untouched (C13).
        if [ ! -f "$WORK_ROOT/stage4_masks/run_manifest.json" ]; then
          echo "!!! no stage4_masks/run_manifest.json — cannot name the run these pre-annotations came from" >&2
          STEP_NAMES+=("CVAT publish"); STEP_STATUS+=("FAILED rc=2 no stage4 manifest to tag (C30)"); STEP_SECS+=(0)
          ABORTED="CVAT publish"; break
        fi
        CVAT_TAG_2D="${CVAT_RUN_TAG:-$(date +%Y%m%dT%H%M%S -r "$WORK_ROOT/stage4_masks/run_manifest.json")}"
        CVAT_WIPE_ARGS=()
        [ "$CVAT_REPLACE" = 1 ] && CVAT_WIPE_ARGS+=(--replace-all-runs)
        run_step "CVAT publish ($CVAT_HOST -> '$CVAT_PIPELINE_PROJECT', run $CVAT_TAG_2D)" fatal \
          "$PY" scripts/cvat_setup.py \
            --host "$CVAT_HOST" --user "$CVAT_USER" \
            --project "$CVAT_PIPELINE_PROJECT" \
            --task-suffix "$CVAT_PIPELINE_SUFFIX" --run-tag "$CVAT_TAG_2D" \
            ${CVAT_WIPE_ARGS[@]+"${CVAT_WIPE_ARGS[@]}"} \
            ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break

        # The answer-key twins. Deliberately NOT --replace: these are the
        # dataset's own human labels, they do not change between runs, and
        # cvat_setup refuses --replace on this suffix anyway (C13). Existing
        # twins are skipped, so this is a no-op unless they are missing —
        # which is exactly the case after --clean-slate.
        if has_ground_truth; then
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
        else
          echo "  no sample_annotation on this substrate — skipping the answer-key twins."
          echo "  (a raw capture has no human labels to project; empty twins would read as agreement)"
          STEP_NAMES+=("CVAT answer-key twins"); STEP_STATUS+=("skipped: substrate has no GT"); STEP_SECS+=(0)
        fi
        ;;

    cvat3d)
        # The 3D counterpart of `cvat`: point-cloud tasks carrying our Stage 8
        # cuboids, plus the human answer key in its own project. Separate from
        # `cvat` because it is far more expensive -- each scene ships its own
        # copy of the cloud and six images (~45-80 MB), and CVAT cannot share
        # frame data between two tasks.
        if [ -z "${CVAT_PASSWORD:-}" ]; then
          echo "!!! CVAT_PASSWORD is unset (.env SECRET block) — cannot publish 3D" >&2
          STEP_NAMES+=("CVAT 3D publish"); STEP_STATUS+=("skipped: no CVAT_PASSWORD"); STEP_SECS+=(0)
          continue
        fi
        # Same freshness rule as `cvat`: the cuboids must be at least as new as
        # the boxes they claim to come from, or a subset run would upload a
        # previous run's geometry under this run's name.
        if [ ! -e "$WORK_ROOT/cvat_export_3d" ] || \
           [ "$WORK_ROOT/stage8_inflate/run_manifest.json" -nt "$WORK_ROOT/cvat_export_3d" ]; then
          run_step "EXPORT cvat_export_3d (stale or missing — rebuilding before publish)" fatal \
            "$PY" scripts/export_cvat_3d.py --taxonomy "$(export_taxonomy)" \
              ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        fi
        # Run-tagged like the 2D publish (C30), off the stage8 manifest this
        # arm's freshness rule already anchors on. Each kept run's tasks hold
        # their own ~45-80 MB point-cloud archive per scene — the storage bill
        # of side-by-side, reclaimed only by --cvat-replace or --clean-slate.
        # --cvat-replace arms --replace-all-runs for OUR tasks only; the
        # answer-key twins are in a DIFFERENT project and cvat_setup_3d never
        # arms any wipe for them (C13): 3D cuboids are not regenerable from a
        # reviewed task by any committed importer, so a rebuild there could
        # only destroy review work.
        # `both` means ours + the answer key. A substrate with no
        # sample_annotation has no answer key to publish, and asking for one
        # would either fail or create empty green tasks that read as a human
        # having agreed with the pipeline.
        if [ ! -f "$WORK_ROOT/stage8_inflate/run_manifest.json" ]; then
          echo "!!! no stage8_inflate/run_manifest.json — cannot name the run these cuboids came from" >&2
          STEP_NAMES+=("CVAT 3D publish"); STEP_STATUS+=("FAILED rc=2 no stage8 manifest to tag (C30)"); STEP_SECS+=(0)
          ABORTED="CVAT 3D publish"; break
        fi
        CVAT_TAG_3D="${CVAT_RUN_TAG:-$(date +%Y%m%dT%H%M%S -r "$WORK_ROOT/stage8_inflate/run_manifest.json")}"
        CVAT_WIPE_ARGS=()
        [ "$CVAT_REPLACE" = 1 ] && CVAT_WIPE_ARGS+=(--replace-all-runs)
        WHICH_3D=both
        has_ground_truth || WHICH_3D=ours
        [ "$WHICH_3D" = ours ] && echo "  no sample_annotation — publishing OUR cuboids only, no answer key"
        run_step "CVAT 3D publish ($CVAT_HOST -> point-cloud tasks, --which $WHICH_3D, run $CVAT_TAG_3D)" fatal \
          "$PY" scripts/cvat_setup_3d.py \
            --host "$CVAT_HOST" --user "$CVAT_USER" --which "$WHICH_3D" \
            --run-tag "$CVAT_TAG_3D" \
            ${CVAT_WIPE_ARGS[@]+"${CVAT_WIPE_ARGS[@]}"} \
            ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;

    road)
        # C33: the road-surface layer. Consumes Stage 1 ALONE — the digit
        # prefix convention (3b/3f/3m/3c = "a variant consuming stage N")
        # would lie here, so the token is `road`, a side-branch like eval/viz.
        # SAM 3 text prompt "paved road" per (keyframe, camera), union mask,
        # plane-gated paint onto the RAW LIDAR_TOP cloud. Opt-in: no archived
        # Results/ cell was measured with it.
        acc
        run_step "STAGE road (road surface: sam3 text 'paved road' -> masks + plane-gated points)" "$WORK_ROOT/stage_road" \
          "$PY" pipeline/stage_road/road.py \
            --out-dir "$WORK_ROOT/stage_road" \
            --paths "$PATHS_CONFIG" \
            ${ACC[@]+"${ACC[@]}"} ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        ;;

    cvatroad)
        # The road layer's publish. The `cvat` prefix is load-bearing:
        # is_cvat_step() picks this arm up for --no-cvat and the clean-slate
        # purge with no extra list to forget. Its OWN project (C33): a surface
        # label in the shared instance project would renumber its categories.
        if [ -z "${CVAT_PASSWORD:-}" ]; then
          echo "!!! CVAT_PASSWORD is unset (.env SECRET block) — cannot publish road" >&2
          STEP_NAMES+=("CVAT road publish"); STEP_STATUS+=("skipped: no CVAT_PASSWORD"); STEP_SECS+=(0)
          continue
        fi
        # Same freshness rule as `cvat`: masks at least as new as the stage
        # tree they claim to describe.
        acc
        if [ ! -e "$WORK_ROOT/cvat_export_road" ] || \
           [ "$WORK_ROOT/stage_road/run_manifest.json" -nt "$WORK_ROOT/cvat_export_road" ]; then
          run_step "EXPORT cvat_export_road (stale or missing — rebuilding before publish)" fatal \
            "$PY" scripts/export_road_coco.py \
              ${ACC[@]+"${ACC[@]}"} ${SCENE_ARGS[@]+"${SCENE_ARGS[@]}"} || break
        fi
        if [ ! -f "$WORK_ROOT/stage_road/run_manifest.json" ]; then
          echo "!!! no stage_road/run_manifest.json — cannot name the run these masks came from" >&2
          STEP_NAMES+=("CVAT road publish"); STEP_STATUS+=("FAILED rc=2 no stage_road manifest to tag (C30)"); STEP_SECS+=(0)
          ABORTED="CVAT road publish"; break
        fi
        CVAT_TAG_ROAD="${CVAT_RUN_TAG:-$(date +%Y%m%dT%H%M%S -r "$WORK_ROOT/stage_road/run_manifest.json")}"
        CVAT_WIPE_ARGS=()
        [ "$CVAT_REPLACE" = 1 ] && CVAT_WIPE_ARGS+=(--replace-all-runs)
        run_step "CVAT road publish ($CVAT_HOST -> '$CVAT_ROAD_PROJECT', run $CVAT_TAG_ROAD)" fatal \
          "$PY" scripts/cvat_setup.py \
            --host "$CVAT_HOST" --user "$CVAT_USER" \
            --export-dir cvat_export_road \
            --project "$CVAT_ROAD_PROJECT" \
            --task-suffix "$CVAT_ROAD_SUFFIX" --run-tag "$CVAT_TAG_ROAD" \
            ${CVAT_WIPE_ARGS[@]+"${CVAT_WIPE_ARGS[@]}"} \
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
for d in stage0_data_probe stage1_ingestion stage3_proposals stage3b_track2d stage3_finetuned stage3_merged stage3_checked stage4_masks stage5_lift stage6_cluster stage7_track stage8_inflate stage_road; do
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
