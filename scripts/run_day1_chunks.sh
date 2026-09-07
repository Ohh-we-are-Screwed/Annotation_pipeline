#!/usr/bin/env bash
# Day-1 chunked annotation run (2026-09-06) — docs/superpowers/specs/2026-09-06-day1-chunked-run-design.md
#
# Runs EVERY step except the VLM check over each chunk of the 2026-09-04 Dhaka
# capture (Dataset/A_nusc/chunk_NNNN), one chunk at a time, and for each chunk
# leaves behind:
#   export/day1_chunk_NNNN/boxes/     nuScenes-format release, every blob a REAL
#                                     FILE (export_release --blobs copy), plus the
#                                     benchmark sidecars the checklist asks for:
#                                     stitch_map.json, double_annotation.json,
#                                     sample_annotation_excluded.json, release_meta.json
#                                     and DELIVERY_NOTE.md
#   export/day1_chunk_NNNN/road/      road-surface lidarseg layer (tables + .bin;
#                                     the blobs are already real files in boxes/)
#   CVAT projects "day1_chunk_NNNN (2D)", "(3D)", "(road)" and the double-annotation
#   pair "day1_chunk_NNNN (3D) — double pass A / B" on $CVAT_HOST
#
# ORDER (spec §9): the release export runs BEFORE the 3D publish. The review task
# has to show the STITCHED identities (export_release writes stitch_map.json and
# export_cvat_3d.py --stitch-map renames the cuboids by it), and the two blank
# double-annotation tasks are cut from the keyframe selection the same export
# writes (double_annotation.json). So `cvat3d` is no longer a CHAIN_STEP: this
# driver publishes the 3D projects itself, after the export.
# That order means <work>/cvat_export_3d does NOT exist when the export runs, so
# the export is told --stage1-dir: Stage 1's own ground-filtered single sweeps
# are what every interpolated point count is measured against (the 3D archive is
# only a copy of them). Those clouds are pruned AFTER the export, below.
#
# Per-chunk ISOLATION: each chunk gets its own generated paths config and its
# own work/out roots under $ROOT_WORK / $ROOT_OUT / $ROOT_PROBE (default
# /home/mt/dhakascenes/{work,out,probe_out}_day1/, overridable per run), so
# no chunk's stage tree is overwritten by the next and any chunk can be resumed
# or republished alone with the wrapper and that chunk's config.
#
# What does NOT run, and why (substrate, not choice):
#   3b   walks 12 Hz sweep frames; this export has ONE image per keyframe.
#   3c   the operator's one exclusion; VLM_CHECK/VLM_USE_CHECKED default 0.
#   The GT halves of `eval` are soft and no-op: there is no sample_annotation.
# Road runs DEGRADED (its ZED refinement keys on rings 10/11; this export
# carries ZED as ring 100+k) — the nuScenes precedent, _SUCCESS.degraded.
#
# Usage:
#   scripts/run_day1_chunks.sh                 # 0006 first (calibration), then 0000..0005
#   scripts/run_day1_chunks.sh 0006            # one chunk
#   scripts/run_day1_chunks.sh 0000 0001       # these, in this order
#   scripts/run_day1_chunks.sh --phase human-import 0000
#                                              # SECOND PASS, after the annotators have
#                                              # finished the CVAT 3D review + double tasks:
#                                              # import_cvat_3d.py, then re-export the release
#                                              # with --human --overwrite-tables (tables and
#                                              # sidecars rewritten in place; blobs untouched)
#   DRY_RUN=1 scripts/run_day1_chunks.sh 0006  # write the config, print the wrapper's plan AND
#                                              # every command the tail would run; runs none of
#                                              # them (works with --phase human-import too)
#
# The CVAT purge is NOT here. It is a one-way door and was run by hand, once,
# before the first chunk (scripts/cvat_purge.py --yes).

set -uo pipefail

# Every root is DERIVED or overridable from the environment, with the value this
# machine has always used as the fallback — so behaviour here is unchanged, and
# a fresh clone (or a git worktree) runs without editing the script. REPO comes
# from where THIS FILE lives, never from $PWD or a hardcoded checkout.
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATASET=${DATASET:-$REPO/Dataset/A_nusc}
EXPORT=${EXPORT:-$REPO/export}
ROOT_WORK=${ROOT_WORK:-/home/mt/dhakascenes/work_day1}
ROOT_OUT=${ROOT_OUT:-/home/mt/dhakascenes/out_day1}
ROOT_PROBE=${ROOT_PROBE:-/home/mt/dhakascenes/probe_out_day1}
LOGS=${LOGS:-$ROOT_WORK/logs}

# The interpreter, in order of preference: an explicit $PY; the ACTIVE conda env
# when it is ano_pipe; ano_pipe under this machine's conda base; the historical
# absolute path; python3. Whichever wins is printed and must be able to import
# numpy — an eleven-chunk overnight run must not discover the wrong interpreter
# at stage 0.
pick_python() {
  local candidate
  if [ -n "${PY:-}" ];                                  then echo "$PY"; return; fi
  if [ -n "${CONDA_PREFIX:-}" ] && [ "${CONDA_PREFIX%/ano_pipe}" != "$CONDA_PREFIX" ] \
     && [ -x "$CONDA_PREFIX/bin/python" ];              then echo "$CONDA_PREFIX/bin/python"; return; fi
  candidate="$(conda info --base 2>/dev/null)/envs/ano_pipe/bin/python"
  if [ -x "$candidate" ];                               then echo "$candidate"; return; fi
  if [ -x /home/mt/miniconda3/envs/ano_pipe/bin/python ]; then
    echo /home/mt/miniconda3/envs/ano_pipe/bin/python; return
  fi
  command -v python3
}
PY=$(pick_python)
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "!!! no python found: set PY=/path/to/python (conda env ano_pipe) and re-run" >&2; exit 2
fi
if ! "$PY" -c 'import numpy' >/dev/null 2>&1; then
  echo "!!! $PY cannot 'import numpy' — that is not the ano_pipe environment." >&2
  echo "    Activate it (conda activate ano_pipe) or set PY=/path/to/ano_pipe/bin/python." >&2
  exit 2
fi
VERSION=v1.0-dhaka          # the version dir inside every chunk, verbatim (read-only)
# The chain runs against a FIXED sibling version: ~5 % of keyframes lack a
# camera image and Stage 0's channels_complete is all-or-nothing per scene.
# scripts/fixup_a_nusc.py drops those samples into $VERSION-fixed beside the
# original (the pilot's v1.0-dhaka-fixed precedent); written once per chunk.
FIXED_VERSION=$VERSION-fixed
# Camera streams the exporter filed under each other's names (see the fixup
# call). Empty = no swap.
SWAP_CHANNELS=${SWAP_CHANNELS-CAM_LEFT CAM_RIGHT}
DAY=day1
# Host side of the CVAT share: `docker volume inspect cvat_cvat_share` -> bind of
# /home/mt/Zami/nuscenes (a symlink into this repo's nuscenes/). Chunks are
# staged under $CVAT_SHARE_ROOT/day1_chunk_NNNN/samples as hard links.
CVAT_SHARE_ROOT=${CVAT_SHARE_ROOT:-/home/mt/Zami/nuscenes}

# Everything but 3b/3c — INCLUDING arm B (3f: the RSUD20K-finetuned detector
# that knows rickshaws and CNGs; 3m: the vocabulary-authority merge, C28/C34),
# which the first calibration run omitted (operator, 2026-09-06 01:55: "did
# you not run the custom rickshaw and CNG class?"). Typed order is execution
# order (run_stages.sh never sorts): arm A, arm B, merge, then the GPU box
# chain on the merged tree, road (consumes Stage 1 alone), the COCO export +
# renders, then the 2D and road publishes.
# NOTE `cvat3d` is deliberately absent: the 3D publish moved AFTER the release
# export (spec §9) so the review task shows the stitched identities and the blank
# double-annotation tasks exist. This driver runs it itself, in publish_3d.
CHAIN_STEPS=${CHAIN_STEPS:-"0 1 3 3f 3m 4 5 6 7 8 road eval viz cvat cvatroad"}

# run          the chain, the export, the publishes (the default)
# human-import the second pass over a chunk whose CVAT 3D tasks are annotated:
#              import_cvat_3d.py + a re-export with --human --overwrite-tables
PHASE=${PHASE:-run}

DEFAULT_ORDER=(0006 0000 0001 0002 0003 0004 0005)
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --phase)   PHASE=${2:?--phase needs a value (run | human-import)}; shift 2 ;;
    --phase=*) PHASE=${1#--phase=}; shift ;;
    *)         ARGS+=("$1"); shift ;;
  esac
done
case "$PHASE" in
  run|human-import) ;;
  *) echo "!!! unknown --phase '$PHASE' (run | human-import)" >&2; exit 2 ;;
esac
CHUNKS=(${ARGS[@]+"${ARGS[@]}"})
[ ${#CHUNKS[@]} -eq 0 ] && CHUNKS=("${DEFAULT_ORDER[@]}")

cd "$REPO" || exit 2
mkdir -p "$LOGS" "$EXPORT"
echo "repo    : $REPO"
echo "python  : $PY  ($("$PY" -V 2>&1))"
echo "roots   : dataset=$DATASET work=$ROOT_WORK out=$ROOT_OUT probe=$ROOT_PROBE export=$EXPORT"

# .env carries CVAT_PASSWORD and the cache roots. Sourced FIRST so that the
# per-chunk exports below win over anything it declares.
if [ -f .env ]; then set -a; . ./.env; set +a; fi
export DHAKASCENES_SUBSTRATE=dhaka6

say() { echo; echo "############ $* ############"; date; }

# DRY_RUN=1 prints the command, shell-quoted, and returns 0 instead of running
# it. Everything the driver does AFTER the chain (the release export, the two
# 3D publishes, the human import) is behind this, because there is no shell test
# harness in this repo and a dry run is the only way to read that tail back
# without spending a live chunk on it.
run_or_echo() {
  if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'DRY_RUN would run:'; printf ' %q' "$@"; echo
    return 0
  fi
  "$@"
}

# run_stages.sh's exit code is a claim; the markers are the record (it exits 0
# on a degraded-but-complete chain and 1 on a refusal). Same reader as
# run_nuscenes_full.sh.
marker_state() {  # stage dir -> clean | degraded | none
  if   [ -e "$1/_SUCCESS" ];          then echo clean
  elif [ -e "$1/_SUCCESS.degraded" ]; then echo degraded
  else echo none
  fi
}

write_paths_config() {  # chunk id -> writes configs/paths_day1_chunk_NNNN.yaml, echoes its path
  local id=$1 cfg=configs/paths_${DAY}_chunk_$1.yaml
  cat > "$cfg" <<EOF
# GENERATED by scripts/run_day1_chunks.sh — the ${DAY} chunk_${id} path contract.
# Selected per run with DHAKASCENES_PATHS_CONFIG=$cfg (and DHAKASCENES_SUBSTRATE=dhaka6).
# Write roots are per-chunk and disjoint from every other substrate's (§1.8).
dataroot: $DATASET/chunk_$id
meta_root: $DATASET/chunk_$id
version: $FIXED_VERSION
work_root: $ROOT_WORK/chunk_$id
out_root: $ROOT_OUT/chunk_$id
probe_out_root: $ROOT_PROBE/chunk_$id
EOF
  echo "$cfg"
}

# The deliverable, and the publishes that read what it wrote. Split out of
# run_chunk so a DRY_RUN can print this tail (run_or_echo) without a live chunk.
release_export() {  # id name chunk work out -> export_release's rc
  local id=$1 name=$2 chunk=$3 work=$4 out=$5
  say "  [$id] 3/5 export_release --blobs copy -> $out/boxes"
  run_or_echo mkdir -p "$out"
  run_or_echo "$PY" scripts/export_release.py \
    --prelabels "$work/stage9_qa" --dataroot "$chunk" --version "$FIXED_VERSION" \
    --out "$out/boxes" --mapper configs/release_category_map.yaml --blobs copy \
    --cvat-export-3d-dir "$work/cvat_export_3d" --stage1-dir "$work/stage1_ingestion" \
    --stage-tree "$work" --chunk-name "$name"
  local rc=$?
  # The exit code is deliberately coarse: export_release exits 2 on checker
  # ERRORS and 0 when the checker only WARNED (zero-instance classes are normal
  # on a Dhaka route), because this driver reads any non-zero as a dead chunk —
  # no 3D publish, no cloud prune, `return 2` at the end of run_chunk.
  echo "export_release rc=$rc  (0 clean or warnings-only, 2 refused/errors)"
  return $rc
}

publish_3d() {  # id work out rel_rc -> 0 (a failed publish never kills the chunk)
  local id=$1 work=$2 out=$3 rel_rc=$4
  if [ "$rel_rc" -gt 1 ] || [ -z "${CVAT_PASSWORD:-}" ]; then
    echo "cvat3d  : skipped (export rc=$rel_rc or no CVAT_PASSWORD)"
    return 0
  fi
  say "  [$id] 4/5 cvat3d: review (pre-filled) + double pass A/B (blank)"
  # Same run tag as the 2D publish (C30): the run that produced the cuboids,
  # read off the Stage 8 manifest. Successive runs land side by side.
  local tag
  if [ -f "$work/stage8_inflate/run_manifest.json" ]; then
    tag=$(date +%Y%m%dT%H%M%S -r "$work/stage8_inflate/run_manifest.json")
  else
    tag='<stage8-run-tag>'   # DRY_RUN only; the live path is gated on this file
  fi
  # The review task, carrying the STITCHED identities: one instance_token per
  # object across the chunk, so a reviewer fixes an identity once. The explicit
  # taxonomy is run_stages.sh's export_taxonomy answer for this chain, named.
  run_or_echo "$PY" scripts/export_cvat_3d.py --taxonomy configs/taxonomy_pilot_dhaka.yaml \
      --stitch-map "$out/boxes/stitch_map.json" \
    && run_or_echo "$PY" -m scripts.cvat_setup_3d --which ours --run-tag "$tag"
  local rc_review=$?
  echo "cvat3d review rc=$rc_review"
  [ $rc_review -ne 0 ] && { echo "cvat3d double: skipped (review publish rc=$rc_review)"; return 0; }

  # The double-annotation pass: the SAME keyframes the export selected
  # (double_annotation.json), blank, in two projects — A and B — for two
  # annotators to fill independently. Never --replace/--reimport: those tasks
  # hold human work and cvat_setup_3d refuses the flags for --which double.
  local dbl=$out/boxes/double_annotation.json
  if [ ! -f "$dbl" ] && [ "${DRY_RUN:-0}" != 1 ]; then
    echo "cvat3d double: skipped ($dbl absent — double.fraction is 0 in configs/release.yaml)"
    return 0
  fi
  # Reuse the blank task.zip when it is already on disk: 45-80 MB of point cloud
  # per scene, and the selection is sticky (export_release reuses an existing
  # double_annotation.json unless --reselect-double). NOT on file presence alone
  # (I7): a rerun under EXPORT_SUFFIX writes a NEW $out, so the export reselects
  # while $work still holds the old clouds, and frames.json is rewritten either
  # way — keeping the archive would then name frame N sample Y while frame N's
  # cloud is keyframe X, and every A/B box would import against the wrong
  # sample_token. --skip-archive-if-frames-match keeps it only while the
  # frames.json beside it lists exactly this selection.
  run_or_echo "$PY" scripts/export_cvat_3d.py --taxonomy configs/taxonomy_pilot_dhaka.yaml \
      --frames "$dbl" --blank --out-subdir cvat_export_3d_double \
      --skip-archive-if-frames-match "$dbl" \
    && run_or_echo "$PY" -m scripts.cvat_setup_3d --which double --run-tag "$tag" \
      ${CVAT_ASSIGNEE_A:+--assignee-a "$CVAT_ASSIGNEE_A"} ${CVAT_ASSIGNEE_B:+--assignee-b "$CVAT_ASSIGNEE_B"}
  echo "cvat3d double rc=$?"
  return 0
}

human_import_chunk() {  # chunk id -> 0 ok | 2 import or re-export failed
  local id=$1 name=${DAY}_chunk_$1 work=$ROOT_WORK/chunk_$1 chunk=$DATASET/chunk_$1
  local out=$EXPORT/${DAY}_chunk_$1${EXPORT_SUFFIX:-} cfg; cfg=$(write_paths_config "$id")
  export DHAKASCENES_PATHS_CONFIG=$cfg
  say "HUMAN IMPORT $id"
  echo "config  : $cfg"
  echo "work    : $work"
  echo "export  : $out"
  # 1. Pull every annotated task the ledger knows (review + double A/B) into
  #    <work_root>/stage10_human/{scenes/<scene>/{verified.jsonl,coverage.json},
  #    import_manifest.json}.
  run_or_echo "$PY" scripts/import_cvat_3d.py --paths "$cfg" \
    || { echo "!!! [$id] import failed"; return 2; }
  # 2. Re-export over the SAME release: tables, sidecars and the delivery note
  #    are rewritten from the human boxes; the blobs on disk are untouched.
  run_or_echo "$PY" scripts/export_release.py \
    --prelabels "$work/stage9_qa" --dataroot "$chunk" --version "$FIXED_VERSION" \
    --out "$out/boxes" --mapper configs/release_category_map.yaml --blobs copy \
    --cvat-export-3d-dir "$work/cvat_export_3d" --stage1-dir "$work/stage1_ingestion" \
    --stage-tree "$work" --chunk-name "$name" \
    --human "$work/stage10_human" --overwrite-tables
  local rc=$?
  echo "re-export rc=$rc"
  return $rc
}

run_chunk() {  # chunk id -> 0 ok | 1 core chain failed | 2 export failed
  local id=$1 chunk=$DATASET/chunk_$1 name=${DAY}_chunk_$1
  # EXPORT_SUFFIX (e.g. "_vlm"): a rerun writes export/day1_chunk_NNNN_vlm/
  # beside the earlier export instead of replacing it, and its CVAT tasks land
  # beside the earlier ones under a new run tag (C30). Nothing already reviewed
  # is deleted — operator, 2026-09-06: "don't delete CVAT exports unless I say so".
  local work=$ROOT_WORK/chunk_$1 out=$EXPORT/${DAY}_chunk_$1${EXPORT_SUFFIX:-}
  local t0=$SECONDS

  say "CHUNK $id  ($name)"
  if [ ! -d "$chunk/samples" ] || [ ! -d "$chunk/$VERSION" ]; then
    echo "!!! $chunk is not a chunk root (no samples/ or $VERSION/). Skipping."; return 1
  fi
  # The validator wants sweeps/ present and readable; an empty dir satisfies
  # it. With the fixed version dir below, the only writes into the dataroot.
  mkdir -p "$chunk/sweeps"
  if [ ! -d "$chunk/$FIXED_VERSION" ]; then
    say "  [$id] fixup: $VERSION -> $FIXED_VERSION (drop keyframes missing a required channel)"
    # --swap-channels: the exporter filed the right-facing stream as CAM_LEFT
    # and the left-facing one as CAM_RIGHT (measured 2026-09-06 by rearward
    # image drift, +68.6 / -80.6 px; operator spotted the scrambled 3D boxes).
    "$PY" scripts/fixup_a_nusc.py --dataroot "$chunk" --version "$VERSION" --out-version "$FIXED_VERSION" \
      ${SWAP_CHANNELS:+--swap-channels $SWAP_CHANNELS}
    local fix_rc=$?
    if [ $fix_rc -ne 0 ]; then echo "!!! [$id] fixup failed rc=$fix_rc"; return 1; fi
  else
    echo "fixup   : $chunk/$FIXED_VERSION present, reused"
  fi

  local cfg; cfg=$(write_paths_config "$id")
  export DHAKASCENES_PATHS_CONFIG=$cfg
  export CVAT_PIPELINE_PROJECT="$name (2D)"
  export CVAT_PIPELINE_3D_PROJECT="$name (3D)"
  export CVAT_ROAD_PROJECT="$name (road)"
  # The 2D and road tasks are created FROM THE CVAT SHARE (no image bytes are
  # copied into CVAT): frame paths resolve against the docker bind at
  # $CVAT_SHARE_ROOT. It still holds pilot_1632's frames under the very same
  # relative names this export uses, and every day-1 chunk repeats those names
  # — so each chunk is staged under its own prefix, as HARD LINKS (same
  # filesystem: no copy, no symlink, nothing for the container to fail to
  # resolve), and the publish is told the prefix via CVAT_SHARE_PREFIX.
  if [ ! -d "$CVAT_SHARE_ROOT" ]; then echo "!!! CVAT share root $CVAT_SHARE_ROOT missing"; return 1; fi
  if [ ! -d "$CVAT_SHARE_ROOT/$name/samples" ]; then
    mkdir -p "$CVAT_SHARE_ROOT/$name" && cp -al "$chunk/samples" "$CVAT_SHARE_ROOT/$name/" \
      || { echo "!!! [$id] could not stage samples into the CVAT share"; return 1; }
    echo "share   : staged $(find "$CVAT_SHARE_ROOT/$name/samples" -type f | wc -l) hard links under $CVAT_SHARE_ROOT/$name/"
  else
    echo "share   : $CVAT_SHARE_ROOT/$name/samples present, reused"
  fi
  export CVAT_SHARE_PREFIX="$name/"
  echo "config  : $cfg"
  echo "work    : $work"
  echo "export  : $out"
  echo "cvat    : '$CVAT_PIPELINE_PROJECT' / '$CVAT_PIPELINE_3D_PROJECT' / '$CVAT_ROAD_PROJECT'"
  echo "steps   : $CHAIN_STEPS"
  # Stage 6 takes epsilon and Stage 8 takes class means from priors_pilot_v0.json
  # and both refuse one not bound to THIS dataroot's fingerprint. Dhaka has no
  # GT to derive it from; it is AUTHORED (handover §6 recipe as code) — after
  # the fixup, so the fingerprint is the fixed tables'. Idempotent.
  run_or_echo "$PY" scripts/author_priors_dhaka.py --paths "$cfg" \
    || { echo "!!! [$id] priors authoring failed"; return 1; }

  if [ "${DRY_RUN:-0}" = 1 ]; then
    PRINT_STEPS=1 scripts/run_stages.sh $CHAIN_STEPS
    say "  [$id] DRY_RUN: the tail this driver runs after the chain (printed, not executed)"
    release_export "$id" "$name" "$chunk" "$work" "$out"
    publish_3d "$id" "$work" "$out" 0
    return 0
  fi
  if [ -d "$out/boxes/$FIXED_VERSION" ] && [ -n "$(ls -A "$out/boxes/$FIXED_VERSION" 2>/dev/null)" ]; then
    echo "!!! $out/boxes/$FIXED_VERSION already exists and is not empty — export_release would refuse."
    echo "!!! Remove it (or the whole $out) to re-export this chunk. Skipping."
    return 2
  fi

  # 1. The chain, publishes included (no --no-cvat).
  say "  [$id] 1/5 chain"
  local stamp; stamp=$(mktemp -p "$LOGS" ".chain_start_${id}_XXXX")
  scripts/run_stages.sh $CHAIN_STEPS
  local chain_rc=$?
  local s8; s8=$(marker_state "$work/stage8_inflate")
  local sr; sr=$(marker_state "$work/stage_road")
  echo "chain rc=$chain_rc   stage8_inflate=$s8   stage_road=$sr"
  # A marker is the record, but only THIS chain's record counts: a work tree
  # reused from an earlier run still holds that run's clean Stage 8 marker,
  # and on 2026-09-06 14:00 an aborted chain (3c refused) sailed past this
  # gate on it and exported the OLD run's boxes under the new run's name.
  if [ "$s8" = none ] || [ ! "$work/stage8_inflate/run_manifest.json" -nt "$stamp" ]; then
    rm -f "$stamp"
    echo "!!! [$id] no Stage 8 completed by THIS chain (marker=$s8, chain rc=$chain_rc) — nothing to gate, nothing to export."; return 1
  fi
  rm -f "$stamp"

  # 2. Stage 9 QA gate -> prelabels.jsonl (the only producer export_release reads).
  say "  [$id] 2/5 stage 9 QA gate"
  local acc=(); [ "$s8" = degraded ] && acc=(--accept-degraded-upstream)
  "$PY" -m pipeline.stage9_qa.gate --paths "$cfg" ${acc[@]+"${acc[@]}"}
  local gate_rc=$?
  echo "stage 9 rc=$gate_rc"
  if [ $gate_rc -ge 2 ]; then echo "!!! [$id] stage 9 REFUSED (rc=$gate_rc). No export."; return 2; fi

  # 3. THE DELIVERABLE — every blob a real file, plus the benchmark sidecars
  #    (stitch_map.json, double_annotation.json, sample_annotation_excluded.json,
  #    release_meta.json, DELIVERY_NOTE.md). It runs BEFORE the 3D publish: §9.
  release_export "$id" "$name" "$chunk" "$work" "$out"
  local rel_rc=$?

  # 4. 3D CVAT: the pre-filled review task (stitched identities from
  #    stitch_map.json) and the two blank double-annotation tasks (frames from
  #    double_annotation.json). Both read what step 3 just wrote.
  publish_3d "$id" "$work" "$out" "$rel_rc"

  # 5. The EXTRAS — everything nuScenes has no notion of, each in its own
  #    named folder beside boxes/ (operator, 2026-09-06: "export what is
  #    compatible in nuScenes and put the extras in folders with appropriate
  #    names such as road").
  #    road/     road-surface layer as nuScenes-lidarseg tables + per-keyframe
  #              .bin masks. Without --link-blobs it writes no symlinks and no
  #              blob copies — the blobs are already real files in boxes/.
  #    coco_2d/  per-scene COCO: Stage 3 boxes + Stage 4 masks in image space,
  #              exactly what the CVAT 2D task received.
  local road_rc=99
  if [ "$sr" != none ]; then
    say "  [$id] 5/5 extras: road/ + coco_2d/"
    local racc=(); [ "$sr" = degraded ] && racc=(--accept-degraded-upstream)
    "$PY" -m scripts.export_road_lidarseg --paths "$cfg" --out "$out/road" ${racc[@]+"${racc[@]}"}
    road_rc=$?
    echo "export_road_lidarseg rc=$road_rc"
  else
    say "  [$id] 5/5 extras: coco_2d/ (road stage wrote no marker — no road/ layer; boxes unaffected)"
  fi
  if [ -d "$work/cvat_export" ]; then
    mkdir -p "$out/coco_2d" && cp -r "$work/cvat_export/." "$out/coco_2d/" && echo "coco_2d: $(ls "$out/coco_2d" | tr '\n' ' ')"
  fi
  # The delivery note export_release generates is the document of record — one
  # writer, regenerated on every re-export (the human-import pass included), so
  # this README is a pointer to it and nothing else.
  cat > "$out/README.md" <<EOF
# $name — machine pre-annotations

See boxes/DELIVERY_NOTE.md for the delivery note the benchmark requires (rule, range,
tiers, classes, identity, attributes, double annotation, anonymisation). Layout:
- boxes/    nuScenes $FIXED_VERSION release (13 tables + samples/ as real files) plus
            sample_annotation_excluded.json, stitch_map.json, double_annotation.json, release_meta.json.
- road/     EXTRA: driveable-surface lidarseg layer. Absent if the road stage wrote no marker.
- coco_2d/  EXTRA: 2D-only COCO layer with the detector's phrase names.
Stage tree: $work
EOF

  # Disk: Stage 1's per-keyframe clouds are ~1.2 GB per 93 keyframes (two
  # copies — with w_acc_count=1 the accumulated cloud IS the single sweep) and
  # nothing downstream of THIS POINT reads them: the box release (step 3, which
  # counts every interpolated box's returns against these very clouds via
  # --stage1-dir), the lidarseg layer and the CVAT 3D archive are all on disk by
  # now. This prune must stay after step 3 for that reason. Pruned by default so
  # six chunks fit; PRUNE_CLOUDS=0 keeps them (re-running Stage 5/6/road on
  # this chunk then needs no Stage 1 rerun).
  if [ "${PRUNE_CLOUDS:-1}" = 1 ] && [ $rel_rc -eq 0 ] && [ -d "$work/stage1_ingestion/clouds" ]; then
    local freed; freed=$(du -sh "$work/stage1_ingestion/clouds" 2>/dev/null | cut -f1)
    rm -rf "$work/stage1_ingestion/clouds" && echo "pruned  : $work/stage1_ingestion/clouds ($freed) — exports are on disk; PRUNE_CLOUDS=0 to keep"
  fi

  local n_links; n_links=$(find "$out" -type l 2>/dev/null | wc -l)
  local n_files; n_files=$(find "$out" -type f 2>/dev/null | wc -l)
  local sz; sz=$(du -sh "$out" 2>/dev/null | cut -f1)
  local mins=$(( (SECONDS - t0) / 60 ))
  echo
  echo "=== CHUNK $id SUMMARY: chain rc=$chain_rc (s8=$s8, road=$sr) gate=$gate_rc release=$rel_rc lidarseg=$road_rc | export: $n_files files, $n_links symlinks, $sz | ${mins} min"
  [ "$n_links" != 0 ] && echo "!!! [$id] export contains symlinks — that violates the hard-copy requirement."
  [ $rel_rc -ne 0 ] && return 2
  return 0
}

STAMP=$(date +%Y%m%dT%H%M%S)
MAIN_LOG=$LOGS/${DAY}_$STAMP.log
{
say "DAY-1 CHUNKED RUN  ($STAMP)  phase: $PHASE  chunks: ${CHUNKS[*]}"
echo "substrate: $DHAKASCENES_SUBSTRATE   VLM_CHECK=${VLM_CHECK:-<default 0>}   VLM_USE_CHECKED=${VLM_USE_CHECKED:-<default 0>}"
echo "log      : $MAIN_LOG"
[ "${DRY_RUN:-0}" = 1 ] && echo "DRY_RUN  : every command below is printed, not executed"
RESULTS=()
for id in "${CHUNKS[@]}"; do
  if [ "$PHASE" = human-import ]; then
    human_import_chunk "$id" 2>&1 | tee -a "$LOGS/chunk_$id.log"
  else
    run_chunk "$id" 2>&1 | tee -a "$LOGS/chunk_$id.log"
  fi
  rc=${PIPESTATUS[0]}
  RESULTS+=("chunk_$id rc=$rc")
  if [ "$rc" = 1 ]; then
    echo "!!! chunk_$id: core chain failed. Not starting the next chunk on a broken profile/config."
    break
  fi
done
say "DONE"
printf '  %s\n' "${RESULTS[@]}"
} 2>&1 | tee -a "$MAIN_LOG"
