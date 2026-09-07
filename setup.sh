#!/usr/bin/env bash
# DhakaScenes annotation pipeline — one-command bootstrap for a fresh clone.
#
#   git clone <repo> && cd <repo> && ./setup.sh
#
# WHY THIS EXISTS
#   Everything about a working checkout is machine-INDEPENDENT except one fact:
#   where the substrate lives. This script derives all the rest — the conda env,
#   the pinned dependency set in its mandated ORDER, .env, the write roots, the
#   two checkpoint FILES, the tmux session every long run is launched into — and
#   then prints, numbered, the handful of things only a human can supply
#   (secrets, and configs/paths.yaml:dataroot/meta_root/version).
#
# CONTRACTS THIS SCRIPT OBEYS — read them before changing anything here
#   requirements.txt         the install ORDER is part of the contract, not a habit
#   requirements-torch.txt   pip's --index-url is process-global -> its own file
#   requirements-devkit.txt  --no-deps, deliberately (stale upstream pins)
#   environment.yml          "conda's job is the interpreter, pip's job is the packages"
#   .env.example             the CANONICAL list of every env var the code reads
#   configs/paths.yaml       §1.8: write roots live OUTSIDE the repo, disjoint from dataroot
#
# NOT `set -e`, ON PURPOSE. Every step reports its own OK/SKIP/WARN/FAIL and the
# run continues wherever it safely can, so one missing tool does not hide the
# other nine problems. Fatal failures accumulate and set the exit code at the end.
set -uo pipefail

# The repo root is wherever this file lives — never a hardcoded path, because the
# whole point of the script is that it runs on a machine nobody has seen yet.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 2
REPO_ROOT=$PWD
ENV_NAME=ano_pipe

# .env.example calls this non-negotiable and says it has already bitten once: the
# env is Python 3.10 and so is the typical system interpreter, so ~/.local
# site-packages shadow the pinned ones. Exported here so EVERY pip and python
# call this script makes is isolated, not just the ones a later shell remembers.
export PYTHONNOUSERSITE=1

# An explicit DHAKASCENES_PATHS_CONFIG in the CALLER's environment selects which
# substrate this bootstrap prepares (the per-chunk configs run_day1_chunks.sh
# generates work exactly this way). .env is sourced further down and would
# otherwise silently overwrite it with the default, so the caller's value is
# stashed here and restored after the source.
PRESET_PATHS_CFG=${DHAKASCENES_PATHS_CONFIG-}

# ---------------------------------------------------------------------------
# Flags
# ---------------------------------------------------------------------------
DO_UPDATE=0 CPU_ONLY=0 WITH_CVAT=0 SKIP_TESTS=0 SKIP_WEIGHTS=0 ASSUME_YES=0
WEIGHTS_DIR=""

usage() {
  cat <<'USAGE'
setup.sh — bring a fresh clone of the DhakaScenes annotation pipeline up to speed.

  ./setup.sh [flags]

Flags:
  --help            Show this text and exit.
  --update          Re-resolve an EXISTING conda env (`conda env update --prune`)
                    and re-run the pip installs. Without it, an existing env is
                    reported and left completely alone — never silently rebuilt.
  --cpu-only        Install the CPU torch wheels instead of the pinned cu124 ones.
                    Allowed, but every GPU stage (2, 3, 3b, 3f, 4, 5, road) and the
                    VRAM-cap contract stop being runnable. Warns loudly.
  --with-cvat       Also clone github.com/cvat-ai/cvat next to the repo and bring
                    the stack up with `docker compose`. Without it, the script only
                    reports whether $CVAT_HOST answers. Nothing is ever deleted.
  --skip-tests      Do not run `pytest tests/ -q`. The import smoke test still runs.
  --skip-weights    Do not download mobile_sam.pt / yolo11x.pt.
  --weights-dir DIR Where the two checkpoint FILES go. Default: the checkpoints
                    directory named by .env (outside the repo — §1.8).
  --yes             Non-interactive: never prompt. Required for an unattended run.

Exit status: 0 if every fatal step passed, 1 otherwise. Safe to re-run; a second
run should report SKIP almost everywhere and finish in seconds.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --help|-h)      usage; exit 0 ;;
    --update)       DO_UPDATE=1 ;;
    --cpu-only)     CPU_ONLY=1 ;;
    --with-cvat)    WITH_CVAT=1 ;;
    --skip-tests)   SKIP_TESTS=1 ;;
    --skip-weights) SKIP_WEIGHTS=1 ;;
    --yes|-y)       ASSUME_YES=1 ;;
    --weights-dir)  WEIGHTS_DIR=${2-}; shift ;;
    *) printf 'setup.sh: unknown flag %s (try --help)\n' "$1" >&2; exit 2 ;;
  esac
  shift
done

# ---------------------------------------------------------------------------
# Reporting. Colour only when stdout is a terminal that admits to having any —
# the operator pipes this into tee inside tmux, where escape codes are noise.
# ---------------------------------------------------------------------------
if [ -t 1 ] && [ -z "${NO_COLOR-}" ] && [ "$(tput colors 2>/dev/null || echo 0)" -ge 8 ]; then
  C_OK=$'\033[1;32m'; C_SKIP=$'\033[1;34m'; C_WARN=$'\033[1;33m'
  C_FAIL=$'\033[1;31m'; C_HEAD=$'\033[1;36m'; C_OFF=$'\033[0m'
else
  C_OK=''; C_SKIP=''; C_WARN=''; C_FAIL=''; C_HEAD=''; C_OFF=''
fi

FAIL_COUNT=0
FAILED=()
WARN_COUNT=0
step() { printf '\n%s=== %s %s\n' "$C_HEAD" "$*" "$C_OFF"; }
ok()   { printf '  %sOK  %s %s\n' "$C_OK"   "$C_OFF" "$*"; }
skip() { printf '  %sSKIP%s %s\n' "$C_SKIP" "$C_OFF" "$*"; }
warn() { printf '  %sWARN%s %s\n' "$C_WARN" "$C_OFF" "$*"; WARN_COUNT=$((WARN_COUNT+1)); }
fail() { printf '  %sFAIL%s %s\n' "$C_FAIL" "$C_OFF" "$*"; FAILED+=("$*"); FAIL_COUNT=$((FAIL_COUNT+1)); }
note() { printf '       %s\n' "$*"; }

confirm() {  # confirm "question" -> 0 yes / 1 no. Non-interactive without --yes is a NO.
  [ "$ASSUME_YES" -eq 1 ] && return 0
  if [ ! -t 0 ]; then note "not a terminal and --yes not given -> answering no"; return 1; fi
  local a; read -r -p "       $* [y/N] " a; case "$a" in y|Y|yes|YES) return 0;; *) return 1;; esac
}

# Scratch outside the repo: §1.8 says nothing this script generates lands in the tree.
TMPD=$(mktemp -d "${TMPDIR:-/tmp}/dhakascenes-setup.XXXXXX") || exit 2
trap 'rm -rf "$TMPD"' EXIT

sha256_of() {  # portable digest; macOS has shasum, Linux sha256sum
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then shasum -a 256 "$1" | awk '{print $1}'
  else echo ""; fi
}

avail_gib() {  # free GiB on the filesystem holding $1 (walks up to the nearest existing dir)
  local p=$1; while [ ! -e "$p" ] && [ "$p" != "/" ]; do p=$(dirname "$p"); done
  df -Pk "$p" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}'
}

printf '%s\n' "DhakaScenes annotation pipeline — bootstrap"
printf '%s\n' "repo:  $REPO_ROOT"
printf '%s\n' "flags: update=$DO_UPDATE cpu_only=$CPU_ONLY with_cvat=$WITH_CVAT skip_tests=$SKIP_TESTS skip_weights=$SKIP_WEIGHTS yes=$ASSUME_YES"

# ===========================================================================
step "1/10  Preflight"
# ===========================================================================
if [ "${BASH_VERSINFO[0]:-0}" -ge 4 ]; then
  ok "bash ${BASH_VERSION}"
else
  fail "bash >= 4 required (found ${BASH_VERSION:-unknown}); this script uses arrays and mapfile"
fi

for t in git tmux; do
  if command -v "$t" >/dev/null 2>&1; then ok "$t  $(command -v $t)"
  else fail "$t not found — required (tmux hosts every long run; see docs/RUNNING.md)"; fi
done

# docker is needed ONLY by CVAT, which is the review server, not the pipeline.
if command -v docker >/dev/null 2>&1; then ok "docker  $(command -v docker)"
else warn "docker not found — CVAT publish steps (run_stages.sh cvat/cvat3d/cvatroad) cannot run"; fi

HAS_GPU=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  HAS_GPU=1
  ok "nvidia-smi  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
else
  warn "no usable nvidia-smi — CPU-only setup is allowed, but these will NOT run:"
  note "stages 2 (OOD embeddings), 3/3b/3f (detectors), 4 (SAM masks), 5 (lift), road;"
  note "the VRAM-cap contract (C1/C19) is meaningless, and DHAKASCENES_RUN_GPU_TESTS must stay 0."
fi

# conda: PATH first, then the usual install roots. A conda that exists but is not
# on PATH is the common case on a fresh box (the installer does not touch
# non-login shells), and failing on that would be a false negative.
CONDA=""
if command -v conda >/dev/null 2>&1; then CONDA=$(command -v conda)
else
  for c in "${CONDA_EXE-}" "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" \
           "$HOME/miniforge3/bin/conda" "$HOME/mambaforge/bin/conda" /opt/conda/bin/conda; do
    [ -n "$c" ] && [ -x "$c" ] && { CONDA=$c; break; }
  done
fi
if [ -n "$CONDA" ]; then
  ok "conda  $CONDA"
  [ -z "$(command -v conda 2>/dev/null)" ] && note "not on PATH; this script uses the absolute path, but add it to your shell rc"
else
  fail "conda not found — the environment contract (environment.yml) is conda-based"
  note "install miniconda, then re-run:"
  note "  curl -fsSLo /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
  note "  bash /tmp/miniconda.sh -b -p \$HOME/miniconda3 && \$HOME/miniconda3/bin/conda init bash"
fi

CONDA_BASE=""; ENV_PREFIX=""; ENV_PY=""
if [ -n "$CONDA" ]; then
  CONDA_BASE=$("$CONDA" info --base 2>/dev/null)
  ENV_PREFIX="$CONDA_BASE/envs/$ENV_NAME"
  ENV_PY="$ENV_PREFIX/bin/python"
fi

# Disk. Three filesystems matter and they are usually not the same one: the repo,
# the conda base (the cu124 wheels alone are ~7 GB), and the write roots (a single
# full run leaves tens of GB of clouds and masks under work_root).
# The work_root read here is deliberately crude — PyYAML does not exist yet. The
# authoritative parse happens in step 5 and any disagreement is caught there.
WORK_ROOT_GUESS=$(awk -F': *' '/^work_root:/ {print $2; exit}' "$REPO_ROOT/configs/paths.yaml" 2>/dev/null)
for spec in "repo:$REPO_ROOT:2" "conda:${CONDA_BASE:-$HOME}:20" "write-roots:${WORK_ROOT_GUESS:-$HOME}:50"; do
  IFS=: read -r label path need <<<"$spec"
  got=$(avail_gib "$path")
  if [ -z "$got" ]; then warn "could not read free space for $label ($path)"
  elif [ "$got" -lt "$need" ]; then
    if [ "$got" -lt 5 ]; then fail "$label filesystem ($path): ${got} GiB free, needs >= ${need} GiB"
    else warn "$label filesystem ($path): ${got} GiB free, ${need} GiB recommended"; fi
  else ok "disk $label ($path): ${got} GiB free"; fi
done

# Python 3.10 is pinned because nuscenes-devkit 1.1.11 has no wheels above 3.11.
if [ -n "$ENV_PY" ] && [ -x "$ENV_PY" ]; then
  v=$("$ENV_PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)
  if [ "$v" = "3.10" ]; then ok "python 3.10 present in $ENV_NAME"
  else fail "$ENV_NAME runs python $v; environment.yml pins 3.10 (devkit has no wheels above 3.11)"; fi
elif [ -n "$CONDA" ]; then
  note "conda will fetch python 3.10 from conda-forge when the env is created (needs network)"
fi

# ===========================================================================
step "2/10  Conda environment '$ENV_NAME'"
# ===========================================================================
# environment.yml's pip: block lists BOTH requirements-torch.txt and
# requirements.txt, which conda hands to ONE pip process — exactly the
# process-global --index-url footgun requirements-torch.txt exists to prevent.
# So the env is created from a copy with the pip block stripped (conda's job is
# the interpreter), and step 3 does pip's job in the mandated order.
ENV_YML_NOPIP="$TMPD/environment-nopip.yml"
awk '
  /^[[:space:]]*-[[:space:]]*pip:[[:space:]]*$/ { match($0, /^[[:space:]]*/); ind = RLENGTH; skip = 1; next }
  skip == 1 {
    if ($0 ~ /^[[:space:]]*$/) next
    match($0, /^[[:space:]]*/)
    if (RLENGTH > ind) next
    skip = 0
  }
  { print }
' "$REPO_ROOT/environment.yml" > "$ENV_YML_NOPIP" 2>/dev/null

ENV_READY=0
if [ -z "$CONDA" ]; then
  fail "skipping env creation — no conda"
elif [ -x "$ENV_PY" ]; then
  ok "env exists: $ENV_PREFIX"
  ENV_READY=1
  if [ "$DO_UPDATE" -eq 1 ]; then
    if confirm "run 'conda env update --prune' on $ENV_NAME? (--prune can UNINSTALL packages)"; then
      if "$CONDA" env update -n "$ENV_NAME" -f "$ENV_YML_NOPIP" --prune; then ok "conda env update --prune"
      else fail "conda env update failed"; fi
    else skip "conda env update declined"; fi
  else
    skip "not touching it — pass --update to run 'conda env update --prune'"
  fi
else
  note "creating $ENV_NAME (python 3.10 + pip only; packages come from pip in step 3)"
  if "$CONDA" env create -n "$ENV_NAME" -f "$ENV_YML_NOPIP"; then
    ok "created $ENV_PREFIX"; ENV_READY=1
  else
    fail "conda env create failed"
  fi
fi
[ "$ENV_READY" -eq 1 ] && ok "interpreter (use this one, always): $ENV_PY"

# ===========================================================================
step "3/10  Pinned dependencies"
# ===========================================================================
# THE ORDER IS THE CONTRACT (requirements.txt header):
#   1. requirements-torch.txt        alone, because --index-url is process-global:
#                                    fold it into requirements.txt and pip weighs the
#                                    PyTorch index for EVERY name in the file, resolving
#                                    by version rather than by index priority.
#   2. requirements.txt              the pinned set, from PyPI
#   3. --no-deps requirements-devkit nuscenes-devkit pins matplotlib<3.6.0 and sam3 pins
#                                    timm>=1.0.17; both bounds are stale and honouring
#                                    them would drag the whole env backwards.
TORCH_REQ="$REPO_ROOT/requirements-torch.txt"
if [ "$CPU_ONLY" -eq 1 ]; then
  # Same file, CPU index, and the +cu124 local versions dropped (the CPU wheels
  # carry no local version). Written to scratch so the tracked file is untouched.
  TORCH_REQ="$TMPD/requirements-torch-cpu.txt"
  sed -e 's#/whl/cu124#/whl/cpu#' -e 's/+cu124//' "$REPO_ROOT/requirements-torch.txt" > "$TORCH_REQ"
  warn "--cpu-only: installing CPU torch wheels. GPU stages will NOT run and no fit claim is quotable."
fi

# Offline satisfaction check, so a second run is fast and touches no index. Every
# `name==version` in the three files must be installed at that version; range pins
# (the C19 HF-ecosystem entries) and git URLs need only be present.
deps_report() {
  "$ENV_PY" - "$REPO_ROOT" "$TORCH_REQ" <<'PYEOF'
import re, sys
from pathlib import Path
import importlib.metadata as im
root, torch_req = Path(sys.argv[1]), Path(sys.argv[2])
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
have = {norm(d.metadata["Name"]): d.version for d in im.distributions() if d.metadata.get("Name")}
problems = []
for f in (torch_req, root / "requirements.txt", root / "requirements-devkit.txt"):
    for line in f.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-"):
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*@\s*", line)          # git+ URL
        if m:
            if norm(m.group(1)) not in have:
                problems.append(f"missing {m.group(1)} ({f.name})")
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s;]+)$", line)      # exact pin
        if m:
            n, want = norm(m.group(1)), m.group(2).split("+")[0]
            if n not in have:
                problems.append(f"missing {m.group(1)}=={want} ({f.name})")
            elif have[n].split("+")[0] != want:
                problems.append(f"{m.group(1)}: have {have[n]}, pinned {want} ({f.name})")
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)[<>=!~]", line)           # range pin
        if m and norm(m.group(1)) not in have:
            problems.append(f"missing {m.group(1)} ({f.name})")
print("\n".join(problems[:20]))
sys.exit(1 if problems else 0)
PYEOF
}

if [ "$ENV_READY" -ne 1 ]; then
  fail "skipping dependency install — no usable env interpreter"
else
  NEED_INSTALL=1
  if [ "$DO_UPDATE" -eq 1 ]; then
    note "--update: reinstalling regardless of what is already resolved"
  elif out=$(deps_report); then
    NEED_INSTALL=0
    skip "every pin in the three requirements files is already satisfied"
    # deps_report compares versions with the local tag stripped, so a cu124 build
    # already installed reads as satisfying the CPU pin. Say so rather than
    # silently downgrading a working GPU env: --update forces the swap.
    if [ "$CPU_ONLY" -eq 1 ] && "$ENV_PY" -c 'import torch,sys; sys.exit(0 if "+" in torch.__version__ else 1)' 2>/dev/null; then
      warn "--cpu-only but a CUDA torch build is already installed; keeping it (use --update to replace it)"
    fi
  else
    note "not yet satisfied:"; printf '%s\n' "$out" | sed 's/^/         /'
  fi

  if [ "$NEED_INSTALL" -eq 1 ]; then
    if "$ENV_PY" -m pip install -r "$TORCH_REQ"; then ok "1/3 requirements-torch.txt"
    else fail "pip install -r requirements-torch.txt failed"; fi
    if "$ENV_PY" -m pip install -r "$REPO_ROOT/requirements.txt"; then ok "2/3 requirements.txt"
    else fail "pip install -r requirements.txt failed"; fi
    if "$ENV_PY" -m pip install --no-deps -r "$REPO_ROOT/requirements-devkit.txt"; then ok "3/3 requirements-devkit.txt (--no-deps)"
    else fail "pip install --no-deps -r requirements-devkit.txt failed"; fi
    if out=$(deps_report); then ok "all pins resolved"
    else warn "pins still unsatisfied after install:"; printf '%s\n' "$out" | sed 's/^/         /'; fi
  fi

  # cvat_sdk is imported by scripts/cvat_setup*.py, cvat_purge.py and
  # import_cvat_3d.py, and `run_stages.sh` publishes to CVAT in its DEFAULT chain
  # — but it appears only in requirements-lock.txt, never in requirements.txt.
  # That is a gap in the manifest (requirements.txt rule 3 says anything imported
  # by this repo belongs there); until it is closed, a fresh clone that skipped
  # this line would fail at the publish step, so install the lock's exact pin.
  if "$ENV_PY" -c 'import cvat_sdk' >/dev/null 2>&1; then
    skip "cvat_sdk present"
  else
    CVAT_PIN=$(awk -F'==' '/^cvat[-_]sdk==/ {print $2; exit}' "$REPO_ROOT/requirements-lock.txt")
    warn "cvat_sdk is missing from requirements.txt but imported by scripts/ — installing the lock pin ${CVAT_PIN:-latest}"
    if "$ENV_PY" -m pip install "cvat_sdk==${CVAT_PIN:?no cvat_sdk pin in requirements-lock.txt}"; then ok "cvat_sdk==$CVAT_PIN"
    else fail "pip install cvat_sdk==$CVAT_PIN failed"; fi
  fi
fi

# ===========================================================================
step "4/10  .env"
# ===========================================================================
ENV_FILE="$REPO_ROOT/.env"
EXAMPLE="$REPO_ROOT/.env.example"

env_get() {  # env_get FILE KEY -> value (empty if unset/absent)
  [ -f "$1" ] || return 0
  awk -v k="$2" -F= '$1 == k { sub("^" k "=", ""); print; exit }' "$1"
}
env_set() {  # env_set FILE KEY VALUE — replace in place, or append. Never duplicates a key.
  local f=$1 k=$2 v=$3
  if grep -qE "^${k}=" "$f" 2>/dev/null; then
    awk -v k="$k" -v v="$v" '{ if (index($0, k "=") == 1) print k "=" v; else print }' "$f" > "$f.tmp" \
      && mv "$f.tmp" "$f"
  else
    printf '%s=%s\n' "$k" "$v" >> "$f"
  fi
}

# Cache roots are siblings of the write roots, exactly as .env.example describes
# them ("siblings of work_root/out_root in configs/paths.yaml, not children of the
# project"). Deriving them from work_root's parent reproduces .env.example's own
# values on the reference machine without hardcoding anybody's home directory.
CACHE_BASE=$(dirname "${WORK_ROOT_GUESS:-$HOME/dhakascenes/work}")
CKPT_DIR_DEFAULT="$CACHE_BASE/cache/checkpoints"

# Thread pins: .env.example records 8 on this 32-core host. nproc/4 clamped to
# [1,8] reproduces that and stops a 4-core laptop from oversubscribing BLAS.
NCPU=$( { command -v nproc >/dev/null 2>&1 && nproc; } || echo 4)
THREADS=$(( NCPU / 4 )); [ "$THREADS" -lt 1 ] && THREADS=1; [ "$THREADS" -gt 8 ] && THREADS=8

# VRAM cap tiers, per the .env.example comment block (C1 + C19):
#   24 GB-class card -> 22000  production budget, headroom for the display server
#   >= 8 GB card     -> 4096   the pilot tier, which is the BINDING 4 GB contract
#   small card / none-> EMPTY  "on the laptop itself the physical card IS the ceiling"
VRAM_CAP=""
if [ "$HAS_GPU" -eq 1 ]; then
  VRAM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  case "${VRAM_TOTAL:-0}" in ''|*[!0-9]*) VRAM_TOTAL=0 ;; esac
  if   [ "$VRAM_TOTAL" -ge 24000 ]; then VRAM_CAP=22000
  elif [ "$VRAM_TOTAL" -ge 8192  ]; then VRAM_CAP=4096
  else VRAM_CAP=""; fi
fi

if [ ! -f "$EXAMPLE" ]; then
  fail ".env.example missing — it is the canonical variable list; cannot derive .env"
elif [ -f "$ENV_FILE" ]; then
  skip ".env exists — not overwritten (it holds this machine's secrets)"
  # Report the delta both ways: a key the example declares but .env lacks is an
  # unset knob; a key the CODE reads that the example does not declare is a hole
  # in the contract .env.example claims to be.
  missing=$(comm -23 \
    <(grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' "$EXAMPLE" | tr -d '=' | sort -u) \
    <(grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' "$ENV_FILE" | tr -d '=' | sort -u))
  if [ -n "$missing" ]; then
    warn "keys in .env.example but not in your .env:"; printf '%s\n' "$missing" | sed 's/^/         /'
  else
    ok "every .env.example key is present in .env"
  fi
  for k in HF_TOKEN CVAT_HOST CVAT_USER CVAT_PASSWORD; do
    [ -z "$(env_get "$ENV_FILE" "$k")" ] && warn "$k is empty in .env — fill it by hand (see the summary)"
  done
else
  cp "$EXAMPLE" "$ENV_FILE" || fail "could not create .env"
  if [ -f "$ENV_FILE" ]; then
    env_set "$ENV_FILE" HF_HOME               "$CACHE_BASE/cache/huggingface"
    env_set "$ENV_FILE" HUGGINGFACE_HUB_CACHE "$CACHE_BASE/cache/huggingface/hub"
    env_set "$ENV_FILE" TORCH_HOME            "$CACHE_BASE/cache/torch"
    env_set "$ENV_FILE" YOLO_CONFIG_DIR       "$CACHE_BASE/cache/ultralytics"
    env_set "$ENV_FILE" MOBILE_SAM_CHECKPOINT "${WEIGHTS_DIR:-$CKPT_DIR_DEFAULT}/mobile_sam.pt"
    env_set "$ENV_FILE" YOLO11_CHECKPOINT     "${WEIGHTS_DIR:-$CKPT_DIR_DEFAULT}/yolo11x.pt"
    env_set "$ENV_FILE" PYTHONNOUSERSITE      1
    env_set "$ENV_FILE" DHAKASCENES_PATHS_CONFIG configs/paths.yaml
    # Determinism block (§1.9): these four remove nondeterminism that has no
    # config meaning. The global SEED is deliberately NOT here — it changes
    # output, so it lives in configs/pipeline_pilot.yaml and in every manifest.
    env_set "$ENV_FILE" CUBLAS_WORKSPACE_CONFIG   ':4096:8'
    env_set "$ENV_FILE" DHAKASCENES_ALLOW_TF32     0
    env_set "$ENV_FILE" DHAKASCENES_CUDNN_BENCHMARK 0
    env_set "$ENV_FILE" TOKENIZERS_PARALLELISM     false
    env_set "$ENV_FILE" OMP_NUM_THREADS "$THREADS"
    env_set "$ENV_FILE" MKL_NUM_THREADS "$THREADS"
    env_set "$ENV_FILE" DHAKASCENES_VRAM_CAP_MIB "$VRAM_CAP"
    # Secrets stay EMPTY. Nothing is ever copied from another machine's .env.
    env_set "$ENV_FILE" HF_TOKEN ""
    # CVAT_HOST/USER/PASSWORD are read by scripts/cvat_*.py but are NOT declared
    # in .env.example. Appended here as empty placeholders so the operator has
    # somewhere to put them; the real fix belongs in .env.example.
    if ! grep -q '^CVAT_HOST=' "$ENV_FILE"; then
      cat >> "$ENV_FILE" <<'CVATEOF'

# ===========================================================================
# CVAT REVIEW SERVER — read by scripts/cvat_setup.py, cvat_setup_3d.py,
# cvat_purge.py and import_cvat_3d.py. NOT yet declared in .env.example
# (added by setup.sh); the default host in those scripts is localhost:8081.
# ===========================================================================
CVAT_HOST=
CVAT_USER=
CVAT_PASSWORD=
CVATEOF
    fi
    ok "created .env from .env.example with machine-derived values"
    note "cache roots -> $CACHE_BASE/cache/{huggingface,torch,ultralytics}"
    note "OMP_NUM_THREADS=MKL_NUM_THREADS=$THREADS (from nproc=$NCPU)"
    if [ -n "$VRAM_CAP" ]; then note "DHAKASCENES_VRAM_CAP_MIB=$VRAM_CAP (card reports ${VRAM_TOTAL} MiB)"
    else note "DHAKASCENES_VRAM_CAP_MIB left EMPTY (no GPU, or the physical card is itself the ceiling)"; fi
    warn "secrets left blank on purpose: HF_TOKEN, CVAT_HOST, CVAT_USER, CVAT_PASSWORD"
  fi
fi

# Everything below reads the resolved .env, so later steps see the operator's
# own values (a pre-existing checkpoint path wins over this script's default).
if [ -f "$ENV_FILE" ]; then set -a; . "$ENV_FILE"; set +a; fi
[ -n "$PRESET_PATHS_CFG" ] && export DHAKASCENES_PATHS_CONFIG="$PRESET_PATHS_CFG"

# ===========================================================================
step "5/10  Write roots (configs/paths.yaml §1.8)"
# ===========================================================================
PATHS_CFG=${DHAKASCENES_PATHS_CONFIG:-configs/paths.yaml}
case "$PATHS_CFG" in /*) : ;; *) PATHS_CFG="$REPO_ROOT/$PATHS_CFG" ;; esac
PATHS_OUT="$TMPD/paths.env"

if [ "$ENV_READY" -ne 1 ]; then
  fail "cannot parse $PATHS_CFG — no env interpreter (PyYAML lives in the env)"
elif [ ! -f "$PATHS_CFG" ]; then
  fail "$PATHS_CFG not found"
else
  # Parsed with PyYAML through the env's interpreter, never grep: a quoted value,
  # an anchor or a comment on the same line all break a regex and none of them
  # break the loader that pipeline/common/paths.py actually uses.
  "$ENV_PY" - "$PATHS_CFG" "$REPO_ROOT" > "$PATHS_OUT" <<'PYEOF'
import os, sys, yaml
cfg, repo = sys.argv[1], os.path.realpath(sys.argv[2])
d = yaml.safe_load(open(cfg)) or {}
errs, out = [], {}
for k in ("dataroot", "meta_root", "version", "work_root", "out_root", "probe_out_root"):
    v = d.get(k)
    if not v:
        errs.append(f"{k} is missing or empty in {cfg}")
    out[k] = str(v) if v else ""
dataroot = os.path.realpath(os.path.expanduser(out["dataroot"])) if out["dataroot"] else ""
def contains(parent, child):
    try:
        return os.path.commonpath([parent, child]) == parent
    except ValueError:
        return False
for k in ("work_root", "out_root", "probe_out_root"):
    if not out[k]:
        continue
    p = os.path.realpath(os.path.expanduser(out[k]))
    out[k + "_abs"] = p
    if contains(repo, p):
        errs.append(f"{k} ({p}) is INSIDE the repo — §1.8 forbids it "
                    "(copytree/rsync -L/a docker build context all dereference it)")
    if dataroot and (contains(dataroot, p) or contains(p, dataroot)):
        errs.append(f"{k} ({p}) is not disjoint from dataroot ({dataroot})")
roots = [(k, out.get(k + "_abs")) for k in ("work_root", "out_root", "probe_out_root")]
for i in range(len(roots)):
    for j in range(i + 1, len(roots)):
        a, b = roots[i][1], roots[j][1]
        if a and b and (a == b or contains(a, b) or contains(b, a)):
            errs.append(f"{roots[i][0]} and {roots[j][0]} are not disjoint ({a} / {b})")
for k, v in out.items():
    print(f"PATH_{k.upper()}={v}")
print(f"PATH_DATAROOT_ABS={dataroot}")
for e in errs:
    print(f"ERR={e}")
PYEOF
  if [ ! -s "$PATHS_OUT" ]; then
    fail "could not parse $PATHS_CFG (is PyYAML installed in $ENV_NAME?)"
  else
    # shellcheck disable=SC1090
    while IFS='=' read -r k v; do case "$k" in PATH_*) eval "$k=\$v" ;; esac; done < "$PATHS_OUT"
    ERRS=$(awk -F'ERR=' '/^ERR=/ {print $2}' "$PATHS_OUT")
    if [ -n "$ERRS" ]; then
      while IFS= read -r e; do fail "paths.yaml: $e"; done <<<"$ERRS"
      note "refusing to create write roots until the contract holds"
    else
      ok "$PATHS_CFG: roots outside the repo and pairwise disjoint from dataroot"
      for r in "${PATH_WORK_ROOT_ABS-}" "${PATH_OUT_ROOT_ABS-}" "${PATH_PROBE_OUT_ROOT_ABS-}"; do
        [ -z "$r" ] && continue
        if [ -d "$r" ]; then skip "exists: $r"
        elif mkdir -p "$r" 2>/dev/null; then ok "created: $r"
        else fail "could not create $r"; fi
      done
    fi
    # dataroot is the ONE per-machine fact this script must not invent. Creating an
    # empty one would turn "you have not pointed me at the data" into stage 0's
    # much less obvious "no usable scenes".
    if [ -n "${PATH_DATAROOT_ABS-}" ] && [ -d "$PATH_DATAROOT_ABS" ]; then
      if [ -d "$PATH_DATAROOT_ABS/${PATH_VERSION-}" ]; then ok "dataroot present with version dir ${PATH_VERSION-}"
      else warn "dataroot $PATH_DATAROOT_ABS has no '${PATH_VERSION-}' directory — version must match on-disk name verbatim"; fi
    else
      warn "dataroot does not exist: ${PATH_DATAROOT_ABS:-<unset>}"
      note "this is the per-machine bit YOU supply — edit $PATHS_CFG (dataroot/meta_root/version)"
    fi
  fi
fi

# ===========================================================================
step "6/10  Model checkpoints"
# ===========================================================================
# Both are FILES, not hub ids, and both adapters refuse anything that is not an
# existing path. YOLO("yolo11x.pt") on a missing file downloads into the CWD —
# i.e. into the repo tree — which §1.8 forbids, which is exactly why the path is
# explicit in .env rather than implicit in ultralytics' default.
# The default destination is the checkpoints directory .env names, OUTSIDE the
# repo. `--weights-dir` overrides it; pointing it at ./weights works (that path is
# git-ignored) but puts ~150 MB inside a tree §1.8 says must stay small.
MOBILE_SAM_DEST=${MOBILE_SAM_CHECKPOINT:-$CKPT_DIR_DEFAULT/mobile_sam.pt}
YOLO11_DEST=${YOLO11_CHECKPOINT:-$CKPT_DIR_DEFAULT/yolo11x.pt}
if [ -n "$WEIGHTS_DIR" ]; then
  MOBILE_SAM_DEST="$WEIGHTS_DIR/mobile_sam.pt"
  YOLO11_DEST="$WEIGHTS_DIR/yolo11x.pt"
fi

# URL and digest come out of the tracked files rather than being retyped here, so
# there is exactly one place to change them: .env.example records the ultralytics
# release URL and the expected sha256; requirements-torch.txt pins the MobileSAM
# commit, and the weights are fetched from that same commit rather than master.
YOLO_URL=$(grep -oE 'https://github\.com/ultralytics/assets/releases/download/[^ ]*yolo11x\.pt' "$EXAMPLE" 2>/dev/null | head -1)
YOLO_SHA=$(grep -oE 'sha256[[:space:]]+[0-9a-f]{64}' "$EXAMPLE" 2>/dev/null | head -1 | awk '{print $2}')
MSAM_COMMIT=$(grep -oE 'MobileSAM\.git@[0-9a-f]{40}' "$REPO_ROOT/requirements-torch.txt" 2>/dev/null | head -1 | cut -d@ -f2)
MSAM_URL="https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/${MSAM_COMMIT:-master}/weights/mobile_sam.pt"

fetch_to() {  # fetch_to URL DEST — atomic: a half-download never lands under the real name
  local url=$1 dest=$2
  mkdir -p "$(dirname "$dest")" || return 1
  if command -v curl >/dev/null 2>&1; then curl -fL --retry 3 --connect-timeout 20 -o "$dest.part" "$url"
  elif command -v wget >/dev/null 2>&1; then wget -q -O "$dest.part" "$url"
  else echo "neither curl nor wget" >&2; return 1; fi || { rm -f "$dest.part"; return 1; }
  mv "$dest.part" "$dest"
}

if [ "$SKIP_WEIGHTS" -eq 1 ]; then
  skip "--skip-weights"
  note "expected: $MOBILE_SAM_DEST"
  note "expected: $YOLO11_DEST"
else
  # --- yolo11x.pt: digest is published, so verify it and fail on mismatch ---
  if [ -f "$YOLO11_DEST" ] && [ -n "$YOLO_SHA" ] && [ "$(sha256_of "$YOLO11_DEST")" = "$YOLO_SHA" ]; then
    skip "yolo11x.pt present, sha256 matches: $YOLO11_DEST"
  elif [ -z "$YOLO_URL" ]; then
    fail "no yolo11x release URL found in .env.example"
  else
    [ -f "$YOLO11_DEST" ] && warn "yolo11x.pt present but sha256 does not match the pin — re-downloading"
    note "downloading $YOLO_URL"
    if fetch_to "$YOLO_URL" "$YOLO11_DEST"; then
      got=$(sha256_of "$YOLO11_DEST")
      if [ -n "$YOLO_SHA" ] && [ "$got" != "$YOLO_SHA" ]; then
        fail "yolo11x.pt sha256 mismatch: got $got, expected $YOLO_SHA"
      else ok "yolo11x.pt -> $YOLO11_DEST (sha256 $got)"; fi
    else fail "yolo11x.pt download failed"; fi
  fi

  # --- mobile_sam.pt: upstream publishes no digest, so record the one we got ---
  if [ -f "$MOBILE_SAM_DEST" ]; then
    skip "mobile_sam.pt present: $MOBILE_SAM_DEST (sha256 $(sha256_of "$MOBILE_SAM_DEST"))"
  else
    note "downloading $MSAM_URL"
    if fetch_to "$MSAM_URL" "$MOBILE_SAM_DEST"; then
      ok "mobile_sam.pt -> $MOBILE_SAM_DEST (sha256 $(sha256_of "$MOBILE_SAM_DEST"))"
      note "upstream publishes no checksum; §1.9 hashes the bytes into every Stage 4 manifest"
    else fail "mobile_sam.pt download failed (pinned commit ${MSAM_COMMIT:-master})"; fi
  fi

  # Write the resolved absolute paths back — but only into keys that are absent or
  # empty. An operator who already pointed these somewhere keeps their value.
  if [ -f "$ENV_FILE" ]; then
    [ -z "$(env_get "$ENV_FILE" MOBILE_SAM_CHECKPOINT)" ] && { env_set "$ENV_FILE" MOBILE_SAM_CHECKPOINT "$MOBILE_SAM_DEST"; ok "wrote MOBILE_SAM_CHECKPOINT into .env"; }
    [ -z "$(env_get "$ENV_FILE" YOLO11_CHECKPOINT)" ]     && { env_set "$ENV_FILE" YOLO11_CHECKPOINT "$YOLO11_DEST"; ok "wrote YOLO11_CHECKPOINT into .env"; }
  fi
fi

# ===========================================================================
step "7/10  Verification"
# ===========================================================================
if [ "$ENV_READY" -ne 1 ]; then
  fail "skipping verification — no env interpreter"
else
  # Import smoke test: the seven imports that fail differently. torch says whether
  # the CUDA wheels actually see the driver; nuscenes and cvat_sdk are the two
  # --no-deps / undeclared installs most likely to be silently absent.
  if "$ENV_PY" - <<'PYEOF'
import sys
import torch, numpy, scipy, yaml, PIL, cvat_sdk, nuscenes
print(f"       python           {sys.version.split()[0]}")
print(f"       torch            {torch.__version__}   cuda.is_available()={torch.cuda.is_available()}")
print(f"       numpy/scipy      {numpy.__version__} / {scipy.__version__}")
print(f"       PyYAML/Pillow    {yaml.__version__} / {PIL.__version__}")
print(f"       cvat_sdk         {getattr(cvat_sdk, '__version__', 'installed')}")
print(f"       nuscenes-devkit  {getattr(nuscenes, '__version__', 'installed')}")
PYEOF
  then ok "import smoke test"
  else fail "import smoke test failed — the env cannot run the pipeline"; fi

  if [ "$SKIP_TESTS" -eq 1 ]; then
    skip "--skip-tests (pytest not run)"
  else
    # The unit tier is GPU-free and synthetic by design; the integration tier stays
    # behind DHAKASCENES_RUN_INTEGRATION so a fresh clone cannot silently skip-and-pass.
    PYTEST_LOG="$TMPD/pytest.log"
    ( cd "$REPO_ROOT" && "$ENV_PY" -m pytest tests/ -q ) 2>&1 | tee "$PYTEST_LOG"
    rc=${PIPESTATUS[0]}
    summary=$(tail -5 "$PYTEST_LOG" | grep -E '(passed|failed|error)' | tail -1)
    if [ "$rc" -eq 0 ]; then ok "pytest tests/ -q — ${summary:-passed}"
    else fail "pytest tests/ -q — ${summary:-exit $rc}"; fi
  fi

  [ -n "${PATH_DATAROOT-}" ] && {
    printf '       resolved %s:\n' "$PATHS_CFG"
    note "  dataroot       ${PATH_DATAROOT-}"
    note "  meta_root      ${PATH_META_ROOT-}"
    note "  version        ${PATH_VERSION-}"
    note "  work_root      ${PATH_WORK_ROOT-}"
    note "  out_root       ${PATH_OUT_ROOT-}"
    note "  probe_out_root ${PATH_PROBE_OUT_ROOT-}"
  }
fi

# ===========================================================================
step "8/10  tmux session 'pipe'"
# ===========================================================================
# Every long run is launched into this session (a chain outlives an ssh session
# only if it is detached). Existing windows are never touched — the operator's
# running chain lives in them.
if ! command -v tmux >/dev/null 2>&1; then
  fail "tmux missing — cannot prepare the run session"
elif tmux has-session -t pipe 2>/dev/null; then
  skip "session 'pipe' exists ($(tmux list-windows -t pipe 2>/dev/null | wc -l) windows, untouched)"
elif tmux new-session -d -s pipe 2>/dev/null; then
  ok "created detached tmux session 'pipe'"
else
  fail "could not create tmux session 'pipe'"
fi

# ===========================================================================
step "9/10  CVAT review server"
# ===========================================================================
CVAT_DIR="$(dirname "$REPO_ROOT")/cvat"
[ -d "$CVAT_DIR" ] || { [ -d "$HOME/cvat" ] && CVAT_DIR="$HOME/cvat"; }
CVAT_URL=${CVAT_HOST:-http://localhost:8081}
if [ "$WITH_CVAT" -eq 1 ]; then
  if ! command -v docker >/dev/null 2>&1; then
    fail "--with-cvat but docker is not installed"
  else
    if [ -d "$CVAT_DIR/.git" ]; then
      skip "CVAT checkout exists: $CVAT_DIR (not updated — a pull can change the schema)"
    elif git clone https://github.com/cvat-ai/cvat "$CVAT_DIR"; then
      ok "cloned CVAT into $CVAT_DIR"
    else
      fail "git clone of cvat-ai/cvat failed"
    fi
    if [ -f "$CVAT_DIR/docker-compose.yml" ]; then
      if ( cd "$CVAT_DIR" && docker compose up -d ); then ok "docker compose up -d ($CVAT_DIR)"
      else fail "docker compose up failed in $CVAT_DIR"; fi
    fi
  fi
else
  skip "--with-cvat not given; only probing $CVAT_URL"
fi
if command -v curl >/dev/null 2>&1; then
  code=$(curl -s -o /dev/null -m 5 -w '%{http_code}' "$CVAT_URL" 2>/dev/null)
  case "$code" in
    000|"") warn "$CVAT_URL does not answer — CVAT publish steps will fail until it does" ;;
    *)      ok "$CVAT_URL answers (HTTP $code)" ;;
  esac
fi
note "create the superuser once, after the stack is up:"
note "  docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'"
note "then put that login in .env as CVAT_HOST / CVAT_USER / CVAT_PASSWORD"

# ===========================================================================
step "10/10  Summary"
# ===========================================================================
printf '\n  Ready:\n'
[ "$ENV_READY" -eq 1 ] && printf '    [x] conda env %s  -> %s\n' "$ENV_NAME" "$ENV_PY" || printf '    [ ] conda env %s\n' "$ENV_NAME"
[ -f "$ENV_FILE" ] && printf '    [x] .env\n' || printf '    [ ] .env\n'
[ -d "${PATH_WORK_ROOT_ABS-/nonexistent}" ] && printf '    [x] write roots\n' || printf '    [ ] write roots\n'
{ [ -f "$YOLO11_DEST" ] && [ -f "$MOBILE_SAM_DEST" ]; } && printf '    [x] checkpoints\n' || printf '    [ ] checkpoints (%s, %s)\n' "$YOLO11_DEST" "$MOBILE_SAM_DEST"
tmux has-session -t pipe 2>/dev/null && printf '    [x] tmux session pipe\n' || printf '    [ ] tmux session pipe\n'

cat <<SUMMARY

  WHAT YOU MUST DO BY HAND — the script cannot guess any of these:

    1. Fill the secrets in .env:
         HF_TOKEN        needed on first download of the gated facebook/sam3 and
                         facebook/sam3.1 repos (Stage 4's default and C26 provider).
         CVAT_HOST       e.g. http://localhost:8081
         CVAT_USER
         CVAT_PASSWORD
       Then, once: $ENV_PY -m huggingface_hub.cli auth login

    2. Point $PATHS_CFG at THIS machine's dataset — the one
       genuinely per-machine fact:
         dataroot:   <blob root: samples/, sweeps/, maps/, can_bus/>
         meta_root:  <directory holding the 13 metadata tables>
         version:    <must equal the on-disk version directory name VERBATIM;
                      a mismatch reads downstream as "no usable scenes", not as
                      a config error>
       The substrate is not redistributable and is not in this repo.

    3. If you will use CVAT: bring the stack up (--with-cvat), create the
       superuser (command printed above), and log in once in a browser.

    4. Sanity-run before anything long:
         cd $REPO_ROOT
         $ENV_PY pipeline/stage0_data_probe/probe.py
         scripts/run_stages.sh --scenes <one-scene> --no-cvat
       Full chain and flags: docs/RUNNING.md; why anything is the way it is:
       docs/DECISIONS.md.
SUMMARY

if [ "$FAIL_COUNT" -gt 0 ]; then
  printf '\n%s%d step(s) FAILED:%s\n' "$C_FAIL" "$FAIL_COUNT" "$C_OFF"
  for f in "${FAILED[@]}"; do printf '    - %s\n' "$f"; done
  exit 1
fi
printf '\n%sbootstrap complete%s (%d warning(s))\n' "$C_OK" "$C_OFF" "$WARN_COUNT"
exit 0
