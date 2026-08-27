#!/usr/bin/env bash
# Download RSUD20K from Kaggle into datasets/rsud20k/. Idempotent: skips if already present.
#
# Requires Kaggle credentials. This client (kaggle 1.7.4.5) needs BOTH a
# username and a key -- it does not accept a bare KGAT_ token. Credentials are
# read from $KAGGLE_CONFIG_DIR/kaggle.json (see scripts/setup_kaggle_auth.sh).
#
# license: RSUD20K is CC BY-NC 4.0 -- research / non-commercial use only.
set -euo pipefail

BUILD_DIR="/home/mt/Zami/Annotation_pipeline/local_yolox_build"
DEST="${BUILD_DIR}/datasets/rsud20k"
SLUG="hasibzunair/rsud20k-bangladesh-road-scene-understanding"
PY="/home/mt/miniconda3/envs/ano_pipe/bin/python"

export KAGGLE_CONFIG_DIR="${KAGGLE_CONFIG_DIR:-${HOME}/.config/kaggle}"

if [ ! -f "${KAGGLE_CONFIG_DIR}/kaggle.json" ]; then
    echo "ERROR: no credentials at ${KAGGLE_CONFIG_DIR}/kaggle.json" >&2
    echo "       run scripts/setup_kaggle_auth.sh first" >&2
    exit 2
fi

if [ -d "${DEST}/images/train" ]; then
    echo "already present at ${DEST}; nothing to do"
    exit 0
fi

mkdir -p "${DEST}"
echo "downloading ${SLUG} -> ${DEST}"
PYTHONNOUSERSITE=1 "${PY}" -c "
import kaggle
kaggle.api.dataset_download_files('${SLUG}', path='${DEST}', unzip=True, quiet=False)
print('download complete')
"

# The archive may nest everything one or two levels down; flatten so that
# images/ and labels/ sit directly under DEST.
if [ ! -d "${DEST}/images" ]; then
    inner="$(find "${DEST}" -maxdepth 3 -type d -name images | head -1)"
    if [ -n "${inner}" ]; then
        parent="$(dirname "${inner}")"
        echo "flattening from ${parent}"
        shopt -s dotglob
        mv "${parent}"/* "${DEST}/"
        shopt -u dotglob
    fi
fi

echo
echo "top-level contents of ${DEST}:"
find "${DEST}" -maxdepth 1 -mindepth 1 | sort
echo
echo "split directories found:"
find "${DEST}" -maxdepth 2 -mindepth 2 -type d | sort
