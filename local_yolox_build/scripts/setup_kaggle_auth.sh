#!/usr/bin/env bash
# Write Kaggle API credentials to $KAGGLE_CONFIG_DIR/kaggle.json with 0600.
#
# The installed client (kaggle 1.7.4.5) requires BOTH `username` and `key`;
# it has no code path for a bare KGAT_ token used alone. The KGAT_ string IS
# the key -- it just has to be paired with the account's username.
#
# Usage:
#   KAGGLE_USERNAME=<your-kaggle-username> KAGGLE_KEY=<KGAT_...> ./setup_kaggle_auth.sh
#
# The credential never appears in this script, in the repository, or in shell
# history if you pass it via a leading space or an env file.
set -euo pipefail

: "${KAGGLE_USERNAME:?set KAGGLE_USERNAME to your Kaggle account username}"
: "${KAGGLE_KEY:?set KAGGLE_KEY to your Kaggle API token}"

CONFIG_DIR="${KAGGLE_CONFIG_DIR:-${HOME}/.config/kaggle}"
mkdir -p "${CONFIG_DIR}"
umask 177
printf '{"username":"%s","key":"%s"}\n' "${KAGGLE_USERNAME}" "${KAGGLE_KEY}" > "${CONFIG_DIR}/kaggle.json"
chmod 600 "${CONFIG_DIR}/kaggle.json"

echo "wrote ${CONFIG_DIR}/kaggle.json (0600)"
echo "verifying authentication ..."
KAGGLE_CONFIG_DIR="${CONFIG_DIR}" PYTHONNOUSERSITE=1 \
    /home/mt/miniconda3/envs/ano_pipe/bin/python -c "
import kaggle
kaggle.api.authenticate()
ds = kaggle.api.dataset_list(search='rsud20k')
print('auth OK -- dataset search returned', len(ds), 'result(s)')
for d in ds[:3]:
    print('   ', d.ref)
"
