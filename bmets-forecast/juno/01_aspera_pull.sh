#!/usr/bin/env bash
# Download UCSD-BMETS-Longitudinal (163 GB, NIfTI) to Juno scratch with IBM Aspera.
# 1) On the TCIA collection page click "DOWNLOAD (163GB)"; TCIA opens an Aspera Faspex package.
#    Copy the package link (https://faspex.cancerimagingarchive.net/aspera/faspex/public/package?context=...).
# 2) Run this script on a Juno login/data-transfer node:  bash juno/01_aspera_pull.sh "<package link>"
# The four small TSVs (clinical, dictionaries, MRI metadata) go to $BMF_SCRATCH/clinical/.
set -euo pipefail
source "$(dirname "$0")/00_env.sh"
LINK="${1:?usage: 01_aspera_pull.sh <faspex package link>}"
mkdir -p "$BMF_DATA" "$BMF_SCRATCH/clinical"
if ! command -v ascli >/dev/null; then
  gem install --user-install aspera-cli && export PATH="$(ruby -e 'print Gem.user_dir')/bin:$PATH"
  ascli config ascp install
fi
# resumable; re-run the same command after an interruption
ascli faspex package receive --url="$LINK" --to-folder="$BMF_DATA" --ts=@json:'{"resume_policy":"sparse_csum","target_rate_kbps":500000}'
find "$BMF_DATA" -maxdepth 1 -type d -name 'BMETS*' | wc -l
echo "Expected 1065 study folders. Now copy the 4 TSVs into $BMF_SCRATCH/clinical/."
