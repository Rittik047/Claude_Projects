#!/usr/bin/env bash
#SBATCH --job-name=bmf-smoke
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
# Smoke test: every CPU stage on 12 seeded patients (>=3 MRIs each), expert segmentations, random3d encoder.
# Writes only to cache_smoke/ and reports_smoke/. Run before run_all.sh:  sbatch juno/00b_smoke.sbatch
set -euo pipefail
source "${BMF_HOME:-$HOME/Claude_Projects/bmets-forecast}/juno/00_env.sh"
S=(--config "$BMF_CONFIG" --subset "${BMF_SMOKE_N:-12}")
for st in audit index splits preprocess track labels features; do bmf "$st" "${S[@]}"; done
bmf embed "${S[@]}" --encoder random3d
for m in P0 P1 P2 P3 P5; do bmf train "${S[@]}" --model "$m" --task T2a; done
bmf train "${S[@]}" --model P1 --task T3
bmf train "${S[@]}" --model O1 --task T4
for st in evaluate explain figures report; do bmf "$st" "${S[@]}"; done
echo "SMOKE OK"
