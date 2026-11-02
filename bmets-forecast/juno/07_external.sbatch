#!/usr/bin/env bash
#SBATCH --job-name=bmf-external
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
# Locked external validation. Run ONLY after the development analysis is final.
# 1) arrange each external cohort into TCIA layout: bmf to-layout --map <cohort_map.csv> --out <dir>
# 2) write configs/external_<cohort>.yaml (paths + same lock_dir as juno.yaml)
source "${BMF_HOME:-$HOME/Claude_Projects/bmets-forecast}/juno/00_env.sh"
for TASK in T2a T2b; do bmf lock --config "$BMF_CONFIG" --task $TASK; done
for EXT in "$BMF_HOME"/configs/external_*.yaml; do
  [ -e "$EXT" ] || continue
  for S in audit index preprocess track labels features; do bmf $S --config "$EXT"; done
  for TASK in T2a T2b; do bmf external --config "$EXT" --task $TASK; done
done
