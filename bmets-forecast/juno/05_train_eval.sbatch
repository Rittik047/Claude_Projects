#!/usr/bin/env bash
#SBATCH --job-name=bmf-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
source "${BMF_HOME:-$HOME/Claude_Projects/bmets-forecast}/juno/00_env.sh"
for TASK in T2a T2b; do
  for M in P0 P1 P2 P3 P5; do bmf train --config "$BMF_CONFIG" --model $M --task $TASK --device cuda; done
  bmf explain --config "$BMF_CONFIG" --task $TASK
done
bmf train --config "$BMF_CONFIG" --model P1 --task T3
bmf train --config "$BMF_CONFIG" --model O1 --task T4
bmf evaluate --config "$BMF_CONFIG"
