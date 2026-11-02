#!/usr/bin/env bash
#SBATCH --job-name=bmf-pre
#SBATCH --array=0-31
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
source "${BMF_HOME:-$HOME/Claude_Projects/bmets-forecast}/juno/00_env.sh"
if [ "${SLURM_ARRAY_TASK_ID:-0}" = "0" ]; then bmf audit --config "$BMF_CONFIG"; fi
bmf index --config "$BMF_CONFIG"
bmf splits --config "$BMF_CONFIG"
bmf preprocess --config "$BMF_CONFIG" --shard "${SLURM_ARRAY_TASK_ID}/32"
