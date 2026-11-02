#!/usr/bin/env bash
#SBATCH --job-name=bmf-index
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
# Writes the shared files (audit.json, timepoints.csv, index.csv, splits.json) ONCE before the preprocessing array.
source "${BMF_HOME:-$HOME/Claude_Projects/bmets-forecast}/juno/00_env.sh"
bmf audit  --config "$BMF_CONFIG"
bmf index  --config "$BMF_CONFIG"
bmf check  --config "$BMF_CONFIG"     # stops the chain if the label scheme or a checkpoint is wrong
bmf splits --config "$BMF_CONFIG"
