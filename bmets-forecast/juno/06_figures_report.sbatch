#!/usr/bin/env bash
#SBATCH --job-name=bmf-report
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
source "${BMF_HOME:-$HOME/Claude_Projects/bmets-forecast}/juno/00_env.sh"
bmf figures --config "$BMF_CONFIG"
bmf report  --config "$BMF_CONFIG"
echo "Copy reports/results.json, reports/tables/ and reports/figures/ back to the repo (aggregate only)."
