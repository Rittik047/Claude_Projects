#!/usr/bin/env bash
#SBATCH --job-name=bmf-segexp
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
source "${BMF_HOME:-$HOME/Claude_Projects/bmets-forecast}/juno/00_env.sh"
bmf seg-export --config "$BMF_CONFIG"
