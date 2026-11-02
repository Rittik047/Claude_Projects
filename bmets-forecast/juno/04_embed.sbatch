#!/usr/bin/env bash
#SBATCH --job-name=bmf-embed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
# Foundation-model weights: put files in $BMF_SCRATCH/weights and name them in configs/juno.yaml (fm.weights).
source "${BMF_HOME:-$HOME/Claude_Projects/bmets-forecast}/juno/00_env.sh"
bmf track    --config "$BMF_CONFIG"
bmf labels   --config "$BMF_CONFIG"
bmf features --config "$BMF_CONFIG"
for ENC in brainsegfounder samed3d random3d; do
  bmf embed --config "$BMF_CONFIG" --encoder $ENC --device cuda || echo "encoder $ENC skipped (weights?)"
done
