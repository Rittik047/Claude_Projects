#!/usr/bin/env bash
#SBATCH --job-name=bmf-segeval
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
# nnU-Net writes out-of-fold validation predictions to fold_k/validation; gather them and score lesion-wise.
source "${BMF_HOME:-$HOME/Claude_Projects/bmets-forecast}/juno/00_env.sh"
for ID in 501 502; do
  M=$([ $ID = 501 ] && echo S0 || echo S2)
  NAME=$(ls "$nnUNet_results" | grep "Dataset${ID}_")
  OUT="$BMF_SCRATCH/cache/seg_pred/$M"; mkdir -p "$OUT"
  cp "$nnUNet_results/$NAME"/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_*/validation/*.nii.gz "$OUT/"
  bmf seg-eval --config "$BMF_CONFIG" --pred-dir "$OUT" --method $M
done
