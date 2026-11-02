#!/usr/bin/env bash
#SBATCH --job-name=bmf-nnunet
#SBATCH --array=0-9                # 0-4: S0 (dataset 501) folds, 5-9: S2 prior-conditioned (502)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A_%a.out
source "${BMF_HOME:-$HOME/Claude_Projects/bmets-forecast}/juno/00_env.sh"
ID=$(( SLURM_ARRAY_TASK_ID < 5 ? 501 : 502 )); FOLD=$(( SLURM_ARRAY_TASK_ID % 5 ))
NAME=$(ls "$nnUNet_raw" | grep "Dataset${ID}_")
if [ ! -d "$nnUNet_preprocessed/$NAME" ]; then
  ( flock 9; [ -d "$nnUNet_preprocessed/$NAME/nnUNetPlans_3d_fullres" ] || \
    nnUNetv2_plan_and_preprocess -d $ID -c 3d_fullres --verify_dataset_integrity ) 9>"$BMF_SCRATCH/plan_${ID}.lock"
fi
cp "$nnUNet_raw/$NAME/splits_final.json" "$nnUNet_preprocessed/$NAME/splits_final.json"   # our patient-level folds
nnUNetv2_train $ID 3d_fullres $FOLD -tr nnUNetTrainer --npz
