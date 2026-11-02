#!/usr/bin/env bash
# Sourced by every Juno job. EDIT the four variables below once.
export JUNO_PARTITION_CPU="${JUNO_PARTITION_CPU:-normal}"     # `sinfo -s` lists Juno partitions
export JUNO_PARTITION_GPU="${JUNO_PARTITION_GPU:-gpu}"        # partition with A100s
export BMF_HOME="${BMF_HOME:-$HOME/Claude_Projects/bmets-forecast}"
export BMF_SCRATCH="${BMF_SCRATCH:-/scratch/$USER/bmets}"     # large, fast storage (>= 600 GB recommended)

export BMF_DATA="$BMF_SCRATCH/UCSD-BMETS-Longitudinal"         # Aspera download lands here
export BMF_CONFIG="$BMF_HOME/configs/juno.yaml"
export nnUNet_raw="$BMF_SCRATCH/cache/nnunet_raw"
export nnUNet_preprocessed="$BMF_SCRATCH/cache/nnunet_preprocessed"
export nnUNet_results="$BMF_SCRATCH/cache/nnunet_results"
mkdir -p "$BMF_SCRATCH/logs" "$nnUNet_preprocessed" "$nnUNet_results"

module purge 2>/dev/null || true
module load cuda 2>/dev/null || true                           # adjust to `module avail cuda` on Juno
if ! command -v conda >/dev/null; then module load anaconda3 2>/dev/null || module load miniconda 2>/dev/null || true; fi
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | grep -q "^bmf "; then
  conda create -y -n bmf python=3.11
  conda activate bmf
  pip install -e "$BMF_HOME[dl,extra,test]"
else
  conda activate bmf
fi
cd "$BMF_HOME"
