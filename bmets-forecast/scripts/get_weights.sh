#!/usr/bin/env bash
# Download foundation-model weights into $1 (default: $BMF_SCRATCH/weights on Juno, MyDrive/bmets/weights on Colab).
#   bash scripts/get_weights.sh /scratch/$USER/bmets/weights
# Then run `bmf check --config <route>.yaml`: it loads each checkpoint and refuses a wrong file.
set -euo pipefail
W="${1:?usage: get_weights.sh WEIGHTS_DIR}"; mkdir -p "$W"; cd "$W"
pip -q install gdown huggingface_hub

# SAM-Med3D-turbo (ViT-B, 128^3, 1 channel): Hugging Face blueyo0/SAM-Med3D, file sam_med3d_turbo.pth
[ -s sam_med3d_turbo.pth ] || curl -L --fail -o sam_med3d_turbo.pth \
  https://huggingface.co/blueyo0/SAM-Med3D/resolve/main/sam_med3d_turbo.pth
# its model code (no pip installer upstream): clone and export SAMMED3D_REPO
[ -d SAM-Med3D ] || git clone -q --depth 1 https://github.com/uni-medical/SAM-Med3D
echo "export SAMMED3D_REPO=$W/SAM-Med3D"

# BrainSegFounder: the authors' Google Drive folder (models + fold JSONs), linked from the repo README
[ -d brainsegfounder_drive ] || gdown --folder -O brainsegfounder_drive \
  https://drive.google.com/drive/folders/1fl3FeMEhv_cnIwrDa5geHPbKL-tHAuQE
echo; echo "BrainSegFounder candidates (pick the BraTS fine-tuned 4-channel SwinUNETR, any one fold):"
find brainsegfounder_drive -type f \( -name '*.pt' -o -name '*.pth' \) -printf '  %s bytes  %p\n' | sort -k3
echo "Then:  ln -sf \"\$PWD/<chosen file>\" $W/brainsegfounder_swinunetr_brats.pt"
echo "Check the channel order in the fold JSON there: expected image list [flair, t1ce, t1, t2]."
