"""Generate the Colab notebooks (run: python colab/_build.py). Every notebook mounts Google Drive and calls `bmf`."""
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).parent
SETUP = r'''# --- setup (run every session) ---
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ['BMF_DRIVE'] = '/content/drive/MyDrive/bmets'                       # change if you use another folder
os.environ['BMF_DATA'] = os.environ['BMF_DRIVE'] + '/UCSD-BMETS-Longitudinal'   # Aspera download target (uploaded to Drive)
os.makedirs(os.environ['BMF_DRIVE'], exist_ok=True)
REPO = '/content/claude_projects'
if not os.path.exists(REPO):
    !git clone -q https://github.com/rittik047/claude_projects.git $REPO
%cd /content/claude_projects/bmets-forecast
!pip -q install -e ".[dl,extra]"
CFG = 'configs/colab.yaml'
!nvidia-smi --query-gpu=name,memory.total --format=csv'''


def nb(title, intro, cells):
    n = nbf.v4.new_notebook()
    n.metadata.update({"accelerator": "GPU", "colab": {"gpuType": "A100"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}})
    n.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}"), nbf.v4.new_code_cell(SETUP)]
    for kind, src in cells:
        n.cells.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    return n


NOTEBOOKS = {
    "01_setup_and_audit.ipynb": nb("01 · Setup, data check and cohort audit",
        "Upload the Aspera download to `MyDrive/bmets/UCSD-BMETS-Longitudinal/` and the four TSVs to `MyDrive/bmets/clinical/`. "
        "You need about 350 GB of Drive space (raw data plus the preprocessed cache). Otherwise run preprocessing on Juno and copy `cache/pp` to Drive.",
        [("code", "!pytest -q -p no:warnings tests  # synthetic tests: must pass"),
         ("code", "!bmf audit --config $CFG\n!bmf index --config $CFG\n!bmf splits --config $CFG"),
         ("code", "import json; print(json.dumps(json.load(open(os.environ['BMF_DRIVE'] + '/reports/audit.json')), indent=1)[:2000])"),
         ("md", "**Check the label values** of the first segmentation before continuing. Expected: {0,1,2,3}, plus 4 for a resection cavity."),
         ("code", "import glob, nibabel as nib, numpy as np\nf = sorted(glob.glob(os.environ['BMF_DATA'] + '/*/*seg*.nii.gz'))[0]\nprint(f, np.unique(nib.load(f).get_fdata()))")]),
    "02_preprocess.ipynb": nb("02 · Preprocessing (resumable)",
        "This stage is CPU-bound. Each session processes one shard and skips studies already done, so re-run it until all 8 shards finish. A full run takes about 6–10 h in total.",
        [("code", "SHARD = 0   # run 0..7, one per session if needed\n!bmf preprocess --config $CFG --shard {SHARD}/8"),
         ("code", "import glob; print(len(glob.glob(os.environ['BMF_DRIVE'] + '/cache/pp/*/seg.nii.gz')), 'of 1065 studies preprocessed')")]),
    "03_segmentation.ipynb": nb("03 · nnU-Net segmentation (one fold per session)",
        "S0 is the 4-channel nnU-Net (dataset 501). S2 is prior-conditioned (dataset 502). Checkpoints are written to Drive, so an interrupted fold resumes with `--c`.",
        [("code", "!bmf seg-export --config $CFG\nos.environ['nnUNet_raw'] = os.environ['BMF_DRIVE'] + '/cache/nnunet_raw'\n"
                  "os.environ['nnUNet_preprocessed'] = os.environ['BMF_DRIVE'] + '/cache/nnunet_preprocessed'\nos.environ['nnUNet_results'] = os.environ['BMF_DRIVE'] + '/cache/nnunet_results'"),
         ("code", "DATASET, FOLD = 501, 0\nimport glob, shutil\nname = os.path.basename(glob.glob(os.environ['nnUNet_raw'] + f'/Dataset{DATASET}_*')[0])\n"
                  "if not os.path.exists(os.environ['nnUNet_preprocessed'] + '/' + name):\n    !nnUNetv2_plan_and_preprocess -d $DATASET -c 3d_fullres --verify_dataset_integrity\n"
                  "shutil.copy(os.environ['nnUNet_raw'] + f'/{name}/splits_final.json', os.environ['nnUNet_preprocessed'] + f'/{name}/splits_final.json')\n"
                  "!nnUNetv2_train $DATASET 3d_fullres $FOLD -tr nnUNetTrainer --npz --c"),
         ("md", "After all 5 folds of a dataset: gather the out-of-fold predictions and score them lesion-wise."),
         ("code", "M = 'S0' if DATASET == 501 else 'S2'\nout = os.environ['BMF_DRIVE'] + f'/cache/seg_pred/{M}'; os.makedirs(out, exist_ok=True)\n"
                  "!cp $nnUNet_results/{name}/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_*/validation/*.nii.gz {out}/\n!bmf seg-eval --config $CFG --pred-dir {out} --method {M}")]),
    "04_embed_train_eval.ipynb": nb("04 · Tracking, labels, foundation-model embeddings, models",
        "Put the foundation-model weights in `MyDrive/bmets/weights/`, using the file names in `configs/colab.yaml`.",
        [("code", "!bmf track --config $CFG\n!bmf labels --config $CFG\n!bmf features --config $CFG"),
         ("code", "for enc in ['brainsegfounder', 'samed3d', 'random3d']:\n    !bmf embed --config $CFG --encoder {enc} --device cuda"),
         ("code", "for task in ['T2a', 'T2b']:\n    for m in ['P0', 'P1', 'P2', 'P3', 'P5']:\n        !bmf train --config $CFG --model {m} --task {task} --device cuda\n    !bmf explain --config $CFG --task {task}\n"
                  "!bmf train --config $CFG --model P1 --task T3\n!bmf train --config $CFG --model O1 --task T4\n!bmf evaluate --config $CFG")]),
    "05_figures_report.ipynb": nb("05 · Figures and aggregate report",
        "This notebook writes `reports/results.json`, `reports/tables/*.csv` and `reports/figures/*.png|pdf` to Drive. These files are aggregate only and safe to commit.",
        [("code", "!bmf figures --config $CFG\n!bmf report --config $CFG\nprint(open(os.environ['BMF_DRIVE'] + '/reports/figures_log.txt').read())"),
         ("code", "from IPython.display import Image, display\nimport glob\nfor p in sorted(glob.glob(os.environ['BMF_DRIVE'] + '/reports/figures/*.png'))[:12]:\n    display(Image(p, width=520))")]),
}

if __name__ == "__main__":
    for name, n in NOTEBOOKS.items():
        nbf.write(n, str(HERE / name))
        print("wrote", name)
