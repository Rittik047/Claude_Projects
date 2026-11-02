# Route 1 — UTD Juno HPC (SLURM)

The full pipeline runs here: download, preprocessing, nnU-Net, embeddings, models, figures. Only aggregate files (`reports/`) come back to git.

## 0. One-time setup

```bash
git clone https://github.com/rittik047/claude_projects.git ~/Claude_Projects
cd ~/Claude_Projects/bmets-forecast
nano juno/00_env.sh          # set JUNO_PARTITION_CPU / JUNO_PARTITION_GPU / BMF_SCRATCH (check: sinfo -s)
source juno/00_env.sh        # creates the conda env "bmf" on first run (~15 min)
pytest -q                    # synthetic tests must pass before touching data
```

## 1. Data (Aspera)

```bash
bash juno/01_aspera_pull.sh "<Faspex package link from the TCIA page>"
# put the 4 TSVs (clinical, clinical dictionary, MRI metadata, metadata dictionary) in $BMF_SCRATCH/clinical/
```

If `ascli` cannot be installed on Juno, download to your laptop with Aspera Connect and copy:
`rsync -avP UCSD-BMETS-Longitudinal/ <netid>@juno.utdallas.edu:/scratch/<netid>/bmets/UCSD-BMETS-Longitudinal/`.

**Verify the label scheme** on the first case before running anything else:
`python -c "import nibabel as n,numpy as np,glob;f=sorted(glob.glob('$BMF_DATA/*/*seg*.nii.gz'))[0];print(f,np.unique(n.load(f).get_fdata()))"`.
The labels should be {0, 1, 2, 3} (plus 4 if there is a resection cavity). If they differ, edit `labels:` in `configs/base.yaml`.

## 2. Foundation-model weights

Download into `$BMF_SCRATCH/weights/`, using the file names in `configs/juno.yaml`:
- BrainSegFounder (SwinUNETR, BraTS fine-tuned checkpoint): https://github.com/lab-smile/BrainSegFounder
- SAM-Med3D (turbo checkpoint): https://github.com/uni-medical/SAM-Med3D. Also install the repository: `pip install git+https://github.com/uni-medical/SAM-Med3D`.

## 3. Run

```bash
bash juno/run_all.sh          # submits 01b (index, once) -> 02 -> 03a -> 03 (10 GPU array jobs) -> 03b, 04 -> 05 -> 06 with dependencies
squeue -u $USER               # monitor; logs in $BMF_SCRATCH/logs
```

| Script | Resources | Approximate time |
|---|---|---|
| 02_preprocess (32-task array) | CPU | 2–4 h |
| 03_seg_train (10 tasks: S0/S2 × 5 folds) | 1×A100 each | 12–24 h each |
| 04_embed | 1×A100 | 3–6 h |
| 05_train_eval | 1×A100 | 2–4 h |
| 06_figures_report | CPU | 15 min |

## 4. External validation (once, at the end)

1. Download a cohort, for example PROTEAS (Zenodo) or Yale-Brain-Mets-Longitudinal (TCIA).
2. Write `map.csv` with columns `study_id,seq,path`. `study_id` must be `CASEID_YY`, and `seq` is one of `t1c|t1n|t2f|t2|seg`.
3. Arrange the files: `bmf to-layout --map map.csv --out $BMF_SCRATCH/ext_proteas`.
4. Copy `configs/juno.yaml` to `configs/external_proteas.yaml` and change `paths`. Keep `lock_dir` unchanged.
5. Run `sbatch -p $JUNO_PARTITION_GPU juno/07_external.sbatch`. The `external` stage refuses to run a second time on the same cohort.

## 5. Bring results back

```bash
cd ~/Claude_Projects && git checkout -b results-$(date +%F)
git add bmets-forecast/reports/results.json bmets-forecast/reports/tables bmets-forecast/reports/figures
git commit -m "results: development run" && git push -u origin HEAD
```
