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
# then: bmf audit / index / check --config $BMF_CONFIG   (label-scheme check; see "Checks" below)
```

If `ascli` cannot be installed on Juno, download to your laptop with Aspera Connect and copy:
`rsync -avP UCSD-BMETS-Longitudinal/ <netid>@juno.utdallas.edu:/scratch/<netid>/bmets/UCSD-BMETS-Longitudinal/`.

**Verify the label scheme** on the first case before running anything else:
`python -c "import nibabel as n,numpy as np,glob;f=sorted(glob.glob('$BMF_DATA/*/*seg*.nii.gz'))[0];print(f,np.unique(n.load(f).get_fdata()))"`.
The labels should be {0, 1, 2, 3} (plus 4 if there is a resection cavity). If they differ, edit `labels:` in `configs/base.yaml`.

## 2. Foundation-model weights

```bash
bash scripts/get_weights.sh $BMF_SCRATCH/weights     # login node (needs internet)
```

- **SAM-Med3D-turbo** (`sam_med3d_turbo.pth`, from Hugging Face `blueyo0/SAM-Med3D`) is downloaded automatically, and the SAM-Med3D code is cloned to `weights/SAM-Med3D`. That repository has no pip installer, so `00_env.sh` exports `SAMMED3D_REPO`.
- **BrainSegFounder:** the script downloads the authors' Google Drive folder and lists every `.pt`/`.pth` file. Pick the **BraTS fine-tuned 4-channel SwinUNETR** checkpoint (any single fold) and link it:
  `ln -sf <file> $BMF_SCRATCH/weights/brainsegfounder_swinunetr_brats.pt`.
  In the fold JSON from the same folder, the `image` list should read *flair, t1ce, t1, t2*: the pipeline feeds channels in that order.
- **Verify** with `bmf check --config $BMF_CONFIG`. Each checkpoint must report `OK matched 1.0 -> 768-d` (BrainSegFounder) or `-> 384-d` (SAM-Med3D). A wrong file is refused (it matches less than 90% of the encoder tensors), never silently loaded.

## 3. Run

Run the smoke test first (12 patients, every CPU stage, about 1 h). It writes only to `cache_smoke/` and `reports_smoke/`, and its log ends with `SMOKE OK`:

```bash
sbatch juno/00b_smoke.sbatch            # BMF_SMOKE_N=20 sbatch ... for a larger subset
```

Then the full run:

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

## Checks (`bmf check`)

`bmf check` writes `reports/check.json` and logs one line per label and per checkpoint. It runs automatically in `01b_index.sbatch` and the smoke test.

**Label scheme.** The pipeline expects BraTS-METS 2025: 1 NETC, 2 SNFH, 3 ET, 4 RC. The check samples 20 segmented studies (`check.n_studies`). It does not trust the label values alone; it also tests each label against MRI contrast physics:

| Rule | Reason |
|---|---|
| values are a subset of {0, 1, 2, 3, 4} | no foreign scheme |
| ET (3) has the highest median t1c z-score of the tumour labels | gadolinium leaks through the disrupted blood–brain barrier, so enhancing tumour is brightest after contrast |
| SNFH (2) is FLAIR-hyperintense (median t2f z > 0) | vasogenic oedema is excess extracellular water |
| label 4 must not be the most enhancing label | catches the old BraTS 2017–2021 scheme (1 NCR, 2 ED, 4 ET) |

If it fails, the log names the rule. Fix it with `labels:` in the route config (for example `labels: {NETC: 1, SNFH: 2, ET: 4, RC: 5}`) or remap the files, then re-run `bmf check`.
