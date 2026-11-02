# Route 2 — Google Colab Pro (A100) + Google Drive

The same `bmf` stages as on Juno, run from notebooks that read and write `MyDrive/bmets/`.

## Drive layout

```
MyDrive/bmets/
  UCSD-BMETS-Longitudinal/   # 1,065 study folders (Aspera download, uploaded to Drive)
  clinical/                  # the 4 TSVs from the TCIA page
  weights/                   # foundation-model checkpoints (names in configs/colab.yaml)
  cache/  reports/           # created by the notebooks
  cache_smoke/  reports_smoke/   # created by 00_smoke_test (safe to delete)
```

**Storage:** raw data 163 GB plus the preprocessed cache (about 150 GB) needs a Google One plan of at least 2 TB. Without it, a hybrid works best: run `02_preprocess` on Juno, then copy only `cache/pp`, `cache/*.csv` and `cache/splits.json` to Drive (about 150 GB) and start from notebook 03 or 04.

## Order

| Notebook | GPU | Sessions |
|---|---|---|
| 00_smoke_test | no (CPU runtime) | 1, about 30–60 min: 12 patients, every CPU stage; writes only `cache_smoke/` and `reports_smoke/` |
| 01_setup_and_audit | optional | 1 |
| 02_preprocess | no (use a CPU runtime to save compute units) | 1–8 shards, resumable |
| 03_segmentation | A100 | 1 per fold (10 total: S0 and S2 × 5 folds), resumes with `--c` |
| 04_embed_train_eval | A100 | 1–2 |
| 05_figures_report | no | 1 |

Every notebook starts with the same setup cell: mount Drive, clone the repo, `pip install -e .`. Re-run it after every disconnect, because all state lives on Drive.

To regenerate the notebooks after code changes, run `python colab/_build.py`.
