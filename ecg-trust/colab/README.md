# Running ecg-trust on Google Colab (GPU)

[`ecg_trust_colab.ipynb`](ecg_trust_colab.ipynb) runs the whole project on a Colab GPU:
- data, splits and caches;
- the feature baselines, the 3-seed ResNet ensemble and the fusion;
- calibration, conformal abstention and the freeze;
- the single-use internal and external tests;
- the 64 figures, the model release and the PDF guide.

It reads the data from Google Drive and writes every result back to Drive as it goes.

Open it in Colab:
`https://colab.research.google.com/github/Rittik047/Claude_Projects/blob/claude/beautiful-carson-3col06/ecg-trust/colab/ecg_trust_colab.ipynb`
(swap the branch for `main` once it is merged).

## What you need

| | |
|---|---|
| Google Drive space | about 10 GB: 6.3 GB of data zips, 3.0 GB of caches, about 0.2 GB per run of outputs |
| Colab runtime | **G4** GPU if your plan offers it. T4, L4 and A100 also work: the code picks up any CUDA device. |
| Time, first session | about 1–1.5 h, including the one-off data download |
| Time, later sessions | about 25–40 min, because the caches are read from Drive |

The times are estimates. GPU training itself is about 5–10 min. Most of the rest is the 2,000-resample patient bootstraps and the noise-robustness sweep, which run on CPU.

## Step 1: put the data on Drive

Create this folder layout in **My Drive**. The notebook fills `cache/` and `outputs/` itself.

```
My Drive/
└── ecg-trust/
    ├── data/
    │   ├── ptbxl.zip         PTB-XL 1.0.3
    │   ├── ptbxl_plus.zip    PTB-XL+ 1.0.1
    │   └── external.zip      ecg-arrhythmia 1.0.0  (only needed for external validation)
    ├── cache/                created by the notebook: 100 Hz signal caches
    └── outputs/              created by the notebook: one folder per run
```

Choose one of three ways to fill `data/`.

### Option A: let the notebook download it (easiest)

Leave `DOWNLOAD_IF_MISSING = True` in the first cell. On the first run the notebook does this for each dataset:
1. Downloads only the files the project needs from the PhysioNet open-data mirror on AWS (`physionet-open.s3.amazonaws.com`), checking each file's SHA-256:
   - PTB-XL: 612 MB;
   - PTB-XL+: 222 MB;
   - ecg-arrhythmia: 5.4 GB.
2. Zips it.
3. Saves it to `My Drive/ecg-trust/data/` as `ptbxl.zip`, `ptbxl_plus.zip` and `external.zip`.

Later runs use those zips and skip the download.

### Option B: download the ZIP files from PhysioNet yourself

| Dataset | Page ("Download the ZIP file" at the bottom) | Save in Drive as |
|---|---|---|
| PTB-XL 1.0.3 | https://physionet.org/content/ptb-xl/1.0.3/ | `ecg-trust/data/ptbxl.zip` |
| PTB-XL+ 1.0.1 | https://physionet.org/content/ptb-xl-plus/1.0.1/ | `ecg-trust/data/ptbxl_plus.zip` |
| ecg-arrhythmia 1.0.0 | https://physionet.org/content/ecg-arrhythmia/1.0.0/ | `ecg-trust/data/external.zip` |

- **Don't unzip.** Upload the zips as they are and rename them as above.
- **Nesting is fine.** PhysioNet zips put everything inside a top folder; the notebook finds the dataset inside it. It looks for these files:
  - `ptbxl_database.csv`;
  - `features/12sl_features.csv`;
  - `WFDBRecords/`.
- **These zips are larger.** They also contain the 500 Hz PTB-XL signals, which the project doesn't use.

### Option C: copy from a machine that already has the data

If you ran `ecgtrust download` locally, zip each folder in `ecg-trust/data/` (`ptbxl`, `ptbxl_plus`, `external`) and upload the zips under the names above.

Unzipped folders in `ecg-trust/data/<name>/` also work. They are much slower, because Drive handles tens of thousands of small files badly.

## Step 2: open the notebook and choose the GPU

1. Open the link at the top of this page.
   - If the repository is private, use **File → Open notebook → GitHub**, tick *Include private repos*, and pick `ecg-trust/colab/ecg_trust_colab.ipynb`. Or upload the `.ipynb` file directly.
2. If the repository is private, add a GitHub token as a Colab secret:
   - **🔑 Secrets → Add new secret**, with the name `GITHUB_TOKEN`;
   - a fine-grained token with read access to the repository is enough;
   - switch on *Notebook access*.
3. Choose **Runtime → Change runtime type → G4** (or another GPU) and save.

## Step 3: run

Check the settings cell, then choose **Runtime → Run all**. Authorise Drive when asked.

| Setting | Default | Meaning |
|---|---|---|
| `DRIVE_ROOT` | `/content/drive/MyDrive/ecg-trust` | the folder from Step 1 |
| `BRANCH` | `claude/beautiful-carson-3col06` | code version to run |
| `DOWNLOAD_IF_MISSING` | `True` | Option A |
| `SAVE_CACHE_TO_DRIVE` | `True` | keep the caches (3.0 GB) on Drive so later runs skip unzipping and preprocessing |
| `RUN_EXTERNAL` | `True` | external validation; `False` skips the 5.4 GB dataset |
| `BUILD_GUIDE` | `True` | render the PDF guide (installs headless Chromium, about 2 min) |
| `N_BOOT` | `2000` | bootstrap resamples; lower it (e.g. 200) for a quick trial run |

The notebook runs these stages in order:
1. Drive and GPU check.
2. Code.
3. Configuration.
4. Data.
5. `prepare` (splits and caches), then `audit`.
6. Baselines, the 3-seed ResNet on the GPU, then fusion.
7. Validation evaluation, then `freeze`.
8. Single-use test runs: internal ResNet, internal fusion and external.
9. Figures, release and guide.
10. Inline preview of the key figures.

## Step 4: find the results on Drive

Each run writes to `My Drive/ecg-trust/outputs/<YYYYMMDD-HHMM>/`:

```
outputs/20261102-1530/
├── reports/
│   ├── results.md                   headline table, calibration, abstention, subgroups, comparator
│   ├── data_audit.md
│   ├── test/ test_fusion/ external/ val_select/ ...   metrics.json, summary.md, CSVs, plots
│   ├── figures/                     64 PNGs in 8 folders + README.md index
│   │   ├── 01_data/  02_training/  03_performance/  04_calibration/
│   │   └── 05_abstention/  06_robustness/  07_explainability/  08_comparator/
│   ├── guide/ecg_trust_guide.pdf    illustrated guide
│   └── test_runs.log                the single-use test log
├── artifacts/
│   ├── models/                      seed checkpoints, baselines, fusion.json, training logs
│   ├── preds/  calibration/
│   └── freeze.json  splits.parquet
└── release/                         servable bundle (about 18 MB) for the API and Streamlit app
```

Results are copied after every stage. If Colab disconnects, everything finished so far is already on Drive.

## Good to know

- **GPU vs the repository's numbers.**
  - Splits and caches are bit-identical to the original CPU run.
  - Training is seeded, but some cuDNN kernels are non-deterministic, so a GPU run can differ slightly from the frozen CPU results in `reports/`. Expect differences in the third decimal of AUROC.
  - Report the numbers from the run you actually did. The repository's `freeze-v1` results remain the reference.
  - To force CPU, add `os.environ["ECGTRUST_DEVICE"] = "cpu"` to the settings cell.
- **Freeze and test lock.** Each notebook run makes its own freeze. Each test set can be evaluated once per model per run. A second call in the same session is refused; pass `--rerun-reason "..."` to repeat it on purpose, and the reason is logged.
- **Disconnects and limits.** Free-tier sessions can time out. Re-run from the top: the data and caches come back from Drive in a few minutes. If you are short of time, set `RUN_EXTERNAL = False` and `N_BOOT = 200` for a trial.
- **Drive quota.** If Drive is nearly full, set `SAVE_CACHE_TO_DRIVE = False`. That saves 3.0 GB, at the cost of rebuilding the caches each session (about 5 min).
- **Not a medical device.** This is research software and must not be used for diagnosis.
