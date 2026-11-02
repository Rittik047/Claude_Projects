# bmets-forecast — design spec

**Date:** 2026-11-02 · **Status:** awaiting spec review
**Author:** Rittik Patra (solo first author) · **Target:** npj Digital Medicine (stretch: Nature Communications)

## 1. Intent

**Working title:** *Forecasting intracranial disease evolution in brain metastases from longitudinal multiparametric MRI with 3D foundation models.*

**Purpose:** a publishable, portfolio-grade study built on a dataset released in October 2026 (UCSD-BMETS-Longitudinal, TCIA, CC BY 4.0).

**Scope (user choice "d"):** three linked tasks.
- **T1:** automatic segmentation and lesion tracking.
- **T2 (headline):** next-scan intracranial progression and new metastases.
- **T3:** lesion-level volume trajectories.
- **Proposed addition, T4:** overall survival. The data contain 153 deaths; see §11, Q1.

**Non-goals:** treatment-effect (causal) claims, or a clinical deployment claim.

| Stated by user | Assumed (correct me) |
|---|---|
| Dataset = UCSD-BMETS-Longitudinal; all three tasks; T2 headline | Segmentation labels follow BraTS-METS 2025 (1 NETC, 2 SNFH, 3 ET, 4 resection cavity) — verified on the first downloaded case |
| External cohorts: use whichever are usable | The user can register for BraTS-METS on Synapse and download PROTEAS (Zenodo) and Yale (TCIA) to Juno |
| Data pulled with Aspera; separate Juno and Colab routes; Colab reads from Google Drive | Juno has SLURM, A100s, and ~2 TB scratch; Colab Pro A100 with Drive holding ≥200 GB |
| Claude cloud credit expires Nov 4 | Work after that continues locally or with regular Claude Code |

## 2. Cohort audit (run 2026-11-02 on the four public TSVs)

| Item | Value |
|---|---|
| Patients / MRI time points | 300 / 1,065 (17 time points are normal scans taken before diagnosis) |
| Post-diagnosis MRIs per patient | 4: 208 · 3: 32 · 2: 60 |
| Post-diagnosis intervals (t → t+1) | **748**; median interval 0.2 years (IQR 0.1–0.2), range 0–2.5; 60 intervals recorded as 0.0 years (ages are given to 0.1-year precision) |
| Radiation within an interval | 1→2: 127/300 · 2→3: 94/240 · 3→4: 26/208 → **247 intervals with radiation, 501 without** |
| Resection before first MRI | 59 |
| Deaths | **153 (51%)**; median time from first MRI to death 0.7 years; median time from last MRI to death 0.3 years |
| Primary tumour | lung 113, breast 78, other 34, melanoma 27, gastrointestinal 22, renal 17, prostate 7, gastrointestinal + lung 1 |
| Sex / age at first MRI | F 167 / M 133; age 61.6 ± 12.9 years |
| Scanners | GE 1,037, Siemens 22, Canon 6; **3T 635 / 1.5T 430**; 8 models |
| Sequence availability | t1c, t1n, t2f, t2w 100%; t1postFSE 99%; ADC 99%; DWI 98%; SWAN 99.5%; ASL 87% |

**Data notes:**
- Folder `BMETSxxxx_01` is the normal pre-diagnosis scan for the 17 patients flagged `MRI without metastases = Yes`. Time point indices are therefore remapped to the post-diagnosis order using the clinical table.
- Timing is age to 0.1-year precision, so intervals are coarse (about ±18 days).
- Treatment information is radiation (yes/no) per interval plus resection before baseline. There is no detail on radiation type, dose or systemic therapy.
- Leakage risk: BraTS-METS 2025 contains UCSD cases, and the same author appears on both datasets. Every BraTS-METS case from UCSD is excluded from any external or pretraining use.

## 3. Tasks, labels and metrics

All labels are derived from the segmentations. Lesions are tracked across time points (§4).

| Task | Unit (n) | Label / target | Primary metric |
|---|---|---|---|
| T1 Segmentation and tracking | Time point (1,048 with metastases) | Voxel labels; lesion correspondence across time points | Lesion-wise Dice and F1 (BraTS-METS protocol), sensitivity by size (<3, 3–10, >10 mm), tracking accuracy |
| **T2a Intracranial progression** | Interval (≤748) | Volumetric RANO-BM progression at t+1: sum of target enhancing-lesion volumes ≥ +73% vs. nadir, **or** any new lesion ≥ 0.05 cc | AUROC, AUPRC, calibration, net benefit |
| **T2b New metastases (distant brain failure)** | Interval | ≥1 new enhancing lesion at t+1 not matched to any lesion at t | Same |
| T3 Lesion trajectory | Lesion × interval | log-ratio of enhancing volume at t+1 vs. t; response class (complete response, partial response ≤ −65%, stable, progression ≥ +73%, volumetric thresholds) | MAE, Spearman ρ, ordinal C-index, interval coverage |
| T4 Overall survival (proposed) | Patient (300; 153 events) | Time from first post-diagnosis MRI to death; censored at the last MRI | Harrell's and Uno's C-index, time-dependent AUC at 6 and 12 months, integrated Brier score |

**Inputs for T2 and T3** at interval t → t+1:
- MRI at t, plus all earlier scans;
- segmentation and lesion features at t;
- radiation in the interval (known at decision time, because it is planned);
- primary tumour, age, sex, resection;
- interval length.

Sensitivity analyses: drop the radiation variable; restrict to intervals of 0.1 year or longer.

**Treatment caveat:** radiation is confounded by indication. It is a predictor, not an intervention. Results are reported stratified by radiation status.

## 4. Pipeline and models

**Preprocessing:** the images are already skull-stripped. Steps:
1. resample to 1 mm isotropic;
2. co-register sequences (t1c is the reference);
3. register longitudinally to the patient's baseline (ANTs rigid + affine);
4. N4 bias correction;
5. z-score inside the brain mask.

Optional sequences (ADC, SWAN, ASL) are added as extra channels, with a missingness mask.

**Lesion tracking:** connected components of ET + NETC → match lesions between registered time points by Hungarian assignment on overlap and centroid distance (threshold 10 mm) → assign track IDs and classify each as new, resolved or merged.

| ID | Model | Task | Notes |
|---|---|---|---|
| S0 | nnU-Net v2 (3d_fullres) | T1 | Strong baseline, 5-fold cross-validation |
| S1 | MedNeXt or a foundation-model encoder (BrainSegFounder / SAM-Med3D) fine-tuned | T1 | Foundation-model comparison |
| S2 | S0 conditioned on the prior time point (prior scan and mask as extra channels) | T1 | Longitudinal-aware; targets new and small lesions |
| P0 | Clinical baseline: lesion count, total ET volume, primary tumour, radiation, age, interval (in the spirit of brain-metastasis velocity) → logistic regression | T2 | Comparator |
| P1 | Radiomics (PyRadiomics; lesion and whole-brain, plus changes over time) + clinical → LightGBM | T2, T3 | SHAP |
| P2 | Frozen 3D foundation-model embeddings (whole brain + per-lesion crops) + clinical → linear / logistic probe | T2, T3 | Foundation models compared: BrainSegFounder, SAM-Med3D encoder, MedSAM2 / BiomedParse-3D (whichever load); embeddings cached |
| P3 | Temporal transformer over the sequence of per-time-point foundation-model embeddings, with clinical and interval tokens | T2, T3 | The longitudinal model |
| P4 | P3 with LoRA fine-tuning of the encoder | T2 | If compute allows |
| P5 | Stacked late fusion of P0 + P1 + P3 | T2 | Tests whether they are complementary |
| O1 | Cox / DeepSurv on P0, P1 and P2 features, plus a temporal survival transformer | T4 | Benchmark against a GPA-like clinical model (no Karnofsky score or extracranial status in the data, so stated as a limitation) |

**Splits:** 5-fold patient-level cross-validation, stratified by primary tumour × number of time points. All intervals and lesions of a patient stay in one fold, and the fold assignment is saved in `splits.json`. Tuning uses inner folds only. External sets are run once, against frozen models.

## 5. External validation (usable cohorts)

| Cohort | Access | What it supports | Use |
|---|---|---|---|
| **BraTS-METS 2025** (1,778 cases, multi-institutional, pre- and post-treatment, NETC/SNFH/ET/RC labels) | Synapse registration | T1 | External segmentation test on **non-UCSD cases only** |
| **PROTEAS** (Sci Data 2025; 40 patients, 744 MRIs, segmentations in BraTS space, radiotherapy plans) | Zenodo | T3, T2 (small) | Longitudinal external test |
| **Yale-Brain-Mets-Longitudinal** (TCIA, DOI 10.7937/3YAT-E768; 1,430 patients, 11,884 studies; clinical workbook; **no segmentations**) | Open (TCIA) | T2, T4 (if the workbook has death dates), via automated segmentation from S2 | Large automated-pipeline external test; reported as "automated labels", with a manual-check subset of 50 interval pairs |
| UCSF-BMSR, Pretreat-MetsToBrain-Masks | TCIA / UCSF | T1 only | Optional; check overlap with BraTS first |

## 6. Explainability and figures (≥35, generated on Juno or Colab)

- **Data:**
  - cohort flow diagram;
  - swimmer plot (time points, radiation, death per patient);
  - interval histogram;
  - scanner and field-strength mosaic;
  - sequence-availability heatmap;
  - primary-tumour × outcome bars;
  - example 4-sequence montage with overlay;
  - lesion-count and volume distributions.
- **T1:**
  - Dice/F1 vs. lesion size;
  - false-negative gallery;
  - tracking Sankey diagram (new / stable / resolved);
  - overlays for each method;
  - performance by field strength.
- **T2:**
  - ROC/PR curves for all models;
  - forest plot of ΔAUROC vs. P0;
  - calibration and decision curves;
  - SHAP beeswarm, bar, dependence and interaction plots;
  - Grad-CAM / integrated-gradient maps on t1c and FLAIR;
  - attention over time points;
  - UMAP of foundation-model embeddings (by primary tumour, outcome, scanner);
  - sequence-ablation bars;
  - subgroup forest plot (primary tumour, sex, radiation, field strength).
- **T3:**
  - per-lesion trajectory spaghetti plots by primary tumour and radiation;
  - trajectory clusters (k-shape or GMM) with centroids;
  - predicted vs. observed scatter with intervals;
  - waterfall plot of best response.
- **T4:**
  - Kaplan–Meier curves by risk tertile and by primary tumour;
  - time-dependent AUC;
  - SHAP for survival.

## 7. Repository layout

```
bmets-forecast/
  pyproject.toml  README.md  MODEL_CARD.md  .gitignore
  configs/                 # base.yaml + juno.yaml + colab.yaml (paths, compute)
  src/bmf/
    data/        clinical.py (TSV parse + timepoint remap), mri_index.py, nifti_io.py, splits.py
    preprocess/  resample, register (ANTs), n4, normalize
    lesions/     components, tracking (Hungarian), rano_bm labels (T2/T3)
    seg/         nnunet adapters (dataset export, train/predict wrappers), prior-conditioned S2
    fm/          encoder registry + embedding extraction + LoRA
    models/      P0–P5, O1
    eval/        metrics (lesion-wise BraTS, AUROC/bootstrap/DeLong, C-index, calibration, DCA)
    explain/     SHAP, Grad-CAM/IG, attention, ablation
    viz/         one function per figure, shared style
    synth.py     synthetic 4D brain + lesions generator for tests
    cli.py       `bmf <stage>`
  juno/                    # ROUTE 1: Aspera pull, conda env, SLURM scripts per stage, README_JUNO.md
  colab/                   # ROUTE 2: Drive-mounted notebooks per stage, README_COLAB.md
  tests/                   # pytest on synthetic data
  docs/                    # spec, plan, analysis plan, guide PDF, manuscript skeleton, TRIPOD+AI
  reports/                 # aggregate results only
```

**CLI stages (shared by both routes):** `audit → index → preprocess → seg-train/seg-predict → track → labels → features → embed → train → evaluate → explain → figures → report`. Each stage is idempotent and cached, and records the git hash.

## 8. Compute plan

| Stage | Where | Estimate |
|---|---|---|
| Aspera download (163 GB) + external datasets | Juno | Hours |
| Preprocessing and registration (1,065 time points) | Juno CPU array | ~3–5 CPU-min per time point → a few hours on 64 cores |
| nnU-Net S0 (5 folds), S2 | A100 | ~12–24 GPU-h each (`3d_fullres`, 250–500 epochs) |
| Foundation-model fine-tune S1 | A100 | ~10–20 GPU-h |
| Embeddings, P2/P3 | A100 | ~5–10 GPU-h |
| P4 LoRA | A100 | ~10–20 GPU-h |
| Radiomics, gradient boosting, SHAP, bootstrap | CPU | Hours |
| Yale external: preprocessing + S2 inference on ~12k studies | Juno | ~1–2 days CPU+GPU |

**Colab route:** the same stages, chunked so each fits a session (≤12 h on A100), with checkpoints and caches on Google Drive. Recommended for the embedding, probe and figure stages. nnU-Net training is better done on Juno.

**Claude cloud credit (now → Nov 4):** all code, tests, both routes, the guide PDF (figures from the real public TSVs, plus synthetic placeholders for image-based figures, clearly labelled), manuscript skeleton, analysis plan, TRIPOD+AI checklist and model card.

## 9. Statistics and reporting

- Bootstrap confidence intervals (2,000 resamples, clustered by patient).
- Holm correction for the pre-specified T2 contrasts: P1, P2, P3 and P5 each vs. P0.
- T3 and the figure-level explainability analyses are exploratory.
- TRIPOD+AI checklist; CLAIM checklist for imaging.
- `docs/analysis_plan.md` is frozen and its hash committed before the first run on real data.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Coarse timing (0.1-year precision) | Use the interval as a covariate; sensitivity analysis excluding 0.0-year intervals |
| Few positive cases for T2b | Count during the audit stage; fall back to T2a as headline if fewer than 60 events |
| Label noise from tracking | Manual QC on a 50-interval sample; report tracking accuracy |
| Scanner shift (1.5T vs. 3T) | Field strength as a stratum; harmonization ablation |
| Overlap with BraTS-METS | Exclude UCSD cases using BraTS site IDs |
| Foundation-model checkpoint licenses or weight availability | Encoder registry with fallbacks; at minimum nnU-Net encoder features |
| Censoring at last MRI (T4) | State as informative-censoring limitation; landmark analysis |

## 11. Open questions

1. Promote T4 (overall survival, 153 events) to a co-primary endpoint alongside T2? This is recommended, because survival is the most robust label in the data.
2. Confirm the label scheme on the first downloaded segmentation file.
