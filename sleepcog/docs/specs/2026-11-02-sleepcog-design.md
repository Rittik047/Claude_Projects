# sleepcog — design spec

**Date:** 2026-11-02 · **Status:** approved design, awaiting spec review
**Author:** Rittik Patra (solo first author) · **Target venue:** npj Digital Medicine (stretch: Nature Communications)

## 1. Intent

**Working title:** *Beyond age: what sleep physiology adds for detecting future cognitive impairment — a multi-site, externally validated comparison of sleep foundation models and interpretable microstructure.*

**Purpose:** a publishable, portfolio-grade study for postdoc and industry applications, built on 2026 data with deep learning and foundation models.

**Success:** a reproducible pipeline; a pre-specified analysis; locked external results; 30+ explanatory figures; a manuscript draft and an illustrated guide PDF.

**Not a goal:** winning the 2026 Challenge (it closed 24 Aug 2026), or making a clinical diagnostic claim.

### What the user stated vs. what I assumed

| Stated by user | Assumed (correct me) |
|---|---|
| Project = PhysioNet 2026 Challenge (PSG → cognitive impairment, CI) | The Challenge training data stays usable for research after the Challenge closes (license to be checked on Kaggle) |
| Deep learning plus foundation models, journal-grade | SleepFM checkpoints are usable for academic work (reported license: CC BY-NC 4.0; check the repo's LICENSE) |
| Framing "beyond age"; external validation on NSRR MrOS and SOF | NSRR access is granted within about 2–4 weeks |
| Claude handles code only; runs happen on Juno or Colab A100 | Juno has SLURM and A100s, and the user can install conda environments there |
| Claude cloud credit expires Nov 4 | Work after Nov 4 continues locally or with regular Claude Code |

## 2. Context and prior art (positioning)

- **SleepFM** (Thapa et al., *Nat Med*, Jan 2026): a foundation model trained on 585k hours of PSG. Reports a dementia C-index of 0.85, but in a Stanford-dominant cohort. The paper says little about how much of the signal comes from age, and gives little physiological interpretation.
- **Sleep-EEG brain-age index** (pooled cohort of 7,071 people, 2025 preprint): a higher brain-age index carries about 39% higher dementia risk per 10 years.
- **2026 Challenge entries** (CinC 2026): hidden-set age-conditioned AUROC about 0.74. One team saw cross-validation AUROC of 0.82 fall to 0.47 on the hidden set, which points to site shift and overfitting.
- **Gap we fill:** an independent, *age-controlled*, multi-site and externally validated head-to-head between foundation models and interpretable physiology. Plus *where* the signal lives (modality, stage, time of night), *what* the foundation-model embedding encodes, and calibration and fairness.

## 3. Data

### 3.1 Primary: PhysioNet Challenge 2026 training data (Human Sleep Project subset)

- **Layout:**
  - `demographics.csv`: BDSPPatientID, SiteID, SessionID, BidsFolder, CreationTime, Age, Sex, Race, Ethnicity, BMI, Cognitive_Impairment, Time_to_Event, Last_Known_Visit_Date, Time_to_Last_Visit;
  - `physiological_data/<site>/<pid>_ses-<s>.edf`;
  - `algorithmic_annotations/<site>/<pid>_ses-<s>_caisr_annotations.edf` (1 Hz `stage_caisr` with W=5, R=4, N1=3, N2=2, N3=1, invalid ≥9; plus `resp_caisr`, `arousal_caisr`, `limb_caisr`, `caisr_prob_*`);
  - `human_annotations/...expert_annotations.edf` (training data only).
- **Label:** a CI ICD code first recorded between 1 and 6 years after the PSG, with at least 2 codes at least 7 days apart. The Challenge's own `create_labels.py` defines it.
- **Channels:** harmonized with the Challenge `channel_table.csv` into categories EEG (F3/F4/C3/C4/O1/O2-M), EOG, chin EMG, leg EMG, ECG, respiratory (abdomen, chest, nasal pressure, airflow, SpO2).
- **Rule:** we use only **algorithmic** annotations as model inputs, so the models are deployable. Human annotations serve only for quality control and agreement checks.

### 3.2 External: NSRR MrOS (men, ≥65) and SOF (women, ≥65)

- These are community cohorts with PSG and later cognitive outcomes (adjudicated dementia, 3MS/MMSE decline). They are used for **one locked test run**.
- The outcome definition differs (adjudicated diagnosis vs. ICD codes). We pre-specify the mapping to "CI within 1–6 years" and report it as a transportability test, not a replication.

### 3.3 Data governance (hard rule)

- Row-level or signal-level data never enters the Claude cloud session or any online LLM service (PhysioNet 2025 LLM policy and the DUA terms).
- Everything that touches data runs on Juno or Colab.
- Only aggregate artifacts come back to the repo: metrics, confidence intervals and summary tables, with cell counts below 10 suppressed.
- `.gitignore` blocks `data/`, `cache/`, `*.edf`, `*.npz` and per-subject outputs.

## 4. Research questions and analyses (pre-specified)

| RQ | Question | Primary metric | Test |
|---|---|---|---|
| RQ1 | Does PSG add signal beyond covariates (age, sex, BMI, site, AHI)? | Challenge age-conditioned AUROC (age gap 2 years); ΔAUROC, ΔAUPRC vs. covariate-only model | Paired bootstrap (2,000) on out-of-fold predictions; likelihood-ratio test in nested logistic models; categorical net reclassification (NRI) |
| RQ2 | Which representation is best? | Same metrics, all models on identical folds | Holm-corrected paired bootstraps |
| RQ3 | Does it generalize? | Leave-one-site-out (LOSO) AUROC per site; external MrOS/SOF AUROC and calibration | Locked models, single external run |
| RQ4 | Where is the signal? | Δ metric under modality ablation and stage/time occlusion; SHAP rankings; embedding probe R² | Bootstrap confidence intervals |
| RQ5 | Can it be trusted? | Calibration slope, calibration intercept, ECE; net benefit; subgroup AUROC gaps (sex, age band, race, site) | Bootstrap confidence intervals; decision curves |

**Secondary analysis:** a time-to-event formulation using Time_to_Event and Time_to_Last_Visit. Discrete-time or Cox model; C-index and time-dependent AUC at 3 and 5 years. This handles censoring, which the binary label ignores.

**Splits:**
- Patient-level 5-fold stratified cross-validation (by label × site), with folds fixed by seed and saved to `splits.json`.
- LOSO across sites.
- All of a patient's sessions stay in the same fold.
- Hyperparameter tuning uses nested inner folds only.

**Leakage guards:** scalers, feature selection and probes are fit inside training folds; the external data is read once, after models are frozen (recorded with a hash in `reports/external_lock.json`).

**Multiplicity:** Holm correction across the RQ2 model contrasts; RQ4 is explicitly exploratory.

**Small-event risk:** an estimated ~80 positive cases if n≈1,090 at 7.5% prevalence; the "large version" of the data may contain more.
- The data audit reports exact event counts and a minimum detectable ΔAUROC.
- If there are fewer than 100 events, the primary contrasts are restricted to 3 pre-registered models: covariate-only (M0), microstructure features with gradient boosting (M1), and SleepFM linear probe (M3).

## 5. Models

| ID | Model | Inputs | Notes |
|---|---|---|---|
| M0 | Covariate-only logistic regression | Age, sex, BMI, site, AHI (automated) | Primary comparator |
| M1 | Microstructure features → gradient boosting (LightGBM) plus elastic-net logistic regression | ~150 features (§6) | SHAP via TreeExplainer |
| M2 | Sleep-EEG brain-age gap | EEG features → age regressor trained on CI-negative training-fold subjects; the gap is the covariate | Replicates the brain-age index idea |
| M3 | SleepFM frozen embeddings → logistic linear probe | Raw PSG resampled to SleepFM's expected input | Patient embedding pooled across 5-min chunks (mean + attention pooling) |
| M4 | SleepFM fine-tuned with LoRA | Same | Class-balanced focal loss; early stopping on inner fold |
| M5 | From-scratch multimodal epoch transformer | Per-modality 1D-CNN epoch encoders → night-level transformer with modality tokens | Same budget as M4 |
| M6 | Late fusion | M0 + M1 + M3 out-of-fold logits → stacked logistic regression | Tests complementarity |

Every model is also refit with covariates appended, so its incremental value can be measured.

## 6. Interpretable microstructure features (M1, M2)

Computed per stage (N2, N3, REM, and NREM pooled) on C3/C4 (with F and O channels where present):
- **Spectral:** relative and absolute band power (δ, θ, α, σ, β), the θ/α "slowing" ratio, 1/f aperiodic exponent (FOOOF).
- **Spindles:** density, amplitude, frequency, duration (YASA).
- **Slow oscillations:** density, amplitude, slope; slow-oscillation–spindle coupling (phase, strength).
- **Sleep architecture:** stage percentages, sleep efficiency, wake after sleep onset (WASO), sleep latency, REM latency, number of transitions, fragmentation index.
- **Respiratory and oxygen:** AHI, oxygen desaturation index (ODI), T90, hypoxic burden (area under the SpO2 baseline).
- **Cardiac:** HRV (RMSSD, SDNN, LF/HF) per stage, heart-rate response to arousals.
- **Movement and arousal:** arousal index, periodic limb movement index.

## 7. Explainability and figures (all generated on Juno; ≥30)

- **Data:**
  - cohort flow diagram (CONSORT-style);
  - site × label composition;
  - age distribution by label;
  - per-site missing-channel heatmap;
  - example 30-second epochs per stage;
  - hypnogram examples for CI and non-CI patients;
  - feature-correlation clustermap;
  - UMAP/t-SNE of SleepFM embeddings colored by site, age and label (to show site effects).
- **Performance:**
  - ROC and PR curves for all models;
  - forest plot of ΔAUROC vs. M0;
  - age-conditioned AUROC by age band;
  - LOSO heatmap;
  - external ROC curves;
  - label-efficiency curves;
  - bootstrap distributions.
- **Calibration and clinical value:** reliability diagrams, calibration slope forest plot, decision curves, risk-score distributions.
- **Explainability:**
  - SHAP beeswarm, bar and dependence plots (top 6), and an interaction heatmap;
  - SHAP by site;
  - modality-ablation bars;
  - stage-occlusion and time-of-night occlusion curves;
  - integrated-gradient saliency over an example night (M4/M5);
  - attention over the night;
  - embedding-probe R² heatmap (which physiological features are linearly decodable from SleepFM).
- **Fairness:** subgroup AUROC forest plot, and calibration by subgroup.
- **Survival:** Kaplan–Meier curves by risk tertile, time-dependent AUC.

## 8. Repo layout (`sleepcog/`)

```
sleepcog/
  pyproject.toml  README.md  MODEL_CARD.md  .gitignore
  configs/            # YAML: paths, folds, models, compute
  src/sleepcog/
    data/             # demographics, EDF/annotation readers, channel harmonization, NSRR adapter, splits
    features/         # spectral, spindles/SO (YASA), architecture, resp/SpO2, HRV
    fm/               # SleepFM wrapper: preprocessing, embedding, LoRA fine-tune
    models/           # M0–M6
    eval/             # challenge metric, bootstrap, DeLong, NRI, calibration, DCA, fairness, survival
    explain/          # SHAP, ablation, occlusion, IG, embedding probes
    viz/              # one function per figure, shared style
    synth.py          # synthetic PSG + demographics generator (tests and demos)
    cli.py            # `sleepcog <stage>` commands
  scripts/juno/       # SLURM job scripts + env setup
  colab/              # end-to-end notebook
  tests/              # pytest on synthetic data (no real data needed)
  docs/               # spec, plan, analysis plan, guide PDF, manuscript skeleton, TRIPOD+AI checklist
  reports/            # aggregate-only results returned from Juno
```

**CLI stages:** `audit → splits → features → embed → train --model Mx → evaluate → explain → figures → report`. Each stage is idempotent and caches to `cache/`, and every run records its git hash.

## 9. Compute plan

| Stage | Where | Estimate |
|---|---|---|
| Data audit, feature extraction (~1–3k nights) | Juno CPU array jobs | ~1–2 CPU-min/night → about 1 day wall-clock on 32 cores |
| SleepFM embedding | A100 | ~3–6 GPU-h |
| M4 LoRA (5 folds + LOSO) | A100 | ~15–30 GPU-h |
| M5 from scratch (5 folds + LOSO) | A100 | ~15–30 GPU-h |
| Gradient boosting, linear models, bootstrap, SHAP | CPU | hours |
| External MrOS/SOF (~3.4k nights) | Juno | features ~1 day; embedding ~6 GPU-h |

**Claude cloud credit (now → Nov 4):** all code, tests (synthetic), configs, SLURM and Colab, the guide PDF (with figures rendered on synthetic data and labeled as such), manuscript skeleton, TRIPOD+AI checklist, analysis plan and model card.

## 10. Testing

- pytest on the synthetic generator: readers, channel mapping, every feature function (known-answer tests, e.g. a sinusoid → band power, an injected spindle → detected), metric parity with the official `evaluate_model.py`, split integrity (no patient overlap), leakage guards, and a CLI smoke test end to end on 20 synthetic nights.
- GitHub Actions CI on CPU (as in ecg-trust).

## 11. Risks

| Risk | Mitigation |
|---|---|
| Few events | Restrict to 3 primary contrasts (M0, M1, M3); report confidence intervals; use the survival formulation |
| Age confounding | Challenge age-conditioned AUROC, covariate models, age-matched sensitivity analysis |
| Site shift | LOSO, ComBat sensitivity analysis on features, site-colored embeddings |
| SleepFM input mismatch (channels, sampling rate) | Adapter with channel-missingness masks; report coverage |
| NSRR access delay | Paper v1 can go out as a preprint with LOSO only; the external test is added before journal submission |
| License (data or SleepFM) | Check before any run; fall back to M5 or a public sleep encoder |

## 12. Deliverables

1. `sleepcog/` code + tests + CI
2. `docs/analysis_plan.md` (frozen before results; hash-committed)
3. `docs/guide/sleepcog_guide.pdf` (illustrated; how to run everything, the science and the method)
4. `docs/manuscript/` npj-style skeleton (Markdown → PDF), with result placeholders filled from `reports/`
5. `docs/TRIPOD_AI_checklist.md`, `MODEL_CARD.md`
6. Juno SLURM and Colab notebook
