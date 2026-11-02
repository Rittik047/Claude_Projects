# Model card: ecg-trust ResNet ensemble (freeze-v1)

The sections follow Mitchell et al., "Model Cards for Model Reporting", FAT* 2019. All numbers are read from `reports/`.

## Model details

| Item | Value |
|---|---|
| Architecture | 1-D residual CNN: 4 stages of 2 basic blocks, widths 64/128/128/128, kernel 7, global average pooling, 5 sigmoid outputs; 1.49 M parameters |
| Ensemble | 3 seeds (0, 1, 2); probabilities averaged (deep ensemble, Lakshminarayanan et al. 2017) |
| Training | AdamW, lr 1e-3, weight decay 1e-2, OneCycle schedule, batch 128, 30 epochs; per-seed checkpoint chosen by val_select macro-AUROC |
| Calibration | Per-class temperature scaling (Guo et al. 2017), fitted on `val_calib` |
| Abstention | Label-conditional (Mondrian) split conformal prediction (Vovk 2005; Angelopoulos & Bates 2023). Each class gets `present`, `absent` or `refer`, at α = 0.10 (served) and α = 0.05 |
| Input | 12-lead ECG, 10 s, in mV. Resampled to 100 Hz; leads normalised with training-set statistics |

**Research variant (not served):** a logit fusion with a gradient-boosted model on the GE 12SL measurements from PTB-XL+. The ResNet logit carries weight w = 0.4, chosen on val_select. It requires 12SL measurements, so it applies to PTB-XL only.

**Version and provenance:**
- freeze commit `aa872cf`, manifest hash `234e641a4a00`;
- served bundle in `release/` (18 MB).

## Intended use

- **Intended:** research and education on trustworthy ECG machine learning (calibration, abstention, robustness, external validation), and portfolio demonstration.
- **Out of scope:** clinical diagnosis or triage, or any decision about a patient. Also out of scope:
  - paediatric ECGs;
  - recordings shorter than 10 s;
  - rhythm diagnosis (the model has no rhythm classes);
  - devices or populations unlike PTB-XL, without re-calibration.

## Factors

- **Groups evaluated:**
  - sex;
  - age band;
  - recording device;
  - PTB-XL signal-quality annotations (baseline drift, static noise, burst noise, electrode problems, pacemaker).
- **Shift factor:** external cohort from China (Chapman University, Shaoxing People's Hospital and Ningbo First Hospital) vs German PTB-XL.
- **Noise factors:** synthetic baseline wander, muscle (EMG) and electrode-motion noise at fixed SNR, plus random lead dropout.

## Metrics

- **Ranking:** per-class and macro AUROC, with 95% CIs from a patient-level bootstrap (2,000 resamples). AUPRC is also reported.
- **Calibration:** ECE with 15 equal-mass bins, and Brier score.
- **Abstention:**
  - conformal coverage, conditional on the true label;
  - refer rate;
  - accuracy on the non-referred ECGs.
- **Comparison with GE 12SL:** model sensitivity at 12SL's own specificity.

## Evaluation data

| Set | Records | Notes |
|---|---|---|
| Internal test: PTB-XL fold 10 | 2,158 | 100% human-validated labels; evaluated once |
| External: PhysioNet ecg-arrhythmia v1.0.0 | 28,646 | strict SNOMED-to-superclass map; 37,331 with the liberal map; evaluated once |

## Training data

- **Training set:** PTB-XL v1.0.3 folds 1–8, 17,084 records. 411 records without any diagnostic superclass were excluded from all splits.
- **Validation:** fold 9, split by patient into `val_select` (1,072 records, model selection) and `val_calib` (1,074 records, calibration and conformal).
- **Labels:** the 5 diagnostic superclasses of the Strodthoff et al. 2021 benchmark. Any diagnostic statement counts; likelihood is ignored.

## Quantitative analyses

| Evaluation | Macro-AUROC [95% CI] | Macro ECE after scaling | Refer rate (α = 0.10) | Positive coverage |
|---|---|---|---|---|
| Internal test, ResNet ensemble | 0.925 [0.917, 0.932] | 0.030 | 12.7% | 0.902 |
| Internal test, fusion | 0.939 [0.933, 0.945] | 0.021 | 9.5% | 0.905 |
| External, ResNet ensemble | 0.949 [0.946, 0.951] | 0.086 | 11.4% | 0.920 (0.878 for NORM, STTC, CD) |

| Class | Internal ResNet AUROC | Fusion AUROC | External AUROC | Sensitivity gain over 12SL at its specificity (ResNet) |
|---|---|---|---|---|
| NORM | 0.945 | 0.953 | 0.967 | +0.228 [+0.160, +0.291] |
| MI | 0.917 | 0.944 | 0.969 | +0.076 [−0.008, +0.126] |
| STTC | 0.941 | 0.943 | 0.928 | +0.269 [+0.203, +0.316] |
| CD | 0.925 | 0.937 | 0.928 | +0.234 [+0.169, +0.282] |
| HYP | 0.897 | 0.919 | 0.951 | +0.126 [+0.055, +0.202] |

**Robustness on the internal test set:**

| Condition | Macro-AUROC | ECE | Refer rate |
|---|---|---|---|
| Clean | 0.925 | – | 12.7% |
| 0 dB SNR, any of the three noise types | 0.912–0.916 | – | – |
| −6 dB EMG | 0.866 | – | – |
| −6 dB electrode motion | 0.886 | 0.245 | 25.5% (falls from 29.0% at 0 dB) |

At −6 dB electrode motion the model becomes over-confident instead of abstaining.

**Subgroups** (descriptive; listed only where 95% CIs do not overlap):

| Class | Group with lower AUROC | Comparison group |
|---|---|---|
| STTC | age 80–89: 0.876 | age 40–59: 0.954 |
| STTC | female: 0.924 | male: 0.955 |
| NORM | CS-12 E device: 0.892 | AT-6 device: 0.966 |

## Ethical considerations

- PTB-XL is from a single German centre (1989–1996). The external cohort is Chinese. Neither represents African, South Asian or Latin American populations, and performance there is unknown.
- Lower STTC performance in older and female patients could translate into unequal error rates if the model were deployed.
- Calibrated probabilities and abstention reduce over-reliance, but only while the input distribution matches calibration. A shift breaks the guarantee silently, which happened on the external cohort.
- The data are de-identified and CC BY 4.0. No new patient data were collected.

## Caveats and recommendations

- Re-calibrate the temperatures and conformal thresholds on local data before any new use. The external results show why.
- Put a signal-quality gate in front of the model. Severe electrode-motion noise produces confident errors.
- Treat the 12SL comparison as indicative: 12SL statements were mapped to superclasses by hand (`configs/12sl_label_map.csv`).
- The external macro-AUROC exceeds the internal one because the strict label map keeps clear-cut cases. It is not evidence of better generalisation.
