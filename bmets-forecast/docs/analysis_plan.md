# Pre-specified analysis plan: bmets-forecast

**Version 1.0, frozen 2026-11-02, before any image data was accessed.** Its SHA-256 is stored in `docs/analysis_plan.sha256`. Any change after the first real-data run must be logged in section 10 with a date and a reason.

## 1. Data

- **Development cohort:** UCSD-BMETS-Longitudinal (TCIA, DOI 10.7937/5ejb-9g94, v1, 2026-10-05). 300 patients; 1,065 MRI time points; 748 post-diagnosis intervals (audit, `reports/audit.json`).
- **Unit of analysis:** the interval t → t+1 for T2; the lesion × interval for T3; the patient for T4.
- **Exclusions:** the 17 normal pre-diagnosis MRIs are excluded from T2–T4 labels. They are kept as segmentation negatives.

## 2. Outcomes (derived from expert segmentations, after longitudinal registration)

- **T2a (primary):** intracranial progression at t+1. Either the sum of enhancing-tumour (ET) volume rises ≥ 73% above the nadir, or any new lesion ≥ 0.05 cc appears (volumetric RANO-BM adaptation).
- **T2b (co-primary):** ≥ 1 new enhancing lesion ≥ 0.05 cc at t+1, not matched to any earlier lesion track (Hungarian matching, 10 mm centroid gate).
- **T3 (secondary):** per-lesion log ratio log((V(t+1) + 0.001) / (V(t) + 0.001)) of ET volume, plus a 4-class volumetric response (complete response CR, partial response PR, stable disease SD, progressive disease PD).
- **T4 (secondary):** overall survival from the first post-diagnosis MRI. Death age is from the clinical table; alive patients are censored at their last MRI.

## 3. Predictors (known at time t)

- **Clinical:** age, sex, primary tumour (7 groups), resection before baseline, radiation planned in the interval, interval length, number of prior MRIs, years since diagnosis.
- **Lesion burden:** number of lesions, total and log ET volume, maximum diameter, change from t−1, new lesions per year so far (brain-metastasis velocity).
- **Radiomics:** PyRadiomics first-order, shape, GLCM and GLSZM features from t1c, t2f and ADC, computed within the tumour core (TC) and whole tumour (WT).
- **Imaging foundation-model embeddings:** whole brain plus the mean over lesions, from BrainSegFounder and SAM-Med3D. A randomly initialised encoder serves as an ablation.

## 4. Models

- **P0:** clinical + lesion-burden logistic regression (the reference model).
- **P1:** LightGBM on all tabular features.
- **P2:** linear probe on foundation-model embeddings + P0 features.
- **P3:** temporal transformer over the sequence of per-time-point embeddings + P0 features.
- **P5:** stacked fusion of P0, P1 and P3.
- **T3:** LightGBM regression (P1).
- **T4:** penalised Cox regression and DeepSurv on baseline features.
- **Segmentation (T1):** nnU-Net S0 (4 channels) vs. S2 (S0 plus the prior time point's t1c and segmentation as input channels).

## 5. Validation

- 5-fold cross-validation at the patient level, stratified by primary tumour × number of MRIs (seed 2026). Tuning happens only inside the training folds.
- External validation runs once, using models locked with `bmf lock`:
  - BraTS-METS 2025, non-UCSD cases only, for T1;
  - PROTEAS, for T2/T3;
  - Yale-Brain-Mets-Longitudinal, with labels from automated segmentation, for T2 and for T4 if survival data are available.

## 6. Primary analysis

- **Metric:** AUROC for T2a and T2b, with 95% CIs from a 2,000-resample bootstrap clustered by patient.
- **Pre-specified contrasts:** P1, P2, P3 and P5 each vs. P0, as paired-bootstrap ΔAUROC with Holm correction (4 tests per task).
- **Hypothesis:** imaging-derived models (P2, P3) improve on P0.

## 7. Secondary analyses

- AUPRC.
- Calibration: ECE, slope and intercept.
- Decision curves for threshold probabilities 0.05–0.60.
- Subgroups: primary tumour, sex, radiation in the interval, field strength. Subgroups are reported only when n ≥ 10 and both outcome classes are present.
- T3: MAE and Spearman ρ.
- T4: Harrell's C-index and IPCW time-dependent AUC at 0.5 and 1 year.
- Explainability is exploratory: SHAP, integrated gradients / Grad-CAM, sequence ablation, temporal attention.

## 8. Sensitivity analyses

- Exclude intervals recorded as 0.0 years.
- Drop the radiation predictor.
- Labels from S2 automated segmentation instead of the expert segmentation.
- 1.5T vs. 3T.
- New-lesion threshold of 0.1 cc.

## 9. Reporting

TRIPOD+AI and CLAIM checklists (`docs/`). No treatment-effect claims: radiation is confounded by indication.

## 10. Amendment log

| Date | Change | Reason |
|---|---|---|
| — | — | — |
