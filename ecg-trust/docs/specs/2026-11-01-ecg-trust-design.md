# ecg-trust — design spec

**Date:** 2026-11-01 · **Status:** awaiting author review · **Owner:** Rittik Patra

A 12-lead ECG classifier that comes with evidence for *when to trust it*. It is an
industry-facing portfolio project for a medtech / digital-health ML role. The selling point
is rigorous, reproducible clinical-ML engineering: patient-level evaluation, calibration,
calibrated abstention, subgroup and robustness analysis, external validation, a commercial-
algorithm comparison, and a deployable service. The goal is not to report a new
state-of-the-art AUC.

## 1. Scope and non-claims

**In scope:** multi-label classification into the five PTB-XL diagnostic superclasses
(NORM, MI, STTC, CD, HYP) from 10 s, 12-lead ECGs, with every analysis in §5.

**Explicit non-claims (stated in README and model card):**
- Not a medical device, not clinically validated, not for diagnostic use.
- PTB-XL labels are derived from reports. Some were initially machine-generated, and
  5,743 / 21,799 records (26 %) were not validated by a human cardiologist. The labels are
  therefore a noisy reference standard, not ground truth.
- Conformal guarantees are marginal and hold only under exchangeability. The external
  dataset is expected to violate exchangeability, and the analysis measures by how much.
- Subgroup results are descriptive. Small subgroups are reported with CIs and are not
  over-interpreted.

**Out of scope:** 500 Hz modelling, the 71-statement / form / rhythm tasks, new
architectures, explainability beyond lead occlusion, and clinical decision thresholds.

## 2. Data (all verified 2026-11-01 on the AWS PhysioNet open-data mirror)

Mirror base: `https://physionet-open.s3.amazonaws.com/`. Direct `physionet.org`, Zenodo,
Kaggle, Hugging Face and UCI are blocked in this cloud environment. All datasets below are
licensed CC BY 4.0 (license files checked).

| Dataset | Version | Role | Verified facts |
|---|---|---|---|
| **PTB-XL** ([Wagner et al. 2020](https://doi.org/10.1038/s41597-020-0495-6)) | 1.0.3 (latest; 1.0.1/1.0.2 also present) | Development + internal test | 21,799 records / 18,869 patients; 10 folds of 2,173–2,198 records; 0 patients span folds; 100 Hz `records100/` used; SHA-256 sums provided |
| **PTB-XL+** ([Strodthoff et al. 2023](https://doi.org/10.1038/s41597-023-02153-8)) | 1.0.1 (latest) | Feature baseline + commercial-algorithm comparator | 21,799 rows keyed by `ecg_id`; features from GE **12SL**, **Uni-G** (Glasgow) and open-source **ECGDeli** (~196 described features); 12SL machine statements per ECG; label mappings to OMOP/SNOMED concept IDs; median beats; fiducial points |
| **Large-scale 12-lead arrhythmia DB** (Chapman University / Shaoxing People's Hospital / Ningbo First Hospital; [PhysioNet `ecg-arrhythmia`](https://physionet.org/content/ecg-arrhythmia/1.0.0/)) | 1.0.0 | **External validation** (different country, hospitals, devices) | 45,152 records; 500 Hz, 12 leads, 10 s (5,000 samples); `.hea` + `.mat`, ~120 KB per record (~5.4 GB total); header has `Age`, `Sex`, `Dx` as SNOMED CT codes; 63 condition codes listed |

**Facts that change the design:**
- PTB-XL `age` is coded **300** for 293 patients aged > 89 (de-identification). These map to
  the age band "> 89" and are never used as a numeric 300.
- **411** PTB-XL records carry no diagnostic-superclass label. They are excluded from training
  and evaluation, and the exclusion count is reported.
- PTB-XL's own artefact annotations (`static_noise` 3,260; `baseline_drift` 1,598;
  `burst_noise` 613; `electrodes_problems` 30) and `pacemaker` (291) allow robustness checks
  on **real** artefacts. There are at least 6 recording devices.
- PTB-XL+ concept IDs are **OMOP concept IDs** (e.g. 4008453 = "SNOMED CT Concept"), *not*
  raw SNOMED CT codes. The external DB uses raw SNOMED CT codes. Cross-dataset label mapping
  is therefore done by an explicit, hand-reviewed acronym → superclass table (§4.3), not by
  ID matching.
- The MIT-BIH Noise Stress Test DB is **not** on the mirror, so §5.5 uses synthetic,
  physically parameterised noise. If `physionet.org` is later allow-listed, real NSTDB noise
  replaces it with no other design change.

## 3. Splits and leakage control

| Split | Source | Use |
|---|---|---|
| train | PTB-XL folds 1–8 | Fit all models and preprocessing statistics |
| val-select (9a) | PTB-XL fold 9, patient-level 50 % | Early stopping, hyperparameters, model selection |
| val-calib (9b) | PTB-XL fold 9, other 50 % of patients | Temperature scaling + conformal calibration only |
| test-internal | PTB-XL fold 10 | Run **once**, after freeze (§7) |
| test-external | `ecg-arrhythmia` | Run **once**, after freeze |

These splits follow the benchmark protocol of [Strodthoff et al. 2021](https://doi.org/10.1109/JBHI.2020.3022989)
(folds 9/10 = validation/test). Verified from metadata: `validated_by_human` = 100 % in
folds 9 and 10, versus 64–68 % in folds 1–8. Evaluation labels are therefore cleaner than
training labels, and the README states this.

Required automated tests:
- No `patient_id` appears in more than one split.
- Normalisation statistics are computed on train only.
- Test files are never read before the freeze tag exists. This is enforced by a guard in
  the evaluation CLI.

## 4. Models

### 4.1 Feature baseline (replaces the hand-crafted baseline in the earlier design)
Gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier` or LightGBM) on
**PTB-XL+ 12SL features**, one-vs-rest per superclass. These are clinically meaningful
measurements (intervals, amplitudes, axes) produced by a commercial algorithm. A secondary
run on **ECGDeli** features tests whether an open-source feature extractor suffices,
because 12SL is not available at deployment time.

### 4.2 Deep model
- Compact 1-D ResNet (PyTorch, CPU), input 12 × 1,000 at 100 Hz, per-lead standardisation.
- Loss: BCE with logits.
- **3 seeds → deep ensemble** (mean of probabilities; [Lakshminarayanan et al. 2017](https://arxiv.org/abs/1612.01474)).
- Reference point: published 1-D CNNs reach ~0.93 macro-AUROC on this task (Strodthoff et
  al. 2021). The exact table values are to be re-checked before quoting them in the README.
  Matching that reference is a sanity check, not a target.

### 4.3 External-label harmonisation (`ecg-arrhythmia` → superclasses)
A versioned CSV maps the 63 acronyms to superclasses. Proposed default mapping, to be
reviewed by the author before external test:
- **CD:** 1AVB, 2AVB*, 3AVB, AVB, LBBB, RBBB, IVB/IDC, LBBBB, LFBBB, WPW, VPE, PRIE
- **HYP:** LVH, RVH, RAH
- **STTC:** STTC, STDD, STE, STTU, TWC, TWO, QTIE
- **MI:** MI, MIBW, MIFW, MILW, MISW
- **NORM (proxy):** `SR` as the only code. This is a proxy, because sinus rhythm alone
  ≠ a normal ECG, and it is reported as such.
- Codes with no PTB-XL superclass equivalent (rhythm, axis, ectopy, etc.) are not used as
  targets.

External metrics are reported per mapped class with prevalence. Classes with fewer than 100
positives are reported as "insufficient". A known caveat is reported rather than corrected:
the source annotations differ in labelling process, and the label shift is expected.

### 4.4 Commercial comparator (GE 12SL statements, PTB-XL+)
The 12SL machine statements are mapped to superclasses via the same reviewed-table approach.
Because 12SL gives binary decisions only, the comparison is made **at its operating point**:
12SL's sensitivity/specificity versus the ensemble's ROC curve and the ensemble's
sensitivity at 12SL's specificity, on the internal test fold. Caveat stated: PTB-XL's
reports were partly machine-initialised, but by the recording devices (e.g. Schiller AT-6),
not by 12SL.

## 5. Trust analyses

1. **Discrimination:** per-class and macro AUROC and AUPRC. 95 % CIs come from a
   **patient-level** bootstrap (2,000 resamples).
2. **Calibration:**
   - Reliability diagrams.
   - Per-class ECE (15 equal-mass bins; the binning sensitivity is also reported).
   - Brier score.
   - All three before and after per-class temperature scaling fitted on 9b
     ([Guo et al. 2017](https://arxiv.org/abs/1706.04599)).
3. **Conformal abstention:** per-class (label-conditional) split conformal on 9b. Each label
   becomes {present, absent, refer} at target α ∈ {0.05, 0.10}. The report gives empirical
   coverage, refer rate, and accuracy on non-referred cases
   ([Angelopoulos & Bates 2021](https://arxiv.org/abs/2107.07511)). The same quantities are
   reported on the external set to measure coverage degradation under dataset shift.
4. **Subgroups:** sex, age band (< 40, 40–59, 60–79, 80–89, > 89), device, the noise flags,
   and pacemaker. Each gets n, AUROC with CI, and ECE. No subgroup is declared different
   without a CI-based comparison; the comparisons are framed as descriptive, with the
   multiplicity caveat.
5. **Robustness to noise (synthetic, physically parameterised):**
   - Noise types:
     - baseline wander: 0.05–0.5 Hz sinusoid mixture + random walk;
     - EMG-like: band-limited Gaussian 20–45 Hz;
     - electrode-motion-like: step and transient bursts;
     - lead dropout: flatline of k leads.
   - Noise is added at SNR = 24, 18, 12, 6, 0, −6 dB, with SNR defined on signal power per
     lead. Unit tests check that the measured SNR is within 0.1 dB of the target.
   - Output: AUROC, ECE and refer rate versus SNR. Key question: does the refer rate rise as
     quality falls, or does the model stay confidently wrong?
   - Plus the real-artefact subgroups from §5.4.
6. **Lead occlusion (light explainability):** the AUROC drop when each lead is zeroed. The
   result is checked against expected physiology (e.g. MI sensitivity to precordial and
   inferior leads) and is not presented as a mechanism.

## 6. Engineering deliverables

```
ecg-trust/
  pyproject.toml           # pinned deps; package `ecgtrust`
  configs/*.yaml           # every run is config-driven, seeded
  src/ecgtrust/
    data/                  # download (checksum-verified), load, label maps, splits
    features/              # PTB-XL+ feature loading
    models/                # baseline, resnet1d, ensemble
    calibrate/             # temperature scaling, conformal
    evaluate/              # metrics, bootstrap, subgroups, external, comparator
    robustness/            # noise generators (SNR-exact), occlusion
    serve/                 # FastAPI app
  app/                     # Streamlit demo
  tests/                   # pytest; tiny fixtures, no network
  reports/                 # data audit, results report, figures
  docs/specs/, docs/plans/
  MODEL_CARD.md, README.md, Dockerfile
.github/workflows/ecg-trust-ci.yml   # repo root; lint (ruff) + pytest on fixtures
```

- **CLI:** `ecgtrust download | audit | train | calibrate | evaluate --split {val,test,external} | serve`.
- **API:** `POST /predict` accepts a WFDB record pair or a `.npy` of shape 12 × 1000 at
  100 Hz (500 Hz input is resampled). It returns calibrated probabilities, conformal
  decisions, and a signal-quality warning.
- **Demo:** 12-lead plot, per-class probabilities, present/absent/refer badges, and a
  quality flag. Example ECGs are drawn from the test fold only after the freeze.
- **Artefacts:** model weights (expected < 20 MB) are committed; data are never committed
  (`.gitignore`).
- **Hosting:** the author deploys to Streamlit Community Cloud. This session cannot.

## 7. Milestones, checkpoints and freeze

| # | Deliverable | Gate |
|---|---|---|
| **M1** | Package skeleton, checksum-verified download of PTB-XL, PTB-XL+ and external set (full, or stratified ≥ 10k records if download is too slow; logged), **data audit report**, leakage tests, feature baseline on 9a | Author checks credit balance; reviews audit + external label map |
| **M2** | ResNet × 3 seeds, ensemble, 9a results vs baseline | **Freeze:** git tag `freeze-v1` fixes models, thresholds, α, mapping tables |
| **M3** | Calibration, conformal, subgroups, robustness, occlusion on val; then **one** run each on internal test and external test; 12SL comparison | Author reviews results report |
| **M4** | API, Docker, Streamlit demo, CI, MODEL_CARD, README with results table and figures | Portfolio-ready |

Credit expiry (per earlier brief): 2026-11-04 23:59 PT. If the budget runs short, cut in
this order, last listed first: occlusion → ECGDeli baseline → 500 Hz resampling path in API
→ Docker. **Never cut:** leakage tests, single-use test discipline, CIs, external
validation, model card.

## 8. Success criteria

- Fully reproducible from `ecgtrust download` → `evaluate` with fixed seeds; CI green.
- Internal test macro-AUROC is reported with a CI, alongside the published reference.
  Falling short is reported honestly; there is no re-tuning on test.
- Calibration before/after, conformal coverage within its finite-sample bound on internal
  test, and measured coverage on external test are all reported.
- At least one clear, defensible finding about trust. Candidates: ECE reduction after
  scaling; refer rate tracking SNR; magnitude of degradation under external shift;
  ensemble vs 12SL at matched specificity.
- README readable by a hiring manager in 2 minutes; model card follows
  [Mitchell et al. 2019](https://arxiv.org/abs/1810.03993).

## 9. Risks

| Risk | Mitigation |
|---|---|
| CPU training too slow | Smaller ResNet width; fewer epochs with early stopping; 1 seed if forced (ensemble then dropped and stated) |
| External download slow / partial | Stratified subsample by `Dx`; log exact record list and checksums |
| External label mapping contestable | Versioned table, author review before test, per-class reporting, proxy labels flagged |
| 12SL mapping ambiguity | Same; comparator reported as indicative, with mapping published |
| Test-set contamination | Freeze tag + CLI guard + single logged test run |
