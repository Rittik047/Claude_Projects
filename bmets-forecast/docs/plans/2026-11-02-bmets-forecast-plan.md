# bmets-forecast Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A tested Python package plus Juno and Colab run routes that take UCSD-BMETS-Longitudinal from raw NIfTI files to T1–T4 results, figures, a guide PDF and a manuscript skeleton.

**Architecture:**
- A `bmf` package with one module per pipeline stage, driven by a single CLI (`bmf <stage> --config`).
- Stages read and write a cache directory, so each stage can be re-run on its own.
- Juno (SLURM) and Colab (Google Drive) are thin route wrappers around the same CLI.
- Heavy dependencies (nnU-Net, torch, ANTs, pyradiomics, foundation-model weights) are optional imports, so the tests run on CPU with synthetic data.

**Tech Stack:** Python 3.11, numpy, pandas, nibabel, SimpleITK, scipy, scikit-learn, lightgbm, lifelines, shap, matplotlib/seaborn, torch (optional), nnunetv2 (optional), antspyx (optional), pyradiomics (optional), monai (optional), pytest, reportlab (PDF).

**Spec:** `bmets-forecast/docs/specs/2026-11-02-bmets-forecast-design.md`

## Global Constraints

- The package root is `bmets-forecast/`, the import name is `bmf`, and the CLI entry point is `bmf`.
- No raw data in git: `.gitignore` covers `data/`, `cache/`, `*.nii*`, `runs/`. The 4 public TSVs are read from `data/clinical/`.
- Splits: 5-fold, patient-level, stratified by primary tumour × number of time points, `seed=2026`, saved to `splits.json`.
- RANO-BM volumetric thresholds:
  - progression: ≥ +73% vs. nadir, or a new lesion ≥ 0.05 cc;
  - partial response: ≤ −65%;
  - complete response: enhancing volume of 0.
- Lesion matching uses Hungarian assignment with a 10 mm centroid gate.
- Labels follow the BraTS-METS 2025 scheme `{1:NETC, 2:SNFH, 3:ET, 4:RC}`, configurable in `configs/base.yaml: labels`.
- Bootstrap: 2,000 resamples, clustered by patient. Holm correction for the T2 contrasts P1, P2, P3 and P5 vs. P0.
- Every figure function signature is `fig_<name>(..., out: Path) -> Path` and saves both PNG (300 dpi) and PDF.
- Commits end with the session attribution trailer.

## Review Focus

1. **Patients with a normal pre-diagnosis scan (17):** folder `_01` must map to "normal", and the post-diagnosis index must shift by one. Pinned in Task 1 (`test_timepoint_remap_with_normal_mri`).
2. **Intervals recorded as 0.0 years (60):** these must be kept, flagged with `zero_interval=True`, and not cause division by zero in growth-rate calculations. Pinned in Task 4 (`test_zero_interval_growth_rate_finite`).
3. **A time point with no enhancing lesion** (complete response, or the normal MRI): the tracker must return an empty track set, and labels must not raise. Pinned in Task 4 (`test_empty_timepoint_tracks`).
4. **A missing optional sequence (ASL, DWI, ADC, SWAN):** the loader returns a zero channel plus a mask bit, never a KeyError. Pinned in Task 2 (`test_missing_optional_sequence_masked`).
5. **Typos in the clinical table** (`"Resection "`, `"resection"`, `"No "`, cp1252 bytes): normalised on load. Pinned in Task 1 (`test_dirty_values_normalized`).

---

### Task 1: Scaffold, clinical parsing, time-point remap and cohort audit

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `configs/base.yaml`, `src/bmf/__init__.py`, `src/bmf/config.py`, `src/bmf/data/clinical.py`, `src/bmf/data/audit.py`
- Test: `tests/test_clinical.py`, `tests/fixtures/clinical_min.tsv` (5 fabricated rows in the same columns, including a patient with a normal MRI, one with 2 MRIs, one dead, and dirty values), `tests/fixtures/mri_meta_min.tsv`

**Interfaces:**
- Produces:
  - `load_config(path: Path) -> dict`, which merges `base.yaml` with a route yaml;
  - `load_clinical(path: Path) -> pd.DataFrame`: one row per patient, columns `case_id, sex, ethnicity, race, primary, dx_method, age_death, has_normal_mri, age_normal, resection, rt_before, age_t1..age_t4, rt_12, rt_23, rt_34`; read with cp1252, values stripped and lower-cased (Yes/No → bool);
  - `timepoints(clin) -> pd.DataFrame`: one row per folder, columns `study_id, case_id, folder_idx, post_idx (0 = normal, 1..4), age, is_normal`;
  - `intervals(clin) -> pd.DataFrame`: columns `case_id, t, t_next, dt_years, rt_between, zero_interval`;
  - `survival_table(clin) -> pd.DataFrame`: columns `case_id, time_years, event`, measured from `age_t1` to death, or censored at the last MRI;
  - `load_mri_meta(path) -> pd.DataFrame`, with sequence columns as booleans;
  - `audit(clin, meta) -> dict`, which also writes `reports/audit.json`.

- [ ] **Step 1: Write the failing tests.**
  - `test_load_clinical_shapes`: 5 rows; `age_t3` is NaN for the patient with 2 MRIs.
  - `test_dirty_values_normalized`: `"Resection "` → `resection`; `"No "` → `False`.
  - `test_timepoint_remap_with_normal_mri`: patient X with a normal MRI → folder `_01` has `is_normal=True, post_idx=0`, and `_02` has `post_idx=1`.
  - `test_intervals`: `dt_years` = age difference; `zero_interval` is True when the difference is 0.0.
  - `test_survival_table`: a dead patient has `event=1` and `time = age_death - age_t1`; an alive patient has `event=0` and `time = age_last - age_t1`.
- [ ] **Step 2:** Run `pytest tests/test_clinical.py -v`. Expected: FAIL (module missing).
- [ ] **Step 3:** Implement the functions above. In `audit()`, reproduce the numbers in spec §2.
- [ ] **Step 4:** Run the tests (PASS). Then run `python -m bmf.data.audit --clinical <real tsv> --meta <real tsv>` locally and check the output: 300 patients, 1,065 time points, 748 intervals, 247 intervals with radiation, 153 deaths.
- [ ] **Step 5:** Commit: `feat(bmf): clinical parsing, timepoint remap, cohort audit`.

### Task 2: MRI index, NIfTI I/O, splits

**Files:**
- Create: `src/bmf/data/mri_index.py`, `src/bmf/data/nifti_io.py`, `src/bmf/data/splits.py`
- Test: `tests/test_index_io_splits.py`

**Interfaces:**
- Consumes: `timepoints()` from Task 1.
- Produces:
  - `SEQS_CORE = ("t1c","t1n","t2f","t2w")` and `SEQS_OPT = ("t1postFSE","dwi","adc","swan","asl")`;
  - `build_index(root: Path, tp: pd.DataFrame) -> pd.DataFrame`, columns `study_id, seq, path, exists`; seg file glob is `*-seg.nii.gz`, configurable;
  - `load_study(index, study_id, seqs, spacing=None) -> (np.ndarray[C,X,Y,Z], np.ndarray[C] mask, affine)`;
  - `load_seg(index, study_id) -> np.ndarray[int]`;
  - `make_splits(clin, n_folds=5, seed=2026) -> dict[case_id, fold]` and `save_splits` / `load_splits`.

- [ ] **Step 1: Write the failing tests** (using tiny NIfTIs written with nibabel to `tmp_path`).
  - `test_build_index_finds_files`.
  - `test_missing_optional_sequence_masked`: `asl` absent → that channel is all zeros and `mask[asl]=0`.
  - `test_splits_patient_level_and_stratified`: no patient appears in two folds; each fold contains each primary tumour that has at least 5 patients.
  - `test_splits_deterministic`: the same seed gives identical splits.
- [ ] **Step 2:** Run (FAIL).
- [ ] **Step 3:** Implement. Splits use `StratifiedKFold` on the string `primary|n_tp`; rare strata collapse to `other|n_tp`.
- [ ] **Step 4:** Run (PASS).
- [ ] **Step 5:** Commit.

### Task 3: Synthetic 4D brain-metastasis generator

**Files:**
- Create: `src/bmf/synth.py`
- Test: `tests/test_synth.py`

**Interfaces:**
- Produces: `make_synthetic_cohort(root: Path, n_patients=6, shape=(48,48,48), seed=0) -> dict`. It writes the TCIA folder and file layout plus a clinical TSV and a metadata TSV in the real column names. Lesions are spheres that grow, shrink or appear, according to a known ground-truth table returned in `dict["truth"]` with columns `case_id, t, lesion_id, volume_cc, is_new`.

- [ ] **Step 1: Write the failing test** `test_synth_roundtrip`: the generated files pass through `load_clinical`, `build_index` and `load_seg`; the ground-truth table contains at least one new lesion and one complete response.
- [ ] **Step 2:** Run (FAIL).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run (PASS).
- [ ] **Step 5:** Commit.

### Task 4: Lesion components, tracking, and RANO-BM / new-lesion labels

**Files:**
- Create: `src/bmf/lesions/components.py`, `src/bmf/lesions/tracking.py`, `src/bmf/lesions/labels.py`
- Test: `tests/test_lesions.py`

**Interfaces:**
- Consumes: `load_seg`, `intervals`, and the synthetic cohort.
- Produces:
  - `lesions(seg, spacing, labels_cfg) -> pd.DataFrame`, columns `lesion_id, centroid_mm(x,y,z), et_cc, tc_cc, wt_cc, max_diam_mm`. Lesions are 26-connected components of ET ∪ NETC; the resection cavity is excluded.
  - `match(prev: DataFrame, curr: DataFrame, gate_mm=10.0) -> list[tuple[prev_id|None, curr_id|None]]`, using `scipy.optimize.linear_sum_assignment` on centroid distance, with pairs beyond the gate rejected.
  - `track_patient(per_tp: list[DataFrame]) -> pd.DataFrame`, columns `track_id, post_idx, lesion_id, status ∈ {stable,new,resolved}`.
  - `interval_labels(tracks, per_tp, intervals) -> pd.DataFrame`, columns `case_id, t, t_next, prog_t2a, newmet_t2b, sum_et_t, sum_et_next, nadir_et`.
  - `lesion_trajectories(tracks, per_tp, intervals) -> pd.DataFrame`, columns `case_id, track_id, t, log_ratio, rano_class, growth_rate_per_yr`.

- [ ] **Step 1: Write the failing tests.**
  - `test_components_volume`: a sphere of radius 5 voxels at 1 mm spacing → `et_cc ≈ 0.5236 ± 5%`.
  - `test_match_gate`: lesions 15 mm apart are not matched.
  - `test_new_lesion_flag`: a synthetic case with a lesion appearing at t2 → `newmet_t2b=1` for interval (1,2).
  - `test_prog_threshold`: sum of ET +80% → `prog_t2a=1`; +60% with no new lesion → 0.
  - `test_new_lesion_volume_floor`: a new lesion of 0.03 cc triggers neither `prog_t2a` nor `newmet_t2b`; one of 0.06 cc triggers both (the 0.05 cc floor applies to both).
  - `test_empty_timepoint_tracks`: an empty segmentation → empty DataFrame, no exception.
  - `test_zero_interval_growth_rate_finite`: `dt=0` → `growth_rate_per_yr` is NaN and `log_ratio` is finite.
  - `test_synth_truth_recovered`: on `make_synthetic_cohort`, tracked new lesions equal `truth.is_new`.
- [ ] **Step 2:** Run (FAIL).
- [ ] **Step 3:** Implement. The nadir is the minimum `sum_et` over post-diagnosis time points up to and including t. Classes: complete response if `sum_et_next == 0`; partial response if ≤ −65% vs. baseline; progression if ≥ +73% vs. nadir or a new lesion; otherwise stable.
- [ ] **Step 4:** Run (PASS).
- [ ] **Step 5:** Commit.

### Task 5: Preprocessing

**Files:**
- Create: `src/bmf/preprocess/pipeline.py`
- Test: `tests/test_preprocess.py`

**Interfaces:**
- Produces:
  - `resample_iso(img_sitk, spacing=1.0, is_label=False)`;
  - `n4(img_sitk, mask)`;
  - `zscore(arr, mask)`;
  - `register_rigid_affine(fixed, moving, backend="sitk"|"ants") -> transform`;
  - `preprocess_study(index, study_id, ref_study_id|None, out_dir) -> Path`. Sequences are aligned to t1c, then to the patient's baseline t1c. The function writes `<out>/<study_id>/<seq>.nii.gz` and `seg.nii.gz`, with the segmentation resampled by nearest neighbour.

- [ ] **Step 1: Write the failing tests.**
  - `test_resample_label_nearest`: only the original label values survive resampling.
  - `test_zscore_inside_mask`: mean ≈ 0 and SD ≈ 1 inside the mask.
  - `test_register_recovers_shift`: a 3-voxel translation on a synthetic volume is recovered to within 1 voxel by the sitk backend.
- [ ] **Step 2:** Run (FAIL).
- [ ] **Step 3:** Implement with SimpleITK by default; ANTs is optional via `antspyx`.
- [ ] **Step 4:** Run (PASS).
- [ ] **Step 5:** Commit.

### Task 6: Segmentation (S0, S1, S2) wrappers and lesion-wise metrics

**Files:**
- Create: `src/bmf/seg/nnunet_export.py`, `src/bmf/seg/run.py`, `src/bmf/eval/seg_metrics.py`
- Test: `tests/test_seg.py`

**Interfaces:**
- Consumes: `load_splits`, the `preprocess_study` outputs.
- Produces:
  - `export_nnunet(pp_root, index, splits, dataset_id=501, prior=False) -> Path`. Writes `imagesTr/labelsTr` and `dataset.json`, plus `splits_final.json` mirroring our folds. With `prior=True` (S2), two extra channels are added: the prior time point's t1c and its segmentation (all zeros at the first time point).
  - `nnunet_cmds(dataset_id, folds, config="3d_fullres", trainer="nnUNetTrainer") -> list[str]` (the plan and train commands that the routes execute).
  - `lesionwise_metrics(pred, gt, spacing, labels=("ET","TC","WT")) -> dict`, following BraTS-METS 2025 lesion-wise Dice and F1 with a 2-voxel dilation for matching. Also `sensitivity_by_size(pred, gt, bins_mm=(3,10)) -> DataFrame`.

- [ ] **Step 1: Write the failing tests.**
  - `test_export_layout`: on the synthetic cohort, the exported files and channel count are 4 (or 6 with the prior).
  - `test_splits_mirrored`: `splits_final.json` folds equal our folds.
  - `test_lesionwise_perfect`: pred = gt → Dice 1, F1 1.
  - `test_lesionwise_missed_lesion`: a missed small lesion lowers F1 and appears in the `<3 mm` bin.
- [ ] **Step 2:** Run (FAIL).
- [ ] **Step 3:** Implement. nnU-Net itself is never imported in the tests.
- [ ] **Step 4:** Run (PASS).
- [ ] **Step 5:** Commit.

### Task 7: Feature tables (clinical, lesion, radiomics)

**Files:**
- Create: `src/bmf/features/tables.py`, `src/bmf/features/radiomics.py`
- Test: `tests/test_features.py`

**Interfaces:**
- Consumes: Task 1 tables, Task 4 lesion and label tables.
- Produces:
  - `clinical_design(clin, intervals) -> DataFrame`, indexed by `(case_id, t)`: age, sex, one-hot primary tumour, resection, `rt_between`, `dt_years`, `zero_interval`, `n_prior_tp`;
  - `lesion_summary(per_tp_lesions) -> DataFrame`, indexed by `(case_id, t)`: count, `sum_et`, `sum_wt`, maximum diameter, edema ratio, deltas vs. the previous time point, and a velocity term (new lesions per year so far);
  - `radiomics_features(pp_study_dir, seg, seqs=("t1c","t2f","adc"), backend="pyradiomics"|"simple") -> dict`. The `"simple"` fallback gives intensity statistics, shape and GLCM via scikit-image.

- [ ] **Step 1: Write the failing tests.**
  - `test_design_no_future_leak`: no column at row t uses information from time point t+1 (checked by perturbing the t+1 segmentation: the features are unchanged).
  - `test_simple_radiomics_keys` (a stable key set).
- [ ] **Step 2:** Run (FAIL).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run (PASS).
- [ ] **Step 5:** Commit.

### Task 8: Foundation-model encoder registry and embeddings

**Files:**
- Create: `src/bmf/fm/registry.py`, `src/bmf/fm/embed.py`, `src/bmf/fm/lora.py`
- Test: `tests/test_fm.py` (skipped if torch is missing)

**Interfaces:**
- Produces:
  - `ENCODERS: dict[str, EncoderSpec]` with keys `brainsegfounder`, `samed3d`, `medsam2`, `nnunet_enc`, `random3d`. `EncoderSpec` has `load(weights_path|None) -> nn.Module`, `in_channels`, `patch`, `out_dim`, `preprocess`.
  - `embed_study(encoder, vol[C,X,Y,Z], seg) -> {"brain": np.ndarray[D], "lesions": {lesion_id: np.ndarray[D]}}`. The brain embedding is the mean of sliding-window patch embeddings; lesion embeddings come from 64³ crops centred on each lesion centroid.
  - `apply_lora(module, r=8, alpha=16, targets=("qkv","proj"))`.
  - Embeddings are cached to `cache/emb/<encoder>/<study_id>.npz`.

- [ ] **Step 1: Write the failing tests.**
  - `test_random3d_embed_shapes`: brain vector of length D, one vector per lesion.
  - `test_lora_trainable_params`: only the LoRA parameters require gradients.
  - `test_registry_missing_weights_message`: a clear error that names the expected path and download URL.
- [ ] **Step 2:** Run (FAIL).
- [ ] **Step 3:** Implement. Real-weight loaders are written but exercised only on Juno or Colab. Weight URLs and paths live in `configs/base.yaml: fm`.
- [ ] **Step 4:** Run (PASS).
- [ ] **Step 5:** Commit.

### Task 9: Prediction models P0–P5 and O1

**Files:**
- Create: `src/bmf/models/tabular.py` (P0 logistic, P1 LightGBM, P2 probe), `src/bmf/models/temporal.py` (P3/P4 transformer), `src/bmf/models/fusion.py` (P5), `src/bmf/models/survival.py` (O1), `src/bmf/models/cv.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: the feature tables, embeddings and splits.
- Produces:
  - `run_cv(model_name, X, y, groups=case_id, splits, task="binary"|"regression"|"survival") -> OOF DataFrame` with columns `case_id, t, y, pred, fold`.
  - `TemporalTransformer(d_in, d_model=128, n_layers=2, n_heads=4, n_clin)`, where `forward(seq[B,T,D], dt[B,T], clin[B,n_clin], mask[B,T]) -> logits[B]`. Time is encoded as sinusoidal features of cumulative years.
  - `stack_oof(oofs: list[DataFrame]) -> DataFrame` (nested: the stacker is trained only on out-of-fold predictions from the training folds).
  - `cox_cv(...)` and `deepsurv_cv(...)` → out-of-fold risk.

- [ ] **Step 1: Write the failing tests.**
  - `test_run_cv_no_patient_leak`.
  - `test_p1_learns_signal`: on a synthetic table where y depends on `sum_et_delta`, AUROC > 0.8.
  - `test_temporal_forward_mask`: changing a masked time step does not change the output.
  - `test_stack_uses_only_train_oof`.
  - `test_cox_cindex_signal`: C-index > 0.7 on synthetic data.
- [ ] **Step 2:** Run (FAIL).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run (PASS).
- [ ] **Step 5:** Commit.

### Task 10: Evaluation and statistics

**Files:**
- Create: `src/bmf/eval/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Produces:
  - `auroc_ci(y, p, groups, n_boot=2000, seed=0) -> (est, lo, hi)`, clustered by patient;
  - `delta_auc(y, p_a, p_b, groups) -> (delta, lo, hi, p)`;
  - `holm(pvals) -> adjusted`;
  - `calibration(y, p, n_bins=10) -> {ece, slope, intercept, table}`;
  - `decision_curve(y, p, thresholds) -> DataFrame`;
  - `cindex_ci(time, event, risk, groups)`;
  - `td_auc(time, event, risk, horizons=(0.5,1.0))`;
  - `subgroup_table(oof, clin, by=("primary","sex","rt_between","field_strength")) -> DataFrame`.

- [ ] **Step 1: Write the failing tests.**
  - `test_auroc_ci_perfect`: estimate 1.0.
  - `test_holm_known`: [0.01, 0.04, 0.03] → [0.03, 0.06, 0.06].
  - `test_calibration_perfect_slope`: slope ≈ 1 on calibrated synthetic data.
  - `test_decision_curve_treat_all`.
  - `test_cluster_bootstrap_resamples_patients`.
- [ ] **Step 2:** Run (FAIL).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run (PASS).
- [ ] **Step 5:** Commit.

### Task 11: Explainability

**Files:**
- Create: `src/bmf/explain/shap_tab.py`, `src/bmf/explain/saliency.py`, `src/bmf/explain/ablation.py`
- Test: `tests/test_explain.py`

**Interfaces:**
- Produces:
  - `shap_values(model, X) -> (values, base, X)`, using TreeExplainer for LightGBM and LinearExplainer for logistic regression;
  - `gradcam3d(model, vol, target_layer)` and `integrated_gradients(model, vol, baseline=0, steps=32)`, each returning a volume the shape of the input;
  - `sequence_ablation(score_fn, X_by_seq) -> DataFrame` (drop one sequence at a time; report Δ metric).

- [ ] **Step 1: Write the failing tests.**
  - `test_shap_additivity`: base + Σ SHAP values ≈ the model's margin.
  - `test_ig_completeness`: Σ IG ≈ f(x) − f(baseline) within 5%, on a tiny torch model.
  - `test_ablation_rows`.
- [ ] **Step 2:** Run (FAIL).
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run (PASS).
- [ ] **Step 5:** Commit.

### Task 12: Figures (≥35 functions), with the data figures rendered now from the real TSVs

**Files:**
- Create: `src/bmf/viz/style.py`, `src/bmf/viz/data_figs.py`, `src/bmf/viz/seg_figs.py`, `src/bmf/viz/pred_figs.py`, `src/bmf/viz/traj_figs.py`, `src/bmf/viz/surv_figs.py`
- Test: `tests/test_viz.py` (every function runs on synthetic inputs and writes a PNG and PDF)

**Interfaces:**
- Produces: `fig_*` functions covering the spec §6 list. The data figures use the real TSVs:
  - cohort flow;
  - swimmer plot;
  - interval histogram;
  - radiation-by-interval bars;
  - primary-tumour bars;
  - age by primary tumour;
  - scanner/field-strength mosaic;
  - sequence-availability heatmap;
  - survival by primary tumour (Kaplan–Meier, from the TSV).
  - The `FIGURES` registry maps each figure name to its function and required inputs.

- [ ] **Step 1: Write the failing test** `test_all_figures_render`, parametrized over `FIGURES`.
- [ ] **Step 2:** Run (FAIL).
- [ ] **Step 3:** Implement. Use the dataviz palette from `style.py`, colour-blind safe, with a consistent font.
- [ ] **Step 4:** Run (PASS). Then run `bmf figures --only data` on the real TSVs and commit the PNGs to `docs/figures/data/` (these are aggregate plots of public data).
- [ ] **Step 5:** Commit.

### Task 13: CLI and configs

**Files:**
- Create: `src/bmf/cli.py`, `configs/juno.yaml`, `configs/colab.yaml`
- Test: `tests/test_cli_e2e.py`

**Interfaces:**
- Produces: `bmf {audit,index,splits,preprocess,seg-export,seg-cmds,seg-eval,track,labels,features,embed,train,evaluate,explain,figures,report} --config <yaml> [--fold k] [--model P1]`.
- `report` writes `reports/results.json` and `reports/tables/*.csv`, with aggregate values only.

- [ ] **Step 1: Write the failing test** `test_e2e_synthetic`: on the synthetic cohort, run the stages `audit → index → splits → preprocess → track → labels → features → embed (random3d) → train P0/P1/P3 → evaluate → figures → report`. Assert that `results.json` contains AUROC entries for P0, P1 and P3.
- [ ] **Step 2:** Run (FAIL).
- [ ] **Step 3:** Implement with argparse. Each stage checks its cache and skips work already done unless `--force` is given.
- [ ] **Step 4:** Run (PASS).
- [ ] **Step 5:** Commit.

### Task 14: Juno route

**Files:**
- Create:
  - `juno/README_JUNO.md`;
  - `juno/00_env.sh` (module load + conda env);
  - `juno/01_aspera_pull.sh` (the `ascp` command template for the TCIA manifest, with resume);
  - `juno/02_preprocess.sbatch` (array job over studies);
  - `juno/03_seg_train.sbatch` (nnU-Net per fold, A100);
  - `juno/04_embed.sbatch`;
  - `juno/05_train_eval.sbatch`;
  - `juno/06_figures_report.sbatch`;
  - `juno/07_external.sbatch`;
  - `juno/run_all.sh` (a dependency chain using `--dependency=afterok`).
- Test: `tests/test_routes.py::test_juno_scripts_reference_cli` (every sbatch file calls `bmf` with a valid stage; `bash -n` syntax check).

- [ ] **Step 1:** Write the failing test.
- [ ] **Step 2:** Run (FAIL).
- [ ] **Step 3:** Write the scripts. Partition, account and paths are variables at the top of `00_env.sh` (`JUNO_PARTITION`, `JUNO_ACCOUNT`, `BMF_DATA`, `BMF_CACHE`).
- [ ] **Step 4:** Run (PASS).
- [ ] **Step 5:** Commit.

### Task 15: Colab route (Google Drive)

**Files:**
- Create: `colab/README_COLAB.md`, plus five notebooks:
  - `colab/01_setup_and_audit.ipynb`: mount Drive, `BMF_DATA=/content/drive/MyDrive/bmets/...`, pip install, audit;
  - `colab/02_preprocess.ipynb` (chunked, resumable);
  - `colab/03_segmentation.ipynb` (nnU-Net one fold per session, checkpoints to Drive);
  - `colab/04_embed_train_eval.ipynb`;
  - `colab/05_figures_report.ipynb`.
- Test: `tests/test_routes.py::test_colab_notebooks_valid` (nbformat validation; each notebook mounts Drive and calls `bmf`).

- [ ] **Step 1:** Write the failing test.
- [ ] **Step 2:** Run (FAIL).
- [ ] **Step 3:** Generate the notebooks with nbformat from `colab/_build.py`.
- [ ] **Step 4:** Run (PASS).
- [ ] **Step 5:** Commit.

### Task 16: Docs, guide PDF, manuscript skeleton, CI

**Files:**
- Create:
  - `README.md`, `MODEL_CARD.md`;
  - `docs/analysis_plan.md` (frozen pre-specification from spec §3, §4, §9);
  - `docs/TRIPOD_AI_checklist.md`, `docs/CLAIM_checklist.md`;
  - `docs/manuscript/manuscript.md` (npj structure with placeholders bound to `reports/results.json` keys);
  - `docs/guide/build_guide.py` → `docs/guide/bmets_forecast_guide.pdf`;
  - `.github/workflows/bmets-ci.yml`.
- Test: `tests/test_docs.py` (the guide builds; the PDF has at least 15 pages; every figure referenced exists).

- [ ] **Step 1:** Write the failing test.
- [ ] **Step 2:** Run (FAIL).
- [ ] **Step 3:** Write the docs. The guide PDF (reportlab) covers:
  - background;
  - the dataset with real audit figures;
  - task definitions with diagrams (lesion tracking and RANO-BM);
  - models;
  - the Juno and Colab step-by-step routes;
  - expected outputs;
  - how to read each figure type, illustrated with synthetic examples labelled as such;
  - reproducibility checklist;
  - timeline to submission.
- [ ] **Step 4:** Run the full suite with `pytest -q` (PASS) and build the PDF.
- [ ] **Step 5:** Commit and push.
