# CLAIM 2024 (Checklist for Artificial Intelligence in Medical Imaging): key items

| Domain | Item | Location |
|---|---|---|
| Title/abstract | AI method + imaging modality identified | Title, Abstract |
| Data | Source, inclusion/exclusion, de-identification | Methods/Data; TCIA page |
| Data | Preprocessing, missing sequences | `bmf.preprocess`; `load_study` masks; sequence-availability figure |
| Data | Partitions (patient-level), leakage control | `splits.json`; `test_run_cv_no_patient_leak`; `test_design_no_future_leak` |
| Ground truth | Definition, annotators, rationale | Neuroradiologist-approved segmentations (source); volumetric RANO-BM labels (analysis plan §2) |
| Model | Architecture, initialisation, pretrained weights and licences | `bmf.fm.registry`; MODEL_CARD |
| Training | Hyper-parameters, augmentation, early stopping | configs/*.yaml; nnU-Net defaults |
| Evaluation | Metrics, CIs, statistical tests | analysis_plan §6–7 |
| Evaluation | External testing | `bmf external` (single locked run) |
| Explainability | Methods + validation of saliency | `bmf.explain`; IG completeness test |
| Failure analysis | Missed lesions, subgroup failures | fn_gallery, subgroup_forest |
| Availability | Code, models, data | Code/Data availability statements |
