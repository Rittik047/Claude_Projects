# TRIPOD+AI checklist (Collins et al., BMJ 2024;385:e078378): where each item is addressed

| # | Item | Location |
|---|---|---|
| 1 | Title: identifies development/evaluation of a prediction model, target population, outcome | manuscript title |
| 2 | Abstract | manuscript Abstract |
| 3a–c | Background, rationale, objectives | Introduction ¶1–4 |
| 4 | Objectives (development + validation) | Introduction ¶4; analysis_plan §6 |
| 5a–b | Data source, dates | Methods/Data; spec §2 (2016–2023, UCSD) |
| 6a–c | Participants, eligibility, treatment | analysis_plan §1; audit.json |
| 7 | Data preparation (preprocessing, harmonisation) | Methods/Preprocessing; `bmf.preprocess` |
| 8a–c | Outcome definition, blinding, timing | analysis_plan §2; `bmf.lesions.labels` |
| 9a–c | Predictors, measurement, timing | analysis_plan §3; `bmf.features` |
| 10 | Sample size | Results/Cohort; events per model reported in results.json |
| 11 | Missing data | `load_study` channel masks; median imputation inside training folds |
| 12a–g | Analysis: model type, tuning, internal validation, performance measures, calibration, fairness, model updating | analysis_plan §4–7; `bmf.models`, `bmf.eval.stats` |
| 13 | Class imbalance | class-weighted losses (P0–P3); AUPRC reported |
| 14 | Fairness | subgroup AUROC/calibration (primary, sex, RT, field strength) |
| 15 | Model output | probabilities; decision curves |
| 16 | Training vs evaluation differences | locked external stage (`bmf lock` / `bmf external`) |
| 17 | Ethics | TCIA de-identified public data; source IRB waiver |
| 18a–c | Open science: funding, conflicts, protocol, registration | analysis_plan.md + SHA-256 (frozen 2026-11-02) |
| 19 | Patient and public involvement | Discussion (not applicable; public data) |
| 20a–b | Participants flow, characteristics | Fig. 1, Table 1 (`reports/audit.json`) |
| 21 | Model development details | Methods/Models; configs/*.yaml |
| 22 | Model specification | code + locked joblib models + model card |
| 23a–b | Performance with CIs; subgroups | results.json; tables/ |
| 24 | Model updating | not performed |
| 25 | Interpretation | Discussion ¶1–3 |
| 26 | Limitations | Discussion ¶5 |
| 27a–c | Usability, implications, future research | Discussion ¶4, ¶6 |
