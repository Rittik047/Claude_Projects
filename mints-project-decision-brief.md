# MINTS project decision brief — first research round

Prepared November 1, 2026 for a biomedical engineering PhD student at UT Dallas working in the MINTS physics lab.

This is a decision brief, not a final project specification. We will select the question together, then investigate its literature, data, scope, and validation requirements in depth before producing the Claude handoff. No publication novelty or experimental performance is established by this review.

**The credit may have a near-term deadline.** If the balance is the Claude Code cloud-session promotion, Anthropic lists expiration as November 4, 2026 at 11:59 p.m. Pacific, or November 5 at 1:59 a.m. Dallas time. It applies to cloud coding sessions, rather than ordinary chat, Cowork, local sessions, or general API calls. Eligible sessions apply the bonus automatically. This is distinct from monthly API credits. Confirm the exact label and expiration in the account before choosing an execution workflow. [Official promotion](https://support.claude.com/en/articles/17152539-cloud-sessions-bonus-credit-promotion), [monthly API credits](https://support.claude.com/en/articles/17154008-monthly-api-credits-for-max-and-team-plans).

**What I reviewed.** I retrieved the public organization inventory, totaling 220 repositories, inspected ten candidate repository READMEs and file inventories, examined additional repository trees, and read code from two analysis notebooks and the sensor-failure implementation. I also consulted relevant research papers and official dataset and Claude documentation. I did not run the notebooks or execute their datasets. Dataset presence is verified; completeness, suitability, and numerical results remain to be audited.

The public work spans environmental sensing, calibration, biometrics, wearable monitoring, and population health. That gives you several credible directions without requiring a new data collection study. [MINTS GitHub](https://github.com/mi3nts), [UT Dallas overview](https://nsm.utdallas.edu/mints-sensors-2023/), [BrainHealth research profile](https://centerforbrainhealth.org/people/david-lary-phd).

## A. Can biometric exposure models generalize to a new session or person?

**Best fit:** a biomedical engineering research project closely connected to existing MINTS work.

**Question:** How well can a small set of biometric measurements estimate nearby PM2.5 concentrations when evaluation holds out entire recording sessions, days, and—if identifiable—participants?

**Existing foundation:** DUEDARE studies environmental concentrations and biometric responses. Its multi-participant repository includes a roughly 5.8 MB preprocessed dataset, a feature-ranking CSV, and a Python notebook. The earlier single-participant study and a later pollutant-estimation paper provide scientific context. The repository README describes its associated manuscript as unpublished; that status needs checking rather than treating the README as a current publication record. [Multi-participant repository](https://github.com/mi3nts/DUEDARE_multiple_participants), [2022 paper](https://www.mdpi.com/1424-8220/22/11/4240), [2024 paper](https://www.mdpi.com/2673-7426/4/2/57).

**Proposed contribution:** a reproducible comparison of random-row evaluation with evaluation on future or separate recordings, using simple baselines, a small biometric feature set, and uncertainty estimates. The notebook inspected uses `train_test_split` on rows. That motivates testing a different deployment question; it does not establish that the original results are incorrect.

**Minimum useful output:** a documented data audit; a table of performance under different splits; errors by recording; and a clear statement of the settings supported by the evidence. A negative result could still guide the lab's next study.

**Validation:** split recordings before preprocessing or feature selection; tune only within training data; preserve time boundaries and an appropriate gap between train and test; compare with a training-set mean, regularized regression, and a tree model. Use a persistence baseline only when a prior measured PM value would actually be available in the proposed deployment. Check confounding by activity, time, and environmental context when those variables exist.

**Decision gate:** identify participant and session metadata. The saved notebook preview shows a timestamp index, but I have not verified recoverable participant labels. Without participant labels, do not promise evaluation on unseen people. Distinguish concentration measured near a participant from actual inhaled or deposited dose. Prediction from observational biometrics does not establish a physiological causal mechanism.

## B. When should we trust a calibrated air-sensor reading?

**Best fit:** a physics and sensing methods project with a clearly available public starting dataset.

**Question:** When a low-cost PM sensor encounters a later time period or a different humidity regime, how do calibration error and prediction-interval coverage change?

**Existing foundation:** the lab's causal feature-selection repository contains PM1 and PM2.5 notebooks, a roughly 67.6 MB CSV, and an MIT license. The corresponding paper describes OPC-N3 calibration against a GRIMM reference. The inspected notebook parses timestamps, aggregates to one-minute measurements, and uses random-row splits for several model evaluations. [Repository](https://github.com/mi3nts/Causality-Driven-Machine-Learning), [2024 paper](https://www.mdpi.com/1424-8220/24/22/7304).

**Proposed contribution:** a careful MINTS-specific benchmark of calibration reliability over time, with a lightweight flag for readings outside the training conditions. Compare raw readings, a simple correction, and a modest tree model. Report MAE, bias, RMSE, interval coverage, and interval width across conditions.

**Validation:** separate training, tuning, interval calibration, and final evaluation periods. Fit feature selection inside training folds. Evaluate dependent time series and distribution shift explicitly: ordinary conformal prediction guarantees do not automatically hold under either. Check whether nominal intervals actually cover reference measurements on the held-out blocks.

**Decision gate:** establish the recording duration, humidity range, device identities, and usable reference overlap. If there is only one device or limited environmental diversity, restrict claims accordingly.

**Novelty caution:** uncertainty quantification for low-cost PM sensors already exists, including recent work on spatial separation and wildfire shift. Simply adding conformal intervals is not an established novel contribution. A publishable angle would need a stronger gap, dataset, comparison, or operational result. [Related 2026 study](https://www.mdpi.com/2073-4433/17/7/692).

## C. How much ECG noise makes HRV unreliable?

**Best fit:** a biomedical signal-processing tool with a public-data prototype that can run independently of lab equipment.

**Question:** How do motion and electrode artifacts affect R-peak detection and HRV estimates, and can a quality rule identify windows where reporting HRV is unreliable?

**Existing foundation:** the MINTS biometrics dashboard includes ECG and HRV processing. PhysioNet's open MIT-BIH Noise Stress Test Database supplies ECGs with known reference beat annotations and controlled noise levels. It is about 67.6 MB uncompressed. [MINTS dashboard](https://github.com/mi3nts/biometricDashboard3), [official dataset](https://physionet.org/content/nstdb/1.0.0/).

**Proposed contribution:** benchmark the lab's existing method and one established alternative, measure peak-detection and HRV error as noise increases, and generate a quality flag with an evidence-backed threshold. An optional replay interface can make the result useful during lab recordings.

**Validation:** report peak precision/recall and timing error, then error in appropriate RR/NN-derived metrics. Explain artifact handling and window length. Split by original underlying ECG record: different noise versions of the same ECG must not appear on both sides of a learning evaluation. The reference records contain arrhythmias, so beat labels and eligibility rules matter when defining normal-to-normal HRV.

**Decision gate:** the stress-test set uses only two source ECG recordings, so it is a useful controlled benchmark with limited subject diversity. Add a separate dataset or approved lab recordings before claiming broad wearable performance. This is a research-quality tool, not an established diagnostic system. Signal-quality gating is also an existing research area; the contribution needs to be defined beyond assembling standard components.

## D. Do environmental mortality models transfer to unseen places and years?

**Best fit:** a population-health and data-science direction.

**Question:** How much does county-level cardiovascular mortality prediction change when testing on geographically separate areas and later years, compared with random row splits?

**Existing foundation:** MINTS's `predicting-cvd` repository describes XGBoost, SHAP, and CAMS/ERA5 atmospheric predictors, and includes preprocessing, model, and revision notebooks plus a combined CSV of roughly 19.2 MB. Related repositories cover respiratory disease, lung cancer, and life expectancy. [CVD repository](https://github.com/mi3nts/predicting-cvd), [life-expectancy repository](https://github.com/mi3nts/Life_Expectancy_Complete).

**Proposed contribution:** a focused external-validity study: demographics-only versus environmental-plus-demographic baselines, geographic and temporal holdouts, and error analysis across counties.

**Validation:** audit outcome definitions, years, county identifiers, suppression, missingness, population weighting, and predictor availability. Keep the same county's repeated years from leaking across inappropriate splits. Account for spatial dependence. Treat SHAP as model attribution, not a causal pollution effect.

**Decision gate:** inspect the existing revision analyses before claiming any missing validation. The repository may already implement parts of this idea. County-level associations cannot establish individual clinical risk, and this is a larger epidemiological commitment than the sensor options.

## E. Can we distinguish sensor malfunction from a real pollution event?

**Best fit:** a practical engineering contribution for the lab's sensor network.

**Question:** Can a detector recognize sensor bias, noise inflation, flatlining, or dropouts quickly while maintaining a low false-alarm rate during genuine environmental changes?

**Existing foundation:** MINTS already has SAFE (`failureAnalysis`). Its README describes rolling statistical checks and physical bounds. The code inspected contains unfinished processing logic and commented-out statistical functions, so the README alone does not establish an operational baseline. [SAFE repository](https://github.com/mi3nts/failureAnalysis).

**Proposed contribution:** complete a minimal offline detector and build a reproducible benchmark against simple bounds and drift rules. Measure false alarms per sensor-day, detection delay, and missed faults. Integrate a lab-facing report only after those results exist.

**Validation:** use real environmental traces, inject clearly labeled simulated faults, and keep a final evaluation set untouched. Evaluate real maintenance events separately if logs are available. Temporal autocorrelation and repeated statistical testing affect alarm rates. A change in one sensor's distribution alone does not establish a device fault; use reference or neighboring measurements when available.

**Decision gate:** real fault labels and maintenance records determine how far the findings can go. Synthetic faults support a controlled benchmark, not proof of real-world fault detection.

## My initial recommendation

Choose **A** for the closest biomedical connection if participant/session metadata is available. Choose **B** for the clearest public-data sensing start. Choose **C** for a self-contained biomedical engineering prototype. Choose **E** for immediate lab engineering utility. Choose **D** if population health is your preferred research direction.

These are candidate contributions, not confirmed literature gaps. The next research round should examine only your top choice or two, checking recent papers, overlap with ongoing lab work, and the data decision gates above.

The lab already has a PM literature-surveillance pipeline, so a generic paper-summary assistant would need a separate, demonstrable contribution to be worthwhile. [Existing literature pipeline](https://github.com/mi3nts/Background_Research_Reports). The current portable air-pack effort is another potential user for an exposure or calibration tool. [Breathe-Easy](https://github.com/mi3nts/Breathe-Easy).

## How we will turn the choice into a Claude package

1. Confirm the credit type and deadline; select the scientific direction.
2. Define your intended result, available time, data access, and programming/compute environment.
3. Audit the selected data and current literature; specify one primary question and a bounded first milestone.
4. Prepare the complete downloadable handoff, including `START_HERE.md`, `CLAUDE.md`, the initial prompt, sequential task prompts, a research protocol, data manifest, source bibliography, environment instructions, evaluation requirements, and a budget/checkpoint plan.
5. Review Claude's first data audit together before expanding the analysis.

The handoff will tell Claude to distinguish proposals from measured results, preserve a final evaluation set, record provenance, and report negative findings. It will define a reproducible scientific output, not require the model to achieve a predetermined accuracy.

All candidates are intended to start with small CPU-based experiments. Actual runtime, cloud environment limits, and credit consumption have not been benchmarked. We will check the balance after a bounded first task rather than assume that a dollar balance guarantees a fixed number of sessions. Cloud coding credit is not a general GPU or hardware purchasing budget. [Claude cloud documentation](https://code.claude.com/docs/en/claude-code-on-the-web).

**For the next conversation round:** choose A–E (or your top two), and provide the credit's exact label and expiration. We can then narrow the question without asking you to design the whole project upfront.