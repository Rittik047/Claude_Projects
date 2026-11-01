# Particle deposition and autonomic physiology: project choices

Research round 1, updated November 1, 2026.

**Your confirmed aims:** produce a research result that could support a paper or thesis; use Claude Code cloud-session credits; connect the project to physiological effects of particulate deposition and autonomic biometric responses; use public data initially, with requested research data or institutionally accessible data as possible extensions. Start with the actual measured MINTS size range and compare true ultrafine particles only if suitable measurements become available.

This brief narrows the earlier broad MINTS repository review. The scientific direction is still for you to choose. None of the proposed contributions has yet passed a complete novelty or data-sufficiency assessment, and no analysis has been run.

## What the research establishes so far

**There is a direct lab foundation.** The public DUEDARE and multi-participant repositories contain biometric and particle-analysis code and preprocessed data. The later repository includes a roughly 5.8 MB dataset and a Python analysis notebook. Its saved preview shows a timestamp index; participant and session mappings still need verification. The inspected notebook uses random row splits for its model evaluations, so a study of performance on separate recordings or people would address an additional deployment question. [DUEDARE](https://github.com/mi3nts/DUEDARE), [multi-participant analysis](https://github.com/mi3nts/DUEDARE_multiple_participants), [pollutant-estimation notebooks](https://github.com/mi3nts/Estimate-Inhaled-PM-and-Gases).

**Ultrafine and submicron particles require distinct measurement claims.** WHO describes UFP as particles below about 100 nm. The Fidas Frog manufacturer gives a lower detection limit around 180 nm. Therefore, its measured bins alone do not directly resolve the UFP range. PM1 and PM2.5 are also not interchangeable with UFP particle number. Any extrapolated smaller-particle distribution must be labeled as an assumption or model estimate. [WHO material](https://cdn.who.int/media/docs/default-source/air-pollution-documents/air-quality-and-health/m1_intro-ambient-ap.pdf?sfvrsn=92a910e8_3), [instrument specifications](https://www.palas.de/en/product/fidasfrog).

**Concentration, dose, and response are different quantities.** Breathing-zone concentration describes the aerosol around a person. Inhaled dose also depends on ventilation and time. Regional deposited dose additionally depends on particle properties, respiratory geometry, and deposition efficiency. Deposition modeling does not establish measured translocation into blood or other organs. The EPA MPPD model is an authoritative modeling reference; its current packaged application requires Windows, so we should not assume it can run inside a Claude cloud container. Validated exported deposition tables may be a practical bridge, subject to checking the software's terms and the model configuration. [EPA model and documentation](https://www.epa.gov/risk/epa-multiple-path-particle-dosimetry-model).

**The physiological evidence is heterogeneous.** A controlled young-adult carbon-UFP study found small, mostly nonsignificant ECG changes. A randomized crossover study in people with diabetes reported reduced high-frequency HRV and delayed heart-rate effects. A multi-study analysis examined associations over preceding hours. The analysis should allow for population, particle composition, activity, and lag differences rather than assume one universal immediate signature. [Healthy adults](https://pmc.ncbi.nlm.nih.gov/articles/PMC2867237/), [diabetes study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4110706/), [multi-study analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC6374365/).

**Deposition–HRV association is an existing idea.** A 2014 commuter study modeled deposited dose and HRV, and a 2021 airport-worker study examined lung-deposited surface area and HRV alongside noise exposure. A useful new project needs a more specific contribution than finding an association or adding a machine-learning algorithm. [Commuter study](https://doi.org/10.1016/j.scitotenv.2013.08.096), [airport-worker study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7967637/).

## Option A — Does modeled deposited dose improve prediction of autonomic responses?

**My recommended scientific direction for your stated PhD interests.**

**Primary question:** Do size-resolved, deposition-weighted exposure metrics improve prediction of a prespecified autonomic endpoint on held-out recordings compared with ambient PM mass and an activity/context baseline?

The direction is exposure → physiological response. It directly addresses physiological response while treating deposition as a model estimate. A first study can use the measured MINTS size range; a subsequent analysis can evaluate true UFP if suitable synchronized data are obtained.

**Bounded first experiment:** choose one primary endpoint—potentially log-RMSSD or SDNN if raw ECG/beat intervals and sufficient recording duration support it—and one small set of lag windows. Compare a context-only baseline with models adding ambient PM, particle-number metrics where available, and modeled regional deposition. Include both an interpretable repeated-measures model and a modest nonlinear model. Select the final endpoint and windows only after auditing the data and relevant signal-processing requirements.

**Useful result:** a reproducible comparison showing whether deposition metrics add predictive information, under which recordings or conditions, with uncertainty and sensitivity to breathing assumptions. A well-supported absence of improvement remains scientifically informative.

**What could make it a contribution:** rigorous validation across recordings or participants; credible ventilation inputs; explicit separation of exposure and physiological signals; sensitivity to particle-size truncation; and, ideally, external evaluation. Each needs a targeted literature check before any novelty claim.

**Main feasibility gates:** synchronized particle distributions and physiology; interpretable units and size-bin boundaries; participant/session identifiers; independent breathing or defensible ventilation scenarios; activity/context measurements; enough independent recordings and exposure variation. Respiratory rate alone does not supply tidal volume or fully determine minute ventilation.

## Option B — Can biometrics estimate exposure in a new person or session?

**Best if your goal is a deployable biometric exposure estimator.**

**Primary question:** How much exposure-estimation performance remains when training and evaluation are separated by participant, session, or day, and the model can flag unreliable predictions?

The direction is physiological measurements → exposure. This follows the lab's existing inverse-modeling work. It can yield a tightly scoped public-data pilot while requested datasets are pending.

**Bounded first experiment:** estimate one exposure target using a small biometric feature set; compare random-row results with valid grouped and temporal splits; assess uncertainty and simple baselines. Refit preprocessing and feature selection within the training partitions. Do not claim evaluation on unseen participants unless their identities are recoverable.

**Useful result:** an evidence-based account of where a biometric estimator transfers, where it fails, and what additional data would be needed for deployment. Accuracy at this task alone does not demonstrate deposited dose or a physiological causal mechanism.

**Main feasibility gates:** meaningful recording groups, sufficient between-person or between-session variation, and exposure values that are independently measured. A true-UFP target requires suitable UFP measurements.

## Option C — How much does missing particle-size information change the conclusion?

**Best if you want the strongest emphasis on aerosol physics and measurement uncertainty.**

**Primary question:** How sensitive are modeled dose metrics and their associations with autonomic outcomes to the instrument's lower size cutoff and uncertain breathing parameters?

**Bounded first experiment:** obtain a dataset with sufficiently resolved smaller-particle distributions, compute modeled dose using all supported sizes, then deliberately censor the distribution at an optical-instrument cutoff. Compare the resulting dose error and the stability of downstream conclusions. Evaluate plausible breathing scenarios separately. Mobility, optical, and aerodynamic diameters need justified treatment rather than being silently equated.

**Useful result:** quantified conditions under which the measured larger-particle fraction supports, weakens, or cannot support a physiological deposition claim. If only simulated distributions are available, the result is a simulation/sensitivity study, not recovery of the actual unmeasured UFP exposure.

**Main feasibility gates:** suitable size-resolved data, defensible deposition parameters, and physiological recordings for a full biological evaluation. Aerosol-only data can validate measurement and dose computations but cannot validate autonomic response.

## Data candidates and access status

| Source | What is verified | Appropriate use | What remains unknown |
|---|---|---|---|
| MINTS DUEDARE / multi-participant data | Public repositories include preprocessed data and analysis code | Initial data audit and measured-range pilot | Participant mappings, raw ECG availability, ventilation, duration, data-specific reuse terms |
| Additional MINTS raw research recordings | Potential access described by you | Stronger synchronization, artifact handling, grouping, and possibly additional instruments | Exact instruments, available channels, permissions, and metadata |
| EPA RECAP | Official catalog lists public deidentified exposure and health endpoints, including HRV | Possible independent fine-PM comparison after inspecting the files | Raw signal availability, size resolution, and compatibility with the proposed endpoint; not verified as a UFP dataset |
| 2026 HVO chamber study | Published respiratory/autonomic analysis; authors state raw supporting data will be made available | Candidate research-data request, subject to particle measurements and data-use conditions | Actual access, available aerosol channels, synchronization, and suitability for the UFP tier |

[RECAP catalog](https://catalog.data.gov/dataset/recap-dataset-subject-exposure-and-health-endpoint-blood-lipids-cardiac-and-lung-data), [HVO study and data statement](https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2026.1710755/full).

Institutional article access does not automatically grant access to a study's underlying participant data. No external researcher has been contacted and no external dataset access has been assumed.

## Design rules for whichever option you choose

- Keep the true-UFP analysis conditional on appropriate measurements; do not fill the missing range with invented observations.
- Preserve participant and recording boundaries, select features and hyperparameters within training data, and keep a final evaluation set untouched.
- Treat participants/sessions as the independent evidence units. Thousands of adjacent time points do not create thousands of independent participants.
- Check activity, temperature, respiration, time of day, and co-pollutants when available. Use a causal diagram to decide the adjustment set: respiration may be a mediator as well as a confounder, depending on the question.
- Avoid circular predictors. If HR or ECG-derived respiration determines estimated dose, test whether that construction mechanically improves prediction of HR/HRV. Prefer independent ventilation inputs and include fixed-ventilation sensitivity analyses.
- Use measured ECG/beat quality and appropriate windows for HRV. Do not interpret every HRV change as a specific sympathetic or parasympathetic mechanism.
- Describe observational results as associations or predictive findings unless the study design supports stronger causal inference. Document uncertainty in modeled dose separately from outcome uncertainty.

## What the $100 cloud credit should produce

The immediate target is a reproducible analysis repository and an audited pilot, with a scientific methods note and figures if the data support them. A complete paper may require additional data and work after the credit expires.

Anthropic's cloud-session promotion expires November 4, 2026 at 11:59 p.m. Pacific (November 5 at 1:59 a.m. Dallas time). It is applied to eligible cloud sessions and is distinct from API credit. Its documentation says there is no separate cloud-container charge. Check the account's displayed expiration and balance as the final authority for your offer. [Official promotion](https://support.claude.com/en/articles/17152539-cloud-sessions-bonus-credit-promotion).

After your choice, the Claude handoff will include an initial prompt, a short `CLAUDE.md`, sequential prompts with explicit completion criteria, a data manifest, a reproducible environment, a prespecified analysis protocol, an annotated bibliography, a draft data-request template, and a credit/checkpoint plan. The first cloud task will audit data and produce a go/no-go report before expensive or exploratory modeling.

**Next decision:** choose A, B, or C. A is my recommendation for physiological impact, B for biometric exposure estimation, and C for deposition-model validity. We will then identify the exact data and one primary endpoint before writing the final execution prompt.
