"""Build the illustrated project guide: python docs/guide/build_guide.py [--out docs/guide/bmets_forecast_guide.pdf]

Real figures come from the public TSVs (docs/figures/data). Model/result figures are SYNTHETIC EXAMPLES
(docs/figures/examples) showing what each output looks like and how to read it; they are replaced by
reports/figures/* after the Juno/Colab run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.enums import TA_CENTER  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,  # noqa: E402
                                TableStyle)

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parent
BLUE = colors.HexColor("#256abf")
INK2 = colors.HexColor("#52514e")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=BLUE, fontSize=17, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=colors.HexColor("#0b0b0b"), fontSize=12.5, spaceBefore=8, spaceAfter=4)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.5, leading=13)
SM = ParagraphStyle("SM", parent=B, fontSize=8, leading=10.5, textColor=INK2)
CAP = ParagraphStyle("CAP", parent=SM, alignment=TA_CENTER, spaceAfter=8)
TITLE = ParagraphStyle("T", parent=H1, fontSize=24, leading=29, alignment=TA_CENTER)


def P(t, s=B):
    return Paragraph(t, s)


def bullets(items, s=B):
    return [Paragraph(f"&bull;&nbsp;{i}", s) for i in items]


def fig(rel: str, caption: str, w=6.2, synthetic=False):
    p = DOCS / rel
    from PIL import Image as PI
    im = PI.open(p)
    h = w * im.height / im.width
    if h > 4.6:
        w, h = w * 4.6 / h, 4.6
    tag = "<font color='#c33636'><b>SYNTHETIC EXAMPLE</b></font> &mdash; " if synthetic else ""
    return KeepTogether([Image(str(p), width=w * inch, height=h * inch), P(tag + caption, CAP)])


def table(rows, widths, header=True, fs=8):
    t = Table([[Paragraph(str(c), ParagraphStyle("c", parent=SM, fontSize=fs, textColor=colors.black)) for c in r] for r in rows],
              colWidths=[w * inch for w in widths])
    st = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c3c2b7")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
          ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        st += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#cde2fb"))]
    t.setStyle(TableStyle(st))
    return t


def pipeline_diagram(out: Path) -> Path:
    from matplotlib.patches import FancyBboxPatch
    stages = [("audit\nindex\nsplits", "#cde2fb"), ("preprocess\n1 mm, reg.,\nN4", "#cde2fb"), ("nnU-Net\nS0 / S2", "#f6d2c3"),
              ("track\nlesions", "#f6d2c3"), ("labels\nT2/T3/T4", "#f6d2c3"), ("features\n+ FM embed", "#cfeee0"),
              ("train\nP0–P5, O1", "#cfeee0"), ("evaluate\nexplain", "#fbe3b3"), ("figures\nreport", "#fbe3b3"), ("lock →\nexternal", "#e5dff7")]
    f, ax = plt.subplots(figsize=(10, 2.2)); ax.axis("off"); ax.set_xlim(0, len(stages) * 1.1); ax.set_ylim(0, 1.4)
    for i, (t, c) in enumerate(stages):
        ax.add_patch(FancyBboxPatch((i * 1.1 + 0.05, 0.35), 0.9, 0.7, boxstyle="round,pad=0.03", fc=c, ec="#898781", lw=0.8))
        ax.text(i * 1.1 + 0.5, 0.7, t, ha="center", va="center", fontsize=7.2)
        if i:
            ax.annotate("", (i * 1.1 + 0.05, 0.7), (i * 1.1 - 0.15, 0.7), arrowprops=dict(arrowstyle="->", color="#52514e"))
    for x, lab in [(0.5, "data (CPU)"), (3.3, "segmentation & labels (GPU)"), (6.6, "modelling (GPU)"), (9.4, "reporting")]:
        ax.text(x * 1.1 / 1.1, 1.22, lab, ha="center", fontsize=8.5, color="#52514e", fontweight="bold")
    f.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb"); plt.close(f)
    return out


def tracking_diagram(out: Path) -> Path:
    import numpy as np
    f, axs = plt.subplots(1, 3, figsize=(8.5, 2.6))
    pts = [[(30, 40, 9), (70, 60, 6)], [(31, 41, 6), (71, 59, 9)], [(70, 60, 12), (40, 80, 5)]]
    for k, (ax, p) in enumerate(zip(axs, pts)):
        ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.add_patch(plt.Circle((50, 50), 47, fc="#f0efec", ec="#c3c2b7"))
        for (x, y, r), col in zip(p, ["#2a78d6", "#eb6834"] if k < 2 else ["#eb6834", "#e34948"]):
            ax.add_patch(plt.Circle((x, y), r, fc=col, alpha=0.75))
        ax.set_title(f"post-dx MRI t{k + 1}", fontsize=9)
    axs[1].text(50, 4, "A: −60% vol · B: +240% → PD", ha="center", fontsize=7.5)
    axs[2].text(50, 4, "A resolved (CR) · C new → T2b = 1", ha="center", fontsize=7.5)
    f.suptitle("Lesion tracking (Hungarian matching, 10 mm gate) and label derivation", fontsize=10)
    f.savefig(out, dpi=200, bbox_inches="tight", facecolor="#fcfcfb"); plt.close(f)
    return out


def build(out: Path):
    audit = json.loads((ROOT / "reports" / "audit.json").read_text())
    gd = DOCS / "guide"
    pipe = pipeline_diagram(gd / "pipeline.png"); trk = tracking_diagram(gd / "tracking.png")
    s = []
    # ---- 1 title
    s += [Spacer(1, 1.3 * inch), P("bmets-forecast", TITLE),
          P("Forecasting intracranial disease evolution in brain metastases from longitudinal multiparametric MRI with 3D foundation models", ParagraphStyle("st", parent=H2, alignment=TA_CENTER)),
          Spacer(1, 0.3 * inch), P("Project guide &mdash; science, data, methods, how to run (Juno &amp; Colab), how to read every figure", ParagraphStyle("c", parent=B, alignment=TA_CENTER)),
          Spacer(1, 0.2 * inch), P("Rittik Patra &middot; Department of Bioengineering, UT Dallas &middot; November 2026", ParagraphStyle("c2", parent=SM, alignment=TA_CENTER)),
          Spacer(1, 0.5 * inch), Image(str(pipe), width=7 * inch, height=1.55 * inch),
          Spacer(1, 0.3 * inch), P("Real figures in this guide are computed from the public UCSD-BMETS-Longitudinal TSVs. Figures labelled "
                                   "<font color='#c33636'><b>SYNTHETIC EXAMPLE</b></font> are rendered by the same code on synthetic inputs to show what each "
                                   "output looks like; the Juno/Colab run regenerates them from the real data in <i>reports/figures/</i>.", SM), PageBreak()]
    # ---- 2 at a glance
    s += [P("1. Project at a glance", H1),
          P("<b>Clinical problem.</b> Patients with brain metastases are followed by MRI every ~2&ndash;3 months. The decisions that matter &mdash; salvage radiosurgery, whole-brain radiotherapy, "
            "changing systemic therapy, shortening the surveillance interval &mdash; depend on whether the disease will progress or seed <i>new</i> metastases before the next scan. "
            "Today this is judged from clinical scores (GPA) and visual comparison."),
          P("<b>Question.</b> Can a model that sees the patient's MRI history forecast the next scan &mdash; and does a 3D imaging <i>foundation model</i> add information beyond lesion burden and radiomics?"),
          table([["Task", "Unit", "Label", "Primary metric"],
                 ["T1 segmentation + tracking", "time point", "voxel labels, lesion correspondence", "lesion-wise Dice / F1, sensitivity by size"],
                 ["<b>T2a progression (primary)</b>", "interval t→t+1", "ΣET ≥ +73 % vs nadir or new lesion ≥ 0.05 cc", "AUROC (patient bootstrap CI)"],
                 ["<b>T2b new metastases</b>", "interval", "≥ 1 new enhancing lesion ≥ 0.05 cc", "AUROC"],
                 ["T3 lesion trajectory", "lesion × interval", "log volume ratio; CR/PR/SD/PD", "MAE, Spearman ρ"],
                 ["T4 overall survival", "patient", "death (153 events), censored at last MRI", "C-index, td-AUC"]], [1.7, 1.0, 2.4, 1.9]),
          Spacer(1, 6), P("<b>Why this is publishable now.</b>", H2)]
    s += bullets(["The dataset was released on TCIA in October 2026; longitudinal, voxel-annotated brain-metastasis MRI was essentially unavailable before 2025.",
                  "Prior longitudinal response work used small single-centre cohorts and hand-crafted features (Kuhn et al., MICCAI-LMID 2025, arXiv:2509.06396).",
                  "Foundation models (BrainSegFounder, SAM-Med3D) have been benchmarked for segmentation, not for forecasting disease evolution.",
                  "Design choices that reviewers at npj Digital Medicine expect: pre-registered analysis plan (hashed), patient-level CV, clustered bootstrap, Holm correction, calibration, decision curves, subgroup fairness, locked external validation, TRIPOD+AI/CLAIM."])
    s += [P("<b>Portfolio value.</b> End-to-end medical-imaging ML: 3D segmentation (nnU-Net), image registration, foundation-model transfer and LoRA, temporal transformers, survival analysis, explainability, HPC (SLURM) and cloud (Colab) execution, reproducible software with ~100 tests and CI.", B), PageBreak()]
    # ---- 3 background
    s += [P("2. Scientific background", H1), P("2.1 Brain metastases and response assessment", H2),
          P("Brain metastases occur in roughly 20&ndash;40 % of patients with solid cancers (lung, breast and melanoma most often). Stereotactic radiosurgery (SRS) controls treated lesions in most cases, "
            "but <i>distant brain failure</i> &mdash; new lesions elsewhere &mdash; occurs in a large fraction of patients within a year and drives salvage treatment. "
            "Response is assessed with RANO-BM (Lin et al., Lancet Oncol 2015): progression is a ≥ 20 % increase in the sum of longest diameters or a new lesion. "
            "Volumetric criteria are more sensitive; a 20 % diameter increase of a sphere corresponds to (1.2)<super>3</super> − 1 ≈ +73 % volume and a 30 % diameter decrease to "
            "1 − 0.7<super>3</super> ≈ −65 % volume, which are the thresholds used here."),
          P("2.2 Prognostic tools", H2),
          P("The Graded Prognostic Assessment (Sperduto et al., JCO 2020) predicts survival from age, performance status, extracranial disease, number of metastases and molecular markers; "
            "brain-metastasis velocity (new lesions per year; Farris et al., IJROBP 2017) predicts survival after distant brain failure. Both are clinical summaries; neither uses the image content. "
            "Our P0 model is a BMV-like clinical-burden baseline built from the segmentations."),
          P("2.3 Foundation models in 3D medical imaging", H2),
          P("Foundation models are large encoders pretrained on many unlabelled or weakly labelled scans that transfer to new tasks with little labelled data. "
            "BrainSegFounder (Cox et al., Med Image Anal 2024) pretrains a SwinUNETR on UK Biobank and BraTS brain MRI; SAM-Med3D (Wang et al., 2023) adapts the Segment-Anything ViT to 3D. "
            "We use them as frozen feature extractors (linear probe, P2), as inputs to a temporal transformer that reads the MRI history (P3), and optionally fine-tune with LoRA (low-rank adapters; Hu et al., 2022)."),
          P("2.4 Why longitudinal modelling", H2),
          P("A single scan shows the current burden; the <i>change</i> between scans (growth rate, new-lesion appearance, oedema evolution) carries information about tumour biology and treatment response. "
            "The temporal transformer encodes each time point with its time since the first MRI (sinusoidal features of years) and attends over the history; a CLS token summarises it for prediction."),
          PageBreak()]
    # ---- 4 dataset
    s += [P("3. The dataset: UCSD-BMETS-Longitudinal", H1),
          P(f"300 adult patients with intracranial metastases imaged at UC San Diego 2016&ndash;2023; 2&ndash;4 post-diagnosis MRIs each, a normal pre-diagnosis MRI for {audit['n_normal_mri']} patients. "
            "Every time point has t1c, t1n, T2-FLAIR (t2f) and T2 (t2w); most also have T1-post FSE, DWI/ADC, susceptibility (SWAN) and ASL. "
            "Images are NIfTI, skull-stripped with HD-BET; segmentations are neuroradiologist-approved. Licence CC BY 4.0 (cite DOI 10.7937/5ejb-9g94)."),
          table([["Audit item (reports/audit.json)", "Value"],
                 ["patients / MRI time points", f"{audit['n_patients']} / {audit['n_timepoints']}"],
                 ["post-dx MRIs per patient (2/3/4)", " / ".join(str(audit['post_mris_per_patient'].get(str(k), audit['post_mris_per_patient'].get(k, 0))) for k in (2, 3, 4))],
                 ["intervals t→t+1 (T2 units)", f"{audit['n_intervals']} (median {audit['interval_years_median'] * 12:.1f} months; {audit['n_zero_intervals']} recorded as 0.0 y)"],
                 ["intervals with cranial radiation", f"{audit['n_rt_intervals']} ({', '.join(f'{k}: {v}' for k, v in audit['rt_by_interval'].items())})"],
                 ["resection before first MRI", str(audit["n_resection_before"])],
                 ["deaths (T4 events)", f"{audit['n_deaths']} (median {audit['median_years_first_mri_to_death']:.1f} y after first MRI)"],
                 ["field strength", ", ".join(f"{k} T: {v}" for k, v in audit["field_strength"].items())]], [3.0, 3.8]),
          Spacer(1, 6), fig("figures/data/cohort_flow.png", "Fig. 3.1 Cohort and analysis units.", 5.0), PageBreak(),
          fig("figures/data/primary_bars.png", "Fig. 3.2 Primary malignancy (7 groups; colour follows the entity throughout the project).", 4.6),
          fig("figures/data/timepoints_per_patient.png", "Fig. 3.3 Longitudinal depth: 208 patients have four post-diagnosis MRIs.", 3.8),
          fig("figures/data/age_by_primary.png", "Fig. 3.4 Age at first post-diagnosis MRI by primary tumour.", 4.8), PageBreak(),
          fig("figures/data/interval_hist.png", "Fig. 3.5 Scan intervals. Ages are given to 0.1 y (~5 weeks), so intervals are quantised; 60 consecutive scans share the same recorded age "
              "(likely planning MRIs shortly after the diagnostic MRI). They are kept and flagged (zero_interval) and removed in a sensitivity analysis.", 5.4),
          fig("figures/data/rt_by_interval.png", "Fig. 3.6 Radiation between consecutive MRIs. Radiation is a predictor known at decision time; it is <b>not</b> analysed as a treatment effect (confounding by indication).", 4.0),
          PageBreak(),
          fig("figures/data/swimmer.png", "Fig. 3.7 Swimmer plot: each line is a patient (sorted by follow-up); dots are MRIs (orange = after an interval with radiation), crosses are deaths.", 5.2),
          PageBreak(),
          fig("figures/data/sequence_availability.png", "Fig. 3.8 Sequence availability by folder index (% of studies). Missing optional sequences are fed as zero channels plus a mask bit.", 5.0),
          fig("figures/data/scanner_mosaic.png", "Fig. 3.9 Scanner models and field strength: 97 % GE; 3 T 60 % / 1.5 T 40 %. Field strength is a pre-specified subgroup.", 5.4),
          PageBreak(),
          fig("figures/data/km_by_primary.png", "Fig. 3.10 Overall survival by primary (Kaplan&ndash;Meier; censored at last MRI because last contact is not in the table).", 5.0),
          fig("figures/data/followup_to_death.png", "Fig. 3.11 Time from first post-diagnosis MRI to death (153 patients).", 4.4), PageBreak()]
    # ---- 5 labels
    s += [P("4. From segmentations to outcomes", H1), Image(str(trk), width=6.8 * inch, height=2.1 * inch), Spacer(1, 4)]
    s += bullets(["<b>Registration.</b> Every time point is rigid+affine registered to the patient's baseline t1c (1 mm grid), so lesions keep their coordinates across scans.",
                  "<b>Lesions.</b> 26-connected components of enhancing tumour (ET) ∪ necrotic core (NETC); resection cavities excluded. Volume, centroid, oedema (SNFH) volume, diameter.",
                  "<b>Tracking.</b> Hungarian assignment on centroid distance to each track's last known position, gate 10 mm. Status: baseline, stable, new, resolved, recurred (re-growth at a resolved site is <i>not</i> a new metastasis).",
                  "<b>T2a</b> = progression at t+1: ΣET(t+1) ≥ 1.73 × nadir, or a new lesion ≥ 0.05 cc. <b>T2b</b> = new lesion ≥ 0.05 cc (distant brain failure).",
                  "<b>T3</b> per lesion: log((V<sub>t+1</sub>+0.001)/(V<sub>t</sub>+0.001)) and CR (0 cc) / PR (≤ −65 % vs first appearance) / PD (≥ +73 % vs nadir) / SD.",
                  "<b>T4</b> overall survival: death age − first-MRI age; alive patients censored at last MRI."])
    s += [fig("figures/examples/tracking_flow.png", "Fig. 4.1 Lesion status per time point (output of <i>bmf track</i>).", 4.4, True),
          fig("figures/examples/montage.png", "Fig. 4.2 Four core sequences with the label overlay (NETC yellow, SNFH blue, ET red).", 6.4, True), PageBreak()]
    # ---- 6 models
    s += [P("5. Models", H1),
          table([["ID", "Inputs", "Model", "Role"],
                 ["P0", "clinical + lesion burden (count, log ΣET, Δ, BMV-like new/yr, RT, interval)", "logistic regression", "reference (clinical-burden baseline)"],
                 ["P1", "P0 + PyRadiomics (t1c, t2f, ADC; TC & WT)", "LightGBM", "hand-crafted imaging"],
                 ["P2", "P0 + FM embedding at t (brain ⊕ mean lesion)", "L2 logistic (inner-CV C)", "foundation-model linear probe"],
                 ["P3", "sequence of FM embeddings t<sub>1</sub>…t + P0 token", "temporal transformer (2×4 heads, d=128)", "longitudinal deep model"],
                 ["P4", "P3 with LoRA-adapted encoder", "LoRA r=8 on qkv/proj", "optional (compute permitting)"],
                 ["P5", "out-of-fold logits of P0, P1, P3", "nested stacking", "complementarity"],
                 ["O1", "baseline clinical + burden (+ embeddings)", "penalised Cox, DeepSurv", "T4 survival"],
                 ["S0/S2", "4 sequences (+ prior t1c & mask)", "nnU-Net v2 3d_fullres", "T1 segmentation"]], [0.5, 2.6, 1.8, 2.0]),
          P("5.1 Foundation-model encoders", H2),
          table([["Encoder", "Pretraining", "Input", "Embedding"],
                 ["BrainSegFounder", "SwinUNETR, UK Biobank + BraTS (Cox et al. 2024)", "4 ch, 96³ patches", "768-d deepest Swin stage, pooled"],
                 ["SAM-Med3D", "3D SAM ViT-B (Wang et al. 2023)", "t1c, 128³", "384-d"],
                 ["nnU-Net encoder", "our S0 (supervised, task-specific)", "4 ch", "320-d bottleneck"],
                 ["random3d", "none (ablation)", "4 ch, 64³", "128-d"]], [1.3, 2.6, 1.3, 1.7]),
          P("Whole-brain embedding = mean over non-overlapping patches with ≥ 10 % brain; lesion embedding = patch centred on each lesion; per time point the model receives [brain ⊕ mean-lesion]. "
            "The random encoder answers the reviewer question \"is it the pretraining or just the architecture?\".", B),
          P("5.2 Leakage guards (tested)", H2)]
    s += bullets(["Patient-level folds (all intervals and lesions of a patient in one fold) &mdash; <i>test_run_cv_no_patient_leak</i>.",
                  "Row t uses only information up to t (plus planned RT and interval length) &mdash; <i>test_design_no_future_leak</i>.",
                  "Scalers, imputers, PCA, probes and stackers fit inside training folds; P5 stacker for fold k never sees fold-k labels &mdash; <i>test_stack_uses_only_train_oof</i>.",
                  "External cohorts run once against hash-locked models; a second run is refused &mdash; <i>test_lock_then_external</i>."])
    s += [PageBreak()]
    # ---- 7 statistics
    s += [P("6. Validation and statistics (pre-specified)", H1),
          P("The analysis plan (docs/analysis_plan.md) was frozen on 2 November 2026 before any image was accessed; its SHA-256 is committed and checked by a test. "
            "Primary endpoint: AUROC for T2a with 95 % CI from 2,000 patient-clustered bootstrap resamples. Pre-specified contrasts P1, P2, P3, P5 vs P0 (paired bootstrap ΔAUROC, Holm). "
            "Secondary: AUPRC, calibration slope/intercept and ECE, decision curves (net benefit across threshold probabilities 0.05&ndash;0.60), subgroups (primary, sex, radiation, field strength; reported if n ≥ 10 with both classes), "
            "T3 MAE/Spearman, T4 C-index and IPCW time-dependent AUC at 6 and 12 months. Sensitivity: drop zero intervals, drop the radiation predictor, automated-segmentation labels, 1.5 T vs 3 T, new-lesion threshold 0.1 cc."),
          fig("figures/examples/roc.png", "Fig. 6.1 ROC curves. Colour follows the model (P0 blue, P1 orange, P3 yellow…). Read: higher-left is better; the legend gives AUROC.", 3.4, True),
          fig("figures/examples/delta_forest.png", "Fig. 6.2 Incremental discrimination vs P0 with 95 % CI and Holm-adjusted p. A CI that excludes 0 = significant gain.", 5.0, True), PageBreak(),
          fig("figures/examples/calibration.png", "Fig. 6.3 Calibration: points on the diagonal = predicted risk equals observed frequency; slope < 1 means over-confident predictions.", 3.4, True),
          fig("figures/examples/decision_curve.png", "Fig. 6.4 Decision curve: net benefit of acting on the model (e.g., earlier MRI) vs treating everyone or no one, across risk thresholds.", 4.6, True),
          fig("figures/examples/pr.png", "Fig. 6.5 Precision&ndash;recall: more informative than ROC when events are rare; dashed line = prevalence.", 3.4, True), PageBreak(),
          fig("figures/examples/subgroup_forest.png", "Fig. 6.6 Subgroup AUROC with CIs (fairness / robustness).", 5.2, True),
          fig("figures/examples/bootstrap_dist.png", "Fig. 6.7 Bootstrap distributions of AUROC (uncertainty).", 4.4, True),
          fig("figures/examples/risk_distribution.png", "Fig. 6.8 Predicted risk by outcome: good separation = little overlap.", 4.4, True), PageBreak()]
    # ---- 8 explainability
    s += [P("7. Explainability: how to read each figure", H1),
          fig("figures/examples/shap_beeswarm.png", "Fig. 7.1 SHAP beeswarm: each dot is an interval; x = contribution to the log-odds of progression; colour = feature value (blue low, red high). "
              "Features sorted by mean |SHAP|. E.g., red dots on the right for d_sum_et = rising burden increases risk.", 5.2, True),
          fig("figures/examples/shap_bar.png", "Fig. 7.2 Global importance (mean |SHAP|).", 4.4, True), PageBreak(),
          fig("figures/examples/shap_dependence.png", "Fig. 7.3 SHAP dependence: how one feature's value maps to its contribution (non-linearity, thresholds).", 6.6, True),
          fig("figures/examples/shap_interaction.png", "Fig. 7.4 SHAP interaction strength between top features (TreeExplainer interaction values).", 4.0, True),
          fig("figures/examples/sequence_ablation.png", "Fig. 7.5 Sequence ablation: AUROC drop when a sequence is removed = its unique information.", 4.0, True), PageBreak(),
          fig("figures/examples/saliency_overlay.png", "Fig. 7.6 Integrated gradients on t1c (top decile of attribution) with the ET contour; IG satisfies completeness (attributions sum to f(x) − f(baseline); tested).", 5.0, True),
          fig("figures/examples/attention_time.png", "Fig. 7.7 Temporal attention: how much the transformer weights each earlier time point.", 4.2, True),
          fig("figures/examples/embedding_umap.png", "Fig. 7.8 UMAP/t-SNE of foundation-model embeddings coloured by primary (or outcome/scanner): checks what the embedding encodes and whether scanner effects dominate.", 3.6, True), PageBreak()]
    # ---- 9 T1, T3, T4 outputs
    s += [P("8. Segmentation, trajectory and survival outputs", H1),
          fig("figures/examples/seg_dice_by_size.png", "Fig. 8.1 Detection sensitivity by lesion size: small (≤ 3 mm) metastases are the clinical failure mode; S2 uses the prior scan to find them.", 4.4, True),
          fig("figures/examples/seg_method_compare.png", "Fig. 8.2 Lesion-wise Dice per method (BraTS-METS protocol).", 3.8, True),
          fig("figures/examples/fn_gallery.png", "Fig. 8.3 Missed lesions gallery (failure analysis).", 4.6, True), PageBreak(),
          fig("figures/examples/spaghetti.png", "Fig. 8.4 Per-lesion volume trajectories by primary; dashed lines at +73 % / −65 %.", 6.4, True),
          fig("figures/examples/trajectory_clusters.png", "Fig. 8.5 Growth-trajectory clusters (median and IQR).", 4.4, True),
          fig("figures/examples/waterfall.png", "Fig. 8.6 Waterfall of best volumetric response per lesion coloured by CR/PR/SD/PD.", 5.6, True), PageBreak(),
          fig("figures/examples/rano_class_bars.png", "Fig. 8.7 Next-scan lesion response distribution by primary.", 4.6, True),
          fig("figures/examples/pred_vs_obs.png", "Fig. 8.8 T3 predicted vs observed log volume ratio.", 3.4, True),
          fig("figures/examples/km_risk_tertiles.png", "Fig. 8.9 T4 Kaplan&ndash;Meier by predicted-risk tertile (log-rank p).", 4.2, True),
          fig("figures/examples/td_auc.png", "Fig. 8.10 Time-dependent AUC (IPCW) at 6 and 12 months.", 3.8, True),
          fig("figures/examples/lesion_count_volume.png", "Fig. 8.11 Lesion count per time point and lesion size distribution.", 6.0, True),
          fig("figures/examples/seg_by_field_strength.png", "Fig. 8.12 Segmentation quality by field strength.", 4.0, True), PageBreak()]
    # ---- 10 running
    s += [P("9. How to run", H1), P("9.1 Route 1 &mdash; UTD Juno (recommended for heavy stages)", H2)]
    s += bullets(["Clone the repo on Juno; edit <i>juno/00_env.sh</i> (partitions from <i>sinfo -s</i>, scratch path); <i>source juno/00_env.sh</i> creates the conda env; run <i>pytest -q</i>.",
                  "Download with Aspera: <i>bash juno/01_aspera_pull.sh \"&lt;Faspex link&gt;\"</i> (or Aspera Connect on a laptop + rsync). Copy the 4 TSVs to <i>$BMF_SCRATCH/clinical/</i>.",
                  "Check the label values of the first segmentation (expected {0,1,2,3}[,4]).",
                  "Put BrainSegFounder and SAM-Med3D checkpoints in <i>$BMF_SCRATCH/weights/</i> (names in configs/juno.yaml).",
                  "Smoke test first: <i>sbatch juno/00b_smoke.sbatch</i> runs every CPU stage on 12 seeded patients (&ge;3 MRIs each), writes only <i>cache_smoke/</i> and <i>reports_smoke/</i>, and ends with SMOKE OK.",
                  "<i>bash juno/run_all.sh</i> submits preprocess → seg-export → nnU-Net (10 GPU jobs) → seg-eval, and embed → train/evaluate → figures/report, chained with SLURM dependencies.",
                  "Commit only <i>reports/results.json</i>, <i>reports/tables/</i>, <i>reports/figures/</i>."])
    s += [P("9.2 Route 2 &mdash; Google Colab Pro (A100) + Drive", H2)]
    s += bullets(["Drive folder <i>MyDrive/bmets/</i> with the data, the TSVs and weights; notebooks 00&ndash;05 in <i>colab/</i> (00 = 12-patient smoke test on a CPU runtime), each starting with the same setup cell (mount, clone, install).",
                  "Raw + preprocessed data need ~350 GB of Drive; otherwise preprocess on Juno and copy <i>cache/pp</i> to Drive.",
                  "nnU-Net: one fold per session, resumable with <i>--c</i>; preprocessing in 8 resumable shards."])
    s += [P("9.3 Compute budget (approximate)", H2),
          table([["Stage", "Hardware", "Time"], ["preprocessing (1,065 studies)", "CPU array (32 tasks)", "2–4 h"],
                 ["nnU-Net S0 + S2, 5 folds each", "10 × A100", "12–24 h per job"], ["FM embeddings (3 encoders)", "A100", "3–6 h"],
                 ["P0–P5, T3, T4, explain", "A100 / CPU", "2–4 h"], ["external: Yale (~12k studies)", "CPU + A100", "1–2 days"]], [2.6, 2.0, 1.8]),
          P("9.4 Command reference", H2),
          P("<font face='Courier' size='8'>bmf audit | index | splits | preprocess [--shard i/n] | seg-export | seg-eval --pred-dir D --method S0 | track | labels | features | "
            "embed --encoder E --device cuda | train --model P0..P5|O1 --task T2a|T2b|T3|T4 | evaluate | explain --task T2a | figures | report | lock --task T2a | external --task T2a | "
            "to-layout --map map.csv --out DIR; any stage + --subset N = smoke run</font>", B), PageBreak()]
    # ---- 11 external, reporting, timeline
    s += [P("10. External validation, reporting and timeline", H1), P("10.1 External cohorts (locked, single run)", H2),
          table([["Cohort", "What it tests", "Notes"],
                 ["BraTS-METS 2025 (non-UCSD sites)", "T1 segmentation", "Synapse registration; exclude UCSD cases (overlap with this dataset)"],
                 ["PROTEAS (Sci Data 2025; 40 pts, 744 MRIs)", "T2/T3 longitudinal", "Zenodo; segmentations in BraTS space"],
                 ["Yale-Brain-Mets-Longitudinal (1,430 pts)", "T2 (+T4 if available)", "no segmentations → S2 automated labels, 50-pair manual check"]], [2.4, 1.6, 2.8]),
          P("10.2 Reporting", H2),
          P("TRIPOD+AI and CLAIM 2024 checklists map every item to a manuscript section (docs/). The manuscript skeleton (docs/manuscript/manuscript.md) contains placeholders such as "
            "{{T2a.models.P3.auroc}} that <i>fill_manuscript.py</i> replaces from reports/results.json, so numbers in the paper always match the run."),
          P("10.3 Suggested timeline", H2),
          table([["Week", "Milestone"], ["1", "Aspera download; preprocess; label-scheme check; nnU-Net S0/S2 training"],
                 ["2", "tracking, labels, audit of event counts (T2a/T2b); embeddings; P0–P5"],
                 ["3", "explainability; sensitivity analyses; figures; draft Results"],
                 ["4", "external validation (PROTEAS, BraTS-METS, Yale); finalize Discussion"],
                 ["5", "internal review (advisor), preprint (medRxiv), submit to npj Digital Medicine"]], [0.8, 6.0]),
          P("10.4 Risks and mitigations", H2)]
    s += bullets(["<b>Few T2b events</b> &rarr; T2a headline; report event counts; regularised models.",
                  "<b>Coarse timing</b> (0.1 y) &rarr; interval as covariate; sensitivity without zero intervals.",
                  "<b>Tracking errors</b> &rarr; manual QC of 50 intervals; tracking accuracy reported.",
                  "<b>Scanner shift</b> &rarr; field-strength subgroup; external cohorts.",
                  "<b>Confounding by treatment</b> &rarr; predictive, not causal, framing."])
    s += [PageBreak(), P("11. References", H1)]
    refs = ["Correia de Verdier M, et al. UCSD-BMETS-Longitudinal (v1). The Cancer Imaging Archive, 2026. doi:10.7937/5ejb-9g94",
            "Lin NU, et al. Response assessment criteria for brain metastases: proposal from the RANO group. Lancet Oncol 2015;16:e270–e278.",
            "Isensee F, et al. nnU-Net. Nat Methods 2021;18:203–211.",
            "Cox J, et al. BrainSegFounder: towards 3D foundation models for neuroimage segmentation. Med Image Anal 2024;97:103301.",
            "Wang H, et al. SAM-Med3D. arXiv:2310.15161, 2023.",
            "Farris M, et al. Brain metastasis velocity. Int J Radiat Oncol Biol Phys 2017;98:131–141.",
            "Sperduto PW, et al. Updated diagnosis-specific GPA. J Clin Oncol 2020;38:3773–3784.",
            "Chadha S, et al. An 11,000-study open-access dataset of longitudinal MRI of brain metastases. arXiv:2506.14021, 2025.",
            "Kuhn L, et al. AI-based response assessment and prediction in longitudinal imaging for brain metastases treated with SRS. arXiv:2509.06396, 2025.",
            "BraTS-METS 2025 Lighthouse Challenge analysis. arXiv:2504.12527, 2025.",
            "Collins GS, et al. TRIPOD+AI statement. BMJ 2024;385:e078378.",
            "Lundberg SM, Lee SI. A unified approach to interpreting model predictions. NeurIPS 2017.",
            "Sundararajan M, Taly A, Yan Q. Axiomatic attribution for deep networks. ICML 2017.",
            "Selvaraju RR, et al. Grad-CAM. ICCV 2017.",
            "Vickers AJ, Elkin EB. Decision curve analysis. Med Decis Making 2006;26:565–574.",
            "Uno H, et al. Evaluating prediction rules for t-year survivors with censored regression models. JASA 2007;102:527–537.",
            "Hu EJ, et al. LoRA: low-rank adaptation of large language models. ICLR 2022.",
            "van Griethuysen JJM, et al. Computational radiomics system to decode the radiographic phenotype. Cancer Res 2017;77:e104–e107.",
            "Katzman JL, et al. DeepSurv. BMC Med Res Methodol 2018;18:24."]
    s += [P(f"[{i}] {r}", SM) for i, r in enumerate(refs, 1)]
    s += [Spacer(1, 8), P("Bibliographic details should be re-verified in a reference manager before submission.", SM)]

    def foot(canvas, doc):
        canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(INK2)
        canvas.drawString(0.7 * inch, 0.45 * inch, "bmets-forecast · project guide")
        canvas.drawRightString(7.8 * inch, 0.45 * inch, f"{doc.page}"); canvas.restoreState()

    SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.7 * inch, rightMargin=0.7 * inch, topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                      title="bmets-forecast guide", author="Rittik Patra").build(s, onFirstPage=foot, onLaterPages=foot)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DOCS / "guide" / "bmets_forecast_guide.pdf"))
    print(build(Path(ap.parse_args().out)))
