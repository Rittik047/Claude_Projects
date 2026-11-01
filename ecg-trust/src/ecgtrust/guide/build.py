"""`ecgtrust guide`: build reports/guide/ecg_trust_guide.pdf from saved reports only."""

from __future__ import annotations

import datetime as dt
import json
import platform
import subprocess
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup

from ecgtrust import CLASSES
from ecgtrust.config import Config
from ecgtrust.guide import figures as F

HERE = Path(__file__).resolve().parent
PENDING = "pending"
DECISIONS = [
    (
        "PhysioNet AWS open-data mirror, every file SHA-256-verified, resumable downloads",
        "physionet.org was not reachable; the proxy cut long transfers",
    ),
    (
        "Patient-level split of fold 9 into model-selection and calibration halves",
        "conformal guarantees need calibration data unused for training or tuning",
    ),
    (
        "External label map completed to all 94 SNOMED codes; strict PTB-XL-faithful primary map, "
        "liberal map as sensitivity analysis",
        "the dataset's condition list missed 43 codes, incl. "
        "complete RBBB and LBBB; PTB-XL treats isolated ST/T findings as form statements",
    ),
    (
        "NORM proxy = sinus-family rhythms only; unlabelled external records excluded",
        "mirrors PTB-XL, where sinus bradycardia can be a normal ECG",
    ),
    (
        "1-D ResNet width 64 (1.49 M parameters), 3 seeds",
        "author's choice after a CPU timing benchmark",
    ),
    (
        "Synthetic, SNR-exact noise instead of NSTDB records",
        "NSTDB was not on the reachable mirror",
    ),
    (
        "GE 12SL map built from statement names cross-checked with PTB-XL+ SNOMED concepts; "
        "ambiguous statements unmapped",
        "keeps the commercial comparison conservative",
    ),
    ("Freeze manifest + single-use test log", "prevents tuning on the test sets"),
]


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _fmt_ci(m: dict) -> tuple[str, str]:
    if not m:
        return PENDING, PENDING
    a = m["discrimination"]["macro_auroc"]
    return f"{a['point']:.3f}", f"{a['lo']:.3f}–{a['hi']:.3f}"


def _mean(v):
    v = [x for x in v if x is not None]
    return sum(v) / len(v) if v else float("nan")


def collect(cfg: Config) -> dict:
    rep = cfg.reports
    test, ext = _load(rep / "test" / "metrics.json"), _load(rep / "external" / "metrics.json")
    val = {k: _load(rep / "val" / f"{k}.json") for k in ("baseline_12sl", "baseline_ecgdeli")}
    val = {k: v for k, v in val.items() if v}
    rs = _load(rep / "val" / "resnet_summary.json")
    if rs:
        val["resnet"] = rs
    summary = _load(rep / "data_summary.json")
    kpi = {}
    kpi["test_auroc"], kpi["test_ci"] = _fmt_ci(test)
    kpi["ext_auroc"], kpi["ext_ci"] = _fmt_ci(ext)
    kpi["ece"] = (
        (
            f"{test['calibration']['macro_ece_pre']:.3f} → "
            f"{test['calibration']['macro_ece_post']:.3f}"
        )
        if test
        else PENDING
    )
    kpi["refer"] = f"{_mean(test['conformal']['0.1']['refer_rate']):.0%}" if test else PENDING
    ctx = {
        "test": test,
        "ext_metrics": ext,
        "val": val,
        "summary": summary,
        "kpi": kpi,
        "px": summary.get("ptbxl", {}),
        "ext": summary.get("external", {}),
    }
    ctx["test_rows"] = (
        [
            (
                r["class"],
                r["n_pos"],
                f"{r['auroc']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}]",
                f"{test['discrimination']['auprc'][i]:.3f}",
                f"{test['calibration']['ece_pre'][i]:.3f} → {test['calibration']['ece_post'][i]:.3f}",
            )
            for i, r in enumerate(test["discrimination"]["per_class"])
        ]
        if test
        else []
    )
    ctx["conformal_rows"] = (
        [
            (
                a,
                f"{_mean(c['cov_pos']):.3f} / {_mean(c['cov_neg']):.3f}",
                f"{_mean(c['refer_rate']):.1%}",
                f"{_mean(c['acc_non_referred']):.3f}",
            )
            for a, c in sorted(test["conformal"].items())
        ]
        if test
        else []
    )
    ctx["summary_text"] = (
        f"On the locked PTB-XL test fold the 3-seed ResNet ensemble reaches macro-AUROC "
        f"{kpi['test_auroc']} (95% CI {kpi['test_ci']}); temperature scaling moves macro ECE "
        f"{kpi['ece']}; at α = 0.10 about {kpi['refer']} of label decisions are referred. "
        f"On the external database macro-AUROC is {kpi['ext_auroc']} (95% CI {kpi['ext_ci']})."
        if test and ext
        else "Final test and external results are pending the locked evaluation runs."
    )
    ctx["summary_rows"] = [
        ("Does it rank cases well?", "Patient-bootstrap AUROC, internal test", kpi["test_auroc"]),
        ("Are probabilities honest?", "ECE before → after temperature scaling", kpi["ece"]),
        (
            "Can it say 'I don't know'?",
            "Label-conditional conformal refer rate, α = 0.10",
            kpi["refer"],
        ),
        ("Does it transfer?", "External database, strict label map", kpi["ext_auroc"]),
        ("Is it robust to noise?", "Synthetic noise at exact SNR; refer rate vs SNR", "see §6.4"),
        ("Better than the cart software?", "Sensitivity at GE 12SL's specificity", "see §6.7"),
    ]
    if ext and test:
        drops = [
            t["auroc"] - e["auroc"]
            for t, e in zip(
                test["discrimination"]["per_class"], ext["discrimination"]["per_class"], strict=True
            )
            if t["auroc"] is not None and e["auroc"] is not None
        ]
        ctx["ext_text"] = (
            f"Per-class AUROC changes by {min(drops):+.3f} to {max(drops):+.3f} "
            "(internal minus external), and calibration error rises, as expected "
            "under label and population shift."
        )
    else:
        ctx["ext_text"] = ""
    return ctx


def build_guide(cfg: Config, author: str = "Rittik Patra") -> Path:
    out = cfg.reports / "guide"
    figdir = out / "fig"
    figdir.mkdir(parents=True, exist_ok=True)
    ctx = collect(cfg)
    rep = cfg.reports
    figs = {
        "prevalence": F.prevalence(ctx["summary"], figdir / "prevalence.png"),
        "validation": F.validation_by_fold(ctx["summary"], figdir / "validation.png"),
        "model_selection": F.model_selection(ctx["val"], figdir / "model_selection.png"),
        "subgroups": F.subgroup_forest(rep / "test" / "subgroups.csv", figdir / "subgroups.png"),
        "comparator": F.comparator(rep / "test" / "comparator.csv", figdir / "comparator.png"),
        "internal_vs_external": F.internal_vs_external(
            ctx["test"], ctx["ext_metrics"], figdir / "internal_vs_external.png"
        ),
    }
    for name in ("roc", "reliability_pre", "reliability_post", "robustness", "occlusion"):
        figs[name] = F.copy_if_exists(rep / "test" / f"{name}.png", figdir / f"{name}.png")

    def fig(name: str, caption: str) -> Markup:
        p = figs.get(name)
        if p is None:
            return Markup(f'<div class="pending">Figure pending: {caption}</div>')
        return Markup(
            f'<figure><img src="fig/{p.name}"><figcaption>{caption}</figcaption></figure>'
        )

    try:
        import torch

        torch_v = torch.__version__.split("+")[0]
    except ImportError:
        torch_v = "n/a"
    freeze = _load(cfg.artifacts / "freeze.json").get("git_commit", "not frozen yet")
    env = Environment(loader=FileSystemLoader(HERE), autoescape=True)
    html = env.get_template("template.html").render(
        **ctx,
        fig=fig,
        fmt_int=lambda v: f"{v:,}" if isinstance(v, int) else PENDING,
        date=dt.date.today().isoformat(),
        author=author,
        decisions=DECISIONS,
        arch=f"width {cfg.train['width']}, stages {cfg.train['blocks']}, kernel 7",
        freeze=str(freeze)[:12],
        py=platform.python_version(),
        torch=torch_v,
        classes=CLASSES,
    )
    (out / "guide.html").write_text(html)
    pdf = out / "ecg_trust_guide.pdf"
    subprocess.run(
        ["node", str(HERE / "render_pdf.js"), str(out / "guide.html"), str(pdf)],
        check=True,
        capture_output=True,
    )
    return pdf


def register(sub) -> None:
    p = sub.add_parser("guide", help="build the illustrated project guide PDF from reports/")

    def run(args):
        from ecgtrust.config import load_config

        print(build_guide(load_config(args.config)))

    p.set_defaults(func=run)
