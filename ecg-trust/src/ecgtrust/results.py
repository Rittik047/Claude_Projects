"""`ecgtrust results`: assemble `reports/results.md` from the saved per-split reports.

Nothing is recomputed here: every number comes from `metrics.json`, `subgroups.csv`,
`robustness.csv`, `comparator.md`, `artifacts/freeze.json` and `reports/test_runs.log`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ecgtrust.config import Config

RUNS = (
    ("Internal test — ResNet ensemble", "test"),
    ("Internal test — ResNet + 12SL fusion", "test_fusion"),
    ("External (ecg-arrhythmia) — ResNet ensemble", "external"),
)
REFERENCE = (
    "Strodthoff et al. 2021, best PTB-XL superdiagnostic model, fold 10",
    "≈0.93",
)


def _load(rep: Path, name: str) -> dict | None:
    f = rep / name / "metrics.json"
    return json.loads(f.read_text()) if f.exists() else None


def _ci(d: dict | None) -> str:
    if not d or d.get("point") is None:
        return "n/a"
    return f"{d['point']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def _mean(v: list) -> str:
    v = [x for x in v if x is not None]
    return f"{sum(v) / len(v):.3f}" if v else "n/a"


def _headline(runs: dict) -> list[str]:
    out = ["## Headline", "", "| evaluation | n | macro-AUROC [95% CI] |", "|---|---|---|"]
    for label, name in RUNS:
        m = runs[name]
        out.append(
            f"| {label} | {m['n']} | {_ci(m['discrimination']['macro_auroc'])} |"
            if m
            else f"| {label} | – | pending |"
        )
    out += [f"| {REFERENCE[0]} | – | {REFERENCE[1]} (reference, not re-run) |", ""]
    return out


def _per_class(runs: dict) -> list[str]:
    present = [(lbl, runs[n]) for lbl, n in RUNS if runs[n]]
    if not present:
        return []
    out = ["## Per-class AUROC", "", "| class | " + " | ".join(lbl for lbl, _ in present) + " |"]
    out.append("|---|" + "---|" * len(present))
    classes = [r["class"] for r in present[0][1]["discrimination"]["per_class"]]
    for i, c in enumerate(classes):
        cells = []
        for _, m in present:
            r = m["discrimination"]["per_class"][i]
            cells.append("n/a" if r["auroc"] is None else _ci({"point": r["auroc"], **r}))
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    return out + [""]


def _calibration(runs: dict) -> list[str]:
    out = [
        "## Calibration and abstention",
        "",
        "Temperature and conformal thresholds were fit on `val_calib` only.",
        "",
        "| evaluation | macro ECE before → after | α | mean coverage (positives) "
        "| mean refer rate |",
        "|---|---|---|---|---|",
    ]
    for label, name in RUNS:
        m = runs[name]
        if not m:
            out.append(f"| {label} | pending | | | |")
            continue
        c = m["calibration"]
        for a, cv in m["conformal"].items():
            out.append(
                f"| {label} | {c['macro_ece_pre']:.3f} → {c['macro_ece_post']:.3f} | {a} | "
                f"{_mean(cv['cov_pos'])} | {_mean(cv['refer_rate'])} |"
            )
    return out + [""]


def _subgroups(rep: Path) -> list[str]:
    out = [
        "## Subgroups",
        "",
        "Descriptive only: listed where two groups of the same attribute have "
        "non-overlapping 95% CIs for a class.",
        "",
    ]
    hits = []
    for label, name in RUNS:
        f = rep / name / "subgroups.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        df = df[(df.status == "ok") & df.lo.notna()]
        for (col, cls), g in df.groupby(["column", "class"]):
            g = g.sort_values("auroc")
            lo_g, hi_g = g.iloc[0], g.iloc[-1]
            if len(g) > 1 and lo_g.hi < hi_g.lo:
                hits.append(
                    f"- {label}, {cls}, {col}: {hi_g.group} {hi_g.auroc:.3f} "
                    f"[{hi_g.lo:.3f}, {hi_g.hi:.3f}] vs {lo_g.group} {lo_g.auroc:.3f} "
                    f"[{lo_g.lo:.3f}, {lo_g.hi:.3f}]"
                )
    return out + (hits or ["- none"]) + [""]


def _robustness(rep: Path) -> list[str]:
    f = rep / "test" / "robustness.csv"
    if not f.exists():
        return ["## Robustness (internal test)", "", "pending", ""]
    df = pd.read_csv(f, dtype={"level": str})
    out = [
        "## Robustness (internal test, ResNet)",
        "",
        "| noise | SNR (dB) | macro-AUROC |",
        "|---|---|---|",
    ]
    for _, r in df[df.level.isin(["6", "0", "clean"])].iterrows():
        out.append(f"| {r.noise} | {r.level} | {r.macro_auroc:.3f} |")
    return out + ["", "Full curves: `reports/test/robustness.csv`, `robustness.png`.", ""]


def _comparator(rep: Path) -> list[str]:
    out = []
    for name in ("test", "test_fusion"):
        f = rep / name / "comparator.md"
        if f.exists():
            body = f.read_text().split("\n", 1)[1].split("Unmapped 12SL statements")[0]
            out += [f"## GE 12SL comparator ({name})", body.strip(), ""]
    return out


def build_results_report(cfg: Config) -> Path:
    rep = cfg.reports
    runs = {name: _load(rep, name) for _, name in RUNS}
    freeze_f = cfg.artifacts / "freeze.json"
    freeze = json.loads(freeze_f.read_text()) if freeze_f.exists() else {}
    log_f = rep / "test_runs.log"
    log = log_f.read_text().strip().splitlines() if log_f.exists() else []
    lines = [
        "# Results",
        "",
        "Generated by `ecgtrust results` from saved reports; nothing is recomputed.",
        "",
        *_headline(runs),
        *_per_class(runs),
        *_calibration(runs),
        *_subgroups(rep),
        *_robustness(rep),
        *_comparator(rep),
        "## Provenance",
        "",
        f"- freeze commit: `{freeze.get('git_commit', 'pending')}`",
        "- test-run log:",
        *[f"  - `{ln}`" for ln in log],
        "",
    ]
    out = rep / "results.md"
    out.write_text("\n".join(lines))
    return out


def register(sub) -> None:
    p = sub.add_parser("results", help="assemble reports/results.md from saved reports")

    def run(args):
        from ecgtrust.config import load_config

        print(build_results_report(load_config(args.config)))

    p.set_defaults(func=run)
