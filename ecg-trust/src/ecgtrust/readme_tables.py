"""Markdown tables for the README, generated from saved reports so numbers cannot drift."""

from __future__ import annotations

import json
from pathlib import Path

ROWS = (
    ("test", "Internal test (PTB-XL fold 10), ResNet ensemble"),
    ("test_fusion", "Internal test, ResNet + 12SL fusion"),
    ("external", "External (ecg-arrhythmia, strict map)"),
)


def render_headline_table(reports_dir: Path) -> str:
    lines = [
        "| Evaluation set | n | Macro-AUROC [95% CI] | Macro ECE (before → after scaling) "
        "| Refer rate at α = 0.10 | Coverage (pos / neg) |",
        "|---|---|---|---|---|---|",
    ]
    for split, label in ROWS:
        p = Path(reports_dir) / split / "metrics.json"
        if not p.exists():
            lines.append(f"| {label} | — | not yet evaluated | — | — | — |")
            continue
        m = json.loads(p.read_text())
        a, c = m["discrimination"]["macro_auroc"], m["calibration"]
        cf = m["conformal"]["0.1"]

        def mean(v):
            v = [x for x in v if x is not None]
            return sum(v) / len(v) if v else float("nan")

        lines.append(
            f"| {label} | {m['n']} | {a['point']:.3f} [{a['lo']:.3f}, {a['hi']:.3f}] | "
            f"{c['macro_ece_pre']:.3f} → {c['macro_ece_post']:.3f} | "
            f"{mean(cf['refer_rate']):.1%} | "
            f"{mean(cf['cov_pos']):.3f} / {mean(cf['cov_neg']):.3f} |"
        )
    return "\n".join(lines)


def register(sub) -> None:
    p = sub.add_parser("readme-tables", help="print README results table from reports/")

    def run(args):
        from ecgtrust.config import load_config

        print(render_headline_table(load_config(args.config).reports))

    p.set_defaults(func=run)
