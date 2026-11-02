"""`ecgtrust figures`: explanatory plots for the data and every analysis.

Figures go to `reports/figures/<group>/`. Groups:

- `01_data`: PTB-XL and external cohort, labels, demographics, signals, 12SL measurements
- `02_training`: learning curves, seed ensemble, fusion weight, model comparison on validation
- `03_performance`: ROC/PR curves, per-class forest plots, score distributions, subgroups
- `04_calibration`: reliability diagrams, ECE per class, temperatures
- `05_abstention`: conformal coverage, refer rates, decision breakdown
- `06_robustness`: noise curves, lead occlusion
- `07_explainability`: integrated gradients, lead importance, embeddings, 12SL feature importance
- `08_comparator`: model vs GE 12SL at 12SL's own operating point

Model-internal attributions (integrated gradients, embeddings, permutation importance) use
`val_select` only. Test-set figures reuse the frozen predictions; nothing is tuned on them.
Every figure is skipped, with a note, if its inputs do not exist yet.
"""

from __future__ import annotations

GROUPS = (
    "01_data",
    "02_training",
    "03_performance",
    "04_calibration",
    "05_abstention",
    "06_robustness",
    "07_explainability",
    "08_comparator",
)


def register(sub) -> None:
    p = sub.add_parser("figures", help="write explanatory plots to reports/figures/")
    p.add_argument("--only", nargs="+", choices=GROUPS, default=None)

    def run(args):
        from ecgtrust.config import load_config
        from ecgtrust.figures.build import build_figures

        made = build_figures(load_config(args.config), args.only)
        print(f"{sum(len(v) for v in made.values())} figures")
        for g, files in made.items():
            print(f"  {g}: {len(files)}")

    p.set_defaults(func=run)
