"""GE 12SL comparator at its own operating point (spec §4.4).

12SL produces binary statements only, so the comparison is: 12SL sensitivity/specificity
versus the model's sensitivity at 12SL's specificity, on the same records. Statements are
mapped through the reviewed table `configs/12sl_label_map.csv`; unmapped statements are
listed in the report. NORM is positive only if no other superclass is asserted.
"""

from __future__ import annotations

import ast
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from ecgtrust import CLASSES
from ecgtrust.evaluate.metrics import bootstrap_samples

MAP_PATH = Path(__file__).resolve().parents[3] / "configs" / "12sl_label_map.csv"


def _load_map(path: Path) -> dict[str, str]:
    with open(path, newline="") as f:
        return {r["statement"]: r["superclass"] for r in csv.DictReader(f)}


def _statements(root: Path) -> pd.Series:
    s = pd.read_csv(Path(root) / "labels" / "12sl_statements.csv", index_col="ecg_id")
    return s["statements"].apply(ast.literal_eval)


def load_12sl_decisions(root: Path, label_map: Path = MAP_PATH) -> pd.DataFrame:
    m = _load_map(label_map)
    rows = []
    for stmts in _statements(root):
        hit = {m[s] for s in stmts if s in m}
        row = {c: c in hit for c in CLASSES}
        row["NORM"] = "NORM" in hit and not (hit - {"NORM"})
        rows.append(row)
    return pd.DataFrame(rows, index=_statements(root).index, columns=list(CLASSES))


def operating_point(y: np.ndarray, d: np.ndarray) -> tuple[float, float]:
    y, d = np.asarray(y).astype(bool), np.asarray(d).astype(bool)
    return float((d & y).sum() / y.sum()), float((~d & ~y).sum() / (~y).sum())


def sens_at_spec(y: np.ndarray, p: np.ndarray, spec: float) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    ok = (1 - fpr) >= spec - 1e-12
    return float(tpr[ok].max()) if ok.any() else 0.0


def comparator_table(y, p, d, groups, n_boot: int = 2000) -> pd.DataFrame:
    y, p, d = np.asarray(y), np.asarray(p), np.asarray(d)
    rows = []
    for c, name in enumerate(CLASSES):
        sens, spec = operating_point(y[:, c], d[:, c])
        model = sens_at_spec(y[:, c], p[:, c], spec)
        both = np.stack([p[:, c], d[:, c]], axis=1)

        def diff(yy, bb):
            s12, sp12 = operating_point(yy, bb[:, 1])
            return sens_at_spec(yy, bb[:, 0], sp12) - s12

        bs = bootstrap_samples(y[:, c], both, groups, diff, n=n_boot)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rows.append(
            {
                "class": name,
                "n_pos": int(y[:, c].sum()),
                "12sl_sens": sens,
                "12sl_spec": spec,
                "model_sens_at_12sl_spec": model,
                "diff": model - sens,
                "lo": float(lo),
                "hi": float(hi),
            }
        )
    return pd.DataFrame(rows)


def analysis(cfg, split: str, ctx: dict) -> dict | None:
    if split not in ("val_select", "test"):
        return None
    root = cfg.dataset_dir("ptbxl_plus")
    ids = ctx["meta"].index
    d = load_12sl_decisions(root).reindex(ids).fillna(False).to_numpy().astype(int)
    t = comparator_table(ctx["y"], ctx["p"], d, ctx["groups"], n_boot=cfg.n_boot)
    mapped = set(_load_map(MAP_PATH))
    unmapped = Counter(s for st in _statements(root).reindex(ids) for s in st if s not in mapped)
    out = cfg.reports / split
    t.to_csv(out / "comparator.csv", index=False)
    lines = [
        f"# GE 12SL comparator — {split}",
        "",
        "Model sensitivity at 12SL's own specificity (patient-bootstrap 95% CI of the "
        "difference). Indicative: the statement mapping is published in "
        "`configs/12sl_label_map.csv`.",
        "",
        "| class | positives | 12SL sens | 12SL spec | model sens @ 12SL spec | Δ [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for r in t.itertuples():
        lines.append(
            f"| {r[1]} | {r.n_pos} | {r._3:.3f} | {r._4:.3f} | {r.model_sens_at_12sl_spec:.3f}"
            f" | {r.diff:+.3f} [{r.lo:+.3f}, {r.hi:+.3f}] |"
        )
    lines += [
        "",
        "Unmapped 12SL statements on these records (count): "
        + ", ".join(f"{k} ({v})" for k, v in unmapped.most_common()),
    ]
    (out / "comparator.md").write_text("\n".join(lines) + "\n")
    return t.to_dict(orient="records")


ANALYSIS = ("comparator", analysis)
