"""Data audit report (plan Task 9): what was downloaded, what the labels look like, and
everything that could bias evaluation, written to `reports/data_audit.md`."""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from ecgtrust import CLASSES
from ecgtrust.config import Config
from ecgtrust.data.ptbxl import load_metadata, load_scp, superclass_labels
from ecgtrust.data.splits import SPLITS, assert_no_patient_leakage, make_splits

NOISE_COLS = ("static_noise", "baseline_drift", "burst_noise", "electrodes_problems")


def _table(rows: list[tuple], header: tuple = ("item", "value")) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def _downloads(cfg: Config) -> str:
    rows = []
    for name in ("ptbxl", "ptbxl_plus", "external"):
        p = cfg.dataset_dir(name) / "download_log.json"
        if p.exists():
            d = json.loads(p.read_text())
            rows.append(
                (
                    name,
                    d.get("version", ""),
                    d["n_selected"],
                    d["n_ok"],
                    d["n_failed"],
                    f"{d['n_bytes'] / 1e6:.1f}",
                )
            )
        else:
            rows.append((name, "", "not downloaded", "", "", ""))
    hdr = ("dataset", "version", "files selected", "verified", "failed", "MB")
    return "## Downloads\n\nAll files verified against the dataset's SHA256SUMS.txt.\n\n" + _table(
        rows, hdr
    )


def _ptbxl(cfg: Config) -> str:
    root = cfg.dataset_dir("ptbxl")
    meta = load_metadata(root)
    y = superclass_labels(meta, load_scp(root))
    splits = make_splits(meta)
    try:
        assert_no_patient_leakage(meta, splits)
        leak = "none (checked)"
    except Exception as e:  # noqa: BLE001
        leak = f"**LEAKAGE: {e}**"
    scp = load_scp(root)
    diag_codes = set(scp.index[scp["diagnostic"] == 1])
    lz = sum(1 for d in meta.scp_codes for k, v in d.items() if v == 0 and k in diag_codes)
    rows = [
        ("records", len(meta)),
        ("patients", meta.patient_id.nunique()),
        ("no superclass label (excluded)", int((~meta.has_label).sum())),
        ("diagnostic SCP codes with likelihood 0 (still counted, benchmark convention)", lz),
        ("age coded 300 (de-identified >89)", int((meta.age >= 300).sum())),
        ("pacemaker flag", int(meta.pacemaker.notna().sum())),
        ("patient leakage across splits", leak),
    ]
    rows += [(f"noise flag: {c}", int(meta[c].notna().sum())) for c in NOISE_COLS]
    prev = []
    for s in SPLITS:
        m = splits == s
        prev.append(
            (
                s,
                int(m.sum()),
                meta.loc[m, "patient_id"].nunique(),
                *(f"{int(y.loc[m, c].sum())} ({y.loc[m, c].mean():.1%})" for c in CLASSES),
            )
        )
    vh = meta.groupby("strat_fold").validated_by_human.mean()
    dev = meta.device.str.strip().value_counts()
    return "\n\n".join(
        [
            "## PTB-XL",
            _table(rows),
            "### Label prevalence per split",
            _table(prev, ("split", "records", "patients", *CLASSES)),
            "### Human-validated labels per fold",
            _table([(int(f), f"{v:.1%}") for f, v in vh.items()], ("fold", "validated_by_human")),
            "### Recording devices",
            _table(list(dev.items()), ("device", "records")),
        ]
    )


def _prevalence(df: pd.DataFrame, cfg: Config, suffix: str = "") -> list[tuple]:
    rows = []
    for c in CLASSES:
        n = int(df[c + suffix].sum())
        status = "ok" if n >= cfg.external_min_pos else "insufficient"
        rows.append((c, n, f"{df[c + suffix].mean():.1%}", status))
    return rows


def _external(cfg: Config) -> str:
    p = cfg.artifacts / "external_meta.parquet"
    if not p.exists():
        return "## External\n\nNot built yet."
    m = pd.read_parquet(p)
    if "has_label" not in m:
        m["has_label"] = m[list(CLASSES)].any(axis=1)
    ok = m[m.excluded_reason == ""]
    ev = ok[ok.has_label]
    reasons = m.loc[m.excluded_reason != "", "excluded_reason"].str.split(":").str[0]
    hdr = ("class", "positives", "prevalence", f"status (min {cfg.external_min_pos} positives)")
    parts = [
        "## External",
        _table(
            [
                ("records", len(m)),
                ("signal OK", len(ok)),
                ("signal excluded", len(m) - len(ok)),
                (
                    "signal OK but no superclass (excluded, as in PTB-XL)",
                    int((~ok.has_label).sum()),
                ),
                ("evaluated (signal OK and labelled)", len(ev)),
            ]
        ),
        "### Signal exclusions by reason",
        _table(list(reasons.value_counts().items()) or [("none", 0)], ("reason", "records")),
        "### Mapped superclass prevalence — primary (strict, PTB-XL-faithful) map\n\n"
        "NORM = proxy: only sinus-family rhythm codes (SR, SB, ST, SA).",
        _table(_prevalence(ev, cfg), hdr),
    ]
    if "has_label_liberal" in m:
        ev_lib = ok[ok.has_label_liberal]
        parts += [
            "### Sensitivity: liberal label map\n\nAdds isolated ST/T-wave findings → STTC and "
            f"voltage-only criteria → HYP. Evaluated records: {len(ev_lib)}.",
            _table(_prevalence(ev_lib, cfg, "_liberal"), hdr),
        ]
    parts += [
        "### Demographics (evaluated, primary map)",
        _table(list(ev.sex.value_counts().items()), ("sex", "records")),
        _table(list(ev.age_band.value_counts().sort_index().items()), ("age band", "records")),
    ]
    return "\n\n".join(parts)


def _vocab_12sl(cfg: Config) -> str:
    p = cfg.dataset_dir("ptbxl_plus") / "labels" / "12sl_statements.csv"
    if not p.exists():
        return "## 12SL statement vocabulary\n\nNot downloaded."
    s = pd.read_csv(p)["statements"].apply(ast.literal_eval)
    cnt = Counter(code for lst in s for code in lst)
    return "## 12SL statement vocabulary\n\nCounts over all ECGs; feeds Task 19's mapping.\n\n" + (
        _table(cnt.most_common(), ("statement", "ECGs"))
    )


def _baseline(cfg: Config) -> str:
    rows = []
    for p in sorted((cfg.reports / "val").glob("baseline_*.json")):
        d = json.loads(p.read_text())
        rows.append(
            (
                d["features"],
                f"{d['macro_auroc'][0]:.3f} [{d['macro_auroc'][1]:.3f}, {d['macro_auroc'][2]:.3f}]",
                *(f"{v:.3f}" for v in d["per_class_auroc"]),
            )
        )
    if not rows:
        return "## Baseline (val_select)\n\nNot run yet."
    return "## Baseline (val_select)\n\nPatient-level bootstrap 95 % CI.\n\n" + _table(
        rows, ("features", "macro AUROC [95% CI]", *CLASSES)
    )


def data_summary(cfg: Config) -> dict:
    """Machine-readable counts used by the guide and README."""
    root = cfg.dataset_dir("ptbxl")
    meta = load_metadata(root)
    y = superclass_labels(meta, load_scp(root))
    splits = make_splits(meta)
    out = {
        "ptbxl": {
            "records": len(meta),
            "patients": int(meta.patient_id.nunique()),
            "unlabelled": int((~meta.has_label).sum()),
            "split_records": {s: int((splits == s).sum()) for s in SPLITS},
            "prevalence": {
                s: {c: float(y.loc[splits == s, c].mean()) for c in CLASSES} for s in SPLITS
            },
            "validated_by_human": {
                int(k): float(v)
                for k, v in meta.groupby("strat_fold").validated_by_human.mean().items()
            },
        }
    }
    p = cfg.artifacts / "external_meta.parquet"
    if p.exists():
        m = pd.read_parquet(p)
        if "has_label" not in m:
            m["has_label"] = m[list(CLASSES)].any(axis=1)
        ok = m[m.excluded_reason == ""]
        ev = ok[ok.has_label]
        ext = {
            "records": len(m),
            "signal_ok": len(ok),
            "evaluated": len(ev),
            "prevalence_strict": {c: float(ev[c].mean()) for c in CLASSES},
        }
        if "has_label_liberal" in m:
            evl = ok[ok.has_label_liberal]
            ext["evaluated_liberal"] = len(evl)
            ext["prevalence_liberal"] = {c: float(evl[c + "_liberal"].mean()) for c in CLASSES}
        out["external"] = ext
    return out


def run_audit(cfg: Config) -> Path:
    out = cfg.reports / "data_audit.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    parts = [
        "# Data audit",
        _downloads(cfg),
        _ptbxl(cfg),
        _external(cfg),
        _vocab_12sl(cfg),
        _baseline(cfg),
    ]
    out.write_text("\n\n".join(parts) + "\n")
    (cfg.reports / "data_summary.json").write_text(json.dumps(data_summary(cfg), indent=2))
    return out


def register(sub) -> None:
    p = sub.add_parser("audit", help="write reports/data_audit.md")

    def run(args):
        from ecgtrust.config import load_config

        print(run_audit(load_config(args.config)))

    p.set_defaults(func=run)
