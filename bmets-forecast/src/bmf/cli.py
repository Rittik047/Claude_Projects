"""`bmf <stage> --config configs/<route>.yaml [options]` — one entry point for both Juno and Colab routes."""
from __future__ import annotations

import argparse
import logging
import sys

from .config import load_config
from .pipeline import STAGES, Ctx, to_layout


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bmf", description=__doc__)
    p.add_argument("stage", choices=sorted(STAGES) + ["to-layout"])
    p.add_argument("--config", default=None, help="route config (merged over configs/base.yaml)")
    p.add_argument("--model", default="P0", help="P0|P1|P2|P3|P5 (T2), P1 (T3), O1 (T4)")
    p.add_argument("--task", default="T2a", help="T2a|T2b|T3|T4")
    p.add_argument("--encoder", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--shard", default=None, help="i/n: process every n-th patient starting at i (array jobs)")
    p.add_argument("--pred-dir", dest="pred_dir", default=None)
    p.add_argument("--method", default="S0")
    p.add_argument("--seg-source", dest="seg_source", default="expert")
    p.add_argument("--force", action="store_true")
    p.add_argument("--subset", type=int, default=None, help="smoke test: keep N seeded patients with >=3 MRIs")
    p.add_argument("--map", default=None, help="to-layout: CSV study_id,seq,path")
    p.add_argument("--out", default=None, help="to-layout: output root")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if a.stage == "to-layout":
        to_layout(a.map, a.out)
        return 0
    cfg = load_config(a.config)
    if a.subset:   # smoke test: few patients, light statistics, separate *_smoke outputs (see Ctx)
        cfg.setdefault("subset", {})["n_patients"] = a.subset
        cfg["n_folds"] = min(int(cfg.get("n_folds", 5)), 3)
        cfg.setdefault("bootstrap", {})["n"] = 200
        cfg.setdefault("train", {})["epochs"] = 5
        cfg["radiomics_backend"] = "simple"
    ctx = Ctx(cfg)
    STAGES[a.stage](ctx, **{k: v for k, v in vars(a).items() if k not in ("stage", "config", "map", "out", "subset")})
    return 0


if __name__ == "__main__":
    sys.exit(main())
