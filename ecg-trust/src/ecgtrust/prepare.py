"""`ecgtrust prepare`: build the splits and the 100 Hz signal caches that training reads.

Outputs, all in `artifacts/`:

- `splits.parquet`: patient-level splits (train folds 1-8, fold 9 split in two, test fold 10)
- `ptbxl_x100.npy` (N, 12, 1000) mV, `ptbxl_index.parquet`: PTB-XL signals in metadata order
- `external_x100.npy`, `external_meta.parquet`: external records resampled 500 → 100 Hz, with
  strict and liberal superclass labels and exclusion reasons

Every step is deterministic, so a rebuild reproduces the frozen run's inputs exactly.
"""

from __future__ import annotations

from pathlib import Path

from ecgtrust.config import Config

LABEL_MAP = Path(__file__).resolve().parents[2] / "configs" / "external_label_map.csv"


def prepare(cfg: Config, workers: int = 8, external: bool = True, force: bool = False) -> dict:
    from ecgtrust.data.ptbxl import load_metadata
    from ecgtrust.data.signals import build_cache
    from ecgtrust.data.splits import assert_no_patient_leakage, make_splits

    art = cfg.artifacts
    art.mkdir(parents=True, exist_ok=True)
    root = cfg.dataset_dir("ptbxl")
    meta = load_metadata(root)
    splits = make_splits(meta)
    assert_no_patient_leakage(meta, splits)
    splits.to_frame().to_parquet(art / "splits.parquet")
    done = {"splits": int(len(splits))}
    if force or not (art / "ptbxl_x100.npy").exists():
        build_cache(meta, root, art / "ptbxl_x100.npy", workers)
    done["ptbxl"] = int(len(meta))
    ext_root = cfg.dataset_dir("external")
    if external and (ext_root / "WFDBRecords").exists():
        if force or not (art / "external_meta.parquet").exists():
            from ecgtrust.data.external import build_external_cache

            build_external_cache(ext_root, LABEL_MAP, art, workers)
        done["external"] = True
    return done


def register(sub) -> None:
    p = sub.add_parser("prepare", help="build splits and 100 Hz signal caches in artifacts/")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--skip-external", action="store_true")
    p.add_argument("--force", action="store_true", help="rebuild caches that already exist")

    def run(args):
        from ecgtrust.config import load_config

        print(
            prepare(
                load_config(args.config),
                workers=args.workers,
                external=not args.skip_external,
                force=args.force,
            )
        )

    p.set_defaults(func=run)
