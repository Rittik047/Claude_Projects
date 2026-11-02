"""Export preprocessed studies to an nnU-Net v2 raw dataset (and command helpers).

S0: 4 channels (t1c, t1n, t2f, t2w). S2 (prior=True): + prior time point t1c and its
segmentation (zeros for a patient's first scan; a normal pre-dx MRI acts as an empty prior) -> longitudinal-aware model.
Folds mirror our patient-level splits through splits_final.json.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

CHANNELS = ("t1c", "t1n", "t2f", "t2w")
LABELS = {"background": 0, "NETC": 1, "SNFH": 2, "ET": 3, "RC": 4}


def export_nnunet(pp_root: Path | str, tp: pd.DataFrame, splits: dict, out_root: Path | str,
                  dataset_id: int = 501, prior: bool = False, name: str | None = None) -> Path:
    pp_root = Path(pp_root)
    name = name or f"Dataset{dataset_id:03d}_BMETS{'prior' if prior else ''}"
    ds = Path(out_root) / name
    (ds / "imagesTr").mkdir(parents=True, exist_ok=True); (ds / "labelsTr").mkdir(exist_ok=True)
    chans = list(CHANNELS) + (["prior_t1c", "prior_seg"] if prior else [])
    cases = []
    for cid, g in tp.groupby("case_id"):          # normal pre-dx MRIs included as empty-label negatives
        prev = None
        for sid in g.sort_values("folder_idx").study_id:
            d = pp_root / sid
            if not (d / "seg.nii.gz").exists():
                prev = sid; continue
            for k, ch in enumerate(CHANNELS):
                shutil.copy(d / f"{ch}.nii.gz", ds / "imagesTr" / f"{sid}_{k:04d}.nii.gz")
            if prior:
                ref = nib.load(str(d / "t1c.nii.gz"))
                for k, (fn, dtype) in enumerate((("t1c.nii.gz", np.float32), ("seg.nii.gz", np.int16)), start=len(CHANNELS)):
                    src = pp_root / prev / fn if prev else None
                    if src is not None and src.exists():
                        shutil.copy(src, ds / "imagesTr" / f"{sid}_{k:04d}.nii.gz")
                    else:
                        nib.save(nib.Nifti1Image(np.zeros(ref.shape, dtype), ref.affine), str(ds / "imagesTr" / f"{sid}_{k:04d}.nii.gz"))
            shutil.copy(d / "seg.nii.gz", ds / "labelsTr" / f"{sid}.nii.gz")
            cases.append(sid)
            prev = sid
    (ds / "dataset.json").write_text(json.dumps({
        "channel_names": {str(i): c for i, c in enumerate(chans)},
        "labels": LABELS, "numTraining": len(cases), "file_ending": ".nii.gz"}, indent=1))
    n_folds = max(splits.values()) + 1
    folds = []
    for k in range(n_folds):
        val = [s for s in cases if splits.get(s.rsplit("_", 1)[0]) == k]
        folds.append({"train": [s for s in cases if s not in val], "val": val})
    (ds / "splits_final.json").write_text(json.dumps(folds, indent=1))
    return ds


def nnunet_cmds(dataset_id: int, folds, config: str = "3d_fullres", trainer: str = "nnUNetTrainer") -> list[str]:
    """Commands executed by the Juno/Colab routes (splits_final.json must be copied to nnUNet_preprocessed)."""
    cmds = [f"nnUNetv2_plan_and_preprocess -d {dataset_id} -c {config} --verify_dataset_integrity"]
    cmds += [f"nnUNetv2_train {dataset_id} {config} {f} -tr {trainer} --npz" for f in folds]
    return cmds


def nnunet_predict_cmd(dataset_id: int, in_dir: str, out_dir: str, fold: int | str = "all",
                       config: str = "3d_fullres", trainer: str = "nnUNetTrainer") -> str:
    f = " ".join(map(str, range(5))) if fold == "all" else str(fold)
    return f"nnUNetv2_predict -i {in_dir} -o {out_dir} -d {dataset_id} -c {config} -tr {trainer} -f {f}"
