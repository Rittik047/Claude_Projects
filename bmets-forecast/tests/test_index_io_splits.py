import numpy as np
import pandas as pd

from bmf.data.clinical import load_clinical, timepoints
from bmf.data.mri_index import SEQS_CORE, SEQS_OPT, build_index, load_seg, load_study
from bmf.data.splits import load_splits, make_splits, save_splits
from conftest import write_nii


def _mini_root(tmp_path, with_asl=False):
    root = tmp_path / "data"
    for sid in ["BMETS9001_01", "BMETS9001_02"]:
        for s in SEQS_CORE + ("adc",):
            write_nii(root / sid / f"{sid}-{s}.nii.gz", np.full((4, 4, 4), 2.0, np.float32))
        if with_asl:
            write_nii(root / sid / f"{sid}-asl.nii.gz", np.ones((4, 4, 4), np.float32))
        write_nii(root / sid / f"{sid}-seg.nii.gz", np.eye(4, dtype=np.uint8)[:, :, None].repeat(4, 2) * 3)
    tp = pd.DataFrame({"study_id": ["BMETS9001_01", "BMETS9001_02"], "case_id": ["BMETS9001"] * 2})
    return root, tp


def test_build_index_finds_files(tmp_path):
    root, tp = _mini_root(tmp_path)
    idx = build_index(root, tp)
    got = idx[idx.exists]
    assert set(got.seq) == set(SEQS_CORE) | {"adc", "seg"}
    assert len(idx) == 2 * (len(SEQS_CORE) + len(SEQS_OPT) + 1)


def test_missing_optional_sequence_masked(tmp_path):
    root, tp = _mini_root(tmp_path)
    idx = build_index(root, tp)
    vol, mask, _ = load_study(idx, "BMETS9001_01", SEQS_CORE + ("asl", "adc"))
    assert vol.shape == (6, 4, 4, 4)
    assert mask.tolist() == [1, 1, 1, 1, 0, 1]
    assert np.all(vol[4] == 0)
    seg = load_seg(idx, "BMETS9001_01")
    assert seg.dtype.kind in "iu" and seg.max() == 3


def test_splits_patient_level_and_stratified():
    rng = np.random.default_rng(0)
    clin = pd.DataFrame({"case_id": [f"P{i:03d}" for i in range(120)],
                         "primary": rng.choice(["lung", "breast", "melanoma", "rare"], 120, p=[.45, .35, .17, .03]),
                         "n_post": rng.choice([2, 3, 4], 120)})
    sp = make_splits(clin, n_folds=5, seed=2026)
    assert set(sp) == set(clin.case_id) and set(sp.values()) == set(range(5))
    df = clin.assign(fold=clin.case_id.map(sp))
    for prim, g in df.groupby("primary"):
        if len(g) >= 25:
            assert g.fold.nunique() == 5


def test_splits_deterministic(tmp_path):
    clin = load_clinical("tests/fixtures/clinical_min.tsv")
    a, b = make_splits(clin, 2, 2026), make_splits(clin, 2, 2026)
    assert a == b
    save_splits(a, tmp_path / "s.json")
    assert load_splits(tmp_path / "s.json") == a
