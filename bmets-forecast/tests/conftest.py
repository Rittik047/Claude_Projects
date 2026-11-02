from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

FIX = Path(__file__).parent / "fixtures"


def write_nii(path: Path, arr, spacing=(1.0, 1.0, 1.0)):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(arr), np.diag([*spacing, 1.0])), str(path))
    return path


@pytest.fixture
def fix_dir():
    return FIX
