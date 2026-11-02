from ecgtrust import CLASSES, LEADS
from ecgtrust.config import load_config


def test_config_loads_constants():
    cfg = load_config()
    assert cfg.versions == {"ptbxl": "1.0.3", "ptbxl_plus": "1.0.1", "external": "1.0.0"}
    assert cfg.alphas == [0.05, 0.10] and cfg.n_boot == 2000 and cfg.ece_bins == 15
    assert cfg.snr_db == [24, 18, 12, 6, 0, -6]


def test_class_and_lead_order():
    assert CLASSES == ("NORM", "MI", "STTC", "CD", "HYP") and len(LEADS) == 12 and LEADS[6] == "V1"
