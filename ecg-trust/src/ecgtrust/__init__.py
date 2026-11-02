"""ecgtrust: trustworthy 12-lead ECG classification on PTB-XL."""

CLASSES: tuple[str, ...] = ("NORM", "MI", "STTC", "CD", "HYP")
LEADS: tuple[str, ...] = (
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
)
FS: int = 100
N_SAMPLES: int = 1000

__version__ = "0.1.0"
