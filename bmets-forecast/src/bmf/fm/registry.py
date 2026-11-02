"""Registry of 3D image encoders used as frozen feature extractors (P2/P3) or LoRA-tuned (P4).

Every encoder maps a tensor [B, C, D, H, W] -> [B, out_dim]. Real foundation-model weights are
downloaded on Juno/Colab (see README_JUNO.md / README_COLAB.md); `random3d` is a deterministic
randomly-initialised CNN used in tests and as a "no pretraining" ablation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from torch import nn


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    in_channels: int
    patch: int                   # cubic patch edge in voxels (1 mm)
    out_dim: int
    url: str
    loader: Callable = field(repr=False)
    note: str = ""
    channels: tuple = ()         # sequence names in the order the encoder was trained on; () = pipeline order


class Random3D(nn.Module):
    def __init__(self, in_ch=4, width=32, out_dim=128, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        chans = [in_ch, width, width * 2, width * 4, out_dim]
        layers = []
        for a, b in zip(chans[:-1], chans[1:]):
            layers += [nn.Conv3d(a, b, 3, stride=2, padding=1), nn.InstanceNorm3d(b), nn.GELU()]
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            for p in self.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * (2.0 / max(1, p[0].numel())) ** 0.5 if p.dim() > 1 else torch.zeros_like(p))

    def forward(self, x):
        return self.net(x).mean(dim=(2, 3, 4))


def _need(path, url):
    if path is None or not Path(path).exists():
        raise FileNotFoundError(f"encoder weights not found at {path}; download from {url} and set fm.weights_dir in the config")


def load_checked(module: nn.Module, ckpt, name: str, min_frac: float = 0.9) -> float:
    """Load `ckpt` into `module`, refusing silent partial loads.

    Accepts a raw state dict or one wrapped under state_dict / model_state_dict / model; strips `module.`.
    Returns the fraction of the module's tensors matched by name and shape; raises if < min_frac, because a
    wrong checkpoint would otherwise leave random weights in place and be reported as a foundation model.
    """
    sd = ckpt
    if isinstance(sd, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(sd.get(key), dict):
                sd = sd[key]; break
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    target = module.state_dict()
    ok = {k: v for k, v in sd.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
    frac = len(ok) / max(len(target), 1)
    if frac < min_frac:
        raise ValueError(f"{name}: checkpoint matched {frac:.0%} of {len(target)} encoder tensors (need {min_frac:.0%}); "
                         f"wrong file or architecture. Example checkpoint keys: {list(sd)[:3]}")
    module.load_state_dict(ok, strict=False)
    return frac


def _sub(sd, prefix: str):
    """Keys under `prefix.` (after unwrapping and stripping `module.`), prefix removed; whole dict if none."""
    if isinstance(sd, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(sd.get(key), dict):
                sd = sd[key]; break
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    sub = {k[len(prefix) + 1:]: v for k, v in sd.items() if k.startswith(prefix + ".")}
    return sub or sd


def _load_random(weights_path=None):
    return Random3D().eval()


def _load_brainsegfounder(weights_path=None):
    """SwinUNETR (feature_size=48) from BrainSegFounder; embedding = pooled deepest Swin stage (768-d)."""
    _need(weights_path, ENCODERS["brainsegfounder"].url)
    from monai.networks.nets import SwinUNETR
    net = SwinUNETR(in_channels=4, out_channels=3, feature_size=48, use_v2=False)
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    frac = load_checked(net.swinViT, _sub(sd, "swinViT"), "brainsegfounder")   # only the encoder is used

    class Enc(nn.Module):
        def __init__(self, swin):
            super().__init__(); self.swin = swin

        def forward(self, x):
            return self.swin(x, normalize=True)[-1].mean(dim=(2, 3, 4))
    enc = Enc(net.swinViT).eval(); enc.load_fraction = frac
    return enc


def _load_samed3d(weights_path=None):
    """SAM-Med3D ViT image encoder (1 channel, 128^3) from the SAM-Med3D repository (pip install from GitHub)."""
    _need(weights_path, ENCODERS["samed3d"].url)
    import os, sys
    repo = os.environ.get("SAMMED3D_REPO")             # SAM-Med3D has no installer: clone it and point here
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)
    from segment_anything.build_sam3D import sam_model_registry3D  # provided by the SAM-Med3D repo
    sam = sam_model_registry3D["vit_b_ori"](checkpoint=None)
    sd = torch.load(weights_path, map_location="cpu", weights_only=False)
    frac = load_checked(sam.image_encoder, _sub(sd, "image_encoder"), "samed3d")

    class Enc(nn.Module):
        def __init__(self, enc):
            super().__init__(); self.enc = enc

        def forward(self, x):
            return self.enc(x[:, :1]).mean(dim=(2, 3, 4))
    enc = Enc(sam.image_encoder).eval(); enc.load_fraction = frac
    return enc


def _load_nnunet_enc(weights_path=None):
    """Encoder of our own nnU-Net S0 model (checkpoint_final.pth in the fold directory): task-specific baseline."""
    _need(weights_path, ENCODERS["nnunet_enc"].url)
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    pred = nnUNetPredictor(device=torch.device("cpu"))
    fold_dir = Path(weights_path).parent
    pred.initialize_from_trained_model_folder(str(fold_dir.parent), use_folds=(int(fold_dir.name.split("_")[-1]),),
                                              checkpoint_name=Path(weights_path).name)

    class Enc(nn.Module):
        def __init__(self, enc):
            super().__init__(); self.enc = enc

        def forward(self, x):
            return self.enc(x)[-1].mean(dim=(2, 3, 4))
    return Enc(pred.network.encoder).eval()


ENCODERS: dict[str, EncoderSpec] = {
    "random3d": EncoderSpec("random3d", 4, 64, 128, "n/a (random init)", _load_random, "no-pretraining ablation"),
    # BraTS fine-tuning (downstream/BraTS/finetuning): roi 128^3, inputs ordered (FLAIR, T1ce, T1, T2) as in
    # brats21_folds.json -- see jsons/remove_flair_t1ce.py, which drops the first two images to remove FLAIR and T1ce.
    "brainsegfounder": EncoderSpec("brainsegfounder", 4, 128, 768, "https://github.com/lab-smile/BrainSegFounder",
                                   _load_brainsegfounder, "SwinUNETR pretrained on UK Biobank + BraTS",
                                   channels=("t2f", "t1c", "t1n", "t2w")),
    "samed3d": EncoderSpec("samed3d", 1, 128, 384, "https://github.com/uni-medical/SAM-Med3D", _load_samed3d,
                           "3D SAM ViT-B encoder (t1c only)"),
    "nnunet_enc": EncoderSpec("nnunet_enc", 4, 96, 320, "trained locally by `bmf seg-train` (nnU-Net v2)",
                              _load_nnunet_enc, "task-specific supervised encoder"),
}


def load_encoder(name: str, weights_path: Path | str | None = None) -> nn.Module:
    if name not in ENCODERS:
        raise KeyError(f"unknown encoder {name}; available: {sorted(ENCODERS)}")
    return ENCODERS[name].loader(weights_path)
