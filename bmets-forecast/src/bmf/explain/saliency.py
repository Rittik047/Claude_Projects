"""Voxel attributions for 3D models: integrated gradients and Grad-CAM."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def integrated_gradients(f, x: torch.Tensor, baseline: torch.Tensor | float = 0.0, steps: int = 32) -> torch.Tensor:
    """f: callable [B,C,D,H,W] -> [B] scalar score. x: [C,D,H,W]. Midpoint Riemann sum."""
    b = torch.full_like(x, float(baseline)) if not torch.is_tensor(baseline) else baseline
    alphas = (torch.arange(steps, dtype=x.dtype) + 0.5) / steps
    grads = torch.zeros_like(x)
    for a in alphas.split(8):
        pts = (b[None] + a.view(-1, 1, 1, 1, 1) * (x - b)[None]).requires_grad_(True)
        g, = torch.autograd.grad(f(pts).sum(), pts)
        grads += g.sum(0)
    return (x - b) * grads / steps


def gradcam3d(f, x: torch.Tensor, target_layer: torch.nn.Module) -> torch.Tensor:
    """Grad-CAM over a 3D conv layer; returns a [D,H,W] map in [0,1] upsampled to the input grid."""
    acts, grads = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0]))
    try:
        score = f(x[None]).sum()
        score.backward()
    finally:
        h1.remove(); h2.remove()
    a, g = acts["a"][0], grads["g"][0]
    cam = F.relu((g.mean(dim=(1, 2, 3), keepdim=True) * a).sum(0))
    cam = F.interpolate(cam[None, None], size=x.shape[1:], mode="trilinear", align_corners=False)[0, 0]
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
