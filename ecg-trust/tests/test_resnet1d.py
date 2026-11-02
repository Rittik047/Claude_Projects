import torch

from ecgtrust.models.resnet1d import ResNet1d


def test_forward_shape():
    assert ResNet1d()(torch.zeros(8, 12, 1000)).shape == (8, 5)


def test_param_budget():
    assert sum(p.numel() for p in ResNet1d().parameters()) < 2_000_000


def test_variable_length():
    assert ResNet1d()(torch.zeros(2, 12, 500)).shape == (2, 5)
    assert ResNet1d()(torch.zeros(2, 12, 250)).shape == (2, 5)


def test_width_scales_capacity():
    small = sum(p.numel() for p in ResNet1d(width=32).parameters())
    assert small < sum(p.numel() for p in ResNet1d(width=64).parameters()) / 2
