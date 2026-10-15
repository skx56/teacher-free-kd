"""Torchvision CNNs, with a CIFAR stem when the input is 32x32."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

BUILDERS = {
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "resnext50_32x4d": models.resnext50_32x4d,
    "mobilenet_v2": models.mobilenet_v2,
    "shufflenet_v2_x1_0": models.shufflenet_v2_x1_0,
}


def build_model(name: str, num_classes: int, cifar_stem: bool) -> nn.Module:
    try:
        builder = BUILDERS[name]
    except KeyError as exc:
        known = ", ".join(BUILDERS)
        raise ValueError(f"Unknown model {name!r}. Choose from: {known}") from exc

    model = builder(weights=None, num_classes=num_classes)
    if cifar_stem:
        _use_cifar_stem(name, model)
    return model


def _use_cifar_stem(name: str, model: nn.Module) -> None:
    """Drop the ImageNet downsample on the first layer so 32x32 inputs survive."""
    if name.startswith(("resnet", "resnext")):
        model.conv1 = nn.Conv2d(3, model.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return

    if name.startswith("mobilenet"):
        first = model.features[0][0]
        model.features[0][0] = _stride1_conv(first)
        return

    if name.startswith("shufflenet"):
        model.conv1[0] = _stride1_conv(model.conv1[0])
        return

    raise ValueError(f"No CIFAR stem for {name}")


def _stride1_conv(conv: nn.Conv2d) -> nn.Conv2d:
    return nn.Conv2d(
        conv.in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=1,
        padding=conv.padding,
        bias=False,
    )
