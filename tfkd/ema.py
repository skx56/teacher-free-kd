"""Exponential-moving-average copy of the student, used as the self-teacher."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1).")
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        student = model.state_dict()
        for name, teacher_value in self.ema.state_dict().items():
            student_value = student[name].detach()
            if teacher_value.dtype.is_floating_point:
                teacher_value.mul_(self.decay).add_(student_value, alpha=1.0 - self.decay)
            else:
                teacher_value.copy_(student_value)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.ema(images)
