"""Loss and EMA checks that do not need a dataset."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from tfkd.ema import ModelEMA
from tfkd.losses import distill_loss, standardize_logits


class LossTests(unittest.TestCase):
    def test_standardize_is_zero_mean_unit_std(self) -> None:
        logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]])
        out = standardize_logits(logits)
        self.assertTrue(torch.allclose(out.mean(dim=-1), torch.zeros(2), atol=1e-5))
        self.assertTrue(torch.allclose(out.std(dim=-1, unbiased=True), torch.ones(2), atol=1e-5))

    def test_identical_logits_have_zero_distill_loss(self) -> None:
        logits = torch.randn(8, 100)
        target = torch.randint(0, 100, (8,))
        loss = distill_loss(logits, logits.clone(), target, temperature=4.0, alpha=1.0, beta=8.0)
        self.assertLess(loss.abs().item(), 1e-4)

    def test_different_logits_have_positive_distill_loss(self) -> None:
        student = torch.randn(8, 100)
        teacher = torch.randn(8, 100)
        target = torch.randint(0, 100, (8,))
        loss = distill_loss(student, teacher, target)
        self.assertGreater(loss.item(), 0.0)

    def test_ema_moves_halfway_at_decay_half(self) -> None:
        model = nn.Linear(4, 3, bias=False)
        teacher = ModelEMA(model, decay=0.5)
        start = model.weight.detach().clone()
        with torch.no_grad():
            model.weight.add_(1.0)
        teacher.update(model)
        expected = start + 0.5
        self.assertTrue(torch.allclose(teacher.ema.weight, expected))


if __name__ == "__main__":
    unittest.main()
