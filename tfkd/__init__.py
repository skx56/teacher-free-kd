"""Teacher-free self-distillation for CNNs."""

from tfkd.ema import ModelEMA
from tfkd.losses import distill_loss, standardize_logits, total_loss

__all__ = ["ModelEMA", "distill_loss", "standardize_logits", "total_loss"]
