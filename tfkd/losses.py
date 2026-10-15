"""Teacher-free distillation losses.

Label smoothing is the virtual teacher on the hard labels. The EMA network
supplies the self-training targets. Those targets are split into a target-class
term and a non-target term, after per-sample logit standardization.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def standardize_logits(logits: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Z-score logits over the class dimension, per sample."""
    mean = logits.mean(dim=-1, keepdim=True)
    std = logits.std(dim=-1, keepdim=True)
    return (logits - mean) / (std + eps)


def dkd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    temperature: float,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Decoupled target-class and non-target-class KL.

    Target-class KL compares the binary probability of the labeled class against
    the rest. Non-target KL compares the distribution over the other classes.
    """
    target = target.reshape(-1)
    gt_mask = torch.zeros_like(student_logits).scatter_(1, target.unsqueeze(1), 1.0)
    other_mask = 1.0 - gt_mask

    pred_student = F.softmax(student_logits / temperature, dim=1)
    pred_teacher = F.softmax(teacher_logits / temperature, dim=1)
    student_binary = _cat_mask(pred_student, gt_mask, other_mask)
    teacher_binary = _cat_mask(pred_teacher, gt_mask, other_mask)
    target_kd = F.kl_div(
        torch.log(student_binary.clamp_min(1e-8)),
        teacher_binary,
        reduction="batchmean",
    )

    log_student_other = F.log_softmax(student_logits / temperature - 1000.0 * gt_mask, dim=1)
    teacher_other = F.softmax(teacher_logits / temperature - 1000.0 * gt_mask, dim=1)
    non_target_kd = F.kl_div(log_student_other, teacher_other, reduction="batchmean")

    scale = temperature ** 2
    return alpha * target_kd * scale + beta * non_target_kd * scale


def distill_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 1.0,
    beta: float = 8.0,
    standardize: bool = True,
) -> torch.Tensor:
    teacher_logits = teacher_logits.detach()
    if standardize:
        student_logits = standardize_logits(student_logits)
        teacher_logits = standardize_logits(teacher_logits)
    return dkd_loss(student_logits, teacher_logits, target, temperature, alpha, beta)


def total_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    target: torch.Tensor,
    label_smoothing: float,
    kd_weight: float,
    temperature: float,
    alpha: float,
    beta: float,
    standardize: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Smoothed cross-entropy plus the self-teacher KL."""
    hard = F.cross_entropy(student_logits, target, label_smoothing=label_smoothing)
    if teacher_logits is None or kd_weight == 0.0:
        zero = student_logits.new_zeros(())
        return hard, hard.detach(), zero
    soft = distill_loss(
        student_logits,
        teacher_logits,
        target,
        temperature=temperature,
        alpha=alpha,
        beta=beta,
        standardize=standardize,
    )
    return hard + kd_weight * soft, hard.detach(), soft.detach()


def _cat_mask(probs: torch.Tensor, gt_mask: torch.Tensor, other_mask: torch.Tensor) -> torch.Tensor:
    target_prob = (probs * gt_mask).sum(dim=1, keepdim=True)
    other_prob = (probs * other_mask).sum(dim=1, keepdim=True)
    return torch.cat([target_prob, other_prob], dim=1)
