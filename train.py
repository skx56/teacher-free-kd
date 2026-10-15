"""Train a CNN with teacher-free self-distillation.

The student is trained from scratch. An EMA copy of its weights is the only
teacher. Hard labels use label smoothing. Soft targets use logit
standardization and a decoupled target / non-target KL.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from tfkd.data import build_loaders
from tfkd.ema import ModelEMA
from tfkd.losses import total_loss
from tfkd.models import BUILDERS, build_model


def main() -> None:
    args = parse_args()
    apply_dataset_defaults(args)
    if args.dry_run:
        dry_run(args)
        return

    set_seed(args.seed)
    device = get_device(args.device)
    pin_memory = device.type == "cuda"
    train_loader, val_loader, num_classes = build_loaders(
        args.dataset,
        args.data,
        args.batch_size,
        args.workers,
        pin_memory,
    )
    model = build_model(args.model, num_classes, cifar_stem=args.dataset == "cifar100").to(device)
    teacher = ModelEMA(model, decay=args.ema_decay)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(args.epochs):
        kd_weight = 0.0 if epoch < args.kd_warmup_epochs else args.kd_weight
        train_loss, train_ce, train_kd = train_one_epoch(
            model, teacher, train_loader, optimizer, device, args, kd_weight
        )
        scheduler.step()
        acc = evaluate(model, val_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch:03d}  loss {train_loss:.4f}  ce {train_ce:.4f}  "
            f"kd {train_kd:.4f}  acc {acc:.2f}  lr {lr:.5f}",
            flush=True,
        )
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "ema": teacher.ema.state_dict(),
            "acc": acc,
            "args": vars(args),
        }
        torch.save(state, checkpoint_dir / "last.pt")
        if acc >= best_acc:
            best_acc = acc
            torch.save(state, checkpoint_dir / "best.pt")

    print(f"best top-1 {best_acc:.2f}", flush=True)


def train_one_epoch(model, teacher, loader, optimizer, device, args, kd_weight: float):
    model.train()
    teacher.ema.eval()
    total = ce_total = kd_total = 0.0
    steps = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        teacher_logits = None
        if kd_weight > 0.0:
            with torch.no_grad():
                teacher_logits = teacher.ema(images)
        loss, ce, kd = total_loss(
            logits,
            teacher_logits,
            targets,
            label_smoothing=args.label_smoothing,
            kd_weight=kd_weight,
            temperature=args.temperature,
            alpha=args.alpha,
            beta=args.beta,
            standardize=not args.no_standardize,
        )
        loss.backward()
        optimizer.step()
        teacher.update(model)
        total += loss.item()
        ce_total += ce.item()
        kd_total += kd.item()
        steps += 1
    return total / steps, ce_total / steps, kd_total / steps


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = 0
    seen = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1)
        correct += (preds == targets).sum().item()
        seen += targets.size(0)
    return 100.0 * correct / seen


def dry_run(args) -> None:
    device = get_device(args.device)
    num_classes = 100 if args.dataset == "cifar100" else 1000
    image_size = 32 if args.dataset == "cifar100" else 224
    model = build_model(args.model, num_classes, cifar_stem=args.dataset == "cifar100").to(device)
    teacher = ModelEMA(model, decay=args.ema_decay)
    images = torch.randn(2, 3, image_size, image_size, device=device)
    targets = torch.randint(0, num_classes, (2,), device=device)
    logits = model(images)
    with torch.no_grad():
        teacher_logits = teacher.ema(images)
    loss, ce, kd = total_loss(
        logits,
        teacher_logits,
        targets,
        label_smoothing=args.label_smoothing,
        kd_weight=args.kd_weight,
        temperature=args.temperature or 4.0,
        alpha=args.alpha or 1.0,
        beta=args.beta or 8.0,
        standardize=not args.no_standardize,
    )
    loss.backward()
    teacher.update(model)
    print(
        f"dry-run ok  model={args.model}  device={device}  "
        f"logits={tuple(logits.shape)}  loss={loss.item():.4f}  ce={ce.item():.4f}  kd={kd.item():.4f}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teacher-free self-distillation")
    parser.add_argument("--dataset", choices=("cifar100", "imagenet"), default="cifar100")
    parser.add_argument("--data", default="data", help="Dataset root. ImageNet must contain train/ and val/.")
    parser.add_argument("--model", choices=tuple(BUILDERS), default="resnet18")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--kd-warmup-epochs", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None, help="Weight on the target-class KL.")
    parser.add_argument("--beta", type=float, default=None, help="Weight on the non-target KL.")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-standardize", action="store_true", help="Skip per-sample logit z-scoring.")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--dry-run", action="store_true", help="One synthetic forward/backward, then exit.")
    return parser.parse_args()


def apply_dataset_defaults(args: argparse.Namespace) -> None:
    if args.dataset == "cifar100":
        defaults = {"epochs": 200, "weight_decay": 5e-4, "kd_warmup_epochs": 20, "temperature": 4.0, "alpha": 1.0, "beta": 8.0}
    else:
        defaults = {"epochs": 100, "weight_decay": 1e-4, "kd_warmup_epochs": 10, "temperature": 1.0, "alpha": 1.0, "beta": 0.5}
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def get_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
