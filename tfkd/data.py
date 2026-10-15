"""CIFAR-100 and ImageNet loaders."""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_loaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader, int]:
    root = Path(data_dir)
    if dataset == "cifar100":
        train_set = datasets.CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=_cifar_transform(train=True),
        )
        val_set = datasets.CIFAR100(
            root=root,
            train=False,
            download=True,
            transform=_cifar_transform(train=False),
        )
        num_classes = 100
    elif dataset == "imagenet":
        train_dir = root / "train"
        val_dir = root / "val"
        if not train_dir.is_dir() or not val_dir.is_dir():
            raise FileNotFoundError(
                f"ImageNet expects {train_dir} and {val_dir} laid out as ImageFolder class directories."
            )
        train_set = datasets.ImageFolder(train_dir, transform=_imagenet_transform(train=True))
        val_set = datasets.ImageFolder(val_dir, transform=_imagenet_transform(train=False))
        num_classes = 1000
    else:
        raise ValueError(f"Unknown dataset {dataset!r}. Choose cifar100 or imagenet.")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin_memory,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, num_classes


def _cifar_transform(train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)
    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose([transforms.ToTensor(), normalize])


def _imagenet_transform(train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
