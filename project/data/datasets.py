from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from utils.seed import seed_worker


DATASET_INFO: Dict[str, Dict[str, object]] = {
    "mnist": {
        "in_channels": 1,
        "image_size": (28, 28),
        "num_classes": 10,
    },
    "fashionmnist": {
        "in_channels": 1,
        "image_size": (28, 28),
        "num_classes": 10,
    },
    "cifar10": {
        "in_channels": 3,
        "image_size": (32, 32),
        "num_classes": 10,
    },
}


def normalize_dataset_name(dataset: str) -> str:
    ds = str(dataset).strip().lower()
    alias_map = {
        "fashion": "fashionmnist",
        "fmnist": "fashionmnist",
        "fashion-mnist": "fashionmnist",
        "fashion_mnist": "fashionmnist",
        "cifar": "cifar10",
        "cifar-10": "cifar10",
    }
    return alias_map.get(ds, ds)


def _parse_image_size(image_size: Optional[Sequence[int]], default_hw: Tuple[int, int]) -> Tuple[int, int]:
    if image_size is None:
        return default_hw
    if len(image_size) != 2:
        raise ValueError("image_size must have length 2, got {}".format(len(image_size)))
    return int(image_size[0]), int(image_size[1])


def get_dataset_info(dataset: str, image_size: Optional[Sequence[int]] = None) -> Dict[str, object]:
    dataset = normalize_dataset_name(dataset)
    if dataset not in DATASET_INFO:
        raise ValueError(f"Unsupported dataset: {dataset}")
    info = DATASET_INFO[dataset].copy()
    info["image_size"] = _parse_image_size(image_size=image_size, default_hw=tuple(info["image_size"]))
    return info


def _build_transform(dataset: str, out_range: str, image_size: Optional[Sequence[int]] = None) -> transforms.Compose:
    dataset = normalize_dataset_name(dataset)
    in_channels = DATASET_INFO[dataset]["in_channels"]
    target_hw = _parse_image_size(image_size=image_size, default_hw=tuple(DATASET_INFO[dataset]["image_size"]))
    tfms: list = []

    if target_hw != tuple(DATASET_INFO[dataset]["image_size"]):
        tfms.append(transforms.Resize(target_hw))

    tfms.append(transforms.ToTensor())

    if out_range == "neg_one_one":
        mean = (0.5,) * int(in_channels)
        std = (0.5,) * int(in_channels)
        tfms.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(tfms)


def get_dataloaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
    out_range: str,
    seed: int,
    val_split: float = 0.1,
    image_size: Optional[Sequence[int]] = None,
) -> tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    dataset = normalize_dataset_name(dataset)
    info = get_dataset_info(dataset, image_size=image_size)
    transform = _build_transform(dataset=dataset, out_range=out_range, image_size=image_size)

    if dataset == "mnist":
        full_train = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
    elif dataset == "fashionmnist":
        full_train = datasets.FashionMNIST(root=data_root, train=True, download=True, transform=transform)
        test_set = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=transform)
    elif dataset == "cifar10":
        full_train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    val_len = max(1, int(len(full_train) * val_split))
    train_len = len(full_train) - val_len
    split_generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full_train, [train_len, val_len], generator=split_generator)

    loader_generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    info["dataset"] = dataset
    info["data_range"] = 1.0 if out_range == "zero_one" else 2.0
    return train_loader, val_loader, test_loader, info
