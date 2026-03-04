from typing import Dict

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
    "cifar10": {
        "in_channels": 3,
        "image_size": (32, 32),
        "num_classes": 10,
    },
}


def get_dataset_info(dataset: str) -> Dict[str, object]:
    if dataset not in DATASET_INFO:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return DATASET_INFO[dataset].copy()


def _build_transform(dataset: str, out_range: str) -> transforms.Compose:
    in_channels = DATASET_INFO[dataset]["in_channels"]
    tfms: list = [transforms.ToTensor()]

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
) -> tuple[DataLoader, DataLoader, DataLoader, Dict[str, object]]:
    dataset = dataset.lower()
    info = get_dataset_info(dataset)
    transform = _build_transform(dataset=dataset, out_range=out_range)

    if dataset == "mnist":
        full_train = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)
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
