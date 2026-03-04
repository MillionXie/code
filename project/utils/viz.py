from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import torch
from torchvision.utils import make_grid, save_image

from utils.io import ensure_dir

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def to_display_range(images: torch.Tensor, out_range: str = "zero_one") -> torch.Tensor:
    images = images.detach().cpu()
    if out_range == "neg_one_one":
        images = (images + 1.0) / 2.0
    return images.clamp(0.0, 1.0)


def save_image_grid(images: torch.Tensor, path: str | Path, nrow: int = 8, out_range: str = "zero_one") -> None:
    path = Path(path)
    ensure_dir(path.parent)
    display = to_display_range(images, out_range=out_range)
    grid = make_grid(display, nrow=nrow, padding=2)
    save_image(grid, path)


def save_reconstruction_comparison(
    inputs: torch.Tensor,
    recons: torch.Tensor,
    path: str | Path,
    max_items: int = 8,
    out_range: str = "zero_one",
) -> None:
    n = min(max_items, inputs.size(0), recons.size(0))
    paired = torch.cat([inputs[:n], recons[:n]], dim=0)
    save_image_grid(paired, path=path, nrow=n, out_range=out_range)


def plot_histogram(
    values: np.ndarray,
    path: str | Path,
    title: str,
    xlabel: str,
    bins: int = 40,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    plt.figure(figsize=(7, 5))
    plt.hist(values, bins=bins, alpha=0.85, edgecolor="black", linewidth=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_eigen_spectrum(eigvals: np.ndarray, path: str | Path, title: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    plt.figure(figsize=(8, 5))
    x = np.arange(1, len(eigvals) + 1)
    plt.plot(x, eigvals, marker="o", markersize=2, linewidth=1)
    plt.title(title)
    plt.xlabel("Eigenvalue Index")
    plt.ylabel("Eigenvalue")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_scatter_2d(
    points: np.ndarray,
    path: str | Path,
    title: str,
    labels: Optional[np.ndarray] = None,
) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    plt.figure(figsize=(7, 6))
    if labels is None:
        plt.scatter(points[:, 0], points[:, 1], s=8, alpha=0.6)
    else:
        unique_labels = np.unique(labels)
        cmap = plt.get_cmap("tab10")
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                points[mask, 0],
                points[mask, 1],
                s=8,
                alpha=0.65,
                label=str(label),
                color=cmap(idx % 10),
            )
        if len(unique_labels) <= 20:
            plt.legend(loc="best", fontsize=8, markerscale=1.5)

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
