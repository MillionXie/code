from __future__ import annotations

from typing import Dict, List

import torch

from utils.viz import save_image_grid


def parse_interp_labels(text: str, num_required: int = 4) -> List[int]:
    labels: List[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            labels.append(int(token))
    if len(labels) < num_required:
        raise ValueError("Need at least {} labels, got '{}'".format(num_required, text))
    return labels[:num_required]


def save_reconstruction_pairs(
    inputs: torch.Tensor,
    recons: torch.Tensor,
    path,
    max_items: int = 128,
    pairs_per_row: int = 8,
    out_range: str = "zero_one",
) -> int:
    n = min(int(max_items), int(inputs.size(0)), int(recons.size(0)))
    if n <= 0:
        return 0
    interleaved = []
    for idx in range(n):
        interleaved.append(inputs[idx : idx + 1])
        interleaved.append(recons[idx : idx + 1])
    grid = torch.cat(interleaved, dim=0)
    nrow = max(2, int(pairs_per_row) * 2)
    save_image_grid(grid, path=path, nrow=nrow, out_range=out_range)
    return n


def bilinear_interpolate_corners(
    z_tl: torch.Tensor,
    z_tr: torch.Tensor,
    z_bl: torch.Tensor,
    z_br: torch.Tensor,
    grid_size: int = 8,
) -> torch.Tensor:
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    alphas = torch.linspace(0.0, 1.0, steps=grid_size, device=z_tl.device, dtype=z_tl.dtype)
    rows = []
    for vy in alphas:
        left = z_tl * (1.0 - vy) + z_bl * vy
        right = z_tr * (1.0 - vy) + z_br * vy
        for vx in alphas:
            z = left * (1.0 - vx) + right * vx
            rows.append(z)
    return torch.cat(rows, dim=0)


def interpolation_neighbor_metrics(images: torch.Tensor, grid_size: int) -> Dict[str, float]:
    if images.dim() != 4:
        raise ValueError("images must be [N,C,H,W], got shape {}".format(tuple(images.shape)))
    if images.size(0) != grid_size * grid_size:
        raise ValueError("N must equal grid_size^2, got N={} grid_size={}".format(images.size(0), grid_size))

    grid = images.view(grid_size, grid_size, images.size(1), images.size(2), images.size(3))
    h_diff = (grid[:, 1:] - grid[:, :-1]).pow(2).mean(dim=(2, 3, 4))
    v_diff = (grid[1:, :] - grid[:-1, :]).pow(2).mean(dim=(2, 3, 4))

    return {
        "neighbor_mse_h_mean": float(h_diff.mean().item()),
        "neighbor_mse_h_std": float(h_diff.std(unbiased=False).item()),
        "neighbor_mse_v_mean": float(v_diff.mean().item()),
        "neighbor_mse_v_std": float(v_diff.std(unbiased=False).item()),
        "neighbor_mse_all_mean": float(torch.cat([h_diff.flatten(), v_diff.flatten()]).mean().item()),
    }

