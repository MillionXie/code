from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F

Reduction = Literal["none", "mean", "sum"]


def _reduce(per_sample: torch.Tensor, reduction: Reduction) -> torch.Tensor:
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError(f"Unsupported reduction: {reduction}")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, reduction: Reduction = "mean") -> torch.Tensor:
    """KL(q(z|x) || p(z)) for diagonal Gaussian posteriors, per sample then reduced."""
    per_sample = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return _reduce(per_sample, reduction)


def reconstruction_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    loss_type: Literal["bce", "mse"] = "mse",
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Reconstruction loss per sample (summed over pixels/channels), then reduced."""
    if loss_type == "bce":
        recon = recon.clamp(1e-6, 1.0 - 1e-6)
        loss_map = F.binary_cross_entropy(recon, target, reduction="none")
    elif loss_type == "mse":
        loss_map = F.mse_loss(recon, target, reduction="none")
    else:
        raise ValueError(f"Unsupported recon loss: {loss_type}")

    per_sample = loss_map.flatten(start_dim=1).sum(dim=1)
    return _reduce(per_sample, reduction)


def mse_loss(recon: torch.Tensor, target: torch.Tensor, reduction: Reduction = "mean") -> torch.Tensor:
    """Pixel-wise mean squared error per sample, then reduced."""
    mse_map = F.mse_loss(recon, target, reduction="none")
    per_sample = mse_map.flatten(start_dim=1).mean(dim=1)
    return _reduce(per_sample, reduction)


def psnr_from_mse(mse: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = torch.clamp(mse, min=1e-12)
    return 20.0 * torch.log10(torch.tensor(data_range, device=mse.device)) - 10.0 * torch.log10(mse)


def batch_mse_psnr(recon: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    per_sample_mse = mse_loss(recon, target, reduction="none")
    per_sample_psnr = psnr_from_mse(per_sample_mse, data_range=data_range)
    return per_sample_mse.mean(), per_sample_psnr.mean()


def _gaussian_kernel_2d(window_size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) * 0.5
    g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    g = g / torch.clamp(g.sum(), min=1e-12)
    kernel_2d = torch.outer(g, g)
    kernel_2d = kernel_2d / torch.clamp(kernel_2d.sum(), min=1e-12)
    return kernel_2d.view(1, 1, window_size, window_size)


def ssim_score(
    recon: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """
    SSIM score in [approximately -1, 1], per sample then reduced.
    This implementation supports both grayscale and RGB tensors in shape [N, C, H, W].
    """
    if recon.shape != target.shape:
        raise ValueError("recon and target must have the same shape, got {} vs {}".format(tuple(recon.shape), tuple(target.shape)))
    if recon.dim() != 4:
        raise ValueError("Expected [N,C,H,W], got shape {}".format(tuple(recon.shape)))
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    c1 = float(0.01 * data_range) ** 2
    c2 = float(0.03 * data_range) ** 2

    n, c, h, w = recon.shape
    ws = int(min(window_size, h, w))
    if ws % 2 == 0:
        ws = max(ws - 1, 1)

    if ws < 3:
        # Fallback for very small spatial sizes.
        x = recon.flatten(start_dim=2)
        y = target.flatten(start_dim=2)
        mu_x = x.mean(dim=2)
        mu_y = y.mean(dim=2)
        x_center = x - mu_x.unsqueeze(-1)
        y_center = y - mu_y.unsqueeze(-1)
        var_x = (x_center * x_center).mean(dim=2)
        var_y = (y_center * y_center).mean(dim=2)
        cov_xy = (x_center * y_center).mean(dim=2)
        num = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
        den = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
        per_channel = num / torch.clamp(den, min=1e-12)
        per_sample = per_channel.mean(dim=1)
        return _reduce(per_sample, reduction)

    kernel = _gaussian_kernel_2d(ws, sigma=sigma, device=recon.device, dtype=recon.dtype)
    kernel = kernel.repeat(c, 1, 1, 1)
    padding = ws // 2

    mu_x = F.conv2d(recon, kernel, stride=1, padding=padding, groups=c)
    mu_y = F.conv2d(target, kernel, stride=1, padding=padding, groups=c)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(recon * recon, kernel, stride=1, padding=padding, groups=c) - mu_x2
    sigma_y2 = F.conv2d(target * target, kernel, stride=1, padding=padding, groups=c) - mu_y2
    sigma_xy = F.conv2d(recon * target, kernel, stride=1, padding=padding, groups=c) - mu_xy

    sigma_x2 = torch.clamp(sigma_x2, min=0.0)
    sigma_y2 = torch.clamp(sigma_y2, min=0.0)

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = num / torch.clamp(den, min=1e-12)

    per_sample = ssim_map.flatten(start_dim=2).mean(dim=2).mean(dim=1)
    return _reduce(per_sample, reduction)
