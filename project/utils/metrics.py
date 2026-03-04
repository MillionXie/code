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
