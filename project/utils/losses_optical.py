from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch

from utils.metrics import reconstruction_loss


def resolve_recon_loss(dataset: str, recon_loss_arg: str) -> str:
    if recon_loss_arg != "auto":
        return recon_loss_arg
    return "bce" if dataset == "mnist" else "mse"


def kl_map_gaussian_prior(
    mu_map: torch.Tensor,
    logvar_map: torch.Tensor,
    prior_type: str = "standard",
    mu0: float = 0.0,
    sigma: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    KL(q||p) where q = N(mu_map, diag(exp(logvar_map))) and p is either:
    - standard Gaussian: N(0, I)
    - biased Gaussian:   N(mu0, sigma^2 I)
    """
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    prior_type = str(prior_type).lower()
    var_q = torch.exp(logvar_map)

    if prior_type in ("standard", "standard_gaussian", "normal"):
        mu_p = 0.0
        var_p = 1.0
    elif prior_type in ("biased", "biased_gaussian", "shifted", "shifted_gaussian"):
        mu_p = float(mu0)
        var_p = float(sigma) * float(sigma)
    else:
        raise ValueError("Unsupported prior_type: {}".format(prior_type))

    diff = mu_map - mu_p
    kl_map = 0.5 * (torch.log(torch.tensor(var_p, device=mu_map.device, dtype=mu_map.dtype)) - logvar_map + (var_q + diff * diff) / var_p - 1.0)
    per_sample = kl_map.flatten(start_dim=1).sum(dim=1)

    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError("Unsupported reduction: {}".format(reduction))


def compute_recon_per_sample(recon: torch.Tensor, target: torch.Tensor, recon_loss_type: str) -> torch.Tensor:
    return reconstruction_loss(recon, target, loss_type=recon_loss_type, reduction="none")


def _tv_per_sample(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = x[:, :, :, 1:] - x[:, :, :, :-1]
    tv = dx.abs().flatten(start_dim=1).mean(dim=1) + dy.abs().flatten(start_dim=1).mean(dim=1)
    return tv


def compute_optical_penalty(
    stage_intensities: Iterable[torch.Tensor],
    mode: str = "tv",
    reduction: str = "mean",
) -> torch.Tensor:
    stage_intensities = list(stage_intensities)
    if len(stage_intensities) == 0:
        zero = torch.zeros(1, device="cpu")
        if reduction == "none":
            return zero
        return zero.squeeze(0)

    penalties = []
    mode = str(mode).lower()

    for intensity in stage_intensities:
        if mode == "l2":
            per_sample = intensity.pow(2).flatten(start_dim=1).mean(dim=1)
        elif mode == "tv":
            per_sample = _tv_per_sample(intensity)
        elif mode == "batch_l2":
            mean_batch = intensity.mean(dim=0, keepdim=True)
            per_sample = (intensity - mean_batch).pow(2).flatten(start_dim=1).mean(dim=1)
        else:
            raise ValueError("Unsupported optical penalty mode: {}".format(mode))
        penalties.append(per_sample)

    stacked = torch.stack(penalties, dim=0).mean(dim=0)

    if reduction == "none":
        return stacked
    if reduction == "mean":
        return stacked.mean()
    if reduction == "sum":
        return stacked.sum()
    raise ValueError("Unsupported reduction: {}".format(reduction))


def sample_map_prior(
    batch_size: int,
    latent_channels: int,
    latent_hw: tuple[int, int],
    prior_cfg: Optional[Dict[str, float]],
    device: torch.device,
) -> torch.Tensor:
    prior_cfg = prior_cfg or {"type": "standard", "mu0": 0.0, "sigma": 1.0}
    prior_type = str(prior_cfg.get("type", "standard")).lower()
    mu0 = float(prior_cfg.get("mu0", 0.0))
    sigma = float(prior_cfg.get("sigma", 1.0))

    shape = (int(batch_size), int(latent_channels), int(latent_hw[0]), int(latent_hw[1]))
    eps = torch.randn(shape, device=device)

    if prior_type in ("standard", "standard_gaussian", "normal"):
        return eps
    if prior_type in ("biased", "biased_gaussian", "shifted", "shifted_gaussian"):
        return mu0 + sigma * eps

    raise ValueError("Unsupported prior type for sampling: {}".format(prior_type))


__all__ = [
    "resolve_recon_loss",
    "kl_map_gaussian_prior",
    "compute_recon_per_sample",
    "compute_optical_penalty",
    "sample_map_prior",
]
