from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F

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


def kl_latent_intensity_biased_gaussian(
    latent_intensity_map: torch.Tensor,
    var_mode: str = "constant",
    var0: float = 1.0,
    prior_mean_m0: float = 0.0,
    prior_sigma0: float = 1.0,
    pre_norm: str = "mean",
    eps: float = 1e-8,
    clamp_nonnegative: bool = True,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    KL(Q||P) on optical latent intensity w (after medium/sensor ROI+pool):
      Q = N(M, diag(var_q))
      P = N(M0, sigma0^2 I)
    where M = flatten(w), M0 is biased mean, and var_q uses diagonal approximation
    without introducing additional networks or sampling.
    """
    if prior_sigma0 <= 0:
        raise ValueError("prior_sigma0 must be > 0")
    if var0 <= 0:
        raise ValueError("var0 must be > 0")

    m = latent_intensity_map.flatten(start_dim=1)
    pre_norm = str(pre_norm).lower()
    if pre_norm == "mean":
        m = m / (m.mean(dim=1, keepdim=True) + float(eps))
    elif pre_norm == "none":
        pass
    else:
        raise ValueError("Unsupported pre_norm: {}".format(pre_norm))

    if clamp_nonnegative:
        m = torch.clamp(m, min=0.0)
    device = m.device
    dtype = m.dtype

    mode = str(var_mode).lower()
    if mode == "constant":
        var_q = torch.full_like(m, float(var0))
    elif mode in ("batch", "batch_var", "batch_diag"):
        v = torch.var(m, dim=0, unbiased=False, keepdim=True)
        var_q = torch.clamp(v, min=1e-8).expand_as(m)
    else:
        raise ValueError("Unsupported var_mode: {}".format(var_mode))

    prior_var = float(prior_sigma0) * float(prior_sigma0)
    log_prior_var = torch.log(torch.tensor(prior_var, device=device, dtype=dtype))
    log_var_q = torch.log(torch.clamp(var_q, min=1e-8))

    diff_sq = (m - float(prior_mean_m0)).pow(2)
    kl_per_dim = 0.5 * (log_prior_var - log_var_q - 1.0 + var_q / prior_var + diff_sq / prior_var)
    kl_per_sample = kl_per_dim.sum(dim=1)

    if reduction == "none":
        return kl_per_sample
    if reduction == "mean":
        return kl_per_sample.mean()
    if reduction == "sum":
        return kl_per_sample.sum()
    raise ValueError("Unsupported reduction: {}".format(reduction))


def compute_recon_per_sample(recon: torch.Tensor, target: torch.Tensor, recon_loss_type: str) -> torch.Tensor:
    return reconstruction_loss(recon, target, loss_type=recon_loss_type, reduction="none")


def _tv_per_sample(x: torch.Tensor) -> torch.Tensor:
    dx = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = x[:, :, :, 1:] - x[:, :, :, :-1]
    tv = dx.abs().flatten(start_dim=1).mean(dim=1) + dy.abs().flatten(start_dim=1).mean(dim=1)
    return tv


def compute_optical_penalty(
    stage_intensity_maps: Iterable[torch.Tensor],
    mode: str = "tv",
    reduction: str = "mean",
) -> torch.Tensor:
    stage_intensity_maps = list(stage_intensity_maps)
    if len(stage_intensity_maps) == 0:
        raise ValueError("stage_intensity_maps is empty; optical penalty requires real intensity stages.")

    penalties = []
    mode = str(mode).lower()

    for intensity in stage_intensity_maps:
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
    prior_cfg: Optional[Dict[str, object]],
    device: torch.device,
    apply_smooth: bool = True,
) -> torch.Tensor:
    prior_cfg = prior_cfg or {"type": "standard", "mu0": 0.0, "sigma": 1.0}
    prior_type = str(prior_cfg.get("type", "standard")).lower()
    mu0 = float(prior_cfg.get("mu0", 0.0))
    sigma = float(prior_cfg.get("sigma", 1.0))

    shape = (int(batch_size), int(latent_channels), int(latent_hw[0]), int(latent_hw[1]))
    eps = torch.randn(shape, device=device)

    if apply_smooth:
        smooth_cfg = prior_cfg.get("spatial_smooth", {}) if isinstance(prior_cfg, dict) else {}
        smooth_type = str(smooth_cfg.get("type", "none")).lower() if isinstance(smooth_cfg, dict) else "none"
        if smooth_type == "gaussian":
            sigma_px = float(smooth_cfg.get("sigma_px", 1.0))
            if sigma_px > 0:
                size = int(max(3, round(6.0 * sigma_px)))
                if size % 2 == 0:
                    size += 1
                coords = torch.arange(size, device=device, dtype=eps.dtype) - (size - 1) * 0.5
                g = torch.exp(-(coords * coords) / (2.0 * sigma_px * sigma_px))
                g = g / g.sum()
                kernel2d = torch.outer(g, g)
                kernel2d = kernel2d / kernel2d.sum()
                kernel = kernel2d.view(1, 1, size, size).repeat(int(latent_channels), 1, 1, 1)
                eps = F.conv2d(eps, kernel, padding=size // 2, groups=int(latent_channels))
        elif smooth_type == "none":
            pass
        else:
            raise ValueError("Unsupported spatial_smooth.type: {}".format(smooth_type))

    if prior_type in ("standard", "standard_gaussian", "normal"):
        return eps
    if prior_type in ("biased", "biased_gaussian", "shifted", "shifted_gaussian"):
        return mu0 + sigma * eps

    raise ValueError("Unsupported prior type for sampling: {}".format(prior_type))


__all__ = [
    "resolve_recon_loss",
    "kl_map_gaussian_prior",
    "kl_latent_intensity_biased_gaussian",
    "compute_recon_per_sample",
    "compute_optical_penalty",
    "sample_map_prior",
]
