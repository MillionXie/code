from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from torchvision.models import Inception_V3_Weights, inception_v3

    _TV_AVAILABLE = True
except Exception:
    _TV_AVAILABLE = False

try:
    from scipy import linalg as scipy_linalg

    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False


def _to_zero_one(images: torch.Tensor, out_range: str) -> torch.Tensor:
    x = images
    if str(out_range).lower() == "neg_one_one":
        x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


def _prepare_for_inception(images: torch.Tensor, out_range: str) -> torch.Tensor:
    x = _to_zero_one(images, out_range=out_range)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - mean) / std


def _build_inception(device: torch.device):
    if not _TV_AVAILABLE:
        raise RuntimeError("torchvision inception is unavailable")
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False, aux_logits=False)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model


@torch.no_grad()
def _extract_features(
    model,
    images: torch.Tensor,
    out_range: str,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    feats = []
    n = images.shape[0]
    for i in range(0, n, batch_size):
        x = images[i : i + batch_size].to(device, non_blocking=True)
        x = _prepare_for_inception(x, out_range=out_range)
        f = model(x)
        if isinstance(f, tuple):
            f = f[0]
        feats.append(f.detach().cpu())
    return torch.cat(feats, dim=0)


def _covariance_torch(x: torch.Tensor) -> torch.Tensor:
    # x: [N, D]
    x = x.to(torch.float64)
    mu = x.mean(dim=0, keepdim=True)
    xc = x - mu
    denom = max(int(x.shape[0]) - 1, 1)
    cov = (xc.T @ xc) / float(denom)
    return cov


def _frechet_distance(mu1: torch.Tensor, cov1: torch.Tensor, mu2: torch.Tensor, cov2: torch.Tensor) -> float:
    mu1 = mu1.to(torch.float64)
    mu2 = mu2.to(torch.float64)
    cov1 = cov1.to(torch.float64)
    cov2 = cov2.to(torch.float64)
    diff = mu1 - mu2

    if _SCIPY_AVAILABLE:
        cov1_np = cov1.detach().cpu().numpy()
        cov2_np = cov2.detach().cpu().numpy()
        covmean, _ = scipy_linalg.sqrtm(cov1_np @ cov2_np, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        tr_covmean = float(np.trace(covmean))
        fid = float(diff.dot(diff).item() + torch.trace(cov1).item() + torch.trace(cov2).item() - 2.0 * tr_covmean)
        return max(fid, 0.0)

    prod = cov1 @ cov2
    eigvals = torch.linalg.eigvals(prod).real
    eigvals = torch.clamp(eigvals, min=0.0)
    tr_covmean = torch.sqrt(eigvals).sum()
    fid = diff.dot(diff) + torch.trace(cov1) + torch.trace(cov2) - 2.0 * tr_covmean
    return max(float(fid.item()), 0.0)


def compute_fid_from_images(
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    out_range: str,
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, object]:
    """
    Compute FID between two image batches.
    Returns a dict with keys: enabled, value, status, n_real, n_fake.
    """
    n = min(int(real_images.shape[0]), int(fake_images.shape[0]))
    if n < 8:
        return {
            "enabled": False,
            "value": None,
            "status": "not_enough_images",
            "n_real": int(real_images.shape[0]),
            "n_fake": int(fake_images.shape[0]),
        }
    try:
        model = _build_inception(device=device)
    except Exception as ex:
        return {
            "enabled": False,
            "value": None,
            "status": "inception_unavailable: {}".format(ex),
            "n_real": int(real_images.shape[0]),
            "n_fake": int(fake_images.shape[0]),
        }

    with torch.no_grad():
        real = real_images[:n]
        fake = fake_images[:n]
        feats_real = _extract_features(model, real, out_range=out_range, device=device, batch_size=int(batch_size))
        feats_fake = _extract_features(model, fake, out_range=out_range, device=device, batch_size=int(batch_size))

        mu_r = feats_real.mean(dim=0)
        mu_f = feats_fake.mean(dim=0)
        cov_r = _covariance_torch(feats_real)
        cov_f = _covariance_torch(feats_fake)
        fid = _frechet_distance(mu_r, cov_r, mu_f, cov_f)

    return {
        "enabled": True,
        "value": float(fid),
        "status": "ok",
        "n_real": int(n),
        "n_fake": int(n),
    }


__all__ = ["compute_fid_from_images"]
