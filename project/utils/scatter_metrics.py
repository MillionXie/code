from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def build_complex_field_from_intensity(intensity: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    E_in = sqrt(I + eps) * exp(i*0)
    intensity: [B, C, H, W] nonnegative tensor
    """
    amp = torch.sqrt(torch.clamp(intensity, min=0.0) + float(eps))
    return torch.complex(amp, torch.zeros_like(amp))


def pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Pearson correlation between two maps.
    a, b shape: [H, W] or [1, H, W]
    """
    if a.dim() == 3:
        a = a[0]
    if b.dim() == 3:
        b = b[0]
    if a.shape != b.shape:
        raise ValueError("pearson_corr shape mismatch: {} vs {}".format(tuple(a.shape), tuple(b.shape)))

    x = a.float().reshape(-1)
    y = b.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt(torch.clamp((x * x).sum(), min=eps) * torch.clamp((y * y).sum(), min=eps))
    return float((x * y).sum().item() / float(denom.item()))


def speckle_contrast(intensity: torch.Tensor, eps: float = 1e-12) -> float:
    """
    C = std(I) / mean(I), computed on one map.
    intensity shape: [H, W] or [1, H, W]
    """
    if intensity.dim() == 3:
        intensity = intensity[0]
    x = intensity.float()
    mean = torch.clamp(x.mean(), min=eps)
    std = x.std(unbiased=False)
    return float((std / mean).item())


def autocorrelation2d(intensity: torch.Tensor) -> torch.Tensor:
    """
    Normalized 2D autocorrelation map, centered by fftshift.
    intensity shape: [H, W] or [1, H, W]
    """
    if intensity.dim() == 3:
        intensity = intensity[0]
    x = intensity.float()
    x = x - x.mean()
    power = torch.fft.fft2(x)
    corr = torch.fft.ifft2(power * torch.conj(power)).real
    denom = torch.clamp((x * x).sum(), min=1e-12)
    corr = corr / denom
    corr = torch.fft.fftshift(corr, dim=(-2, -1))
    return corr


def autocorr_center_fwhm(corr: torch.Tensor) -> float:
    """
    Estimate autocorrelation width by center-line FWHM (in pixels).
    """
    if corr.dim() == 3:
        corr = corr[0]
    h, w = int(corr.shape[-2]), int(corr.shape[-1])
    cy, cx = h // 2, w // 2
    row = corr[cy].float()
    peak = float(row[cx].item())
    if peak <= 0:
        return 0.0
    half = 0.5 * peak

    left = cx
    while left > 0 and float(row[left].item()) >= half:
        left -= 1
    right = cx
    while right < w - 1 and float(row[right].item()) >= half:
        right += 1
    return float(max(right - left, 0))


def resize_2d_map(x: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
    """
    x: [H,W] or [1,H,W] -> output [H2,W2]
    """
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(0)
    else:
        raise ValueError("resize_2d_map expects 2D/3D tensor, got {}".format(tuple(x.shape)))
    y = F.interpolate(x, size=hw, mode="area")
    return y[0, 0]


def translate_zero_fill(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """
    Translate image/map with zero padding (no wrap-around).
    x: [H,W] or [1,H,W]
    dx > 0 means shift right, dy > 0 means shift down.
    """
    squeeze = False
    if x.dim() == 2:
        x = x.unsqueeze(0)
        squeeze = True
    if x.dim() != 3:
        raise ValueError("translate_zero_fill expects 2D/3D tensor, got {}".format(tuple(x.shape)))

    c, h, w = x.shape
    out = torch.zeros_like(x)

    src_x0 = max(0, -int(dx))
    src_x1 = min(w, w - int(dx)) if dx >= 0 else w
    dst_x0 = max(0, int(dx))
    dst_x1 = min(w, w + int(dx)) if dx < 0 else w

    src_y0 = max(0, -int(dy))
    src_y1 = min(h, h - int(dy)) if dy >= 0 else h
    dst_y0 = max(0, int(dy))
    dst_y1 = min(h, h + int(dy)) if dy < 0 else h

    if src_x1 > src_x0 and src_y1 > src_y0 and dst_x1 > dst_x0 and dst_y1 > dst_y0:
        out[:, dst_y0:dst_y1, dst_x0:dst_x1] = x[:, src_y0:src_y1, src_x0:src_x1]

    if squeeze:
        return out[0]
    return out


def to_numpy_image(x: torch.Tensor) -> np.ndarray:
    """
    Convert map to numpy HxW for plotting.
    """
    if x.dim() == 3:
        x = x[0]
    return x.detach().cpu().numpy()

