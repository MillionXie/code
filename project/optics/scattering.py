from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityScattering(nn.Module):
    def forward(self, E_complex: torch.Tensor) -> torch.Tensor:
        return E_complex


class IIDPhaseMask(nn.Module):
    def __init__(self, phase_mode: str = "uniform", phase_sigma: float = 1.0):
        super().__init__()
        self.phase_mode = phase_mode
        self.phase_sigma = float(phase_sigma)

    def forward(self, E_complex: torch.Tensor) -> torch.Tensor:
        if self.phase_mode == "uniform":
            phi = torch.rand_like(E_complex.real) * (2.0 * math.pi)
        elif self.phase_mode == "normal":
            phi = torch.randn_like(E_complex.real) * self.phase_sigma
        else:
            raise ValueError("Unsupported phase_mode: {}".format(self.phase_mode))
        phase_mask = torch.exp(1j * phi)
        return E_complex * phase_mask


class CorrelatedPhaseMask(nn.Module):
    def __init__(self, corr_len_px: float = 3.0, phase_sigma: float = 1.0):
        super().__init__()
        self.corr_len_px = float(max(corr_len_px, 1e-3))
        self.phase_sigma = float(phase_sigma)

    def _gaussian_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        sigma = max(self.corr_len_px, 0.5)
        size = int(max(3, round(sigma * 6.0)))
        if size % 2 == 0:
            size += 1

        coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) * 0.5
        g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        g = g / g.sum()
        kernel = torch.outer(g, g)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def forward(self, E_complex: torch.Tensor) -> torch.Tensor:
        bsz, channels, height, width = E_complex.shape
        kernel = self._gaussian_kernel(device=E_complex.device, dtype=E_complex.real.dtype)
        pad = kernel.shape[-1] // 2

        noise = torch.randn(bsz * channels, 1, height, width, device=E_complex.device, dtype=E_complex.real.dtype)
        smoothed = F.conv2d(noise, kernel, padding=pad)
        smoothed = smoothed.view(bsz, channels, height, width)

        smoothed = smoothed / (smoothed.std(dim=(-2, -1), keepdim=True) + 1e-6)
        phi = smoothed * self.phase_sigma
        phase_mask = torch.exp(1j * phi)
        return E_complex * phase_mask


class AngleLimitedScattering(nn.Module):
    def __init__(self, wavelength_nm: float, pixel_pitch_um: float, na: float = 0.25, max_k: float | None = None):
        super().__init__()
        self.wavelength_m = float(wavelength_nm) * 1e-9
        self.pixel_pitch_m = float(pixel_pitch_um) * 1e-6
        self.na = float(na)
        self.max_k = max_k

    def forward(self, E_complex: torch.Tensor) -> torch.Tensor:
        _, _, height, width = E_complex.shape

        fx = torch.fft.fftfreq(width, d=self.pixel_pitch_m, device=E_complex.device, dtype=E_complex.real.dtype)
        fy = torch.fft.fftfreq(height, d=self.pixel_pitch_m, device=E_complex.device, dtype=E_complex.real.dtype)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")
        freq_radius = torch.sqrt(FX * FX + FY * FY)

        if self.max_k is not None:
            f_cut = float(self.max_k)
        else:
            f_cut = self.na / max(self.wavelength_m, 1e-12)

        pupil = (freq_radius <= f_cut).to(E_complex.real.dtype)
        pupil = pupil.unsqueeze(0).unsqueeze(0)

        spectrum = torch.fft.fft2(E_complex)
        filtered = torch.fft.ifft2(spectrum * pupil)
        return filtered


def build_scatterer(scatter_cfg: Dict[str, Any], wavelength_nm: float, pixel_pitch_um: float) -> nn.Module:
    scatter_type = str(scatter_cfg.get("type", "iid_phase")).lower()

    if scatter_type in ("none", "identity"):
        return IdentityScattering()
    if scatter_type in ("iid", "iid_phase", "iidphasemask"):
        return IIDPhaseMask(
            phase_mode=str(scatter_cfg.get("phase_mode", "uniform")).lower(),
            phase_sigma=float(scatter_cfg.get("phase_sigma", 1.0)),
        )
    if scatter_type in ("correlated", "correlated_phase", "correlatedphasemask"):
        return CorrelatedPhaseMask(
            corr_len_px=float(scatter_cfg.get("corr_len_px", 3.0)),
            phase_sigma=float(scatter_cfg.get("phase_sigma", 1.0)),
        )
    if scatter_type in ("angle_limited", "anglelimited", "na", "pupil"):
        return AngleLimitedScattering(
            wavelength_nm=wavelength_nm,
            pixel_pitch_um=pixel_pitch_um,
            na=float(scatter_cfg.get("na", 0.25)),
            max_k=scatter_cfg.get("max_k", None),
        )

    raise ValueError("Unsupported scatter type: {}".format(scatter_type))


__all__ = [
    "IdentityScattering",
    "IIDPhaseMask",
    "CorrelatedPhaseMask",
    "AngleLimitedScattering",
    "build_scatterer",
]
