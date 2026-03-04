from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityScattering(nn.Module):
    def __init__(self, static: bool = True):
        super().__init__()
        self.static = bool(static)

    def forward(self, E_complex: torch.Tensor) -> torch.Tensor:
        return E_complex


class IIDPhaseMask(nn.Module):
    def __init__(self, phase_mode: str = "uniform", phase_sigma: float = 1.0, static: bool = True):
        super().__init__()
        self.phase_mode = phase_mode
        self.phase_sigma = float(phase_sigma)
        self.static = bool(static)
        self.register_buffer("_cached_phase_mask", None, persistent=False)
        self._cached_meta: Optional[Tuple[int, int, str, str]] = None

    def _sample_phase_mask(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.phase_mode == "uniform":
            phi = torch.rand(1, 1, height, width, device=device, dtype=dtype) * (2.0 * math.pi)
        elif self.phase_mode == "normal":
            phi = torch.randn(1, 1, height, width, device=device, dtype=dtype) * self.phase_sigma
        else:
            raise ValueError("Unsupported phase_mode: {}".format(self.phase_mode))
        return torch.exp(1j * phi)

    def forward(self, E_complex: torch.Tensor) -> torch.Tensor:
        _, _, height, width = E_complex.shape
        meta = (height, width, str(E_complex.device), str(E_complex.real.dtype))
        if self.static:
            if self._cached_phase_mask is None or self._cached_meta != meta:
                self._cached_phase_mask = self._sample_phase_mask(height, width, E_complex.device, E_complex.real.dtype)
                self._cached_meta = meta
            phase_mask = self._cached_phase_mask
        else:
            phase_mask = self._sample_phase_mask(height, width, E_complex.device, E_complex.real.dtype)
        return E_complex * phase_mask


class CorrelatedPhaseMask(nn.Module):
    def __init__(self, corr_len_px: float = 3.0, phase_sigma: float = 1.0, static: bool = True):
        super().__init__()
        self.corr_len_px = float(max(corr_len_px, 1e-3))
        self.phase_sigma = float(phase_sigma)
        self.static = bool(static)
        self.register_buffer("_cached_phase_mask", None, persistent=False)
        self._cached_meta: Optional[Tuple[int, int, str, str]] = None

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

    def _sample_phase_mask(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        kernel = self._gaussian_kernel(device=device, dtype=dtype)
        pad = kernel.shape[-1] // 2

        noise = torch.randn(1, 1, height, width, device=device, dtype=dtype)
        smoothed = F.conv2d(noise, kernel, padding=pad)
        smoothed = smoothed / (smoothed.std(dim=(-2, -1), keepdim=True) + 1e-6)
        phi = smoothed * self.phase_sigma
        phase_mask = torch.exp(1j * phi)
        return phase_mask

    def forward(self, E_complex: torch.Tensor) -> torch.Tensor:
        _, _, height, width = E_complex.shape
        meta = (height, width, str(E_complex.device), str(E_complex.real.dtype))
        if self.static:
            if self._cached_phase_mask is None or self._cached_meta != meta:
                self._cached_phase_mask = self._sample_phase_mask(height, width, E_complex.device, E_complex.real.dtype)
                self._cached_meta = meta
            phase_mask = self._cached_phase_mask
        else:
            phase_mask = self._sample_phase_mask(height, width, E_complex.device, E_complex.real.dtype)
        return E_complex * phase_mask


class AngleLimitedScattering(nn.Module):
    def __init__(
        self,
        wavelength_nm: float,
        pixel_pitch_um: float,
        na: float = 0.25,
        max_k: Optional[float] = None,
        static: bool = True,
    ):
        super().__init__()
        self.wavelength_m = float(wavelength_nm) * 1e-9
        self.pixel_pitch_m = float(pixel_pitch_um) * 1e-6
        self.na = float(na)
        self.max_k = max_k
        self.static = bool(static)
        self.register_buffer("_cached_pupil", None, persistent=False)
        self._cached_meta: Optional[Tuple[int, int, str, str]] = None

    def _build_pupil(self, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        fx = torch.fft.fftfreq(width, d=self.pixel_pitch_m, device=device, dtype=dtype)
        fy = torch.fft.fftfreq(height, d=self.pixel_pitch_m, device=device, dtype=dtype)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")
        freq_radius = torch.sqrt(FX * FX + FY * FY)

        if self.max_k is not None:
            f_cut = float(self.max_k)
        else:
            f_cut = self.na / max(self.wavelength_m, 1e-12)

        pupil = (freq_radius <= f_cut).to(dtype).unsqueeze(0).unsqueeze(0)
        return pupil

    def forward(self, E_complex: torch.Tensor) -> torch.Tensor:
        _, _, height, width = E_complex.shape
        meta = (height, width, str(E_complex.device), str(E_complex.real.dtype))
        if self.static:
            if self._cached_pupil is None or self._cached_meta != meta:
                self._cached_pupil = self._build_pupil(height, width, E_complex.device, E_complex.real.dtype)
                self._cached_meta = meta
            pupil = self._cached_pupil
        else:
            pupil = self._build_pupil(height, width, E_complex.device, E_complex.real.dtype)

        spectrum = torch.fft.fft2(E_complex)
        filtered = torch.fft.ifft2(spectrum * pupil)
        return filtered


def build_scatterer(scatter_cfg: Dict[str, Any], wavelength_nm: float, pixel_pitch_um: float) -> nn.Module:
    scatter_type = str(scatter_cfg.get("type", "iid_phase")).lower()
    static = bool(scatter_cfg.get("static", True))

    if scatter_type in ("none", "identity"):
        return IdentityScattering(static=static)
    if scatter_type in ("iid", "iid_phase", "iidphasemask"):
        return IIDPhaseMask(
            phase_mode=str(scatter_cfg.get("phase_mode", "uniform")).lower(),
            phase_sigma=float(scatter_cfg.get("phase_sigma", 1.0)),
            static=static,
        )
    if scatter_type in ("correlated", "correlated_phase", "correlatedphasemask"):
        return CorrelatedPhaseMask(
            corr_len_px=float(scatter_cfg.get("corr_len_px", 3.0)),
            phase_sigma=float(scatter_cfg.get("phase_sigma", 1.0)),
            static=static,
        )
    if scatter_type in ("angle_limited", "anglelimited", "na", "pupil"):
        return AngleLimitedScattering(
            wavelength_nm=wavelength_nm,
            pixel_pitch_um=pixel_pitch_um,
            na=float(scatter_cfg.get("na", 0.25)),
            max_k=scatter_cfg.get("max_k", None),
            static=static,
        )

    raise ValueError("Unsupported scatter type: {}".format(scatter_type))


__all__ = [
    "IdentityScattering",
    "IIDPhaseMask",
    "CorrelatedPhaseMask",
    "AngleLimitedScattering",
    "build_scatterer",
]
