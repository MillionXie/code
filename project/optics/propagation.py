from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _complex_interpolate(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    real = F.interpolate(x.real, size=size, mode="bilinear", align_corners=False)
    imag = F.interpolate(x.imag, size=size, mode="bilinear", align_corners=False)
    return torch.complex(real, imag)


def _complex_pad(x: torch.Tensor, pad: tuple[int, int, int, int]) -> torch.Tensor:
    return torch.complex(F.pad(x.real, pad, mode="constant", value=0.0), F.pad(x.imag, pad, mode="constant", value=0.0))


def _distance_lowpass_limits(wavelength_m: float, z_m: float, dx_m: float, dy_m: float, height: int, width: int) -> tuple[float, float]:
    # Distance-related cutoff approximation used by band-limited angular spectrum variants.
    lx = max(width * dx_m, 1e-12)
    ly = max(height * dy_m, 1e-12)
    z_abs = abs(z_m)

    fx_limit = 1.0 / (wavelength_m * math.sqrt(1.0 + (2.0 * z_abs / lx) ** 2))
    fy_limit = 1.0 / (wavelength_m * math.sqrt(1.0 + (2.0 * z_abs / ly) ** 2))
    return fx_limit, fy_limit


def angular_spectrum_propagate(
    E_complex: torch.Tensor,
    wavelength_nm: float,
    pixel_pitch_um: float,
    z_mm: float,
    pad_factor: float,
    bandlimit: bool = True,
    upsample_factor: int = 1,
) -> torch.Tensor:
    """
    Band-limited angular spectrum propagation with zero-padding and optional pre-upsampling.

    Args:
        E_complex: complex field [B, C, H, W]
        wavelength_nm: wavelength in nm
        pixel_pitch_um: sensor pitch in um
        z_mm: propagation distance in mm
        pad_factor: spatial padding factor, e.g., 2.0
        bandlimit: enable distance-related low-pass filter
        upsample_factor: optional upsampling before propagation; cropped back to original FOV

    Returns:
        Complex field with same shape as input.
    """
    if not torch.is_complex(E_complex):
        raise ValueError("E_complex must be a complex tensor")

    if abs(float(z_mm)) < 1e-12:
        return E_complex

    if upsample_factor < 1:
        raise ValueError("upsample_factor must be >= 1")

    bsz, channels, h0, w0 = E_complex.shape
    field = E_complex

    wavelength_m = float(wavelength_nm) * 1e-9
    dx_m = float(pixel_pitch_um) * 1e-6
    dy_m = float(pixel_pitch_um) * 1e-6
    z_m = float(z_mm) * 1e-3

    if upsample_factor > 1:
        hu = int(round(h0 * upsample_factor))
        wu = int(round(w0 * upsample_factor))
        field = _complex_interpolate(field, size=(hu, wu))
        dx_m = dx_m / float(upsample_factor)
        dy_m = dy_m / float(upsample_factor)

    _, _, h, w = field.shape

    if pad_factor < 1.0:
        raise ValueError("pad_factor must be >= 1")
    pad_h = max(int(round((pad_factor - 1.0) * h / 2.0)), 0)
    pad_w = max(int(round((pad_factor - 1.0) * w / 2.0)), 0)

    if pad_h > 0 or pad_w > 0:
        field_pad = _complex_pad(field, pad=(pad_w, pad_w, pad_h, pad_h))
    else:
        field_pad = field

    _, _, hp, wp = field_pad.shape

    fx = torch.fft.fftfreq(wp, d=dx_m, device=field_pad.device, dtype=field_pad.real.dtype)
    fy = torch.fft.fftfreq(hp, d=dy_m, device=field_pad.device, dtype=field_pad.real.dtype)
    FY, FX = torch.meshgrid(fy, fx, indexing="ij")

    inv_lambda = 1.0 / wavelength_m
    kz_sq = inv_lambda * inv_lambda - (FX * FX + FY * FY)
    propagating = kz_sq > 0
    kz = torch.sqrt(torch.clamp(kz_sq, min=0.0))

    phase = 2.0 * math.pi * z_m * kz
    transfer = torch.exp(1j * phase)

    mask = propagating
    if bandlimit:
        fx_limit, fy_limit = _distance_lowpass_limits(wavelength_m, z_m, dx_m, dy_m, hp, wp)
        lowpass = (torch.abs(FX) <= fx_limit) & (torch.abs(FY) <= fy_limit)
        mask = mask & lowpass

    transfer = transfer * mask.to(transfer.dtype)
    transfer = transfer.unsqueeze(0).unsqueeze(0)

    field_fft = torch.fft.fft2(field_pad)
    out_pad = torch.fft.ifft2(field_fft * transfer)

    if pad_h > 0 or pad_w > 0:
        out = out_pad[:, :, pad_h : pad_h + h, pad_w : pad_w + w]
    else:
        out = out_pad

    if upsample_factor > 1:
        out = _complex_interpolate(out, size=(h0, w0))

    return out


__all__ = ["angular_spectrum_propagate"]
