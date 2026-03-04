from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def detect_intensity(E_complex: torch.Tensor) -> torch.Tensor:
    return E_complex.real.pow(2) + E_complex.imag.pow(2)


def center_crop(x: torch.Tensor, crop_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = x.shape[-2:]
    ch = min(int(crop_hw[0]), h)
    cw = min(int(crop_hw[1]), w)
    top = max((h - ch) // 2, 0)
    left = max((w - cw) // 2, 0)
    return x[:, :, top : top + ch, left : left + cw]


class IntensitySensor(nn.Module):
    """
    Differentiable sensor model:
    detect/noise -> fixed ROI crop -> pooling (microlens array) -> normalize.
    """

    def __init__(
        self,
        roi_hw: Optional[Tuple[int, int]] = None,
        pool_type: str = "avg",
        pool_kernel: int = 2,
        pool_stride: int = 2,
        out_hw: Optional[Tuple[int, int]] = None,
        normalize: str = "mean",
        noise_model: str = "none",
        poisson_scale: float = 1.0,
        read_noise_std: float = 0.0,
    ):
        super().__init__()
        self.roi_hw = roi_hw
        self.pool_type = pool_type
        self.pool_kernel = int(max(pool_kernel, 1))
        self.pool_stride = int(max(pool_stride, 1))
        self.out_hw = out_hw
        self.normalize = normalize
        self.noise_model = noise_model
        self.poisson_scale = float(poisson_scale)
        self.read_noise_std = float(read_noise_std)

    def _apply_noise(self, intensity: torch.Tensor) -> torch.Tensor:
        if self.noise_model == "none":
            return intensity

        if self.noise_model == "poisson_gaussian_approx":
            var = torch.clamp(intensity, min=0.0) * self.poisson_scale + self.read_noise_std * self.read_noise_std
            std = torch.sqrt(torch.clamp(var, min=1e-8))
            noisy = intensity + torch.randn_like(intensity) * std
            return torch.clamp(noisy, min=0.0)

        raise ValueError("Unsupported noise_model: {}".format(self.noise_model))

    def _apply_pooling(self, intensity: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "none":
            return intensity
        if self.pool_type == "avg":
            return F.avg_pool2d(intensity, kernel_size=self.pool_kernel, stride=self.pool_stride)
        if self.pool_type == "max":
            return F.max_pool2d(intensity, kernel_size=self.pool_kernel, stride=self.pool_stride)
        if self.pool_type == "conv_stride":
            channels = intensity.shape[1]
            weight = torch.ones(
                channels,
                1,
                self.pool_kernel,
                self.pool_kernel,
                device=intensity.device,
                dtype=intensity.dtype,
            )
            weight = weight / float(self.pool_kernel * self.pool_kernel)
            return F.conv2d(intensity, weight, stride=self.pool_stride, padding=0, groups=channels)

        raise ValueError("Unsupported pool_type: {}".format(self.pool_type))

    def _normalize(self, intensity: torch.Tensor) -> torch.Tensor:
        if self.normalize == "none":
            return intensity

        if self.normalize == "mean":
            denom = intensity.mean(dim=(-2, -1), keepdim=True) + 1e-6
            return intensity / denom

        if self.normalize == "log1p":
            return torch.log1p(torch.clamp(intensity, min=0.0))

        if self.normalize == "log1p_mean":
            out = torch.log1p(torch.clamp(intensity, min=0.0))
            denom = out.mean(dim=(-2, -1), keepdim=True) + 1e-6
            return out / denom

        raise ValueError("Unsupported normalize mode: {}".format(self.normalize))

    def forward(self, intensity: torch.Tensor, return_info: bool = False):
        # Keep raw ROI intensity for physical losses/inspection.
        roi_raw = intensity
        if self.roi_hw is not None:
            roi_raw = center_crop(roi_raw, self.roi_hw)

        # Sensor noise is applied after raw ROI extraction.
        x = self._apply_noise(roi_raw)
        x = self._apply_pooling(x)

        if self.out_hw is not None and x.shape[-2:] != self.out_hw:
            x = F.interpolate(x, size=self.out_hw, mode="area")

        # Optical latent w must remain non-negative.
        latent_intensity = torch.clamp(x, min=0.0)
        final_latent = self._normalize(latent_intensity)

        if return_info:
            return final_latent, {
                "roi_raw_intensity": roi_raw,
                "latent_intensity_map": latent_intensity,
                "final_latent_map": final_latent,
            }
        return final_latent


__all__ = ["detect_intensity", "IntensitySensor", "center_crop"]
