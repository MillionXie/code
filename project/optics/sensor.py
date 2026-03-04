from __future__ import annotations

from typing import List, Optional, Tuple, Union

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


def _to_hw(value: Union[int, Tuple[int, int], List[int]]) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Expected length 2, got {}".format(len(value)))
        return max(int(value[0]), 1), max(int(value[1]), 1)
    v = max(int(value), 1)
    return v, v


class IntensitySensor(nn.Module):
    """
    Differentiable sensor model:
    detect/noise -> optional extra ROI crop -> pooling (microlens array) -> optional digital resize -> normalize.

    Notes:
    - roi_hw is an optional extra FOV crop and is NOT the propagation padding/unpadding step.
    - out_hw is optional digital post-processing resize; keep None to avoid interpolation.
    """

    def __init__(
        self,
        roi_hw: Optional[Tuple[int, int]] = None,
        pool_type: str = "avg",
        pool_kernel: Union[int, Tuple[int, int], List[int]] = 2,
        pool_stride: Union[int, Tuple[int, int], List[int]] = 2,
        pool_reduce: str = "mean",
        out_hw: Optional[Tuple[int, int]] = None,
        expected_hw: Optional[Tuple[int, int]] = None,
        normalize: str = "mean",
        noise_model: str = "none",
        poisson_scale: float = 1.0,
        read_noise_std: float = 0.0,
    ):
        super().__init__()
        self.roi_hw = roi_hw
        self.pool_type = str(pool_type).lower()
        self.pool_kernel = _to_hw(pool_kernel)
        self.pool_stride = _to_hw(pool_stride)
        self.pool_reduce = str(pool_reduce).lower()
        self.out_hw = out_hw
        self.expected_hw = expected_hw
        self.normalize = str(normalize).lower()
        self.noise_model = str(noise_model).lower()
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
        k_hw = self.pool_kernel
        s_hw = self.pool_stride
        cell_area = float(k_hw[0] * k_hw[1])

        if self.pool_type == "none":
            return intensity

        if self.pool_reduce == "sum" and self.pool_type in ("avg", "max", "conv_stride"):
            pooled = F.avg_pool2d(intensity, kernel_size=k_hw, stride=s_hw)
            return pooled * cell_area

        if self.pool_reduce not in ("mean", "sum"):
            raise ValueError("Unsupported pool_reduce: {}".format(self.pool_reduce))

        if self.pool_type == "avg":
            return F.avg_pool2d(intensity, kernel_size=k_hw, stride=s_hw)
        if self.pool_type == "max":
            return F.max_pool2d(intensity, kernel_size=k_hw, stride=s_hw)
        if self.pool_type == "conv_stride":
            channels = intensity.shape[1]
            weight = torch.ones(
                channels,
                1,
                k_hw[0],
                k_hw[1],
                device=intensity.device,
                dtype=intensity.dtype,
            )
            weight = weight / cell_area
            return F.conv2d(intensity, weight, stride=s_hw, padding=0, groups=channels)

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
        roi_raw = torch.clamp(intensity, min=0.0)
        if self.roi_hw is not None:
            roi_raw = center_crop(roi_raw, self.roi_hw)

        # Sensor noise is applied after raw ROI extraction.
        x = self._apply_noise(roi_raw)
        pooled = torch.clamp(self._apply_pooling(x), min=0.0)

        pooled_raw = pooled
        if self.out_hw is not None and pooled.shape[-2:] != self.out_hw:
            pooled = F.interpolate(pooled, size=self.out_hw, mode="area")
        elif self.out_hw is None and self.expected_hw is not None and pooled.shape[-2:] != self.expected_hw:
            raise ValueError(
                "Pooled sensor map size {} does not match latent_hw {}. "
                "请通过 pool_kernel/pool_stride 让 pooled size == latent_hw，或显式设置 out_hw 做数字缩放。".format(
                    tuple(pooled.shape[-2:]), tuple(self.expected_hw)
                )
            )

        # Optical latent w must remain non-negative.
        latent_intensity = torch.clamp(pooled, min=0.0)
        final_latent = self._normalize(latent_intensity)

        if return_info:
            return final_latent, {
                "roi_raw_intensity": roi_raw,
                "roi_intensity": roi_raw,
                "pooled_intensity": pooled_raw,
                "final_intensity": final_latent,
                "latent_intensity_map": latent_intensity,
                "final_latent_map": final_latent,
            }
        return final_latent


__all__ = ["detect_intensity", "IntensitySensor", "center_crop"]
