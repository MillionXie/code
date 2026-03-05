from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def detect_intensity(E_complex: torch.Tensor) -> torch.Tensor:
    return E_complex.real.pow(2) + E_complex.imag.pow(2)


def _to_hw(value: Union[int, Tuple[int, int], List[int]]) -> Tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("Expected length 2, got {}".format(len(value)))
        return max(int(value[0]), 1), max(int(value[1]), 1)
    v = max(int(value), 1)
    return v, v


class IntensitySensor(nn.Module):
    """
    Simplified sensor model:
    detector intensity -> optional microlens-style pooling -> latent map.

    Notes:
    - No ROI crop, no extra normalize branch.
    - No extra out_hw interpolation branch.
    - To use full-resolution latent, set pool_type=none and latent_hw equal to field size.
    """

    def __init__(
        self,
        pool_type: str = "avg",
        pool_kernel: Union[int, Tuple[int, int], List[int]] = 2,
        pool_stride: Union[int, Tuple[int, int], List[int]] = 2,
        pool_reduce: str = "mean",
        expected_hw: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.pool_type = str(pool_type).lower()
        self.pool_kernel = _to_hw(pool_kernel)
        self.pool_stride = _to_hw(pool_stride)
        self.pool_reduce = str(pool_reduce).lower()
        self.expected_hw = expected_hw

    def _apply_pooling(self, intensity: torch.Tensor) -> torch.Tensor:
        if self.pool_type == "none":
            pooled = intensity
        elif self.pool_type == "avg":
            pooled = F.avg_pool2d(intensity, kernel_size=self.pool_kernel, stride=self.pool_stride)
        elif self.pool_type == "max":
            pooled = F.max_pool2d(intensity, kernel_size=self.pool_kernel, stride=self.pool_stride)
        elif self.pool_type == "conv_stride":
            channels = intensity.shape[1]
            k_h, k_w = self.pool_kernel
            weight = torch.ones(
                channels,
                1,
                k_h,
                k_w,
                device=intensity.device,
                dtype=intensity.dtype,
            ) / float(k_h * k_w)
            pooled = F.conv2d(intensity, weight, stride=self.pool_stride, padding=0, groups=channels)
        else:
            raise ValueError("Unsupported pool_type: {}".format(self.pool_type))

        if self.pool_reduce not in ("mean", "sum"):
            raise ValueError("Unsupported pool_reduce: {}".format(self.pool_reduce))
        if self.pool_reduce == "sum" and self.pool_type != "none":
            k_h, k_w = self.pool_kernel
            pooled = pooled * float(k_h * k_w)

        return pooled

    def forward(self, intensity: torch.Tensor, return_info: bool = False):
        detector_intensity = torch.clamp(intensity, min=0.0)
        pooled_intensity = torch.clamp(self._apply_pooling(detector_intensity), min=0.0)
        latent_map = pooled_intensity

        if self.expected_hw is not None and latent_map.shape[-2:] != self.expected_hw:
            raise ValueError(
                "Latent size {} != expected latent_hw {}. "
                "请调整 pool_type/pool_kernel/pool_stride 或 model.latent_hw。".format(
                    tuple(latent_map.shape[-2:]), tuple(self.expected_hw)
                )
            )

        if return_info:
            return latent_map, {
                "detector_intensity": detector_intensity,
                "pooled_intensity": pooled_intensity,
                "latent_intensity_map": latent_map,
                "final_latent_map": latent_map,
            }
        return latent_map


__all__ = ["detect_intensity", "IntensitySensor"]
