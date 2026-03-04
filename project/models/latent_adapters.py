from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from optics.propagation import angular_spectrum_propagate
from optics.scattering import build_scatterer
from optics.sensor import IntensitySensor, detect_intensity


class IdentityAdapter(nn.Module):
    def forward(self, z_map: torch.Tensor, return_info: bool = False):
        if return_info:
            intensity = torch.clamp(z_map.pow(2), min=0.0)
            return z_map, {
                "stage_intensity_names": ["identity"],
                "stage_intensity_maps": [intensity],
                "latent_intensity_map": intensity,
                "final_latent_map": z_map,
            }
        return z_map


class OpticalOLSAdapter(nn.Module):
    """
    Optical latent adapter pipeline:
    resize -> propagate -> scatter -> propagate -> detect -> roi/pool -> decoder-size map.
    """

    def __init__(
        self,
        latent_shape: Tuple[int, int, int],
        resize_hw: Optional[Tuple[int, int]],
        field_init_mode: str,
        direct_mode: str,
        wavelength_nm: float,
        pixel_pitch_um: float,
        z1_mm: float,
        z2_mm: float,
        pad_factor: float,
        bandlimit: bool = True,
        upsample_factor: int = 1,
        scatter_cfg: Optional[Dict[str, Any]] = None,
        sensor_cfg: Optional[Dict[str, Any]] = None,
        output_center: bool = False,
    ):
        super().__init__()
        self.latent_channels, self.latent_h, self.latent_w = latent_shape
        self.resize_hw = resize_hw
        self.field_init_mode = str(field_init_mode).lower()
        self.direct_mode = str(direct_mode).lower()

        self.wavelength_nm = float(wavelength_nm)
        self.pixel_pitch_um = float(pixel_pitch_um)
        self.z1_mm = float(z1_mm)
        self.z2_mm = float(z2_mm)
        self.pad_factor = float(pad_factor)
        self.bandlimit = bool(bandlimit)
        self.upsample_factor = int(max(1, upsample_factor))
        self.output_center = bool(output_center)

        scatter_cfg = scatter_cfg or {"type": "iid_phase"}
        sensor_cfg = sensor_cfg or {}
        self.full_res_source = str(sensor_cfg.get("full_res_source", "roi_intensity")).lower()

        self.scatterer = build_scatterer(scatter_cfg, self.wavelength_nm, self.pixel_pitch_um)

        roi_hw = sensor_cfg.get("roi_hw", None)
        out_hw = sensor_cfg.get("out_hw", None)
        pool_kernel = sensor_cfg.get("pool_kernel", 2)
        pool_stride = sensor_cfg.get("pool_stride", 2)
        if isinstance(pool_kernel, list):
            pool_kernel = tuple(pool_kernel)
        if isinstance(pool_stride, list):
            pool_stride = tuple(pool_stride)
        self.sensor = IntensitySensor(
            roi_hw=tuple(roi_hw) if roi_hw is not None else None,
            pool_type=str(sensor_cfg.get("pool_type", "avg")).lower(),
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            pool_reduce=str(sensor_cfg.get("pool_reduce", "mean")).lower(),
            out_hw=tuple(out_hw) if out_hw is not None else None,
            expected_hw=(self.latent_h, self.latent_w) if self.direct_mode == "latent_hw" else None,
            normalize=str(sensor_cfg.get("normalize", "mean")).lower(),
            noise_model=str(sensor_cfg.get("noise_model", "none")).lower(),
            poisson_scale=float(sensor_cfg.get("poisson_scale", 1.0)),
            read_noise_std=float(sensor_cfg.get("read_noise_std", 0.0)),
        )

    def _to_complex_field(self, z_map: torch.Tensor) -> torch.Tensor:
        if self.field_init_mode == "real":
            real = z_map
        elif self.field_init_mode == "sqrt_positive":
            real = torch.sqrt(torch.clamp(z_map, min=0.0) + 1e-8)
        else:
            raise ValueError("Unsupported field_init_mode: {}".format(self.field_init_mode))
        return torch.complex(real, torch.zeros_like(real))

    def _resize_field(self, E: torch.Tensor, target_hw: Optional[Tuple[int, int]]) -> torch.Tensor:
        if target_hw is None or E.shape[-2:] == target_hw:
            return E
        real = F.interpolate(E.real, size=target_hw, mode="bilinear", align_corners=False)
        imag = F.interpolate(E.imag, size=target_hw, mode="bilinear", align_corners=False)
        return torch.complex(real, imag)

    def forward(self, z_map: torch.Tensor, return_info: bool = False):
        E0 = self._to_complex_field(z_map)
        E0 = self._resize_field(E0, self.resize_hw)

        I0 = detect_intensity(E0)

        E1 = angular_spectrum_propagate(
            E_complex=E0,
            wavelength_nm=self.wavelength_nm,
            pixel_pitch_um=self.pixel_pitch_um,
            z_mm=self.z1_mm,
            pad_factor=self.pad_factor,
            bandlimit=self.bandlimit,
            upsample_factor=self.upsample_factor,
        )
        I1 = detect_intensity(E1)

        E2 = self.scatterer(E1)
        I2 = detect_intensity(E2)

        E3 = angular_spectrum_propagate(
            E_complex=E2,
            wavelength_nm=self.wavelength_nm,
            pixel_pitch_um=self.pixel_pitch_um,
            z_mm=self.z2_mm,
            pad_factor=self.pad_factor,
            bandlimit=self.bandlimit,
            upsample_factor=self.upsample_factor,
        )
        I3 = detect_intensity(E3)

        sensor_final_map, sensor_info = self.sensor(I3, return_info=True)
        latent_intensity_map = sensor_info["latent_intensity_map"]
        roi_intensity = sensor_info["roi_intensity"]

        if latent_intensity_map.shape[1] != self.latent_channels:
            raise ValueError(
                "Optical latent intensity channel mismatch: got {}, expected {}".format(
                    latent_intensity_map.shape[1], self.latent_channels
                )
            )

        if self.direct_mode == "latent_hw":
            final_latent_map = sensor_final_map
            if final_latent_map.shape[-2:] != (self.latent_h, self.latent_w):
                raise ValueError(
                    "Optical sensor output size {} does not match latent_hw {}. "
                    "请通过 pool_kernel/pool_stride 让 pooled size == latent_hw，或显式设置 out_hw 做数字缩放。".format(
                        tuple(final_latent_map.shape[-2:]), (self.latent_h, self.latent_w)
                    )
                )
            if latent_intensity_map.shape[-2:] != (self.latent_h, self.latent_w):
                raise ValueError(
                    "Optical latent intensity size {} does not match latent_hw {}. "
                    "请通过 pool_kernel/pool_stride 让 pooled size == latent_hw，或显式设置 out_hw 做数字缩放。".format(
                        tuple(latent_intensity_map.shape[-2:]), (self.latent_h, self.latent_w)
                    )
                )
            if self.output_center:
                final_latent_map = final_latent_map - final_latent_map.mean(dim=(-2, -1), keepdim=True)
        elif self.direct_mode == "full_res":
            if self.full_res_source == "roi_intensity":
                final_latent_map = sensor_info["roi_intensity"]
            elif self.full_res_source == "pooled_intensity":
                final_latent_map = sensor_info["pooled_intensity"]
            elif self.full_res_source in ("final_intensity", "sensor_final"):
                final_latent_map = sensor_info["final_intensity"]
            else:
                raise ValueError("Unsupported sensor.full_res_source: {}".format(self.full_res_source))
            # Keep physical semantics in full-res mode: no centering on intensity maps.
        else:
            raise ValueError("Unsupported optics.direct_mode: {}".format(self.direct_mode))

        if final_latent_map.shape[1] != self.latent_channels:
            raise ValueError(
                "Optical adapter channel mismatch: got {}, expected {}".format(final_latent_map.shape[1], self.latent_channels)
            )

        if return_info:
            return final_latent_map, {
                "stage_intensity_names": ["input", "prop1", "scatter", "prop2", "sensor_roi"],
                "stage_intensity_maps": [I0, I1, I2, I3, roi_intensity],
                "latent_intensity_map": latent_intensity_map,
                "final_latent_map": final_latent_map,
                "sensor": sensor_info,
            }

        return final_latent_map


__all__ = ["IdentityAdapter", "OpticalOLSAdapter"]
