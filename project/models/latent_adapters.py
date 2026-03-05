from __future__ import annotations

import math
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
    Optical latent encoder pipeline:
    image/feature -> resize -> (prop + phase)x2 -> propagate -> static scatter ->
    short propagate -> detect intensity -> optional pooling -> latent mean mu_w (nonnegative)
    -> sample z_w = mu_w + sigma * eps with fixed sigma.
    """

    def __init__(
        self,
        latent_shape: Tuple[int, int, int],
        resize_hw: Optional[Tuple[int, int]],
        field_init_mode: str,
        wavelength_nm: float,
        pixel_pitch_um: float,
        z1_mm: float,
        z2_mm: float,
        pad_factor: float,
        bandlimit: bool = True,
        upsample_factor: int = 1,
        scatter_cfg: Optional[Dict[str, Any]] = None,
        sensor_cfg: Optional[Dict[str, Any]] = None,
        diffraction_z_mm: Optional[Tuple[float, float]] = None,
        z_to_scatter_mm: Optional[float] = None,
        z_to_sensor_mm: float = 1.0,
        phase_trainable: bool = True,
        phase_init: str = "uniform",
        posterior_sigma: float = 0.1,
    ):
        super().__init__()
        self.latent_channels, self.latent_h, self.latent_w = latent_shape
        self.resize_hw = resize_hw
        self.field_init_mode = str(field_init_mode).lower()

        self.wavelength_nm = float(wavelength_nm)
        self.pixel_pitch_um = float(pixel_pitch_um)
        if self.resize_hw is None:
            self.resize_hw = (self.latent_h, self.latent_w)
        self.field_h, self.field_w = int(self.resize_hw[0]), int(self.resize_hw[1])
        if self.field_h <= 0 or self.field_w <= 0:
            raise ValueError("resize_hw must be positive, got {}".format(self.resize_hw))

        if diffraction_z_mm is None:
            self.diffraction_z_mm = (float(z1_mm), float(z1_mm))
        else:
            if len(diffraction_z_mm) != 2:
                raise ValueError("diffraction_z_mm must contain 2 distances, got {}".format(len(diffraction_z_mm)))
            self.diffraction_z_mm = (float(diffraction_z_mm[0]), float(diffraction_z_mm[1]))

        self.z_to_scatter_mm = float(z2_mm if z_to_scatter_mm is None else z_to_scatter_mm)
        self.z_to_sensor_mm = float(z_to_sensor_mm)
        self.pad_factor = float(pad_factor)
        self.bandlimit = bool(bandlimit)
        self.upsample_factor = int(max(1, upsample_factor))
        self.posterior_sigma = float(posterior_sigma)
        if self.posterior_sigma < 0:
            raise ValueError("posterior_sigma must be >= 0")

        scatter_cfg = scatter_cfg or {"type": "iid_phase"}
        sensor_cfg = sensor_cfg or {}
        self.scatterer = build_scatterer(scatter_cfg, self.wavelength_nm, self.pixel_pitch_um)

        pool_kernel = sensor_cfg.get("pool_kernel", 2)
        pool_stride = sensor_cfg.get("pool_stride", 2)
        if isinstance(pool_kernel, list):
            pool_kernel = tuple(pool_kernel)
        if isinstance(pool_stride, list):
            pool_stride = tuple(pool_stride)
        self.sensor = IntensitySensor(
            pool_type=str(sensor_cfg.get("pool_type", "avg")).lower(),
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
            pool_reduce=str(sensor_cfg.get("pool_reduce", "mean")).lower(),
            expected_hw=(self.latent_h, self.latent_w),
        )

        self.phase_trainable = bool(phase_trainable)
        self.phase_init = str(phase_init).lower()
        phase_params = []
        for _ in range(2):
            if self.phase_init in ("uniform", "random"):
                init = torch.rand(1, 1, self.field_h, self.field_w) * (2.0 * math.pi)
            elif self.phase_init in ("zeros", "zero"):
                init = torch.zeros(1, 1, self.field_h, self.field_w)
            else:
                raise ValueError("Unsupported phase_init: {}".format(self.phase_init))
            phase_params.append(nn.Parameter(init, requires_grad=self.phase_trainable))
        self.phase_masks = nn.ParameterList(phase_params)

    def _align_input_channels(self, x: torch.Tensor) -> torch.Tensor:
        c_in = x.shape[1]
        if c_in == self.latent_channels:
            return x
        if self.latent_channels == 1:
            return x.mean(dim=1, keepdim=True)
        if c_in == 1:
            return x.repeat(1, self.latent_channels, 1, 1)
        if c_in > self.latent_channels:
            return x[:, : self.latent_channels]
        pad = self.latent_channels - c_in
        zeros = torch.zeros(x.shape[0], pad, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        return torch.cat([x, zeros], dim=1)

    def _to_complex_field(self, z_map: torch.Tensor) -> torch.Tensor:
        if self.field_init_mode == "real":
            real = torch.clamp(z_map, min=0.0)
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

    def _run_optical_pipeline(self, field_map: torch.Tensor, return_info: bool = False):
        E0 = self._to_complex_field(field_map)
        E0 = self._resize_field(E0, self.resize_hw)
        I0 = detect_intensity(E0)

        stage_names = ["field_input"]
        stage_maps = [I0]

        E = E0
        for idx, z_mm in enumerate(self.diffraction_z_mm):
            E = angular_spectrum_propagate(
                E_complex=E,
                wavelength_nm=self.wavelength_nm,
                pixel_pitch_um=self.pixel_pitch_um,
                z_mm=z_mm,
                pad_factor=self.pad_factor,
                bandlimit=self.bandlimit,
                upsample_factor=self.upsample_factor,
            )
            phase = self.phase_masks[idx]
            phase_mask = torch.exp(1j * phase).to(E.dtype)
            E = E * phase_mask
            stage_names.append("diff{}".format(idx + 1))
            stage_maps.append(detect_intensity(E))

        E = angular_spectrum_propagate(
            E_complex=E,
            wavelength_nm=self.wavelength_nm,
            pixel_pitch_um=self.pixel_pitch_um,
            z_mm=self.z_to_scatter_mm,
            pad_factor=self.pad_factor,
            bandlimit=self.bandlimit,
            upsample_factor=self.upsample_factor,
        )
        I_before_scatter = detect_intensity(E)
        stage_names.append("before_scatter")
        stage_maps.append(I_before_scatter)

        E = self.scatterer(E)
        I_scatter = detect_intensity(E)
        stage_names.append("after_scatter")
        stage_maps.append(I_scatter)

        E = angular_spectrum_propagate(
            E_complex=E,
            wavelength_nm=self.wavelength_nm,
            pixel_pitch_um=self.pixel_pitch_um,
            z_mm=self.z_to_sensor_mm,
            pad_factor=self.pad_factor,
            bandlimit=self.bandlimit,
            upsample_factor=self.upsample_factor,
        )
        I_sensor = detect_intensity(E)
        stage_names.append("sensor_pre_pool")
        stage_maps.append(I_sensor)

        latent_mean_map, sensor_info = self.sensor(I_sensor, return_info=True)
        latent_mean_map = torch.clamp(latent_mean_map, min=0.0)

        if latent_mean_map.shape[1] != self.latent_channels:
            raise ValueError(
                "Optical latent intensity channel mismatch: got {}, expected {}".format(
                    latent_mean_map.shape[1], self.latent_channels
                )
            )

        if return_info:
            return latent_mean_map, {
                "stage_intensity_names": stage_names + ["latent_mean"],
                "stage_intensity_maps": stage_maps + [latent_mean_map],
                "latent_intensity_map": latent_mean_map,
                "latent_mean_map": latent_mean_map,
                "sensor": sensor_info,
            }
        return latent_mean_map

    def _sample_latent(self, latent_mean_map: torch.Tensor, sample_posterior: bool, posterior_sigma: Optional[float]) -> Tuple[torch.Tensor, float]:
        sigma = self.posterior_sigma if posterior_sigma is None else float(posterior_sigma)
        if sigma < 0:
            raise ValueError("posterior_sigma must be >= 0")
        if not sample_posterior or sigma == 0.0:
            return latent_mean_map, sigma
        eps = torch.randn_like(latent_mean_map)
        return latent_mean_map + sigma * eps, sigma

    def encode_from_input(
        self,
        x: torch.Tensor,
        return_info: bool = False,
        sample_posterior: bool = True,
        posterior_sigma: Optional[float] = None,
    ):
        field_map = self._align_input_channels(x)
        latent_mean_map, info = self._run_optical_pipeline(field_map, return_info=True)
        latent_sample_map, sigma = self._sample_latent(
            latent_mean_map=latent_mean_map,
            sample_posterior=sample_posterior,
            posterior_sigma=posterior_sigma,
        )
        if return_info:
            info.update(
                {
                    "field_input_map": field_map,
                    "posterior_sigma": sigma,
                    "final_latent_map": latent_sample_map,
                }
            )
            return latent_sample_map, info
        return latent_sample_map

    def forward(
        self,
        z_map: torch.Tensor,
        return_info: bool = False,
        sample_posterior: bool = False,
        posterior_sigma: Optional[float] = None,
    ):
        latent_mean_map, info = self._run_optical_pipeline(z_map, return_info=True)
        latent_sample_map, sigma = self._sample_latent(
            latent_mean_map=latent_mean_map,
            sample_posterior=sample_posterior,
            posterior_sigma=posterior_sigma,
        )
        if return_info:
            info.update({"posterior_sigma": sigma, "final_latent_map": latent_sample_map})
            return latent_sample_map, info
        return latent_sample_map


class OpticalDiffractionDecoder(nn.Module):
    """
    Pure optical decoder:
    latent intensity map -> complex field -> (prop + phase)xN -> sensor intensity -> reconstructed image.
    """

    def __init__(
        self,
        latent_shape: Tuple[int, int, int],
        out_channels: int,
        output_hw: Tuple[int, int],
        field_hw: Optional[Tuple[int, int]],
        field_init_mode: str,
        wavelength_nm: float,
        pixel_pitch_um: float,
        layer_z_mm: Tuple[float, float, float, float],
        z_to_sensor_mm: float,
        pad_factor: float,
        bandlimit: bool = True,
        upsample_factor: int = 1,
        phase_trainable: bool = True,
        phase_init: str = "uniform",
        out_range: str = "zero_one",
    ):
        super().__init__()
        self.latent_channels, self.latent_h, self.latent_w = latent_shape
        self.out_channels = int(out_channels)
        self.output_hw = (int(output_hw[0]), int(output_hw[1]))
        self.field_hw = field_hw if field_hw is not None else self.output_hw
        self.field_hw = (int(self.field_hw[0]), int(self.field_hw[1]))
        self.field_init_mode = str(field_init_mode).lower()
        self.wavelength_nm = float(wavelength_nm)
        self.pixel_pitch_um = float(pixel_pitch_um)
        self.layer_z_mm = tuple(float(z) for z in layer_z_mm)
        self.z_to_sensor_mm = float(z_to_sensor_mm)
        self.pad_factor = float(pad_factor)
        self.bandlimit = bool(bandlimit)
        self.upsample_factor = int(max(1, upsample_factor))
        self.out_range = str(out_range)
        self.phase_trainable = bool(phase_trainable)
        self.phase_init = str(phase_init).lower()

        phase_params = []
        for _ in range(len(self.layer_z_mm)):
            if self.phase_init in ("uniform", "random"):
                init = torch.rand(1, 1, self.field_hw[0], self.field_hw[1]) * (2.0 * math.pi)
            elif self.phase_init in ("zeros", "zero"):
                init = torch.zeros(1, 1, self.field_hw[0], self.field_hw[1])
            else:
                raise ValueError("Unsupported phase_init: {}".format(self.phase_init))
            phase_params.append(nn.Parameter(init, requires_grad=self.phase_trainable))
        self.phase_masks = nn.ParameterList(phase_params)

        # Monotonic intensity-to-image mapping without convolution.
        self.log_gain = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def _resize_field(self, E: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if E.shape[-2:] == target_hw:
            return E
        real = F.interpolate(E.real, size=target_hw, mode="bilinear", align_corners=False)
        imag = F.interpolate(E.imag, size=target_hw, mode="bilinear", align_corners=False)
        return torch.complex(real, imag)

    def _align_channels(self, x: torch.Tensor, target_channels: int) -> torch.Tensor:
        c_in = x.shape[1]
        if c_in == target_channels:
            return x
        if c_in == 1 and target_channels > 1:
            return x.repeat(1, target_channels, 1, 1)
        if c_in > target_channels:
            return x[:, :target_channels]
        pad = target_channels - c_in
        zeros = torch.zeros(x.shape[0], pad, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype)
        return torch.cat([x, zeros], dim=1)

    def _latent_to_complex(self, z_latent: torch.Tensor) -> torch.Tensor:
        z_latent = self._align_channels(z_latent, self.latent_channels)
        if self.field_init_mode in ("sqrt_positive", "sqrt"):
            amp = torch.sqrt(torch.clamp(z_latent, min=0.0) + 1e-8)
        elif self.field_init_mode == "real":
            amp = z_latent
        else:
            raise ValueError("Unsupported decoder field_init_mode: {}".format(self.field_init_mode))
        return torch.complex(amp, torch.zeros_like(amp))

    def _intensity_to_output(self, intensity: torch.Tensor) -> torch.Tensor:
        gain = torch.exp(self.log_gain).to(dtype=intensity.dtype, device=intensity.device)
        x = gain * intensity + self.bias.to(dtype=intensity.dtype, device=intensity.device)
        if self.out_range == "zero_one":
            return torch.sigmoid(x)
        if self.out_range == "neg_one_one":
            return torch.tanh(x)
        raise ValueError("Unsupported out_range: {}".format(self.out_range))

    def forward(self, z_latent: torch.Tensor, return_info: bool = False):
        E = self._latent_to_complex(z_latent)
        E = self._resize_field(E, self.field_hw)

        stage_names = []
        stage_maps = []
        for idx, z_mm in enumerate(self.layer_z_mm):
            E = angular_spectrum_propagate(
                E_complex=E,
                wavelength_nm=self.wavelength_nm,
                pixel_pitch_um=self.pixel_pitch_um,
                z_mm=float(z_mm),
                pad_factor=self.pad_factor,
                bandlimit=self.bandlimit,
                upsample_factor=self.upsample_factor,
            )
            phase = self.phase_masks[idx]
            phase_mask = torch.exp(1j * phase).to(E.dtype)
            E = E * phase_mask
            stage_names.append("dec_diff{}".format(idx + 1))
            stage_maps.append(detect_intensity(E))

        E = angular_spectrum_propagate(
            E_complex=E,
            wavelength_nm=self.wavelength_nm,
            pixel_pitch_um=self.pixel_pitch_um,
            z_mm=self.z_to_sensor_mm,
            pad_factor=self.pad_factor,
            bandlimit=self.bandlimit,
            upsample_factor=self.upsample_factor,
        )
        I_sensor = detect_intensity(E)
        stage_names.append("dec_sensor")
        stage_maps.append(I_sensor)

        x_hat = self._intensity_to_output(I_sensor)
        x_hat = self._align_channels(x_hat, self.out_channels)
        if x_hat.shape[-2:] != self.output_hw:
            x_hat = F.interpolate(x_hat, size=self.output_hw, mode="bilinear", align_corners=False)

        if return_info:
            return x_hat, {
                "stage_intensity_names": stage_names,
                "stage_intensity_maps": stage_maps,
                "recon_intensity": I_sensor,
            }
        return x_hat

    def decode(self, z_latent: torch.Tensor, return_info: bool = False):
        return self.forward(z_latent, return_info=return_info)


__all__ = ["IdentityAdapter", "OpticalOLSAdapter", "OpticalDiffractionDecoder"]
