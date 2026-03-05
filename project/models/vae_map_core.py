from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VAEMapCore(nn.Module):
    """
    Shared encoder/decoder map-latent VAE core.

    - encode(x) -> (mu_map, logvar_map) with shape [B, C_lat, H_lat, W_lat]
    - reparameterize(mu_map, logvar_map) -> z_map
    - decode(z_map) -> x_hat
    """

    def __init__(
        self,
        in_channels: int,
        input_size: Tuple[int, int],
        latent_channels: int,
        latent_hw: Tuple[int, int],
        encoder_channels: Sequence[int] = (32, 64),
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        decoder_mode: str = "deconv",
        out_range: str = "zero_one",
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.latent_channels = int(latent_channels)
        self.latent_hw = (int(latent_hw[0]), int(latent_hw[1]))
        self.decoder_mode = str(decoder_mode).lower()
        self.out_range = out_range

        if not encoder_channels:
            raise ValueError("encoder_channels cannot be empty")

        enc_layers = []
        prev_c = self.in_channels
        for out_c in encoder_channels:
            enc_layers.append(ConvBlock(prev_c, int(out_c), stride=2))
            prev_c = int(out_c)
        self.encoder = nn.Sequential(*enc_layers)

        self.enc_to_mu = nn.Conv2d(prev_c, self.latent_channels, kernel_size=1)
        self.enc_to_logvar = nn.Conv2d(prev_c, self.latent_channels, kernel_size=1)

        dec_layers = []
        prev_c = self.latent_channels
        if self.decoder_mode == "deconv":
            for out_c in decoder_channels:
                dec_layers.append(DeconvBlock(prev_c, int(out_c)))
                prev_c = int(out_c)
            self.decoder = nn.Sequential(*dec_layers)
        elif self.decoder_mode == "conv_refine":
            for out_c in decoder_channels:
                dec_layers.append(ConvBlock(prev_c, int(out_c), stride=1))
                prev_c = int(out_c)
            self.decoder = nn.Sequential(*dec_layers)
        else:
            raise ValueError("Unsupported decoder_mode: {}".format(self.decoder_mode))

        self.dec_out = nn.Conv2d(prev_c, self.in_channels, kernel_size=3, stride=1, padding=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = F.adaptive_avg_pool2d(h, self.latent_hw)
        mu_map = self.enc_to_mu(h)
        logvar_map = self.enc_to_logvar(h)
        return mu_map, logvar_map

    def reparameterize(self, mu_map: torch.Tensor, logvar_map: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar_map)
        eps = torch.randn_like(std)
        return mu_map + eps * std

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_range == "zero_one":
            return torch.sigmoid(x)
        if self.out_range == "neg_one_one":
            return torch.tanh(x)
        raise ValueError("Unsupported out_range: {}".format(self.out_range))

    def decode(self, z_map: torch.Tensor) -> torch.Tensor:
        h = self.decoder(z_map)
        x_hat = self.dec_out(h)
        if x_hat.shape[-2:] != self.input_size:
            x_hat = F.interpolate(x_hat, size=self.input_size, mode="bilinear", align_corners=False)
        return self._apply_output_activation(x_hat)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_map, logvar_map = self.encode(x)
        z_map = self.reparameterize(mu_map, logvar_map)
        x_hat = self.decode(z_map)
        return x_hat, mu_map, logvar_map, z_map


__all__ = ["VAEMapCore"]
