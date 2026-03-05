from __future__ import annotations

from typing import Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvVAE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        input_size: Tuple[int, int],
        latent_dim: int = 100,
        model_size: Literal["tiny", "small"] = "tiny",
        out_range: Literal["zero_one", "neg_one_one"] = "zero_one",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.model_size = model_size
        self.out_range = out_range

        if model_size == "tiny":
            # Fair-comparison electronic baseline: ~24k trainable params (latent_dim=64).
            channels = [4, 6, 14]
            hidden_dim = 32
        elif model_size == "small":
            channels = [6, 10, 16]
            hidden_dim = 48
        else:
            raise ValueError(f"Unsupported model_size: {model_size}")

        self.channels = channels

        encoder_layers: list[nn.Module] = []
        prev_c = in_channels
        for c in channels:
            encoder_layers.extend(
                [
                    nn.Conv2d(prev_c, c, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                ]
            )
            prev_c = c
        self.encoder = nn.Sequential(*encoder_layers)

        prev_training = self.encoder.training
        self.encoder.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, *input_size)
            enc_out = self.encoder(dummy)
        self.encoder.train(prev_training)
        self.enc_feature_shape = tuple(enc_out.shape[1:])
        self.enc_feature_dim = int(enc_out[0].numel())

        self.fc_enc = nn.Linear(self.enc_feature_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, hidden_dim)
        self.fc_unflatten = nn.Linear(hidden_dim, self.enc_feature_dim)

        decoder_layers: list[nn.Module] = []
        reverse_channels = list(reversed(channels))
        prev_c = reverse_channels[0]
        for c in reverse_channels[1:]:
            decoder_layers.extend(
                [
                    nn.ConvTranspose2d(prev_c, c, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(c),
                    nn.ReLU(inplace=True),
                ]
            )
            prev_c = c

        decoder_layers.append(nn.ConvTranspose2d(prev_c, in_channels, kernel_size=4, stride=2, padding=1))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        h = h.flatten(start_dim=1)
        h = F.relu(self.fc_enc(h), inplace=True)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def _apply_output_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_range == "zero_one":
            return torch.sigmoid(x)
        if self.out_range == "neg_one_one":
            return torch.tanh(x)
        raise ValueError(f"Unsupported out_range: {self.out_range}")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc_dec(z), inplace=True)
        h = self.fc_unflatten(h)
        h = h.view(-1, *self.enc_feature_shape)
        x_hat = self.decoder(h)

        if x_hat.shape[-2:] != self.input_size:
            x_hat = F.interpolate(x_hat, size=self.input_size, mode="bilinear", align_corners=False)

        return self._apply_output_activation(x_hat)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z

    def sample_prior(self, num_samples: int, device: torch.device) -> torch.Tensor:
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)
