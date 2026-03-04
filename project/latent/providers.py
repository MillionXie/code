from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class LatentBatch:
    z: torch.Tensor
    mu: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None


class LatentProvider(ABC):
    """Unified latent provider interface for future extensions (e.g. scattering latents)."""

    @abstractmethod
    def get_latent(
        self,
        model: Optional[torch.nn.Module] = None,
        x: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> LatentBatch:
        raise NotImplementedError


class GaussianPriorProvider(LatentProvider):
    """z ~ N(0, I)."""

    def __init__(self, latent_dim: int):
        self.latent_dim = latent_dim

    def get_latent(
        self,
        model: Optional[torch.nn.Module] = None,
        x: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> LatentBatch:
        if num_samples is None:
            if x is None:
                raise ValueError("Either x or num_samples must be provided for GaussianPriorProvider.")
            num_samples = x.size(0)

        if device is None:
            if x is not None:
                device = x.device
            elif model is not None:
                device = next(model.parameters()).device
            else:
                device = torch.device("cpu")

        z = torch.randn(num_samples, self.latent_dim, device=device)
        return LatentBatch(z=z)


class EncoderPosteriorProvider(LatentProvider):
    """Latents from encoder posterior q(z|x)."""

    def __init__(self, sample: bool = True):
        self.sample = sample

    def get_latent(
        self,
        model: Optional[torch.nn.Module] = None,
        x: Optional[torch.Tensor] = None,
        num_samples: Optional[int] = None,
        device: Optional[torch.device] = None,
    ) -> LatentBatch:
        del num_samples, device
        if model is None:
            raise ValueError("model must be provided for EncoderPosteriorProvider.")
        if x is None:
            raise ValueError("x must be provided for EncoderPosteriorProvider.")

        mu, logvar = model.encode(x)
        z = model.reparameterize(mu, logvar) if self.sample else mu
        return LatentBatch(z=z, mu=mu, logvar=logvar)
