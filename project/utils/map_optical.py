from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim

from models import OpticalOLSAdapter, VAEMapCore
from utils.losses_optical import sample_map_prior
from utils.metrics import mse_loss, psnr_from_mse
from utils.viz import save_image_grid, save_reconstruction_comparison


def build_map_core_from_cfg(cfg: dict, dataset_info: dict) -> VAEMapCore:
    model_cfg = cfg.get("model", {})
    return VAEMapCore(
        in_channels=int(dataset_info["in_channels"]),
        input_size=tuple(dataset_info["image_size"]),
        latent_channels=int(model_cfg.get("latent_channels", 16)),
        latent_hw=tuple(model_cfg.get("latent_hw", [4, 4])),
        encoder_channels=tuple(model_cfg.get("encoder_channels", [32, 64])),
        decoder_channels=tuple(model_cfg.get("decoder_channels", [256, 128, 64, 32])),
        decoder_mode=str(model_cfg.get("decoder_mode", "deconv")),
        out_range=str(cfg.get("data", {}).get("out_range", "zero_one")),
    )


def build_optical_adapter_from_cfg(cfg: dict, model: VAEMapCore) -> OpticalOLSAdapter:
    optics_cfg = cfg.get("optics", {})
    if not optics_cfg:
        raise ValueError("Optical config missing: require top-level 'optics' section")
    sensor_cfg = optics_cfg.get("sensor", {})
    resize_hw_cfg = optics_cfg.get("resize_hw", [model.latent_hw[0], model.latent_hw[1]])
    resize_hw = tuple(resize_hw_cfg) if resize_hw_cfg is not None else None

    return OpticalOLSAdapter(
        latent_shape=(model.latent_channels, model.latent_hw[0], model.latent_hw[1]),
        resize_hw=resize_hw,
        field_init_mode=str(optics_cfg.get("field_init_mode", "real")),
        wavelength_nm=float(optics_cfg.get("wavelength_nm", 532.0)),
        pixel_pitch_um=float(optics_cfg.get("pixel_pitch_um", 8.0)),
        z1_mm=float(optics_cfg.get("z1_mm", 20.0)),
        z2_mm=float(optics_cfg.get("z2_mm", 20.0)),
        pad_factor=float(optics_cfg.get("pad_factor", 2.0)),
        bandlimit=bool(optics_cfg.get("bandlimit", True)),
        upsample_factor=int(optics_cfg.get("upsample_factor", 1)),
        scatter_cfg=optics_cfg.get("scatter", {}),
        sensor_cfg=sensor_cfg,
    )


@torch.no_grad()
def evaluate_map_loader(
    model: VAEMapCore,
    adapter,
    loader,
    device: torch.device,
    data_range: float,
) -> dict:
    model.eval()
    adapter.eval()

    mse_sum = 0.0
    psnr_sum = 0.0
    count = 0

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        mu_map, logvar_map = model.encode(x)
        z_map = model.reparameterize(mu_map, logvar_map)
        z_mid = adapter(z_map)
        recon = model.decode(z_mid)

        mse_per_sample = mse_loss(recon, x, reduction="none")
        psnr_per_sample = psnr_from_mse(mse_per_sample, data_range=data_range)

        mse_sum += mse_per_sample.sum().item()
        psnr_sum += psnr_per_sample.sum().item()
        count += x.size(0)

    return {"mse": mse_sum / max(count, 1), "psnr": psnr_sum / max(count, 1)}


@torch.no_grad()
def save_optical_stage_visualization(
    inputs: torch.Tensor,
    optics_info: dict,
    path: Path,
    max_items: int = 6,
    out_range: str = "zero_one",
    logger=None,
) -> None:
    stages = optics_info.get("stage_intensity_maps", [])
    stage_names = optics_info.get("stage_intensity_names", [])
    final_latent = optics_info.get("final_latent_map", None)

    if len(stages) == 0 or final_latent is None:
        return

    def _prepare_image_row(x: torch.Tensor) -> torch.Tensor:
        x = x[:max_items, :1]
        if out_range == "neg_one_one":
            x = (x + 1.0) / 2.0
        return torch.clamp(x, 0.0, 1.0)

    def _prepare_map_row(x: torch.Tensor, use_log1p: bool = False) -> torch.Tensor:
        x = x[:max_items, :1]
        if use_log1p:
            x = torch.log1p(torch.clamp(x, min=0.0))
        xmin = x.amin(dim=(-2, -1), keepdim=True)
        xmax = x.amax(dim=(-2, -1), keepdim=True)
        return torch.clamp((x - xmin) / (xmax - xmin + 1e-6), 0.0, 1.0)

    def _stack_rows(rows):
        target_h = max(int(r.shape[-2]) for r in rows)
        target_w = max(int(r.shape[-1]) for r in rows)
        resized = []
        for r in rows:
            if r.shape[-2:] != (target_h, target_w):
                r = torch.nn.functional.interpolate(r, size=(target_h, target_w), mode="bilinear", align_corners=False)
            resized.append(r)
        return torch.cat(resized, dim=0)

    field_input = stages[stage_names.index("field_input")] if "field_input" in stage_names else stages[0]
    scatter = stages[stage_names.index("scatter")] if "scatter" in stage_names else stages[min(len(stages) - 1, 2)]

    rows = [
        _prepare_image_row(inputs),
        _prepare_map_row(field_input, use_log1p=True),
        _prepare_map_row(scatter, use_log1p=True),
        _prepare_map_row(final_latent, use_log1p=False),
    ]
    grid = _stack_rows(rows)
    save_image_grid(grid, path=path, nrow=max_items, out_range="zero_one")

    if logger is not None:
        logger.info(
            "optics.png semantics: 4 rows x %d cols, channel0 only; "
            "rows=[input_image, field_input_intensity(before_propagation), "
            "scatter_intensity(after_scatter), final_latent_map(input_to_decoder)], cols=samples",
            max_items,
        )


def build_decoder_prior_cfg(prior_cfg: dict, klw_cfg: dict) -> dict:
    mu0 = float(klw_cfg.get("m0", prior_cfg.get("mu0", 0.0)))
    sigma0 = float(klw_cfg.get("prior_sigma0", prior_cfg.get("sigma", 1.0)))
    return {
        "type": "biased_gaussian",
        "mu0": mu0,
        "sigma": sigma0,
        "spatial_smooth": prior_cfg.get("spatial_smooth", {}),
    }


@torch.no_grad()
def save_epoch_visuals_optical(
    model: VAEMapCore,
    adapter: OpticalOLSAdapter,
    epoch: int,
    fixed_inputs: torch.Tensor,
    val_loader,
    out_range: str,
    prior_cfg: dict,
    recon_dir: Path,
    sample_dir: Path,
    interp_dir: Path,
    optics_dir: Path,
    device: torch.device,
    sample_apply_smooth: bool = False,
    sample_prior_space: str = "z_map",
    decoder_prior_cfg: Optional[dict] = None,
    logger=None,
) -> None:
    model.eval()
    adapter.eval()

    mu_fixed, logvar_fixed = model.encode(fixed_inputs)
    z_fixed = model.reparameterize(mu_fixed, logvar_fixed)
    z_fixed_mid, info_fixed = adapter(z_fixed, return_info=True)
    recon_fixed = model.decode(z_fixed_mid)

    save_reconstruction_comparison(
        inputs=fixed_inputs,
        recons=recon_fixed,
        path=recon_dir / "epoch_{:03d}_recon.png".format(epoch),
        max_items=8,
        out_range=out_range,
    )

    save_optical_stage_visualization(
        inputs=fixed_inputs,
        optics_info=info_fixed,
        path=optics_dir / "epoch_{:03d}_optics.png".format(epoch),
        max_items=6,
        out_range=out_range,
        logger=logger,
    )

    sample_prior_space = str(sample_prior_space).lower()
    if sample_prior_space == "decoder":
        z_mid_prior = sample_map_prior(
            batch_size=64,
            latent_channels=model.latent_channels,
            latent_hw=model.latent_hw,
            prior_cfg=decoder_prior_cfg or prior_cfg,
            device=device,
            apply_smooth=sample_apply_smooth,
        )
        sampled = model.decode(z_mid_prior)
    else:
        z_prior = sample_map_prior(
            batch_size=64,
            latent_channels=model.latent_channels,
            latent_hw=model.latent_hw,
            prior_cfg=prior_cfg,
            device=device,
            apply_smooth=sample_apply_smooth,
        )
        sampled = model.decode(adapter(z_prior))
    save_image_grid(sampled, path=sample_dir / "epoch_{:03d}_sample.png".format(epoch), nrow=8, out_range=out_range)

    interp_batch = next(iter(val_loader))[0]
    if interp_batch.size(0) < 2:
        return

    idx = torch.randperm(interp_batch.size(0))[:2]
    pair = interp_batch[idx].to(device)
    mu_pair, _ = model.encode(pair)
    z1, z2 = mu_pair[0:1], mu_pair[1:2]

    alphas = torch.linspace(0.0, 1.0, steps=10, device=device).view(10, 1, 1, 1)
    z_interp = z1 * (1.0 - alphas) + z2 * alphas
    interp_images = model.decode(adapter(z_interp))
    save_image_grid(
        interp_images,
        path=interp_dir / "epoch_{:03d}_interp.png".format(epoch),
        nrow=10,
        out_range=out_range,
    )


def save_map_checkpoint(
    path: Path,
    model: VAEMapCore,
    adapter: OpticalOLSAdapter,
    optimizer: optim.Optimizer,
    epoch: int,
    cfg: dict,
    dataset_info: dict,
    trainable_params: int,
    epoch_metrics: dict,
) -> None:
    payload = {
        "epoch": epoch,
        "config": cfg,
        "dataset_info": dataset_info,
        "trainable_params": trainable_params,
        "model_state_dict": model.state_dict(),
        "adapter_state_dict": adapter.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch_metrics": epoch_metrics,
    }
    torch.save(payload, path)


__all__ = [
    "build_map_core_from_cfg",
    "build_optical_adapter_from_cfg",
    "evaluate_map_loader",
    "save_optical_stage_visualization",
    "build_decoder_prior_cfg",
    "save_epoch_visuals_optical",
    "save_map_checkpoint",
]
