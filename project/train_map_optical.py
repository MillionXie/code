from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import torch
import torch.optim as optim
from tqdm import tqdm

from data.datasets import get_dataloaders
from models import OpticalOLSAdapter, VAEMapCore
from utils.config import apply_cli_overrides, load_config
from utils.io import append_row_csv, count_trainable_params, ensure_dir, now_timestamp, save_json, write_summary_csv
from utils.logger import create_logger
from utils.losses_optical import (
    compute_optical_penalty,
    compute_recon_per_sample,
    kl_latent_intensity_biased_gaussian,
    resolve_recon_loss,
    sample_map_prior,
)
from utils.metrics import mse_loss, psnr_from_mse
from utils.seed import set_seed
from utils.viz import save_image_grid, save_reconstruction_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train map-latent optical baseline (Optical OLS adapter)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model_from_cfg(cfg: dict, dataset_info: dict) -> VAEMapCore:
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


def build_optical_adapter(cfg: dict, model: VAEMapCore) -> OpticalOLSAdapter:
    optics_cfg = cfg.get("optics", {})
    if not optics_cfg:
        raise ValueError("Optical config missing: require top-level 'optics' section")
    direct_mode = str(optics_cfg.get("direct_mode", "latent_hw")).lower()
    sensor_cfg = optics_cfg.get("sensor", {})
    if direct_mode == "latent_hw":
        out_hw_cfg = sensor_cfg.get("out_hw", None)
        if out_hw_cfg is not None:
            out_hw = tuple(out_hw_cfg)
            if out_hw != tuple(model.latent_hw):
                raise ValueError(
                    "optics.sensor.out_hw {} must equal model.latent_hw {} when explicitly enabled in optics.direct_mode='latent_hw'".format(
                        out_hw, tuple(model.latent_hw)
                    )
                )
    return OpticalOLSAdapter(
        latent_shape=(model.latent_channels, model.latent_hw[0], model.latent_hw[1]),
        resize_hw=tuple(optics_cfg.get("resize_hw", [model.latent_hw[0], model.latent_hw[1]])),
        field_init_mode=str(optics_cfg.get("field_init_mode", "real")),
        direct_mode=direct_mode,
        wavelength_nm=float(optics_cfg.get("wavelength_nm", 532.0)),
        pixel_pitch_um=float(optics_cfg.get("pixel_pitch_um", 8.0)),
        z1_mm=float(optics_cfg.get("z1_mm", 20.0)),
        z2_mm=float(optics_cfg.get("z2_mm", 20.0)),
        pad_factor=float(optics_cfg.get("pad_factor", 2.0)),
        bandlimit=bool(optics_cfg.get("bandlimit", True)),
        upsample_factor=int(optics_cfg.get("upsample_factor", 1)),
        scatter_cfg=optics_cfg.get("scatter", {}),
        sensor_cfg=sensor_cfg,
        output_center=bool(optics_cfg.get("output_center", True)),
    )


@torch.no_grad()
def evaluate_loader(model: VAEMapCore, adapter: OpticalOLSAdapter, loader, device: torch.device, data_range: float) -> dict:
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
    optics_show: str = "simple",
    logger=None,
) -> None:
    stages = optics_info.get("stage_intensity_maps", [])
    stage_names = optics_info.get("stage_intensity_names", [])
    final_latent = optics_info.get("final_latent_map", None)

    if len(stages) == 0 or final_latent is None:
        return

    def _prepare_image_row(x: torch.Tensor) -> torch.Tensor:
        x = x[:max_items, :1]  # channel0 only
        if out_range == "neg_one_one":
            x = (x + 1.0) / 2.0
        x = torch.clamp(x, 0.0, 1.0)
        return x

    def _prepare_map_row(x: torch.Tensor, use_log1p: bool = False) -> torch.Tensor:
        x = x[:max_items, :1]  # channel0 only
        if use_log1p:
            x = torch.log1p(torch.clamp(x, min=0.0))
        xmin = x.amin(dim=(-2, -1), keepdim=True)
        xmax = x.amax(dim=(-2, -1), keepdim=True)
        x = (x - xmin) / (xmax - xmin + 1e-6)
        x = torch.clamp(x, 0.0, 1.0)
        return x

    def _stack_rows(rows: List[torch.Tensor]) -> torch.Tensor:
        target_h = max(int(r.shape[-2]) for r in rows)
        target_w = max(int(r.shape[-1]) for r in rows)
        resized = []
        for r in rows:
            if r.shape[-2:] != (target_h, target_w):
                r = torch.nn.functional.interpolate(r, size=(target_h, target_w), mode="bilinear", align_corners=False)
            resized.append(r)
        return torch.cat(resized, dim=0)

    field_input = stages[stage_names.index("input")] if "input" in stage_names else stages[0]
    scatter = stages[stage_names.index("scatter")] if "scatter" in stage_names else stages[min(len(stages) - 1, 2)]
    final_latent = final_latent

    # Fixed 4-row visualization:
    # 1) input image, 2) field input intensity(before propagation), 3) scatter intensity, 4) final latent for decoder.
    rows = [
        _prepare_image_row(inputs),
        _prepare_map_row(field_input, use_log1p=True),
        _prepare_map_row(scatter, use_log1p=True),
        _prepare_map_row(final_latent, use_log1p=False),
    ]
    main_grid = _stack_rows(rows)
    save_image_grid(main_grid, path=path, nrow=max_items, out_range="zero_one")

    if logger is not None:
        logger.info(
            "optics.png semantics: 4 rows x %d cols, channel0 only; "
            "rows=[input_image, field_input_intensity(before_propagation), scatter_intensity(after_scatter), final_latent_map(input_to_decoder)], cols=samples",
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
def save_epoch_visuals(
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
    optics_show: str = "stage",
    sample_apply_smooth: bool = False,
    sample_prior_space: str = "z_map",
    decoder_prior_cfg: Optional[dict] = None,
    direct_mode: str = "latent_hw",
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
        optics_show=optics_show,
        logger=logger,
    )

    sample_prior_space = str(sample_prior_space).lower()
    if sample_prior_space == "decoder" and str(direct_mode).lower() == "latent_hw":
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


def save_checkpoint(
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


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(
        cfg,
        dataset=args.dataset,
        outdir=args.outdir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    dataset = str(cfg.get("dataset", "mnist")).lower()
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})
    prior_cfg = loss_cfg.get("prior", {"type": "biased_gaussian", "mu0": 0.25, "sigma": 1.0})
    klw_cfg = loss_cfg.get("kl_w", {})
    viz_cfg = cfg.get("viz", {})

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    out_range = str(data_cfg.get("out_range", "zero_one"))
    recon_loss_type = resolve_recon_loss(dataset, str(train_cfg.get("recon_loss", "auto")))

    if recon_loss_type == "bce" and out_range != "zero_one":
        raise ValueError("BCE requires out_range=zero_one")

    if cfg.get("outdir") is None:
        cfg["outdir"] = str(Path("outputs") / "map_optical_{}_{}".format(dataset, now_timestamp()))

    run_dir = ensure_dir(cfg["outdir"])
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    recon_dir = ensure_dir(run_dir / "reconstructions")
    sample_dir = ensure_dir(run_dir / "samples")
    interp_dir = ensure_dir(run_dir / "interpolations")
    optics_dir = ensure_dir(run_dir / "optics")
    log_dir = ensure_dir(run_dir / "logs")

    logger = create_logger("train_map_optical", outdir=log_dir, filename="train_map_optical.log")
    logger.info("Config: %s", cfg)

    device = select_device()
    logger.info("Device: %s", device)

    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        dataset=dataset,
        data_root=args.data_root,
        batch_size=int(train_cfg.get("batch_size", 128)),
        num_workers=int(train_cfg.get("num_workers", 4)),
        out_range=out_range,
        seed=seed,
        image_size=data_cfg.get("image_size", [64, 64]),
    )

    model = build_model_from_cfg(cfg, dataset_info).to(device)
    adapter = build_optical_adapter(cfg, model).to(device)
    if str(cfg.get("optics", {}).get("direct_mode", "latent_hw")).lower() == "full_res" and model.decoder_mode != "conv_refine":
        logger.info(
            "Warning: optics.direct_mode=full_res is typically paired with model.decoder_mode=conv_refine; current decoder_mode=%s",
            model.decoder_mode,
        )

    optics_cfg = cfg.get("optics", {})
    scatter_cfg = optics_cfg.get("scatter", {})
    sensor_cfg = optics_cfg.get("sensor", {})
    logger.info(
        "Optics | field_init_mode=%s scatter_type=%s scatter.static=%s resize_hw=%s pad_factor=%s bandlimit=%s upsample_factor=%s direct_mode=%s",
        str(optics_cfg.get("field_init_mode", "real")),
        str(scatter_cfg.get("type", "iid_phase")),
        bool(getattr(adapter.scatterer, "static", scatter_cfg.get("static", True))),
        tuple(optics_cfg.get("resize_hw", [model.latent_hw[0], model.latent_hw[1]])),
        float(optics_cfg.get("pad_factor", 2.0)),
        bool(optics_cfg.get("bandlimit", True)),
        int(optics_cfg.get("upsample_factor", 1)),
        str(optics_cfg.get("direct_mode", "latent_hw")),
    )
    logger.info(
        "Sensor | latent_hw=%s roi_hw=%s pool_type=%s pool_kernel=%s pool_stride=%s pool_reduce=%s out_hw=%s",
        tuple(model.latent_hw),
        sensor_cfg.get("roi_hw", None),
        str(sensor_cfg.get("pool_type", "avg")),
        sensor_cfg.get("pool_kernel", 2),
        sensor_cfg.get("pool_stride", 2),
        str(sensor_cfg.get("pool_reduce", "mean")),
        sensor_cfg.get("out_hw", None),
    )

    trainable_params = count_trainable_params(model) + count_trainable_params(adapter)
    logger.info("Trainable parameters: %d", trainable_params)

    lr = float(train_cfg.get("lr", 1e-3))
    optimizer = optim.Adam(list(model.parameters()) + list(adapter.parameters()), lr=lr)

    alpha = float(loss_cfg.get("alpha", 1.0))
    beta = float(loss_cfg.get("beta", 1.0))
    gamma = float(loss_cfg.get("gamma", 0.1))
    penalty_mode = str(loss_cfg.get("optical_penalty", {}).get("mode", "tv"))
    kl_var_mode = str(klw_cfg.get("var_mode", "constant"))
    kl_target = str(klw_cfg.get("target", "final_latent")).lower()
    kl_m0 = float(klw_cfg.get("m0", prior_cfg.get("mu0", 0.25)))
    kl_prior_sigma0 = float(klw_cfg.get("prior_sigma0", prior_cfg.get("sigma", 1.0)))
    if "pre_norm" in klw_cfg:
        kl_pre_norm = str(klw_cfg.get("pre_norm"))
    else:
        kl_pre_norm = "none" if kl_target == "final_latent" else "mean"
    kl_var0 = float(klw_cfg.get("var0", kl_prior_sigma0 * kl_prior_sigma0))
    kl_clamp_nonnegative = bool(klw_cfg.get("clamp_nonnegative", kl_target != "final_latent"))
    if kl_target not in ("latent_intensity", "final_latent"):
        raise ValueError("loss.kl_w.target must be one of: latent_intensity|final_latent")

    sample_apply_smooth_in_train = bool(prior_cfg.get("apply_smooth_in_train", False))
    direct_mode = str(optics_cfg.get("direct_mode", "latent_hw")).lower()
    sample_cfg = cfg.get("sample", {})
    sample_prior_space = str(sample_cfg.get("prior_space", "auto")).lower()
    if sample_prior_space == "auto":
        sample_prior_space = "decoder" if (kl_target == "final_latent" and direct_mode == "latent_hw") else "z_map"
    if sample_prior_space not in ("decoder", "z_map"):
        raise ValueError("sample.prior_space must be one of: auto|decoder|z_map")
    if sample_prior_space == "decoder" and direct_mode != "latent_hw":
        logger.info(
            "sample.prior_space=decoder is only supported in optics.direct_mode=latent_hw. Falling back to z_map sampling."
        )
        sample_prior_space = "z_map"
    decoder_prior_cfg = build_decoder_prior_cfg(prior_cfg=prior_cfg, klw_cfg=klw_cfg)

    optics_show = str(viz_cfg.get("optics_show", "stage")).lower()
    if optics_show not in ("simple", "stage", "sensor_roi", "sensor_pooled", "both"):
        raise ValueError("viz.optics_show must be one of: simple|stage|sensor_roi|sensor_pooled|both")
    logger.info(
        "KL_w | target=%s var_mode=%s var0=%.6f M0=%.6f prior_sigma0=%.6f pre_norm=%s clamp_nonnegative=%s",
        kl_target,
        kl_var_mode,
        kl_var0,
        kl_m0,
        kl_prior_sigma0,
        kl_pre_norm,
        kl_clamp_nonnegative,
    )
    logger.info("Sampling | prior_space=%s", sample_prior_space)

    epochs = int(train_cfg.get("epochs", 30))
    data_range = float(dataset_info.get("data_range", 1.0))

    fixed_inputs = next(iter(val_loader))[0][:8].to(device)

    history_csv = run_dir / "history.csv"
    history = []
    best_val_mse = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        model.train()
        adapter.train()

        loss_sum = 0.0
        recon_sum = 0.0
        kl_sum = 0.0
        op_sum = 0.0
        count = 0
        latent_w_min = float("inf")
        latent_w_max = float("-inf")
        latent_w_mean_sum = 0.0
        latent_w_std_sum = 0.0

        pbar = tqdm(train_loader, desc="Epoch {}/{}".format(epoch, epochs), leave=False)
        for x, _ in pbar:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            mu_map, logvar_map = model.encode(x)
            z_map = model.reparameterize(mu_map, logvar_map)
            z_mid, optics_info = adapter(z_map, return_info=True)
            recon = model.decode(z_mid)

            if not torch.isfinite(recon).all():
                raise RuntimeError("Non-finite recon detected")

            recon_per_sample = compute_recon_per_sample(recon, x, recon_loss_type)
            latent_intensity_map = optics_info["latent_intensity_map"]
            if kl_target == "final_latent":
                kl_input_map = optics_info["final_latent_map"]
            else:
                kl_input_map = latent_intensity_map
            kl_per_sample = kl_latent_intensity_biased_gaussian(
                latent_intensity_map=kl_input_map,
                var_mode=kl_var_mode,
                var0=kl_var0,
                prior_mean_m0=kl_m0,
                prior_sigma0=kl_prior_sigma0,
                pre_norm=kl_pre_norm,
                clamp_nonnegative=kl_clamp_nonnegative,
                reduction="none",
            )
            op_per_sample = compute_optical_penalty(
                optics_info.get("stage_intensity_maps", []),
                mode=penalty_mode,
                reduction="none",
            )

            loss_per_sample = alpha * kl_per_sample + beta * recon_per_sample + gamma * op_per_sample
            loss = loss_per_sample.mean()

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss detected")

            loss.backward()
            optimizer.step()

            bsz = x.size(0)
            count += bsz
            loss_sum += loss_per_sample.sum().item()
            recon_sum += recon_per_sample.sum().item()
            kl_sum += kl_per_sample.sum().item()
            op_sum += op_per_sample.sum().item()
            latent_w_min = min(latent_w_min, float(latent_intensity_map.min().item()))
            latent_w_max = max(latent_w_max, float(latent_intensity_map.max().item()))
            latent_w_mean_sum += float(latent_intensity_map.mean().item()) * bsz
            latent_w_std_sum += float(latent_intensity_map.std(unbiased=False).item()) * bsz

            pbar.set_postfix(
                loss="{:.4f}".format(loss_sum / max(count, 1)),
                recon="{:.4f}".format(recon_sum / max(count, 1)),
                kl="{:.4f}".format(kl_sum / max(count, 1)),
                op="{:.4f}".format(op_sum / max(count, 1)),
            )

        train_metrics = {
            "loss": loss_sum / max(count, 1),
            "recon": recon_sum / max(count, 1),
            "kl": kl_sum / max(count, 1),
            "op": op_sum / max(count, 1),
            "latent_w_min": latent_w_min if count > 0 else 0.0,
            "latent_w_max": latent_w_max if count > 0 else 0.0,
            "latent_w_mean": latent_w_mean_sum / max(count, 1),
            "latent_w_std": latent_w_std_sum / max(count, 1),
        }
        val_metrics = evaluate_loader(model, adapter, val_loader, device, data_range)

        row = {
            "epoch": epoch,
            "loss": train_metrics["loss"],
            "recon": train_metrics["recon"],
            "kl": train_metrics["kl"],
            "op": train_metrics["op"],
            "latent_w_min": train_metrics["latent_w_min"],
            "latent_w_max": train_metrics["latent_w_max"],
            "latent_w_mean": train_metrics["latent_w_mean"],
            "latent_w_std": train_metrics["latent_w_std"],
            "val_mse": val_metrics["mse"],
            "val_psnr": val_metrics["psnr"],
        }
        history.append(row)
        append_row_csv(history_csv, row)

        logger.info(
            "Epoch %d/%d | loss=%.6f recon=%.6f kl=%.6f op=%.6f val_mse=%.6f val_psnr=%.3f latent_w[min=%.6f max=%.6f mean=%.6f std=%.6f]",
            epoch,
            epochs,
            row["loss"],
            row["recon"],
            row["kl"],
            row["op"],
            row["val_mse"],
            row["val_psnr"],
            row["latent_w_min"],
            row["latent_w_max"],
            row["latent_w_mean"],
            row["latent_w_std"],
        )

        save_epoch_visuals(
            model=model,
            adapter=adapter,
            epoch=epoch,
            fixed_inputs=fixed_inputs,
            val_loader=val_loader,
            out_range=out_range,
            prior_cfg=prior_cfg,
            recon_dir=recon_dir,
            sample_dir=sample_dir,
            interp_dir=interp_dir,
            optics_dir=optics_dir,
            device=device,
            optics_show=optics_show,
            sample_apply_smooth=sample_apply_smooth_in_train,
            sample_prior_space=sample_prior_space,
            decoder_prior_cfg=decoder_prior_cfg,
            direct_mode=direct_mode,
            logger=logger,
        )

        save_checkpoint(
            path=ckpt_dir / "last.pt",
            model=model,
            adapter=adapter,
            optimizer=optimizer,
            epoch=epoch,
            cfg=cfg,
            dataset_info=dataset_info,
            trainable_params=trainable_params,
            epoch_metrics=row,
        )

        if row["val_mse"] < best_val_mse:
            best_val_mse = row["val_mse"]
            best_epoch = epoch
            save_checkpoint(
                path=ckpt_dir / "best.pt",
                model=model,
                adapter=adapter,
                optimizer=optimizer,
                epoch=epoch,
                cfg=cfg,
                dataset_info=dataset_info,
                trainable_params=trainable_params,
                epoch_metrics=row,
            )

    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    adapter.load_state_dict(best_ckpt["adapter_state_dict"])

    test_metrics = evaluate_loader(model, adapter, test_loader, device, data_range)

    results = {
        "config": cfg,
        "trainable_params": trainable_params,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "test_mse": test_metrics["mse"],
        "test_psnr": test_metrics["psnr"],
        "history": history,
    }
    save_json(results, run_dir / "results.json")

    summary_row = {
        "dataset": dataset,
        "mode": "map_optical",
        "epochs": epochs,
        "latent_channels": model.latent_channels,
        "latent_h": model.latent_hw[0],
        "latent_w": model.latent_hw[1],
        "trainable_params": trainable_params,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "test_mse": test_metrics["mse"],
        "test_psnr": test_metrics["psnr"],
        "outdir": str(run_dir),
    }
    write_summary_csv(run_dir / "summary.csv", [summary_row])

    logger.info("Training completed. Best epoch: %d", best_epoch)
    logger.info("Artifacts saved to: %s", run_dir)


if __name__ == "__main__":
    main()
