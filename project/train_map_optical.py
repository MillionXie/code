from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from data.datasets import get_dataloaders
from utils.config import apply_cli_overrides, load_config
from utils.io import append_row_csv, count_trainable_params, ensure_dir, now_timestamp, save_json, write_summary_csv
from utils.logger import create_logger
from utils.losses_optical import (
    compute_optical_penalty,
    compute_recon_per_sample,
    kl_latent_intensity_biased_gaussian,
    resolve_recon_loss,
)
from utils.map_optical import (
    build_decoder_prior_cfg,
    build_map_core_from_cfg,
    build_optical_adapter_from_cfg,
    evaluate_map_loader,
    save_epoch_visuals_optical,
    save_map_checkpoint,
    save_optical_phase_parameters,
)
from utils.seed import set_seed


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
    optics_cfg = cfg.get("optics", {})

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
    phase_dir = ensure_dir(run_dir / "optics_params")
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

    model = build_map_core_from_cfg(cfg, dataset_info).to(device)
    adapter = build_optical_adapter_from_cfg(cfg, model).to(device)
    model_arch = str(cfg.get("model", {}).get("arch", "conv")).lower()
    if hasattr(model, "encoder") and hasattr(model, "enc_to_mu") and hasattr(model, "enc_to_logvar"):
        # Backward-compatible path: conv decoder model instance.
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.enc_to_mu.parameters():
            p.requires_grad = False
        for p in model.enc_to_logvar.parameters():
            p.requires_grad = False
    scatter_cfg = optics_cfg.get("scatter", {})
    sensor_cfg = optics_cfg.get("sensor", {})
    diff_cfg = optics_cfg.get("diffractive_layers", {})
    diff_z_cfg = diff_cfg.get("z_mm", [optics_cfg.get("z1_mm", 20.0), optics_cfg.get("z1_mm", 20.0)])
    resize_hw_cfg = optics_cfg.get("resize_hw", [model.latent_hw[0], model.latent_hw[1]])
    resize_hw_display = tuple(resize_hw_cfg) if resize_hw_cfg is not None else None
    logger.info(
        "Optics | field_init_mode=%s resize_hw=%s pixel_pitch_um=%.3f wavelength_nm=%.1f pad_factor=%s bandlimit=%s upsample_factor=%s",
        str(optics_cfg.get("field_init_mode", "real")),
        resize_hw_display,
        float(optics_cfg.get("pixel_pitch_um", 8.0)),
        float(optics_cfg.get("wavelength_nm", 532.0)),
        float(optics_cfg.get("pad_factor", 2.0)),
        bool(optics_cfg.get("bandlimit", True)),
        int(optics_cfg.get("upsample_factor", 1)),
    )
    logger.info(
        "DiffractiveEncoder | phase_layers=2 trainable=%s init=%s z_mm=%s z_to_scatter_mm=%s z_to_sensor_mm=%s posterior_sigma=%s",
        bool(diff_cfg.get("trainable", True)),
        str(diff_cfg.get("init", "uniform")),
        tuple(diff_z_cfg),
        float(optics_cfg.get("z_to_scatter_mm", optics_cfg.get("z2_mm", 20.0))),
        float(optics_cfg.get("z_to_sensor_mm", 1.0)),
        float(optics_cfg.get("posterior_sigma", loss_cfg.get("posterior_sigma", 0.1))),
    )
    if model_arch in ("pure_optical", "optical_only", "optical"):
        dec_cfg = optics_cfg.get("decoder", {})
        logger.info(
            "DiffractiveDecoder | phase_layers=4 field_hw=%s latent_to_field_mode=%s layer_z_mm=%s z_to_sensor_mm=%s",
            tuple(dec_cfg.get("resize_hw", optics_cfg.get("resize_hw", data_cfg.get("image_size", [64, 64])))),
            str(dec_cfg.get("latent_to_field_mode", "repeat")),
            tuple(dec_cfg.get("layer_z_mm", [20.0, 20.0, 20.0, 20.0])),
            float(dec_cfg.get("z_to_sensor_mm", 20.0)),
        )
    logger.info(
        "Scatter+Sensor | scatter_type=%s scatter.static=%s latent_hw=%s pool_type=%s pool_kernel=%s pool_stride=%s pool_reduce=%s",
        str(scatter_cfg.get("type", "iid_phase")),
        bool(getattr(adapter.scatterer, "static", scatter_cfg.get("static", True))),
        tuple(model.latent_hw),
        str(sensor_cfg.get("pool_type", "avg")),
        sensor_cfg.get("pool_kernel", 2),
        sensor_cfg.get("pool_stride", 2),
        str(sensor_cfg.get("pool_reduce", "mean")),
    )

    trainable_params = count_trainable_params(model) + count_trainable_params(adapter)
    logger.info("Trainable parameters: %d", trainable_params)
    if model_arch in ("pure_optical", "optical_only", "optical"):
        logger.info("Model trainable modules: optical encoder + optical decoder (pure optical VAE)")
        logger.info(
            "Optical mode model config usage: latent_channels/latent_hw and optics.decoder.* are active; "
            "conv encoder/decoder settings are ignored."
        )
    else:
        logger.info("Model trainable modules: decoder + optical adapter (encoder frozen)")
        logger.info(
            "Optical mode model config usage: latent_channels/latent_hw/decoder_* are active; "
            "encoder_channels are kept for compatibility but not used in optical forward."
        )

    train_params = [p for p in model.parameters() if p.requires_grad] + [p for p in adapter.parameters() if p.requires_grad]
    optimizer = optim.Adam(train_params, lr=float(train_cfg.get("lr", 1e-3)))

    alpha = float(loss_cfg.get("alpha", 1.0))
    beta = float(loss_cfg.get("beta", 1.0))
    gamma = float(loss_cfg.get("gamma", 0.1))
    penalty_mode = str(loss_cfg.get("optical_penalty", {}).get("mode", "tv"))
    posterior_sigma = float(loss_cfg.get("posterior_sigma", optics_cfg.get("posterior_sigma", 0.1)))
    if posterior_sigma < 0:
        raise ValueError("loss.posterior_sigma must be >= 0")

    kl_var_mode = str(klw_cfg.get("var_mode", "constant"))
    kl_target = str(klw_cfg.get("target", "latent_mean")).lower()
    if kl_target == "final_latent":
        kl_target = "latent_mean"
    if kl_target not in ("latent_mean", "latent_intensity"):
        raise ValueError("loss.kl_w.target must be one of: latent_mean|latent_intensity")
    kl_m0 = float(klw_cfg.get("m0", prior_cfg.get("mu0", 0.25)))
    kl_prior_sigma0 = float(klw_cfg.get("prior_sigma0", prior_cfg.get("sigma", 1.0)))
    kl_pre_norm = str(klw_cfg.get("pre_norm", "mean"))
    kl_var0 = float(klw_cfg.get("var0", posterior_sigma * posterior_sigma if posterior_sigma > 0 else 1e-8))
    kl_clamp_nonnegative = bool(klw_cfg.get("clamp_nonnegative", True))

    sample_apply_smooth_in_train = bool(prior_cfg.get("apply_smooth_in_train", False))
    sample_prior_space = "decoder"
    decoder_prior_cfg = build_decoder_prior_cfg(prior_cfg=prior_cfg, klw_cfg=klw_cfg)

    logger.info(
        "KL_w | target=%s var_mode=%s var0=%.6f M0=%.6f prior_sigma0=%.6f pre_norm=%s clamp_nonnegative=%s posterior_sigma=%s",
        kl_target,
        kl_var_mode,
        kl_var0,
        kl_m0,
        kl_prior_sigma0,
        kl_pre_norm,
        kl_clamp_nonnegative,
        posterior_sigma,
    )
    logger.info(
        "Sampling | prior_space=%s P(z)=N(mu0=%.6f, sigma=%.6f)",
        sample_prior_space,
        float(decoder_prior_cfg.get("mu0", 0.0)),
        float(decoder_prior_cfg.get("sigma", 1.0)),
    )
    if kl_pre_norm != "none":
        logger.warning(
            "KL pre_norm=%s may improve stability but can weaken absolute-scale matching to sampling prior. "
            "If sampling quality is poor, try loss.kl_w.pre_norm=none.",
            kl_pre_norm,
        )

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

            z_mid, optics_info = adapter.encode_from_input(
                x,
                return_info=True,
                sample_posterior=True,
                posterior_sigma=posterior_sigma,
            )
            recon = model.decode(z_mid)

            if not torch.isfinite(recon).all():
                raise RuntimeError("Non-finite recon detected")

            recon_per_sample = compute_recon_per_sample(recon, x, recon_loss_type)
            latent_intensity_map = optics_info["latent_intensity_map"]
            kl_input_map = optics_info.get("latent_mean_map", latent_intensity_map)

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
        val_metrics = evaluate_map_loader(model, adapter, val_loader, device, data_range)

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

        save_epoch_visuals_optical(
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
            sample_apply_smooth=sample_apply_smooth_in_train,
            sample_prior_space=sample_prior_space,
            decoder_prior_cfg=decoder_prior_cfg,
            logger=logger,
        )
        save_optical_phase_parameters(
            epoch=epoch,
            adapter=adapter,
            model=model,
            outdir=phase_dir,
            logger=logger,
        )

        save_map_checkpoint(
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
            save_map_checkpoint(
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
    test_metrics = evaluate_map_loader(model, adapter, test_loader, device, data_range)

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
