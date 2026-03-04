from __future__ import annotations

import argparse
from pathlib import Path

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
    kl_map_gaussian_prior,
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
        encoder_channels=tuple(model_cfg.get("encoder_channels", [32, 64, 128])),
        decoder_channels=tuple(model_cfg.get("decoder_channels", [128, 64])),
        out_range=str(cfg.get("data", {}).get("out_range", "zero_one")),
    )


def build_optical_adapter(cfg: dict, model: VAEMapCore) -> OpticalOLSAdapter:
    optics_cfg = cfg.get("optics", {})
    if not optics_cfg:
        raise ValueError("Optical config missing: require top-level 'optics' section")

    return OpticalOLSAdapter(
        latent_shape=(model.latent_channels, model.latent_hw[0], model.latent_hw[1]),
        resize_hw=tuple(optics_cfg.get("resize_hw", [model.latent_hw[0], model.latent_hw[1]])),
        wavelength_nm=float(optics_cfg.get("wavelength_nm", 532.0)),
        pixel_pitch_um=float(optics_cfg.get("pixel_pitch_um", 8.0)),
        z1_mm=float(optics_cfg.get("z1_mm", 20.0)),
        z2_mm=float(optics_cfg.get("z2_mm", 20.0)),
        pad_factor=float(optics_cfg.get("pad_factor", 2.0)),
        bandlimit=bool(optics_cfg.get("bandlimit", True)),
        upsample_factor=int(optics_cfg.get("upsample_factor", 1)),
        scatter_cfg=optics_cfg.get("scatter", {}),
        sensor_cfg=optics_cfg.get("sensor", {}),
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
def save_optical_stage_visualization(optics_info: dict, path: Path, max_items: int = 6) -> None:
    stages = optics_info.get("stage_intensities", [])
    stage_names = optics_info.get("stage_names", [])

    if len(stages) == 0:
        return

    # show representative stages for presentation: prop1, scatter, sensor
    idx_list = []
    for target in ("prop1", "scatter", "sensor"):
        if target in stage_names:
            idx_list.append(stage_names.index(target))
    if not idx_list:
        idx_list = [0, min(len(stages) - 1, 1), len(stages) - 1]

    vis = []
    target_h = 0
    target_w = 0
    for idx in idx_list:
        stage = stages[idx]
        stage = stage[:max_items, :1]
        stage = stage / (stage.mean(dim=(-2, -1), keepdim=True) + 1e-6)
        stage = torch.clamp(stage, 0.0, 3.0) / 3.0
        target_h = max(target_h, int(stage.shape[-2]))
        target_w = max(target_w, int(stage.shape[-1]))
        vis.append(stage)

    # Different optical stages can have different resolutions (e.g. sensor-pooled output),
    # resize to a common canvas before concatenation for visualization.
    vis_resized = []
    for stage in vis:
        if stage.shape[-2:] != (target_h, target_w):
            stage = torch.nn.functional.interpolate(stage, size=(target_h, target_w), mode="nearest")
        vis_resized.append(stage)

    grid_tensor = torch.cat(vis_resized, dim=0)
    save_image_grid(grid_tensor, path=path, nrow=max_items, out_range="zero_one")


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

    save_optical_stage_visualization(info_fixed, path=optics_dir / "epoch_{:03d}_optics.png".format(epoch), max_items=6)

    z_prior = sample_map_prior(
        batch_size=64,
        latent_channels=model.latent_channels,
        latent_hw=model.latent_hw,
        prior_cfg=prior_cfg,
        device=device,
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
    )

    model = build_model_from_cfg(cfg, dataset_info).to(device)
    adapter = build_optical_adapter(cfg, model).to(device)

    trainable_params = count_trainable_params(model) + count_trainable_params(adapter)
    logger.info("Trainable parameters: %d", trainable_params)

    lr = float(train_cfg.get("lr", 1e-3))
    optimizer = optim.Adam(list(model.parameters()) + list(adapter.parameters()), lr=lr)

    alpha = float(loss_cfg.get("alpha", 1.0))
    beta = float(loss_cfg.get("beta", 1.0))
    gamma = float(loss_cfg.get("gamma", 0.1))
    penalty_mode = str(loss_cfg.get("optical_penalty", {}).get("mode", "tv"))

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
            kl_per_sample = kl_map_gaussian_prior(
                mu_map,
                logvar_map,
                prior_type=str(prior_cfg.get("type", "biased_gaussian")),
                mu0=float(prior_cfg.get("mu0", 0.25)),
                sigma=float(prior_cfg.get("sigma", 1.0)),
                reduction="none",
            )
            op_per_sample = compute_optical_penalty(
                optics_info.get("stage_intensities", []),
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
        }
        val_metrics = evaluate_loader(model, adapter, val_loader, device, data_range)

        row = {
            "epoch": epoch,
            "loss": train_metrics["loss"],
            "recon": train_metrics["recon"],
            "kl": train_metrics["kl"],
            "op": train_metrics["op"],
            "val_mse": val_metrics["mse"],
            "val_psnr": val_metrics["psnr"],
        }
        history.append(row)
        append_row_csv(history_csv, row)

        logger.info(
            "Epoch %d/%d | loss=%.6f recon=%.6f kl=%.6f op=%.6f val_mse=%.6f val_psnr=%.3f",
            epoch,
            epochs,
            row["loss"],
            row["recon"],
            row["kl"],
            row["op"],
            row["val_mse"],
            row["val_psnr"],
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
