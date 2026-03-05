from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from data.datasets import get_dataloaders
from models import IdentityAdapter, VAEMapCore
from utils.config import apply_cli_overrides, load_config
from utils.io import append_row_csv, count_trainable_params, ensure_dir, now_timestamp, save_json, write_summary_csv
from utils.logger import create_logger
from utils.losses_optical import (
    compute_recon_per_sample,
    kl_map_gaussian_prior,
    resolve_recon_loss,
    sample_map_prior,
)
from utils.metrics import mse_loss, psnr_from_mse
from utils.seed import set_seed
from utils.viz import save_image_grid, save_reconstruction_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train map-latent electronic baseline (Identity adapter)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashionmnist", "cifar10"], default=None)
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


@torch.no_grad()
def evaluate_loader(model: VAEMapCore, adapter: IdentityAdapter, loader, device: torch.device, data_range: float) -> dict:
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
def save_epoch_visuals(
    model: VAEMapCore,
    adapter: IdentityAdapter,
    epoch: int,
    fixed_inputs: torch.Tensor,
    val_loader,
    out_range: str,
    prior_cfg: dict,
    recon_dir: Path,
    sample_dir: Path,
    interp_dir: Path,
    device: torch.device,
) -> None:
    model.eval()
    adapter.eval()

    mu_fixed, logvar_fixed = model.encode(fixed_inputs)
    z_fixed = model.reparameterize(mu_fixed, logvar_fixed)
    z_fixed_mid = adapter(z_fixed)
    recon_fixed = model.decode(z_fixed_mid)
    save_reconstruction_comparison(
        inputs=fixed_inputs,
        recons=recon_fixed,
        path=recon_dir / "epoch_{:03d}_recon.png".format(epoch),
        max_items=8,
        out_range=out_range,
    )

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
    adapter: IdentityAdapter,
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
    prior_cfg = loss_cfg.get("prior", {"type": "standard", "mu0": 0.0, "sigma": 1.0})

    seed = int(train_cfg.get("seed", 42))
    set_seed(seed)

    out_range = str(data_cfg.get("out_range", "zero_one"))
    recon_loss_type = resolve_recon_loss(dataset, str(train_cfg.get("recon_loss", "auto")))

    if recon_loss_type == "bce" and out_range != "zero_one":
        raise ValueError("BCE requires out_range=zero_one")

    if cfg.get("outdir") is None:
        cfg["outdir"] = str(Path("outputs") / "map_electronic_{}_{}".format(dataset, now_timestamp()))

    run_dir = ensure_dir(cfg["outdir"])
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    recon_dir = ensure_dir(run_dir / "reconstructions")
    sample_dir = ensure_dir(run_dir / "samples")
    interp_dir = ensure_dir(run_dir / "interpolations")
    log_dir = ensure_dir(run_dir / "logs")

    logger = create_logger("train_map_electronic", outdir=log_dir, filename="train_map_electronic.log")
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
    adapter = IdentityAdapter().to(device)

    trainable_params = count_trainable_params(model) + count_trainable_params(adapter)
    logger.info("Trainable parameters: %d", trainable_params)

    lr = float(train_cfg.get("lr", 1e-3))
    optimizer = optim.Adam(list(model.parameters()) + list(adapter.parameters()), lr=lr)

    alpha = float(loss_cfg.get("alpha", 1.0))
    beta = float(loss_cfg.get("beta", 1.0))
    gamma = float(loss_cfg.get("gamma", 0.0))

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
            z_mid = adapter(z_map)
            recon = model.decode(z_mid)

            if not torch.isfinite(recon).all():
                raise RuntimeError("Non-finite recon detected")

            recon_per_sample = compute_recon_per_sample(recon, x, recon_loss_type)
            kl_per_sample = kl_map_gaussian_prior(
                mu_map,
                logvar_map,
                prior_type=str(prior_cfg.get("type", "standard")),
                mu0=float(prior_cfg.get("mu0", 0.0)),
                sigma=float(prior_cfg.get("sigma", 1.0)),
                reduction="none",
            )
            op_per_sample = torch.zeros_like(kl_per_sample)

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
            "Epoch %d/%d | loss=%.6f recon=%.6f kl=%.6f val_mse=%.6f val_psnr=%.3f",
            epoch,
            epochs,
            row["loss"],
            row["recon"],
            row["kl"],
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
        "mode": "map_electronic",
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
