import argparse
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from data.datasets import get_dataloaders
from models import ConvVAE
from utils.io import append_row_csv, count_trainable_params, ensure_dir, now_timestamp, save_json, write_summary_csv
from utils.logger import create_logger, log_args
from utils.metrics import kl_divergence, mse_loss, psnr_from_mse, reconstruction_loss
from utils.seed import set_seed
from utils.viz import save_image_grid, save_reconstruction_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ConvVAE on MNIST/FashionMNIST/CIFAR-10")
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashionmnist", "cifar10"], default="mnist")
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--model_size", type=str, choices=["tiny", "small"], default="tiny")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--recon_loss", type=str, choices=["auto", "bce", "mse"], default="auto")
    parser.add_argument("--out_range", type=str, choices=["zero_one", "neg_one_one"], default="zero_one") # 注意：当使用 `BCE` 时必须使用 `--out_range zero_one`。
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--outdir", type=str, default=None)
    return parser.parse_args()


def resolve_recon_loss(dataset: str, recon_loss_arg: str) -> str:
    if recon_loss_arg != "auto":
        return recon_loss_arg
    return "bce" if dataset in ("mnist", "fashionmnist") else "mse"


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(
    path: Path,
    model: ConvVAE,
    optimizer: optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    dataset_info: dict,
    trainable_params: int,
    epoch_metrics: dict,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
        "dataset_info": dataset_info,
        "trainable_params": trainable_params,
        "epoch_metrics": epoch_metrics,
    }
    torch.save(payload, path)


@torch.no_grad()
def evaluate_loader(model: ConvVAE, loader, device: torch.device, data_range: float) -> dict:
    model.eval()
    mse_sum = 0.0
    psnr_sum = 0.0
    count = 0

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        recon, _, _, _ = model(x)

        mse_per_sample = mse_loss(recon, x, reduction="none")
        psnr_per_sample = psnr_from_mse(mse_per_sample, data_range=data_range)

        mse_sum += mse_per_sample.sum().item()
        psnr_sum += psnr_per_sample.sum().item()
        count += x.size(0)

    return {
        "mse": mse_sum / max(count, 1),
        "psnr": psnr_sum / max(count, 1),
    }


@torch.no_grad()
def save_epoch_visuals(
    model: ConvVAE,
    epoch: int,
    fixed_inputs: torch.Tensor,
    val_loader,
    out_range: str,
    recon_dir: Path,
    sample_dir: Path,
    interp_dir: Path,
    latent_dir: Path,
    device: torch.device,
) -> None:
    model.eval()

    recon_fixed, _, _, _ = model(fixed_inputs)
    save_reconstruction_comparison(
        inputs=fixed_inputs,
        recons=recon_fixed,
        path=recon_dir / f"epoch_{epoch:03d}_recon.png",
        max_items=8,
        out_range=out_range,
    )

    sampled = model.sample_prior(num_samples=64, device=device)
    save_image_grid(
        sampled,
        path=sample_dir / f"epoch_{epoch:03d}_sample.png",
        nrow=8,
        out_range=out_range,
    )

    with torch.no_grad():
        mu_fixed, logvar_fixed = model.encode(fixed_inputs)
        kl_fixed = kl_divergence(mu_fixed, logvar_fixed, reduction="none")

    def _to_heatmap_tensor(x: torch.Tensor) -> torch.Tensor:
        x = x.detach().cpu()
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min) / (x_max - x_min + 1e-8)
        return x.unsqueeze(0).unsqueeze(0)

    save_image_grid(
        _to_heatmap_tensor(mu_fixed),
        path=latent_dir / f"epoch_{epoch:03d}_mu_heatmap.png",
        nrow=1,
        out_range="zero_one",
    )
    save_image_grid(
        _to_heatmap_tensor(logvar_fixed),
        path=latent_dir / f"epoch_{epoch:03d}_logvar_heatmap.png",
        nrow=1,
        out_range="zero_one",
    )
    save_image_grid(
        _to_heatmap_tensor(kl_fixed.unsqueeze(1)),
        path=latent_dir / f"epoch_{epoch:03d}_kl_heatmap.png",
        nrow=1,
        out_range="zero_one",
    )

    interp_batch = next(iter(val_loader))[0]
    if interp_batch.size(0) < 2:
        return

    idx = torch.randperm(interp_batch.size(0))[:2]
    pair = interp_batch[idx].to(device)
    mu_pair, _ = model.encode(pair)
    z1, z2 = mu_pair[0:1], mu_pair[1:2]

    alphas = torch.linspace(0.0, 1.0, steps=10, device=device).unsqueeze(1)
    z_interp = z1 * (1.0 - alphas) + z2 * alphas
    interp_images = model.decode(z_interp)
    save_image_grid(
        interp_images,
        path=interp_dir / f"epoch_{epoch:03d}_interp.png",
        nrow=10,
        out_range=out_range,
    )


def main() -> None:
    args = parse_args()

    if args.outdir is None:
        args.outdir = str(Path("outputs") / f"{args.dataset}_vae_{args.model_size}_{now_timestamp()}")

    recon_loss_type = resolve_recon_loss(dataset=args.dataset, recon_loss_arg=args.recon_loss)
    if recon_loss_type == "bce" and args.out_range != "zero_one":
        raise ValueError("BCE requires out_range=zero_one so reconstruction is in [0,1].")

    run_dir = ensure_dir(args.outdir)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    recon_dir = ensure_dir(run_dir / "reconstructions")
    sample_dir = ensure_dir(run_dir / "samples")
    interp_dir = ensure_dir(run_dir / "interpolations")
    latent_dir = ensure_dir(run_dir / "latent_viz")
    log_dir = ensure_dir(run_dir / "logs")

    logger = create_logger("train_vae", outdir=log_dir, filename="train.log")

    set_seed(args.seed)
    device = select_device()

    logger.info("Device: %s", device)
    logger.info("Resolved recon loss: %s", recon_loss_type)
    log_args(logger, args)

    train_loader, val_loader, test_loader, dataset_info = get_dataloaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        out_range=args.out_range,
        seed=args.seed,
    )

    model = ConvVAE(
        in_channels=int(dataset_info["in_channels"]),
        input_size=dataset_info["image_size"],
        latent_dim=args.latent_dim,
        model_size=args.model_size,
        out_range=args.out_range,
    ).to(device)

    trainable_params = count_trainable_params(model)
    logger.info("Trainable parameters: %d", trainable_params)
    if args.model_size == "tiny" and args.latent_dim == 64:
        logger.info("Fair baseline profile active: tiny + latent_dim=64 (~24k params).")
    elif args.model_size == "small" and args.latent_dim == 100:
        logger.info("High-capacity profile active: small + latent_dim=100 (~242k params).")
    else:
        logger.warning(
            "Current setting deviates from preset reference profiles (tiny+64 or small+100). params=%d",
            trainable_params,
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    fixed_inputs = next(iter(val_loader))[0][:8].to(device)
    history = []
    history_csv = run_dir / "history.csv"
    best_val_mse = float("inf")
    best_epoch = -1
    data_range = float(dataset_info["data_range"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        recon_sum = 0.0
        kl_sum = 0.0
        count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for x, _ in pbar:
            x = x.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            recon, mu, logvar, _ = model(x)
            if not torch.isfinite(recon).all():
                logger.error("Non-finite reconstruction detected at epoch %d.", epoch)
                raise RuntimeError("Training aborted due to non-finite reconstruction output.")

            recon_per_sample = reconstruction_loss(recon, x, loss_type=recon_loss_type, reduction="none")
            kl_per_sample = kl_divergence(mu, logvar, reduction="none")
            loss_per_sample = recon_per_sample + args.beta * kl_per_sample
            loss = loss_per_sample.mean()

            if not torch.isfinite(loss):
                logger.error("Non-finite loss detected at epoch %d.", epoch)
                raise RuntimeError("Training aborted due to NaN/Inf loss.")

            loss.backward()
            optimizer.step()

            batch_size = x.size(0)
            loss_sum += loss_per_sample.sum().item()
            recon_sum += recon_per_sample.sum().item()
            kl_sum += kl_per_sample.sum().item()
            count += batch_size

            pbar.set_postfix(
                loss=f"{(loss_sum / max(count, 1)):.4f}",
                recon=f"{(recon_sum / max(count, 1)):.4f}",
                kl=f"{(kl_sum / max(count, 1)):.4f}",
            )

        train_metrics = {
            "loss": loss_sum / max(count, 1),
            "recon": recon_sum / max(count, 1),
            "kl": kl_sum / max(count, 1),
        }

        val_metrics = evaluate_loader(model=model, loader=val_loader, device=device, data_range=data_range)

        epoch_row = {
            "epoch": epoch,
            "loss": train_metrics["loss"],
            "recon": train_metrics["recon"],
            "kl": train_metrics["kl"],
            "val_mse": val_metrics["mse"],
            "val_psnr": val_metrics["psnr"],
        }
        history.append(epoch_row)
        append_row_csv(history_csv, epoch_row)

        logger.info(
            "Epoch %d/%d | loss=%.6f recon=%.6f kl=%.6f val_mse=%.6f val_psnr=%.3f",
            epoch,
            args.epochs,
            train_metrics["loss"],
            train_metrics["recon"],
            train_metrics["kl"],
            val_metrics["mse"],
            val_metrics["psnr"],
        )

        save_epoch_visuals(
            model=model,
            epoch=epoch,
            fixed_inputs=fixed_inputs,
            val_loader=val_loader,
            out_range=args.out_range,
            recon_dir=recon_dir,
            sample_dir=sample_dir,
            interp_dir=interp_dir,
            latent_dir=latent_dir,
            device=device,
        )

        save_checkpoint(
            path=ckpt_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            dataset_info=dataset_info,
            trainable_params=trainable_params,
            epoch_metrics=epoch_row,
        )

        if val_metrics["mse"] < best_val_mse:
            best_val_mse = val_metrics["mse"]
            best_epoch = epoch
            save_checkpoint(
                path=ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                dataset_info=dataset_info,
                trainable_params=trainable_params,
                epoch_metrics=epoch_row,
            )

    best_ckpt = torch.load(ckpt_dir / "best.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate_loader(model=model, loader=test_loader, device=device, data_range=data_range)

    results = {
        "config": vars(args),
        "resolved_recon_loss": recon_loss_type,
        "device": str(device),
        "trainable_params": trainable_params,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "best_val_psnr": next((r["val_psnr"] for r in history if r["epoch"] == best_epoch), None),
        "test_mse": test_metrics["mse"],
        "test_psnr": test_metrics["psnr"],
        "history": history,
    }

    save_json(results, run_dir / "results.json")

    summary_row = {
        "dataset": args.dataset,
        "model_size": args.model_size,
        "latent_dim": args.latent_dim,
        "beta": args.beta,
        "epochs": args.epochs,
        "resolved_recon_loss": recon_loss_type,
        "out_range": args.out_range,
        "seed": args.seed,
        "trainable_params": trainable_params,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_mse,
        "test_mse": test_metrics["mse"],
        "test_psnr": test_metrics["psnr"],
        "outdir": str(run_dir),
    }
    write_summary_csv(run_dir / "summary.csv", [summary_row])

    logger.info("Training completed. Best epoch: %d, test_mse=%.6f, test_psnr=%.3f", best_epoch, test_metrics["mse"], test_metrics["psnr"])
    logger.info("Artifacts saved to: %s", run_dir)


if __name__ == "__main__":
    main()
