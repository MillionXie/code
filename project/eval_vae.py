import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from data.datasets import get_dataloaders
from models import ConvVAE
from utils.io import flatten_dict, now_timestamp, save_json, write_summary_csv
from utils.logger import create_logger
from utils.metrics import mse_loss, psnr_from_mse
from utils.seed import set_seed
from utils.viz import save_reconstruction_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ConvVAE checkpoint on test set")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashionmnist", "cifar10"], default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=None)
    return parser.parse_args()


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ckpt_path = Path(args.checkpoint)
    if args.outdir is None:
        args.outdir = str(ckpt_path.parent / f"eval_{now_timestamp()}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("eval_vae", outdir=outdir, filename="eval.log")

    device = select_device()
    checkpoint = torch.load(ckpt_path, map_location=device)
    train_args = checkpoint.get("args", {})

    dataset = args.dataset or train_args.get("dataset")
    if dataset is None:
        raise ValueError("Dataset not found. Please provide --dataset explicitly.")

    out_range = train_args.get("out_range", "zero_one")
    latent_dim = int(train_args.get("latent_dim", 100))
    model_size = train_args.get("model_size", "tiny")

    _, _, test_loader, dataset_info = get_dataloaders(
        dataset=dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        out_range=out_range,
        seed=args.seed,
    )

    model = ConvVAE(
        in_channels=int(dataset_info["in_channels"]),
        input_size=dataset_info["image_size"],
        latent_dim=latent_dim,
        model_size=model_size,
        out_range=out_range,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_range = float(dataset_info["data_range"])
    mse_sum = 0.0
    psnr_sum = 0.0
    count = 0
    first_batch_saved = False

    with torch.no_grad():
        for x, _ in tqdm(test_loader, desc="Evaluating", leave=False):
            x = x.to(device, non_blocking=True)
            recon, _, _, _ = model(x)

            mse_per_sample = mse_loss(recon, x, reduction="none")
            psnr_per_sample = psnr_from_mse(mse_per_sample, data_range=data_range)

            mse_sum += mse_per_sample.sum().item()
            psnr_sum += psnr_per_sample.sum().item()
            count += x.size(0)

            if not first_batch_saved:
                save_reconstruction_comparison(
                    inputs=x,
                    recons=recon,
                    path=outdir / "reconstruction_test.png",
                    max_items=8,
                    out_range=out_range,
                )
                first_batch_saved = True

    metrics = {
        "dataset": dataset,
        "checkpoint": str(ckpt_path),
        "mse": mse_sum / max(count, 1),
        "psnr": psnr_sum / max(count, 1),
        "samples": count,
        "device": str(device),
    }

    save_json(metrics, outdir / "results.json")
    write_summary_csv(outdir / "summary.csv", [flatten_dict(metrics)])

    logger.info("Evaluation done | mse=%.6f psnr=%.3f", metrics["mse"], metrics["psnr"])
    logger.info("Results saved to %s", outdir)


if __name__ == "__main__":
    main()
