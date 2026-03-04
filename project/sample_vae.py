import argparse
from pathlib import Path

import torch

from data.datasets import get_dataset_info
from latent import GaussianPriorProvider
from models import ConvVAE
from utils.io import now_timestamp, save_json
from utils.logger import create_logger
from utils.seed import set_seed
from utils.viz import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images from VAE prior")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--grid_size", type=int, default=8)
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
        args.outdir = str(ckpt_path.parent / f"sample_{now_timestamp()}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("sample_vae", outdir=outdir, filename="sample.log")

    device = select_device()
    checkpoint = torch.load(ckpt_path, map_location=device)
    train_args = checkpoint.get("args", {})

    dataset = train_args.get("dataset")
    if dataset is None:
        raise ValueError("Cannot infer dataset from checkpoint args.")

    info = get_dataset_info(dataset)
    model = ConvVAE(
        in_channels=int(info["in_channels"]),
        input_size=info["image_size"],
        latent_dim=int(train_args.get("latent_dim", 100)),
        model_size=train_args.get("model_size", "tiny"),
        out_range=train_args.get("out_range", "zero_one"),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    provider = GaussianPriorProvider(latent_dim=model.latent_dim)

    with torch.no_grad():
        latent = provider.get_latent(model=model, num_samples=args.n_samples, device=device)
        samples = model.decode(latent.z)

    sample_path = outdir / "samples.png"
    save_image_grid(samples, path=sample_path, nrow=args.grid_size, out_range=model.out_range)

    meta = {
        "checkpoint": str(ckpt_path),
        "dataset": dataset,
        "n_samples": args.n_samples,
        "grid_size": args.grid_size,
        "seed": args.seed,
        "device": str(device),
        "sample_path": str(sample_path),
    }
    save_json(meta, outdir / "results.json")

    logger.info("Generated %d samples to %s", args.n_samples, sample_path)


if __name__ == "__main__":
    main()
