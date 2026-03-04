from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.datasets import get_dataset_info
from models import IdentityAdapter, VAEMapCore
from utils.config import load_config
from utils.io import now_timestamp, save_json
from utils.logger import create_logger
from utils.losses_optical import sample_map_prior
from utils.seed import set_seed
from utils.viz import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample map-latent electronic model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg: dict) -> VAEMapCore:
    dataset = str(cfg.get("dataset", "mnist")).lower()
    info = get_dataset_info(dataset)
    model_cfg = cfg.get("model", {})
    out_range = str(cfg.get("data", {}).get("out_range", "zero_one"))

    return VAEMapCore(
        in_channels=int(info["in_channels"]),
        input_size=tuple(info["image_size"]),
        latent_channels=int(model_cfg.get("latent_channels", 16)),
        latent_hw=tuple(model_cfg.get("latent_hw", [4, 4])),
        encoder_channels=tuple(model_cfg.get("encoder_channels", [32, 64, 128])),
        decoder_channels=tuple(model_cfg.get("decoder_channels", [128, 64])),
        decoder_mode=str(model_cfg.get("decoder_mode", "deconv")),
        out_range=out_range,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    ckpt_path = Path(args.checkpoint)

    if args.outdir is None:
        args.outdir = str(ckpt_path.parent / "sample_map_electronic_{}".format(now_timestamp()))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("sample_map_electronic", outdir=outdir, filename="sample_map_electronic.log")

    device = select_device()
    model = build_model(cfg).to(device)
    adapter = IdentityAdapter().to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "adapter_state_dict" in checkpoint:
        adapter.load_state_dict(checkpoint["adapter_state_dict"])

    model.eval()
    adapter.eval()

    prior_cfg = cfg.get("loss", {}).get("prior", {"type": "standard", "mu0": 0.0, "sigma": 1.0})

    with torch.no_grad():
        z_map = sample_map_prior(
            batch_size=args.n_samples,
            latent_channels=model.latent_channels,
            latent_hw=model.latent_hw,
            prior_cfg=prior_cfg,
            device=device,
        )
        samples = model.decode(adapter(z_map))

    sample_path = outdir / "samples.png"
    save_image_grid(
        samples,
        path=sample_path,
        nrow=args.grid_size,
        out_range=str(cfg.get("data", {}).get("out_range", "zero_one")),
    )

    meta = {
        "config": args.config,
        "checkpoint": str(ckpt_path),
        "n_samples": args.n_samples,
        "grid_size": args.grid_size,
        "seed": args.seed,
        "sample_path": str(sample_path),
        "device": str(device),
    }
    save_json(meta, outdir / "results.json")
    logger.info("Saved samples to %s", sample_path)


if __name__ == "__main__":
    main()
