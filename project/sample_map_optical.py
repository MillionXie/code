from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.datasets import get_dataset_info
from utils.config import load_config
from utils.io import now_timestamp, save_json
from utils.logger import create_logger
from utils.losses_optical import sample_map_prior
from utils.map_optical import build_decoder_prior_cfg, build_map_core_from_cfg, build_optical_adapter_from_cfg
from utils.seed import set_seed
from utils.viz import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample map-latent optical model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=64)
    parser.add_argument("--grid_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    ckpt_path = Path(args.checkpoint)

    if args.outdir is None:
        args.outdir = str(ckpt_path.parent / "sample_map_optical_{}".format(now_timestamp()))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("sample_map_optical", outdir=outdir, filename="sample_map_optical.log")

    device = select_device()
    dataset = str(cfg.get("dataset", "mnist")).lower()
    dataset_info = get_dataset_info(dataset, image_size=cfg.get("data", {}).get("image_size", [64, 64]))
    model = build_map_core_from_cfg(cfg, dataset_info).to(device)
    adapter = build_optical_adapter_from_cfg(cfg, model).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "adapter_state_dict" in checkpoint:
        adapter.load_state_dict(checkpoint["adapter_state_dict"])

    model.eval()
    adapter.eval()

    loss_cfg = cfg.get("loss", {})
    prior_cfg = loss_cfg.get("prior", {"type": "biased_gaussian", "mu0": 0.25, "sigma": 1.0})
    klw_cfg = loss_cfg.get("kl_w", {})
    kl_target = str(klw_cfg.get("target", "latent_mean")).lower()
    if kl_target == "final_latent":
        kl_target = "latent_mean"
    sample_prior_space = "decoder"

    decoder_prior_cfg = build_decoder_prior_cfg(prior_cfg=prior_cfg, klw_cfg=klw_cfg)
    sample_apply_smooth = bool(prior_cfg.get("apply_smooth_in_sample", prior_cfg.get("apply_smooth_in_train", False)))
    smooth_cfg = decoder_prior_cfg.get("spatial_smooth", {}) if isinstance(decoder_prior_cfg, dict) else {}
    smooth_type = str(smooth_cfg.get("type", "none")).lower() if isinstance(smooth_cfg, dict) else "none"
    logger.info(
        "Sampling from prior P(z)=N(mu0=%.6f, sigma=%.6f) | kl_target=%s | apply_smooth=%s smooth_type=%s",
        float(decoder_prior_cfg["mu0"]),
        float(decoder_prior_cfg["sigma"]),
        kl_target,
        sample_apply_smooth,
        smooth_type,
    )

    with torch.no_grad():
        z_mid = sample_map_prior(
            batch_size=args.n_samples,
            latent_channels=model.latent_channels,
            latent_hw=model.latent_hw,
            prior_cfg=decoder_prior_cfg,
            device=device,
            apply_smooth=sample_apply_smooth,
        )
        samples = model.decode(z_mid)

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
        "sample_prior_space": sample_prior_space,
        "kl_target": kl_target,
        "sample_apply_smooth": sample_apply_smooth,
        "sample_smooth_type": smooth_type,
        "sample_prior_mu0": float(decoder_prior_cfg["mu0"]),
        "sample_prior_sigma": float(decoder_prior_cfg["sigma"]),
    }
    save_json(meta, outdir / "results.json")
    logger.info("Saved samples to %s", sample_path)


if __name__ == "__main__":
    main()
