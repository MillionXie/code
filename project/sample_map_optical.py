from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.datasets import get_dataset_info
from models import OpticalOLSAdapter, VAEMapCore
from utils.config import load_config
from utils.io import now_timestamp, save_json
from utils.logger import create_logger
from utils.losses_optical import sample_map_prior
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


def build_adapter(cfg: dict, model: VAEMapCore) -> OpticalOLSAdapter:
    optics_cfg = cfg.get("optics", {})
    if not optics_cfg:
        raise ValueError("Missing optics section in config")
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
    model = build_model(cfg).to(device)
    adapter = build_adapter(cfg, model).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "adapter_state_dict" in checkpoint:
        adapter.load_state_dict(checkpoint["adapter_state_dict"])

    model.eval()
    adapter.eval()

    loss_cfg = cfg.get("loss", {})
    prior_cfg = loss_cfg.get("prior", {"type": "biased_gaussian", "mu0": 0.25, "sigma": 1.0})
    klw_cfg = loss_cfg.get("kl_w", {})
    direct_mode = str(cfg.get("optics", {}).get("direct_mode", "latent_hw")).lower()
    kl_target = str(klw_cfg.get("target", "final_latent")).lower()
    sample_prior_space = str(cfg.get("sample", {}).get("prior_space", "auto")).lower()
    if sample_prior_space == "auto":
        sample_prior_space = "decoder" if (kl_target == "final_latent" and direct_mode == "latent_hw") else "z_map"
    if sample_prior_space not in ("decoder", "z_map"):
        raise ValueError("sample.prior_space must be one of: auto|decoder|z_map")
    if sample_prior_space == "decoder" and direct_mode != "latent_hw":
        logger.info("sample.prior_space=decoder is not supported for direct_mode=%s, fallback to z_map", direct_mode)
        sample_prior_space = "z_map"

    decoder_prior_cfg = {
        "type": "biased_gaussian",
        "mu0": float(klw_cfg.get("m0", prior_cfg.get("mu0", 0.0))),
        "sigma": float(klw_cfg.get("prior_sigma0", prior_cfg.get("sigma", 1.0))),
        "spatial_smooth": prior_cfg.get("spatial_smooth", {}),
    }

    with torch.no_grad():
        if sample_prior_space == "decoder":
            z_mid = sample_map_prior(
                batch_size=args.n_samples,
                latent_channels=model.latent_channels,
                latent_hw=model.latent_hw,
                prior_cfg=decoder_prior_cfg,
                device=device,
                apply_smooth=True,
            )
            samples = model.decode(z_mid)
        else:
            z_map = sample_map_prior(
                batch_size=args.n_samples,
                latent_channels=model.latent_channels,
                latent_hw=model.latent_hw,
                prior_cfg=prior_cfg,
                device=device,
                apply_smooth=True,
            )
            z_mid = adapter(z_map)
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
    }
    save_json(meta, outdir / "results.json")
    logger.info("Saved samples to %s", sample_path)


if __name__ == "__main__":
    main()
