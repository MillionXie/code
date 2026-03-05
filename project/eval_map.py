from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from data.datasets import get_dataloaders
from models import IdentityAdapter, OpticalOLSAdapter, VAEMapCore
from utils.config import load_config
from utils.io import flatten_dict, now_timestamp, save_json, write_summary_csv
from utils.logger import create_logger
from utils.metrics import mse_loss, psnr_from_mse
from utils.seed import set_seed
from utils.viz import save_reconstruction_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate map-latent model")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["electronic", "optical"], required=True)
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=None)
    return parser.parse_args()


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(cfg: dict, dataset_info: dict) -> VAEMapCore:
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


def build_adapter(cfg: dict, model: VAEMapCore, mode: str):
    if mode == "electronic":
        return IdentityAdapter()

    optics_cfg = cfg.get("optics", {})
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
    dataset = args.dataset or cfg.get("dataset", None)
    if dataset is None:
        raise ValueError("dataset missing in config and CLI")

    ckpt_path = Path(args.checkpoint)
    if args.outdir is None:
        args.outdir = str(ckpt_path.parent / "eval_map_{}_{}".format(args.mode, now_timestamp()))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("eval_map", outdir=outdir, filename="eval_map.log")

    device = select_device()
    out_range = str(cfg.get("data", {}).get("out_range", "zero_one"))

    _, _, test_loader, dataset_info = get_dataloaders(
        dataset=dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        out_range=out_range,
        seed=args.seed,
        image_size=cfg.get("data", {}).get("image_size", [64, 64]),
    )

    model = build_model(cfg, dataset_info).to(device)
    adapter = build_adapter(cfg, model, args.mode).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "adapter_state_dict" in checkpoint:
        adapter.load_state_dict(checkpoint["adapter_state_dict"])

    model.eval()
    adapter.eval()

    data_range = float(dataset_info["data_range"])
    mse_sum = 0.0
    psnr_sum = 0.0
    count = 0
    saved = False

    with torch.no_grad():
        for x, _ in tqdm(test_loader, desc="Evaluating", leave=False):
            x = x.to(device, non_blocking=True)
            mu_map, logvar_map = model.encode(x)
            z_map = model.reparameterize(mu_map, logvar_map)
            z_mid = adapter(z_map)
            recon = model.decode(z_mid)

            mse_ps = mse_loss(recon, x, reduction="none")
            psnr_ps = psnr_from_mse(mse_ps, data_range=data_range)

            mse_sum += mse_ps.sum().item()
            psnr_sum += psnr_ps.sum().item()
            count += x.size(0)

            if not saved:
                save_reconstruction_comparison(
                    inputs=x,
                    recons=recon,
                    path=outdir / "reconstruction_test.png",
                    max_items=8,
                    out_range=out_range,
                )
                saved = True

    metrics = {
        "mode": args.mode,
        "dataset": dataset,
        "checkpoint": str(ckpt_path),
        "mse": mse_sum / max(count, 1),
        "psnr": psnr_sum / max(count, 1),
        "samples": count,
        "device": str(device),
    }

    save_json(metrics, outdir / "results.json")
    write_summary_csv(outdir / "summary.csv", [flatten_dict(metrics)])

    logger.info("Eval map done | mse=%.6f psnr=%.3f", metrics["mse"], metrics["psnr"])


if __name__ == "__main__":
    main()
