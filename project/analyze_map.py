from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.datasets import get_dataloaders
from models import OpticalOLSAdapter, VAEMapCore
from utils.config import load_config
from utils.io import flatten_dict, now_timestamp, save_json, write_summary_csv
from utils.logger import create_logger
from utils.losses_optical import kl_latent_intensity_biased_gaussian, kl_map_gaussian_prior
from utils.metrics import mse_loss, psnr_from_mse
from utils.seed import set_seed
from utils.viz import plot_eigen_spectrum, plot_histogram, plot_scatter_2d

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze map-latent statistics")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--run_tsne", action="store_true")
    parser.add_argument("--max_points", type=int, default=5000)
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


def build_optical_adapter_if_needed(cfg: dict, model: VAEMapCore):
    optics_cfg = cfg.get("optics", None)
    if not optics_cfg:
        return None

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


def pca_2d(x: np.ndarray, seed: int) -> np.ndarray:
    if SKLEARN_AVAILABLE:
        return PCA(n_components=2, random_state=seed).fit_transform(x)

    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    dataset = args.dataset or cfg.get("dataset", None)
    if dataset is None:
        raise ValueError("dataset missing in config and CLI")

    ckpt_path = Path(args.checkpoint)
    if args.outdir is None:
        args.outdir = str(ckpt_path.parent / "analyze_map_{}".format(now_timestamp()))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("analyze_map", outdir=outdir, filename="analyze_map.log")

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
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    adapter = build_optical_adapter_if_needed(cfg, model)
    if adapter is not None:
        adapter = adapter.to(device)
        if "adapter_state_dict" in checkpoint:
            adapter.load_state_dict(checkpoint["adapter_state_dict"])
        adapter.eval()

    prior_cfg = cfg.get("loss", {}).get("prior", {"type": "standard", "mu0": 0.0, "sigma": 1.0})
    klw_cfg = cfg.get("loss", {}).get("kl_w", {})
    kl_prior_sigma0 = float(klw_cfg.get("prior_sigma0", prior_cfg.get("sigma", 1.0)))
    kl_var0 = float(klw_cfg.get("var0", kl_prior_sigma0 * kl_prior_sigma0))
    kl_target = str(klw_cfg.get("target", "final_latent")).lower()
    if "pre_norm" in klw_cfg:
        kl_pre_norm = str(klw_cfg.get("pre_norm"))
    else:
        kl_pre_norm = "none" if kl_target == "final_latent" else "mean"
    kl_clamp_nonnegative = bool(klw_cfg.get("clamp_nonnegative", kl_target != "final_latent"))
    if kl_target not in ("latent_intensity", "final_latent"):
        raise ValueError("loss.kl_w.target must be one of: latent_intensity|final_latent")

    mu_list = []
    logvar_list = []
    label_list = []
    kl_list = []

    mse_sum = 0.0
    psnr_sum = 0.0
    count = 0
    data_range = float(dataset_info.get("data_range", 1.0))

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Extracting map latents", leave=False):
            x = x.to(device, non_blocking=True)

            mu_map, logvar_map = model.encode(x)
            z_map = model.reparameterize(mu_map, logvar_map)
            if adapter is not None:
                z_mid, info = adapter(z_map, return_info=True)
                if kl_target == "final_latent":
                    latent_for_stats = info["final_latent_map"]
                else:
                    latent_for_stats = info["latent_intensity_map"]
                recon = model.decode(z_mid)
                kl_ps = kl_latent_intensity_biased_gaussian(
                    latent_intensity_map=latent_for_stats,
                    var_mode=str(klw_cfg.get("var_mode", "constant")),
                    var0=kl_var0,
                    prior_mean_m0=float(klw_cfg.get("m0", prior_cfg.get("mu0", 0.0))),
                    prior_sigma0=kl_prior_sigma0,
                    pre_norm=kl_pre_norm,
                    clamp_nonnegative=kl_clamp_nonnegative,
                    reduction="none",
                )
            else:
                latent_for_stats = mu_map
                recon = model.decode(z_map)
                kl_ps = kl_map_gaussian_prior(
                    mu_map,
                    logvar_map,
                    prior_type=str(prior_cfg.get("type", "standard")),
                    mu0=float(prior_cfg.get("mu0", 0.0)),
                    sigma=float(prior_cfg.get("sigma", 1.0)),
                    reduction="none",
                )

            mse_ps = mse_loss(recon, x, reduction="none")
            psnr_ps = psnr_from_mse(mse_ps, data_range=data_range)

            mse_sum += mse_ps.sum().item()
            psnr_sum += psnr_ps.sum().item()
            count += x.size(0)

            mu_list.append(latent_for_stats.flatten(start_dim=1).cpu().numpy())
            logvar_list.append(logvar_map.flatten(start_dim=1).cpu().numpy())
            label_list.append(y.numpy())
            kl_list.append(kl_ps.cpu().numpy())

    mu_all = np.concatenate(mu_list, axis=0)
    logvar_all = np.concatenate(logvar_list, axis=0)
    labels_all = np.concatenate(label_list, axis=0)
    kl_all = np.concatenate(kl_list, axis=0)

    dim_mean = mu_all.mean(axis=0)
    dim_var = mu_all.var(axis=0)

    plot_histogram(dim_mean, outdir / "mu_dim_mean_hist.png", "Histogram of Map-Latent Dimension Means", "Mean")
    plot_histogram(dim_var, outdir / "mu_dim_var_hist.png", "Histogram of Map-Latent Dimension Variances", "Variance")
    plot_histogram(kl_all, outdir / "kl_distribution_hist.png", "KL Distribution Across Samples", "KL")

    cov = np.cov(mu_all, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    plot_eigen_spectrum(eigvals, outdir / "mu_cov_eigenspectrum.png", "Eigenvalue Spectrum of Cov(mu_map)")

    rng = np.random.default_rng(args.seed)
    if len(mu_all) > args.max_points:
        idx = rng.choice(len(mu_all), size=args.max_points, replace=False)
        mu_vis = mu_all[idx]
        label_vis = labels_all[idx]
    else:
        mu_vis = mu_all
        label_vis = labels_all

    pca_pts = pca_2d(mu_vis, seed=args.seed)
    plot_scatter_2d(pca_pts, outdir / "pca_2d.png", "PCA (2D) of Map-Latent mu", labels=label_vis)

    tsne_saved = False
    if args.run_tsne:
        if SKLEARN_AVAILABLE:
            tsne_n = min(len(mu_vis), 3000)
            idx = rng.choice(len(mu_vis), size=tsne_n, replace=False) if len(mu_vis) > tsne_n else np.arange(len(mu_vis))
            tsne = TSNE(n_components=2, random_state=args.seed, init="pca", learning_rate="auto")
            tsne_pts = tsne.fit_transform(mu_vis[idx])
            plot_scatter_2d(tsne_pts, outdir / "tsne_2d.png", "t-SNE (2D) of Map-Latent mu", labels=label_vis[idx])
            tsne_saved = True
        else:
            logger.info("t-SNE skipped: scikit-learn not installed")

    summary = {
        "checkpoint": str(ckpt_path),
        "dataset": dataset,
        "samples": int(len(mu_all)),
        "latent_source": ("optical_" + kl_target) if adapter is not None else "encoder_mu_map",
        "latent_dim_flat": int(mu_all.shape[1]),
        "params_trainable": int(checkpoint.get("trainable_params", -1)),
        "mse": mse_sum / max(count, 1),
        "psnr": psnr_sum / max(count, 1),
        "latent_stats": {
            "mu_global_mean": float(mu_all.mean()),
            "mu_global_std": float(mu_all.std()),
            "logvar_global_mean": float(logvar_all.mean()),
            "logvar_global_std": float(logvar_all.std()),
            "mu_dim_mean_mean": float(dim_mean.mean()),
            "mu_dim_mean_std": float(dim_mean.std()),
            "mu_dim_var_mean": float(dim_var.mean()),
            "mu_dim_var_std": float(dim_var.std()),
            "kl_mean": float(kl_all.mean()),
            "kl_std": float(kl_all.std()),
            "kl_min": float(kl_all.min()),
            "kl_max": float(kl_all.max()),
            "cov_trace": float(np.trace(cov)),
            "eig_max": float(eigvals.max()),
            "eig_min": float(eigvals.min()),
            "eig_top10": [float(x) for x in eigvals[:10]],
        },
        "artifacts": {
            "mu_dim_mean_hist": str(outdir / "mu_dim_mean_hist.png"),
            "mu_dim_var_hist": str(outdir / "mu_dim_var_hist.png"),
            "kl_distribution_hist": str(outdir / "kl_distribution_hist.png"),
            "mu_cov_eigenspectrum": str(outdir / "mu_cov_eigenspectrum.png"),
            "pca_2d": str(outdir / "pca_2d.png"),
            "tsne_2d": str(outdir / "tsne_2d.png") if tsne_saved else None,
        },
    }

    save_json(summary, outdir / "summary.json")
    write_summary_csv(outdir / "summary.csv", [flatten_dict(summary)])
    logger.info("Analyze map done | mse=%.6f psnr=%.3f", summary["mse"], summary["psnr"])


if __name__ == "__main__":
    main()
