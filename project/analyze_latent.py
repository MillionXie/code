import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.datasets import get_dataloaders
from models import ConvVAE
from utils.io import flatten_dict, now_timestamp, save_json, write_summary_csv
from utils.logger import create_logger
from utils.metrics import kl_divergence, mse_loss, psnr_from_mse
from utils.seed import set_seed
from utils.viz import plot_eigen_spectrum, plot_histogram, plot_scatter_2d

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent statistics and visualization for VAE")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashionmnist", "cifar10"], default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--run_tsne", dest="run_tsne", action="store_true")
    parser.add_argument("--no_tsne", dest="run_tsne", action="store_false")
    parser.add_argument("--max_points", type=int, default=5000)
    parser.add_argument("--tsne_points", type=int, default=3000)
    parser.set_defaults(run_tsne=True)
    return parser.parse_args()


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pca_2d(mu: np.ndarray, seed: int) -> np.ndarray:
    if SKLEARN_AVAILABLE:
        pca = PCA(n_components=2, random_state=seed)
        return pca.fit_transform(mu)

    centered = mu - mu.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ckpt_path = Path(args.checkpoint)
    if args.outdir is None:
        args.outdir = str(ckpt_path.parent / f"analyze_{now_timestamp()}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("analyze_latent", outdir=outdir, filename="analyze.log")

    device = select_device()
    checkpoint = torch.load(ckpt_path, map_location=device)
    train_args = checkpoint.get("args", {})

    dataset = args.dataset or train_args.get("dataset")
    if dataset is None:
        raise ValueError("Dataset not found. Please provide --dataset.")

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

    mu_list = []
    logvar_list = []
    label_list = []
    kl_list = []
    mse_list = []
    psnr_list = []

    mse_sum = 0.0
    psnr_sum = 0.0
    count = 0
    data_range = float(dataset_info["data_range"])

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Extracting latents", leave=False):
            x = x.to(device, non_blocking=True)
            mu, logvar = model.encode(x)
            recon = model.decode(mu)

            mse_per_sample = mse_loss(recon, x, reduction="none")
            psnr_per_sample = psnr_from_mse(mse_per_sample, data_range=data_range)
            kl_per_sample = kl_divergence(mu, logvar, reduction="none")

            mse_sum += mse_per_sample.sum().item()
            psnr_sum += psnr_per_sample.sum().item()
            count += x.size(0)

            mu_list.append(mu.cpu().numpy())
            logvar_list.append(logvar.cpu().numpy())
            label_list.append(y.numpy())
            kl_list.append(kl_per_sample.cpu().numpy())
            mse_list.append(mse_per_sample.cpu().numpy())
            psnr_list.append(psnr_per_sample.cpu().numpy())

    mu_all = np.concatenate(mu_list, axis=0)
    logvar_all = np.concatenate(logvar_list, axis=0)
    labels_all = np.concatenate(label_list, axis=0)
    kl_all = np.concatenate(kl_list, axis=0)
    mse_all = np.concatenate(mse_list, axis=0)
    psnr_all = np.concatenate(psnr_list, axis=0)

    mu_dim_mean = mu_all.mean(axis=0)
    mu_dim_var = mu_all.var(axis=0)

    plot_histogram(
        mu_dim_mean,
        path=outdir / "mu_dim_mean_hist.png",
        title="Histogram of Latent Dimension Means (mu)",
        xlabel="Mean Value per Dimension",
        bins=40,
    )
    plot_histogram(
        mu_dim_var,
        path=outdir / "mu_dim_var_hist.png",
        title="Histogram of Latent Dimension Variances (mu)",
        xlabel="Variance per Dimension",
        bins=40,
    )
    plot_histogram(
        kl_all,
        path=outdir / "kl_distribution_hist.png",
        title="KL Distribution Across Samples",
        xlabel="KL(q(z|x) || p(z))",
        bins=50,
    )

    cov = np.cov(mu_all, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    plot_eigen_spectrum(
        eigvals,
        path=outdir / "mu_cov_eigenspectrum.png",
        title="Eigenvalue Spectrum of Cov(mu)",
    )

    rng = np.random.default_rng(args.seed)
    if len(mu_all) > args.max_points:
        idx = rng.choice(len(mu_all), size=args.max_points, replace=False)
        mu_vis = mu_all[idx]
        labels_vis = labels_all[idx]
    else:
        mu_vis = mu_all
        labels_vis = labels_all

    pca_points = pca_2d(mu_vis, seed=args.seed)
    plot_scatter_2d(
        pca_points,
        path=outdir / "pca_2d.png",
        title="PCA (2D) of Latent mu",
        labels=labels_vis,
    )
    pca_rows = []
    for idx in range(len(pca_points)):
        pca_rows.append(
            {
                "index": int(idx),
                "label": int(labels_vis[idx]),
                "x": float(pca_points[idx, 0]),
                "y": float(pca_points[idx, 1]),
            }
        )
    write_summary_csv(outdir / "embedding_pca.csv", pca_rows)

    tsne_saved = False
    tsne_points = None
    labels_tsne = None
    if args.run_tsne:
        if SKLEARN_AVAILABLE:
            tsne_n = min(len(mu_vis), int(args.tsne_points))
            tsne_idx = rng.choice(len(mu_vis), size=tsne_n, replace=False) if len(mu_vis) > tsne_n else np.arange(len(mu_vis))
            mu_tsne = mu_vis[tsne_idx]
            labels_tsne = labels_vis[tsne_idx]
            tsne = TSNE(n_components=2, random_state=args.seed, init="pca", learning_rate="auto")
            tsne_points = tsne.fit_transform(mu_tsne)
            plot_scatter_2d(
                tsne_points,
                path=outdir / "tsne_2d.png",
                title="t-SNE (2D) of Latent mu",
                labels=labels_tsne,
            )
            tsne_rows = []
            for idx in range(len(tsne_points)):
                tsne_rows.append(
                    {
                        "index": int(idx),
                        "label": int(labels_tsne[idx]),
                        "x": float(tsne_points[idx, 0]),
                        "y": float(tsne_points[idx, 1]),
                    }
                )
            write_summary_csv(outdir / "embedding_tsne.csv", tsne_rows)
            tsne_saved = True
        else:
            logger.info("t-SNE skipped: scikit-learn is not installed.")

    per_sample_rows = []
    for idx in range(len(labels_all)):
        per_sample_rows.append(
            {
                "index": int(idx),
                "label": int(labels_all[idx]),
                "mse": float(mse_all[idx]),
                "psnr": float(psnr_all[idx]),
                "kl": float(kl_all[idx]),
            }
        )
    write_summary_csv(outdir / "per_sample_metrics.csv", per_sample_rows)

    class_rows = []
    for label in np.unique(labels_all):
        mask = labels_all == label
        class_rows.append(
            {
                "label": int(label),
                "count": int(mask.sum()),
                "mse_mean": float(mse_all[mask].mean()),
                "psnr_mean": float(psnr_all[mask].mean()),
                "kl_mean": float(kl_all[mask].mean()),
                "mu_norm_mean": float(np.linalg.norm(mu_all[mask], axis=1).mean()),
            }
        )
    write_summary_csv(outdir / "class_metrics.csv", class_rows)

    np.savez_compressed(
        outdir / "latent_arrays.npz",
        mu=mu_all,
        logvar=logvar_all,
        labels=labels_all,
        kl=kl_all,
        mse=mse_all,
        psnr=psnr_all,
        pca_points=pca_points,
        pca_labels=labels_vis,
        tsne_points=tsne_points if tsne_points is not None else np.empty((0, 2), dtype=np.float32),
        tsne_labels=labels_tsne if labels_tsne is not None else np.empty((0,), dtype=np.int64),
    )

    summary = {
        "checkpoint": str(ckpt_path),
        "dataset": dataset,
        "samples": int(len(mu_all)),
        "latent_dim": int(mu_all.shape[1]),
        "params_trainable": int(checkpoint.get("trainable_params", -1)),
        "mse": mse_sum / max(count, 1),
        "psnr": psnr_sum / max(count, 1),
        "latent_stats": {
            "mu_global_mean": float(mu_all.mean()),
            "mu_global_std": float(mu_all.std()),
            "logvar_global_mean": float(logvar_all.mean()),
            "logvar_global_std": float(logvar_all.std()),
            "mu_dim_mean_mean": float(mu_dim_mean.mean()),
            "mu_dim_mean_std": float(mu_dim_mean.std()),
            "mu_dim_var_mean": float(mu_dim_var.mean()),
            "mu_dim_var_std": float(mu_dim_var.std()),
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
            "latent_arrays_npz": str(outdir / "latent_arrays.npz"),
            "per_sample_metrics_csv": str(outdir / "per_sample_metrics.csv"),
            "class_metrics_csv": str(outdir / "class_metrics.csv"),
            "embedding_pca_csv": str(outdir / "embedding_pca.csv"),
            "embedding_tsne_csv": str(outdir / "embedding_tsne.csv") if tsne_saved else None,
        },
    }

    save_json(summary, outdir / "summary.json")
    write_summary_csv(outdir / "summary.csv", [flatten_dict(summary)])

    logger.info(
        "Latent analysis complete | mse=%.6f psnr=%.3f kl_mean=%.4f",
        summary["mse"],
        summary["psnr"],
        summary["latent_stats"]["kl_mean"],
    )
    logger.info("Saved analysis artifacts to %s", outdir)


if __name__ == "__main__":
    main()
