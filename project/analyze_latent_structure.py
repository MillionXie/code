from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils.io import save_json, write_summary_csv
from utils.latent_compare import (
    build_test_loader,
    ensure_analysis_dirs,
    extract_decoder_latent,
    flatten_latent,
    load_analysis_config,
    load_model_bundle_from_checkpoint,
    save_config_used,
)
from utils.seed import set_seed

matplotlib.use("Agg")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task4: Analyze latent structure (PCA/t-SNE/statistics)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashionmnist", "cifar10"], default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--tsne_perplexity", type=float, default=None)
    return parser.parse_args()


def _resolve_runtime(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    analysis_cfg = cfg.get("analysis", {})
    dataset = args.dataset or cfg.get("dataset", analysis_cfg.get("dataset", None))
    batch_size = int(args.batch_size if args.batch_size is not None else analysis_cfg.get("batch_size", 256))
    num_workers = int(args.num_workers if args.num_workers is not None else analysis_cfg.get("num_workers", 4))
    seed = int(args.seed if args.seed is not None else analysis_cfg.get("seed", 42))
    num_samples = int(args.num_samples if args.num_samples is not None else analysis_cfg.get("num_samples", 1000))
    tsne_perplexity = float(
        args.tsne_perplexity if args.tsne_perplexity is not None else analysis_cfg.get("tsne_perplexity", 30.0)
    )
    outdir = args.outdir or cfg.get("outdir", analysis_cfg.get("outdir", None))
    if outdir is None:
        outdir = str(Path("outputs") / "task4_latent_structure")
    save_pdf = bool(analysis_cfg.get("save_pdf", True))
    include_optical_intensity = bool(analysis_cfg.get("include_optical_intensity", False))
    return {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": seed,
        "num_samples": num_samples,
        "tsne_perplexity": tsne_perplexity,
        "outdir": outdir,
        "save_pdf": save_pdf,
        "include_optical_intensity": include_optical_intensity,
    }


def _scatter_plot(points: np.ndarray, labels: np.ndarray, title: str, out_path: Path, save_pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=160)
    uniq = np.unique(labels)
    cmap = plt.get_cmap("tab10")
    for i, cls in enumerate(uniq):
        mask = labels == cls
        ax.scatter(points[mask, 0], points[mask, 1], s=10, alpha=0.75, label=str(int(cls)), color=cmap(i % 10))
    if len(uniq) <= 12:
        ax.legend(loc="best", fontsize=8, frameon=False)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Dim-1")
    ax.set_ylabel("Dim-2")
    ax.grid(alpha=0.2, linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _hist_plot(sample_means: np.ndarray, sample_stds: np.ndarray, title: str, out_path: Path, save_pdf: bool) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=160)
    axes[0].hist(sample_means, bins=40, color="#4C72B0", alpha=0.85, edgecolor="black", linewidth=0.4)
    axes[0].set_title("Per-sample Mean")
    axes[0].set_xlabel("mean")
    axes[0].set_ylabel("count")
    axes[1].hist(sample_stds, bins=40, color="#DD8452", alpha=0.85, edgecolor="black", linewidth=0.4)
    axes[1].set_title("Per-sample Std")
    axes[1].set_xlabel("std")
    axes[1].set_ylabel("count")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_center_distance_matrix(
    matrix: np.ndarray,
    labels: List[int],
    title: str,
    out_path: Path,
    save_pdf: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=160)
    im = ax.imshow(matrix, cmap="viridis")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([str(int(v)) for v in labels], fontsize=8)
    ax.set_yticklabels([str(int(v)) for v in labels], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _pca_2d(x: np.ndarray, seed: int) -> np.ndarray:
    if SKLEARN_OK:
        return PCA(n_components=2, random_state=seed).fit_transform(x)
    x0 = x - x.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    return x0 @ vt[:2].T


def _tsne_2d(x: np.ndarray, seed: int, perplexity: float) -> np.ndarray:
    if not SKLEARN_OK:
        raise RuntimeError("scikit-learn unavailable for t-SNE")
    n = x.shape[0]
    if n < 5:
        raise ValueError("Need at least 5 samples for t-SNE")
    p = min(float(perplexity), float((n - 1) / 3.0))
    p = max(2.0, p)
    pre_dim = min(50, x.shape[1], n - 1)
    x_pre = PCA(n_components=pre_dim, random_state=seed).fit_transform(x) if pre_dim >= 2 else x
    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=p)
    return tsne.fit_transform(x_pre)


def _class_center_distance(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    cls = sorted([int(v) for v in np.unique(y)])
    centers = []
    for c in cls:
        mask = y == c
        centers.append(x[mask].mean(axis=0))
    centers_arr = np.stack(centers, axis=0)
    diff = centers_arr[:, None, :] - centers_arr[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    return dist, cls


def _save_embedding_csv(points: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    rows = []
    for i in range(points.shape[0]):
        rows.append({"index": int(i), "label": int(labels[i]), "x": float(points[i, 0]), "y": float(points[i, 1])})
    write_summary_csv(out_path, rows)


def _analyze_one_mode(
    mode_name: str,
    latent: np.ndarray,
    labels: np.ndarray,
    fig_dir: Path,
    metrics_dir: Path,
    seed: int,
    tsne_perplexity: float,
    save_pdf: bool,
) -> Dict[str, Any]:
    pca_pts = _pca_2d(latent, seed=seed)
    _scatter_plot(
        pca_pts,
        labels,
        title="{} Latent PCA".format(mode_name),
        out_path=fig_dir / "latent_pca_{}.png".format(mode_name),
        save_pdf=save_pdf,
    )
    _save_embedding_csv(pca_pts, labels, metrics_dir / "embedding_pca_{}.csv".format(mode_name))

    tsne_ok = False
    tsne_pts = np.empty((0, 2), dtype=np.float32)
    if SKLEARN_OK and latent.shape[0] >= 5:
        try:
            tsne_pts = _tsne_2d(latent, seed=seed, perplexity=tsne_perplexity)
            _scatter_plot(
                tsne_pts,
                labels,
                title="{} Latent t-SNE".format(mode_name),
                out_path=fig_dir / "latent_tsne_{}.png".format(mode_name),
                save_pdf=save_pdf,
            )
            _save_embedding_csv(tsne_pts, labels, metrics_dir / "embedding_tsne_{}.csv".format(mode_name))
            tsne_ok = True
        except Exception:
            tsne_ok = False

    sample_means = latent.mean(axis=1)
    sample_stds = latent.std(axis=1)
    _hist_plot(
        sample_means,
        sample_stds,
        title="{} Latent Distribution".format(mode_name),
        out_path=fig_dir / "latent_hist_{}.png".format(mode_name),
        save_pdf=save_pdf,
    )

    center_dist, cls = _class_center_distance(latent, labels)
    _plot_center_distance_matrix(
        matrix=center_dist,
        labels=cls,
        title="{} Class Center Distance".format(mode_name),
        out_path=fig_dir / "latent_center_dist_{}.png".format(mode_name),
        save_pdf=save_pdf,
    )
    dist_rows = []
    for i, c1 in enumerate(cls):
        row = {"class": int(c1)}
        for j, c2 in enumerate(cls):
            row["d_to_{}".format(int(c2))] = float(center_dist[i, j])
        dist_rows.append(row)
    write_summary_csv(metrics_dir / "class_center_dist_{}.csv".format(mode_name), dist_rows)

    sil = None
    if SKLEARN_OK and len(np.unique(labels)) >= 2 and latent.shape[0] >= 10:
        try:
            sil = float(silhouette_score(latent, labels))
        except Exception:
            sil = None

    stats = {
        "mode": mode_name,
        "n_samples": int(latent.shape[0]),
        "latent_dim": int(latent.shape[1]),
        "global_mean": float(latent.mean()),
        "global_std": float(latent.std()),
        "sample_mean_mean": float(sample_means.mean()),
        "sample_mean_std": float(sample_means.std()),
        "sample_std_mean": float(sample_stds.mean()),
        "sample_std_std": float(sample_stds.std()),
        "silhouette": sil,
        "tsne_available": bool(tsne_ok),
    }

    save_json(stats, metrics_dir / "latent_stats_{}.json".format(mode_name))
    write_summary_csv(metrics_dir / "latent_stats_{}.csv".format(mode_name), [stats])

    np.savez_compressed(
        metrics_dir / "latent_embeddings_{}.npz".format(mode_name),
        latent=latent.astype(np.float32),
        labels=labels.astype(np.int64),
        pca=pca_pts.astype(np.float32),
        tsne=tsne_pts.astype(np.float32),
        class_center_distance=center_dist.astype(np.float32),
        class_ids=np.asarray(cls, dtype=np.int64),
    )
    return stats


def main() -> None:
    args = parse_args()
    cfg = load_analysis_config(args.config)
    runtime = _resolve_runtime(cfg, args)
    set_seed(int(runtime["seed"]))

    ckpt_cfg = cfg.get("checkpoints", {})
    ckpt_e = ckpt_cfg.get("electronic", None)
    ckpt_o = ckpt_cfg.get("optical", None)
    if ckpt_e is None or ckpt_o is None:
        raise ValueError("config.checkpoints.electronic and config.checkpoints.optical are required")

    outdir = Path(runtime["outdir"])
    fig_dir, metrics_dir = ensure_analysis_dirs(outdir)
    save_config_used({"input_config": cfg, "runtime": runtime}, outdir / "config_used.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle_e = load_model_bundle_from_checkpoint(ckpt_e, device=device, mode="electronic")
    bundle_o = load_model_bundle_from_checkpoint(ckpt_o, device=device, mode="optical")

    dataset = runtime["dataset"] or bundle_e["dataset"]
    if dataset is None:
        raise ValueError("Dataset is missing in args/config/checkpoint.")
    if bundle_o["out_range"] != bundle_e["out_range"] or tuple(bundle_o["image_size"]) != tuple(bundle_e["image_size"]):
        raise ValueError("Electronic/optical image setting mismatch; cannot compare directly.")

    test_loader, _ = build_test_loader(
        dataset=dataset,
        data_root=args.data_root,
        out_range=bundle_e["out_range"],
        image_size=bundle_e["image_size"],
        batch_size=int(runtime["batch_size"]),
        num_workers=int(runtime["num_workers"]),
        seed=int(runtime["seed"]),
    )

    target_n = int(runtime["num_samples"])
    lat_e: List[np.ndarray] = []
    lat_o: List[np.ndarray] = []
    lat_oi: List[np.ndarray] = []
    labels_all: List[np.ndarray] = []
    count = 0

    with torch.no_grad():
        for x, y in test_loader:
            if count >= target_n:
                break
            take = min(target_n - count, x.size(0))
            if take <= 0:
                break
            xb = x[:take].to(device, non_blocking=True)
            yb = y[:take]

            z_e, _ = extract_decoder_latent(bundle_e, xb, return_info=True)
            z_o, info_o = extract_decoder_latent(bundle_o, xb, return_info=True)

            lat_e.append(flatten_latent(z_e))
            lat_o.append(flatten_latent(z_o))
            if bool(runtime["include_optical_intensity"]) and isinstance(info_o, dict) and "latent_intensity_map" in info_o:
                lat_oi.append(flatten_latent(info_o["latent_intensity_map"]))
            labels_all.append(yb.numpy())
            count += take

    if count == 0:
        raise RuntimeError("No samples collected from test set.")

    labels_np = np.concatenate(labels_all, axis=0)
    reprs: Dict[str, np.ndarray] = {
        "electronic": np.concatenate(lat_e, axis=0),
        "optical": np.concatenate(lat_o, axis=0),
    }
    if len(lat_oi) > 0:
        reprs["optical_intensity"] = np.concatenate(lat_oi, axis=0)

    summary_rows = []
    summary_json = {"dataset": dataset, "num_samples": int(count), "modes": {}}
    for mode_name, latent_np in reprs.items():
        stats = _analyze_one_mode(
            mode_name=mode_name,
            latent=latent_np,
            labels=labels_np,
            fig_dir=fig_dir,
            metrics_dir=metrics_dir,
            seed=int(runtime["seed"]),
            tsne_perplexity=float(runtime["tsne_perplexity"]),
            save_pdf=bool(runtime["save_pdf"]),
        )
        summary_rows.append(stats)
        summary_json["modes"][mode_name] = stats

    write_summary_csv(metrics_dir / "latent_stats_summary.csv", summary_rows)
    save_json(summary_json, metrics_dir / "latent_stats_summary.json")
    print("Saved latent structure analysis to: {}".format(outdir))


if __name__ == "__main__":
    main()
