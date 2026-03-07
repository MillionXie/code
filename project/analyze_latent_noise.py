from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid, save_image

from utils.io import save_json, write_summary_csv
from utils.latent_compare import (
    build_test_loader,
    decode_from_latent,
    ensure_analysis_dirs,
    evaluate_batch_metrics,
    extract_decoder_latent,
    fetch_dataset_indices,
    load_analysis_config,
    load_model_bundle_from_checkpoint,
    save_config_used,
    to_display_range,
)
from utils.seed import set_seed

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task4: Latent noise robustness analysis")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashionmnist", "cifar10"], default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--noise_sigmas", type=str, default=None, help="Comma-separated list, e.g. 0,0.05,0.1")
    parser.add_argument("--interp_steps", type=int, default=None)
    parser.add_argument("--interp_noise_sigma", type=float, default=None)
    return parser.parse_args()


def _resolve_runtime(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    analysis_cfg = cfg.get("analysis", {})
    dataset = args.dataset or cfg.get("dataset", analysis_cfg.get("dataset", None))
    batch_size = int(args.batch_size if args.batch_size is not None else analysis_cfg.get("batch_size", 256))
    num_workers = int(args.num_workers if args.num_workers is not None else analysis_cfg.get("num_workers", 4))
    seed = int(args.seed if args.seed is not None else analysis_cfg.get("seed", 42))
    num_samples = int(args.num_samples if args.num_samples is not None else analysis_cfg.get("num_samples", 1000))
    interp_steps = int(args.interp_steps if args.interp_steps is not None else analysis_cfg.get("interp_steps", 9))
    interp_noise_sigma = float(
        args.interp_noise_sigma if args.interp_noise_sigma is not None else analysis_cfg.get("interp_noise_sigma", 0.0)
    )
    outdir = args.outdir or cfg.get("outdir", analysis_cfg.get("outdir", None))
    if outdir is None:
        outdir = str(Path("outputs") / "task4_latent_noise")
    if args.noise_sigmas is not None:
        noise_sigmas = [float(v.strip()) for v in str(args.noise_sigmas).split(",") if v.strip()]
    else:
        noise_sigmas = [float(v) for v in analysis_cfg.get("noise_sigmas", [0.0, 0.05, 0.1, 0.2, 0.3])]
    if len(noise_sigmas) == 0:
        noise_sigmas = [0.0]
    fixed_indices = [int(v) for v in analysis_cfg.get("fixed_indices", [0, 1])]
    if len(fixed_indices) < 2:
        fixed_indices = [0, 1]
    save_pdf = bool(analysis_cfg.get("save_pdf", True))
    return {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": seed,
        "num_samples": num_samples,
        "noise_sigmas": noise_sigmas,
        "interp_steps": interp_steps,
        "interp_noise_sigma": interp_noise_sigma,
        "fixed_indices": fixed_indices,
        "outdir": outdir,
        "save_pdf": save_pdf,
    }


def _save_grid_png_pdf(images: torch.Tensor, out_prefix: Path, out_range: str, nrow: int, save_pdf: bool) -> None:
    vis = to_display_range(images.detach().cpu(), out_range=out_range)
    grid = make_grid(vis, nrow=nrow, padding=2)
    png_path = out_prefix.with_suffix(".png")
    save_image(grid, png_path)
    if save_pdf:
        fig, ax = plt.subplots(figsize=(max(4, nrow * 1.2), 3.5), dpi=160)
        g = grid.detach().cpu()
        if g.shape[0] == 1:
            ax.imshow(g[0].numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(np.transpose(g.numpy(), (1, 2, 0)))
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_prefix.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)


def _plot_noise_curves(rows: List[Dict[str, Any]], out_path: Path, save_pdf: bool) -> None:
    modes = sorted(list({str(r["mode"]) for r in rows}))
    sigmas = sorted(list({float(r["sigma"]) for r in rows}))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), dpi=160)
    metrics = [("mse", "MSE"), ("psnr", "PSNR"), ("ssim", "SSIM")]
    colors = {"electronic": "#4C72B0", "optical": "#DD8452"}

    for m_idx, (key, title) in enumerate(metrics):
        ax = axes[m_idx]
        for mode in modes:
            vals = []
            for s in sigmas:
                found = [r for r in rows if str(r["mode"]) == mode and float(r["sigma"]) == float(s)]
                vals.append(float(found[0][key]) if len(found) > 0 else np.nan)
            ax.plot(sigmas, vals, marker="o", linewidth=1.8, label=mode, color=colors.get(mode, None))
        ax.set_title(title)
        ax.set_xlabel("Noise sigma")
        ax.grid(alpha=0.25, linewidth=0.5)
    axes[0].set_ylabel("Metric")
    axes[-1].legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    if save_pdf:
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _evaluate_noise_for_mode(
    bundle: Dict[str, Any],
    test_loader,
    data_range: float,
    sigmas: List[float],
    num_samples: int,
) -> List[Dict[str, Any]]:
    device = next(bundle["model"].parameters()).device
    rows: List[Dict[str, Any]] = []

    for sigma in sigmas:
        mse_sum = 0.0
        psnr_sum = 0.0
        ssim_sum = 0.0
        count = 0
        with torch.no_grad():
            for x, _ in test_loader:
                if count >= num_samples:
                    break
                take = min(num_samples - count, x.size(0))
                if take <= 0:
                    break
                xb = x[:take].to(device, non_blocking=True)
                z, _ = extract_decoder_latent(bundle, xb, return_info=False)
                if float(sigma) > 0.0:
                    z = z + float(sigma) * torch.randn_like(z)
                recon = decode_from_latent(bundle, z)
                m = evaluate_batch_metrics(recon, xb, data_range=data_range)
                mse_sum += m["mse"].sum().item()
                psnr_sum += m["psnr"].sum().item()
                ssim_sum += m["ssim"].sum().item()
                count += take

        rows.append(
            {
                "mode": bundle["mode"],
                "sigma": float(sigma),
                "samples": int(count),
                "mse": float(mse_sum / max(count, 1)),
                "psnr": float(psnr_sum / max(count, 1)),
                "ssim": float(ssim_sum / max(count, 1)),
            }
        )
    return rows


def _find_interp_pair(test_loader, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor, int, int, int, int]:
    x_fix, y_fix, valid_idx = fetch_dataset_indices(test_loader, indices)
    if x_fix.shape[0] >= 2:
        return x_fix[0:1], x_fix[1:2], int(y_fix[0].item()), int(y_fix[1].item()), valid_idx[0], valid_idx[1]

    # fallback: first two samples from test set
    all_x = []
    all_y = []
    for x, y in test_loader:
        all_x.append(x)
        all_y.append(y)
        if sum(int(v.shape[0]) for v in all_x) >= 2:
            break
    x_cat = torch.cat(all_x, dim=0)
    y_cat = torch.cat(all_y, dim=0)
    return x_cat[0:1], x_cat[1:2], int(y_cat[0].item()), int(y_cat[1].item()), 0, 1


def _interp_and_save(
    bundle: Dict[str, Any],
    x1: torch.Tensor,
    x2: torch.Tensor,
    out_prefix: Path,
    out_range: str,
    steps: int,
    sigma_interp: float,
    save_pdf: bool,
) -> Dict[str, Any]:
    device = next(bundle["model"].parameters()).device
    with torch.no_grad():
        xb = torch.cat([x1, x2], dim=0).to(device)
        z_pair, _ = extract_decoder_latent(bundle, xb, return_info=False)
        z1, z2 = z_pair[0:1], z_pair[1:2]
        ts = torch.linspace(0.0, 1.0, steps=steps, device=device)
        zs = []
        latent_maps = []
        for t in ts:
            zt = z1 * (1.0 - t) + z2 * t
            if sigma_interp > 0:
                zt = zt + sigma_interp * torch.randn_like(zt)
            zs.append(zt)
            latent_maps.append(zt.mean(dim=1, keepdim=True))
        z_interp = torch.cat(zs, dim=0)
        recon = decode_from_latent(bundle, z_interp)
        latent_grid = torch.cat(latent_maps, dim=0)

    _save_grid_png_pdf(recon, out_prefix=out_prefix, out_range=out_range, nrow=steps, save_pdf=save_pdf)
    _save_grid_png_pdf(
        to_display_range(latent_grid, "zero_one"),
        out_prefix=Path(str(out_prefix) + "_latent"),
        out_range="zero_one",
        nrow=steps,
        save_pdf=save_pdf,
    )
    return {
        "steps": int(steps),
        "sigma_interp": float(sigma_interp),
        "recon_shape": list(recon.shape),
        "latent_shape": list(latent_grid.shape),
    }


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

    test_loader, dataset_info = build_test_loader(
        dataset=dataset,
        data_root=args.data_root,
        out_range=bundle_e["out_range"],
        image_size=bundle_e["image_size"],
        batch_size=int(runtime["batch_size"]),
        num_workers=int(runtime["num_workers"]),
        seed=int(runtime["seed"]),
    )

    data_range = float(dataset_info.get("data_range", 1.0))
    noise_rows = []
    noise_rows.extend(
        _evaluate_noise_for_mode(
            bundle=bundle_e,
            test_loader=test_loader,
            data_range=data_range,
            sigmas=[float(v) for v in runtime["noise_sigmas"]],
            num_samples=int(runtime["num_samples"]),
        )
    )
    noise_rows.extend(
        _evaluate_noise_for_mode(
            bundle=bundle_o,
            test_loader=test_loader,
            data_range=data_range,
            sigmas=[float(v) for v in runtime["noise_sigmas"]],
            num_samples=int(runtime["num_samples"]),
        )
    )

    write_summary_csv(metrics_dir / "noise_recon_metrics.csv", noise_rows)
    save_json({"rows": noise_rows}, metrics_dir / "noise_recon_metrics.json")
    _plot_noise_curves(
        rows=noise_rows,
        out_path=fig_dir / "noise_recon_curves.png",
        save_pdf=bool(runtime["save_pdf"]),
    )

    x1, x2, y1, y2, idx1, idx2 = _find_interp_pair(test_loader, runtime["fixed_indices"])
    interp_e = _interp_and_save(
        bundle=bundle_e,
        x1=x1,
        x2=x2,
        out_prefix=fig_dir / "interpolation_electronic",
        out_range=bundle_e["out_range"],
        steps=int(runtime["interp_steps"]),
        sigma_interp=float(runtime["interp_noise_sigma"]),
        save_pdf=bool(runtime["save_pdf"]),
    )
    interp_o = _interp_and_save(
        bundle=bundle_o,
        x1=x1,
        x2=x2,
        out_prefix=fig_dir / "interpolation_optical",
        out_range=bundle_o["out_range"],
        steps=int(runtime["interp_steps"]),
        sigma_interp=float(runtime["interp_noise_sigma"]),
        save_pdf=bool(runtime["save_pdf"]),
    )

    summary = {
        "dataset": dataset,
        "num_samples_recon_noise": int(runtime["num_samples"]),
        "noise_sigmas": [float(v) for v in runtime["noise_sigmas"]],
        "interpolation_pair": {
            "index_1": int(idx1),
            "index_2": int(idx2),
            "label_1": int(y1),
            "label_2": int(y2),
        },
        "interpolation_electronic": interp_e,
        "interpolation_optical": interp_o,
    }
    save_json(summary, metrics_dir / "noise_summary.json")

    np.savez_compressed(
        metrics_dir / "noise_recon_curves.npz",
        sigmas=np.asarray([float(v) for v in runtime["noise_sigmas"]], dtype=np.float32),
        electronic_mse=np.asarray([r["mse"] for r in noise_rows if r["mode"] == "electronic"], dtype=np.float32),
        optical_mse=np.asarray([r["mse"] for r in noise_rows if r["mode"] == "optical"], dtype=np.float32),
        electronic_psnr=np.asarray([r["psnr"] for r in noise_rows if r["mode"] == "electronic"], dtype=np.float32),
        optical_psnr=np.asarray([r["psnr"] for r in noise_rows if r["mode"] == "optical"], dtype=np.float32),
        electronic_ssim=np.asarray([r["ssim"] for r in noise_rows if r["mode"] == "electronic"], dtype=np.float32),
        optical_ssim=np.asarray([r["ssim"] for r in noise_rows if r["mode"] == "optical"], dtype=np.float32),
    )
    print("Saved latent noise analysis to: {}".format(outdir))


if __name__ == "__main__":
    main()
