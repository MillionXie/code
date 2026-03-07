from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid, save_image
import math

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
    resolve_checkpoint_groups,
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
    mode_label: str,
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
                "mode": mode_label,
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
    model_name: str,
    x1: torch.Tensor,
    x2: torch.Tensor,
    out_prefix: Path,
    out_range: str,
    steps: int,
    sigma_interp: float,
    save_pdf: bool,
) -> Dict[str, Any]:
    def _latent_to_vis_map(z: torch.Tensor) -> torch.Tensor:
        # z: [B,C,H,W] (map) or [B,D] (vector)
        if z.dim() == 4:
            return z.mean(dim=1, keepdim=True)
        if z.dim() == 2:
            b, d = z.shape
            side = int(math.ceil(math.sqrt(float(d))))
            pad = side * side - d
            if pad > 0:
                z = torch.cat([z, torch.zeros(b, pad, device=z.device, dtype=z.dtype)], dim=1)
            return z.view(b, 1, side, side)
        raise ValueError("Unsupported latent shape for visualization: {}".format(tuple(z.shape)))

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
            latent_maps.append(_latent_to_vis_map(zt))
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
        "model_name": model_name,
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

    ckpt_grp = resolve_checkpoint_groups(cfg)
    ckpt_e = ckpt_grp.get("electronic", None)
    optical_entries = ckpt_grp.get("opticals", [])
    if ckpt_e is None or len(optical_entries) == 0:
        raise ValueError("Need checkpoints.electronic and at least one optical checkpoint (optical/opticals).")

    outdir = Path(runtime["outdir"])
    fig_dir, metrics_dir = ensure_analysis_dirs(outdir)
    save_config_used({"input_config": cfg, "runtime": runtime}, outdir / "config_used.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle_e = load_model_bundle_from_checkpoint(ckpt_e, device=device, mode="electronic")
    optical_bundles = []
    for item in optical_entries:
        optical_bundles.append({"name": item["name"], "bundle": load_model_bundle_from_checkpoint(item["path"], device=device, mode="optical")})

    dataset = runtime["dataset"] or bundle_e["dataset"]
    if dataset is None:
        raise ValueError("Dataset is missing in args/config/checkpoint.")

    entries: List[Dict[str, Any]] = [{"name": "electronic", "tag": "electronic", "bundle": bundle_e}]
    for item in optical_bundles:
        entries.append({"name": item["name"], "tag": "optical", "bundle": item["bundle"]})

    loader_cache: Dict[str, Any] = {}

    noise_rows = []
    for entry in entries:
        bundle = entry["bundle"]
        if (runtime["dataset"] or bundle["dataset"]) != dataset:
            raise ValueError(
                "Dataset mismatch for {}: expected {}, got {}".format(entry["name"], dataset, bundle["dataset"])
            )
        cache_key = "{}|{}|{}x{}".format(
            dataset,
            bundle.get("out_range"),
            int(bundle["image_size"][0]),
            int(bundle["image_size"][1]),
        )
        if cache_key not in loader_cache:
            test_loader, dataset_info = build_test_loader(
                dataset=dataset,
                data_root=args.data_root,
                out_range=bundle["out_range"],
                image_size=bundle["image_size"],
                batch_size=int(runtime["batch_size"]),
                num_workers=int(runtime["num_workers"]),
                seed=int(runtime["seed"]),
            )
            loader_cache[cache_key] = (test_loader, dataset_info)
        else:
            test_loader, dataset_info = loader_cache[cache_key]

        mode_label = "{}:{}".format(entry["tag"], entry["name"])
        noise_rows.extend(
            _evaluate_noise_for_mode(
                bundle=bundle,
                mode_label=mode_label,
                test_loader=test_loader,
                data_range=float(dataset_info.get("data_range", 1.0)),
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

    interp_items = []
    for entry in entries:
        bundle = entry["bundle"]
        cache_key = "{}|{}|{}x{}".format(
            dataset,
            bundle.get("out_range"),
            int(bundle["image_size"][0]),
            int(bundle["image_size"][1]),
        )
        test_loader, _ = loader_cache[cache_key]
        x1, x2, y1, y2, idx1, idx2 = _find_interp_pair(test_loader, runtime["fixed_indices"])
        safe_name = str(entry["name"]).replace("/", "_").replace("\\", "_").replace(" ", "_")
        prefix = fig_dir / "interpolation_{}_{}".format(entry["tag"], safe_name)
        info = _interp_and_save(
            bundle=bundle,
            model_name=entry["name"],
            x1=x1,
            x2=x2,
            out_prefix=prefix,
            out_range=bundle["out_range"],
            steps=int(runtime["interp_steps"]),
            sigma_interp=float(runtime["interp_noise_sigma"]),
            save_pdf=bool(runtime["save_pdf"]),
        )
        info.update(
            {
                "mode_tag": entry["tag"],
                "model_name": entry["name"],
                "index_1": int(idx1),
                "index_2": int(idx2),
                "label_1": int(y1),
                "label_2": int(y2),
            }
        )
        interp_items.append(info)

    summary = {
        "dataset": dataset,
        "num_samples_recon_noise": int(runtime["num_samples"]),
        "noise_sigmas": [float(v) for v in runtime["noise_sigmas"]],
        "interpolations": interp_items,
    }
    save_json(summary, metrics_dir / "noise_summary.json")

    npz_data = {
        "sigmas": np.asarray([float(v) for v in runtime["noise_sigmas"]], dtype=np.float32),
    }
    mode_names = sorted(list({str(r["mode"]) for r in noise_rows}))
    for mode_name in mode_names:
        safe_mode = mode_name.replace(":", "_").replace("/", "_").replace("\\", "_").replace(" ", "_")
        npz_data["{}_mse".format(safe_mode)] = np.asarray(
            [r["mse"] for r in noise_rows if str(r["mode"]) == mode_name], dtype=np.float32
        )
        npz_data["{}_psnr".format(safe_mode)] = np.asarray(
            [r["psnr"] for r in noise_rows if str(r["mode"]) == mode_name], dtype=np.float32
        )
        npz_data["{}_ssim".format(safe_mode)] = np.asarray(
            [r["ssim"] for r in noise_rows if str(r["mode"]) == mode_name], dtype=np.float32
        )
    np.savez_compressed(metrics_dir / "noise_recon_curves.npz", **npz_data)
    print("Saved latent noise analysis to: {}".format(outdir))


if __name__ == "__main__":
    main()
