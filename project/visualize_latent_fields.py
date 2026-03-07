from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import math

from utils.io import save_json, write_summary_csv
from utils.latent_compare import (
    build_test_loader,
    decode_from_latent,
    ensure_analysis_dirs,
    extract_decoder_latent,
    fetch_dataset_indices,
    load_analysis_config,
    load_model_bundle_from_checkpoint,
    resolve_checkpoint_groups,
    normalize_log_intensity_for_display,
    pick_stage_intensity_map,
    save_config_used,
    to_display_range,
)
from utils.seed import set_seed

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task4: Visualize electronic vs optical latent pipelines")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashionmnist", "cifar10"], default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--mode", type=str, choices=["electronic", "optical", "both"], default="both")
    parser.add_argument("--sample_indices", type=str, default=None, help="Comma-separated sample indices")
    return parser.parse_args()


def _resolve_runtime(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    analysis_cfg = cfg.get("analysis", {})
    dataset = args.dataset or cfg.get("dataset", analysis_cfg.get("dataset", None))
    batch_size = int(args.batch_size if args.batch_size is not None else analysis_cfg.get("batch_size", 256))
    num_workers = int(args.num_workers if args.num_workers is not None else analysis_cfg.get("num_workers", 4))
    seed = int(args.seed if args.seed is not None else analysis_cfg.get("seed", 42))
    outdir = args.outdir or cfg.get("outdir", analysis_cfg.get("outdir", None))
    if outdir is None:
        outdir = str(Path("outputs") / "task4_visualize_latent_fields")

    if args.sample_indices is not None:
        fixed_indices = [int(x.strip()) for x in str(args.sample_indices).split(",") if x.strip()]
    else:
        fixed_indices = [int(v) for v in analysis_cfg.get("fixed_indices", [0, 1, 2, 3, 4])]
    if len(fixed_indices) == 0:
        fixed_indices = [0]

    save_pdf = bool(analysis_cfg.get("save_pdf", True))
    optical_vis = analysis_cfg.get("optical_vis", {})
    electronic_vis = analysis_cfg.get("electronic_vis", {})
    return {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": seed,
        "outdir": outdir,
        "fixed_indices": fixed_indices,
        "save_pdf": save_pdf,
        "optical_vis": optical_vis,
        "electronic_vis": electronic_vis,
    }


def _resolve_bundles(
    cfg: Dict[str, Any],
    device: torch.device,
    mode: str,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    ckpt_grp = resolve_checkpoint_groups(cfg)
    bundle_e: Optional[Dict[str, Any]] = None
    optical_bundles: List[Dict[str, Any]] = []

    if mode in ("electronic", "both"):
        ckpt_e = ckpt_grp.get("electronic", None)
        if ckpt_e is None:
            raise ValueError("config.checkpoints.electronic is required for electronic/both mode")
        bundle_e = load_model_bundle_from_checkpoint(ckpt_e, device=device, mode="electronic")

    if mode in ("optical", "both"):
        optical_entries = ckpt_grp.get("opticals", [])
        if len(optical_entries) == 0:
            raise ValueError("At least one optical checkpoint is required for optical/both mode")
        for item in optical_entries:
            bundle_o = load_model_bundle_from_checkpoint(item["path"], device=device, mode="optical")
            optical_bundles.append({"name": item["name"], "bundle": bundle_o})

    return bundle_e, optical_bundles


def _tensor_to_plot_image(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu()
    if x.dim() == 4:
        x = x[0]
    if x.dim() != 3:
        raise ValueError("Expected [C,H,W] or [1,C,H,W], got {}".format(tuple(x.shape)))
    if x.shape[0] == 1:
        return x[0].numpy()
    if x.shape[0] == 3:
        return np.transpose(x.numpy(), (1, 2, 0))
    return x[0].numpy()


def _show_image_axis(ax, img: np.ndarray, title: str) -> None:
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    else:
        ax.imshow(np.clip(img, 0.0, 1.0))
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _show_intensity_axis(ax, intensity_map: torch.Tensor, title: str, cmap: str) -> None:
    if intensity_map.dim() == 3:
        intensity_map = intensity_map.unsqueeze(0)
    show = normalize_log_intensity_for_display(torch.clamp(intensity_map[:1, :1], min=0.0))[0, 0].detach().cpu().numpy()
    ax.imshow(show, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def _build_optical_figure(
    bundle: Dict[str, Any],
    model_name: str,
    x_sample: torch.Tensor,
    sample_index: int,
    fig_dir: Path,
    vis_cfg: Dict[str, Any],
    save_pdf: bool,
) -> None:
    device = next(bundle["model"].parameters()).device
    out_range = bundle["out_range"]
    cmap = str(vis_cfg.get("cmap", "magma"))
    _ = bool(vis_cfg.get("use_log1p", True))

    with torch.no_grad():
        x = x_sample.unsqueeze(0).to(device)
        z_mid, info = extract_decoder_latent(bundle, x, return_info=True)
        recon = decode_from_latent(bundle, z_mid)

    input_img = _tensor_to_plot_image(to_display_range(x.detach().cpu(), out_range)[0])
    recon_img = _tensor_to_plot_image(to_display_range(recon.detach().cpu(), out_range)[0])

    i0 = pick_stage_intensity_map(info or {}, ["field_input", "input_intensity", "I0"])
    i3 = pick_stage_intensity_map(info or {}, ["sensor_pre_pool", "I3", "after_prop2", "before_sensor"])

    sensor_info = (info or {}).get("sensor", {})
    latent_intensity = sensor_info.get("pooled_intensity", None)
    if latent_intensity is None:
        latent_intensity = (info or {}).get("latent_intensity_map", None)
    if latent_intensity is None:
        latent_intensity = z_mid

    if i0 is None:
        i0 = torch.clamp(x, min=0.0)
    if i3 is None:
        i3 = torch.clamp(latent_intensity, min=0.0)

    plt.rcParams.update({"font.size": 10, "figure.dpi": 160})
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.2))
    _show_image_axis(axes[0], input_img, "Input")
    _show_intensity_axis(axes[1], i0.detach().cpu(), r"Input Field $I_0$", cmap=cmap)
    _show_intensity_axis(axes[2], i3.detach().cpu(), r"Scattered Field $I_3$", cmap=cmap)
    _show_intensity_axis(axes[3], latent_intensity.detach().cpu(), "Optical Latent", cmap=cmap)
    _show_image_axis(axes[4], recon_img, "Reconstruction")
    fig.suptitle("Optical Pipeline | {} | sample={}".format(model_name, sample_index), fontsize=11, y=1.02)
    fig.tight_layout()

    safe = str(model_name).replace("/", "_").replace("\\", "_").replace(" ", "_")
    png_path = fig_dir / "optical_pipeline_{}_sample_{:04d}.png".format(safe, sample_index)
    fig.savefig(png_path, bbox_inches="tight")
    if save_pdf:
        fig.savefig(fig_dir / "optical_pipeline_{}_sample_{:04d}.pdf".format(safe, sample_index), bbox_inches="tight")
    plt.close(fig)


def _build_electronic_figure(
    bundle: Dict[str, Any],
    model_name: str,
    x_sample: torch.Tensor,
    sample_index: int,
    fig_dir: Path,
    vis_cfg: Dict[str, Any],
    save_pdf: bool,
) -> None:
    device = next(bundle["model"].parameters()).device
    out_range = bundle["out_range"]
    topk = max(1, int(vis_cfg.get("topk_channels", 4)))

    with torch.no_grad():
        x = x_sample.unsqueeze(0).to(device)
        z_mid, _ = extract_decoder_latent(bundle, x, return_info=True)
        recon = decode_from_latent(bundle, z_mid)

    input_img = _tensor_to_plot_image(to_display_range(x.detach().cpu(), out_range)[0])
    recon_img = _tensor_to_plot_image(to_display_range(recon.detach().cpu(), out_range)[0])

    latent = z_mid.detach().cpu()
    vector_latent = latent.dim() == 2
    if not vector_latent:
        # map-latent case: [B,C,H,W]
        lat0 = latent[0]
        mean_map = lat0.mean(dim=0)
        ch_scores = lat0.flatten(start_dim=1).std(dim=1)
        topk = min(topk, lat0.shape[0])
        top_idx = torch.topk(ch_scores, k=topk, largest=True).indices.tolist()
    else:
        # vector-latent case: [B,D], reshape to square heatmap for visualization.
        vec = latent[0]
        d = int(vec.numel())
        side = int(math.ceil(math.sqrt(float(d))))
        pad = side * side - d
        if pad > 0:
            vec = torch.cat([vec, torch.zeros(pad, dtype=vec.dtype)], dim=0)
        mean_map = vec.view(side, side)
        topk = min(topk, d)
        top_idx = torch.topk(torch.abs(latent[0]), k=topk, largest=True).indices.tolist()

    plt.rcParams.update({"font.size": 10, "figure.dpi": 160})
    fig = plt.figure(figsize=(14, 3.4))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.0, 1.0, 1.2, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    _show_image_axis(ax0, input_img, "Input")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(mean_map.numpy(), cmap="viridis")
    ax1.set_title("Latent Heatmap", fontsize=10)
    ax1.axis("off")

    subgs = gs[0, 2].subgridspec(2, 2)
    for i in range(4):
        ax = fig.add_subplot(subgs[i // 2, i % 2])
        if i < len(top_idx):
            idx = top_idx[i]
            if not vector_latent:
                ax.imshow(lat0[idx].numpy(), cmap="inferno")
                ax.set_title("ch{}".format(idx), fontsize=8)
            else:
                value = float(latent[0, idx].item())
                tile = np.ones((10, 10), dtype=np.float32) * value
                ax.imshow(tile, cmap="inferno")
                ax.set_title("z{}".format(idx), fontsize=8)
        ax.axis("off")

    ax3 = fig.add_subplot(gs[0, 3])
    _show_image_axis(ax3, recon_img, "Reconstruction")

    fig.suptitle("Electronic Pipeline | {} | sample={}".format(model_name, sample_index), fontsize=11, y=1.02)
    fig.tight_layout()

    safe = str(model_name).replace("/", "_").replace("\\", "_").replace(" ", "_")
    png_path = fig_dir / "electronic_pipeline_{}_sample_{:04d}.png".format(safe, sample_index)
    fig.savefig(png_path, bbox_inches="tight")
    if save_pdf:
        fig.savefig(fig_dir / "electronic_pipeline_{}_sample_{:04d}.pdf".format(safe, sample_index), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cfg = load_analysis_config(args.config)
    runtime = _resolve_runtime(cfg, args)
    set_seed(int(runtime["seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle_e, optical_bundles = _resolve_bundles(cfg=cfg, device=device, mode=str(args.mode))

    bundle_ref = bundle_e if bundle_e is not None else (optical_bundles[0]["bundle"] if len(optical_bundles) > 0 else None)
    if bundle_ref is None:
        raise RuntimeError("No bundle is loaded. Check mode/checkpoint configuration.")

    dataset = runtime["dataset"] or bundle_ref["dataset"]

    outdir = Path(runtime["outdir"])
    fig_dir, metrics_dir = ensure_analysis_dirs(outdir)
    save_config_used({"input_config": cfg, "runtime": runtime, "mode": args.mode}, outdir / "config_used.yaml")

    sample_rows = []
    loader_cache: Dict[str, Any] = {}
    entries: List[Dict[str, Any]] = []
    if bundle_e is not None:
        entries.append({"tag": "electronic", "name": "electronic", "bundle": bundle_e})
    for item in optical_bundles:
        entries.append({"tag": "optical", "name": item["name"], "bundle": item["bundle"]})

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
            test_loader, _ = build_test_loader(
                dataset=dataset,
                data_root=args.data_root,
                out_range=bundle["out_range"],
                image_size=bundle["image_size"],
                batch_size=int(runtime["batch_size"]),
                num_workers=int(runtime["num_workers"]),
                seed=int(runtime["seed"]),
            )
            loader_cache[cache_key] = test_loader
        else:
            test_loader = loader_cache[cache_key]

        samples_x, samples_y, valid_indices = fetch_dataset_indices(test_loader, runtime["fixed_indices"])
        if samples_x.numel() == 0:
            continue
        for i, sample_index in enumerate(valid_indices):
            x_sample = samples_x[i]
            sample_rows.append(
                {
                    "model_tag": entry["tag"],
                    "model_name": entry["name"],
                    "sample_index": int(sample_index),
                    "label": int(samples_y[i].item()),
                }
            )
            if entry["tag"] == "optical":
                _build_optical_figure(
                    bundle=bundle,
                    model_name=entry["name"],
                    x_sample=x_sample,
                    sample_index=sample_index,
                    fig_dir=fig_dir,
                    vis_cfg=runtime["optical_vis"],
                    save_pdf=bool(runtime["save_pdf"]),
                )
            else:
                _build_electronic_figure(
                    bundle=bundle,
                    model_name=entry["name"],
                    x_sample=x_sample,
                    sample_index=sample_index,
                    fig_dir=fig_dir,
                    vis_cfg=runtime["electronic_vis"],
                    save_pdf=bool(runtime["save_pdf"]),
                )

    write_summary_csv(metrics_dir / "visualized_samples.csv", sample_rows)
    save_json({"samples": sample_rows}, metrics_dir / "visualized_samples.json")
    print("Saved visualization figures to: {}".format(fig_dir))


if __name__ == "__main__":
    main()
