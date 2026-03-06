from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data.datasets import get_dataloaders
from models import ConvVAE
from utils.eval_tools import (
    bilinear_interpolate_corners,
    interpolation_neighbor_metrics,
    parse_interp_labels,
    save_reconstruction_pairs,
)
from utils.io import flatten_dict, now_timestamp, save_json, write_summary_csv
from utils.logger import create_logger
from utils.metrics import kl_divergence, mse_loss, psnr_from_mse
from utils.seed import set_seed
from utils.viz import save_image_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ConvVAE checkpoint on test set (reconstruction + interpolation)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashionmnist", "cifar10"], default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--num_recon_images", type=int, default=128)
    parser.add_argument("--recon_pairs_per_row", type=int, default=8)
    parser.add_argument("--interp_grid_size", type=int, default=8)
    parser.add_argument("--interp_labels", type=str, default="0,1,2,3")
    parser.add_argument("--n_interp_panels", type=int, default=8)
    parser.add_argument("--class_bank_per_label", type=int, default=32)
    return parser.parse_args()


def select_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sample_corner_images(class_bank: dict, labels: list[int], rng: np.random.Generator) -> list[torch.Tensor]:
    corner_images = []
    for label in labels:
        bank = class_bank.get(int(label), [])
        if len(bank) == 0:
            raise RuntimeError("No samples for label {} in class bank.".format(label))
        idx = int(rng.integers(0, len(bank)))
        corner_images.append(bank[idx])
    return corner_images


def _build_label_panels(
    available_labels: list[int],
    base_labels: list[int],
    n_panels: int,
    rng: np.random.Generator,
) -> list[list[int]]:
    if len(available_labels) < 4:
        return []
    panels: list[list[int]] = []

    valid_base = [int(x) for x in base_labels if int(x) in available_labels]
    if len(valid_base) < 4:
        valid_base = available_labels[:4]
    panels.append(valid_base[:4])

    for _ in range(max(int(n_panels) - 1, 0)):
        picks = rng.choice(available_labels, size=4, replace=False).tolist()
        panels.append([int(x) for x in picks])
    return panels


def _select_class_prototype(model: ConvVAE, candidates: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    if len(candidates) == 0:
        raise RuntimeError("Cannot select prototype from empty candidates.")

    with torch.no_grad():
        batch = torch.cat(candidates, dim=0).to(device)
        z, _ = model.encode(batch)
        center = z.mean(dim=0, keepdim=True)
        dist = (z - center).pow(2).sum(dim=1)
        idx = int(torch.argmin(dist).item())
    return candidates[idx]


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    ckpt_path = Path(args.checkpoint)
    if args.outdir is None:
        args.outdir = str(ckpt_path.parent / f"eval_{now_timestamp()}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    recon_dir = outdir / "reconstructions"
    interp_dir = outdir / "interpolations"
    recon_dir.mkdir(parents=True, exist_ok=True)
    interp_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("eval_vae", outdir=outdir, filename="eval.log")

    device = select_device()
    checkpoint = torch.load(ckpt_path, map_location=device)
    train_args = checkpoint.get("args", {})

    dataset = args.dataset or train_args.get("dataset")
    if dataset is None:
        raise ValueError("Dataset not found. Please provide --dataset explicitly.")

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

    data_range = float(dataset_info["data_range"])
    mse_sum = 0.0
    psnr_sum = 0.0
    kl_sum = 0.0
    count = 0

    mse_all = []
    psnr_all = []
    kl_all = []
    labels_all = []

    recon_inputs = []
    recon_outputs = []
    recon_labels = []
    recon_collected = 0

    class_bank: dict[int, list[torch.Tensor]] = {}

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating", leave=False):
            x = x.to(device, non_blocking=True)
            mu, logvar = model.encode(x)
            recon = model.decode(mu)

            mse_per_sample = mse_loss(recon, x, reduction="none")
            psnr_per_sample = psnr_from_mse(mse_per_sample, data_range=data_range)
            kl_per_sample = kl_divergence(mu, logvar, reduction="none")

            mse_sum += mse_per_sample.sum().item()
            psnr_sum += psnr_per_sample.sum().item()
            kl_sum += kl_per_sample.sum().item()
            count += x.size(0)

            mse_all.append(mse_per_sample.cpu())
            psnr_all.append(psnr_per_sample.cpu())
            kl_all.append(kl_per_sample.cpu())
            labels_all.append(y.cpu())

            if recon_collected < int(args.num_recon_images):
                take = min(int(args.num_recon_images) - recon_collected, x.size(0))
                recon_inputs.append(x[:take].detach().cpu())
                recon_outputs.append(recon[:take].detach().cpu())
                recon_labels.append(y[:take].detach().cpu())
                recon_collected += take

            y_cpu = y.cpu()
            x_cpu = x.detach().cpu()
            for idx in range(y_cpu.size(0)):
                label = int(y_cpu[idx].item())
                bank = class_bank.setdefault(label, [])
                if len(bank) < int(args.class_bank_per_label):
                    bank.append(x_cpu[idx : idx + 1].clone())

    mse_np = torch.cat(mse_all, dim=0).numpy() if mse_all else np.zeros((0,), dtype=np.float32)
    psnr_np = torch.cat(psnr_all, dim=0).numpy() if psnr_all else np.zeros((0,), dtype=np.float32)
    kl_np = torch.cat(kl_all, dim=0).numpy() if kl_all else np.zeros((0,), dtype=np.float32)
    label_np = torch.cat(labels_all, dim=0).numpy() if labels_all else np.zeros((0,), dtype=np.int64)

    recon_saved = 0
    if len(recon_inputs) > 0 and len(recon_outputs) > 0:
        recon_input_cat = torch.cat(recon_inputs, dim=0)
        recon_output_cat = torch.cat(recon_outputs, dim=0)
        recon_label_cat = torch.cat(recon_labels, dim=0) if len(recon_labels) > 0 else None

        if recon_label_cat is not None:
            order = torch.argsort(recon_label_cat)
            recon_input_cat = recon_input_cat[order]
            recon_output_cat = recon_output_cat[order]

        recon_saved = save_reconstruction_pairs(
            inputs=recon_input_cat,
            recons=recon_output_cat,
            path=recon_dir / "reconstruction_pairs_sorted.png",
            max_items=int(args.num_recon_images),
            pairs_per_row=int(args.recon_pairs_per_row),
            out_range=out_range,
        )
        save_reconstruction_pairs(
            inputs=recon_input_cat,
            recons=recon_output_cat,
            path=recon_dir / "reconstruction_pairs_preview.png",
            max_items=min(16, int(args.num_recon_images)),
            pairs_per_row=4,
            out_range=out_range,
        )

    rng = np.random.default_rng(args.seed)
    base_labels = parse_interp_labels(args.interp_labels, num_required=4)
    available_labels = sorted([int(k) for k, v in class_bank.items() if len(v) > 0])
    label_panels = _build_label_panels(available_labels, base_labels, int(args.n_interp_panels), rng)
    prototypes: dict[int, torch.Tensor] = {}
    for label in available_labels:
        prototypes[label] = _select_class_prototype(model, class_bank[label], device=device)

    panel_rows = []
    for panel_idx, labels in enumerate(label_panels):
        corners = [prototypes[int(lb)] for lb in labels]
        tl, tr, bl, br = corners[0], corners[1], corners[2], corners[3]

        with torch.no_grad():
            z_tl, _ = model.encode(tl.to(device))
            z_tr, _ = model.encode(tr.to(device))
            z_bl, _ = model.encode(bl.to(device))
            z_br, _ = model.encode(br.to(device))
            z_grid = bilinear_interpolate_corners(
                z_tl=z_tl,
                z_tr=z_tr,
                z_bl=z_bl,
                z_br=z_br,
                grid_size=int(args.interp_grid_size),
            )
            imgs = model.decode(z_grid).detach().cpu()

        panel_path = interp_dir / "interp_panel_{:03d}.png".format(panel_idx)
        corner_path = interp_dir / "interp_panel_{:03d}_corners.png".format(panel_idx)
        save_image_grid(imgs, path=panel_path, nrow=int(args.interp_grid_size), out_range=out_range)
        save_image_grid(torch.cat([tl, tr, bl, br], dim=0), path=corner_path, nrow=4, out_range=out_range)

        smoothness = interpolation_neighbor_metrics(imgs, grid_size=int(args.interp_grid_size))
        panel_rows.append(
            {
                "panel_index": panel_idx,
                "corner_labels": {
                    "top_left": int(labels[0]),
                    "top_right": int(labels[1]),
                    "bottom_left": int(labels[2]),
                    "bottom_right": int(labels[3]),
                },
                "panel_image": str(panel_path),
                "corner_image": str(corner_path),
                "smoothness": smoothness,
            }
        )

    if len(panel_rows) > 0:
        (interp_dir / "interp_panel_main.png").write_bytes((interp_dir / "interp_panel_000.png").read_bytes())
        (interp_dir / "interp_panel_main_corners.png").write_bytes((interp_dir / "interp_panel_000_corners.png").read_bytes())

    interp_agg = {}
    if len(panel_rows) > 0:
        keys = list(panel_rows[0]["smoothness"].keys())
        for key in keys:
            interp_agg[key] = float(np.mean([row["smoothness"][key] for row in panel_rows]))

    per_sample_rows = []
    for idx in range(int(len(label_np))):
        per_sample_rows.append(
            {
                "index": idx,
                "label": int(label_np[idx]),
                "mse": float(mse_np[idx]),
                "psnr": float(psnr_np[idx]),
                "kl": float(kl_np[idx]),
            }
        )
    write_summary_csv(outdir / "per_sample_metrics.csv", per_sample_rows)

    metrics = {
        "dataset": dataset,
        "checkpoint": str(ckpt_path),
        "mse": mse_sum / max(count, 1),
        "psnr": psnr_sum / max(count, 1),
        "kl_mean": kl_sum / max(count, 1),
        "samples": count,
        "device": str(device),
        "reconstruction": {
            "requested": int(args.num_recon_images),
            "saved_pairs": int(recon_saved),
            "pairs_per_row": int(args.recon_pairs_per_row),
            "artifact_sorted": str(recon_dir / "reconstruction_pairs_sorted.png"),
            "artifact_preview": str(recon_dir / "reconstruction_pairs_preview.png"),
        },
        "interpolation": {
            "grid_size": int(args.interp_grid_size),
            "requested_panels": int(args.n_interp_panels),
            "saved_panels": len(panel_rows),
            "base_corner_labels": [int(x) for x in base_labels],
            "aggregate_smoothness": interp_agg,
            "panels": panel_rows,
        },
        "artifacts": {
            "per_sample_metrics_csv": str(outdir / "per_sample_metrics.csv"),
            "reconstruction_dir": str(recon_dir),
            "interpolation_dir": str(interp_dir),
            "interp_main_panel": str(interp_dir / "interp_panel_main.png") if len(panel_rows) > 0 else None,
            "interp_main_corners": str(interp_dir / "interp_panel_main_corners.png") if len(panel_rows) > 0 else None,
        },
    }

    save_json(metrics, outdir / "results.json")
    write_summary_csv(outdir / "summary.csv", [flatten_dict(metrics)])

    logger.info(
        "Evaluation done | mse=%.6f psnr=%.3f kl=%.4f recon_pairs=%d interp_panels=%d",
        metrics["mse"],
        metrics["psnr"],
        metrics["kl_mean"],
        metrics["reconstruction"]["saved_pairs"],
        metrics["interpolation"]["saved_panels"],
    )
    logger.info("Results saved to %s", outdir)


if __name__ == "__main__":
    main()
