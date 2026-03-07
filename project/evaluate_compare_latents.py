from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
from data.datasets import normalize_dataset_name

from utils.io import save_json, write_summary_csv
from utils.latent_compare import (
    build_test_loader,
    decode_from_latent,
    ensure_analysis_dirs,
    evaluate_batch_metrics,
    extract_decoder_latent,
    load_analysis_config,
    load_model_bundle_from_checkpoint,
    resolve_checkpoint_groups,
    save_config_used,
)
from utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task4: Compare normal latent vs scattering latent on test set")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashionmnist", "cifar10"], default=None)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def _resolve_runtime(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    analysis_cfg = cfg.get("analysis", {})
    dataset = args.dataset or cfg.get("dataset", analysis_cfg.get("dataset", None))
    if dataset is not None:
        dataset = normalize_dataset_name(dataset)
    batch_size = int(args.batch_size if args.batch_size is not None else analysis_cfg.get("batch_size", 256))
    num_workers = int(args.num_workers if args.num_workers is not None else analysis_cfg.get("num_workers", 4))
    seed = int(args.seed if args.seed is not None else analysis_cfg.get("seed", 42))
    outdir = args.outdir or cfg.get("outdir", analysis_cfg.get("outdir", None))
    if outdir is None:
        outdir = str(Path("outputs") / "task4_compare_latents")
    return {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "seed": seed,
        "outdir": outdir,
    }


def _evaluate_bundle(
    bundle: Dict[str, Any],
    test_loader,
    data_range: float,
    device: torch.device,
) -> Dict[str, Any]:
    mse_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    count = 0

    per_rows: List[Dict[str, Any]] = []
    idx_global = 0

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Evaluating {}".format(bundle["mode"]), leave=False):
            x = x.to(device, non_blocking=True)
            z, _ = extract_decoder_latent(bundle, x, return_info=False)
            recon = decode_from_latent(bundle, z)
            metrics = evaluate_batch_metrics(recon, x, data_range=data_range)

            mse_ps = metrics["mse"]
            psnr_ps = metrics["psnr"]
            ssim_ps = metrics["ssim"]

            bsz = x.size(0)
            count += bsz
            mse_sum += mse_ps.sum().item()
            psnr_sum += psnr_ps.sum().item()
            ssim_sum += ssim_ps.sum().item()

            y_np = y.detach().cpu().numpy()
            mse_np = mse_ps.detach().cpu().numpy()
            psnr_np = psnr_ps.detach().cpu().numpy()
            ssim_np = ssim_ps.detach().cpu().numpy()
            for i in range(bsz):
                per_rows.append(
                    {
                        "index": idx_global + i,
                        "label": int(y_np[i]),
                        "mode": bundle["mode"],
                        "mse": float(mse_np[i]),
                        "psnr": float(psnr_np[i]),
                        "ssim": float(ssim_np[i]),
                    }
                )
            idx_global += bsz

    return {
        "test_mse": mse_sum / max(count, 1),
        "test_psnr": psnr_sum / max(count, 1),
        "test_ssim": ssim_sum / max(count, 1),
        "samples": count,
        "per_sample_rows": per_rows,
    }


def main() -> None:
    args = parse_args()
    cfg = load_analysis_config(args.config)
    runtime = _resolve_runtime(cfg, args)
    set_seed(int(runtime["seed"]))

    ckpt_grp = resolve_checkpoint_groups(cfg)
    ckpt_elec = ckpt_grp.get("electronic", None)
    optical_entries = ckpt_grp.get("opticals", [])
    if ckpt_elec is None or len(optical_entries) == 0:
        raise ValueError("Need checkpoints.electronic and at least one optical checkpoint (optical/opticals).")

    outdir = Path(runtime["outdir"])
    fig_dir, metrics_dir = ensure_analysis_dirs(outdir)
    _ = fig_dir  # this script only outputs metrics; keep folder structure consistent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bundle_e = load_model_bundle_from_checkpoint(ckpt_elec, device=device, mode="electronic")
    bundles_to_eval: List[Dict[str, Any]] = []
    bundles_to_eval.append({"tag": "electronic", "name": "electronic", "bundle": bundle_e})
    for item in optical_entries:
        b = load_model_bundle_from_checkpoint(item["path"], device=device, mode="optical")
        bundles_to_eval.append({"tag": "optical", "name": item["name"], "bundle": b})

    loader_cache: Dict[str, Any] = {}
    rows = []
    per_sample_dict: Dict[str, List[Dict[str, Any]]] = {}
    dataset = runtime["dataset"] or bundle_e["dataset"]
    if dataset is None:
        raise ValueError("Dataset is missing in args/config/checkpoint.")

    for entry in bundles_to_eval:
        bundle = entry["bundle"]
        bundle_dataset = runtime["dataset"] or bundle["dataset"]
        if bundle_dataset != dataset:
            raise ValueError(
                "Dataset mismatch among checkpoints: expected {}, got {} for {}".format(
                    dataset, bundle_dataset, entry["name"]
                )
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

        res = _evaluate_bundle(
            bundle,
            test_loader=test_loader,
            data_range=float(dataset_info.get("data_range", 1.0)),
            device=device,
        )
        latent_hw = bundle.get("latent_hw", None)
        rows.append(
            {
                "dataset": dataset,
                "tag": entry["tag"],
                "name": entry["name"],
                "mode": bundle["mode"],
                "family": bundle.get("family", "map"),
                "latent_channels": bundle.get("latent_channels", None),
                "latent_h": int(latent_hw[0]) if latent_hw is not None else None,
                "latent_w": int(latent_hw[1]) if latent_hw is not None else None,
                "out_range": bundle.get("out_range"),
                "image_h": int(bundle["image_size"][0]) if bundle.get("image_size", None) is not None else None,
                "image_w": int(bundle["image_size"][1]) if bundle.get("image_size", None) is not None else None,
                "checkpoint": bundle["checkpoint"],
                "test_mse": float(res["test_mse"]),
                "test_psnr": float(res["test_psnr"]),
                "test_ssim": float(res["test_ssim"]),
                "samples": int(res["samples"]),
                "fid": None,
            }
        )
        per_sample_dict[entry["name"]] = res["per_sample_rows"]

    write_summary_csv(metrics_dir / "summary_metrics.csv", rows)
    save_json({"rows": rows}, metrics_dir / "summary_metrics.json")
    for name, ps_rows in per_sample_dict.items():
        safe_name = str(name).replace("/", "_").replace("\\", "_").replace(" ", "_")
        write_summary_csv(metrics_dir / "per_sample_metrics_{}.csv".format(safe_name), ps_rows)

    used_cfg = {
        "input_config": cfg,
        "runtime": {
            **runtime,
            "data_root": args.data_root,
            "device": str(device),
            "resolved_dataset": dataset,
            "num_models": len(bundles_to_eval),
        },
    }
    save_config_used(used_cfg, outdir / "config_used.yaml")
    print("Saved compare metrics to: {}".format(metrics_dir))


if __name__ == "__main__":
    main()
