from __future__ import annotations

"""
该实验用于验证散射层建模是否具备合理的 speckle 统计特性与相关性特征，
支撑报告中对相关散射薄相位屏模型的物理解释。
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import numpy as np
import torch
from torchvision import datasets, transforms

from optics.propagation import angular_spectrum_propagate
from optics.scattering import build_scatterer
from optics.sensor import IntensitySensor, detect_intensity
from utils.config import load_config
from utils.io import now_timestamp, save_json, write_summary_csv
from utils.scatter_metrics import (
    autocorr_center_fwhm,
    autocorrelation2d,
    build_complex_field_from_intensity,
    pearson_corr,
    resize_2d_map,
    speckle_contrast,
    to_numpy_image,
    translate_zero_fill,
)
from utils.seed import set_seed

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scatter media physical characterization (no VAE training)")
    parser.add_argument("--config", type=str, default="./configs/scatter_characterization_mnist.yaml")
    parser.add_argument("--dataset", type=str, choices=["mnist", "fashionmnist"], default=None)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--outdir", type=str, default=None)
    return parser.parse_args()


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def _resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = load_config(args.config)
    if args.dataset is not None:
        cfg["dataset"] = args.dataset
    if args.sample_index is not None:
        cfg["sample_index"] = int(args.sample_index)
    if args.outdir is not None:
        cfg["outdir"] = args.outdir
    if cfg.get("outdir", None) is None:
        cfg["outdir"] = str(Path("outputs") / "scatter_characterization_{}_{}".format(cfg.get("dataset", "mnist"), now_timestamp()))
    return cfg


def _load_single_sample(dataset_name: str, data_root: str, resize_hw: Tuple[int, int], sample_index: int) -> Tuple[torch.Tensor, int]:
    tfm = transforms.Compose([transforms.Resize(resize_hw), transforms.ToTensor()])
    name = str(dataset_name).lower()
    if name == "mnist":
        ds = datasets.MNIST(root=data_root, train=False, download=True, transform=tfm)
    elif name == "fashionmnist":
        ds = datasets.FashionMNIST(root=data_root, train=False, download=True, transform=tfm)
    else:
        raise ValueError("Unsupported dataset: {}".format(dataset_name))

    idx = int(max(0, min(int(sample_index), len(ds) - 1)))
    x, y = ds[idx]
    return x.unsqueeze(0), int(y)


def _optical_forward(
    e_in: torch.Tensor,
    scatter_cfg: Dict[str, Any],
    optics_cfg: Dict[str, Any],
    sensor: IntensitySensor | None,
) -> Dict[str, torch.Tensor]:
    e1 = angular_spectrum_propagate(
        E_complex=e_in,
        wavelength_nm=float(optics_cfg["wavelength_nm"]),
        pixel_pitch_um=float(optics_cfg["pixel_pitch_um"]),
        z_mm=float(optics_cfg["z1_mm"]),
        pad_factor=float(optics_cfg["pad_factor"]),
        bandlimit=bool(optics_cfg["bandlimit"]),
        upsample_factor=int(optics_cfg.get("upsample_factor", 1)),
    )

    scatterer = build_scatterer(
        scatter_cfg=scatter_cfg,
        wavelength_nm=float(optics_cfg["wavelength_nm"]),
        pixel_pitch_um=float(optics_cfg["pixel_pitch_um"]),
    ).to(e_in.device)
    e2 = scatterer(e1)

    e3 = angular_spectrum_propagate(
        E_complex=e2,
        wavelength_nm=float(optics_cfg["wavelength_nm"]),
        pixel_pitch_um=float(optics_cfg["pixel_pitch_um"]),
        z_mm=float(optics_cfg["z2_mm"]),
        pad_factor=float(optics_cfg["pad_factor"]),
        bandlimit=bool(optics_cfg["bandlimit"]),
        upsample_factor=int(optics_cfg.get("upsample_factor", 1)),
    )
    i3 = detect_intensity(e3)

    pooled = None
    if sensor is not None:
        pooled = sensor(i3)

    return {
        "I3": i3,
        "pooled": pooled if pooled is not None else i3,
    }


def _build_scatter_cases(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    scatter_cfg = cfg["scatter"]
    static = bool(scatter_cfg.get("static", True))
    phase_mode = str(scatter_cfg.get("phase_mode", "uniform"))
    sigma_phi = float(scatter_cfg.get("sigma_phi", 1.0))
    lc_list = [float(v) for v in scatter_cfg.get("lc_list", [1.0, 3.0, 5.0])]

    cases: List[Dict[str, Any]] = []
    cases.append(
        {
            "name": "iid",
            "title": "iid phase",
            "scatter_cfg": {
                "type": "iid_phase",
                "phase_mode": phase_mode,
                "phase_sigma": sigma_phi,
                "static": static,
            },
        }
    )
    for lc in lc_list:
        cases.append(
            {
                "name": "corr_lc_{:.1f}".format(lc),
                "title": "correlated lc={:.1f}px".format(lc),
                "scatter_cfg": {
                    "type": "correlated_phase",
                    "corr_len_px": lc,
                    "phase_sigma": sigma_phi,
                    "static": static,
                },
            }
        )
    return cases


def _make_sensor(cfg: Dict[str, Any]) -> IntensitySensor | None:
    pool_cfg = cfg.get("pooling", {})
    if not bool(pool_cfg.get("use_pooling", True)):
        return None
    return IntensitySensor(
        pool_type=str(pool_cfg.get("pool_type", "avg")),
        pool_kernel=pool_cfg.get("pool_kernel", 8),
        pool_stride=pool_cfg.get("pool_stride", 8),
        pool_reduce=str(pool_cfg.get("pool_reduce", "mean")),
        expected_hw=None,
    )


def _plot_fixed_input_grid(
    out_path: Path,
    input_img: torch.Tensor,
    i0: torch.Tensor,
    rows: List[Dict[str, Any]],
) -> None:
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 5, figsize=(15, 3.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ["input x", "I0=|E_in|^2", "I3 detector", "pooled_intensity", "autocorrelation"]
    for c in range(5):
        axes[0, c].set_title(col_titles[c], fontsize=10)

    input_np = to_numpy_image(input_img)
    i0_np = to_numpy_image(i0)

    for r, row in enumerate(rows):
        i3_np = to_numpy_image(row["I3"])
        pool_np = to_numpy_image(row["pooled"])
        ac_np = to_numpy_image(row["autocorr"])

        axes[r, 0].imshow(input_np, cmap="gray")
        axes[r, 1].imshow(i0_np, cmap="gray")
        axes[r, 2].imshow(np.log1p(np.clip(i3_np, 0, None)), cmap="magma")
        axes[r, 3].imshow(np.log1p(np.clip(pool_np, 0, None)), cmap="viridis")
        axes[r, 4].imshow(ac_np, cmap="coolwarm", vmin=-0.2, vmax=1.0)

        for c in range(5):
            axes[r, c].axis("off")
        axes[r, 0].set_ylabel(row["title"], fontsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_autocorr_maps(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    n = len(rows)
    n_cols = min(4, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.8 * n_cols, 3.8 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for idx, row in enumerate(rows):
        rr = idx // n_cols
        cc = idx % n_cols
        ax = axes[rr, cc]
        ax.imshow(to_numpy_image(row["autocorr"]), cmap="coolwarm", vmin=-0.2, vmax=1.0)
        ax.set_title(row["title"], fontsize=10)
        ax.axis("off")

    for idx in range(n, n_rows * n_cols):
        rr = idx // n_cols
        cc = idx % n_cols
        axes[rr, cc].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_memory_curves(out_path: Path, dx_list: List[int], rows: List[Dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    for row in rows:
        axes[0].plot(dx_list, row["raw_corr"], marker="o", linewidth=1.5, label=row["title"])
        axes[1].plot(dx_list, row["comp_corr"], marker="o", linewidth=1.5, label=row["title"])

    axes[0].set_title("Raw correlation vs dx")
    axes[1].set_title("Compensated correlation vs dx")
    for ax in axes:
        ax.set_xlabel("Input shift dx (px)")
        ax.set_ylabel("Correlation with dx=0 output")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def run_experiment_fixed_input(
    x: torch.Tensor,
    cfg: Dict[str, Any],
    outdir: Path,
) -> List[Dict[str, Any]]:
    sensor = _make_sensor(cfg)
    cases = _build_scatter_cases(cfg)
    eps = float(cfg["data"].get("eps", 1e-8))
    optics_cfg = cfg["optics"]

    i_in = x.clamp(0.0, 1.0)
    e_in = build_complex_field_from_intensity(i_in, eps=eps)
    i0 = detect_intensity(e_in)

    grid_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for case in cases:
        out = _optical_forward(e_in=e_in, scatter_cfg=case["scatter_cfg"], optics_cfg=optics_cfg, sensor=sensor)
        i3 = out["I3"][0, 0]
        pooled = out["pooled"][0, 0]
        autocorr = autocorrelation2d(i3)

        i3_down = resize_2d_map(i3, hw=tuple(i_in.shape[-2:]))
        pool_down = pooled
        input_down_pool = resize_2d_map(i_in[0, 0], hw=tuple(pool_down.shape[-2:]))

        row = {
            "name": case["name"],
            "title": case["title"],
            "I3": i3,
            "pooled": pooled,
            "autocorr": autocorr,
            "speckle_contrast": speckle_contrast(i3),
            "autocorr_width_px": autocorr_center_fwhm(autocorr),
            "corr_i3_vs_input": pearson_corr(i3_down, i_in[0, 0]),
            "corr_pooled_vs_input": pearson_corr(pool_down, input_down_pool),
        }
        grid_rows.append(row)
        csv_rows.append(
            {
                "scatter_name": case["name"],
                "scatter_title": case["title"],
                "speckle_contrast": row["speckle_contrast"],
                "autocorr_width_px": row["autocorr_width_px"],
                "corr_i3_vs_input": row["corr_i3_vs_input"],
                "corr_pooled_vs_input": row["corr_pooled_vs_input"],
            }
        )

    _plot_fixed_input_grid(
        out_path=outdir / "comparison_fixed_input.png",
        input_img=i_in[0, 0],
        i0=i0[0, 0],
        rows=grid_rows,
    )
    _plot_autocorr_maps(out_path=outdir / "autocorrelation_fixed_input.png", rows=grid_rows)
    write_summary_csv(outdir / "metrics_fixed_input.csv", csv_rows)
    return grid_rows


def run_experiment_memory_effect(
    x: torch.Tensor,
    cfg: Dict[str, Any],
    outdir: Path,
) -> List[Dict[str, Any]]:
    sensor = _make_sensor(cfg)
    cases = _build_scatter_cases(cfg)
    eps = float(cfg["data"].get("eps", 1e-8))
    optics_cfg = cfg["optics"]
    mem_cfg = cfg.get("memory_effect", {})
    dx_list = [int(v) for v in mem_cfg.get("dx_list", [-6, -4, -2, 0, 2, 4, 6])]
    dy = int(mem_cfg.get("dy", 0))

    i_in = x.clamp(0.0, 1.0)
    results_rows: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []

    for case in cases:
        raw_corr_list: List[float] = []
        comp_corr_list: List[float] = []

        # Always compute dx=0 reference first, so every curve has the same x-length.
        x_ref = translate_zero_fill(i_in[0, 0], dx=0, dy=dy).unsqueeze(0).unsqueeze(0)
        e_ref = build_complex_field_from_intensity(x_ref, eps=eps)
        ref_out = _optical_forward(e_in=e_ref, scatter_cfg=case["scatter_cfg"], optics_cfg=optics_cfg, sensor=sensor)
        i_ref = ref_out["I3"][0, 0]

        for dx in dx_list:
            x_shift = translate_zero_fill(i_in[0, 0], dx=dx, dy=dy).unsqueeze(0).unsqueeze(0)
            e_in = build_complex_field_from_intensity(x_shift, eps=eps)
            out = _optical_forward(e_in=e_in, scatter_cfg=case["scatter_cfg"], optics_cfg=optics_cfg, sensor=sensor)
            i3 = out["I3"][0, 0]

            raw = pearson_corr(i3, i_ref)
            comp = pearson_corr(translate_zero_fill(i3, dx=-dx, dy=-dy), i_ref)
            raw_corr_list.append(raw)
            comp_corr_list.append(comp)

            csv_rows.append(
                {
                    "scatter_name": case["name"],
                    "scatter_title": case["title"],
                    "dx": int(dx),
                    "dy": int(dy),
                    "raw_corr": float(raw),
                    "compensated_corr": float(comp),
                    "raw_over_comp": float(raw / comp) if abs(comp) > 1e-12 else 0.0,
                }
            )

        results_rows.append(
            {
                "name": case["name"],
                "title": case["title"],
                "raw_corr": raw_corr_list,
                "comp_corr": comp_corr_list,
            }
        )

    _plot_memory_curves(out_path=outdir / "memory_effect_curves.png", dx_list=dx_list, rows=results_rows)
    write_summary_csv(outdir / "metrics_memory_effect.csv", csv_rows)
    return results_rows


def main() -> None:
    args = parse_args()
    cfg = _resolve_config(args)
    set_seed(int(cfg.get("seed", 42)))

    outdir = Path(cfg["outdir"])
    outdir.mkdir(parents=True, exist_ok=True)

    dataset = str(cfg.get("dataset", "mnist"))
    data_root = str(cfg.get("data_root", "./data"))
    resize_hw = tuple(int(v) for v in cfg.get("data", {}).get("resize_hw", [64, 64]))
    sample_index = int(cfg.get("sample_index", 0))

    x, label = _load_single_sample(
        dataset_name=dataset,
        data_root=data_root,
        resize_hw=resize_hw,
        sample_index=sample_index,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = x.to(device)

    fixed_rows = run_experiment_fixed_input(x=x, cfg=cfg, outdir=outdir)
    memory_rows = run_experiment_memory_effect(x=x, cfg=cfg, outdir=outdir)

    cfg_used = _deep_update(
        cfg,
        {
            "runtime": {
                "device": str(device),
                "resolved_dataset": dataset,
                "resolved_sample_index": sample_index,
                "resolved_label": int(label),
            }
        },
    )
    with (outdir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_used, f, sort_keys=False, allow_unicode=True)

    summary = {
        "dataset": dataset,
        "sample_index": sample_index,
        "sample_label": int(label),
        "outdir": str(outdir),
        "num_scatter_cases": len(fixed_rows),
        "artifacts": {
            "comparison_fixed_input": str(outdir / "comparison_fixed_input.png"),
            "autocorrelation_fixed_input": str(outdir / "autocorrelation_fixed_input.png"),
            "memory_effect_curves": str(outdir / "memory_effect_curves.png"),
            "metrics_fixed_input_csv": str(outdir / "metrics_fixed_input.csv"),
            "metrics_memory_effect_csv": str(outdir / "metrics_memory_effect.csv"),
            "config_used_yaml": str(outdir / "config_used.yaml"),
        },
    }
    save_json(summary, outdir / "summary.json")
    print("Saved scatter characterization artifacts to: {}".format(outdir))


if __name__ == "__main__":
    main()
