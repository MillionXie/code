from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Dict, List, Sequence, Tuple

import matplotlib
import torch

# Make local package imports work for:
# 1) python optics/test_scattering_memory.py
# 2) python -m optics.test_scattering_memory
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from optics.propagation import angular_spectrum_propagate
from optics.scattering import build_scatterer
from optics.sensor import detect_intensity
from utils.io import now_timestamp, save_json, write_summary_csv
from utils.seed import set_seed
from utils.viz import save_image_grid

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_csv_floats(text: str) -> List[float]:
    out = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    if len(out) == 0:
        raise ValueError("Expect non-empty float list, got '{}'".format(text))
    return out


def _normalize_batch_for_grid(x: torch.Tensor) -> torch.Tensor:
    x = torch.log1p(torch.clamp(x, min=0.0))
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return torch.clamp((x - x_min) / (x_max - x_min + 1e-8), 0.0, 1.0)


def _build_tilted_input_field(
    field_hw: Tuple[int, int],
    tilts_cpp: torch.Tensor,
    aperture_sigma_px: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    h, w = int(field_hw[0]), int(field_hw[1])
    yy = torch.arange(h, device=device, dtype=dtype) - (h - 1) * 0.5
    xx = torch.arange(w, device=device, dtype=dtype) - (w - 1) * 0.5
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")

    sigma = max(float(aperture_sigma_px), 1.0)
    amp = torch.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    amp = amp / torch.clamp(amp.max(), min=1e-8)
    amp = amp.view(1, 1, h, w)

    phase = 2.0 * math.pi * tilts_cpp.view(-1, 1, 1, 1) * xx.view(1, 1, h, w)
    real = amp * torch.cos(phase)
    imag = amp * torch.sin(phase)
    return torch.complex(real, imag)


def _peak_normalized_xcorr(ref: torch.Tensor, cur: torch.Tensor) -> Tuple[float, int, int]:
    ref = ref - ref.mean()
    cur = cur - cur.mean()
    denom = torch.sqrt(torch.clamp((ref * ref).sum(), min=1e-12) * torch.clamp((cur * cur).sum(), min=1e-12))
    corr = torch.fft.ifft2(torch.fft.fft2(ref) * torch.conj(torch.fft.fft2(cur))).real
    corr = corr / torch.clamp(denom, min=1e-12)

    h, w = int(corr.shape[-2]), int(corr.shape[-1])
    flat_idx = int(torch.argmax(corr).item())
    iy = flat_idx // w
    ix = flat_idx % w
    if iy > h // 2:
        iy -= h
    if ix > w // 2:
        ix -= w
    peak = float(corr.view(-1)[flat_idx].item())
    return peak, int(ix), int(iy)


def _run_optical_chain(
    e_in: torch.Tensor,
    scatter_cfg: Dict[str, object],
    wavelength_nm: float,
    pixel_pitch_um: float,
    distance_a_mm: float,
    distance_b_mm: float,
    pad_factor: float,
    bandlimit: bool,
    upsample_factor: int,
) -> torch.Tensor:
    scatterer = build_scatterer(scatter_cfg, wavelength_nm=wavelength_nm, pixel_pitch_um=pixel_pitch_um).to(e_in.device)

    e_a = angular_spectrum_propagate(
        E_complex=e_in,
        wavelength_nm=wavelength_nm,
        pixel_pitch_um=pixel_pitch_um,
        z_mm=distance_a_mm,
        pad_factor=pad_factor,
        bandlimit=bandlimit,
        upsample_factor=upsample_factor,
    )
    e_s = scatterer(e_a)
    e_b = angular_spectrum_propagate(
        E_complex=e_s,
        wavelength_nm=wavelength_nm,
        pixel_pitch_um=pixel_pitch_um,
        z_mm=distance_b_mm,
        pad_factor=pad_factor,
        bandlimit=bandlimit,
        upsample_factor=upsample_factor,
    )
    return detect_intensity(e_b)


def _build_cases(
    phase_sigma: float,
    phase_mode: str,
    corr_lens_px: Sequence[float],
    na_values: Sequence[float],
    static: bool,
) -> List[Dict[str, object]]:
    cases: List[Dict[str, object]] = []
    cases.append(
        {
            "name": "iid_phase",
            "cfg": {
                "type": "iid_phase",
                "phase_mode": str(phase_mode),
                "phase_sigma": float(phase_sigma),
                "static": bool(static),
            },
        }
    )
    for corr in corr_lens_px:
        cases.append(
            {
                "name": "correlated_phase_corr_{:.2f}px".format(float(corr)),
                "cfg": {
                    "type": "correlated_phase",
                    "corr_len_px": float(corr),
                    "phase_sigma": float(phase_sigma),
                    "static": bool(static),
                },
            }
        )
    for na in na_values:
        cases.append(
            {
                "name": "angle_limited_na_{:.3f}".format(float(na)),
                "cfg": {
                    "type": "angle_limited",
                    "na": float(na),
                    "static": bool(static),
                },
            }
        )
    return cases


def _plot_curves(x: List[float], rows: List[Dict[str, object]], out_path: Path, key: str, ylabel: str) -> None:
    plt.figure(figsize=(8.0, 5.0))
    for row in rows:
        y = row[key]
        plt.plot(x, y, marker="o", linewidth=1.2, markersize=3, label=row["name"])
    plt.xlabel("Input tilt (cycles/pixel)")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scattering memory-effect probe: propagate(a)->scatter->propagate(b)")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--field_hw", type=int, nargs=2, default=[200, 200], metavar=("H", "W"))
    parser.add_argument("--wavelength_nm", type=float, default=532.0)
    parser.add_argument("--pixel_pitch_um", type=float, default=8.0)
    parser.add_argument("--distance_a_mm", type=float, default=20.0)
    parser.add_argument("--distance_b_mm", type=float, default=5.0)
    parser.add_argument("--pad_factor", type=float, default=2.0)
    parser.add_argument("--upsample_factor", type=int, default=1)
    parser.add_argument("--bandlimit", action="store_true", default=True)
    parser.add_argument("--no_bandlimit", action="store_false", dest="bandlimit")
    parser.add_argument("--n_tilts", type=int, default=11)
    parser.add_argument("--tilt_max_cpp", type=float, default=0.03)
    parser.add_argument("--aperture_sigma_px", type=float, default=45.0)
    parser.add_argument("--phase_sigma", type=float, default=1.0)
    parser.add_argument("--phase_mode", type=str, choices=["uniform", "normal"], default="uniform")
    parser.add_argument("--corr_lens_px", type=str, default="1.0,3.0,6.0")
    parser.add_argument("--na_values", type=str, default="0.08,0.15,0.25")
    parser.add_argument("--dynamic_scatter", action="store_true")
    parser.add_argument("--memory_threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))

    if args.outdir is None:
        args.outdir = str(Path("outputs") / "optics_memory_{}".format(now_timestamp()))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    corr_lens_px = _parse_csv_floats(args.corr_lens_px)
    na_values = _parse_csv_floats(args.na_values)
    static = not bool(args.dynamic_scatter)

    tilts = torch.linspace(-float(args.tilt_max_cpp), float(args.tilt_max_cpp), int(args.n_tilts), device=device, dtype=dtype)
    zero_idx = int(args.n_tilts // 2)

    e_in = _build_tilted_input_field(
        field_hw=(int(args.field_hw[0]), int(args.field_hw[1])),
        tilts_cpp=tilts,
        aperture_sigma_px=float(args.aperture_sigma_px),
        device=device,
        dtype=dtype,
    )

    cases = _build_cases(
        phase_sigma=float(args.phase_sigma),
        phase_mode=str(args.phase_mode),
        corr_lens_px=corr_lens_px,
        na_values=na_values,
        static=static,
    )

    all_rows: List[Dict[str, object]] = []
    csv_rows: List[Dict[str, object]] = []
    for case in cases:
        name = str(case["name"])
        scatter_cfg = case["cfg"]

        with torch.no_grad():
            intensity = _run_optical_chain(
                e_in=e_in,
                scatter_cfg=scatter_cfg,
                wavelength_nm=float(args.wavelength_nm),
                pixel_pitch_um=float(args.pixel_pitch_um),
                distance_a_mm=float(args.distance_a_mm),
                distance_b_mm=float(args.distance_b_mm),
                pad_factor=float(args.pad_factor),
                bandlimit=bool(args.bandlimit),
                upsample_factor=int(args.upsample_factor),
            )

        ref = intensity[zero_idx, 0]
        peak_corrs: List[float] = []
        peak_shift_x: List[int] = []
        peak_shift_y: List[int] = []
        for idx in range(intensity.shape[0]):
            peak, sx, sy = _peak_normalized_xcorr(ref, intensity[idx, 0])
            peak_corrs.append(peak)
            peak_shift_x.append(sx)
            peak_shift_y.append(sy)

        valid = [abs(float(tilts[i].item())) for i, c in enumerate(peak_corrs) if c >= float(args.memory_threshold)]
        memory_width = max(valid) if len(valid) > 0 else 0.0

        vis = _normalize_batch_for_grid(intensity[:, :1].detach().cpu())
        save_image_grid(
            vis,
            path=outdir / "{}_intensity_grid.png".format(name),
            nrow=int(args.n_tilts),
            out_range="zero_one",
        )

        row = {
            "name": name,
            "scatter_cfg": scatter_cfg,
            "peak_corr": peak_corrs,
            "peak_shift_x_px": peak_shift_x,
            "peak_shift_y_px": peak_shift_y,
            "memory_threshold": float(args.memory_threshold),
            "memory_width_cpp": float(memory_width),
            "corr_at_zero_tilt": float(peak_corrs[zero_idx]),
        }
        all_rows.append(row)
        csv_rows.append(
            {
                "name": name,
                "scatter_type": str(scatter_cfg.get("type", "")),
                "static": bool(scatter_cfg.get("static", True)),
                "corr_len_px": float(scatter_cfg.get("corr_len_px", -1.0)) if "corr_len_px" in scatter_cfg else "",
                "na": float(scatter_cfg.get("na", -1.0)) if "na" in scatter_cfg else "",
                "memory_threshold": float(args.memory_threshold),
                "memory_width_cpp": float(memory_width),
                "corr_at_zero_tilt": float(peak_corrs[zero_idx]),
            }
        )

    tilt_list = [float(v.item()) for v in tilts.detach().cpu()]
    _plot_curves(
        x=tilt_list,
        rows=all_rows,
        out_path=outdir / "memory_peak_corr_vs_tilt.png",
        key="peak_corr",
        ylabel="Peak normalized cross-correlation",
    )
    _plot_curves(
        x=tilt_list,
        rows=all_rows,
        out_path=outdir / "memory_peak_shift_x_vs_tilt.png",
        key="peak_shift_x_px",
        ylabel="Peak shift X (px)",
    )

    result = {
        "meta": {
            "device": str(device),
            "field_hw": [int(args.field_hw[0]), int(args.field_hw[1])],
            "wavelength_nm": float(args.wavelength_nm),
            "pixel_pitch_um": float(args.pixel_pitch_um),
            "distance_a_mm": float(args.distance_a_mm),
            "distance_b_mm": float(args.distance_b_mm),
            "pad_factor": float(args.pad_factor),
            "bandlimit": bool(args.bandlimit),
            "upsample_factor": int(args.upsample_factor),
            "n_tilts": int(args.n_tilts),
            "tilt_max_cpp": float(args.tilt_max_cpp),
            "tilt_values_cpp": tilt_list,
            "aperture_sigma_px": float(args.aperture_sigma_px),
            "memory_threshold": float(args.memory_threshold),
        },
        "cases": all_rows,
    }
    save_json(result, outdir / "summary.json")
    write_summary_csv(outdir / "summary.csv", csv_rows)

    print("Saved memory-effect analysis to: {}".format(outdir))


if __name__ == "__main__":
    main()
