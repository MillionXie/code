from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from utils.io import now_timestamp, save_json, write_summary_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch evaluate all experiment runs under outputs/")
    parser.add_argument("--outputs_root", type=str, default="./outputs")
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compute_interp_fid", dest="compute_interp_fid", action="store_true")
    parser.add_argument("--no_interp_fid", dest="compute_interp_fid", action="store_false")
    parser.add_argument("--fid_max_images", type=int, default=512)
    parser.add_argument("--fid_batch_size", type=int, default=64)
    parser.add_argument("--re_eval", action="store_true")
    parser.set_defaults(compute_interp_fid=True)
    return parser.parse_args()


def _r3(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    try:
        return float("{:.3f}".format(float(x)))
    except Exception:
        return None


def _parse_run_name(name: str) -> Dict[str, Any]:
    n = name.lower()
    dataset = None
    for ds in ("fashionmnist", "mnist", "cifar10"):
        if ds in n:
            dataset = ds
            break

    scatter_type = None
    if "iid" in n:
        scatter_type = "iid"
    elif "corr" in n:
        scatter_type = "corr"

    corr_len_px = None
    m_corr = re.search(r"corr[_-]?(\d+(?:\.\d+)?)", n)
    if m_corr is not None:
        corr_len_px = float(m_corr.group(1))

    l_mm = None
    m_l = re.search(r"(?:^|_)l[_-]?(\d+(?:\.\d+)?)", n)
    if m_l is not None:
        l_mm = float(m_l.group(1))

    no_pool = "nopool" in n
    return {
        "dataset_in_name": dataset,
        "scatter_type_in_name": scatter_type,
        "corr_len_px_in_name": corr_len_px,
        "l_mm_in_name": l_mm,
        "no_pool_in_name": no_pool,
    }


def _discover_runs(outputs_root: Path) -> List[Path]:
    runs = []
    for ckpt in outputs_root.rglob("checkpoints/best.pt"):
        run_dir = ckpt.parent.parent
        runs.append(run_dir)
    runs = sorted(set(runs))
    return runs


def _inspect_run(run_dir: Path) -> Dict[str, Any]:
    ckpt = run_dir / "checkpoints" / "best.pt"
    if not ckpt.exists():
        return {"valid": False, "reason": "missing_checkpoint"}

    try:
        obj = torch.load(ckpt, map_location="cpu")
    except Exception as ex:
        return {"valid": False, "reason": "checkpoint_load_failed: {}".format(ex)}

    run_name = run_dir.name
    meta_name = _parse_run_name(run_name)
    dataset = meta_name["dataset_in_name"]
    mode = None
    run_type = "unknown"
    latent_family = None
    cfg_dataset = None
    cfg_seed = None
    cfg_latent_hw = None
    cfg_scatter_type = None
    cfg_corr_len_px = None
    cfg_z_to_sensor_mm = None
    cfg_pool_type = None
    cfg_pool_kernel = None
    cfg_pool_stride = None

    if isinstance(obj, dict) and "args" in obj:
        # vector VAE baseline
        run_type = "vae_vector"
        latent_family = "ordinary"
        cfg_args = obj.get("args", {})
        dataset = dataset or cfg_args.get("dataset", None)
        cfg_dataset = cfg_args.get("dataset", None)
        cfg_seed = cfg_args.get("seed", None)
    elif isinstance(obj, dict) and "config" in obj and isinstance(obj.get("config"), dict):
        cfg = obj["config"]
        dataset = dataset or cfg.get("dataset", None)
        cfg_dataset = cfg.get("dataset", None)
        cfg_seed = cfg.get("train", {}).get("seed", None)
        cfg_latent_hw = cfg.get("model", {}).get("latent_hw", None)
        optics_cfg = cfg.get("optics", {})
        scatter_cfg = optics_cfg.get("scatter", {})
        sensor_cfg = optics_cfg.get("sensor", {})
        cfg_scatter_type = scatter_cfg.get("type", None)
        cfg_corr_len_px = scatter_cfg.get("corr_len_px", None)
        cfg_z_to_sensor_mm = optics_cfg.get("z_to_sensor_mm", None)
        cfg_pool_type = sensor_cfg.get("pool_type", None)
        cfg_pool_kernel = sensor_cfg.get("pool_kernel", None)
        cfg_pool_stride = sensor_cfg.get("pool_stride", None)
        if cfg.get("optics", None) is not None:
            run_type = "map_optical"
            mode = "optical"
            latent_family = "scattering"
        else:
            run_type = "map_electronic"
            mode = "electronic"
            latent_family = "ordinary"
    else:
        return {"valid": False, "reason": "unrecognized_checkpoint_format"}

    return {
        "valid": True,
        "run_dir": run_dir,
        "run_name": run_name,
        "checkpoint": ckpt,
        "dataset": dataset,
        "dataset_cfg": cfg_dataset,
        "seed_cfg": cfg_seed,
        "run_type": run_type,
        "mode": mode,
        "latent_family": latent_family,
        "latent_hw_cfg": cfg_latent_hw,
        "scatter_type_cfg": cfg_scatter_type,
        "corr_len_px_cfg": cfg_corr_len_px,
        "z_to_sensor_mm_cfg": cfg_z_to_sensor_mm,
        "pool_type_cfg": cfg_pool_type,
        "pool_kernel_cfg": cfg_pool_kernel,
        "pool_stride_cfg": cfg_pool_stride,
        **meta_name,
    }


def _eval_command(
    script_root: Path,
    run_meta: Dict[str, Any],
    eval_outdir: Path,
    args: argparse.Namespace,
) -> List[str]:
    py = sys.executable
    dataset = run_meta.get("dataset", None)
    ckpt = str(run_meta["checkpoint"])
    common = [
        "--checkpoint",
        ckpt,
        "--data_root",
        str(args.data_root),
        "--batch_size",
        str(int(args.batch_size)),
        "--num_workers",
        str(int(args.num_workers)),
        "--seed",
        str(int(args.seed)),
        "--outdir",
        str(eval_outdir),
    ]
    if dataset is not None:
        common.extend(["--dataset", str(dataset)])
    if args.compute_interp_fid:
        common.extend(
            [
                "--compute_interp_fid",
                "--fid_max_images",
                str(int(args.fid_max_images)),
                "--fid_batch_size",
                str(int(args.fid_batch_size)),
            ]
        )

    if run_meta["run_type"] == "vae_vector":
        return [py, str(script_root / "eval_vae.py")] + common
    if run_meta["run_type"] in ("map_electronic", "map_optical"):
        mode = "electronic" if run_meta["run_type"] == "map_electronic" else "optical"
        return [py, str(script_root / "eval_map.py"), "--mode", mode] + common
    raise ValueError("Unsupported run_type: {}".format(run_meta["run_type"]))


def _load_eval_results(eval_outdir: Path) -> Dict[str, Any]:
    path = eval_outdir / "results.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _has_valid_interp_fid(results: Dict[str, Any]) -> bool:
    inter = results.get("interpolation", {})
    metrics = inter.get("metrics", {})
    fid = metrics.get("fid", {})
    if not isinstance(fid, dict):
        return False
    status = str(fid.get("status", ""))
    val = fid.get("value", None)
    if val is None:
        return False
    if status.startswith("inception_unavailable"):
        return False
    return True


def _extract_metric(results: Dict[str, Any], key: str, default: Optional[float] = None) -> Optional[float]:
    # Reconstruction metrics
    rec = results.get("reconstruction", {})
    rec_m = rec.get("metrics", {})
    if key in rec_m:
        return rec_m.get(key, default)
    if key in results:
        return results.get(key, default)
    return default


def _extract_interp_fid(results: Dict[str, Any]) -> Tuple[Optional[float], str]:
    inter = results.get("interpolation", {})
    metrics = inter.get("metrics", {})
    fid = metrics.get("fid", {})
    if isinstance(fid, dict):
        return fid.get("value", None), str(fid.get("status", "unknown"))
    return None, "missing"


def _extract_interp_neighbor_metric(results: Dict[str, Any]) -> Optional[float]:
    inter = results.get("interpolation", {})
    metrics = inter.get("metrics", {})
    smooth = metrics.get("aggregate_smoothness", inter.get("aggregate_smoothness", {}))
    if not isinstance(smooth, dict):
        return None
    if "neighbor_mse_all_mean" in smooth:
        return smooth.get("neighbor_mse_all_mean")
    for k in ("neighbor_mse_h_mean", "neighbor_mse_v_mean"):
        if k in smooth:
            return smooth[k]
    return None


def _sha1_short(path: Path, max_bytes: int = 2_000_000) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha1()
    try:
        with path.open("rb") as f:
            remain = int(max_bytes)
            while remain > 0:
                chunk = f.read(min(1 << 20, remain))
                if not chunk:
                    break
                h.update(chunk)
                remain -= len(chunk)
        return h.hexdigest()[:12]
    except Exception:
        return None


def _group_stats(values: List[float]) -> Dict[str, Optional[float]]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    if n == 1:
        return {"n": 1, "mean": values[0], "std": 0.0, "ci95": 0.0}
    mean = sum(values) / float(n)
    var = sum((v - mean) ** 2 for v in values) / float(n - 1)
    std = var ** 0.5
    ci95 = 1.96 * std / (n ** 0.5)
    return {"n": n, "mean": mean, "std": std, "ci95": ci95}


def _aggregate_ci(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    metrics = ["recon_mse", "recon_psnr", "recon_ssim", "recon_kl", "interp_neighbor_mse", "interp_fid"]
    groups = defaultdict(list)
    for row in rows:
        key = (
            row.get("dataset"),
            row.get("latent_family"),
            row.get("run_type"),
            row.get("scatter_type"),
            row.get("corr_len_px"),
            row.get("l_mm"),
            row.get("no_pool"),
        )
        groups[key].append(row)

    out_rows = []
    for key, g_rows in groups.items():
        row = {
            "dataset": key[0],
            "latent_family": key[1],
            "run_type": key[2],
            "scatter_type": key[3],
            "corr_len_px": key[4],
            "l_mm": key[5],
            "no_pool": key[6],
            "n_runs": len(g_rows),
        }
        for m in metrics:
            vals = [float(r[m]) for r in g_rows if r.get(m) is not None]
            st = _group_stats(vals)
            row["{}_mean".format(m)] = _r3(st["mean"])
            row["{}_std".format(m)] = _r3(st["std"])
            row["{}_ci95".format(m)] = _r3(st["ci95"])
            row["{}_n".format(m)] = int(st["n"])
        out_rows.append(row)
    out_rows.sort(key=lambda x: (str(x.get("dataset")), str(x.get("latent_family")), str(x.get("run_type"))))
    return out_rows


def _build_latent_vs_scatter(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Compare scattering runs against ordinary baseline mean within each dataset.
    metrics = ["recon_mse", "recon_psnr", "recon_ssim", "interp_neighbor_mse", "interp_fid"]
    ordinary_by_dataset: Dict[str, Dict[str, float]] = {}
    for ds in sorted({r.get("dataset") for r in rows if r.get("dataset") is not None}):
        baseline_rows = [r for r in rows if r.get("dataset") == ds and r.get("latent_family") == "ordinary"]
        if len(baseline_rows) == 0:
            continue
        avg = {}
        for m in metrics:
            vals = [float(r[m]) for r in baseline_rows if r.get(m) is not None]
            avg[m] = sum(vals) / float(len(vals)) if len(vals) > 0 else None
        ordinary_by_dataset[ds] = avg

    out = []
    for r in rows:
        if r.get("latent_family") != "scattering":
            continue
        ds = r.get("dataset")
        if ds not in ordinary_by_dataset:
            continue
        base = ordinary_by_dataset[ds]
        row = {
            "dataset": ds,
            "run_name": r.get("run_name"),
            "run_type": r.get("run_type"),
            "scatter_type": r.get("scatter_type"),
            "corr_len_px": r.get("corr_len_px"),
            "l_mm": r.get("l_mm"),
            "no_pool": r.get("no_pool"),
        }
        for m in metrics:
            row["ordinary_{}".format(m)] = _r3(base.get(m))
            row["scattering_{}".format(m)] = _r3(r.get(m))
            if base.get(m) is None or r.get(m) is None:
                row["delta_{}".format(m)] = None
            else:
                row["delta_{}".format(m)] = _r3(float(r.get(m)) - float(base[m]))
        out.append(row)
    out.sort(key=lambda x: (str(x.get("dataset")), str(x.get("run_name"))))
    return out


def _format_row_numbers_3dp(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            if isinstance(v, bool) or v is None or isinstance(v, int):
                d[k] = v
            elif isinstance(v, float):
                d[k] = _r3(v)
            else:
                d[k] = v
        out.append(d)
    return out


def main() -> None:
    args = parse_args()
    outputs_root = Path(args.outputs_root).resolve()
    if args.outdir is None:
        outdir = outputs_root / ("batch_eval_" + now_timestamp())
    else:
        outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    eval_runs_dir = outdir / "per_run_evals"
    eval_runs_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = _discover_runs(outputs_root)
    script_root = Path(__file__).resolve().parent

    run_rows_raw: List[Dict[str, Any]] = []
    skipped_rows: List[Dict[str, Any]] = []

    for run_dir in run_dirs:
        inspect = _inspect_run(run_dir)
        if not inspect.get("valid", False):
            skipped_rows.append({"run_dir": str(run_dir), "reason": inspect.get("reason", "invalid")})
            continue

        run_name = str(inspect["run_name"])
        try:
            run_rel = run_dir.relative_to(outputs_root)
            run_key = str(run_rel).replace("/", "__")
        except Exception:
            run_key = run_name
        eval_outdir = eval_runs_dir / run_key
        results_path = eval_outdir / "results.json"

        need_eval = bool(args.re_eval) or (not results_path.exists())
        if (not need_eval) and bool(args.compute_interp_fid):
            try:
                old_results = _load_eval_results(eval_outdir)
                if not _has_valid_interp_fid(old_results):
                    need_eval = True
            except Exception:
                need_eval = True

        if need_eval:
            cmd = _eval_command(script_root=script_root, run_meta=inspect, eval_outdir=eval_outdir, args=args)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                skipped_rows.append(
                    {
                        "run_dir": str(run_dir),
                        "run_name": run_name,
                        "reason": "eval_failed",
                        "stderr": proc.stderr[-1000:],
                    }
                )
                continue

        if not results_path.exists():
            skipped_rows.append({"run_dir": str(run_dir), "run_name": run_name, "reason": "missing_eval_results"})
            continue

        try:
            results = _load_eval_results(eval_outdir)
        except Exception as ex:
            skipped_rows.append({"run_dir": str(run_dir), "run_name": run_name, "reason": "parse_results_failed: {}".format(ex)})
            continue

        interp_fid_value, interp_fid_status = _extract_interp_fid(results)
        recon_preview = Path(str(results.get("reconstruction", {}).get("artifact_preview", "")))
        if not recon_preview.is_absolute():
            recon_preview = eval_outdir / "reconstructions" / "reconstruction_pairs_preview.png"
        row = {
            "run_name": run_name,
            "run_key": run_key,
            "run_dir": str(run_dir),
            "eval_dir": str(eval_outdir),
            "checkpoint": str(inspect["checkpoint"]),
            "checkpoint_sha1_12": _sha1_short(Path(inspect["checkpoint"])),
            "recon_preview_sha1_12": _sha1_short(recon_preview),
            "dataset": inspect.get("dataset"),
            "dataset_cfg": inspect.get("dataset_cfg"),
            "seed_cfg": inspect.get("seed_cfg"),
            "run_type": inspect.get("run_type"),
            "mode": inspect.get("mode"),
            "latent_family": inspect.get("latent_family"),
            "latent_hw_cfg": str(inspect.get("latent_hw_cfg")),
            "scatter_type_cfg": inspect.get("scatter_type_cfg"),
            "corr_len_px_cfg": inspect.get("corr_len_px_cfg"),
            "z_to_sensor_mm_cfg": inspect.get("z_to_sensor_mm_cfg"),
            "pool_type_cfg": inspect.get("pool_type_cfg"),
            "pool_kernel_cfg": str(inspect.get("pool_kernel_cfg")),
            "pool_stride_cfg": str(inspect.get("pool_stride_cfg")),
            "scatter_type": inspect.get("scatter_type_in_name"),
            "corr_len_px": inspect.get("corr_len_px_in_name"),
            "l_mm": inspect.get("l_mm_in_name"),
            "no_pool": inspect.get("no_pool_in_name"),
            "recon_mse": _extract_metric(results, "mse"),
            "recon_psnr": _extract_metric(results, "psnr"),
            "recon_ssim": _extract_metric(results, "ssim"),
            "recon_kl": _extract_metric(results, "kl_mean"),
            "recon_samples": int(results.get("reconstruction", {}).get("metrics", {}).get("samples", results.get("samples", 0))),
            "interp_panels": int(results.get("interpolation", {}).get("saved_panels", 0)),
            "interp_neighbor_mse": _extract_interp_neighbor_metric(results),
            "interp_fid": interp_fid_value,
            "interp_fid_status": interp_fid_status,
        }
        run_rows_raw.append(row)

    # Aggregate on raw values first to avoid precision-loss in CI.
    ci_rows_raw = _aggregate_ci(run_rows_raw)
    contrast_rows_raw = _build_latent_vs_scatter(run_rows_raw)

    run_rows = _format_row_numbers_3dp(run_rows_raw)
    ci_rows = _format_row_numbers_3dp(ci_rows_raw)
    contrast_rows = _format_row_numbers_3dp(contrast_rows_raw)
    skipped_rows_round = _format_row_numbers_3dp(skipped_rows)

    write_summary_csv(outdir / "all_runs_metrics_raw.csv", run_rows_raw)
    write_summary_csv(outdir / "grouped_ci_summary_raw.csv", ci_rows_raw)
    write_summary_csv(outdir / "latent_vs_scatter_comparison_raw.csv", contrast_rows_raw)
    write_summary_csv(outdir / "all_runs_metrics.csv", run_rows)
    write_summary_csv(outdir / "grouped_ci_summary.csv", ci_rows)
    write_summary_csv(outdir / "latent_vs_scatter_comparison.csv", contrast_rows)
    write_summary_csv(outdir / "skipped_runs.csv", skipped_rows_round)

    save_json({"rows": run_rows_raw}, outdir / "all_runs_metrics_raw.json")
    save_json({"rows": ci_rows_raw}, outdir / "grouped_ci_summary_raw.json")
    save_json({"rows": contrast_rows_raw}, outdir / "latent_vs_scatter_comparison_raw.json")
    save_json({"rows": run_rows}, outdir / "all_runs_metrics.json")
    save_json({"rows": ci_rows}, outdir / "grouped_ci_summary.json")
    save_json({"rows": contrast_rows}, outdir / "latent_vs_scatter_comparison.json")
    save_json({"rows": skipped_rows_round}, outdir / "skipped_runs.json")

    used_cfg = {
        "outputs_root": str(outputs_root),
        "data_root": str(args.data_root),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "seed": int(args.seed),
        "compute_interp_fid": bool(args.compute_interp_fid),
        "fid_max_images": int(args.fid_max_images),
        "fid_batch_size": int(args.fid_batch_size),
        "re_eval": bool(args.re_eval),
        "num_discovered_runs": int(len(run_dirs)),
        "num_evaluated_runs": int(len(run_rows_raw)),
        "num_skipped_runs": int(len(skipped_rows)),
    }
    with (outdir / "config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(used_cfg, f, sort_keys=False, allow_unicode=True)

    print("Batch evaluation done.")
    print("outdir: {}".format(outdir))
    print("evaluated_runs: {} | skipped_runs: {}".format(len(run_rows_raw), len(skipped_rows)))


if __name__ == "__main__":
    main()
