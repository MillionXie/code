from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

from data.datasets import get_dataloaders
from models import IdentityAdapter
from utils.config import load_config
from utils.map_optical import build_map_core_from_cfg, build_optical_adapter_from_cfg
from utils.metrics import mse_loss, psnr_from_mse, ssim_score


def load_analysis_config(path: str) -> Dict[str, Any]:
    cfg = load_config(path)
    cfg.setdefault("analysis", {})
    cfg.setdefault("checkpoints", {})
    return cfg


def save_config_used(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def ensure_analysis_dirs(outdir: str | Path) -> Tuple[Path, Path]:
    root = Path(outdir)
    fig_dir = root / "figures"
    metrics_dir = root / "metrics"
    fig_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, metrics_dir


def infer_mode_from_cfg(cfg: Dict[str, Any]) -> str:
    return "optical" if cfg.get("optics", None) is not None else "electronic"


def load_model_bundle_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    ckpt_path = Path(checkpoint_path)
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("config", None)
    if not isinstance(cfg, dict):
        raise ValueError("Checkpoint {} missing `config` dict.".format(ckpt_path))

    resolved_mode = str(mode) if mode is not None else infer_mode_from_cfg(cfg)
    resolved_mode = resolved_mode.lower()
    if resolved_mode not in ("electronic", "optical"):
        raise ValueError("Unsupported mode: {}".format(resolved_mode))

    data_cfg = cfg.get("data", {})
    dataset = str(cfg.get("dataset", "mnist")).lower()
    out_range = str(data_cfg.get("out_range", "zero_one"))
    image_size = tuple(int(v) for v in data_cfg.get("image_size", [64, 64]))

    dataset_info = {
        "in_channels": 1 if dataset in ("mnist", "fashionmnist") else 3,
        "image_size": image_size,
    }
    model = build_map_core_from_cfg(cfg, dataset_info).to(device)
    if resolved_mode == "optical":
        adapter = build_optical_adapter_from_cfg(cfg, model).to(device)
    else:
        adapter = IdentityAdapter().to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    if "adapter_state_dict" in ckpt:
        adapter.load_state_dict(ckpt["adapter_state_dict"])
    model.eval()
    adapter.eval()

    latent_hw = getattr(model, "latent_hw", None)
    latent_channels = getattr(model, "latent_channels", None)
    if resolved_mode == "optical" and hasattr(adapter, "latent_h") and hasattr(adapter, "latent_w"):
        latent_hw = (int(adapter.latent_h), int(adapter.latent_w))
    if resolved_mode == "optical" and hasattr(adapter, "latent_channels"):
        latent_channels = int(adapter.latent_channels)

    return {
        "mode": resolved_mode,
        "checkpoint": str(ckpt_path),
        "config": cfg,
        "dataset": dataset,
        "out_range": out_range,
        "image_size": image_size,
        "model": model,
        "adapter": adapter,
        "latent_hw": latent_hw,
        "latent_channels": latent_channels,
    }


def build_test_loader(
    dataset: str,
    data_root: str,
    out_range: str,
    image_size: Tuple[int, int],
    batch_size: int,
    num_workers: int,
    seed: int,
):
    _, _, test_loader, dataset_info = get_dataloaders(
        dataset=dataset,
        data_root=data_root,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        out_range=out_range,
        seed=int(seed),
        image_size=list(image_size),
    )
    return test_loader, dataset_info


def extract_decoder_latent(
    bundle: Dict[str, Any],
    x: torch.Tensor,
    return_info: bool = False,
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    model = bundle["model"]
    adapter = bundle["adapter"]
    mode = bundle["mode"]

    if mode == "optical":
        z, info = adapter.encode_from_input(x, return_info=True, sample_posterior=False)
        if return_info:
            return z, info
        return z, None

    mu_map, logvar_map = model.encode(x)
    z = adapter(mu_map)
    info = {
        "mu_map": mu_map,
        "logvar_map": logvar_map,
        "final_latent_map": z,
        "latent_mean_map": mu_map,
        "latent_intensity_map": torch.clamp(mu_map, min=0.0),
    }
    if return_info:
        return z, info
    return z, None


def decode_from_latent(bundle: Dict[str, Any], z: torch.Tensor) -> torch.Tensor:
    return bundle["model"].decode(z)


def evaluate_batch_metrics(
    recon: torch.Tensor,
    target: torch.Tensor,
    data_range: float,
) -> Dict[str, torch.Tensor]:
    mse_ps = mse_loss(recon, target, reduction="none")
    psnr_ps = psnr_from_mse(mse_ps, data_range=float(data_range))
    ssim_ps = ssim_score(recon, target, data_range=float(data_range), reduction="none")
    return {"mse": mse_ps, "psnr": psnr_ps, "ssim": ssim_ps}


def collect_first_n_from_loader(loader, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    count = 0
    target_n = int(max(n, 0))
    for x, y in loader:
        if count >= target_n:
            break
        take = min(target_n - count, x.size(0))
        xs.append(x[:take].clone())
        ys.append(y[:take].clone())
        count += take
    if len(xs) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def pick_stage_intensity_map(info: Dict[str, Any], preferred_names: List[str]) -> Optional[torch.Tensor]:
    names = info.get("stage_intensity_names", [])
    maps = info.get("stage_intensity_maps", [])
    if not isinstance(names, list) or not isinstance(maps, list) or len(maps) == 0:
        return None
    for key in preferred_names:
        if key in names:
            return maps[names.index(key)]
    return maps[-1]


def normalize_log_intensity_for_display(x: torch.Tensor) -> torch.Tensor:
    x = torch.log1p(torch.clamp(x, min=0.0))
    x_min = x.amin(dim=(-2, -1), keepdim=True)
    x_max = x.amax(dim=(-2, -1), keepdim=True)
    return torch.clamp((x - x_min) / (x_max - x_min + 1e-8), 0.0, 1.0)


def flatten_latent(z: torch.Tensor) -> np.ndarray:
    return z.detach().cpu().flatten(start_dim=1).numpy()


def to_display_range(x: torch.Tensor, out_range: str) -> torch.Tensor:
    if str(out_range).lower() == "neg_one_one":
        x = (x + 1.0) / 2.0
    return torch.clamp(x, 0.0, 1.0)


def fetch_dataset_indices(test_loader, indices: Sequence[int]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    dataset = getattr(test_loader, "dataset", None)
    if dataset is None:
        raise ValueError("test_loader has no dataset attribute.")

    valid_indices: List[int] = []
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    n_total = len(dataset)
    for idx in indices:
        i = int(idx)
        if i < 0 or i >= n_total:
            continue
        x, y = dataset[i]
        xs.append(x.unsqueeze(0))
        ys.append(torch.tensor([int(y)], dtype=torch.long))
        valid_indices.append(i)

    if len(xs) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long), []
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0), valid_indices
