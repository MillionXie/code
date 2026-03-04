from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a dict-like YAML object")
    return cfg


def apply_cli_overrides(
    cfg: Dict[str, Any],
    dataset: Optional[str] = None,
    outdir: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    out = deepcopy(cfg)
    out.setdefault("train", {})

    if dataset is not None:
        out["dataset"] = dataset
    if outdir is not None:
        out["outdir"] = outdir
    if epochs is not None:
        out["train"]["epochs"] = int(epochs)
    if batch_size is not None:
        out["train"]["batch_size"] = int(batch_size)
    if lr is not None:
        out["train"]["lr"] = float(lr)
    if seed is not None:
        out["train"]["seed"] = int(seed)

    return out


__all__ = ["load_config", "apply_cli_overrides"]
