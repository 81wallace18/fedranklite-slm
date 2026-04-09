from __future__ import annotations

import copy
import os
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def _resolve_vars(cfg: dict) -> dict:
    raw = yaml.dump(cfg)
    for key in ("experiment_name", "seed"):
        raw = raw.replace(f"${{{key}}}", str(cfg.get(key, "")))
    return yaml.safe_load(raw)


def _load_raw(path: Path) -> dict:
    """Load and merge YAML chain without resolving variables."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    base_ref = cfg.pop("_base_", None)
    if base_ref:
        base_path = (path.parent / base_ref).resolve()
        base_cfg = _load_raw(base_path)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = _load_raw(path)
    cfg = _resolve_vars(cfg)
    return cfg
