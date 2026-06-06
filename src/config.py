import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    "dataset": {
        "name": "Cityscapes",
        "data_root": "~/datasets/cityscapes",
        "mode": "fine",
        "train_split": "train",
        "eval_split": "val",
        "num_classes": 20,
        "image_size": [128, 256],
        "num_workers": 4,
    },
    "training": {
        "result_dir": "outputs/csfm",
        "epochs": 150,
        "batch_size": 4,
        "amp": False,
        "amp_dtype": "bf16",
        "use_checkpoint": False,
        "resume": None,
        "resume_mode": "full",
        "resume_reset_lr": None,
        "grad_clip_norm": 1.0,
        "seed": 0,
    },
    "optimizer": {
        "name": "AdamW",
        "lr": 3.0e-5,
        "weight_decay": 1.0e-4,
    },
    "scheduler": {
        "name": "warmup_cosine",
        "warmup_epochs": 10,
        "eta_min": 1.0e-6,
    },
    "flow_matching": {
        "lam": 0.5,
        "alpha_1": 100.0,
        "alpha_0": 10.0,
        "a": 1.0,
        "b": 0.0,
        "sampler": "euler",
        "z0_mode": "generator",
        "z0_noise_std": 0.5,
        "var_weight": 5.0,
        "align_weight": 1.0,
    },
    "model": {
        "image_channels": 3,
        "fusion_channels": 128,
        "rrdb_blocks": 15,
        "rrdb_growth_channels": 32,
        "unet_base_channels": 128,
        "unet_channel_mults": [1, 2, 4, 8],
        "num_res_blocks": 3,
        "time_emb_dim": 256,
        "attn_levels": [3, 4],
        "dropout": 0.0,
        "num_heads": 4,
    },
    "source_generator": {
        "image_channels": 3,
        "fusion_channels": 128,
        "rrdb_blocks": 10,
        "rrdb_growth_channels": 32,
        "source_base_channels": 96,
        "source_channel_mults": [1, 4, 8],
        "num_res_blocks": 3,
        "attn_levels": [3],
        "dropout": 0.0,
        "num_heads": 4,
        "fixed_std": 0.5,
        "logvar_min": -6.0,
        "logvar_max": 3.0,
    },
    "evaluation": {
        "ckpt_path": None,
        "save_dir": "results/evaluation",
        "split": "val",
        "batch_size": 4,
        "num_steps": 50,
        "num_visualize": 16,
        "num_snap_points": 5,
        "decode_z0": False,
        "sample_seed": None,
        "limit_batches": None,
    },
    "wandb": {
        "use_wandb": False,
        "project": "conditional-source-flow-matching-segmentation",
        "name": None,
        "run_id": None,
        "resume": "allow",
    },
}


def deep_merge(base: Mapping[str, Any], update: Mapping[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], MutableMapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Optional[str]) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return config

    with open(path, "r", encoding="utf-8") as handle:
        user_config = json.load(handle)
    if not isinstance(user_config, dict):
        raise ValueError(f"Config must contain a JSON object: {path}")
    return deep_merge(config, user_config)


def set_nested(config: MutableMapping[str, Any], path: str, value: Any) -> None:
    keys = path.split(".")
    current = config
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def apply_overrides(
    config: Dict[str, Any],
    args: Any,
    mapping: Mapping[str, str],
) -> Dict[str, Any]:
    updated = copy.deepcopy(config)
    for arg_name, config_path in mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            set_nested(updated, config_path, value)
    return updated


def save_config(config: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
