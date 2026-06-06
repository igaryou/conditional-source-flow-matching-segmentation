import copy
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from config import DEFAULT_CONFIG, deep_merge
from models import CSFMSourceGenerator, SegDiffModel


def _canonicalize_source_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    source = copy.deepcopy(dict(config))
    if "source_base_channels" not in source and "unet_base_channels" in source:
        source["source_base_channels"] = source.pop("unet_base_channels")
    if "source_channel_mults" not in source and "unet_channel_mults" in source:
        source["source_channel_mults"] = source.pop("unet_channel_mults")
    return source


def _legacy_checkpoint_config(saved: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    model_config = saved.get("model_config")
    if isinstance(model_config, Mapping):
        model = model_config.get("model")
        if isinstance(model, Mapping):
            normalized["model"] = copy.deepcopy(dict(model))

        source = model_config.get("source_generator")
        if isinstance(source, Mapping):
            normalized["source_generator"] = _canonicalize_source_config(source)

    top_level_mapping = {
        "num_classes": ("dataset", "num_classes"),
        "image_size": ("dataset", "image_size"),
        "batch_size": ("training", "batch_size"),
        "epochs": ("training", "epochs"),
        "base_lr": ("optimizer", "lr"),
        "lr": ("optimizer", "lr"),
        "weight_decay": ("optimizer", "weight_decay"),
        "eta_min": ("scheduler", "eta_min"),
        "lam": ("flow_matching", "lam"),
        "alpha_1": ("flow_matching", "alpha_1"),
        "alpha_0": ("flow_matching", "alpha_0"),
        "a": ("flow_matching", "a"),
        "b": ("flow_matching", "b"),
        "sampler": ("flow_matching", "sampler"),
        "z0_mode": ("flow_matching", "z0_mode"),
        "z0_noise_std": ("flow_matching", "z0_noise_std"),
        "var_weight": ("flow_matching", "var_weight"),
        "align_weight": ("flow_matching", "align_weight"),
        "use_amp": ("training", "amp"),
        "amp_dtype": ("training", "amp_dtype"),
        "use_checkpoint": ("training", "use_checkpoint"),
    }
    for old_key, (section, new_key) in top_level_mapping.items():
        if old_key in saved:
            value = copy.deepcopy(saved[old_key])
            if old_key == "image_size" and isinstance(value, str):
                parts = value.lower().split("x")
                if len(parts) == 2:
                    value = [int(parts[0]), int(parts[1])]
            normalized.setdefault(section, {})[new_key] = value

    return normalized


def checkpoint_config(
    checkpoint: Mapping[str, Any],
    fallback: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], bool]:
    base = copy.deepcopy(dict(fallback or DEFAULT_CONFIG))
    saved = checkpoint.get("config")
    if not isinstance(saved, Mapping):
        return base, False

    if isinstance(saved.get("model"), Mapping) and isinstance(
        saved.get("source_generator"), Mapping
    ):
        normalized = copy.deepcopy(dict(saved))
        normalized["source_generator"] = _canonicalize_source_config(
            normalized["source_generator"]
        )
    else:
        normalized = _legacy_checkpoint_config(saved)
    return deep_merge(base, normalized), bool(normalized)


def model_configs(config: Mapping[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    num_classes = int(config["dataset"]["num_classes"])
    use_checkpoint = bool(config["training"].get("use_checkpoint", False))

    model_config = copy.deepcopy(dict(config["model"]))
    model_config["mask_channels"] = num_classes - 1
    model_config["use_checkpoint"] = use_checkpoint
    model_config["unet_channel_mults"] = tuple(model_config["unet_channel_mults"])
    model_config["attn_levels"] = tuple(model_config["attn_levels"])

    source_config = _canonicalize_source_config(config["source_generator"])
    source_config["mask_channels"] = num_classes - 1
    source_config["use_checkpoint"] = use_checkpoint
    source_config["source_channel_mults"] = tuple(
        source_config["source_channel_mults"]
    )
    source_config["attn_levels"] = tuple(source_config["attn_levels"])
    return model_config, source_config


def build_models(
    config: Mapping[str, Any],
    device: torch.device,
) -> Tuple[SegDiffModel, CSFMSourceGenerator]:
    model_config, source_config = model_configs(config)
    model = SegDiffModel(**model_config).to(device)
    source_generator = CSFMSourceGenerator(**source_config).to(device)
    return model, source_generator


def build_source_generator(
    config: Mapping[str, Any],
    device: torch.device,
) -> CSFMSourceGenerator:
    _, source_config = model_configs(config)
    return CSFMSourceGenerator(**source_config).to(device)


def load_model_states(
    checkpoint: Mapping[str, Any],
    model: SegDiffModel,
    source_generator: CSFMSourceGenerator,
) -> None:
    model.load_state_dict(checkpoint["model"])
    source_generator.load_state_dict(checkpoint["source_generator"])


def source_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    state = checkpoint.get("source_generator")
    if isinstance(state, Mapping):
        return state
    return checkpoint
