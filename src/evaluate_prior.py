import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from builders import (
    build_source_generator,
    checkpoint_config,
    source_state_dict,
)
from config import apply_overrides, load_config, save_config
from dataset import Cityscapes20ClassDataset
from flow_matching import ConditionalSourceFlowMatching
from metrics import SegmentationMetrics
from visualization import save_prior_grid, save_raw_mask


CLI_OVERRIDES = {
    "data_root": "dataset.data_root",
    "result_dir": "training.result_dir",
    "num_classes": "dataset.num_classes",
    "image_size": "dataset.image_size",
    "batch_size": "evaluation.batch_size",
    "num_workers": "dataset.num_workers",
    "ckpt_path": "evaluation.ckpt_path",
    "save_dir": "evaluation.save_dir",
    "split": "evaluation.split",
    "num_visualize": "evaluation.num_visualize",
    "decode_z0": "evaluation.decode_z0",
    "sample_seed": "evaluation.sample_seed",
    "limit_batches": "evaluation.limit_batches",
    "lam": "flow_matching.lam",
    "alpha_1": "flow_matching.alpha_1",
    "alpha_0": "flow_matching.alpha_0",
    "z0_noise_std": "flow_matching.z0_noise_std",
    "amp": "training.amp",
    "amp_dtype": "training.amp_dtype",
    "use_checkpoint": "training.use_checkpoint",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Decode and evaluate the image-conditioned source mean or a sampled "
            "source latent before Flow Matching integration."
        )
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--image_size", type=int, nargs=2, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default=None
    )
    parser.add_argument("--num_visualize", type=int, default=None)
    parser.add_argument(
        "--decode_z0",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Decode a stochastic z0 sample instead of the source mean.",
    )
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--alpha_1", type=float, default=None)
    parser.add_argument("--alpha_0", type=float, default=None)
    parser.add_argument("--z0_noise_std", type=float, default=None)
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--amp_dtype", choices=["bf16", "fp16"], default=None
    )
    parser.add_argument(
        "--use_checkpoint",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def write_metrics(result: Dict[str, Any], save_dir: Path) -> None:
    with open(save_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    with open(save_dir / "metrics.txt", "w", encoding="utf-8") as handle:
        handle.write("# Source prior evaluation\n")
        for key in (
            "split",
            "checkpoint",
            "eval_mode",
            "decode_z0",
            "sample_seed",
            "total_samples",
            "pixel_acc",
            "mIoU",
            "mAcc",
            "eval_z_z1_mse",
            "eval_z_z1_align_mse",
            "eval_z_z1_cos_pixel",
            "sigma_mean",
            "sigma_min_batchavg",
            "sigma_max_batchavg",
            "decode_mean_confidence",
        ):
            value = result[key]
            if isinstance(value, float):
                handle.write(f"{key:24s}: {value:.6f}\n")
            else:
                handle.write(f"{key:24s}: {value}\n")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    requested_config = apply_overrides(
        load_config(args.config), args, CLI_OVERRIDES
    )
    checkpoint_path = requested_config["evaluation"].get("ckpt_path")
    if checkpoint_path is None:
        checkpoint_path = str(
            Path(requested_config["training"]["result_dir"]) / "model_final.pth"
        )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_resolved, has_checkpoint_config = checkpoint_config(
        checkpoint, requested_config
    )
    config = requested_config
    if has_checkpoint_config:
        config["source_generator"] = checkpoint_resolved["source_generator"]
        config["dataset"]["num_classes"] = checkpoint_resolved["dataset"][
            "num_classes"
        ]
        config["dataset"]["image_size"] = checkpoint_resolved["dataset"][
            "image_size"
        ]
        config["flow_matching"] = checkpoint_resolved["flow_matching"]
        config = apply_overrides(config, args, CLI_OVERRIDES)
        print("Loaded source-generator architecture from checkpoint.")
    else:
        print("Checkpoint has no config; using config-file fallback architecture.")

    dataset_config = config["dataset"]
    evaluation_config = config["evaluation"]
    training_config = config["training"]
    flow_config = config["flow_matching"]
    if args.z0_noise_std is not None:
        config["source_generator"]["fixed_std"] = args.z0_noise_std
    fixed_std = config["source_generator"].get("fixed_std")
    if fixed_std is not None:
        flow_config["z0_noise_std"] = fixed_std
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_seed = evaluation_config.get("sample_seed")
    if sample_seed is not None:
        torch.manual_seed(int(sample_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(sample_seed))

    amp_enabled = bool(training_config["amp"] and device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if training_config["amp_dtype"] == "bf16"
        else torch.float16
    )
    dataset_config["data_root"] = os.path.expanduser(dataset_config["data_root"])

    dataset = Cityscapes20ClassDataset(
        root=dataset_config["data_root"],
        split=evaluation_config["split"],
        mode=dataset_config["mode"],
        image_size=tuple(dataset_config["image_size"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(evaluation_config["batch_size"]),
        shuffle=False,
        num_workers=int(dataset_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    source = build_source_generator(config, device)
    source.load_state_dict(source_state_dict(checkpoint), strict=True)
    source.eval()
    flow = ConditionalSourceFlowMatching(
        K=int(dataset_config["num_classes"]),
        lam=float(flow_config["lam"]),
        alpha_1=float(flow_config["alpha_1"]),
        alpha_0=float(flow_config["alpha_0"]),
        device=device,
    )

    save_dir = Path(evaluation_config["save_dir"])
    visualizations_dir = save_dir / "visualizations"
    masks_dir = save_dir / "pred_masks"
    save_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, save_dir / "config.json")

    decode_z0 = bool(evaluation_config["decode_z0"])
    noise_std = float(flow_config["z0_noise_std"])
    eval_mode = "z0" if decode_z0 else "mu"
    prediction_title = "decode(z0)" if decode_z0 else "decode(mu)"
    metrics = SegmentationMetrics(int(dataset_config["num_classes"]))

    totals = {
        "samples": 0,
        "eval_z_z1_mse": 0.0,
        "eval_z_z1_align_mse": 0.0,
        "eval_z_z1_cos_pixel": 0.0,
        "eval_z_abs": 0.0,
        "mu_abs": 0.0,
        "z1_abs": 0.0,
        "sigma_mean": 0.0,
        "sigma_min_batchavg": 0.0,
        "sigma_max_batchavg": 0.0,
        "decode_mean_confidence": 0.0,
    }
    visualized = 0
    limit_batches = evaluation_config.get("limit_batches")

    for batch_index, (image, _, target) in enumerate(
        tqdm(loader, desc=f"{eval_mode}-prior:{evaluation_config['split']}")
    ):
        if limit_batches is not None and batch_index >= int(limit_batches):
            break
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).long()
        batch_size = image.shape[0]

        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            z0, mu, logvar = source(image)
            z_eval = z0 if decode_z0 else mu
            probabilities, prediction = flow.decode_ilr(z_eval)

        metrics.update(prediction, target)
        z1 = flow.masks_to_z1(target).to(
            device=z_eval.device, dtype=z_eval.dtype
        )
        eval_float = z_eval.float()
        mu_float = mu.float()
        z1_float = z1.float()
        eval_normalized = F.normalize(eval_float, p=2, dim=1)
        z1_normalized = F.normalize(z1_float, p=2, dim=1)
        confidence = probabilities.max(dim=1).values.float()
        sigma = torch.exp(0.5 * logvar.float())

        totals["samples"] += batch_size
        totals["eval_z_z1_mse"] += (
            float(F.mse_loss(eval_float, z1_float).item()) * batch_size
        )
        totals["eval_z_z1_align_mse"] += (
            float(F.mse_loss(eval_normalized, z1_normalized).item())
            * batch_size
        )
        totals["eval_z_z1_cos_pixel"] += (
            float((eval_normalized * z1_normalized).sum(dim=1).mean().item())
            * batch_size
        )
        totals["eval_z_abs"] += float(eval_float.abs().mean().item()) * batch_size
        totals["mu_abs"] += float(mu_float.abs().mean().item()) * batch_size
        totals["z1_abs"] += float(z1_float.abs().mean().item()) * batch_size
        totals["sigma_mean"] += float(sigma.mean().item()) * batch_size
        totals["sigma_min_batchavg"] += float(sigma.min().item()) * batch_size
        totals["sigma_max_batchavg"] += float(sigma.max().item()) * batch_size
        totals["decode_mean_confidence"] += (
            float(confidence.mean().item()) * batch_size
        )

        remaining = max(
            0, int(evaluation_config["num_visualize"]) - visualized
        )
        for item_index in range(min(remaining, batch_size)):
            stem = (
                f"{evaluation_config['split']}_batch{batch_index:04d}"
                f"_idx{item_index:02d}"
            )
            save_prior_grid(
                image=image[item_index].cpu(),
                target=target[item_index].cpu(),
                prediction=prediction[item_index].cpu(),
                confidence=confidence[item_index].cpu(),
                save_path=visualizations_dir / f"{stem}.png",
                num_classes=int(dataset_config["num_classes"]),
                prediction_title=prediction_title,
            )
            save_raw_mask(
                prediction[item_index], masks_dir / f"{stem}_pred.png"
            )
            visualized += 1

    result = metrics.compute()
    denominator = max(int(totals["samples"]), 1)
    result.update(
        {
            "split": evaluation_config["split"],
            "checkpoint": str(checkpoint_path),
            "eval_mode": eval_mode,
            "decode_z0": decode_z0,
            "sample_seed": sample_seed,
            "total_samples": int(totals["samples"]),
            "visualized": visualized,
            "z0_noise_std": noise_std,
        }
    )
    for key, value in totals.items():
        if key != "samples":
            result[key] = value / denominator
    write_metrics(result, save_dir)
    print(
        f"Prior evaluation complete: pixel_acc={result['pixel_acc']:.6f}, "
        f"mIoU={result['mIoU']:.6f}, mAcc={result['mAcc']:.6f}"
    )
    print(f"Results saved to {save_dir}.")


if __name__ == "__main__":
    main()
