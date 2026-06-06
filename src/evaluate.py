import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from builders import build_models, checkpoint_config, load_model_states
from config import apply_overrides, load_config, save_config
from dataset import Cityscapes20ClassDataset
from flow_matching import ConditionalSourceFlowMatching
from metrics import SegmentationMetrics
from visualization import save_trajectory_grid


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
    "num_steps": "evaluation.num_steps",
    "num_visualize": "evaluation.num_visualize",
    "num_snap_points": "evaluation.num_snap_points",
    "limit_batches": "evaluation.limit_batches",
    "lam": "flow_matching.lam",
    "alpha_1": "flow_matching.alpha_1",
    "alpha_0": "flow_matching.alpha_0",
    "a": "flow_matching.a",
    "b": "flow_matching.b",
    "sampler": "flow_matching.sampler",
    "z0_mode": "flow_matching.z0_mode",
    "z0_noise_std": "flow_matching.z0_noise_std",
    "var_weight": "flow_matching.var_weight",
    "align_weight": "flow_matching.align_weight",
    "amp": "training.amp",
    "amp_dtype": "training.amp_dtype",
    "use_checkpoint": "training.use_checkpoint",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained conditional-source Flow Matching model."
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
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--num_visualize", type=int, default=None)
    parser.add_argument("--num_snap_points", type=int, default=None)
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--alpha_1", type=float, default=None)
    parser.add_argument("--alpha_0", type=float, default=None)
    parser.add_argument("--a", type=float, default=None)
    parser.add_argument("--b", type=float, default=None)
    parser.add_argument(
        "--sampler", choices=["euler", "midpoint", "RK4", "rk4"], default=None
    )
    parser.add_argument(
        "--z0_mode",
        choices=["generator", "gaussian", "dirichlet", "uniform"],
        default=None,
    )
    parser.add_argument("--z0_noise_std", type=float, default=None)
    parser.add_argument("--var_weight", type=float, default=None)
    parser.add_argument("--align_weight", type=float, default=None)
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
        handle.write(f"split      : {result['split']}\n")
        handle.write(f"num_steps  : {result['num_steps']}\n")
        handle.write(f"pixel_acc  : {result['pixel_acc']:.6f}\n")
        handle.write(f"mIoU       : {result['mIoU']:.6f}\n")
        handle.write(f"mAcc       : {result['mAcc']:.6f}\n")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    requested_config = apply_overrides(
        load_config(args.config), args, CLI_OVERRIDES
    )
    evaluation_config = requested_config["evaluation"]
    checkpoint_path = evaluation_config.get("ckpt_path")
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
        config["model"] = checkpoint_resolved["model"]
        config["source_generator"] = checkpoint_resolved["source_generator"]
        config["dataset"]["num_classes"] = checkpoint_resolved["dataset"][
            "num_classes"
        ]
        config["dataset"]["image_size"] = checkpoint_resolved["dataset"][
            "image_size"
        ]
        config["flow_matching"] = checkpoint_resolved["flow_matching"]
        config = apply_overrides(config, args, CLI_OVERRIDES)
        print("Loaded model architecture and Flow Matching defaults from checkpoint.")
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

    model, source_generator = build_models(config, device)
    load_model_states(checkpoint, model, source_generator)
    model.eval()
    source_generator.eval()

    flow = ConditionalSourceFlowMatching(
        K=int(dataset_config["num_classes"]),
        a=float(flow_config["a"]),
        b=float(flow_config["b"]),
        lam=float(flow_config["lam"]),
        alpha_1=float(flow_config["alpha_1"]),
        alpha_0=float(flow_config["alpha_0"]),
        device=device,
        z0_mode=flow_config["z0_mode"],
        sampler=flow_config["sampler"],
    )

    save_dir = Path(evaluation_config["save_dir"])
    visualizations_dir = save_dir / "visualizations"
    save_dir.mkdir(parents=True, exist_ok=True)
    visualizations_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, save_dir / "config.json")

    metrics = SegmentationMetrics(int(dataset_config["num_classes"]))
    visualized = 0
    total_samples = 0
    limit_batches = evaluation_config.get("limit_batches")

    for batch_index, (image, _, target) in enumerate(
        tqdm(loader, desc=f"evaluate:{evaluation_config['split']}")
    ):
        if limit_batches is not None and batch_index >= int(limit_batches):
            break
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type, dtype=amp_dtype, enabled=amp_enabled
        ):
            trajectory = flow.sampling_method(
                model=model,
                source_generator=source_generator,
                image=image,
                num_steps=int(evaluation_config["num_steps"]),
                return_traj=True,
            )
        prediction = trajectory[-1]
        metrics.update(prediction, target)
        total_samples += image.shape[0]

        remaining = max(
            0, int(evaluation_config["num_visualize"]) - visualized
        )
        for item_index in range(min(remaining, image.shape[0])):
            save_trajectory_grid(
                image=image[item_index].cpu(),
                target=target[item_index].cpu(),
                trajectory=trajectory[:, item_index],
                save_path=visualizations_dir
                / (
                    f"{evaluation_config['split']}_batch{batch_index:04d}"
                    f"_idx{item_index:02d}.png"
                ),
                num_snap_points=int(evaluation_config["num_snap_points"]),
                num_classes=int(dataset_config["num_classes"]),
            )
            visualized += 1

    result = metrics.compute()
    result.update(
        {
            "split": evaluation_config["split"],
            "checkpoint": str(checkpoint_path),
            "num_steps": int(evaluation_config["num_steps"]),
            "sampler": flow_config["sampler"],
            "z0_mode": flow_config["z0_mode"],
            "z0_noise_std": float(flow_config["z0_noise_std"]),
            "batch_size": int(evaluation_config["batch_size"]),
            "total_samples": total_samples,
            "num_visualize": visualized,
            "num_snap_points": int(evaluation_config["num_snap_points"]),
        }
    )
    write_metrics(result, save_dir)
    print(
        f"Evaluation complete: pixel_acc={result['pixel_acc']:.6f}, "
        f"mIoU={result['mIoU']:.6f}, mAcc={result['mAcc']:.6f}"
    )
    print(f"Results saved to {save_dir}.")


if __name__ == "__main__":
    main()
