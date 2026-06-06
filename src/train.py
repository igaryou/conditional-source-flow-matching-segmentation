import argparse
import os
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from builders import build_models, checkpoint_config, load_model_states
from config import apply_overrides, load_config, save_config
from dataset import Cityscapes20ClassDataset
from flow_matching import ConditionalSourceFlowMatching
from visualization import save_loss_curve


CLI_OVERRIDES = {
    "data_root": "dataset.data_root",
    "result_dir": "training.result_dir",
    "num_classes": "dataset.num_classes",
    "image_size": "dataset.image_size",
    "epochs": "training.epochs",
    "batch_size": "training.batch_size",
    "num_workers": "dataset.num_workers",
    "lr": "optimizer.lr",
    "weight_decay": "optimizer.weight_decay",
    "warmup_epochs": "scheduler.warmup_epochs",
    "eta_min": "scheduler.eta_min",
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
    "resume": "training.resume",
    "resume_mode": "training.resume_mode",
    "resume_reset_lr": "training.resume_reset_lr",
    "use_wandb": "wandb.use_wandb",
    "wandb_project": "wandb.project",
    "wandb_name": "wandb.name",
    "wandb_run_id": "wandb.run_id",
    "wandb_resume": "wandb.resume",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train conditional-source Flow Matching for semantic segmentation."
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--result_dir", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=None)
    parser.add_argument("--image_size", type=int, nargs=2, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--eta_min", type=float, default=None)
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
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--resume_mode", choices=["full", "model_only"], default=None
    )
    parser.add_argument("--resume_reset_lr", type=float, default=None)
    parser.add_argument(
        "--use_wandb",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument(
        "--wandb_resume",
        choices=["allow", "must", "never", "auto"],
        default=None,
    )
    return parser.parse_args()


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    parameters = module.parameters()
    if trainable_only:
        parameters = (parameter for parameter in parameters if parameter.requires_grad)
    return sum(parameter.numel() for parameter in parameters)


def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    eta_min: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative")

    warmup_epochs = min(warmup_epochs, max(epochs - 1, 0))
    if warmup_epochs == 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1), eta_min=eta_min
        )

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs - warmup_epochs, 1),
        eta_min=eta_min,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def initialize_wandb(config: Mapping[str, Any]):
    wandb_config = config["wandb"]
    if not wandb_config.get("use_wandb", False):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb is not installed; continuing with local logging only.")
        return None

    return wandb.init(
        project=wandb_config["project"],
        name=wandb_config.get("name"),
        id=wandb_config.get("run_id"),
        resume=(
            wandb_config.get("resume")
            if wandb_config.get("run_id") is not None
            else None
        ),
        config=dict(config),
    )


def checkpoint_payload(
    model: torch.nn.Module,
    source_generator: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    loss_best: float,
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "model": model.state_dict(),
        "source_generator": source_generator.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "loss_best": loss_best,
        "config": dict(config),
    }


def main() -> None:
    args = parse_args()
    requested_config = apply_overrides(
        load_config(args.config), args, CLI_OVERRIDES
    )

    resume_path = requested_config["training"].get("resume")
    resume_checkpoint: Optional[Mapping[str, Any]] = None
    config = requested_config
    if resume_path:
        resume_checkpoint = torch.load(resume_path, map_location="cpu")
        checkpoint_resolved, has_config = checkpoint_config(
            resume_checkpoint, requested_config
        )
        if has_config:
            config["model"] = checkpoint_resolved["model"]
            config["source_generator"] = checkpoint_resolved[
                "source_generator"
            ]
            config["dataset"]["num_classes"] = checkpoint_resolved["dataset"][
                "num_classes"
            ]
            print("Using model architecture stored in the resume checkpoint.")
        else:
            print("Resume checkpoint has no config; using the requested config.")

    training_config = config["training"]
    dataset_config = config["dataset"]
    optimizer_config = config["optimizer"]
    scheduler_config = config["scheduler"]
    flow_config = config["flow_matching"]
    if args.z0_noise_std is not None:
        config["source_generator"]["fixed_std"] = args.z0_noise_std
    fixed_std = config["source_generator"].get("fixed_std")
    if fixed_std is not None:
        flow_config["z0_noise_std"] = fixed_std
    if (
        resume_path
        and training_config["resume_mode"] == "model_only"
        and training_config.get("resume_reset_lr") is not None
    ):
        optimizer_config["lr"] = float(training_config["resume_reset_lr"])

    seed = int(training_config.get("seed", 0))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_requested = bool(training_config["amp"])
    amp_enabled = amp_requested and device.type == "cuda"
    amp_dtype = training_config["amp_dtype"]
    amp_torch_dtype = (
        torch.bfloat16 if amp_dtype == "bf16" else torch.float16
    )
    if amp_requested and not amp_enabled:
        print("AMP requested but CUDA is unavailable; using fp32.")

    result_dir = Path(training_config["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    dataset_config["data_root"] = os.path.expanduser(dataset_config["data_root"])

    dataset = Cityscapes20ClassDataset(
        root=dataset_config["data_root"],
        split=dataset_config["train_split"],
        mode=dataset_config["mode"],
        image_size=tuple(dataset_config["image_size"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        num_workers=int(dataset_config["num_workers"]),
        pin_memory=device.type == "cuda",
    )

    model, source_generator = build_models(config, device)
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

    parameters = list(model.parameters()) + list(source_generator.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(optimizer_config["lr"]),
        weight_decay=float(optimizer_config["weight_decay"]),
    )
    scheduler = build_scheduler(
        optimizer,
        epochs=int(training_config["epochs"]),
        warmup_epochs=int(scheduler_config["warmup_epochs"]),
        eta_min=float(scheduler_config["eta_min"]),
    )

    start_epoch = 0
    loss_best = float("inf")
    resume_mode = training_config["resume_mode"]
    if resume_checkpoint is not None:
        load_model_states(resume_checkpoint, model, source_generator)
        checkpoint_epoch = int(resume_checkpoint.get("epoch", -1))
        if resume_mode == "full":
            if "optimizer" not in resume_checkpoint or "scheduler" not in resume_checkpoint:
                raise KeyError(
                    "Full resume requires optimizer and scheduler checkpoint states."
                )
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
            move_optimizer_state_to_device(optimizer, device)
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
            start_epoch = checkpoint_epoch + 1
            loss_best = float(resume_checkpoint.get("loss_best", loss_best))
        else:
            start_epoch = 0
            loss_best = float("inf")
        print(
            f"Loaded resume checkpoint at epoch {checkpoint_epoch + 1} "
            f"with mode '{resume_mode}'."
        )

    config["runtime"] = {
        "device": str(device),
        "amp_enabled": amp_enabled,
        "model_parameters": count_parameters(model),
        "source_generator_parameters": count_parameters(source_generator),
        "trainable_parameters": sum(
            count_parameters(module, trainable_only=True)
            for module in (model, source_generator)
        ),
    }
    save_config(config, result_dir / "config.json")
    wandb_run = initialize_wandb(config)

    scaler = None
    if amp_enabled and amp_dtype == "fp16":
        scaler = torch.cuda.amp.GradScaler()

    losses = []
    log_path = result_dir / "train_log.txt"
    epochs = int(training_config["epochs"])
    grad_clip_norm = float(training_config.get("grad_clip_norm", 1.0))

    for epoch in range(start_epoch, epochs):
        model.train()
        source_generator.train()
        accumulated = {
            "loss": 0.0,
            "loss_mse": 0.0,
            "loss_var": 0.0,
            "loss_align": 0.0,
            "weighted_var": 0.0,
            "weighted_align": 0.0,
            "var_weight_dyn": 0.0,
            "align_weight_dyn": 0.0,
            "aux_warmup": 0.0,
            "ema_mse": 0.0,
            "ema_var": 0.0,
            "ema_align": 0.0,
            "pred_velocity": 0.0,
            "target_velocity": 0.0,
            "zt_abs": 0.0,
            "z1_abs": 0.0,
            "mu_abs": 0.0,
            "mu_min": 0.0,
            "mu_max": 0.0,
            "logvar_mean": 0.0,
            "logvar_min": 0.0,
            "logvar_max": 0.0,
            "sigma_mean": 0.0,
            "sigma_min": 0.0,
            "sigma_max": 0.0,
            "grad_norm": 0.0,
        }
        batch_count = 0

        progress = tqdm(loader, desc=f"train epoch {epoch + 1}/{epochs}")
        for image, onehot, _ in progress:
            optimizer.zero_grad(set_to_none=True)
            image = image.to(device, non_blocking=True)
            onehot = onehot.to(device=device, dtype=image.dtype, non_blocking=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_torch_dtype,
                enabled=amp_enabled,
            ):
                loss, logs = flow.loss(
                    onehot=onehot,
                    image=image,
                    model=model,
                    source_generator=source_generator,
                    var_weight=float(flow_config["var_weight"]),
                    align_weight=float(flow_config["align_weight"]),
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, grad_clip_norm
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    parameters, grad_clip_norm
                )
                optimizer.step()

            for key in accumulated:
                if key == "grad_norm":
                    accumulated[key] += float(grad_norm)
                elif key in logs:
                    accumulated[key] += logs[key]
            batch_count += 1
            progress.set_postfix(loss=f"{logs['loss']:.5f}")

        scheduler.step()
        for key in accumulated:
            accumulated[key] /= max(batch_count, 1)
        losses.append(accumulated["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        log_line = (
            f"epoch:{epoch + 1} loss:{accumulated['loss']:.6f} "
            f"loss_mse:{accumulated['loss_mse']:.6f} "
            f"loss_var:{accumulated['loss_var']:.6f} "
            f"loss_align:{accumulated['loss_align']:.6f} "
            f"lr:{current_lr:.8f} grad_norm:{accumulated['grad_norm']:.4f} "
            f"pred_velocity:{accumulated['pred_velocity']:.4f} "
            f"target_velocity:{accumulated['target_velocity']:.4f} "
            f"mu_abs:{accumulated['mu_abs']:.4f} "
            f"sigma_mean:{accumulated['sigma_mean']:.4f}"
        )
        print(log_line)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(log_line + "\n")

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": accumulated["loss"],
                    "train/loss_velocity": accumulated["loss_mse"],
                    "train/loss_var": accumulated["loss_var"],
                    "train/loss_align": accumulated["loss_align"],
                    "train/lr": current_lr,
                    "train/grad_norm": accumulated["grad_norm"],
                    "train/weighted_var": accumulated["weighted_var"],
                    "train/weighted_align": accumulated["weighted_align"],
                    "source_generator/mu_abs": accumulated["mu_abs"],
                    "source_generator/sigma_mean": accumulated["sigma_mean"],
                }
            )

        if (epoch + 1) >= 20 and accumulated["loss"] < loss_best:
            loss_best = accumulated["loss"]
            torch.save(
                checkpoint_payload(
                    model,
                    source_generator,
                    optimizer,
                    scheduler,
                    epoch,
                    loss_best,
                    config,
                ),
                result_dir / "model.pth",
            )

    final_epoch = max(epochs - 1, start_epoch - 1)
    torch.save(
        checkpoint_payload(
            model,
            source_generator,
            optimizer,
            scheduler,
            final_epoch,
            loss_best,
            config,
        ),
        result_dir / "model_final.pth",
    )
    if losses:
        save_loss_curve(losses, result_dir / "loss_curve.png")
    if wandb_run is not None:
        wandb_run.finish()

    print(f"Training complete. Checkpoints and config saved to {result_dir}.")


if __name__ == "__main__":
    main()
