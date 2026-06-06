from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_loss_curve(losses: Sequence[float], save_path: Path) -> None:
    plt = _pyplot()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.plot(range(1, len(losses) + 1), losses, label="loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.grid(True)
    axis.legend()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _snapshot_indices(num_frames: int, num_snap_points: int) -> List[int]:
    if num_frames <= 1:
        return [0]
    if num_snap_points <= 1:
        return [num_frames - 1]
    indices = torch.linspace(0, num_frames - 1, steps=num_snap_points)
    output: List[int] = []
    for index in torch.round(indices).to(torch.int64).tolist():
        if index not in output:
            output.append(index)
    return output


@torch.no_grad()
def save_trajectory_grid(
    image: torch.Tensor,
    target: torch.Tensor,
    trajectory: torch.Tensor,
    save_path: Path,
    num_snap_points: int,
    num_classes: int,
) -> None:
    plt = _pyplot()
    if trajectory.ndim != 3:
        raise ValueError("trajectory must have shape [T, H, W]")

    snapshot_ids = _snapshot_indices(trajectory.shape[0], num_snap_points)
    num_columns = len(snapshot_ids) + 2
    fig, axes = plt.subplots(1, num_columns, figsize=(2.7 * num_columns, 3.5))

    axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("image")
    axes[1].imshow(target.cpu().numpy(), vmin=0, vmax=num_classes - 1)
    axes[1].set_title("ground truth")

    for axis in axes[:2]:
        axis.axis("off")

    for axis_index, trajectory_index in enumerate(snapshot_ids, start=2):
        axes[axis_index].imshow(
            trajectory[trajectory_index].cpu().numpy(),
            vmin=0,
            vmax=num_classes - 1,
        )
        axes[axis_index].set_title(f"step {trajectory_index + 1}")
        axes[axis_index].axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_prior_grid(
    image: torch.Tensor,
    target: torch.Tensor,
    prediction: torch.Tensor,
    confidence: torch.Tensor,
    save_path: Path,
    num_classes: int,
    prediction_title: str,
) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("image")
    axes[1].imshow(target.cpu().numpy(), vmin=0, vmax=num_classes - 1)
    axes[1].set_title("ground truth")
    axes[2].imshow(prediction.cpu().numpy(), vmin=0, vmax=num_classes - 1)
    axes[2].set_title(prediction_title)
    confidence_image = axes[3].imshow(
        confidence.cpu().numpy(), vmin=0.0, vmax=1.0
    )
    axes[3].set_title("max probability")
    for axis in axes:
        axis.axis("off")
    fig.colorbar(confidence_image, ax=axes[3], fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def save_raw_mask(mask: torch.Tensor, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    array = mask.detach().cpu().numpy().astype(np.uint8)
    Image.fromarray(array).save(save_path)
