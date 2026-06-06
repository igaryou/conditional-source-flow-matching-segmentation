from typing import Dict

import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.confmat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    @torch.no_grad()
    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred = pred.reshape(-1).cpu()
        target = target.reshape(-1).cpu()
        valid = (target >= 0) & (target < self.num_classes)
        pred = pred[valid]
        target = target[valid]
        indices = target * self.num_classes + pred
        bins = torch.bincount(indices, minlength=self.num_classes**2)
        self.confmat += bins.reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, object]:
        conf = self.confmat.float()
        true_positive = conf.diag()
        ground_truth = conf.sum(dim=1)
        predicted = conf.sum(dim=0)
        union = ground_truth + predicted - true_positive

        iou = true_positive / union.clamp_min(1.0)
        class_acc = true_positive / ground_truth.clamp_min(1.0)
        pixel_acc = true_positive.sum() / conf.sum().clamp_min(1.0)

        return {
            "pixel_acc": float(pixel_acc.item()),
            "mIoU": float(iou.mean().item()),
            "mAcc": float(class_acc.mean().item()),
            "IoU_per_class": [float(value.item()) for value in iou],
            "Acc_per_class": [float(value.item()) for value in class_acc],
            "confusion_matrix": conf.to(torch.int64).tolist(),
        }

