"""
Unified Loss & Metrics for Background Removal / Image Matting.
Combines Binary Cross Entropy (BCE) with Soft Dice Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice Loss on continuous sigmoid probabilities."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)

        intersection = (probs * targets).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + targets.sum(dim=dims)

        dice = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        return (1.0 - dice).mean()


class BCEDiceLoss(nn.Module):
    """Weighted BCE + Soft Dice Loss."""

    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = self.dice(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


@torch.no_grad()
def dice_score_tensor(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    """Computes binary Dice coefficient metric (F1-score) as a tensor."""
    preds = (torch.sigmoid(logits) >= threshold).float()
    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    total = preds.sum(dim=dims) + targets.sum(dim=dims)
    return ((2.0 * intersection + eps) / (total + eps)).mean()


@torch.no_grad()
def iou_score_tensor(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    """Computes Intersection over Union (Jaccard Index) as a tensor."""
    preds = (torch.sigmoid(logits) >= threshold).float()
    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims) - intersection
    return ((intersection + eps) / (union + eps)).mean()


@torch.no_grad()
def dice_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """Computes binary Dice coefficient metric (F1-score)."""
    return dice_score_tensor(logits, targets, threshold=threshold, eps=eps).item()


@torch.no_grad()
def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """Computes Intersection over Union (Jaccard Index)."""
    return iou_score_tensor(logits, targets, threshold=threshold, eps=eps).item()


if __name__ == "__main__":
    targets = (torch.rand(4, 1, 64, 64) > 0.5).float()
    logits = torch.randn(4, 1, 64, 64)
    loss_fn = BCEDiceLoss()
    loss = loss_fn(logits, targets)
    dice = dice_score(logits, targets)
    iou = iou_score(logits, targets)
    print(f"Loss: {loss.item():.4f} | Dice: {dice:.4f} | IoU: {iou:.4f}")
