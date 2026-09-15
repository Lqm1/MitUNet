"""Segmentation scores including a max-pool boundary IoU."""

from __future__ import annotations

from dataclasses import dataclass

import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class SegmentationScores:
    """Aggregated binary segmentation metrics."""

    iou: float
    boundary_iou: float
    recall: float
    precision: float
    accuracy: float


def boundary_iou(
    predicted: torch.Tensor,
    target: torch.Tensor,
    dilation_ratio: float = 0.02,
) -> float:
    """Boundary IoU via morphological erosion with max-pooling.

    Both inputs are binarized at 0.5. Boundaries are ``mask - eroded(mask)``
    where erosion is implemented as ``1 - max_pool(1 - mask)``. The dilation
    radius scales with the image diagonal so the metric stays resolution
    invariant.
    """
    with torch.no_grad():
        predictions = (predicted.detach() > 0.5).float()
        references = (target.detach() > 0.5).float()
        if predictions.dim() == 3:
            predictions = predictions.unsqueeze(1)
            references = references.unsqueeze(1)
        _, _, height, width = predictions.shape
        diagonal = float((height**2 + width**2) ** 0.5)
        radius = max(1, int(round(diagonal * dilation_ratio)))
        kernel = 2 * radius + 1
        predicted_eroded = 1.0 - functional.max_pool2d(
            1.0 - predictions, kernel_size=kernel, stride=1, padding=radius
        )
        target_eroded = 1.0 - functional.max_pool2d(
            1.0 - references, kernel_size=kernel, stride=1, padding=radius
        )
        predicted_edge = (predictions - predicted_eroded).flatten(1)
        target_edge = (references - target_eroded).flatten(1)
        intersection = (predicted_edge * target_edge).sum(dim=1)
        union = (predicted_edge + target_edge).clamp(0, 1).sum(dim=1)
        return float((intersection / (union + 1e-7)).mean().item())


def summarize_scores(
    true_positive: torch.Tensor,
    false_positive: torch.Tensor,
    false_negative: torch.Tensor,
    true_negative: torch.Tensor,
    boundary_value: float,
) -> SegmentationScores:
    """Convert accumulated confusion counts to a score bundle."""
    iou = smp.metrics.iou_score(
        true_positive, false_positive, false_negative, true_negative, reduction="micro"
    )
    recall = smp.metrics.recall(
        true_positive, false_positive, false_negative, true_negative, reduction="micro"
    )
    precision = smp.metrics.precision(
        true_positive, false_positive, false_negative, true_negative, reduction="micro"
    )
    accuracy = smp.metrics.accuracy(
        true_positive, false_positive, false_negative, true_negative, reduction="micro"
    )
    return SegmentationScores(
        iou=float(iou.item()),
        boundary_iou=float(boundary_value),
        recall=float(recall.item()),
        precision=float(precision.item()),
        accuracy=float(accuracy.item()),
    )


def binarize_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Apply sigmoid + threshold to raw model logits."""
    return (torch.sigmoid(logits) > threshold).float()
