"""Segmentation loss factory with type-safe configuration."""

from __future__ import annotations

import segmentation_models_pytorch as smp
from torch import nn

SUPPORTED_LOSSES = ("dice", "focal", "lovasz", "tversky")


def build_loss(
    name: str,
    tversky_alpha: float = 0.6,
    tversky_beta: float = 0.4,
    tversky_gamma: float = 1.0,
) -> nn.Module:
    """Create a binary segmentation loss by name.

    ``tversky_*`` arguments control the asymmetric penalty: ``alpha``
    weights false negatives (recall) and ``beta`` weights false positives
    (precision). The paper optimum is ``alpha=0.6`` / ``beta=0.4``.
    """
    normalized = name.strip().lower()
    if normalized == "dice":
        return smp.losses.DiceLoss(mode="binary")
    if normalized == "focal":
        return smp.losses.FocalLoss(mode="binary")
    if normalized == "lovasz":
        return smp.losses.LovaszLoss(mode="binary")
    if normalized == "tversky":
        if not 0.0 < tversky_alpha < 1.0 or not 0.0 < tversky_beta < 1.0:
            raise ValueError("Tversky alpha and beta must lie in (0, 1)")
        return smp.losses.TverskyLoss(
            mode="binary",
            alpha=float(tversky_alpha),
            beta=float(tversky_beta),
            gamma=float(tversky_gamma),
        )
    raise ValueError(f"Unknown loss '{name}' (expected one of {SUPPORTED_LOSSES})")


def loss_display_name(loss: nn.Module) -> str:
    """Short filesystem-safe name for checkpoint filenames."""
    return type(loss).__name__.lower().replace("loss", "")
