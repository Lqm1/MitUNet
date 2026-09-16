"""Hybrid MitUNet architecture: MiT-B4 encoder inside a U-Net decoder."""

from __future__ import annotations

from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch import nn

from mitunet.config import ModelConfig
from mitunet.support import load_checkpoint_state


def build_wall_segmenter(config: ModelConfig | None = None) -> nn.Module:
    """Build the hybrid wall segmenter.

    A SegFormer model is instantiated first so its ImageNet-pretrained
    Mix-Transformer encoder can be transplanted into a U-Net with an
    scSE-attended decoder. This mirrors the paper pipeline while keeping a
    single construction entry point for training and inference.
    """
    active = config or ModelConfig()
    encoder_weights = active.encoder_weights
    # The donor SegFormer carries the pretrained transformer encoder.
    donor = smp.Segformer(
        encoder_name=active.encoder_name,
        encoder_weights=encoder_weights,
        in_channels=active.in_channels,
        classes=active.num_classes,
    )
    segmenter = smp.Unet(
        encoder_name=active.encoder_name,
        encoder_weights=None,
        in_channels=active.in_channels,
        classes=active.num_classes,
        decoder_attention_type=active.decoder_attention,
    )
    segmenter.encoder = donor.encoder
    return segmenter


def load_wall_segmenter_weights(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    strict: bool = True,
) -> nn.Module:
    """Load wall-segmentation weights into a MitUNet model."""
    state = load_checkpoint_state(checkpoint_path, device)
    model.load_state_dict(state, strict=strict)
    return model


def count_trainable_parameters(model: nn.Module) -> int:
    """Count trainable parameters for logging."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model_from_checkpoint(
    checkpoint: str | Path, device: torch.device, config: ModelConfig | None = None
) -> nn.Module:
    """Avoid donor models/pretrained downloads for complete checkpoints.

    Partial checkpoints retain the original initialization and permissive loading.
    Probe a single U-Net without downloading pretrained weights. Preserve RNG
    state so a partial-checkpoint fallback uses the legacy initialization.
    """
    active = config or ModelConfig()
    state = load_checkpoint_state(checkpoint, torch.device("cpu"))
    with torch.random.fork_rng(devices=[]):
        model = smp.Unet(
            encoder_name=active.encoder_name,
            encoder_weights=None,
            in_channels=active.in_channels,
            classes=active.num_classes,
            decoder_attention_type=active.decoder_attention,
        )
    expected = model.state_dict()
    if expected.keys() == state.keys() and all(
        expected[key].shape == state[key].shape and expected[key].dtype == state[key].dtype
        for key in expected
    ):
        model.load_state_dict(state, strict=True, assign=True)
        return model.to(device)
    model = build_wall_segmenter(active)
    model.load_state_dict(state, strict=False)
    return model.to(device)
