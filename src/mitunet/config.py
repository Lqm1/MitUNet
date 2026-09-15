"""Typed configuration containers for MitUNet training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """Hybrid encoder-decoder layout.

    The encoder is a Mix-Transformer (MiT-B4) borrowed from SegFormer and
    transplanted into a U-Net decoder with scSE attention.
    """

    encoder_name: str = "mit_b4"
    encoder_weights: str | None = "imagenet"
    in_channels: int = 3
    num_classes: int = 1
    decoder_attention: str = "scse"


@dataclass(frozen=True)
class AugmentationConfig:
    """Preprocessing and augmentation settings."""

    image_size: int = 512
    training_scale_range: tuple[float, float] = (0.9, 1.1)
    training_shift_limit: float = 0.05
    training_rotation_degrees: int = 15
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    opening_thickness_px: int = 30
    closing_kernel_size: int = 5


@dataclass(frozen=True)
class TrainingConfig:
    """Training loop hyperparameters."""

    max_epochs: int = 30
    batch_size: int = 4
    learning_rate: float = 1e-4
    fine_tune_learning_rate: float = 1e-5
    weight_decay: float = 0.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 3
    num_workers: int = 2
    seed: int = 42
    validation_split: float = 0.2
    checkpoint_dir: str = "checkpoints"
    loss_name: str = "tversky"
    tversky_alpha: float = 0.6
    tversky_beta: float = 0.4
    tversky_gamma: float = 1.0
    model: ModelConfig = field(default_factory=ModelConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)


@dataclass(frozen=True)
class InferenceConfig:
    """Inference-time settings."""

    image_size: int = 512
    threshold: float = 0.5
    device: str = "auto"
