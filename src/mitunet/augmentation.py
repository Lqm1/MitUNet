"""Albumentations pipelines for training, validation and inference."""

from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from mitunet.config import AugmentationConfig


def build_training_transforms(config: AugmentationConfig) -> A.Compose:
    """Geometric + photometric training pipeline at the configured size."""
    lower_scale, upper_scale = config.training_scale_range
    rotation = config.training_rotation_degrees
    shift = config.training_shift_limit
    return A.Compose(
        [
            A.Affine(
                scale=(lower_scale, upper_scale),
                translate_percent=(-shift, shift),
                rotate=(-rotation, rotation),
                p=0.7,
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.Perspective(p=0.3),
            A.OneOf(
                [
                    A.ElasticTransform(p=1.0, alpha=120, sigma=120 * 0.05),
                    A.GridDistortion(p=1.0),
                ],
                p=0.2,
            ),
            A.RandomBrightnessContrast(p=0.5),
            A.CLAHE(p=0.2),
            A.OneOf(
                [
                    A.GaussNoise(p=1.0),
                    A.ISONoise(p=1.0),
                ],
                p=0.3,
            ),
            A.Resize(config.image_size, config.image_size),
            A.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
            ToTensorV2(),
        ]
    )


def build_validation_transforms(config: AugmentationConfig) -> A.Compose:
    """Deterministic resize + normalization for validation and testing."""
    return A.Compose(
        [
            A.Resize(config.image_size, config.image_size),
            A.Normalize(mean=config.imagenet_mean, std=config.imagenet_std),
            ToTensorV2(),
        ]
    )


def build_inference_transforms(config: AugmentationConfig) -> A.Compose:
    """Inference preprocessing without any stochastic augmentation."""
    return build_validation_transforms(config)
