import numpy as np

from mitunet.augmentation import build_inference_transforms, build_training_transforms
from mitunet.config import AugmentationConfig


def test_training_transforms_produce_tensors() -> None:
    config = AugmentationConfig(image_size=128)
    transforms = build_training_transforms(config)
    image = (np.random.rand(160, 160, 3) * 255).astype(np.uint8)
    mask = (np.random.rand(160, 160) > 0.5).astype(np.uint8)
    augmented = transforms(image=image, mask=mask)
    assert augmented["image"].shape == (3, 128, 128)
    assert augmented["mask"].shape == (128, 128)


def test_inference_transforms_deterministic_shape() -> None:
    config = AugmentationConfig(image_size=128)
    transforms = build_inference_transforms(config)
    image = (np.random.rand(200, 120, 3) * 255).astype(np.uint8)
    augmented = transforms(image=image)
    assert augmented["image"].shape == (3, 128, 128)
