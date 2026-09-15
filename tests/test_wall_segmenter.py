import numpy as np
import torch

from mitunet.architecture import build_wall_segmenter
from mitunet.config import ModelConfig
from mitunet.objectives import SUPPORTED_LOSSES, build_loss
from mitunet.scores import boundary_iou


def test_wall_segmenter_forward_shape() -> None:
    model = build_wall_segmenter(ModelConfig(encoder_name="mit_b4", encoder_weights=None)).eval()
    dummy = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        logits = model(dummy)
    assert logits.shape == (1, 1, 128, 128)


def test_loss_factory_supported_names() -> None:
    for name in SUPPORTED_LOSSES:
        loss = build_loss(name)
        dummy = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()
        value = loss(dummy, target)
        assert torch.isfinite(value).all()


def test_boundary_iou_perfect_overlap() -> None:
    prediction = torch.zeros(1, 1, 64, 64)
    prediction[:, :, 16:48, 16:48] = 1.0
    target = prediction.clone()
    assert boundary_iou(prediction, target) == 1.0


def test_boundary_iou_empty_masks() -> None:
    prediction = torch.zeros(1, 1, 32, 32)
    target = torch.zeros(1, 1, 32, 32)
    value = boundary_iou(prediction, target)
    assert np.isfinite(value)
