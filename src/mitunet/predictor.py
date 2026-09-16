"""Inference entry points plus ONNX export for the wall segmenter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from mitunet.architecture import build_wall_segmenter, model_from_checkpoint
from mitunet.augmentation import build_inference_transforms
from mitunet.config import AugmentationConfig, InferenceConfig, ModelConfig


class WallPredictor:
    """Thresholded wall-mask predictor wrapping a MitUNet model."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        inference: InferenceConfig | None = None,
        preprocessing: AugmentationConfig | None = None,
    ) -> None:
        self.model = model.eval().to(device)
        self.device = device
        self.inference = inference or InferenceConfig()
        self.preprocessing = preprocessing or AugmentationConfig(
            image_size=self.inference.image_size
        )
        self.transforms = build_inference_transforms(self.preprocessing)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: torch.device,
        model_config: ModelConfig | None = None,
        inference: InferenceConfig | None = None,
        preprocessing: AugmentationConfig | None = None,
    ) -> WallPredictor:
        """Build a predictor from a ``.pth`` wall-segmentation checkpoint."""
        model = model_from_checkpoint(checkpoint_path, device, model_config)
        return cls(model, device, inference=inference, preprocessing=preprocessing)

    @classmethod
    def from_pretrained_weights(
        cls,
        device: torch.device,
        model_config: ModelConfig | None = None,
        inference: InferenceConfig | None = None,
    ) -> WallPredictor:
        """Build an ImageNet-initialized predictor (no wall fine-tuning)."""
        model = build_wall_segmenter(model_config or ModelConfig())
        return cls(model, device, inference=inference)

    @torch.inference_mode()
    def predict_proba(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return a wall probability map in the model's input resolution."""
        augmented = self.transforms(image=np.asarray(image_rgb))
        tensor = augmented["image"].unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy()
        return np.asarray(probs, dtype=np.float32)

    def predict_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return a binary {0, 1} wall mask."""
        proba = self.predict_proba(image_rgb)
        return (proba > self.inference.threshold).astype(np.uint8)


def export_wall_segmenter_onnx(
    checkpoint_path: str | Path,
    output_path: str | Path,
    model_config: ModelConfig | None = None,
    image_size: int = 512,
    opset: int = 18,
) -> float:
    """Export a checkpoint to ONNX and verify numerical parity on CPU.

    Returns the maximum absolute difference between Torch and ONNX outputs.
    """
    import onnx
    import onnxruntime as ort

    device = torch.device("cpu")
    model = model_from_checkpoint(checkpoint_path, device, model_config)
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    batch_dim = torch.export.Dim("batch")
    torch.onnx.export(  # type: ignore[arg-type]
        model,
        (dummy,),
        str(output),
        input_names=["image"],
        output_names=["logits"],
        dynamic_shapes={"x": {0: batch_dim}},
        opset_version=opset,
    )
    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        expected = model(dummy).numpy()
    actual = session.run(["logits"], {"image": dummy.numpy()})[0]
    max_diff = float(np.abs(expected - actual).max())
    return max_diff
