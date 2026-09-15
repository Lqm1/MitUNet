"""Standalone ONNX inference sample (NumPy + onnxruntime only)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def preprocess(image_path: Path, image_size: int = 512) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    return np.transpose(image, (2, 0, 1))[None].astype(np.float32)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="MitUNet ONNX sample")
    parser.add_argument("model", type=Path)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("onnx_mask.png"))
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    tensor = preprocess(args.image, args.size)
    (logits,) = session.run(["logits"], {"image": tensor})
    mask = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(np.uint8)[0, 0] * 255
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), mask)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
