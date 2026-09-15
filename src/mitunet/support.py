"""Shared helpers: seeding, devices, image I/O and checkpoints."""

from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch


def set_random_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``/``cpu``/``cuda`` to a concrete torch device."""
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized in {"cpu", "cuda"}:
        if normalized == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but torch.cuda.is_available() is False")
        return torch.device(normalized)
    raise ValueError(f"Unknown device '{requested}' (expected auto, cpu or cuda)")


def load_rgb_image(path: str | Path) -> np.ndarray:
    """Load an image from disk as an RGB uint8 array."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_binary_mask(mask: np.ndarray, path: str | Path) -> None:
    """Save a binary {0, 1} mask as a grayscale PNG."""
    output = (np.asarray(mask) > 0.5).astype(np.uint8) * 255
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), output)


def save_overlay(image_rgb: np.ndarray, mask: np.ndarray, path: str | Path) -> None:
    """Save a red overlay of the predicted wall mask on the input image."""
    binary = (np.asarray(mask) > 0.5).astype(np.uint8)
    overlay = np.asarray(image_rgb).copy()
    overlay[binary > 0] = (overlay[binary > 0] * 0.5 + np.array([255, 60, 60]) * 0.5).astype(
        np.uint8
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def extract_state_dict(payload: object) -> dict:
    """Unwrap a checkpoint payload to a plain state dict."""
    if (
        isinstance(payload, dict)
        and "model_state" in payload
        and isinstance(payload["model_state"], dict)
    ):
        return dict(payload["model_state"])
    if isinstance(payload, dict) and all(isinstance(key, str) for key in payload):
        # Heuristic: plain state dicts hold tensors, training checkpoints hold metadata.
        values = list(payload.values())
        if values and all(torch.is_tensor(value) for value in values):
            return dict(payload)
        # Some legacy files store {"model_state": ...} or bare state dicts only.
        candidate_keys = [key for key in payload if "state" in key.lower()]
        for key in candidate_keys:
            nested = payload[key]
            if isinstance(nested, dict):
                return dict(nested)
    if isinstance(payload, dict):
        return dict(payload)
    raise ValueError(f"Unsupported checkpoint payload type: {type(payload)!r}")


def load_checkpoint_state(path: str | Path, device: torch.device) -> dict:
    """Load a checkpoint file and return its state dict on the given device."""
    payload = torch.load(str(path), map_location=device, weights_only=False)
    return extract_state_dict(payload)
