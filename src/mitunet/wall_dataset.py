"""COCO wall-segmentation datasets with opening subtraction."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset

from mitunet.target_cache import TargetCache, file_version

TransformFn = Callable[..., dict]


class CocoWallDataset(Dataset[tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]]):
    """Binary wall masks derived from COCO polygon annotations.

    Wall polygons form the foreground. Door and window polygons are
    thickened along their minor axis and subtracted so the network learns
    continuous wall segments without openings. A morphological closing pass
    removes hairline gaps left by the subtraction.
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotation_path: str | Path,
        transforms: TransformFn | None = None,
        opening_thickness_px: int = 30,
        closing_kernel_size: int = 5,
        target_cache_mb: int = 256,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.transforms = transforms
        self.opening_thickness_px = int(opening_thickness_px)
        self.closing_kernel_size = int(closing_kernel_size)
        self.coco = COCO(str(annotation_path))
        self.annotation_path = Path(annotation_path)
        self.annotation_version = file_version(annotation_path)
        self.target_cache = TargetCache(target_cache_mb)
        self.image_ids: list[int] = self.coco.getImgIds()
        if not self.image_ids:
            raise ValueError(f"No images found in {annotation_path}")
        self.wall_category_ids = set(self.coco.getCatIds(catNms=["wall"]))
        self.door_category_ids = set(self.coco.getCatIds(catNms=["door"]))
        self.window_category_ids = set(self.coco.getCatIds(catNms=["window"]))
        self.opening_category_ids = self.door_category_ids | self.window_category_ids

    def __len__(self) -> int:
        return len(self.image_ids)

    def build_opening_mask(self, annotations: list[dict], height: int, width: int) -> np.ndarray:
        """Rasterize thickened door/window boxes as an exclusion mask."""
        exclusion = np.zeros((height, width), dtype=np.uint8)
        for annotation in annotations:
            if annotation["category_id"] not in self.opening_category_ids:
                continue
            instance_mask = (self.coco.annToMask(annotation) > 0).astype(np.uint8) * 255
            _, binary = cv2.threshold(instance_mask, 0, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            center, size, angle = cv2.minAreaRect(contours[0])
            short_side, long_side = float(size[0]), float(size[1])
            padding = float(self.opening_thickness_px)
            if short_side <= long_side:
                expanded = (short_side + padding, long_side)
            else:
                expanded = (short_side, long_side + padding)
            box = np.intp(cv2.boxPoints((center, expanded, angle)))  # type: ignore[call-overload]
            cv2.drawContours(exclusion, [box], 0, 255, -1)  # type: ignore[list-item, arg-type]
        return exclusion

    def build_wall_mask(self, annotations: list[dict], height: int, width: int) -> np.ndarray:
        """Combine wall polygons, subtract openings and close small gaps."""
        wall = np.zeros((height, width), dtype=np.uint8)
        for annotation in annotations:
            if annotation["category_id"] in self.wall_category_ids:
                wall = np.maximum(wall, (self.coco.annToMask(annotation) > 0).astype(np.uint8))
        exclusion = self.build_opening_mask(annotations, height, width)
        # Scale to a common 0/255 range before subtraction to avoid depth bugs.
        wall_scaled = (wall * 255).astype(np.uint8)
        cleaned = cv2.subtract(wall_scaled, exclusion)
        kernel = np.ones((self.closing_kernel_size, self.closing_kernel_size), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        return (cleaned > 127).astype(np.uint8)

    def __getitem__(self, index: int) -> tuple:
        if file_version(self.annotation_path) != self.annotation_version:
            raise RuntimeError("COCO annotations changed during use; recreate the dataset")
        image_id = self.image_ids[index]
        info = self.coco.loadImgs(image_id)[0]
        image_path = self.image_dir / info["file_name"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            fallback = (index + 1) % len(self)
            return self.__getitem__(fallback)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = int(info["height"]), int(info["width"])
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        key = (
            image_id,
            self.annotation_version,
            height,
            width,
            self.opening_thickness_px,
            self.closing_kernel_size,
        )
        cached = self.target_cache.get(key)
        if cached is None:
            mask = self.build_wall_mask(annotations, height, width)
            self.target_cache.put(key, (mask,))
        else:
            (mask,) = cached
        if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            return augmented["image"], augmented["mask"]
        return image, mask


class MaskFolderDataset(Dataset[tuple]):
    """Simple image/mask folder pairs for quick experiments.

    Expects ``image_dir`` with RGB images and ``mask_dir`` with matching
    filenames (any extension is matched by stem).
    """

    def __init__(
        self,
        image_dir: str | Path,
        mask_dir: str | Path,
        transforms: TransformFn | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transforms = transforms
        mask_by_stem = {path.stem: path for path in self.mask_dir.iterdir() if path.is_file()}
        self.pairs: list[tuple[Path, Path]] = []
        for image_path in sorted(self.image_dir.iterdir()):
            if not image_path.is_file():
                continue
            mask_path = mask_by_stem.get(image_path.stem)
            if mask_path is not None:
                self.pairs.append((image_path, mask_path))
        if not self.pairs:
            raise ValueError(f"No image/mask pairs under {image_dir} and {mask_dir}")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple:
        image_path, mask_path = self.pairs[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask = (mask > 127).astype(np.uint8)
        if self.transforms is not None:
            augmented = self.transforms(image=image, mask=mask)
            return augmented["image"], augmented["mask"]
        return image, mask


class TransformedView(Dataset[tuple]):
    """Lazily apply Albumentations transforms to another dataset view."""

    def __init__(self, source: Dataset, transforms: TransformFn | None) -> None:
        self.source = source
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.source)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> tuple:
        image, mask = self.source[index]
        if self.transforms is None:
            return image, mask
        augmented = self.transforms(image=np.asarray(image), mask=np.asarray(mask))
        return augmented["image"], augmented["mask"]


def discover_coco_splits(root: str | Path) -> dict[str, tuple[Path, Path]]:
    """Find ``train``/``valid``/``test`` COCO splits below a root directory."""
    root_path = Path(root)
    splits: dict[str, tuple[Path, Path]] = {}
    for name in ("train", "valid", "test", "val"):
        candidate_dir = root_path / name
        annotation = candidate_dir / "_annotations.coco.json"
        if candidate_dir.is_dir() and annotation.is_file():
            key = "valid" if name == "val" else name
            splits[key] = (candidate_dir, annotation)
    if not splits and (root_path / "_annotations.coco.json").is_file():
        splits["train"] = (root_path, root_path / "_annotations.coco.json")
    return splits


def seed_worker(worker_id: int) -> None:
    """Seed DataLoader workers deterministically."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    import random

    random.seed(worker_seed)
    os.environ["PYTHONHASHSEED"] = str(worker_seed)
