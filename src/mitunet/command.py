"""Typer-based command line for training, inference and export."""

from __future__ import annotations

from pathlib import Path

import torch
import typer
from torch.utils.data import DataLoader, random_split

from mitunet.architecture import build_wall_segmenter, count_trainable_parameters
from mitunet.augmentation import build_training_transforms, build_validation_transforms
from mitunet.config import AugmentationConfig, InferenceConfig, ModelConfig, TrainingConfig
from mitunet.engine import evaluate_dataset, fit_wall_segmenter
from mitunet.objectives import SUPPORTED_LOSSES, build_loss
from mitunet.predictor import WallPredictor, export_wall_segmenter_onnx
from mitunet.support import (
    load_rgb_image,
    resolve_device,
    save_binary_mask,
    save_overlay,
    set_random_seed,
)
from mitunet.wall_dataset import (
    CocoWallDataset,
    MaskFolderDataset,
    TransformedView,
    discover_coco_splits,
    seed_worker,
)

app = typer.Typer(no_args_is_help=True)
export_app = typer.Typer(no_args_is_help=True, help="Model export commands.")
app.add_typer(export_app, name="export")


def _build_coco_loaders(
    dataset_root: Path,
    training: TrainingConfig,
    device: torch.device,  # noqa: ARG001 - kept for symmetric CLI plumbing
) -> tuple[DataLoader, DataLoader]:
    splits = discover_coco_splits(dataset_root)
    if "train" not in splits:
        raise typer.BadParameter(f"No COCO split found under {dataset_root}")
    train_transforms = build_training_transforms(training.augmentation)
    valid_transforms = build_validation_transforms(training.augmentation)
    if "valid" in splits:
        train_dir, train_ann = splits["train"]
        valid_dir, valid_ann = splits["valid"]
        train_base = CocoWallDataset(
            train_dir,
            train_ann,
            transforms=None,
            opening_thickness_px=training.augmentation.opening_thickness_px,
            closing_kernel_size=training.augmentation.closing_kernel_size,
        )
        valid_base = CocoWallDataset(
            valid_dir,
            valid_ann,
            transforms=None,
            opening_thickness_px=training.augmentation.opening_thickness_px,
            closing_kernel_size=training.augmentation.closing_kernel_size,
        )
        train_dataset: torch.utils.data.Dataset = TransformedView(train_base, train_transforms)
        valid_dataset: torch.utils.data.Dataset = TransformedView(valid_base, valid_transforms)
    else:
        # Single COCO annotation: deterministic 80/20 split like the paper notebook.
        single_dir, single_ann = splits["train"]
        combined = CocoWallDataset(
            single_dir,
            single_ann,
            transforms=None,
            opening_thickness_px=training.augmentation.opening_thickness_px,
            closing_kernel_size=training.augmentation.closing_kernel_size,
        )
        generator = torch.Generator().manual_seed(training.seed)
        train_size = int((1.0 - training.validation_split) * len(combined))
        train_subset, valid_subset = random_split(
            combined, [train_size, len(combined) - train_size], generator=generator
        )
        train_dataset = TransformedView(train_subset, train_transforms)
        valid_dataset = TransformedView(valid_subset, valid_transforms)
    generator_seed = torch.Generator().manual_seed(training.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=training.batch_size,
        shuffle=True,
        num_workers=training.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator_seed,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=training.batch_size,
        shuffle=False,
        num_workers=training.num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
    )
    return train_loader, valid_loader


@app.command()
def train(
    dataset_root: Path = typer.Argument(..., help="COCO dataset root (train/valid/test)"),
    checkpoint_dir: Path = typer.Option(Path("checkpoints"), help="Where to store checkpoints"),
    epochs: int = typer.Option(30, help="Maximum training epochs"),
    batch_size: int = typer.Option(4, help="Batch size"),
    learning_rate: float = typer.Option(1e-4, help="Adam learning rate (base training)"),
    fine_tune: Path | None = typer.Option(None, help="Checkpoint to fine-tune from"),
    fine_tune_lr: float = typer.Option(1e-5, help="Adam learning rate for fine-tuning"),
    loss: str = typer.Option("tversky", help=f"Loss name {SUPPORTED_LOSSES}"),
    tversky_alpha: float = typer.Option(0.6, help="Tversky false-negative weight"),
    tversky_beta: float = typer.Option(0.4, help="Tversky false-positive weight"),
    image_size: int = typer.Option(512, help="Square training resolution"),
    encoder: str = typer.Option("mit_b4", help="SMP encoder name"),
    seed: int = typer.Option(42, help="Random seed"),
    num_workers: int = typer.Option(2, help="DataLoader workers (use 0 on Windows CPU)"),
    device: str = typer.Option("auto", help="Compute device (auto, cpu or cuda)"),
) -> None:
    """Train the hybrid wall segmenter on a COCO floor-plan dataset."""
    active_device = resolve_device(device)
    if active_device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    set_random_seed(seed)
    augmentation = AugmentationConfig(image_size=image_size)
    training = TrainingConfig(
        max_epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        fine_tune_learning_rate=fine_tune_lr,
        seed=seed,
        checkpoint_dir=str(checkpoint_dir),
        loss_name=loss,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        num_workers=num_workers,
        model=ModelConfig(encoder_name=encoder),
        augmentation=augmentation,
    )
    train_loader, valid_loader = _build_coco_loaders(dataset_root, training, active_device)
    typer.echo(f"train_batches={len(train_loader)} valid_batches={len(valid_loader)}")
    model = build_wall_segmenter(training.model)
    if fine_tune is not None:
        from mitunet.architecture import load_wall_segmenter_weights

        load_wall_segmenter_weights(model, fine_tune, active_device, strict=False)
        effective_lr = training.fine_tune_learning_rate
        typer.echo(f"Fine-tuning from {fine_tune} (lr={effective_lr})")
    else:
        effective_lr = training.learning_rate
    model.to(active_device)
    typer.echo(f"parameters={count_trainable_parameters(model)} device={active_device.type}")
    loss_fn = build_loss(loss, tversky_alpha=tversky_alpha, tversky_beta=tversky_beta)
    optimizer = torch.optim.Adam(model.parameters(), lr=effective_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=training.scheduler_factor,
        patience=training.scheduler_patience,
    )
    best_path, _, best_iou = fit_wall_segmenter(
        model,
        train_loader,
        valid_loader,
        loss_fn,
        optimizer,
        scheduler,
        active_device,
        max_epochs=training.max_epochs,
        checkpoint_dir=training.checkpoint_dir,
    )
    typer.echo(f"Best checkpoint: {best_path} (valid_iou={best_iou:.4f})")


@app.command()
def infer(
    checkpoint: Path = typer.Argument(..., help="Wall-segmentation checkpoint"),
    image: Path = typer.Argument(..., help="Input floor-plan image"),
    output_dir: Path = typer.Option(Path("outputs"), help="Where to write predictions"),
    image_size: int = typer.Option(512, help="Model input resolution"),
    threshold: float = typer.Option(0.5, help="Sigmoid threshold"),
    encoder: str = typer.Option("mit_b4", help="SMP encoder name"),
    device: str = typer.Option("auto", help="Compute device (auto, cpu or cuda)"),
) -> None:
    """Run wall segmentation on a single image."""
    active_device = resolve_device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    inference = InferenceConfig(image_size=image_size, threshold=threshold, device=device)
    predictor = WallPredictor.from_checkpoint(
        checkpoint,
        active_device,
        model_config=ModelConfig(encoder_name=encoder),
        inference=inference,
        preprocessing=AugmentationConfig(image_size=image_size),
    )
    rgb = load_rgb_image(image)
    mask = predictor.predict_mask(rgb)
    proba = predictor.predict_proba(rgb)
    save_binary_mask(mask, output_dir / f"{image.stem}_wall.png")
    save_overlay(rgb, mask, output_dir / f"{image.stem}_overlay.png")
    typer.echo(
        f"wrote {image.stem}_wall.png wall_ratio={float(mask.mean()):.4f} "
        f"proba_mean={float(proba.mean()):.4f}"
    )


@app.command(name="evaluate")
def evaluate(
    checkpoint: Path = typer.Argument(..., help="Wall-segmentation checkpoint"),
    dataset_root: Path = typer.Argument(..., help="COCO dataset root or image/mask folders"),
    image_size: int = typer.Option(512, help="Model input resolution"),
    batch_size: int = typer.Option(4, help="Batch size"),
    num_workers: int = typer.Option(2, help="DataLoader workers"),
    encoder: str = typer.Option("mit_b4", help="SMP encoder name"),
    device: str = typer.Option("auto", help="Compute device (auto, cpu or cuda)"),
) -> None:
    """Evaluate a checkpoint with IoU, boundary IoU, recall, precision and accuracy."""
    active_device = resolve_device(device)
    augmentation = AugmentationConfig(image_size=image_size)
    valid_transforms = build_validation_transforms(augmentation)
    splits = discover_coco_splits(dataset_root)
    if splits:
        key = "valid" if "valid" in splits else "train" if "train" in splits else next(iter(splits))
        image_dir, annotation = splits[key]
        dataset: torch.utils.data.Dataset = CocoWallDataset(image_dir, annotation, transforms=None)
        dataset = TransformedView(dataset, valid_transforms)
        typer.echo(f"Evaluating COCO split '{key}' with {len(dataset)} samples")
    else:
        # Fall back to image/mask folder layout: <root>/images + <root>/masks.
        image_dir = dataset_root / "images" if (dataset_root / "images").is_dir() else dataset_root
        mask_dir = (
            dataset_root / "masks" if (dataset_root / "masks").is_dir() else dataset_root / "labels"
        )
        folder = MaskFolderDataset(image_dir, mask_dir, transforms=None)
        dataset = TransformedView(folder, valid_transforms)
        typer.echo(f"Evaluating folder pairs with {len(dataset)} samples")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
    )
    predictor = WallPredictor.from_checkpoint(
        checkpoint,
        active_device,
        model_config=ModelConfig(encoder_name=encoder),
        preprocessing=augmentation,
    )
    scores = evaluate_dataset(predictor.model, loader, active_device)
    typer.echo(
        f"iou={scores.iou:.4f} boundary_iou={scores.boundary_iou:.4f} "
        f"recall={scores.recall:.4f} precision={scores.precision:.4f} accuracy={scores.accuracy:.4f}"
    )


@export_app.command("onnx")
def export_onnx(
    checkpoint: Path = typer.Argument(..., help="Wall-segmentation checkpoint"),
    output: Path = typer.Argument(..., help="Destination .onnx path"),
    image_size: int = typer.Option(512, help="Square input resolution"),
    encoder: str = typer.Option("mit_b4", help="SMP encoder name"),
    opset: int = typer.Option(18, help="ONNX opset version"),
) -> None:
    """Export a checkpoint to ONNX and verify Torch parity."""
    max_diff = export_wall_segmenter_onnx(
        checkpoint,
        output,
        model_config=ModelConfig(encoder_name=encoder),
        image_size=image_size,
        opset=opset,
    )
    typer.echo(f"exported {output} (opset={opset}) max_abs_diff={max_diff:.2e}")
    if max_diff > 1e-3:
        raise typer.BadParameter(f"ONNX parity check failed: diff={max_diff:.2e}")


@app.command()
def smoke(
    device: str = typer.Option("auto", help="Compute device (auto, cpu or cuda)"),
    image_size: int = typer.Option(512, help="Square input resolution"),
    encoder: str = typer.Option("mit_b4", help="SMP encoder name"),
) -> None:
    """Quick forward pass to verify installation (ImageNet encoder, no checkpoint)."""
    active_device = resolve_device(device)
    model = build_wall_segmenter(ModelConfig(encoder_name=encoder)).to(active_device).eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=active_device)
    with torch.no_grad():
        logits = model(dummy)
    typer.echo(
        f"device={active_device.type} cuda_available={torch.cuda.is_available()} "
        f"logits={tuple(logits.shape)} expected=(1, 1, {image_size}, {image_size})"
    )


if __name__ == "__main__":
    app()
