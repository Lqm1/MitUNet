"""Training and validation loops with checkpointing and benchmarking."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from mitunet.objectives import loss_display_name
from mitunet.scores import (
    SegmentationScores,
    binarize_logits,
    boundary_iou_tensor,
    summarize_scores,
)
from mitunet.tensorboard import create_writer, should_log_images

# Must stay in sync with AugmentationConfig defaults (used to unnormalize
# samples for TensorBoard visualization).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class EpochHistory:
    """Per-epoch training curves."""

    train_loss: list[float] = field(default_factory=list)
    train_iou: list[float] = field(default_factory=list)
    valid_loss: list[float] = field(default_factory=list)
    valid_iou: list[float] = field(default_factory=list)
    valid_boundary_iou: list[float] = field(default_factory=list)
    valid_fps: list[float] = field(default_factory=list)
    valid_vram_mb: list[float] = field(default_factory=list)


def _accumulate_confusion(
    logits: torch.Tensor, masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    predictions = binarize_logits(logits)
    true_positive, false_positive, false_negative, true_negative = smp.metrics.get_stats(
        predictions.long(), masks.long(), mode="binary"
    )
    return (
        true_positive.sum().detach(),
        false_positive.sum().detach(),
        false_negative.sum().detach(),
        true_negative.sum().detach(),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run a single training epoch and return (loss, IoU)."""
    model.train()
    running_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_tp = torch.zeros((), dtype=torch.int64, device=device)
    total_fp = torch.zeros((), dtype=torch.int64, device=device)
    total_fn = torch.zeros((), dtype=torch.int64, device=device)
    total_tn = torch.zeros((), dtype=torch.int64, device=device)
    progress = tqdm(loader, desc="Training", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).unsqueeze(1).float()
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().double() * images.size(0)
        tp, fp, fn, tn = _accumulate_confusion(outputs, masks)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    epoch_iou = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn)
    return running_loss.item() / len(loader.dataset), float(epoch_iou.item())  # type: ignore[arg-type]


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
    max_images: int = 4,
) -> tuple[float, SegmentationScores, float, float]:
    """Run validation and return (loss, scores, fps, peak VRAM in MiB)."""
    model.eval()
    running_loss = torch.zeros((), dtype=torch.float64, device=device)
    total_tp = torch.zeros((), dtype=torch.int64, device=device)
    total_fp = torch.zeros((), dtype=torch.int64, device=device)
    total_fn = torch.zeros((), dtype=torch.int64, device=device)
    total_tn = torch.zeros((), dtype=torch.int64, device=device)
    boundary_sum = torch.zeros((), dtype=torch.float64, device=device)
    batch_count = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    progress = tqdm(loader, desc="Validation", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).unsqueeze(1).float()
        outputs = model(images)
        if samples is not None and not samples:
            samples.append(
                (
                    images[:max_images].detach().cpu().clone(),
                    masks[:max_images].detach().cpu().clone(),
                    outputs[:max_images].sigmoid().detach().cpu().clone(),
                )
            )
        loss = loss_fn(outputs, masks)
        running_loss += loss.detach().double() * images.size(0)
        predictions = binarize_logits(outputs)
        tp, fp, fn, tn = smp.metrics.get_stats(predictions.long(), masks.long(), mode="binary")
        total_tp += tp.sum().detach()
        total_fp += fp.sum().detach()
        total_fn += fn.sum().detach()
        total_tn += tn.sum().detach()
        boundary_sum += boundary_iou_tensor(predictions, masks).double()
        batch_count += 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - start, 1e-6)
    dataset_size = len(loader.dataset)  # type: ignore[arg-type]
    fps = dataset_size / elapsed
    peak_vram = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    )
    mean_boundary = boundary_sum.item() / max(batch_count, 1)
    scores = summarize_scores(total_tp, total_fp, total_fn, total_tn, mean_boundary)
    return running_loss.item() / dataset_size, scores, fps, peak_vram


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> SegmentationScores:
    """Compute full-dataset metrics without a loss function."""
    model.eval()
    total_tp = torch.zeros((), dtype=torch.int64, device=device)
    total_fp = torch.zeros((), dtype=torch.int64, device=device)
    total_fn = torch.zeros((), dtype=torch.int64, device=device)
    total_tn = torch.zeros((), dtype=torch.int64, device=device)
    boundary_sum = torch.zeros((), dtype=torch.float64, device=device)
    batch_count = 0
    for images, masks in tqdm(loader, desc="Metrics", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).unsqueeze(1).float()
        outputs = model(images)
        predictions = (torch.sigmoid(outputs) > threshold).float()
        tp, fp, fn, tn = smp.metrics.get_stats(predictions.long(), masks.long(), mode="binary")
        total_tp += tp.sum().detach()
        total_fp += fp.sum().detach()
        total_fn += fn.sum().detach()
        total_tn += tn.sum().detach()
        boundary_sum += boundary_iou_tensor(predictions, masks).double()
        batch_count += 1
    return summarize_scores(
        total_tp, total_fp, total_fn, total_tn, boundary_sum.item() / max(batch_count, 1)
    )


@torch.no_grad()
def benchmark_throughput(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Measure inference FPS and peak VRAM over one validation pass."""
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    frames = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        _ = model(images)
        frames += images.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - start, 1e-6)
    peak_vram = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    )
    return frames / elapsed, peak_vram


def _unnormalize_for_display(
    images: torch.Tensor,
    image_mean: tuple[float, float, float] = _IMAGENET_MEAN,
    image_std: tuple[float, float, float] = _IMAGENET_STD,
) -> torch.Tensor:
    """Revert ImageNet normalization so logged samples look natural."""
    mean = torch.tensor(image_mean, device=images.device).view(1, -1, 1, 1)
    std = torch.tensor(image_std, device=images.device).view(1, -1, 1, 1)
    return (images * std + mean).clamp(0.0, 1.0)


@torch.no_grad()
def _log_sample_images(
    writer: SummaryWriter,
    tag_prefix: str,
    sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    global_step: int,
    image_mean: tuple[float, float, float] = _IMAGENET_MEAN,
    image_std: tuple[float, float, float] = _IMAGENET_STD,
) -> None:
    """Log one batch of input / target / predicted-probability images."""
    images, masks, probabilities = sample
    display = _unnormalize_for_display(images.float(), image_mean, image_std)
    target = masks.float()
    if target.dim() == 3:
        target = target.unsqueeze(1)
    writer.add_images(f"{tag_prefix}/input", display, global_step)
    writer.add_images(f"{tag_prefix}/target", target, global_step)
    writer.add_images(f"{tag_prefix}/pred_proba", probabilities, global_step)


def fit_wall_segmenter(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    device: torch.device,
    max_epochs: int,
    checkpoint_dir: str | Path,
    model_tag: str = "mitunet",
    tensorboard_dir: str | Path | None = None,
    log_images_every_n_epochs: int = 5,
    max_log_images: int = 4,
    tensorboard_run_name: str = "mitunet",
    tensorboard_flush_secs: int = 30,
    training_metadata: dict | None = None,
    image_mean: tuple[float, float, float] = _IMAGENET_MEAN,
    image_std: tuple[float, float, float] = _IMAGENET_STD,
) -> tuple[Path, EpochHistory, float]:
    """Train until ``max_epochs`` and keep the best-IoU checkpoint.

    TensorBoard logging follows https://docs.pytorch.org/docs/stable/tensorboard.html:
    ``SummaryWriter`` writes event files under ``tensorboard_dir`` and scalars are
    grouped hierarchically (``Loss/train``, ``Metrics/valid_iou``, ...). Visualize with::

        tensorboard --logdir=<checkpoint_dir>/tensorboard

    Notes on coverage:

    * Loss breakdown: the loss factory (``mitunet.objectives``) returns a single
      SMP scalar loss (dice/focal/lovasz/tversky), so there are no sub-components
      to decompose. What is logged is the train/valid split plus the full metric
      breakdown (IoU, boundary IoU, recall, precision, accuracy).
    * Learning rate: every optimizer param group is logged (``Optimizer/lr`` for a
      single group, ``Optimizer/lr_group{i}`` otherwise).
    * Images: ``Samples/valid/{input,target,pred_proba}`` via
      ``add_images`` every ``log_images_every_n_epochs`` epochs (and epoch 1).

    Pass ``tensorboard_dir=None`` to disable logging. The ``train`` CLI resolves
    its default (``None``) to ``<checkpoint_dir>/tensorboard`` unless
    ``--no-tensorboard`` is given.
    """
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    loss_tag = loss_display_name(loss_fn)
    history = EpochHistory()
    best_iou = 0.0
    best_path = checkpoint_root / f"{model_tag}_{loss_tag}_best.pth"
    should_log_images(1, log_images_every_n_epochs, max_log_images)
    writer = create_writer(tensorboard_dir, tensorboard_run_name, tensorboard_flush_secs)
    try:
        if writer is not None:
            writer.add_text(
                "Config/training",
                json.dumps(
                    training_metadata
                    or {
                        "model": model_tag,
                        "loss": loss_tag,
                        "max_epochs": max_epochs,
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    indent=2,
                ),
                0,
            )
        for epoch in range(1, max_epochs + 1):
            samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] | None = (
                []
                if writer is not None
                and should_log_images(epoch, log_images_every_n_epochs, max_log_images)
                else None
            )
            train_loss, train_iou = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
            valid_loss, valid_scores, fps, vram = validate_one_epoch(
                model, valid_loader, loss_fn, device, samples=samples, max_images=max_log_images
            )
            scheduler.step(valid_scores.iou)
            history.train_loss.append(train_loss)
            history.train_iou.append(train_iou)
            history.valid_loss.append(valid_loss)
            history.valid_iou.append(valid_scores.iou)
            history.valid_boundary_iou.append(valid_scores.boundary_iou)
            history.valid_fps.append(fps)
            history.valid_vram_mb.append(vram)
            print(
                f"Epoch {epoch}/{max_epochs} "
                f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} "
                f"valid_loss={valid_loss:.4f} valid_iou={valid_scores.iou:.4f} "
                f"boundary_iou={valid_scores.boundary_iou:.4f} fps={fps:.1f} vram={vram:.0f}MiB"
            )
            if writer is not None:
                writer.add_scalar("Loss/train", train_loss, epoch)
                writer.add_scalar("Loss/valid", valid_loss, epoch)
                writer.add_scalar("Metrics/train_iou", train_iou, epoch)
                writer.add_scalar("Metrics/valid_iou", valid_scores.iou, epoch)
                writer.add_scalar("Metrics/valid_boundary_iou", valid_scores.boundary_iou, epoch)
                writer.add_scalar("Metrics/valid_recall", valid_scores.recall, epoch)
                writer.add_scalar("Metrics/valid_precision", valid_scores.precision, epoch)
                writer.add_scalar("Metrics/valid_accuracy", valid_scores.accuracy, epoch)
                param_groups = optimizer.param_groups
                for group_index, group in enumerate(param_groups):
                    tag = (
                        "Optimizer/lr"
                        if len(param_groups) == 1
                        else f"Optimizer/lr_group{group_index}"
                    )
                    writer.add_scalar(tag, float(group["lr"]), epoch)
                writer.add_scalar("Sys/fps", fps, epoch)
                writer.add_scalar("Sys/vram_mb", vram, epoch)
                if samples:
                    _log_sample_images(
                        writer,
                        "Samples/valid",
                        samples[0],
                        epoch,
                        image_mean=image_mean,
                        image_std=image_std,
                    )
                writer.flush()
            if valid_scores.iou > best_iou:
                best_iou = valid_scores.iou
                if epoch > 1:
                    scored_name = (
                        f"{model_tag}_{loss_tag}_{valid_scores.iou:.4f}".replace("0.", "")
                        + f"_{epoch}E.pth"
                    )
                    best_path = checkpoint_root / scored_name
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "valid_iou": valid_scores.iou,
                            "valid_boundary_iou": valid_scores.boundary_iou,
                            "epoch": epoch,
                        },
                        best_path,
                    )
                    print(f"Saved new best checkpoint: {best_path} (IoU={best_iou:.4f})")
    finally:
        if writer is not None:
            writer.close()
    if not best_path.is_file():
        torch.save(
            {"model_state": model.state_dict(), "valid_iou": best_iou, "epoch": max_epochs},
            best_path,
        )
    return best_path, history, best_iou
