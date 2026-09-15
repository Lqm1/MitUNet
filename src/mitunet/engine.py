"""Training and validation loops with checkpointing and benchmarking."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import segmentation_models_pytorch as smp
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from mitunet.objectives import loss_display_name
from mitunet.scores import SegmentationScores, binarize_logits, boundary_iou, summarize_scores


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    predictions = binarize_logits(logits)
    true_positive, false_positive, false_negative, true_negative = smp.metrics.get_stats(
        predictions.long(), masks.long(), mode="binary"
    )
    return (
        true_positive.sum().detach().cpu(),
        false_positive.sum().detach().cpu(),
        false_negative.sum().detach().cpu(),
        true_negative.sum().detach().cpu(),
        predictions.detach().cpu(),
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
    running_loss = 0.0
    total_tp = torch.tensor(0.0)
    total_fp = torch.tensor(0.0)
    total_fn = torch.tensor(0.0)
    total_tn = torch.tensor(0.0)
    progress = tqdm(loader, desc="Training", leave=False)
    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1).float()
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += float(loss.item()) * images.size(0)
        tp, fp, fn, tn, _ = _accumulate_confusion(outputs, masks)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        progress.set_postfix(loss=f"{loss.item():.4f}")
    epoch_iou = smp.metrics.iou_score(total_tp, total_fp, total_fn, total_tn)
    return running_loss / len(loader.dataset), float(epoch_iou.item())  # type: ignore[arg-type]


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, SegmentationScores, float, float]:
    """Run validation and return (loss, scores, fps, peak VRAM in MiB)."""
    model.eval()
    running_loss = 0.0
    total_tp = torch.tensor(0.0)
    total_fp = torch.tensor(0.0)
    total_fn = torch.tensor(0.0)
    total_tn = torch.tensor(0.0)
    boundary_sum = 0.0
    batch_count = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    progress = tqdm(loader, desc="Validation", leave=False)
    for images, masks in progress:
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1).float()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        running_loss += float(loss.item()) * images.size(0)
        predictions = binarize_logits(outputs)
        tp, fp, fn, tn = smp.metrics.get_stats(predictions.long(), masks.long(), mode="binary")
        total_tp += tp.sum().detach().cpu()
        total_fp += fp.sum().detach().cpu()
        total_fn += fn.sum().detach().cpu()
        total_tn += tn.sum().detach().cpu()
        boundary_sum += boundary_iou(predictions, masks)
        batch_count += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - start, 1e-6)
    dataset_size = len(loader.dataset)  # type: ignore[arg-type]
    fps = dataset_size / elapsed
    peak_vram = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    )
    mean_boundary = boundary_sum / max(batch_count, 1)
    scores = summarize_scores(total_tp, total_fp, total_fn, total_tn, mean_boundary)
    return running_loss / dataset_size, scores, fps, peak_vram


@torch.no_grad()
def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> SegmentationScores:
    """Compute full-dataset metrics without a loss function."""
    model.eval()
    total_tp = torch.tensor(0.0)
    total_fp = torch.tensor(0.0)
    total_fn = torch.tensor(0.0)
    total_tn = torch.tensor(0.0)
    boundary_sum = 0.0
    batch_count = 0
    for images, masks in tqdm(loader, desc="Metrics", leave=False):
        images = images.to(device)
        masks = masks.to(device).unsqueeze(1).float()
        outputs = model(images)
        predictions = (torch.sigmoid(outputs) > threshold).float()
        tp, fp, fn, tn = smp.metrics.get_stats(predictions.long(), masks.long(), mode="binary")
        total_tp += tp.sum().detach().cpu()
        total_fp += fp.sum().detach().cpu()
        total_fn += fn.sum().detach().cpu()
        total_tn += tn.sum().detach().cpu()
        boundary_sum += boundary_iou(predictions, masks)
        batch_count += 1
    return summarize_scores(
        total_tp, total_fp, total_fn, total_tn, boundary_sum / max(batch_count, 1)
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
        images = images.to(device)
        _ = model(images)
        frames += images.size(0)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = max(time.perf_counter() - start, 1e-6)
    peak_vram = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    )
    return frames / elapsed, peak_vram


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
) -> tuple[Path, EpochHistory, float]:
    """Train until ``max_epochs`` and keep the best-IoU checkpoint."""
    checkpoint_root = Path(checkpoint_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    loss_tag = loss_display_name(loss_fn)
    history = EpochHistory()
    best_iou = 0.0
    best_path = checkpoint_root / f"{model_tag}_{loss_tag}_best.pth"
    for epoch in range(1, max_epochs + 1):
        train_loss, train_iou = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        valid_loss, valid_scores, fps, vram = validate_one_epoch(
            model, valid_loader, loss_fn, device
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
    if not best_path.is_file():
        torch.save(
            {"model_state": model.state_dict(), "valid_iou": best_iou, "epoch": max_epochs},
            best_path,
        )
    return best_path, history, best_iou
