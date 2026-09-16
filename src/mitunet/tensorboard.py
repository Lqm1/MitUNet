"""TensorBoard run allocation shared by the training entry points."""

import re
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp

from torch.utils.tensorboard import SummaryWriter


def create_writer(
    log_root: str | Path | None, run_name: str, flush_secs: int = 30
) -> SummaryWriter | None:
    """Create an isolated run; None disables logging without creating directories."""
    if flush_secs < 1:
        raise ValueError("tensorboard_flush_secs must be positive")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError(
            "tensorboard_run_name must use letters, digits, dots, underscores or hyphens"
        )
    if log_root is None:
        return None
    if not str(log_root).strip():
        raise ValueError("tensorboard_dir must not be empty; use None to disable logging")
    root = Path(log_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = mkdtemp(prefix=f"{run_name}_{timestamp}_", dir=root)
    writer = SummaryWriter(log_dir=run_dir, flush_secs=flush_secs)
    print(f"TensorBoard logging to {run_dir}")
    return writer


def should_log_images(epoch: int, every: int, maximum: int) -> bool:
    """Zero interval or image count disables image logging."""
    if every < 0 or maximum < 0:
        raise ValueError("TensorBoard image interval and count must be nonnegative")
    return every > 0 and maximum > 0 and (epoch == 1 or epoch % every == 0)
