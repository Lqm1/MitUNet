import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mitunet.tensorboard import create_writer, should_log_images


def test_training_logs_bounded_samples_without_extra_forward(tmp_path):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from mitunet.engine import fit_wall_segmenter
    from mitunet.objectives import build_loss

    model = torch.nn.Conv2d(3, 1, 1)
    calls = []
    model.register_forward_hook(lambda *_: calls.append(1))
    loader = DataLoader(
        TensorDataset(
            torch.zeros(4, 3, 8, 8),
            torch.ones(4, 8, 8),
        ),
        batch_size=4,
    )
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
    path, _, _ = fit_wall_segmenter(
        model,
        loader,
        loader,
        build_loss("tversky"),
        optimizer,
        scheduler,
        torch.device("cpu"),
        1,
        tmp_path / "checkpoints",
        tensorboard_dir=tmp_path / "logs",
        max_log_images=1,
    )
    assert path.is_file()
    assert len(calls) == 2  # training and validation only
    run = next((tmp_path / "logs").iterdir())
    events = EventAccumulator(str(run)).Reload()
    assert events.Scalars("Loss/valid")[0].step == 1
    assert events.Images("Samples/valid/input")[0].width == 8
    assert events.Images("Samples/valid/target")[0].width == 8


def test_runs_are_isolated_and_events_readable(tmp_path):
    first = create_writer(tmp_path, "experiment")
    second = create_writer(tmp_path, "experiment")
    assert first is not None and second is not None
    try:
        assert first.log_dir != second.log_dir
        first.add_scalar("Loss/train", 0.25, 3)
        first.flush()
        events = EventAccumulator(first.log_dir).Reload()
        assert events.Scalars("Loss/train")[0].step == 3
        assert events.Scalars("Loss/train")[0].value == 0.25
    finally:
        first.close()
        second.close()


def test_disabled_and_invalid_options(tmp_path):
    assert create_writer(None, "experiment") is None
    assert not list(tmp_path.iterdir())
    with pytest.raises(ValueError):
        create_writer(tmp_path, "../escape")
    with pytest.raises(ValueError):
        create_writer("", "experiment")
    assert should_log_images(1, 5, 4)
    assert should_log_images(5, 5, 4)
    assert not should_log_images(2, 5, 4)
    assert not should_log_images(1, 0, 4)
    assert not should_log_images(1, 5, 0)
