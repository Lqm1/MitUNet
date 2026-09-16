import json
import pickle

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mitunet.architecture import build_wall_segmenter, model_from_checkpoint
from mitunet.config import InferenceConfig, ModelConfig
from mitunet.engine import fit_wall_segmenter
from mitunet.objectives import build_loss
from mitunet.predictor import WallPredictor
from mitunet.target_cache import TargetCache
from mitunet.wall_dataset import CocoWallDataset


def test_cache_is_bounded_and_returns_copies():
    cache = TargetCache(1)
    source = np.ones((512, 1024), dtype=np.uint8)
    cache.put((1,), (source,))
    source[:] = 7
    assert cache.get((1,))[0][0, 0] == 1
    cache.get((1,))[0][:] = 9
    assert cache.get((1,))[0][0, 0] == 1
    cache.put((2,), (source,))
    cache.put((3,), (source,))
    assert cache.get((1,)) is None
    assert cache.bytes <= cache.max_bytes
    assert not pickle.loads(pickle.dumps(cache)).entries
    disabled = TargetCache(0)
    disabled.put((1,), (source,))
    assert disabled.get((1,)) is None


def test_coco_cached_masks_match_uncached(tmp_path, monkeypatch):
    cv2.imwrite(str(tmp_path / "image.png"), np.full((64, 64, 3), 200, dtype=np.uint8))
    annotation = tmp_path / "annotations.json"
    annotation.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "image.png", "height": 64, "width": 64}],
                "categories": [{"id": 1, "name": "wall"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "iscrowd": 0,
                        "area": 1024,
                        "bbox": [8, 8, 32, 32],
                        "segmentation": [[8, 8, 40, 8, 40, 40, 8, 40]],
                    }
                ],
            }
        )
    )
    cached = CocoWallDataset(tmp_path, annotation)
    uncached = CocoWallDataset(tmp_path, annotation, target_cache_mb=0)
    reference = uncached[0]
    first = cached[0]
    monkeypatch.setattr(
        cached, "build_wall_mask", lambda *_: (_ for _ in ()).throw(AssertionError("cache missed"))
    )
    second = cached[0]
    assert np.array_equal(first[1], reference[1])
    assert np.array_equal(second[1], reference[1])


def test_real_mit_b4_training_checkpoint_and_fast_loading(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(5)
    config = ModelConfig(encoder_weights=None)
    model = build_wall_segmenter(config)
    loader = DataLoader(
        TensorDataset(torch.randn(2, 3, 64, 64), torch.ones(2, 64, 64)), batch_size=2
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max")
    best, history, _ = fit_wall_segmenter(
        model,
        loader,
        loader,
        build_loss("tversky"),
        optimizer,
        scheduler,
        torch.device("cpu"),
        1,
        tmp_path,
    )
    assert np.isfinite(history.train_loss[0])
    from mitunet import architecture

    monkeypatch.setattr(
        architecture,
        "build_wall_segmenter",
        lambda *_: (_ for _ in ()).throw(AssertionError("legacy allocation")),
    )
    restored = model_from_checkpoint(best, torch.device("cpu"), config)
    model.eval()
    with torch.inference_mode():
        image = torch.randn(1, 3, 64, 64)
        torch.testing.assert_close(restored.eval()(image), model(image), rtol=0, atol=0)
    predictor = WallPredictor(restored, torch.device("cpu"), InferenceConfig(image_size=64))
    output = predictor.predict_proba(np.zeros((64, 64, 3), dtype=np.uint8))
    assert output.shape == (64, 64) and np.isfinite(output).all()


def test_train_metrics_and_weights_match_reference():
    import copy

    import segmentation_models_pytorch as smp

    from mitunet.engine import train_one_epoch

    torch.manual_seed(7)
    model = torch.nn.Conv2d(3, 1, 1)
    reference = copy.deepcopy(model)
    data = TensorDataset(torch.randn(4, 3, 8, 8), torch.randint(0, 2, (4, 8, 8)))
    loader = DataLoader(data, batch_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    reference_optimizer = torch.optim.Adam(reference.parameters(), lr=1e-4)
    loss_fn = build_loss("tversky")
    total = 0.0
    counts = [torch.tensor(0.0) for _ in range(4)]
    for images, masks in loader:
        masks = masks.unsqueeze(1).float()
        reference_optimizer.zero_grad()
        raw = reference(images)
        loss = loss_fn(raw, masks)
        loss.backward()
        reference_optimizer.step()
        total += loss.item() * len(images)
        stats = smp.metrics.get_stats((raw.sigmoid() > 0.5).long(), masks.long(), mode="binary")
        for accumulated, value in zip(counts, stats, strict=True):
            accumulated += value.sum()
    loss, iou = train_one_epoch(model, loader, loss_fn, optimizer, torch.device("cpu"))
    assert loss == total / 4
    assert iou == smp.metrics.iou_score(*counts).item()
    for actual, expected in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_infer_cli_only_runs_one_prediction(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from mitunet import command

    calls = []

    class Predictor:
        def predict_proba(self, image):
            calls.append(image.shape)
            return np.full((8, 8), 0.7, dtype=np.float32)

    monkeypatch.setattr(command.WallPredictor, "from_checkpoint", lambda *a, **kw: Predictor())
    monkeypatch.setattr(command, "load_rgb_image", lambda *_: np.zeros((8, 8, 3), dtype=np.uint8))
    result = CliRunner().invoke(
        command.app,
        [
            "infer",
            str(tmp_path / "dummy.pth"),
            str(tmp_path / "image.png"),
            "--output-dir",
            str(tmp_path / "out"),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert (tmp_path / "out" / "image_wall.png").is_file()
