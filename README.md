# MitUNet

Hybrid Mix-Transformer + U-Net wall segmentation for floor plans, rebuilt as a
modern Python/PyTorch project with `uv`.

Derived from [**Enhancing Floor Plan Recognition: A Hybrid Mix-Transformer and
U-Net Approach for Precise Wall Segmentation**](https://doi.org/10.1007/s00138-026-01815-y)
([arXiv:2512.02413](https://arxiv.org/abs/2512.02413)). No notebook code was
copied verbatim; the pipeline below reimplements the published method with
clean naming, type hints and tested modules.

## Model

- MiT-B4 encoder transplanted from SegFormer into a U-Net decoder.
- scSE decoder attention, 3-channel input, single wall logit output.
- Tversky loss (α=0.6, β=0.4) as the paper optimum; Dice/Focal/Lovász available.
- Boundary IoU (dilation ratio 0.02) alongside IoU/recall/precision/accuracy.

## Layout

- `src/mitunet/architecture.py` — hybrid model construction
- `src/mitunet/wall_dataset.py` — COCO wall masks with opening subtraction
- `src/mitunet/augmentation.py` — train/valid/inference pipelines
- `src/mitunet/objectives.py` — loss factory
- `src/mitunet/scores.py` — segmentation + boundary metrics
- `src/mitunet/engine.py` — training/validation loops
- `src/mitunet/predictor.py` — inference wrapper + ONNX export
- `src/mitunet/command.py` — `train`, `infer`, `evaluate`, `smoke`, `export onnx`
- `src/mitunet/support.py` — seeding, devices, image and checkpoint helpers
- `src/mitunet/config.py` — typed configuration containers

## Setup

```bash
uv sync --extra cu132  # CUDA 13.2 builds (Linux/Windows)
uv sync --extra cpu    # CPU-only builds
```

## Usage

```bash
uv run --extra cu132 mitunet smoke --device cuda
uv run --extra cu132 mitunet train ./datasets/coco --epochs 30 --device cuda
uv run --extra cu132 mitunet train ./datasets/coco --fine-tune ./checkpoints/base.pth --device cuda
uv run --extra cu132 mitunet infer ./checkpoints/best.pth ./sample.png --output-dir ./outputs
uv run --extra cu132 mitunet evaluate ./checkpoints/best.pth ./datasets/coco
uv run --extra cu132 mitunet export onnx ./checkpoints/best.pth ./outputs/model.onnx
```

Dataset roots accept COCO layouts (`train/`, `valid/`, `test/` each with
`_annotations.coco.json`) or a single folder with `_annotations.coco.json`.
Evaluation also accepts `images/` + `masks/` folder pairs.

## Testing

```bash
uv run --extra cu132 pytest -q
uvx ruff check src tests
uvx ruff format --check src tests
uv run --extra cu132 mypy src
```

## License

Code is MIT (see `LICENSE`). Pretrained weights trained on CubiCasa5k remain
under CC-BY-NC 4.0; commercial use of the weights is restricted.
