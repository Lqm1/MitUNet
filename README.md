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

## TensorBoard

The train CLI enables logging by default. Each invocation creates an isolated
`<checkpoint_dir>/tensorboard/<run_name>_<timestamp>_<unique_id>/` directory.
Python APIs use `tensorboard_dir=None` to disable logging; pass an explicit
directory to enable it. An empty string is not a disable flag.

Both projects expose the same options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--tensorboard-dir` | `<checkpoint_dir>/tensorboard` | Log root |
| `--tensorboard-run-name` | `mitunet` | Run prefix; letters, digits, dots, underscores, hyphens |
| `--no-tensorboard` | false | Disable all event logging |
| `--tensorboard-image-every` | 5 | Images on epoch 1 and every N epochs; 0 disables |
| `--tensorboard-max-images` | 4 | Maximum images from the first validation batch; 0 disables |
| `--tensorboard-flush-secs` | 30 | Background flush interval; also flush after each epoch |

```bash
uv run --extra cu132 mitunet train ./datasets/coco --tensorboard-run-name experiment-1
uv run --extra cu132 tensorboard --logdir checkpoints/tensorboard
uv run --extra cu132 mitunet train ./datasets/coco --tensorboard-dir ./logs/tb --tensorboard-image-every 10
uv run --extra cu132 mitunet train ./datasets/coco --no-tensorboard
```

Use `--extra cpu` instead for CPU environments.
Scalars use epoch steps and common `Loss/train`, `Loss/valid`, `Optimizer/lr`,
`Metrics/valid_*` tags. Validation images use `Samples/valid/*`.
Samples are collected during validation without another data-loader pass.
New invocations, including resumed or fine-tuned training, always create a new
run; previous event files are never purged. Fine-tuning starts at epoch 1.

MitUNet additionally records training IoU, boundary IoU, recall, precision, accuracy, FPS and VRAM. Its selected loss is a single scalar.
Model graph tracing is not performed during training.
See the [PyTorch TensorBoard documentation](https://docs.pytorch.org/docs/stable/tensorboard.html).

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

## Performance options

Training accepts `--target-cache-mb 256` and `--persistent-workers`.
The cache stores deterministic targets before random augmentation. Its limit is
per dataset instance per process, not a global RAM limit; multiply the limit by
the number of training and validation workers when budgeting memory.
Use `--target-cache-mb 0` to disable it. Workers start with empty caches.
With the default non-persistent workers, their caches are discarded each epoch.
For multi-epoch cache reuse, enable persistent workers or use `--num-workers 0`.

Persistent workers are opt-in because keeping worker RNG state changes the
augmentation sequence compared with recreating workers. They are automatically
disabled when the worker count is zero. The dataset files must remain stable
during a run. COCO annotation changes raise an error requesting dataset recreation.

Loss and confusion statistics accumulate on the compute device; floating loss
sums use float64 to retain the previous Python accumulator precision.
Training progress no longer transfers loss to the CPU for every batch.
Inference uses inference mode without changing the output format or precision.
Complete checkpoints use one U-Net and skip the donor/pretrained download; partial checkpoints retain legacy initialization. Fresh training retains the existing initialization to preserve seed-dependent weights.
