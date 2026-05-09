# PyTorch Pipeline: Pizza, Steak, Sushi

Reusable CNN training pipeline for a Food-101 subset with custom data loading,
TensorBoard experiment tracking, TinyVGG, and EfficientNet_B0 transfer learning.

## Environment Setup

Use a virtual environment so PyTorch, torchvision, TensorBoard, pytest, and the
plotting dependencies do not mix with your system Python.

Recommended Python version: **Python 3.11**. PyTorch may not have wheels for the
newest Python releases, so avoid using Python 3.14 for this project.

From the repo root:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python3.11` is not installed, install it first:

```bash
brew install python@3.11
```

After activation, confirm the environment is using the venv Python:

```bash
which python
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

If installation fails with `No space left on device`, free disk space and rerun:

```bash
python -m pip install -r requirements.txt
```

Every new terminal session needs activation before running training:

```bash
source .venv/bin/activate
```

## Dataset Download

The best match for this assignment is the official Food-101 dataset on Hugging
Face: `ethz/food101`. It contains 101 food categories, including `pizza`,
`steak`, and `sushi`, with 750 train and 250 test images per class in the
original dataset.

Create the required 225-images-per-class subset:

```bash
python -m pip install datasets
python scripts/prepare_food101_subset.py \
  --dataset ethz/food101 \
  --output-dir data/pizza_steak_sushi \
  --classes pizza steak sushi \
  --images-per-class 225
```

This writes:

```text
data/pizza_steak_sushi/
├── pizza/
├── steak/
└── sushi/
```

If you see `ModuleNotFoundError: No module named 'datasets'`, the active venv
does not have Hugging Face Datasets installed yet. Run:

```bash
source .venv/bin/activate
python -m pip install datasets
```

If pip reports `No space left on device`, free disk space before retrying. The
download step needs enough room for Python package dependencies plus the image
subset under `data/`.

## Training Setup

After activating `.venv`, run from the repo root:

```bash
python train.py --data-dir data/pizza_steak_sushi --epochs 30 --lr 0.001
```

The `--data-dir` path must already exist and contain one subfolder per class:

```text
data/
└── pizza_steak_sushi/
    ├── pizza/
    │   ├── image_001.jpg
    │   └── ...
    ├── steak/
    │   ├── image_001.jpg
    │   └── ...
    └── sushi/
        ├── image_001.jpg
        └── ...
```

For the assignment dataset, place the Food-101 subset there with 225 images per
class. The repo ignores `data/` because image datasets are too large to commit.

The TA smoke test is supported:

```bash
python train.py --epochs 5 --lr 0.001
```

By default, the CLI looks for `data/pizza_steak_sushi/class_name/*.jpg`. You can
also pass explicit split folders:

```bash
python train.py --train-dir data/train --test-dir data/test --epochs 30
```

TensorBoard logs are written to `experiments/`:

```bash
tensorboard --logdir experiments
```

Run tests with coverage:

```bash
pytest
```

## Experiment Grid

Run three TinyVGG hyperparameter experiments:

```bash
python train.py --data-dir data/pizza_steak_sushi --epochs 30 --compare-grid
```

This logs each run under a unique TensorBoard name and writes:

- `loss_curves.png`
- `experiment_comparison.png`
- `experiments/**/summary.json`
- `models/best_model.pth`

## Accuracy Table

Fill these with the final metrics after training on the 225-image-per-class
Food-101 subset.

| Run | Model | Learning rate | Hidden units | Epochs | Best test accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| tinyvgg-lr-0.001-hidden-10 | TinyVGG | 0.001 | 10 | 30 | TBD |
| tinyvgg-lr-0.0005-hidden-10 | TinyVGG | 0.0005 | 10 | 30 | TBD |
| tinyvgg-lr-0.001-hidden-20 | TinyVGG | 0.001 | 20 | 30 | TBD |
| effnetb0-transfer | EfficientNet_B0 | 0.001 | n/a | 10 | TBD |

Best config: update after the grid run. In general, the winning TinyVGG config
should balance enough hidden units to learn food texture/shape cues without
overfitting the small 675-image dataset.

## Transfer Learning Comparison

EfficientNet_B0 is available through:

```bash
python train.py --data-dir data/pizza_steak_sushi --model effnetb0 --epochs 10 --lr 0.001
```

The base feature extractor is frozen and the classifier head is replaced for
three classes. EfficientNet usually reaches higher accuracy faster because its
features were pretrained on ImageNet, but each epoch is heavier than TinyVGG and
the checkpoint has more parameters. TinyVGG is faster and easier to inspect;
EfficientNet is the better accuracy choice when transfer-learning weights are
allowed.

## Inference

After training, run top-1 inference with confidence on up to three custom images:

```bash
python predict.py path/to/image1.jpg path/to/image2.jpg path/to/image3.jpg \
  --checkpoint models/best_model.pth \
  --classes pizza steak sushi
```

## Known Limitations

- The repository does not include the Food-101 image files.
- `loss_curves.png`, `experiment_comparison.png`, and final accuracy values must
  be regenerated after real training in the grading environment.
- EfficientNet_B0 default weights require torchvision to download or already
  cache pretrained weights.
