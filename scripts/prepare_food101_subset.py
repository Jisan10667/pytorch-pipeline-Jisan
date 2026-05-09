"""Download a Food-101 subset from Hugging Face into class folders.

This creates the layout expected by train.py:

data/pizza_steak_sushi/
  pizza/*.jpg
  steak/*.jpg
  sushi/*.jpg
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare pizza/steak/sushi images from Hugging Face Food-101.")
    parser.add_argument("--dataset", default="ethz/food101", help="Hugging Face dataset id.")
    parser.add_argument("--output-dir", default="data/pizza_steak_sushi", help="Output class-folder directory.")
    parser.add_argument("--classes", nargs="+", default=["pizza", "steak", "sushi"])
    parser.add_argument("--images-per-class", type=int, default=225)
    parser.add_argument("--split", default="train", help="Food-101 split to sample from.")
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: datasets\n\n"
            "Install it inside the active virtual environment:\n"
            "  python -m pip install datasets\n\n"
            "If installation fails with 'No space left on device', free disk space first. "
            "The current project also needs room for the downloaded image subset."
        ) from exc

    dataset = load_dataset(args.dataset, split=args.split)
    label_feature = dataset.features["label"]
    label_names = label_feature.names
    wanted = set(args.classes)
    wanted_label_ids = {label_names.index(class_name): class_name for class_name in args.classes}
    counts = {class_name: 0 for class_name in args.classes}

    for example_index, example in enumerate(dataset):
        label_id = int(example["label"])
        class_name = wanted_label_ids.get(label_id)
        if class_name is None or counts[class_name] >= args.images_per_class:
            continue

        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        image = example["image"].convert("RGB")
        image_path = class_dir / f"{safe_name(class_name)}_{counts[class_name]:04d}_{example_index:06d}.jpg"
        image.save(image_path, quality=95)
        counts[class_name] += 1

        if all(counts[class_name] >= args.images_per_class for class_name in wanted):
            break

    print(f"Saved subset to: {output_dir}")
    for class_name in args.classes:
        print(f"{class_name}: {counts[class_name]} images")

    missing = [class_name for class_name, count in counts.items() if count < args.images_per_class]
    if missing:
        raise RuntimeError(f"Not enough images found for: {', '.join(missing)}")


if __name__ == "__main__":
    main()
