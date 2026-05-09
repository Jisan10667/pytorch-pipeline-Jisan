"""Run inference on custom images with a saved checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from going_moduler import data_setup, model_builder, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Food-101 subset classes for custom images.")
    parser.add_argument("images", nargs="+", help="Image paths to classify.")
    parser.add_argument("--checkpoint", default="models/best_model.pth")
    parser.add_argument("--classes", nargs="+", default=["pizza", "steak", "sushi"])
    parser.add_argument("--model", choices=["tinyvgg", "effnetb0"], default="tinyvgg")
    parser.add_argument("--hidden-units", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    image_size = 224 if args.model == "effnetb0" else args.image_size
    if args.model == "effnetb0":
        model = model_builder.create_effnetb0(output_shape=len(args.classes))
    else:
        model = model_builder.create_tinyvgg(
            hidden_units=args.hidden_units,
            output_shape=len(args.classes),
            image_size=image_size,
        )
    model = utils.load_model(model, args.checkpoint, device=device)
    transform = data_setup.build_transforms(image_size=image_size, augment=False)

    for image_path in args.images[:3]:
        prediction, confidence = utils.predict_image(model, Path(image_path), args.classes, transform, device=device)
        print(f"{image_path}: {prediction} ({confidence:.2%})")


if __name__ == "__main__":
    main()
