"""Command-line training entry point."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

try:
    from going_modular import data_setup, engine, model_builder, utils
except ImportError:
    import data_setup
    import engine
    import model_builder
    import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TinyVGG or EfficientNet on a custom image dataset.")
    parser.add_argument("--data-dir", type=str, default=None, help="Single class-folder dataset to split into train/test.")
    parser.add_argument("--train-dir", type=str, default=None, help="Training directory with class subfolders.")
    parser.add_argument("--test-dir", type=str, default=None, help="Testing directory with class subfolders.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden-units", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model", choices=["tinyvgg", "effnetb0"], default="tinyvgg")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--experiments-dir", type=str, default="experiments")
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--compare-grid", action="store_true", help="Run 3 TinyVGG lr x hidden-units experiments.")
    return parser.parse_args()


def _default_data_dir(args: argparse.Namespace) -> None:
    if args.data_dir is None and args.train_dir is None:
        candidate = Path("data") / "pizza_steak_sushi"
        args.data_dir = str(candidate)


def build_model(model_name: str, class_count: int, hidden_units: int, image_size: int) -> nn.Module:
    if model_name == "effnetb0":
        return model_builder.create_effnetb0(output_shape=class_count)
    return model_builder.create_tinyvgg(hidden_units=hidden_units, output_shape=class_count, image_size=image_size)


def run_training(args: argparse.Namespace, lr: float, hidden_units: int, run_name: str) -> dict:
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=224 if args.model == "effnetb0" else args.image_size,
        augment=not args.no_augment,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(args.model, len(class_names), hidden_units, 224 if args.model == "effnetb0" else args.image_size)
    optimizer = torch.optim.Adam(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    writer = engine.make_writer(
        experiment_name=run_name,
        model_name=args.model,
        extra=f"lr_{lr}_hidden_{hidden_units}",
        log_dir=args.experiments_dir,
    )
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=args.epochs,
        device=device,
        writer=writer,
    )
    summary = engine.summarize_results(results)
    summary.update(
        {
            "run_name": run_name,
            "model": args.model,
            "learning_rate": lr,
            "hidden_units": hidden_units,
            "class_names": class_names,
        }
    )
    utils.save_json(summary, Path(args.experiments_dir) / run_name / "summary.json")
    utils.plot_loss_curves(results, save_path="loss_curves.png")
    utils.save_model(model, args.models_dir, "best_model.pth")
    return summary


def main() -> None:
    args = parse_args()
    _default_data_dir(args)
    if args.model == "effnetb0" and args.epochs == 30:
        args.epochs = 10

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if args.compare_grid:
        grid = [(0.001, 10), (0.0005, 10), (0.001, 20)]
        summaries = []
        for lr, hidden_units in grid:
            run_name = f"tinyvgg-grid-{timestamp}-lr-{lr}-hidden-{hidden_units}"
            summaries.append(run_training(args, lr=lr, hidden_units=hidden_units, run_name=run_name))
        utils.plot_experiment_comparison(summaries, save_path="experiment_comparison.png")
        utils.save_json({"experiments": summaries}, Path(args.experiments_dir) / f"comparison-{timestamp}.json")
    else:
        run_name = args.run_name or f"{args.model}-{timestamp}"
        run_training(args, lr=args.lr, hidden_units=args.hidden_units, run_name=run_name)


if __name__ == "__main__":
    main()
