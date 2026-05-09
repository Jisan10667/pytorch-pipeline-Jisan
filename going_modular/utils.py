"""Utility helpers for checkpoints, plots, and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn


def save_model(model: nn.Module, target_dir: str | Path, model_name: str = "best_model.pth") -> Path:
    """Save a model state_dict to ``target_dir / model_name``."""

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / model_name
    torch.save(model.state_dict(), model_path)
    return model_path


def load_model(
    model: nn.Module,
    model_path: str | Path,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load a state_dict checkpoint into an instantiated model."""

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def save_json(data: dict, path: str | Path) -> None:
    """Persist a dictionary as formatted JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def plot_loss_curves(results: dict[str, list[float]], save_path: str | Path = "loss_curves.png") -> Path:
    """Plot train/test loss and accuracy curves."""

    epochs = range(1, len(results["train_loss"]) + 1)
    save_path = Path(save_path)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, results["train_loss"], label="train")
    plt.plot(epochs, results["test_loss"], label="test")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, results["train_acc"], label="train")
    plt.plot(epochs, results["test_acc"], label="test")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def plot_experiment_comparison(
    experiment_summaries: list[dict],
    save_path: str | Path = "experiment_comparison.png",
) -> Path:
    """Plot best test accuracy for each experiment run."""

    save_path = Path(save_path)
    names = [summary["run_name"] for summary in experiment_summaries]
    accuracies = [summary["best_test_acc"] for summary in experiment_summaries]
    plt.figure(figsize=(max(8, len(names) * 2.5), 5))
    plt.bar(names, accuracies)
    plt.ylabel("Best test accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path


def predict_image(
    model: nn.Module,
    image_path: str | Path,
    class_names: list[str],
    transform: Callable,
    device: torch.device | str = "cpu",
) -> tuple[str, float]:
    """Return top-1 class name and confidence for one image."""

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.to(device)
    model.eval()
    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        confidence, prediction = probabilities.max(dim=1)
    return class_names[prediction.item()], confidence.item()
