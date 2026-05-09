"""Training and evaluation loops.

All model fitting logic lives here so ``train.py`` stays as orchestration only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm


def accuracy_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Calculate top-1 accuracy for a batch."""

    # The largest logit is the model's predicted class.
    correct = torch.eq(y_pred.argmax(dim=1), y_true).sum().item()
    return correct / len(y_true)


def train_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
) -> tuple[float, float]:
    """Train for one epoch and return average loss and accuracy."""

    # train() enables layers such as Dropout/BatchNorm to behave in training
    # mode. TinyVGG does not use them, but EfficientNet's head can.
    model.train()
    train_loss = 0.0
    train_acc = 0.0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        y_logits = model(X)
        loss = loss_fn(y_logits, y)
        # Accumulate per-batch values; divide by number of batches at the end.
        train_loss += loss.item()
        train_acc += accuracy_fn(y_logits, y)

        # Standard PyTorch optimization loop: clear old gradients, backprop the
        # current loss, then update trainable parameters.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc


def test_step(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device | str,
) -> tuple[float, float]:
    """Evaluate for one epoch and return average loss and accuracy."""

    # eval() disables training-only behavior; inference_mode() saves memory by
    # not tracking gradients during validation.
    model.eval()
    test_loss = 0.0
    test_acc = 0.0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            test_logits = model(X)
            test_loss += loss_fn(test_logits, y).item()
            test_acc += accuracy_fn(test_logits, y)

    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    epochs: int,
    device: torch.device | str,
    writer: SummaryWriter | None = None,
) -> dict[str, list[float]]:
    """Train and evaluate a model for ``epochs``."""

    # Keep metric history in plain lists so plotting, JSON summaries, and tests
    # can use the same return value.
    results: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    model.to(device)
    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        if writer is not None:
            # TensorBoard groups train/test curves under the same scalar chart.
            writer.add_scalars("Loss", {"train": train_loss, "test": test_loss}, epoch)
            writer.add_scalars("Accuracy", {"train": train_acc, "test": test_acc}, epoch)
            writer.flush()

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

    if writer is not None:
        writer.close()
    return results


def make_writer(
    experiment_name: str,
    model_name: str,
    extra: str | None = None,
    log_dir: str | Path = "experiments",
) -> SummaryWriter:
    """Create a TensorBoard writer with a unique, readable path."""

    # Example path:
    # experiments/tinyvgg-grid-.../tinyvgg/lr_0.001_hidden_20
    path = Path(log_dir) / experiment_name / model_name
    if extra:
        path = path / extra
    return SummaryWriter(log_dir=str(path))


def summarize_results(results: dict[str, list[float]]) -> dict[str, Any]:
    """Return final and best metrics for README/table generation."""

    # Best epoch is based on validation accuracy, because final-epoch accuracy
    # can dip after the model has already reached its strongest checkpoint.
    best_epoch = max(range(len(results["test_acc"])), key=lambda index: results["test_acc"][index])
    return {
        "final_train_loss": results["train_loss"][-1],
        "final_train_acc": results["train_acc"][-1],
        "final_test_loss": results["test_loss"][-1],
        "final_test_acc": results["test_acc"][-1],
        "best_test_acc": results["test_acc"][best_epoch],
        "best_epoch": best_epoch + 1,
    }
