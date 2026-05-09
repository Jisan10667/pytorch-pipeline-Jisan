"""Dataset and DataLoader utilities for custom image folders.

The project intentionally avoids ``torchvision.datasets.ImageFolder`` so the
dataset class can also work from arbitrary file paths collected elsewhere.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Iterable

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageDataset(Dataset):
    """A custom image dataset backed by explicit file paths and labels."""

    def __init__(
        self,
        image_paths: Iterable[str | Path],
        labels: Iterable[int | str],
        class_names: list[str] | None = None,
        transform: Callable | None = None,
    ) -> None:
        self.image_paths = [Path(path) for path in image_paths]
        raw_labels = list(labels)
        if len(self.image_paths) != len(raw_labels):
            raise ValueError("image_paths and labels must have the same length.")
        if not self.image_paths:
            raise ValueError("ImageDataset needs at least one image path.")

        if class_names is None:
            if all(isinstance(label, int) for label in raw_labels):
                class_names = [str(index) for index in sorted(set(raw_labels))]
            else:
                class_names = sorted({str(label) for label in raw_labels})

        self.class_names = list(class_names)
        self.class_to_idx = {class_name: index for index, class_name in enumerate(self.class_names)}
        self.labels = [label if isinstance(label, int) else self.class_to_idx[str(label)] for label in raw_labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


def build_transforms(
    image_size: int = 64,
    augment: bool = False,
    normalize: bool = True,
) -> transforms.Compose:
    """Create torchvision transforms for training or evaluation."""

    transform_steps: list[Callable] = [transforms.Resize((image_size, image_size))]
    if augment:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.TrivialAugmentWide(),
            ]
        )
    transform_steps.append(transforms.ToTensor())
    if normalize:
        transform_steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(transform_steps)


def find_classes(directory: str | Path) -> list[str]:
    """Return sorted class folder names from a directory."""

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {directory}. "
            "Create it with class subfolders such as "
            "'data/pizza_steak_sushi/pizza', 'data/pizza_steak_sushi/steak', "
            "and 'data/pizza_steak_sushi/sushi', or pass --data-dir/--train-dir "
            "to the folder that actually contains your images."
        )
    class_names = sorted(path.name for path in directory.iterdir() if path.is_dir())
    if not class_names:
        raise FileNotFoundError(f"No class directories found in {directory}.")
    return class_names


def list_image_paths(directory: str | Path, class_names: list[str] | None = None) -> tuple[list[Path], list[str], list[str]]:
    """Collect image paths and string labels from class-named subdirectories."""

    directory = Path(directory)
    class_names = class_names or find_classes(directory)
    paths: list[Path] = []
    labels: list[str] = []
    for class_name in class_names:
        class_dir = directory / class_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(image_path)
                labels.append(class_name)
    if not paths:
        raise FileNotFoundError(f"No images found under {directory}.")
    return paths, labels, class_names


def stratified_split(
    image_paths: list[Path],
    labels: list[str],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[Path], list[str], list[Path], list[str]]:
    """Split paths into train/test subsets while preserving class balance."""

    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    by_label: dict[str, list[Path]] = {}
    for path, label in zip(image_paths, labels):
        by_label.setdefault(label, []).append(path)

    train_paths: list[Path] = []
    train_labels: list[str] = []
    test_paths: list[Path] = []
    test_labels: list[str] = []
    for label, paths in by_label.items():
        shuffled = paths[:]
        rng.shuffle(shuffled)
        split_index = max(1, int(len(shuffled) * train_ratio))
        if len(shuffled) > 1:
            split_index = min(split_index, len(shuffled) - 1)
        label_train = shuffled[:split_index]
        label_test = shuffled[split_index:]
        train_paths.extend(label_train)
        train_labels.extend([label] * len(label_train))
        test_paths.extend(label_test)
        test_labels.extend([label] * len(label_test))

    return train_paths, train_labels, test_paths, test_labels


def create_dataloaders(
    train_dir: str | Path | None = None,
    test_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    train_ratio: float = 0.8,
    transform: Callable | None = None,
    train_transform: Callable | None = None,
    test_transform: Callable | None = None,
    batch_size: int = 32,
    image_size: int = 64,
    augment: bool = True,
    normalize: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create train/test DataLoaders from split folders or one custom folder.

    Expected folder layouts:
    - ``train_dir/class_name/*.jpg`` and ``test_dir/class_name/*.jpg``
    - or ``data_dir/class_name/*.jpg`` with an internal stratified split.
    """

    if train_dir is None and data_dir is None:
        raise ValueError("Provide either train_dir/test_dir or data_dir.")

    if train_dir is not None:
        class_names = find_classes(train_dir)
        train_paths, train_labels, class_names = list_image_paths(train_dir, class_names)
        if test_dir is None:
            train_paths, train_labels, test_paths, test_labels = stratified_split(
                train_paths, train_labels, train_ratio=train_ratio, seed=seed
            )
        else:
            test_paths, test_labels, _ = list_image_paths(test_dir, class_names)
    else:
        all_paths, all_labels, class_names = list_image_paths(data_dir)
        train_paths, train_labels, test_paths, test_labels = stratified_split(
            all_paths, all_labels, train_ratio=train_ratio, seed=seed
        )

    if transform is not None:
        train_transform = train_transform or transform
        test_transform = test_transform or transform
    train_transform = train_transform or build_transforms(image_size=image_size, augment=augment, normalize=normalize)
    test_transform = test_transform or build_transforms(image_size=image_size, augment=False, normalize=normalize)

    train_dataset = ImageDataset(train_paths, train_labels, class_names=class_names, transform=train_transform)
    test_dataset = ImageDataset(test_paths, test_labels, class_names=class_names, transform=test_transform)
    pin_memory = torch.cuda.is_available() if pin_memory is None else pin_memory

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_dataloader, test_dataloader, class_names
