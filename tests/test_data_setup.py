from pathlib import Path

import pytest
import torch
from PIL import Image

from going_moduler.data_setup import ImageDataset, build_transforms, create_dataloaders


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color=color).save(path)


def test_custom_image_dataset_loads_tensor_and_label(tmp_path: Path) -> None:
    image_path = tmp_path / "pizza" / "sample.jpg"
    _write_image(image_path, (255, 0, 0))
    transform = build_transforms(image_size=16, augment=False, normalize=False)

    dataset = ImageDataset([image_path], ["pizza"], class_names=["pizza"], transform=transform)

    image_tensor, label = dataset[0]
    assert isinstance(image_tensor, torch.Tensor)
    assert image_tensor.shape == (3, 16, 16)
    assert label == 0


def test_create_dataloaders_splits_class_folders(tmp_path: Path) -> None:
    for class_name, color in {"pizza": (255, 0, 0), "steak": (0, 255, 0), "sushi": (0, 0, 255)}.items():
        for index in range(4):
            _write_image(tmp_path / class_name / f"{index}.jpg", color)

    train_loader, test_loader, class_names = create_dataloaders(
        data_dir=tmp_path,
        batch_size=2,
        image_size=16,
        augment=False,
        normalize=False,
        seed=123,
    )

    images, labels = next(iter(train_loader))
    assert class_names == ["pizza", "steak", "sushi"]
    assert images.shape == (2, 3, 16, 16)
    assert labels.dtype == torch.long
    assert len(train_loader.dataset) == 9
    assert len(test_loader.dataset) == 3


def test_image_dataset_rejects_mismatched_lengths(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ImageDataset([tmp_path / "missing.jpg"], [])
