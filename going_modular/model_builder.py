"""Model definitions for the image classification experiments."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class TinyVGG(nn.Module):
    """TinyVGG-style convolutional network used in the course lessons."""

    def __init__(
        self,
        input_shape: int = 3,
        hidden_units: int = 10,
        output_shape: int = 3,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        flattened_features = hidden_units * (image_size // 4) * (image_size // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_features, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


def create_tinyvgg(
    input_shape: int = 3,
    hidden_units: int = 10,
    output_shape: int = 3,
    image_size: int = 64,
) -> TinyVGG:
    """Factory for TinyVGG."""

    return TinyVGG(
        input_shape=input_shape,
        hidden_units=hidden_units,
        output_shape=output_shape,
        image_size=image_size,
    )


def create_effnetb0(
    output_shape: int = 3,
    freeze_base: bool = True,
    weights: models.EfficientNet_B0_Weights | None = models.EfficientNet_B0_Weights.DEFAULT,
) -> nn.Module:
    """Create an EfficientNet_B0 transfer-learning model for 3 classes."""

    model = models.efficientnet_b0(weights=weights)
    if freeze_base:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=in_features, out_features=output_shape),
    )
    return model
