"""Model definitions for the image classification experiments."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models


class TinyVGG(nn.Module):
    """TinyVGG-style convolutional network used in the course lessons.

    The network has two convolution blocks followed by one linear classifier.
    ``hidden_units`` controls the number of feature maps and therefore the
    model capacity.
    """

    def __init__(
        self,
        input_shape: int = 3,
        hidden_units: int = 10,
        output_shape: int = 3,
        image_size: int = 64,
    ) -> None:
        super().__init__()
        # Each conv block keeps spatial size with padding=1, then MaxPool halves
        # height and width. Two blocks reduce image_size by a factor of 4.
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
        # After two MaxPool layers, a 64x64 image becomes 16x16. The classifier
        # needs that flattened feature count for its Linear layer.
        flattened_features = hidden_units * (image_size // 4) * (image_size // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=flattened_features, out_features=output_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return raw logits. CrossEntropyLoss expects logits and applies the
        # softmax operation internally.
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))


def create_tinyvgg(
    input_shape: int = 3,
    hidden_units: int = 10,
    output_shape: int = 3,
    image_size: int = 64,
) -> TinyVGG:
    """Factory for TinyVGG.

    Keeping model creation in a function makes the CLI and tests simpler and
    gives all callers the same defaults.
    """

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
    """Create an EfficientNet_B0 transfer-learning model for 3 classes.

    The pretrained feature extractor stays intact. Only the classifier head is
    replaced so the model predicts this project's food classes.
    """

    model = models.efficientnet_b0(weights=weights)
    if freeze_base:
        # Freezing keeps ImageNet features fixed and trains only the new head.
        # This is faster and works well for small image datasets.
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    # EfficientNet_B0's classifier is Dropout + Linear. Replace the Linear layer
    # with one whose output size equals the number of target classes.
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features=in_features, out_features=output_shape),
    )
    return model
