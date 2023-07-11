"""Implementation of VGG16 in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7369575."""

from __future__ import annotations

import torch
import torchvision


def vgg16mod(num_classes: int) -> torch.nn.Module:
    """Create modified VGG16 model.

    The classifier of this model is
        Linear (25,088, 4096)
        ReLU -> Dropout
        Linear (1024, num_classes)
    """
    model = torchvision.models.vgg16()
    model.classifier = model.classifier[:4]
    in_features = model.classifier[0].in_features
    model.classifier[0] = torch.nn.Linear(in_features, 1024)
    model.classifier[3] = torch.nn.Linear(1024, num_classes)
    return model
