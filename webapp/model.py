from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class CalorieRegressor(nn.Module):
    """ImageNet-pretrained ResNet-18 backbone + calorie regression head (+ optional Food-101 cls)."""

    def __init__(self, mode: str = "rgb", pretrained: bool = True, num_classes: int = 0):
        super().__init__()
        if mode not in {"rgb", "rgbd"}:
            raise ValueError("mode must be one of {'rgb', 'rgbd'}")
        self.mode = mode
        self.num_classes = num_classes

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self._adapt_input_layer_if_needed()
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.reg_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )
        self.cls_head = None
        if num_classes > 0:
            self.cls_head = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(256, num_classes),
            )

    def _adapt_input_layer_if_needed(self) -> None:
        if self.mode != "rgbd":
            return
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=4,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:4] = old_conv.weight.mean(dim=1, keepdim=True)
        self.backbone.conv1 = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        preds = self.reg_head(features)
        return preds

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        if self.cls_head is None:
            raise RuntimeError("Classification head is not enabled for this model.")
        features = self.backbone(x)
        return self.cls_head(features)

    @property
    def has_classifier(self) -> bool:
        return self.cls_head is not None
