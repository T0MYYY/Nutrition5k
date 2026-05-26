import torch
import torch.nn as nn
from typing import Optional
from .backbone import create_backbone


class MassRegressor(nn.Module):
    """Mass regression model for Exp 4.

    backbone -> optionally concat volume scalar -> FC(->4096) -> FC(->4096) -> FC(->1).
    """

    def __init__(self, use_volume: bool = True, backbone_name: str = 'inception_v3'):
        super().__init__()
        self.use_volume = use_volume
        self.backbone = create_backbone(backbone_name, in_channels=3)
        in_feat = self.backbone.OUT_FEATURES + (1 if use_volume else 0)
        self.head = nn.Sequential(
            nn.Linear(in_feat, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),    nn.ReLU(inplace=True),
            nn.Linear(4096, 1),
        )

    def forward(self, x: torch.Tensor, volume: Optional[torch.Tensor]) -> torch.Tensor:
        if self.use_volume and volume is None:
            raise ValueError("volume tensor required when use_volume=True")
        feat = self.backbone(x)
        if self.use_volume:
            feat = torch.cat([feat, volume.view(-1, 1)], dim=1)
        return self.head(feat).squeeze(1)
