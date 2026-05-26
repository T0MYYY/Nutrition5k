import torch
import torch.nn as nn
from typing import List, Dict
from .backbone import create_backbone


class MultitaskNutritionNet(nn.Module):
    """Multi-task regression network for nutrition estimation.

    Architecture:
        backbone (default: InceptionV3, 8192 features for 256x256 input)
        -> shared FC(in_feat->4096) -> FC(4096->4096)
        -> per-task: FC(4096->4096) -> FC(4096->1)
    """

    def __init__(self, tasks: List[str], in_channels: int = 3,
                 backbone_name: str = 'inception_v3'):
        super().__init__()
        self.tasks = tasks
        self.backbone = create_backbone(backbone_name, in_channels)
        in_feat = self.backbone.OUT_FEATURES

        self.shared = nn.Sequential(
            nn.Linear(in_feat, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),   nn.ReLU(inplace=True),
        )
        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(4096, 4096), nn.ReLU(inplace=True),
                nn.Linear(4096, 1),
            )
            for task in tasks
        })

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat = self.backbone(x)
        shared = self.shared(feat)
        return {task: self.heads[task](shared).squeeze(1) for task in self.tasks}
