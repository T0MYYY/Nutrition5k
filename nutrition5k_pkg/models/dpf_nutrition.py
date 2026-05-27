from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet101_Weights, resnet101
from torchvision.models.resnet import Bottleneck, ResNet


class ResNetFeatureStream(nn.Module):
    def __init__(self, in_channels: int, pretrained: bool = True,
                 resnet_layers: Optional[Sequence[int]] = None):
        super().__init__()
        if resnet_layers is None:
            weights = ResNet101_Weights.IMAGENET1K_V2 if pretrained else None
            base = resnet101(weights=weights)
        else:
            base = ResNet(Bottleneck, list(resnet_layers))

        if in_channels != 3:
            orig = base.conv1.weight.data.clone()
            base.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained and orig.shape[1] == 3 and in_channels == 1:
                base.conv1.weight.data.copy_(orig.mean(dim=1, keepdim=True))

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return [c2, c3, c4, c5]


class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, in_channels // reduction)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid(),
        )
        self.project = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        joint = rgb + depth
        channel_weight = self.channel_attention(joint)
        spatial_input = torch.cat([joint.mean(dim=1, keepdim=True), joint.amax(dim=1, keepdim=True)], dim=1)
        spatial_weight = self.spatial_attention(spatial_input)
        rgb = rgb * channel_weight * spatial_weight
        depth = depth * channel_weight * spatial_weight
        return self.project(torch.cat([rgb, depth], dim=1))


class DPFNutritionNet(nn.Module):
    def __init__(
        self,
        tasks: List[str],
        pretrained: bool = True,
        fusion_channels: int = 512,
        head_hidden: int = 512,
        food2k_resnet101_path: Optional[str] = None,
        resnet_layers: Optional[Sequence[int]] = None,
    ):
        super().__init__()
        self.tasks = tasks
        self.pretrained = pretrained
        self.rgb_stream = ResNetFeatureStream(3, pretrained=pretrained, resnet_layers=resnet_layers)
        self.depth_stream = ResNetFeatureStream(1, pretrained=pretrained, resnet_layers=resnet_layers)
        self.stage_channels = [256, 512, 1024, 2048]
        self.cabs = nn.ModuleList([
            CrossAttentionBlock(in_ch, fusion_channels)
            for in_ch in self.stage_channels
        ])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.heads = nn.ModuleDict({
            task: nn.Sequential(
                nn.Linear(fusion_channels, head_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(head_hidden, 1),
            )
            for task in tasks
        })
        self._load_food2k_or_log_fallback(food2k_resnet101_path)

    def _load_food2k_or_log_fallback(self, path: Optional[str]) -> None:
        fallback = 'ImageNet ResNet101' if self.pretrained else 'random ResNet'
        if not path:
            print(f'Food2K ResNet101 path not provided; using {fallback} initialization.')
            return
        import os
        if not os.path.isfile(path):
            print(f'Food2K ResNet101 weights not found at {path}; using {fallback} initialization.')
            return
        checkpoint = torch.load(path, map_location='cpu')
        state = self._extract_resnet_state(checkpoint)
        if not state:
            print(f'Food2K ResNet101 weights at {path} did not contain ResNet keys; using {fallback} initialization.')
            return
        stream_state = self.rgb_stream.state_dict()
        matched_keys = [
            key for key, value in state.items()
            if key in stream_state and stream_state[key].shape == value.shape
        ]
        if len(matched_keys) < len(stream_state) // 2:
            print(f'Food2K ResNet101 weights at {path} matched too few ResNet layers; using {fallback} initialization.')
            return
        self.rgb_stream.load_state_dict(state, strict=False)
        depth_state = dict(state)
        if 'conv1.weight' in depth_state and depth_state['conv1.weight'].shape[1] == 3:
            depth_state['conv1.weight'] = depth_state['conv1.weight'].mean(dim=1, keepdim=True)
        self.depth_stream.load_state_dict(depth_state, strict=False)
        print(f'Loaded Food2K ResNet101 weights from {path}.')

    @staticmethod
    def _extract_resnet_state(checkpoint) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ('state_dict', 'model_state_dict', 'model', 'net', 'network'):
                nested = checkpoint.get(key)
                if isinstance(nested, dict):
                    checkpoint = nested
                    break
        if not isinstance(checkpoint, dict):
            return {}

        prefixes = (
            'module.',
            'model.',
            'backbone.',
            'encoder.',
            'encoder_q.',
            'resnet.',
            'base_model.',
        )
        valid_roots = ('conv1.', 'bn1.', 'layer1.', 'layer2.', 'layer3.', 'layer4.')
        state = {}
        for key, value in checkpoint.items():
            clean_key = key
            changed = True
            while changed:
                changed = False
                for prefix in prefixes:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                        changed = True
            if clean_key.startswith(valid_roots):
                state[clean_key] = value
        return state

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> Dict[str, torch.Tensor]:
        rgb_features = self.rgb_stream(rgb)
        depth_features = self.depth_stream(depth)
        fused = None
        for rgb_feat, depth_feat, cab in zip(rgb_features, depth_features, self.cabs):
            current = cab(rgb_feat, depth_feat)
            if fused is None:
                fused = current
            else:
                fused = F.interpolate(fused, size=current.shape[-2:], mode='bilinear', align_corners=False)
                fused = fused + current
        pooled = self.pool(fused).flatten(1)
        return {task: self.heads[task](pooled).squeeze(1) for task in self.tasks}
