import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights


class InceptionV3Backbone(nn.Module):
    """InceptionV3 feature extractor up to Mixed_7c.

    Output: (B, 8192) for 256x256 input.
    Supports in_channels=4 for RGB-D experiments (Exp 3).
    """

    OUT_FEATURES = 8192  # 2x2x2048 for 256x256 input

    def __init__(self, in_channels: int = 3):
        super().__init__()
        # torchvision>=0.14 forces aux_logits=True when loading pretrained weights;
        # load with aux_logits=True then disable the aux branch.
        base = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
        base.aux_logits = False
        base.AuxLogits = None

        if in_channels != 3:
            orig_weight = base.Conv2d_1a_3x3.conv.weight.data.clone()  # (32, 3, 3, 3)
            new_conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, bias=False)
            nn.init.zeros_(new_conv.weight)
            new_conv.weight.data[:, :3] = orig_weight
            base.Conv2d_1a_3x3.conv = new_conv

        self.conv1 = base.Conv2d_1a_3x3
        self.conv2 = base.Conv2d_2a_3x3
        self.conv3 = base.Conv2d_2b_3x3
        self.pool1 = base.maxpool1
        self.conv4 = base.Conv2d_3b_1x1
        self.conv5 = base.Conv2d_4a_3x3
        self.pool2 = base.maxpool2
        self.mixed5b = base.Mixed_5b
        self.mixed5c = base.Mixed_5c
        self.mixed5d = base.Mixed_5d
        self.mixed6a = base.Mixed_6a
        self.mixed6b = base.Mixed_6b
        self.mixed6c = base.Mixed_6c
        self.mixed6d = base.Mixed_6d
        self.mixed6e = base.Mixed_6e
        self.mixed7a = base.Mixed_7a
        self.mixed7b = base.Mixed_7b
        self.mixed7c = base.Mixed_7c
        self.pool3 = nn.AvgPool2d(kernel_size=3, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool2(x)
        x = self.mixed5b(x)
        x = self.mixed5c(x)
        x = self.mixed5d(x)
        x = self.mixed6a(x)
        x = self.mixed6b(x)
        x = self.mixed6c(x)
        x = self.mixed6d(x)
        x = self.mixed6e(x)
        x = self.mixed7a(x)
        x = self.mixed7b(x)
        x = self.mixed7c(x)
        x = self.pool3(x)
        return x.flatten(1)


class TimmBackbone(nn.Module):
    """timm-backed feature extractor for ablation experiments (e.g. ConvNeXt, Swin-V2)."""

    def __init__(self, model_name: str, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        import timm
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_channels,
            global_pool='avg',
        )
        self.OUT_FEATURES: int = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def create_backbone(name: str, in_channels: int = 3) -> nn.Module:
    """Factory: 'inception_v3' returns InceptionV3Backbone; anything else goes through timm."""
    if name == 'inception_v3':
        return InceptionV3Backbone(in_channels=in_channels)
    try:
        return TimmBackbone(name, in_channels=in_channels)
    except ImportError:
        raise ImportError(
            f"pip install timm  (required for backbone '{name}')"
        )
