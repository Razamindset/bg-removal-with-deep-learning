"""
Self-contained U-Net with ResNet-34 backbone for Background Removal / Matting.
Consolidated into a single readable file without nested folder hierarchies.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x 2"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    Upsampling via Bilinear Interpolation + Conv to eliminate checkerboard artifacts,
    concatenates skip features, then refines with DoubleConv.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ResNetEncoder(nn.Module):
    """Extracts hierarchical multi-scale feature maps using pretrained ResNet-34."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        resnet = models.resnet34(weights=weights)

        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # [64, H/2, W/2]
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1                                         # [64, H/4, W/4]
        self.layer2 = resnet.layer2                                         # [128, H/8, W/8]
        self.layer3 = resnet.layer3                                         # [256, H/16, W/16]
        self.layer4 = resnet.layer4                                         # [512, H/32, W/32]

    def forward(self, x):
        f0 = self.layer0(x)
        f1 = self.layer1(self.maxpool(f0))
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f0, f1, f2, f3, f4]


class ResNetUNet(nn.Module):
    """Complete U-Net: ResNet-34 Backbone + Smooth Bilinear Decoder."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.encoder = ResNetEncoder(pretrained=pretrained)

        self.d1 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.d2 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.d3 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.d4 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)

        # Final upsample to original resolution (H/2 -> H)
        self.d5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DoubleConv(32, 16),
        )
        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        f0, f1, f2, f3, f4 = self.encoder(x)

        d1 = self.d1(f4, f3)
        d2 = self.d2(d1, f2)
        d3 = self.d3(d2, f1)
        d4 = self.d4(d3, f0)
        d5 = self.d5(d4)

        return self.head(d5)


if __name__ == "__main__":
    model = ResNetUNet(pretrained=False)
    dummy_input = torch.randn(2, 3, 256, 256)
    out = model(dummy_input)
    print("Input shape: ", dummy_input.shape)
    print("Output shape:", out.shape)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
