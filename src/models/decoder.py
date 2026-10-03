import torch
import torch.nn as nn
from src.models.double_conv import DoubleConv


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        # The resnet skips are differnt dimension
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):

        x = self.up(x)

        # Concatenate with the encoder's skip connection along the channel axis (dim=1)
        x = torch.concat([x, skip], dim=1)

        out = self.conv(x)

        return out