import torch
import torch.nn as nn
from src.models.encoder import ResNetEncoder
from src.models.decoder import DecoderBlock

class ResNetUnet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.encoder = ResNetEncoder(pretrained=pretrained)

        self.d1 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.d2 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.d3 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.d4 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=32)
        self.d5 = DecoderBlock(in_channels=32, skip_channels=3, out_channels=16)

        self.head = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)

    def forward(self, x):
        f0, f1, f2, f3, f4 = self.encoder(x)

        d1 = self.d1(f4, f3)
        d2 = self.d2(d1, f2)
        d3 = self.d3(d2, f1)
        d4 = self.d4(d3, f0)
        d5 = self.d5(d4, x)

        final = self.head(d5)

        return final

    
if __name__ == "__main__":
    model = ResNetUnet()
    out = model(torch.randn(2, 3, 256, 256))
    print(out.shape)
    print(sum(p.numel() for p in model.parameters()))