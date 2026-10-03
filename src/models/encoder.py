import torch
import torchvision.models as models
import torch.nn as nn

class ResNetEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        
        weights = models.ResNet34_Weights.DEFAULT
        model = models.resnet34(weights=weights)
        
        self.layer0 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
        )
        self.maxpool = model.maxpool

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4


    def forward(self, x):
        f0 = self.layer0(x)

        f1 = self.layer1(self.maxpool(f0))

        f2 = self.layer2(f1)

        f3 = self.layer3(f2)

        f4 = self.layer4(f3)

        return [f0, f1, f2, f3, f4]


if __name__ == "__main__":
    enc = ResNetEncoder()

    feats = enc(torch.randn(1, 3, 256, 256))
    for f in feats:
        print(f.shape)
