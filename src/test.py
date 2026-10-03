import torch
import torchvision.models as models

weights = models.ResNet34_Weights.DEFAULT
model = models.resnet34(weights=weights)

model.eval()

x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    print("Inout shape", x.shape)
    x = model.conv1(x)
    print("conv1", x.shape)

    x = model.bn1(x)
    print("bn1", x.shape)
    
    x = model.relu(x)
    print("relu", x.shape)

    x = model.maxpool(x)
    print("maxPool1", x.shape)

    # Layer 1
    x = model.layer1(x)
    print("layer 1", x.shape)

    x = model.layer2(x)
    print("layer 2", x.shape)

    x = model.layer3(x)
    print("layer 3", x.shape)

    x = model.layer4(x)
    print("layer 4", x.shape)


# Now if iw ant to use this as the backbone for my bg removal tool 
# i would need to connect these skip connections with the decoder layers...

# from wht i understand in u net we used the output of the conv bocks before relu i think 
# But here the e dont have access to individual output of each conv op

# for the bottom up appraoch maybe 
# as the input of the last layer is already added to the putout of it that we will pass throug the bottle neck 
# assuming we add another bottle neck maybe 
# we can take the output of each layer and concat it to the input of each upscale convolution maybe??