import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from modules.base_model.deeplabv3_representation_constrative import Deeplabv3plus_representation


device = torch.device("cuda")
model = Deeplabv3plus_representation(40,
                pretrained_model=None,
                norm_layer=nn.BatchNorm2d, type_backbone='resnet101')

# model.to(device)
model.eval()
# summary(model, (3,128,128))
left = torch.randn(2, 3, 128, 128)
# left.to(device)
# right = torch.randn(2, 3, 128, 128)

# print(model.branch1)

pred, representation = model(left)

print(pred)
print(pred.shape)
print(representation.shape)
