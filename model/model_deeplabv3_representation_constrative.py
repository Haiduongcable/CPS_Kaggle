# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import partial
from collections import OrderedDict
from config.config import config
from modules.base_model import resnet50, resnet101
from modules.base_model.deeplabv3_representation_constrative import Deeplabv3plus_representation
from torchsummary import summary

class Network(nn.Module):
    def __init__(self, num_classes, norm_layer, pretrained_model=None,type_backbone = 'resnet101'):
        super(Network, self).__init__()
        self.branch1 = Deeplabv3plus_representation(num_classes, norm_layer, pretrained_model, type_backbone)
        self.branch2 = Deeplabv3plus_representation(num_classes, norm_layer, pretrained_model,type_backbone)

    def forward(self, data, step=1):
        if not self.training:
            # print("evaluation")
            if step == 1:
                pred, _ = self.branch1(data)
            elif step == 2:
                pred, _ = self.branch2(data)
            return pred
        # print('training')
        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)


# if __name__ == '__main__':
#     device = torch.device("cuda")
#     model = Network(40, criterion=nn.CrossEntropyLoss(),
#                     pretrained_model="self_supervised_weight/epoch_76.pth",
#                     norm_layer=nn.BatchNorm2d)
   
#     # model.to(device)
#     model.eval()
#     # summary(model, (3,128,128))
#     left = torch.randn(2, 3, 128, 128)
#     # left.to(device)
#     # right = torch.randn(2, 3, 128, 128)

#     # print(model.branch1)

#     out = model(left)
#     print(out.shape)
