# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import partial
from collections import OrderedDict
from config import config
from modules.base_model import resnet50
from torchsummary import summary
from modules.base_model.segformer_representation import EncoderDecoderSegformer
from modules.base_model import resnet50, resnet101
from modules.base_model.deeplabv3_representation_constrative import Deeplabv3plus_representation
from torch.nn import BatchNorm2d

class NetworkSegformerDeeplabRepresentation(nn.Module):
    def __init__(self, num_classes,type_backbone_deeplabv3, pretrained_segformer = None, pretrained_deeplabv3 = None):
        super(NetworkSegformerDeeplabRepresentation, self).__init__()
        self.branch1 = Deeplabv3plus_representation(num_classes, BatchNorm2d,\
                             pretrained_deeplabv3 ,type_backbone= type_backbone_deeplabv3)
        self.branch2 = EncoderDecoderSegformer(backbone_cfg='mitb2',\
                             num_classes = num_classes, pretrained=pretrained_segformer)

    def forward(self, data, step=1):
        # if not self.training:
        #     pred1 = self.branch1(data)
        #     return pred1

        if step == 1:
            return self.branch1(data)
        elif step == 2:
            return self.branch2(data)


if __name__ == '__main__':
    device = torch.device("cuda")
    # model = Network(40, criterion=nn.CrossEntropyLoss(),
    #                 pretrained_model=None,
    #                 norm_layer=nn.BatchNorm2d)
   
    # # model.to(device)
    # model.eval()
    # # summary(model, (3,128,128))
    # left = torch.randn(2, 3, 128, 128)
    # # left.to(device)
    # # right = torch.randn(2, 3, 128, 128)

    # # print(model.branch1)

    # out_2 = model(left, step = 1)
    # print(out_2.shape)

    # out_1 = model(left, step = 2)
    # print(out_1.shape)

