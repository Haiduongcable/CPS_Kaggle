# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from functools import partial
from collections import OrderedDict
from config.config import config
from modules.base_model import resnet50
from modules.base_model.encoder_decoder_deeplabv3 import EncoderNetwork, DecoderNetwork, VATDecoderNetwork



class Network(nn.Module):
    def __init__(self, num_classes, norm_layer, pretrained_model=None):
        super(Network, self).__init__()
        self.encoder_1 = EncoderNetwork(num_classes=num_classes, norm_layer=norm_layer,\
                                        pretrained_model=pretrained_model, back_bone = 50)
        self.encoder_2 = EncoderNetwork(num_classes=num_classes, norm_layer=norm_layer,\
                                        pretrained_model=pretrained_model, back_bone = 50)
        
        self.decoder_1 = VATDecoderNetwork(num_classes=num_classes)
        self.decoder_2 = DecoderNetwork(num_classes=num_classes)

    def forward(self, data, step=1):
        b, c, h, w = data.shape
        if not self.training:
            encode_feature = self.encoder_2(data)
            output = self.decoder_2(encode_feature,  data_shape = (h,w))
            return output
        if step == 1:
            encode_feature = self.encoder_1(data)
            output = self.decoder_1(encode_feature, data_shape = (h,w) ,t_model = self.decoder_2)
        elif step == 2:
            encode_feature = self.encoder_2(data)
            output = self.decoder_2(encode_feature,  data_shape = (h,w))
        return output
            

if __name__ == '__main__':
    device = torch.device("cuda")
    model = Network(40, criterion=nn.CrossEntropyLoss(),
                    pretrained_model=None,
                    norm_layer=nn.BatchNorm2d)
   
    # model.to(device)
    model.eval()
    # summary(model, (3,128,128))
    left = torch.randn(2, 3, 128, 128)
    # left.to(device)
    # right = torch.randn(2, 3, 128, 128)

    # print(model.branch1)

    out_2 = model(left, step = 1)
    print(out_2.shape)

    out_1 = model(left, step = 2)
    print(out_1.shape)

