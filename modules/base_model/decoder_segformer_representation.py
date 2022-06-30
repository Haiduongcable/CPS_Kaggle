# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
import torch.nn.functional as F

def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None):
    if isinstance(size, torch.Size):
        size = tuple(int(x) for x in size)
    
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x 

class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__()
        self.in_channels = kwargs['in_channels']
        self.in_index = kwargs['in_index']
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, kwargs['num_classes'], kernel_size=1)
        self.dropout = nn.Dropout2d(kwargs['dropout_ratio'])
        
        self.representation = nn.Sequential(
            nn.Conv2d(embedding_dim, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1)
        )

    def _transform_inputs(self, inputs):
        '''
        Transform input
        '''
        inputs = [inputs[i] for i in self.in_index]
        return inputs
    
    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        if not self.training:
            x = self.dropout(_c)
            x = self.linear_pred(x)
            return x 
        
        feature_representation = self.representation(_c)
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x, feature_representation


if __name__ == '__main__':
    device = torch.device("cuda")
    decoder_head = SegFormerHead(feature_strides=[4, 8, 16, 32], in_channels=[64, 128, 320, 512],dropout_ratio=0.1,decoder_params=dict(embed_dim=768), num_classes=150, in_index = [0, 1, 2, 3])
    decoder_head.to(device)
   
    # model.to(device)
    #backbone.eval()
    # summary(backbone, (3,128,128))
    feature_1 = torch.randn(4,64,128,128).to(device)
    feature_2 = torch.randn(4,128,64,64).to(device)
    feature_3 = torch.randn(4,320,32,32).to(device)
    feature_4 = torch.randn(4,512,16,16).to(device)
    input_feature = [feature_1, feature_2, feature_3, feature_4]
    pred, representation = decoder_head(input_feature)
    print(pred.shape)
    print(representation.shape)
