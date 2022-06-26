from pip import main
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, SegformerConfig
from PIL import Image
import requests
import torch
# from utils_segformer_pascalVoc import id2label, label2id
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

def create_id_label():
    l_label = ['background', 'aeroplane', 'bicycle', 'bird',
                    'boat',
                    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable',
                    'dog', 'horse', 'motorbike', 'person',
                    'pottedplant',
                    'sheep', 'sofa', 'train', 'tv/monitor']
    id2label = {}
    label2id = {}
    for index, item_label in enumerate(l_label):
        id2label[index] = item_label
        label2id[item_label] = index
    return id2label,label2id

class SegFormer_Customize(nn.Module):
    def __init__(self, num_label):
        '''
        Customize model with custom num_labels, load pretrain from nvidia mit-b2. 
        
        '''
        if num_label == 1:
            id2label = {0: 'data'}
            label2id = {'data': 0}
        else:
            id2label,label2id = create_id_label()
        super(SegFormer_Customize, self).__init__()
        # print(id2label)
        self.backbone = SegformerForSemanticSegmentation.from_pretrained("nvidia/mit-b2",\
                        ignore_mismatched_sizes=True, num_labels=num_label,\
                        id2label=id2label, label2id=label2id,
                        reshape_last_stage=True)
    def forward(self, inputs):
        
        '''
        Customize forward of Segformer:
        Args: inputs: (b * c * h * w)
        Returns: outputs: (b * c * h * w)
        '''
        b, c, h, w = inputs.shape
        logits = self.backbone(inputs).logits
        outputs =  F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=True)
        return outputs

if __name__ == '__main__':
    device = torch.device("cuda")
    model = SegFormer_Customize(1)
   
    # model.to(device)
    model.eval()
    # summary(model, (3,128,128))
    left = torch.randn(4, 3, 512, 512)
    # left.to(device)
    # right = torch.randn(2, 3, 128, 128)

    # print(model.branch1)

    out = model(left)
    print(out.shape)