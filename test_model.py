import os
import torch 
import torch.nn as nn
import torch.nn.functional as F
from model.model_segformer_representation_contrastive import NetworkSegformerRepresentation
from config.config_contrastive_unreliable import config

device = torch.device("cuda")
path_pretrained = "pretrained/mit_b2.pth"
model = NetworkSegformerRepresentation(config.num_classes, pretrained = path_pretrained)
# model.to(device)
# model.eval()
# summary(model, (3,128,128))
left = torch.randn(2, 3, 128, 128)
# left.to(device)
# right = torch.randn(2, 3, 128, 128)

# print(model.branch1)

pred, rep = model(left)

print(pred.shape)
print(rep.shape)
# print(representation.shape)
