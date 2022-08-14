import os 
import numpy as np 
import time 
from tqdm import tqdm 
import torch 
import torch.nn.functional as F
# log_dataset = ["kvasir","etis","clinicdb","cvc-300","cvc-colondb"]
# log_weight = np.array([100, 196, 62, 60, 380])
# log_weight = log_weight / np.sum(log_weight)
# dice_iou = "0.823	0.8829	0.6292	0.7144	0.7623	0.8389	0.7723	0.8543	0.6354	0.7278"

# def get_dice_miou(str_dice_iou):
#     tmp_value = dice_iou.split("\t")
#     mDice = 0
#     mIOU = 0
#     for index in range(5):
#         mIOU += float(tmp_value[index * 2]) * log_weight[index]
#         mDice += float(tmp_value[index * 2+ 1]) * log_weight[index]
#     print("mIOU: ", mIOU)
#     print("mDice: ", mDice)
        
    

# get_dice_miou(dice_iou)

output_model = torch.rand(4,21,512,512)
prob = F.softmax(output_model, dim=1)

entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=1)
print(entropy.shape)
alpha_t = 0.3 
print(entropy.cpu().numpy().flatten().shape)
high_thresh = np.percentile(entropy.cpu().numpy().flatten(), 100 - alpha_t,)
print(high_thresh)