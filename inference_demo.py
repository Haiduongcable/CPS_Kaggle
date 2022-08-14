from __future__ import division
import os.path as osp
import os
import sys
import time
import argparse
import math
from tqdm import tqdm
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn import BatchNorm2d
from config.config_contrastive_unreliable import config

from utils.pyt_utils import ensure_dir
from utils.visualize import show_img
from utils.evaluator import Evaluator
from utils.metric import hist_info
from PIL import Image


from model.model_segformer_representation_contrastive import NetworkSegformerRepresentation

# from model.model_deeplabv3_representation_constrative import Network
# from model.model import Network
from dataloader.dataloader import VOC

from utils.load_save_checkpoint import save_bestcheckpoint, load_only_checkpoint

from evaluation.eval_function import SegEvaluator
from dataloader.dataloader import VOC
from dataloader.dataloader import ValPre
import cv2

def filter_entropy(pred, threshold_entropy):
    
    prob = torch.softmax(pred, dim=0)
    print("Entropy prob: ", prob.shape)
    pred_class = prob.argmax(0)
    entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=0)
    print("Entropy entropy: ", entropy.shape)
    print("Entropy pred class", pred_class.shape)
    percent = 80
    thresh = np.percentile(
        entropy[pred_class != 255].detach().cpu().numpy().flatten(), percent)
    thresh_mask = entropy.ge(thresh).bool() * (pred_class != 255).bool()
    pred_class[thresh_mask] = 255
    return pred_class.numpy()


# def get_class_colors(*args):
#     def uint82bin(n, count=8):
#         """returns the binary of integer n, count refers to amount of bits"""
#         return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])
#     N = 21
#     cmap = np.zeros((N, 3), dtype=np.uint8)
#     for i in range(N):
#         r, g, b = 0, 0, 0
#         id = i
#         for j in range(7):
#             str_id = uint82bin(id)
#             r = r ^ (np.uint8(str_id[-1]) << (7 - j))
#             g = g ^ (np.uint8(str_id[-2]) << (7 - j))
#             b = b ^ (np.uint8(str_id[-3]) << (7 - j))
#             id = id >> 3
#         cmap[i, 0] = r
#         cmap[i, 1] = g
#         cmap[i, 2] = b
#     class_colors = cmap.tolist()
#     return class_colors[1:]

def convert_predict_2_class_color(predict_map):
    predict_map = np.expand_dims(predict_map, axis = 2)
    # print("Expand shape", np.shape(predict_map))
    # predict_map_3_channel = np.concatenate((predict_map,predict_map, predict_map), axis=2)
    # print(np.shape(predict_map_3_channel))
    l_class_color = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],
                [128, 0, 0], [128, 0, 128], [128, 128, 0],
                [128, 128, 128],
                [0, 0, 64], [0, 0, 192], [0, 128, 64],
                [0, 128, 192],
                [128, 0, 64], [128, 0, 192], [128, 128, 64],
                [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0],
                [0, 192, 128], [128, 64, 0]]

    l_class_name = ['background', 'aeroplane', 'bicycle', 'bird',
                'boat',
                'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable',
                'dog', 'horse', 'motorbike', 'person',
                'pottedplant',
                'sheep', 'sofa', 'train', 'tv/monitor']
    visualize_map = np.zeros((np.shape(predict_map)[0], np.shape(predict_map)[1],3), dtype=np.uint8) + 255
    
    for index_class in range(len(l_class_color)):
        visualize_map = np.where(predict_map == index_class , l_class_color[index_class], visualize_map)
    return visualize_map

def normalize(img, mean, std):
    img = img.astype(np.float32) / 255.0
    img = img - mean
    img = img / std

    return img

def preprocess(image):
    image_height = 512
    image_width = 512
    image = cv2.resize(image, (512, 512))
    image_mean = np.array([0.485, 0.456, 0.406])
    image_std = np.array([0.229, 0.224, 0.225])
    image = normalize(image, image_mean, image_std)
    return image




def inference_model(model_cps, image, device):
    image_processed = preprocess(image)
    print(np.shape(image_processed))
    image_processed = image_processed.transpose(2, 0, 1)
    input_data = np.ascontiguousarray(image_processed[None, :, :, :],
                                          dtype=np.float32)
    input_data = torch.FloatTensor(input_data).cuda(device)
    print(input_data.shape)
    with torch.no_grad():
        score = model_cps(input_data)
        log_score = score.detach().cpu()[0]
        print(score.shape)
        score = score[0]
        pred = score.argmax(0).cpu().numpy()
        
    return pred, log_score


path_pretrained_mitb2 = 'pretrained/mit_b2.pth'
network =  NetworkSegformerRepresentation(config.num_classes, pretrained = path_pretrained_mitb2)
# network  = Network(config.num_classes,
#                 pretrained_model=None,
#                 norm_layer=BatchNorm2d, type_backbone='resnet50')
# network  = Network(config.num_classes,
#                 pretrained_model=None,
#                 norm_layer=BatchNorm2d)
path_checkpoint = "weights/best_contrastive_segformer_b2_update_30_07.pth"

model = load_only_checkpoint(path_checkpoint, network)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

path_input_image = "/home/asilla/duongnh/project/CrossPseudo_UpdateBranch/DATA_CPS/pascal_voc/val/image/2007_000042.jpg"
image = cv2.imread(path_input_image)
H, W, _ = np.shape(image)
print(np.shape(image))
pred, prob = inference_model(model, image, device)

pred_threshold_entropy = filter_entropy(prob, 20)

print("Pred: ",np.shape(pred_threshold_entropy))
# pred = np.array(pred_threshold_entropy, dtype = np.uint8)

visualize_pred = convert_predict_2_class_color(pred_threshold_entropy.astype(np.uint8))
visualize_pred = np.array(visualize_pred, dtype=np.uint8)
print(np.shape(visualize_pred))
visualize_pred = cv2.resize(visualize_pred, (W, H))
cv2.imwrite("Log_result/test_pred.png", visualize_pred)
# result_img = Image.fromarray(pred_threshold_entropy.astype(np.uint8), mode='P')

# class_colors = get_class_colors()
# palette_list = list(np.array(class_colors).flat)
# if len(palette_list) < 768:
#     palette_list += [0] * (768 - len(palette_list))
# result_img.putpalette(palette_list)
# result_img = result_img.resize((W, H))
# result_img.save("Log_result/test_pred.png")
# np_result_img = np.array(result_img)




