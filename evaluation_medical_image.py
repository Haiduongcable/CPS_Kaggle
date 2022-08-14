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
from config.config import config
from model.model import Network
# from model.model_deeplabv3_adv import Network
# from model.model_segformer_2_branch import Network
from model.model_segformer_representation_contrastive import NetworkSegformerRepresentation
# from model.model_segformer_deeplabv3 import Network
from dataloader.dataloader import VOC
from utils.init_func import init_weight, group_weight
from lr_policy import WarmUpPolyLR
import imageio
from modules.seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from utils.load_save_checkpoint import load_only_checkpoint, save_bestcheckpoint

import cv2
from torch.nn import BatchNorm2d
from tensorboardX import SummaryWriter
from evaluation.eval_function import SegEvaluator
from dataloader.harnetmseg_loader import get_loader, test_dataset

def test(model, path, device, path_save_dataset):
    
    ##### put ur data_path of TestDataSet/Kvasir here #####
    data_path = path
    #####                                             #####
    
    model.eval()
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, 352)
    b=0.0
    mIOU = 0.0
    for i in tqdm(range(test_loader.size)):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.to(device)
        
        res  = model(image, step=1)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        image_res = res
        log_res = np.array(image_res) * 255
        image_res = np.zeros((log_res.shape[0], log_res.shape[1], 3))
        image_res[:,:,0] = log_res
        image_res[:,:,1] = log_res
        image_res[:,:,2] = log_res
        image_res = np.uint8(image_res)
        
        
        res2  = model(image, step=2)
        
        res2 = F.upsample(res2, size=gt.shape, mode='bilinear', align_corners=False)
        res2 = res2.sigmoid().data.cpu().numpy().squeeze()
        res2 = (res2 - res2.min()) / (res2.max() - res2.min() + 1e-8)
        image_res2 = res2
        input = res2
        log_res2 = np.array(image_res2) * 255
        image_res2 = np.zeros((log_res2.shape[0], log_res2.shape[1], 3))
        image_res2[:,:,0] = log_res2
        image_res2[:,:,1] = log_res2
        image_res2[:,:,2] = log_res2
        image_res2 = np.uint8(image_res2)
        
        
        image_log = cv2.imread(image_root + name)
        
        
        target = np.array(gt)
        vs_gt = target * 255
        image_vs_gt = np.zeros((log_res.shape[0], log_res.shape[1], 3))
        image_vs_gt[:,:,0] = vs_gt
        image_vs_gt[:,:,1] = vs_gt
        image_vs_gt[:,:,2] = vs_gt
        image_vs_gt = np.uint8(image_vs_gt)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        color = (0, 0, 255)
        thickness = 5
        
        stack_image = np.zeros((image_log.shape[0],10, 3),dtype = np.uint8) + 255
        image_res = cv2.putText(image_res, 'Predict', (0, 50), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        image_vs_gt = cv2.putText(image_vs_gt, 'Groundtruth', (0, 50), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        
        im_vsall = cv2.hconcat([image_log,stack_image, image_res,stack_image,image_res2,stack_image, image_vs_gt])
        
        
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input,(-1))
        target_flat = np.reshape(target,(-1))
 
        intersection = (input_flat*target_flat)
        
        loss =  (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)
        IOU = float((intersection.sum() + smooth) / (input.sum() + target.sum() - intersection.sum() + smooth))
        a =  '{:.4f}'.format(loss)
        
        mIOU += IOU
        im_vsall = cv2.putText(im_vsall, "Dice: " + str(round(loss,2)), (0, 50), font, 
                        fontScale, color, thickness, cv2.LINE_AA)
        # if IOU < 0.3:
        cv2.imwrite(path_save_dataset+"/" + name, im_vsall)
        a = float(a)
        b = b + a
        
    return b/test_loader.size, mIOU/test_loader.size





num_class = 1
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
num_class = 1
network = NetworkSegformerRepresentation(num_class)
# network = Network(num_class,pretrained_model=config.pretrained_model, norm_layer=BatchNorm2d)

# path_resnet101 = "/home/asilla/duongnh/project/Analys_COCO/tmp_folder/DATA_CPS/pytorch-weight/resnet101_v1c.pth"
# # define and init the model
# num_classes = 1
# network = Network(num_classes, criterion=None,
#                 pretrained_model=path_resnet101,
#                 norm_layer=BatchNorm2d)
network.to(device)
# 


s_epoch = 0
lambda_cross_entropy = 1
model = load_only_checkpoint("weights/last_model_segformer_contrastive_unreliable.pth", network)
model.eval()

dir_test = "../Dataset/TestDataset"
path_save = "Visualize/"
for dataset in os.listdir(dir_test):
    path_test = dir_test + "/" + dataset
    path_save_dataset = path_save + "/" + dataset
    if not os.path.exists(path_save_dataset):
        os.mkdir(path_save_dataset)
    meanDice, meanIOU = test(model, path_test, device, path_save_dataset)
    print("Mean Dice of {} dataset: {}".format(dataset,meanDice))
    print("Mean IOU of {} dataset: {}".format(dataset,meanIOU))