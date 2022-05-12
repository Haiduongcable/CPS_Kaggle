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
from config import config
from model.model import Network
from dataloader.dataloader import VOC
from utils.init_func import init_weight, group_weight
from lr_policy import WarmUpPolyLR

from modules.seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from utils.load_save_checkpoint import load_checkpoint, save_bestcheckpoint

from torch.nn import BatchNorm2d
from tensorboardX import SummaryWriter
from eval_function import SegEvaluator
from dataloader.harnetmseg_loader import get_loader, test_dataset

from utils.losses import structure_loss


import wandb

os.environ["WANDB_API_KEY"] = "351cc1ebc0d966d49152a4c1937915dd4e7b4ef5"

wandb.login(key="351cc1ebc0d966d49152a4c1937915dd4e7b4ef5")

wandb.init(project = "Medical Image Deeplabv3+")



cudnn.benchmark = True

seed = config.seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


def test(model, path):
    
    ##### put ur data_path of TestDataSet/Kvasir here #####
    data_path = path
    #####                                             #####
    
    model.eval()
    image_root = '{}/images/'.format(data_path)
    gt_root = '{}/masks/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, 352)
    b=0.0
    for i in range(100):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        
        res  = model(image, step=1)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        
        input = res
        target = np.array(gt)
        N = gt.shape
        smooth = 1
        input_flat = np.reshape(input,(-1))
        target_flat = np.reshape(target,(-1))
 
        intersection = (input_flat*target_flat)
        
        loss =  (2 * intersection.sum() + smooth) / (input.sum() + target.sum() + smooth)

        a =  '{:.4f}'.format(loss)
        a = float(a)
        b = b + a
        
    return b/100



#Load dataset
train_path = "../Dataset/TrainDataset"
test_path = "../Dataset/TestDataset/Kvasir"
train_image_root = train_path +  "/image"
train_gts_root = train_path + "/mask"
train_loader = get_loader(train_image_root, train_gts_root, batchsize=8,\
                        trainsize=352, augmentation = True, supervised = True)    

unsupervised_train_loader = get_loader(train_image_root, train_gts_root, batchsize=8,\
                        trainsize=352, augmentation = True, supervised = False)  


    
# define and init the model
num_class = 1
model = Network(num_class,
                pretrained_model=config.pretrained_model,
                norm_layer=BatchNorm2d)
init_weight(model.branch1.business_layer, nn.init.kaiming_normal_,
            BatchNorm2d, config.bn_eps, config.bn_momentum,
            mode='fan_in', nonlinearity='relu')
init_weight(model.branch2.business_layer, nn.init.kaiming_normal_,
            BatchNorm2d, config.bn_eps, config.bn_momentum,
            mode='fan_in', nonlinearity='relu')
# define the learning rate
base_lr = config.lr
# define the two optimizers
params_list_l = []
params_list_l = group_weight(params_list_l, model.branch1.backbone,
                            BatchNorm2d, base_lr)
for module in model.branch1.business_layer:
    params_list_l = group_weight(params_list_l, module, BatchNorm2d,
                                base_lr)        # head lr * 10
optimizer_l = torch.optim.SGD(params_list_l,
                            lr=base_lr,
                            momentum=config.momentum,
                            weight_decay=config.weight_decay)


params_list_r = []
params_list_r = group_weight(params_list_r, model.branch2.backbone,
                            BatchNorm2d, base_lr)
for module in model.branch2.business_layer:
    params_list_r = group_weight(params_list_r, module, BatchNorm2d,
                                base_lr)        # head lr * 10
optimizer_r = torch.optim.SGD(params_list_r,
                            lr=base_lr,
                            momentum=config.momentum,
                            weight_decay=config.weight_decay)
# config lr policy
total_iteration = config.nepochs * config.niters_per_epoch
lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print('begin train')

s_epoch = 0

best_dice = -1
for epoch in range(s_epoch, config.nepochs):
    model.train()
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    number_iter_per_epoch = len(train_loader)
    pbar = tqdm(range(number_iter_per_epoch), file=sys.stdout, bar_format=bar_format)


    dataloader = iter(train_loader)
    unsupervised_dataloader = iter(unsupervised_train_loader)

    sum_loss_sup_l = 0
    sum_loss_sup_r = 0
    sum_cps = 0

    ''' supervised part '''
    for idx in pbar:
        optimizer_l.zero_grad()
        optimizer_r.zero_grad()
        start_time = time.time()

        minibatch = dataloader.next()
        unsup_minibatch = unsupervised_dataloader.next()
        imgs, gts = minibatch
        unsup_imgs, _ = unsup_minibatch
        imgs = imgs.cuda()
        gts = gts.cuda()
        unsup_imgs = unsup_imgs.cuda()
        
        b, c, h, w = imgs.shape
        _, pred_sup_l = model(imgs, step=1)
        _, pred_unsup_l = model(unsup_imgs, step=1)
        _, pred_sup_r = model(imgs, step=2)
        _, pred_unsup_r = model(unsup_imgs, step=2)

        ### cps loss ###
        pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
        pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
        # _, max_l = torch.max(pred_l, dim=1)
        # _, max_r = torch.max(pred_r, dim=1)
        # max_l = max_l.long()
        # max_r = max_r.long()
        pseudo_gts_r = pred_r.sigmoid()
        pseudo_gts_l = pred_l.sigmoid()
        cps_loss = structure_loss(pred_l, pseudo_gts_r) + structure_loss(pred_r, pseudo_gts_l)
        # dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
        

        loss_sup_l = structure_loss(pred_sup_l, gts)
        loss_sup_r = structure_loss(pred_sup_r, gts)
        current_idx = epoch * config.niters_per_epoch + idx
        lr = lr_policy.get_lr(current_idx)

        # reset the learning rate
        optimizer_l.param_groups[0]['lr'] = lr
        optimizer_l.param_groups[1]['lr'] = lr
        for i in range(2, len(optimizer_l.param_groups)):
            optimizer_l.param_groups[i]['lr'] = lr
        optimizer_r.param_groups[0]['lr'] = lr
        optimizer_r.param_groups[1]['lr'] = lr
        for i in range(2, len(optimizer_r.param_groups)):
            optimizer_r.param_groups[i]['lr'] = lr

        loss = loss_sup_l + loss_sup_r + cps_loss * config.cps_weight
        # print(loss.item())
        loss.backward()
        optimizer_l.step()
        optimizer_r.step()
        sum_loss_sup_l += loss_sup_l.item()
        sum_loss_sup_r += loss_sup_r.item()
        sum_cps += cps_loss.item()
        # pbar.set_description(print_str, refresh=False)

        end_time = time.time()
    print("Losss supervised loss left epoch {}: {}".format(epoch, sum_loss_sup_l / number_iter_per_epoch))
    print("Losss supervised loss right epoch {}: {}".format(epoch, sum_loss_sup_r / number_iter_per_epoch))
    print("Losss crosspseudo loss epoch {}: {}".format(epoch, sum_cps / number_iter_per_epoch))
    with torch.no_grad():   
        meandice = test(model,test_path)
        if meandice > best_dice:
            bestdice = meandice
            #save checkpoint 
            save_bestcheckpoint(model, optimizer_l, optimizer_r)
        print("Dice score:   ", meandice)
    wandb.log({"mDice deeplabv3+":  meandice, "epoch": epoch})
    wandb.log({"Supervised Training Loss left":  sum_loss_sup_l / len(pbar),"epoch": epoch})
    wandb.log({"Supervised Training Loss right":  sum_loss_sup_r / len(pbar),"epoch": epoch})
    wandb.log({"Supervised Training Loss CPS":  sum_cps / len(pbar),"epoch": epoch})
