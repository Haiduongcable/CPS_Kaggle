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
from model.model_segformer_2_branch import Network
from dataloader.dataloader import VOC
from utils.init_func import init_weight, group_weight
from lr_policy import WarmUpPolyLR

from modules.seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from utils.load_save_checkpoint import load_checkpoint, save_bestcheckpoint

from torch.nn import BatchNorm2d
from tensorboardX import SummaryWriter
from dataloader.harnetmseg_loader import get_loader, test_dataset

from utils.losses import structure_loss

from torch.nn.functional import binary_cross_entropy_with_logits

if config["use_wandb"]:
    import wandb

    os.environ["WANDB_API_KEY"] = "88d5e168a5043d5ca6a1d3e4050ec957a3e702d4"

    wandb.login(key="88d5e168a5043d5ca6a1d3e4050ec957a3e702d4")

    wandb.init(project = "Medical Image Segformer setting threshold 0.3")



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
# path_pretrain_self_supervised = ".pth"
train_loader = get_loader(train_image_root, train_gts_root, batchsize=8,\
                        trainsize=352, augmentation = True, supervised = True)    

unsupervised_train_loader = get_loader(train_image_root, train_gts_root, batchsize=8,\
                        trainsize=352, augmentation = True, supervised = False)  


path_save = "medical_weight/best_segformer_threshold_0.3.pth"
# define and init the model
num_class = 1
model = Network(num_class)
# define the learning rate
base_lr = 0.01
# define the two optimizers
optimizer_l = torch.optim.SGD(model.branch1.parameters(),
                            lr=base_lr,
                            momentum=config.momentum,
                            weight_decay=config.weight_decay)

NUM_EPOCH = 128
NUM_ITER_PER_EPOCH = len(train_loader)

optimizer_r = torch.optim.SGD(model.branch2.parameters(),
                            lr=base_lr,
                            momentum=config.momentum,
                            weight_decay=config.weight_decay)
# config lr policy
total_iteration = NUM_EPOCH * NUM_ITER_PER_EPOCH
lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, NUM_ITER_PER_EPOCH * 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print('begin train')

s_epoch = 0

best_dice = -1
for epoch in range(s_epoch, NUM_EPOCH):
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
        pred_sup_l = model(imgs, step=1)
        pred_unsup_l = model(unsup_imgs, step=1)
        pred_sup_r = model(imgs, step=2)
        pred_unsup_r = model(unsup_imgs, step=2)

        ### cps loss ###
        pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
        pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
        # _, max_l = torch.max(pred_l, dim=1)
        # _, max_r = torch.max(pred_r, dim=1)
        # max_l = max_l.long()
        # max_r = max_r.long()
        with torch.no_grad():
            pseudo_gts_r = pred_r.sigmoid()
            pseudo_gts_r = torch.where(pseudo_gts_r > 0.3, torch.ones_like(pseudo_gts_r), torch.zeros_like(pseudo_gts_r))
            pseudo_gts_l = pred_l.sigmoid()
            pseudo_gts_l = torch.where(pseudo_gts_l > 0.3, torch.ones_like(pseudo_gts_r), torch.zeros_like(pseudo_gts_r))
        # loss_unsup, pass_rate, neg_loss = semi_ce_loss(pred_l, pseudo_gts_r)
        cps_loss = structure_loss(pred_l, pseudo_gts_r) + structure_loss(pred_r, pseudo_gts_l)
        # dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
        

        loss_sup_l = structure_loss(pred_sup_l, gts)
        loss_sup_r = structure_loss(pred_sup_r, gts)
        current_idx = epoch * NUM_ITER_PER_EPOCH + idx
        lr = lr_policy.get_lr(current_idx)

        # reset the learning rate
        optimizer_l.param_groups[0]['lr'] = lr
        optimizer_r.param_groups[0]['lr'] = lr
    
        loss = loss_sup_l + loss_sup_r + cps_loss 
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
            save_bestcheckpoint(model, optimizer_l, optimizer_r, path_save)
        print("Dice score:   ", meandice)
    
    if config["use_wandb"]:
        wandb.log({"mDice segformer":  meandice, "epoch": epoch})
        wandb.log({"Supervised Training Loss left":  sum_loss_sup_l / len(pbar),"epoch": epoch})
        wandb.log({"Supervised Training Loss right":  sum_loss_sup_r / len(pbar),"epoch": epoch})
        wandb.log({"Supervised Training Loss CPS":  sum_cps / len(pbar),"epoch": epoch})
