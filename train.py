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
from dataloader.dataloader import get_train_loader
# from model_efficientnet_backbone import Network
from model.model import Network
from dataloader.dataloader import VOC
from utils.init_func import init_weight, group_weight
from lr_policy import WarmUpPolyLR

from modules.seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from utils.load_save_checkpoint import load_checkpoint, save_checkpoint
from utils.losses import dice_loss, structure_loss
# from seg_opr.sync_bn import DataParallelModel
'''
Eval import
'''
from eval_function import SegEvaluator
from dataloader.dataloader import VOC
from dataloader.dataloader import ValPre

from torch.nn import BatchNorm2d
from tensorboardX import SummaryWriter

import wandb

os.environ["WANDB_API_KEY"] = "351cc1ebc0d966d49152a4c1937915dd4e7b4ef5"

wandb.login(key="351cc1ebc0d966d49152a4c1937915dd4e7b4ef5")

wandb.init(project = "Cross Pseudo Label ratio label 4")


cudnn.benchmark = True

seed = config.seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# data loader + unsupervised data loader
train_loader, train_sampler = get_train_loader(VOC, train_source=config.train_source, unsupervised=False)
unsupervised_train_loader, unsupervised_train_sampler = get_train_loader(VOC, train_source=config.unsup_source, unsupervised=True)

# config network and crooss entropy loss
cross_entropy = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)



# define and init the model
model = Network(config.num_classes,
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
lambda_cross_entropy = 2

for epoch in range(s_epoch, config.nepochs):
    model.train()
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

    # if is_debug:
    #     pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
    # else:
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)


    dataloader = iter(train_loader)
    unsupervised_dataloader = iter(unsupervised_train_loader)

    sum_loss_sup_l = 0
    sum_loss_sup_r = 0
    sum_cps = 0
    sum_cross_entropy_loss_sup_l = 0
    sum_dice_loss_sup_l = 0
    sum_cross_entropy_loss_sup_r = 0
    sum_dice_loss_sup_r = 0

    ''' supervised part '''
    for idx in pbar:
        optimizer_l.zero_grad()
        optimizer_r.zero_grad()
        start_time = time.time()

        minibatch = dataloader.next()
        unsup_minibatch = unsupervised_dataloader.next()
        imgs = minibatch['data']
        gts = minibatch['label']
        unsup_imgs = unsup_minibatch['data']
        imgs = imgs.cuda(non_blocking=True)
        unsup_imgs = unsup_imgs.cuda(non_blocking=True)
        gts = gts.cuda(non_blocking=True)


        b, c, h, w = imgs.shape
        _, pred_sup_l = model(imgs, step=1)
        _, pred_unsup_l = model(unsup_imgs, step=1)
        _, pred_sup_r = model(imgs, step=2)
        _, pred_unsup_r = model(unsup_imgs, step=2)

        ### cps loss ###
        pred_l = torch.cat([pred_sup_l, pred_unsup_l], dim=0)
        pred_r = torch.cat([pred_sup_r, pred_unsup_r], dim=0)
        _, max_l = torch.max(pred_l, dim=1) #( b * c* h * w)
        _, max_r = torch.max(pred_r, dim=1)
        max_l = max_l.long()
        max_r = max_r.long()
        
        #calculate loss
    #     test_loss = dice_loss(max_sup_r, gts)
    #     print(test_loss)
    #     break
    # break
        #cross entropy loss
        cps_crossentropy_loss = cross_entropy(pred_l, max_r) + cross_entropy(pred_r, max_l)
        #cross pseudo label diceloss
        cps_dice_loss = dice_loss(max_r, max_l)
        # print("Cps dice loss: ", cps_dice_loss)

        cps_loss = (lambda_cross_entropy * cps_crossentropy_loss + cps_dice_loss) * config.cps_weight

        ### standard cross entropy loss ###
        crossentropy_sup_l_loss = cross_entropy(pred_sup_l, gts)
        dice_sup_l_loss = dice_loss(max_sup_l, gts)
        # print("Supervised dice loss left: ", dice_sup_l_loss)

        loss_sup_l = lambda_cross_entropy * crossentropy_sup_l_loss + dice_sup_l_loss

        cross_entropy_sup_r_loss = cross_entropy(pred_sup_r, gts)
        dice_sup_r_loss = dice_loss(max_sup_r, gts)
        # print("Supervised dice loss right: ", dice_sup_r_loss)

        loss_sup_r = lambda_cross_entropy * cross_entropy_sup_r_loss + dice_sup_r_loss

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

        loss = loss_sup_l + loss_sup_r + cps_loss
        # print(loss_sup.item())
        loss.backward()
        optimizer_l.step()
        optimizer_r.step()
        sum_loss_sup_l += loss_sup_l.item()
        sum_loss_sup_r += loss_sup_r.item()
        sum_cps += cps_loss.item()
        
        #Log Iter loss:
        iter_cross_entropy_loss = (cps_crossentropy_loss.item() +\
                                 crossentropy_sup_l_loss.item() +\
                                 cross_entropy_sup_r_loss.item())/3
        iter_dice_loss = (cps_dice_loss.item() + dice_sup_l_loss.item() +\
                         dice_sup_r_loss.item())/3
        sum_cross_entropy_loss_sup_l += crossentropy_sup_l_loss.item()
        sum_cross_entropy_loss_sup_r += cross_entropy_sup_r_loss.item()
        sum_dice_loss_sup_r += dice_sup_r_loss.item()
        sum_dice_loss_sup_l += dice_sup_l_loss.item()
        # print("Cross entropy loss in iter {}:   {}".format(idx, iter_cross_entropy_loss))
        # print("Dice loss in iter {}:    {}".format(idx, iter_dice_loss))

        end_time = time.time()
    #Eval 

    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}

    val_pre = ValPre()
    dataset = VOC(data_setting, 'val', val_pre, training=False)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, None,
                                 config.eval_scale_array, config.eval_flip,
                                 ["cuda"], False, None,
                                 False)
        m_IOU_deeplabv3_1 = segmentor.run_model(model.branch2)
        m_IOU_deeplabv3_2 = segmentor.run_model(model.branch1)
    print("mIOU deeplabv3 branch 1",m_IOU_deeplabv3_1)
    print("mIOU deeplabv3 branch 1", m_IOU_deeplabv3_2)
    save_checkpoint(model, optimizer_l, optimizer_r, epoch)

    wandb.log({"Supervised Training Loss Deeplabv3+ 1":  sum_loss_sup_l / len(pbar)})
    wandb.log({"Supervised Training Loss Deeplabv3+ 2":  sum_loss_sup_r / len(pbar)})

    wandb.log({"Supervised Cross Entropy Loss Deeplabv3+ 1":  sum_cross_entropy_loss_sup_l / len(pbar)})
    wandb.log({"Supervised Cross Entropy Loss Deeplabv3+ 2":  sum_cross_entropy_loss_sup_r / len(pbar)})

    wandb.log({"Supervised Dice Loss Deeplabv3+ 1":  sum_dice_loss_sup_l / len(pbar)})
    wandb.log({"Supervised  Dice Loss Deeplabv3+ 2":  sum_dice_loss_sup_r / len(pbar)})

    wandb.log({"Supervised Training Loss CPS":  sum_cps / len(pbar)})
    wandb.log({"mIOU val deeplabv3 1": m_IOU_deeplabv3_1})
    wandb.log({"mIOU val deeplabv3 2": m_IOU_deeplabv3_2})

    