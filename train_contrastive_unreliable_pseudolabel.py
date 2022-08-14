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

from config.config_contrastive_unreliable import config
from dataloader.dataloader import get_train_loader

from model.model_deeplabv3_representation_constrative import Network
from dataloader.dataloader import VOC
from utils.init_func import init_weight, group_weight
from lr_policy import WarmUpPolyLR

from utils.load_save_checkpoint import save_bestcheckpoint
from torch.nn import BatchNorm2d
from evaluation.eval_function import SegEvaluator
from dataloader.dataloader import VOC
from dataloader.dataloader import ValPre
from utils.train_step_contrastive_learning import train_step
if config.use_wandb:
    import wandb

    os.environ["WANDB_API_KEY"] = "88d5e168a5043d5ca6a1d3e4050ec957a3e702d4"

    wandb.login(key="88d5e168a5043d5ca6a1d3e4050ec957a3e702d4")

    wandb.init(project = "Cross Pseudo Label Deeplabv3+ ResNet 101 Unreliable PseudoLabel Contrastive learning ")


cudnn.benchmark = True

seed = config.seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# data loader + unsupervised data loader
train_loader, train_sampler = get_train_loader(VOC, train_source=config.train_source, unsupervised=False)
unsupervised_train_loader, unsupervised_train_sampler = get_train_loader(VOC, train_source=config.unsup_source, unsupervised=True)

# config network and criterion
supervised_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)



# define and init the model
model = Network(config.num_classes,
                pretrained_model=config.pretrained_model,
                norm_layer=BatchNorm2d, type_backbone='resnet50')
init_weight(model.branch1.business_layer, nn.init.kaiming_normal_,
            BatchNorm2d, config.bn_eps, config.bn_momentum,
            mode='fan_in', nonlinearity='relu')
init_weight(model.branch2.business_layer, nn.init.kaiming_normal_,
            BatchNorm2d, config.bn_eps, config.bn_momentum,
            mode='fan_in', nonlinearity='relu')


base_lr = config.lr
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

total_iteration = config.total_epoch * config.niters_per_epoch
lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print('begin train')
s_epoch = 0

memory_bank = []
queue_ptrlis = []
queue_size = []
for i in range(config.num_classes):
    memory_bank.append([torch.zeros(0, 256)])
    queue_size.append(30000)
    queue_ptrlis.append(torch.zeros(1, dtype=torch.long))
queue_size[0] = 50000
path_save = "weights/last_model_contrastive_unreliable_resnet50.pth"
best_mIOU = 0 
for epoch in range(s_epoch, config.total_epoch):
    model.train()
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

    pbar = tqdm(range(config.niters_per_epoch))


    dataloader = iter(train_loader)
    unsupervised_dataloader = iter(unsupervised_train_loader)

    sum_loss_sup_l = 0
    sum_loss_sup_r = 0
    sum_unsup_l = 0
    sum_unsup_r = 0
    sum_contrastive_loss_l = 0
    sum_contrastive_loss_r = 0

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
        sup_loss_l, unsup_loss_l, contrastive_loss_l =  train_step(model,imgs, gts, unsup_imgs,\
                                                                    epoch,supervised_criterion,\
                                                                    memory_bank, queue_ptrlis, queue_size,\
                                                                    trained_model = 'left')
        sup_loss_r, unsup_loss_r, contrastive_loss_r =  train_step(model,imgs, gts, unsup_imgs,\
                                                                    epoch,supervised_criterion,\
                                                                    memory_bank, queue_ptrlis, queue_size,\
                                                                    trained_model = 'right')
        
        
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

        loss = sup_loss_l + sup_loss_r + unsup_loss_l + unsup_loss_r + contrastive_loss_l + contrastive_loss_r
        loss.backward()
        optimizer_l.step()
        optimizer_r.step()
        sum_loss_sup_l += sup_loss_l.item()
        sum_loss_sup_r += sup_loss_r.item()
        sum_unsup_l += unsup_loss_l.item()
        sum_unsup_r += unsup_loss_r.item()
        sum_contrastive_loss_l += contrastive_loss_l.item()
        sum_contrastive_loss_r += contrastive_loss_r.item()

        end_time = time.time()

    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}

    val_pre = ValPre()
    dataset = VOC(data_setting, 'val', val_pre, training=False)
    model.eval()
    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, None,
                                 config.eval_scale_array, config.eval_flip,
                                 ["cuda"], False, None,
                                 False)
        m_IOU_deeplabv3_1, mDice_deeplabv3_1 = segmentor.run_model(model.branch2)
        m_IOU_deeplabv3_2, mDice_deeplabv3_2 = segmentor.run_model(model.branch1)
        average_mIOU = (m_IOU_deeplabv3_1 + m_IOU_deeplabv3_2)/2
        if average_mIOU > best_mIOU:
            best_mIOU = average_mIOU
            save_bestcheckpoint(model, optimizer_l, optimizer_r, "weights/best_contrastive_deeplab_resnet50.pth")
    print("mIOU deeplabv3 branch 1",m_IOU_deeplabv3_1)
    print("mIOU deeplabv3 branch 2", mDice_deeplabv3_2)
    print("mDice deeplabv3 branch 1",m_IOU_deeplabv3_1)
    print("mDice deeplabv3 branch 2", mDice_deeplabv3_2)
    save_bestcheckpoint(model, optimizer_l, optimizer_r, path_save)
    
    if config.use_wandb:
        wandb.log({"mIOU deeplabv3+ 1":  m_IOU_deeplabv3_1, "epoch": epoch})
        wandb.log({"mIOU deeplabv3+ 2":  m_IOU_deeplabv3_2, "epoch": epoch})
        wandb.log({"mDice deeplabv3+ 1":  mDice_deeplabv3_1, "epoch": epoch})
        wandb.log({"mDice deeplabv3+ 2":  mDice_deeplabv3_2, "epoch": epoch})
        wandb.log({"Supervised Training Loss left":  sum_loss_sup_l / len(pbar),"epoch": epoch})
        wandb.log({"Supervised Training Loss right":  sum_loss_sup_r / len(pbar),"epoch": epoch})
        wandb.log({"UnSupervised Training Loss left":  sum_unsup_l / len(pbar),"epoch": epoch})
        wandb.log({"UnSupervised Training Loss right":  sum_unsup_r / len(pbar),"epoch": epoch})
        wandb.log({"Contrastive Training Loss left":  sum_contrastive_loss_l / len(pbar),"epoch": epoch})
        wandb.log({"Contrastive Training Loss right":  sum_contrastive_loss_r / len(pbar),"epoch": epoch})
