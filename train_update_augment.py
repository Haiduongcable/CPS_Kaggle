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
from dataloader import get_train_loader
# from model_efficientnet_backbone import Network
from model import Network
from dataloader_update_augmentation import VOC
from utils.init_func import init_weight, group_weight
from lr_policy import WarmUpPolyLR

from modules.seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from utils.load_save_checkpoint import load_checkpoint, save_checkpoint
# from seg_opr.sync_bn import DataParallelModel
from torch.nn import BatchNorm2d
from tensorboardX import SummaryWriter

import wandb

os.environ["WANDB_API_KEY"] = "351cc1ebc0d966d49152a4c1937915dd4e7b4ef5"

wandb.login(key="351cc1ebc0d966d49152a4c1937915dd4e7b4ef5")

wandb.init(project = "Cross Pseudo Label change learning rate + Augment")


cudnn.benchmark = True

seed = config.seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# data loader + unsupervised data loader
train_loader, train_sampler = get_train_loader(VOC, train_source=config.train_source, unsupervised=False)
unsupervised_train_loader, unsupervised_train_sampler = get_train_loader(VOC, train_source=config.unsup_source, unsupervised=True)

# config network and criterion
criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
criterion_csst = nn.MSELoss(reduction='mean')


# define and init the model
model = Network(config.num_classes, criterion=criterion,
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

# engine.register_state(dataloader=train_loader, model=model,
#                         optimizer_l=optimizer_l, optimizer_r=optimizer_r)
# if engine.continue_state_object:
#     engine.restore_checkpoint()

# model.train()
# network = Network(config.num_classes, criterion=criterion,
#                 pretrained_model=config.pretrained_model,
#                 norm_layer=BatchNorm2d)

# # define the learning rate
# base_lr = config.lr
# # define the two optimizers
# params_list_l = []
# params_list_l = group_weight(params_list_l, model.branch1.backbone,
#                             BatchNorm2d, base_lr)
# for module in model.branch1.business_layer:
#     params_list_l = group_weight(params_list_l, module, BatchNorm2d,
#                                 base_lr)        # head lr * 10
# optimizer_l = torch.optim.SGD(params_list_l,
#                             lr=base_lr,
#                             momentum=config.momentum,
#                             weight_decay=config.weight_decay)
# params_list_r = []
# params_list_r = group_weight(params_list_r, model.branch2.backbone,
#                             BatchNorm2d, base_lr)
# for module in model.branch2.business_layer:
#     params_list_r = group_weight(params_list_r, module, BatchNorm2d,
#                                 base_lr)        # head lr * 10
# optimizer_r = torch.optim.SGD(params_list_r,
#                             lr=base_lr,
#                             momentum=config.momentum,
#                             weight_decay=config.weight_decay)
# # config lr policy
# total_iteration = config.nepochs * config.niters_per_epoch
# lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

print('begin train')

# model.train()
# dataloader = iter(train_loader)
# unsupervised_dataloader = iter(unsupervised_train_loader)
# minibatch = dataloader.next()
# unsup_minibatch = unsupervised_dataloader.next()
# imgs = minibatch['data']
# gts = minibatch['label']
# unsup_imgs = unsup_minibatch['data']
# imgs = imgs.cuda(non_blocking=True)
# unsup_imgs = unsup_imgs.cuda(non_blocking=True)
# gts = gts.cuda(non_blocking=True)


# b, c, h, w = imgs.shape
# _, pred_sup_l = model(imgs, step=1)
# print(pred_sup_l[0,0,:10,:10])
s_epoch = 0

# save_checkpoint(model, optimizer_l, optimizer_r, epoch)
# model , optimizer_l, optimizer_r, s_epoch = \
#     load_checkpoint("/kaggle/input/pretrained-cps/checkpoint_epoch_14.pth", network, optimizer_l, optimizer_r, s_epoch)
model.train()
# _, pred_sup_l = model(imgs, step=1)
# print(pred_sup_l[0,0,:10,:10])
for epoch in range(s_epoch, config.nepochs):
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'

    # if is_debug:
    #     pbar = tqdm(range(10), file=sys.stdout, bar_format=bar_format)
    # else:
    pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout, bar_format=bar_format)


    dataloader = iter(train_loader)
    unsupervised_dataloader = iter(unsupervised_train_loader)

    sum_loss_sup = 0
    sum_loss_sup_r = 0
    sum_cps = 0

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
        _, max_l = torch.max(pred_l, dim=1)
        _, max_r = torch.max(pred_r, dim=1)
        max_l = max_l.long()
        max_r = max_r.long()
        cps_loss = criterion(pred_l, max_r) + criterion(pred_r, max_l)
        # dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
        cps_loss = cps_loss
        cps_loss = cps_loss * config.cps_weight

        ### standard cross entropy loss ###
        loss_sup = criterion(pred_sup_l, gts)
        # dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
        loss_sup = loss_sup

        loss_sup_r = criterion(pred_sup_r, gts)
        # dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
        loss_sup_r = loss_sup_r

        unlabeled_loss = False

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

        loss = loss_sup + loss_sup_r + cps_loss
        # print(loss_sup.item())
        loss.backward()
        optimizer_l.step()
        optimizer_r.step()

        # print_str = 'Epoch{}/{}'.format(epoch, config.nepochs) \
        #             + ' Iter{}/{}:'.format(idx + 1, config.niters_per_epoch) \
        #             + ' lr=%.2e' % lr \
        #             + ' loss_sup=%.2f' % loss_sup.item() \
        #             + ' loss_sup_r=%.2f' % loss_sup_r.item() \
        #                 + ' loss_cps=%.4f' % cps_loss.item()

        sum_loss_sup += loss_sup.item()
        sum_loss_sup_r += loss_sup_r.item()
        sum_cps += cps_loss.item()
        # pbar.set_description(print_str, refresh=False)

        end_time = time.time()
    save_checkpoint(model, optimizer_l, optimizer_r, epoch)
    wandb.log({"Supervised Training Loss":  sum_loss_sup / len(pbar)})
    wandb.log({"Supervised Training Loss right":  sum_loss_sup_r / len(pbar)})
    wandb.log({"Supervised Training Loss CPS":  sum_cps / len(pbar)})
