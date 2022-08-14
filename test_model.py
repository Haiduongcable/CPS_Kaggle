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

from model.model_segformer_representation_contrastive import NetworkSegformerRepresentation
from dataloader.dataloader import VOC
from utils.init_func import init_weight, group_weight
from lr_policy import WarmUpPolyLR

from utils.load_save_checkpoint import save_bestcheckpoint
from torch.nn import BatchNorm2d
from evaluation.eval_function import SegEvaluator
from dataloader.dataloader import VOC
from dataloader.dataloader import ValPre


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

s_epoch = 0
print(config.niters_per_epoch)
print(len(train_loader))
print(len(unsupervised_train_loader))
for epoch in range(s_epoch, config.total_epoch):
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
        minibatch = dataloader.next()
        unsup_minibatch = unsupervised_dataloader.next()
        imgs = minibatch['data']
        gts = minibatch['label']
        unsup_imgs = unsup_minibatch['data']
       
