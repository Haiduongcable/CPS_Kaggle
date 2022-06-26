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
from config.config_selfsupervised import config
from model.model import SingleNetwork
from dataloader.dataloader import VOC
from utils.init_func import init_weight, group_weight
from lr_policy import WarmUpPolyLR

from modules.seg_opr.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from utils.load_save_checkpoint import load_checkpoint, save_only_checkpoint

from torch.nn import BatchNorm2d
from dataloader.dataloader_context_restoration import get_loader
if config["use_wandb"]:
    import wandb

    os.environ["WANDB_API_KEY"] = "351cc1ebc0d966d49152a4c1937915dd4e7b4ef5"

    wandb.login(key="351cc1ebc0d966d49152a4c1937915dd4e7b4ef5")

    wandb.init(project = "Self supervised learning Medical Image")



    cudnn.benchmark = True

seed = config.seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)


#Load dataset
train_path = "../Dataset/TrainDataset"
train_image_root = train_path +  "/image/"
train_loader = get_loader(train_image_root, batchsize=config.batch_size, trainsize=352)    

criterion = nn.MSELoss()
    
# define and init the model
num_classes = 3 
norm_layer = BatchNorm2d
model = SingleNetwork(num_classes, norm_layer, pretrained_model=config.pretrained_model)

init_weight(model.business_layer, nn.init.kaiming_normal_,
            BatchNorm2d, config.bn_eps, config.bn_momentum,
            mode='fan_in', nonlinearity='relu')


base_lr = config.lr
params_list = []
params_list = group_weight(params_list, model.backbone,
                            BatchNorm2d, base_lr)
for module in model.business_layer:
    params_list = group_weight(params_list, module, BatchNorm2d,
                                base_lr)        # head lr * 10
optimizer = torch.optim.SGD(params_list,
                            lr=base_lr,
                            momentum=config.momentum,
                            weight_decay=config.weight_decay)
num_iter_per_epoch = len(train_loader)
total_iteration = config.nepochs * num_iter_per_epoch
lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, num_iter_per_epoch * config.warm_up_epoch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print('begin train')

s_epoch = 0
for epoch in range(s_epoch, 256):
    model.train()
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    number_iter_per_epoch = len(train_loader)
    pbar = tqdm(range(number_iter_per_epoch), file=sys.stdout, bar_format=bar_format)
    dataloader = iter(train_loader)
    total_loss = 0
    ''' supervised part '''
    for idx in pbar:
        optimizer.zero_grad()
        start_time = time.time()

        minibatch = dataloader.next()
        imgs, gts = minibatch
        imgs = imgs.cuda()
        gts = gts.cuda()
        
        b, c, h, w = imgs.shape
        _, pred = model(imgs)
        # pred = F.sigmoid(pred)
        loss = criterion(pred, gts)
        total_loss += loss.item()
        current_idx = epoch * number_iter_per_epoch + idx
        lr = lr_policy.get_lr(current_idx)

        # reset the learning rate
        optimizer.param_groups[0]['lr'] = lr
        optimizer.param_groups[1]['lr'] = lr
        for i in range(2, len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = lr

        loss.backward()
        optimizer.step()
    total_loss = total_loss / len(pbar)
    print("Epoch {} Loss training: {}".format(epoch, total_loss))
    if config["use_wandb"]:
        wandb.log({"Self supervised loss":  total_loss,"epoch": epoch})
    path_save_checkpoint = "self_supervised_weight/epoch_{}.pth".format(epoch)
    save_only_checkpoint(path_save_checkpoint, model)
        
    
    
