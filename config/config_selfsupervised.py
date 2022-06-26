# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import time
import math
import numpy as np
from easydict import EasyDict as edict
import argparse

C = edict()
config = C
cfg = C

C.seed = 12345
C.use_wandb = False

C.dataset_path = "/home/asilla/duongnh/project/Analys_COCO/tmp_folder/DATA_CPS/pascal_voc"
C.img_root_folder = "/home/asilla/duongnh/project/Analys_COCO/tmp_folder/DATA_CPS/pascal_voc"
C.gt_root_folder = "/home/asilla/duongnh/project/Analys_COCO/tmp_folder/DATA_CPS/pascal_voc"
C.pretrained_model = "/home/asilla/duongnh/project/Analys_COCO/tmp_folder/DATA_CPS/pytorch-weight/resnet50_v1c.pth"
C.path_save_checkpoint = "weights"

C.fix_bias = True
C.bn_eps = 1e-5
C.bn_momentum = 0.1

C.unsup_weight = 0
C.cps_weight = 1.5
C.momentum = 0.9
C.weight_decay = 1e-4
C.nepochs = 128
C.lr_power = 0.9
C.lr = 0.005
C.warm_up_epoch = 2
C.batch_size = 8