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
C.use_wandb = True
''' Logging'''
C.snapshot_dir = "Log"
C.log_dir = "Log"
exp_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
C.log_file = C.log_dir + '/log_' + exp_time + '.log'
C.link_log_file = C.log_file + '/log_last.log'
C.val_log_file = C.log_dir + '/val_' + exp_time + '.log'
C.link_val_log_file = C.log_dir + '/val_last.log'

'''Contrastive loss'''
#config contrastive loss and unsupervised loss
C.drop_percent = 80 
C.ratio_aug_cutmix = 0.4
C.unsupervised_loss_weight = 1
C.low_entropy_threshold = 20
C.current_class_threshold = 0.3 
C.current_class_negative_threshold = 1
C.low_rank = 3
C.high_rank = 20 
C.temperature = 0.5
C.num_queries = 256
C.num_negatives = 50


""" Data Dir and Weight Dir """
C.dataset_path = "/home/asilla/duongnh/project/Analys_COCO/tmp_folder/DATA_CPS/pascal_voc"
C.img_root_folder = "/home/asilla/duongnh/project/Analys_COCO/tmp_folder/DATA_CPS/pascal_voc"
C.gt_root_folder = "/home/asilla/duongnh/project/Analys_COCO/tmp_folder/DATA_CPS/pascal_voc"
C.pretrained_model = "/home/asilla/duongnh/project/Analys_COCO/tmp_folder/DATA_CPS/pytorch-weight/resnet50_v1c.pth"
C.path_save_checkpoint = "weights"


''' Experiments Setting '''
C.labeled_ratio = 4   # ratio of labeled set
C.train_source = osp.join(C.dataset_path, "subset_train_aug/train_aug_labeled_1-{}.txt".format(C.labeled_ratio))
C.unsup_source = osp.join(C.dataset_path, "subset_train_aug/train_aug_unlabeled_1-{}.txt".format(C.labeled_ratio))

C.eval_source = osp.join(C.dataset_path, "val.txt")
C.is_test = False
C.fix_bias = True
C.bn_eps = 1e-5
C.bn_momentum = 0.1

''' Image Config '''
C.num_classes = 21
C.background = 0
C.image_mean = np.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
C.image_std = np.array([0.229, 0.224, 0.225])
C.image_height = 512
C.image_width = 512
C.num_train_imgs = 10582 // C.labeled_ratio
C.num_eval_imgs = 1449
C.num_unsup_imgs = 10582 - C.num_train_imgs     # unsupervised samples

"""Train Config"""
if os.getenv('learning_rate'):
    C.lr = float(os.environ['learning_rate'])
else:
    C.lr = 0.00075

C.batch_size = 4
C.lr_power = 0.9
C.momentum = 0.9
C.weight_decay = 1e-4



#C.nepochs = 50
C.total_epoch = 64

#Test code
C.max_samples = max(C.num_train_imgs, C.num_unsup_imgs)    

# C.max_samples = 300

# Define the iterations in an epoch
C.cold_start = 0
C.niters_per_epoch = int(math.ceil(C.max_samples * 1.0 // C.batch_size))
C.num_workers = 8
C.train_scale_array = [0.5, 0.75, 1, 1.5, 1.75, 2.0]
C.warm_up_epoch = 0

''' Eval Config '''
C.eval_iter = 30
C.eval_stride_rate = 2 / 3
C.eval_scale_array = [1] #[1, 0.75, 1.25]
C.eval_flip = False
C.eval_base_size = 512
C.eval_crop_size = 512

"""Display Config"""
if os.getenv('snapshot_iter'):
    C.snapshot_iter = int(os.environ['snapshot_iter'])
else:
    C.snapshot_iter = 2

C.record_info_iter = 20
C.display_iter = 50