#!/usr/bin/env python3
# encoding: utf-8
import os
import cv2
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from utils.evaluator import Evaluator
from utils.metric import hist_info, compute_score
from dataloader.dataloader import VOC
from dataloader.dataloader import ValPre
from model.model import Network
from utils.dataset_process import get_class_colors
# from model_efficientnet_backbone import Network

import sys
import torch
from torch.utils.data import dataloader
from torch.multiprocessing import reductions
from multiprocessing.reduction import ForkingPickler
from PIL import Image

default_collate_func = dataloader.default_collate

def default_collate_override(batch):
  dataloader._use_shared_memory = False
  return default_collate_func(batch)

setattr(dataloader, 'default_collate', default_collate_override)

for t in torch._storage_classes:
  if sys.version_info[0] == 2:
    if t in ForkingPickler.dispatch:
        del ForkingPickler.dispatch[t]
  else:
    if t in ForkingPickler._extra_reducers:
        del ForkingPickler._extra_reducers[t]



class SegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']
        pred = self.sliding_eval(img, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp,
                        'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(self.save_path)
            ensure_dir(self.save_path+'_color')

            fn = name + '.png'

            'save colored result'
            result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            class_colors = get_class_colors()
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path+'_color', fn))

            'save raw result'
            cv2.imwrite(os.path.join(self.save_path, fn), pred)

        if True:
            
            colors = get_class_colors()
            print(len(colors))
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iu, mean_IU, _, mean_pixel_acc = compute_score(hist, correct,
                                                       labeled)
        # print(len(dataset.get_class_names()))
        result_line = print_iou(iu, mean_pixel_acc,
                                self.dataset.get_class_names(), True)
        meanIU = np.nanmean(iu)
        return result_line, meanIU
    
    def run_model(self, model):
        """There are four evaluation modes:
            1.only eval a .pth model: -e *.pth
            2.only eval a certain epoch: -e epoch
            3.eval all epochs in a given section: -e start_epoch-end_epoch
            4.eval all epochs from a certain started epoch: -e start_epoch-
            """
        model.eval()
        self.val_func = model
        result_line, meanIU = self.single_process_evalutation()
        return meanIU

    

def get_num_checkpoint(name_checkpoint):
    num_checkpoint = int(name_checkpoint[17:-4])
    return num_checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='1', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default=None)
    args = parser.parse_args()
    
    path_model = "/home/haiduong/Documents/DoAn/TorchSemiSeg/SaveCheckpoint/checkpoint_epoch_44_warmup.pth"
    path_folder_checkpoint = "weights"
    all_dev = ["cuda"]

    network = Network(config.num_classes, criterion=None, norm_layer=nn.BatchNorm2d)
    data_setting = {'img_root': config.img_root_folder,
                    'gt_root': config.gt_root_folder,
                    'train_source': config.train_source,
                    'eval_source': config.eval_source}
    val_pre = ValPre()
    dataset = VOC(data_setting, 'val', val_pre, training=False)

    with torch.no_grad():
        segmentor = SegEvaluator(dataset, config.num_classes, config.image_mean,
                                 config.image_std, network,
                                 config.eval_scale_array, config.eval_flip,
                                 all_dev, args.verbose, args.save_path,
                                 args.show_image)
        
        # for name_checkpoint in sorted(os.listdir(path_folder_checkpoint)):
        #     num_checkpoint =  get_num_checkpoint(name_checkpoint)
        #     if num_checkpoint >= 34 and num_checkpoint <= 40 :
        #         path_checkpoint = path_folder_checkpoint + "/" + name_checkpoint
        #         print("Load checkpoint: ", name_checkpoint)
        segmentor.run(path_model)
