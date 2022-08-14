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
from torch.nn import BatchNorm2d
from config.config_contrastive_unreliable import config

from utils.pyt_utils import ensure_dir
from utils.visualize import show_img
from utils.evaluator import Evaluator
from utils.metric import hist_info
from PIL import Image
from utils.metric import hist_info, compute_score, compute_score_dice_IOU


#$from model.model_segformer_representation_contrastive import NetworkSegformerRepresentation

# from model.model_deeplabv3_representation_constrative import Network
from model.model import Network
from dataloader.dataloader import VOC

from utils.load_save_checkpoint import save_bestcheckpoint, load_only_checkpoint

from evaluation.eval_function import SegEvaluator
from dataloader.dataloader import VOC
from dataloader.dataloader import ValPre
import cv2

def get_class_colors(*args):
    def uint82bin(n, count=8):
        """returns the binary of integer n, count refers to amount of bits"""
        return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])
    N = 21
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    class_colors = cmap.tolist()
    return class_colors[1:]



def convert_predict_2_class_color(predict_map):
    predict_map = np.expand_dims(predict_map, axis = 2)
    # print("Expand shape", np.shape(predict_map))
    # predict_map_3_channel = np.concatenate((predict_map,predict_map, predict_map), axis=2)
    # print(np.shape(predict_map_3_channel))
    l_class_color = [[0, 0, 0], [0, 0, 128], [0, 128, 0], [0, 128, 128],
                [128, 0, 0], [128, 0, 128], [128, 128, 0],
                [128, 128, 128],
                [0, 0, 64], [0, 0, 192], [0, 128, 64],
                [0, 128, 192],
                [128, 0, 64], [128, 0, 192], [128, 128, 64],
                [128, 128, 192], [0, 64, 0], [0, 64, 128],
                [0, 192, 0],
                [0, 192, 128], [128, 64, 0]]

    l_class_name = ['background', 'aeroplane', 'bicycle', 'bird',
                'boat',
                'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                'diningtable',
                'dog', 'horse', 'motorbike', 'person',
                'pottedplant',
                'sheep', 'sofa', 'train', 'tv/monitor']
    visualize_map = np.zeros((np.shape(predict_map)[0], np.shape(predict_map)[1],3), dtype=np.uint8) + 255
    
    for index_class in range(len(l_class_color)):
        visualize_map = np.where(predict_map == index_class , l_class_color[index_class], visualize_map)
    return visualize_map

    
    

class SegEvaluator_visualize(SegEvaluator):
    def get_eval_per_image(self, results_dict):
        hist = np.zeros((config.num_classes, config.num_classes))
        correct = 0
        labeled = 0
        hist += results_dict['hist']
        correct += results_dict['correct']
        labeled += results_dict['labeled']

        iu, mean_IU, _, mean_pixel_acc, meanDice = compute_score_dice_IOU(hist, correct,
                                                       labeled)
        meanIU = np.nanmean(iu)
        return meanIU


    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        name = data['fn']
        pred = self.sliding_eval(img, config.eval_crop_size, config.eval_stride_rate, device)
        hist_tmp, labeled_tmp, correct_tmp = hist_info(config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp,
                        'correct': correct_tmp}
        mean_iu_log = self.get_eval_per_image(results_dict)
        

        # if self.save_path is not None:
        save_path = "/home/asilla/duongnh/project/CrossPseudo_UpdateBranch/CPS_Kaggle/Log_result/Log_resnet101"
        save_path_color = save_path + "_color"
        save_raw_img = save_path + "_rawimg"
        save_label = save_path + "_color_label"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(save_path_color):
            os.mkdir(save_path_color)
        if not os.path.exists(save_raw_img):
            os.mkdir(save_raw_img)
        if not os.path.exists(save_label):
            os.mkdir(save_label)
        fn = name + '.png'

        'save colored result'
        result_img = convert_predict_2_class_color(pred.astype(np.uint8))
        
        cv2.imwrite(os.path.join(save_path+'_color', name + "_" + str(mean_iu_log) + ".png"), result_img)
        # result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
        # class_colors = get_class_colors()
        # palette_list = list(np.array(class_colors).flat)
        # if len(palette_list) < 768:
        #     palette_list += [0] * (768 - len(palette_list))
        # result_img.putpalette(palette_list)
        # result_img.save(os.path.join(save_path+'_color', fn))
        # np_result_img = np.array(result_img)


        'save colored label'
        # print(np.shape(label))
        visualize_label = convert_predict_2_class_color(label.astype(np.uint8))
        cv2.imwrite(os.path.join(save_label, fn), visualize_label)
        # label_img = Image.fromarray(label.astype(np.uint8), mode='P')
        # class_colors = get_class_colors()
        # palette_list = list(np.array(class_colors).flat)
        # if len(palette_list) < 768:
        #     palette_list += [0] * (768 - len(palette_list))
        # label_img.putpalette(palette_list)
        # b, g, r = label_img.split()
        # im = Image.merge("RGB", (r, g, b))
        # label_img = label_img[:,:,::-1]
        # label_img = label_img.convert('RGB')
        # np_label_img = np.array(label_img)
        # np_label_img = cv2.cvtColor(np_label_img, cv2.COLOR_RGB2BGR)
        #cv2.imwrite(os.path.join(save_label, fn), np_label_img)
        # label_img.save(os.path.join(save_label, fn))
        # np_result_img = np.array(result_img)
        # print(np_result_img)
        #concat and save raw img, label, pred 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mask_label = np.zeros_like(img)
        mask_pred = np.zeros_like(img)
        # mask_label[:,:,0] = label
        # mask_label[:,:,1] = label
        # mask_label[:,:,2] = label
        # mask_pred[:,:,0] = np_result_img
        # mask_pred[:,:,1] = np_result_img
        # mask_pred[:,:,2] = np_result_img
        # concat_img = cv2.hconcat([img, mask_label, np_result_img])
        'save raw result'
        cv2.imwrite(os.path.join(save_raw_img, fn), img)
        cv2.imwrite(os.path.join(save_path, fn), pred)
        return results_dict


#path_pretrained_mitb2 = 'pretrained/mit_b2.pth'
#network =  NetworkSegformerRepresentation(config.num_classes, pretrained = path_pretrained_mitb2)
# network  = Network(config.num_classes,
#                 pretrained_model=None,
#                 norm_layer=BatchNorm2d, type_backbone='resnet101')
network  = Network(config.num_classes,
                pretrained_model=None,
                norm_layer=BatchNorm2d)
path_checkpoint = "weights/best_deeplab_resnet101.pth"

model = load_only_checkpoint(path_checkpoint, network)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
val_pre = ValPre()
data_setting = {'img_root': config.img_root_folder,
                'gt_root': config.gt_root_folder,
                'train_source': config.train_source,
                'eval_source': config.eval_source}
dataset = VOC(data_setting, 'val', val_pre, training=False)
with torch.no_grad():
    segmentor = SegEvaluator_visualize(dataset, config.num_classes, config.image_mean,
                                config.image_std, None,
                                config.eval_scale_array, config.eval_flip,
                                ["cuda"], False, None,
                                False)
    m_IOU_segformer_1, mDice_segformer_1 = segmentor.run_model(model.branch1)
    #m_IOU_segformer_2, mDice_segformer_2 = segmentor.run_model(model.branch2)
    print(m_IOU_segformer_1, mDice_segformer_1)
    # print(m_IOU_segformer_2, mDice_segformer_2)
