import numpy as np 
import os 
import time 
import cv2 
import torch 
import random 

def expand_dataloader(labeled_image, labeled_mask, length_unlabeled_data):
    '''
    Expand and truncated dataset of labeled data 
    Ratio between labeled data and unlabeled data: 1:4
    Expand size of labeled data to avoid unbalance data and noise training in training
    Args: labeled_dataset: numpy array: array to dataloader 
          unlabeled_dataset: numpy array: array to dataloader with None masking labeled
    Return: expand_labeled_dataset
    '''
    seed = 10 
    random.seed(seed)
    length_labeled_data = len(labeled_image)
    ratio_data = length_unlabeled_data // length_labeled_data
    #random choice rest dataloader 
    #Update labeled masking follow labeled image
    number_rest_dataset = length_unlabeled_data - length_labeled_data * ratio_data 
    l_index = [index for index in range(length_labeled_data)]
    random_index_rest_dataset = random.choices(l_index, k=number_rest_dataset)
    expand_labeled_image = labeled_image * ratio_data + [labeled_image[item] for item in random_index_rest_dataset]
    expand_labeled_mask = labeled_mask * ratio_data + [labeled_mask[item] for item in random_index_rest_dataset]
    return expand_labeled_image, expand_labeled_mask


def label_onehot(inputs, num_segments):
    batch_size, im_h, im_w = inputs.shape
    outputs = torch.zeros((num_segments, batch_size, im_h, im_w)).cuda()

    inputs_temp = inputs.clone()
    inputs_temp[inputs == 255] = 0
    outputs.scatter_(0, inputs_temp.unsqueeze(1), 1.0)
    outputs[:, inputs == 255] = 0

    return outputs.permute(1, 0, 2, 3)

