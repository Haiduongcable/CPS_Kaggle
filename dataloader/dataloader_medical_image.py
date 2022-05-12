import os
import cv2
import torch
import numpy as np
from torch.utils import data
import random

from utils.process_data import expand_dataloader
from modules.datasets.BaseDataset import BaseDataset
import random 
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import config
# def get_train_loader(unsupervised=False):
#         train_dataset = dataset(data_setting, "train", train_preprocess, config.tot_samples, unsupervised=unsupervised)
#     else:
#         train_dataset = dataset(data_setting, "train", train_preprocess,
#                                 config.max_samples, unsupervised=unsupervised)

#     train_sampler = None
#     is_shuffle = True
#     batch_size = config.batch_size

#     train_loader = data.DataLoader(train_dataset,
#                                    batch_size=batch_size,
#                                    num_workers=config.num_workers,
#                                    drop_last=True,
#                                    shuffle=is_shuffle,
#                                    pin_memory=True,
#                                    sampler=train_sampler,
#                                    collate_fn=collate_fn)

#     return train_loader, train_sampler

def get_train_loader(dataset):
    batch_size = config.batch_size 
    train_loader = data.DataLoader(dataset,
                                   batch_size=batch_size,
                                   num_workers=config.num_workers,
                                   drop_last=True,
                                   shuffle=True,
                                   pin_memory=True)
    
    return train_loader

class MedicalImageDataset(Dataset):
    def __init__(self, supervised = True, validation = False):
        self.supervised = supervised
        self.validation = validation
        self.path_dataset = "/home/asilla/duongnh/project/Analys_COCO/tmp_folder/CrossPseudo_UpdateBranch/Dataset/TrainDataset"
        self.path_test_dataset = "/home/asilla/duongnh/project/Analys_COCO/tmp_folder/CrossPseudo_UpdateBranch/Dataset/TestDataset"
        
        self.l_nimage = os.listdir(self.path_dataset + "/image")
        # self.l_test_nimage = os.listdir(self.path_test_dataset  + "/image")
        seed = 10 
        random.seed(10)
        random.shuffle(self.l_nimage)
        #Get length dataset
        self.length_dataset = len(self.l_nimage)
        self.num_label_data = self.length_dataset // 4
        self.num_unlabel_data = self.length_dataset - self.num_label_data 
        # self.num_val_data = len(self.l_test_nimage)
        
        self.l_nimage_labeled = self.l_nimage[:self.num_label_data]
        self.l_nimage_unlabled = self.l_nimage[self.num_label_data:]
        l_path_label_image = [self.path_dataset + "/image/" +\
                                    item for item in self.l_nimage_labeled]
        l_path_label_mask = [self.path_dataset + "/mask/" +\
                                    item for item in self.l_nimage_labeled]
        self.l_path_unlabel_image = [self.path_dataset + "/image/" +\
                                    item for item in self.l_nimage_unlabled]
        
        self.l_path_label_image, self.l_path_label_mask = expand_dataloader(l_path_label_image,\
                                                        l_path_label_mask, self.l_path_unlabel_image)
        # print(len(self.l_path_label_image), len(self.l_path_label_mask))
        # print(len(self.l_path_unlabel_image))
        # self.l_path_val_image = [self.path_test_dataset + "/image/" +\
        #                             item for item in self.l_test_nimage]
        # self.l_path_val_mask = [self.path_test_dataset + "/mask/" +\
        #                             item for item in self.l_test_nimage]
        
    def __len__(self):
        return len(self.l_path_label_image)
        
    def __getitem__(self, idx):
        if self.validation:
            return None
        if self.supervised:
            #call mask and call image
            path_image = self.l_path_label_image[idx]
            path_mask = self.l_path_label_mask[idx]
            image = cv2.imread(path_image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(path_mask)[:,:,0]
            mask = np.asarray(mask, np.float32)
            mask /= 255.0
            # mask = np.where(mask > 125, 1, 0)
            # mask = np.asarray(mask, np.float32)
            # mask /= (mask.max() + 1e-8)
            # mask = np.array(mask).astype(np.int64)
            sample = {"image": image, "mask": mask}
            encoded_input = self.train_transform(sample)
        else:
            #call only image, mask = blank 
            path_image = self.l_path_unlabel_image[idx]
            image = cv2.imread(path_image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = np.zeros_like(image)[:,:,0]
            
            # mask = np.where(mask > 125, 1, 0)
            mask = np.asarray(mask, np.float32)
            mask /= 255.0
            max_value = np.max(mask)
            if max_value <0 or max_value > 1:
                print(max_value)
            # mask = np.array(mask).astype(np.int64)
            sample = {"image": image, "mask": mask}
            encoded_input = self.train_transform(sample)
        return encoded_input
       
    def train_transform(self, sample):
        train_transform = A.Compose([
                A.Resize(height=352, width=352),
                # A.GaussianBlur(blur_limit=(0,0.1), p = 0.25),
                A.Affine(translate_percent= (-0.1, 0.1), cval = 0, cval_mask= 255, p = 0.3),
                A.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.15, p = 0.25),
                A.HorizontalFlip(p = 0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()])
        transfomerd = train_transform(image=sample["image"], mask=sample["mask"])
        return {'image': transfomerd['image'], 'label': transfomerd['mask']}
    
    def val_transform(self, sample):
        val_transform = A.Compose([
            A.Resize(height=352, width=352),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()])
        transfomerd = val_transform(image=sample["image"], mask=sample["mask"])
        return {'image': transfomerd['image'], 'label': transfomerd['mask']}


if __name__ == '__main__':
    dataset = MedicalImageDataset(supervised=True, validation=False)
    
    train_loader = data.DataLoader(dataset,
                                   batch_size=4,
                                   num_workers=8,
                                   drop_last=True,
                                   shuffle=True,
                                   pin_memory=True)
    