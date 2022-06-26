#from dataloader.dataloader_medical_image import MedicalImageDataset, get_train_loader
from torch.utils import data
import numpy as np 
import cv2
import torch 
import imgaug.augmenters as iaa
from model.model import Network 

from torch.nn import BatchNorm2d
if __name__ == '__main__':
    device = torch.device("cuda")
    model = Network(1, pretrained_model="/home/asilla/duongnh/project/Analys_COCO/tmp_folder/CrossPseudo_UpdateBranch/CPS_Kaggle/self_supervised_weight/epoch_76.pth", norm_layer=BatchNorm2d)
   
    # model.to(device)
    model.eval()
    # summary(model, (3,128,128))
    left = torch.randn(2, 3, 128, 128)
    # left.to(device)
    # right = torch.randn(2, 3, 128, 128)

    # print(model.branch1)

    out = model(left)
    print(out.shape)
    
    