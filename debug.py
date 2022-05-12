from dataloader.dataloader_medical_image import MedicalImageDataset, get_train_loader
from torch.utils import data
import numpy as np 
import cv2
import torch 

if __name__ == '__main__':
    dataset = MedicalImageDataset(supervised=True, validation=False)
    print(len(dataset))
    
    # unlabeled_dataset = MedicalImageDataset(supervised=False, validation=False)
    # train_loader = data.DataLoader(dataset,
    #                                batch_size=4,
    #                                num_workers=0,
    #                                drop_last=True,
    #                                shuffle=True,
    #                                pin_memory=True)
    
    # # train_loader = get_train_loader(dataset)
    # dataloader = iter(train_loader)
    # minibatch = dataloader.next()
    # imgs = minibatch['image']
    # gts = minibatch['label']
    # gts_np = gts.detach().cpu().numpy()
    # print(np.unique(gts_np))
    sample = dataset[102]
    image = sample["image"]
    mask = sample["label"]
    # print(np.shape(image))
    # print(np.shape(mask))
    mask = mask.detach().cpu().numpy()
    print(np.unique(mask))
    
    image = image.permute(1, 2,0)
    image = image.detach().cpu().numpy()
    print(image.shape)
    cv2.imwrite("mask.jpg", mask)
    cv2.imwrite("image.jpg", image)
    
    