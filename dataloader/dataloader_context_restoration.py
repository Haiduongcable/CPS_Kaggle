import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
from process_data import expand_dataloader
import cv2
import imgaug.augmenters as iaa

class PolypDataset_selfsupervised(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, trainsize):
        self.trainsize = trainsize
        self.image_root = image_root
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = self.images
        self.n_image = sorted(os.listdir(self.image_root))
        seed = 10 
        random.seed(10)
        random.shuffle(self.n_image)
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.filter_files()
        print("Length: ", len(self.images), len(self.gts) )
        self.size = len(self.images)
        self.img_transform = transforms.Compose([
            transforms.RandomRotation(45, resample=False, expand=False, center=None, fill=None),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((self.trainsize, self.trainsize)),])
            # transforms.ToTensor(),])
            # transforms.Normalize([0.485, 0.456, 0.406],
            #                         [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.RandomRotation(45, resample=False, expand=False, center=None, fill=None),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Resize((self.trainsize, self.trainsize)),])
            # transforms.ToTensor()])
        self.gt_augmentation = iaa.CoarseDropout((0.04, 0.12), size_percent=(0.05, 0.25))
            

    
    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.rgb_loader(self.gts[index])
        #Convert gt to cv2 and augmentaion random drop out
        gt_np = np.array(gt)
        gt_np_dropout = self.gt_augmentation.augment(images=[gt_np])[0]
        gt_dropout = Image.fromarray(gt_np_dropout)
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        gt_dropout = self.gt_transform(gt_dropout)
        return image, gt_dropout

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            images.append(img_path)
            gts.append(gt_path)
        self.images = images
        self.gts = gts
        
    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size



def get_loader(image_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True):
    
    dataset = PolypDataset_selfsupervised(image_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader

if __name__ == '__main__':
    image_root = "/home/asilla/duongnh/project/Analys_COCO/tmp_folder/CrossPseudo_UpdateBranch/Dataset/TrainDataset/image/"
    dataset = PolypDataset_selfsupervised(image_root, trainsize = 352)
    sample_1 = dataset[0]
    image, gt = sample_1 
    image = np.array(image)
    gt = np.array(gt)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    gt = cv2.cvtColor(gt, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test_in.png", image)
    cv2.imwrite("test_out.png", gt)
    
    # print(train_loader)
        