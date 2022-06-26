import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import random
import torch
from .process_data import expand_dataloader
import cv2

class PolypDataset(data.Dataset):
    """
    dataloader for polyp segmentation tasks
    """
    def __init__(self, image_root, gt_root, trainsize,\
                augmentations, supervised = True, use_offline_aug = False):
        self.use_offline_aug = use_offline_aug
        self.trainsize = trainsize
        self.augmentations = augmentations
        print('Used augmentation offline: ', self.use_offline_aug)
        print(self.augmentations)
        self.image_root = image_root
        self.gt_root = gt_root
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.n_image = sorted(os.listdir(self.image_root))
        seed = 10
        random.seed(seed)
        random.shuffle(self.n_image)
        self.images = sorted(self.images)
        
        self.gts = sorted(self.gts)
        self.ratio_labeled = 0.2 # 0.2 vs 0.8
        self.length_dataset = len(self.n_image)
        self.length_labeled_dataset = int(self.ratio_labeled * self.length_dataset)
        self.length_unlabeled_dataset = self.length_dataset - self.length_labeled_dataset
        if supervised: 
            self.images , self.gts = self.filter_labeled_dataset()
        else:
            self.images , self.gts = self.filter_unlabeled_dataset()
        print("Length: ", len(self.images), len(self.gts) )
        # print(self.images[:10])
        # print(self.gts[:10])
        # print("DONNNE")
        
        
        self.size = len(self.images)
        if self.augmentations:
            print('Using RandomRotation, RandomFlip')
            self.img_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            self.gt_transform = transforms.Compose([
                transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            
        else:
            print('no augmentation')
            self.img_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])])
            
            self.gt_transform = transforms.Compose([
                transforms.Resize((self.trainsize, self.trainsize)),
                transforms.ToTensor()])
            

    
    def __getitem__(self, index):
        
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.img_transform is not None:
            image = self.img_transform(image)
            
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        for img_path, gt_path in zip(self.images, self.gts):
            images.append(img_path)
            gts.append(gt_path)
        self.images = images
        self.gts = gts
        
        
    def filter_labeled_dataset(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        l_nimage_labeled = self.n_image[:self.length_labeled_dataset]
        # if self.use_offline_aug:
        #     #CONFIG 
        #     PATH_OFFLINE_AUG_image = "../Dataset/Augmentation/Mosaic_augmentation/image"
        #     PATH_OFFLINE_AUG_mask = "../Dataset/Augmentation/Mosaic_augmentation/mask"
        #     l_offline_aug_image = []
        #     l_offline_aug_mask = []
        #     for nimage in os.listdir(PATH_OFFLINE_AUG_image):
        #         path_aug_image = PATH_OFFLINE_AUG_image + "/" + nimage
        #         path_aug_mask = PATH_OFFLINE_AUG_mask + "/" + nimage
        #         l_offline_aug_image.append(path_aug_image)
        #         l_offline_aug_mask.append(path_aug_mask)
        #     print("Add offline Augmentation data: ", len(l_offline_aug_image), len(l_offline_aug_mask))
        for nameimage in l_nimage_labeled:
            path_images = self.image_root +"/"+ nameimage
            path_gts = self.gt_root +"/"+ nameimage
            images.append(path_images)
            gts.append(path_gts)
        # if self.use_offline_aug:
        #     images += l_offline_aug_image
        #     gts += l_offline_aug_mask
        images, gts = expand_dataloader(images,gts, self.length_unlabeled_dataset)
        
        return images, gts
    
    def filter_unlabeled_dataset(self):
        assert len(self.images) == len(self.gts)
        images = []
        gts = []
        l_nimage_unlabeled = self.n_image[self.length_labeled_dataset:]
        # l_n_labimage = [self.path_dataset + "/image/" +\
        #                             item for item in l_nimage_labeled]
        for nameimage in l_nimage_unlabeled:
            path_images = self.image_root +"/"+ nameimage
            path_gts = self.gt_root +"/"+ nameimage
            images.append(path_images)
            gts.append(path_gts)
        return images, gts

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



def get_loader(image_root, gt_root, batchsize, trainsize, shuffle=True, num_workers=4, pin_memory=True, augmentation=True, supervised=False):
    
    dataset = PolypDataset(image_root, gt_root, trainsize, augmentation, supervised = supervised)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.tif') or f.endswith('.png')]
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name
    
    
    # def gamma_correction(self, image_pillow, gamma = 0.65):
    #     image_rgb = np.array()
    #     invGamma = 1.0 / gamma
    #     table = np.array([((i / 255.0) ** invGamma) * 255
    #         for i in np.arange(0, 256)]).astype("uint8")
    #     # apply gamma correction using the lookup table
    #     return cv2.LUT(image, table)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


if __name__ == '__main__':
    image_root = "/home/asilla/duongnh/project/Analys_COCO/tmp_folder/CrossPseudo_UpdateBranch/Dataset/TrainDataset/image"
    gt_root = "/home/asilla/duongnh/project/Analys_COCO/tmp_folder/CrossPseudo_UpdateBranch/Dataset/TrainDataset/mask"
    train_loader = get_loader(image_root, gt_root, batchsize=16, trainsize=352, augmentation = True, supervised = True)
    unsupervised_train_loader = get_loader(image_root, gt_root, batchsize=16, trainsize=352, augmentation = True, supervised = False)
    # print(train_loader)
        