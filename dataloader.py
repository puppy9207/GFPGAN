import torch
import os
import cv2
import torchvision.transforms as transforms
import numpy as np
import math

from PIL import Image

from GFPDeg import random_mixed_kernels, random_add_gaussian_noise, random_add_jpg_compression

# file format check
def check_image_file(filename: str):
    return any(filename.endswith(extension) for extension in [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"])

def deg_process(image):
    # openCV
    image = np.asarray(image)
    image = np.transpose(image,(1,2,0))
    img_gt = image
    h,w,c = img_gt.shape

    # Blur Kernel Setting
    kernel = random_mixed_kernels(
            ["iso","aniso"],
            [0.5,0.5],
            41,
            [0.2,10],
            [0.2,10], [-math.pi, math.pi],
            noise_range=None)
    # Blur Kernel
    img_lq = cv2.filter2D(img_gt, -1, kernel)

    # 1~8 scale Downscale
    scale = np.random.uniform(1, 8)
    img_lq = cv2.resize(img_lq, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)

    # Random Gaussian Noise
    img_lq = random_add_gaussian_noise(img_lq, [0, 15])

    # Random JPEG Noise
    img_lq = random_add_jpg_compression(img_lq, [60, 100])

    # original
    img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)

    return img_lq

class TrainDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, opt,device):
        super(TrainDataset, self).__init__()
        self.filenames = [os.path.join(opt["train_root_path"], x) for x in os.listdir(os.path.join(opt["train_root_path"])) if check_image_file(x)]
        self.hq_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        self.lq_transforms = transforms.Compose([
            transforms.Lambda(deg_process),
            transforms.ToTensor()
        ])

    def __getitem__(self,idx):
        hq = cv2.imread(self.filenames[idx])
        hq = cv2.cvtColor(hq,cv2.COLOR_BGR2RGB)
        hq = self.hq_transforms(hq)
        lq = self.lq_transforms(hq)
        return lq, hq
    
    def __len__(self):
        return len(self.filenames)