from dataloader import TrainDataset
import torch
import torchvision.utils as vutils
from torch import nn as nn
from torch.nn import functional as F

import time
import math
import os
from model.stylegan2_arch import StyleGAN2Discriminator
from model.gfpganv1_arch import GFPGANv1, FacialComponentDiscriminator
from model.resnetArcFace import ResNetArcFace
from util import L1Loss, GANLoss, PerceptualLoss

def get_roi_regions(self, eye_out_size=80, mouth_out_size=120):
        # hard code
        face_ratio = int(self.opt['network_g']['out_size'] / 512)
        eye_out_size *= face_ratio
        mouth_out_size *= face_ratio

        rois_eyes = []
        rois_mouths = []
        for b in range(self.loc_left_eyes.size(0)):  # loop for batch size
            # left eye and right eye
            img_inds = self.loc_left_eyes.new_full((2, 1), b)
            bbox = torch.stack([self.loc_left_eyes[b, :], self.loc_right_eyes[b, :]], dim=0)  # shape: (2, 4)
            rois = torch.cat([img_inds, bbox], dim=-1)  # shape: (2, 5)
            rois_eyes.append(rois)
            # mouse
            img_inds = self.loc_left_eyes.new_full((1, 1), b)
            rois = torch.cat([img_inds, self.loc_mouths[b:b + 1, :]], dim=-1)  # shape: (1, 5)
            rois_mouths.append(rois)

        rois_eyes = torch.cat(rois_eyes, 0).to(self.device)
        rois_mouths = torch.cat(rois_mouths, 0).to(self.device)

        # real images
        all_eyes = roi_align(self.gt, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        self.left_eyes_gt = all_eyes[0::2, :, :, :]
        self.right_eyes_gt = all_eyes[1::2, :, :, :]
        self.mouths_gt = roi_align(self.gt, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio
        # output
        all_eyes = roi_align(self.output, boxes=rois_eyes, output_size=eye_out_size) * face_ratio
        self.left_eyes = all_eyes[0::2, :, :, :]
        self.right_eyes = all_eyes[1::2, :, :, :]
        self.mouths = roi_align(self.output, boxes=rois_mouths, output_size=mouth_out_size) * face_ratio

os.environ['CUDA_VISIBLE_DEVICES'] = str("1")
device = torch.device("cuda:0")
opt = {"train_root_path":"./example"}

out_size = 512
log_size = int(math.log(out_size,2))

discriminator = StyleGAN2Discriminator(out_size)
for p in discriminator.parameters():
    p.requires_grad = False

generator = GFPGANv1(out_size,decoder_load_path="./model/pre-train/StyleGAN2_512_Cmul1_FFHQ_B12G4_scratch_800k.pth")

d_left_eye = FacialComponentDiscriminator()
d_right_eye = FacialComponentDiscriminator()
d_mouth = FacialComponentDiscriminator()
identity = ResNetArcFace("IRBlock",[2,2,2,2],False)
# pre-train

cri_pixel = L1Loss(loss_weight=1e-1,reduction="mean")
layer_weights = {
    'conv1_2': 0.1,
    'conv2_2': 0.1,
    'conv3_4': 1,
    'conv4_4': 1,
    'conv5_4': 1}

cri_percept = PerceptualLoss(layer_weights,style_weight=50)
cri_l1 = L1Loss(loss_weight=1,reduction="mean")
cri_component = GANLoss("vanilla",loss_weight=1)
cri_gan = GANLoss("wgan_softplus",loss_weight=1e-1)

opt_g = torch.optim.Adam(generator.parameters(),lr=2e-3)
opt_d = torch.optim.Adam(discriminator.parameters(),lr=2e-3)
opt_d_left_eye = torch.optim.Adam(d_left_eye.parameters(),lr=2e-3)
opt_d_right_eye = torch.optim.Adam(d_right_eye.parameters(),lr=2e-3)
opt_d_mouth = torch.optim.Adam(d_mouth.parameters(),lr=2e-3)
#parameter setting

dataset = TrainDataset(opt,device)
dataloader = torch.utils.data.DataLoader(dataset, 
            batch_size=1,
            shuffle=True,
            pin_memory=True,
            sampler=None,
            num_workers=8)

for epoch in range(100):
    for i, (lq,hq) in enumerate(dataloader):
        lq = lq.to(device)
        hq = hq.to(device)

        opt_g.zero_grad()