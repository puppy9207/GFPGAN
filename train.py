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


os.environ['CUDA_VISIBLE_DEVICES'] = str("1")
device = torch.device("cuda:0")
opt = {"train_root_path":"./example"}

out_size = 512
log_size = int(math.log(out_size,2))

discriminator = StyleGAN2Discriminator(out_size)
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
        vutils.save_image(lq.detach(), "result_lq.png")
        vutils.save_image(hq.detach(), "result_hq.png")
        time.sleep(1)