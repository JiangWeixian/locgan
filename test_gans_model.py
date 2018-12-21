from options.opt_gans import Optgans
from modules.gans_model import GansModel
from data.pascal_gans import PascalVOC, AnnotationTransform
from modules.util.scafford import parsetInt
from data.augmentions import mask_collate

import torch
import torchvision.transforms as transforms
import os

# inti datasets
transforms = transforms.Compose([
    transforms.Scale(size=(Optgans.fine_size, Optgans.fine_size)),
    transforms.ToTensor()])

datasets = PascalVOC(transform=transforms, target_transform=transforms, anno_transform=AnnotationTransform())
datasets_size = len(datasets)
dataloader = torch.utils.data.DataLoader(datasets, batch_size = Optgans.batch_size, shuffle=True, collate_fn=mask_collate, num_workers=2)

# init network
net_maskgans = GansModel(Optgans)

# network info
print (net_maskgans)

# train or continue train
if Optgans.mode == 'TEST':
    for i, (source, target, anno, wh) in enumerate(dataloader):
        net_maskgans.test(source, target)
        if i >= 2000:
            break