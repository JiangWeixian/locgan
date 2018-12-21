from options.opt_detection import Optdetection
from modules.detection_model import DetectModel
from modules.util.scafford import parsetInt
from data.ssd_pascal import VOCDetection, AnnotationTransform, MaskTransform
from data.config import mean, std
from data.augmentions import mask_collate, SSDAugmentation

import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import os
import torchvision.utils as vutils

# Init datatype
if Optdetection.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # cudnn.benchmark = True
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# inti datasets
mask_transformer = transforms.Compose([
        # TODO: Scale
        transforms.ToTensor()])

datasets = VOCDetection(transform=SSDAugmentation(Optdetection.fine_size), target_transform=AnnotationTransform(), mask_transform=MaskTransform(mask_transform=mask_transformer, size=Optdetection.fine_size, mask_size=64))
datasets_size = len(datasets)
print(datasets_size)
dataloader = torch.utils.data.DataLoader(datasets, batch_size = Optdetection.batch_size, shuffle=True, collate_fn=mask_collate, num_workers=2)

# # init network
net = DetectModel(Optdetection)

# #network info
print (net)


# train
for epoch in range(160, Optdetection.epochs):
    for i, (source, target, anno, wh) in enumerate(dataloader):
        net.train(source, target, target, anno, wh)
        if net.cnt % Optdetection.save_epoch_freq == 0:
            net.save(label='TRAIN', epoch=epoch)