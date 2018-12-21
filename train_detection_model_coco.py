from options.opt_detection import Optdetection
from modules.detection_model import DetectModel
from modules.util.scafford import parsetInt
# from data.coco14 import COCOVoc, COCOAnnotationTransform
# from data.coco01 import COCOVoc
from data.cocorgb import COCOVoc, COCOMaskTransform
from data.config import mean, std
from data.augmentions import target_collate

import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
from modules.util.scafford import progress_bar

# Init datatype
if Optdetection.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# inti datasets
mask_transformer = transforms.Compose([
        # TODO: Scale
        transforms.Scale((512, 512)),
        transforms.ToTensor()])
source_transformer = transforms.Compose([
        # TODO: Scale
        transforms.Scale((512, 512)),
        transforms.ToTensor()])

datasets = COCOVoc(source_transform=source_transformer, mask_transform=mask_transformer)
datasets_size = len(datasets)
print(datasets_size)
dataloader = torch.utils.data.DataLoader(datasets, batch_size = Optdetection.batch_size, shuffle=True, collate_fn=target_collate, num_workers=1)
# # init network
net = DetectModel(Optdetection)

# network info
print (net)

# train
for epoch in range(Optdetection.epochs):
    for i, (source, target, anno, wh) in enumerate(dataloader):
        net.train(source, target, target, anno, wh)
        loss = net.current_errors()
        # progress_bar(i, len(datasets)/Optdetection.batch_size, 'LossD: {:3f} | LossG: {:3f} | Lossfakeloc: {:3f} | Lossrealloc: {:3f}'.format(loss['loss_D'], loss['loss_G'], loss['loss_fake_loc'], loss['loss_real_loc']))
        # progress_bar(i, len(datasets)/Optdetection.batch_size, 'LossD: {:3f} | LossG: {:3f}'.format(loss['loss_D'], loss['loss_G']))
        progress_bar(i, len(datasets)/Optdetection.batch_size, 'Loss: {:3f}'.format(loss['loss']))        
        if net.cnt % Optdetection.save_epoch_freq == 0:
            net.save(label='TRAIN', epoch=epoch)