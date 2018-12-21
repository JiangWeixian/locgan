from options.opt_detection import Optdetection
from modules.detection_model import DetectModel
from modules.util.scafford import progress_bar
from data.split import VOCDetection, AnnotationTransform, Split
from data.config import mean, std
from data.augmentions import noise_collate, SSDAugmentation, NoiseAugmentation

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
source_transformer = transforms.Compose([
        # TODO: Scale
        transforms.Resize((321, 321)),
        transforms.ToTensor()])

datasets = VOCDetection(source_transform=source_transformer, anno_transform=AnnotationTransform(), mask_transform=Split(size=321))
datasets_size = len(datasets)
print(datasets_size)
dataloader = torch.utils.data.DataLoader(datasets, batch_size = Optdetection.batch_size, shuffle=True, collate_fn=noise_collate, num_workers=2)

# init network
net = DetectModel(Optdetection)

# network info
print (net)


# train
for epoch in range(Optdetection.epochs):
    for i, (source, target, anno, wh) in enumerate(dataloader):
        net.train(source, target, target, anno, wh)
        loss = net.current_errors()
        progress_bar(i, len(datasets)/Optdetection.batch_size, 'LossD: {:3f} | LossG: {:3f}'.format(loss['loss_D'], loss['loss_G']))
        if net.cnt % Optdetection.save_epoch_freq == 0:
            net.save(label='TRAIN', epoch=epoch)