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
        transforms.Resize((256, 256)),
        transforms.ToTensor()])

datasets = VOCDetection(source_transform=source_transformer, anno_transform=AnnotationTransform(), mask_transform=Split(size=256))
datasets_size = len(datasets)
print(datasets_size)
dataloader = torch.utils.data.DataLoader(datasets, batch_size = Optdetection.batch_size, shuffle=True, collate_fn=noise_collate, num_workers=2)


for i, (source, target, anno, wh) in enumerate(dataloader):
    vutils.save_image(source, 'source.png')
    if i == 0:
        break