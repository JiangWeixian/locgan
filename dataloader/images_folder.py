"""IMAGE Dataset Classes

Original author: Francisco Massa
https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py

Updated by: Ellis Brown, Max deGroot
"""

import os
import os.path
import sys
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from config import IMAGE_PATH, FINE_SIZE
from PIL import Image, ImageDraw, ImageFont

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

source_transformer = transforms.Compose([
        transforms.Resize((FINE_SIZE, FINE_SIZE)),
        transforms.ToTensor()])

mask_transformer = source_transformer

class IMAGELOADER(data.Dataset):
    """COCO Bus Dataset Object
    """

    def __init__(self,
                 root=IMAGE_PATH, image_sets=[('2014', 'bus')],
                 source_transform=source_transformer,
                 mask_transform=mask_transformer,
                 dataset_name='coco_bus'):
        self.root = root
        self.image_set = image_sets
        self.source_transform = source_transform
        self.mask_transform = mask_transform
        self.name = dataset_name
        self.ids = list()
        # FIXME: need modify
        for (year, name) in image_sets:
            root_mask_path = os.path.join(self.root, 'train_mask' + year)
            root_img_path = os.path.join(self.root, 'train' + year)
            for line in open(os.path.join(root_mask_path, 'train_' + name + '.txt')):
                appendix = '.jpg'
                single_img_path = os.path.join(root_img_path, line.strip() + appendix)
                single_mask_path = os.path.join(root_mask_path, name, line.strip() + appendix)
                self.ids.append((single_img_path, single_mask_path))

    def __getitem__(self, index):
        im, mask, wh = self.pull_item(index)

        return im, mask, wh

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id, mask_id = self.ids[index]

        img = Image.open(img_id).convert('RGB')
        width, height = img.size
        mask = Image.open(mask_id).resize((width, height)).convert('L')

        if self.source_transform is not None:
            img = self.source_transform(img)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        wh = np.array([float(width), float(height)])
        
        return img, mask, wh

if __name__ == '__main__':
    BUS_DATASET_LOADER = IMAGELOADER()
    print(len(BUS_DATASET_LOADER))
    print(BUS_DATASET_LOADER.pull_item(0))