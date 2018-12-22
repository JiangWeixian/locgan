"""VOC Dataset Classes

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
from PIL import Image, ImageDraw, ImageFont
import numpy as np
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

voc_datasets_root = '/home/eric/Documents/datasets/VOCdevkit'
VOC_CLASSES = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

mask_transformer = transforms.Compose([
        # TODO: Scale
        transforms.Scale((256, 256)),
        transforms.ToTensor()])

source_transformer = transforms.Compose([
        # TODO: Scale
        transforms.Scale((321, 321)),
        transforms.ToTensor()])

class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                cur_pt = cur_pt / float(width) if i % 2 == 0 else cur_pt / float(height)
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        return res  # [[xmin, ymin, xmax, ymax, label_ind], ... ]


class MaskTransform(object):
    def __init__(self, mask_transform=None, fine_size=321):
        self.mask_transform = mask_transform
        self.fine_size = fine_size

    def __call__(self, source, anno):
        for i, bbox in enumerate(anno):
            xmin = int(bbox[0]*self.fine_size)
            ymin = int(bbox[1]*self.fine_size)
            xmax = int(bbox[2]*self.fine_size)
            ymax = int(bbox[3]*self.fine_size)
            mask = source.crop((xmin, ymin, xmax, ymax))
            mask = self.mask_transform(mask)
            mask = mask.unsqueeze(0)
            if i == 0:
                masks = mask
            else:
                masks = torch.cat([masks, mask], dim=0)
        return masks   


class VOCDetection(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root=voc_datasets_root, image_sets=[('2012', 'trainval'), ('2007', 'trainval')], source_transform=None, anno_transform=None, mask_transform=None,
                 dataset_name='VOC0712'):
        self.root = root
        self.image_set = image_sets
        self.source_transform = source_transform
        self.anno_transform = anno_transform
        self.mask_transform = mask_transform
        self.name = dataset_name
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self._maskpath = os.path.join('%s', 'SegmentationClass', '%s.png')
        self.ids = list()
        for (year, name) in image_sets:
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Segmentation', name + '.txt')):
                self.ids.append((rootpath, line.strip()))

    def __getitem__(self, index):
        im, mask, anno, wh = self.pull_item(index)

        return im, mask, anno, wh

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        img_id = self.ids[index]

        anno = ET.parse(self._annopath % img_id).getroot()
        img = Image.open(self._imgpath % img_id)
        mask = Image.open(self._maskpath % img_id)
        target = mask.convert('RGB')
        width, height = img.size

        if self.anno_transform is not None:
            anno = self.anno_transform(anno, width, height)

        if self.mask_transform is not None:
            bg, fg = self.mask_transform(img, mask, target)    

        if self.source_transform is not None:
            anno = np.array(anno)
            img = self.source_transform(img)

        wh = np.array([float(width), float(height)])
        
        return img, fg, anno, wh


class Split(object):
    def __init__(self, size=256):
        self.mode = 'split'
        self.size = size
        self.transformer = transforms.Compose([
            # TODO: Scale
            transforms.Scale((self.size, self.size)),
            transforms.ToTensor()])

    def _split(self, source, mask, target):
        source = self.transformer(source)
        target = self.transformer(target)
        mask = self.transformer(mask)
        fg = mask.gt(0)
        bg = 1- fg
        bg = bg.expand_as(target)
        fg = fg.type(type(target)) * source
        bg = bg.type(type(source)) * source
        return bg, fg
    
    def __call__(self, source, mask, target):
        bg, fg = self._split(source, mask, target)
        return bg, fg