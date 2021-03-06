from __future__ import print_function


import errno
import hashlib
import os
import sys
import tarfile
import numpy as np
import torchvision.transforms as transforms

import torch
import torch.utils.data as data
from PIL import Image

from six.moves import urllib

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from config import coco_datasets_root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "."))
from pycocotools.coco import COCO

CLASSES = [
    'airplane','bicycle','bird','boat',
    'bottle','bus','car','cat','chair',
    'cow','dining table','dog','horse',
    'motorcycle','person','potted plant',
    'sheep','skateboard','train','tv']

LABEL_MAP = {
    5: 'airplane',
    2: 'bicycle',
    16: 'bird',
    9: 'boat',
    44: 'bottle',
    6: 'bus',
    3: 'car',
    17: 'cat',
    62: 'chair',
    21: 'cow',
    67: 'dining table',
    18: 'dog',
    19: 'horse',
    4: 'motorcycle',
    1: 'person',
    64: 'potted plant',
    20: 'sheep',
    41: 'skateboard',
    7: 'train',
    72: 'tv'
}

class COCOAnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    @Params:
    - class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
    - keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
    - height (int): height
    - width (int): width
    """

    def __init__(self, class_to_ind=None):
        self.class_to_ind = class_to_ind

    def __call__(self, bboxs, width, height):
        """
        @Params:
        - target (annotation) : the target annotation to be made usable
                will be an ET.Element
        @Returns:
        a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = []
        for bbox in bboxs:
            bndbox = []
            for i, cur_pt in enumerate(bbox[:-1]):
                cur_pt = cur_pt / float(width) if i % 2 == 0 else cur_pt / float(height)
                cur_pt = 0 if cur_pt <= 0 else cur_pt
                cur_pt = 1 if cur_pt >= 1 else cur_pt
                bndbox.append(cur_pt)
            label_name = LABEL_MAP[bbox[-1]]
            label_idx = self.class_to_ind[label_name]
            bndbox.append(label_idx)
            res += [bndbox]
        return res

class COCOMaskTransform(object):
    def __init__(self, mask_transform=None):
        self.mask_transform = mask_transform

    def _get_seg(self, targets, bboxs):
        for i, mask in enumerate(targets):
            bbox = bboxs[i]
            label_name = LABEL_MAP[bbox[-1]]
            label_color = Image.open('{}/{}.jpg'.format(coco_datasets_root, label_name)).convert('RGB')
            np.clip(mask, 0, 255, out=mask)
            mask = Image.fromarray(mask, mode='L')
            mask = self.mask_transform(mask)
            color = self.mask_transform(label_color)
            target = mask*color
            if i == 0:
                targets = target
                masks = mask
            else:
                targets += target
                masks += mask
        return masks, targets
    
    def _get_bg(self, source, masks):
        source = self.mask_transform(source)
        fg = masks.gt(0)
        bg = 1- fg
        bg = bg.expand_as(source)
        bg = bg.type(type(source)) * source
        return bg

    def __call__(self, source, targets, bboxs):
        masks, fg = self._get_seg(targets, bboxs)
        bg = self._get_bg(source, masks)
        return bg+fg    

class COCOTargetTransform(object):
    def __init__(self, mask_transform=None):
        self.mask_transform = mask_transform

    def __call__(self, source, bboxs):
        for i, bbox in enumerate(bboxs):
            mask = source.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
            mask = self.mask_transform(mask)
            mask = mask.unsqueeze(0)
            if i == 0:
                masks = mask
            else:
                masks = torch.cat([masks, mask], dim=0)
        return masks    
          
class COCOVoc(data.Dataset):
    '''Load pascal voc dataset, just like dataset.mnist

    @Params:
    - root: the datasets path
    - train: use this datasets for mode-train or not
    - transform/traget_transform/anno_transform: transform source-img/seg-img/annotation for better use
    - download: download this datasets if you need
    '''

    def __init__(self,
                 root=coco_datasets_root,
                 mode='train',
                 source_transform=None,
                 mask_transform=None,
                 download=False):
        self.root = root
        self.train = True if mode == 'train' else False
        self.source_transform = source_transform
        image_dir = os.path.join(root, '{}2014'.format(mode))
        anno_file = os.path.join(root, 'annotations/instances_{}2014.json'.format(mode))
        split_file = split_file = os.path.join(root, '21classes.npz')
        self.coco = COCO(anno_file)
        self.catIds = self.coco.getCatIds(catNms=CLASSES)
        print(len(self.catIds))
        self.anno_transform = COCOAnnotationTransform(class_to_ind=dict(zip(CLASSES, range(len(self.catIds)))))
        self.mask_transform = COCOMaskTransform(mask_transform)

        # Get  images List
        data = np.load(split_file)
        for key, val in data.items():
            self.imgIds = val
        
        self.images = []
        self.masks = []
        for imgId in self.imgIds:
            img = self.coco.loadImgs(int(imgId))[0]
            img_name = img['file_name']
            img_path = os.path.join(image_dir, img_name)
            if os.path.isfile(img_path):
                self.images.append(img_path)
                self.masks.append(int(imgId))
        
        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        '''Load data from datasets, and return [source_img, seg_img, annotation, wh]

        @Params:
        - index: the index of which source_img

        @Returns:
        format like [source_img, seg_img, annotation, wh]
        '''
        img = Image.open(self.images[index]).convert('RGB')
        masks, annos = self._get_bbox(self.masks[index])
        width, height = img.size

        
        if self.mask_transform is not None:
            mask = self.mask_transform(img, masks, annos)
        if self.source_transform is not None:
            img = self.source_transform(img)
        # todo(bdd) : perhaps transformations should be applied differently to masks? 
        if self.anno_transform is not None:
            anno = self.anno_transform(annos, width, height)
            anno = np.array(anno)
        wh = np.array([width, height])

        return img, mask, anno, wh

    def _get_bbox(self, imgId):
        annIds = self.coco.getAnnIds(imgIds=imgId, catIds=self.catIds, iscrowd=False)
        anns = self.coco.loadAnns(annIds)
        coords = []
        masks = []
        gap = 255
        for ele in anns:
            xmin, ymin, width, height = ele['bbox']
            xmax = xmin+width
            ymax = ymin+height
            catId = ele['category_id']
            masks.append(self.coco.annToMask(ele)*gap)
            coords.append([int(xmin), int(ymin), int(xmax), int(ymax), int(catId)])
        return masks, coords

    def __len__(self):
        return len(self.images)

class Combine(object):
    def __init__(self, size=256):
        self.mode = 'combine'
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
        fg = fg.type(type(target)) * target
        bg = bg.type(type(source)) * source
        return bg, fg
    
    def __call__(self, source, mask, target):
        bg, fg = self._split(source, mask, target)
        return bg + fg        


if __name__ == '__main__':
    import torch
    import torchvision.transforms as transforms
    import torchvision.utils as vutils
    from config import coco_datasets_root, mean, std
    mask_transformer = transforms.Compose([
            # TODO: Scale
            transforms.Scale((300, 300)),
            transforms.ToTensor()])
    source_transformer = transforms.Compose([
            # TODO: Scale
            transforms.Scale((300, 300)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    datasets = COCOVoc(source_transform=source_transformer, mask_transform=mask_transformer, anno_transform=COCOAnnotationTransform(keep_difficult=True))
    # datasets = COCOVoc()
    datasets_size = len(datasets)
    print(datasets_size)
    # dataloader = torch.utils.data.DataLoader(datasets, batch_size = 1, shuffle=True, num_workers=2)

    print(datasets[0][0].shape, datasets[0][1].shape)
    input = datasets[0][1]
    source = datasets[0][0]
    print(input)
    for i in range(1, 21):
        mask = torch.zeros(self.batch_size, 1, self.fine_size, self.fine_size)
        low_to_high = input[0, (i-1), :, :].gt(0)
        mask.masked_fill_(low_to_high, 1)
        target = mask * source
        if i == 1:
            targets = target
        else:
            targets = torch.cat([targets, target], dim=1)
    for i in range(20):
            vutils.save_image(targets[:, i*3:(i+1)*3, :, :], 'real{}.png'.format(i))