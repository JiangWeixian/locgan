from options.opt_test import opt
from modules.detection_model import DetectModel
from modules.util.scafford import parsetInt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from PIL import Image, ImageDraw

import numpy as np
import skimage.io as io
import pylab
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import os
import numpy as np
import json

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

# Init datatype
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'boat', 
    'bird', 'cat', 'dog', 'horse', 
    'sheep', 'cow', 'skateboard', 'bottle', 
    'chair', 'potted plant', 'dining table', 'tv'
]

if opt.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

# inti datasets
source_transformer = transforms.Compose([
        # TODO: Scale
        transforms.Scale((opt.fine_size, opt.fine_size)),
        transforms.ToTensor()])

# init network
net = DetectModel(opt)

# load datasets
root = '/home/eric/Documents/datasets/coco/'
mode = 'val'
image_dir = os.path.join(root, '{}2014'.format(mode))
anno_file = os.path.join(root, 'annotations/instances_{}2014.json'.format(mode))


cocoGt = COCO(anno_file)
# eval
cocoDt = cocoGt.loadRes('./cocores.json')
# imgIds= cocoGt.getImgIds()
imgIds = 262148
cocoEval = COCOeval(cocoGt,cocoDt, 'bbox')
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
