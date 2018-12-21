from options.opt_test import opt
from modules.detection_model import DetectModel
from modules.util.scafford import parsetInt
from pycocotools.coco import COCO
from PIL import Image, ImageDraw

import torch
import torchvision.transforms as transforms
import os
import numpy as np
import json

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


coco = COCO(anno_file)
catIds = coco.getCatIds()
imgIds = coco.getImgIds()
print(len(imgIds))

imgs = []
for imgId in imgIds:
        img = coco.loadImgs(int(imgId))[0]
        img_name = img['file_name']
        img_path = os.path.join(image_dir, img_name)
        imgs.append(img_path)

res = []
for i, img in enumerate(imgs):
    img = Image.open(imgs[i]).convert('RGB')
    w, h = img.size
    img = source_transformer(img)
    img = img.cuda()
    singleRes = net.test(img, imgIds[i], labelmap=catIds, wh = [w, h])
    print(singleRes)
    if singleRes:
        res += singleRes
    if i == 100:
        break

print(len(res))
with open('cocores.json', 'w') as f:
    jsondata = json.dump(res, f)