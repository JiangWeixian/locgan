"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
from options.opt_detection import Optdetection
from modules.network import define_netG
from modules.funcs.detection import Detect
from modules.funcs.prior_box import PriorBox
from options.cfg_priorbox import v6 as cfg

# from data.pascal import VOCDetection, AnnotationTransform, MaskTransform
# from data.augmentions import BaseAugmentation, NoiseAugmentation
from data.ssd_pascal import VOC2012Test, AnnotationTransform, MaskTransform, VOC_CLASSES
from data.augmentions_ssd import BaseAugmentation
from PIL import Image, ImageDraw, ImageFont

import sys
import os
import time
import argparse
import numpy as np
import pickle

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


labelmap = (  # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='/media/eric/GANS/epoch_290_TRAIN_DETECT_netG_.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default='/home/eric/Documents/datasets/VOCtest/VOCdevkit', help='Location of VOC root directory')
parser.add_argument('--fine_size', default=321, help='Location of VOC root directory')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

detect = Detect(21, 0, 200, 0.45)
priors = PriorBox(cfg)
priorboxes = Variable(priors.train(), volatile=True)

transformer = transforms.Compose([
    transforms.ToTensor()
])    

YEAR = '2007'
save_folder = args.save_folder + 'VOC' + YEAR
set_type = 'test'
labelmap = VOC_CLASSES

def test_net(save_folder, net, cuda, testset, top_k,
             im_size=321, thresh=0.05):
    """Test a Fast R-CNN network on an image database."""
    filename = save_folder+'{}.txt'.format(YEAR+set_type)
    num_images = len(testset)
    for i in range(num_images):
        print('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img, wh = testset.pull_item(i)

        x = transformer(img)
        x = Variable(x.unsqueeze(0))
        img_index = testset.ids[i][1]
        img_name = os.path.join(testset.ids[i][0], 'JPEGImages', testset.ids[i][1]+'.jpg')
        img = Image.open(img_name)
        draw = ImageDraw.Draw(img)
        w, h = wh

        if cuda:
            x = x.cuda()

        _, conf, loc = net(x)      # forward pass
        detections = detect(loc, F.softmax(conf.view(-1,  21)), priorboxes).data
        # scale each detection back up to the image
        scale = torch.Tensor([w, h, w, h])
        pred_num = 0
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.5:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write(str(img_name)+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                draw.rectangle(pt)
                draw.rectangle((pt[0], pt[1], pt[0]+12*len(label_name)/2, pt[1]+12), fill='white')
                draw.text((pt[0], pt[1]), text=label_name, fill='black')            
                j += 1
        save_path = save_folder+'{}_{}.jpg'.format(YEAR, img_index)    
        img.save(save_path)    

def draw(save_folder, net, cuda, im_size=321, thresh=0.05):
    filename = save_folder+'{}.txt'.format(YEAR+set_type)

if __name__ == '__main__':
    # load net
    num_classes = 21 # +1 background
    mask_transformer = transforms.Compose([
        # TODO: Scale
        transforms.Resize((64, 64)),
        transforms.ToTensor()])
    # datasets = VOCDetection(root=args.voc_root, image_sets=[('2007', 'test')], source_transform=BaseAugmentation(321), anno_transform=AnnotationTransform(), mask_transform=NoiseAugmentation())        
    datasets = VOC2012Test(root=args.voc_root, image_sets=[('2012', 'test')], transform=BaseAugmentation(321))            
    datasets_size = len(datasets)
    print(datasets_size)
    net = define_netG('res101', 3)
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    if args.cuda:
        net = net.cuda()
    # evaluation
    test_net(args.save_folder, net, args.cuda, datasets, args.top_k, 300,
             thresh=args.confidence_threshold)
