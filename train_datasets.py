from data.coco import COCOVoc, COCOAnnotationTransform, COCOTestVoc
from data.augmentions import mask_collate
import torch
import torchvision.transforms as transforms
import numpy as np

from modules.util.scafford import set_npz


def train_datasets_npz(path):
    '''In coco.py, set the classname=bus, so bus datasets, bus imgs, bus annos

    @Params:
    - path: npz file path
    '''
    transform = transforms.Compose([
        transforms.Scale(size=(288, 288)),
        transforms.ToTensor()])
    datasets = COCOVoc(transform=transform, target_transform=transform, anno_transform=COCOAnnotationTransform(keep_difficult=True))
    class_recs = {}
    for i in range(0, len(datasets.images)):
        img_path = datasets.images[i].split('/')[-1]
        anno = datasets[i][2][:, :-1]
        det = [False] * anno.shape[0]
        class_recs[img_path] = {'bbox': anno, 'det': det}
    set_npz(class_recs, path)    

def train_valdatasets_npz(path):
    '''In coco.py, set the classname=bus, so bus datasets, bus imgs, bus annos

    @Params:
    - path: npz file path
    '''
    transform = transforms.Compose([
        transforms.Scale(size=(288, 288)),
        transforms.ToTensor()])
    datasets = COCOVoc(train=False, transform=transform, target_transform=transform, anno_transform=COCOAnnotationTransform(keep_difficult=True))
    class_recs = {}
    for i in range(0, len(datasets.images)):
        img_path = datasets.images[i].split('/')[-1]
        if img_path == 'COCO_val2014_000000205782.jpg':
            print(img_path)
        anno = datasets[i][2][:, :-1]
        det = [False] * anno.shape[0]
        class_recs[img_path] = {'bbox': anno, 'det': det}
    set_npz(class_recs, path)    

if __name__ == '__main__':
    train_valdatasets_npz('busval.npz')
    # train_datasets_npz('bus.npz')
    # data = np.load('bus.npz')
    # print(data.files)
    # for key, val in data.items():
    #     print(val)