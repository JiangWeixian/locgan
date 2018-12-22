datasets_root = '/home/eric/Desktop/Project-PY/pro-py27/01GANs/gans-detection/data/'
coco_datasets_root = '/home/eric/Documents/datasets/coco/'  

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'boat', 
    'bird', 'cat', 'dog', 'horse', 
    'sheep', 'cow', 'skateboard', 'bottle', 
    'chair', 'potted plant', 'dining table', 'tv'
]

gap =255
mean = [ 119.39717555/255,  114.11720509/255,  105.33135092/255]
std=[ 62.62723199/255,  61.18791482/255,  61.99810061/255]

# IMAGE_FOLDER_LOADER
IMAGE_PATH = '/media/eric/Elements/datasets/coco'
FINE_SIZE = 300
