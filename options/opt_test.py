import argparse
from .cfg_gans import grah_netG, grah_netD, grah_netL, grah_netC
from .cfg_priorbox import v1, v2, v3, v4, v6

mode = 'train'
parser = argparse.ArgumentParser()
parser.add_argument('--name', default='coco-rgb21', help='the name of this model, deside the savepath of .pth .png files')
parser.add_argument('--mode', default='test', help='train mode: train/pre-train/test/continue')
parser.add_argument('--fine_size', default=300, help='the img will be resize as [fine_size, fine_size]')
parser.add_argument('--input_dim', default=3, help='the nums of color channels, RGB imgs channels is 3')
parser.add_argument('--batch_size', default=1, help='the batch size of training data')
parser.add_argument('--cfg_priors', default=v6, help='the config for Priorboxes')
parser.add_argument('--output_dim', default=1, help='the nums of color channels of netG')
parser.add_argument('--cuda', default=True, help='support cuda or not')
parser.add_argument('--cc', default=False, help='support tensorboard or not')
parser.add_argument('--which_model_netG', default='resmask', help='deside which netG class to build netG')
parser.add_argument('--networkG_config', default=grah_netG['base'], help='define the grah of netG, the map of config and grah writed in cfg_gans.py')
parser.add_argument('--which_model_netD', default='fm', help='deside which netD class to build netD')
parser.add_argument('--networkD_config', default=grah_netD['mask'], help='define the grah of netD, the map of config and grah writed in cfg_gans.py')
parser.add_argument('--save_dir', default='testpoints/', help='the save dir of train-output')
parser.add_argument('--detection_path', default='detection.txt', help='the detect output result of model')
parser.add_argument('--eval_path', default='eval.txt', help='the all detect output result of model')
parser.add_argument('--save_epoch_freq', default=1, help='frequency of saving results')
parser.add_argument('--g_network_path', default='/home/eric/Documents/PY27/weights/res101-dssd-coco/epoch_4_TRAIN_DETECT_netG_.pth', help='For continue train mode, the path of netG weights')
parser.add_argument('--d_network_path', default='/home/eric/Documents/STABLE/gans-detection-mask/weights/mask-up/epoch99_TRAIN_DETECT_netD.pth', help='For continue train mode, the path of netD weights')
parser.add_argument('--m_network_path', default='weights/epoch_laterest_TRAIN_DETECT_netM.pth', help='For continue train mode, the path of netL weights')
parser.add_argument('--num_classes', default=21, help='class num')
parser.add_argument('--nms_thresh', default=0.5, help='nms for detection')
parser.add_argument('--overlap_thresh', default=0.5, help='threshold for match')
parser.add_argument('--bkg_label', default=0, help='the label for background')
parser.add_argument('--detect_threshold', default=0.5, help='threshold for mask priorboxes')

opt = parser.parse_args()