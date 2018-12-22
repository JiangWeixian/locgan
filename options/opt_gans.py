import argparse
import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from dataloader.config import FINE_SIZE, BATCH_SZIE

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='mask', help='the name of this model, deside the savepath of .pth .png files')
parser.add_argument('--mode', default='train', help='train mode: train/pre-train/test/continue')
parser.add_argument('--batch_size', default=BATCH_SZIE, help='the batch size of training data')
parser.add_argument('--fine_size', default=FINE_SIZE, help='the img will be resize as [fine_size, fine_size]')
parser.add_argument('--epochs', default=150, help='total train 200 epochs')
parser.add_argument('--input_dim', default=3, help='the nums of color channels, RGB imgs channels is 3')
parser.add_argument('--output_dim', default=1, help='the nums of color channels of netG')
parser.add_argument('--beta_gans', default=0.5, help='the beta in optim')
parser.add_argument('--lr', default=2e-4, help='learning rate')
parser.add_argument('--cuda', default=True, help='support cuda or not')
parser.add_argument('--cc', default=False, help='support tensorboard or not')
parser.add_argument('--default_rate', default=0.1)
parser.add_argument('--improve_rate', default=0.5)
parser.add_argument('--which_model_netG', default='mask', help='deside which netG class to build netG')
parser.add_argument('--which_model_netD', default='fm', help='deside which netD class to build netD')
parser.add_argument('--save_dir', default='checkpoints/', help='the save dir of train-output')
parser.add_argument('--save_epoch_freq', default=100, help='frequency of saving results')
parser.add_argument('--g_network_path', default='')
parser.add_argument('--d_network_path', default='')

opts = parser.parse_args()