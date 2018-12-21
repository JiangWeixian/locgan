import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
import torch
import functools

from util.weight import gans_weights_init, classify_weights_init
from util.build import create_gans_network_grah, create_location_network_grah
# from data.config import CLASSES

# define network
def define_netG(which_model_netG, x_dim):
    '''build network G

    @Params:
    - which_model_netG: decide use whick netG class to define netG
    - network_config: the network grah config for netG
    - x_dim: the dim of input data for network

    @Returns:
    the network G
    '''
    if which_model_netG == 'mask':
        netG = NetMask(x_dim, 1)
        netG.apply(gans_weights_init)
    elif which_model_netG == 'unet':
        norm_layer = get_norm_layer('batch')
        netG = UnetGenerator(3, 3, 8, 64, norm_layer=norm_layer, use_dropout=False, gpu_ids=[0])
        netG.apply(gans_weights_init)
    elif which_model_netG == 'unetmask':
        norm_layer = get_norm_layer('batch')
        netG = UnetMask()
        netG.apply(gans_weights_init)    
    elif which_model_netG == 'mask2':
        netG = NetMask2(x_dim, 1)
        netG.apply(gans_weights_init)    
    elif which_model_netG == 'res18':
        netG = ResNet18Dssd(BasicBlock, [2, 2, 2, 2])
        netG.apply(gans_weights_init)
    elif which_model_netG == 'res101':
        netG = ResNet101Dssd(Bottleneck, [3, 4, 23, 3])
        netG.apply(gans_weights_init)
    return netG


def define_netD(which_model_netD, network_config, x_dim):
    '''build network D

    @Params:
    - which_model_netD: decide use whick netD class to define netD
    - network_config: the network grah config for netD
    - x_dim: the dim of input data for network

    @Returns:
    the network D
    '''
    network = create_gans_network_grah(network_config, x_dim)[0]
    if which_model_netD == 'fm':
        netD =  GanFMDiscriminator(layers=network)
        netD.apply(gans_weights_init)
    elif which_model_netD == 'fm2':
        netD =  GanFMDiscriminator2(layers=network)
        netD.apply(gans_weights_init)
    elif which_model_netD == 'res101':
        netD = ResNet101Dis(Bottleneck, [3, 4, 23, 3])
    return netD

def define_netL(which_model_netL, x_dim):
    '''build network L
    '''
    if which_model_netL == 'res18':
        netL =  NetLocation()
    return netL

class ResNet101Dis(nn.Module):
    def __init__(self, block, layers, num_classes=21):
        super(ResNet101Dis, self).__init__()
        self.resnet = ResNet(block=block, layers=layers, num_classes=1000)
        self.num_classes = num_classes

        # extras
        self.extras = nn.ModuleList([
            nn.Conv2d(2048, 512, 1, 1),
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.Conv2d(1024, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.Conv2d(512, 128, 1, 1),
            nn.Conv2d(128, 256, 3, 2, 1)
        ])

        # deconv modules - 7&8
        self.levelconv7_x = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.deconv8_x = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 3, 2, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        # deconv modules - 6&7
        self.levelconv6_x = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.deconv7_x = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, 2, 1, output_padding=1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        # deconv modules - 5&6
        self.levelconv5_x = nn.Sequential(
            nn.Conv2d(2048, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.deconv6_x = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, 2, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        # deconv modules - 4&5
        self.levelconv4_x = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.deconv5_x = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, 2, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        # deconv modules - 3&4
        self.levelconv3_x = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.deconv4_x = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, 2, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.levelconv_lists = [self.levelconv7_x, self.levelconv6_x, self.levelconv5_x, self.levelconv4_x, self.levelconv3_x]
        self.deconv_lists = [self.deconv8_x, self.deconv7_x, self.deconv6_x,self.deconv5_x, self.deconv4_x]

        self.dense = nn.ModuleList([
            nn.Conv2d(512, 1024, 1, 1),
            nn.Conv2d(512, 1024, 1, 1),
            nn.Conv2d(512, 1024, 1, 1),
            nn.Conv2d(512, 1024, 1, 1),
            nn.Conv2d(512, 1024, 1, 1),
            nn.Conv2d(256, 1024, 1, 1)
        ])

        self.res = nn.ModuleList([
            # loc&conf1
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1),
            # loc&conf2
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1),
            # loc&conf3
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1),
            # loc&conf4
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1),
            # loc&conf5
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1),
            #loc&conf6 
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1)
        ])

        self.score_conv = nn.ModuleList([
            nn.Conv2d(1024, 4*num_classes, 1, 1),
            nn.Conv2d(1024, 8*num_classes, 1, 1),
            nn.Conv2d(1024, 8*num_classes, 1, 1),
            nn.Conv2d(1024, 8*num_classes, 1, 1),
            nn.Conv2d(1024, 4*num_classes, 1, 1),
            nn.Conv2d(1024, 4*num_classes, 1, 1)
        ])
        self.bbox_conv = nn.ModuleList([
            nn.Conv2d(1024, 4*4, 1, 1),
            nn.Conv2d(1024, 8*4, 1, 1),
            nn.Conv2d(1024, 8*4, 1, 1),
            nn.Conv2d(1024, 8*4, 1, 1),
            nn.Conv2d(1024, 4*4, 1, 1),
            nn.Conv2d(1024, 4*4, 1, 1)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, input):
        sources = []
        fts = self.resnet(input)
        x = fts[-1]

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                fts.append(x)
        x = fts[-1]
        sources.append(x)

        for i, d, l in zip([4, 3, 2, 1, 0], self.deconv_lists, self.levelconv_lists):
            x = F.relu(d(x) + l(fts[i]))
            sources.insert(0, x)
        
        res = []
        for i in range(len(sources)):
            x = sources[i]
            for j in range(i*3, (i+1)*3):
                x = self.res[j](x)
            res.append(x)
        
        loc = []
        conf = []
        for x, y, d, c, l in zip(sources, res, self.dense, self.score_conv, self.bbox_conv):
            x = y + d(x)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        return F.sigmoid(sources[0]), sources, conf , loc


class ResNet101Dssd512(nn.Module):
    def __init__(self, block, layers, num_classes=21):
        super(ResNet101Dssd512, self).__init__()
        self.resnet = ResNet(block=block, layers=layers, num_classes=1000)
        self.num_classes = num_classes

        # extras
        self.extras = nn.ModuleList([
            nn.Conv2d(2048, 512, 1, 1),
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.Conv2d(1024, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.Conv2d(512, 128, 1, 1),
            nn.Conv2d(128, 256, 3, 2, 1)
        ])

        # deconv modules - 7&8
        self.levelconv7_x = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.deconv8_x = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 4, 2, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        # deconv modules - 6&7
        self.levelconv6_x = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.deconv7_x = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        # deconv modules - 5&6
        self.levelconv5_x = nn.Sequential(
            nn.Conv2d(2048, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.deconv6_x = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        # deconv modules - 4&5
        self.levelconv4_x = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.deconv5_x = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        # deconv modules - 3&4
        self.levelconv3_x = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.deconv4_x = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.levelconv_lists = [self.levelconv7_x, self.levelconv6_x, self.levelconv5_x, self.levelconv4_x, self.levelconv3_x]
        self.deconv_lists = [self.deconv8_x, self.deconv7_x, self.deconv6_x,self.deconv5_x, self.deconv4_x]

        self.dense = nn.ModuleList([
            nn.Conv2d(512, 1024, 1, 1),
            nn.Conv2d(512, 1024, 1, 1),
            nn.Conv2d(512, 1024, 1, 1),
            nn.Conv2d(512, 1024, 1, 1),
            nn.Conv2d(512, 1024, 1, 1),
            nn.Conv2d(256, 1024, 1, 1)
        ])

        self.res = nn.ModuleList([
            # loc&conf1
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1),
            # loc&conf2
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1),
            # loc&conf3
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1),
            # loc&conf4
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1),
            # loc&conf5
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1),
            #loc&conf6 
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1)
        ])

        self.score_conv = nn.ModuleList([
            nn.Conv2d(1024, 4*num_classes, 1, 1),
            nn.Conv2d(1024, 8*num_classes, 1, 1),
            nn.Conv2d(1024, 8*num_classes, 1, 1),
            nn.Conv2d(1024, 8*num_classes, 1, 1),
            nn.Conv2d(1024, 4*num_classes, 1, 1),
            nn.Conv2d(1024, 4*num_classes, 1, 1)
        ])
        self.bbox_conv = nn.ModuleList([
            nn.Conv2d(1024, 4*4, 1, 1),
            nn.Conv2d(1024, 8*4, 1, 1),
            nn.Conv2d(1024, 8*4, 1, 1),
            nn.Conv2d(1024, 8*4, 1, 1),
            nn.Conv2d(1024, 4*4, 1, 1),
            nn.Conv2d(1024, 4*4, 1, 1)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, input):
        sources = []
        fts = self.resnet(input)
        x = fts[-1]

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                fts.append(x)
        x = fts[-1]
        sources.append(x)

        for i, d, l in zip([4, 3, 2, 1, 0], self.deconv_lists, self.levelconv_lists):
            x = F.relu(d(x) + l(fts[i]))
            sources.insert(0, x)
        
        res = []
        for i in range(len(sources)):
            x = sources[i]
            for j in range(i*3, (i+1)*3):
                x = self.res[j](x)
            res.append(x)
        
        loc = []
        conf = []
        for x, y, d, c, l in zip(sources, res, self.dense, self.score_conv, self.bbox_conv):
            x = y + d(x)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())   

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        return sources[0], conf , loc

class ResNet101Dssd(nn.Module):
    def __init__(self, block, layers, num_classes=21):
        super(ResNet101Dssd, self).__init__()
        self.resnet = ResNet(block=block, layers=layers, num_classes=1000)
        self.num_classes = num_classes

        # extras
        self.extras = nn.ModuleList([
            nn.Conv2d(2048, 512, 1, 1),
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.Conv2d(1024, 256, 1, 1),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.Conv2d(512, 128, 1, 1),
            nn.Conv2d(128, 256, 3, 2, 1)
        ])

        # deconv modules - 7&8
        self.levelconv7_x = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.deconv8_x = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 3, 2, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        # deconv modules - 6&7
        self.levelconv6_x = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.deconv7_x = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, 2, 1, output_padding=1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        # deconv modules - 5&6
        self.levelconv5_x = nn.Sequential(
            nn.Conv2d(2048, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.deconv6_x = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, 2, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        # deconv modules - 4&5
        self.levelconv4_x = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.deconv5_x = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, 2, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        # deconv modules - 3&4
        self.levelconv3_x = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.deconv4_x = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 3, 2, 1),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
        )

        self.levelconv_lists = [self.levelconv7_x, self.levelconv6_x, self.levelconv5_x, self.levelconv4_x, self.levelconv3_x]
        self.deconv_lists = [self.deconv8_x, self.deconv7_x, self.deconv6_x,self.deconv5_x, self.deconv4_x]

        self.dense = nn.ModuleList([
            nn.Conv2d(512, 1024, 1, 1),
            nn.Conv2d(512, 1024, 1, 1),
            nn.Conv2d(512, 1024, 1, 1),
            nn.Conv2d(512, 1024, 1, 1),
            nn.Conv2d(512, 1024, 1, 1),
            nn.Conv2d(256, 1024, 1, 1)
        ])

        self.res = nn.ModuleList([
            # loc&conf1
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1),
            # loc&conf2
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1),
            # loc&conf3
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1),
            # loc&conf4
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1),
            # loc&conf5
            nn.Conv2d(512, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1),
            #loc&conf6 
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 1024, 1, 1)
        ])

        self.score_conv = nn.ModuleList([
            nn.Conv2d(1024, 4*num_classes, 1, 1),
            nn.Conv2d(1024, 8*num_classes, 1, 1),
            nn.Conv2d(1024, 8*num_classes, 1, 1),
            nn.Conv2d(1024, 8*num_classes, 1, 1),
            nn.Conv2d(1024, 4*num_classes, 1, 1),
            nn.Conv2d(1024, 4*num_classes, 1, 1)
        ])
        self.bbox_conv = nn.ModuleList([
            nn.Conv2d(1024, 4*4, 1, 1),
            nn.Conv2d(1024, 8*4, 1, 1),
            nn.Conv2d(1024, 8*4, 1, 1),
            nn.Conv2d(1024, 8*4, 1, 1),
            nn.Conv2d(1024, 4*4, 1, 1),
            nn.Conv2d(1024, 4*4, 1, 1)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, input):
        sources = []
        fts = self.resnet(input)
        x = fts[-1]

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                fts.append(x)
        x = fts[-1]
        sources.append(x)

        for i, d, l in zip([4, 3, 2, 1, 0], self.deconv_lists, self.levelconv_lists):
            x = F.relu(d(x) + l(fts[i]))
            sources.insert(0, x)
        
        res = []
        for i in range(len(sources)):
            x = sources[i]
            for j in range(i*3, (i+1)*3):
                x = self.res[j](x)
            res.append(x)
        
        loc = []
        conf = []
        for x, y, d, c, l in zip(sources, res, self.dense, self.score_conv, self.bbox_conv):
            x = y + d(x)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())   

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        return sources[0], conf , loc

class ResNet18Dssd(nn.Module):
    def __init__(self, block, layers, num_classes=21):
        super(ResNet18Dssd, self).__init__()
        self.resnet = ResNet(block=block, layers=layers, num_classes=num_classes)
        self.num_classes = num_classes

        # extras
        self.extras = nn.ModuleList([
            nn.Conv2d(512, 128, 1, 1),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.Conv2d(256, 128, 1, 1),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.Conv2d(256, 128, 1, 1),
            nn.Conv2d(128, 256, 3, 2, 1)
        ])

        # deconv modules - 7&8
        self.levelconv7_x = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )

        self.deconv8_x = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )

        # deconv modules - 6&7
        self.levelconv6_x = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )

        self.deconv7_x = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )

        # deconv modules - 5&6
        self.levelconv5_x = nn.Sequential(
            nn.Conv2d(512, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )

        self.deconv6_x = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, 1, output_padding=1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )

        # deconv modules - 4&5
        self.levelconv4_x = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )

        self.deconv5_x = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )

        # deconv modules - 3&4
        self.levelconv3_x = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )

        self.deconv4_x = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 3, 2, 1, output_padding=1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )

        self.levelconv_lists = [self.levelconv7_x, self.levelconv6_x, self.levelconv5_x, self.levelconv4_x, self.levelconv3_x]
        self.deconv_lists = [self.deconv8_x, self.deconv7_x, self.deconv6_x,self.deconv5_x, self.deconv4_x]

        self.dense = nn.ModuleList([
            nn.Conv2d(128, 256, 1, 1),
            nn.Conv2d(128, 256, 1, 1),
            nn.Conv2d(128, 256, 1, 1),
            nn.Conv2d(128, 256, 1, 1),
            nn.Conv2d(128, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1)
        ])

        self.res = nn.ModuleList([
            # loc&conf1
            nn.Conv2d(128, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            # loc&conf2
            nn.Conv2d(128, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            # loc&conf3
            nn.Conv2d(128, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            # loc&conf4
            nn.Conv2d(128, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            # loc&conf5
            nn.Conv2d(128, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            #loc&conf6 
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1),
            nn.Conv2d(256, 256, 1, 1)
        ])

        self.score_conv = nn.ModuleList([
            nn.Conv2d(256, 4*num_classes, 1, 1),
            nn.Conv2d(256, 8*num_classes, 1, 1),
            nn.Conv2d(256, 8*num_classes, 1, 1),
            nn.Conv2d(256, 8*num_classes, 1, 1),
            nn.Conv2d(256, 4*num_classes, 1, 1),
            nn.Conv2d(256, 4*num_classes, 1, 1)
        ])
        self.bbox_conv = nn.ModuleList([
            nn.Conv2d(256, 4*4, 1, 1),
            nn.Conv2d(256, 8*4, 1, 1),
            nn.Conv2d(256, 8*4, 1, 1),
            nn.Conv2d(256, 8*4, 1, 1),
            nn.Conv2d(256, 4*4, 1, 1),
            nn.Conv2d(256, 4*4, 1, 1)
        ])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, input):
        sources = []
        fts = self.resnet(input)
        x = fts[-1]

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                fts.append(x)
        x = fts[-1]
        sources.append(x)
        for i, d, l in zip([4, 3, 2, 1, 0], self.deconv_lists, self.levelconv_lists):
            x = F.relu(d(x) + l(fts[i]))
            sources.insert(0, x)
        
        res = []
        for i in range(len(sources)):
            x = sources[i]
            for j in range(i*3, (i+1)*3):
                x = self.res[j](x)
            res.append(x)
        
        loc = []
        conf = []
        # error
        for x, y, d, c, l in zip(sources, res, self.dense, self.score_conv, self.bbox_conv):
            x = y + d(x)
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        loc = loc.view(loc.size(0), -1, 4)
        conf = conf.view(conf.size(0), -1, self.num_classes)
        return sources[0], conf , loc

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        feats = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        feats.append(x)
        x = self.layer3(x)
        feats.append(x)
        x = self.layer4(x)
        feats.append(x)

        return feats

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer        

class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)

class NetMask(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(NetMask, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc

        self.deconvmodel = nn.Sequential(
            # *2
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 1, 1, 0),
            nn.Tanh()
        )

    
    def forward(self, input):
        x = self.deconvmodel(input)
        x = x.view(x.size(0), 3, 64, 64)
        return x

class NetMask2(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(NetMask2, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc

        self.deconvmodel = nn.Sequential(
            # *2
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 7, 1),
            nn.Tanh()
        )

    
    def forward(self, input):
        x = self.deconvmodel(input)
        return x        

class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# netD
class DCDiscriminator(nn.Module):
    def __init__(self, ngpu=0, ndf=64, nc=3):
        super(DCDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input) 
        return output.view(-1, 1).squeeze(1) 

class GanFMDiscriminator(nn.Module):
    '''The netD class

    @Params:
     - layers: a array of network grah
    '''
    def __init__(self, layers):
        super(GanFMDiscriminator, self).__init__()
        self.length = len(layers)
        network_parts = []
        for i in range(self.length):
            part = nn.Sequential(*layers[i])
            network_parts.append(part)
        self.main = nn.ModuleList(network_parts)
        self.conf = nn.Linear(512*4*4, 21)

    def forward(self, input):
        x = input
        outputs = []
        for index in range(self.length):
            x = self.main[index](x)
            outputs.append(x)
        x = outputs[-2].view(outputs[-2].size(0), -1)
        conf = self.conf(x)
        return outputs[-1], outputs[:-1], conf

class GanFMDiscriminator2(nn.Module):
    '''The netD class

    @Params:
     - layers: a array of network grah
    '''
    def __init__(self, layers):
        super(GanFMDiscriminator2, self).__init__()
        self.length = len(layers)
        network_parts = []
        for i in range(self.length):
            part = nn.Sequential(*layers[i])
            network_parts.append(part)
        self.main = nn.ModuleList(network_parts)

    def forward(self, input):
        x = input
        outputs = []
        for index in range(self.length):
            x = self.main[index](x)
            outputs.append(x)
        return outputs[-1], outputs[:-1]


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# net NORM
class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x /= norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class L1Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L1Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.norm(1, dim=1, keepdim=True)
        x = x / norm
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

class NetL1Norms(nn.Module):
    '''The norm layers networks

    @Params:
     - layers: a array of network grah
    '''
    def __init__(self, layers, scale):
        super(NetL1Norms, self).__init__()
        self.scale = scale or 1
        self.norm1 = L1Norm(layers[0], self.scale)
        self.norm2 = L1Norm(layers[1], self.scale)
        self.norm3 = L1Norm(layers[2], self.scale)
        self.norm4 = L1Norm(layers[3], self.scale)
        self.norm5 = L1Norm(layers[4], self.scale)
        self.norm6 = L1Norm(layers[5], self.scale)
        self.norms = [self.norm1, self.norm2, self.norm3, self.norm4, self.norm5, self.norm6]
    
    def forward(self, feats):
        output = []
        for (x, l) in zip (feats, self.norms):
            output.append(l.forward(x))
        return output

class UnetMask(nn.Module):
    def __init__(self):
        super(UnetMask, self).__init__()
        self.size = 321
        norm_layer = get_norm_layer('batch')
        self.unet = UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False, gpu_ids=[0])
    
    def _upsample(self, input):
        output = F.upsample(input, size=(int(self.size), int(self.size)), mode='bilinear')
        return output

    def forward(self, input):
        x = self.unet(input)
        # x = self._upsample(x)
        return x

if __name__ == '__main__':
    # NETG
    # netG = ResnetGeneratorMASK(3,1)
    # input = Variable(torch.randn(1, 3, 300, 300))
    # feats, conf, loc = netG.forward(input)
    # for item in feats:
    #     print(item.size())

    # netG = ResNet(BasicBlock, [2, 2, 2, 2])
    # input = Variable(torch.randn(1, 3, 300, 300))
    # x = netG.forward(input)
    # print(x.size())

    netG = ResNet101Dssd512(Bottleneck, [3, 4, 23, 3])
    # netG = ResNet18Dssd(BasicBlock, [2, 2, 2, 2])
    # print(netG)
    input = Variable(torch.randn(1, 3, 512, 512))
    netG.forward(input)

    # print(ResNet(Bottleneck, [3, 4, 23, 3]))

    # NET M
    # netM = NetMask2(512, 3)
    # print(netM)
    # input = Variable(torch.randn(1, 512, 41, 41))
    # print(netM.forward(input).size())
    # fm = [[('CONV', 64, 4, 2, 1), 'LR', ('CONV', 64*2, 4, 2, 1), 'B','LR'], [('CONV', 64*4, 4, 2, 1), 'B','LR'], [('CONV', 64*8, 4, 2, 1), 'B','LR'], [('CONV', 1, 4, 1, 0), 'S']]
    # net = define_netD('fm', fm, 3)
    # input = Variable(torch.randn(1, 3, 64, 64))
    # net.forward(input)
    # print(net)

    # netM = test(0)
    # print(netM)
    # input = Variable(torch.randn(1, 3, 64, 64))
    # netM.forward(input)

    # NET L
    # netl = define_netL('res18', 512)
    # print(netl)
    # input = [Variable(torch.randn(128, 128, 7, 7)), Variable(torch.randn(128, 128, 3, 3)), Variable(torch.randn(128, 128, 2, 2))]
    # netM = define_netG('unet', 3)
    # print(netM)
    # input = Variable(torch.randn(1, 3, 321, 321))
    # print(netM.forward(input).size())
    # net = UnetMask()
    # input = Variable(torch.randn(1, 3, 256, 256))
    # print(net.forward(input).size())