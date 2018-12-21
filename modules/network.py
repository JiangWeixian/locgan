import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import math
import torch
import functools

from config import grah_netD
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
        netG = ResnetGenerator(3, 1)
        netG.apply(gans_weights_init)
    return netG


def define_netD(which_model_netD, x_dim):
    '''build network D

    @Params:
    - which_model_netD: decide use whick netD class to define netD
    - network_config: the network grah config for netD
    - x_dim: the dim of input data for network

    @Returns:
    the network D
    '''
    network = create_gans_network_grah(grah_netD[which_model_netD], x_dim)[0]
    if which_model_netD == 'fm':
        netD = GanFMDiscriminator(layers=network)
        netD.apply(gans_weights_init)
    return netD

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

# Res
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            x = self.model(input)
            return x


# Define a resnet block
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

if __name__ == '__main__':
    netG = ResnetGenerator(3, 1)
    input = Variable(torch.randn(1, 3, 300, 300))
    print(netG(input))
    netD = define_netD('fm', 1)
    print(netD)
