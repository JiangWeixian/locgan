import torch.nn as nn
import torch

# network weight init
def gans_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #print m.weight.data.size()
        m.weight.data.normal_(0.0, 0.02)
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1 or  classname.find('InstanceNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def classify_weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

def xavier(param):
    nn.init.xavier_normal(param)

def load_weights(net, weights, layer_target, weights_target):
    assert len(layer_target) == len(weights_target)
    for (l, t) in zip(layer_target, weights_target):
        if l not in net.state_dict().keys() or t not in weights.keys():
            raise KeyError('unexpected key "{} {}" in state_dict'
                               .format(l, t))
        else:
            try:
                net.state_dict()[l].copy_(weights[t])
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                          name, net.state_dict()[name].size(), weights[t].size()))
                raise