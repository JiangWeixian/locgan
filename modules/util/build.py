import torch.nn as nn

fine_size = 288
def create_gans_network_grah(cfg, x_dim = 0):
    '''create gans network grah by cfg

    @Params:
    - cfg: the config of network grah, default saved in `./options/cfg_gans.py`. 
    - x_dim: the dim of input data

    @Return:
    a array, stored the network grah
    '''
    layers = []
    i_dim = x_dim
    for v in cfg:
        if v == 'R':
            layers += [nn.ReLU(True)]
        elif v == 'LR':
            layers += [nn.LeakyReLU(0.2, inplace=True)]
        elif v == 'S':
            layers += [nn.Sigmoid()]
        elif v == 'TH':
            layers += [nn.Tanh()]
        elif v == 'B':
            layers += [nn.BatchNorm2d(i_dim)]
        elif v == 'P':
            layers += [nn.MaxPool2d(2, stride=2, ceil_mode=True)]
        elif v == 'AP':
            layers += [nn.AvgPool2d(fine_size/4)]
        elif type(v) == tuple:
            layer_type, o_dim, k, s, p = v
            if layer_type == 'DCONV':
                layers += [nn.ConvTranspose2d(i_dim, o_dim, kernel_size=k, stride=s, padding=p, bias=False)]
                i_dim = o_dim
            elif layer_type == 'CONV':
                layers += [nn.Conv2d(i_dim, o_dim, kernel_size=k, stride=s, padding=p, bias=False)]
                i_dim = o_dim
        elif type(v) == list:
            layer, o_dim = create_gans_network_grah(v, i_dim)
            i_dim = o_dim
            layers.append(layer)
    return layers, i_dim

def create_location_network_grah(cfg, base_dim = 128, expansion=2):
    '''create location network grah by cfg

    @Params:
    - cfg: the config of network grah, default saved in `./options/cfg_gans.py`. 
    - base_dim: the dim of input data

    @Return:
    a array, stored the network grah
    '''
    layers = []
    i_dim = base_dim
    for v in cfg:
        layer_type, o_dim, k, s, p = v
        layers += [nn.Conv2d(i_dim, o_dim, kernel_size=k, stride=s, padding=1)]
        i_dim *= expansion
    return layers

if __name__ == '__main__':
    cfg = [[('CONV', 64, 4, 2, 1), 'LR'], [('CONV', 64, 4, 2, 1), 'LR']]
    cfg1 = [('CONV', 64, 4, 2, 1), 'LR', ('CONV', 64*2, 4, 2, 1), 'B', ('CONV', 64*4, 4, 2, 1), 'LR', ('CONV', 64*8, 4, 2, 1), 'LR', ('CONV', 100, 4, 1, 0), 'B', 'LR', ('DCONV', 64*8, 4, 1, 0), 'B','R', ('DCONV', 64*4, 4, 2, 1), 'B','R', ('DCONV', 64*2, 4, 2, 1), 'B','R', ('DCONV', 64, 4, 2, 1), 'B','R', ('DCONV', 3, 4, 2, 1), 'S']
    cfg_location = [('CONV', 16, 3, 1, 1), ('CONV', 16, 3, 1, 1), ('CONV', 24, 3, 1, 1)]
    #print (len(create_gans_network_grah(cfg, 3)))