import torch

from modules.network import NetLocation, NetMask
from torch.autograd import Variable

# netL = NetLocation()
# input = [Variable(torch.randn(1, 128, 24, 24)), Variable(torch.randn(1, 256, 12, 12)), Variable(torch.randn(1, 512, 6, 6))]

# netL.forward(input)

netM = NetMask(512, 3)
input = [Variable(torch.randn(128, 128,  24, 24)), Variable(torch.randn(128, 256, 12, 12)), Variable(torch.randn(128, 512, 6, 6))]
