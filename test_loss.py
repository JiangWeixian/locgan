from modules.loss.loss import GANConfLoss
import torch
from torch.autograd import Variable

pred = Variable(torch.randn(32, 21))
gt = Variable(torch.ones(32, 1).type(torch.LongTensor))
print(pred, gt)
criterion = GANConfLoss(21, 0, True)
criterion(pred, gt)