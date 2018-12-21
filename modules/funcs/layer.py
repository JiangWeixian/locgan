import torch
import torch.nn.functional as F
import numpy as np

def layer_upsample(features, rpnbox, scale, o_size):
    num = rpnbox.size(0)
    i_size = features.size(-1)
    for i in range(num):
        bbox =(rpnbox[i, 1:]*scale).clamp(min=0, max=i_size).int()
        xmin, ymin, xmax, ymax = bbox.data
        h = max(ymax - ymin, 1)
        w = max(xmax - xmin, 1)
        gt_foi = features[:, :, int(ymin):int(ymin+h), int(xmin):int(xmin+w)]
        foi = F.upsample(gt_foi, size=(int(o_size), int(o_size)), mode='bilinear')
        if i == 0:
            fois = foi
        else:
            fois = torch.cat([fois, foi], dim=0)
    return fois
