'''
train locgan entry file
'''
import torch
from dataloader.images_folder import IMAGELOADER
from options.opt_gans import opts
from modules.mask_model import MASKMODEL
from dataloader.augmentions import mask_collate

Datasets = IMAGELOADER()
Dataloader = torch.utils.data.DataLoader(
    Datasets,
    batch_size=opts.batch_size,
    shuffle=True,
    collate_fn=mask_collate,
    num_workers=2)
Network = MASKMODEL(opts)

for epoch in range(opts.epochs):
  for i, (source, mask, wh) in enumerate(Dataloader):
        print (i, source, mask, wh)
        if i == 10:
            break