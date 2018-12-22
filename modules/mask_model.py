import os
import glob

import glog as log
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable
import sys

from .loss.loss import GANLoss, criterion_fmloss
from .funcs.layer import layer_upsample
from pycrayon import CrayonClient

from .base_model import BaseModel
from .network import define_netD, define_netG

class MASKMODEL(BaseModel):
    '''Generate Mask
    '''
    def __init__(self, opt):
        super(MASKMODEL, self).__init__(opt)
        # init attributes
        self.mode = opt.mode
        self.batch_size = opt.batch_size
        self.fine_size = opt.fine_size
        self.cuda = opt.cuda
        self.cnt = 0

        # init input
        self.source = self.Tensor(opt.batch_size, opt.input_dim, opt.fine_size, opt.fine_size)
        self.mask = self.Tensor(opt.batch_size, opt.input_dim, opt.fine_size, opt.fine_size)

        # init network
        self.netG = define_netG(opt.which_model_netG, 3)
        self.netD = define_netD(opt.which_model_netD, 1)

        # load pretrain network
        if opt.g_network_path:
            self.load_networkG(opt.g_network_path)
        if opt.d_network_path:
            self.load_networkD(opt.d_network_path)
        
        # init solver
        if opt.mode == 'train':
            self.G_solver = torch.optim.Adam(self.netG.parameters(), lr = self.opt.lr, betas=(self.opt.beta_gans, 0.999))
            self.D_solver = torch.optim.Adam(self.netD.parameters(), lr = self.opt.lr, betas=(self.opt.beta_gans, 0.999))
        
        # init loss functions
        self.criterionGAN = GANLoss(tensor = self.Tensor, cuda=opt.cuda)

        # init tesorboard
        if opt.cc:
            self.cc = CrayonClient(hostname="localhost")
            self.cc.remove_all_experiments()
            try:
                self.G_exp = self.cc.open_experiment('g_loss')
            except Exception, e:
                self.G_exp = self.cc.create_experiment('g_loss')  
            try:
                self.D_exp = self.cc.open_experiment('d_loss')
            except Exception, e:
                self.D_exp = self.cc.create_experiment('d_loss')

        # support cuda
        if opt.cuda:
            self.netD.cuda()
            self.netG.cuda()
            self.mask = self.mask.cuda()
            self.source = self.source.cuda()
        
        # support variable
        self.source = Variable(self.source)
        self.mask = Variable(self.mask)

        log.info("Training Detect Model")
    
    def draft_data(self, source, mask, wh):
        self.batch_size = source.size(0)
        self.source.data.resize_(source.size()).copy_(source)
        self.mask.data.resize_(mask.size()).copy_(mask)

    def backward_D(self):
        # fake
        self.fake = self.netG.forward(self.source)

        # get fake loss
        self.pred_fake, _ = self.netD.forward(self.fake)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # real
        self.pred_real, self.feats_real = self.netD.forward(self.mask)
        
        # get real loss
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # get d loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        
        # backward D
        self.loss_D.backward(retain_graph=True)

        self.cnt += 1

    def backward_G(self):
        # fake
        pred_fake, feats_fake = self.netD.forward(self.fake)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # get fm loss
        self.loss_fm = criterion_fmloss(self.feats_real, feats_fake, cuda=self.cuda)

        # get g loss
        self.loss_G = (self.loss_G_GAN) * 0.1 +  (self.loss_fm) * 0.9

        # backward G
        self.loss_G.backward(retain_graph=True)

    def train(self, source, mask, wh):
        # draft data from datasets
        self.draft_data(source, mask, wh)

        # Backward G
        self.D_solver.zero_grad()
        self.backward_D()
        self.D_solver.step()

        self.G_solver.zero_grad()
        self.backward_G()
        self.G_solver.step()

        # Visual
        # self.visual()

    def save_image(self, label, epoch):
        log.info("Saving MG(source/real_mask/fake_mask) - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        vutils.save_image(self.source.data, '{}/epoch{}_{}_mask_source.png'.format(self.save_dir, epoch, label))
        vutils.save_image(self.mask.data, '{}/epoch{}_{}_mask_real_mask.png'.format(self.save_dir, epoch, label))
        vutils.save_image(self.fake.data, '{}/epoch{}_{}_mask_fake_mask.png'.format(self.save_dir, epoch, label))
    
    def save_network(self, label, epoch='laterest'):
        log.info("Saving netG - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        torch.save(self.netG.state_dict(), '{}/epoch_{}_{}_mask_netG.pth'.format(self.save_dir, epoch, label))
        torch.save(self.netD.state_dict(), '{}/epoch_{}_{}_mask_netD.pth'.format(self.save_dir, epoch, label))
    
    def save(self, label, epoch):
        log.info("*" * 50)
        log.info("Epoch: {}  Iters: {}".format(epoch, self.cnt))
        save_dir = os.path.join(self.save_dir, self.name())
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_image(label, epoch)
        self.save_network(label, epoch)

    def load_networkG(self, g_network_path):
        '''load network for netG/D, H mean hotmap
        '''
        log.info("Loading network weights for netG")
        if self.cuda:
            self.netG.load_state_dict(torch.load(g_network_path))
        else:
            self.netG.load_state_dict(torch.load(g_network_path, map_location=lambda storage, loc: storage))

    def load_networkD(self, d_network_path):
        '''load network for netG/D, H mean hotmap
        '''
        log.info("Loading network weights for netd")
        if self.cuda:
            self.netD.load_state_dict(torch.load(d_network_path))
        else:
            self.netD.load_state_dict(torch.load(d_network_path, map_location=lambda storage, loc: storage))
    
    def load_pre_network(self, pretrained_path):
        '''load network for netG/D, H mean hotmap
        '''
        log.info("Loading pretrained network weights for resnet18 in netG")
        if self.cuda:
            self.netG.resnet.load_state_dict(torch.load(pretrained_path))
        else:
            self.netG.resnet.load_state_dict(torch.load(pretrained_path, map_location=lambda storage, loc: storage))

    def current_errors(self):
        return [('loss_G', self.loss_G.data[0]), ('loss_D', self.loss_D.data[0])]

    def visual(self):
        self.D_exp.add_scalar_value('D_loc_loss', self.loss_D.data[0], step=self.cnt)
        self.G_exp.add_scalar_value('G_loc_loss', self.loss_G.data[0], step=self.cnt)
    
    def name(self):
        return self.opt.name + 'model'

    def __str__(self):
        return 'netG:\n {} netD: {}'.format(self.netG.__str__(), self.netD.__str__())