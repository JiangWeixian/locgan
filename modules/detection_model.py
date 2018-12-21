import os
import glob

import random
import numpy.random as npr
import glog as log
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable

from .loss.loss import GANLoss, criterion_fmloss, MultiBoxLoss, SimpleMultiBoxLoss, GANConfLoss
from .util.bbox import box_copy, mask_crop
from .funcs.prior_box import PriorBox
from .funcs.detection import Detect
from .funcs.rpn import RPN
from .funcs.layer import layer_upsample
# from roi_pooling.modules.roi_pool import RoIPool
# from pycrayon import CrayonClient

from .base_model import BaseModel
from .network import define_netD, define_netG, define_netL
from options.cfg_priorbox import v6
from options.cfg_gans import *

class DetectModel(BaseModel):
    '''
    Full image gan
    '''
    def __init__(self, opt):
        super(DetectModel, self).__init__(opt)
        # init attributes
        self.mode = opt.mode
        self.batch_size = opt.batch_size
        self.fine_size = opt.fine_size
        self.cuda = opt.cuda
        self.num_classes = opt.num_classes
        self.cfg = v6
        self.priors = PriorBox(self.cfg)
        self.priorboxes = Variable(self.priors.train(), volatile=True)
        self.cnt = 0
        self.beta = 9

        # init input
        self.source = self.Tensor(opt.batch_size, opt.input_dim, opt.fine_size, opt.fine_size)
        self.resize_source = self.Tensor(opt.batch_size, opt.input_dim, 321, 321)
        self.anno = []
        self.mask = []
        self.target = []

        # init network
        self.netG = define_netG('res101', 3)
        self.netM = define_netG('mask2', 3)
        self.netD = define_netG('res101', 3)

        # self.load_pre_network(opt.pretrained_path)
        self.load_networkD(opt.d_network_path)
        self.load_networkM(opt.m_network_path)
        self.load_networkG(opt.g_network_path)
        

        # init optimizer
        if opt.mode == 'train':
            self.G_solver = torch.optim.SGD(list(self.netG.parameters()) + list(self.netM.parameters()+list(self.netD.parameters())), lr = self.opt.lr)
            # self.D_solver = torch.optim.SGD(self.netD.parameters(), lr = self.opt.lr)
            # self.Loc_solver = torch.optim.SGD(self.netG.parameters(), lr=opt.lr)
        
        # init loss
        self.criterionMultiBoxLoss = MultiBoxLoss(opt.num_classes, opt.overlap_thresh,
                                                                                            opt.bkg_label, self.cfg, opt.cuda)                                                                           
        self.criterionL1 = nn.L1Loss()

        # init Modules
        self.detect = Detect(self.num_classes, 0, 200, opt.nms_thresh)

        # support cuda
        if opt.cuda:
            # self.netD.cuda()
            self.netG.cuda()
            self.netM.cuda()
            self.criterionL1.cuda()
            self.source = self.source.cuda()
            self.resize_source = self.resize_source.cuda()
        
        # support variable
        self.source = Variable(self.source)
        self.resize_source = Variable(self.resize_source)

        log.info("Training Detect Model")
    
    def draft_data(self, source, mask, target, anno, wh):
        self.batch_size = source.size(0)
        self.source.data.resize_(source.size()).copy_(source)

        if self.cuda:
            self.anno = [Variable(element.cuda(), volatile=True) for element in anno]
        else:
            self.anno = [Variable(element, volatile=True) for element in anno]

    def backward_D(self):
        '''In PriorBox type
        '''
        # REAL RPN
        self.feats_g, self.real_conf, self.real_loc = self.netG.forward(self.source)
        self.fake_target = self.netM.forward(self.feats_g)

        # Real
        self.real_loss_l, self.real_loss_c = self.criterionMultiBoxLoss.forward(self.real_loc, self.real_conf, self.priorboxes, self.anno)

        self.cnt += 1

    def backward_G(self):
        # Fake
        _, self.fake_conf, self.fake_loc = self.netD.forward(self.fake_target)
        #loss loc
        self.fake_loss_l, self.fake_loss_c = self.criterionMultiBoxLoss.forward(self.fake_loc, self.fake_conf, self.priorboxes, self.anno)
        self.loss_l1 = self.criterionL1(self.fake_target, self.source)

        # Loss D
        self.loss = self.fake_loss_l + self.fake_loss_c + self.real_loss_l + self.real_loss_c + self.loss_l1
        # self.loss_G = self.loss_G_GAN * 0.1 + (self.fake_loss_l + self.fake_loss_c) * 0.9

        # Backward G
        self.loss_G.backward()

    def test(self, source, mask, target, anno, wh):
        # set input
        self.draft_data(source, mask, target, anno, wh)

        self.feats_g, self.real_conf, self.real_loc = self.netG.forward(self.source)
        self.fake_target = self.netM.forward(self.feats_g)
        self._swap_channels()
 

    def backward_SSD(self):
        self.feats_g, self.fake_conf, self.fake_loc = self.netG.forward(self.source)
        self.fake_loss_l, self.fake_loss_c = self.criterionMultiBoxLoss.forward(self.fake_loc, self.fake_conf, self.priorboxes, self.anno)

        self.loss_G = self.fake_loss_l +  self.fake_loss_c
        self.loss_G.backward()
        self.cnt +=1

    def train(self, source, mask, target, anno, wh):
        # set input
        self.draft_data(source, mask, target, anno, wh)

        # Backward D&L
        self.G_solver.zero_grad()
        self.backward_D() 
        self.backward_G()       
        self.G_solver.step()

        if self.cnt % 10 == 0:
            print(self.loss.data[0])

    def train2(self, source, mask, target, anno, wh):
        self.draft_data(source, mask, target, anno, wh)
        self.Loc_solver.zero_grad()
        self.backward_SSD()
        self.Loc_solver.step()    

    def _swap_channels(self):
        channels = [0, 1, 2]
        fg = self.fake_target.data
        fg[fg < 0.01] = 0
        bg = (1- fg)*self.source.data
        fg = fg*self.source.data
        # bright
        if npr.randint(2):
            alpha = npr.uniform(-0.5, 0.5)
            bg = bg + alpha
            fg = fg + alpha  

        # contrast
        if npr.randint(2):
            alpha = npr.uniform(0.5, 1.5)
            fg = fg * alpha

        # saturation
        if npr.randint(2):
            alpha = npr.uniform(0.5, 1.5)
            fg[:, 0, :, :] = fg[:, 0, :, :] * alpha    
      
        # swap
        if npr.randint(2):
            random.shuffle(channels)
            fg = fg[:, channels, :, :]

        resize_source = (bg + fg).clamp_(min=0, max=1)    
        self.resize_source.data.resize_(resize_source.size()).copy_(resize_source)

    def save_location(self):
        # output = self.detect(self.fake_loc, F.softmax(self.fake_conf.view(-1,  self.num_classes)), self.priorboxes)
        log.info("The detection result in GD: {}".format(self.fake_boxes/self.fine_size))
        log.info("The datasets result in Val: {}".format(self.anno))

    def save_image(self, label, epoch):
        log.info("Saving TRIMG(source/real_target/fake_target) - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        vutils.save_image(self.source.data, '{}/epoch{}_{}_DETECT_source.png'.format(self.save_dir, epoch, label))
        # cat = torch.cat([self.fake_target, self.real_target], dim=2)
        # vutils.save_image(cat.data, '{}/epoch{}_{}_DETECT_G.png'.format(self.save_dir, epoch, label))
        # vutils.save_image(self.real_target.data, '{}/epoch{}_{}_DETECT_real.png'.format(self.save_dir, epoch, label))
        vutils.save_image(self.fake_target.data, '{}/epoch{}_{}_DETECT_fake.png'.format(self.save_dir, epoch, label))
        # vutils.save_image(self.resize_source.data, '{}/epoch{}_{}_DETECT_resize.png'.format(self.save_dir, epoch, label))
    
    def save_network(self, label, epoch='laterest'):
        log.info("Saving netG - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        torch.save(self.netG.state_dict(), '{}/epoch_{}_{}_DETECT_netG_.pth'.format(self.save_dir, epoch, label))
        log.info("Saving netM - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        torch.save(self.netM.state_dict(), '{}/epoch_{}_{}_DETECT_netM.pth'.format(self.save_dir, epoch, label))
        log.info("Saving netD - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        torch.save(self.netD.state_dict(), '{}/epoch_{}_{}_DETECT_netD.pth'.format(self.save_dir, epoch, label))
        
    
    def save(self, label, epoch):
        log.info("*" * 50)
        log.info("Epoch: {}  Iters: {}".format(epoch, self.cnt))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_image(label, epoch)
        self.save_network(label, epoch=epoch)
        # self.save_location()

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

    def load_networkM(self, m_network_path):
        '''load network for netM, H mean hotmap
        '''
        log.info("Loading network weights for netm")
        if self.cuda:
            self.netM.load_state_dict(torch.load(m_network_path))
        else:
            self.netM.load_state_dict(torch.load(m_network_path, map_location=lambda storage, loc: storage))
    
    def load_pre_network(self, pretrained_path):
        '''load network for netG/D, H mean hotmap
        '''
        log.info("Loading pretrained network weights for resnet18 in netG")
        if self.cuda:
            self.netG.resnet.load_state_dict(torch.load(pretrained_path))
        else:
            self.netG.resnet.load_state_dict(torch.load(pretrained_path, map_location=lambda storage, loc: storage))

    def current_errors(self):
        # return {'loss_G': self.loss_G.data[0], 'loss_D': self.loss_D.data[0], 'loss_fake_loc': (self.fake_loss_c+self.fake_loss_l).data[0], 'loss_real_loc': (self.fake_loss_c+self.fake_loss_l).data[0]}
        return {'loss_G': self.loss_G.data[0], 'loss_D': self.loss_D.data[0]}
        
    def current_errors2(self):
        return {'loss_loc': (self.fake_loss_c+self.fake_loss_l).data[0]}    

    def visual(self):
        self.D_loc_exp.add_scalar_value('D_loc_loss', self.loss_D.data[0], step=self.cnt)
        self.G_loc_exp.add_scalar_value('G_loc_loss', self.loss_G.data[0], step=self.cnt)
        self.loc_exp.add_scalar_value('loc_loss', self.fake_loss_l.data[0]+self.fake_loss_c.data[0], step=self.cnt)
    
    def name(self):
        return 'DetectionModel'

    def __str__(self):
        netG_grah =  self.netG.__str__()
        netD_grah = self.netM.__str__()
        return 'netG:\n {}\n netD:\n {}\n'.format(netG_grah, netD_grah)