import os
import glob

import glog as log
import torch
import torch.nn as nn
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable

from loss.loss import GANLoss, criterion_fmloss, MultiBoxLoss
from funcs.prior_box import PriorBox
from funcs.detection import Detect
from pycrayon import CrayonClient

from .base_model import BaseModel
from .network import define_netD, define_netC, define_netL

class LocModel(BaseModel):
    def __init__(self, opt):
        super(LocModel, self).__init__(opt)
        # init attributes
        self.mode = opt.mode
        self.batch_size = opt.batch_size
        self.fine_size = opt.fine_size
        self.cuda = opt.cuda
        self.priors = PriorBox(opt.cfg_priors)
        self.priorboxes = Variable(self.priors.train(), volatile=True)
        self.cnt = 0

        # init input
        self.source = self.Tensor(opt.batch_size, opt.input_dim, opt.fine_size, opt.fine_size)
        self.anno = []

        # init network
        self.netL = define_netL(opt.which_model_netL, opt.networkL_config)
        self.netD = define_netD(opt.which_model_netD, opt.networkD_config, opt.input_dim)
        self.netC = define_netC(opt.which_model_netC, opt.networkC_config)

        # init optimizer
        if opt.mode == 'train':
            self.L_solver = torch.optim.Adam(list(self.netL.parameters())+list(self.netD.parameters())+list(self.netC.parameters()), lr = self.opt.lr, betas=(self.opt.beta_gans, 0.999))
        elif opt.mode == 'continue':
            self.load_networkL(opt.l_network_path)
            self.load_networkD(opt.d_network_path)
            self.load_networkC(opt.c_network_path)
            self.L_solver = torch.optim.Adam(list(self.netL.parameters())+list(self.netD.parameters())+list(self.netC.parameters()), lr = self.opt.lr, betas=(self.opt.beta_gans, 0.999))

        
        # init loss
        self.criterionMultiBoxLoss = MultiBoxLoss(opt.num_classes, opt.overlap_thresh,
                                                                                               opt.bkg_label, self.priorboxes, 
                                                                                               opt.mask_overlap_thresh, opt.cfg_priors, opt.cuda)

        # init Detection
        self.detect = Detect(2, 0, 200, opt.overlap_thresh)

        # init tesorboard
        if opt.cc:
            self.cc = CrayonClient(hostname="localhost")
            self.cc.remove_all_experiments()
            try:
                self.L_exp = self.cc.open_experiment('L_loss')
            except Exception, e:
                self.L_exp = self.cc.create_experiment('L_loss')

        # support cuda
        if opt.cuda:
            self.netL.cuda()
            self.netD.cuda()
            self.netC.cuda()
            self.source = self.source.cuda()
        
        # support variable
        self.source = Variable(self.source)

        log.info("Training Loc Model")
    
    def draft_data(self, source, target, anno, wh):
        self.batch_size = source.size(0)
        self.source.data.resize_(source.size()).copy_(source)
        if self.cuda:
            self.anno = [Variable(element.cuda(), volatile=True) for element in anno]
        else:
            self.anno = [Variable(element, volatile=True) for element in anno]

    def backward_L(self):
        # Real
        self.pred_real, self.feats_real = self.netD.forward(self.source)
        # Real-loc
        self.real_loc = self.netL(self.feats_real)
        self.real_conf = self.netC(self.feats_real)
        self.real_loss_l, self.real_loss_c = self.criterionMultiBoxLoss.forward(self.real_loc, self.real_conf, self.source, self.anno)
        # Loss L
        self.loss_L = self.real_loss_l + self.real_loss_c

        # Backward D
        self.loss_L.backward(retain_graph=True)

        self.cnt += 1

    def train(self, source, target, anno, wh):
        # set input
        self.draft_data(source, target, anno, wh)

        # Backward D
        self.L_solver.zero_grad()
        self.backward_L()
        self.L_solver.step()

        # Visual
        self.visual()

    def save_location(self):
        output = self.detect(self.real_loc, F.softmax(self.real_conf.view(-1, 2)), self.priorboxes)
        log.info("Val the NetL")
        log.info("The detection result in Val: {}".format(output[0][1]))
        log.info("The datasets result in Val: {}".format(self.anno))
    
    def save_network(self, label, epoch):
        log.info("Saving netD - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        torch.save(self.netD.state_dict(), '{}/epoch{}_{}_DETECT_netD.pth'.format(self.save_dir, epoch, label))
        log.info("Saving netL - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        torch.save(self.netL.state_dict(), '{}/epoch{}_{}_DETECT_netL.pth'.format(self.save_dir, epoch, label))
        log.info("Saving netC - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        torch.save(self.netC.state_dict(), '{}/epoch{}_{}_DETECT_netC.pth'.format(self.save_dir, epoch, label))
    
    def save(self, label, epoch):
        log.info("*" * 50)
        log.info("Epoch: {}  Iters: {}".format(epoch, self.cnt))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_network(label, epoch)
        self.save_location()

    def load_networkL(self, l_network_path):
        log.info("Loading network weights for netL")
        self.netL.load_state_dict(torch.load(l_network_path))

    def load_networkC(self, c_network_path):
        log.info("Loading network weights for netC")
        self.netC.load_state_dict(torch.load(c_network_path))

    def load_networkD(self, d_network_path):
        '''load network for netG/D, H mean hotmap
        '''
        log.info("Loading network weights for netG/D")
        self.netD.load_state_dict(torch.load(d_network_path))

    def current_errors(self):
        return [ ('loss_L', self.loss_L.data[0])]

    def visual(self):
        self.L_exp.add_scalar_value('L_loss', self.loss_L.data[0], step=self.cnt)
    
    def name(self):
        return 'LocModel'

    def __str__(self):
        netD_grah = self.netD.__str__()
        netL_grah = self.netL.__str__()
        netC_grah = self.netC.__str__()
        return 'netD:\n {}\n netL:\n {}\n netC:\n {}'.format(netD_grah, netL_grah, netC_grah)