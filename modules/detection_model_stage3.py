import os
import glob

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
    '''GanS+ObjDetection

    @Attributes:
    - is_load: flag for already load netD/G weights
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

        # init input
        self.source = self.Tensor(opt.batch_size, opt.input_dim, opt.fine_size, opt.fine_size)
        self.tmp = self.Tensor(opt.batch_size, opt.input_dim, opt.fine_size, opt.fine_size)
        self.resize_source = self.Tensor(opt.batch_size, opt.input_dim, opt.fine_size, opt.fine_size)
        self.anno = []
        self.mask = []
        self.target = []

        # init network
        self.netG = define_netG('res101', 3)
        self.netM = define_netG('mask', 512)
        self.netD = define_netD('fm', grah_netD['fm'], 3)
        self.netM2 = define_netG('mask2', 512)
        self.netD2 = define_netD('fm2', grah_netD['fm'], 3)

        # self.load_pre_network(opt.pretrained_path)
        # self.load_networkD(opt.d_network_path)
        # self.load_networkM(opt.m_network_path)
        # self.load_networkG(opt.g_network_path)
        

        # init optimizer
        if opt.mode == 'train':
            self.Loc_solver = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
            self.G_solver = torch.optim.Adam(list(self.netM.parameters())+list(self.netM2.parameters()), lr = self.opt.lr, betas=(self.opt.beta_gans, 0.999))
            self.D_solver = torch.optim.Adam(self.netD.parameters(), lr = self.opt.lr, betas=(self.opt.beta_gans, 0.999))
            self.D_solver2 = torch.optim.Adam(self.netD2.parameters(), lr = self.opt.lr, betas=(self.opt.beta_gans, 0.999))
        
        # init loss
        self.criterionMultiBoxLoss = MultiBoxLoss(opt.num_classes, opt.overlap_thresh,
                                                                                            opt.bkg_label, self.cfg, opt.cuda)
        self.criterionGANConf =  GANConfLoss(opt.num_classes, opt.bkg_label, opt.cuda)                                                                                   
        self.criterionGAN = GANLoss(tensor = self.Tensor, cuda=opt.cuda)
        self.criterionL1 = nn.L1Loss()

        # init Modules
        self.detect = Detect(self.num_classes, 0, 200, opt.nms_thresh)
        self.rpn = RPN(self.num_classes, 0, 200, 10, opt.nms_thresh)

        # support cuda
        if opt.cuda:
            self.netD.cuda()
            self.netG.cuda()
            self.netM.cuda()
            self.netD2.cuda()
            self.netM2.cuda()
            self.criterionL1.cuda()
            self.source = self.source.cuda()
            self.tmp = self.tmp.cuda()
            self.resize_source = self.resize_source.cuda()
        
        # support variable
        self.source = Variable(self.source)
        self.resize_source = Variable(self.resize_source)
        self.tmp = Variable(self.tmp, requires_grad=False)

        log.info("Training Detect Model")
    
    def draft_data(self, source, mask, target, anno, wh):
        self.batch_size = source.size(0)
        self.source.data.resize_(source.size()).copy_(source)

        if self.cuda:
            self.mask = [Variable(element.cuda(), volatile=True) for element in mask]
        else:
            self.mask = [Variable(element, volatile=True) for element in target]

        if self.cuda:
            self.target = [Variable(element.cuda(), volatile=True) for element in target]
        else:
            self.target = [Variable(element, volatile=True) for element in target]

        if self.cuda:
            self.anno = [Variable(element.cuda(), volatile=True) for element in anno]
        else:
            self.anno = [Variable(element, volatile=True) for element in anno]

    def backward_D(self):
        '''In PriorBox type
        '''
        # Fake RPN
        self.feats_g, self.fake_conf, self.fake_loc = self.netG.forward(self.source)
        self.fake_boxes = self.rpn(self.fake_loc, F.softmax(self.fake_conf.view(-1, self.num_classes)), self.priorboxes)
        self.rpn_boxes, self.rpn_mask, self.rpn_target, self.rpn_label = self.rpn.mask_match(self.anno, self.mask, self.target, self.cuda)

        # Get FakeMask
        pooled_features = layer_upsample(self.feats_g, self.rpn_boxes, 1/8.0, 8)
        self.fake_target =  self.netM.forward(pooled_features)

        # Get FakeLoss
        self.pred_fake, _, _ = self.netD.forward(self.fake_target)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        # self.real_target = (self.rpn_mask*self.rpn_target)[:self.num, :, :, :]
        self.pred_real, self.feats_real, self.conf_real = self.netD.forward(self.rpn_target)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_conf = self.criterionGANConf(self.conf_real, self.rpn_label)

        # Loss D
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 + self.loss_D_conf
        # Backward D
        self.loss_D.backward(retain_graph=True)

        self.cnt += 1

    def backward_G(self):
        # Fake
        pred_fake, feats_fake, _ = self.netD.forward(self.fake_target)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # loss fm
        self.loss_fm = criterion_fmloss(self.feats_real, feats_fake, cuda=self.cuda)
        # loss l1
        self.loss_l1 = self.criterionL1(self.fake_target, self.rpn_target)

        # Loss D
        self.loss_G = self.loss_G_GAN +  (self.loss_fm + self.loss_l1) * 10
        # self.loss_G = self.loss_G_GAN * 0.1 + (self.fake_loss_l + self.fake_loss_c) * 0.9

        # Backward G
        self.loss_G.backward()

    def train(self, source, mask, target, anno, wh):
        # set input
        self.draft_data(source, mask, target, anno, wh)

        # Backward D&L
        self.D_solver.zero_grad()
        self.backward_D()        
        self.D_solver.step()

        # Backward G
        self.G_solver.zero_grad()
        self.backward_G()
        self.G_solver.step()

        self._mask_pick(source, self.fake_target, self.rpn_boxes)

            
    def backward_D2(self):
        '''In PriorBox type
        '''
        # Fake RPN
        self.feats_g, self.fake_conf, self.fake_loc = self.netG.forward(self.resize_source)

        # Get FakeMask
        self.fake_target =  self.netM2.forward(self.feats_g)
        # self.fake_target = (self.fake_target*self.rpn_target)[:self.num, :, :, :]

        # Get FakeLoss
        self.pred_fake, _ = self.netD2.forward(self.fake_target)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)
        self.fake_loss_l, self.fake_loss_c = self.criterionMultiBoxLoss.forward(self.fake_loc, self.fake_conf, self.priorboxes, self.anno)

        # Real
        # self.real_target = (self.rpn_mask*self.rpn_target)[:self.num, :, :, :]
        self.pred_real, self.feats_real = self.netD2.forward(self.source)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Loss D
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_loc = self.fake_loss_l + self.fake_loss_c
        # Backward D
        self.loss_D.backward(retain_graph=True)

        self.cnt += 1

    def backward_G2(self):
        # Fake
        pred_fake, feats_fake = self.netD2.forward(self.fake_target)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # loss fm
        self.loss_fm = criterion_fmloss(self.feats_real, feats_fake, cuda=self.cuda)
        # loss l1
        self.loss_l1 = self.criterionL1(self.fake_target, self.source)

        # Loss D
        self.loss_G = self.loss_G_GAN +  (self.loss_fm + self.loss_l1) * 10
        # self.loss_G = self.loss_G_GAN * 0.1 + (self.fake_loss_l + self.fake_loss_c) * 0.9

        # Backward G
        self.loss_G.backward()

    def train2(self):

        # Backward D&L
        self.D2_solver2.zero_grad()
        self.backward_D2()        
        self.D2_solver2.step()

        # Backward G
        self.G_solver.zero_grad()
        self.backward_G2()
        self.G_solver.step()

        self.Loc_solver.zero_grad()
        self.loss_loc.backward()
        self.Loc_solver.step()        

    def _validate_netD(self):
        conf_pred = F.softmax(self.conf_real.view(-1, num_classes))
        best_pred, best_pred_idx = conf_pred.max(1, keepdim=True)

        gt_label = torch.LongTensor(self.rpn_label.size())

        # Support Cuda
        if self.cuda:
            gt_label = gt_label.cuda()
        gt_label = Variable(gt_label, requires_grad=False)    
        gt_label.data.copy_(self.rpn_label.data)
        predicted = best_pred_idx.eq(gt_label).data.cpu().sum()

        print('netD-Class-Acc: {:.3f}'.format(predicted/float(self.num)))

    def _mask_pick(self, source, masks, bboxes): 
        for i in range(masks.size(0)):
            mask = masks[i, :, :, :].unsqueeze(0)
            xmin, ymin, xmax, ymax = bboxes[i, 1:].clamp(min=0, max=self.fine_size).int().data
            h = max(ymax - ymin, 1)
            w = max(xmax - xmin, 1)
            resize_mask = mask.data.resize_(1, 3, int(h), int(w))
            source[:, :, int(ymin):int(ymin+h), int(xmin):int(xmin+w)]= resize_mask.data
        self.resize_source.data.resize_(source.size()).copy_(source)    
        vutils.save_image(self.resize_source.data, 'source.png')
            

    def _mask_crop(self, input, label):
        # print(input)
        label = label.gt(0)
        label = label.unsqueeze(label.dim()).unsqueeze(label.dim()).unsqueeze(label.dim()).expand_as(input).type_as(input)
        return input*label

    def save_location(self):
        # output = self.detect(self.fake_loc, F.softmax(self.fake_conf.view(-1,  self.num_classes)), self.priorboxes)
        log.info("The detection result in GD: {}".format(self.fake_boxes/self.fine_size))
        log.info("The datasets result in Val: {}".format(self.anno))

    def save_image(self, label, epoch):
        log.info("Saving TRIMG(source/real_target/fake_target) - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        vutils.save_image(self.source.data, '{}/epoch{}_{}_DETECT_source.png'.format(self.save_dir, epoch, label))
        # cat = torch.cat([self.fake_target, self.real_target], dim=2)
        # vutils.save_image(cat.data, '{}/epoch{}_{}_DETECT_G.png'.format(self.save_dir, epoch, label))
        vutils.save_image(self.real_target.data, '{}/epoch{}_{}_DETECT_real.png'.format(self.save_dir, epoch, label))
        vutils.save_image(self.fake_target.data, '{}/epoch{}_{}_DETECT_fake.png'.format(self.save_dir, epoch, label))
    
    def save_network(self, label, epoch='laterest'):
        log.info("Saving netG - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        torch.save(self.netG.state_dict(), '{}/epoch_{}_{}_DETECT_netG_.pth'.format(self.save_dir, epoch, label))
        log.info("Saving netM - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        torch.save(self.netM.state_dict(), '{}/epoch_{}_{}_DETECT_netM.pth'.format(self.save_dir, epoch, label))
        log.info("Saving netD - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        torch.save(self.netD.state_dict(), '{}/epoch_{}_{}_DETECT_netD.pth'.format(self.save_dir, epoch, label))
        log.info("Saving netM - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        torch.save(self.netM2.state_dict(), '{}/epoch_{}_{}_DETECT_netM2.pth'.format(self.save_dir, epoch, label))
        log.info("Saving netD - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        torch.save(self.netD2.state_dict(), '{}/epoch_{}_{}_DETECT_netD2.pth'.format(self.save_dir, epoch, label))
        
    
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
        return {'loss_G': self.loss_G.data[0], 'loss_D': self.loss_D.data[0], 'loss_loc': (self.fake_loss_c+self.fake_loss_l).data[0]}

    def visual(self):
        self.D_loc_exp.add_scalar_value('D_loc_loss', self.loss_D.data[0], step=self.cnt)
        self.G_loc_exp.add_scalar_value('G_loc_loss', self.loss_G.data[0], step=self.cnt)
        self.loc_exp.add_scalar_value('loc_loss', self.fake_loss_l.data[0]+self.fake_loss_c.data[0], step=self.cnt)
    
    def name(self):
        return 'DetectionModel'

    def __str__(self):
        netG_grah = self.netG.__str__() + self.netM.__str__()
        netD_grah = self.netD.__str__()
        return 'netG:\n {}\n netD:\n {}\n'.format(netG_grah, netD_grah)