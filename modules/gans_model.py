import os

import glog as log
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable

from loss.loss import GANLoss, criterion_fmloss
from pycrayon import CrayonClient

from .base_model import BaseModel
from .network import define_netD, define_netG


class GansModel(BaseModel):
    def __init__(self, opt):
        super(GansModel, self).__init__(opt)
        # init attributes
        self.mode = opt.mode
        self.batch_size = opt.batch_size
        self.fine_size = opt.fine_size
        self.cnt = 0

        # init input
        self.source = self.Tensor(opt.batch_size, opt.input_dim, opt.fine_size, opt.fine_size)
        self.real_target = self.Tensor(opt.batch_size, opt.input_dim, opt.fine_size, opt.fine_size)

        # init network
        self.netG = define_netG(opt.which_model_netG, opt.networkG_config, opt.input_dim)
        self.netD = define_netD(opt.which_model_netD, opt.networkD_config, opt.output_dim)

        # init optim
        if self.mode == 'train':
            self.G_solver = torch.optim.Adam(self.netG.parameters(), lr = opt.lr, betas=(opt.beta_gans, 0.999))
            self.D_solver = torch.optim.Adam(self.netD.parameters(), lr = opt.lr, betas=(opt.beta_gans, 0.999))
        elif self.mode == 'continue':
            self.load_network(opt.g_network_path, opt.d_network_path)
            self.G_solver = torch.optim.Adam(self.netG.parameters(), lr = opt.lr, betas=(opt.beta_gans, 0.999))
            self.D_solver = torch.optim.Adam(self.netD.parameters(), lr = opt.lr, betas=(opt.beta_gans, 0.999))
        elif self.mode == 'test':
            self.load_network(opt.g_network_path, opt.d_network_path)

        # init loss function/obj
        self.criterionL1 = nn.L1Loss()
        self.criterionGAN = GANLoss(tensor = self.Tensor, cuda=opt.cuda)

        # init tensorboard
        if opt.cc:
            self.cc = CrayonClient(hostname="localhost")
            self.cc.remove_all_experiments()
            try:
                self.D_exp = self.cc.open_experiment('D_loss')
            except:
                self.D_exp = self.cc.create_experiment('D_loss')
            try:
                self.G_exp = self.cc.open_experiment('G_loss')
            except:
                self.G_exp = self.cc.create_experiment('G_loss')

        # support cuda
        if opt.cuda:
            self.netG.cuda()
            self.netD.cuda()
            self.criterionL1.cuda()
            self.source = self.source.cuda()
            self.real_target = self.real_target.cuda()
        
        # variable input
        self.source = Variable(self.source)
        self.real_target = Variable(self.real_target)

        log.info("Training Gans Model!")

    def draft_data(self, source, target):
        self.batch_size = source.size(0)
        self.source.data.resize_(source.size()).copy_(source)
        self.real_target.data.resize_(target.size()).copy_(target)
    
    def backward_D(self):
        # Fake
        self.fake_target = self.netG.forward(self.source)
        print (self.fake_target.size())
        self.pred_fake, _ = self.netD.forward(self.fake_target)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        self.pred_real, self.feats_real = self.netD.forward(self.real_target)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)
        
        # Loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward(retain_variables=True)

        self.cnt += 1

    def backward_G(self):
        # Fake
        pred_fake, feats_fake = self.netD.forward(self.fake_target)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # FM loss
        self.loss_fm = criterion_fmloss(self.feats_real, feats_fake, cuda=self.opt.cuda)

        # Loss
        self.loss_G = self.loss_G_GAN + self.loss_fm
        self.loss_G.backward(retain_variables=True)
    
    def train(self, source, target):
        # Set input for network
        self.draft_data(source, target)

        # Backward D
        self.D_solver.zero_grad()
        self.backward_D()
        self.D_solver.step()

        # Backward G
        self.G_solver.zero_grad()
        self.backward_G()
        self.G_solver.step()
        self.visual()

    def test(self, source, target, epoch, label='TEST'):
        self.draft_data(source, target)
        self.fake_target = self.netG.forward(self.source)
        self.save_image(label, epoch)
    
    def save_image(self, label, epoch):
        log.info("Saving TRIMG(source/real_target/fake_target) - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        vutils.save_image(self.source.data, '{}/{}_GAN_source_epoch{}.png'.format(self.save_dir, label, epoch))
        vutils.save_image(self.real_target.data, '{}/{}_GAN_real_epoch{}.png'.format(self.save_dir, label, epoch))
        vutils.save_image(self.fake_target.data, '{}/{}_GAN_fake_epoch{}.png'.format(self.save_dir, label, epoch))

    def save_network(self, label, epoch):
        log.info("Saving netG - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        torch.save(self.netG.state_dict(), '{}/{}_GAN_netG_epoch{}.pth'.format(self.save_dir, label, epoch))
        log.info("Saving netD - [epochs: {}  cnt: {}] in {}".format(epoch, self.cnt, self.save_dir))
        torch.save(self.netD.state_dict(), '{}/{}_GAN_netD_epoch{}.pth'.format(self.save_dir, label, epoch))

    def save(self, label, epoch):
        log.info("*" * 50)
        log.info("Epoch: {}  Iters: {}".format(epoch, self.cnt))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.save_image(label, epoch)
        self.save_network(label, epoch)
    
    def load_network(self, g_network_path, d_network_path):
        log.info("Loading network weights for netG/D")
        self.netG.load_state_dict(torch.load(g_network_path))
        self.netD.load_state_dict(torch.load(d_network_path))

    def current_errors(self):
        return [('loss_G_GAN', self.loss_G_GAN.data[0]), 
                        ('loss_fm', self.loss_fm.data[0]),
                        ('loss_D_fake', self.loss_D_fake.data[0]),
                        ('loss_D_real', self.loss_D_real.data[0])
        ]

    def visual(self):
        self.D_exp.add_scalar_value('D_loss', self.loss_D.data[0], step=self.cnt)
        self.G_exp.add_scalar_value('G_loss', self.loss_G.data[0], step=self.cnt)
        
    def name(self):
        return 'GansDetection'
        
    def __str__(self):
        netG_grah = self.netG.__str__()
        netD_grah = self.netD.__str__()
        return 'netG:\n {}\n netD:\n {}'.format(netG_grah, netD_grah)