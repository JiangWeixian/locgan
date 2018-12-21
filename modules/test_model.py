import os
import glob

import glog as log
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.autograd import Variable

from .util.bbox import box_copy, mask_crop
from .funcs.prior_box import PriorBox
from .funcs.detection import Detect, Annotate
from .funcs.rpn import RPN
from .funcs.layer import layer_upsample
from roi_pooling.modules.roi_pool import RoIPool
from pycrayon import CrayonClient
import matplotlib.pyplot as plt
from options.cfg_priorbox import v4, v5

from .base_model import BaseModel
from .network import define_netD, define_netG, define_netL, define_netC, define_netGDetection

class TestModel(BaseModel):
    '''GanS+ObjDetection

    @Attributes:
    - is_load: flag for already load netD/G weights
    '''
    def __init__(self, opt):
        super(TestModel, self).__init__(opt)
        # init attributes
        self.mode = opt.mode
        self.detect_threshold = opt.detect_threshold
        self.fine_size = opt.fine_size
        self.cuda = opt.cuda
        self.detection_path = os.path.join(self.save_dir, opt.detection_path)
        self.eval_path = os.path.join(self.save_dir, opt.eval_path)
        self.priors = PriorBox(v5)
        self.priorboxes = Variable(self.priors.train(), volatile=True)

        # init input
        self.source = self.Tensor(1, opt.input_dim, opt.fine_size, opt.fine_size)

        # init network
        self.netG = define_netG(opt.which_model_netG, opt.networkG_config, opt.input_dim)
        self.netM = define_netG('mask', opt.networkG_config, 256)

        self.load_networkG(opt.g_network_path)
        self.load_networkM(opt.m_network_path)

        # init Modules
        self.detect = Detect(2, 0, 200, opt.nms_thresh)

        # support cuda
        if opt.cuda:
            self.netG.cuda()
            self.netD.cuda()
            self.netM.cuda()
            self.source = self.source.cuda()
        
        # support variable
        self.source = Variable(self.source)

        log.info("Test Detection Model")
    
    def draft_data(self, source, target):
        self.batch_size = source.size(0)
        self.source.data.resize_(source.size()).copy_(source)
        self.target = target.squeeze(0)

    def test(self, source, target):
        # set input
        self.draft_data(source, target)

        self.feats_g, self.fake_conf, self.fake_loc = self.netG.forward(self.source)
        self.output = self.detect(self.fake_loc, F.softmax(self.fake_conf.view(-1,2)), self.priorboxes)
    
    def display(self, img, filename):
        '''Visual anno on source image

        @Params:
        - img: (PIL.Image) source image
        '''
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # Store bboxes which score big than threshold
        coords = []
        # Store all bboxes
        results = []
        monitor = Annotate(img)
        detections = self.output.data
        for i in range(1, detections.size(1)):
            # For each boxes
            j = 0
            while detections[0, i, j, 0] > 0:
                score = detections[0, i, j, 0]
                if self.cuda:
                    pt = (detections[0, i, j, 1:]*self.fine_size).clamp(min=0, max=self.fine_size-1).cpu().numpy()
                else:
                    pt = (detections[0, i, j, 1:]*self.fine_size).clamp(min=0, max=self.fine_size-1).numpy()
                coord = (pt[0], pt[1], pt[2], pt[3], str(score))
                if score >= self.detect_threshold:
                    coords.append(coord)
                results.append(coord)
                j += 1
            for ele in coords:
                with open(self.detection_path, mode='a') as f:
                    f.write('{} '.format(filename) + ' '.join(str(c) for c in ele)+'\n')
            for ele in results:
                with open(self.eval_path, mode='a') as f:
                    f.write('{} '.format(filename) + ' '.join(str(c) for c in ele)+'\n')
            monitor.draw_rects(coords)
            save_path = os.path.join(self.save_dir, filename)
            plt.savefig(save_path)
            if len(coords):
                self._upsample(coords)

    def _upsample(self, coords):
        scale = 1.0/4
        i_size = self.feats_g.size(-1)
        o_size = 14
        for i, coord in enumerate(coords):
            xmin, ymin, xmax, ymax = coord[:-1]
            xmin, ymin, xmax, ymax = xmin*scale, ymin*scale, xmax*scale, ymax*scale
            h = max(ymax - ymin, 1)
            w = max(xmax - xmin, 1)
            gt_foi = self.feats_g[:, :, int(ymin):int(ymin+h), int(xmin):int(xmin+w)]
            foi = F.upsample(gt_foi, size=(int(o_size), int(o_size)), mode='bilinear')
            if i == 0:
                fois = foi
            else:
                fois = torch.cat([fois, foi], dim=0)
        self.fake_target = self.netM.forward(fois)*255     

    def save_mask_image(self, filename):
        mask_save_dir = os.path.join(self.save_dir, 'mask')
        real_save_dir = os.path.join(self.save_dir, 'real')
        if not os.path.exists(mask_save_dir):
            os.makedirs(mask_save_dir)
        if not os.path.exists(real_save_dir):
            os.makedirs(real_save_dir)
        vutils.save_image(self.fake_target.data, '{}/{}'.format(mask_save_dir, filename))
        vutils.save_image(self.target, '{}/{}'.format(real_save_dir, filename))

    def load_networkL(self, l_network_path):
        log.info("Loading network weights for netL")
        if self.cuda:
            self.netL.load_state_dict(torch.load(l_network_path))
        else:
            self.netL.load_state_dict(torch.load(l_network_path, map_location=lambda storage, loc: storage))

    def load_networkG(self, g_network_path):
        '''load network for netG/D, H mean hotmap
        '''
        log.info("Loading network weights for netG/D")
        if self.cuda:
            self.netG.load_state_dict(torch.load(g_network_path))
        else:
            self.netG.load_state_dict(torch.load(g_network_path, map_location=lambda storage, loc: storage))

    def load_networkM(self, m_network_path):
        '''load network for netG/D, H mean hotmap
        '''
        log.info("Loading network weights for netM")
        if self.cuda:
            self.netM.load_state_dict(torch.load(m_network_path))
        else:
            self.netM.load_state_dict(torch.load(m_network_path, map_location=lambda storage, loc: storage))
    
    def load_networkD(self, d_network_path):
        '''load network for netG/D, H mean hotmap
        '''
        log.info("Loading network weights for netG/D")
        if self.cuda:
            self.netD.load_state_dict(torch.load(d_network_path))
        else:
            self.netD.load_state_dict(torch.load(d_network_path, map_location=lambda storage, loc: storage))
    
    def name(self):
        return 'TestModel'

    def __str__(self):
        netG_grah = self.netG.__str__()
        netM_grah = self.netM.__str__()
        return 'netG:\n {}\n netM:\n {}\n '.format(netG_grah, netM_grah)