import torch
import torch.nn as nn
from torch.autograd import Variable
from ..util import bbox

class GANConfLoss(nn.Module):
    def __init__(self, num_classes, bkg_label, use_gpu=True):
        super(GANConfLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.background_label = bkg_label
        
        # init loss
        self.criterionLossC = nn.CrossEntropyLoss(size_average=False)
        if use_gpu:
            self.criterionLossC.cuda()
    def forward(self, conf_pred, conf_gt):
            conf_p = conf_pred.view(-1, self.num_classes)
            targets_weighted = torch.LongTensor(conf_gt.size())

        # Support Cuda
            if self.use_gpu:
                targets_weighted = targets_weighted.cuda()
            targets_weighted = Variable(targets_weighted, requires_grad=False)    
            targets_weighted.data.copy_(conf_gt.data)
            loss_c = self.criterionLossC(conf_p, targets_weighted)
            return loss_c / conf_pred.size(0)

class GANLoss(nn.Module):
    '''Wrap for BCEloss(now), the netD output can not be [batch_size, 1]
    ...So, this class is for more complex dim output like [batch_size, x, x]

    @Attributes:
    - real_label/fake_label: true/false
    - real_label_var/fake_label_var: true label/fake label, variable type 
    - loss: defalut(now) is BCELOSS
    '''
    def __init__(self, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, type='BCE', cuda=False):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.cuda = cuda
        self.Tensor = tensor
        if self.cuda:
            self.loss = nn.BCELoss().cuda()
        else:
            self.loss = nn.BCELoss()
            

    def get_target_tensor(self, input, target_is_real):
        '''if target_is_real, compute true data's loss, else compute fake data's loss
        '''
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                if self.cuda:
                    real_tensor = real_tensor.cuda()
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                if self.cuda:
                    fake_tensor = fake_tensor.cuda()
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, bkg_label, cfg, use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.negpos_ratio = 3
        print(cfg)
        self.image_size = cfg['min_dim']
        self.feature_maps = cfg['feature_maps']
        self.variance = cfg['variance']
        self.aspect_ratios = cfg['aspect_ratios']
        self.eachlayersbox_length = [f*f*(2+len(ar)) for f, ar in zip(cfg['feature_maps'], cfg['aspect_ratios'])]
        
        # init loss
        self.criterionLossL = nn.SmoothL1Loss(size_average=False)
        self.criterionLossC = nn.CrossEntropyLoss(size_average=False)

        if use_gpu:
            self.criterionLossL.cuda()
            self.criterionLossC.cuda()
    
    def forward(self, loc_pred, conf_pred, prior_boxes, annos):
        # Init data
        loc_data = loc_pred
        conf_data = conf_pred
        num = loc_data.size(0)
        default_prior_boxes = prior_boxes[:loc_data.size(1), :]
        num_priors = default_prior_boxes.size(0)
        num_classes = self.num_classes
        
        loc_t = torch.Tensor(num, num_priors, 4)
        # After filtered by mask_labels, matched by bbox.match
        conf_t = torch.LongTensor(num, num_priors)

        # Support Cuda
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        
        for idx in range(num):
            # Get batch data
            anno = annos[idx][:, :-1].data
            labels = annos[idx][:, -1].data
            defaults = default_prior_boxes.data
            # Match anno for each priorbox, and set label for each one. 
            bbox.match(self.threshold, anno, defaults, self.variance, labels,
                                       loc_t, conf_t, idx)

        # Support variable
        loc_t = Variable(loc_t, requires_grad = False)
        conf_t = Variable(conf_t, requires_grad=False)

        # Get the priorbox(obj one)
        pos = conf_t > 0
        num_pos = pos.data.sum()
        
        # Filter loc_t by pos
        # Tips: If filter, pos must has same size as loc_data, just expand!
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)

        # LOC Prediction
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = self.criterionLossL(loc_p, loc_t)

        # CONF Prediction
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = bbox.log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = self.criterionLossC(conf_p, targets_weighted)

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c

class SimpleMultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, bkg_label, 
                                 mask_overlap_thresh, cfg, use_gpu=True):
        super(SimpleMultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.mask_overlap = mask_overlap_thresh
        self.background_label = bkg_label
        self.negpos_ratio = 3
        self.image_size = cfg['min_dim']
        self.feature_maps = cfg['feature_maps']
        self.variance = cfg['variance']
        self.aspect_ratios = cfg['aspect_ratios']
        self.eachlayersbox_length = [f*f*(2+len(ar)) for f, ar in zip(cfg['feature_maps'], cfg['aspect_ratios'])]
        
        # init loss
        self.criterionLossL = nn.SmoothL1Loss(size_average=False)
        self.criterionLossC = nn.CrossEntropyLoss(size_average=False)

        if use_gpu:
            self.criterionLossL.cuda()
            self.criterionLossC.cuda()
    
    def forward(self, loc_pred, conf_pred, roi_data):
        # Init data
        loc_data = loc_pred
        conf_data = conf_pred
        rois = roi_data[0].data
        labels = roi_data[1].data
        num_priors = loc_pred.size(0)
        num_classes = self.num_classes
        

        loc_t = torch.Tensor(num_priors, 4)
        # After filtered by mask_labels, matched by bbox.match
        conf_t = torch.LongTensor(num_priors)

        loc_t = rois
        conf_t = labels.view(-1, 1)

        # Support Cuda
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        # Support variable
        loc_t = Variable(loc_t, requires_grad = False)
        conf_t = Variable(conf_t, requires_grad=False)

        # Get the priorbox(obj one)
        pos = conf_t > 0
        num_pos = pos.data.sum()
        
        # Filter loc_t by pos
        # Tips: If filter, pos must has same size as loc_data, just expand!
        pos_idx = pos.expand_as(loc_data)

        # LOC Prediction
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = self.criterionLossL(loc_p, loc_t)

        # CONF Prediction
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = bbox.log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        pos = pos.view(1, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(1, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos = pos.view(-1, 1)
        neg = neg.view(-1, 1)
        pos_idx = pos.expand_as(conf_data)
        neg_idx = neg.expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = self.criterionLossC(conf_p, targets_weighted)

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c

def criterion_fmloss(real_feats, fake_feats, criterion='HingeEmbeddingLoss', cuda=False):
    '''Compute distance bwtween real_feats and fake_feats, instead of l1loss

    - Params:
    @real_feats: real img's features, **not the last output of netD, and is hidden-layers's output**
    @fake_feats: same as upone, but just from fake imgs
    @criterion: criterion type, defalyt is `HingeEmbeddingLoss`
    '''
    if criterion == 'HingeEmbeddingLoss':
        criterion = nn.HingeEmbeddingLoss()
    losses = 0
    for real_feat, fake_feat in zip(real_feats, fake_feats):
        l2 = (real_feat.mean(0) - fake_feat.mean(0)) * (real_feat.mean(0) - fake_feat.mean(0))
        loss = criterion(l2, Variable(torch.ones(l2.size())).cuda())
        losses += loss
    return losses

