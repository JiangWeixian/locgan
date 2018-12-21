import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
from ..util.bbox import decode, nms_preN, match, sample_rois, sample_rois_mask, sample_rois_mask_nonp
from options.cfg_priorbox import v6 as cfg

class RPN(Function):
    def __init__(self, num_classes, bkg_label, top_k, pre_N, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.pre_N = pre_N
        self.threshold = 0.5
        self.fg_nums = 32
        self.rois_nums = 128
        self.nms_thresh = nms_thresh
        self.conf_thresh = 0
        if self.nms_thresh <= 0:
            raise ValueError('nms_thresh must be negative')
        self.variance = cfg['variance']
        self.img_size = cfg['min_dim']
    
    def forward(self, loc_data, conf_data, prior_data):
        num = loc_data.size(0)
        num_priors = prior_data.size(0)
        all_boxes = []
        if num == 1:
            # size batch x num_classes x num_priors
            conf_data = conf_data.t().contiguous().unsqueeze(0)
        elif num > 1:
            conf_data = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)
        
        for i in range(num):
            decode_boxes = decode(loc_data[i], prior_data, self.variance)
            # Filter all prior-box, get the overlap score
            conf_scores = conf_data[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decode_boxes)
                boxes = decode_boxes[l_mask].view(-1, 4).clamp(max=1, min=0)
                ids, count = nms_preN(boxes, scores, top_k=self.top_k, pre_N=self.pre_N)
                all_boxes.append(boxes[ids[:count]])
            # idx of highest scoring and non-overlapping boxes per class
            self.output = torch.cat(all_boxes, 0)
        return self.output

    def mask_match(self, annos, masks, targets, use_gpu=False):
        # Match output with gt_boxes(anno)
        num = self.output.size(0)

        for idx in range(num):
            anno = annos[idx][:, :-1].data
            mask = masks[idx].data
            target = targets[idx].data
            labels = annos[idx][:, -1].data
            defaults = torch.cat((self.output, anno), 0)
            # rois, masks_target, targets_target, labels, bbox_targets= sample_rois_mask(self.threshold, anno, mask, target, defaults, labels, idx, fg_rois_per_image=self.fg_nums, rois_per_image=self.rois_nums, use_gpu=use_gpu)
            rois, masks_target, targets_target, labels= sample_rois_mask_nonp(self.threshold, anno, mask, target, defaults, labels, idx, fg_rois_per_image=self.fg_nums, use_gpu=use_gpu)
            
        # Support cuda
        if use_gpu:
            rois = rois.cuda()
            labels = labels.cuda()
            masks_target = masks_target.cuda()
            targets_target = targets_target.cuda()
            bbox_targets = bbox_targets.cuda()
        # Support variable
        rois = rois*self.img_size
        rois = Variable(torch.cat((torch.zeros(rois.size(0), 1), rois), 1), requires_grad = False)
        labels = Variable(labels, requires_grad=False)
        masks_target = Variable(masks_target, requires_grad=False)
        targets_target = Variable(targets_target)
        return rois, masks_target, targets_target, labels