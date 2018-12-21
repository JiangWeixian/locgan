import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
from ..util.bbox import decode, nms, mask_filter, mask_detect, bbox_transform_inv, validate
from options.cfg_priorbox import v6 as cfg

import matplotlib.pyplot as plt
import numpy
from matplotlib.patches import Rectangle

class Detect(Function):
    def __init__(self, num_classes, bkg_label, top_k, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.conf_thresh = 0.01
        if self.nms_thresh <= 0:
            raise ValueError('nms_thresh must be negative')
        self.variance = cfg['variance']
        self.img_size = cfg['min_dim']
        self.boxes = []
    
    def forward(self, loc_data, conf_data, prior_data):
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)
        self.output = torch.zeros(num, self.num_classes, self.top_k, 5)
        if num == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.t().contiguous().unsqueeze(0)
        else:
            conf_preds = conf_data.view(num, num_priors,
                                        self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                self.output[i, cl, :count] = \
                    torch.cat((scores[ids[:count]].unsqueeze(1),
                               boxes[ids[:count]]), 1)
        return self.output

    def validate(self, annos, threshold):
        anno = annos[0][:, :-1].data
        recs = 0
        precs = 0
        for cl in range(1, self.num_classes):
            rec, prec = validate(threshold, anno, self.boxes[cl-1]) 
            recs = recs + rec
            precs = precs + prec
        
        return recs / float(self.num_classes-1), precs/float(self.num_classes-1)
    

class Annotate(object):
    def __init__(self, frame):
        self.img = frame.resize((300, 300))
        self.rect = Rectangle((0,0), 1, 1, edgecolor='red', fill=False)
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.rect_coor = []
        self.press = False
        self.fig = plt.figure()
        self.fig.set_size_inches(1, 1, forward=False)
        self.ax = plt.Axes(self.fig, [0., 0., 1., 1.])
        self.ax.set_axis_off()
        self.ax.add_patch(self.rect)
        self.fig.add_axes(self.ax)
        self.ax.imshow(self.img)
        
    def draw_rects(self, coor_arrs):
        '''draw rect, and rect's arr from coor_arrs
        '''
        for index, coor in enumerate(coor_arrs):
            rect = Rectangle((0,0), 1, 1, edgecolor='red', fill=False)
            self.ax.add_patch(rect)
            rect.set_width(coor[2] - coor[0])
            rect.set_height(coor[3] - coor[1])
            rect.set_xy((coor[0], coor[1]))
        self.ax.figure.canvas.draw()

def detect(img, target, detections, scale, filename, img_filename, show=True, use_gpu=False, labelmap=['bus']):
    '''Store and convert the detections from netL

    @Params:
    - detections: output from netL
    - scale: origin image size
    - filename: store the result in this file
    - use_gpu: netL forward in GPU type
    - labelmap: label name list
    '''
    annotate = Annotate(img)
    # For each class
    for i in range(1, detections.size(1)):
        # For each boxes
        results = mask_detect(detections[0, i, :, :], target, use_gpu=use_gpu, min_dim=scale)
        for coord in results:
            with open(filename, mode='a') as f:
                f.write('{} '.format(filename) + ' '.join(str(c) for c in coord)+'\n')
        annotate.draw_rects(results)
        if show:
            plt.show()
        else:
            if img_filename:
                plt.savefig(img_filename)