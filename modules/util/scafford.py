import numpy as np
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def parsetInt(str):
    '''Get the int number in string

    @Params:
    - str: the origin str

    @Returns:
    the int in str
    '''
    return int(''.join([x for x in str if x.isdigit()]))

def set_npz(dicts, path):
    '''convert dicts to `.npz` file, the dicts should be like
    ... {'one': data, 'two': data}, because some np.array can not saved into `.json`

    @Params:
    - dicts: the dicts should be like {'one': data, 'two': data}
    - path: `.npz` file path, such as `/to/xx.npz`
    '''
    np.savez(path, **dicts)

def verification(detpath, npzpath, overthresh=0.5, fine_size=288, use_07_metric=True):
    npos = 0
    class_recs = np.load(npzpath)
    for key, val in class_recs.items():
        R = val[()]
        BB = R['bbox']
        npos += BB.shape[0]
    with open(detpath, 'r') as f:
        lines = f.readlines()
    if any(lines) == 1:
        splitlines = [x.strip().split(' ') for x in lines]
        image_ids = [x[0].split('/')[-1] for x in splitlines]
        confidence = np.array([float(x[-1]) for x in splitlines])
        BB = np.array([[float(z) for z in x[1:-1]] for x in splitlines])

        # sort by confidence
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)
        img_list = []
        for d in range(nd):
            R = class_recs[image_ids[d]][()]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)*fine_size
            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin, 0.)
                ih = np.maximum(iymax - iymin, 0.)
                inters = iw * ih
                uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                       (BBGT[:, 2] - BBGT[:, 0]) *
                       (BBGT[:, 3] - BBGT[:, 1]) - inters)
                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

                if ovmax > overthresh:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
                else:
                    fp[d] = 1.

        # compute precision recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, use_07_metric)
    else:
        rec = -1.
        prec = -1.
        ap = -1.
    return rec, prec, ap

def voc_ap(rec, prec, use_07_metric=True):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

# coco-bus dataset pixel iou
# (real)val-bus datasets in `/media/eric/Elements/coco/val_mask2014/bus`
# (fake) genrate from netG
def pixel_iou(ft, rt):
    '''Compute pixel iou between [rt, ft], and ft and rt are (target) 1dim's imgs 

    @Params:
    - rt: (tensor) target images from datasets
    - ft: (tensor) target images from net
    '''
    pos_ft = ft.gt(0.01).type()
    pos_rt = rt > 0.01
    area_ft = pos_ft.sum()
    area_rt = pos_rt.sum()
    pos_inter =  pos_ft[pos_rt]
    inter = pos_inter.sum()
    union = area_ft + area_rt - inter
    overlap = inter/float(union)
    return overlap

def pixel_acc(ft, rt):
    '''Compute pixel accuracy between [rt, ft], and ft and rt are (target) 1dim's imgs 

    @Params:
    - rt: (tensor) target images from datasets
    - ft: (tensor) target images from net
    '''
    pos_ft = ft > 0.01
    pos_rt = rt > 0.01
    neg_ft = ft <= 0.01
    neg_rt = rt <= 0.01
    area = ft.size()[1] * ft.size()[2]
    pos_inter = pos_ft[pos_rt]
    neg_inter = neg_ft[neg_rt]
    inter = pos_inter.sum() + neg_inter.sum()
    acc = inter/float(area)
    return acc