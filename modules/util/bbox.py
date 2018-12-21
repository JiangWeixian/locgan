import torch
import numpy as np
import numpy.random as npr
import torch.nn.functional as F


#############
# About Image
#############
def mask_filter(boxes, target, min_dim, mask_overlap=0, debug=False, use_gpu=False):
    '''Filter prior box by mask, as if the boxes already scale into image_size
    ...Steps: 
        1) Count the num of nonzero elements on masks
        2) Count overlap = the nums / the area size of prior box
        3) Compare overlap with mask_overlap, overlap > mask_overlap, set label = 1

    @Params:
    - boxes(torch.tensor): default_origin_prior_boxes = the prior box after scale
    - target(torch.tensor): mask image
    - min_dim(int): image size
    - mask_overlap(float): the threshold of overlap

    @Returns:
    (torch.tensor): the labels of boxes, label = 1 mean objects, label = 0 mean background
    '''
    num_priors = boxes.size(0)
    boxes_labels = torch.zeros([num_priors, 2])
    for index in range(num_priors):
        element = boxes[index, :].clamp_(max=min_dim, min=0)
        xmin = int(element[0])
        ymin = int(element[1])
        xmax = int(element[2])
        ymax = int(element[3])
        cut_size = (xmax - xmin) * (ymax - ymin)
        # Forbidden using [0, 0, 0, 0] to slice target
        if not (ymax - ymin) or not (xmax - xmin) or (ymax - ymin) < 0 or (xmax - xmin) < 0:
            mask_size = 0
            overlap = 0
            label = 0
        else:
            cut_target = target.squeeze()[ymin:ymax, xmin:xmax]
            pos = cut_target > 0.001
            N = pos.sum()
            # If all zero elements in cut
            if N == 0:
                mask_size = 0
                overlap = 0
                label = 0
            else:
                overlap = N / (cut_size * 1.0)
                overlap = 0 if overlap >= 1 else overlap
                mask_size = N
                if mask_overlap:
                    if overlap > mask_overlap:
                        label = 1
                    else:
                        label = 0

        if mask_overlap:
            boxes_labels[index, 0] = label
            boxes_labels[index, 1] = label
        else:
            boxes_labels[index, 0] = mask_size
            boxes_labels[index, 1] = overlap
    return boxes_labels

def mask_detect(detections, target, use_gpu=False, min_dim=288, threshold=0.6):
    '''Detect boxes by mask target
    ...If scores(one in detectoins) gt threshold, do not apply mask detect
    ...If all scores in detections small than threshold, apply mask detect

    @Params:
    - detections:(tensor) [scores, xmin, ymin, xmax, ymax]
    - target:(var) result from netG
    - use_gpu: GPU mode
    - min_dim: the fine size of img
    - threshold: the switch threshold for applying mask detect

    @Return:
    annos:(list) [(xmin, ymin, xmax, ymax)]
    '''
    j = 0
    annos = []
    pred = 0
    while detections[j, 0] > 0.1:
        score = detections[j, 0]
        pt = (detections[j, 1:]*min_dim).clamp_(max=min_dim, min=0)
        if use_gpu:
            pt = pt.cpu().numpy()
        else:
            pt = pt.numpy()
        coords = [pt[0], pt[1], pt[2], pt[3], score]
        if score >= threshold:
            annos.append(coords)
            pred = 0.2
        # Apply Mask detect
        if score > pred and score <= threshold:
            xmin, ymin, xmax, ymax = (int(x) for x in coords[:-1])
            if (ymax - ymin) and (xmax - xmin) and (ymax - ymin) > 0 and (xmax - xmin) > 0:
                cut_size = (xmax - xmin) * (ymax - ymin)
                cut_target = target.squeeze()[ymin:ymax, xmin:xmax]
                pos = cut_target > 0.001
                N = pos.sum()
                overlap = N / (cut_size * 1.0)
                coords[-1] = overlap
                if overlap > threshold:
                    annos.append(coords)
        j += 1
    return annos

def eval_result(detections, target, use_gpu=False, min_dim=288, threshold=0.6):
    '''Detect boxes by mask target
    ...If scores(one in detectoins) gt threshold, do not apply mask detect
    ...If all scores in detections small than threshold, apply mask detect

    @Params:
    - detections:(tensor) [scores, xmin, ymin, xmax, ymax]
    - target:(var) result from netG
    - use_gpu: GPU mode
    - min_dim: the fine size of img
    - threshold: the switch threshold for applying mask detect

    @Return:
    annos:(list) [(xmin, ymin, xmax, ymax)]
    '''
    j = 0
    annos = []
    pred = 0
    while detections[j, 0] > 0.1:
        score = detections[j, 0]
        pt = (detections[j, 1:]*min_dim).clamp_(max=min_dim, min=0)
        if use_gpu:
            pt = pt.cpu().numpy()
        else:
            pt = pt.numpy()
        coords = [pt[0], pt[1], pt[2], pt[3], score]
        if score >= threshold:
            annos.append(coords)
        # Apply Mask detect
        elif target.dim():
            xmin, ymin, xmax, ymax = (int(x) for x in coords[:-1])
            if (ymax - ymin) and (xmax - xmin) and (ymax - ymin) > 0 and (xmax - xmin) > 0:
                cut_size = (xmax - xmin) * (ymax - ymin)
                cut_target = target.squeeze()[ymin:ymax, xmin:xmax]
                pos = cut_target > 0.001
                N = pos.sum()
                overlap = N / (cut_size * 1.0)
                coords[-1] = overlap
                annos.append(coords)
        elif not target.dim():
            annos.append(coords)
        j += 1
    return annos

def box_crop(feats, boxes, count, min_dim):
    '''Crop feats by boxes([xmin, ymin, xmax, ymax] data type)

    @Params:
    - feats:(Var) features from network layers 
    - boxes:(tensor) each element is [xmin, ymin, xmax, ymax] data type
    - count: (int) the num of boxes
    - min_dim:(int) the size of feats
    '''
    crop_featsdata = []
    if count > 0:
        for index in range(count):
            element = torch.round(boxes[index, :]).clamp_(max=min_dim, min=0)
            xmin = int(element[0])
            ymin = int(element[1])
            xmax = int(element[2])
            ymax = int(element[3])
            if not (ymax - ymin) or not (xmax - xmin) or (ymax - ymin) < 0 or (xmax - xmin) < 0:
                feat = torch.autograd.Variable(torch.zeros([]))
            else:
                feat = feats[:, :, ymin:ymax, xmin:xmax]
            crop_featsdata.append(feat)
    else:
        crop_featsdata.append(feats)
    return crop_featsdata
        

def box_copy(masks, rpnboxes, img_size):
    '''Combine masks to target, the location of masks come from boxes

    @Params:
    - masks:(list) copy data, any size
    - rpnboxes:(tensor) [batch, num_classes, num_boxes, 5], first value mean overlap, and the remian mean boxes location
    - img_size:(int) the size of target

    @Returns:
    targets:(tensor) cat target into targets as a batch
    '''
    # Batch
    num = len(masks)
    targets = torch.zeros([1, 1, img_size, img_size])
    targets = torch.autograd.Variable(targets)
    for i in range(num):
        # Mask nums for each Batch
        target = torch.zeros([1, 1, img_size, img_size])
        mask = masks[i]
        boxes = rpnboxes[i]
        num_masks = len(mask)
        # Combine each crop mask
        for j in range(num_masks):
            element = mask[j]
            if element.dim() == 0:
                continue
            else:
                new_boxes = torch.zeros([4])
                h, w = element.size()[2:]
                new_boxes[:2] = torch.floor(boxes)[1, j, 1:3]
                new_boxes[2] =  w
                new_boxes[3] = h
                # [xmin, ymin, xmax, ymax]
                new_boxes[2:] += new_boxes[:2]
                new_boxes = new_boxes.clamp_(max=img_size, min=0)
                # use new [w, h] to crop element
                wh = new_boxes[2:] - new_boxes[:2]
                # [padding_left, padding_top, padding_right, padding_bottom]
                new_boxes[2:] = img_size - new_boxes[2:]
                try:
                    if j == 0:
                        target = F.pad(element[:, :, 0:int(wh[1]), 0:int(wh[0])], (int(new_boxes[0]), int(new_boxes[2]), int(new_boxes[1]), int(new_boxes[3])), mode='constant', value=0)
                    else:
                        target += F.pad(element[:, :, 0:int(wh[1]), 0:int(wh[0])], (int(new_boxes[0]), int(new_boxes[2]), int(new_boxes[1]), int(new_boxes[3])), mode='constant', value=0)
                except:
                    print(new_boxes)
        if i == 0:
            targets = target
        else:
            targets = torch.cat([targets, target], dim=0)
    return targets

def box_stitch(boxes):
    '''Stitch all boxes in boxes to a big box

    @Params:
    - boxes:(Tensor) boxes list

    @Returns:
    (Tensor) a new big box
    '''
    box = torch.zeros([1, 4])
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    box[:, 0] = torch.min(x1)
    box[:, 1]  = torch.min(y1)
    box[:, 2]  = torch.max(x2)
    box[:, 3]  = torch.max(y2)
    return box

def box_mask(boxes, count, img_size):
    mask = torch.zeros([1, 1, img_size, img_size])
    if count > 0:
        for index in range(count):
            element = torch.round(boxes[index, :]).clamp_(max=img_size, min=0)
            xmin = int(element[0])
            ymin = int(element[1])
            xmax = int(element[2])
            ymax = int(element[3])
            if not (ymax - ymin) or not (xmax - xmin) or (ymax - ymin) < 0 or (xmax - xmin) < 0:
                pass
            else:
                mask[:, :, ymin:ymax, xmin:xmax] = torch.ones([ymax-ymin, xmax-xmin])
    mask = torch.autograd.Variable(mask)
    return mask

def mask_crop(img, mask):
    '''Crop img by mask

    @Params:
    - img: (variable) the source image
    - mask: (variable) the binary mask image
    '''
    img_crop = mask * img
    return img_crop

def divide_target(target):
    '''Divide 0 and 1 in target, make target([1, x, x]) become target([2, x, x])
    ...First dim mean bg mask, second means obj mask

    @Params:
    - target: (tensor) mask image from datasets 

    @Returns:
    oneshot(tensor): new target shape-like [2, x, x]
    '''
    n = 2
    bg = (1 - target).clamp(min=0, max=1)
    oneshot = torch.cat([bg, target], dim=1)
    return oneshot


#############
# About BBox
#############
# Below fork from ssd.pytorch
def point_form(boxes):
    '''Fork from ssd.pytorch
    ...Convert [cx, cy, w, h] into [xmin, ymin, xmax, ymax]

    @Params:
    - boxes: defalut_origin_prior_boxes

    @Returns:
    boxes in [xmin, ymin, xmax, ymax] point form
    '''
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax

def intersect(box_a, box_b):
    """ Fork from ssd.pytorch 
    ...We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    
    @Params:
    - box_a: (tensor) bounding boxes, Shape: [A,4].
    - box_b: (tensor) bounding boxes, Shape: [B,4].

    @Return:
    - (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Fork from ssd.pytorch 
    ...Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    
    @Params:
    - box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
    - box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    
    @Return:
    - jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.

    @Params:
    - loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
    - priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    @Return:
    decoded bounding box predictions
    """
    # print(variances[0])
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes. 
    make [xmin, ymin, xmax, ymax] into [cx, cy, w, h] with variance

    @Params:
    - matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
    - priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
    - variances: (list[float]) Variances of priorboxes
    @Return:
    encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

def validate(threshold, truths, priors):
    '''Compute acc or other features
    '''
    # jaccard index
    overlaps = jaccard(
        truths,
        priors
    )
    ntruths = truths.size(0)
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    all_pos = best_truth_overlap > threshold
    # tp is correct
    tp_pos = best_prior_overlap > threshold 
    ntp = tp_pos.sum() if tp_pos.dim() else 0
    nfp = (all_pos.sum() - ntp) if all_pos.dim() else 1
    prec = ntp / float(nfp + ntp)
    rec = ntp / float(ntruths)
    return rec, prec

def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    '''Fork from ssd.pytorch box_util.py
    Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    @Params:
    - threshold: (float) The overlap threshold used when mathing boxes.
    - truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
    - priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
    - variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
    - labels: (tensor) All the class labels for the image, Shape: [num_obj].
    - mask_labels: (tensor) labels for all prior box filtered by mask images
    - loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
    - conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
    - mask_conf_t: mask_labels
    - match_conf_t: same as origin ssd.protorch box_utils.match's conf_t 
    - idx: (int) current batch index
    '''
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx] + 1
    conf[best_truth_overlap < threshold] = 0
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc
    conf_t[idx] = conf

def sample_rois(threshold, truths, priors, labels, idx, fg_rois_per_image, rois_per_image, use_gpu=False):
    '''Fork from ssd.pytorch box_util.py
    Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    @Params:
    - threshold: (float) The overlap threshold used when mathing boxes.
    - truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
    - priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
    - variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
    - labels: (tensor) All the class labels for the image, Shape: [num_obj].
    - mask_labels: (tensor) labels for all prior box filtered by mask images
    - loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
    - conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
    - mask_conf_t: mask_labels
    - match_conf_t: same as origin ssd.protorch box_utils.match's conf_t 
    - idx: (int) current batch index
    '''
    # jaccard index
    overlaps = jaccard(
        truths,
        priors
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    conf = labels[best_truth_idx] + 1
    conf[best_truth_overlap < threshold] = 0
    # Apply numpy choice
    if not use_gpu:
        best_truth_overlap = best_truth_overlap.numpy()
        conf = conf.numpy()
        rois = priors.numpy()
        gt_rois = matches.numpy()
    else:
        best_truth_overlap = best_truth_overlap.cpu().numpy()
        conf = conf.cpu().numpy()
        rois = priors.cpu().numpy()
        gt_rois = matches.cpu().numpy()
    fg_inds = np.where(best_truth_overlap >= 0.5)[0]
    fg_rois_per_this_image = int(min(fg_rois_per_image, fg_inds.size))
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image)
    bg_inds = np.where(best_truth_overlap < 0.5)[0]
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image)
    keep_inds = np.append(fg_inds, bg_inds)
    conf = conf[keep_inds]
    rois = rois[keep_inds]
    gt_rois = gt_rois[keep_inds]
    bbox_target_data = _compute_targets(rois, gt_rois)
    return torch.from_numpy(rois), torch.from_numpy(conf).type(torch.LongTensor), torch.from_numpy(bbox_target_data)

def sample_rois_mask_nonp(threshold, truths, bmasks, tmasks, priors, labels, idx, fg_rois_per_image, use_gpu=False):
    # jaccard index
    overlaps = jaccard(
        truths,
        priors
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    keep_inds = best_truth_idx[best_truth_overlap > threshold]
    num = int(min(keep_inds.dim(), fg_rois_per_image))
    keep_inds = keep_inds[torch.randperm(keep_inds.dim()).cuda()][:num]
    if keep_inds.dim() == 1:
        keep_inds = torch.cat([keep_inds, keep_inds], 0)
    conf = labels[keep_inds] + 1
    rois = truths[keep_inds]
    rpn_bmasks = bmasks[keep_inds]
    rpn_tmasks = tmasks[keep_inds]
    return rois, rpn_bmasks, rpn_tmasks, conf

def sample_rois_mask(threshold, truths, masks, targets, priors, labels, idx, fg_rois_per_image, rois_per_image, use_gpu=False):
    '''Fork from ssd.pytorch box_util.py
    Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    @Params:
    - threshold: (float) The overlap threshold used when mathing boxes.
    - truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
    - priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
    - variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
    - labels: (tensor) All the class labels for the image, Shape: [num_obj].
    - mask_labels: (tensor) labels for all prior box filtered by mask images
    - loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
    - conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
    - mask_conf_t: mask_labels
    - match_conf_t: same as origin ssd.protorch box_utils.match's conf_t 
    - idx: (int) current batch index
    '''
    # jaccard index
    overlaps = jaccard(
        truths,
        priors
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]
    mask_matches = masks[best_truth_idx]
    target_matches = targets[best_truth_idx]
    conf = labels[best_truth_idx] + 1
    conf[best_truth_overlap < threshold] = 0
    # Apply numpy choice
    if not use_gpu:
        best_truth_overlap = best_truth_overlap.numpy()
        conf = conf.numpy()
        rois = priors.numpy()
        gt_rois = matches.numpy()
        gt_masks = mask_matches.numpy()
        gt_targets = target_matches.numpy()
    else:
        best_truth_overlap = best_truth_overlap.cpu().numpy()
        conf = conf.cpu().numpy()
        rois = priors.cpu().numpy()
        gt_rois = matches.cpu().numpy()
        gt_masks = mask_matches.cpu().numpy()
        gt_targets = target_matches.cpu().numpy()

    fg_inds = np.where(best_truth_overlap >= 0.5)[0]
    fg_rois_per_this_image = fg_rois_per_image#int(min(fg_rois_per_image, fg_inds.size))
    if fg_inds.size > 0:
        fg_inds = npr.choice(fg_inds, size=fg_rois_per_this_image)
    bg_inds = np.where(best_truth_overlap < 0.5)[0]
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = min(bg_rois_per_this_image, bg_inds.size)
    if bg_inds.size > 0:
        bg_inds = npr.choice(bg_inds, size=bg_rois_per_this_image)
    keep_inds = np.append(fg_inds, bg_inds)
    conf = conf[keep_inds]
    rois = rois[keep_inds]
    gt_rois = gt_rois[keep_inds]
    gt_masks = gt_masks[keep_inds]
    gt_targets = gt_targets[keep_inds]
    bbox_target_data = _compute_targets(rois, gt_rois)
    return torch.from_numpy(rois), torch.from_numpy(gt_masks), torch.from_numpy(gt_targets), torch.from_numpy(conf).type(torch.LongTensor), torch.from_numpy(bbox_target_data)

def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    return targets

def bbox_transform(ex_rois, gt_rois):
    """
    computes the distance from ground-truth boxes to the given boxes, normed by their size
    :param ex_rois: n * 4 numpy array, given boxes
    :param gt_rois: n * 4 numpy array, ground-truth boxes
    :return: deltas: n * 4 numpy array, ground-truth boxes
    """
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    # assert np.min(ex_widths) > 0.1 and np.min(ex_heights) > 0.1, \
    #     'Invalid boxes found: {} {}'. \
    #         format(ex_rois[np.argmin(ex_widths), :], ex_rois[np.argmin(ex_heights), :])

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets

def bbox_transform_inv(boxes, deltas):
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    pred_boxes = torch.zeros(deltas.size())
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    @Params:
    - boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
    - scores: (tensor) The class predscores for the img, Shape:[num_priors].
    - overlap: (float) The overlap thresh for suppressing unnecessary boxes.
    - top_k: (int) The Maximum number of box preds to consider.
    
    @Return:
    The indices of the kept boxes with respect to num_priors.
    """
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

def nms_preN(boxes, scores, overlap=0.7, top_k=2000, pre_N = 200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    @Params:
    - boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
    - scores: (tensor) The class predscores for the img, Shape:[num_priors].
    - overlap: (float) The overlap thresh for suppressing unnecessary boxes.
    - top_k: (int) The Maximum number of box preds to consider.
    
    @Return:
    The indices of the kept boxes with respect to num_priors.
    """
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    if count > pre_N:
        count = pre_N
    return keep, count

if __name__ == '__main__':
    # [xmin, ymin, xmax, ymax]
    box_a = torch.Tensor([[0, 0, 4, 4], [4, 0, 8, 4], [0, 4, 4, 8], [4, 4, 8, 8]])
    box_b = torch.Tensor([[2, 2, 4, 4], [2, 2, 6, 6], [6, 2, 8, 8], [6, 6, 8, 8]])
    # print (jaccard(box_a, box_b))
    # [cx, cy, w, h]
    box_c = torch.Tensor([[3, 3, 2, 2], [4, 4, 4, 4], [7, 5, 2, 6], [7, 7, 2, 2]])
    match(0.5, box_a, box_c, 1, 1, 1, 1, 1)

    
