import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utility.func import np_softmax
from net.lib.box.process import clip_boxes, filter_boxes
from net.lib.nms.gpu_nms.gpu_nms import gpu_nms
from net.lib.box.overlap.cython_overlap.cython_box_overlap import cython_box_overlap
from net.loss import weighted_focal_loss_for_cross_entropy, weighted_smooth_l1


# ------------------------------ NET ------------------------------
class RpnMultiHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(RpnMultiHead, self).__init__()

        self.num_scales = len(cfg.rpn_scales)
        self.num_bases = [len(b) for b in cfg.rpn_base_apsect_ratios]

        self.convs  = nn.ModuleList()
        self.logits = nn.ModuleList()
        self.deltas = nn.ModuleList()

        for l in range(self.num_scales):
            channels = in_channels*2
            self.convs.append(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1))
            self.logits.append(
                nn.Sequential(
                    nn.Conv2d(channels, 2*self.num_bases[l], kernel_size=3, padding=1),
                )
            )
            self.deltas.append(
                nn.Sequential(
                    nn.Conv2d(channels, 4*2*self.num_bases[l], kernel_size=3, padding=1),
                )
            )

    def forward(self, fs):
        """
        :param fs: [p2, p3, p4, p5]
        :return:
            logits_flat:    base: 1 : 1  1 : 2  2 : 1
                        p2[0][0]: [f_submit, b] [f_submit, b] [f_submit, b],
                        p2[0][1]: [f_submit, b] [f_submit, b] [f_submit, b],
                                         ...
                        p2[16][16]:
                           ...
                        p5[:][:]
            shape: (B, N, 2)

            f_submit = foreground prob
            b = background prob

            deltas_flat:    base:   1 : 1    1 : 2    2 : 1
                        p2[0][0]: [df, db] [df, db] [df, db]
                        p2[0][1]: [df, db] [df, db] [df, db]
                                         ...
                        p2[16][16]:
                           ...
                        p5[:][:]
            shape: (B, N, 2, 4)

            df = foreground deltas
            db = background deltas
            in which df, db = [cx, cy, w, h]

            a total of N = (16*16*3 + 32*32*3 + 64*64*3 + 128*128*3) proposals in an input
        """
        batch_size = len(fs[0])

        logits_flat = []
        deltas_flat = []
        for l in range(self.num_scales):  # apply multibox head to feature maps
            f = fs[l]
            f = F.relu(self.convs[l](f))

            f = F.dropout(f, p=0.5, training=self.training)
            logit = self.logits[l](f)
            delta = self.deltas[l](f)

            logit_flat = logit.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
            delta_flat = delta.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2, 4)
            logits_flat.append(logit_flat)
            deltas_flat.append(delta_flat)

        logits_flat = torch.cat(logits_flat, 1)
        deltas_flat = torch.cat(deltas_flat, 1)

        return logits_flat, deltas_flat


# ------------------------------ Creating Anchor Boxes ------------------------------

def _make_bases(base_size, base_apsect_ratios):
    """
    make base anchor boxes for each base size & ratio
    :param:
        base_size:
            anchor base size
            e.g. 16
        base_apsect_ratios:
            height/width ratio
            e.g. [(1, 1), (1, 2), (2, 1)]
    :return:
        list of bases, each base has 4 coordinates, a total of
        len(base_apsect_ratios) bases. e.g.
        [[ -8.,  -8.,   8.,   8.],
         [ -8., -16.,   8.,  16.],
         [-16.,  -8.,  16.,   8.]]
    """
    bases = []
    for ratio in base_apsect_ratios:
        w = ratio[0] * base_size
        h = ratio[1] * base_size
        rw = round(w / 2)
        rh = round(h / 2)
        base = (-rw, -rh, rw, rh)
        bases.append(base)

    bases = np.array(bases, np.float32)
    return bases


def _make_anchor_boxes(f, scale, bases):
    """
    make anchor boxes on every pixel of the feature map
    :param:
        f_submit: feature of size (B, C, H, W)
        scale:
            zoom scale from feature map to image,
            used to define stride on image. e.g. 4
        bases:
            base anchor boxes. e.g.
            [[ -8.,  -8.,   8.,   8.],
             [ -8., -16.,   8.,  16.],
             [-16.,  -8.,  16.,   8.]]
    :return:
        list of anchor boxes on input image
        shape: H * W * len(base_apsect_ratios)
    """
    anchor_boxes = []
    _, _, H, W = f.size()
    for y, x in itertools.product(range(H), range(W)):
        cx = x * scale
        cy = y * scale
        for b in bases:
            x0, y0, x1, y1 = b
            x0 += cx
            y0 += cy
            x1 += cx
            y1 += cy
            anchor_boxes.append([x0, y0, x1, y1])

    anchor_boxes = np.array(anchor_boxes, np.float32)
    return anchor_boxes


def fpn_make_anchor_boxes(fs, cfg):
    """
    :param: fs: (num_scales, B, C, H, W) a batch of features
    create region proposals from all 4 feature maps from FPN
    total of (16*16*3 + 32*32*3 + 64*64*3 + 128*128*3) boxes
    """
    rpn_anchor_boxes = []
    num_scales = len(cfg.rpn_scales)
    for l in range(num_scales):
        bases = _make_bases(cfg.rpn_base_sizes[l], cfg.rpn_base_apsect_ratios[l])
        boxes = _make_anchor_boxes(fs[l], cfg.rpn_scales[l], bases)
        rpn_anchor_boxes.append(boxes)

    rpn_anchor_boxes = np.vstack(rpn_anchor_boxes)
    return rpn_anchor_boxes


# ------------------------------ bbox regression -------------------------------------
# "UnitBox: An Advanced Object Detection Network" - Jiahui Yu, Yuning Jiang, Zhangyang Wang, Zhimin Cao, Thomas Huang
#  https://arxiv.org/abs/1608.01471
def rpn_encode(window, truth_box):
    cx = (window[:,0] + window[:,2])/2
    cy = (window[:,1] + window[:,3])/2
    w  = (window[:,2] - window[:,0]+1)
    h  = (window[:,3] - window[:,1]+1)

    target = (truth_box - np.column_stack([cx,cy,cx,cy]))/np.column_stack([w,h,w,h])
    target = target*np.array([-1,-1,1,1],np.float32)
    return target


def rpn_decode(window, delta):
    cx = (window[:,0] + window[:,2])/2
    cy = (window[:,1] + window[:,3])/2
    w  = (window[:,2] - window[:,0]+1)
    h  = (window[:,3] - window[:,1]+1)

    delta = delta*np.array([-1,-1,1,1],np.float32)
    box   = delta*np.column_stack([w,h,w,h]) + np.column_stack([cx,cy,cx,cy])
    return box


# ------------------------------ Non-Maximum Supression -------------------------------------
def rpn_nms(cfg, mode, images, anchor_boxes, logits_flat, deltas_flat):
    """
    This function:
    1. Do non-maximum suppression on given window and logistic score
    2. filter small rpn_proposals, crop border
    3. bbox regression

    :param cfg: configure
    :param mode: mode. e.g. 'train', 'test', 'eval'
    :param images: a batch of input images
    :param anchor_boxes: all anchor boxes in a batch, list of coords, e.g.
               [[x0, y0, x1, y1], ...], a total of 16*16*3 + 32*32*3 + 64*64*3 + 128*128*3
    :param logits_flat: (B, N, 2) NOT nomalized
               [[0.7, 0.5], ...]
    :param deltas_flat: (B, N, 2, 4)
               [[[t1, t2, t3, t4], [t1, t2, t3, t4]], ...]
    :return: all proposals in a batch. e.g.
        [i, x0, y0, x1, y1, score, label]
        proposals[0]:   image idx in the batch
        proposals[1:5]: bbox
        proposals[5]:   probability of foreground (background skipped)
        proposals[6]:   class label
    """
    if mode in ['train']:
        nms_prob_threshold = cfg.rpn_train_nms_pre_score_threshold
        nms_overlap_threshold = cfg.rpn_train_nms_overlap_threshold
        nms_min_size = cfg.rpn_train_nms_min_size

    elif mode in ['valid', 'test', 'eval']:
        nms_prob_threshold = cfg.rpn_test_nms_pre_score_threshold
        nms_overlap_threshold = cfg.rpn_test_nms_overlap_threshold
        nms_min_size = cfg.rpn_test_nms_min_size

        if mode in ['eval']:
            nms_prob_threshold = 0.05  # set low numbe r to make roc curve.
    else:
        raise ValueError('rpn_nms(): invalid mode = %s?' % mode)

    num_classes = cfg.num_classes
    logits = logits_flat.data.cpu().numpy()
    deltas = deltas_flat.data.cpu().numpy()
    batch_size, _, height, width = images.size()

    # non-max suppression
    rpn_proposals = []
    for b in range(batch_size):
        pic_proposals = [np.empty((0, 7), np.float32)]
        prob = np_softmax(logits[b])
        delta = deltas[b]

        # skip background
        for c in range(1, num_classes):
            index = np.where(prob[:, c] > nms_prob_threshold)[0]
            if len(index) > 0:
                w = anchor_boxes[index]
                p = prob[index, c].reshape(-1, 1)
                d = delta[index, c]
                # bbox regression, do some clip/filter
                box = rpn_decode(w, d)
                box = clip_boxes(box, width, height)  # take care of borders
                keep = filter_boxes(box, min_size=nms_min_size)  # get rid of small boxes

                if len(keep) > 0:
                    box = box[keep]
                    p = p[keep]
                    keep = gpu_nms(np.hstack((box, p)), nms_overlap_threshold)

                    proposal = np.zeros((len(keep), 7), np.float32)
                    proposal[:, 0] = b
                    proposal[:, 1:5] = np.around(box[keep], 0)
                    proposal[:, 5] = p[keep, 0]
                    proposal[:, 6] = c
                    pic_proposals.append(proposal)

        pic_proposals = np.vstack(pic_proposals)
        rpn_proposals.append(pic_proposals)

    rpn_proposals = Variable(torch.from_numpy(np.vstack(rpn_proposals))).cuda()
    return rpn_proposals


# ------------------------------ Labeling Anchor Boxes ------------------------------
def _make_one_rpn_target(cfg, image, anchor_boxes, truth_boxes, truth_labels):
    """
    labeling windows for one image
    :param image: input image
    :param anchor_boxes: list of bboxes e.g. [x0, y0, x1, y1]
    :param truth_boxes: list of boxes, e.g. [x0, y0, x1, y1]
    :param truth_labels: 1 for sure
    :return:
        label: 1 for pos, 0 for neg
        label_assign: which truth box is assigned to the window
        label_weight: pos=1, neg \in (0, 1] by rareness, otherwise 0 (don't care)
        target: bboxes' offsets
        target_weight: same as label_weight
    """
    num_anchor_boxes = len(anchor_boxes)
    label = np.zeros((num_anchor_boxes,), np.float32)
    label_assign = np.zeros((num_anchor_boxes,), np.int32)
    label_weight = np.ones((num_anchor_boxes,), np.float32)  # <todo> why use 1 for init ?
    target = np.zeros((num_anchor_boxes, 4), np.float32)
    target_weight = np.zeros((num_anchor_boxes,), np.float32)

    num_truth_box = len(truth_boxes)
    if num_truth_box != 0:
        _, height, width = image.size()

        overlap = cython_box_overlap(anchor_boxes, truth_boxes)
        argmax_overlap = np.argmax(overlap, 1)
        max_overlap = overlap[np.arange(num_anchor_boxes), argmax_overlap]
        # label 1/0 for each anchor
        bg_index = max_overlap < cfg.rpn_train_bg_thresh_high
        label[bg_index] = 0
        label_weight[bg_index] = 1

        fg_index = max_overlap >= cfg.rpn_train_fg_thresh_low
        label[fg_index] = 1
        label_weight[fg_index] = 1
        label_assign[...] = argmax_overlap

        # for each truth, anchor_boxes with highest overlap, include multiple maxs
        # re-assign less overlapped gt to anchor_boxes
        argmax_overlap = np.argmax(overlap, 0)
        max_overlap = overlap[argmax_overlap, np.arange(num_truth_box)]
        anchor_assignto_gt, gt_assignto_anchor = np.where(overlap == max_overlap)

        fg_index = anchor_assignto_gt
        label[fg_index] = 1
        label_weight[fg_index] = 1
        label_assign[fg_index] = gt_assignto_anchor

        # regression
        fg_index = np.where(label != 0)
        target_window = anchor_boxes[fg_index]
        target_truth_box = truth_boxes[label_assign[fg_index]]
        target[fg_index] = rpn_encode(target_window, target_truth_box)
        target_weight[fg_index] = 1

        # don't care
        invalid_truth_label = np.where(truth_labels < 0)[0]
        invalid_index = np.isin(label_assign, invalid_truth_label) & (label != 0)
        label_weight[invalid_index] = 0
        target_weight[invalid_index] = 0

        # weights for class balancing
        fg_index = np.where((label_weight != 0) & (label != 0))[0]
        bg_index = np.where((label_weight != 0) & (label == 0))[0]

        num_fg = len(fg_index)
        num_bg = len(bg_index)
        label_weight[fg_index] = 1
        label_weight[bg_index] = num_fg / num_bg

        if cfg.rpn_train_scale_balance:
            # weights for scale balancing
            num_scales = len(cfg.rpn_scales)
            num_bases = [len(b) for b in cfg.rpn_base_apsect_ratios]
            start = 0
            for l in range(num_scales):
                h, w = int(height // 2 ** l), int(width // 2 ** l)
                end = start + h * w * num_bases[l]
                label_weight[start:end] *= (2**l)**2
                start = end

        # task balancing
        target_weight[fg_index] = label_weight[fg_index]

    # save
    label = Variable(torch.from_numpy(label)).cuda()
    label_assign = Variable(torch.from_numpy(label_assign)).cuda()
    label_weight = Variable(torch.from_numpy(label_weight)).cuda()
    target = Variable(torch.from_numpy(target)).cuda()
    target_weight = Variable(torch.from_numpy(target_weight)).cuda()
    return label, label_assign, label_weight, target, target_weight


def make_rpn_target(cfg, images, anchor_boxes, truth_boxes_batch, truth_labels_batch):
    """
    labeling anchor boxes for a batch of images
    :param cfg: configuration
    :param images: a batch of input images
    :param anchor_boxes: list of anchor boxes e.g. [x0, y0, x1, y1]
    :param truth_boxes_batch: list of boxes, e.g. [x0, y0, x1, y1]
    :param truth_labels_batch: 1 for sure
    :return:
        label: 1 for pos, 0 for neg
        label_assign: which truth box is assigned to the window
        label_weight: pos=1, neg \in (0, 1] by rareness, otherwise 0 (don't care)
        target: bboxes' offsets
        target_weight: same as label_weight
    """
    anchor_labels = []
    anchor_label_assigns = []
    anchor_label_weights = []
    anchor_targets = []
    anchor_targets_weights = []

    batch_size = len(truth_boxes_batch)
    for b in range(batch_size):
        image = images[b]
        truth_boxes = truth_boxes_batch[b]
        truth_labels = truth_labels_batch[b]

        label, label_assign, label_weight, target, targets_weight = \
            _make_one_rpn_target(cfg, image, anchor_boxes, truth_boxes, truth_labels)

        anchor_labels.append(label.view(1, -1))
        anchor_label_assigns.append(label_assign.view(1, -1))
        anchor_label_weights.append(label_weight.view(1, -1))
        anchor_targets.append(target.view(1, -1, 4))
        anchor_targets_weights.append(targets_weight.view(1, -1))

    anchor_labels = torch.cat(anchor_labels, 0)
    anchor_label_assigns = torch.cat(anchor_label_assigns, 0)
    anchor_label_weights = torch.cat(anchor_label_weights, 0)
    anchor_targets = torch.cat(anchor_targets, 0)
    anchor_targets_weights = torch.cat(anchor_targets_weights, 0)

    return anchor_labels, anchor_label_assigns, anchor_label_weights, anchor_targets, anchor_targets_weights


# ----------------------------------------- Loss -------------------------------------------
def rpn_cls_loss(logits, labels, label_weights):
    """
    :param logits: (B, N, 2),    unnormalized foreground/background score
    :param labels: (B, N)        {0, 1} for bg/fg
    :param label_weights: (B, N) float \in (0,1] for rareness, otherwise 0 (don't care)
    :return: float
    """
    batch_size, num_anchors, num_classes = logits.size()
    labels = labels.long()
    batch_num_anchors = batch_size * num_anchors

    # classification ---
    logits = logits.view(batch_num_anchors, num_classes)
    labels = labels.view(batch_num_anchors, 1)
    label_weights = label_weights.view(batch_num_anchors, 1)

    return weighted_focal_loss_for_cross_entropy(logits, labels, label_weights)


def rpn_reg_loss(labels, deltas, target_deltas, target_weights, delta_sigma=3.0):
    """
    :param labels: (B, N) {0, 1} for bg/fg. used to catch positive samples
    :param deltas: (B, N, 2, 4) bbox regression
    :param target_deltas: (B, N, 4) target deltas from make_rpn_target
    :param target_weights: (B, N) float \in (0,1] for rareness, otherwise 0 (don't care)
    :param delta_sigma: float
    :return: float
    """
    batch_size, num_anchors, num_classes, num_deltas = deltas.size()
    assert num_deltas == 4
    labels = labels.long()
    batch_num_anchors = batch_size * num_anchors

    # one-hot encode
    labels = labels.view(batch_num_anchors, 1)
    deltas = deltas.view(batch_num_anchors, num_classes, 4)
    target_deltas = target_deltas.view(batch_num_anchors, 4)
    target_weights = target_weights.view(batch_num_anchors, 1)
    # calc positive samples only
    index = (labels != 0).nonzero()[:, 0]
    deltas = deltas[index]
    target_deltas = target_deltas[index]
    target_weights = target_weights[index].expand((-1, 4)).contiguous()

    select = labels[index].view(-1, 1).expand((-1, 4)).contiguous().view(-1, 1, 4)
    deltas = deltas.gather(1, select).view(-1, 4)

    return weighted_smooth_l1(deltas, target_deltas, target_weights, delta_sigma)