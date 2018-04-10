import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
# faster-rcnn box encode/decode, nms, softmax
from net.lib.box.process import is_small_box, clip_boxes, filter_boxes, bbox_encode, bbox_decode
from net.lib.box.overlap.cython_overlap.cython_box_overlap import cython_box_overlap
from net.lib.nms.gpu_nms.gpu_nms import gpu_nms
from utility.func import np_softmax


# ------------------------------ NET ------------------------------
class RcnnHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(RcnnHead, self).__init__()
        self.num_classes = cfg.num_classes
        self.crop_size   = cfg.rcnn_crop_size

        self.fc1 = nn.Linear(in_channels*self.crop_size*self.crop_size, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.logit = nn.Linear(1024, self.num_classes)
        self.delta = nn.Linear(1024, self.num_classes*4)

    def forward(self, crops):
        """
        :param crops: cropped feature maps
            shape = (B, C, crop_size, crop_size)
        :return:
            logits: (B, num_classes)
            delta: (B, num_classes*4)
        """
        # flatten each cropped feature map into C*crop_size*crop_size
        x = crops.view(crops.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.dropout(x, 0.5, training=self.training)
        logits = self.logit(x)
        deltas = self.delta(x)

        return logits, deltas


# ------------------------------ Non-max supress ------------------------------
def rcnn_nms(cfg, mode, images, rpn_proposals, logits, deltas):
    """
    :param cfg: configure
    :param mode: mode. e.g. 'train', 'test', 'eval'
    :param images: a batch of input images
    :param rpn_proposals: rpn proposals (N, ) [i, x0, y0, x1, y1, score, label]
    :param logits:
    :param deltas:
    :return:
        [i, x0, y0, x1, y1, score, label, 0]
    """
    if mode in ['train']:
        nms_prob_threshold = cfg.rcnn_train_nms_pre_score_threshold
        nms_overlap_threshold = cfg.rcnn_train_nms_overlap_threshold
        nms_min_size = cfg.rcnn_train_nms_min_size

    elif mode in ['valid', 'test', 'eval']:
        nms_prob_threshold = cfg.rcnn_test_nms_pre_score_threshold
        nms_overlap_threshold = cfg.rcnn_test_nms_overlap_threshold
        nms_min_size = cfg.rcnn_test_nms_min_size

        if mode in ['eval']:
            nms_prob_threshold = 0.05  # set low numbe r to make roc curve.

    else:
        raise ValueError('rcnn_nms(): invalid mode = %s?' % mode)

    num_classes = cfg.num_classes
    logits = logits.cpu().data.numpy()
    deltas = deltas.cpu().data.numpy().reshape(-1, num_classes, 4)
    batch_size, _, height, width = images.size()

    rpn_proposals = rpn_proposals.cpu().data.numpy()

    # non-max suppression
    rcnn_proposals = []
    for b in range(batch_size):
        pic_proposals = [np.empty((0, 8), np.float32)]
        select_batch = np.where(rpn_proposals[:, 0] == b)[0]
        if len(select_batch) > 0:
            proposal = rpn_proposals[select_batch]
            prob  = np_softmax(logits[select_batch])  # <todo>why use np_sigmoid?
            delta = deltas[select_batch]

            # skip background
            for c in range(1, num_classes):
                index = np.where(prob[:, c] > nms_prob_threshold)[0]
                if len(index) > 0:
                    p = prob[index, c].reshape(-1, 1)
                    d = delta[index, c]
                    # bbox regression, clip & filter
                    box = bbox_decode(proposal[index, 1:5], d)
                    box = clip_boxes(box, width, height)
                    keep = filter_boxes(box, min_size=nms_min_size)

                    if len(keep) > 0:
                        box = box[keep]
                        p = p[keep]
                        keep = gpu_nms(np.hstack((box, p)), nms_overlap_threshold)

                        detection = np.zeros((len(keep), 8), np.float32)
                        detection[:, 0] = b
                        detection[:, 1:5] = np.around(box[keep], 0)
                        detection[:, 5] = p[keep, 0]  # p[:, 0]
                        detection[:, 6] = c
                        detection[:, 7] = 0 #spare
                        pic_proposals.append(detection)

        pic_proposals = np.vstack(pic_proposals)
        rcnn_proposals.append(pic_proposals)

    rcnn_proposals = Variable(torch.from_numpy(np.vstack(rcnn_proposals))).cuda()
    return rcnn_proposals


# ------------------------------ labeling proposals ------------------------------
def _add_truth_box_to_proposal(proposal, img_idx, truth_box, truth_label, score=-1):
    """
    :param proposal: proposals fot ONE IMAGE. e.g.
        [i, x0, y0, x1, y1, score, label]
    :param img_idx: image index in the batch
    :param truth_box:
    :param truth_label:
    :param score:
    :return:
    """
    if len(truth_box) != 0:
        truth = np.zeros((len(truth_box), 8), np.float32)
        truth[:, 0] = img_idx
        truth[:, 1:5] = truth_box
        truth[:, 5] = score  # 1
        truth[:, 6] = truth_label
        truth[:, 7] = 0  #spare
    else:
        truth = np.zeros((0, 8), np.float32)

    sampled_proposal = np.vstack([proposal, truth])
    return sampled_proposal


def _make_one_rcnn_target(cfg, image, proposals, truth_boxes, truth_labels):
    """
    make rcnn target for ONE IMAGE, sampling labels
    https://github.com/ruotianluo/pytorch-faster-rcnn
    :param image: input image
    :param proposals: i is the index if image in batch:
        [i, x0, y0, x1, y1, score, label, 0]
    :param truth_boxes: list of boxes, e.g.
        [x0, y0, x1, y1]
    :param truth_labels: label of each truth box
    :return:
        sampled_proposal: 1 for pos, 0 for neg
        sampled_label: label of sampled truth box
        sampled_assign: which truth box is assigned to the sampled proposal
        sampled_target: bboxes' offsets from sampled proposals to truth boxes
    """
    sampled_proposal = Variable(torch.FloatTensor((0, 8))).cuda()
    sampled_label = Variable(torch.LongTensor((0, 1))).cuda()
    sampled_assign = np.array((0, 1), np.int32)
    sampled_target = Variable(torch.FloatTensor((0, 4))).cuda()

    if len(truth_boxes) == 0 or len(proposals) == 0:
        return sampled_proposal, sampled_label, sampled_assign, sampled_target

    # filter invalid proposals
    _, height, width = image.size()
    num_proposal = len(proposals)

    valid = []
    for i in range(num_proposal):
        box = proposals[i, 1:5]
        if not (is_small_box(box, min_size=cfg.mask_train_min_size)):
            valid.append(i)

    if len(valid) == 0:
        return sampled_proposal, sampled_label, sampled_assign, sampled_target

    proposals = proposals[valid]
    # assign fg/bg to each proposal
    num_proposal = len(proposals)
    box = proposals[:, 1:5]
    # for each bbox, the index of gt which has max overlap with it
    overlap = cython_box_overlap(box, truth_boxes)
    argmax_overlap = np.argmax(overlap, 1)
    max_overlap = overlap[np.arange(num_proposal), argmax_overlap]

    fg_index = np.where(max_overlap >= cfg.rcnn_train_fg_thresh_low)[0]
    bg_index = np.where((max_overlap < cfg.rcnn_train_bg_thresh_high) & \
                        (max_overlap >= cfg.rcnn_train_bg_thresh_low))[0]

    # sampling for class balance
    num_classes = cfg.num_classes
    num = cfg.rcnn_train_batch_size
    num_fg = int(np.round(cfg.rcnn_train_fg_fraction * cfg.rcnn_train_batch_size))

    # Small modification to the original version where we ensure a fixed number of regions are sampled
    # https://github.com/precedenceguo/mx-rcnn/commit/3853477d9155c1f340241c04de148166d146901d
    fg_length = len(fg_index)
    bg_length = len(bg_index)

    if fg_length > 0 and bg_length > 0:
        num_fg = min(num_fg, fg_length)
        fg_index = fg_index[
            np.random.choice(fg_length, size=num_fg, replace=fg_length < num_fg)
        ]
        num_bg = num - num_fg
        bg_index = bg_index[
            np.random.choice(bg_length, size=num_bg, replace=bg_length < num_bg)
        ]
    # no bgs
    elif fg_length > 0:
        num_fg = num
        num_bg = 0
        fg_index = fg_index[
            np.random.choice(fg_length, size=num_fg, replace=fg_length < num_fg)
        ]
    # no fgs
    elif bg_length > 0:
        num_fg = 0
        num_bg = num
        bg_index = bg_index[
            np.random.choice(bg_length, size=num_bg, replace=bg_length < num_bg)
        ]
    # no bgs and no fgs?
    else:
        num_fg = 0
        num_bg = num
        bg_index = np.random.choice(num_proposal, size=num_bg, replace=num_proposal < num_bg)

    assert ((num_fg + num_bg) == num)

    # selecting both fg and bg
    index = np.concatenate([fg_index, bg_index], 0)
    sampled_proposal = proposals[index]

    # label
    sampled_assign = argmax_overlap[index]
    sampled_label = truth_labels[sampled_assign]
    sampled_label[num_fg:] = 0  # Clamp labels for the background to 0

    # target
    if num_fg > 0:
        target_truth_box = truth_boxes[sampled_assign[:num_fg]]
        target_box = sampled_proposal[:num_fg][:, 1:5]
        sampled_target = bbox_encode(target_box, target_truth_box)

    sampled_target = Variable(torch.from_numpy(sampled_target)).cuda()
    sampled_label = Variable(torch.from_numpy(sampled_label)).long().cuda()
    sampled_proposal = Variable(torch.from_numpy(sampled_proposal)).cuda()
    return sampled_proposal, sampled_label, sampled_assign, sampled_target


def make_rcnn_target(cfg, images, proposals, truth_boxes, truth_labels):
    """
    a sampled subset of proposals, with it's_train corresponding truth label and offsets
    :param images: (B, 3, H, W), BGR mode
    :param proposals: (B, 8), [i, x0, y0, x1, y1, score, label, 0]
    :param truth_boxes: (B, _, 4)
    :param truth_labels: (B, _, 1)
    :return:
    """
    # <todo> take care of don't care ground truth. Here, we only ignore them  ----
    truth_boxes = copy.deepcopy(truth_boxes)
    truth_labels = copy.deepcopy(truth_labels)
    batch_size = len(images)
    # filter truth labels is 0 todo: do we really need to check this?
    for img_idx in range(batch_size):
        index = np.where(truth_labels[img_idx] > 0)[0]
        truth_boxes[img_idx] = truth_boxes[img_idx][index]
        truth_labels[img_idx] = truth_labels[img_idx][index]

    proposals = proposals.cpu().data.numpy()
    sampled_proposals = []
    sampled_labels =    []
    sampled_assigns =   []
    sampled_targets =   []

    batch_size = len(truth_boxes)
    for img_idx in range(batch_size):
        image            = images[img_idx]
        img_truth_boxes  = truth_boxes[img_idx]
        img_truth_labels = truth_labels[img_idx]

        if len(img_truth_boxes) != 0:
            if len(proposals) == 0:
                img_proposals = np.zeros((0, 8), np.float32)
            else:
                img_proposals = proposals[proposals[:, 0] == img_idx]

            img_proposals = _add_truth_box_to_proposal(img_proposals, img_idx, img_truth_boxes, img_truth_labels)

            sampled_proposal, sampled_label, sampled_assign, sampled_target = \
                _make_one_rcnn_target(cfg, image, img_proposals, img_truth_boxes, img_truth_labels)

            sampled_proposals.append(sampled_proposal)
            sampled_labels.append(sampled_label)
            sampled_assigns.append(sampled_assign)
            sampled_targets.append(sampled_target)

    sampled_proposals = torch.cat(sampled_proposals, 0)
    sampled_labels = torch.cat(sampled_labels, 0)
    sampled_targets = torch.cat(sampled_targets, 0)
    sampled_assigns = np.hstack(sampled_assigns)

    return sampled_proposals, sampled_labels, sampled_assigns, sampled_targets


# ------------------------------ Calculate loss ------------------------------
def rcnn_reg_loss(labels, deltas, targets, deltas_sigma=1.0):
    """
    :param labels: (\sum B_i*num_proposals_i, )
    :param deltas: (\sum B_i*num_proposals_i, num_classes*4)
    :param targets:(\sum B_i*num_proposals_i, 4)
    :param deltas_sigma: float
    :return: float
    """
    batch_size, num_classes_mul_4 = deltas.size()
    num_classes = num_classes_mul_4 // 4
    deltas = deltas.view(batch_size, num_classes, 4)

    num_pos_proposals = len(labels.nonzero())
    if num_pos_proposals > 0:
        # one hot encode. select could also seen as mask matrix
        select = Variable(torch.zeros((batch_size, num_classes))).cuda()
        select.scatter_(1, labels.view(-1, 1), 1)
        select[:, 0] = 0  # bg is 0, label starts from 1

        select = select.view(batch_size, num_classes, 1).expand((batch_size, num_classes, 4)).contiguous().byte()
        deltas = deltas[select].view(-1, 4)

        deltas_sigma2 = deltas_sigma * deltas_sigma
        return F.smooth_l1_loss(deltas * deltas_sigma2, targets * deltas_sigma2,
                                size_average=False) / deltas_sigma2 / num_pos_proposals
    else:
        return Variable(torch.cuda.FloatTensor(1).zero_()).sum()


def rcnn_cls_loss(logits, labels):
    """
    :param logits: (\sum B_i*num_proposals_i, num_classes)
    :param labels: (\sum B_i*num_proposals_i, )
    :return: float
    """
    return F.cross_entropy(logits, labels, size_average=True)