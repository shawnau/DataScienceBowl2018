import copy
import cv2
import numpy as np
import skimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.lib.box.process import is_small_box
from net.lib.box.overlap.cython_overlap.cython_box_overlap import cython_box_overlap
from utility.func import np_sigmoid
from net.loss import binary_cross_entropy_with_logits

# ----------------------------- NET -----------------------------
class MaskHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(MaskHead, self).__init__()
        self.num_classes = cfg.num_classes

        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, stride=1)
        self.bn1   = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn2   = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1)
        self.bn4   = nn.BatchNorm2d(256)

        self.up    = nn.ConvTranspose2d(256, 256, kernel_size=4, padding=1, stride=2, bias=False)
        self.logit = nn.Conv2d(256, self.num_classes, kernel_size=1, padding=0, stride=1)

    def forward(self, crops):
        """
        :param crops: (B, feature_channels, crop_size, crop_size) cropped feature map
        :return:
            logits (B, num_classes, 2*crop_size, 2*crop_size)
        """
        x = F.relu(self.bn1(self.conv1(crops)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        x = F.relu(self.bn3(self.conv3(x)), inplace=True)
        x = F.relu(self.bn4(self.conv4(x)), inplace=True)
        x = self.up(x)
        logits = self.logit(x)

        return logits


# ----------------------------- NMS -----------------------------
def make_empty_masks(cfg, mode, inputs):
    masks = []
    batch_size, C, H, W = inputs.size()
    for b in range(batch_size):
        mask = np.zeros((H, W), np.float32)
        masks.append(mask)
    return masks


def instance_to_binary(instance, threshold, min_area):
    binary = instance > threshold
    label  = skimage.morphology.label(binary)
    num_labels = label.max()
    if num_labels>0:
        areas    = [(label==c+1).sum() for c in range(num_labels)]
        max_area = max(areas)

        for c in range(num_labels):
            if areas[c] != max_area:
                binary[label==c+1]=0
            else:
                if max_area<min_area:
                    binary[label==c+1]=0

    #<todo> fill holes? ---

    return binary


def mask_nms(cfg, images, proposals, mask_logits):
    """
    1. do non-maximum suppression to remove overlapping segmentations
    2. resize the masks from mask head output (28*28) into proposal size
    3. paste the masks into input image
    #<todo> better nms for mask
    :param cfg:
    :param images: (B, C, H, W)
    :param proposals: (B, 7) [i, x0, y0, x1, y1, score, label]
    :param mask_logits: (B, num_classes, 2*crop_size, 2*crop_size)
    :return:
        b_multi_masks: (B, H, W) masks labelled with 1,2,...N (total number of masks)
        b_mask_instances: (B*N, H, W) masks with prob
        b_mask_proposals: (B*N, ) proposals
    """
    overlap_threshold   = cfg.mask_test_nms_overlap_threshold
    pre_score_threshold = cfg.mask_test_nms_pre_score_threshold
    mask_threshold      = cfg.mask_test_mask_threshold
    mask_min_area       = cfg.mask_test_mask_min_area

    proposals   = proposals.cpu().data.numpy()
    mask_logits = mask_logits.cpu().data.numpy()
    mask_probs  = np_sigmoid(mask_logits)

    b_multi_masks = []
    b_mask_proposals = []
    b_mask_instances = []
    batch_size, C, H, W = images.size()
    for b in range(batch_size):
        multi_masks = np.zeros((H, W), np.float32)  # multi masks for a image
        mask_proposals = []  # proposals for a image
        mask_instances = []  # instances for a image
        num_keeps = 0

        index = np.where((proposals[:, 0] == b) & (proposals[:, 5] > pre_score_threshold))[0]
        if len(index) != 0:
            instances = []    # all instances
            boxes = []        # all boxes
            for i in index:
                mask = np.zeros((H, W), np.bool)

                x0, y0, x1, y1 = proposals[i, 1:5].astype(np.int32)
                h, w = y1-y0+1, x1-x0+1
                label = int(proposals[i, 6])    # get label of the instance
                crop = mask_probs[i, label]     # get mask channel of the label
                crop = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
                # crop = crop > mask_threshold  # turn prob feature map into 0/1 mask
                mask[y0:y1+1, x0:x1+1] = crop   # paste mask into empty mask

                instances.append(mask)
                boxes.append([x0, y0, x1, y1])

            # compute box overlap, do nms
            L = len(index)
            binary = [instance_to_binary(m, mask_threshold, mask_min_area) for m in instances]
            boxes = np.array(boxes, np.float32)
            box_overlap = cython_box_overlap(boxes, boxes)
            instance_overlap = np.zeros((L, L), np.float32)

            # calculate instance overlapping iou
            for i in range(L):
                instance_overlap[i, i] = 1
                for j in range(i+1, L):
                    if box_overlap[i, j] < 0.01:
                        continue

                    x0 = int(min(boxes[i, 0], boxes[j, 0]))
                    y0 = int(min(boxes[i, 1], boxes[j, 1]))
                    x1 = int(max(boxes[i, 2], boxes[j, 2]))
                    y1 = int(max(boxes[i, 3], boxes[j, 3]))

                    mi = binary[i][y0:y1, x0:x1]
                    mj = binary[j][y0:y1, x0:x1]

                    intersection = (mi & mj).sum()
                    union = (mi | mj).sum()
                    instance_overlap[i, j] = intersection/(union + 1e-12)
                    instance_overlap[j, i] = instance_overlap[i, j]

            # non-max-suppression to remove overlapping segmentation
            score = proposals[index, 5]
            sort_idx = list(np.argsort(-score))

            # https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
            keep = []
            while len(sort_idx) > 0:
                i = sort_idx[0]
                keep.append(i)
                delete_index = list(np.where(instance_overlap[i] > overlap_threshold)[0])
                sort_idx = [e for e in sort_idx if e not in delete_index]
            # filter instances & proposals
            num_keeps = len(keep)
            for i in range(num_keeps):
                k = keep[i]
                multi_masks[np.where(binary[k])] = i + 1
                mask_instances.append(instances[k].reshape(1, H, W))

                t = index[k]
                b, x0, y0, x1, y1, score, label = proposals[t]
                mask_proposals.append(np.array([b, x0, y0, x1, y1, score, label], np.float32))

        if num_keeps==0:
            mask_proposals = np.zeros((0,7  ),np.float32)
            mask_instances = np.zeros((0,H,W),np.float32)
        else:
            mask_proposals = np.vstack(mask_proposals)
            mask_instances = np.vstack(mask_instances)

        b_mask_proposals.append(mask_proposals)
        b_mask_instances.append(mask_instances)
        b_multi_masks.append(multi_masks)

    b_mask_proposals = Variable(torch.from_numpy(np.vstack(b_mask_proposals))).cuda()
    return b_multi_masks, b_mask_instances, b_mask_proposals


# ----------------------------- Label Target -----------------------------
def _add_truth_box_to_proposal(cfg, proposal, b, truth_box, truth_label, score=-1):
    if len(truth_box) !=0:
        truth = np.zeros((len(truth_box),7),np.float32)
        truth[:,0  ] = b
        truth[:,1:5] = truth_box
        truth[:,5  ] = score #1  #
        truth[:,6  ] = truth_label
    else:
        truth = np.zeros((0,7),np.float32)

    sampled_proposal = np.vstack([proposal,truth])
    return sampled_proposal


# <todo> mask crop should match align kernel (same wait to handle non-integer pixel location (e.g. 23.5, 32.1))
def _crop_instance(instance, box, size, threshold=0.5):
    """
    return the ground truth mask for mask head
    :param instance: one mask of (H, W) of input image, e.g.
        [[0, 1, 1, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]]
        for a 3x3 image
    :param box: bbox on input image. e.g.
        [x0, y0, x1, y1]
    :param size: size of the output of maskhead. e.g. 28*28
    :param threshold: used to define pos/neg pixels of the mask
    :return: cropped & resized mask into size of the mask head output
    """
    H, W = instance.shape
    x0, y0, x1, y1 = np.rint(box).astype(np.int32)
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(W, x1)
    y1 = min(H, y1)

    #<todo> filter this
    if 1:
        if x0 == x1:
            x0 = x0-1
            x1 = x1+1
            x0 = max(0, x0)
            x1 = min(W, x1)
        if y0 == y1:
            y0 = y0-1
            y1 = y1+1
            y0 = max(0, y0)
            y1 = min(H, y1)

    #print(x0,y0,x1,y1)
    crop = instance[y0:y1+1,x0:x1+1]
    crop = cv2.resize(crop,(size,size))
    crop = (crop > threshold).astype(np.float32)
    return crop


# cpu version
def _make_one_mask_target(cfg, mode, image, proposals, truth_box, truth_label, truth_instance):
    """
    make mask targets for one image.
    1. assign truth box to each proposals by threshold for fg/bg
    2. crop assigned instance into bbox size
    3. resize to maskhead's_train output size.
    :param image: image as (H, W, C) numpy array
    :param proposals: list of regional proposals generated by RCNN. e.g.
        [[i, x0, y0, x1, y1, score, label], ...]
    :param truth_box: list of truth boxes. e.g.
        [[x0, y0, x1, y1], ...]
    :param truth_label: 1s
        maskhead are used to predict mask,
        all masks are positive proposals. (foreground)
        here we have 2 classes so it's_train fixed to 1
    :param truth_instance: list of truth instances, (H, W)
    :return:
        sampled_proposal: same as proposals
        sampled_label: same as truth_label
        sampled_instance: cropped instance, matching maskhead's_train output
        sampled_assign: index of truth_box each proposals belongs to
    """
    sampled_proposal = Variable(torch.FloatTensor(0, 7)).cuda()
    sampled_label    = Variable(torch.LongTensor (0, 1)).cuda()
    sampled_instance = Variable(torch.FloatTensor(0, 1, 1)).cuda()

    if len(truth_box) == 0 or len(proposals) == 0:
        return sampled_proposal, sampled_label, sampled_instance

    # filter invalid proposals like small proposals
    _, height, width = image.size()
    num_proposal = len(proposals)

    valid = []
    for i in range(num_proposal):
        box = proposals[i, 1:5]
        if not(is_small_box(box, min_size=cfg.mask_train_min_size)):  # is_small_box_at_boundary
            valid.append(i)

    if len(valid) == 0:
        return sampled_proposal, sampled_label, sampled_instance

    proposals = proposals[valid]
    # assign bbox to proposals by overlap threshold
    num_proposal = len(proposals)
    box = proposals[:, 1:5]
    # for each bbox, the index of gt which has max overlap with it
    overlap = cython_box_overlap(box, truth_box)
    argmax_overlap = np.argmax(overlap, 1)
    max_overlap = overlap[np.arange(num_proposal), argmax_overlap]

    fg_index = np.where(max_overlap >= cfg.mask_train_fg_thresh_low)[0]

    if len(fg_index) == 0:
        return sampled_proposal, sampled_label, sampled_instance

    fg_length = len(fg_index)
    num_fg = cfg.mask_train_batch_size
    fg_index = fg_index[
        np.random.choice(fg_length, size=num_fg, replace=fg_length < num_fg)
    ]

    sampled_proposal = proposals[fg_index]
    sampled_assign   = argmax_overlap[fg_index]     # assign a gt to each bbox
    sampled_label    = truth_label[sampled_assign]  # assign gt's_train label to each bbox
    sampled_instance = []
    for i in range(len(fg_index)):
        instance = truth_instance[sampled_assign[i]]  # for each positive bbox, find instance it belongs to
        box  = sampled_proposal[i, 1:5]
        crop = _crop_instance(instance, box, cfg.mask_size)  # crop the instance by box
        sampled_instance.append(crop[np.newaxis, :, :])

    # save
    sampled_instance = np.vstack(sampled_instance)
    sampled_proposal = Variable(torch.from_numpy(sampled_proposal)).cuda()
    sampled_label    = Variable(torch.from_numpy(sampled_label)).long().cuda()
    sampled_instance = Variable(torch.from_numpy(sampled_instance)).cuda()
    return sampled_proposal, sampled_label, sampled_instance


def make_mask_target(cfg, mode, images, proposals, truth_boxes, truth_labels, truth_instances):
    """
    :param:
        images:
        proposals:
        truth_boxes:
        truth_labels:
        truth_instances:
    :return:
        sampled_proposals: bbox which has overlap with one gt > threshold
        sampled_labels: class label of the bbox
        sampled_assigns: which gt the bbox is assigned to (seems not used)
        sampled_instances: cropped/resized instance mask into mask output (28*28)
    """
    #<todo> take care of don't care ground truth. Here, we only ignore them  ---
    truth_boxes     = copy.deepcopy(truth_boxes)
    truth_labels    = copy.deepcopy(truth_labels)
    truth_instances = copy.deepcopy(truth_instances)
    batch_size = len(images)
    for b in range(batch_size):
        index = np.where(truth_labels[b]>0)[0]
        truth_boxes [b] = truth_boxes [b][index]
        truth_labels[b] = truth_labels[b][index]
        truth_instances[b] = truth_instances[b][index]

    proposals = proposals.cpu().data.numpy()
    sampled_proposals  = []
    sampled_labels     = []
    sampled_assigns    = []
    sampled_instances  = []

    batch_size = len(truth_boxes)
    for b in range(batch_size):
        image          = images[b]
        truth_box      = truth_boxes[b]
        truth_label    = truth_labels[b]
        truth_instance = truth_instances[b]

        if len(truth_box) != 0:
            if len(proposals)==0:
                proposal = np.zeros((0,7),np.float32)
            else:
                proposal = proposals[proposals[:,0]==b]

            # proposal = _add_truth_box_to_proposal(cfg, proposal, b, truth_box, truth_label)

            sampled_proposal, sampled_label, sampled_instance = \
                _make_one_mask_target(cfg, mode, image, proposal, truth_box, truth_label, truth_instance)

            sampled_proposals.append(sampled_proposal)
            sampled_labels.append(sampled_label)
            sampled_instances.append(sampled_instance)

    sampled_proposals = torch.cat(sampled_proposals,0)
    sampled_labels    = torch.cat(sampled_labels,0)
    sampled_instances = torch.cat(sampled_instances,0)

    return sampled_proposals, sampled_labels, sampled_instances


# ---------------------------- Loss ----------------------------
def mask_loss(logits, labels, instances):
    """
    :param logits:  (\sum B_i*num_proposals_i, num_classes, 2*crop_size, 2*crop_size)
    :param labels:  (\sum B_i*num_proposals_i, )
    :param instances: (\sum B_i*num_proposals_i, 2*crop_size, 2*crop_size)
    :return:
    """
    batch_size, num_classes = logits.size(0), logits.size(1)

    logits_flat = logits.view(batch_size, num_classes, -1)
    dim = logits_flat.size(2)

    # one hot encode
    select = Variable(torch.zeros((batch_size, num_classes))).cuda()
    select.scatter_(1, labels.view(-1, 1), 1)
    select[:, 0] = 0
    select = select.view(batch_size, num_classes, 1).expand((batch_size, num_classes, dim)).contiguous().byte()

    logits_flat = logits_flat[select].view(-1)
    labels_flat = instances.view(-1)

    return binary_cross_entropy_with_logits(logits_flat, labels_flat)