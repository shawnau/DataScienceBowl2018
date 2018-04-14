import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parallel.data_parallel import data_parallel

from net.layer.SE_ResNeXt_FPN import SEResNeXtFPN
from net.layer.crop import CropRoi
from net.layer.rpn  import RpnMultiHead, make_rpn_target,  rpn_nms,  rpn_cls_loss,  rpn_reg_loss, fpn_make_anchor_boxes
from net.layer.rcnn import RcnnHead,     make_rcnn_target, rcnn_nms, rcnn_cls_loss, rcnn_reg_loss
from net.layer.mask import MaskHead,     make_mask_target, mask_nms, mask_loss, make_empty_masks


class MaskRcnnNet(nn.Module):

    def __init__(self, cfg):
        super(MaskRcnnNet, self).__init__()
        self.version = 'net version \'mask-rcnn-se-resnext50-fpn\''
        self.cfg  = cfg
        self.mode = 'train'

        feature_channels = 256
        crop_channels = feature_channels
        self.feature_net = SEResNeXtFPN(cfg, [3, 4, 6, 3])
        self.rpn_head    = RpnMultiHead(cfg, feature_channels)
        self.rcnn_crop   = CropRoi  (cfg, cfg.rcnn_crop_size)
        self.rcnn_head   = RcnnHead (cfg, crop_channels)
        self.mask_crop   = CropRoi  (cfg, cfg.mask_crop_size)
        self.mask_head   = MaskHead (cfg, crop_channels)

    def forward(self, images, truth_boxes=None, truth_labels=None, truth_instances=None):
        features = self.feature_net(images)

        # rpn proposals -------------------------------------------
        self.rpn_logits_flat, self.rpn_deltas_flat = self.rpn_head(features)
        self.anchor_boxes = fpn_make_anchor_boxes(features, self.cfg)
        self.rpn_proposals = rpn_nms(self.cfg, self.mode, images,
                                     self.anchor_boxes,
                                     self.rpn_logits_flat,
                                     self.rpn_deltas_flat)

        # make tagets for rpn and rcnn ------------------------------------------------
        if self.mode in ['train', 'valid']:
            self.rpn_labels, \
            self.rpn_label_assigns, \
            self.rpn_label_weights, \
            self.rpn_targets, \
            self.rpn_targets_weights = \
                make_rpn_target(self.cfg, images, self.anchor_boxes, truth_boxes, truth_labels)

            self.sampled_rcnn_proposals, \
            self.sampled_rcnn_labels, \
            self.sampled_rcnn_assigns, \
            self.sampled_rcnn_targets = \
                make_rcnn_target(self.cfg, images, self.rpn_proposals, truth_boxes, truth_labels)

            self.rpn_proposals = self.sampled_rcnn_proposals  # use sampled proposals for training

        # rcnn proposals ------------------------------------------------
        self.rcnn_proposals = self.rpn_proposals # for eval only when no porposals
        if len(self.rpn_proposals) > 0:
            rcnn_crops = self.rcnn_crop(features, self.rpn_proposals)
            self.rcnn_logits, self.rcnn_deltas = self.rcnn_head(rcnn_crops)
            self.rcnn_proposals = rcnn_nms(self.cfg, self.mode, images,
                                           self.rpn_proposals,
                                           self.rcnn_logits,
                                           self.rcnn_deltas)

        # make targets for mask head ------------------------------------
        if self.mode in ['train', 'valid']:
            self.sampled_rcnn_proposals, \
            self.sampled_mask_labels, \
            self.sampled_mask_instances,   = \
                make_mask_target(self.cfg, self.mode, images,
                                 self.rcnn_proposals,
                                 truth_boxes,
                                 truth_labels,
                                 truth_instances)
            self.rcnn_proposals = self.sampled_rcnn_proposals

        # segmentation  -------------------------------------------
        self.detections = self.rcnn_proposals
        self.masks = make_empty_masks(self.cfg, self.mode, images)
        self.mask_instances = []

        if len(self.rcnn_proposals) > 0:
            mask_crops = self.mask_crop(features, self.detections)
            self.mask_logits = self.mask_head(mask_crops)
            self.masks, self.mask_instances, self.mask_proposals = mask_nms(self.cfg, images, self.rcnn_proposals, self.mask_logits)
            self.detections = self.mask_proposals

    def loss(self):
        self.rpn_cls_loss = rpn_cls_loss(self.rpn_logits_flat,
                                         self.rpn_labels,
                                         self.rpn_label_weights)

        self.rpn_reg_loss = rpn_reg_loss(self.rpn_labels,
                                         self.rpn_deltas_flat,
                                         self.rpn_targets,
                                         self.rpn_targets_weights)

        self.rcnn_cls_loss = rcnn_cls_loss(self.rcnn_logits,
                                           self.sampled_rcnn_labels)

        self.rcnn_reg_loss = rcnn_reg_loss(self.sampled_rcnn_labels,
                                           self.rcnn_deltas,
                                           self.sampled_rcnn_targets)

        self.mask_cls_loss  = mask_loss(self.mask_logits,
                                        self.sampled_mask_labels,
                                        self.sampled_mask_instances)

        self.total_loss = self.rpn_cls_loss + \
                          self.rpn_reg_loss + \
                          self.rcnn_cls_loss + \
                          self.rcnn_reg_loss + \
                          self.mask_cls_loss

        return self.total_loss


    def forward_mask(self, images, rcnn_proposals):
        cfg  = self.cfg
        mode = self.mode
        self.rcnn_proposals = rcnn_proposals

        features = data_parallel(self.feature_net, images)
        self.detections = self.rcnn_proposals
        self.masks      = make_empty_masks(cfg, mode, images)
        if len(self.rcnn_proposals) > 0:
              mask_crops = self.mask_crop(features, self.detections)
              self.mask_logits = data_parallel(self.mask_head, mask_crops)
              self.masks,  self.mask_instances, self.mask_proposals  = mask_nms(cfg, images, self.rcnn_proposals, self.mask_logits)
              self.detections = self.mask_proposals


    def set_mode(self, mode ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
        else:
            raise NotImplementedError

    def load_pretrain(self, pretrain_file, skip=[]):
        pretrain_state_dict = torch.load(pretrain_file)
        state_dict = self.state_dict()

        keys = list(state_dict.keys())
        for key in keys:
            if any(s in key for s in skip):
                continue
            state_dict[key] = pretrain_state_dict[key]

        self.load_state_dict(state_dict)
