from common import *
from net.lib.roi_align_pool_tf.module import RoIAlign as Crop
from .configuration import *
from .layer.rpn_multi_nms     import *
from .layer.rpn_multi_target  import *
from .layer.rpn_multi_loss    import *
from .layer.rcnn_nms     import *
from .layer.rcnn_target  import *
from .layer.rcnn_loss    import *
from .layer.mask_nms    import *
from .layer.mask_target import *
from .layer.mask_loss   import *


# P layers ## ---------------------------
class LateralBlock(nn.Module):
    def __init__(self, c_planes, p_planes, out_planes ):
        super(LateralBlock, self).__init__()
        self.lateral = nn.Conv2d(c_planes,  p_planes,   kernel_size=1, padding=0, stride=1)
        self.top     = nn.Conv2d(p_planes,  out_planes, kernel_size=3, padding=1, stride=1)

    def forward(self, c , p):
        _,_,H,W = c.size()
        c = self.lateral(c)
        p = F.upsample(p, scale_factor=2,mode='nearest')
        p = p[:,:,:H,:W] + c
        p = self.top(p)

        return p

## C layers ## ---------------------------
class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, out_planes, is_downsample=False, stride=1):
        super(BottleneckBlock, self).__init__()
        self.is_downsample = is_downsample

        self.bn1   = nn.BatchNorm2d(in_planes,eps = 2e-5)
        self.conv1 = nn.Conv2d(in_planes,     planes, kernel_size=1, padding=0, stride=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes,eps = 2e-5)
        self.conv2 = nn.Conv2d(   planes,     planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn3   = nn.BatchNorm2d(planes,eps = 2e-5)
        self.conv3 = nn.Conv2d(   planes, out_planes, kernel_size=1, padding=0, stride=1, bias=False)

        if is_downsample:
            self.downsample = nn.Conv2d(in_planes, out_planes, kernel_size=1, padding=0, stride=stride, bias=False)

    def forward(self, x):

        x = F.relu(self.bn1(x),inplace=True)
        z = self.conv1(x)
        z = F.relu(self.bn2(z),inplace=True)
        z = self.conv2(z)
        z = F.relu(self.bn3(z),inplace=True)
        z = self.conv3(z)

        if self.is_downsample:
            z += self.downsample(x)
        else:
            z += x

        return z


def make_layer_c0(in_planes, out_planes):
    layers = [
        nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)


def make_layer_c(in_planes, planes, out_planes, num_blocks, stride):
    layers = []
    layers.append(BottleneckBlock(in_planes, planes, out_planes, is_downsample=True, stride=stride))
    for i in range(1, num_blocks):
        layers.append(BottleneckBlock(out_planes, planes, out_planes))

    return nn.Sequential(*layers)


class FeatureNet(nn.Module):

    def __init__(self, cfg, in_channels, out_channels=256 ):
        super(FeatureNet, self).__init__()
        self.cfg=cfg

        # bottom-top
        self.layer_c0 = make_layer_c0(in_channels, 64)

        self.layer_c1 = make_layer_c(   64,  64,  256, num_blocks=3, stride=1)  #out =  64*4 =  256
        self.layer_c2 = make_layer_c(  256, 128,  512, num_blocks=4, stride=2)  #out = 128*4 =  512
        self.layer_c3 = make_layer_c(  512, 256, 1024, num_blocks=6, stride=2)  #out = 256*4 = 1024
        self.layer_c4 = make_layer_c( 1024, 512, 2048, num_blocks=3, stride=2)  #out = 512*4 = 2048

        # top-down
        self.layer_p4 = nn.Conv2d   ( 2048, out_channels, kernel_size=1, stride=1, padding=0)
        self.layer_p3 = LateralBlock( 1024, out_channels, out_channels)
        self.layer_p2 = LateralBlock(  512, out_channels, out_channels)
        self.layer_p1 = LateralBlock(  256, out_channels, out_channels)

    def forward(self, x):
        #pass                        #; print('input ',   x.size())
        c0 = self.layer_c0 (x)       #; print('layer_c0 ',c0.size())
                                     #
        c1 = self.layer_c1(c0)       #; print('layer_c1 ',c1.size())
        c2 = self.layer_c2(c1)       #; print('layer_c2 ',c2.size())
        c3 = self.layer_c3(c2)       #; print('layer_c3 ',c3.size())
        c4 = self.layer_c4(c3)       #; print('layer_c4 ',c4.size())

        p4 = self.layer_p4(c4)       #; print('layer_p4 ',p4.size())
        p3 = self.layer_p3(c3, p4)   #; print('layer_p3 ',p3.size())
        p2 = self.layer_p2(c2, p3)   #; print('layer_p2 ',p2.size())
        p1 = self.layer_p1(c1, p2)   #; print('layer_p1 ',p1.size())

        features = [p1,p2,p3,p4]
        assert(len(self.cfg.rpn_scales) == len(features))

        return features


# various head ##########################################
class RpnMultiHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(RpnMultiHead, self).__init__()

        self.num_classes = cfg.num_classes
        self.num_scales  = len(cfg.rpn_scales)
        self.num_bases   = [len(b) for b in cfg.rpn_base_apsect_ratios]

        self.convs  = nn.ModuleList()
        self.logits = nn.ModuleList()
        self.deltas = nn.ModuleList()
        for l in range(self.num_scales):
            channels = in_channels*2
            self.convs.append ( nn.Conv2d(in_channels, channels, kernel_size=3, padding=1) )
            self.logits.append(
                nn.Sequential(
                    nn.Conv2d(channels, self.num_bases[l]*self.num_classes,   kernel_size=3, padding=1) ,
                )
            )
            self.deltas.append(
                nn.Sequential(
                    nn.Conv2d(channels, self.num_bases[l]*self.num_classes*4, kernel_size=3, padding=1),
                )
            )

    def forward(self, fs):
        batch_size = len(fs[0])

        logits_flat = []
        probs_flat  = []
        deltas_flat = []
        for l in range(self.num_scales):  # apply multibox head to feature maps
            f = fs[l]
            f = F.relu(self.convs[l](f))

            f = F.dropout(f, p=0.5, training=self.training)
            logit = self.logits[l](f)
            delta = self.deltas[l](f)

            logit_flat = logit.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)
            delta_flat = delta.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes, 4)
            logits_flat.append(logit_flat)
            deltas_flat.append(delta_flat)

        logits_flat = torch.cat(logits_flat, 1)
        deltas_flat = torch.cat(deltas_flat, 1)

        return logits_flat, deltas_flat


# https://qiita.com/yu4u/items/5cbe9db166a5d72f9eb8
class CropRoi(nn.Module):
    """
    RoiAlign
    :param:
        cfg: configure
        crop_size: size of output of RoiAlign. e.g. 14 (means 14*14)
    :Input:
        fs: features from FPN-ResNet50, e.g. [p1, p2, p3, p4]
        proposals: proposals. e.g.
            [i, x0, y0, x1, y1, score, label]
    :return:
        cropped feature map using RoiAlign
    """
    def __init__(self, cfg, crop_size):
        super(CropRoi, self).__init__()
        self.num_scales = len(cfg.rpn_scales)
        self.crop_size  = crop_size
        self.sizes      = cfg.rpn_base_sizes
        self.scales     = cfg.rpn_scales

        self.crops = nn.ModuleList()
        for l in range(self.num_scales):
            self.crops.append(
                Crop(self.crop_size, self.crop_size, 1/self.scales[l])
            )

    def forward(self, fs, proposals):
        num_proposals = len(proposals)

        # this is  complicated. we need to decide for a given roi,
        # which of the p0,p1, ..p3 layers to pool from
        boxes = proposals.detach().data[:, 1:5]
        sizes = boxes[:, 2:]-boxes[:, :2]
        sizes = torch.sqrt(sizes[:, 0]*sizes[:, 1])
        distances = torch.abs(sizes.view(num_proposals, 1).expand(num_proposals, 4) \
                              - torch.from_numpy(np.array(self.sizes, np.float32)).cuda())
        min_distances, min_index = distances.min(1)

        rois = proposals.detach().data[:, 0:5]
        rois = Variable(rois)

        crops   = []
        indices = []
        for l in range(self.num_scales):
            index = (min_index == l).nonzero()

            if len(index) > 0:
                crop = self.crops[l](fs[l], rois[index].view(-1,5))
                crops.append(crop)
                indices.append(index)

        crops   = torch.cat(crops,0)
        indices = torch.cat(indices,0).view(-1)
        crops   = crops[torch.sort(indices)[1]]
        # crops = torch.index_select(crops,0,index)
        return crops


class RcnnHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(RcnnHead, self).__init__()
        self.num_classes = cfg.num_classes
        self.crop_size   = cfg.rcnn_crop_size

        self.fc1 = nn.Linear(in_channels*self.crop_size*self.crop_size,1024)
        self.fc2 = nn.Linear(1024,1024)
        self.logit = nn.Linear(1024,self.num_classes)
        self.delta = nn.Linear(1024,self.num_classes*4)

    def forward(self, crops):

        x = crops.view(crops.size(0), -1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = F.dropout(x,0.5,training=self.training)
        logits = self.logit(x)
        deltas = self.delta(x)

        return logits, deltas


class MaskHead(nn.Module):

    def __init__(self, cfg, in_channels):
        super(MaskHead, self).__init__()
        self.num_classes = cfg.num_classes

        self.conv1 = nn.Conv2d( in_channels,256, kernel_size=3, padding=1, stride=1)
        self.bn1   = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256,256, kernel_size=3, padding=1, stride=1)
        self.bn2   = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256,256, kernel_size=3, padding=1, stride=1)
        self.bn3   = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256,256, kernel_size=3, padding=1, stride=1)
        self.bn4   = nn.BatchNorm2d(256)

        self.up    = nn.ConvTranspose2d(256,256, kernel_size=4, padding=1, stride=2, bias=False)
        self.logit = nn.Conv2d( 256, self.num_classes, kernel_size=1, padding=0, stride=1)

    def forward(self, crops):
        x = F.relu(self.bn1(self.conv1(crops)),inplace=True)
        x = F.relu(self.bn2(self.conv2(x)),inplace=True)
        x = F.relu(self.bn3(self.conv3(x)),inplace=True)
        x = F.relu(self.bn4(self.conv4(x)),inplace=True)
        x = self.up(x)
        logits = self.logit(x)

        return logits


class MaskRcnnNet(nn.Module):

    def __init__(self, cfg):
        super(MaskRcnnNet, self).__init__()
        self.version = 'net version \'mask-rcnn-resnet50-fpn\''
        self.cfg  = cfg
        self.mode = 'train'

        feature_channels = 128
        crop_channels = feature_channels
        self.feature_net = FeatureNet(cfg, 3, feature_channels)
        self.rpn_head    = RpnMultiHead(cfg,feature_channels)
        self.rcnn_crop   = CropRoi  (cfg, cfg.rcnn_crop_size)
        self.rcnn_head   = RcnnHead (cfg, crop_channels)
        self.mask_crop   = CropRoi  (cfg, cfg.mask_crop_size)
        self.mask_head   = MaskHead (cfg, crop_channels)

    def forward(self, inputs,
                truth_boxes=None,
                truth_labels=None,
                truth_instances=None):

        cfg  = self.cfg
        mode = self.mode

        # features
        features = data_parallel(self.feature_net, inputs)

        # rpn proposals -------------------------------------------
        self.rpn_logits_flat, self.rpn_deltas_flat = data_parallel(self.rpn_head, features)
        self.rpn_window    = make_rpn_windows(cfg, features)
        self.rpn_proposals = rpn_nms(cfg, mode, inputs,
                                     self.rpn_window,
                                     self.rpn_logits_flat,
                                     self.rpn_deltas_flat)

        if mode in ['train', 'valid']:
            self.rpn_labels, \
            self.rpn_label_assigns, \
            self.rpn_label_weights, \
            self.rpn_targets, \
            self.rpn_target_weights = \
                make_rpn_target(cfg, mode, inputs,
                                self.rpn_window,
                                truth_boxes,
                                truth_labels)

            self.rpn_proposals, \
            self.rcnn_labels, \
            self.rcnn_assigns, \
            self.rcnn_targets  = \
                make_rcnn_target(cfg, mode, inputs,
                                 self.rpn_proposals,
                                 truth_boxes,
                                 truth_labels)

        # rcnn proposals ------------------------------------------------
        self.rcnn_proposals = self.rpn_proposals
        if len(self.rpn_proposals) > 0:
            rcnn_crops = self.rcnn_crop(features, self.rpn_proposals)
            self.rcnn_logits, self.rcnn_deltas = data_parallel(self.rcnn_head, rcnn_crops)
            self.rcnn_proposals = rcnn_nms(cfg, mode, inputs,
                                           self.rpn_proposals,
                                           self.rcnn_logits,
                                           self.rcnn_deltas)

        if mode in ['train', 'valid']:
            self.rcnn_proposals, \
            self.mask_labels, \
            self.mask_assigns, \
            self.mask_instances,   = \
                make_mask_target(cfg, mode, inputs,
                                 self.rcnn_proposals,
                                 truth_boxes,
                                 truth_labels,
                                 truth_instances)

        #segmentation  -------------------------------------------
        self.detections = self.rcnn_proposals
        self.masks      = make_empty_masks(cfg, mode, inputs)

        if len(self.detections) > 0:
              mask_crops = self.mask_crop(features, self.detections)
              self.mask_logits = data_parallel(self.mask_head, mask_crops)
              self.masks = mask_nms(cfg, mode, inputs,
                                    self.detections,
                                    self.mask_logits) #<todo> better nms for mask

    def loss(self):

        self.rpn_cls_loss, self.rpn_reg_loss = \
           rpn_loss(self.rpn_logits_flat,
                    self.rpn_deltas_flat,
                    self.rpn_labels,
                    self.rpn_label_weights,
                    self.rpn_targets,
                    self.rpn_target_weights)

        self.rcnn_cls_loss, self.rcnn_reg_loss = \
            rcnn_loss(self.rcnn_logits,
                      self.rcnn_deltas,
                      self.rcnn_labels,
                      self.rcnn_targets)

        ## self.mask_cls_loss = Variable(torch.cuda.FloatTensor(1).zero_()).sum()
        self.mask_cls_loss  = \
            mask_loss(self.mask_logits,
                      self.mask_labels,
                      self.mask_instances)

        self.total_loss = self.rpn_cls_loss + \
                          self.rpn_reg_loss + \
                          self.rcnn_cls_loss + \
                          self.rcnn_reg_loss + \
                          self.mask_cls_loss

        return self.total_loss


    #<todo> freeze bn for imagenet pretrain
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
        #raise NotImplementedError




