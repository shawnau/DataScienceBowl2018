from common import *
import itertools

from net.lib.box.process import *


# make base size for anchor boxes
def make_bases(base_size, base_apsect_ratios):
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
        rw = round(w/2)
        rh = round(h/2)
        base =(-rw, -rh, rw, rh, )
        bases.append(base)

    bases = np.array(bases,np.float32)
    return bases


def make_windows(f, scale, bases):
    """
    make anchor boxes on every pixel of the feature map
    :param:
        f: feature of size (B, C, H, W)
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
    """
    windows  = []
    _, _, H, W = f.size()
    for y, x in itertools.product(range(H),range(W)):
        cx = x*scale
        cy = y*scale
        for b in bases:
            x0,y0,x1,y1 = b
            x0 += cx
            y0 += cy
            x1 += cx
            y1 += cy
            windows.append([x0,y0,x1,y1])

    windows = np.array(windows, np.float32)
    return windows


def make_rpn_windows(cfg, fs):
    """
    create region proposals from all 4 feature maps to original image
    """
    rpn_windows = []
    num_scales = len(cfg.rpn_scales)
    for l in range(num_scales):
        bases   = make_bases(cfg.rpn_base_sizes[l], cfg.rpn_base_apsect_ratios[l])
        windows = make_windows(fs[l], cfg.rpn_scales[l], bases)
        rpn_windows.append(windows)

    rpn_windows = np.vstack(rpn_windows)

    return rpn_windows


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


def rpn_nms(cfg, mode, inputs, window, logits_flat, deltas_flat):
    """
    This function:
    1. Do non-maximum suppression on given window and logistic score
    2. filter small proposals
    :param cfg: configure
    :param mode: mode. e.g. 'train', 'test', 'eval'
    :param inputs: input images
    :param window: all regional proposals, list of coords, e.g.
               [[x0, y0, x1, y1], ...]
    :param logits_flat: score for each windows. e.g.
               [[0.7, 0.3], ...] for foreground, background
    :param deltas_flat: bbox regression offset for each bbox. e.g.
               [[t1, t2, t3, t4], ...]
    :return: list of proposals. e.g.
        [i, x0, y0, x1, y1, score, label] (scale_level)
        proposal[0]:   idx in the batch
        proposal[1:5]: bbox
        proposal[5]:   probability of foreground (background skipped)
        proposal[6]:   class label

    """
    if mode in ['train',]:
        nms_pre_score_threshold = cfg.rpn_train_nms_pre_score_threshold
        nms_overlap_threshold   = cfg.rpn_train_nms_overlap_threshold
        nms_min_size            = cfg.rpn_train_nms_min_size

    elif mode in ['eval', 'valid', 'test',]:
        nms_pre_score_threshold = cfg.rpn_test_nms_pre_score_threshold
        nms_overlap_threshold   = cfg.rpn_test_nms_overlap_threshold
        nms_min_size            = cfg.rpn_test_nms_min_size

        if mode in ['eval']:
            nms_pre_score_threshold = 0.05  # set low numbe r to make roc curve.
    else:
        raise ValueError('rpn_nms(): invalid mode = %s?'%mode)

    logits = logits_flat.data.cpu().numpy()
    deltas = deltas_flat.data.cpu().numpy()
    batch_size,_,height, width = inputs.size()
    num_classes = cfg.num_classes

    proposals = []
    for b in range(batch_size):
        proposal = [np.empty((0,7),np.float32),]

        ps = np_softmax(logits[b])
        ds = deltas[b]

        for c in range(1, num_classes):  # skip background  # num_classes
            index = np.where(ps[:,c] > nms_pre_score_threshold)[0]
            if len(index) > 0:
                p = ps[index, c].reshape(-1,1)
                d = ds[index, c]
                w = window[index]
                box = rpn_decode(w, d)
                box = clip_boxes(box, width, height)  # take care of borders

                keep = filter_boxes(box, min_size=nms_min_size)  # get rid of small boxes
                if len(keep)>0:
                    box  = box[keep]
                    p    = p  [keep]
                    keep = gpu_nms(np.hstack((box, p)), nms_overlap_threshold)

                    prop = np.zeros((len(keep),7),np.float32)
                    prop[:,0  ] = b
                    prop[:,1:5] = np.around(box[keep],0)
                    prop[:,5  ] = p[keep,0]
                    prop[:,6  ] = c
                    proposal.append(prop)

        proposal = np.vstack(proposal)
        proposals.append(proposal)

    proposals = Variable(torch.from_numpy(np.vstack(proposals))).cuda()
    return proposals
