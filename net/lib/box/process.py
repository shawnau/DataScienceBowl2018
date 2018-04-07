import numpy as np
import torch


def torch_clip_proposals(proposals, index, width, height):
    proposals = torch.stack((
         proposals[index,0],
         proposals[index,1].clamp(0, width  - 1),
         proposals[index,2].clamp(0, height - 1),
         proposals[index,3].clamp(0, width  - 1),
         proposals[index,4].clamp(0, height - 1),
         proposals[index,5],
         proposals[index,6],
    ), 1)
    return proposals


# python
def clip_boxes(boxes, width, height):
    """
    Clip process to image boundaries.
    Used in rpn_nms and rcnn_nms
    :param boxes: proposals
    :param width: input's_train width
    :param height: input's_train height
    :return: cropped proposals to fit input's_train border
    """
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width  - 1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height - 1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width  - 1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height - 1)
    return boxes


def bbox_encode(bboxes, targets):
    """
    :param bboxes: bboxes
    :param targets: target ground truth boxes
    :return: deltas
    """

    bw = bboxes[:, 2] - bboxes[:, 0] + 1.0
    bh = bboxes[:, 3] - bboxes[:, 1] + 1.0
    bx = bboxes[:, 0] + 0.5 * bw
    by = bboxes[:, 1] + 0.5 * bh

    tw = targets[:, 2] - targets[:, 0] + 1.0
    th = targets[:, 3] - targets[:, 1] + 1.0
    tx = targets[:, 0] + 0.5 * tw
    ty = targets[:, 1] + 0.5 * th

    dx = (tx - bx) / bw
    dy = (ty - by) / bh
    dw = np.log(tw / bw)
    dh = np.log(th / bh)

    deltas = np.vstack((dx, dy, dw, dh)).transpose()
    return deltas


def bbox_decode(bboxes, deltas):
    """
    :param bboxes: bounding boxes
    :param deltas: bbox regression deltas
    :return: refined bboxes
    """
    num   = len(bboxes)
    predictions = np.zeros((num,4), dtype=np.float32)
    # if num == 0: return predictions  #not possible?

    bw = bboxes[:, 2] - bboxes[:, 0] + 1.0
    bh = bboxes[:, 3] - bboxes[:, 1] + 1.0
    bx = bboxes[:, 0] + 0.5 * bw
    by = bboxes[:, 1] + 0.5 * bh
    bw = bw[:, np.newaxis]
    bh = bh[:, np.newaxis]
    bx = bx[:, np.newaxis]
    by = by[:, np.newaxis]

    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
    
    x = dx * bw + bx
    y = dy * bh + by
    dw = np.clip(dw, -10, 10)
    dh = np.clip(dh, -10, 10)
    w = np.exp(dw) * bw
    h = np.exp(dh) * bh

    predictions[:, 0::4] = x - 0.5 * w  # x0,y0,x1,y1
    predictions[:, 1::4] = y - 0.5 * h
    predictions[:, 2::4] = x + 0.5 * w
    predictions[:, 3::4] = y + 0.5 * h

    return predictions


def filter_boxes(boxes, min_size):
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def is_small_box_at_boundary(box,W,H, min_size):
    x0,y0,x1,y1 = box
    w = (x1-x0)+1
    h = (y1-y0)+1
    aspect = max(w,h)/min(w,h)
    area = w*h
    return ((x0==0 or x1==W-1) and (w<min_size)) or \
           ((y0==0 or y1==H-1) and (h<min_size))


def is_small_box(box, min_size):
    x0,y0,x1,y1 = box
    w = (x1-x0)+1
    h = (y1-y0)+1
    aspect = max(w,h)/min(w,h)
    area = w*h
    return (w <min_size or h<min_size)


def is_big_box(box, max_size):
    x0,y0,x1,y1 = box
    w = (x1-x0)+1
    h = (y1-y0)+1
    aspect = max(w,h)/min(w,h)
    area = w*h
    return (w>max_size or h>max_size)
