from net.lib.box.process import *


# faster-rcnn box encode/decode
from utility.func import np_sigmoid


def rcnn_encode(window, truth_box):
    return box_transform(window, truth_box)


def rcnn_decode(window, delta):
    return box_transform_inv(window, delta)


# this is in cpu: <todo> change to gpu ?
def rcnn_nms(cfg, mode, inputs, proposals, logits, deltas ):

    if mode in ['train',]:
        nms_pre_score_threshold = cfg.rcnn_train_nms_pre_score_threshold
        nms_overlap_threshold   = cfg.rcnn_train_nms_overlap_threshold
        nms_min_size = cfg.rcnn_train_nms_min_size

    elif mode in ['valid', 'test','eval']:
        nms_pre_score_threshold = cfg.rcnn_test_nms_pre_score_threshold
        nms_overlap_threshold   = cfg.rcnn_test_nms_overlap_threshold
        nms_min_size = cfg.rcnn_test_nms_min_size

        if mode in ['eval']:
            nms_pre_score_threshold = 0.05 # set low numbe r to make roc curve.

    else:
        raise ValueError('rcnn_nms(): invalid mode = %s?'%mode)


    batch_size, _, height, width = inputs.size() #original image width
    num_classes = cfg.num_classes

    probs     = np_sigmoid(logits.cpu().data.numpy())
    deltas    = deltas.cpu().data.numpy().reshape(-1, num_classes,4)
    proposals = proposals.cpu().data.numpy()

    # non-max suppression
    detections = []
    for b in range(batch_size):
        detection = [np.empty((0,7),np.float32),]

        index = np.where(proposals[:,0]==b)[0]
        if len(index)>0:
            prob  = probs [index]
            delta = deltas[index]
            proposal = proposals[index]

            for j in range(1,num_classes): #skip background
                idx = np.where(prob[:,j] > nms_pre_score_threshold)[0]
                if len(idx) > 0:
                    p = prob[idx, j].reshape(-1,1)
                    d = delta[idx, j]
                    box = rcnn_decode(proposal[idx,1:5], d)
                    box = clip_boxes(box, width, height)

                    keep = filter_boxes(box, min_size = nms_min_size)
                    num  = len(keep)
                    if num > 0:
                        box  = box[keep]
                        p    = p[keep]
                        keep = gpu_nms(np.hstack((box, p)), nms_overlap_threshold)

                        det = np.zeros((num,7),np.float32)
                        det[:,0  ] = b
                        det[:,1:5] = np.around(box,0)
                        det[:,5  ] = p[:,0]
                        det[:,6  ] = j
                        detection.append(det)

        detection = np.vstack(detection)

        ##limit to MAX_PER_IMAGE detections over all classes
        # if nms_max_per_image > 0:
        #     if len(detection) > nms_max_per_image:
        #         threshold = np.sort(detection[:,4])[-nms_max_per_image]
        #         keep = np.where(detection[:,4] >= threshold)[0]
        #         detection = detection[keep, :]

        detections.append(detection)

    detections = Variable(torch.from_numpy(np.vstack(detections))).cuda()
    return detections
