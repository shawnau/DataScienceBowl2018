from common import *
from utility.draw import *
import itertools

if __name__ == '__main__':
    from rpn_multi_nms     import *

else:
    from .rpn_multi_nms    import *


# cpu version
def make_one_rpn_target(cfg, mode, input, window, truth_box, truth_label):
    """
    given bboxes, return positive/negative samples (balanced)
    :return:
        label: 1 for pos, 0 for neg for now
        label_assign:
        label_weight: bboxes' indices to consider as pos/neg is 1, other is 0 (negleted)
        target: bboxes' offsets
        target_weight: bboxes' indices to consider as pos/neg is 1, other is 0 (negleted)
    """
    num_window = len(window)
    label         = np.zeros((num_window, ), np.float32)
    label_assign  = np.zeros((num_window, ), np.int32)
    label_weight  = np.ones ((num_window, ), np.float32)
    target        = np.zeros((num_window,4), np.float32)
    target_weight = np.zeros((num_window, ), np.float32)


    num_truth_box = len(truth_box)
    if num_truth_box != 0:

        _,height,width = input.size()

        # "SSD: Single Shot MultiBox Detector" - Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy
        #   -- see Table.3
        #
        # allowed_border=0
        # invalid_index = (
        #     (window[:,0] < allowed_border)    | \
        #     (window[:,1] < allowed_border)    | \
        #     (window[:,2] > width-1  - allowed_border) | \
        #     (window[:,3] > height-1 - allowed_border))
        # label_weight [invalid_index]=0
        # target_weight[invalid_index]=0

        # classification
        # bg
        overlap        = cython_box_overlap(window, truth_box)
        argmax_overlap = np.argmax(overlap,1)
        max_overlap    = overlap[np.arange(num_window),argmax_overlap]

        bg_index = max_overlap < cfg.rpn_train_bg_thresh_high
        label[bg_index] = 0
        label_weight[bg_index] = 1

        # fg
        fg_index = max_overlap >= cfg.rpn_train_fg_thresh_low
        label[fg_index] = 1  #<todo> extend to multi-class ... need to modify regression below too
        label_weight[fg_index] = 1
        label_assign[...] = argmax_overlap

        # fg: for each truth, window with highest overlap, include multiple maxs
        argmax_overlap = np.argmax(overlap,0)
        max_overlap    = overlap[argmax_overlap,np.arange(num_truth_box)]
        argmax_overlap, a = np.where(overlap==max_overlap)

        fg_index = argmax_overlap
        label       [fg_index] = 1
        label_weight[fg_index] = 1
        label_assign[fg_index] = a

        # regression
        fg_index         = np.where(label!=0)
        target_window    = window[fg_index]
        target_truth_box = truth_box[label_assign[fg_index]]
        target[fg_index] = rpn_encode(target_window, target_truth_box)
        target_weight[fg_index] = 1

        # don't care
        invalid_truth_label = np.where(truth_label<0)[0]
        invalid_index = np.isin(label_assign, invalid_truth_label) & (label!=0)
        label_weight [invalid_index]=0
        target_weight[invalid_index]=0

        # class balancing
        if 1:
            fg_index = np.where( (label_weight!=0) & (label!=0) )[0]
            bg_index = np.where( (label_weight!=0) & (label==0) )[0]

            num_fg = len(fg_index)
            num_bg = len(bg_index)
            label_weight[fg_index] = 1
            label_weight[bg_index] = num_fg/num_bg

            #scale balancing
            num_scales = len(cfg.rpn_scales)
            num_bases  = [len(b) for b in cfg.rpn_base_apsect_ratios]
            start = 0
            for l in range(num_scales):
                h,w = int(height//2**l),int(width//2**l)
                end = start+ h*w*num_bases[l]
                ## label_weight[start:end] *= (2**l)**2
                start=end

        #task balancing
        target_weight[fg_index] = label_weight[fg_index]

    # save
    label          = Variable(torch.from_numpy(label)).cuda()
    label_assign   = Variable(torch.from_numpy(label_assign)).cuda()
    label_weight   = Variable(torch.from_numpy(label_weight)).cuda()
    target         = Variable(torch.from_numpy(target)).cuda()
    target_weight  = Variable(torch.from_numpy(target_weight)).cuda()
    return  label, label_assign, label_weight, target, target_weight


def make_rpn_target(cfg, mode, inputs, window, truth_boxes, truth_labels):

    rpn_labels = []
    rpn_label_assigns = []
    rpn_label_weights = []
    rpn_targets = []
    rpn_targets_weights = []

    batch_size = len(truth_boxes)
    for b in range(batch_size):
        input = inputs[b]
        truth_box   = truth_boxes[b]
        truth_label = truth_labels[b]

        rpn_label, rpn_label_assign, rpn_label_weight, rpn_target, rpn_targets_weight = \
            make_one_rpn_target(cfg, mode, input, window, truth_box, truth_label)

        rpn_labels.append(rpn_label.view(1,-1))
        rpn_label_assigns.append(rpn_label_assign.view(1,-1))
        rpn_label_weights.append(rpn_label_weight.view(1,-1))
        rpn_targets.append(rpn_target.view(1,-1,4))
        rpn_targets_weights.append(rpn_targets_weight.view(1,-1))


    rpn_labels          = torch.cat(rpn_labels, 0)
    rpn_label_assigns   = torch.cat(rpn_label_assigns, 0)
    rpn_label_weights   = torch.cat(rpn_label_weights, 0)
    rpn_targets         = torch.cat(rpn_targets, 0)
    rpn_targets_weights = torch.cat(rpn_targets_weights, 0)

    return rpn_labels, rpn_label_assigns, rpn_label_weights, rpn_targets, rpn_targets_weights
