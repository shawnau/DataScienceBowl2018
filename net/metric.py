from net.lib.box.process import *


# https://www.kaggle.com/wcukierski/example-metric-implementation
def compute_precision(threshold, iou):
    matches = iou > threshold
    true_positives  = np.sum(matches, axis=1) == 1  # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Missed objects
    false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
    tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    return tp, fp, fn


def print_precision(precision):
    print('thresh   prec    TP    FP    FN')
    print('---------------------------------')
    for (t, p, tp, fp, fn) in precision:
        print('%0.2f     %0.2f   %3d   %3d   %3d'%(t, p, tp, fp, fn))


def compute_average_precision_for_mask(predict, truth, t_range=np.arange(0.5, 1.0, 0.05)):
    """
    :param predict: multi_masks for each instance
    :param truth: multi_masks for ground truth
    :param t_range: thresholds
    :return:
    """
    num_truth   = len(np.unique(truth))
    num_predict = len(np.unique(predict))

    # Compute intersection between all objects
    intersection = np.histogram2d(truth.flatten(), predict.flatten(), bins=(num_truth, num_predict))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(truth,   bins = num_truth  )[0]
    area_pred = np.histogram(predict, bins = num_predict)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred,  0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    precision = []
    average_precision = 0
    for t in t_range:
        tp, fp, fn = compute_precision(t, iou)
        p = tp / (tp + fp + fn)
        precision.append((t, p, tp, fp, fn))
        average_precision += p

    average_precision /= len(precision)
    return average_precision, precision


HIT =1
MISS=0
TP=1
FP=0
INVALID=-1


def compute_precision_for_box(box, truth_box, truth_label, threshold=[0.5]):

    num_truth_box = len(truth_box)
    num_box       = len(box      )

    overlap = cython_box_overlap(box, truth_box)
    argmax_overlap = np.argmax(overlap,0)
    max_overlap    = overlap[argmax_overlap,np.arange(num_truth_box)]

    invalid_truth_box     = truth_box[truth_label<0]
    invalid_valid_overlap = cython_box_overlap(box, invalid_truth_box)

    precision=[]
    recall=[]
    result=[]
    truth_result=[]

    for t in threshold:
        truth_r = np.ones(num_truth_box, np.int32)
        r       = np.ones(num_box, np.int32)

        # truth_result
        truth_r[...] = INVALID
        truth_r[(max_overlap <t) & (truth_label>0)] = MISS
        truth_r[(max_overlap>=t) & (truth_label>0)] = HIT

        # result
        r[...] = FP
        r[argmax_overlap[truth_r==HIT]] = TP

        index = np.where(r==FP)[0]
        if len(index)>0:
            index = index[np.where(invalid_valid_overlap[index]>t)[0]]
            r[index] = INVALID

        num_truth = (truth_r!=INVALID).sum()
        num_hit   = (truth_r==HIT    ).sum()
        num_miss  = (truth_r==MISS   ).sum()
        rec       = num_hit / num_truth

        num_tp = (r==TP).sum()
        num_fp = (r==FP).sum()
        prec = num_tp / max(num_tp + num_fp + num_miss, 1e-12)

        precision.append(prec)
        recall.append(rec)
        result.append(r)
        truth_result.append(truth_r)

        # if len(thresholds)==1:
        #     precisions = precisions[0]
        #     recalls = recalls[0]
        #     results = results[0]
        #     truth_results = truth_results[0]

    return  precision, recall, result, truth_result


def compute_hit_fp_for_box(proposals, truth_boxes, truth_labels):

    score =[]
    hit =[]
    fp  =[]
    num_miss=0

    for (proposal, truth_box, truth_label) in zip(proposals, truth_boxes, truth_labels):

        box   = proposal[:,1:5]
        precision, recall, result, truth_result = compute_precision_for_box(box, truth_box, truth_label, threshold=[0.5])
        result, truth_result = result[0], truth_result[0]

        s = proposal[:,5]
        N = len(result)
        h = np.zeros(N)
        f = np.zeros(N)
        h[np.where(result==HIT)]=1
        f[np.where(result==FP )]=1

        num_miss = (truth_result==MISS).sum()
        hit = hit + list(h)
        fp  = fp  + list(f)
        score = score  + list(s)

    return hit, fp, score, num_miss
