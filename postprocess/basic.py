import numpy

from dataset.annotate import mask_to_inner_contour


def filter_small(multi_mask, threshold):
    num_masks = int(multi_mask.max())

    j=0
    for i in range(num_masks):
        thresh = (multi_mask==(i+1))

        area = thresh.sum()
        if area < threshold:
            multi_mask[thresh]=0
        else:
            multi_mask[thresh]=(j+1)
            j = j+1

    return multi_mask


def shrink_by_one(multi_mask):

    multi_mask_shrink = np.zeros(multi_mask.shape, np.int32)

    num = int(multi_mask.max())
    for m in range(num):
        mask = (multi_mask == m+1)
        contour = mask_to_inner_contour(mask)
        thresh = thresh & (~contour)
        multi_mask_shrink[thresh] = m+1

    return multi_mask_shrink