from common import *


def prob_delta_to_candidates( prob, delta, heads, threshold=0.4):
    num_heads =len(heads)

    candidates = []
    for h in range(num_heads):
        index = np.where(prob[h]>threshold)
        if len(index[0])!=0:
            y,x      = index[0],index[1]
            dx       = delta[h,0][index]
            dy       = delta[h,1][index]
            dminor_r = delta[h,2][index]
            dmajor_r = delta[h,3][index]
            sin      = delta[h,4][index]
            cos      = delta[h,5][index]

            r  = heads[h]
            rr = max(1,int(0.33*2**r))

            cx      = (dx*rr+x)
            cy      = (dy*rr+y)
            minor_r = (2**(dminor_r+r))
            major_r = (2**(dmajor_r+r))
            angle   = (np.arctan2(sin,cos)/np.pi*180)
            score   = prob[h][index]

            c = np.vstack((cx,cy,minor_r,major_r,angle,score))
            candidates.append(c.T)

    candidates = np.concatenate(candidates)
    return candidates


# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppress(candidates, min_distance_threshold=0.25):

    if len(candidates) == 0:
        return []

    cx     =  candidates[:,0]
    cy     =  candidates[:,1]
    log2r  = np.log2 ( (candidates[:,2] + candidates[:,3])/2)
    score  =  candidates[:,5]
    indices = np.argsort(-score)  #decreasing

    select = []
    while len(indices) > 0:
        i = indices[0]
        select.append(i)

        # last added
        distances = ((cx[indices] - cx[i])**2 \
                +    (cy[indices] - cy[i])**2 \
                +    (log2r[indices] - log2r[i])**2)**0.5


        # delete all  candidates that is nearby
        #remove = np.where(distances < min_distance_threshold*log2r[i])
        remove = np.where(distances < 6)  #<todo> a good thresholding mnethods
        indices = np.delete( indices, remove )

    nms = candidates[select]
    return nms


def nms_to_original_size( nms, image, original_image):
    h,w = image.shape[:2]
    original_h,original_w = original_image.shape[:2]

    scale_x = original_w/w
    scale_y = original_h/h
    nms[:,0] *= scale_x
    nms[:,1] *= scale_y
    nms[:,2] *= scale_x
    nms[:,3] *= scale_y

    return nms


def nms_to_label( nms, image ):

    H,W = image.shape[:2]
    label = np.zeros((H,W,3), np.uint8)

    num_mns = len(nms)
    assert(num_mns<256*256)

    nms = nms.astype(np.int32)
    for i in range(num_mns):
        candidate = nms[i]
        b,g,r = 0, (i+1)//256, (i+1)%256
        cx,cy, minor_r,major_r, angle, score = candidate
        cv2.ellipse(label, (cx,cy), (minor_r,major_r), angle, 0, 360, (b,g,r), -1)

    label = label.astype(np.int32)
    label = label[:,:,1]*256 + label[:,:,2]

    return label


# # overlaps
# #https://stackoverflow.com/questions/25349178/calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation
# #http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
#
# def one_box_overlap(box1,box2):
#
#     ''' Calculate the Intersection over Union (IoU) of two bounding boxes.
#
#     '''
#     #box=x0,y0,x1,y1
#     assert box1[0] < box1[2]
#     assert box1[1] < box1[3]
#     assert box2[0] < box2[2]
#     assert box2[1] < box2[3]
#
#     # determine the coordinates of the intersection rectangle
#     x_left   = max(box1[0], box2[0])
#     y_top    = max(box1[1], box2[1])
#     x_right  = min(box1[2], box2[2])
#     y_bottom = min(box1[3], box2[3])
#
#     if x_right < x_left or y_bottom < y_top:
#         return 0.0
#
#     # The intersection of two axis-aligned bounding boxes is always an
#     # axis-aligned bounding box
#     intersection_area = (x_right - x_left+1) * (y_bottom - y_top+1)
#
#     # compute the area of both AABBs
#     area1 = (box1[2] - box1[0]+1) * (box1[3] - box1[1]+1)
#     area2 = (box2[2] - box2[0]+1) * (box2[3] - box2[1]+1)
#
#     # compute the intersection over union by taking the intersection
#     # area and dividing it by the sum of prediction + ground-truth
#     # areas - the interesection area
#     iou = intersection_area / float(area1 + area2 - intersection_area)
#     assert iou >= 0.0
#     assert iou <= 1.0
#     return iou