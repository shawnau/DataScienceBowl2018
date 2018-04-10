import os
import cv2
import numpy as np
import matplotlib.cm


# draw -----------------------------------
from matplotlib import pyplot as plt

from dataset.reader import instance_to_multi_mask
from net.layer.rpn import rpn_decode
from utility.metric import compute_precision_for_box, HIT, MISS, INVALID, TP, FP, compute_precision


def image_show(name, image, resize=1):
    H,W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize*W), round(resize*H))


def draw_shadow_text(img, text, pt,  fontScale, color, thickness, color1=None, thickness1=None):
    if color1 is None: color1=(0,0,0)
    if thickness1 is None: thickness1 = thickness+2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color,  thickness,  cv2.LINE_AA)


def draw_screen_rect(image, pt1, pt2, color, alpha=0.5):
    x1, y1 = pt1
    x2, y2 = pt2
    image[y1:y2,x1:x2,:] = (1-alpha)*image[y1:y2,x1:x2,:] + (alpha)*np.array(color, np.uint8)


def to_color(s, color=None):

    if type(color) in [str] or color is None:
        #https://matplotlib.org/xkcd/examples/color/colormaps_reference.html

        if color is None: color='cool'
        color = matplotlib.get_cmap(color)(s)
        b = int(255*color[2])
        g = int(255*color[1])
        r = int(255*color[0])

    elif type(color) in [list,tuple]:
        b = int(s*color[0])
        g = int(s*color[1])
        r = int(s*color[2])

    return b,g,r


def draw_multi_proposal_metric(cfg, image, proposal, truth_box, truth_label,
                          color0=[0,255,255], color1=[255,0,255], color2=[255,255,0],thickness=1):

    H,W = image.shape[:2]
    image_truth    = image.copy()  #yellow
    image_proposal = image.copy()  #pink
    image_hit      = image.copy()  #cyan
    image_miss     = image.copy()  #yellow
    image_fp       = image.copy()  #pink
    image_invalid  = image.copy()  #white
    precision = 0

    if len(proposal)>0 and len(truth_box)>0:

        thresholds=[0.5,]

        box = proposal[:,1:5]
        precisions, recalls, results, truth_results = \
            compute_precision_for_box(box, truth_box, truth_label, thresholds)

        #for precision, recall, result, truth_result, threshold in zip(precisions, recalls, results, truth_results, thresholds):

        if 1:
            precision, recall, result, truth_result, threshold = \
                precisions[0], recalls[0], results[0], truth_results[0], thresholds[0]


            for i,b in enumerate(truth_box):
                x0,y0,x1,y1 = b.astype(np.int32)

                if truth_result[i]==HIT:
                    cv2.rectangle(image_truth,(x0,y0), (x1,y1), color0, thickness)
                    draw_screen_rect(image_hit,(x0,y0), (x1,y1), color2, 0.25)

                if truth_result[i]==MISS:
                    cv2.rectangle(image_truth,(x0,y0), (x1,y1), color0, thickness)
                    cv2.rectangle(image_miss,(x0,y0), (x1,y1), color0, thickness)

                if truth_result[i]==INVALID:
                    draw_screen_rect(image_invalid,(x0,y0), (x1,y1), (255,255,255), 0.5)

            for i,b in enumerate(box):
                x0,y0,x1,y1 = b.astype(np.int32)
                cv2.rectangle(image_proposal,(x0,y0), (x1,y1), color1, thickness)

                if result[i]==TP:
                    cv2.rectangle(image_hit,(x0,y0), (x1,y1), color2, thickness)

                if result[i]==FP:
                    cv2.rectangle(image_fp,(x0,y0), (x1,y1), color1, thickness) #255,0,255

                if result[i]==INVALID:
                    cv2.rectangle(image_invalid,(x0,y0), (x1,y1), (255,255,255), thickness)

    draw_shadow_text(image_truth, 'truth',  (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(image_proposal,'proposal', (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(image_hit, 'hit',  (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(image_miss,'miss', (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(image_fp,  'fp',   (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(image_invalid, 'n.a.', (5,15),0.5, (255,255,255), 1)

    all = np.hstack([image_truth,image_proposal,image_hit,image_miss,image_fp,image_invalid])
    draw_shadow_text(all,'%0.2f prec@0.5'%precision, (5,H-15),0.5, (255,255,255), 1)
    return all


def draw_mask_metric(cfg, image, mask, truth_box, truth_label, truth_instance):

    H,W = image.shape[:2]
    overlay_truth   = np.zeros((H,W,3),np.uint8)  #yellow
    overlay_mask    = np.zeros((H,W,3),np.uint8)  #pink
    overlay_error   = np.zeros((H,W,3),np.uint8)
    overlay_metric  = np.zeros((H,W,3),np.uint8)
    overlay_contour = image.copy()
    average_overlap   = 0
    average_precision = 0
    precision_50 = 0
    precision_70 = 0

    if len(truth_box)>0:

        #pixel error: fp and miss
        truth_mask = instance_to_multi_mask(truth_instance)
        truth      = truth_mask!=0

        predict = mask!=0
        hit  = truth & predict
        miss = truth & (~predict)
        fp   = (~truth) & predict

        overlay_error[hit ]=[128,128,0]
        overlay_error[fp  ]=[255,0,255]
        overlay_error[miss]=[0,255,255]

        # truth and mask ---
        overlay_mask [predict]=[64,0,64]
        overlay_truth[truth  ]=[0,64,64]
        overlay_mask  = multi_mask_to_contour_overlay(mask,overlay_mask,[255,0,255])
        overlay_truth = multi_mask_to_contour_overlay(truth_mask,overlay_truth,[0,255,255])

        overlay_contour = multi_mask_to_contour_overlay(mask,overlay_contour,[255,0,255])

        # metric -----
        predict = mask
        truth   = instance_to_multi_mask(truth_instance)

        num_truth   = len(np.unique(truth  ))-1
        num_predict = len(np.unique(predict))-1

        if num_predict!=0:
            intersection = np.histogram2d(truth.flatten(), predict.flatten(), bins=(num_truth+1, num_predict+1))[0]

            # Compute areas (needed for finding the union between all objects)
            area_true = np.histogram(truth,   bins = num_truth  +1)[0]
            area_pred = np.histogram(predict, bins = num_predict+1)[0]
            area_true = np.expand_dims(area_true, -1)
            area_pred = np.expand_dims(area_pred,  0)
            union = area_true + area_pred - intersection
            intersection = intersection[1:,1:]   # Exclude background from the analysis
            union = union[1:,1:]
            union[union == 0] = 1e-9
            iou = intersection / union # Compute the intersection over union

            precision = {}
            average_precision = 0
            thresholds = np.arange(0.5, 1.0, 0.05)
            for t in thresholds:
                tp, fp, fn = compute_precision(t, iou)
                prec = tp / (tp + fp + fn)
                precision[round(t,2) ]=prec
                average_precision += prec
            average_precision /= len(thresholds)
            precision_50 = precision[0.50]
            precision_70 = precision[0.70]


            #iou = num_truth, num_predict
            overlap = np.max(iou,1)
            assign  = np.argmax(iou,1)
            #print(overlap)

            for t in range(num_truth):
                s = overlap[t]
                if s>0.5:
                    color = to_color(max(0.0,(overlap[t]-0.5)/0.5), [255,255,255])
                else:
                    color = [0,0,255] #to_color(max(0.0,(0.5-overlap[t])/0.5), [0,255,0])
                overlay_metric[truth_instance[t]>0]=color

            overlay_metric = multi_mask_to_contour_overlay(mask,overlay_metric,[255,0,255])
            average_overlap = overlap.mean()

    draw_shadow_text(overlay_truth,   'truth',  (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(overlay_mask,    'mask',   (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(overlay_error,   'error',  (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(overlay_metric,  '%0.2f iou '%average_overlap,    (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(overlay_contour, 'contour', (5,15),0.5, (255,255,255), 1)

    all = np.hstack((overlay_truth, overlay_mask, overlay_error, overlay_metric, overlay_contour, image, ))
    draw_shadow_text(all,'%0.2f prec@0.5'%(precision_50), (5,H-45),0.5, (255,255,255), 1)
    draw_shadow_text(all,'%0.2f prec@0.7'%(precision_70), (5,H-30),0.5, (255,255,255), 1)
    draw_shadow_text(all,'%0.2f prec'%average_precision,  (5,H-15),0.5, (255,255,0), 1)
    #image_show('all mask : image, truth, predict, error, metric',all,1)

    return all


# rpn-------------------------------------------
def unflat_to_c3(data, num_bases, scales, H, W):
    dtype =data.dtype
    data=data.astype(np.float32)

    datas=[]

    num_scales = len(scales)
    start = 0
    for l in range(num_scales):
        h,w = int(H/scales[l]), int(W/scales[l])
        c = num_bases[l]

        size = h*w*c
        d = data[start:start+size].reshape(h,w,c)
        start = start+size

        if c==1:
            d = d*np.array([1,1,1])

        elif c==3:
            pass

        elif c==4:
            d = np.dstack((
                (d[:,:,0]+d[:,:,1])/2,
                 d[:,:,2],
                 d[:,:,3],
            ))

        elif c==5:
            d = np.dstack((
                 d[:,:,0],
                (d[:,:,1]+d[:,:,2])/2,
                (d[:,:,3]+d[:,:,4])/2,
            ))


        elif c==6:
            d = np.dstack((
                (d[:,:,0]+d[:,:,1])/2,
                (d[:,:,2]+d[:,:,3])/2,
                (d[:,:,4]+d[:,:,5])/2,
            ))

        else:
            raise NotImplementedError

        d = d.astype(dtype)
        datas.append(d)

    return datas


def draw_multi_rpn_prob(cfg, image, rpn_prob_flat):

    H,W = image.shape[:2]
    scales = cfg.rpn_scales
    num_scales = len(cfg.rpn_scales)
    num_bases  = [len(b) for b in cfg.rpn_base_apsect_ratios]

    rpn_prob = (rpn_prob_flat[:,1]*255).astype(np.uint8)
    rpn_prob = unflat_to_c3(rpn_prob, num_bases, scales, H, W)

    ## -pyramid -
    pyramid=[]
    for l in range(num_scales):
        pyramid.append(cv2.resize(image, None, fx=1/scales[l],fy=1/scales[l]))

    all = []
    for l in range(num_scales):
        a = np.vstack((
            pyramid[l],
            rpn_prob[l],
        ))

        all.append(
            cv2.resize(a, None, fx=scales[l],fy=scales[l],interpolation=cv2.INTER_NEAREST)
        )

    all = np.hstack(all).astype(np.uint8)
    draw_shadow_text(all,'rpn-prob', (5,15),0.5, (255,255,255), 1)

    return all


def draw_multi_rpn_delta(cfg, image, rpn_prob_flat, rpn_delta_flat, window, color=[255,255,255]):

    threshold = cfg.rpn_test_nms_pre_score_threshold

    image_box   = image.copy()
    image_point = image.copy()
    index = np.where(rpn_prob_flat>threshold)[0]
    for i in index:
        l = np.argmax(rpn_prob_flat[i])
        if l==0: continue

        w = window[i]
        t = rpn_delta_flat[i,l]
        b = rpn_decode(w.reshape(1,4), t.reshape(1,4))
        x0,y0,x1,y1 = b.reshape(-1).astype(np.int32)

        wx0,wy0,wx1,wy1 = window[i].astype(np.int32)

        cx = (wx0+wx1)//2
        cy = (wy0+wy1)//2

        #cv2.rectangle(image,(w[0], w[1]), (w[2], w[3]), (255,255,255), 1)
        cv2.rectangle(image_box,(x0,y0), (x1,y1), color, 1)
        image_point[cy,cx]= color


    draw_shadow_text(image_box,   'rpn-box', (5,15),0.5, (255,255,255), 1)
    draw_shadow_text(image_point, 'point',   (5,15),0.5, (255,255,255), 1)
    all = np.hstack([image_box,image_point])
    return all


# pink
def draw_multi_rpn_proposal(cfg, image, proposal):

    image = image.copy()
    for p in proposal:
        x0,y0,x1,y1 = p[1:5].astype(np.int32)
        score = p[5]
        color = to_color(score, [255,0,255])
        cv2.rectangle(image, (x0,y0), (x1,y1), color, 1)

    return image


# yellow
def draw_truth_box(cfg, image, truth_box, truth_label):

    image = image.copy()
    if len(truth_box)>0:
        for b,l in zip(truth_box, truth_label):
            x0,y0,x1,y1 = b.astype(np.int32)
            if l <=0 : continue
            cv2.rectangle(image, (x0,y0), (x1,y1), [0,255,255], 1)

    return image


def normalize(data):
    data = (data-data.min())/(data.max()-data.min())
    return data


def draw_rpn_target_truth_box(image, truth_box, truth_label):
    image = image.copy()
    for b,l in zip(truth_box, truth_label):
        x0,y0,x1,y1 = np.round(b).astype(np.int32)
        if l >0:
            cv2.rectangle(image,(x0,y0), (x1,y1), (0,0,255), 1)
        else:
            cv2.rectangle(image,(x0,y0), (x1,y1), (255,255,255), 1)
            # draw_dotted_rect(image,(x0,y0), (x1,y1), (0,0,255), )

    return image


def draw_rpn_target_label(cfg, image, window, label, label_assign, label_weight):

    H,W = image.shape[:2]
    scales = cfg.rpn_scales
    num_scales = len(cfg.rpn_scales)
    num_bases  = [len(b) for b in cfg.rpn_base_apsect_ratios]

    label         = (normalize(label) * 255).astype(np.uint8)
    label_assign  = (normalize(label_assign) * 255).astype(np.uint8)
    label_weight  = (normalize(label_weight) * 255).astype(np.uint8)
    labels        = unflat_to_c3(label, num_bases, scales, H, W)
    label_assigns = unflat_to_c3(label_assign, num_bases, scales, H, W)
    label_weights = unflat_to_c3(label_weight, num_bases, scales, H, W)

    all_label = []
    for l in range(num_scales):
        pyramid = cv2.resize(image, None, fx=1/scales[l],fy=1/scales[l])

        a = np.vstack((
            pyramid,
            labels[l],
            label_assigns[l],
            label_weights[l],
        ))

        a = cv2.resize(a, None, fx=scales[l],fy=scales[l],interpolation=cv2.INTER_NEAREST)
        all_label.append(a)

    all_label = np.hstack(all_label)
    return all_label


def draw_rpn_target_target(cfg, image, window, target, target_weight):

    H,W = image.shape[:2]
    scales = cfg.rpn_scales
    num_scales = len(cfg.rpn_scales)
    num_bases  = [len(b) for b in cfg.rpn_base_apsect_ratios]

    target_weight  = (normalize(target_weight) * 255).astype(np.uint8)
    target_weights = unflat_to_c3(target_weight, num_bases, scales, H, W)

    all=[]
    for l in range(num_scales):
        pyramid = cv2.resize(image, None, fx=1/scales[l],fy=1/scales[l])

        a = np.vstack((
            pyramid,
            target_weights[l],
        ))

        a = cv2.resize(a, None, fx=scales[l],fy=scales[l],interpolation=cv2.INTER_NEAREST)
        all.append(a)

    all = np.hstack(all)
    return all


def draw_rpn_target_target1(cfg, image, window, target, target_weight, is_before=False, is_after=True):

    image = image.copy()

    index = np.where(target_weight!=0)[0]
    for i in index:
        w = window[i]
        t = target[i]
        b = rpn_decode(w.reshape(1,4), t.reshape(1,4))
        b = b.reshape(-1).astype(np.int32)

        if is_before:
            cv2.rectangle(image,(w[0], w[1]), (w[2], w[3]), (0,0,255), 1)
            #cv2.circle(image,((w[0]+w[2])//2, (w[1]+w[3])//2),2, (0,0,255), -1, cv2.LINE_AA)

        if is_after:
            cv2.rectangle(image,(b[0], b[1]), (b[2], b[3]), (0,255,255), 1)

    return image


def multi_mask_to_color_overlay(multi_mask, image=None, color=None):

    height,width = multi_mask.shape[:2]
    overlay = np.zeros((height,width,3),np.uint8) if image is None else image.copy()
    num_masks = int(multi_mask.max())
    if num_masks==0: return overlay

    if type(color) in [str] or color is None:
        #https://matplotlib.org/xkcd/examples/color/colormaps_reference.html

        if color is None: color='summer'  #'cool' #'brg'
        color = plt.get_cmap(color)(np.arange(0,1,1/num_masks))
        color = np.array(color[:,:3])*255
        color = np.fliplr(color)
        #np.random.shuffle(color)

    elif type(color) in [list,tuple]:
        color = [ color for i in range(num_masks) ]

    for i in range(num_masks):
        mask = multi_mask==i+1
        overlay[mask]=color[i]
        #overlay = instance[:,:,np.newaxis]*np.array( color[i] ) +  (1-instance[:,:,np.newaxis])*overlay

    return overlay


def multi_mask_to_contour_overlay(multi_mask, image=None, color=[255,255,255]):

    height,width = multi_mask.shape[:2]
    overlay = np.zeros((height,width,3),np.uint8) if image is None else image.copy()
    num_masks = int(multi_mask.max())
    if num_masks==0: return overlay

    for i in range(num_masks):
        mask = multi_mask==i+1
        contour = mask_to_inner_contour(mask)
        overlay[contour]=color

    return overlay


def mask_to_outer_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = (~mask) & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour


def mask_to_inner_contour(mask):
    pad = np.lib.pad(mask, ((1, 1), (1, 1)), 'reflect')
    contour = mask & (
            (pad[1:-1,1:-1] != pad[:-2,1:-1]) \
          | (pad[1:-1,1:-1] != pad[2:,1:-1])  \
          | (pad[1:-1,1:-1] != pad[1:-1,:-2]) \
          | (pad[1:-1,1:-1] != pad[1:-1,2:])
    )
    return contour


def draw_predict_mask(threshold, image, mask, detection):

    H,W = image.shape[:2]

    box_overlay     = image.copy()
    contour_overlay = image.copy()
    color_overlay   = np.zeros((H,W,3),np.uint8)

    if len(detection)>0:
        colors = matplotlib.cm.get_cmap('hot')
        multi_mask = mask

        for i,d in enumerate(detection):
            mask = (multi_mask==i+1)
            contour = mask_to_inner_contour(mask)
            score   = d[5]
            s       = max(0,(score-threshold)/(1-threshold))

            color = colors(s)   #(0,1,0)  #
            color = int(color[2]*255),int(color[1]*255),int(color[0]*255)
            contour_overlay[contour] = color
            color_overlay[mask]      = color
            color_overlay[contour]   = [255,255,255]

            x0,y0,x1,y1 = d[1:5].astype(np.int32)
            cv2.rectangle(box_overlay, (x0,y0), (x1,y1), color, 1)

    all = np.hstack([image, color_overlay, contour_overlay, box_overlay])
    draw_shadow_text(all, 'threshold=%0.3f'%threshold,  (5,15),0.5, (255,255,255), 1)

    return all