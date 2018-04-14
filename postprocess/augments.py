from common import *
from utility.draw import *

from dataset.reader import *
from dataset.transform import *
from net.layer.mask import *

##--------------------------------------------------------------
AUG_FACTOR = 16

def draw_proposal(image,proposal):
    image = image.copy()
    for p in proposal:
        x0,y0,x1,y1 = p[1:5].astype(np.int32)
        cv2.rectangle(image,(x0,y0),(x1,y1),(255,255,255),1)
    return image


def scale_to_factor(image, scale_x, scale_y, factor=16):
    height,width = image.shape[:2]
    h = math.ceil(scale_y*height/factor)*factor
    w = math.ceil(scale_x*width/factor)*factor
    image = cv2.resize(image,(w,h))
    return image


def undo_flip_transpose(height, width, image=None, proposal=None, mask=None, instance=None, type=0):
    undo=[
        (0, 1,0,  0,1,),
        (3, 0,1,  1,0,),
        (2, 1,0,  0,1,),
        (1, 0,1,  1,0,),
        (4, 1,0,  0,1,),
        (5, 1,0,  0,1,),
        (6, 0,1,  1,0,),
        (7, 0,1,  1,0,),
    ]
    t, m0, m1,m2, m3  = undo[type]
    H = m0*height + m1*width
    W = m2*height + m3*width
    return do_flip_transpose(H, W, image, proposal, mask, instance, t )

"""
def do_flip_transpose(height, width, image=None, proposal=None, mask=None, instance=None, type=0):
    #choose one of the 8 cases

    if image    is not None: image=image.copy()
    if proposal is not None: proposal=proposal.copy()
    if mask     is not None: mask=mask.copy()
    if instance is not None: instance=instance.copy()

    if proposal is not None:
        x0 = proposal[:,1]
        y0 = proposal[:,2]
        x1 = proposal[:,3]
        y1 = proposal[:,4]

    if type==1: #rotate90
        if image is not None:
            image = image.transpose(1,0,2)
            image = cv2.flip(image,1)

        if mask is not None:
            #mask = np.rot90(mask,k=1)
            mask = mask.transpose(1,0)
            mask = np.fliplr(mask)


        if instance is not None:
            instance = instance.transpose(1,2,0)
            #instance = np.rot90(instance,k=1)
            instance = instance.transpose(1,0,2)
            instance = np.fliplr(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            x0, y0, x1, y1  =  y0, x0,  y1, x1
            x0, x1 = height-1-x0, height-1-x1

    if type==2: #rotate180
        if image is not None:
            image = cv2.flip(image,-1)

        if mask is not None:
            mask = np.rot90(mask,k=2)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            instance = np.rot90(instance,k=2)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            x0, x1 = width -1-x0, width -1-x1
            y0, y1 = height-1-y0, height-1-y1

    if type==3: #rotate270
        if image is not None:
            image = image.transpose(1,0,2)
            image = cv2.flip(image,0)

        if mask is not None:
            #mask = np.rot90(mask,k=3)
            mask = mask.transpose(1,0)
            mask = np.flipud(mask)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            #instance = np.rot90(instance,k=3)
            instance = instance.transpose(1,0,2)
            instance = np.flipud(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            x0, y0, x1, y1  =  y0, x0,  y1, x1
            y0, y1 = width-1-y0, width-1-y1

    if type==4: #flip left-right
        if image is not None:
            image = cv2.flip(image,1)

        if mask is not None:
            mask = np.fliplr(mask)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            instance = np.fliplr(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            x0, x1 = width -1-x0, width -1-x1

    if type==5: #flip up-down
        if image is not None:
            image = cv2.flip(image,0)

        if mask is not None:
            mask = np.flipud(mask)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            instance = np.flipud(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            y0, y1 = height-1-y0, height-1-y1

    if type==6:
        if image is not None:
            image = cv2.flip(image,1)
            image = image.transpose(1,0,2)
            image = cv2.flip(image,1)

        if mask is not None:
            mask = cv2.flip(mask,1)
            mask = mask.transpose(1,0)
            mask = cv2.flip(mask,1)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            instance = np.fliplr(instance)
            instance = np.rot90(instance,k=3)
            #instance = np.fliplr(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            x0, x1 = width -1-x0, width -1-x1
            x0, y0, x1, y1  =  y0, x0,  y1, x1
            x0, x1 = height -1-x0, height -1-x1

    if type==7:
        if image is not None:
            image = cv2.flip(image,0)
            image = image.transpose(1,0,2)
            image = cv2.flip(image,1)

        if mask is not None:
            mask = cv2.flip(mask,0)
            mask = mask.transpose(1,0)
            mask = cv2.flip(mask,1)

        if instance is not None:
            instance = instance.transpose(1,2,0)
            instance = np.flipud(instance)
            instance = np.rot90(instance,k=3)
            #instance = np.fliplr(instance)
            instance = instance.transpose(2,0,1)

        if proposal is not None:
            y0, y1 = height-1-y0, height-1-y1
            x0, y0, x1, y1  =  y0, x0,  y1, x1
            x0, x1 = height -1-x0, height -1-x1


    if proposal is not None:
        x0,x1 = np.minimum(x0,x1), np.maximum(x0,x1)
        y0,y1 = np.minimum(y0,y1), np.maximum(y0,y1)
        proposal[:,1] = x0
        proposal[:,2] = y0
        proposal[:,3] = x1
        proposal[:,4] = y1

    out=[]
    if image    is not None: out.append(image)
    if proposal is not None: out.append(proposal)
    if mask     is not None: out.append(mask)
    if instance is not None: out.append(instance)
    if len(out)==1: out=out[0]

    return out
"""
#my change
def do_flip_transpose(height, width, image=None, proposal=None, mask=None, instance=None, type=0):
    # choose one of the 8 cases

    if image is not None and len(image)!=0: image = image.copy()
    if proposal is not None  and len(proposal)!=0: proposal = proposal.copy()
    if mask is not None  and len(mask)!=0: mask = mask.copy()
    if instance is not None  and len(instance)!=0: instance = instance.copy()

    if proposal is not None  and len(proposal)!=0  :
        x0 = proposal[:, 1]
        y0 = proposal[:, 2]
        x1 = proposal[:, 3]
        y1 = proposal[:, 4]

    if type == 1:  # rotate90
        if image is not None and len(image)!=0:
            image = image.transpose(1, 0, 2)
            image = cv2.flip(image, 1)

        if mask is not None  and len(mask)!=0:
            # mask = np.rot90(mask,k=1)
            mask = mask.transpose(1, 0)
            mask = np.fliplr(mask)

        if instance is not None and len(proposal)!=0:
            instance = instance.transpose(1, 2, 0)
            # instance = np.rot90(instance,k=1)
            instance = instance.transpose(1, 0, 2)
            instance = np.fliplr(instance)
            instance = instance.transpose(2, 0, 1)

        if proposal is not None  and len(proposal)!=0 :
            x0, y0, x1, y1 = y0, x0, y1, x1
            x0, x1 = height - 1 - x0, height - 1 - x1

    if type == 2:  # rotate180
        if image is not None  and len(image)!=0:
            image = cv2.flip(image, -1)

        if mask is not None  and len(mask)!=0  :
            mask = np.rot90(mask, k=2)

        if instance is not None  and len(instance)!=0:
            instance = instance.transpose(1, 2, 0)
            instance = np.rot90(instance, k=2)
            instance = instance.transpose(2, 0, 1)

        if proposal is not None  and len(proposal)!=0 :
            x0, x1 = width - 1 - x0, width - 1 - x1
            y0, y1 = height - 1 - y0, height - 1 - y1

    if type == 3:  # rotate270
        if image is not None  and len(image)!=0:
            image = image.transpose(1, 0, 2)
            image = cv2.flip(image, 0)

        if mask is not None  and len(mask)!=0  :
            # mask = np.rot90(mask,k=3)
            mask = mask.transpose(1, 0)
            mask = np.flipud(mask)

        if instance is not None and len(instance)!=0:
            instance = instance.transpose(1, 2, 0)
            # instance = np.rot90(instance,k=3)
            instance = instance.transpose(1, 0, 2)
            instance = np.flipud(instance)
            instance = instance.transpose(2, 0, 1)

        if proposal is not None  and len(proposal)!=0 :
            x0, y0, x1, y1 = y0, x0, y1, x1
            y0, y1 = width - 1 - y0, width - 1 - y1

    if type == 4:  # flip left-right
        if image is not None  and len(image)!=0:
            image = cv2.flip(image, 1)

        if mask is not None  and len(mask)!=0:
            mask = np.fliplr(mask)

        if instance is not None  and len(instance)!=0:
            instance = instance.transpose(1, 2, 0)
            instance = np.fliplr(instance)
            instance = instance.transpose(2, 0, 1)

        if proposal is not None  and len(proposal)!=0:
            x0, x1 = width - 1 - x0, width - 1 - x1

    if type == 5:  # flip up-down
        if image is not None  and len(image)!=0  and len(image)!=0:
            image = cv2.flip(image, 0)

        if mask is not None  and len(mask)!=0 :
            mask = np.flipud(mask)

        if instance is not None  and len(instance)!=0:
            instance = instance.transpose(1, 2, 0)
            instance = np.flipud(instance)
            instance = instance.transpose(2, 0, 1)

        if proposal is not None  and len(proposal)!=0 :
            y0, y1 = height - 1 - y0, height - 1 - y1

    if type == 6:
        if image is not None  and len(image)!=0:
            image = cv2.flip(image, 1)
            image = image.transpose(1, 0, 2)
            image = cv2.flip(image, 1)

        if mask is not None  and len(mask)!=0 :
            mask = cv2.flip(mask, 1)
            mask = mask.transpose(1, 0)
            mask = cv2.flip(mask, 1)

        if instance is not None  and len(instance)!=0:
            instance = instance.transpose(1, 2, 0)
            instance = np.fliplr(instance)
            instance = np.rot90(instance, k=3)
            # instance = np.fliplr(instance)
            instance = instance.transpose(2, 0, 1)

        if proposal is not None  and len(proposal)!=0:
            x0, x1 = width - 1 - x0, width - 1 - x1
            x0, y0, x1, y1 = y0, x0, y1, x1
            x0, x1 = height - 1 - x0, height - 1 - x1

    if type == 7:
        if image is not None  and len(image)!=0:
            image = cv2.flip(image, 0)
            image = image.transpose(1, 0, 2)
            image = cv2.flip(image, 1)

        if mask is not None  and len(mask)!=0 :
            mask = cv2.flip(mask, 0)
            mask = mask.transpose(1, 0)
            mask = cv2.flip(mask, 1)

        if instance is not None  and len(instance)!=0:
            instance = instance.transpose(1, 2, 0)
            instance = np.flipud(instance)
            instance = np.rot90(instance, k=3)
            # instance = np.fliplr(instance)
            instance = instance.transpose(2, 0, 1)

        if proposal is not None  and len(proposal)!=0 :
            y0, y1 = height - 1 - y0, height - 1 - y1
            x0, y0, x1, y1 = y0, x0, y1, x1
            x0, x1 = height - 1 - x0, height - 1 - x1

    if proposal is not None  and len(proposal)!=0 :
        x0, x1 = np.minimum(x0, x1), np.maximum(x0, x1)
        y0, y1 = np.minimum(y0, y1), np.maximum(y0, y1)
        proposal[:, 1] = x0
        proposal[:, 2] = y0
        proposal[:, 3] = x1
        proposal[:, 4] = y1

    out = []
    if image is not None : out.append(image)
    if proposal is not None  : out.append(proposal)
    if mask is not None   : out.append(mask)
    if instance is not None : out.append(instance)
    if len(out) == 1: out = out[0]

    return out

def do_test_augment_identity(image, proposal=None):
    height,width = image.shape[:2]
    h = math.ceil(height/AUG_FACTOR)*AUG_FACTOR
    w = math.ceil(width /AUG_FACTOR)*AUG_FACTOR
    dx = w-width
    dy = h-height

    image = cv2.copyMakeBorder(image, left=0, top=0, right=dx, bottom=dy,
                               borderType= cv2.BORDER_REFLECT101, value=[0,0,0] )

    if proposal is not None and len(proposal)!=0:
        h,w = image.shape[:2]
        proposal = proposal.copy()
        x1,y1 = proposal[:,3],proposal[:,4]
        x1[np.where(x1>width -1-dx)[0]]=w-1 #dx
        y1[np.where(y1>height-1-dy)[0]]=h-1 #dy
        proposal[:,3] = x1
        proposal[:,4] = y1
        return image, proposal
    else:
        return image


def undo_test_augment_identity(net, image):

    height,width = image.shape[:2]

    rcnn_proposals = net.rcnn_proposals.cpu().numpy()
    detections     = net.detections.data.cpu().numpy()
    multi_masks    = net.masks[0].copy()
    if len(net.mask_instances) > 0:
        instances      = net.mask_instances[0].copy()
    else:
        instances = []
    
    if len(rcnn_proposals) > 0:
        rcnn_proposals[:,1]=np.clip(rcnn_proposals[:,1],0,width -1)
        rcnn_proposals[:,2]=np.clip(rcnn_proposals[:,2],0,height-1)
        rcnn_proposals[:,3]=np.clip(rcnn_proposals[:,3],0,width -1)
        rcnn_proposals[:,4]=np.clip(rcnn_proposals[:,4],0,height-1)
    if len(rcnn_proposals) > 0:
        detections[:,1]=np.clip(detections[:,1],0,width -1)
        detections[:,2]=np.clip(detections[:,2],0,height-1)
        detections[:,3]=np.clip(detections[:,3],0,width -1)
        detections[:,4]=np.clip(detections[:,4],0,height-1)

    multi_masks = multi_masks[0:height, 0:width]
    if len(net.mask_instances) > 0:
        instances = instances[:, 0:height, 0:width]
    else:
        instances = []

    return rcnn_proposals, detections, multi_masks, instances


def do_test_augment_flip_transpose(image, proposal=None, type=0):
    height,width = image.shape[:2]
    image = do_flip_transpose(height,width, image=image, type=type)

    if proposal is not None:
        proposal = do_flip_transpose(height,width, proposal=proposal, type=type)

    return do_test_augment_identity(image, proposal)


def undo_test_augment_flip_transpose(net, image, type=0):
    height,width = image.shape[:2]
    undo=[
        (0, 1,0,  0,1,),
        (3, 0,1,  1,0,),
        (2, 1,0,  0,1,),
        (1, 0,1,  1,0,),
        (4, 1,0,  0,1,),
        (5, 1,0,  0,1,),
        (6, 0,1,  1,0,),
        (7, 0,1,  1,0,),
    ]
    t, m0, m1,m2, m3  = undo[type]
    H = m0*height + m1*width
    W = m2*height + m3*width

    dummy_image = np.zeros((H,W,3),np.uint8)
    rcnn_proposal, detection, mask, instance = undo_test_augment_identity(net, dummy_image)
    #detection, mask, instance = do_flip_transpose(H,W, proposal=detection, mask=mask, instance=instance, type=t)
    #my change
    out= do_flip_transpose(H, W, proposal=detection, mask=mask, instance=instance, type=t)
    if(len(out)>=1):
        detection, mask, instance = out
    else:
        detection=[]
        mask=[]
        instance=[]
    rcnn_proposal             = do_flip_transpose(H,W, proposal=rcnn_proposal, type=t)

    return rcnn_proposal, detection, mask, instance


def do_test_augment_scale(image, proposal=None, scale_x=1, scale_y=1):
    height,width = image.shape[:2]

    image = scale_to_factor(image, scale_x, scale_y, factor=AUG_FACTOR)
    if proposal is not None:
        H,W = image.shape[:2]
        x0 = proposal[:,1]
        y0 = proposal[:,2]
        x1 = proposal[:,3]
        y1 = proposal[:,4]
        proposal[:,1] = np.round(x0 * (W-1)/(width -1))
        proposal[:,2] = np.round(y0 * (H-1)/(height-1))
        proposal[:,3] = np.round(x1 * (W-1)/(width -1))
        proposal[:,4] = np.round(y1 * (H-1)/(height-1))
        return image, proposal

    else :
        return image


def undo_test_augment_scale(net, image, scale_x=1, scale_y=1):

    def scale_mask(H,W, detection, mask_prob):

        multi_mask = np.zeros((H,W),np.int32)
        instance_prob = []

        num_detection = len(detection)
        for n in range(num_detection):
            _,x0,y0,x1,y1,score,label,k = detection[n]
            x0 = int(round(x0))
            y0 = int(round(y0))
            x1 = int(round(x1))
            y1 = int(round(y1))
            label = int(label)
            k = int(k)
            h, w  = y1-y0+1, x1-x0+1

            crop  = mask_prob[k, label]
            crop  = cv2.resize(crop, (w,h), interpolation=cv2.INTER_LINEAR)

            mask = np.zeros((H,W),np.float32)
            mask[y0:y1+1,x0:x1+1] = crop
            instance_prob.append(mask.reshape(1,H,W))

            binary = instance_to_binary(mask, 
                                        threshold=net.cfg.mask_test_mask_threshold,
                                        min_area=net.cfg.mask_test_mask_min_area)
            multi_mask[np.where(binary)] = n+1

        instance_prob = np.vstack(instance_prob)
        return multi_mask, instance_prob

    # ----
    rcnn_proposal = net.rcnn_proposals.cpu().numpy()
    detection     = net.detections.data.cpu().numpy()
    mask          = net.masks[0]
    if rcnn_proposal is not None and len(rcnn_proposal)!=0:
        mask_logit = net.mask_logits.cpu().data.numpy()
        mask_prob  = np_sigmoid(mask_logit)


        height,width = image.shape[:2]
        H,W  = mask.shape[:2]
        scale_x = (width -1)/(W-1)
        scale_y = (height-1)/(H-1)
    #MY CHANGE
    #if rcnn_proposal is not None:
        x0 = rcnn_proposal[:,1]
        y0 = rcnn_proposal[:,2]
        x1 = rcnn_proposal[:,3]
        y1 = rcnn_proposal[:,4]
        rcnn_proposal[:,1] = np.round(x0 * scale_x)
        rcnn_proposal[:,2] = np.round(y0 * scale_y)
        rcnn_proposal[:,3] = np.round(x1 * scale_x)
        rcnn_proposal[:,4] = np.round(y1 * scale_y)
    #if detection is not None:
        x0 = detection[:,1]
        y0 = detection[:,2]
        x1 = detection[:,3]
        y1 = detection[:,4]
        detection[:,1] = np.round(x0 * scale_x)
        detection[:,2] = np.round(y0 * scale_y)
        detection[:,3] = np.round(x1 * scale_x)
        detection[:,4] = np.round(y1 * scale_y)

        mask, instance = scale_mask(height, width,  detection, mask_prob, )
    else:
        instance=mask
    return rcnn_proposal, detection, mask, instance
