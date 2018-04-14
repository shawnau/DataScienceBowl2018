import os, sys
import glob
sys.path.append('..')
from configuration import Configuration
from dataset.reader import *
from dataset.folder import TrainFolder
from utility.draw import *
from net.lib.nms.cython_nms.cython_nms import cython_nms
#ensemble =======================================================


def check_valid(proposal, width, height):

    num_proposal = len(proposal)

    keep=[]
    for n in range(num_proposal):
        b,x0,y0,x1,y1,score,label,aux = proposal[n]
        w = x1 - x0
        h = y1 - y0

        valid = 1
        if (w*h <10) or \
           (x0<w/4          and h/w>3.5) or \
           (x1>width-1-w/4  and h/w>3.5) or \
           (y0<h/4          and w/h>3.5) or \
           (y1>height-1-h/4 and w/h>3.5) or \
           0: valid=0

        if valid:
            keep.append(n)

    return keep


#remove border
def make_tight_proposal(proposal, border, width, height):
    proposal = proposal.copy()
    x0 = proposal[:,1]
    y0 = proposal[:,2]
    x1 = proposal[:,3]
    y1 = proposal[:,4]
    w = x1 - x0
    h = y1 - y0
    x = (x1 + x0)/2
    y = (y1 + y0)/2

    aa =  w - h
    bb = (w + h)/(2*border+1)
    ww = np.clip((aa + bb)/2, 0, np.inf)
    hh = np.clip((bb - aa)/2, 0, np.inf)

    is_boundy_x0 = x0 < w/4
    is_boundy_x1 = x1 > width-1 -w/4
    is_boundy_y0 = y0 < h/4
    is_boundy_y1 = y1 > height-1-h/4

    proposal[:, 1] = is_boundy_x0*x0 + (1-is_boundy_x0)*(x - ww/2)
    proposal[:, 2] = is_boundy_y0*y0 + (1-is_boundy_y0)*(y - hh/2)
    proposal[:, 3] = is_boundy_x1*x1 + (1-is_boundy_x1)*(x + ww/2)
    proposal[:, 4] = is_boundy_y1*y1 + (1-is_boundy_y1)*(y + hh/2)
    return proposal


def run_ensemble_box():
    cfg = Configuration()
    f_eval = TrainFolder(os.path.join(cfg.result_dir, cfg.model_name))
    out_dir = os.path.join(f_eval.folder_name, 'predict', 'box_ensemble')
    ensemble_dirs = [ os.path.join(f_eval.folder_name, 'predict', 'box_'+e) for e in cfg.test_augment_names ]

    split = cfg.valid_split
    ids = read_list_from_file(os.path.join(cfg.split_dir, split), comment='#')
    
    #setup ---------------------------------------
    os.makedirs(os.path.join(out_dir, 'proposal', 'overlays'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'proposal', 'npys'), exist_ok=True)
    
    for row in ids:
        folder, name = row.split('/')[-2:]
        print(name)
        image = cv2.imread(os.path.join(cfg.data_dir, folder, 'images', '%s.png' % name), cv2.IMREAD_COLOR)
        height, width= image.shape[:2]

        image1 = image.copy()
        image2 = image.copy()
        image3 = image.copy()

        detections=[]
        tight_detections=[]
        for t, dir in enumerate(ensemble_dirs):
            npy_file = os.path.join(dir, 'detections', "%s.npy"%name)
            detection = np.load(npy_file)
            #my change
            if detection is not None and len(detection)!=0:
                tight_detection = make_tight_proposal(detection, 0.25, width, height)

                keep = check_valid(tight_detection, width, height)
                #if len(keep)==0: continue
                detection       = detection[keep]
                tight_detection = tight_detection[keep]

                detections.append(detection)
                tight_detections.append(tight_detection)

                for n in range(len(detection)):
                    _,x0,y0,x1,y1,score,label,k = tight_detection[n]

                    color = to_color((score-0.5)/0.5,(0,255,255))
                    cv2.rectangle(image2, (x0,y0), (x1,y1), color, 1)

                    if t == 0:
                        cv2.rectangle(image1, (x0,y0), (x1,y1), color, 1)
                # image_show('image1',image1,1)
                # cv2.waitKey(0)

        if len(detections) > 0:
            detections = np.vstack(detections)
            tight_detections = np.vstack(tight_detections)
            rois = tight_detections[:,1:6]  #detections[:,1:6]
            keep = cython_nms(rois, 0.5)
            # for i in keep:
            #     #_,x0,y0,x1,y1,score,label,k = detections[i]
            #     x0,y0,x1,y1,score = rois[i]
            #     cv2.rectangle(image3, (x0,y0), (x1,y1), (0,255,0), 1)

            nms = detections[keep]
            tight_nms = tight_detections[keep]

            for i in range(len(nms)):
                x0,y0,x1,y1,score = tight_nms[i][1:6]
                #x0,y0,x1,y1 = small_nms_box[n]
                color = to_color((score-0.5)/0.5,(0,255,255))
                cv2.rectangle(image3, (x0,y0), (x1,y1), color, 1)

        draw_shadow_text(image1, 'original',  (5,15),0.5, (255,255,255), 1)
        draw_shadow_text(image2, 'all',       (5,15),0.5, (255,255,255), 1)
        draw_shadow_text(image3, 'all(nms)',  (5,15),0.5, (255,255,255), 1)

        all = np.hstack([image1,image3,image2])

        #image_show('all',all,1)
        #cv2.waitKey(0)

        ##save results ---------------------------------------
        np.save(os.path.join(out_dir, 'proposal', 'npys', "%s.npy"%name),detections)
        cv2.imwrite(os.path.join(out_dir, 'proposal', 'overlays', "%s.png"%name),all)


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_ensemble_box()
    print('\nsucess!')