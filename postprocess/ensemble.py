import os, sys
import glob
sys.path.append(os.path.dirname(__file__))
from configuration import Configuration
from dataset.reader import *
from dataset.folder import TrainFolder
from utility.draw import *
from net.lib.nms.cython_nms.cython_nms import cython_nms
#ensemble =======================================================

class Cluster(object):
    def __init__(self):
        super(Cluster, self).__init__()
        self.members=[]
        self.center ={}

    def add_item(self, proposal, instance):
        if self.center =={}:
            self.members = [{
                'proposal': proposal, 'instance': instance
            },]
            self.center  = {
                'union_proposal': proposal, 'union_instance':instance,
                'inter_proposal': proposal, 'inter_instance':instance,
            }
        else:
            self.members.append({
                'proposal': proposal, 'instance': instance
            })
            center_union_proposal = self.center['union_proposal'].copy()
            center_union_instance = self.center['union_instance'].copy()
            center_inter_proposal = self.center['inter_proposal'].copy()
            center_inter_instance = self.center['inter_instance'].copy()

            self.center['union_proposal'] = [
                min(center_union_proposal[1],proposal[1]),
                min(center_union_proposal[2],proposal[2]),
                max(center_union_proposal[3],proposal[3]),
                max(center_union_proposal[4],proposal[4]),
                max(center_union_proposal[5],proposal[5]),
            ]
            self.center['union_instance'] = np.maximum(center_union_instance , instance )

            self.center['inter_proposal'] = [
                max(center_inter_proposal[1],proposal[1]),
                max(center_inter_proposal[2],proposal[2]),
                min(center_inter_proposal[3],proposal[3]),
                min(center_inter_proposal[4],proposal[4]),
                min(center_inter_proposal[5],proposal[5]),
            ]
            self.center['inter_instance'] = np.minimum(center_inter_instance , instance )

    def distance(self, proposal, instance, type='union'):

        if type=='union':
            center_proposal = self.center['inter_proposal']
            center_instance = self.center['inter_instance']
        elif type=='inter':
            center_proposal = self.center['box']
            center_instance = self.center['union']
        else:
            raise NotImplementedError

        x0 = int(max(proposal[1],center_proposal[1]))
        y0 = int(max(proposal[2],center_proposal[2]))
        x1 = int(min(proposal[3],center_proposal[3]))
        y1 = int(min(proposal[4],center_proposal[4]))

        w = max(0,x1-x0)
        h = max(0,y1-y0)
        box_intersection = w*h
        if box_intersection<0.01: return 0

        x0 = int(min(proposal[0],center_proposal[0]))
        y0 = int(min(proposal[1],center_proposal[1]))
        x1 = int(max(proposal[2],center_proposal[2]))
        y1 = int(max(proposal[3],center_proposal[3]))

        i0 = center_instance[y0:y1,x0:x1]  #center_inter[y0:y1,x0:x1]
        i1 = instance[y0:y1,x0:x1]>0.5

        intersection = np.logical_and(i0, i1).sum()
        area    = np.logical_or(i0, i1).sum()
        overlap = intersection/(area + 1e-12)

        return overlap


def do_clustering( proposals, instances, threshold=0.5, type='union'):

    clusters = []
    num_augments   = len(instances)
    for n in range(0, num_augments):
        proposal = proposals[n]
        instance = instances[n]

        num = len(instance)
        for i in range(num):
            p, m = proposal[i],instance[i]

            is_group = 0
            for c in clusters:
                iou = c.distance(p, m, type)

                if iou>threshold:
                    c.add_item(p, m)
                    is_group=1

            if is_group == 0:
                c = Cluster()
                c.add_item(p, m)
                clusters.append(c)

    return clusters


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
    out_dir = os.path.join(f_eval.folder_name, 'predict', 'xx_ensemble')

    ensemble_dirs = [
        'xx_normal',
        'xx_flip_transpose_1',
        'xx_flip_transpose_2',
        'xx_flip_transpose_3',
        'xx_flip_transpose_4',
        'xx_flip_transpose_5',
        'xx_flip_transpose_6',
        'xx_flip_transpose_7',
        'xx_scale_0.8',
        'xx_scale_1.2',
        'xx_scale_0.5',
        'xx_scale_1.8',
    ]
    ensemble_dirs = [ os.path.join(f_eval.folder_name, 'predict', e) for e in ensemble_dirs ]

    #setup ---------------------------------------
    os.makedirs(os.path.join(out_dir, 'proposal', 'overlays'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'proposal', 'npys'), exist_ok=True)

    names = glob.glob(ensemble_dirs[0] + '/overlays/*/')
    names = [n.split('/')[-2]for n in names]
    sorted(names)

    num_ensemble = len(ensemble_dirs)
    for name in names:
        print(name)
        image_file = os.path.join(ensemble_dirs[0], 'overlays', name, '%s.png'%name)
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)
        height, width= image.shape[:2]

        image1 = image.copy()
        image2 = image.copy()
        image3 = image.copy()

        detections=[]
        tight_detections=[]
        for t, dir in enumerate(ensemble_dirs):
            npy_file = os.path.join(dir, 'detections', "%s.npy"%name)
            detection = np.load(npy_file)
            tight_detection = make_tight_proposal(detection, 0.25, width, height)

            keep = check_valid(tight_detection, width, height)
            #if len(keep)==0: continue
            detection       = detection[keep]
            tight_detection = tight_detection[keep]

            detections.append(detection)
            tight_detections.append(tight_detection)

            for n in range(len(detection)):
                _,x0,y0,x1,y1,score,label = tight_detection[n]

                color = to_color((score-0.5)/0.5,(0,255,255))
                cv2.rectangle(image2, (x0,y0), (x1,y1), color, 1)

                if t == 0:
                    cv2.rectangle(image1, (x0,y0), (x1,y1), color, 1)
            # image_show('image1',image1,1)
            # cv2.waitKey(0)

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


def run_ensemble_xxx():
    cfg = Configuration()
    f_eval = TrainFolder(os.path.join(cfg.result_dir, cfg.model_name))
    out_dir = os.path.join(f_eval.folder_name, 'predict', 'xx_ensemble1')

    ensemble_dirs = [
        'xx_normal',
        'xx_flip_transpose_1',
        'xx_flip_transpose_2',
        'xx_flip_transpose_3',
        'xx_flip_transpose_4',
        'xx_flip_transpose_5',
        'xx_flip_transpose_6',
        'xx_flip_transpose_7',
        'xx_scale_0.8',
        'xx_scale_1.2',
        'xx_scale_0.5',
        'xx_scale_1.8',
    ]
    ensemble_dirs = [ os.path.join(f_eval.folder_name, 'predict', e) for e in ensemble_dirs ]

    #setup ---------------------------------------
    os.makedirs(os.path.join(out_dir, 'proposal', 'overlays'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'proposal', 'npys'), exist_ok=True)

    names = glob.glob(ensemble_dirs[0] + '/overlays/*/')
    names = [n.split('/')[-2]for n in names]
    sorted(names)

    num_ensemble = len(ensemble_dirs)
    for name in names:
        print(name)
        image_file = os.path.join(ensemble_dirs[0], 'overlays', name, '%s.png' % name)
        image = cv2.imread(image_file,cv2.IMREAD_COLOR)
        height, width= image.shape[:2]

        overall = np.zeros((height, width, 3),np.float32)
        for t,dir in enumerate(ensemble_dirs):
            image_file = os.path.join(dir, 'overlays', "%s.png" % name)
            image = cv2.imread(image_file,cv2.IMREAD_COLOR)
            overall += image

        overall = overall/num_ensemble

        #image_show('overall',overall,1)
        #image_show('image2',image2,1)
        #image_show('image3',image3,1)
        #image_show('all',all,1)
        #cv2.waitKey(0)


def run_make_l2_data():
    cfg = Configuration()
    f_eval = TrainFolder(os.path.join(cfg.result_dir, cfg.model_name))
    out_dir = os.path.join(f_eval.folder_name, 'predict', 'xx_l2_data')

    ensemble_dirs = [
        'xx_normal',
        'xx_flip_transpose_1',
        'xx_flip_transpose_2',
        'xx_flip_transpose_3',
        'xx_flip_transpose_4',
        'xx_flip_transpose_5',
        'xx_flip_transpose_6',
        'xx_flip_transpose_7',
        'xx_scale_0.8',
        'xx_scale_1.2',
        'xx_scale_0.5',
        'xx_scale_1.8',
    ]
    #ensemble_dirs = [os.path.join(f_eval.folder_name, 'predict', e) for e in ensemble_dirs]
    ensemble_dirs = [os.path.join(f_eval.folder_name, 'predict', 'xx_ensemble_'+e) for e in ensemble_dirs]

    #setup ---------------------------------------
    os.makedirs(out_dir +'/ensemble_data_overlays', exist_ok=True)
    os.makedirs(out_dir +'/ensemble_data', exist_ok=True)

    split = 'test_black_white_53'  #'BBBC006'   #'valid1_ids_gray2_43' #
    ids = read_list_from_file(os.path.join(cfg.split_dir, split), comment='#')

    for i in range(len(ids)):
        folder, name = ids[i].split('/')[-2:]
        print('%05d %s'%(i,name))

        image = cv2.imread(os.path.join(cfg.data_dir, folder, 'images', '%s.png' % name), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]

        instances=[]
        proposals=[]
        for t,dir in enumerate(ensemble_dirs):
            instance = np.load(os.path.join(dir, 'instances', '%s.npy'%name))
            proposal = np.load(os.path.join(dir, 'detections', '%s.npy'%name))
            assert(len(proposal)==len(instance))

            instances.append(instance)
            proposals.append(proposal)

        clusters = do_clustering(proposals, instances, threshold=0.5, type='union')

        # ensemble instance
        ensemble_instances      = []
        ensemble_instance_edges = []
        for c in clusters:
            num_members = len(c.members)

            ensemble_instance = np.zeros((height, width),np.float32)
            ensemble_instance_edge = np.zeros((height, width),np.float32)
            for j in range(num_members):
                m = c.members[j]['instance']

                kernel = np.ones((3,3),np.float32)
                me = m - cv2.erode(m,kernel)
                md = m - cv2.dilate(m,kernel)
                diff =(me - md)*m

                ensemble_instance += m
                ensemble_instance_edge += diff

            ensemble_instances.append(ensemble_instance)
            ensemble_instance_edges.append(ensemble_instance_edge)

        ensemble_instances = np.array(ensemble_instances)
        ensemble_instance_edges = np.array(ensemble_instance_edges)

        sum_instance      = ensemble_instances.sum(axis=0)
        sum_instance_edge = ensemble_instance_edges.sum(axis=0)

        gray0 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray1 = (sum_instance/sum_instance.max()*255).astype(np.uint8)
        gray2 = (sum_instance_edge/sum_instance_edge.max()*255).astype(np.uint8)
        all = np.hstack([gray0,gray1,gray2])
        #image_show('all',all,1)

        #save as train data
        data  = cv2.merge((gray0, gray1, gray2))

        cv2.imwrite(os.path.join(out_dir, 'ensemble_data_overlays', '%s.png'%name), all)
        cv2.imwrite(os.path.join(out_dir, 'ensemble_data',          '%s.png'%name), data)


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_ensemble_box()
    print('\nsucess!')