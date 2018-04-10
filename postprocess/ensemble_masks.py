import sys
sys.path.append('..')
from configuration import Configuration
from dataset.reader import *
from dataset.folder import TrainFolder
from utility.draw import *
from net.lib.nms.cython_nms.cython_nms import cython_nms
from net.layer.mask import instance_to_binary

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


def ensemble_masks():
    cfg = Configuration()
    f_eval = TrainFolder(os.path.join(cfg.result_dir, cfg.model_name))
    out_dir = os.path.join(f_eval.folder_name, 'predict', 'ensemble_all')

    ensemble_dirs = [
        'normal',
        'flip_transpose_1',
        'flip_transpose_2',
        'flip_transpose_3',
        'flip_transpose_4',
        'flip_transpose_5',
        'flip_transpose_6',
        'flip_transpose_7',
        'scale_0.8',
        'scale_1.2',
        'scale_0.5',
        'scale_1.8',
    ]
    #ensemble_dirs = [os.path.join(f_eval.folder_name, 'predict', e) for e in ensemble_dirs]
    ensemble_dirs = [os.path.join(f_eval.folder_name, 'predict', 'mask_ensemble_'+e) for e in ensemble_dirs]

    #setup ---------------------------------------
    os.makedirs(out_dir +'/ensemble_data_overlays', exist_ok=True)
    os.makedirs(out_dir +'/ensemble_data', exist_ok=True)

    split = 'valid_test'#'test_black_white_53'  #'BBBC006'   #'valid1_ids_gray2_43' #
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
        ensemble_binary         = []
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

            ensemble_avg = ensemble_instance / num_members
            binary = instance_to_binary(ensemble_avg,
                                        cfg.mask_test_mask_threshold,
                                        cfg.mask_test_mask_min_area)

            ensemble_binary.append(binary)
            ensemble_instances.append(ensemble_instance)
            ensemble_instance_edges.append(ensemble_instance_edge)

        multi_mask = instance_to_multi_mask(np.array(ensemble_binary))
        ensemble_instances = np.array(ensemble_instances)
        ensemble_instance_edges = np.array(ensemble_instance_edges)

        sum_instance      = ensemble_instances.sum(axis=0)
        sum_instance_edge = ensemble_instance_edges.sum(axis=0)

        gray0 = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        gray1 = (sum_instance/sum_instance.max()*255).astype(np.uint8)
        gray2 = (sum_instance_edge/sum_instance_edge.max()*255).astype(np.uint8)
        all = np.hstack([gray0,gray1,gray2])

        #save as train data
        data  = cv2.merge((gray0, gray1, gray2))

        cv2.imwrite(os.path.join(out_dir, 'ensemble_data_overlays', '%s.png'%name), all)
        cv2.imwrite(os.path.join(out_dir, 'ensemble_data',          '%s.png'%name), data)
        np.save(    os.path.join(out_dir, 'ensemble_masks',         '%s.npy'%name), multi_mask)


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    ensemble_masks()
    print('\nsucess!')