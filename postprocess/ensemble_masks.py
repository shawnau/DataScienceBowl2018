import sys, operator
sys.path.append('..')
from scipy.ndimage.morphology import binary_fill_holes
from configuration import Configuration
from dataset.reader import *
from dataset.folder import TrainFolder
from utility.draw import *
from net.lib.nms.cython_nms.cython_nms import cython_nms
from net.layer.mask import instance_to_binary


class MaskCluster:
    def __init__(self):
        super(MaskCluster, self).__init__()
        self.members = []
        self.union = None
        self.union_size = None

    def add(self, instance):
        if self.members == []:
            self.members.append(instance)
            self.union = instance
            self.union_size = instance.sum()
        else:
            self.members.append(instance)
            self.union = np.logical_or(self.union, instance)
            self.union_size = self.union.sum()


def clustering_masks(instances, iou_threshold=0.3, overlap_threshold=0.3):
    """
    :param instances: numpy array of instances
    :return:
    """
    clusters = []
    num = instances.shape[0]

    instance_sizes = []
    for i in range(num):
        instance = instances[i]
        instance_sizes.append((i, instance.sum()))
    sorted_sizes = sorted(instance_sizes, key=lambda tup: tup[1], reverse=True)


    for i, instance_size in sorted_sizes:
        instance = instances[i]

        added_to_group = False
        for c in clusters:
            cluster_size = c.union_size
            inter = np.logical_and(c.union, instance).sum()
            union = np.logical_or(c.union, instance).sum()
            iou = inter / (union + 1e-12)

            if (inter / cluster_size > overlap_threshold) or \
                    (inter / instance_size > overlap_threshold) or \
                    (iou > iou_threshold):
                c.add(instance)
                added_to_group = True

        if added_to_group == False:
            c = MaskCluster()
            c.add(instance)
            clusters.append(c)

    return clusters


def fill_holes(instances):
    for i in range(instances.shape[0]):
        instances[i] = binary_fill_holes(instances[i]).astype(np.float32)
    return instances


def filter_small(proposals, instances, min_threshold=0.0001, area_threshold=10):
    """
    :param instances: numpy array of 0/1 instance in one image
    :param area_threshold: do filter if max mask / min mask > this
    :param min_threshold: min area ratio
    :return: filtered instances
    """
    H, W = instances[0].shape[:2]
    min_size = (H * W) * min_threshold

    keep_instances = []
    keep_proposals = []
    max_size = 0

    for i in range(instances.shape[0]):
        size = instances[i].sum()
        if size > max_size:
            max_size = size

    if (max_size / min_size) > area_threshold:
        for i in range(instances.shape[0]):
            size = instances[i].sum()
            if size / (H * W) > min_threshold:
                keep_instances.append(instances[i])
                keep_proposals.append(proposals[i])
    else:
        keep_instances = instances
        keep_proposals = proposals

    keep_proposals = np.array(keep_proposals)
    keep_instances = np.array(keep_instances)

    return keep_proposals, keep_instances


def ensemble_masks():
    cfg = Configuration()
    f_eval = TrainFolder(os.path.join(cfg.result_dir, cfg.model_name))
    out_dir = os.path.join(f_eval.folder_name, 'predict', 'ensemble_all')

    #ensemble_dirs = [os.path.join(f_eval.folder_name, 'predict', e) for e in ensemble_dirs]
    ensemble_dirs = [os.path.join(f_eval.folder_name, 'predict', 'mask_ensemble_'+e) for e in cfg.test_augment_names]

    #setup ---------------------------------------
    os.makedirs(out_dir +'/ensemble_data_overlays', exist_ok=True)
    os.makedirs(out_dir +'/ensemble_data', exist_ok=True)
    os.makedirs(out_dir +'/ensemble_masks', exist_ok=True)

    split = cfg.valid_split#'test_black_white_53'
    ids = read_list_from_file(os.path.join(cfg.split_dir, split), comment='#')

    for i in range(len(ids)):
        folder, name = ids[i].split('/')[-2:]
        print('%05d %s'%(i,name))

        image = cv2.imread(os.path.join(cfg.data_dir, folder, 'images', '%s.png' % name), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]

        instances=[]
        proposals=[]
        for t,dir in enumerate(ensemble_dirs):
            instance_prob = np.load(os.path.join(dir, 'instances',   '%s.npy'%name))
            multi_mask    = np.load(os.path.join(dir, 'multi_masks', '%s.npy'%name))
            instance      = (instance_prob > cfg.mask_test_mask_threshold).astype(np.float32)
            proposal      = np.load(os.path.join(dir, 'detections',  '%s.npy'%name))
            assert(len(proposal)==len(instance))

            instances.append(instance)
            proposals.append(proposal)

        all_proposals = np.concatenate(proposals)
        all_instances = np.concatenate(instances)

        # filter small noises
        all_proposals, all_instances = filter_small(all_proposals, all_instances)

        # nms
        rois = all_proposals[:, 1:6]
        keep = cython_nms(rois, 0.5)
        all_instances = all_instances[keep]
        
        # fill holes
        all_instances = fill_holes(all_instances)

        # mask cluster
        clusters = clustering_masks(all_instances, iou_threshold=0.5)

        # ensemble instance
        ensemble_instances = []  # list of summed up instance clusters
        ensemble_instance_edges = []
        for c in clusters:
            num_members = len(c.members)
            ensemble_instance = np.zeros((height, width), np.float32)  # summed up one cluster
            ensemble_instance_edge = np.zeros((height, width), np.float32)
            for j in range(num_members):
                m = c.members[j]  # members in the same cluster
                kernel = np.ones((3, 3), np.float32)
                me = m - cv2.erode(m, kernel)
                md = m - cv2.dilate(m, kernel)
                diff = (me - md) * m

                ensemble_instance += m
                ensemble_instance_edge += diff
            # convert sum_up of one cluster to binary
            binary_instance = ((ensemble_instance / num_members) > 0.1).astype(np.float32)
            ensemble_instances.append(binary_instance)
            ensemble_instance_edges.append(ensemble_instance_edge)

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
        multi_mask = instance_to_multi_mask(ensemble_instances)
        cv2.imwrite(os.path.join(out_dir, 'ensemble_data_overlays', '%s.png'%name), all)
        cv2.imwrite(os.path.join(out_dir, 'ensemble_data',          '%s.png'%name), data)
        np.save(    os.path.join(out_dir, 'ensemble_masks',         '%s.npy'%name), multi_mask)


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    ensemble_masks()
    print('\nsucess!')