import sys, operator

sys.path.append('..')
from scipy.ndimage.morphology import binary_fill_holes
from configuration import Configuration
from dataset.reader import *
from dataset.folder import TrainFolder
from utility.draw import *
from net.lib.nms.cython_nms.cython_nms import cython_nms
from net.layer.mask import instance_to_binary

from multiprocessing import Pool
from numba import jit


class MaskCluster:
    def __init__(self):
        super(MaskCluster, self).__init__()
        self.members = []
        self.core = None
        self.core_size = None

    def add(self, instance, type='union'):
        """
        :param instance: numpy array of one instance
        :param type:
            union: core is union
            intersect: core is intersection
        :return:
        """
        if self.members == []:
            self.members.append(instance)
            self.core = instance
            self.core_size = instance.sum()
        else:
            self.members.append(instance)
            if type == 'union':
                self.core = np.logical_or(self.core, instance)
            elif type == 'intersect':
                self.core = np.logical_and(self.core, instance)
            else:
                raise NotImplementedError
            self.core_size = self.core.sum()


def clustering_masks(instances, iou_threshold=0.5, overlap_threshold=0.8):
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
            cluster_size = c.core_size
            inter = np.logical_and(c.core, instance).sum()
            union = np.logical_or(c.core, instance).sum()
            iou = inter / (union + 1e-12)

            if ((inter / cluster_size) > overlap_threshold) or \
                    ((inter / instance_size) > overlap_threshold) or \
                    (iou > iou_threshold):
                c.add(instance)
                added_to_group = True

        if added_to_group == False:
            c = MaskCluster()
            c.add(instance)
            clusters.append(c)

    return clusters

@jit
def fill_holes(instances):
    for i in range(instances.shape[0]):
        instances[i] = binary_fill_holes(instances[i]).astype(np.float32)
    return instances

@jit
def filter_small(proposals, instances, area_threshold=36):
    """
    :param instances: numpy array of 0/1 instance in one image
    :param area_threshold: do filter if max mask / min mask > this
    :param min_threshold: min area ratio
    :return: filtered instances
    """
    H, W = instances[0].shape[:2]

    keep_instances = []
    keep_proposals = []
    max_size = 0
    min_size = H*W

    for i in range(instances.shape[0]):
        size = instances[i].sum()
        if size > max_size:
            max_size = size
        elif size < min_size:
            min_size = size

    size_threshold = max_size / area_threshold
    
    if (max_size / min_size) > area_threshold:
        for i in range(instances.shape[0]):
            size = instances[i].sum()
            
            if size > size_threshold:
                #print('%d: %d'%(i, size), ' > ', size_threshold, 'append')
                keep_instances.append(instances[i])
                keep_proposals.append(proposals[i])
            else:
                pass
                #print('%d: %d'%(i, size), ' < ', size_threshold, 'exclude')
    else:
        keep_instances = instances
        keep_proposals = proposals

    keep_proposals = np.array(keep_proposals)
    keep_instances = np.array(keep_instances)

    return keep_proposals, keep_instances


def ensemble_one_mask(packed):
    cfg, folder_name, img_id = packed
    ensemble_dirs = [os.path.join(folder_name, 'predict', 'mask_ensemble_' + e) for e in cfg.test_augment_names]
    out_dir = os.path.join(folder_name, 'predict', 'ensemble_all')

    folder, name = img_id.split('/')[-2:]
    
    #if os.path.isfile(os.path.join(out_dir, 'ensemble_masks', '%s.npy' % name)):
    #    print('skip: ', name[:6])
    #    return

    image = cv2.imread(os.path.join(cfg.data_dir, folder, 'images', '%s.png' % name), cv2.IMREAD_COLOR)
    height, width = image.shape[:2]

    instances = []
    proposals = []
    for t, dir in enumerate(ensemble_dirs):
        instance_prob = np.load(os.path.join(dir, 'instances', '%s.npy' % name))
        instance = (instance_prob > cfg.mask_test_mask_threshold).astype(np.float32)
        proposal = np.load(os.path.join(dir, 'detections', '%s.npy' % name))
        assert (len(proposal) == len(instance))

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
    # all_instances = fill_holes(all_instances)

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
        binary_instance = ((ensemble_instance / num_members) > 0.2).astype(np.float32)
        ensemble_instances.append(binary_instance)
        ensemble_instance_edges.append(ensemble_instance_edge)

    ensemble_instances = np.array(ensemble_instances)
    ensemble_instance_edges = np.array(ensemble_instance_edges)

    sum_instance = ensemble_instances.sum(axis=0)
    sum_instance_edge = ensemble_instance_edges.sum(axis=0)

    gray1 = (sum_instance / sum_instance.max() * 255).astype(np.uint8)
    rgb1 = cv2.cvtColor(gray1, cv2.COLOR_GRAY2RGB)
    gray2 = (sum_instance_edge / sum_instance_edge.max() * 255).astype(np.uint8)
    rgb2 = cv2.cvtColor(gray2, cv2.COLOR_GRAY2RGB)

    w, h, _ = rgb2.shape
    m = rgb2 > 0
    c = np.tile([0, 255, 0], [w, h, 1])
    i = image * (1 - m) + c * m

    all = np.hstack([image, i, rgb1])

    # save as train data
    # data = cv2.merge((image, gray1, gray2))
    multi_mask = instance_to_multi_mask(ensemble_instances)
    cv2.imwrite(os.path.join(out_dir, 'ensemble_data_overlays', '%s.png' % name), all)
    # cv2.imwrite(os.path.join(out_dir, 'ensemble_data', '%s.png' % name), data)
    # np.save(os.path.join(out_dir, 'ensemble_instances', '%s.npy' % name), ensemble_instances)
    np.save(os.path.join(out_dir, 'ensemble_masks', '%s.npy' % name), multi_mask)
    print('Done')

def ensemble_masks(multiprocess=True):
    cfg = Configuration()
    f_eval = TrainFolder(os.path.join(cfg.result_dir, cfg.model_name))
    out_dir = os.path.join(f_eval.folder_name, 'predict', 'ensemble_all')

    # setup ---------------------------------------
    os.makedirs(out_dir + '/ensemble_data_overlays', exist_ok=True)
    os.makedirs(out_dir + '/ensemble_data', exist_ok=True)
    os.makedirs(out_dir + '/ensemble_masks', exist_ok=True)

    split = cfg.valid_split  # 'test_black_white_53'
    ids = read_list_from_file(os.path.join(cfg.split_dir, split), comment='#')

    if multiprocess:
        pool = Pool()
        pool.map(ensemble_one_mask, [(cfg, f_eval.folder_name, img_id) for img_id in ids])
    else:
        for i, packed in enumerate([(cfg, f_eval.folder_name, img_id) for img_id in ids]):
            print('%04d'%i, end='')
            ensemble_one_mask(packed)


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    ensemble_masks(multiprocess=False)
    print('\nsucess!')