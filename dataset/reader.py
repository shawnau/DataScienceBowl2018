import os
from timeit import default_timer as timer
import numpy as np
from numba import jit
import cv2
from torch.utils.data.dataset import Dataset
from utility.file import read_list_from_file
from net.lib.box.process import is_small_box_at_boundary, is_small_box, is_big_box


MIN_SIZE =  6
MAX_SIZE =  128  # np.inf
IGNORE_BOUNDARY = -1
IGNORE_SMALL    = -2
IGNORE_BIG      = -3


class ScienceDataset(Dataset):
    """
    train mode:
        :return:
        image: (H, W, C) numpy array
        multi_mask: a map records masks. e.g.
            [[0, 1, 1, 0],
             [2, 0, 0, 3],
             [2, 0, 3, 3]]
            for 3 masks in a 4*4 input
        meta: not used
        index: index of the image (unique)
    """
    def __init__(self, cfg, split, transform=None, mode='train'):
        super(ScienceDataset, self).__init__()
        start = timer()

        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.mode = mode

        # read split
        self.ids = read_list_from_file(os.path.join(self.cfg.split_dir, split), comment='#')

        # print
        print('\ttime = %0.2f min' % ((timer() - start) / 60))
        print('\tnum_ids = %d' % (len(self.ids)))
        print('')

    def __getitem__(self, index):
        folder_name = self.ids[index]
        name   = folder_name.split('/')[-1]
        folder = folder_name.split('/')[0]
        image_path = os.path.join(self.cfg.data_dir, folder, 'images', '%s.png' % name)
        image  = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if self.mode in ['train']:
            multi_mask_path = os.path.join(self.cfg.data_dir, folder, 'multi_masks', '%s.npy' % name)
            multi_mask = np.load(multi_mask_path).astype(np.int32)
            meta = '<not_used>'

            if self.transform is not None:
                return self.transform(image, multi_mask, meta, index)
            else:
                return image, multi_mask, meta, index

        if self.mode in ['test']:
            if self.transform is not None:
                return self.transform(image, index)
            else:
                return image, index

    def __len__(self):
        return len(self.ids)


def multi_mask_to_annotation(multi_mask):
    """
    :param multi_mask: a map records masks. e.g.
        [[0, 1, 1, 0],
         [2, 0, 0, 3],
         [2, 0, 3, 3]]
        for 3 masks in a 4*4 input
    :return:
        box: lists of diameter coords. e.g.
            [[x0, y0, x1, y1], ...]
        label: currently all labels are 1 (for foreground only)
        instance: list of one vs all masks. e.g.
            [[[0, 1, 1, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0]], ...]
            for thr first mask of all masks, a total of 3 lists in this case
    """
    H,W      = multi_mask.shape[:2]
    box      = []
    label    = []
    instance = []

    num_masks = multi_mask.max()
    for i in range(num_masks):
        mask = (multi_mask == (i+1))
        if mask.sum() > 1:

            y, x = np.where(mask)
            y0 = y.min()
            y1 = y.max()
            x0 = x.min()
            x1 = x.max()
            w = (x1-x0)+1
            h = (y1-y0)+1

            # border = max(1, round(0.1*min(w,h)))
            # border = 0
            border = max(2, round(0.2*(w+h)/2))

            x0 = x0-border
            x1 = x1+border
            y0 = y0-border
            y1 = y1+border

            # clip
            x0 = max(0,x0)
            y0 = max(0,y0)
            x1 = min(W-1,x1)
            y1 = min(H-1,y1)

            # label
            l = 1  # <todo> support multiclass later ... ?
            if is_small_box_at_boundary((x0,y0,x1,y1),W,H,MIN_SIZE):
                l = IGNORE_BOUNDARY
                continue  # completely ignore!
            elif is_small_box((x0,y0,x1,y1),MIN_SIZE):
                l = IGNORE_SMALL
                continue
            elif is_big_box((x0,y0,x1,y1),MAX_SIZE):
                l = IGNORE_BIG
                continue

            box.append([x0,y0,x1,y1])
            label.append(l)
            instance.append(mask)

    box      = np.array(box,np.float32)
    label    = np.array(label,np.float32)
    instance = np.array(instance,np.float32)

    if len(box)==0:
        box      = np.zeros((0,4),np.float32)
        label    = np.zeros((0,1),np.float32)
        instance = np.zeros((0,H,W),np.float32)

    return box, label, instance


def instance_to_multi_mask(instances):
    """
    :param
    instance: list of one vs all masks. e.g.
            [[0, 1, 1, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]], ...
    :return:
    multi_mask: a map records masks. e.g.
        [[0, 1, 1, 0],
         [2, 0, 0, 3],
         [2, 0, 3, 3]]
        for 3 masks in a 4*4 input
    """
    H,W = instances.shape[1:3]
    multi_mask = np.zeros((H,W),np.int32)

    # sort masks   
    instance_sizes = []
    num_masks = len(instances)
    for i in range(num_masks):
        instance = instances[i]
        instance_sizes.append((i, instance.sum()))
    sorted_sizes = sorted(instance_sizes, key=lambda tup: tup[1], reverse=True)
    
    for j, item in enumerate(sorted_sizes):
        multi_mask[instances[item[0]] > 0] = j+1

    return multi_mask

@jit
def multi_mask_to_instance(multi_mask):
    H, W = multi_mask.shape[:2]
    num_masks = len(np.unique(multi_mask))

    instances = []
    for i in range(num_masks):
        instance = (multi_mask == (i+1))
        if instance.sum() > 1:
            instances.append(instance)
    
    if len(instances) > 0:
        instances = np.array(instances, np.float32)
    else:
        instances = np.zeros((0, H, W), np.float32)
    return instances