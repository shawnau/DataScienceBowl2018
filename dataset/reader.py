import numpy

from net.lib.box.process import is_small_box_at_boundary, is_small_box, is_big_box
from utility.file import *

from net.lib.box.process import *


MIN_SIZE =  6
MAX_SIZE =  128  # np.inf
IGNORE_BOUNDARY = -1
IGNORE_SMALL    = -2
IGNORE_BIG      = -3


class ScienceDataset(Dataset):

    def __init__(self, split, transform=None, mode='train'):
        super(ScienceDataset, self).__init__()
        start = timer()

        self.split = split
        self.transform = transform
        self.mode = mode

        # read split
        self.ids = read_list_from_file(os.path.join(SPLIT_DIR, split), comment='#')

        # print
        print('\ttime = %0.2f min' % ((timer() - start) / 60))
        print('\tnum_ids = %d' % (len(self.ids)))
        print('')

    def __getitem__(self, index):
        id     = self.ids[index]
        name   = id.split('/')[-1]
        folder = id.split('/')[0]
        image_path = os.path.join(IMAGE_DIR, folder, 'images', '%s.png' % name)
        image  = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if self.mode in ['train']:
            multi_mask_path = os.path.join(IMAGE_DIR, folder, 'multi_masks', '%s.npy' % name)
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
            [[0, 1, 1, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
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

            border = max(2, round(0.2*(w+h)/2))
            #border = max(1, round(0.1*min(w,h)))
            #border = 0
            x0 = x0-border
            x1 = x1+border
            y0 = y0-border
            y1 = y1+border

            #clip
            x0 = max(0,x0)
            y0 = max(0,y0)
            x1 = min(W-1,x1)
            y1 = min(H-1,y1)

            #label
            l = 1 #<todo> support multiclass later ... ?
            if is_small_box_at_boundary((x0,y0,x1,y1),W,H,MIN_SIZE):
                l = IGNORE_BOUNDARY
                continue  #completely ignore!
            elif is_small_box((x0,y0,x1,y1),MIN_SIZE):
                l = IGNORE_SMALL
                continue
            elif is_big_box((x0,y0,x1,y1),MAX_SIZE):
                l = IGNORE_BIG
                continue

            # add --------------------
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