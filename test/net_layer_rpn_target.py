from dataset.reader import multi_mask_to_annotation
from dataset.transform import *

from net.draw import *


def check_make_one_rpn_target(input, window, truth_box, truth_label,
                              cfg,
                              label, label_assign, label_weight,
                              target, target_weight):
    image = input.data.cpu().numpy() * 255
    image = image.transpose((1, 2, 0)).astype(np.uint8).copy()

    all1 = draw_rpn_target_truth_box(image, truth_box, truth_label)
    all2 = draw_rpn_target_label(cfg, image, window, label, label_assign, label_weight)
    all3 = draw_rpn_target_target(cfg, image, window, target, target_weight)
    all4 = draw_rpn_target_target1(cfg, image, window, target, target_weight)

    image_show('all1', all1, 1)
    image_show('all2', all2, 1)
    image_show('all3', all3, 1)
    image_show('all4', all4, 1)
    cv2.waitKey(0)


def check_layer():
    image_id = '3ebd2ab34ba86e515feb79ffdeb7fc303a074a98ba39949b905dbde3ff4b7ec0'

    dir = '/root/share/project/kaggle/science2018/data/image/stage1_train'
    image_file = dir + '/' + image_id + '/images/' + image_id + '.png'
    npy_file   = dir + '/' + image_id + '/multi_mask.npy'

    multi_mask0 = np.load(npy_file)
    image0      = cv2.imread(image_file,cv2.IMREAD_COLOR)

    batch_size =4
    H,W = 256,256
    images = []
    multi_masks = []
    inputs = []
    boxes  = []
    labels = []
    instances = []
    for b in range(batch_size):
        image, multi_mask = random_crop_transform(image0, multi_mask0, W, H)
        box, label, instance = multi_mask_to_annotation(multi_mask)
        input = Variable(torch.from_numpy(image.transpose((2,0,1))).float().div(255)).cuda()

        label[[5,12,14,18]]=-1  #dummy ignore

        images.append(image)
        inputs.append(input)
        multi_masks.append(multi_mask)
        boxes.append(box)
        labels.append(label)
        instances.append(instance)

        # print information ---
        N = len(label)
        for n in range(N):
            print( '%d  :  %s  %d'%(n, box[n], label[n]),)
        print('')

    #dummy features
    in_channels = 256
    num_heads = 4
    feature_heights = [ int(H//2**i) for i in range(num_heads) ]
    feature_widths  = [ int(W//2**i) for i in range(num_heads) ]
    ps = []
    for h,w in zip(feature_heights,feature_widths):
        p = np.random.uniform(-1,1,size=(batch_size,in_channels,h,w)).astype(np.float32)
        p = Variable(torch.from_numpy(p)).cuda()
        ps.append(p)

    # check layer
    cfg = type('', (object,), {})() #Configuration() #default configuration
    cfg.rpn_num_heads  = num_heads
    cfg.rpn_num_bases  = 3
    cfg.rpn_base_sizes = [ 8, 16, 32, 64 ] #radius
    cfg.rpn_base_apsect_ratios = [1, 0.5,  2]
    cfg.rpn_strides    = [ 1,  2,  4,  8 ]



    cfg.rpn_train_bg_thresh_high = 0.5
    cfg.rpn_train_fg_thresh_low  = 0.5


    #start here --------------------------
    bases, windows = make_rpn_windows(cfg, ps)
    rpn_labels, rpn_label_assigns, rpn_label_weights, rpn_targets, rpn_targets_weights = \
        make_rpn_target(cfg, inputs, windows, boxes, labels)


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    check_layer()

    print('sucess')