from dataset.annotate import multi_mask_to_color_overlay, multi_mask_to_contour_overlay
from dataset.reader import *
from configuration import Configuration


def run_make_test_annotation(cfg):
    ids = read_list_from_file(os.path.join(SPLIT_DIR, cfg.annotation_test_split), comment='#')
    data_dir = os.path.join(IMAGE_DIR, 'stage1_test')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)

    for i in range(len(ids)):
        # load from download folder
        folder = ids[i].split('/')[0]
        name   = ids[i].split('/')[1]
        image_path = os.path.join(DOWNLOAD_DIR, '%s/%s/images/%s.png' % (folder, name, name))
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # show and save into image folder
        cv2.imwrite(os.path.join(data_dir, 'images/%s.png' % name), image)
        print('\rannotate: ', i)

    print('run_make_test_annotation success!')


def run_make_train_annotation(cfg):
    ids = read_list_from_file(os.path.join(SPLIT_DIR, cfg.annotation_train_split), comment='#')

    data_dir = os.path.join(IMAGE_DIR, 'stage1_train')
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'multi_masks'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'overlays'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'images'), exist_ok=True)

    for i in range(len(ids)):
        # load image and mask files
        id = ids[i]

        folder = id.split('/')[0]
        name   = id.split('/')[-1]
        image_files = glob.glob(os.path.join(DOWNLOAD_DIR, '%s/%s/images/*.png' % (folder, name)))
        assert(len(image_files) == 1)
        image_file = image_files[0]

        # image
        image = cv2.imread(image_file, cv2.IMREAD_COLOR)
        H, W, C = image.shape
        multi_mask = np.zeros((H, W), np.int32)

        mask_files = glob.glob(os.path.join(DOWNLOAD_DIR, '%s/%s/masks/*.png' % (folder, name)))
        mask_files.sort()
        for j in range(len(mask_files)):
            mask_file = mask_files[j]
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            multi_mask[np.where(mask > 128)] = j+1

        # check
        color_overlay   = multi_mask_to_color_overlay  (multi_mask, color='summer')
        color1_overlay  = multi_mask_to_contour_overlay(multi_mask, color_overlay, [255,255,255])
        contour_overlay = multi_mask_to_contour_overlay(multi_mask, image, [0,255,0])
        all = np.hstack((image, contour_overlay,color1_overlay,)).astype(np.uint8)

        # dump images and multi-masks
        cv2.imwrite(os.path.join(data_dir, 'images', '%s.png' % name), image)
        cv2.imwrite(os.path.join(data_dir, 'multi_masks', '%s.png' % name), color_overlay)
        np.save(    os.path.join(data_dir, 'multi_masks', '%s.npy' % name), multi_mask)
        cv2.imwrite(os.path.join(data_dir, 'overlays', '%s.png' % name), all)
        print('\rannotate: ', i)

    print('run_make_train_annotation success!')


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    cfg = Configuration()
    run_make_train_annotation(cfg)
    run_make_test_annotation(cfg)
    print('Done')
