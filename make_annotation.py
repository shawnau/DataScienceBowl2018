import os
import numpy as np
import cv2

from utility.file import read_list_from_file
from dataset.annotate import multi_mask_to_color_overlay, multi_mask_to_contour_overlay
from dataset.folder import SourceFolder, DataFolder
from configuration import Configuration


def run_make_test_annotation(cfg):
    ids = read_list_from_file(os.path.join(cfg.split_dir, cfg.annotation_test_split), comment='#')
    s_test = SourceFolder(cfg.source_test_dir)
    d_test = DataFolder(os.path.join(cfg.data_dir, 'stage1_test'))
    for i in range(len(ids)):
        folder = ids[i].split('/')[0]
        name   = ids[i].split('/')[1]
        image = s_test.get_image(name)

        # show and save into image folder_name
        cv2.imwrite(os.path.join(d_test.image_folder, '%s.png' % name), image)
        print('\rannotate: ', i)

    print('run_make_test_annotation success!')


def run_make_train_annotation(cfg):
    ids = read_list_from_file(os.path.join(cfg.split_dir, cfg.annotation_train_split), comment='#')

    s_train = SourceFolder(cfg.source_train_dir)
    d_train = DataFolder(os.path.join(cfg.data_dir, 'stage1_train'))

    for i in range(len(ids)):
        # load image and mask files
        id = ids[i]

        folder = id.split('/')[0]
        name   = id.split('/')[-1]
        image = s_train.get_image(name)
        multi_masks = s_train.get_masks(name)

        # check
        color_overlay   = multi_mask_to_color_overlay  (multi_masks, color='summer')
        color1_overlay  = multi_mask_to_contour_overlay(multi_masks, color_overlay, [255,255,255])
        contour_overlay = multi_mask_to_contour_overlay(multi_masks, image, [0,255,0])
        stacked_overlay = np.hstack((image, contour_overlay,color1_overlay,)).astype(np.uint8)

        # dump images and multi-masks
        cv2.imwrite(os.path.join(d_train.image_folder,  '%s.png' % name), image)
        cv2.imwrite(os.path.join(d_train.mask_folder,   '%s.png' % name), color_overlay)
        np.save(    os.path.join(d_train.mask_folder,   '%s.npy' % name), multi_masks)
        cv2.imwrite(os.path.join(d_train.overlay_folder,'%s.png' % name), stacked_overlay)
        print('\rannotate: ', i)

    print('run_make_train_annotation success!')


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    cfg = Configuration()
    run_make_train_annotation(cfg)
    run_make_test_annotation(cfg)
    print('Done')
