import os
import numpy as np
import cv2

from utility.file import read_list_from_file
from utility.draw import multi_mask_to_color_overlay, multi_mask_to_contour_overlay
from dataset.folder import SourceFolder, DataFolder
from configuration import Configuration


def run_make_test_annotation(cfg, norm=False):
    ids = read_list_from_file(os.path.join(cfg.split_dir, cfg.annotation_test_split), comment='#')
    s_test = SourceFolder(cfg.source_test_dir)
    d_test = DataFolder(os.path.join(cfg.data_dir, 'stage2_test_final'))
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


def run_make_extra_annotation(cfg, min_mask_threshold):
    ids = read_list_from_file(os.path.join(cfg.split_dir, cfg.annotation_extra_split), comment='#')

    s_extra = SourceFolder(cfg.source_extra_dir)
    d_extra = DataFolder(os.path.join(cfg.data_dir, 'extra_data'))

    extra_train_270 = []
    for row in ids:
        # load image and mask files
        folder = row.split('/')[0]
        name   = row.split('/')[-1]
        image = s_extra.get_image(name)
        multi_masks = s_extra.get_masks(name)

        for i in range(3):
            for j in range(3):
                x = i * 256
                y = j * 256
                img_idx = i * 3 + j
                image_crop = image[y:y + 256, x:x + 256]
                multi_mask_crop = multi_masks[y:y + 256, x:x + 256]
                mask_idx = np.unique(multi_mask_crop)
                assert (mask_idx[0] == 0)
                mask_idx = mask_idx[1:]

                re_assigned_multi_mask = np.zeros((256, 256), np.int32)
                for x in range(len(mask_idx)):
                    mask_size = np.sum(multi_mask_crop == mask_idx[x])
                    if mask_size < min_mask_threshold:
                        continue
                    re_assigned_multi_mask[np.where(multi_mask_crop == mask_idx[x])] = x + 1

                # check
                color_overlay   = multi_mask_to_color_overlay  (re_assigned_multi_mask, color='summer')
                color1_overlay  = multi_mask_to_contour_overlay(re_assigned_multi_mask, color_overlay, [255,255,255])
                contour_overlay = multi_mask_to_contour_overlay(re_assigned_multi_mask, image_crop, [0,255,0])
                stacked_overlay = np.hstack((image_crop, contour_overlay,color1_overlay,)).astype(np.uint8)
                
                # dump images and multi-masks
                extra_train_270.append('%s_%s'%(name, img_idx))
                cv2.imwrite(os.path.join(d_extra.image_folder,  '%s_%s.png' % (name, img_idx)), image_crop)
                cv2.imwrite(os.path.join(d_extra.mask_folder,   '%s_%s.png' % (name, img_idx)), color_overlay)
                np.save(    os.path.join(d_extra.mask_folder,   '%s_%s.npy' % (name, img_idx)), re_assigned_multi_mask)
                cv2.imwrite(os.path.join(d_extra.overlay_folder,'%s_%s.png' % (name, img_idx)), stacked_overlay)
                print('\r %s_%s' % (name, img_idx))
    print('run_make_extra_annotation success!')


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    cfg = Configuration()
    #run_make_train_annotation(cfg)
    run_make_test_annotation(cfg)
    #run_make_extra_annotation(cfg, 8)
    print('Done')
