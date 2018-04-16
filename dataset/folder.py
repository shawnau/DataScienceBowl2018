import os
from glob import glob
import numpy as np
import cv2


class SourceFolder:
    def __init__(self, folder_name):
        self.folder_name = folder_name

    def get_image(self, img_id, flags=cv2.IMREAD_COLOR):
        img_folder = os.path.join(self.folder_name, img_id, 'images')
        img_files = glob(os.path.join(img_folder, '*.png'))
        if len(img_files) == 0:
            img_files = glob(os.path.join(img_folder, '*.tif'))
        assert len(img_files) == 1
        img_file = img_files[0]

        return cv2.imread(img_file, flags)

    def get_masks(self, img_id):
        img = self.get_image(img_id)
        H, W, C = img.shape
        multi_mask = np.zeros((H, W), np.int32)

        mask_folder = os.path.join(self.folder_name, img_id, 'masks')
        mask_files = glob(os.path.join(mask_folder, '*.png'))
        mask_files.sort()

        for j in range(len(mask_files)):
            mask_file = mask_files[j]
            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            mh, mw = mask.shape
            assert (mh == H) and (mw == W)
            multi_mask[np.where(mask > 128)] = j+1

        return multi_mask


class DataFolder:
    def __init__(self, folder_name):
        self.folder_name = folder_name
        self.image_folder = os.path.join(self.folder_name, 'images')
        self.mask_folder = os.path.join(self.folder_name, 'multi_masks')
        self.overlay_folder = os.path.join(self.folder_name, 'overlays')

        os.makedirs(self.folder_name, exist_ok=True)
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.mask_folder, exist_ok=True)
        os.makedirs(self.overlay_folder, exist_ok=True)

    def get_image(self, img_id, flags=cv2.IMREAD_COLOR):
        # img_file = glob(os.path.join(self.image_folder, '%s.png')%img_id)
        # assert len(img_files) == 1
        # img_file = img_file[0]
        return cv2.imread(os.path.join(self.image_folder, '%s.png')%img_id, flags)

    def get_masks(self, img_id):
        return np.load(os.path.join(self.mask_folder, '%s.npy')%img_id).astype(np.int32)


class TrainFolder:
    def __init__(self, folder_name):
        """
        save model results
        :param folder_name: absolute path of model
        """
        self.folder_name = folder_name
        self.checkpoint_dir = os.path.join(self.folder_name, 'checkpoint')
        self.train_result = os.path.join(self.folder_name, 'train')
        self.backup = os.path.join(self.folder_name, 'backup')

        self.evaluate_dir = os.path.join(self.folder_name, 'evaluate')
        self.evaluate_npy_dir = os.path.join(self.evaluate_dir, 'npys')
        self.evaluate_overlay_dir = os.path.join(self.evaluate_dir, 'overlay')

        self.submit_dir = os.path.join(self.folder_name, 'submit')
        self.submit_npy_dir = os.path.join(self.submit_dir, 'npys')
        self.submit_instance_dir = os.path.join(self.submit_dir, 'post_instances')
        self.submit_masks_dir = os.path.join(self.submit_dir, 'post_multi_masks')
        self.submit_overlay_dir = os.path.join(self.submit_dir, 'overlay')
        self.submit_psds_dir = os.path.join(self.submit_dir, 'psds')

        os.makedirs(self.folder_name, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.train_result, exist_ok=True)
        os.makedirs(self.backup, exist_ok=True)

        os.makedirs(self.evaluate_dir, exist_ok=True)
        os.makedirs(self.evaluate_npy_dir, exist_ok=True)
        os.makedirs(self.evaluate_overlay_dir, exist_ok=True)

        os.makedirs(self.submit_dir, exist_ok=True)
        os.makedirs(self.submit_npy_dir, exist_ok=True)
        os.makedirs(self.submit_masks_dir, exist_ok=True)
        os.makedirs(self.submit_instance_dir, exist_ok=True)
        os.makedirs(self.submit_overlay_dir, exist_ok=True)
        os.makedirs(os.path.join(self.submit_dir, 'psds'), exist_ok=True)


if __name__ == '__main__':
    from utility.draw import multi_mask_to_color_overlay, multi_mask_to_contour_overlay

    src_root_dir = '/Users/Shawn/Downloads'
    s_train = SourceFolder(os.path.join(src_root_dir, 'stage1_train'))
    img_id = '0a7d30b252359a10fd298b638b90cb9ada3acced4e0c0e5a3692013f432ee4e9'
    img = s_train.get_image(img_id)
    multi_masks = s_train.get_masks(img_id)

    color_overlay = multi_mask_to_color_overlay(multi_masks, color='summer')
    color1_overlay = multi_mask_to_contour_overlay(multi_masks, color_overlay, [255, 255, 255])
    contour_overlay = multi_mask_to_contour_overlay(multi_masks, img, [0, 255, 0])
    all = np.hstack((img, contour_overlay, color1_overlay,)).astype(np.uint8)

    d_train = DataFolder(os.path.join(src_root_dir, 'data', 'stage1_train'))
    cv2.imwrite(os.path.join(d_train.image_folder, '%s.png'%img_id), img)
    cv2.imwrite(os.path.join(d_train.mask_folder, '%s.png'%img_id), color_overlay)
    np.save(os.path.join(d_train.mask_folder, '%s.npy' % img_id), multi_masks)
    cv2.imwrite(os.path.join(d_train.overlay_folder, '%s.png' % img_id), all)

