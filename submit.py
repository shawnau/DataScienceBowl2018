import os
import sys
from utility.draw import multi_mask_to_color_overlay, multi_mask_to_contour_overlay
from dataset.folder import DataFolder
from net.lib.box.process import torch_clip_proposals

sys.path.append(os.path.dirname(__file__))

from train import *


# https://www.kaggle.com/c/data-science-bowl-2018/discussion/48561#
def run_length_encode(x):
    bs = np.where(x.T.flatten())[0]

    rle = []
    prev = -2
    for b in bs:
        if b > prev+1:
            rle.extend((b + 1, 0))
        rle[-1] += 1
        prev = b

    if len(rle) != 0 and rle[-1]+rle[-2] == x.size:
        rle[-2] = rle[-2] - 1

    rle = ' '.join([str(r) for r in rle])
    return rle


def run_length_decode(rle, H, W, fill_value=255):

    mask = np.zeros((H * W), np.uint8)
    rle = np.array([int(s) for s in rle.split(' ')]).reshape(-1, 2)
    for r in rle:
        start = r[0]-1
        end = start + r[1]
        mask[start : end] = fill_value
    mask = mask.reshape(W, H).T # H, W need to swap as transposing.
    return mask


cfg = Configuration()
work_dir = os.path.join(cfg.result_dir, cfg.model_name)
f_submit = TrainFolder(work_dir)


rows = read_list_from_file(os.path.join(cfg.split_dir, cfg.submit_split), comment='#')
all_test_ids = [x.split('/')[-1] for x in rows]


# overwrite functions
def revert(net, images):
    #undo test-time-augmentation (e.g. unpad or scale back to input image size, etc)
    batch_size = len(images)
    for b in range(batch_size):
        image  = images[b]
        height,width  = image.shape[:2]
        index = (net.detections[:,0]==b).nonzero().view(-1)
        net.detections = torch_clip_proposals(net.detections, index, width, height)

        net.masks[b] = net.masks[b][:height,:width]

    return net, image


def submit_augment(image, index):
    pad_image = pad_to_factor(image, factor=16)
    input = torch.from_numpy(pad_image.transpose((2,0,1))).float().div(255)
    return input, image, index


def submit_collate(batch):

    batch_size = len(batch)
    inputs    = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    images    =             [batch[b][1]for b in range(batch_size)]
    indices   =             [batch[b][2]for b in range(batch_size)]

    return [inputs, images, indices]


def run_submit():

    # setup  ---------------------------
    initial_checkpoint = os.path.join(f_submit.checkpoint_dir, cfg.submit_checkpoint)

    os.makedirs(os.path.join(work_dir, 'submit', 'overlays'), exist_ok=True)
    os.makedirs(os.path.join(work_dir, 'submit', 'npys'), exist_ok=True)

    log = Logger()
    log.open(os.path.join(work_dir, 'log.submit.txt'), mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % work_dir)
    log.write('\n')

    # net ----------------------------------------
    net = MaskRcnnNet(cfg).cuda()

    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    net.load_state_dict(torch.load(initial_checkpoint,
                                   map_location=lambda storage, loc: storage))

    log.write('%s\n\n'%(type(net)))
    log.write('\n')

    # dataset ----------------------------------------
    log.write('** dataset setting **\n')

    test_dataset = ScienceDataset(cfg, cfg.submit_split, mode='test', transform=submit_augment)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler     = SequentialSampler(test_dataset),
                        batch_size  = 1,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = submit_collate)

    log.write('\ttest_dataset.split = %s\n'%(test_dataset.split))
    log.write('\tlen(test_dataset)  = %d\n'%(len(test_dataset)))
    log.write('\n')

    # start evaluation here! ----------------------------------------
    log.write('** start evaluation here! **\n')
    start = timer()

    test_num = len(test_loader.dataset)
    for i, (inputs, images, indices) in enumerate(test_loader, 0):

        print('\rpredicting: %10d/%d (%0.0f %%)  %0.2f min'%(i, test_num-1, 100*i/(test_num-1),
                         (timer() - start) / 60), end='',flush=True)
        time.sleep(0.01)

        net.set_mode('test')
        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            net(inputs)

        # save results ---------------------------------------
        batch_size = len(indices)
        assert(batch_size == 1)
        revert(net, images)

        batch_size, C, H, W = inputs.size()
        masks = net.masks

        for b in range(batch_size):
            # image0 = (inputs[b].transpose((1,2,0))*255).astype(np.uint8)
            image = images[b]
            mask = masks[b]

            contour_overlay  = multi_mask_to_contour_overlay(mask, image, color=[0,255,0])
            color_overlay    = multi_mask_to_color_overlay(mask, color='summer')
            color1_overlay   = multi_mask_to_contour_overlay(mask, color_overlay, color=[255,255,255])
            all_overlays = np.hstack((image,contour_overlay,color1_overlay))

            img_id = test_dataset.ids[indices[b]]
            name =img_id.split('/')[-1]

            np.save(os.path.join(f_submit.submit_npy_dir, '%s.npy'%name), mask)
            cv2.imwrite(os.path.join(f_submit.submit_overlay_dir, '%s.png'%name), all_overlays)

            # save psd
            cv2.imwrite(os.path.join(f_submit.submit_psds_dir, '%s.png'%name), image)
            cv2.imwrite(os.path.join(f_submit.submit_psds_dir, '%s.mask.png'%name), color_overlay)
            cv2.imwrite(os.path.join(f_submit.submit_psds_dir, '%s.contour.png'%name), contour_overlay)
            print('save %s'%name[:4])

    assert(test_num == len(test_loader.sampler))

    log.write('initial_checkpoint  = %s\n'%(initial_checkpoint))
    log.write('test_num  = %d\n'%(test_num))
    log.write('\n')


def run_npy_to_sumbit_csv():
    """
    post process & create csv file
    """
    # load data from image directory
    f_data = DataFolder(os.path.join(cfg.data_dir, 'stage1_test'))
    # load masks from submit directory
    csv_file = os.path.join(f_submit.folder_name, cfg.submit_csv_name)

    all_num = 0
    cvs_ImageId = []
    cvs_EncodedPixels = []

    npy_files = glob.glob(f_submit.submit_npy_dir + '/*.npy')
    ids = [npy_file.split('/')[-1].replace('.npy', '') for npy_file in npy_files]
    for npy_file in npy_files:
        name = npy_file.split('/')[-1].replace('.npy', '')

        multi_mask = np.load(npy_file)

        # <todo> ---------------------------------
        # post process here
        # multi_mask = filter_small(multi_mask, 8)

        num_masks = int(multi_mask.max())
        for m in range(num_masks):
            rle = run_length_encode(multi_mask == m+1)
            cvs_ImageId.append(name)
            cvs_EncodedPixels.append(rle)
        all_num += num_masks

        # debug ------------------------------------
        print(all_num, num_masks)  # GT is 4152?
        image_file = os.path.join(f_data.image_folder, '%s.png'%name)
        image = cv2.imread(image_file)
        color_overlay   = multi_mask_to_color_overlay(multi_mask)
        color1_overlay  = multi_mask_to_contour_overlay(multi_mask, color_overlay)
        contour_overlay = multi_mask_to_contour_overlay(multi_mask, image, [0,255,0])
        all_contour = np.hstack((image, contour_overlay, color1_overlay)).astype(np.uint8)

        cv2.imwrite(os.path.join(f_submit.submit_overlay_dir, '%s.png' % id), all_contour)
    # submission csv  ----------------------------

    # kaggle submission requires all test image to be listed!
    for t in all_test_ids:
        if t not in ids:
            cvs_ImageId.append(t)
            cvs_EncodedPixels.append('')  # null

    df = pd.DataFrame({'ImageId': cvs_ImageId, 'EncodedPixels': cvs_EncodedPixels})
    df.to_csv(csv_file, index=False, columns=['ImageId', 'EncodedPixels'])


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_submit()
    run_npy_to_sumbit_csv()
    print('\nsucess!')
