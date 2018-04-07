import sys
from utility.draw import *
from utility.metric import compute_average_precision_for_mask

sys.path.append(os.path.dirname(__file__))
from train import *


def eval_augment(image, multi_mask, meta, index):
    pad_image = pad_to_factor(image, factor=16)
    input = torch.from_numpy(pad_image.transpose((2,0,1))).float().div(255)
    box, label, instance = multi_mask_to_annotation(multi_mask)
    return input, box, label, instance, meta, image, index


def eval_collate(batch):
    batch_size = len(batch)
    inputs    = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    boxes     =             [batch[b][1]for b in range(batch_size)]
    labels    =             [batch[b][2]for b in range(batch_size)]
    instances =             [batch[b][3]for b in range(batch_size)]
    metas     =             [batch[b][4]for b in range(batch_size)]
    images    =             [batch[b][5]for b in range(batch_size)]
    indices   =             [batch[b][6]for b in range(batch_size)]

    return [inputs, boxes, labels, instances, metas, images, indices]


def run_evaluate():

    cfg = Configuration()
    work_dir = os.path.join(cfg.result_dir, cfg.model_name)
    f = TrainFolder(work_dir)
    initial_checkpoint = os.path.join(f.checkpoint_dir, cfg.valid_checkpoint)

    log = Logger()
    log.open(work_dir+'/log.evaluate.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % work_dir)
    log.write('\n')

    # net ------------------------------
    # cfg.rpn_train_nms_pre_score_threshold = 0.8 #0.885#0.5
    # cfg.rpn_test_nms_pre_score_threshold  = 0.8 #0.885#0.5
    net = MaskRcnnNet(cfg).cuda()
    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n\n'%(type(net)))
    log.write('\n')

    # dataset ----------------------------------------
    log.write('** dataset setting **\n')

    test_dataset = ScienceDataset(cfg, cfg.valid_split, mode='train', transform=eval_augment)
    test_loader  = DataLoader(
                        test_dataset,
                        sampler = SequentialSampler(test_dataset),
                        batch_size  = 1,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = eval_collate)

    log.write('\ttest_dataset.split = %s\n'%(test_dataset.split))
    log.write('\tlen(test_dataset)  = %d\n'%(len(test_dataset)))
    log.write('\n')

    # start evaluation here!
    log.write('** start evaluation here! **\n')
    mask_average_precisions = []
    box_precisions_50 = []

    test_num  = 0
    test_loss = np.zeros(5, np.float32)

    for i, (inputs, truth_boxes, truth_labels, truth_instances, metas, images, indices) in enumerate(test_loader, 0):
        if all((truth_label > 0).sum() == 0 for truth_label in truth_labels):
            continue

        net.set_mode('test')
        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            net(inputs, truth_boxes,  truth_labels, truth_instances)
        # save results ---------------------------------------
        batch_size = len(indices)
        assert(batch_size == 1)  # currently support batch_size==1 only
        batch_size,C,H,W = inputs.size()
        masks      = net.masks
        detections = net.detections.cpu().numpy()

        for b in range(batch_size):
            image  = images[b]
            mask = masks[b]

            index = np.where(detections[:, 0] == b)[0]
            detection = detections[index]
            box = detection[:,1:5]

            truth_mask = instance_to_multi_mask(truth_instances[b])
            truth_box  = truth_boxes[b]
            truth_label= truth_labels[b]
            truth_instance= truth_instances[b]

            mask_average_precision, mask_precision = \
                compute_average_precision_for_mask(mask, truth_mask, t_range=np.arange(0.5, 1.0, 0.05))

            box_precision, box_recall, box_result, truth_box_result = \
                compute_precision_for_box(box, truth_box, truth_label, threshold=[0.5])

            box_precision = box_precision[0]

            mask_average_precisions.append(mask_average_precision)
            box_precisions_50.append(box_precision)

            # print results --------------------------------------------
            img_id = test_dataset.ids[indices[b]]
            name = img_id.split('/')[-1]
            print('%d\t%s\t%0.5f  (%0.5f)' % (i, name, mask_average_precision, box_precision))

            # save data
            np.save(os.path.join(f.evaluate_npy_dir, '%s.npy' % name), mask)

            # draw prediction ------------------------------------------
            contour_overlay  = multi_mask_to_contour_overlay(mask, image, color=[0,255,0])
            color_overlay    = multi_mask_to_color_overlay(mask, color='summer')
            color1_overlay   = multi_mask_to_contour_overlay(mask, color_overlay, color=[255,255,255])

            all1 = np.hstack((image, contour_overlay, color1_overlay))

            all6 = draw_multi_proposal_metric(cfg, image, detection,
                                              truth_box, truth_label,
                                              [0,255,255], [255,0,255], [255,255,0])
            all7 = draw_mask_metric(cfg, image, mask,
                                    truth_box, truth_label, truth_instance)

            cv2.imwrite(os.path.join(f.evaluate_overlay_dir, '%s.png' % name), all1)
            cv2.imwrite(os.path.join(f.evaluate_overlay_dir, '%s.png' % name), all6)
            cv2.imwrite(os.path.join(f.evaluate_overlay_dir, '%s.png' % name), all7)

        test_num += batch_size

    # assert(test_num == len(test_loader.sampler))
    test_loss = test_loss/test_num

    log.write('initial_checkpoint  = %s\n'%(initial_checkpoint))
    log.write('test_loss = %0.5f\n'%(test_loss[0]))
    log.write('test_num  = %d\n'%(test_num))
    log.write('\n')

    mask_average_precisions = np.array(mask_average_precisions)
    box_precisions_50 = np.array(box_precisions_50)
    log.write('-------------\n')
    log.write('mask_average_precision = %0.5f\n' % mask_average_precisions.mean())
    log.write('box_precision@0.5 = %0.5f\n' % box_precisions_50.mean())
    log.write('\n')


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    run_evaluate()
    print('\nsucess!')

