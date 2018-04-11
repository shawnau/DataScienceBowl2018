import os, sys
sys.path.append('..')

from configuration import Configuration
from dataset.folder import TrainFolder
from utility.file import Logger
from net.model import MaskRcnnNet
from postprocess.augments import *



def run_predict():
    cfg = Configuration()
    f_eval = TrainFolder(os.path.join(cfg.result_dir, cfg.model_name))
    out_dir = f_eval.folder_name
    initial_checkpoint = os.path.join(f_eval.checkpoint_dir, cfg.valid_checkpoint)

    # augment -----------------------------------------------------------------------------------------------------
    augments=[
        ('normal',           do_test_augment_identity,       undo_test_augment_identity,       {         } ),
        ('flip_transpose_1', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':1,} ),
        ('flip_transpose_2', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':2,} ),
        ('flip_transpose_3', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':3,} ),
        ('flip_transpose_4', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':4,} ),
        ('flip_transpose_5', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':5,} ),
        ('flip_transpose_6', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':6,} ),
        ('flip_transpose_7', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type':7,} ),
        ('scale_0.8',        do_test_augment_scale,  undo_test_augment_scale,     { 'scale_x': 0.8, 'scale_y': 0.8  } ),
        ('scale_1.2',        do_test_augment_scale,  undo_test_augment_scale,     { 'scale_x': 1.2, 'scale_y': 1.2  } ),
        ('scale_0.5',        do_test_augment_scale,  undo_test_augment_scale,     { 'scale_x': 0.5, 'scale_y': 0.5  } ),
        ('scale_1.8',        do_test_augment_scale,  undo_test_augment_scale,     { 'scale_x': 1.8, 'scale_y': 1.8  } ),
    ]

    split = cfg.valid_split#'valid_black_white_44'#'test_black_white_53' # 


    #start experiments here! ###########################################################
    log = Logger()
    log.open(out_dir+'/log.evaluate.txt',mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')


    ## net ------------------------------
    cfg.rcnn_test_nms_pre_score_threshold = 0.5
    cfg.mask_test_nms_pre_score_threshold = cfg.rcnn_test_nms_pre_score_threshold

    net = MaskRcnnNet(cfg).cuda()
    if initial_checkpoint is not None:
        log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n\n'%(type(net)))
    log.write('\n')


    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    ids = read_list_from_file(os.path.join(cfg.split_dir, split), comment='#')
    log.write('\ttsplit   = %s\n'%(split))
    log.write('\tlen(ids) = %d\n'%(len(ids)))
    log.write('\n')


    for tag_name, do_test_augment, undo_test_augment, params in augments:

        ## setup  --------------------------
        tag = 'box_%s'%tag_name
        os.makedirs(os.path.join(out_dir, 'predict', tag, 'overlays'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'predict', tag, 'predicts'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'predict', tag, 'rcnn_proposals'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'predict', tag, 'detections'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'predict', tag, 'masks'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'predict', tag, 'instances'), exist_ok=True)


        log.write('** start evaluation here @%s! **\n'%tag)
        for i in range(len(ids)):
            folder, name = ids[i].split('/')[-2:]
            print('%03d %s'%(i,name))
            image = cv2.imread(os.path.join(cfg.data_dir, folder, 'images', '%s.png' % name), cv2.IMREAD_COLOR)
            ## augment --------------------------------------
            augment_image  = do_test_augment(image, proposal=None,  **params)

            net.set_mode('test')
            with torch.no_grad():
                input = torch.from_numpy(augment_image.transpose((2,0,1))).float().div(255).unsqueeze(0)
                input = Variable(input).cuda()
                net.forward(input)

            rcnn_proposal, detection, mask, instance  = undo_test_augment(net, image, **params)

            ## save results ---------------------------------------
            #np.save(os.path.join(out_dir, 'predict', tag, 'rcnn_proposals', '%s.npy'%name),rcnn_proposal)
            #np.save(os.path.join(out_dir, 'predict', tag, 'masks',          '%s.npy'%name),mask)
            np.save(os.path.join(out_dir, 'predict', tag, 'detections',     '%s.npy'%name),detection)
            #np.save(os.path.join(out_dir, 'predict', tag, 'instances',      '%s.npy'%name),instance)

            if 1:
                threshold = cfg.rcnn_test_nms_pre_score_threshold
                all2 = draw_predict_mask(threshold, image, mask, detection)

                ## save
                #cv2.imwrite(os.path.join(out_dir, 'predict', tag, 'predicts', '%s.png'%name), all2)

                if 1:
                    color_overlay   = multi_mask_to_color_overlay(mask)
                    color1_overlay  = multi_mask_to_contour_overlay(mask, color_overlay)
                    contour_overlay = multi_mask_to_contour_overlay(mask, image, [0,255,0])

                    mask_score = instance.sum(0)
                    #mask_score = cv2.cvtColor((np.clip(mask_score,0,1)*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
                    mask_score = cv2.cvtColor((mask_score/mask_score.max()*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)

                    all = np.hstack((image, contour_overlay, color1_overlay, mask_score)).astype(np.uint8)
                    #image_show('overlays',all)

                    #psd
                    #os.makedirs(os.path.join(out_dir, 'predict', 'overlays'), exist_ok=True)
                    #cv2.imwrite(os.path.join(out_dir, 'predict', tag, 'overlays', '%s.png'%name),all)

                    #os.makedirs(os.path.join(out_dir, 'predict', tag, 'overlays', name), exist_ok=True)
                    #cv2.imwrite(os.path.join(out_dir, 'predict', tag, 'overlays', name, "%s.png"%name),image)
                    #cv2.imwrite(os.path.join(out_dir, 'predict', tag, 'overlays', name, "%s.mask.png"%name),color_overlay)
                    #cv2.imwrite(os.path.join(out_dir, 'predict', tag, 'overlays', name, "%s.contour.png"%name),contour_overlay)

        #assert(test_num == len(test_loader.sampler))
        log.write('-------------\n')
        log.write('initial_checkpoint  = %s\n'%(initial_checkpoint))
        log.write('tag=%s\n'%tag)
        log.write('\n')

5
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_predict()

    print('\nsucess!')