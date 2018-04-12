import os, sys
sys.path.append('..')

from configuration import Configuration
from dataset.folder import TrainFolder
from utility.file import Logger
from net.model import MaskRcnnNet
from postprocess.augments import *



def run_predict_mask_only():
    cfg = Configuration()
    f_eval = TrainFolder(os.path.join(cfg.result_dir, cfg.model_name))
    out_dir = f_eval.folder_name
    initial_checkpoint = os.path.join(f_eval.checkpoint_dir, cfg.valid_checkpoint)

    propsal_dir = os.path.join(out_dir, 'predict', 'box_ensemble', 'proposal', 'npys')

    # augment -----------------------------------------------------------------------------------------------------
    split = cfg.valid_split#'valid_black_white_44' #'valid_test'#'test_black_white_53'


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

    for tag_name, do_test_augment, undo_test_augment, params in cfg.test_augments:

        ## setup  --------------------------
        tag = 'mask_ensemble_%s'%tag_name   ##tag = 'test1_ids_gray2_53-00011000_model'
        os.makedirs(os.path.join(out_dir, 'predict', tag, 'overlays'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'predict', tag, 'predicts'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'predict', tag, 'rcnn_proposals'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'predict', tag, 'detections'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'predict', tag, 'multi_masks'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'predict', tag, 'instances'), exist_ok=True)


        log.write('** start evaluation here @%s! **\n'%tag)
        for i in range(len(ids)):
            folder, name = ids[i].split('/')[-2:]
            print('%03d %s'%(i,name))

            image    = cv2.imread(os.path.join(cfg.data_dir, folder, 'images', '%s.png' % name), cv2.IMREAD_COLOR)
            proposal = np.load(os.path.join(propsal_dir, '%s.npy'%name))
            ## augment --------------------------------------
            augment_image, augment_propsal = do_test_augment(image, proposal,  **params)

            net.set_mode('test')
            with torch.no_grad():
                input = torch.from_numpy(augment_image.transpose((2,0,1))).float().div(255).unsqueeze(0)
                input = Variable(input).cuda()
                rcnn_proposal = Variable(torch.from_numpy(augment_propsal)).cuda()
                net.forward_mask(input,rcnn_proposal)

            rcnn_proposal, detection, mask, instance  = undo_test_augment(net, image, **params)

            ##save results ---------------------------------------
            np.save(os.path.join(out_dir, 'predict', tag, 'multi_masks',    '%s.npy'%name),mask)
            np.save(os.path.join(out_dir, 'predict', tag, 'detections',     '%s.npy'%name),detection)
            #np.save(os.path.join(out_dir, 'predict', tag, 'rcnn_proposals', '%s.npy'%name),rcnn_proposal)
            np.save(os.path.join(out_dir, 'predict', tag, 'instances',      '%s.npy'%name),instance)

            if 0:
                threshold = cfg.rcnn_test_nms_pre_score_threshold  #0.8
                #all1 = draw_predict_proposal(threshold, image, rcnn_proposal)
                all2 = draw_predict_mask(threshold, image, mask, detection)

                ## save
                #cv2.imwrite(os.path.join(out_dir, 'predict', tag, 'predicts', '%s.png'%name), all2)
                # image_show('predict_mask',all2)

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
                    #cv2.imwrite(os.path.join(out_dir, 'predict', tag, 'overlays', '%s.png' % name), all)

                    #os.makedirs(os.path.join(out_dir, 'predict', tag, 'overlays', name), exist_ok=True)
                    #cv2.imwrite(os.path.join(out_dir, 'predict', tag, 'overlays', name, "%s.png" % name), image)
                    #cv2.imwrite(os.path.join(out_dir, 'predict', tag, 'overlays', name, "%s.mask.png" % name), color_overlay)
                    #cv2.imwrite(os.path.join(out_dir, 'predict', tag, 'overlays', name, "%s.contour.png" % name), contour_overlay)

        #assert(test_num == len(test_loader.sampler))
        log.write('-------------\n')
        log.write('initial_checkpoint  = %s\n'%(initial_checkpoint))
        log.write('tag=%s\n'%tag)
        log.write('\n')


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_predict_mask_only()

    print('\nsucess!')