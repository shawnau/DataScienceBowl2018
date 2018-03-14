## evaluate post process here ####-------------------------------------
# def run_evaluate_map():
#
#     out_dir = RESULTS_DIR + '/mask-rcnn-gray-011b-drop1'
#     split   = 'valid1_ids_gray_only_43'
#
#     #------------------------------------------------------------------
#     log = Logger()
#     log.open(out_dir+'/log.evaluate.txt',mode='a')
#
#     #os.makedirs(out_dir +'/eval/'+split+'/label', exist_ok=True)
#     #os.makedirs(out_dir +'/eval/'+split+'/final', exist_ok=True)
#
#
#     image_files = glob.glob(out_dir + '/submit/npys/*.png')
#     image_files.sort()
#
#     average_precisions = []
#     for image_file in image_files:
#         #image_file = image_dir + '/0a849e0eb15faa8a6d7329c3dd66aabe9a294cccb52ed30a90c8ca99092ae732.png'
#
#         name  = image_file.split('/')[-1].replace('.png','')
#
#         image   = cv2.imread(DATA_DIR + '/image/stage1_train/' + name + '/images/' + name +'.png')
#         truth   = np.load(DATA_DIR    + '/image/stage1_train/' + name + '/multi_mask.npy').astype(np.int32)
#         predict = np.load(out_dir     + '/submit/npys/' + name + '.npy').astype(np.int32)
#         assert(predict.shape == truth.shape)
#         assert(predict.shape[:2] == image.shape[:2])
#
#
#         #image_show('image',image)
#         #image_show('mask',mask)
#         #cv2.waitKey(0)
#
#
#         #baseline labeling  -------------------------
#
#
#         # fill hole, file small, etc ...
#         # label = filter_small(label, threshold=15)
#
#
#         average_precision, precision = compute_average_precision(predict, truth)
#         average_precisions.append(average_precision)
#
#         #save and show  -------------------------
#         print(average_precision)
#
#         # overlay = (skimage.color.label2rgb(label, bg_label=0, bg_color=(0, 0, 0))*255).astype(np.uint8)
#         # cv2.imwrite(out_dir +'/eval/'+split+'/label/' + name + '.png',overlay)
#         # np.save    (out_dir +'/eval/'+split+'/label/' + name + '.npy',label)
#
#
#         # overlay1 = draw_label_contour (image, label )
#         # mask  = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
#         # final = np.hstack((image, overlay1, overlay, mask))
#         # final = final.astype(np.uint8)
#         # cv2.imwrite(out_dir +'/eval/'+split+'/final/' + name + '.png',final)
#         #
#         #
#         # image_show('image',image)
#         # image_show('mask',mask)
#         # image_show('overlay',overlay)
#         # cv2.waitKey(1)
#
#     ##----------------------------------------------
#     average_precisions = np.array(average_precisions)
#     log.write('-------------\n')
#     log.write('average_precision = %0.5f\n'%average_precisions.mean())
#     log.write('\n')