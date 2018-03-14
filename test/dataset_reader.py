from utility.draw import *
from dataset.annotate import multi_mask_to_color_overlay, multi_mask_to_contour_overlay
from dataset.reader import *


def run_check_dataset_reader():

    def augment(image, multi_mask, meta, index):
        box, label, instance = multi_mask_to_annotation(multi_mask)
        return image, multi_mask, box, label, instance, meta, index

    dataset = ScienceDataset(
        'train1_ids_gray2_500', mode='train',
        #'disk0_ids_dummy_9', mode='train',
        #'merge1_1', mode='train',
        transform = augment,
    )
    #sampler = SequentialSampler(dataset)
    sampler = RandomSampler(dataset)


    for n in iter(sampler):
    #for n in range(10):
    #n=0
    #while 1:
        image, multi_mask, box, label, instance, meta, index = dataset[n]

        print('n=%d------------------------------------------'%n)
        print('meta : ', meta)

        contour_overlay = multi_mask_to_contour_overlay(multi_mask,image,color=[0,0,255])
        color_overlay   =   multi_mask_to_color_overlay(multi_mask)
        image_show('image',np.hstack([image,color_overlay,contour_overlay]))

        num_masks  = len(instance)
        for i in range(num_masks):
            x0,y0,x1,y1 = box[i]
            print('label[i], box[i] : ', label[i], box[i])

            instance1 = cv2.cvtColor((instance[i]*255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
            image1           = image.copy()
            color_overlay1   = color_overlay.copy()
            contour_overlay1 = contour_overlay.copy()

            cv2.rectangle(instance1,(x0,y0),(x1,y1),(0,255,255),2)
            cv2.rectangle(image1,(x0,y0),(x1,y1),(0,255,255),2)
            cv2.rectangle(color_overlay1,(x0,y0),(x1,y1),(0,255,255),2)
            cv2.rectangle(contour_overlay1,(x0,y0),(x1,y1),(0,255,255),2)
            image_show('instance[i]',np.hstack([instance1, image1,color_overlay1,contour_overlay1]))
            cv2.waitKey(0)