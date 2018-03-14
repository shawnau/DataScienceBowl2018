from net.metric import *
from submit import run_length_encode, run_length_decode


def run_check_run_length_encode():

    name = 'b98681c74842c4058bd2f88b06063731c26a90da083b1ef348e0ec734c58752b'

    npy_file = DATA_DIR + '/image/stage1_train/' + name + '/multi_mask.npy'
    multi_mask = np.load(npy_file)

    cvs_EncodedPixels = []
    num =int( multi_mask.max())
    for m in range(num):
        rle = run_length_encode(multi_mask==m+1)
        cvs_EncodedPixels.append(rle)
    cvs_EncodedPixels.sort()

    #reference encoding from 'stage1_train_labels.csv'
    df = pd.read_csv (DATA_DIR + '/__download__/stage1_train_labels.csv')
    df = df.loc[df['ImageId'] == name]

    reference_cvs_EncodedPixels = df['EncodedPixels'].values
    reference_cvs_EncodedPixels.sort()

    print('reference_cvs_EncodedPixels\n', reference_cvs_EncodedPixels)
    print('')
    print('cvs_EncodedPixels\n', cvs_EncodedPixels)
    print('')

    print(reference_cvs_EncodedPixels==cvs_EncodedPixels)


def run_check_run_length_decode():

    name = 'b98681c74842c4058bd2f88b06063731c26a90da083b1ef348e0ec734c58752b'

    npy_file = DATA_DIR + '/image/stage1_train/' + name + '/multi_mask.npy'
    multi_mask = np.load(npy_file)
    H,W = multi_mask.shape[:2]

    cvs_EncodedPixels = []
    num =int( multi_mask.max())
    for m in range(num):
        rle = run_length_encode(multi_mask==m+1)
        cvs_EncodedPixels.append(rle)



    #reference encoding from 'stage1_train_labels.csv'
    df = pd.read_csv (DATA_DIR + '/__download__/stage1_train_labels.csv')
    df = df.loc[df['ImageId'] == name]

    reference_cvs_EncodedPixels = df['EncodedPixels'].values
    reference_cvs_EncodedPixels.sort()

    reference_multi_mask = np.zeros((H, W), np.int32)
    for rle in reference_cvs_EncodedPixels:
        thresh = run_length_decode(rle, H, W, fill_value=255)
        id = cvs_EncodedPixels.index(rle)
        reference_multi_mask[thresh>128] = id+1



    reference_multi_mask = reference_multi_mask.astype(np.float32)
    reference_multi_mask = reference_multi_mask/reference_multi_mask.max()*255
    multi_mask = multi_mask.astype(np.float32)
    multi_mask = multi_mask/multi_mask.max()*255


    print((reference_multi_mask!=multi_mask).sum())

    image_show('multi_mask', multi_mask,2)
    image_show('reference_multi_mask', reference_multi_mask,2)
    image_show('diff', (reference_multi_mask!=multi_mask)*255,2)
    cv2.waitKey(0)


def run_check_compute_precision_for_box():

    H,W = 256,256

    truth_label = np.array([1,1,2,1,-1],np.float32)
    truth_box = np.array([
        [ 10, 10,0,0,],
        [100, 10,0,0,],
        [ 50, 50,0,0,],
        [ 10,100,0,0,],
        [100,100,0,0,],
    ],np.float32)
    truth_box[:,2] = truth_box[:,0]+25
    truth_box[:,3] = truth_box[:,1]+25


    box = np.zeros((7,4),np.float32)
    box[:5] = truth_box[[0,1,2,4,3]] + np.random.uniform(-10,10,size=(5,4))
    box[ 5] = [ 10, 10, 80, 80,]
    box[ 6] = [100,100,180,180,]

    thresholds=[0.3,0.5,0.6]
    precisions, recalls, results, truth_results = \
        compute_precision_for_box(box, truth_box, truth_label, thresholds)

    for precision, recall, result, truth_result, threshold in zip(precisions, recalls, results, truth_results, thresholds):
        print('')
        print('threshold ', threshold)
        print('precision ', precision)
        print('recall    ', recall)

        image  = np.zeros((H,W,3),np.uint8)
        for i,b in enumerate(truth_box):
            x0,y0,x1,y1 = b.astype(np.int32)
            if truth_result[i]==HIT:
                draw_screen_rect(image,(x0,y0), (x1,y1), (0,255,255), 0.5)
            if truth_result[i]==MISS:
                draw_screen_rect(image,(x0,y0), (x1,y1), (0,0,255), 0.5)
            if truth_result[i]==INVALID:
                draw_screen_rect(image,(x0,y0), (x1,y1), (255,255,255), 0.5)


        for i,b in enumerate(box):
            x0,y0,x1,y1 = b.astype(np.int32)
            if result[i]==TP:
                cv2.rectangle(image,(x0,y0), (x1,y1), (0,255,255), 1)
            if result[i]==FP:
                cv2.rectangle(image,(x0,y0), (x1,y1), (0,0,255), 1)
            if result[i]==INVALID:
                draw_dotted_rect(image,(x0,y0), (x1,y1), (255,255,255), 1)


        image_show("image_box",image,1)
        cv2.waitKey(0)


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_run_length_encode()
    run_check_run_length_decode()
    run_check_compute_precision_for_box()

    print('\nsucess!')