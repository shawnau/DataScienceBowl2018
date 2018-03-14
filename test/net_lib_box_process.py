from net.lib.box.process import *


def run_check_nms():
    H,W = 480,640
    num_objects = 4
    rois = []
    for n in range(num_objects):
        w = np.random.randint(64,256)
        h = np.random.randint(64,256)
        x0 = np.random.randint(0,W-w)
        y0 = np.random.randint(0,H-h)
        x1 = x0 + w
        y1 = y0 + h
        gt = [x0,y0,x1,y1]

        M = np.random.randint(10,20)
        for m in range(M):
            dw = int(np.random.uniform(0.5,2)*w)
            dh = int(np.random.uniform(0.5,2)*h)
            dx = int(np.random.uniform(-1,1)*w*0.5)
            dy = int(np.random.uniform(-1,1)*h*0.5)
            xx0 = x0 - dw//2 + dx
            yy0 = y0 - dh//2 + dy
            xx1 = xx0 + w+dw
            yy1 = yy0 + h+dh
            score = np.random.uniform(0.5,2)

            rois.append([xx0,yy0,xx1,yy1, score])
            pass

    rois = np.array(rois).astype(np.float32)


    if 1:
        keep = gpu_nms(rois, 0.5)
        print('gpu_nms    :', keep)
        keep = cython_nms(rois, 0.5)
        print('cython_nms :', keep)

        #gpu     [52, 39, 48, 43, 47, 21, 9, 6, 32, 8, 16, 36, 28, 29, 53, 41]
        #py      [52, 39, 48, 43, 47, 21, 9, 6, 32, 8, 16, 36, 28, 29, 53, 41]
        #cython  [52, 39, 48, 43, 47, 21, 9, 6, 32, 8, 16, 36, 28, 29, 53, 41]

    if 1:
        rois = torch.from_numpy(rois).cuda()
        keep = torch_nms(rois, 0.5)

        keep = keep.cpu().numpy()
        print('torch_nms  :', keep)
        #torch [52 39 48 43 47 21  9  6 32  8 16 36 28 29 53 41]


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_check_nms()

    print('sucess!')