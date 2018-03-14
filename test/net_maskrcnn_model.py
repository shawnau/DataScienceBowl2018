from net.resnet50_mask_rcnn.model import *


def run_check_feature_net():

    batch_size = 4
    C, H, W = 3, 256, 256
    feature_channels = 128

    x = torch.randn(batch_size,C,H,W)
    inputs = Variable(x).cuda()

    cfg = Configuration()
    feature_net = FeatureNet(cfg, C, feature_channels).cuda()

    ps = feature_net(inputs)

    print('')
    num_heads = len(ps)
    for i in range(num_heads):
        p = ps[i]
        print(i, p.size())


def run_check_multi_rpn_head():

    batch_size  = 8
    in_channels = 128
    H,W = 256, 256
    num_scales = 4
    feature_heights = [ int(H//2**l) for l in range(num_scales) ]
    feature_widths  = [ int(W//2**l) for l in range(num_scales) ]

    fs = []
    for h,w in zip(feature_heights,feature_widths):
        f = np.random.uniform(-1,1,size=(batch_size,in_channels,h,w)).astype(np.float32)
        f = Variable(torch.from_numpy(f)).cuda()
        fs.append(f)

    cfg = Configuration()
    rpn_head = RpnMultiHead(cfg, in_channels).cuda()
    logits_flat, deltas_flat = rpn_head(fs)

    print('logits_flat ',logits_flat.size())
    print('deltas_flat ',deltas_flat.size())
    print('')


def run_check_crop_head():

    # feature maps
    batch_size   = 4
    in_channels  = 128
    out_channels = 256
    H,W = 256, 256
    num_scales = 4
    feature_heights = [ int(H//2**l) for l in range(num_scales) ]
    feature_widths  = [ int(W//2**l) for l in range(num_scales) ]

    fs = []
    for h,w in zip(feature_heights,feature_widths):
        f = np.random.uniform(-1,1,size=(batch_size,in_channels,h,w)).astype(np.float32)
        f = Variable(torch.from_numpy(f)).cuda()
        fs.append(f)

    #proposal i,x0,y0,x1,y1,score, label
    proposals = []
    for b in range(batch_size):

        num_proposals = 4
        xs = np.random.randint(0,64,num_proposals)
        ys = np.random.randint(0,64,num_proposals)
        sizes  = np.random.randint(8,64,num_proposals)
        scores = np.random.uniform(0,1,num_proposals)

        proposal = np.zeros((num_proposals,7),np.float32)
        proposal[:,0] = b
        proposal[:,1] = xs
        proposal[:,2] = ys
        proposal[:,3] = xs+sizes
        proposal[:,4] = ys+sizes
        proposal[:,5] = scores
        proposal[:,6] = 1
        proposals.append(proposal)

    proposals = np.vstack(proposals)
    proposals = Variable(torch.from_numpy(proposals)).cuda()

    cfg      = Configuration()
    crop_net = CropRoi(cfg).cuda()
    crops    = crop_net(fs, proposals)

    print('crops', crops.size())
    print ('')

    crops     = crops.data.cpu().numpy()
    proposals = proposals.data.cpu().numpy()

    #for m in range(num_proposals):
    for m in range(8):
        crop     = crops[m]
        proposal = proposals[m]

        i,x0,y0,x1,y1,score,label = proposal

        print ('i=%d, x0=%3d, y0=%3d, x1=%3d, y1=%3d, score=%0.2f'%(i,x0,y0,x1,y1,score) )
        print (crop[0,0,:5] )
        print ('')


def run_check_rcnn_head():

    num_rois     = 100
    in_channels  = 256
    crop_size    = 14

    crops = np.random.uniform(-1,1,size=(num_rois,in_channels, crop_size,crop_size)).astype(np.float32)
    crops = Variable(torch.from_numpy(crops)).cuda()

    cfg = Configuration()
    assert(crop_size==cfg.rcnn_crop_size)

    rcnn_head = RcnnHead(cfg, in_channels).cuda()
    logits, deltas = rcnn_head(crops)

    print('logits ',logits.size())
    print('deltas ',deltas.size())
    print('')


def run_check_mask_head():

    num_rois    = 100
    in_channels = 256
    crop_size   = 14

    crops = np.random.uniform(-1,1,size=(num_rois, in_channels, crop_size, crop_size)).astype(np.float32)
    crops = Variable(torch.from_numpy(crops)).cuda()

    cfg = Configuration()
    assert(crop_size==cfg.crop_size)

    mask_head = MaskHead(cfg, in_channels).cuda()
    logits = mask_head(crops)

    print('logits ',logits.size())
    print('')


def run_check_mask_net():

    batch_size, C, H, W = 1, 3, 128,128
    feature_channels = 64
    inputs = np.random.uniform(-1,1,size=(batch_size, C, H, W)).astype(np.float32)
    inputs = Variable(torch.from_numpy(inputs)).cuda()

    cfg = Configuration()
    mask_net = MaskRcnnNet(cfg).cuda()

    mask_net.set_mode('eval')
    mask_net(inputs)

    print('rpn_logits_flat ',mask_net.rpn_logits_flat.size())
    print('rpn_probs_flat  ',mask_net.rpn_probs_flat.size())
    print('rpn_deltas_flat ',mask_net.rpn_deltas_flat.size())
    print('')


if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    run_check_feature_net()
    run_check_multi_rpn_head()
    run_check_crop_head()
    run_check_rcnn_head()
    run_check_mask_head()
    run_check_mask_net()
