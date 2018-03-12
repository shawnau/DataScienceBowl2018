import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2' #'3,2,1,0'

from common import *
from utility.file import *
from dataset.reader import *
from net.scheduler import *
from net.metric import *


# -------------------------------------------------------------------------------------
#WIDTH, HEIGHT = 128,128
#WIDTH, HEIGHT = 192,192
WIDTH, HEIGHT = 256,256

from net.resnet50_mask_rcnn.configuration import Configuration
from net.resnet50_mask_rcnn.draw import *
from net.resnet50_mask_rcnn.model import *
# -------------------------------------------------------------------------------------


class TrainFolder:
    def __init__(self, model_name):
        self.work_dir = os.path.join(RESULTS_DIR, model_name)
        self.checkpoint = os.path.join(self.work_dir, 'checkpoint')
        self.train_result = os.path.join(self.work_dir, 'train')
        self.backup = os.path.join(self.work_dir, 'backup')

        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.checkpoint, exist_ok=True)
        os.makedirs(self.train_result, exist_ok=True)
        os.makedirs(self.backup, exist_ok=True)
        backup_project_as_zip(PROJECT_PATH, os.path.join(self.work_dir, 'backup', 'code.train.%s.zip' % IDENTIFIER))


def train_augment(image, multi_mask, meta, index):

    image, multi_mask = \
        random_shift_scale_rotate_transform(
            image, multi_mask,
            shift_limit=[0, 0],
            scale_limit=[1/2, 2],
            rotate_limit=[-45, 45],
            borderMode=cv2.BORDER_REFLECT_101,
            u=0.5) #borderMode=cv2.BORDER_CONSTANT

    # overlay = multi_mask_to_color_overlay(multi_mask,color='cool')
    # overlay1 = multi_mask_to_color_overlay(multi_mask1,color='cool')
    # image_show('overlay',overlay)
    # image_show('overlay1',overlay1)
    # cv2.waitKey(0)

    image, multi_mask = random_crop_transform(image, multi_mask, WIDTH, HEIGHT, u=0.5)
    image, multi_mask = random_horizontal_flip_transform(image, multi_mask, 0.5)
    image, multi_mask = random_vertical_flip_transform(image, multi_mask, 0.5)
    image, multi_mask = random_rotate90_transform(image, multi_mask, 0.5)
    ##image,  multi_mask = fix_crop_transform2(image, multi_mask, -1,-1,WIDTH, HEIGHT)

    input = torch.from_numpy(image.transpose((2,0,1))).float().div(255)
    box, label, instance = multi_mask_to_annotation(multi_mask)

    return input, box, label, instance, meta, index


def valid_augment(image, multi_mask, meta, index):

    image,  multi_mask = fix_crop_transform(image, multi_mask, -1, -1, WIDTH, HEIGHT)
    input = torch.from_numpy(image.transpose((2,0,1))).float().div(255)
    box, label, instance  = multi_mask_to_annotation(multi_mask)

    return input, box, label, instance, meta, index


def make_collate(batch):

    batch_size = len(batch)
    #for b in range(batch_size): print (batch[b][0].size())
    inputs    = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    boxes     =             [batch[b][1]for b in range(batch_size)]
    labels    =             [batch[b][2]for b in range(batch_size)]
    instances =             [batch[b][3]for b in range(batch_size)]
    metas     =             [batch[b][4]for b in range(batch_size)]
    indices   =             [batch[b][5]for b in range(batch_size)]

    return [inputs, boxes, labels, instances, metas, indices]


def evaluate(net, test_loader):

    test_num  = 0
    test_loss = np.zeros(6, np.float32)
    test_acc  = 0
    for i, (inputs, truth_boxes, truth_labels, truth_instances, metas, indices) in enumerate(test_loader, 0):

        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            net(inputs, truth_boxes,  truth_labels, truth_instances )
            loss = net.loss(inputs, truth_boxes,  truth_labels, truth_instances)

        # acc    = dice_loss(masks, labels) #todo

        batch_size = len(indices)
        test_acc  += 0 # batch_size*acc[0][0]
        test_loss += batch_size*np.array((
                           loss.cpu().data.numpy(),
                           net.rpn_cls_loss.cpu().data.numpy(),
                           net.rpn_reg_loss.cpu().data.numpy(),
                           net.rcnn_cls_loss.cpu().data.numpy(),
                           net.rcnn_reg_loss.cpu().data.numpy(),
                           net.mask_cls_loss.cpu().data.numpy(),
                         ))
        test_num  += batch_size

    assert(test_num == len(test_loader.sampler))
    test_acc  = test_acc/test_num
    test_loss = test_loss/test_num
    return test_loss, test_acc


def run_train(model_name, train_split, valid_split, checkpoint=None, pretrain_file=None):

    work_dir = os.path.join(RESULTS_DIR, model_name)
    skip = ['crop', 'mask']

    # setup
    f = TrainFolder(model_name)

    log = Logger()
    log.open(os.path.join(work_dir, 'log.train.txt'), mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % work_dir)
    log.write('\n')

    # net -------------------------------------------------
    log.write('** net setting **\n')
    cfg = Configuration()
    net = MaskRcnnNet(cfg).cuda()

    if checkpoint is not None:
        checkpoint = os.path.join(f.checkpoint, checkpoint)
        log.write('\tcheckpoint = %s\n' % checkpoint)
        net.load_state_dict(torch.load(checkpoint,
                                       map_location=lambda storage, loc: storage))

    if pretrain_file is not None:
        pretrain_file = os.path.join(f.checkpoint, pretrain_file)
        log.write('\tpretrain_file = %s\n' % pretrain_file)
        net.load_pretrain(pretrain_file, skip)

    log.write('%s\n\n' % type(net))
    log.write('%s\n' % net.version)
    log.write('\n')

    # optimiser
    iter_accum  = 1
    batch_size  = 16

    num_iters   = 1000*1000
    iter_smooth = 20
    iter_log    = 50
    iter_valid  = 100
    iter_save   = [0, num_iters-1] + list(range(0, num_iters, 500))  # 1*1000

    LR = None  #LR = StepLR([ (0, 0.01),  (200, 0.001),  (300, -1)])
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.01/iter_accum,
                          momentum=0.9,
                          weight_decay=0.0001
                          )

    start_iter  = 0
    start_epoch = 0.
    if checkpoint is not None:
        checkpoint  = torch.load(checkpoint.replace('_model.pth', '_optimizer.pth'))
        start_iter  = checkpoint['iter' ]
        start_epoch = checkpoint['epoch']

        rate = get_learning_rate(optimizer)  #load all except learning rate
        optimizer.load_state_dict(checkpoint['optimizer'])
        adjust_learning_rate(optimizer, rate)

    # dataset -------------------------------------------------
    log.write('** dataset setting **\n')

    train_dataset = ScienceDataset(train_split, mode='train', transform=train_augment)
    train_loader  = DataLoader(
                        train_dataset,
                        sampler     = RandomSampler(train_dataset),
                        batch_size  = batch_size,
                        drop_last   = True,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = make_collate)

    valid_dataset = ScienceDataset(valid_split, mode='train', transform=valid_augment)
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler     = SequentialSampler(valid_dataset),
                        batch_size  = batch_size,
                        drop_last   = False,
                        num_workers = 4,
                        pin_memory  = True,
                        collate_fn  = make_collate)

    log.write('\tWIDTH, HEIGHT = %d, %d\n' % (WIDTH, HEIGHT))
    log.write('\ttrain_dataset.split = %s\n' % train_dataset.split)
    log.write('\tvalid_dataset.split = %s\n' % valid_dataset.split)
    log.write('\tlen(train_dataset)  = %d\n' % len(train_dataset))
    log.write('\tlen(valid_dataset)  = %d\n' % len(valid_dataset))
    log.write('\tlen(train_loader)   = %d\n' % len(train_loader))
    log.write('\tlen(valid_loader)   = %d\n' % len(valid_loader))
    log.write('\tbatch_size  = %d\n' % batch_size)
    log.write('\titer_accum  = %d\n' % iter_accum)
    log.write('\tbatch_size*iter_accum  = %d\n' % (batch_size*iter_accum))
    log.write('\n')

    # start training here! -------------------------------------------------
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n' % str(optimizer))
    log.write(' momentum=%f\n' % optimizer.param_groups[0]['momentum'])
    log.write(' LR=%s\n\n' % str(LR))

    log.write(' images_per_epoch = %d\n\n' % len(train_dataset))
    log.write(' rate    iter   epoch  num   | valid_loss               | train_loss               | batch_loss               |  time          \n')
    log.write('-------------------------------------------------------------------------------------------------------------------------------\n')

    train_loss  = np.zeros(6,np.float32)
    train_acc   = 0.0
    valid_loss  = np.zeros(6,np.float32)
    valid_acc   = 0.0
    batch_loss  = np.zeros(6,np.float32)
    batch_acc   = 0.0
    rate = 0

    start = timer()
    j = 0
    i = 0

    while i < num_iters:  # loop over the dataset multiple times
        sum_train_loss = np.zeros(6,np.float32)
        sum_train_acc  = 0.0
        sum = 0

        net.set_mode('train')
        optimizer.zero_grad()
        for inputs, truth_boxes, truth_labels, truth_instances, metas, indices in train_loader:
            if all(len(b) == 0 for b in truth_boxes):
                continue

            batch_size = len(indices)
            i = j/iter_accum + start_iter
            epoch = (i-start_iter)*batch_size*iter_accum/len(train_dataset) + start_epoch
            num_products = epoch*len(train_dataset)

            if i % iter_valid == 0:
                net.set_mode('valid')
                valid_loss, valid_acc = evaluate(net, valid_loader)
                net.set_mode('train')

                print('\r', end='', flush=True)
                log.write('%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %s\n' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5],#valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],#train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5],#batch_acc,
                         time_to_str((timer() - start)/60)))
                time.sleep(0.01)

            if i in iter_save:
                model_config_path = os.path.join(f.checkpoint, 'configuration.pkl')
                model_path = os.path.join(f.checkpoint, '%08d_model.pth'%i)
                optimizer_path = os.path.join(f.checkpoint, '%08d_optimizer.pth'%i)

                torch.save(net.state_dict(), model_path)
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter'     : i,
                    'epoch'    : epoch,
                    }, optimizer_path)
                with open(model_config_path, 'wb') as pickle_file:
                    pickle.dump(cfg, pickle_file, pickle.HIGHEST_PROTOCOL)

            # learning rate schduler -------------------------------------------------
            if LR is not None:
                lr = LR.get_rate(i)
                if lr < 0:
                    break
                adjust_learning_rate(optimizer, lr/iter_accum)
            rate = get_learning_rate(optimizer)*iter_accum

            # one iteration update  -------------------------------------------------
            inputs = Variable(inputs).cuda()
            net(inputs, truth_boxes, truth_labels, truth_instances)
            loss = net.loss()

            # accumulated update
            loss.backward()
            if j % iter_accum == 0:
                # torch.nn.utils.clip_grad_norm(net.parameters(), 1)
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  -------------------------------------------------
            batch_acc  = 0 #acc[0][0]
            batch_loss = np.array((
                           loss.cpu().data.numpy(),
                           net.rpn_cls_loss.cpu().data.numpy(),
                           net.rpn_reg_loss.cpu().data.numpy(),
                           net.rcnn_cls_loss.cpu().data.numpy(),
                           net.rcnn_reg_loss.cpu().data.numpy(),
                           net.mask_cls_loss.cpu().data.numpy(),
                         ))
            sum_train_loss += batch_loss
            sum_train_acc  += batch_acc
            sum += 1
            if i % iter_smooth == 0:
                train_loss = sum_train_loss/sum
                train_acc  = sum_train_acc /sum
                sum_train_loss = np.zeros(6,np.float32)
                sum_train_acc  = 0.
                sum = 0

            print('\r%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %s  %d,%d,%s' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5], # valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5], # train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5], # batch_acc,
                         time_to_str((timer() - start)/60) ,i,j, ''), end='', flush=True)#str(inputs.size()))
            j = j+1
        pass  # end of one data loader
    pass  # end of all iterations

    # save last
    if 1:
        model_config_path = os.path.join(work_dir, 'checkpoint', 'configuration.pkl')
        model_path = os.path.join(work_dir, 'checkpoint', '%d_model.pth'%i)
        optimizer_path = os.path.join(work_dir, 'checkpoint', '%08d_optimizer.pth'%i)

        torch.save(net.state_dict(), model_path)
        torch.save({
            'optimizer': optimizer.state_dict(),
            'iter'     : i,
            'epoch'    : epoch,
        }, optimizer_path)
        with open(model_config_path, 'wb') as pickle_file:
            pickle.dump(cfg, pickle_file, pickle.HIGHEST_PROTOCOL)

    log.write('\n')


if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_train(model_name='mask-rcnn-50-gray500-02',
              train_split='train1_ids_gray2_500',
              valid_split='valid1_ids_gray2_43',
              pretrain_file=None,
              checkpoint=None
              )

    print('\nsucess!')
