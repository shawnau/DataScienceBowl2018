from dataset.folder import TrainFolder
from dataset.transform import *
from dataset.reader import *
from utility.file import *
from utility.scheduler import *
from configuration import Configuration
from net.model import MaskRcnnNet
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  #'3,2' #'3,2,1,0'


WIDTH, HEIGHT = 256, 256

def train_augment(image, multi_mask, meta, index):

    image, multi_mask = \
        random_shift_scale_rotate_transform(
            image, multi_mask,
            shift_limit=[0, 0],
            scale_limit=[1/2, 2],
            rotate_limit=[-45, 45],
            borderMode=cv2.BORDER_REFLECT_101,
            u=0.5)

    image, multi_mask = random_crop_transform(image, multi_mask, WIDTH, HEIGHT, u=0.5)
    image, multi_mask = random_horizontal_flip_transform(image, multi_mask, 0.5)
    image, multi_mask = random_vertical_flip_transform(image, multi_mask, 0.5)
    image, multi_mask = random_rotate90_transform(image, multi_mask, 0.5)

    input_image = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255)
    box, label, instance = multi_mask_to_annotation(multi_mask)

    return input_image, box, label, instance, meta, index


def valid_augment(image, multi_mask, meta, index):

    image, multi_mask = fix_crop_transform(image, multi_mask, -1, -1, WIDTH, HEIGHT)
    input_image = torch.from_numpy(image.transpose((2,0,1))).float().div(255)
    box, label, instance  = multi_mask_to_annotation(multi_mask)

    return input_image, box, label, instance, meta, index


def make_collate(batch):

    batch_size = len(batch)
    inputs    = torch.stack([batch[b][0]for b in range(batch_size)], 0)
    boxes     =             [batch[b][1]for b in range(batch_size)]
    labels    =             [batch[b][2]for b in range(batch_size)]
    instances =             [batch[b][3]for b in range(batch_size)]
    metas     =             [batch[b][4]for b in range(batch_size)]
    indices   =             [batch[b][5]for b in range(batch_size)]

    return [inputs, boxes, labels, instances, metas, indices]


def validate(net, test_loader):

    test_num  = 0
    test_loss = np.zeros(6, np.float32)

    for i, (inputs, truth_boxes, truth_labels, truth_instances, metas, indices) in enumerate(test_loader, 0):

        with torch.no_grad():
            inputs = Variable(inputs).cuda()
            net(inputs, truth_boxes, truth_labels, truth_instances)
            loss = net.loss()

        batch_size = len(indices)
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
    test_loss = test_loss/test_num
    return test_loss


def run_train():

    cfg = Configuration()
    f = TrainFolder(os.path.join(cfg.result_dir, cfg.model_name))
    skip = ['crop', 'mask']

    log = Logger()
    log.open(os.path.join(f.folder_name, 'log.train.txt'), mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('** some experiment setting **\n')
    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\tout_dir      = %s\n' % f.folder_name)
    log.write('\n')

    # net -------------------------------------------------
    log.write('** net setting **\n')
    net = MaskRcnnNet(cfg).cuda()

    if cfg.checkpoint is not None:
        checkpoint = os.path.join(f.checkpoint_dir, cfg.checkpoint)
        log.write('\tcheckpoint_dir = %s\n' % checkpoint)
        net.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))

    if cfg.pretrain is not None:
        pretrain_file = os.path.join(f.checkpoint_dir, cfg.pretrain)
        log.write('\tpretrain_file = %s\n' % pretrain_file)
        net.load_pretrain(pretrain_file, skip)

    # optimiser -------------------------------------------------
    #optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
    #                      lr=cfg.lr/cfg.iter_accum,
    #                      amsgrad=True,
    #                      weight_decay=0.0001
    #                      )
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=cfg.lr / cfg.iter_accum,
                          momentum=0.9,
                          weight_decay=0.0001
                          )
    
    lr_scheduler = cfg.lr_scheduler

    start_iter  = 0
    start_epoch = 0.
    if cfg.checkpoint is not None:
        checkpoint = os.path.join(f.checkpoint_dir, cfg.checkpoint)
        checkpoint_optim  = torch.load(checkpoint.replace('_model.pth', '_optimizer.pth'))
        start_iter  = checkpoint_optim['iter' ]
        start_epoch = checkpoint_optim['epoch']

        rate = get_learning_rate(optimizer)  # load all except learning rate
        optimizer.load_state_dict(checkpoint_optim['optimizer'])
        adjust_learning_rate(optimizer, rate)

    # dataset -------------------------------------------------
    log.write('** dataset setting **\n')
    train_dataset = ScienceDataset(cfg, cfg.train_split, mode='train', transform=train_augment)
    train_loader  = DataLoader(
                        train_dataset,
                        sampler=RandomSampler(train_dataset),
                        batch_size=cfg.batch_size,
                        drop_last=True,
                        num_workers=4,
                        pin_memory=True,
                        collate_fn=make_collate)

    valid_dataset = ScienceDataset(cfg, cfg.valid_split, mode='train', transform=valid_augment)
    valid_loader  = DataLoader(
                        valid_dataset,
                        sampler=SequentialSampler(valid_dataset),
                        batch_size=cfg.batch_size,
                        drop_last=False,
                        num_workers=4,
                        pin_memory=True,
                        collate_fn=make_collate)

    log.write('\tWIDTH, HEIGHT = %d, %d\n' % (WIDTH, HEIGHT))
    log.write('\ttrain_dataset.split = %s\n' % train_dataset.split)
    log.write('\tvalid_dataset.split = %s\n' % valid_dataset.split)
    log.write('\tlen(train_dataset)  = %d\n' % len(train_dataset))
    log.write('\tlen(valid_dataset)  = %d\n' % len(valid_dataset))
    log.write('\tlen(train_loader)   = %d\n' % len(train_loader))
    log.write('\tlen(valid_loader)   = %d\n' % len(valid_loader))
    log.write('\tbatch_size  = %d\n' % cfg.batch_size)
    log.write('\titer_accum  = %d\n' % cfg.iter_accum)
    log.write('\tbatch_size*iter_accum  = %d\n' % (cfg.batch_size*cfg.iter_accum))
    log.write('\n')

    # start training here! -------------------------------------------------
    log.write('** start training here! **\n')
    log.write(' optimizer=%s\n' % str(optimizer))
    # log.write(' momentum=%f\n' % optimizer.param_groups[0]['momentum'])
    log.write(' lr_scheduler=%s\n\n' % str(lr_scheduler))

    log.write(' images_per_epoch = %d\n\n' % len(train_dataset))
    log.write(' rate    iter   epoch  num   | valid_loss rpnc rpnr rcnnc rcnnr mask | train_loss rpnc rpnr rcnnc rcnnr mask | batch_loss rpnc rpnr rcnnc rcnnr mask |  time          \n')
    log.write('-------------------------------------------------------------------------------------------------------------------------------------------------------------------\n')

    train_loss  = np.zeros(6,np.float32)
    valid_loss  = np.zeros(6,np.float32)
    batch_loss  = np.zeros(6,np.float32)
    rate = 0

    start = timer()
    j = 0  # accum counter
    i = 0  # iter  counter

    while i < cfg.num_iters:  # loop over the dataset multiple times
        sum_train_loss = np.zeros(6,np.float32)
        sum_train_acc  = 0.0
        batch_sum = 0

        net.set_mode('train')
        optimizer.zero_grad()
        for inputs, truth_boxes, truth_labels, truth_instances, metas, indices in train_loader:
            if all(len(b) == 0 for b in truth_boxes):
                continue
            batch_size = len(indices)
            i = j/cfg.iter_accum + start_iter
            epoch = (i-start_iter)*batch_size*cfg.iter_accum/len(train_dataset) + start_epoch
            num_products = epoch*len(train_dataset)
            # validate iter -------------------------------------------------
            if i % cfg.iter_valid == 0:
                net.set_mode('valid')
                valid_loss = validate(net, valid_loader)
                net.set_mode('train')

                print('\r', end='', flush=True)
                log.write('%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %s\n' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5],#valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5],#train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5],#batch_acc,
                         time_to_str((timer() - start)/60)))
                time.sleep(0.01)
            # save checkpoint_dir -------------------------------------------------
            if i in cfg.iter_save:
                model_config_path = os.path.join(f.checkpoint_dir, 'configuration.pkl')
                model_path = os.path.join(f.checkpoint_dir, '%08d_model.pth' % i)
                optimizer_path = os.path.join(f.checkpoint_dir, '%08d_optimizer.pth' % i)

                torch.save(net.state_dict(), model_path)
                torch.save({
                    'optimizer': optimizer.state_dict(),
                    'iter'     : i,
                    'epoch'    : epoch,
                    }, optimizer_path)
                with open(model_config_path, 'wb') as pickle_file:
                    pickle.dump(cfg, pickle_file, pickle.HIGHEST_PROTOCOL)

            # learning rate schduler -------------------------------------------------
            if lr_scheduler is not None:
                lr = lr_scheduler.get_rate(i)
                if lr < 0:
                    break
                adjust_learning_rate(optimizer, lr/cfg.iter_accum)
            rate = get_learning_rate(optimizer)*cfg.iter_accum

            # one iteration update  -------------------------------------------------
            inputs = Variable(inputs).cuda()
            net(inputs, truth_boxes, truth_labels, truth_instances)
            loss = net.loss()

            # accumulated update
            loss.backward()
            if j % cfg.iter_accum == 0:
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
            batch_sum += 1
            if i % cfg.iter_smooth == 0:
                train_loss = sum_train_loss/batch_sum
                sum_train_loss = np.zeros(6,np.float32)
                sum_train_acc  = 0.
                batch_sum = 0

            print('\r%0.4f %5.1f k %6.1f %4.1f m | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %0.3f   %0.2f %0.2f   %0.2f %0.2f   %0.2f | %s  %d,%d,%s' % (\
                         rate, i/1000, epoch, num_products/1000000,
                         valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3], valid_loss[4], valid_loss[5], # valid_acc,
                         train_loss[0], train_loss[1], train_loss[2], train_loss[3], train_loss[4], train_loss[5], # train_acc,
                         batch_loss[0], batch_loss[1], batch_loss[2], batch_loss[3], batch_loss[4], batch_loss[5], # batch_acc,
                         time_to_str((timer() - start)/60), i, j, ''), end='', flush=True)#str(inputs.size()))
            j = j+1
        pass  # end of one data loader
    pass  # end of all iterations

    # save last
    if 1:
        model_config_path = os.path.join(f.folder_name, 'checkpoint', 'configuration.pkl')
        model_path = os.path.join(f.folder_name, 'checkpoint', '%d_model.pth' % i)
        optimizer_path = os.path.join(f.folder_name, 'checkpoint', '%08d_optimizer.pth' % i)

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

    run_train()

    print('\nsucess!')
