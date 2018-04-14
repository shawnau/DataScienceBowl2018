import os
import configparser
from utility.scheduler import StepLR
from postprocess.augments import *


class Configuration(object):

    def __init__(self):
        super(Configuration, self).__init__()
        self.version = 'configuration version \'mask-rcnn-resnet50-fpn-kaggle\''
        # source data downloaded from kaggle
        self.source_dir = '/root/xiaoxuan/kaggle/data/__download__'
        self.source_train_dir = os.path.join(self.source_dir, 'stage1_train')
        self.source_test_dir = os.path.join(self.source_dir, 'stage2_test')
        self.source_extra_dir = os.path.join(self.source_dir, 'extra_data')

        # root directory to store data & result
        self.root_dir = '/root/xiaoxuan/kaggle'
        # data directory to store preprocessed data
        self.data_dir = os.path.join(self.root_dir, 'data')
        self.split_dir = os.path.join(self.data_dir, 'splits')
        # result directory to store model info
        self.result_dir = os.path.join('/', 'root', 'super_large')

        # net
        # include background class
        self.num_classes = 2

        # multi-rpn  --------------------------------------------------------
        # base size of the anchor box on input image (2*a, diameter?)
        self.rpn_base_sizes = [8, 16, 32, 64]
        # 4 dirrerent zoom scales from each feature map to input.
        # used to get stride of anchor boxes
        # e.g. 2 for p1, 4 for p2, 8 for p3, 16 for p4.
        # the smaller the feature map is, the bigger the anchor box will be
        self.rpn_scales = [2, 4, 8, 16]

        aspect = lambda s,x: (s*1/x**0.5,s*x**0.5)
        # for slim cells
        self.rpn_base_apsect_ratios = [
            [(1, 1)],
            [(1, 1), aspect(2 ** 0.25, 2), aspect(2 ** 0.25, 0.5), ],
            [(1, 1), aspect(2 ** 0.5, 1), aspect(2 ** 0.25, 2), aspect(2 ** 0.25, 0.5), aspect(2 ** 0.25, 3),
             aspect(2 ** 0.25, 0.25), ],
            [(1, 1), aspect(2 ** 0.5, 1), aspect(2 ** 0.25, 2), aspect(2 ** 0.25, 0.5), ],
        ]

        #self.rpn_base_apsect_ratios = [
        #    [(1,1) ],
        #    [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
        #    [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
        #    [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
        #]

        self.rpn_train_bg_thresh_high = 0.5
        self.rpn_train_fg_thresh_low  = 0.5

        self.rpn_train_scale_balance = False

        self.rpn_train_nms_pre_score_threshold = 0.50
        self.rpn_train_nms_overlap_threshold   = 0.85  # higher for more proposals for mask training
        self.rpn_train_nms_min_size = 5

        self.rpn_test_nms_pre_score_threshold = 0.60
        self.rpn_test_nms_overlap_threshold   = 0.75
        self.rpn_test_nms_min_size = 5

        # rcnn ------------------------------------------------------------------
        self.rcnn_crop_size         = 14
        self.rcnn_train_batch_size  = 32  # per image
        self.rcnn_train_fg_fraction = 0.5
        self.rcnn_train_fg_thresh_low  = 0.5
        self.rcnn_train_bg_thresh_high = 0.5
        self.rcnn_train_bg_thresh_low  = 0.0

        self.rcnn_train_nms_pre_score_threshold = 0.05
        self.rcnn_train_nms_overlap_threshold   = 0.85  # high for more proposals for mask
        self.rcnn_train_nms_min_size = 8

        self.rcnn_test_nms_pre_score_threshold = 0.50
        self.rcnn_test_nms_overlap_threshold   = 0.85
        self.rcnn_test_nms_min_size = 8

        # mask ------------------------------------------------------------------
        self.mask_crop_size            = 14  # input of mask head
        self.mask_train_batch_size     = 32  # per image
        self.mask_size                 = 28  # out put of mask head
        self.mask_train_min_size       = 8
        self.mask_train_fg_thresh_low  = self.rpn_train_fg_thresh_low

        self.mask_test_nms_pre_score_threshold = 0.1
        self.mask_test_nms_overlap_threshold = 0.2
        self.mask_test_mask_threshold  = 0.5
        self.mask_test_mask_min_area = 8

        # annotation
        self.annotation_train_split = 'train_all_664'
        self.annotation_test_split = 'test2_all_3019'
        self.annotation_extra_split = 'extra_raw'

        # training --------------------------------------------------------------
        self.model_name = '4-10'#'4-10-color-mixed'

        self.train_split = 'train_color_mixed_338'
        self.valid_split = 'test2_bw_to_ensemble_791'#'test2_color_219'
        self.pretrain = None
        self.checkpoint = None

        # optim -----------------------------------------------------------------
        self.lr = 0.01
        self.iter_accum = 1  # learning rate = lr/iter_accum
        self.batch_size = 10
        self.num_iters = 35000
        self.iter_smooth = 20  # calculate smoothed loss over each 20 iter
        self.iter_valid = 100
        self.iter_save = list(range(0, self.num_iters, 1000)) + [self.num_iters]
        self.lr_scheduler = StepLR([ (0, 0.01),  (8000, 0.001),  (20000, 0.0001)])

        # validation  -----------------------------------------------------------
        self.valid_checkpoint = '00021000_model.pth'#'00017000_model.pth'
        # submit ----------------------------------------------------------------
        self.submit_checkpoint = self.valid_checkpoint
        self.submit_split = None#'test2_color_219'
        self.submit_csv_name = None#'submission-BW2799-only.csv'

        # test time augments
        self.test_augments = [
            ('normal', do_test_augment_identity, undo_test_augment_identity, {}),
            #('flip_transpose_1', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type': 1, }),
            ('flip_transpose_2', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type': 2, }),
            #('flip_transpose_3', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type': 3, }),
            ('flip_transpose_4', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type': 4, }),
            #('flip_transpose_5', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type': 5, }),
            #('flip_transpose_6', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type': 6, }),
            #('flip_transpose_7', do_test_augment_flip_transpose, undo_test_augment_flip_transpose, {'type': 7, }),
            #('scale_0.8', do_test_augment_scale, undo_test_augment_scale, {'scale_x': 0.8, 'scale_y': 0.8}),
            #('scale_1.2', do_test_augment_scale, undo_test_augment_scale, {'scale_x': 1.2, 'scale_y': 1.2}),
            ('scale_0.5', do_test_augment_scale, undo_test_augment_scale, {'scale_x': 0.8, 'scale_y': 0.8}),
            ('scale_1.8', do_test_augment_scale, undo_test_augment_scale, {'scale_x': 1.2, 'scale_y': 1.2}),
        ]
        self.test_augment_names = [t[0] for t in self.test_augments]

    def __repr__(self):
        d = self.__dict__.copy()
        str = ''
        for k, v in d.items():
            str += '%32s = %s\n' % (k,v)

        return str

    def save(self, file):
        d = self.__dict__.copy()
        config = configparser.ConfigParser()
        config['all'] = d
        with open(file, 'w') as f:
            config.write(f)

    def load(self, file):
        # config = configparser.ConfigParser()
        # config.read(file)
        #
        # d = config['all']
        # self.num_classes     = eval(d['num_classes'])
        # self.multi_num_heads = eval(d['multi_num_heads'])

        raise NotImplementedError
