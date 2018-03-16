from common import *
from net.scheduler import StepLR
import configparser


class Configuration(object):

    def __init__(self):
        super(Configuration, self).__init__()
        self.version = 'configuration version \'mask-rcnn-resnet50-fpn-kaggle\''

        # net
        # include background class
        self.num_classes = 2

        # multi-rpn  --------------------------------------------------------
        # base size of the anchor box on imput image (2*a, diameter?)
        self.rpn_base_sizes = [8, 16, 32, 64]
        # 4 dirrerent zoom scales from each feature map to input.
        # used to get stride of anchor boxes
        # e.g. 2 for p1, 4 for p2, 8 for p3, 16 for p4.
        # the smaller the feature map is, the bigger the anchor box will be
        self.rpn_scales = [2, 4, 8, 16]

        aspect = lambda s,x: (s*1/x**0.5,s*x**0.5)
        # self.rpn_base_apsect_ratios = [
        #     [(1,1) ],
        #     [(1,1),                    aspect(2**0.33,2), aspect(2**0.33,0.5),],
        #     [(1,1), aspect(2**0.66,1), aspect(2**0.33,2), aspect(2**0.33,0.5),aspect(2**0.33,3), aspect(2**0.33,0.33),  ],
        #     [(1,1), aspect(2**0.66,1), aspect(2**0.33,2), aspect(2**0.33,0.5),],
        # ]
        self.rpn_base_apsect_ratios = [
            [(1,1) ],
            [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
            [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
            [(1,1), aspect(2**0.5,2), aspect(2**0.5,0.5),],
        ]

        self.rpn_train_bg_thresh_high = 0.5
        self.rpn_train_fg_thresh_low  = 0.5

        self.rpn_train_nms_pre_score_threshold = 0.7
        self.rpn_train_nms_overlap_threshold   = 0.8  # higher for more proposals for mask training
        self.rpn_train_nms_min_size = 5

        self.rpn_test_nms_pre_score_threshold = 0.8
        self.rpn_test_nms_overlap_threshold   = 0.5
        self.rpn_test_nms_min_size = 5

        # rcnn ------------------------------------------------------------------
        self.rcnn_crop_size         = 14
        self.rcnn_train_batch_size  = 64  # per image
        self.rcnn_train_fg_fraction = 0.5
        self.rcnn_train_fg_thresh_low  = 0.5
        self.rcnn_train_bg_thresh_high = 0.5
        self.rcnn_train_bg_thresh_low  = 0.0

        self.rcnn_train_nms_pre_score_threshold = 0.05
        self.rcnn_train_nms_overlap_threshold   = 0.8  # high for more proposals for mask
        self.rcnn_train_nms_min_size = 5

        self.rcnn_test_nms_pre_score_threshold = 0.3
        self.rcnn_test_nms_overlap_threshold   = 0.5
        self.rcnn_test_nms_min_size = 5

        # mask ------------------------------------------------------------------
        self.mask_crop_size            = 14
        self.mask_train_batch_size     = 64  # per image
        self.mask_size                 = 28  # per image
        self.mask_train_min_size       = 5
        self.mask_train_fg_thresh_low  = self.rpn_train_fg_thresh_low

        self.mask_test_nms_pre_score_threshold = 0.4
        self.mask_test_nms_overlap_threshold = 0.1
        self.mask_test_mask_threshold  = 0.5

        # annotation
        self.annotation_train_split = 'train_ids_all_670'
        self.annotation_test_split = 'test_ids_all_65'

        # training --------------------------------------------------------------
        self.model_name = 'mask-rcnn-50-gray500-02'
        self.model_name = '3-16'

        self.train_split = 'train_ids_gray_500'
        self.valid_split = 'valid_ids_gray_43'
        self.pretrain = None
        self.checkpoint = None

        # optim -----------------------------------------------------------------
        self.lr = 0.01
        self.iter_accum = 1  # learning rate = lr/iter_accum
        self.batch_size = 12
        self.num_iters = 1000 * 1000
        self.iter_smooth = 20  # calculate smoothed loss over each 20 iter
        self.iter_valid = 100
        self.iter_save = list(range(0, self.num_iters, 500)) + [self.num_iters]
        self.lr_scheduler =  StepLR([ (0, 0.01),  (5000, 0.001),  (10000, 0.001)])

        # validation  -----------------------------------------------------------
        self.valid_checkpoint = '0004600_model.pth'

        # submit ----------------------------------------------------------------
        self.submit_checkpoint = '0004600_model.pth'
        self.submit_split = 'test_ids_gray_53'
        self.submit_csv_name = 'submission-gray53-only.csv'

    def __repr__(self):
        d = self.__dict__.copy()
        str=''
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
