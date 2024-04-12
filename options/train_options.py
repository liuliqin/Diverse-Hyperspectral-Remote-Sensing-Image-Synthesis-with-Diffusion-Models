from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def __init__(self):
        self.initialized = False
    def initialize(self,parser):

        BaseOptions.initialize(self,parser)
        parser.add_argument('--batch_size', type=int, default=4, help='train batch size')
        parser.add_argument('--crop_size', type=int, default=256, help='crop size')
        parser.add_argument('--start_epoch', type=int, default=0, help='number of start training epoch')
        parser.add_argument('--lrd_start_epoch', type=int, default=400, help='number of start lr decay epoch')
        parser.add_argument('--train_epoch', type=int, default=1801, help='number of train epochs')
        parser.add_argument('--D_start_epoch', type=int, default=0, help='start D epoch')
        parser.add_argument('--G_iter', type=int, default=3, help='G iters peer D iter')
        parser.add_argument('--lrD', type=float, default=0.00001, help='learning rate, default=0.00001')  #learning rates
        parser.add_argument('--lrG', type=float, default=0.00001, help='learning rate, default=0.00001')
        # parser.add_argument('--lrR', type=float, default=0.00001, help='learning rate, default=0.00001')
        parser.add_argument('--G_Decay_epoch', type=int, default=1000, help='learning rate decay epoch, default=100')
        parser.add_argument('--D_Decay_epoch', type=int, default=1000, help='learning rate decay epoch, default=100')
        parser.add_argument('--spe_inter_size', type=int, default=16, help='spetral select inter of spectral discriminator')

        parser.add_argument('--L1_lambda', type=float, default=10, help='lambda for L1 loss')
        parser.add_argument('--weight_l1', default=False, help='whether use weight for different band in L1 loss')
        parser.add_argument('--Cos_lambda', type=float, default=100, help='lambda for cosine loss')
        parser.add_argument('--Ssim_lambda', type=float, default=0, help='lambda for ssim loss')
        parser.add_argument('--Hsv_lambda', type=float, default=0, help='lambda for hsv loss')
        parser.add_argument('--Abun_reg_lambda', type=float, default=0, help='lambda for regulazation of abundance')
        parser.add_argument('--Res_reg_lambda', type=float, default=0, help='lambda for regulazation of residual spectral')
        parser.add_argument('--spe_lambda', type=float, default=1, help='lambda for spectral discriminator loss')
        parser.add_argument('--save_epoch', type=int, default=200, help='save model each save_epoch')
        parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for Adam optimizer')  # 0.5
        parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
        self.initialized = True
        return parser
