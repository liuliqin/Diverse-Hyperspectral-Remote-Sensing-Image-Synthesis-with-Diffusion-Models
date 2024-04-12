from .base_options import BaseOptions

class InferOptions(BaseOptions):
    def __init__(self):
        self.initialized = False

    def initialize(self,parser):
        BaseOptions.initialize(self,parser)
        parser.add_argument('--crop_size', type=int, default=256, help='input size')
        parser.add_argument('--model_epoch', default='1801', help='Which model to test?')
        parser.add_argument('--test_flag', default='test_mini', help='train or test_mini')  # _mini
        parser.add_argument('--abun_flag', default=False, help='save abundance mat')
        parser.add_argument('--gen_time', type=int, default=10, help='Which epoch model to latent')
        parser.add_argument('--latent_root', default='20230517_atten', help='which diffusion root')
        parser.add_argument('--latent_epoch', default='004000', help='Which epoch model to latent')
        parser.add_argument('--cal_performance', default=True, help='cal performance?')

        self.initialized =True
        return parser