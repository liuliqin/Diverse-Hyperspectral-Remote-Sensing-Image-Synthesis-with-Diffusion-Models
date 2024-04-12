import argparse,os

class BaseOptions():
    def __init__(self):
        self.initialized = False
    def initialize(self,parser):
        parser.add_argument('--dataset', required=False, default='grss_dfc_2018', help='')#dataset
        parser.add_argument('--device', required=False, default='cuda:0', help='device GPU ID')
        parser.add_argument('--input_channel', type=int, default=48, help='input msi channel')# 224
        parser.add_argument('--lib_path', required=False, default='spe_grss_345.mat', help='spectral lib path') # spe_chose_345
        parser.add_argument('--abun_num', type=int, default=345, help='spectral lib path')
        parser.add_argument('--ngf', type=int, default=128)
        parser.add_argument('--ndf', type=int, default=128)
        parser.add_argument('--downscale_factor', type=int, default=2, help='down scale factor 2 ** downscale_factor')
        parser.add_argument('--z_channel', type=int, default=256, help='channel of latent codes before mapping')
        parser.add_argument('--latent_channel', type=int, default=16, help='channel of latent codes')
        parser.add_argument('--embed_num', type=int, default=1024, help='channel of latent codes')
        parser.add_argument('--save_root', required=False, default='results', help='results save path')
        parser.add_argument('--identy_root', required=False, default='20230407_latent16', help='identy path')#
        self.initialized = True
        return parser
    def parse(self,save = True):
        if not self.initialized:
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt = parser.parse_args()
        if save:
            root = os.path.join(opt.save_root, opt.identy_root)  #
            if not os.path.isdir(root):
                os.mkdir(root)
            file_name = os.path.join(root, 'options.txt')
            with open(file_name, 'w') as f:
                for key, value in opt.__dict__.items():
                    f.write(key + '\t' + str(value) + '\n')
        self.opt = opt
        return self.opt
