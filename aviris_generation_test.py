import torch
from datasets.datasets import ImageDataset
from interface.aviris_generation import AvirisGeneration
#from utils.util import write_img
from options.test_options import TestOptions
import h5py
import numpy as np



if __name__=='__main__':
    option = TestOptions()
    opt = option.parse(save = False)
    print(opt)

    test_dataset = ImageDataset('F:/datasets/'+opt.dataset+'/'+opt.test_flag ,opt.crop_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    data = h5py.File(opt.lib_path)
    spe_lib = torch.tensor(data['spe_grss']).float().cuda()
    bright = torch.Tensor(np.random.rand(48)).unsqueeze(1).float().cuda()
    aviris_gen = AvirisGeneration(spe_lib, opt, bright,mode='Test')
    # build networks
    aviris_gen.build_networks()
    aviris_gen.load_checkpoint(opt.model_epoch)
    print('test start!')
    aviris_gen.encode(test_loader)
    # aviris_gen.eval_one_epoch(test_loader)

