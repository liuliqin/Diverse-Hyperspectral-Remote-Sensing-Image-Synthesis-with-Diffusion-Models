import torch
import os
# from datasets.datasets import ImageDataset
from datasets.image_datasets import ImageDataset
from interface.aviris_generation import AvirisGeneration
#from utils.util import write_img
from options.infer_options import InferOptions

import h5py
import numpy as np
# import time

if __name__=='__main__':
    option = InferOptions()
    opt = option.parse(save = False)
    print(opt)
    # T1=time.time()
    data = h5py.File(opt.lib_path)
    spe_lib = torch.tensor(data['spe_grss']).float().cuda()
    bright = torch.Tensor(np.random.rand(48)).unsqueeze(1).float().cuda()
    aviris_gen = AvirisGeneration(spe_lib, opt, bright, mode='Test')
    # build networks
    aviris_gen.build_networks()
    aviris_gen.load_checkpoint(opt.model_epoch)
    # Use the diffusion model to generate latent code
    data_dir = os.path.join('F:/datasets',opt.dataset,opt.test_flag)
    code_dir = os.path.join(opt.save_root,opt.identy_root,opt.test_flag)
    eval_dir = os.path.join(opt.save_root,opt.latent_root)
    # os.system("python ../guided-diffusion-hyper-syns/image_sample.py --data_dir %s --code_dir %s --eval_dir %s --model_step" %(data_dir,code_dir,eval_dir,opt.latent_epoch))

    for t in range(0,opt.gen_time):
        os.system("E:\conda_envs/new_envs_swinT/python image_sample.py --data_dir %s --code_dir %s --eval_dir %s --model_step %s" % (data_dir, code_dir, eval_dir, opt.latent_epoch))
        latent_path =os.path.join(eval_dir,opt.latent_epoch)
        # latent_path = os.path.join(opt.save_root,'opt.identy_root,opt.test_flag)
        test_dataset = ImageDataset(opt.crop_size,latent_path,data_dir)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
        aviris_gen.interface_one_epoch(test_loader,t)

