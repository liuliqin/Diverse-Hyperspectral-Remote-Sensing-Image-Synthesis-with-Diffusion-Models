import torch
import os
from datasets.datasets import ImageDataset
import h5py
from interface.aviris_generation import AvirisGeneration
from options.train_options import TrainOptions
from scipy import io as scio

if __name__ =="__main__":
    option = TrainOptions()
    opt = option.parse()
    print(opt)#disp parameters

    #data loaders
    train_dataset = ImageDataset('F:/datasets/'+opt.dataset+'/train',opt.crop_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_dataset = ImageDataset('F:/datasets/' + opt.dataset + '/test', 512)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # spe_lib load
    data = h5py.File(opt.lib_path)
    spe_lib = torch.tensor(data['spe_grss']).float().cuda()

    # band weight
    if opt.weight_l1:
        coff=h5py.File('mean_std_mini_train.mat') # band weights for l1 loss
        loss_weight = torch.tensor(coff['std_history']).unsqueeze(0).unsqueeze(2).float().cuda()
    else:
        loss_weight =torch.ones(48,1)

    # bright_initial

    ini_saa = torch.tensor(scio.loadmat('bright.mat')['bright']).float().cuda().t()
    # bright = h5py.File('ini_bright.mat')
    # ini_saa = torch.tensor(bright['bright']).float().cuda()# 224*1

    # ESTABLISH MODEL
    aviris_gen = AvirisGeneration(spe_lib,opt,ini_saa,weight_band=loss_weight)
    # Initial network
    aviris_gen.build_networks()
    # Set optimizer
    aviris_gen.set_optimizers()
    # load model
    aviris_gen.load_checkpoint()
    # set_logger
    aviris_gen.set_logger(train_loader,test_loader)
    # training loop
    aviris_gen.training_loop(train_loader,test_loader)
    # save log
    aviris_gen.save_log()