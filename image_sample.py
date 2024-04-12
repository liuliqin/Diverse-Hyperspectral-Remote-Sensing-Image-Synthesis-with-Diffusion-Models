"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import scipy.io as scio
import numpy as np
import torch as th
import torch.distributed as dist
import sys
sys.path.append('..')
from utils import dist_util
from datasets.latent_image_datasets import load_data
from utils.logger_utils import write_img
from utils.calcul_metric import Cal

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
# import pdb


def main():
    # pdb.set_trace()
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    model_path = os.path.join(args.eval_dir,'model'+args.model_step+'.pt')
    save_path = os.path.join(args.eval_dir, args.model_step)
    os.makedirs(save_path, exist_ok=True)
    eval_file = os.path.join(save_path, 'performance.txt')
    # cal_performance = Cal(num_range=65535)
    # logger.configure(dir=args.eval_dir)
    print("creating model and diffusion...")

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    data = load_data(
        data_dir=args.data_dir,
        code_dir=args.code_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        deterministic=True,
    )
    device = th.device("cuda:%s" % args.gpu_ids[0]
                            if th.cuda.is_available() and len(args.gpu_ids) > 0
                            else "cpu")
    # data_mean = 3652.5
    # data_std = 2450.0
    # data_mean = data_mean + data_std
    # data_std = 2.5 * data_std
    print("sampling...")
    # all_images = th.zeros(args.num_images,48,args.img_size,args.img_size)
    # all_names =[]
    # test_psnr = 0
    mse_total = 0

    with open(eval_file, 'w') as f:
        # f.write(f'img_size {args.image_size}')
        f.write('\n')
    for i in range(args.num_images//args.batch_size):

        batch_img,cond_img,batch_name=next(data)
        cond_img = cond_img.to(device)
        batch_img =batch_img.to(device)
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (args.batch_size, 16, args.image_size//(2**args.down_factor), args.image_size//(2**args.down_factor)),
            clip_denoised=args.clip_denoised,
            model_kwargs=cond_img,
        )
        # sample = (sample*data_std+data_mean).clamp(0,65535).to(th.int16)
        # sample = ((sample + 1) * (4095 / 2)).clamp(0, 65535).to(th.int16)
        print(f'successfully generate batch {(i):2d}')
        # psnr = batch_PSNR(batch_img,sample,data_range=4095)

        for i,name in enumerate(batch_name):
        #     gen_hyper = (sample[i].cpu().detach().numpy()).astype('float64')
        #     real_hyper = (batch_img[i].cpu().detach().numpy() * data_std + data_mean).astype('float64')
        #     rmse, mrae, sam, mssim, mpsnr = cal_performance.calcul_one_img(gen_hyper, real_hyper)
        #     print(mpsnr)
        #     test_psnr += mpsnr
        #     with open(eval_file, 'a') as f:
        #         f.write(f'{name} {(rmse):.4f} {(mrae):.4f} {(sam):.4f} {(mssim):.6f} {(mpsnr):.6f}')
        #         f.write('\n')

            # img_path = os.path.join(save_path,name.replace('.tif','pre.tif'))
            # real_img_path = os.path.join(save_path,name)
            # write_img(img_path, gen_hyper)
            # write_img(real_img_path,real_hyper)
            gen_latent = (sample[i].cpu().detach().numpy())*0.24
            real_latent = (batch_img[i].cpu().detach().numpy())*0.24
            latent_path =os.path.join(save_path,name.replace('.tif','.mat'))
            scio.savemat(latent_path,{'code':gen_latent})
            rmse = np.mean((real_latent-gen_latent)**2)
            mse_total+=rmse
            with open(eval_file, 'a') as f:
                f.write(f'{(rmse):.4f}')
                f.write('\n')
    # mean_rmse, mean_mrae, mean_sam, mean_ssim, mean_psnr = cal_performance.calcul_mean()
    mean_mse = mse_total/args.num_images
    with open(eval_file, 'a') as f:
        f.write(f'mean of all: {(mean_mse):.4f}')

    print("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised = True,# whether clip to -1,1
        data_dir = 'F:/datasets/grss_dfc_2018/test_mini',
        code_dir = "results/20230407_latent16/test_mini",
        eval_dir = 'results/20230517_atten',
        batch_size = 12,
        use_ddim = False,
        model_step='004000',
        num_images = 12,
        gpu_ids=(0,),
        down_factor =2,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
