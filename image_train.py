"""
Train a diffusion model on images.
"""
import argparse
from datasets.latent_image_datasets import load_data
from utils import logger
import torch
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from utils.train_util import TrainLoop


def main():
    args = create_argparser().parse_args()

    # dist_util.setup_dist()
    logger.configure(dir = args.work_dir)#注册logger

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )#create model and diffusion phrase UNetModel, Spaced diffusion
    gpu_ids = (0,),
    device = torch.device("cuda:%s" % gpu_ids[0]
                          if torch.cuda.is_available() and len(gpu_ids) > 0
                          else "cpu")
    model = model.to(device)
    # model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        code_dir=args.code_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data = data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        # ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir = "F:/datasets/grss_dfc_2018/train_mini",
        code_dir = "results/20230407_latent16/train_mini",
        work_dir = "results/20230517_atten",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=32,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        save_interval=10000,#10000
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
