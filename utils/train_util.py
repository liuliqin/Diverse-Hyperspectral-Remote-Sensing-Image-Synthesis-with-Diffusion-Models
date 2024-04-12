import copy
import functools
import os

import torch as th
from torch.optim import AdamW

from . import logger
from networks.fp16_util import MixedPrecisionTrainer
from guided_diffusion.resample import LossAwareSampler, UniformSampler
from .logger_utils import Logger

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        log_interval,
        save_interval,
        resume_checkpoint,
        gpu_ids=(0,),
        use_fp16=False,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        per_eval_steps=-1,
        do_eval_func=None,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size

        self._load_parameters()
        self.use_fp16 = use_fp16
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = th.device("cuda:%s" % gpu_ids[0]
                                if th.cuda.is_available() and len(gpu_ids) > 0
                                else "cpu")
        self.model = self.model.to(self.device)
        self.per_eval_steps = per_eval_steps
        self.do_eval_func = do_eval_func
        # self.train_logger = Logger(100000 if self.lr_anneal_steps==0 else self.lr_anneal_steps,1,window_name=logger.get_dir().split('/')[-1])


    def _load_parameters(self):
        if find_resume_checkpoint():
            # 首先看看在log目录里面有没有step.txt，若有说明可以加载本目录下的modellatest
            self.resume_step = find_resume_checkpoint()
            resume_checkpoint = os.path.join(get_blob_logdir(), 'modellatest.pt')
        elif self.resume_checkpoint:
            resume_checkpoint = self.resume_checkpoint
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
        else:
            resume_checkpoint = None

        if resume_checkpoint:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                th.load(
                    resume_checkpoint, map_location='cpu'
                )
            )

    def _load_optimizer_state(self):
        main_checkpoint = get_blob_logdir() or self.resume_checkpoint

        opt_checkpoint = os.path.join(
            os.path.dirname(main_checkpoint), f"optlatest.pt"
        )
        if os.path.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(
                opt_checkpoint, map_location='cpu'
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond, _ = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
                self.save(is_latest=True)
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
            # print(self.step)
            if self.per_eval_steps != -1 and \
                    ((self.step + self.resume_step) % self.per_eval_steps
                     == self.per_eval_steps - 1):
                out_vis_folder = os.path.join(get_blob_logdir(), f'vis_step_{self.step+self.resume_step}')
                os.makedirs(out_vis_folder, exist_ok=True)
                self.do_eval_func(out_dir=out_vis_folder)
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(self.device)
            # micro_cond = {
            #     k: v[i : i + self.microbatch].to(self.device)
            #     for k, v in cond.items()
            # }
            micro_cond = cond[i : i + self.microbatch].to(self.device)
            # last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], self.device)

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            # loss = losses["loss"]
            # mse_loss = (losses["mse"] * weights).mean()
            # logger.logkv_mean("total_loss", losses["loss"].mean().item())
            logger.logkv_mean("mse_loss", losses["mse"].mean().item())
            # logger.logkv_mean("l1_loss", losses["L1"].item())
            # logger.logkv_mean("sam_loss", losses["SAM"].item())
            logger.logkv_mean("ssim_loss", losses["SSIM"].item())
            # self.train_logger.log({"train_loss":loss,"sam_loss":losses["SAM"],"ssim_loss":losses["SSIM"],"l1_loss":losses["L1"],"noise_mse":mse_loss}, images={})
            logger.logkv("loss", loss.item())
            # log_loss_dict(
            #     self.diffusion, t, {k: v * weights for k, v in losses.items()}
            # )
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self, is_latest=False):
        name = f'{(self.step+self.resume_step):06d}'
        if is_latest:
            name = 'latest'
        # else:
        #     # save window
        #     self.train_logger.save(os.path.join(get_blob_logdir(), name + '.log'), logger.get_dir().split('/')[-1])

        with open(os.path.join(get_blob_logdir(), 'step.txt'), 'w') as f:
            f.write(f'{(self.step+self.resume_step):06d}')

        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            logger.log(f"saving model...")
            filename = f"model{name}.pt"
            th.save(state_dict, os.path.join(get_blob_logdir(), filename))

        save_checkpoint(self.mp_trainer.master_params)

        th.save(self.opt.state_dict(),  os.path.join(get_blob_logdir(), f"opt{name}.pt"))


class BasicCDTrainLoop:
    def __init__(
        self,
        *,
        model,
        data,
        batch_size,
        lr,
        log_interval,
        save_interval,
        resume_checkpoint,
        gpu_ids=(0,),
        use_fp16=False,
        weight_decay=0.05,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size

        self._load_parameters()
        self.use_fp16 = use_fp16
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
        )
        from guided_diffusion.losses import cross_entropy
        self.cal_loss = cross_entropy
        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.

        self.device = th.device("cuda:%s" % gpu_ids[0]
                                if th.cuda.is_available() and len(gpu_ids) > 0
                                else "cpu")
        self.model = self.model.to(self.device)

    def _load_parameters(self):
        if find_resume_checkpoint():
            # 首先看看在log目录里面有没有step.txt，若有说明可以加载本目录下的modellatest
            self.resume_step = find_resume_checkpoint()
            resume_checkpoint = os.path.join(get_blob_logdir(), 'modellatest.pt')
        elif self.resume_checkpoint:
            resume_checkpoint = self.resume_checkpoint
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
        else:
            resume_checkpoint = None

        if resume_checkpoint:
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(
                th.load(
                    resume_checkpoint, map_location='cpu'
                )
            )

    def _load_optimizer_state(self):
        main_checkpoint = get_blob_logdir() or self.resume_checkpoint

        opt_checkpoint = os.path.join(
            os.path.dirname(main_checkpoint), f"optlatest.pt"
        )
        if os.path.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(
                opt_checkpoint, map_location='cpu'
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
                self.save(is_latest=True)
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        self.mp_trainer.optimize(self.opt)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()

        pred = self.model(batch.to(self.device))
        loss = self.cal_loss(pred, cond.to(self.device))
        logger.logkv('loss', loss.item())

        self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self, is_latest=False):
        name = f'{(self.step+self.resume_step):06d}'
        if is_latest:
            name = 'latest'

        with open(os.path.join(get_blob_logdir(), 'step.txt'), 'w') as f:
            f.write(f'{(self.step+self.resume_step):06d}')

        def save_checkpoint(params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            logger.log(f"saving model...")
            filename = f"model{name}.pt"
            th.save(state_dict, os.path.join(get_blob_logdir(), filename))

        save_checkpoint(self.mp_trainer.master_params)

        th.save(self.opt.state_dict(),  os.path.join(get_blob_logdir(), f"opt{name}.pt"))


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    step_txt = os.path.join(get_blob_logdir(), 'step.txt')
    if os.path.exists(step_txt):
        with open(step_txt, 'r') as f:
            step = f.read()
            return int(step)
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
