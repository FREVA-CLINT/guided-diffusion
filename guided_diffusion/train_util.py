import copy
import functools
import os

import blobfile as bf
import torch
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .evaluation import plot_snapshot_images
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from guided_diffusion import config as cfg
from tqdm import tqdm

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
            ema_rate,
            log_interval,
            save_model_interval,
            resume_iter,
            use_fp16=False,
            fp16_scale_growth=1e-3,
            schedule_sampler=None,
            weight_decay=0.0,
            lr_anneal_steps=0,
            max_iter
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_model_interval = save_model_interval
        self.resume_iter = resume_iter
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.max_steps = max_iter
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = torch.cuda.is_available()

        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_iter:
            self.load_model()
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if torch.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

        dist_util.sync_params(self.model.parameters())

    def load_model(self):
        logger.log(f"loading model from iteration: {self.resume_iter}...")

        # load model
        self.model.load_state_dict(dist_util.load_state_dict(
            "{}/model{}.pth".format(get_snapshot_dir(), self.resume_iter), map_location=dist_util.dev()
        ))

        # load optimizer
        self.opt.load_state_dict(dist_util.load_state_dict(
            "{}/opt{}.pth".format(get_snapshot_dir(), self.resume_iter), map_location=dist_util.dev()
        ))

        # load ema params
        self.ema_params = []
        for rate in self.ema_rate:
            self.ema_params.append(copy.deepcopy(self.mp_trainer.master_params))
            if dist.get_rank() == 0:
                self.mp_trainer.state_dict_to_master_params(dist_util.load_state_dict(
                    "{}/ema_{}_{}.pth".format(get_snapshot_dir(), rate, self.resume_iter),
                    map_location=dist_util.dev()
                ))
            dist_util.sync_params(self.ema_params[-1])

    def run_loop(self):
        pbar = tqdm(range(self.resume_iter, self.max_steps))

        for i in pbar:
            batch, support_batch, cond = next(self.data)
            self.run_step(batch, support_batch, cond)
            if cfg.save_snapshot_image_interval and self.step % cfg.save_snapshot_image_interval == 0:
                plot_snapshot_images(support_batch, self.model, self.diffusion, "iter_{}".format(self.step))
            if self.log_interval and self.step % self.log_interval == 0 and self.step > 0:
                logger.dumpkvs()
            if self.step % self.save_model_interval == 0 and self.step > 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_model_interval != 0:
            self.save()

    def run_step(self, batch, support_batch, cond):
        self.forward_backward(batch, support_batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, support_batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            if not cfg.n_classes:
                micro_support = support_batch[i : i + self.microbatch].to(dist_util.dev())
            else:
                micro_support = None
            micro_cond = {
                k: v[i: i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                micro_support,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_iter) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_iter)
        logger.logkv("samples", (self.step + self.resume_iter + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step + self.resume_iter)}.pth"
                else:
                    filename = f"ema_{rate}_{(self.step + self.resume_iter)}.pth"
                with bf.BlobFile(bf.join(get_snapshot_dir(), filename), "wb") as f:
                    torch.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                    bf.join(get_snapshot_dir(), f"opt{(self.step + self.resume_iter)}.pth"),
                    "wb",
            ) as f:
                torch.save(self.opt.state_dict(), f)

        dist.barrier()


def get_snapshot_dir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return "{}/ckpt/".format(cfg.snapshot_dir)


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
