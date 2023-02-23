"""
Train a diffusion model on images.
"""

import os

from torch.utils.data import DataLoader

from guided_diffusion import config as cfg
from guided_diffusion import dist_util, logger
from guided_diffusion.netcdfloader import FrevaNetCDFLoader, InfiniteSampler, EVANetCDFLoader
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    create_model, create_gaussian_diffusion,
)
from guided_diffusion.train_util import TrainLoop


def main(arg_file=None):
    cfg.set_train_args(arg_file)
    for subdir in ("", "/images", "/ckpt"):
        outdir = cfg.snapshot_dir + subdir
        if not os.path.exists(outdir):
            os.makedirs(outdir)

    dist_util.setup_dist()
    logger.configure()
    logger.log("creating data loader...")

    if cfg.data_root_dir:
        dataset = EVANetCDFLoader(data_root=cfg.data_root_dir, data_in_names=cfg.img_names,
                                  data_in_types=cfg.data_types, ensembles=cfg.train_ensembles, ssis=cfg.train_ssis,
                                  locations=cfg.locations)
    else:
        dataset = FrevaNetCDFLoader(project=cfg.freva_project, model=cfg.freva_model,
                                    experiment=cfg.freva_experiment, time_frequency=cfg.freva_time_frequency,
                                    data_types=cfg.data_types, gt_ensembles=cfg.gt_ensembles, realm=cfg.freva_realm,
                                    support_ensemble=cfg.support_ensemble,
                                    split_timesteps=cfg.split_timesteps)

    data = iter(DataLoader(dataset, batch_size=cfg.batch_size,
                           sampler=InfiniteSampler(len(dataset)),
                           num_workers=cfg.n_threads))

    logger.log("creating model and diffusion...")
    model = create_model(
        image_size=cfg.img_sizes,
        num_channels=cfg.num_channels,
        num_res_blocks=cfg.n_residual_blocks,
        channel_mult=cfg.conv_factors,
        learn_sigma=False,
        use_checkpoint=False,
        attention_resolutions=cfg.attention_res,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        dropout=cfg.dropout,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
    )
    diffusion = create_gaussian_diffusion(
        steps=cfg.diffusion_steps,
        learn_sigma=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing=False,
    )

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=cfg.batch_size,
        microbatch=-1, # -1 disables microbatches
        lr=cfg.lr,
        ema_rate="0.9999",
        log_interval=cfg.log_interval,
        save_model_interval=cfg.save_model_interval,
        resume_iter=cfg.resume_iter,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=schedule_sampler,
        weight_decay=0.0,
        lr_anneal_steps=0,
        max_iter=cfg.max_iter
    ).run_loop()


if __name__ == "__main__":
    main()
