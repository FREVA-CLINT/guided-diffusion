"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch
import torch.distributed as dist
import xarray as xr

from guided_diffusion import config as cfg
from guided_diffusion import dist_util, logger
from guided_diffusion.evaluation import plot_images
from guided_diffusion.netcdfloader import EVANetCDFLoader, FrevaNetCDFLoader
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model,
    add_dict_to_argparser,
    create_gaussian_diffusion,
)


def main(arg_file=None):
    cfg.set_evaluate_args(arg_file)
    if not os.path.exists(cfg.eval_dir):
        os.makedirs(cfg.eval_dir)

    dist_util.setup_dist()
    logger.configure()

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

    model.load_state_dict(
        dist_util.load_state_dict("{}/ckpt/{}".format(cfg.snapshot_dir, cfg.model_name), map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    if cfg.data_root_dir:
        dataset = EVANetCDFLoader(data_root=cfg.data_root_dir, data_in_names=cfg.img_names,
                                  data_in_types=cfg.data_types, ensembles=cfg.train_ensembles, ssis=cfg.train_ssis,
                                  locations=cfg.locations)
    else:
        dataset = FrevaNetCDFLoader(project=cfg.freva_project, model=cfg.freva_model,
                                    experiment=cfg.freva_experiment, time_frequency=cfg.freva_time_frequency,
                                    data_types=cfg.data_types, gt_ensembles=cfg.gt_ensembles, realm=cfg.freva_realm,
                                    support_ensemble=cfg.val_ensemble,
                                    split_timesteps=cfg.split_timesteps)

    logger.log("sampling...")

    in_out_channels = len(cfg.data_types)
    if cfg.split_timesteps and not cfg.lstm:
        in_out_channels *= cfg.split_timesteps
    all_samples = []
    all_sample_names = []
    while len(all_samples) * cfg.batch_size < len(cfg.sample_names):
        sample_fn = (
            diffusion.p_sample_loop
        )
        model_kwargs = {}
        if cfg.n_classes:
            classes = torch.tensor(cfg.batch_size * [cfg.classes[(location, ssi)] for location in cfg.sample_locations
                                                     for ssi in cfg.sample_ssis
                                                     if (location == 'ne' and ssi == 0.0) or (
                                                             location != 'ne' and ssi != 0.0)])
            model_kwargs["y"] = classes.to(cfg.device)
            samples = sample_fn(
                model, None,
                (len(classes), in_out_channels, cfg.img_sizes[0], cfg.img_sizes[1]),
                clip_denoised=True,
                model_kwargs=model_kwargs,
            )
            sample_classes = [(location, int(ssi) if ssi % 2 == 0 else ssi)
                              for location in cfg.sample_locations for ssi in cfg.sample_ssis
                              if (location == 'ne' and ssi == 0.0) or (
                                      location != 'ne' and ssi != 0.0)]
            sample_names = []
            for ensemble in cfg.sample_names[len(all_samples)*cfg.batch_size:len(all_samples)*cfg.batch_size+cfg.batch_size]:
                sample_names += ['deva{}ssi{}{}_echam6_BOT_mm_'.format(int(sample_class[1]) if sample_class[1] % 2 == 0 else sample_class[1], sample_class[0] if sample_class[0] != "ne" else "", ensemble) for sample_class in sample_classes]
            all_sample_names.append(sample_names)
            all_samples.append(samples)
        else:
            support_image = [dataset.__getitem__(ensemble=cfg.gt_ensembles[0], timechunk=t)[1].to(cfg.device).repeat(cfg.batch_size, 1, 1, 1) for t in
                             cfg.sample_time_chunks]
            support_image = torch.cat(support_image)

            samples = sample_fn(
                model, support_image,
                (cfg.batch_size * len(cfg.sample_time_chunks), in_out_channels, cfg.img_sizes[0], cfg.img_sizes[1]),
                clip_denoised=True,
                model_kwargs={},
            )
            all_samples.append(samples)

    if cfg.create_output_files:
        for i in range(len(all_samples)):
            for j in range(len(all_samples[i])):
                sample = torch.stack(torch.split(all_samples[i][j], cfg.split_timesteps, dim=0), dim=0)
                create_outputs(sample, dataset, all_sample_names[i][j], dataset.xr_dss)

    if cfg.create_eval_images:
        if not os.path.exists('{}/images'.format(cfg.eval_dir)):
            os.makedirs('{}/images'.format(cfg.eval_dir))
        for t in cfg.sample_time_chunks:
            assert t < dataset.time_chunks
            support_image = dataset.__getitem__(ensemble=cfg.gt_ensembles[0], timechunk=t)[1].to(cfg.device)
            samples_by_timestep = torch.cat([sample[t*cfg.batch_size:(t+1)*cfg.batch_size, :, :, :] for sample in all_samples])
            gt_images = torch.stack([dataset.__getitem__(ensemble=ens, timechunk=t)[0] for ens in cfg.gt_ensembles]
                                    ).to(cfg.device)

            samples_by_timestep = torch.stack(torch.split(samples_by_timestep, cfg.split_timesteps, dim=1), dim=2)
            gt_images = torch.stack(torch.split(gt_images, cfg.split_timesteps, dim=1), dim=2)
            support_image = torch.stack(torch.split(support_image, cfg.split_timesteps, dim=0), dim=1)

            # mean over ensemble
            gan_ensemble_mean = torch.mean(samples_by_timestep, dim=0)
            gt_ensemble_mean = torch.mean(gt_images, dim=0)
            mean_over_ens = torch.stack([gan_ensemble_mean, gt_ensemble_mean], dim=0)
            plot_images(mean_over_ens, 'mean_over_ens', ["Mean Pred", "Mean GT"], cfg.eval_dir)

            # mean over ensemble and time
            mean_over_ens_and_time = torch.mean(mean_over_ens, dim=1).unsqueeze(1)
            plot_images(mean_over_ens_and_time, 'mean_over_ens_and_time', ["Mean Pred", "Mean GT"], cfg.eval_dir)

            # difference support
            diff_support = torch.sub(support_image.unsqueeze(0), samples_by_timestep)
            plot_images(diff_support, 'diff_support', cfg.sample_names, cfg.eval_dir)

            # difference between generated images
            diff_gan_images = torch.sub(gan_ensemble_mean.unsqueeze(0), samples_by_timestep)
            plot_images(diff_gan_images, 'diff_gan_images', cfg.sample_names, cfg.eval_dir)
            diff_ensembles = torch.sub(gt_ensemble_mean.unsqueeze(0), gt_images)
            plot_images(diff_ensembles[:diff_gan_images.shape[0]], 'diff_ensembles', cfg.gt_ensembles, cfg.eval_dir)

    dist.barrier()
    logger.log("sampling complete")


def create_outputs(sample, data_set, file_name, xr_dss):
    for v in range(len(cfg.data_types)):
        data_type = cfg.data_types[v]

        ds = xr_dss[1].copy()

        if cfg.normalization:
            sample[v, :, :, :] = data_set.img_normalizer.renormalize(sample[v, :, :, :], v)

        ds[data_type] = xr.DataArray(sample.to(torch.device('cpu')).detach().numpy()[v, :, :, :],
                                     dims=xr_dss[1].coords.dims, coords=xr_dss[1].coords)
        ds["time"] = xr_dss[0]["time"].values

        for var in xr_dss[0].keys():
            ds[var] = xr_dss[0][var]

        ds.attrs["history"] = "Infilled using CRAI (Climate Reconstruction AI: " \
                              "https://github.com/FREVA-CLINT/climatereconstructionAI)\n" + ds.attrs["history"]
        ds.to_netcdf('{}/{}{}_1992.nc'.format(cfg.eval_dir, file_name, data_type))


if __name__ == "__main__":
    main()
