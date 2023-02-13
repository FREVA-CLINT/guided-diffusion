import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import pyplot

from guided_diffusion import config as cfg


def plot_snapshot_images(gt_image, model, diffusion, filename):
    model.eval()

    sample_fn = (
        diffusion.p_sample_loop
    )
    sample = sample_fn(model,
            (cfg.n_images, 3, cfg.img_sizes[0], cfg.img_sizes[0]),
            clip_denoised=True,
            model_kwargs={},
    )
    gt_image = gt_image.unsqueeze(0).repeat(cfg.n_images, 1, 1, 1)
    plot_images(gt_image, sample, filename)


def plot_images(gt_images, images, filename):
    images = images.to(torch.device('cpu'))

    fig, axes = plt.subplots(nrows=2, ncols=images.shape[0],
                             figsize=(images.shape[0] * 2, 4))

    # plot and save data
    fig.patch.set_facecolor('black')
    for i in range(images.shape[0]):
        if images.shape[0] > 1:
            axes[0][i].imshow(images[i, 0], vmin=-1, vmax=1)
            axes[1][i].imshow(gt_images[i, 0], vmin=-1, vmax=1)
        else:
            axes[0].imshow(images[i, 0], vmin=-1, vmax=1)
            axes[1].imshow(gt_images[i, 0], vmin=-1, vmax=1)
    plt.savefig('{}/images/{}.jpg'.format(cfg.snapshot_dir, filename))

    plt.clf()
    plt.close('all')


def generate_images(model, n_images, support_img=None):
    if cfg.gan_type == "lstm-gan":
        support_img = support_img.unsqueeze(0).repeat(n_images, 1, 1, 1, 1)
        noise = torch.randn(support_img.shape, device=cfg.device)
        noise = torch.cat([support_img, noise], dim=2)
    else:
        noise = torch.randn(n_images, cfg.seed_size, 1, 1, device=cfg.device)

    with torch.no_grad():
        output = model(noise)
    return output
