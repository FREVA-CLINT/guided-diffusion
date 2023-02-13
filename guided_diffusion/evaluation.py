import matplotlib.pyplot as plt
import numpy as np
import torch

from guided_diffusion import config as cfg


def plot_snapshot_images(model, diffusion, filename):
    model.eval()

    sample_fn = (
        diffusion.p_sample_loop
    )
    sample = sample_fn(model,
            (cfg.n_images, 3, cfg.img_sizes[0], cfg.img_sizes[0]),
            clip_denoised=True,
            model_kwargs={},
    )
    plot_images(sample, filename)


def plot_images(images, filename):
    images = images.to(torch.device('cpu'))

    fig, axes = plt.subplots(ncols=images.shape[0],
                             figsize=(images.shape[0] * 2, 2))

    # plot and save data
    fig.patch.set_facecolor('black')
    for i in range(images.shape[0]):
        if images.shape[0] > 1:
            axes[i].imshow(torch.movedim(images[i], 0, 2))
        else:
            axes.imshow(torch.movedim(images[i], 0, 2))
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
