import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import pyplot

from guided_diffusion import config as cfg


def plot_snapshot_images(gt_image, model, diffusion, filename):
    in_out_channels = len(cfg.data_types)
    if cfg.split_timesteps and not cfg.lstm:
        in_out_channels *= cfg.split_timesteps

    model.eval()

    sample_fn = (
        diffusion.p_sample_loop
    )
    output = sample_fn(model,
            (cfg.n_images, in_out_channels, cfg.img_sizes[0], cfg.img_sizes[0]),
            clip_denoised=True,
            model_kwargs={},
    )
    output = output.unsqueeze(1)
    gt_image = gt_image.unsqueeze(0)
    if not cfg.lstm:
        output = torch.transpose(output, 1, 2)
        gt_image = torch.transpose(gt_image, 0, 1)
    output = torch.cat([output.to(cfg.device), gt_image.unsqueeze(0).to(cfg.device)], dim=0)
    plot_images(output, filename)


def plot_images(images, filename):
    images = images.to(torch.device('cpu'))
    fig, axes = plt.subplots(nrows=images.shape[0], ncols=images.shape[1],
                             figsize=(images.shape[4] / 50 * images.shape[1], images.shape[3] * images.shape[0] / 50))

    # plot and save data
    fig.patch.set_facecolor('black')
    for c in range(images.shape[2]):
        vmin = -1
        vmax = 1

        # plot and save data
        fig.patch.set_facecolor('black')
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                if images.shape[0] > 1 and images.shape[1] > 1:
                    axes[i, j].axis("off")
                    if cfg.test:
                        axes[i, j].imshow(images[i])
                    else:
                        axes[i, j].imshow(np.squeeze(images[i, j, c, :, :]), vmin=vmin, vmax=vmax)
                elif images.shape[1] > 1:
                    axes[j].axis("off")
                    if cfg.test:
                        axes[j].imshow(images[j])
                    else:
                        axes[j].imshow(np.squeeze(images[i, j, c, :, :]), vmin=vmin, vmax=vmax)
                else:
                    axes[i].axis("off")
                    if cfg.test:
                        axes[i].imshow(images[i])
                    else:
                        axes[i].imshow(np.squeeze(images[i, j, c, :, :]), vmin=vmin, vmax=vmax)
        plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0, right=1, bottom=0, top=1)
        plt.savefig('{}/images/{}_{}.jpg'.format(cfg.snapshot_dir, filename, c), bbox_inches='tight')

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
