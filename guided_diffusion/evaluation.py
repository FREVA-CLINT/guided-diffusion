import matplotlib.pyplot as plt
import numpy as np
import torch

from guided_diffusion import config as cfg


def plot_snapshot_images(gt_img, model, diffusion, filename, cond):

    in_out_channels = len(cfg.data_types)
    if cfg.split_timesteps and not cfg.lstm:
        in_out_channels *= cfg.split_timesteps

    model.eval()

    sample_fn = (
        diffusion.p_sample_loop
    )

    if cfg.n_classes is None:
        output = sample_fn(model, gt_img.repeat(cfg.n_images, 1, 1, 1).to(cfg.device),
                           (cfg.n_images, in_out_channels, cfg.img_sizes[0], cfg.img_sizes[1]),
                           clip_denoised=True,
                           model_kwargs={},
                           )
        plot_names = (cfg.n_images + 1) * [""]
    else:
        model_kwargs = {}
        classes = torch.randint(
            low=0, high=cfg.n_classes, size=(cfg.n_images,), device=cfg.device
        )
        model_kwargs["y"] = classes
        output = sample_fn(model, None,
                           (cfg.n_images, in_out_channels, cfg.img_sizes[0], cfg.img_sizes[1]),
                           clip_denoised=True,
                           model_kwargs=model_kwargs,
                           )
        classes = torch.cat([classes, torch.tensor([cond["y"][0]]).to(cfg.device)])
        plot_names = ["{} ssi{}".format(get_class_from_name([k for k, v in cfg.classes.items() if c == v][0][0]), [k for k, v in cfg.classes.items() if c == v][0][1]) for c in classes]
    output = output.unsqueeze(1).to(cfg.device)
    gt_img = gt_img.unsqueeze(0)
    if not cfg.lstm:
        output = torch.transpose(output, 1, 2)
        gt_img = torch.transpose(gt_img, 0, 1)
    output = torch.cat([output.to(cfg.device), gt_img.to(cfg.device).unsqueeze(0)], dim=0)
    plot_images(output, filename, plot_names, cfg.snapshot_dir)


def plot_images(images, filename, plot_names, directory):
    images = images.to(torch.device('cpu'))
    fig, axes = plt.subplots(nrows=images.shape[0], ncols=images.shape[1],
                             figsize=(images.shape[4] / 50 * images.shape[1], images.shape[3] * images.shape[0] / 50))
    font_size = 24

    # plot and save data
    fig.patch.set_facecolor('black')
    for c in range(images.shape[2]):
        if cfg.vlim:
            vmin = cfg.vlim[0]
            vmax = cfg.vlim[1]
        else:
            vmin = torch.min(images)
            vmax = torch.max(images)

        # plot and save data
        fig.patch.set_facecolor('black')
        for i in range(images.shape[0]):
            for j in range(images.shape[1]):
                if images.shape[0] > 1 and images.shape[1] > 1:
                    if plot_names[i]:
                        axes[i, 0].text(-1, 0.5, plot_names[i],
                                size=font_size, va="center", transform=axes[i, 0].transAxes, color="white")
                    axes[i, j].axis("off")
                    axes[i, j].imshow(np.squeeze(images[i, j, c, :, :]), vmin=vmin, vmax=vmax)
                elif images.shape[1] > 1:
                    if plot_names[i]:
                        axes[0].text(-1, 0.5, plot_names[i],
                                size=font_size, va="center", transform=axes[0].transAxes, color="white")
                    axes[j].axis("off")
                    axes[j].imshow(np.squeeze(images[i, j, c, :, :]), vmin=vmin, vmax=vmax)
                else:
                    if plot_names[i]:
                        axes[i].text(-1, 0.5, plot_names[i],
                                size=font_size, va="center", transform=axes[i].transAxes, color="white")
                    axes[i].axis("off")
                    axes[i].imshow(np.squeeze(images[i, j, c, :, :]), vmin=vmin, vmax=vmax)
        plt.subplots_adjust(wspace=0.01, hspace=0.01, left=0, right=1, bottom=0, top=1)
        plt.savefig('{}/images/{}_{}.jpg'.format(directory, filename, cfg.data_types[c]), bbox_inches='tight')

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

def get_class_from_name(gt_class):
    if gt_class == 'nh':
        return "NH"
    elif gt_class == 'sh':
        return "SH"
    elif gt_class == 'ne':
        return "No Eru"
    else:
        return "Trop"