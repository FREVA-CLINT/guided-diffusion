import numpy as np
from torchvision import transforms
from . import config as cfg


class DataNormalizer:
    def __init__(self, data, normalization):
        self.data_std, self.data_mean, self.data_min, self.data_max, self.data_tf = [], [], [], [], []

        self.normalization = normalization

        for i in range(len(data)):
            self.data_mean.append(np.nanmean(data[i]))
            self.data_std.append(np.nanstd(data[i]))
            self.data_min.append(np.nanmin(data[i]))
            self.data_max.append(np.nanmax(data[i] - self.data_min[-1]))
            if normalization == 'std':
                self.data_tf.append(transforms.Normalize(mean=[self.data_mean[-1]], std=[self.data_std[-1]]))
            elif normalization == 'img':
                self.data_tf.append(transforms.Normalize(mean=0.5, std=0.5))
            elif normalization == 'custom':
                self.data_tf.append(transforms.Normalize(mean=cfg.custom_mean, std=cfg.custom_std))

    def normalize(self, data, index):
        if self.normalization == 'std' or cfg.normalization == 'custom':
            return self.data_tf[index](data)
        elif self.normalization == 'img':
            return self.data_tf[index]((data - self.data_min[index]) / self.data_max[index])
        else:
            return data

    def renormalize(self, data, index):
        if self.normalization == 'std':
            return self.data_std[index] * data + self.data_mean[index]
        elif self.normalization == 'img':
            return (0.5 * data + 0.5) * self.data_max[index] + self.data_min[index]
        elif self.normalization == 'custom':
            return cfg.custom_std * data + cfg.custom_mean
        else:
            return data
