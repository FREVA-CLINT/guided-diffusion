import os

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, Sampler

from guided_diffusion.normalizer import DataNormalizer
from guided_diffusion.load_data_paths import load_paths
from guided_diffusion import config as cfg


def load_steadymask(path, mask_name, data_type, device):
    if mask_name is None:
        return None
    else:
        steady_mask, _ = load_netcdf(path, [mask_name], [data_type])
        return torch.from_numpy(steady_mask[0]).to(device)


class InfiniteSampler(Sampler):
    def __init__(self, num_samples, data_source=None):
        super().__init__(data_source)
        self.num_samples = num_samples

    def __iter__(self):
        return iter(self.loop())

    def __len__(self):
        return 2 ** 31

    def loop(self):
        i = 0
        np.random.seed(cfg.random_seed)
        order = np.random.permutation(self.num_samples)
        while True:
            yield order[i]
            i += 1
            if i >= self.num_samples:
                np.random.seed(cfg.random_seed)
                order = np.random.permutation(self.num_samples)
                i = 0


def nc_loadchecker(filename, data_type, keep_dss=False):
    basename = filename.split("/")[-1]

    if not os.path.isfile(filename):
        print('File {} not found.'.format(filename))

    try:
        # We use load_dataset instead of open_dataset because of lazy transpose
        ds = xr.load_dataset(filename, decode_times=True)
        if ('r0' in basename or 'r1' in basename) and 'i1850p3' in basename:
            ds[data_type].values = np.flip(ds[data_type].values, axis=1)

    except Exception:
        raise ValueError('Impossible to read {}.'
                         '\nPlease, check that it is a netCDF file and it is not corrupted.'.format(basename))

    ds1 = ds

    if keep_dss:
        dtype = ds[data_type].dtype
        ds = ds.drop_vars(data_type)
        ds[data_type] = np.empty(0, dtype=dtype)
        return [ds, ds1], [ds1[data_type].values]
    else:
        return None, [ds1[data_type].values]


def load_netcdf(data_paths, data_types, keep_dss=False):
    if data_paths is None:
        return None, None
    else:
        ndata = len(data_paths)
        assert ndata == len(data_types)

        dss, data = nc_loadchecker(data_paths[0], data_types[0], keep_dss=keep_dss)
        lengths = [len(data[0])]
        for i in range(1, ndata):
            data += nc_loadchecker(data_paths[i], data_types[i])[1]
            lengths.append(len(data[-1]))

        if keep_dss:
            return dss, data, lengths[0]
        else:
            return data, lengths[0]


class FrevaNetCDFLoader(Dataset):
    def __init__(self, project, model, experiment, time_frequency, data_types, gt_ensembles, support_ensemble,
                 split_timesteps=0, mode='train'):
        super(FrevaNetCDFLoader, self).__init__()

        self.data_types = data_types
        self.split_timesteps = split_timesteps
        self.time_chunks = 0
        self.img_data, self.support_data = [], []

        for type in data_types:
            ensemble_data = []
            for ensemble in gt_ensembles + [support_ensemble]:
                paths = load_paths(project=project, model=model, experiment=experiment, time_frequency=time_frequency,
                                   variable=type, ensemble=ensemble)
                paths = list(paths)
                data, _ = load_netcdf(paths, len(paths) * [type])
                data = np.concatenate(data)[:, :cfg.img_sizes[0], :cfg.img_sizes[1]]
                data = np.split(data, len(data) // split_timesteps)
                self.time_chunks = len(data)
                if ensemble == support_ensemble:
                    self.support_data.append(data)
                else:
                    ensemble_data += data

            self.img_data.append(ensemble_data)
            self.img_length = len(ensemble_data)

        self.img_normalizer = DataNormalizer(self.img_data, cfg.normalization)
        self.support_data_normalizer = DataNormalizer(self.support_data, cfg.normalization)

    def load_data(self, ind_data, index, data, normalizer):
        image = data[ind_data][index]
        image = torch.from_numpy(np.nan_to_num(image))
        return normalizer.normalize(image, ind_data)

    def __getitem__(self, index=None, ensemble=None, timechunk=None):
        if ensemble and timechunk:
            ensemble_index = cfg.gt_ensembles.index(ensemble)
            index = ensemble_index * self.time_chunks + timechunk

        support_index = index % self.time_chunks

        images = []
        support_images = []
        for i in range(len(self.data_types)):
            images.append(self.load_data(i, index, self.img_data, self.img_normalizer))
            support_images.append(self.load_data(i, support_index, self.support_data, self.support_data_normalizer))

        if cfg.lstm:
            images = torch.stack(images, dim=1)
            support_images = torch.stack(support_images, dim=1)
        else:
            images = torch.cat(images).unsqueeze(0)
            support_images = torch.cat(support_images).unsqueeze(0)
        return images.squeeze(0), support_images.squeeze(0), {}

    def __len__(self):
        return self.img_length
