# Standard library
import datetime as dt
import os
import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr


def coarse_size(hw, factor):
    """ERA5 grid size after coarsening by `factor`, rounded up.

    A_theta has to emit exactly this shape, since the guidance residual is
    y - A(x0): 77x77 at factor 1, 39x39 (~50 km) at factor 2.
    """
    return tuple(-(-int(s) // int(factor)) for s in hw)


class CerraPriorDataset(torch.utils.data.Dataset):
    """
    Super-Resolution Dataset (Upsampled ERA5 + CERRA Forcing).
    
    Logic:
    1. Loads Low-Res ERA5 (85x85) and Time Embeddings.
    2. Upsamples ERA5+Time to High-Res (384x384) using Bicubic interpolation.
    3. Concatenates with High-Res CERRA Forcing (Orography).
    4. Returns (Full_Input, CERRA_Target).
    """
    def __init__(self, 
                 root_dir_cerra,
                 root_dir_era5,
                 split,
                 region,
                 cerra_vars=['u10', 'v10', 't2m', 'sshf', 'zust', 'sp'],
                 era5_vars=['u10', 'v10', 't2m', 'sshf', 'zust', 'sp'],
                 n_samples=None,
                 crop=False,
                 era5_coarsen=1):
        super().__init__()

        # Factor by which ERA5 is area-averaged after loading. 1 leaves it at its
        # native ~25 km (77x77); 2 gives ~50 km (39x39). This is the knob behind
        # the "different coarse source" experiment: the coarse field changes, so
        # A_theta is retrained for it, and the diffusion prior -- which never sees
        # ERA5 -- is reused unchanged.
        self.era5_coarsen = int(era5_coarsen or 1)

        self.cerra_vars = cerra_vars
        self.era5_vars = era5_vars
        # Paths
        cerra_path = os.path.join(root_dir_cerra, split, f"{region}.nc")
        era5_path = os.path.join(root_dir_era5, split, f"{region}.nc")
        cerra_orography_path = os.path.join(root_dir_cerra, split, f"static_{region}.nc")

        # 1. Load Statistics
        self.era5_mean = np.load(os.path.join(root_dir_era5, "statistics", "dynamic_mean.npy"))
        self.era5_std = np.load(os.path.join(root_dir_era5, "statistics", "dynamic_std.npy"))

        self.era5_orography_mean = np.load(os.path.join(root_dir_era5, "statistics", "forcing_mean.npy"))
        self.era5_orography_std = np.load(os.path.join(root_dir_era5, "statistics", "forcing_std.npy"))

        # 2. Open Datasets (Lazy Xarray)
        self.cerra_ds = xr.open_dataset(cerra_path, engine="h5netcdf")
        self.era5_ds = xr.open_dataset(era5_path, engine="h5netcdf")

        # 3. Load Static Data into RAM (Optimization)
        # We perform the static normalization ONCE here to save CPU cycles in __getitem__
        ds_static_cerra = xr.open_dataset(cerra_orography_path, engine="h5netcdf")
        raw_static_cerra = ds_static_cerra['orog'].values.astype(np.float32)[None, ...]
        self.cerra_orography = torch.from_numpy((raw_static_cerra - self.era5_orography_mean) / self.era5_orography_std)
        ds_static_cerra.close()

        # 4. Temporal subsampling
        total = len(self.cerra_ds.time)
        if n_samples is not None and n_samples < total:
            rng = np.random.default_rng(seed=42)
            self.indices = np.sort(rng.choice(total, size=n_samples, replace=False))
        else:
            self.indices = np.arange(total)

    def __len__(self):
        return len(self.indices)


    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])
        # 1. Load Dynamic Data
        cerra = torch.from_numpy(self._load_dynamic_step(self.cerra_ds, self.cerra_vars, real_idx))
        era5 = torch.from_numpy(self._load_dynamic_step(self.era5_ds, self.era5_vars, real_idx))
        # 2. Normalize Dynamic Data
        cerra = (cerra - self.era5_mean[:, None, None]) / self.era5_std[:, None, None]
        era5 = (era5 - self.era5_mean[:, None, None]) / self.era5_std[:, None, None]
        # 3. Coarsen ERA5, if asked. adaptive_avg_pool2d rather than avg_pool2d:
        # 77 is not divisible by 2, and this area-averages over uneven bins
        # instead of dropping the remainder.
        if self.era5_coarsen > 1:
            era5 = F.adaptive_avg_pool2d(era5, coarse_size(era5.shape[-2:],
                                                           self.era5_coarsen))

        return cerra, self.cerra_orography, era5

    
    def _load_dynamic_step(self, dataset, variables, idx):
        """Lazy load specific time step."""
        data_sel = dataset[variables].isel(time=idx)
        return data_sel.to_array().values.astype(np.float32)
    
    def close(self):
        if self.cerra_ds: self.cerra_ds.close()