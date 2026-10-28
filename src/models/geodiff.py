import importlib
import os

import nvtx
import torch
import pytorch_lightning as pl
from src.utils import vis
from src.utils import constants
import random

import matplotlib.pyplot as plt

import torch.nn.functional as F

from physicsnemo.models.diffusion import EDMPrecond
from typing import Union, List

from src.utils.utils import compute_psd_2d, compute_derived_fields, compute_crps_empirical, compute_log_spectral_distance
from src.utils.sampling import diffusion_step, stochastic_sampler_DPS
import wandb
import numpy as np




network_module = importlib.import_module("physicsnemo.models.diffusion")



class GeoDiffWrapper(pl.LightningModule):
    """
    Improved preconditioning proposed in the paper "Elucidating the Design Space of
    Diffusion-Based Generative Models" (EDM).

    This is a variant of `EDMPrecond` that is specifically designed for prior learning.
    It wraps a neural network that predicts the denoised high-resolution image
    given a noisy high-resolution image, and additional conditioning that includes a
    orography.

    Parameters
    ----------
    model_type : str, optional
        Class name of the underlying model. Must be one of the following:
        'SongUNet', 'SongUNetPosEmbd', 'SongUNetPosLtEmbd', 'DhariwalUNet'.
        Defaults to 'SongUNetPosEmbd'.
    sigma_data : float, optional
        Expected standard deviation of the training data, by default 0.5.
    sigma_min : float, optional
        Minimum supported noise level, by default 0.0.
    sigma_max : float, optional
        Maximum supported noise level, by default inf.
    **model_kwargs : dict
        Keyword arguments passed to the underlying model `__init__` method.

    See Also
    --------
    For information on model types and their usage:
    :class:`~physicsnemo.models.diffusion.SongUNet`: Basic U-Net for diffusion models
    :class:`~physicsnemo.models.diffusion.SongUNetPosEmbd`: U-Net with positional embeddings
    :class:`~physicsnemo.models.diffusion.SongUNetPosLtEmbd`: U-Net with positional and lead-time embeddings

    Please refer to the documentation of these classes for details on how to call
    and use these models directly.

    Note
    ----
    References:
    - Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    - Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(self, args):
        super().__init__()
        self.P_mean = 0.0 # mean value for noise level computation
        self.P_std = 1.2 # standard deviation for noise level computation
        self.sigma_data = 1.0 # standard deviation for data weighting
        self.sigma_min = 0.002  # minimum noise level
        self.sigma_max = 800  # maximum noise level
        self.wandb_project = args.wandb_project
        self.img_out_channels = args.img_out_channels
        
        self.lr = args.lr
        
        self.model_kwargs = {
            'checkpoint_level': args.checkpoint_level,
            # 'gridtype': args.gridtype,
            # 'N_grid_channels': args.N_grid_channels,
            'embedding_type': args.embedding_type,
            'model_channels': args.model_channels,
            'channel_mult': args.channel_mult,
            'attn_resolutions': args.attn_resolutions,
        }

        model_class = getattr(network_module, args.model_type)
        self.model = model_class(
            img_resolution=args.img_resolution,
            in_channels=args.img_out_channels + 1, # the +1 is for the orography
            out_channels=args.img_out_channels,
            **self.model_kwargs,
        )

        self.num_ensembles = getattr(args, 'num_ensembles', 1)
        self.zeta_lr = getattr(args, 'zeta_lr', 100.0)
        # "channel": unit-norm gradient per (sample, variable); "global": DPS,
        # the raw gradient. The ablation of the paper turns on this switch alone.
        self.grad_norm = getattr(args, 'grad_norm', 'channel')

        self._test_preds = []
        self._test_targets = []

        self.a_operator = None

        self.scaling_fn = self._scaling_fn

        # Per-channel loss weights: upweight smooth fields (t2m, sp)
        # that contribute less gradient signal to the denoising loss.
        # Order: [u10, v10, t2m, sshf, zust, sp]
        if getattr(args, 'channel_weights', None) is not None:
            channel_weights = torch.tensor(args.channel_weights, dtype=torch.float32)
        else:
            channel_weights = torch.ones(args.img_out_channels)
        self.register_buffer(
            "channel_weights", channel_weights.reshape(1, -1, 1, 1)
        )

        # De-standardization buffers for physical-unit metrics
        stats_dir = os.path.join(args.dataset_era5, "statistics")
        data_mean = torch.from_numpy(
            np.load(os.path.join(stats_dir, "dynamic_mean.npy"))
        ).float().reshape(1, -1, 1, 1)
        data_std = torch.from_numpy(
            np.load(os.path.join(stats_dir, "dynamic_std.npy"))
        ).float().reshape(1, -1, 1, 1)
        self.register_buffer("data_mean", data_mean, persistent=False)
        self.register_buffer("data_std", data_std, persistent=False)


    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=[0.9, 0.999], eps=1e-8
            )
        return opt

    @staticmethod
    def _scaling_fn(
        x: torch.Tensor, orography: torch.Tensor, c_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Scale input tensors by first scaling the high-resolution tensor and then
        concatenating with the orography tensor.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image of shape (B, C_hr, H, W).
        orography : torch.Tensor
            Low-resolution image of shape (B, 1, H, W).
        c_in : torch.Tensor
            Scaling factor of shape (B, 1, 1, 1).

        Returns
        -------
        torch.Tensor
            Scaled and concatenated tensor of shape (B, C_in+C_out, H, W).
        """
        return torch.cat([c_in * x, orography.to(x.dtype)], dim=1)

    @nvtx.annotate(message="EDMPrecondSuperResolution", color="orange")
    def forward(
        self,
        x: torch.Tensor,
        orography: torch.Tensor,
        sigma: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        """
        Forward pass of the EDMPrecondSuperResolution model wrapper.

        This method applies the EDM preconditioning to compute the denoised image
        from a noisy high-resolution image and orography.

        Parameters
        ----------
        x : torch.Tensor
            Noisy high-resolution image of shape (B, C_hr, H, W). The number of
            channels `C_hr` should be equal to `img_out_channels`.
        orography : torch.Tensor
            Orography of shape (B, 1, H, W).
        sigma : torch.Tensor
            Noise level of shape (B) or (B, 1) or (B, 1, 1, 1).
        force_fp32 : bool, optional
            Whether to force FP32 precision regardless of the `use_fp16` attribute,
            by default False.
        **model_kwargs : dict
            Additional keyword arguments to pass to the underlying model
            `self.model` forward method.

        Returns
        -------
        torch.Tensor
            Denoised high-resolution image of shape (B, C_hr, H, W).

        Raises
        ------
        ValueError
            If the model output dtype doesn't match the expected dtype.
        """
        # Concatenate input channels
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2) 
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        # if img_lr is None:
        #     arg = c_in * x
        # else:
        arg = self.scaling_fn(x, orography, c_in)

        # Pad spatial dims up to the next multiple of 32 so U-Net skip shapes align.
        H, W = arg.shape[-2:]
        pad_h = (-H) % 32
        pad_w = (-W) % 32
        arg = F.pad(arg, (0, pad_w, 0, pad_h), mode="reflect")

        F_x = self.model(
            arg,
            c_noise.flatten(),
            class_labels=None,
            **model_kwargs,
        )
        F_x = F_x[..., :H, :W]

        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x
    
    def training_step(self, batch, *args):
        cerra, cerra_orography, _ = batch
        
        rnd_normal = torch.randn([cerra.shape[0], 1, 1, 1], device=cerra.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        latent = cerra + torch.randn_like(cerra) * sigma
        
        D_x = self(
            latent,
            cerra_orography,
            sigma,
        )
        
        train_loss = torch.mean(weight * self.channel_weights * (D_x - cerra) ** 2)
        
        log_dict = {
            "train_loss": train_loss,
        }
        self.log_dict(
            log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        return train_loss

    def validation_step(self, batch, *args):
        cerra, cerra_orography, _ = batch
        
        rnd_normal = torch.randn([cerra.shape[0], 1, 1, 1], device=cerra.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        latent = cerra + torch.randn_like(cerra) * sigma
        
        D_x = self(
            latent,
            cerra_orography,
            sigma,
        )
        
        val_loss = torch.mean(weight * (D_x - cerra) ** 2)

        # Log aggregate and per-channel validation loss
        val_log_dict = {
            "val_loss": val_loss
        }
        per_ch = torch.mean(weight * (D_x - cerra) ** 2, dim=(0, 2, 3))
        for i, name in enumerate(constants.PARAM_NAMES_SHORT_CERRA):
            val_log_dict[f"val_loss/{name}"] = per_ch[i]
        self.log_dict(
            val_log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        
        
        batch_idx = args[0]
        
        # Plot some example predictions using prior and encoder
        if (
            self.trainer.is_global_zero
            and batch_idx == 0
            and self.current_epoch % 10 == 0
            and self.wandb_project is not None
        ):
            self.load_metrics_and_plots(D_x, cerra, mask=None)
            
            
    def test_step(self, batch, batch_idx):
        cerra_target, orography, era5 = batch

        batch_size = era5.shape[0]
        n_ens = self.num_ensembles

        # Create n_ens seed batches, each with batch_size seeds
        rank_batches = [
            [batch_idx * n_ens * batch_size + j * batch_size + i
             for i in range(batch_size)]
            for j in range(n_ens)
        ]

        full_pred = diffusion_step(
            net=self,
            sampler_fn=stochastic_sampler_DPS,
            img_shape=cerra_target.shape[-2:],
            img_out_channels=self.img_out_channels,
            rank_batches=rank_batches,
            img_lr=era5,
            conditions=orography,
            rank=self.global_rank,
            device=self.device,
            zeta_lr=self.zeta_lr,
            a_operator=self.a_operator,
            channel_weights=self.channel_weights,
            grad_norm=self.grad_norm,
        )

        # Reshape (n_ens * B, C, H, W) → (n_ens, B, C, H, W), take ensemble mean
        ensemble_preds = full_pred.view(n_ens, batch_size, *full_pred.shape[1:])
        ensemble_mean = ensemble_preds.mean(dim=0)  # (B, C, H, W)

        self.load_metrics_and_plots_test(ensemble_mean, cerra_target, era5)

        # De-standardize to physical units for metrics and PSD
        ens_phys = ensemble_preds * self.data_std.unsqueeze(0) + self.data_mean.unsqueeze(0)
        mean_phys = ens_phys.mean(dim=0)
        target_phys = cerra_target * self.data_std + self.data_mean

        # Total metrics
        region = getattr(self, "current_region", "unknown")
        rmse = torch.sqrt(F.mse_loss(mean_phys, target_phys))
        crps = compute_crps_empirical(ens_phys, target_phys)
        rmse_shift = self._shift_tolerant_rmse(mean_phys, target_phys)
        self.log(f"test/{region}/rmse", rmse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"test/{region}/crps", crps, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"test/{region}/rmse_shift", rmse_shift,
                    on_step=False, on_epoch=True, sync_dist=True)

        # Per-variable metrics
        for var_i, var_name in enumerate(constants.PARAM_NAMES_SHORT_CERRA):
            var_rmse = torch.sqrt(F.mse_loss(mean_phys[:, var_i], target_phys[:, var_i]))
            var_crps = compute_crps_empirical(ens_phys[:, :, var_i], target_phys[:, var_i])
            var_rmse_shift = self._shift_tolerant_rmse(mean_phys[:, var_i], target_phys[:, var_i])
            self.log(f"test/{region}/{var_name}/rmse", var_rmse,
                        on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"test/{region}/{var_name}/crps", var_crps,
                        on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"test/{region}/{var_name}/rmse_shift", var_rmse_shift,
                        on_step=False, on_epoch=True, sync_dist=True)

        # Accumulate de-standardized data for PSD at epoch end
        self._test_preds.append(ens_phys.detach().cpu())
        self._test_targets.append(target_phys.detach().cpu())

        return {f"test/{region}/rmse": rmse.detach(), f"test/{region}/crps": crps.detach()}
    
    
    def _log_psd_scalars(self, k, psd_values, region, var_name, source_label):
        """Log PSD as scalars in log10 space with wavenumber as custom x-axis."""
        k_np = np.asarray(k[1:], dtype=np.float64)
        psd_np = np.asarray(psd_values[1:], dtype=np.float64)

        metric_key = f"psd/{region}/{var_name}/{source_label}"
        x_key = f"psd/{region}/{var_name}/log10_wavenumber"
        wandb.define_metric(x_key, hidden=True)
        wandb.define_metric(metric_key, step_metric=x_key)

        log_k = np.log10(k_np)
        log_psd = np.log10(psd_np)

        for ki, pi in zip(log_k, log_psd):
            wandb.log({x_key: float(ki), metric_key: float(pi)})

    def on_test_epoch_end(self):
        region = getattr(self, "current_region", "unknown")

        if self.trainer.is_global_zero and len(self._test_preds) > 0:
            all_preds = torch.cat(self._test_preds, dim=1)    # (n_ens, N_total, C, H, W)
            all_targets = torch.cat(self._test_targets, dim=0) # (N_total, C, H, W)
            n_ens = all_preds.shape[0]

            var_names = constants.PARAM_NAMES_SHORT_CERRA
            run_name = wandb.run.name if wandb.run else "prediction"

            if self.wandb_project is not None and not self.trainer.sanity_checking:
                # PSD for raw variables — mean of individual member PSDs
                for var_i, var_name in enumerate(var_names):
                    psd_members = []
                    for m in range(n_ens):
                        k_pred, psd_m = compute_psd_2d(all_preds[m, :, var_i, :, :])
                        psd_members.append(psd_m)
                    psd_pred = np.mean(psd_members, axis=0)
                    k_tgt, psd_tgt = compute_psd_2d(all_targets[:, var_i, :, :])

                    self._log_psd_scalars(k_tgt, psd_tgt, region, var_name, "CERRA")
                    self._log_psd_scalars(k_pred, psd_pred, region, var_name, run_name)

                    lsd = compute_log_spectral_distance(psd_pred, psd_tgt)
                    # sync_dist=False: this branch only runs on rank 0 (is_global_zero guard above),
                    # so an all_reduce here would hang waiting for other ranks that never call it.
                    self.log(f"test/{region}/{var_name}/lsd", lsd, on_step=False, on_epoch=True, sync_dist=False)

                # PSD for derived quantities — mean of individual member PSDs
                for derived_name in constants.DERIVED_VAR_NAMES:
                    psd_members = []
                    for m in range(n_ens):
                        pred_derived_m = compute_derived_fields(all_preds[m], var_names)
                        k_pred, psd_m = compute_psd_2d(pred_derived_m[derived_name])
                        psd_members.append(psd_m)
                    psd_pred = np.mean(psd_members, axis=0)

                    tgt_derived = compute_derived_fields(all_targets, var_names)
                    k_tgt, psd_tgt = compute_psd_2d(tgt_derived[derived_name])

                    self._log_psd_scalars(k_tgt, psd_tgt, region, derived_name, "CERRA")
                    self._log_psd_scalars(k_pred, psd_pred, region, derived_name, run_name)

                    lsd = compute_log_spectral_distance(psd_pred, psd_tgt)
                    self.log(f"test/{region}/{derived_name}/lsd", lsd, on_step=False, on_epoch=True, sync_dist=False)

        # Clear accumulators for next region
        self._test_preds.clear()
        self._test_targets.clear()

    @staticmethod
    def round_sigma(sigma: Union[float, List, torch.Tensor]) -> torch.Tensor:
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float, List, torch.Tensor]
            Sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            Tensor representation of sigma values.

        See Also
        --------
        EDMPrecond.round_sigma
        """
        return EDMPrecond.round_sigma(sigma)
            
            
            
    @staticmethod
    def _shift_tolerant_rmse(pred, target, max_shift=2):
        """Per-sample minimum MSE over a small translation window of pred,
        then sqrt of the batch mean. Crops to the interior so no boundary
        artifacts. pred/target share shape; last two dims are spatial."""
        H, W = pred.shape[-2:]
        h, w = H - 2 * max_shift, W - 2 * max_shift
        tgt = target[..., max_shift:max_shift + h, max_shift:max_shift + w]
        reduce_dims = tuple(range(1, tgt.ndim))
        mses = []
        for dy in range(-max_shift, max_shift + 1):
            for dx in range(-max_shift, max_shift + 1):
                y0 = max_shift + dy
                x0 = max_shift + dx
                p = pred[..., y0:y0 + h, x0:x0 + w]
                mses.append(((p - tgt) ** 2).mean(dim=reduce_dims))
        best = torch.stack(mses, dim=0).min(dim=0).values
        return torch.sqrt(best.mean())

    def load_metrics_and_plots(self, prediction, target, mask):
        # Reshape from (B, C, H, W) to (B, num_grid_nodes, C)
        prediction = prediction.permute(0, 2, 3, 1).flatten(1, 2)
        target = target.permute(0, 2, 3, 1).flatten(1, 2)
        
        # Plot samples
        log_plot_dict = {}
        
        var_i = random.randint(0, len(constants.PARAM_NAMES_SHORT_CERRA) - 1)
        var_name = constants.PARAM_NAMES_SHORT_CERRA[var_i]
        var_unit = constants.PARAM_UNITS_CERRA[var_i]
        
        sample = random.randint(0, prediction.shape[0] - 1)
        
        #select one mask grid
        mask_grid = mask[sample, :, :].flatten(1,2).squeeze(0) if mask is not None else None
        
        
        pred_states = prediction[sample, :, var_i]
        target_state = target[sample, :, var_i]
        
        plot_title = f"{var_name} ({var_unit})"
        
        # Make plots
        log_plot_dict[f"pred_{var_name}"] = vis.plot_ensemble_prediction(
            pred_states,
            target_state,
            mask=mask_grid,
            title=f"{plot_title} (prior)",
        )
        
        if not self.trainer.sanity_checking:
            self.logger.experiment.log(log_plot_dict)
        
        plt.close("all") 


    def load_metrics_and_plots_test(self, prediction, cerra, era5):
        # Reshape from (B, C, H, W) to (B, num_grid_nodes, C)
        prediction = prediction.permute(0, 2, 3, 1).flatten(1, 2)
        cerra = cerra.permute(0, 2, 3, 1).flatten(1, 2)
        era5 = era5.permute(0, 2, 3, 1).flatten(1, 2)
        
        # Plot samples
        log_plot_dict = {}
        
        var_i = random.randint(0, len(constants.PARAM_NAMES_SHORT_CERRA) - 1)
        var_name = constants.PARAM_NAMES_SHORT_CERRA[var_i]
        var_unit = constants.PARAM_UNITS_CERRA[var_i]
        
        sample = random.randint(0, prediction.shape[0] - 1)        
        
        pred_states = prediction[sample, :, var_i]
        cerra_state = cerra[sample, :, var_i]
        era5_state = era5[sample, :, var_i]
        
        plot_title = f"{var_name} ({var_unit})"
        
        # Make plots
        log_plot_dict[f"pred_{var_name}"] = vis.plot_era5_cerra_prediction(
            pred_states,
            cerra_state,
            era5_state,
            title=f"{plot_title} (prior)",
        )

        if not self.trainer.sanity_checking:
            self.logger.experiment.log(log_plot_dict)

        plt.close("all")
