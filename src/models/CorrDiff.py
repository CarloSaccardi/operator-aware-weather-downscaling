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
from src.utils.sampling import diffusion_step, stochastic_sampler_CondDiff
from src.models.geounet import GeoUNetWrapper
import wandb
import numpy as np

network_module = importlib.import_module("physicsnemo.models.diffusion")



class CorrDiffWrapper(pl.LightningModule):
    """
    CorrDiff: residual-corrective diffusion for downscaling.

    Two-stage model following Mardani et al. (2023):
      1. A frozen deterministic UNet (`GeoUNetWrapper`) predicts the
         conditional mean of the high-resolution field given ERA5.
      2. A diffusion model is trained to denoise the *residual*
         (cerra - mean_pred). At inference, sampled residuals are added
         back to the deterministic mean to recover the full prediction.

    The diffusion network itself uses the same EDM preconditioning as
    `DiffusionWrapper` and is conditioned on the (orography-augmented) ERA5
    input through channel concatenation.
    """

    def __init__(self, args):
        super().__init__()
        self.P_mean = 0.0
        self.P_std = 1.2
        self.sigma_data = 1.0
        self.sigma_min = 0.002
        self.sigma_max = 800
        self.wandb_project = args.wandb_project
        self.img_out_channels = args.img_out_channels

        self.lr = args.lr

        self.model_kwargs = {
            'checkpoint_level': args.checkpoint_level,
            'embedding_type': args.embedding_type,
            'model_channels': args.model_channels,
            'channel_mult': args.channel_mult,
            'attn_resolutions': args.attn_resolutions,
        }

        model_class = getattr(network_module, args.model_type)
        self.model = model_class(
            img_resolution=args.img_resolution,
            in_channels=args.img_out_channels + args.img_out_channels + 1,  # noisy residual + era5 (dyn) + orography
            out_channels=args.img_out_channels,
            **self.model_kwargs,
        )

        self.num_ensembles = getattr(args, 'num_ensembles', 1)

        self._test_preds = []
        self._test_targets = []

        self.scaling_fn = self._scaling_fn

        # Per-channel loss weights
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

        # ----- Frozen deterministic backbone (GeoUNet) -----
        assert getattr(args, 'geounet_checkpoint', None) is not None, \
            "CorrDiff requires `geounet_checkpoint` to point to a trained GeoUNet checkpoint."
        self.geounet = GeoUNetWrapper.load_from_checkpoint(
            args.geounet_checkpoint, args=args
        )
        for p in self.geounet.parameters():
            p.requires_grad = False
        self.geounet.eval()

    def train(self, mode: bool = True):
        # Keep the frozen backbone in eval mode regardless of parent .train() calls.
        super().train(mode)
        self.geounet.eval()
        return self

    @torch.no_grad()
    def _mean_prediction(self, era5: torch.Tensor) -> torch.Tensor:
        """Run the frozen deterministic UNet to get the conditional mean."""
        return self.geounet(era5).detach()

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=[0.9, 0.999], eps=1e-8
        )
        return opt

    @staticmethod
    def _scaling_fn(
        x: torch.Tensor, era5: torch.Tensor, c_in: torch.Tensor
    ) -> torch.Tensor:
        """
        Concatenate the (scaled) noisy residual with the ERA5 conditioning
        (which already contains orography as its last channel).
        """
        return torch.cat([c_in * x, era5.to(x.dtype)], dim=1)

    @nvtx.annotate(message="CorrDiff", color="orange")
    def forward(
        self,
        x: torch.Tensor,
        era5: torch.Tensor,
        sigma: torch.Tensor,
        force_fp32: bool = False,
        **model_kwargs: dict,
    ) -> torch.Tensor:
        """
        EDM-preconditioned forward pass. `x` here is a noisy *residual*
        (cerra - mean_pred + noise), not a noisy CERRA field.
        """
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        arg = self.scaling_fn(x, era5, c_in)

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
        cerra, cerra_orography, era5 = batch
        #### Upsample era5 and concatenate with orography ####
        era5 = F.interpolate(era5, size=cerra.shape[-2:], mode='bicubic', align_corners=False)
        era5 = torch.cat([era5, cerra_orography], dim=1)
        ######################################################
        # Deterministic mean prediction (no grad, frozen backbone)
        mean_pred = self._mean_prediction(era5)
        residual = cerra - mean_pred

        rnd_normal = torch.randn([residual.shape[0], 1, 1, 1], device=residual.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        latent = residual + torch.randn_like(residual) * sigma

        D_x = self(
            latent,
            era5,
            sigma,
        )

        train_loss = torch.mean(weight * self.channel_weights * (D_x - residual) ** 2)

        log_dict = {
            "train_loss": train_loss,
        }
        self.log_dict(
            log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        return train_loss

    def validation_step(self, batch, *args):
        cerra, cerra_orography, era5 = batch
        #### Upsample era5 and concatenate with orography ####
        era5 = F.interpolate(era5, size=cerra.shape[-2:], mode='bicubic', align_corners=False)
        era5 = torch.cat([era5, cerra_orography], dim=1)
        ######################################################
        mean_pred = self._mean_prediction(era5)
        residual = cerra - mean_pred

        rnd_normal = torch.randn([residual.shape[0], 1, 1, 1], device=residual.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        latent = residual + torch.randn_like(residual) * sigma

        D_x = self(
            latent,
            era5,
            sigma,
        )

        val_loss = torch.mean(weight * (D_x - residual) ** 2)

        val_log_dict = {
            "val_loss": val_loss
        }
        per_ch = torch.mean(weight * (D_x - residual) ** 2, dim=(0, 2, 3))
        for i, name in enumerate(constants.PARAM_NAMES_SHORT_CERRA):
            val_log_dict[f"val_loss/{name}"] = per_ch[i]
        self.log_dict(
            val_log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )

        batch_idx = args[0]

        if (
            self.trainer.is_global_zero
            and batch_idx == 0
            and self.current_epoch % 10 == 0
            and self.wandb_project is not None
        ):
            # Reconstruct full-field prediction for plotting:
            # at validation we don't sample — we just visualize the
            # denoised residual added back to the deterministic mean.
            full_pred = mean_pred + D_x
            self.load_metrics_and_plots(full_pred, cerra, mask=None)

    def test_step(self, batch, batch_idx):
        cerra, cerra_orography, era5 = batch
        #### Upsample era5 and concatenate with orography ####
        era5 = F.interpolate(era5, size=cerra.shape[-2:], mode='bicubic', align_corners=False)
        era5 = torch.cat([era5, cerra_orography], dim=1)
        ######################################################
        batch_size = era5.shape[0]
        n_ens = self.num_ensembles

        # Deterministic mean prediction (single forward pass, shared by all members)
        mean_pred = self._mean_prediction(era5)

        rank_batches = [
            [batch_idx * n_ens * batch_size + j * batch_size + i
             for i in range(batch_size)]
            for j in range(n_ens)
        ]

        # Sample residuals from the diffusion model.
        # Note: `diffusion_step` calls `self(...)` (i.e. our forward), which
        # produces denoised residuals — not denoised CERRA. We add the
        # deterministic mean back below.
        sampled_residuals = diffusion_step(
            net=self,
            sampler_fn=stochastic_sampler_CondDiff,
            img_shape=cerra.shape[-2:],
            img_out_channels=self.img_out_channels,
            rank_batches=rank_batches,
            img_lr=era5,
            conditions=None,
            rank=self.global_rank,
            device=self.device,
        )

        # (n_ens * B, C, H, W) -> (n_ens, B, C, H, W)
        sampled_residuals = sampled_residuals.view(
            n_ens, batch_size, *sampled_residuals.shape[1:]
        )

        # Add deterministic mean back to each member to get full predictions
        ensemble_preds = sampled_residuals + mean_pred.unsqueeze(0)
        ensemble_mean = ensemble_preds.mean(dim=0)

        self.load_metrics_and_plots_test(ensemble_mean, cerra, era5)

        # De-standardize to physical units for metrics and PSD
        ens_phys = ensemble_preds * self.data_std.unsqueeze(0) + self.data_mean.unsqueeze(0)
        mean_phys = ens_phys.mean(dim=0)
        target_phys = cerra * self.data_std + self.data_mean

        region = getattr(self, "current_region", "unknown")
        rmse = torch.sqrt(F.mse_loss(mean_phys, target_phys))
        crps = compute_crps_empirical(ens_phys, target_phys)
        rmse_shift = self._shift_tolerant_rmse(mean_phys, target_phys)
        self.log(f"test/{region}/rmse", rmse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"test/{region}/crps", crps, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"test/{region}/rmse_shift", rmse_shift,
                 on_step=False, on_epoch=True, sync_dist=True)

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
            all_preds = torch.cat(self._test_preds, dim=1)
            all_targets = torch.cat(self._test_targets, dim=0)
            n_ens = all_preds.shape[0]

            var_names = constants.PARAM_NAMES_SHORT_CERRA
            run_name = wandb.run.name if wandb.run else "prediction"

            if self.wandb_project is not None and not self.trainer.sanity_checking:
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

        self._test_preds.clear()
        self._test_targets.clear()

    @staticmethod
    def round_sigma(sigma: Union[float, List, torch.Tensor]) -> torch.Tensor:
        return EDMPrecond.round_sigma(sigma)

    @staticmethod
    def _shift_tolerant_rmse(pred, target, max_shift=2):
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
        prediction = prediction.permute(0, 2, 3, 1).flatten(1, 2)
        target = target.permute(0, 2, 3, 1).flatten(1, 2)

        log_plot_dict = {}

        var_i = random.randint(0, len(constants.PARAM_NAMES_SHORT_CERRA) - 1)
        var_name = constants.PARAM_NAMES_SHORT_CERRA[var_i]
        var_unit = constants.PARAM_UNITS_CERRA[var_i]

        sample = random.randint(0, prediction.shape[0] - 1)

        mask_grid = mask[sample, :, :].flatten(1, 2).squeeze(0) if mask is not None else None

        pred_states = prediction[sample, :, var_i]
        target_state = target[sample, :, var_i]

        plot_title = f"{var_name} ({var_unit})"

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
        prediction = prediction.permute(0, 2, 3, 1).flatten(1, 2)
        cerra = cerra.permute(0, 2, 3, 1).flatten(1, 2)
        era5 = era5.permute(0, 2, 3, 1).flatten(1, 2)

        log_plot_dict = {}

        var_i = random.randint(0, len(constants.PARAM_NAMES_SHORT_CERRA) - 1)
        var_name = constants.PARAM_NAMES_SHORT_CERRA[var_i]
        var_unit = constants.PARAM_UNITS_CERRA[var_i]

        sample = random.randint(0, prediction.shape[0] - 1)

        pred_states = prediction[sample, :, var_i]
        cerra_state = cerra[sample, :, var_i]
        era5_state = era5[sample, :, var_i]

        plot_title = f"{var_name} ({var_unit})"

        log_plot_dict[f"pred_{var_name}"] = vis.plot_era5_cerra_prediction(
            pred_states,
            cerra_state,
            era5_state,
            title=f"{plot_title} (prior)",
        )

        if not self.trainer.sanity_checking:
            self.logger.experiment.log(log_plot_dict)

        plt.close("all")
