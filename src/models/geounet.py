import torch
import pytorch_lightning as pl
import importlib
import wandb
import matplotlib.pyplot as plt
from src.utils import constants
from src.utils import vis
import random
import os
import torch.nn.functional as F
from argparse import Namespace
from typing import List
from src.utils.utils import compute_psd_2d, compute_derived_fields, compute_crps_empirical, compute_log_spectral_distance
import numpy as np


network_module = importlib.import_module("physicsnemo.models.diffusion")




class GeoUNetWrapper(pl.LightningModule):
    def __init__(self,args):
        super().__init__()
        
        # for compatibility with older versions that took only 1 dimension
        if isinstance(args.img_resolution, int):
            self.img_shape_x = self.img_shape_y = args.img_resolution
        else:
            self.img_shape_y = args.img_resolution[0]
            self.img_shape_x = args.img_resolution[1]

        self.img_in_channels = args.img_in_channels
        self.img_out_channels = args.img_out_channels
        self.lr = args.lr
        self.wandb_project = args.wandb_project
        self._test_preds = []
        self._test_targets = []

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
            in_channels=args.img_out_channels + 1,  # +1 for orography
            out_channels=args.img_out_channels,
            **self.model_kwargs,
        )
        

    def forward(self, x: torch.Tensor, **model_kwargs: dict) -> torch.Tensor:
        # Pad spatial dims up to the next multiple of 32 so U-Net skip shapes align.
        H, W = x.shape[-2:]
        pad_h = (-H) % 32
        pad_w = (-W) % 32
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        D_x = self.model(
            x_padded,
            torch.zeros(x_padded.shape[0], device=x_padded.device),  # noise_labels (unused with embedding_type=zero)
            None,                                                    # class_labels
            **model_kwargs,
        )
        return D_x[..., :H, :W].to(torch.float32)

    def training_step(self, batch, *args):
        cerra, cerra_orography, era5 = batch
        #### Upsample era5 and concatenate with orography ####
        era5 = F.interpolate(era5, size=cerra.shape[-2:], mode='bicubic', align_corners=False)
        era5 = torch.cat([era5, cerra_orography], dim=1)
        ######################################################
            
        D_x = self(era5)
        
        loss = F.mse_loss(D_x, cerra)
        
        log_dict = {
            "train_loss_epoch": loss,
        }
        self.log_dict(
            log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True
        )
        return loss

    def validation_step(self, batch, *args):        
        cerra, cerra_orography, era5 = batch
        #### Upsample era5 and concatenate with orography ####
        era5 = F.interpolate(era5, size=cerra.shape[-2:], mode='bicubic', align_corners=False)
        era5 = torch.cat([era5, cerra_orography], dim=1)
        ######################################################
            
        D_x = self(era5)
        
        val_loss = F.mse_loss(D_x, cerra)
        
        val_log_dict = {
            "val_loss": val_loss,
        }
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
        cerra, cerra_orography, era5 = batch
        #### Upsample era5 and concatenate with orography ####
        era5 = F.interpolate(era5, size=cerra.shape[-2:], mode='bilinear', align_corners=False)
        era5 = torch.cat([era5, cerra_orography], dim=1)
        ######################################################
            
        full_pred = self(era5)
        
        # Log region-grouped images to a WandB table
        self.load_metrics_and_plots_test(full_pred, cerra, era5)

        # 5. Metrics — treat deterministic prediction as single ensemble member
        ensemble_preds = full_pred.unsqueeze(0)  # (1, B, C, H, W)

        # De-standardize to physical units for metrics and PSD
        ens_phys = ensemble_preds * self.data_std.unsqueeze(0) + self.data_mean.unsqueeze(0)
        mean_phys = ens_phys.squeeze(0)
        target_phys = cerra * self.data_std + self.data_mean

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
    
    
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return opt
            
            
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
            wandb.log(log_plot_dict)
        
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

