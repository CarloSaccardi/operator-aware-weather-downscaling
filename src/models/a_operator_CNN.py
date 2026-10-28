import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import random

from src.utils import vis, constants


class AOperatorCNN(nn.Module):
    """
    Learned bias-correction operator A_theta.
    First area-averages CERRA to ERA5 resolution, then learns a correction
    in low-res space: (B, 7, 77, 77) -> (B, 6, 77, 77).
    Residual architecture: learns a correction on top of area-averaged CERRA.
    """

    def __init__(self, in_channels=7, out_channels=6, target_size=(77, 77), hidden_channels=16):
        super().__init__()
        self.target_size = target_size
        self.out_channels = out_channels

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1),
        )

        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x):
        x_lr = F.interpolate(x, size=self.target_size, mode='area')
        skip = x_lr[:, :self.out_channels]
        correction = self.net(x_lr)
        return self.scale * (skip + correction) + self.bias


class AOperatorWrapper(pl.LightningModule):
    """PyTorch Lightning wrapper for the learned A operator."""

    def __init__(self, args):
        super().__init__()
        self.lr = args.lr
        self.wandb_project = args.wandb_project
        self.img_out_channels = args.img_out_channels

        in_channels = args.img_out_channels + 1  # +1 for orography
        # The grid A_theta maps onto: the ERA5 grid, whatever resolution the
        # coarse product is at. (77, 77) is ERA5 at ~25 km; the ~50 km arm sets
        # a_target_size: [39, 39] and era5_coarsen: 2 in its config.
        target_size = getattr(args, "a_target_size", None) or (77, 77)
        self.model = AOperatorCNN(
            in_channels=in_channels,
            out_channels=args.img_out_channels,
            target_size=tuple(target_size),
        )
        
        # Per-channel loss weights (same as diffusion model)
        if getattr(args, 'channel_weights', None) is not None:
            channel_weights = torch.tensor(args.channel_weights, dtype=torch.float32)
        else:
            channel_weights = torch.ones(args.img_out_channels)
        self.register_buffer(
            "channel_weights", channel_weights.reshape(1, -1, 1, 1)
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=[0.9, 0.999], eps=1e-8)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, *args):
        cerra, cerra_orography, era5 = batch
        # Build input: concatenate CERRA fields with orography
        x = torch.cat([cerra, cerra_orography], dim=1)  
        pred = self(x)
        loss = torch.mean(self.channel_weights * (pred - era5) ** 2) #weight variables. 
        self.log_dict({"train_loss": loss}, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, *args):
        cerra, cerra_orography, era5 = batch
        x = torch.cat([cerra, cerra_orography], dim=1)  
        pred = self(x)
        val_loss = F.mse_loss(pred, era5)

        val_log_dict = {"val_loss": val_loss}
        per_ch = F.mse_loss(pred, era5, reduction="none").mean(dim=(0, 2, 3))
        for i, name in enumerate(constants.PARAM_NAMES_SHORT_CERRA):
            val_log_dict[f"val_loss/{name}"] = per_ch[i]
        self.log_dict(val_log_dict, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        batch_idx = args[0]
        if (
            self.trainer.is_global_zero
            and batch_idx == 0
            and self.current_epoch % 10 == 0
            and self.wandb_project is not None
        ):
            self.load_metrics_and_plots(pred, era5, mask=None)

    def test_step(self, batch, batch_idx):
        cerra, cerra_orography, era5 = batch
        x = torch.cat([cerra, cerra_orography], dim=1)  
        pred = self(x)

        region = getattr(self, "current_region", "unknown")
        rmse = torch.sqrt(F.mse_loss(pred, era5))
        self.log(f"test/{region}/rmse", rmse, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        per_ch_mse = F.mse_loss(pred, era5, reduction="none").mean(dim=(0, 2, 3))
        for var_i, var_name in enumerate(constants.PARAM_NAMES_SHORT_CERRA):
            var_rmse = torch.sqrt(per_ch_mse[var_i])
            self.log(f"test/{region}/{var_name}/rmse", var_rmse, on_step=False, on_epoch=True, sync_dist=True)

        return {f"test/{region}/rmse": rmse.detach()}

    def load_metrics_and_plots(self, prediction, target, mask):
        import matplotlib.pyplot as plt

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
            pred_states, target_state, mask=mask_grid, title=f"{plot_title} (A_operator)",
        )

        if not self.trainer.sanity_checking:
            self.logger.experiment.log(log_plot_dict)

        plt.close("all")


class PooledAOperator(nn.Module):
    """A_theta trained at one coarse resolution, area-averaged onto a coarser one.

    The ~50 km ERA5 field is *defined* as the ~25 km field area-averaged by 2
    (CerraPriorDataset.__getitem__). Pooling P is linear, so the ~50 km target is
    exactly P(y_25), and wrapping the ~25 km operator gives

        y_50 - P(A_25(x))  =  P(y_25 - A_25(x)),

    the ~25 km residual measured on the coarser grid. Nothing is retrained:
    A_theta is reused as-is and only the grid its output lands on changes, which
    makes the coarse-source swap an inference-time change alone.

    Training A_theta on the ~50 km grid directly (a_operator_train_50km.yaml) is
    the alternative. It is not more correct -- it differs only in that its 3x3
    convolutions would act at ~50 km rather than ~25 km spacing.
    """

    def __init__(self, operator, out_size):
        super().__init__()
        self.operator = operator
        self.out_size = tuple(out_size)

    def forward(self, x):
        return F.adaptive_avg_pool2d(self.operator(x), self.out_size)


def load_a_operator(args, map_location=None):
    """Build the frozen A_theta the DPS likelihood uses, or None for bicubic.

    `a_target_size` is the grid the guidance residual y - A(x0) lives on.
    `a_operator_native_size` is the grid the *checkpoint* was trained to emit;
    it defaults to `a_target_size`, i.e. an operator trained for the grid it is
    evaluated on.

    When they differ the operator is built at its native size and its output is
    pooled down. Building it at the target size instead would load without error
    -- target_size is not a parameter, only conv weights are -- and silently give
    a different operator, so the two sizes are kept apart deliberately.
    """
    ckpt = getattr(args, "a_operator_ckpt", None)
    if ckpt is None:
        return None

    target = tuple(getattr(args, "a_target_size", None) or (77, 77))
    native = tuple(getattr(args, "a_operator_native_size", None) or target)

    build_args = copy.copy(args)
    build_args.a_target_size = list(native)

    kwargs = {"map_location": map_location} if map_location is not None else {}
    wrapper = AOperatorWrapper.load_from_checkpoint(ckpt, args=build_args, **kwargs)
    wrapper.eval()
    for p in wrapper.parameters():
        p.requires_grad_(False)

    operator = wrapper.model
    if native != target:
        operator = PooledAOperator(operator, target).eval()
    return operator
