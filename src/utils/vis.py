# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

# First-party
from . import constants, utils


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_ensemble_prediction(
    prediction, target, mask, title=None, vrange=None
):
    """
    Plot example predictions, ground truth, mean and std.-dev.
    from ensemble forecast

    prediction: (S, N_grid,)
    target: (N_grid,)
    ens_mean: (N_grid,)
    ens_std: (N_grid,)
    obs_mask: (N_grid,)
    (optional) title: title of plot
    (optional) vrange: tuple of length with common min and max of values
        (not for std.)
    """
    
    target_masked = target * (~mask) if mask is not None else target - prediction
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(25, 15),
        #subplot_kw={"projection": constants.LAMBERT_PROJ},
    )
    axes = axes.flatten()

    # Plot target, ensemble mean and std.
    gt_im = plot_on_axis(
        axes[0],
        target_masked,
        vmin=target.min().item(),
        vmax=target.max().item(),
        ax_title="Ground Truth - Masked" if mask is not None else "Ground Truth - Residual",
    )
    plot_on_axis(
        axes[1],
        target,
        vmin=target.min().item(),
        vmax=target.max().item(),
        ax_title="Ground Truth",
    )
    plot_on_axis(
        axes[2],
        prediction,
        vmin=prediction.min().item(),
        vmax=prediction.max().item(),
        ax_title="Prediction",
    )
    # Turn off unused axes
    for ax in axes[(3 + prediction.shape[0]) :]:
        ax.axis("off")

    # Add colorbars
    values_cbar = fig.colorbar(
        gt_im, ax=axes[:3], aspect=60, location="bottom", shrink=0.9
    )
    values_cbar.ax.tick_params(labelsize=10)
    #std_cbar = fig.colorbar(std_im, aspect=30, location="bottom", shrink=0.9)
    #std_cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig


def plot_psd_comparison(k, psd_gt, psd_pred, title="PSD",
                        label_gt="CERRA", label_pred="Prediction"):
    """
    Log-log plot of radially averaged 2D PSD: ground truth vs prediction.

    Parameters
    ----------
    k : np.ndarray
        Wavenumber bin centers.
    psd_gt : np.ndarray
        Ground truth PSD.
    psd_pred : np.ndarray
        Prediction PSD.
    title : str
        Plot title.
    label_gt, label_pred : str
        Curve labels.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    # Skip first bin (k~0) to avoid log(0)
    ax.loglog(k[1:], psd_gt[1:], lw=2, c='k', label=label_gt)
    ax.loglog(k[1:], psd_pred[1:], lw=2, label=label_pred)
    ax.set_xlabel(r"Wavenumber $|\mathbf{k}|$")
    ax.set_ylabel("PSD")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def psd_to_wandb_table(k, psd_gt, psd_pred, label_gt="CERRA", label_pred="Prediction"):
    """
    Create a wandb.Table from PSD data so that W&B can overlay curves across runs.

    Returns
    -------
    wandb.Table
        Table with columns ["wavenumber", "psd", "source"] suitable for a
        W&B line plot (Custom Chart → vega "line" with x=wavenumber, y=psd, color=source).
    """
    # Skip first bin (k~0)
    k_np = np.asarray(k[1:], dtype=np.float64)
    gt_np = np.asarray(psd_gt[1:], dtype=np.float64)
    pred_np = np.asarray(psd_pred[1:], dtype=np.float64)

    table = wandb.Table(columns=["wavenumber", "psd", "source"])
    for ki, gi in zip(k_np, gt_np):
        table.add_data(float(ki), float(gi), label_gt)
    for ki, pi in zip(k_np, pred_np):
        table.add_data(float(ki), float(pi), label_pred)
    return table


def plot_on_axis(ax, data, vmin=None, vmax=None, ax_title=None):
    """
    Plot weather state on given axis
    """
    #ax.coastlines()  # Add coastline outlines
    H = W = int(data.shape[0]**0.5)
    data_grid = data.reshape(H, W).to(torch.float32).cpu().numpy()
    im = ax.imshow(
        data_grid,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap="plasma",
    )

    if ax_title:
        ax.set_title(ax_title, size=15)
    return im





@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_era5_cerra_prediction(
    prediction, cerra, era5, title=None, vrange=None
):
    """
    Plot example predictions, ground truth, mean and std.-dev.
    from ensemble forecast

    prediction: (S, N_grid,)
    cerra: (N_grid,)
    era5: (N_grid,)
    (optional) title: title of plot
    (optional) vrange: tuple of length with common min and max of values
        (not for std.)
    """
    
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(25, 15),
        #subplot_kw={"projection": constants.LAMBERT_PROJ},
    )
    axes = axes.flatten()

    # Plot target, ensemble mean and std.
    gt_im = plot_on_axis(
        axes[0],
        era5,
        vmin=era5.min().item(),
        vmax=era5.max().item(),
        ax_title="ERA5",
    )
    plot_on_axis(
        axes[1],
        cerra,
        vmin=cerra.min().item(),
        vmax=cerra.max().item(),
        ax_title="CERRA",
    )
    plot_on_axis(
        axes[2],
        prediction,
        vmin=prediction.min().item(),
        vmax=prediction.max().item(),
        ax_title="Prediction",
    )
    # Turn off unused axes
    for ax in axes[(3 + prediction.shape[0]) :]:
        ax.axis("off")

    # Add colorbars
    values_cbar = fig.colorbar(
        gt_im, ax=axes[:3], aspect=60, location="bottom", shrink=0.9
    )
    values_cbar.ax.tick_params(labelsize=10)
    #std_cbar = fig.colorbar(std_im, aspect=30, location="bottom", shrink=0.9)
    #std_cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig