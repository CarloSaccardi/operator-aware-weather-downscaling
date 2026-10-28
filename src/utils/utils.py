from tueplots import bundles, figsizes
from . import constants
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Callable
from physicsnemo.utils.patching import GridPatching2D
from physicsnemo.utils.generative import StackedRandomGenerator
import tqdm
import numpy as np
import nvtx




def fractional_plot_bundle(fraction):
    """
    Get the tueplots bundle, but with figure width as a fraction of
    the page width.
    """
    bundle = bundles.neurips2023(usetex=False, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (
        original_figsize[0] / fraction,
        original_figsize[1],
    )
    return bundle

def init_wandb_metrics(wandb_logger):
    """
    Set up wandb metrics to track
    """
    experiment = wandb_logger.experiment
    experiment.define_metric("val_mean_loss", summary="min")
    # for step in constants.VAL_STEP_LOG_ERRORS:
    #     experiment.define_metric(f"val_loss_unroll{step}", summary="min")



def fractional_plot_bundle(fraction):
    """
    Get the tueplots bundle, but with figure width as a fraction of
    the page width.
    """
    bundle = bundles.neurips2023(usetex=False, family="serif")
    bundle.update(figsizes.neurips2023())
    original_figsize = bundle["figure.figsize"]
    bundle["figure.figsize"] = (
        original_figsize[0] / fraction,
        original_figsize[1],
    )
    return bundle


def compute_crps_empirical(ensemble_preds: Tensor, target: Tensor) -> Tensor:
    """
    Compute empirical CRPS using the sort-based formula.

    CRPS = E|X - y| - 0.5 * E|X - X'|

    Parameters
    ----------
    ensemble_preds : Tensor
        Shape (M, ...) where M is the ensemble size.
    target : Tensor
        Shape (...) matching ensemble_preds[0].

    Returns
    -------
    Tensor
        Scalar CRPS averaged over all dimensions.
    """
    M = ensemble_preds.shape[0]

    # Term 1: mean absolute error to observation
    mae_term = torch.abs(ensemble_preds - target.unsqueeze(0)).mean(dim=0)

    # Term 2: ensemble spread via sort-based formula
    if M > 1:
        sorted_ens, _ = torch.sort(ensemble_preds, dim=0)
        k = torch.arange(M, device=sorted_ens.device, dtype=sorted_ens.dtype)
        weights = (2 * k + 1 - M) / (M * M)
        shape = [M] + [1] * (sorted_ens.dim() - 1)
        spread_term = (weights.reshape(shape) * sorted_ens).sum(dim=0)
    else:
        spread_term = torch.zeros_like(mae_term)

    return (mae_term - spread_term).mean()


def compute_psd_2d(data: torch.Tensor, dx: float = 1.0):
    """
    Compute radially averaged 2D Power Spectral Density.

    Parameters
    ----------
    data : torch.Tensor
        Shape (N, H, W) — N samples.
    dx : float
        Grid spacing (same in both directions).

    Returns
    -------
    k_centers : np.ndarray
        Wavenumber bin centers.
    psd_radial : np.ndarray
        Radially averaged PSD.
    """
    N, H, W = data.shape

    # 2D FFT, normalize by grid size
    fft_2d = torch.fft.fft2(data) / (H * W)
    psd_2d = torch.abs(fft_2d) ** 2
    psd_2d = psd_2d.mean(dim=0)  # average over samples → (H, W)

    # Wavenumber grids
    kx = torch.fft.fftfreq(W, d=dx)
    ky = torch.fft.fftfreq(H, d=dx)
    KY, KX = torch.meshgrid(ky, kx, indexing='ij')
    K = torch.sqrt(KX ** 2 + KY ** 2)

    # Radial binning
    k_max = min(kx.abs().max(), ky.abs().max())
    n_bins = min(H, W) // 2
    k_bin_edges = torch.linspace(0, k_max, n_bins + 1)
    k_centers = 0.5 * (k_bin_edges[:-1] + k_bin_edges[1:])

    psd_radial = torch.zeros(n_bins)
    for i in range(n_bins):
        mask = (K >= k_bin_edges[i]) & (K < k_bin_edges[i + 1])
        if mask.any():
            psd_radial[i] = psd_2d[mask].mean()

    return k_centers.numpy(), psd_radial.numpy()


def compute_log_spectral_distance(psd_pred: np.ndarray, psd_target: np.ndarray) -> float:
    """
    Log-spectral distance (LSD) between two radially averaged PSDs.

    LSD = sqrt(mean((log10(psd_pred) - log10(psd_target)) ** 2))

    The k=0 bin is excluded, matching the convention used elsewhere when
    logging PSD in log10 space (see `_log_psd_scalars` in the model classes).

    Parameters
    ----------
    psd_pred : np.ndarray
        Radially averaged PSD of the prediction, as returned by compute_psd_2d.
    psd_target : np.ndarray
        Radially averaged PSD of the target, as returned by compute_psd_2d.

    Returns
    -------
    float
        Scalar LSD value.
    """
    psd_pred = np.asarray(psd_pred, dtype=np.float64)[1:]
    psd_target = np.asarray(psd_target, dtype=np.float64)[1:]
    log_diff = np.log10(psd_pred) - np.log10(psd_target)
    return float(np.sqrt(np.mean(log_diff ** 2)))


def compute_derived_fields(data: torch.Tensor, var_names: list) -> dict:
    """
    Compute derived meteorological fields from raw variables.

    Parameters
    ----------
    data : torch.Tensor
        Shape (N, C, H, W).
    var_names : list
        List of variable names matching channel order.

    Returns
    -------
    dict
        Keys: 'vorticity', 'divergence', 'kinetic_energy'.
        Values: torch.Tensor of shape (N, H, W).
    """
    u_idx = var_names.index("u_wind")
    v_idx = var_names.index("v_wind")
    u = data[:, u_idx, :, :]  # (N, H, W)
    v = data[:, v_idx, :, :]

    dvdx = torch.gradient(v, dim=-1)[0]
    dudy = torch.gradient(u, dim=-2)[0]
    dudx = torch.gradient(u, dim=-1)[0]
    dvdy = torch.gradient(v, dim=-2)[0]

    derived = {
        "vorticity": dvdx - dudy,
        "divergence": dudx + dvdy,
        "kinetic_energy": 0.5 * (u ** 2 + v ** 2),
    }

    # Cn2 (atmospheric turbulence) — dry-air approximation
    t2m = data[:, var_names.index("t2m"), :, :]    # K
    sshf = -data[:, var_names.index("sshf"), :, :] / 1.216e3  # K m/s (convert from W/m² to K m/s using specific heat capacity of air)
    ustar = data[:, var_names.index("zust"), :, :]  # m/s
    sp = data[:, var_names.index("sp"), :, :]  / 100     # HPa

    A = 7.9e-5         # K/Pa
    Z = 2.0            # reference height m
    KAPPA = 0.4        # von Kármán constant
    G = 9.81           # gravitational acceleration m/s²

    # Obukhov length (preserve sign, guard against division by zero)
    theta_v = t2m
    ustar_safe = ustar.clamp(min=1e-6)
    sshf_safe = torch.where(sshf.abs() < 1e-6, torch.full_like(sshf, 1e-6), sshf)
    L = -(ustar_safe ** 3 * theta_v) / (KAPPA * G * sshf_safe)

    # stability parameter, clamp to avoid extreme values
    zeta = (Z / L).clamp(-10.0, 10.0)

    # Wyngaard (1971) stability function
    g_zeta = torch.where(
        zeta < 0,
        4.9 * (1 - 6.1 * zeta) ** (-2.0 / 3.0),  # unstable
        4.9 * (1 + 2.2 * zeta),                  # stable
    )

    CT2 = (-sshf / ustar_safe) ** 2 * Z ** (-2.0 / 3.0) * g_zeta
    Cn2 = (A * sp / t2m ** 2) ** 2 * CT2

    # Replace any remaining non-finite values with zero
    Cn2 = torch.where(torch.isfinite(Cn2), Cn2, torch.zeros_like(Cn2))

    derived["Cn2"] = Cn2

    return derived
