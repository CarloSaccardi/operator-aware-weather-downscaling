import torch
import torch.nn.functional as F
from torch import Tensor
from physicsnemo.utils.generative import StackedRandomGenerator
import tqdm
import nvtx
from typing import Optional, Callable



def stochastic_sampler_DPS(
    net: torch.nn.Module,
    latents: Tensor,
    img_lr: Tensor,
    conditions: Tensor,
    num_steps: int = 18,
    sigma_min: float = 0.002,
    sigma_max: float = 800,
    rho: float = 7,
    zeta_lr: float = 10,
    a_operator: Optional[torch.nn.Module] = None,
    channel_weights: Optional[Tensor] = None,
    grad_norm: str = "channel",
) -> Tensor:
    """EDM Heun sampler with diffusion posterior sampling on the low-resolution field.

    `grad_norm` selects how the guidance gradient is scaled before it enters the
    drift:

        "channel"  divide each channel by its own Frobenius norm, so every
                   variable gets a unit-length step. A single `zeta_lr` is then a
                   scale-free displacement budget shared by all of them.
        "global"   DPS (Chung et al., ICLR 2023): no normalisation. The step is
                   zeta * grad of the un-squared residual norm, which is exactly
                   DPS's zeta_i * grad ||y - A(x0)||^2 with zeta_i = zeta' / ||y -
                   A(x0)||, since grad ||u||^2 = 2 ||u|| grad ||u|| and the factor
                   2 is absorbed into zeta. The raw per-channel magnitudes survive,
                   so channels with large gradients dominate the step.

    The two arms put `zeta_lr` on different scales — one multiplies a unit vector,
    the other a raw gradient — so a value tuned for one says nothing about the other.

    `a_operator` is the forward model inside the likelihood: the learned A_theta
    when given, and bicubic downsampling when None.
    """

    if grad_norm not in ("channel", "global"):
        raise ValueError(
            f"grad_norm must be 'channel' or 'global', got {grad_norm!r}"
        )
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )

    img_lr = img_lr.to(latents.device)
    x_next = latents * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):

        x_hat = x_next
        t_hat = t_cur

        def compute_guidance_gradient(x_hat, t_hat):
            x_hat = x_hat.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                denoised = net(x_hat, conditions, t_hat)

                ### LOSS era5 ###
                if zeta_lr > 0:
                    a_X0 = a_operator(torch.cat([denoised, conditions], dim=1)) if a_operator is not None else F.interpolate(denoised, size=img_lr.shape[-2:], mode='bicubic', align_corners=False)
                    # Per-sample residual norm, summed: each sample then gets its
                    # own grad ||y - A(x0)||. A norm over the whole batch would tie
                    # the samples together through a 1/||residual|| that only
                    # cancels under the "channel" normalisation below.
                    loss_lr = torch.linalg.vector_norm(img_lr - a_X0, dim=(-3, -2, -1)).sum()
                    torch.use_deterministic_algorithms(False)
                    grad_lr = torch.autograd.grad(outputs=loss_lr, inputs=x_hat)[0]
                    torch.use_deterministic_algorithms(True)
                    if grad_norm == "channel":
                        grad_lr = grad_lr / (torch.linalg.vector_norm(grad_lr, dim=(-2, -1), keepdim=True) + 1e-8)
                    # "global" leaves grad_lr untouched — that is the DPS step.
                else:
                    grad_lr = torch.zeros_like(x_hat)

                return grad_lr, denoised

        grad_lr, denoised = compute_guidance_gradient(x_hat, t_hat)
        d_cur = (x_hat - denoised) / t_hat + zeta_lr * grad_lr
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            grad_lr, denoised = compute_guidance_gradient(x_next, t_next)
            d_prime = (x_next - denoised) / t_next + zeta_lr * grad_lr
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def stochastic_sampler_CondDiff(
    net: torch.nn.Module,
    latents: Tensor,
    img_lr: Tensor,
    conditions: Tensor,
    num_steps: int = 18, 
    sigma_min: float = 0.002,
    sigma_max: float = 80,
    rho: float = 5,
    # S_churn: float = 0, #10
    # S_min: float = 0,
    # S_max: float = float("inf"),
    # S_noise: float = 1,
) -> Tensor:

    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  


    img_lr = img_lr.to(latents.device)

    x_next = latents * t_steps[0]
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        
        # x_cur = x_next
        # gamma = S_churn / num_steps if S_min <= t_cur <= S_max else 0
        # t_hat = net.round_sigma(t_cur + gamma * t_cur)
        # x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)
        
        x_hat = x_next
        t_hat = t_cur

        denoised = net(x_hat, img_lr, t_hat) 
        
        d_cur = ((x_hat - denoised) / t_hat)
        x_next = (x_hat + (t_next - t_hat) * d_cur )

        if i < num_steps - 1:
            denoised = net(x_next, img_lr, t_next) 
            d_prime = (x_next - denoised) / t_next 
            x_next = (x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime))
            
    return x_next



def diffusion_step(
    net: torch.nn.Module,
    sampler_fn: callable,
    img_shape: tuple,
    img_out_channels: int,
    rank_batches: list,
    img_lr: torch.Tensor,
    conditions: torch.Tensor,
    rank: int,
    device: torch.device,
    **sampler_kwargs,
) -> torch.Tensor:

    """
    Generate images using diffusion techniques as described in the relevant paper.

    This function applies a diffusion model to generate high-resolution images based on
    low-resolution inputs. It supports optional conditioning on high-resolution mean
    predictions and lead time labels.

    For each low-resolution sample in `img_lr`, the function generates multiple
    high-resolution samples, with different random seeds, specified in `rank_batches`.
    The function then concatenates these high-resolution samples across the batch dimension.

    Parameters
    ----------
    net : torch.nn.Module
        The diffusion model network.
    sampler_fn : callable
        Function used to sample images from the diffusion model.
    img_shape : tuple
        Shape of the images, (height, width).
    img_out_channels : int
        Number of output channels for the image.
    rank_batches : list
        List of batches of seeds to process.
    img_lr : torch.Tensor
        Low-resolution input image with shape (seed_batch_size, channels_lr, height, width).
    rank : int, optional
        Rank of the current process for distributed processing.
    device : torch.device, optional
        Device to perform computations.
    mean_hr : torch.Tensor, optional
        High-resolution mean tensor to be used as an additional input,
        with shape (1, channels_hr, height, width). Default is None.
    lead_time_label : torch.Tensor, optional
        Lead time label tensor for temporal conditioning,
        with shape (batch_size, lead_time_dims). Default is None.

    Returns
    -------
    torch.Tensor
        Generated images concatenated across batches with shape
        (seed_batch_size * len(rank_batches), out_channels, height, width).
    """


    img_lr = img_lr.to(memory_format=torch.channels_last)

    # Loop over batches
    all_images = []
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(rank != 0)):
        with nvtx.annotate(f"generate {len(all_images)}", color="rapids"):
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            # Initialize random generator, and generate latents
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn(
                [
                    img_lr.shape[0],
                    img_out_channels,
                    img_shape[0],
                    img_shape[1],
                ],
                device=device,
            ).to(memory_format=torch.channels_last)

            with torch.no_grad():
                images = sampler_fn(
                    net, latents, img_lr, conditions, **sampler_kwargs,
                )
            all_images.append(images)
    return torch.cat(all_images)