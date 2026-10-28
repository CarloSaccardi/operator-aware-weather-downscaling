# Operator-Aware Posterior Sampling for Multivariate Weather Downscaling

Code for the paper accepted at the **NeurIPS 2026 Tackling Climate Change with Machine Learning (TCCML) workshop**.

We downscale ERA5 weather fields to CERRA resolution using diffusion posterior sampling. A learned forward operator captures differences between the two reanalysis systems, while per-channel gradient normalization allows a single guidance strength across six weather variables. Both operate within the sampling guidance loop, keeping the high-resolution diffusion prior unchanged.

The repository includes the diffusion model, learned operator, conditional baselines, and example configurations. Set your own dataset and checkpoint paths in `yaml_configs/` before running; data and pretrained weights are not included.
