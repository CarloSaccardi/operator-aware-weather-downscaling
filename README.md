# Operator-Aware Posterior Sampling for Multivariate Weather Downscaling

Code for the paper by **Carlo Saccardi, Rudolf Saathof, Geethu Joseph, and Justin Dauwels**, accepted at the **NeurIPS 2026 Tackling Climate Change with Machine Learning (TCCML) workshop**.

We downscale ERA5 weather fields (~25 km) to CERRA resolution (~5.5 km) using diffusion posterior sampling. The method combines a high-resolution diffusion prior with two changes to the sampling guidance:

- **A learned forward operator** maps high-resolution fields to the coarse source, capturing differences between reanalysis systems that a fixed downsampler cannot represent.
- **Per-channel gradient normalization** balances guidance across weather variables, allowing a single guidance strength.

The prior and forward operator are trained separately and frozen during inference. Coarse weather fields enter through the guidance loop, keeping the high-resolution prior independent of the coarse source.

Experiments cover Central Europe and Iberia and six surface variables: 10 m wind components (`u10`, `v10`), 2 m temperature (`t2m`), surface sensible heat flux (`sshf`), friction velocity (`zust`), and surface pressure (`sp`). Orography provides static geographical conditioning.

## Repository

- `main.py`: training and evaluation for GeoDiff and the conditional diffusion, CorrDiff, and GeoUNet baselines.
- `train_operator.py`: training and evaluation of the learned forward operator.
- `src/`: model implementations, dataset loading, sampling, and utilities.
- `yaml_configs/`: example training and evaluation configurations.
- `physicsnemo/`: vendored NVIDIA PhysicsNeMo components.

## Usage

Dependencies are listed in `requirements.txt`. Prepare your ERA5/CERRA data in the format expected by `src/data/dataset.py`, then set `dataset_cerra` and `dataset_era5` in the relevant YAML files. For evaluation, also set `load` and `a_operator_ckpt` to your trained checkpoints.

```bash
# Train the high-resolution diffusion prior
python main.py --config yaml_configs/GeoDiff/geodiff_train.yaml

# Train the forward operator
python train_operator.py --config yaml_configs/AOperator/a_operator_train.yaml

# Evaluate with posterior sampling
python main.py --config yaml_configs/GeoDiff/geodiff_test.yaml
```

The configurations are starting points; adjust settings for your data and hardware. Data and pretrained weights are not included.
