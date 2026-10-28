# Standard library
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import random
import time
from argparse import ArgumentParser

# Third-party
import pytorch_lightning as pl
import torch
from lightning_fabric.utilities import seed
import numpy as np

# First-party
from src.utils import utils
from src.models import DiffusionWrapper, GeoUNetWrapper, GeoDiffWrapper, CorrDiffWrapper, AOperatorWrapper, load_a_operator
from src.data import CerraPriorDataset
import yaml

MODELS = {
    "Diffusion": DiffusionWrapper,
    "GeoUNet": GeoUNetWrapper,
    "GeoDiff": GeoDiffWrapper,
    "CorrDiff": CorrDiffWrapper,
}


def get_args():
    """
    Main function for training and evaluating models
    """
    parser = ArgumentParser(
        description="Train or evaluate NeurWP models for LAM"
    )
    # General options
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (e.g., yaml_configs/good_runs/UNet/UNet_test.yaml)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Override config/CLI values using key=value pairs. "
            "Can be repeated. Values are parsed as YAML "
            "(e.g., --override lr=1e-4 --override channel_mult=[1,2,2])"
        ),
    )
    parser.add_argument(
        "--dataset_cerra",
        type=str,
        default="data/CERRA",
        help="Dataset, corresponding to name in data directory "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--dataset_era5",
        type=str,
        default="data/ERA5",
        help="Dataset, corresponding to name in data directory "
        "(default: meps_example)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="graph_efm",
        help="Model architecture to train/evaluate (default: graph_lam)",
    )
    parser.add_argument(
        "--subset_ds",
        type=int,
        default=0,
        help="Use only a small subset of the dataset, for debugging"
        "(default: 0=false)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of workers in data loader (default: 4)",
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to load model parameters from (default: None)",
    )
    parser.add_argument(
        "--restore_opt",
        type=int,
        default=0,
        help="If optimizer state should be restored with model "
        "(default: 0 (false))",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Numerical precision to use for model (32/16/bf16) (default: 32)",
    )
    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="Eval model on given data split (val/test) "
        "(default: None (train model))",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="neural-lam",
        help="Wandb project name",
    )
    parser.add_argument(
        "--run_name",  
        type=str,
        default=None,
        help="Name of the run",
    )
    ########################################################
    # DATASET #
    parser.add_argument(
        "--output_variables",  
        type=list,
        default=None,
        help="List of output variables to predict",
    )
    parser.add_argument(
        "--img_in_channels",  
        type=int,
        default=5,
        help="Number of input channels",
    )
    parser.add_argument(
        "--img_out_channels",  
        type=int,
        default=5,
        help="Number of output channels",
    )
    parser.add_argument(
        "--img_resolution",  
        type=list,
        default=[300, 300],
        help="Resolution of the input images",
    )
    parser.add_argument(
        "--crop",
        type=bool,
        default=False,
        help="Crop the input regions such that we can obtain 5 non-overlapping regions",
    )
    parser.add_argument(
        "--era5_coarsen",
        type=int,
        default=1,
        help="Area-average ERA5 by this factor on load. 1 = native ~25km (77x77), "
             "2 = ~50km (39x39). Changing it changes the coarse source, so A_theta "
             "must be retrained for it; the diffusion prior never sees ERA5 and is "
             "reused unchanged.",
    )
    parser.add_argument(
        "--a_target_size",
        type=int,
        nargs="+",
        default=None,
        help="Grid the guidance residual y - A(x0) lives on, matching the ERA5 "
             "grid after --era5_coarsen (default 77 77; use 39 39 with "
             "--era5_coarsen 2)",
    )
    parser.add_argument(
        "--a_operator_native_size",
        type=int,
        nargs="+",
        default=None,
        help="Grid --a_operator_ckpt was trained to emit, when it differs from "
             "--a_target_size. The operator is then built at this size and its "
             "output area-averaged onto --a_target_size. Set it to 77 77 to reuse "
             "the ~25 km A_theta against the ~50 km coarse source: that field is "
             "itself the ~25 km one area-averaged, so pooling the operator's "
             "output is exact and no retraining is needed. Defaults to "
             "--a_target_size (operator trained for the grid it is used on).",
    )
    ########################################################
    # MODEL #
    parser.add_argument(
        "--N_grid_channels",  
        type=int,
        default=4,
        help="Number of grid channels",
    )
    parser.add_argument(
        "--embedding_type",  
        type=str,
        default="zero",
        help="List of output variables to predict",
    )
    parser.add_argument(
        "--model_channels",  
        type=int,
        default=64,
        help="Number of model channels",
    )
    parser.add_argument(
        "--channel_mult",  
        type=list,
        default=[1, 2, 2],
        help="List of channel multipliers",
    )
    parser.add_argument(
        "--attn_resolutions",  
        type=list,
        default=[16],
        help="List of attention resolutions",
    )
    parser.add_argument(
        "--model_type",  
        type=str,
        default="SongUNetPosEmbd",
        help="List of attention resolutions",
    )
    ########################################################
    # TRAINING #
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run "
        "(default: 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4, help="learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size (default: 4)"
    )
    parser.add_argument(
        "--lr_decay", type=int, default=1, help="learning rate decay (default: 1)"
    )
    parser.add_argument(
        "--lr_rampup", type=int, default=0, help="learning rate rampup (default: 0)"
    )
    parser.add_argument(
        "--gridtype",
        type=str,
        help="Type of positional grid to use: 'sinusoidal', 'learnable', 'linear', or 'test'.",
    )
    parser.add_argument(
        "--num_ensembles",
        type=int,
        default=32,
        help="Number of ensembles to generate with diffusion",
    )
    parser.add_argument(
        "--zeta_lr",
        type=float,
        default=100.0,
        help="DPS guidance strength for low-resolution observations (default: 100.0)",
    )
    parser.add_argument(
        "--grad_norm",
        type=str,
        default="channel",
        choices=["channel", "global"],
        help="How the guidance gradient is scaled: 'channel' = unit norm per "
        "(sample, variable), 'global' = DPS, the raw gradient unnormalized "
        "(default: channel)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to resume training from (default: None)",
    )
    parser.add_argument(
        "--channel_weights",
        type=list,
        default=None,
        help="Per-channel loss weights for diffusion training (default: None = equal weights)",
    )
    parser.add_argument(
        "--data_subset_len",
        type=int,
        default=None,
        help="Number of random data samples to use from the dataset (default: None = use all)",
    )
    parser.add_argument(
        "--a_operator_ckpt",
        type=str,
        default=None,
        help="Path to trained A operator checkpoint (default: None = use area downsampling)",
    )
    parser.add_argument(
        "--checkpoint_level",
        type=int,
        default=0,
        help="Checkpoint level for the model (default: 0)",
    )
    parser.add_argument(
        "--regression_net",
        type=str,
        default=None,
        help="Path to load model parameters from regression step.",
    )


    args, _ = parser.parse_known_args()
    if args.config is not None:
        with open(str(args.config), "r") as f:
            config = yaml.safe_load(f) or {}
        parser.set_defaults(**config)
    args = parser.parse_args()
    if args.override:
        update_args(args, parse_overrides(args.override))
    return args


def worker_init_fn(worker_id):
    """Seed each worker's NumPy and Python random state."""
    worker_seed = args.seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def main(args):
    # Asserts for arguments
    assert args.model in MODELS, f"Unknown model: {args.model}"
    assert args.eval in (
        None,
        "val",
        "test",
    ), f"Unknown eval setting: {args.eval}"

    # Get an (actual) random run id as a unique identifier
    random_run_id = random.randint(0, 9999)
    num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    devices = torch.cuda.device_count()
    print(f"Using {devices} GPUs")

    # Set seed
    seed.seed_everything(args.seed)

    # Instantiate model + trainer
    if torch.cuda.is_available():
        device_name = "gpu"
    else:
        device_name = "cpu"

    # Load model parameters Use new args for model
    model_class = MODELS[args.model]
    if args.load:
        model = model_class.load_from_checkpoint(args.load, args=args, strict=False)
        if args.restore_opt:
            model.opt_state = torch.load(args.load)["optimizer_states"][0]
    else:
        model = model_class(args)

    # Load learned A operator CNN (not part of diffusion state dict). Frozen, and
    # pooled onto a_target_size when the checkpoint was trained on a finer grid.
    if getattr(args, 'a_operator_ckpt', None) is not None:
        model.a_operator = load_a_operator(args)

    prefix = "subset-" if args.subset_ds else ""
    prefix += args.run_name if hasattr(args, "run_name") else ""
    if args.eval:
        prefix = prefix + f"eval-{args.eval}-"
    run_name = (
        f"{prefix}-{args.model}-"
        f"{time.strftime('%m_%d_%H')}-{random_run_id:04d}"
    )

    # Callbacks for saving model checkpoint
    callbacks = []
    callbacks.append(
        pl.callbacks.ModelCheckpoint(
            dirpath=f"saved_models/{run_name}",
            filename="min_val_loss",
            monitor="val_loss",
            mode="min",
            save_last=True,
        )
    )
    
    if args.wandb_project is not None:
        logger = pl.loggers.WandbLogger(
            project=args.wandb_project, name=run_name, config=args
        )
    else:
        logger = pl.loggers.TensorBoardLogger(
            save_dir="DebugLogs/", name=run_name
        )  # or CSVLogger


    # Training strategy
    # If doing pure autoencoder training (kl_beta = 0), the prior network is not
    # used at all in producing the loss. This is desired, but DDP complains.
    strategy = "ddp"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=False,
        strategy=strategy,
        accelerator=device_name,
        devices=devices,
        num_nodes=num_nodes,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
        inference_mode=False, # This is important for the model to be able to compute gradients
    )

    # Only init once, on rank 0 only
    if trainer.global_rank == 0 and isinstance(logger, pl.loggers.WandbLogger):
        utils.init_wandb_metrics(logger)  # Do after wandb.init

    if args.eval:
        regions = ["Greenland", "WestRussia", "Turkey", "Iberia", "CentralEurope"] #, 

        test_dataset_list = []

        for region in regions:
            # Load data
            test_dataset = CerraPriorDataset(
                args.dataset_cerra,
                args.dataset_era5,
                region=region,
                split="test",
                n_samples=args.data_subset_len,
                crop=args.crop,
                era5_coarsen=args.era5_coarsen,
            )
            
            test_dataset_list.append(test_dataset)
            
            #test_dataset = torch.utils.data.ConcatDataset(test_dataset_list)

            test_loader = torch.utils.data.DataLoader(  
                test_dataset,
                args.batch_size,
                shuffle=False,
                num_workers=args.n_workers,
            )
        
            # Tag region so test_step can log per-region metrics
            model.current_region = region
            trainer.test(model, dataloaders=test_loader)
            
        test_loader.dataset.close()

    else:
        
        regions = ["Iberia", "CentralEurope"]
        train_dataset_list = []
        val_dataset_list = []

        for region in regions:
            # Load data
            dataset_region_train = CerraPriorDataset(
                args.dataset_cerra,
                args.dataset_era5,
                region=region,
                split="train",
                n_samples=None,
                crop=args.crop,
                era5_coarsen=args.era5_coarsen,
            )

            dataset_region_val = CerraPriorDataset(
                args.dataset_cerra,
                args.dataset_era5,
                region=region,
                split="val",
                n_samples=None,
                crop=args.crop,
                era5_coarsen=args.era5_coarsen,
            )
            
            train_dataset_list.append(dataset_region_train)
            val_dataset_list.append(dataset_region_val)
            
        train_dataset = torch.utils.data.ConcatDataset(train_dataset_list)
        val_dataset = torch.utils.data.ConcatDataset(val_dataset_list)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            args.batch_size,
            shuffle=True,
            num_workers=args.n_workers,
            worker_init_fn=worker_init_fn
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
            worker_init_fn=worker_init_fn
        )
        
        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path= args.resume if args.resume else None,
            
        )
        
        train_loader.dataset.close()
        val_loader.dataset.close()


def update_args(args, config_dict):
    for key, val in config_dict.items():
        setattr(args, key, val)  
  

def parse_overrides(override_list):
    overrides = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Override must be in key=value format: {item}")
        key, raw_value = item.split("=", 1)
        overrides[key] = yaml.safe_load(raw_value)
    return overrides


if __name__ == '__main__':
    
    args = get_args()
    main(args)