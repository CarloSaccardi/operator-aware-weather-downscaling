# Standard library
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
from src.models.a_operator_CNN import AOperatorWrapper
from src.data import CerraPriorDataset
import os
import yaml


def get_args():
    """
    Main function for training and evaluating the learned A operator.
    """
    parser = ArgumentParser(
        description="Train or evaluate the learned A operator for downscaling"
    )
    # General options
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help=(
            "Override config/CLI values using key=value pairs. "
            "Can be repeated. Values are parsed as YAML "
            "(e.g., --override lr=1e-4)"
        ),
    )
    parser.add_argument(
        "--dataset_cerra",
        type=str,
        default="data/CERRA",
        help="Path to CERRA dataset directory",
    )
    parser.add_argument(
        "--dataset_era5",
        type=str,
        default="data/ERA5",
        help="Path to ERA5 dataset directory",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="AOperator",
        help="Model architecture",
    )
    parser.add_argument(
        "--subset_ds",
        type=int,
        default=0,
        help="Use only a small subset of the dataset, for debugging (default: 0=false)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--n_workers",
        type=int,
        default=8,
        help="Number of workers in data loader (default: 8)",
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
        help="If optimizer state should be restored with model (default: 0 (false))",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Numerical precision to use for model (32/16/bf16) (default: 16-mixed)",
    )
    # Evaluation options
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="Eval model on given data split (val/test) (default: None (train model))",
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
        "--img_in_channels",
        type=int,
        default=7,
        help="Number of input channels (6 vars + 1 orography)",
    )
    parser.add_argument(
        "--img_out_channels",
        type=int,
        default=6,
        help="Number of output channels",
    )
    parser.add_argument(
        "--img_resolution",
        type=list,
        default=[320, 320],
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
             "2 = ~50km (39x39). A_theta is fitted to whatever comes out, so this "
             "is the only thing that has to change to target a coarser source.",
    )
    parser.add_argument(
        "--a_target_size",
        type=int,
        nargs="+",
        default=None,
        help="Grid A_theta emits, matching the ERA5 grid after --era5_coarsen "
             "(default 77 77; use 39 39 with --era5_coarsen 2)",
    )
    ########################################################
    # TRAINING #
    parser.add_argument(
        "--val_interval",
        type=int,
        default=1,
        help="Number of epochs training between each validation run (default: 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=2e-4, help="learning rate (default: 2e-4)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="upper epoch limit (default: 200)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="batch size (default: 8)"
    )
    parser.add_argument(
        "--data_subset_len",
        type=int,
        default=None,
        help="Number of random data samples to use from the dataset (default: None = use all)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to resume training from (default: None)",
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

    # Load model parameters
    if args.load:
        model = AOperatorWrapper.load_from_checkpoint(args.load, args=args)
        if args.restore_opt:
            model.opt_state = torch.load(args.load)["optimizer_states"][0]
    else:
        model = AOperatorWrapper(args)

    prefix = "subset-" if args.subset_ds else ""
    prefix += args.run_name if hasattr(args, "run_name") and args.run_name else ""
    if args.eval:
        prefix = prefix + f"eval-{args.eval}-"
    run_name = (
        f"{prefix}-AOperator-"
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
        )

    strategy = "ddp"

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        deterministic=True,
        strategy=strategy,
        accelerator=device_name,
        devices=devices,
        num_nodes=num_nodes,
        logger=logger,
        log_every_n_steps=1,
        callbacks=callbacks,
        check_val_every_n_epoch=args.val_interval,
        precision=args.precision,
    )

    # Only init once, on rank 0 only
    if trainer.global_rank == 0 and isinstance(logger, pl.loggers.WandbLogger):
        utils.init_wandb_metrics(logger)

    if args.eval:
        regions = ["WestRussia", "Greenland", "Iberia", "Turkey", "CentralEurope"]

        for region in regions:
            test_dataset = CerraPriorDataset(
                args.dataset_cerra,
                args.dataset_era5,
                region=region,
                split="test",
                n_samples=args.data_subset_len,
                crop=args.crop,
                era5_coarsen=args.era5_coarsen,
            )

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                args.batch_size,
                shuffle=False,
                num_workers=args.n_workers,
            )

            model.current_region = region
            trainer.test(model, dataloaders=test_loader)

        test_loader.dataset.close()

    else:
        regions = ["Iberia", "CentralEurope"]
        train_dataset_list = []
        val_dataset_list = []

        for region in regions:
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
            worker_init_fn=worker_init_fn,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            args.batch_size,
            shuffle=False,
            num_workers=args.n_workers,
            worker_init_fn=worker_init_fn,
        )

        trainer.fit(
            model=model,
            train_dataloaders=train_loader,
            val_dataloaders=val_loader,
            ckpt_path=args.resume if args.resume else None,
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


if __name__ == "__main__":
    args = get_args()
    main(args)
