import argparse
import os
from datetime import datetime, timedelta
import time

import numpy as np
import torch
import wandb
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from graph_gen.data.orderings import (
    order_graphs, ORDER_FUNCS,
)
from graph_gen.data.nx_dataset import GraphDiffusionDataset
from graph_gen.models.gine_stack_score_matching import GINEStackDiffusion, evaluate_diffusion_sampling
from graph_gen.data.data_utils import train_val_test_split, set_seeds
from graph_gen.data import DATASETS
from graph_gen.models.diffusion_beta_schedules import BETA_SCHEDULES


PROJECT = "gine_diffusion_results"


def main(
    edge_augmentation: str,
    lr: float,
    wd: float,
    pe_dim: int,
    epochs: int,
    batch_size: int,
    order: str,
    data_name: str,
    workers: int,
    version: str,
    hidden_dim: int,
    empirical_bw: bool,
    beta_schedule: str,
    time_steps: int,
    replicate: int,
):
    start = time.time()
    order_func = ORDER_FUNCS[order]
    betas = BETA_SCHEDULES[beta_schedule](time_steps)

    # prep. data
    print("Preparing data...")
    graph_getter, num_repetitions = DATASETS[data_name]
    set_seeds(42)
    graphs = graph_getter()
    train_graphs, val_graphs, test_graphs = train_val_test_split(graphs)
    train_ordered_graphs = order_graphs(
        train_graphs, num_repetitions=num_repetitions, order_func=order_func,
    )
    val_ordered_graphs = order_graphs(
        val_graphs, num_repetitions=num_repetitions, order_func=order_func,
    )
    bw = None
    train_dset = GraphDiffusionDataset(
        train_ordered_graphs, bw=bw, pe_dim=pe_dim,
        edge_augmentation=edge_augmentation, empirical_bandwidth=empirical_bw,
        betas=betas,
    )
    val_dset = GraphDiffusionDataset(
        val_ordered_graphs, bw=bw, pe_dim=pe_dim,
        edge_augmentation=edge_augmentation, empirical_bandwidth=empirical_bw,
        betas=betas,
    )
    train_dl = DataLoader(
        train_dset, batch_size=batch_size, num_workers=workers,
        shuffle=True,
    )
    val_dl = DataLoader(
        val_dset, batch_size=batch_size, num_workers=workers,
    )

    # set up trainer
    print("Preparing trainer...")
    output_folder = os.path.join(os.path.expanduser("~"), "scratch/graph_gen_logs")
    model_name = f"Diffusion_empircalBW-{empirical_bw}_order-{order}_data-{data_name}_version-{version}_replicate-{replicate}"
    wandb_logger = WandbLogger(save_dir=output_folder, project=PROJECT, name=model_name)
    wandb_logger.experiment.config.update({
        "order": order, "edge_augmentation": edge_augmentation or "none", "data_name": data_name,
        "replicate": replicate, "empirical_bw": empirical_bw, "time_steps": time_steps,
        "beta_schedule": beta_schedule,
    })
    loggers = [
        CSVLogger(save_dir=output_folder, name=model_name),
        wandb_logger,
    ]
    checkpoint = ModelCheckpoint(dirpath=output_folder, filename=model_name)
    callbacks = [
        LearningRateMonitor(),
        checkpoint,
    ]
    trainer = Trainer(
        accelerator="gpu", devices=1,
        max_epochs=epochs,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=1,
        enable_progress_bar=False,
        limit_train_batches=30,
        limit_val_batches=9,
        max_time=timedelta(hours=1),
    )

    # set up model
    torch.manual_seed(replicate)
    model = GINEStackDiffusion(
        node_features=pe_dim, hidden_dim=hidden_dim,
        epochs=epochs, lr=lr, wd=wd,
    ).to("cuda")

    # train model
    print(f"Training model {datetime.now()}...")
    trainer.fit(
        model, train_dataloaders=train_dl,
        val_dataloaders=val_dl,
    )

    # evaluate model
    print(f"Evaluating model {datetime.now()}...")
    model = model.load_from_checkpoint(checkpoint.best_model_path).to("cuda")
    # MMDs
    n = 256
    skeleton_dset = GraphDiffusionDataset(
        train_ordered_graphs[:n], bw=bw, edge_augmentation=edge_augmentation,
        empirical_bandwidth=empirical_bw, pe_dim=pe_dim, betas=betas,
    )
    skeleton_loader = DataLoader(skeleton_dset, shuffle=False, batch_size=n)
    skeleton_graphs = next(iter(skeleton_loader))
    mmd_results = evaluate_diffusion_sampling(
        model.to("cuda"), betas.to("cuda"),
        skeleton_graphs.to("cuda"), test_graphs[:256],
    )
    mean_mmd = np.mean(list(mmd_results.values()))
    mmd_results["mean_mmd"] = mean_mmd
    wandb_logger.log_metrics(mmd_results)
    wandb_logger.log_metrics({
        "total_time": time.time() - start,
    })
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser(description="GINE diffusion training")
    parser.add_argument(
        "--edge_augmentation", type=str, required=False, default="none",
        choices=["none", "complete"],
    )
    parser.add_argument(
        "--lr", type=float, required=True,
        help="Learning rate",
    )
    parser.add_argument(
        "--wd", type=float, required=False,
        help="Weight decay", default=0,
    )
    parser.add_argument("--pe_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--order", type=str, required=True,
        choices=list(ORDER_FUNCS),
        help="Which node ordering, e.g. C-M for Cuthill-McKee",
    )
    parser.add_argument(
        "--data_name", type=str, required=True,
        choices=list(DATASETS),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--version", type=str, required=True,
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128,
        help="Hidden dimension of Graphite model",
    )
    parser.add_argument(
        "--empirical_bw", action="store_true",
        help="Whether to use the graph's bandwidth to restrict output space"
    )
    parser.add_argument(
        "--beta_schedule", type=str, default="cosine",
        help="Variance schedule for diffusion",
        choices=list(BETA_SCHEDULES),
    )
    parser.add_argument(
        "--time_steps", type=int, required=False, default=200,
        help="Number of diffusion time steps",
    )
    parser.add_argument(
        "--replicate", required=True, type=int,
    )

    args = parser.parse_args()
    main(**args.__dict__)
