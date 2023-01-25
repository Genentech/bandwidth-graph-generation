import argparse
import os
from datetime import datetime, timedelta
import time

import torch
from torch_geometric.loader import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from graph_gen.data.orderings import (
    order_graphs, ORDER_FUNCS,
)
from graph_gen.data.nx_dataset import GraphAEDataset
from graph_gen.models.gine_stack import GINEStackVAE
from graph_gen.data.data_utils import train_val_test_split, set_seeds
from graph_gen.data import DATASETS


PROJECT = "gine_vae_results"


def train_evaluate(
    edge_augmentation: str,
    kl_weight: float,
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
    replicate: int,
    sigma: float,
):
    start = time.time()
    # wandb sweep
    order_func = ORDER_FUNCS[order]

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
    test_ordered_graphs = order_graphs(
        test_graphs, num_repetitions=1, order_func=order_func,
    )
    bw = None
    train_dset = GraphAEDataset(
        train_ordered_graphs, bw=bw, pe_dim=pe_dim,
        edge_augmentation=edge_augmentation, empirical_bandwidth=empirical_bw,
    )
    val_dset = GraphAEDataset(
        val_ordered_graphs, bw=bw, pe_dim=pe_dim,
        edge_augmentation=edge_augmentation, empirical_bandwidth=empirical_bw,
    )
    test_dset = GraphAEDataset(
        test_ordered_graphs, bw=bw, pe_dim=pe_dim, edge_augmentation=edge_augmentation,
        empirical_bandwidth=empirical_bw,
    )
    train_dl = DataLoader(
        train_dset, batch_size=batch_size, num_workers=workers,
        shuffle=True,
    )
    val_dl = DataLoader(
        val_dset, batch_size=batch_size, num_workers=workers,
    )
    test_dl = DataLoader(
        test_dset, batch_size=batch_size, num_workers=workers,
    )

    # set up trainer
    print("Preparing trainer...")
    output_folder = os.path.join(os.path.expanduser("~"), "scratch/graph_gen_logs")
    model_name = f"Graphite_empircalBW-{empirical_bw}_order-{order}_data-{data_name}_version-{version}_replicate-{replicate}"
    wandb_logger = WandbLogger(save_dir=output_folder, project=PROJECT, name=model_name)
    wandb_logger.experiment.config.update({
        "order": order, "edge_augmentation": edge_augmentation or "none", "data_name": data_name,
        "replicate": replicate, "empirical_bw": empirical_bw,
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
    model = GINEStackVAE(
        node_features=pe_dim, hidden_dim=hidden_dim, kl_weight=kl_weight,
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
    # auprc
    trainer.test(model, test_dl)
    # MMDs
    n = 256
    skeleton_dset = GraphAEDataset(
        train_ordered_graphs[:n], bw=bw, edge_augmentation=edge_augmentation,
        empirical_bandwidth=empirical_bw, pe_dim=pe_dim,
    )
    skeleton_loader = DataLoader(skeleton_dset, shuffle=False, batch_size=n)
    skeleton_graphs = next(iter(skeleton_loader))
    mmd_results = model.evaluate_sigma(sigma=sigma, skeleton_graphs=skeleton_graphs, real_graphs=test_graphs[:n])
    wandb_logger.log_metrics(mmd_results)


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser(description="GINEVAE hyperopt using wandb")
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
    parser.add_argument(
        "--edge_augmentation", type=str, required=False, default="none",
        choices=["none", "complete"],
    )
    parser.add_argument(
        "--hidden_dim", type=int, required=True,
        help="Hidden dimension of Graphite model",
    )
    parser.add_argument(
        "--pe_dim", type=int, required=False, default=16,
        help="Positional encoding dimension",
    )
    parser.add_argument(
        "--version", type=str, required=True,
    )
    parser.add_argument(
        "--empirical_bw", action="store_true",
        help="Whether to use the graph's bandwidth to restrict output space"
    )
    parser.add_argument(
        "--replicate", required=True, type=int,
    )
    parser.add_argument(
        "--kl_weight", type=float, required=True,
        help="Weight of KL-divergence in the loss",
    )
    parser.add_argument(
        "--lr", type=float, required=True,
        help="Learning rate",
    )
    parser.add_argument(
        "--wd", type=float, required=False,
        help="Weight decay", default=0,
    )
    parser.add_argument(
        "--sigma", required=True, type=float,
        help="Latent sampling standard deviation"
    )
    parser.add_argument(
        "--workers", type=int, required=False, default=4,
    )
    args = parser.parse_args()
    # run model
    train_evaluate(
        edge_augmentation=args.edge_augmentation, kl_weight=args.kl_weight, lr=args.lr, wd=args.wd,
        pe_dim=args.pe_dim, epochs=args.epochs, batch_size=args.batch_size, order=args.order,
        data_name=args.data_name, workers=args.workers, version=args.version,
        hidden_dim=args.hidden_dim, empirical_bw=args.empirical_bw, replicate=args.replicate,
        sigma=args.sigma,
    )
