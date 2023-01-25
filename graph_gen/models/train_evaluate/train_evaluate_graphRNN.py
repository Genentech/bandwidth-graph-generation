import os
import argparse

import torch
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from graph_gen.data.orderings import (
    order_graphs, ORDER_FUNCS,
)
from graph_gen.data.nx_dataset import GraphRNNDataset
from graph_gen.models.graph_rnn import GraphRNNSimple
from graph_gen.data.data_utils import train_val_test_split, set_seeds
from graph_gen.data import DATASETS


PROJECT = "graphRNNResults"


def train_evaluate(
    lr: float,
    wd: float,
    epochs: int,
    batch_size: int,
    order: str,
    data_name: str,
    workers: int,
    version: str,
    replicate: int,
    temperature: float,
    bw: int = None,
):
    # prep. data
    print("Preparing data...")
    graph_getter, num_repetitions = DATASETS[data_name]
    set_seeds(42)
    graphs = graph_getter()
    max_len = max(map(len, graphs))
    train_graphs, val_graphs, test_graphs = train_val_test_split(graphs)
    order_func = ORDER_FUNCS[order]
    train_ordered_graphs = order_graphs(
        train_graphs, num_repetitions=num_repetitions, order_func=order_func,
    )
    val_ordered_graphs = order_graphs(
        val_graphs, num_repetitions=num_repetitions, order_func=order_func,
    )
    test_ordered_graphs = order_graphs(
        test_graphs, num_repetitions=1, order_func=order_func,
    )
    bw = bw or max(  # max bw across splits
        max(g.bw for g in ordered_graphs)
        for ordered_graphs
        in [train_ordered_graphs, val_ordered_graphs, test_ordered_graphs]
    )
    train_dset = GraphRNNDataset(train_ordered_graphs, bw=bw)
    val_dset = GraphRNNDataset(val_ordered_graphs, bw=bw)
    test_dset = GraphRNNDataset(test_ordered_graphs, bw=bw)
    train_dl = DataLoader(
        train_dset, batch_size=batch_size, num_workers=workers,
        shuffle=True, collate_fn=GraphRNNDataset.collate_fn,
    )
    val_dl = DataLoader(
        val_dset, batch_size=batch_size, num_workers=workers,
        collate_fn=GraphRNNDataset.collate_fn, shuffle=False,
    )
    test_dl = DataLoader(
        test_dset, batch_size=batch_size, num_workers=workers,
        collate_fn=GraphRNNDataset.collate_fn, shuffle=False,
    )

    # set up trainer
    print("Preparing trainer...")
    model_name = f"GraphRNN_bw-{bw}_order-{order}_data-{data_name}_version-{version}_replicate-{replicate}"
    output_folder = os.path.join(os.path.expanduser("~"), "scratch/graph_gen_logs", model_name)
    os.makedirs(output_folder, exist_ok=True)
    wandb_logger = WandbLogger(save_dir=output_folder, project=PROJECT, name=model_name)
    wandb_logger.experiment.config.update({
        "order": order, "data_name": data_name, "replicate": replicate,
    })
    loggers = [
        CSVLogger(save_dir=output_folder, name=model_name), wandb_logger,
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
    )

    # set up model
    torch.manual_seed(replicate)
    model = GraphRNNSimple(
        # hidden_dim=32,
        hidden_dim=128,
        epochs=epochs, bw=bw, lr=lr,
        wd=wd,
    )

    # fit model
    print("Training model...")
    trainer.fit(
        model, train_dataloaders=train_dl,
        val_dataloaders=val_dl,
    )

    # evaluate model
    print("Evaluating model...")
    model = model.load_from_checkpoint(checkpoint.best_model_path)
    model.to("cuda")
    # MMD
    n = 256
    mmd_results = model.evaluate_temperature(
        temperature, real_graphs=test_graphs[:n], max_graph_len=max_len,
    )
    mmds = [mmd for name, mmd in mmd_results.items() if "mmd" in name]
    mean_mmd = np.mean(mmds)
    for logger in loggers:
        logger.log_metrics({
            "mean_mmd": mean_mmd,
        })
        logger.log_metrics({
            name: mmd for name, mmd in mmd_results.items()
            if "mmd" in name
        })
    # AUPRC
    trainer.test(model, test_dl)


if __name__ == "__main__":
    # parse
    parser = argparse.ArgumentParser(description="GraphRNN training + results")
    parser.add_argument(
        "--lr", type=float, required=True,
        help="Learning rate",
    )
    parser.add_argument(
        "--wd", type=float, required=False,
        help="Weight decay", default=0,
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch_size", type=int, required=False, default=32,
    )
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
        "--workers", type=int, required=False, default=4,
    )
    parser.add_argument(
        "--version", required=True, type=str,
    )
    parser.add_argument(
        "--replicate", type=int, required=True,
    )
    parser.add_argument(
        "--temperature", type=float, required=True,
        help="Temperature of RNN sampling",
    )
    parser.add_argument(
        "--bw", type=int, required=False, default=None,
        help="Manually set the bandwidth to potentially sub-maximal values.",
    )
    args = parser.parse_args()

    # run
    train_evaluate(
        lr=args.lr, wd=args.wd, epochs=args.epochs, batch_size=args.batch_size,
        order=args.order, data_name=args.data_name, workers=args.workers, version=args.version,
        replicate=args.replicate, temperature=args.temperature, bw=args.bw,
    )
