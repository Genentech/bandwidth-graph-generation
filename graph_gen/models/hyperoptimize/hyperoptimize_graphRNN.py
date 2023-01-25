import os
import argparse

import wandb
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


PROJECT = "graphRNN_sweeps"


def main():
    # wandb sweep
    run = wandb.init(project=PROJECT)
    lr = wandb.config.lr
    wd = wandb.config.wd
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    order = wandb.config.order
    order_func = ORDER_FUNCS[order]
    data_name = wandb.config.data_name
    workers = wandb.config.workers
    version = wandb.config.version

    # prep. data
    print("Preparing data...")
    graph_getter, num_repetitions = DATASETS[data_name]
    set_seeds(42)
    graphs = graph_getter()
    max_len = max(map(len, graphs))
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
    bw = max(  # max bw across splits
        max(g.bw for g in ordered_graphs)
        for ordered_graphs
        in [train_ordered_graphs, val_ordered_graphs, test_ordered_graphs]
    )
    train_dset = GraphRNNDataset(train_ordered_graphs, bw=bw)
    val_dset = GraphRNNDataset(val_ordered_graphs, bw=bw)
    train_dl = DataLoader(
        train_dset, batch_size=batch_size, num_workers=workers,
        shuffle=True, collate_fn=GraphRNNDataset.collate_fn,
    )
    val_dl = DataLoader(
        val_dset, batch_size=batch_size, num_workers=workers,
        collate_fn=GraphRNNDataset.collate_fn,
    )

    # set up trainer
    print("Preparing trainer...")
    output_folder = os.path.join(os.path.expanduser("~"), "scratch/graph_gen_logs")
    model_name = f"GraphRNN_bw-{bw}_order-{order}_data-{data_name}_version-{version}"
    loggers = [
        CSVLogger(save_dir=output_folder, name=model_name),
        WandbLogger(save_dir=output_folder, project=PROJECT, name=model_name),
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
    df, temp, mmd_mean = model.find_best_temperature(
        val_graphs[:256], max_graph_len=max_len,
    )
    df = df.set_index("temperature")
    for logger in loggers:
        logger.log_metrics({"mean_mmd": mmd_mean, "best_sampling_temp": temp})
        for col in df:
            logger.log_metrics({
                col: df.loc[temp][col],
            })
    wandb.finish()
    print("Done!")


if __name__ == "__main__":
    # get args
    parser = argparse.ArgumentParser(description="GraphRNN hyperopt using wandb")
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
        "--version", type=str, required=True,
    )
    parser.add_argument(
        "--count", type=int, required=True,
        help="How many hyperoptimization steps to take"
    )
    args = parser.parse_args()
    # set up sweep config
    name = f"BayesSweepGraphRNN_data-{args.data_name}_order-{args.order}_version-{args.version}"
    sweep_configuration = {
        'method': 'bayes',
        'name': name,
        'metric': {"goal": "minimize", "name": "mean_mmd"},
        'parameters': {
            'lr': {
                'distribution': "log_uniform_values",
                'max': 1e-2, 'min': 1e-4,
            },
            "epochs": {"value": args.epochs},
            "batch_size": {"value": args.batch_size},
            "order": {"value": args.order},
            "data_name": {"value": args.data_name},
            "version": {"value": args.version},
            "workers": {"value": 4},
            'wd': {
                'distribution': "log_uniform_values",
                'max': 1e-1, 'min': 1e-5,
            }
        }
    }
    # run the sweep!
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT)
    wandb.agent(sweep_id, function=main, count=args.count)
