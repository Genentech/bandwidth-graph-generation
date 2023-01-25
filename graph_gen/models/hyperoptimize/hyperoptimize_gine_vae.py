import argparse
import os
from datetime import datetime, timedelta
import time


import wandb
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


PROJECT = "gine_vae_sweeps"


def main():
    start = time.time()
    # wandb sweep
    run = wandb.init(project=PROJECT)
    edge_augmentation = wandb.config.edge_augmentation
    kl_weight = wandb.config.kl_weight
    lr = wandb.config.lr
    wd = wandb.config.wd
    pe_dim = wandb.config.pe_dim
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    order = wandb.config.order
    order_func = ORDER_FUNCS[order]
    data_name = wandb.config.data_name
    workers = wandb.config.workers
    version = wandb.config.version
    hidden_dim = wandb.config.hidden_dim
    empirical_bw = wandb.config.empirical_bw

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
    train_dset = GraphAEDataset(
        train_ordered_graphs, bw=bw, pe_dim=pe_dim,
        edge_augmentation=edge_augmentation, empirical_bandwidth=empirical_bw,
    )
    val_dset = GraphAEDataset(
        val_ordered_graphs, bw=bw, pe_dim=pe_dim,
        edge_augmentation=edge_augmentation, empirical_bandwidth=empirical_bw,
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
    model_name = f"Graphite_empircalBW-{empirical_bw}_order-{order}_data-{data_name}_version-{version}"
    wandb_logger = WandbLogger(save_dir=output_folder, project=PROJECT, name=model_name)
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
    eval_results = trainer.test(model, val_dl)[0]
    auprc = eval_results["test/auprc"]
    # MMDs
    n = 256
    skeleton_dset = GraphAEDataset(
        train_ordered_graphs[:n], bw=bw, edge_augmentation=edge_augmentation,
        empirical_bandwidth=empirical_bw, pe_dim=pe_dim,
    )
    skeleton_loader = DataLoader(skeleton_dset, shuffle=False, batch_size=n)
    skeleton_graphs = next(iter(skeleton_loader))
    df = model.to("cuda").find_best_sigma(
        skeleton_graphs=skeleton_graphs.to("cuda"),
        real_graphs=[g.graph for g in val_ordered_graphs[:n]],
        sigmas=[1.0],
    )
    df = df.set_index("sigma")
    mean_mmd = df.mean(axis=1).min()
    sigma = df.mean(axis=1).idxmin()
    hyperopt_target = mean_mmd - auprc
    for logger in loggers:
        logger.log_metrics({
            "mean_mmd": mean_mmd, "sigma": sigma,
            "mean_mmd_minus_auprc": hyperopt_target,
            "final_auprc": auprc,
        })
        for col in df:
            logger.log_metrics({
                col: df.loc[sigma][col],
            })
    wandb_logger.log_metrics({
        "total_time": time.time() - start,
    })
    wandb.finish()
    print("Done!")


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
        "--version", type=str, required=True,
    )
    parser.add_argument(
        "--count", type=int, required=True,
        help="How many hyperoptimization steps to take",
    )
    parser.add_argument(
        "--empirical_bw", action="store_true",
        help="Whether to use the graph's bandwidth to restrict output space"
    )
    args = parser.parse_args()

    # set up sweep
    name = f"SweepGINEVae_data-{args.data_name}_order-{args.order}_edgeAug-{args.edge_augmentation}_version-{args.version}"
    sweep_configuration = {
        'method': 'bayes',
        'name': name,
        'metric': {"goal": "minimize", "name": "mean_mmd_minus_auprc"},
        'parameters': {
            'lr': {
                'distribution': "log_uniform_values",
                'max': 1e-2, 'min': 1e-4,
            },
            "kl_weight": {
                'distribution': "log_uniform_values",
                'max': 1, 'min': 1e-5,
            },
            "epochs": {"value": args.epochs},
            "batch_size": {"value": args.batch_size},
            "order": {"value": args.order},
            "data_name": {"value": args.data_name},
            "version": {"value": args.version},
            "workers": {"value": 4},
            'wd': {"value": 0},
            "edge_augmentation": {"value": args.edge_augmentation},
            "pe_dim": {"value": 16},
            "hidden_dim": {"value": args.hidden_dim},
            "empirical_bw": {"value": args.empirical_bw},
        },
    }
    # run the sweep!
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT)
    wandb.agent(sweep_id, function=main, count=args.count)
