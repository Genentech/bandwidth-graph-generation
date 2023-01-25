import networkx as nx
import numpy as np
from tqdm.auto import tqdm
import torch
import pandas as pd
from torch import nn
from torchmetrics.functional.classification import (
    binary_average_precision, binary_recall, binary_precision,
)
from graph_gen.models.model_utils import (
    apply_func_to_packed_sequence,
)
from torch.nn.utils.rnn import (
    PackedSequence, pad_packed_sequence,
)

from graph_gen.models.model_utils import BaseLightningModule
from graph_gen.data.bandwidth import BandFlatten, graph_from_bandflat
from graph_gen.analysis.mmd import evaluate_sampled_graphs


class RowMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ):
        super().__init__()
        self.in_layer = nn.Linear(input_dim, hidden_dim)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.out_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_layer(x)
        # conditional to handle packed sequences
        if x.ndim == 3:
            x = torch.transpose(x, 2, 1)
            x = self.bn(x)
            x = torch.transpose(x, 2, 1)
        elif x.ndim == 2:
            x = self.bn(x)
        x = self.relu(x)
        return self.out_layer(x)


class GraphRNNSimpleModule(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        bw: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.row_in_mlp = RowMLP(bw + 1, hidden_size, hidden_size)
        self.graph_state_rnn = nn.GRU(
            input_size=hidden_size,
            batch_first=True,
            num_layers=4,
            hidden_size=hidden_size,
        )
        self.row_out_mlp = RowMLP(hidden_size, hidden_size, bw + 1)
        self.bw = bw

    def forward(self, bw_flat_adj: PackedSequence) -> PackedSequence:
        """
        :param bw_flat_adj: band flattened adjacency matrix
        :return: edge probability logits
        """
        # graph state
        gru_in = apply_func_to_packed_sequence(
            self.row_in_mlp, bw_flat_adj,
        )
        gru_hidden = self.graph_state_rnn(gru_in)[0]  # batch size x seq len x hidden
        edge_out = apply_func_to_packed_sequence(
            self.row_out_mlp, gru_hidden,
        )
        return edge_out

    def unpacked_forward(self, bw_flat_adj: torch.Tensor) -> torch.Tensor:
        gru_in = self.row_in_mlp(bw_flat_adj)
        gru_hidden = self.graph_state_rnn(gru_in)[0]
        return self.row_out_mlp(gru_hidden)


class GraphRNNSimple(BaseLightningModule):
    def __init__(
        self,
        hidden_dim: int,
        epochs: int,
        bw: int,
        lr: float = 1e-3,
        wd: float = 1e-2,
    ):
        super().__init__(epochs, lr, wd)
        # model
        self.bw = bw
        self.graph_rnn = GraphRNNSimpleModule(hidden_size=hidden_dim, bw=self.bw)
        # optimizer
        self.loss = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def forward(self, band_flat_adj: PackedSequence) -> torch.Tensor:
        return self.graph_rnn(band_flat_adj)

    def get_loss(
        self,
        batch,
        prefix: str,
    ):
        packed_input, packed_output = batch
        pred = self(packed_input)
        loss = self.loss(pred.data, packed_output.data)
        # logging
        sigmoid_adj_pred = torch.sigmoid(pred.data)
        self.log(f"{prefix}/loss", loss, prog_bar=True)
        self.log(
            f"{prefix}/auprc",
            binary_average_precision(sigmoid_adj_pred, packed_output.data),
            prog_bar=True,
        )
        self.log(
            f"{prefix}/precision",
            binary_precision(sigmoid_adj_pred, packed_output.data),
            prog_bar=True,
        )
        self.log(
            f"{prefix}/recall",
            binary_recall(sigmoid_adj_pred, packed_output.data),
            prog_bar=True,
        )
        return loss

    @torch.no_grad()
    def test_step(self, batch, _=None):
        packed_input, packed_output = batch
        pred = self(packed_input)
        sigmoid_adj_pred = torch.sigmoid(pred.data).squeeze().cpu()
        actual = packed_output.data.squeeze().cpu()
        # batchwise AUPRC
        batch_auprc = binary_average_precision(sigmoid_adj_pred, actual)
        # per graph AUPRC
        packed_edge_probs = apply_func_to_packed_sequence(torch.sigmoid, pred)
        unpacked_edge_probs, _ = pad_packed_sequence(packed_edge_probs, batch_first=True)
        unpacked_edge_actual, _ = pad_packed_sequence(packed_output, batch_first=True)
        per_graph_auprc = torch.tensor([
            binary_average_precision(prob, act)
            for prob, act in zip(unpacked_edge_probs, unpacked_edge_actual)
        ]).mean().item()
        # return
        return {
            "edge_prob_pred": sigmoid_adj_pred, "edges": actual,
            "per_batch_auprc": batch_auprc,
            "mean_per_graph_auprc": per_graph_auprc,
        }

    def test_epoch_end(self, outputs):
        # whole dataset auprc
        probs = torch.cat([batch["edge_prob_pred"] for batch in outputs], dim=0)
        actual = torch.cat([batch["edges"] for batch in outputs], dim=0)
        self.log("test/auprc", binary_average_precision(probs, actual))
        # per graph auprc
        self.log(
            "test/per_graph_auprc",
            np.mean([batch["mean_per_graph_auprc"] for batch in outputs]),
        )
        # per batch auprc
        self.log(
            "test/per_batch_auprc",
            np.mean([batch["per_batch_auprc"] for batch in outputs]),
        )


    @torch.no_grad()
    def sample(self, n: int, steps: int, temperature: float = 1) -> torch.Tensor:
        x = torch.zeros((n, 1, self.bw + 1), device=self.device)
        x[:, 0, 0] = 1  # start token
        for i in range(steps):
            next_x = self.graph_rnn.unpacked_forward(x)
            next_p = torch.sigmoid(next_x[:, -1] * 1 / temperature)
            sampled = torch.bernoulli(next_p).unsqueeze(1)
            x = torch.cat([x, sampled], dim=1)
        return x

    def evaluate_temperature(
        self, temp: float, real_graphs: list[nx.Graph],
        max_graph_len: int,
    ) -> dict[str, float]:
        samples = self.sample(len(real_graphs), steps=max_graph_len, temperature=temp)
        bf = BandFlatten(self.bw)
        sampled_graphs = []
        for sample in samples:
            sampled_no_start_token = sample[1:]
            stop_idx = sampled_no_start_token[:, 0].argmax()
            sampled_clean = sampled_no_start_token[:stop_idx, 1:].cpu()
            try:
                graph = graph_from_bandflat(sampled_clean, bf)
            except ValueError:
                continue
            sampled_graphs.append(graph)
        if not sampled_graphs:
            return {}
        return evaluate_sampled_graphs(sampled_graphs, real_graphs)

    def find_best_temperature(
        self, real_graphs: list[nx.Graph], max_graph_len: int,
        temperatures: list[float] = None,
    ) -> tuple[pd.DataFrame, float, float]:
        temperatures = temperatures or np.arange(0.1, 2.1, 0.1)
        results = []
        for temp in tqdm(temperatures, desc="Sampling at different temps"):
            result = self.evaluate_temperature(temp, real_graphs, max_graph_len)
            result["temperature"] = temp
            results.append(result)
        df = pd.DataFrame(results)
        gb = df.groupby("temperature").mean().mean(axis=1)
        best_temp = gb.idxmin()
        best_mean_mmd = gb.min()
        return df, best_temp, best_mean_mmd

    def on_train_end(self) -> None:
        print("DONE!")
