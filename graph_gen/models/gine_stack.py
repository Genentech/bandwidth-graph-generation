import networkx as nx
from tqdm.auto import tqdm
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import MetaLayer, GINEConv, GINConv
import torch_geometric as pyg
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
from torchmetrics.functional.classification import (
    binary_average_precision, binary_recall, binary_precision,
)

from graph_gen.models.model_utils import BaseLightningModule
from graph_gen.analysis.mmd import evaluate_sampled_graphs
from graph_gen.data.data_utils import clean_graph_from_adj


class GINEMLP(nn.Module):
    def __init__(
        self, hidden: int,
    ):
        super().__init__()
        self.hidden = hidden
        self.in_channels = hidden  # used by GINEConv
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 2 * hidden),
            nn.BatchNorm1d(2 * hidden),
            nn.GELU(),
            nn.Linear(2 * hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class GINEStack(nn.Module):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        hidden: int,
        n_layers: int = 3,
    ):
        super().__init__()
        self.node_embed = nn.Sequential(
            nn.Linear(node_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.gine_convs = nn.ModuleList([
            GINEConv(
                GINEMLP(hidden),
                train_eps=True,
                edge_dim=edge_features,
            )
            for _ in range(n_layers)
        ])

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        x = self.node_embed(x)
        intermediates = [x]
        for i, layer in enumerate(self.gine_convs):
            # print(i, x.abs().max().item())
            x = layer(x, edge_index=edge_index, edge_attr=edge_attr)
            intermediates.append(x)
        concat_x = torch.cat(intermediates, dim=-1)
        return concat_x


class VAEBottleneck(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden: int,
    ):
        super().__init__()
        self.mu = nn.Linear(input_dim, hidden)
        self.log_sigma = nn.Linear(input_dim, hidden)
        nn.init.zeros_(self.log_sigma.weight)
        nn.init.zeros_(self.log_sigma.bias)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mu(z), self.log_sigma(z)

    def forward_noised(
        self, z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param z:
        :return: sampled z, sigma, kl divergence
        """
        mu, log_sigma = self(z)
        sigma = torch.exp(log_sigma)
        kl_full_size = sigma**2 + mu**2 - log_sigma - 1/2
        kl = kl_full_size.sum()
        return mu + sigma * torch.randn_like(mu), sigma, kl


class EdgeLogits(nn.Module):
    def __init__(
        self,
        node_dim: int,
        output_dim: int = 1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, output_dim),
        )

    def forward(
        self, src: torch.Tensor, dest: torch.Tensor, edge_attr: torch.Tensor,
        u, batch,
    ) -> torch.Tensor:
        x1 = torch.cat([src, dest], dim=-1)
        x2 = torch.cat([dest, src], dim=-1)
        return self.mlp(x1) + self.mlp(x2)


############################################################
##                     GINE VAE code                      ##
############################################################


class GINEDecoder(nn.Module):
    def __init__(
        self,
        node_features: int,
        hidden: int,
        n_layers: int = 3,
    ):
        super().__init__()
        # this will embed the node features to add to the encoder outputs
        self.node_embed = nn.Sequential(
            nn.Linear(node_features, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        edge_features = 2
        self.gine_stack = GINEStack(
            node_features=hidden, edge_features=edge_features,
            n_layers=n_layers, hidden=hidden,
        )
        self.pre_final_edge_kernel = nn.Sequential(
            nn.Linear(hidden * (n_layers + 2), hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
        )
        self.edge_kernel = MetaLayer(edge_model=EdgeLogits(node_dim=hidden + node_features))

    def forward(
        self, x: torch.Tensor, z: torch.Tensor,
        edge_index: torch.Tensor, is_virtual_edge: torch.Tensor,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        """

        :param x: usually positional encoding
        :param z: output of decoder + bottleneck
        :param edge_index:
        :param is_virtual_edge: bool vector for whether edge_index refer to virtual edge or potential output edge
        :param batch: batch index
        :return:
        """
        # build edge attributes to one-hot encode `is_virtual_edge`
        edge_attr = torch.zeros((edge_index.shape[1], 2), dtype=x.dtype, device=x.device, requires_grad=False)
        real_edge = ~is_virtual_edge
        edge_attr[real_edge, 0] = 1
        edge_attr[is_virtual_edge, 1] = 1
        og_x = x
        # get embedding of positional encoding
        pe_embed = self.node_embed(x)
        x = pe_embed + z
        x = self.gine_stack(
            x, edge_index, edge_attr,
        )
        x = torch.cat([pe_embed, x], dim=-1)
        x = self.pre_final_edge_kernel(x)
        restricted_edge_index = edge_index[:, real_edge]  # restrict edge kernel to non-virtual edges
        x = torch.cat([x, og_x], dim=-1)
        edge_logits = self.edge_kernel(x, edge_index=restricted_edge_index, batch=batch)[1]
        return edge_logits


class GINEStackVAE(BaseLightningModule):
    def __init__(
        self,
        node_features: int,
        hidden_dim: int,
        kl_weight: float,
        epochs: int,
        n_layers: int = 3,
        lr: float = 1e-3,
        wd: float = 1e-2,
    ):
        super().__init__(epochs, lr, wd)
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        # model
        self.encoder = GINEStack(
            node_features=node_features, edge_features=3,
            hidden=hidden_dim, n_layers=n_layers,
        )
        self.kl_weight = kl_weight
        self.bottleneck = VAEBottleneck(
            input_dim=(n_layers + 1) * hidden_dim,
            hidden=hidden_dim,
        )
        self.decoder = GINEDecoder(
            node_features=node_features, hidden=hidden_dim,
            n_layers=n_layers,
        )
        self.loss = nn.BCEWithLogitsLoss()

    def get_loss(self, batch: pyg.data.Data, prefix: str) -> torch.Tensor:
        # 3 edge types are: in original graph, in low BW graph, virtual edge
        edge_attr = F.one_hot(batch.edge_type.to(torch.long), num_classes=3).to(batch.x.dtype)
        # encoder
        original_edge = batch.edge_type == 0
        encoder_out = self.encoder(
            x=batch.x, edge_attr=edge_attr[original_edge], edge_index=batch.edge_index[:, original_edge],
        )
        # bottleneck
        z, sigma, kl = self.bottleneck.forward_noised(encoder_out)
        self.log(f"{prefix}/kl", kl, prog_bar=True)
        self.log(f"{prefix}/sigma_mean", sigma.mean(), prog_bar=True)
        output_edge = batch.edge_type <= 1
        weighted_kl = kl * self.kl_weight / output_edge.sum()
        self.log(f"{prefix}/weighted_kl", weighted_kl, prog_bar=True)
        loss = weighted_kl
        # decoder
        is_virtual_edge = batch.edge_type == 2
        edge_logits = self.decoder(
            x=batch.x, z=z, is_virtual_edge=is_virtual_edge, batch=batch.batch,
            edge_index=batch.edge_index,
        ).squeeze()
        # loss computation
        target = (batch.edge_type[output_edge].squeeze() == 0).to(torch.float32)
        bce = self.loss(edge_logits, target)
        loss = loss + bce
        self.log(f"{prefix}/loss", loss, prog_bar=True)
        self.log(f"{prefix}/bce", bce, prog_bar=True)
        # logging
        edge_probs = torch.sigmoid(edge_logits)
        self.log(
            f"{prefix}/auprc",
            binary_average_precision(edge_probs, target),
            prog_bar=True,
        )
        self.log(
            f"{prefix}/precision",
            binary_precision(edge_probs, target),
            prog_bar=True,
        )
        self.log(
            f"{prefix}/recall",
            binary_recall(edge_probs, target),
            prog_bar=True,
        )
        return loss

    def forward(self, batch: pyg.data.Data) -> tuple[torch.Tensor, torch.Tensor]:
        # 3 edge types are: in original graph, in low BW graph, virtual edge
        edge_attr = F.one_hot(batch.edge_type.to(torch.long), num_classes=3).to(batch.x.dtype)
        # encoder
        original_edge = batch.edge_type == 0
        encoder_out = self.encoder(
            x=batch.x, edge_attr=edge_attr[original_edge], edge_index=batch.edge_index[:, original_edge],
        )
        # bottleneck
        z, sigma = self.bottleneck(encoder_out)
        # decoder
        is_virtual_edge = batch.edge_type == 2
        edge_logits = self.decoder(
            x=batch.x, z=z, is_virtual_edge=is_virtual_edge, batch=batch.batch,
            edge_index=batch.edge_index,
        ).squeeze()
        return z, edge_logits

    @torch.no_grad()
    def sample(self, batch: pyg.data.Data, sigma: float = 1) -> torch.Tensor:
        z_shape = (batch.x.shape[0], self.hidden_dim)
        z = torch.randn(z_shape, dtype=torch.float32, device=self.device) * sigma
        is_virtual_edge = batch.edge_type == 2
        edge_logits = self.decoder(
            x=batch.x, z=z, is_virtual_edge=is_virtual_edge, batch=batch.batch,
            edge_index=batch.edge_index,
        ).squeeze()
        return edge_logits

    @torch.no_grad()
    def evaluate_sigma(
        self, sigma: float, skeleton_graphs: pyg.data.Data, real_graphs: list[nx.Graph],
    ) -> dict[str, float]:
        samples = self.sample(skeleton_graphs, sigma=sigma)
        non_virtual_idx = skeleton_graphs.edge_type <= 1
        edge_probs = torch.sigmoid(samples)
        adjs = to_dense_adj(
            edge_index=skeleton_graphs.edge_index[:, non_virtual_idx],
            edge_attr=edge_probs, batch=skeleton_graphs.batch,
        )
        sampled_graphs = [
            clean_graph_from_adj(adj.squeeze().cpu().numpy() > 0.5)
            for adj in adjs
        ]
        sampled_graphs = [graph for graph in sampled_graphs if len(graph) > 0]
        if len(sampled_graphs) == 0:
            result = {}
        else:
            result = evaluate_sampled_graphs(sampled_graphs, real_graphs)
        result["sigma"] = sigma
        return result

    def find_best_sigma(
        self,
        skeleton_graphs: pyg.data.Data,
        real_graphs: list[nx.Graph],
        sigmas: list[float] = None,
    ) -> pd.DataFrame:
        sigmas = sigmas or [0.1, 0.5, 1, 1.5, 2]
        results = []
        for sigma in tqdm(
            sigmas, desc="sampling at different variances",
        ):
            sigma_results = self.evaluate_sigma(sigma, skeleton_graphs, real_graphs)
            results.append(sigma_results)
        return pd.DataFrame(results)
