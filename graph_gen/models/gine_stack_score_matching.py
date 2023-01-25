import torch
from torch import nn
import torch_geometric as pyg
from torch_geometric.nn import MetaLayer
import torch.nn.functional as F
from tqdm.auto import tqdm
import networkx as nx
from torch_geometric.utils import to_dense_adj
from graph_gen.analysis.mmd import evaluate_sampled_graphs
from graph_gen.data.data_utils import clean_graph_from_adj

from graph_gen.models.gine_stack import GINEStack
from graph_gen.models.model_utils import BaseLightningModule, SinusoidalPositionEmbeddings


class EdgePredictor(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        output_dim: int = 1,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, output_dim),
        )

    def forward(
        self, src: torch.Tensor, dest: torch.Tensor, edge_attr: torch.Tensor,
        u, batch,
    ) -> torch.Tensor:
        x1 = torch.cat([src, dest, edge_attr], dim=-1)
        x2 = torch.cat([dest, src, edge_attr], dim=-1)
        return self.mlp(x1) + self.mlp(x2)


class GINEStackDiffusion(BaseLightningModule):
    def __init__(
        self,
        node_features: int,
        hidden_dim: int,
        epochs: int,
        n_layers: int = 3,
        lr: float = 1e-3,
        wd: float = 1e-2,
    ):
        super().__init__(epochs, lr, wd)
        super().save_hyperparameters()
        self.time_MLP = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.node_MLP = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.GELU(),
        )
        self.gine_stack = GINEStack(
            node_features=hidden_dim, edge_features=3,
            hidden=hidden_dim, n_layers=n_layers,
        )
        self.pre_edge_kernel = nn.Sequential(
            nn.Linear(hidden_dim * (n_layers + 3), hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
        )
        self.edge_kernel = MetaLayer(
            edge_model=EdgePredictor(node_dim=hidden_dim + node_features, edge_dim=hidden_dim),
        )

    def forward(self, batch: pyg.data.Data) -> torch.Tensor:
        """

        :param batch: batch from nx_dataset.GraphDiffusionDataset
        :return: predicted noise
        """
        # time_embed = self.time_MLP(batch.noise_scale[batch.batch])
        time_embed = self.time_MLP(batch.time_fraction[batch.batch])
        pe_embed = self.node_MLP(batch.x)
        x = pe_embed + time_embed
        gine_stack_out = self.gine_stack(
            x=x, edge_attr=batch.edge_attr,
            edge_index=batch.edge_index,
        )
        x = torch.cat([gine_stack_out, pe_embed, time_embed], dim=-1)
        x = self.pre_edge_kernel(x)
        x = torch.cat([x, batch.x], dim=-1)
        real_edge = batch.edge_type < 2
        edge_embed = self.edge_embed(batch.edge_attr[real_edge, :1])
        noise_pred = self.edge_kernel(
            x=x, edge_index=batch.edge_index[:, real_edge],
            edge_attr=edge_embed,
            batch=batch.batch,
        )[1]
        return noise_pred.squeeze()

    def get_loss(self, batch, prefix: str) -> torch.Tensor:
        noise_pred = self(batch)
        mse = F.mse_loss(noise_pred, batch.edge_attr_noise)
        self.log(f"{prefix}/mse", mse, prog_bar=True)
        self.log(f"{prefix}/loss", mse, prog_bar=True)
        return mse


def evaluate_diffusion_sampling(
    model: GINEStackDiffusion,
    betas: torch.Tensor,
    skeleton_graphs: pyg.data.Data,
    real_graphs: list[nx.Graph],
) -> dict[str, float]:
    sampler = GINEStackDiffusionSampler(model, betas)
    sampled_edge_attr_list = sampler.p_sample_loop(skeleton_graphs)
    edge_index = skeleton_graphs.edge_index[:, skeleton_graphs.edge_type < 2]
    adjs = to_dense_adj(
        edge_index=edge_index,
        edge_attr=sampled_edge_attr_list[-1],  # take the first time index
        batch=skeleton_graphs.batch,
    )
    sampled_graphs = [
        clean_graph_from_adj(adj.squeeze().cpu().numpy() > 0.0)
        for adj in adjs
    ]
    sampled_graphs = [graph for graph in sampled_graphs if len(graph) > 0]
    if len(sampled_graphs) == 0:
        return {}
    return evaluate_sampled_graphs(sampled_graphs, real_graphs)


class GINEStackDiffusionSampler:
    def __init__(
        self,
        model: GINEStackDiffusion,
        betas: torch.Tensor,
    ):
        self.model = model
        self.betas = betas
        alphas = 1 - self.betas
        self.cumprod_alphas = torch.cumprod(alphas, dim=0)

    @torch.no_grad()
    def p_sample(
        self,
        batch: pyg.data.Data,
        time_index: int,
    ):
        # prep noise
        beta = self.betas[time_index]
        alpha = 1 - beta
        score_scaler = (1 - alpha) / torch.sqrt(1 - self.cumprod_alphas[time_index])
        noise_scaler = torch.sqrt(beta)
        # noise_scale is used by model as time conditioning
        # cumprod_alpha = self.cumprod_alphas[time_index]
        # model_noise_scaler = torch.torch.sqrt(1 - cumprod_alpha)
        batch_size = len(torch.unique(batch.batch))
        # batch.noise_scale = model_noise_scaler.repeat(batch_size)
        batch.time_fraction = (torch.tensor(time_index, device=self.model.device) / len(self.betas)).repeat(batch_size)
        # get the mean from the re-parameterized model
        edge_score = self.model(batch)
        real_edge_idx = batch.edge_type < 2
        noisy_edges = batch.edge_attr[real_edge_idx, 0]
        edge_model_mean = 1 / torch.sqrt(1 / alpha) * (
            noisy_edges - edge_score * score_scaler
        )
        # inject noise
        # edge noise uses assumption that edges are ordered [edge_1, backward edge_1, ...]
        n_edges = len(noisy_edges) // 2
        edge_noise = torch.randn_like(edge_model_mean[:n_edges])
        edge_noise_tiled = (
            torch.repeat_interleave(edge_noise, 2).unsqueeze(-1) if time_index > 0
            else torch.zeros_like(noisy_edges)
        ).squeeze()
        # combine noise and mean
        new_edge_attr = noise_scaler * edge_noise_tiled + edge_model_mean
        return new_edge_attr

    @torch.no_grad()
    def p_sample_loop(
        self, batch: pyg.data.Data,
    ) -> list[torch.Tensor]:
        # prepare initial random noise
        real_edge_idx = batch.edge_type < 2
        n_edges = int(real_edge_idx.sum().item()) // 2
        initial_noise = torch.randn_like(
            batch.edge_attr[real_edge_idx, 0][:n_edges],
        ).repeat_interleave(2)
        batch.edge_attr[real_edge_idx, 0] = initial_noise
        edge_attr_list = [initial_noise]
        # run backward model
        n_steps = len(self.betas)
        for time_index in tqdm(reversed(range(n_steps)), total=n_steps):
            edge_attr = self.p_sample(batch, time_index)
            edge_attr_list.append(edge_attr)
            batch.edge_attr[real_edge_idx, 0] = edge_attr
        return edge_attr_list
