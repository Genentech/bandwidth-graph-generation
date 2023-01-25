from typing import Callable

from tqdm.auto import tqdm
import torch
from torch.nn.utils.rnn import pack_sequence, PackedSequence
from torch_geometric.data import Data
from torch.utils.data import Dataset
from graph_gen.data.data_utils import (
    add_complete_edge_set, set_seeds,
    add_edge_set_to_data,
)
from graph_gen.data.orderings import OrderedGraph
from graph_gen.data.bandwidth import bw_edges, BandFlatten
from graph_gen.models.positional_encoding import PositionalEncoding


class GraphAEDataset(Dataset):
    def __init__(
        self, ordered_graphs: list[OrderedGraph],
        bw: int = None, pe_dim: int = 16,
        edge_augmentation: str = None,
        empirical_bandwidth: bool = True,
    ):
        """

        :param ordered_graphs:
        :param edge_augmentation:
        :param bw: bandwidth of output graph.
            If None and empirical_bw is False, bw = n - 1, so no restriction
            If None and empirical_bw is True, bw = phi(G)
        :param pe_dim:
        :param empirical_bandwidth:
            whether to use max bandwidth or each graph's bandwidth for output edges
        """
        super().__init__()
        self.ordered_graphs = ordered_graphs
        self.pe_dim = pe_dim
        self.bw = bw
        self.empirical_bw = empirical_bandwidth
        self.edge_augmentation: Callable[[Data, int], Data] = {
            "complete": add_complete_edge_set,
            None: lambda data, _: data,
            "none": lambda data, _: data,
        }[edge_augmentation]
        self.ordered_graphs = ordered_graphs
        self.data = self.build_data()

    def build_datum(self, ordered_graph: OrderedGraph) -> Data:
        data = ordered_graph.to_data()
        n = data.num_nodes
        # pe = SinusoidalPositionEmbeddings(self.pe_dim)(torch.linspace(0, 1, n))
        pe = PositionalEncoding(self.pe_dim, max_len=n)()
        data = Data(x=pe, edge_index=data.edge_index, num_nodes=n)
        if self.empirical_bw:
            bw = ordered_graph.bw
        else:
            bw = self.bw or n - 1  # n - 1 case gives no restriction
        data = add_edge_set_to_data(
            data, set(bw_edges(data.num_nodes, bw)), edge_type=1,
        )
        # potential random action
        set_seeds(ordered_graph.seed)
        data = self.edge_augmentation(data, 2)
        return data

    def build_data(self) -> list[Data]:
        return list(map(
            self.build_datum,
            tqdm(self.ordered_graphs, desc="Converting Graphs -> Data"),
        ))

    def __getitem__(self, idx: int) -> Data:
        return self.data[idx]

    def __len__(self) -> int:
        return len(self.data)


class GraphDiffusionDataset(GraphAEDataset):
    def __init__(
        self, ordered_graphs: list[OrderedGraph],
        betas: torch.Tensor,
        bw: int = None, pe_dim: int = 16,
        edge_augmentation: str = None,
        empirical_bandwidth: bool = True,
    ):
        """

        :param ordered_graphs:
        :param edge_augmentation:
        :param bw: bandwidth of output graph
        :param pe_dim:
        :param empirical_bandwidth:
            whether to use max bandwidth or each graph's bandwidth for output edges
        """
        super().__init__(
            ordered_graphs, bw=bw, pe_dim=pe_dim,
            edge_augmentation=edge_augmentation,
            empirical_bandwidth=empirical_bandwidth,
        )
        self.betas = betas
        alphas = 1 - self.betas
        self.cumprod_alphas = torch.cumprod(alphas, dim=0)

    def build_datum(self, ordered_graph: OrderedGraph) -> Data:
        """
        Makes sure edges are interleaved (i, j), (j, i), (k, l), (l, k), ...
        """
        datum = super().build_datum(ordered_graph)
        new_edge_index = []
        new_edge_type = []
        for (i, j), edge_type in zip(datum.edge_index.T, datum.edge_type):
            if j > i:
                continue
            new_edge_index.append([i, j])
            new_edge_index.append([j, i])
            new_edge_type += [edge_type, edge_type]
        new_datum = Data(
            x=datum.x,
            edge_type=torch.tensor(new_edge_type, dtype=torch.int),
            edge_index=torch.tensor(new_edge_index, dtype=torch.long).T
        )
        return new_datum

    def add_noise(self, data: Data, time_index: int = None) -> Data:
        time_index = (
            torch.randint(0, len(self.cumprod_alphas), (1,)) if time_index is None
            else torch.tensor([time_index], dtype=torch.long)
        )
        time_fraction = time_index / len(self.betas)
        cumprod_alpha = self.cumprod_alphas[time_index]
        mean_scaler = torch.sqrt(cumprod_alpha)
        noise_scaler = torch.sqrt(1 - cumprod_alpha)
        # noise edge attrs
        real_edge_types = data.edge_type[data.edge_type < 2]
        real_edge_attr = torch.empty_like(real_edge_types, dtype=torch.float32)
        real_edge_attr[real_edge_types == 0] = 1  # in original graph
        real_edge_attr[real_edge_types == 1] = -1  # in bandwidth graph
        n_real_edges = real_edge_attr.shape[0] // 2
        attr_noise = torch.randn(n_real_edges)
        attr_noise_tiled = torch.repeat_interleave(attr_noise, 2)
        noised_edge_attr = noise_scaler * attr_noise_tiled + mean_scaler * real_edge_attr
        #  final_edge_attr should:
        #  * have shape: edge_index.shape[1], 3
        final_edge_attr = torch.zeros(data.edge_index.shape[1], 3)
        #  * final_edge_attr[:, 0] should be noise for real edges and 0 for virtual edges
        final_edge_attr[:attr_noise_tiled.shape[0], 0] = noised_edge_attr
        #  * final_edge_attr[:, 1:] should be one-hot encoding of if real or virtual edges
        final_edge_attr[data.edge_type <= 1, 1] = 1
        final_edge_attr[data.edge_type == 2, 2] = 1
        return Data(
            x=data.x, edge_index=data.edge_index,
            edge_attr_noise=attr_noise_tiled,  # this is what we will try to predict
            edge_attr=final_edge_attr,  # this is the noisy input to predict noise from
            noise_scale=noise_scaler,
            time_index=time_index,
            time_fraction=time_fraction,
            edge_type=data.edge_type,
        )

    def __getitem__(self, idx: int) -> Data:
        data = super().__getitem__(idx)
        return self.add_noise(data)


class GraphRNNDataset(Dataset):
    def __init__(
        self, ordered_graphs: list[OrderedGraph],
        bw: int,
    ):
        super().__init__()
        self.band_flatten = BandFlatten(bw=bw)
        print("Flattening graphs...")
        self.band_flat_adjs = self.flatten_graphs(ordered_graphs)

    def flatten_graphs(
        self, ordered_graphs: list[OrderedGraph],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        band_flat_adjs = []
        for ordered_graph in tqdm(ordered_graphs):
            A = ordered_graph.to_adjacency()
            flat_adj = self.band_flatten.flatten(A)
            inp = torch.zeros(
                (flat_adj.shape[0] + 1, flat_adj.shape[1] + 1), dtype=torch.float32,
            )
            out = torch.zeros(
                (flat_adj.shape[0] + 1, flat_adj.shape[1] + 1), dtype=torch.float32,
            )
            inp[1:, 1:] = flat_adj
            inp[0, 0] = 1  # add start token
            out[:-1, 1:] = flat_adj
            out[-1, 0] = 1  # add stop token
            band_flat_adjs.append((inp, out))
        return band_flat_adjs

    def __len__(self) -> int:
        return len(self.band_flat_adjs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.band_flat_adjs[idx]

    @staticmethod
    def collate_fn(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[PackedSequence, PackedSequence]:
        inps, outs = [], []
        for inp, out in batch:
            inps.append(inp)
            outs.append(out)
        return pack_sequence(inps, enforce_sorted=False), pack_sequence(outs, enforce_sorted=False)