import numpy as np
import networkx as nx
import torch


class BandFlatten:
    def __init__(self, bw: int, forward: bool = False):
        self.bw = bw
        self.forward = forward

    def flatten(self, A: torch.Tensor) -> torch.Tensor:
        return (
            self.flatten_forward(A, self.bw) if self.forward
            else self.flatten_backward(A, self.bw)
        )

    def unflatten(self, band_flat_A: torch.Tensor) -> torch.Tensor:
        return (
            self.unflatten_forward(band_flat_A) if self.forward
            else self.unflatten_backward(band_flat_A)
        )

    @staticmethod
    def flatten_forward(A: torch.Tensor, bw: int) -> torch.Tensor:
        n = A.shape[0]
        out = torch.zeros((n, bw) + A.shape[2:], dtype=A.dtype, device=A.device)
        for i in range(n):
            append_len = min(bw, n - i - 1)
            if append_len > 0:
                out[i, :append_len] = A[i, i + 1: i + 1 + append_len]
        return out

    @staticmethod
    def unflatten_forward(band_flat_A: torch.Tensor) -> torch.Tensor:
        n, bw = band_flat_A.shape[:2]
        out = torch.zeros((n, n) + band_flat_A.shape[2:], dtype=band_flat_A.dtype, device=band_flat_A.device)
        for i in range(n):
            append_len = min(bw, n - i - 1)
            if append_len > 0:
                out[i, i + 1: i + 1 + append_len] = band_flat_A[i, :append_len]
        out = out + out.T
        return out

    @staticmethod
    def flatten_backward(A: torch.Tensor, bw: int) -> torch.Tensor:
        n = A.shape[0]
        out = torch.zeros((n, bw) + A.shape[2:], dtype=A.dtype, device=A.device)
        for i in range(n):
            append_len = min(bw, i)
            if append_len > 0:
                out[i, :append_len] = torch.flip(A[i, i - append_len: i], (0,))
        return out

    @staticmethod
    def unflatten_backward(band_flat_A: torch.Tensor) -> torch.Tensor:
        n, bw = band_flat_A.shape[:2]
        out = torch.zeros((n, n) + band_flat_A.shape[2:], dtype=band_flat_A.dtype, device=band_flat_A.device)
        for i in range(n):
            append_len = min(bw, i)
            if append_len > 0:
                out[i, i - append_len: i] = torch.flip(band_flat_A[i, :append_len], (0,))
        out = out + out.T
        return out


def bw_from_adj(A: np.ndarray) -> int:
    """calculate bandwidth from adjacency matrix"""
    band_sizes = np.arange(A.shape[0]) - A.argmax(axis=1)
    return band_sizes.max()


def graph_from_bandflat(flat_adj: torch.Tensor, bf) -> nx.Graph:
    G = nx.from_numpy_array(bf.unflatten(flat_adj).numpy())
    G.remove_nodes_from(list(nx.isolates(G)))
    G = G.subgraph(max(nx.connected_components(G), key=len))
    G = nx.convert_node_labels_to_integers(G)
    return G


def bw_edges(n: int, bw: int) -> list[tuple[int, int]]:
    edges = []
    for i in range(n):
        for j in range(bw):
            idx_1 = i
            idx_2 = i + j + 1
            if idx_2 >= n:
                continue
            edges.append((idx_1, idx_2))
            edges.append((idx_2, idx_1))
    return edges