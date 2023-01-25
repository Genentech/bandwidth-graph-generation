import os

import torch_geometric as pyg
import networkx as nx
from torch_geometric.utils import to_networkx


def get_DD_data(
    root: str = os.path.join(os.path.expanduser("~"), "scratch"),
) -> list[nx.Graph]:
    dd_data = pyg.datasets.TUDataset(root, name="DD")
    nx_graphs = [
        to_networkx(datum, to_undirected=True)
        for datum in dd_data
        if 100 <= datum.num_nodes <= 500
    ]
    print("initial length", len(dd_data), "-> filtered length", len(nx_graphs))
    return nx_graphs


def get_PROTEINS_data(
    root: str = os.path.join(os.path.expanduser("~"), "scratch"),
) -> list[nx.Graph]:
    dd_data = pyg.datasets.TUDataset(root, name="PROTEINS")
    nx_graphs = [
        to_networkx(datum, to_undirected=True)
        for datum in dd_data
        if 10 <= datum.num_nodes <= 125
    ]
    nx_graphs = [graph for graph in nx_graphs if nx.number_connected_components(graph) == 1]
    print("initial length", len(dd_data), "-> filtered length", len(nx_graphs))
    return nx_graphs


def get_ENZYMES_data(
    root: str = os.path.join(os.path.expanduser("~"), "scratch"),
) -> list[nx.Graph]:
    dd_data = pyg.datasets.TUDataset(root, name="ENZYMES")
    nx_graphs = [
        to_networkx(datum, to_undirected=True)
        for datum in dd_data
        if 10 <= datum.num_nodes <= 125
    ]
    nx_graphs = [graph for graph in nx_graphs if nx.number_connected_components(graph) == 1]
    print("initial length", len(dd_data), "-> filtered length", len(nx_graphs))
    return nx_graphs
