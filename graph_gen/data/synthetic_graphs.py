import scipy
import numpy as np
import networkx as nx


def create_community_graph(
    num_nodes=np.arange(12, 21), num_comms=2,
    intra_comm_edge_prob=0.3, inter_comm_edge_frac=0.05
):
    """
    Creates a community graph following this paper:
    https://arxiv.org/abs/1802.08773
    The default values give the definition of a "community-small" graph in the
    above paper. Each community is a Erdos-Renyi graph, with a certain set
    number of edges connecting the communities sparsely (drawn uniformly).
    All nodes will be given a feature vector of all 1s.
    Arguments:
        `num_nodes`: number of nodes in the graph, or an array to sample from
        `num_comms`: number of communities to create
        `intra_comm_edge_prob`: probability of edge in Erdos-Renyi graph for
            each community
        `inter_comm_edge_frac`: number of edges to draw between each pair of
            communities, as a fraction of `num_nodes`; edges are drawn uniformly
            at random between communities
    Returns a NetworkX Graph with NumPy arrays as node attributes.
    """
    if type(num_nodes) is not int:
        num_nodes = np.random.choice(num_nodes)

    # Create communities
    exp_size = int(num_nodes / num_comms)
    comm_sizes = []
    total_size = 0
    g = nx.empty_graph()
    while total_size < num_nodes:
        size = min(exp_size, num_nodes - total_size)
        g = nx.disjoint_union(
            g, nx.erdos_renyi_graph(size, intra_comm_edge_prob)
        )
        comm_sizes.append(size)
        total_size += size

    # Link together communities
    node_inds = np.cumsum(comm_sizes)
    num_inter_edges = int(num_nodes * inter_comm_edge_frac)
    for i in range(num_comms):
        for j in range(i):
            i_nodes = np.arange(node_inds[i - 1] if i else 0, node_inds[i])
            j_nodes = np.arange(node_inds[j - 1] if j else 0, node_inds[j])
            for _ in range(num_inter_edges):
                g.add_edge(
                    np.random.choice(i_nodes), np.random.choice(j_nodes)
                )
    # largest connected component
    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()
    g = nx.convert_node_labels_to_integers(g)
    return g


def get_community_graphs(n: int = 1500) -> list[nx.Graph]:
    return [
        create_community_graph(
            num_nodes=np.arange(60, 161),
        ) for _ in range(n)
    ]


def create_planar_graph(num_nodes=64):
    """
    Creates a planar graph using the Delaunay triangulation algorithm.
    All nodes will be given a feature vector of all 1s.
    Arguments:
        `node_dim`: size of node feature vector
        `num_nodes`: number of nodes in the graph, or an array to sample from
    Returns a NetworkX Graph with NumPy arrays as node attributes.
    """
    if type(num_nodes) is not int:
        num_nodes = np.random.choice(num_nodes)

    # Sample points uniformly at random from unit square
    points = np.random.rand(num_nodes, 2)
    # Perform Delaunay triangulation
    tri = scipy.spatial.Delaunay(points)
    # Create graph and add edges from triangulation result
    g = nx.empty_graph(num_nodes)
    indptr, indices = tri.vertex_neighbor_vertices
    for i in range(num_nodes):
        for j in indices[indptr[i]:indptr[i + 1]]:
            g.add_edge(i, j)
    return nx.convert_node_labels_to_integers(g)


def get_planar_graphs(n: int = 1500) -> list[nx.Graph]:
    np.random.seed(0)
    return [
        create_planar_graph() for _ in range(n)
    ]


def create_sbm_graph(
    num_blocks_arr=np.arange(2, 6), block_size_arr=np.arange(20, 41),
    intra_block_edge_prob=0.3, inter_block_edge_prob=0.05
):
    """
    Creates a stochastic-block-model graph, where the number of blocks and size
    of blocks is drawn randomly.
    All nodes will be given a feature vector of all 1s.
    Arguments:
        `node_dim`: size of node feature vector
        `num_blocks_arr`: iterable containing possible numbers of blocks to have
            (selected uniformly)
        `block_size_arr`: iterable containing possible block sizes for each
            block (selected uniformly per block)
        `intra_block_edge_prob`: probability of edge within blocks
        `inter_block_edge_prob`: probability of edge between blocks
    Returns a NetworkX Graph with NumPy arrays as node attributes.
    """
    num_blocks = np.random.choice(num_blocks_arr)
    block_sizes = np.random.choice(block_size_arr, num_blocks, replace=True)

    # Create matrix of edge probabilities between blocks
    p = np.full((len(block_sizes), len(block_sizes)), inter_block_edge_prob)
    np.fill_diagonal(p, intra_block_edge_prob)

    # Create SBM graph
    g = nx.stochastic_block_model(block_sizes, p)

    # Delete these two attributes, or else conversion to PyTorch Geometric Data
    # object will fail
    del g.graph["partition"]
    del g.graph["name"]

    return nx.convert_node_labels_to_integers(g)


def get_sbm_graphs(n: int = 1500) -> list[nx.Graph]:
    np.random.seed(0)
    return [
        create_sbm_graph() for _ in range(n)
    ]


def get_2d_grid_graphs() -> list[nx.Graph]:
    widths = np.arange(10, 20 + 1)
    graphs = []
    for width in widths:
        for height in range(min(widths), width + 1):
            graph = nx.grid_2d_graph(width, height)
            graphs.append(nx.convert_node_labels_to_integers(graph))
    return graphs
