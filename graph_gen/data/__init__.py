from graph_gen.data.mol_utils import get_zinc_graphs, get_peptide_graphs
from graph_gen.data.protein_data import get_DD_data, get_PROTEINS_data, get_ENZYMES_data
from graph_gen.data.synthetic_graphs import (
    get_2d_grid_graphs, get_community_graphs, get_planar_graphs,
)


DATASETS = {  # name: (graphs, num_repetitions)
    "grid2d": (
        get_2d_grid_graphs, 5,
    ),
    "community2": (
        get_community_graphs, 1,
    ),
    "planar": (
        get_planar_graphs, 1,
    ),
    "SBM": (
        get_planar_graphs, 1,
    ),
    "zinc250k": (
        get_zinc_graphs, 1,
    ),
    "DD": (
        get_DD_data, 1,
    ),
    "peptides": (
        get_peptide_graphs, 1,
    ),
    "PROTEINS": (
        get_PROTEINS_data, 1,
    ),
    "ENZYMES": (
        get_ENZYMES_data, 1,
    )
}
