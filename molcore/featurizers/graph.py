import torch
from torch_geometric.data import Data
from molcore._molcore import PyMolGraph, mol_to_graph_arrays


def to_pyg_data(graph: PyMolGraph) -> Data:
    """
    Rust graph arrays → PyG Data. Zero-copy at every stage.
    node_feats: (N, 4) float32  — [atomic_num, is_aromatic, formal_charge, num_hs]
    edge_index: (2, E) int64    — COO, bidirectional
    edge_attr:  (E, 4) float32  — bond type one-hot
    """
    node_feats_np, edge_index_np, edge_feats_np = mol_to_graph_arrays(graph)
    return Data(
        x          = torch.from_numpy(node_feats_np),
        edge_index = torch.from_numpy(edge_index_np),
        edge_attr  = torch.from_numpy(edge_feats_np),
    )


def to_dgl_graph(graph: PyMolGraph):
    """DGL graph — requires dgl package."""
    import dgl
    node_feats_np, edge_index_np, edge_feats_np = mol_to_graph_arrays(graph)
    src = edge_index_np[0]
    dst = edge_index_np[1]
    g = dgl.graph((src, dst))
    g.ndata["feat"] = torch.from_numpy(node_feats_np)
    g.edata["feat"] = torch.from_numpy(edge_feats_np)
    return g
