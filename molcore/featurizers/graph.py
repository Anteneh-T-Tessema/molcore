import torch
from torch_geometric.data import Data
from molcore._molcore import PyMolGraph, mol_to_graph_arrays


def to_pyg_data(graph: PyMolGraph) -> Data:
    """
    Rust graph arrays → PyG Data. Zero-copy at every stage.
    node_feats: (N, 9) float32
        [atomic_num, is_aromatic, formal_charge, num_hs,
         degree, in_ring, hybridization, chirality, mass_norm]
    edge_index: (2, E) int64    — COO, bidirectional
    edge_attr:  (E, 4) float32  — bond type one-hot [single, double, triple, aromatic]
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


_ATOM_TYPES = {6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl",
               35: "Br", 53: "I"}
_OTHER = "other"


def to_pyg_hetero(graph: PyMolGraph):
    """
    Heterogeneous PyG graph partitioned by atom type.

    Node sets   : one per element symbol (C, N, O, F, P, S, Cl, Br, I, other)
    Edge sets   : ('C','bond','N') etc. — one per (src_type, dst_type) pair present
    node features: same 4-vector as to_pyg_data, stored under 'x'
    edge features: same 4-vector bond one-hot, stored under 'edge_attr'

    Requires: torch_geometric >= 2.0
    """
    from torch_geometric.data import HeteroData
    import numpy as np

    node_feats_np, edge_index_np, edge_feats_np = mol_to_graph_arrays(graph)
    n_atoms = node_feats_np.shape[0]

    # Map global atom index → type label
    type_labels = [
        _ATOM_TYPES.get(int(node_feats_np[i, 0]), _OTHER)
        for i in range(n_atoms)
    ]

    # Local index within each type
    type_to_globals: dict[str, list[int]] = {}
    global_to_local: list[int] = []
    for g_idx, t in enumerate(type_labels):
        local = len(type_to_globals.setdefault(t, []))
        type_to_globals[t].append(g_idx)
        global_to_local.append(local)

    data = HeteroData()

    # Node features
    for t, g_indices in type_to_globals.items():
        data[t].x = torch.from_numpy(node_feats_np[g_indices])

    # Edges: group by (src_type, dst_type)
    n_edges = edge_index_np.shape[1]
    edge_groups: dict[tuple[str, str], tuple[list[int], list[int], list[int]]] = {}
    for e in range(n_edges):
        sg, dg = int(edge_index_np[0, e]), int(edge_index_np[1, e])
        st, dt = type_labels[sg], type_labels[dg]
        key = (st, dt)
        if key not in edge_groups:
            edge_groups[key] = ([], [], [])
        edge_groups[key][0].append(global_to_local[sg])
        edge_groups[key][1].append(global_to_local[dg])
        edge_groups[key][2].append(e)

    for (st, dt), (srcs, dsts, e_indices) in edge_groups.items():
        ei = torch.tensor([srcs, dsts], dtype=torch.long)
        ef = torch.from_numpy(edge_feats_np[e_indices])
        data[st, "bond", dt].edge_index = ei
        data[st, "bond", dt].edge_attr = ef

    return data
