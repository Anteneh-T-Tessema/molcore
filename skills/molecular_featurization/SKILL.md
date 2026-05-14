# Skill: molecular_featurization

## When to invoke
When an agent needs to convert molecules into graph tensors for GNN training or inference
(PyG `Data`, PyG `HeteroData`, or DGL graphs).

## Inputs
| Field | Type | Description |
|---|---|---|
| `smiles` | `str` | Single SMILES string |
| `hetero` | `bool` | Return `HeteroData` partitioned by atom type (default: False) |

## Outputs — homogeneous (`to_pyg`)
`torch_geometric.data.Data` with:

| Tensor | Shape | dtype | Description |
|---|---|---|---|
| `x` | `(N, 9)` | float32 | Node features (see layout below) |
| `edge_index` | `(2, E)` | int64 | COO bidirectional edge list |
| `edge_attr` | `(E, 4)` | float32 | Bond one-hot `[single, double, triple, aromatic]` |

### Node feature layout (`x` columns)
| idx | Feature | Notes |
|---|---|---|
| 0 | `atomic_num` | Raw Z |
| 1 | `is_aromatic` | 0 / 1 |
| 2 | `formal_charge` | Signed |
| 3 | `num_hs` | Implicit H count |
| 4 | `degree` | Heavy-atom neighbor count |
| 5 | `in_ring` | 0 / 1 (BFS bridge-detection) |
| 6 | `hybridization` | 0=unknown · 1=sp · 2=sp2 · 3=sp3 |
| 7 | `chirality` | 0=none · 1=@ (S) · 2=@@ (R) |
| 8 | `mass_norm` | atomic_mass / 100 |

## Outputs — heterogeneous (`to_pyg_hetero`)
`torch_geometric.data.HeteroData` partitioned by element symbol
(C, N, O, F, P, S, Cl, Br, I, other).

- `data[element].x` : `(n_elem, 9)` float32
- `data[src, "bond", dst].edge_index` : `(2, e)` int64
- `data[src, "bond", dst].edge_attr`  : `(e, 4)` float32

## Entry points
```python
from molcore.molecule import Mol

mol  = Mol.from_smiles("c1ccccc1C(=O)O")
data = mol.to_pyg()          # HeteroData: mol.to_pyg_hetero()
dgl  = mol.to_dgl()
```

## Guardrails
- All arrays are zero-copy from Rust. Do not `.clone()` before passing to a model.
- `edge_index` dtype must stay `torch.long` — PyG asserts this.
- `NODE_FEAT_DIM = 9`; update any model `in_features` accordingly.
- For legacy models trained on 4-feature vectors, re-train or add a projection layer.
