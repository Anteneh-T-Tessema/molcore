# Skill: molecular_featurization

## When to invoke
When an agent needs to convert molecules into graph tensors for GNN training or inference
(PyG Data objects or DGL graphs).

## Inputs
| Field | Type | Description |
|---|---|---|
| `smiles` | `str` | Single SMILES string |

## Outputs
`torch_geometric.data.Data` with:
- `x`: `(N, 4)` float32 — node features `[atomic_num, is_aromatic, formal_charge, num_hs]`
- `edge_index`: `(2, E)` int64 — COO bidirectional edge list
- `edge_attr`: `(E, 4)` float32 — bond type one-hot `[single, double, triple, aromatic]`

## Entry point
```python
from molcore.molecule import Mol
data = Mol.from_smiles("c1ccccc1C(=O)O").to_pyg()
```

## Guardrails
- All arrays are zero-copy from Rust. Do not call `.clone()` before passing to a model.
- `edge_index` dtype must remain `torch.long` — PyG asserts this.
