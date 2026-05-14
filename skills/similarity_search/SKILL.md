# Skill: similarity_search

## When to invoke
When an agent needs to rank a compound library by similarity to a query molecule,
for virtual screening or nearest-neighbor lookup.

## Inputs
| Field | Type | Description |
|---|---|---|
| `query_smiles` | `list[str]` | Query molecules (Q) |
| `library_smiles` | `list[str]` | Library to screen (L) |
| `top_k` | `int` | Return top K hits per query (default: 10) |
| `threshold` | `float` | Tanimoto cutoff — discard hits below this (default: 0.0) |

## Outputs
For each query: list of `(smiles, tanimoto_score)` tuples, sorted descending.

## Entry point
```python
from molcore.pipeline import featurize_smiles
from molcore._molcore import tanimoto_matrix

q_fps = featurize_smiles(query_smiles).numpy()
l_fps = featurize_smiles(library_smiles).numpy()
sim   = tanimoto_matrix(q_fps, l_fps)   # (Q, L) float32
```

## Performance targets
- 10k × 10k Tanimoto matrix: ≥ 10× faster than RDKit BulkTanimoto
