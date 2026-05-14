# Skill: property_prediction

## When to invoke
When an agent needs fast molecular descriptors (MW, LogP, TPSA) for filtering,
lead optimization context, or Lipinski Ro5 assessment.

## Inputs
| Field | Type | Description |
|---|---|---|
| `smiles` | `list[str]` | Batch of SMILES |
| `backend` | `str` | `"rust"` (fast, approx) or `"rdkit"` (exact) |

## Outputs
`torch.Tensor` of shape `(N, 3)`:
- col 0: MW (g/mol)
- col 1: LogP (Crippen)
- col 2: heavy atom count (Rust) or TPSA (RDKit backend)

## Entry point
```python
from molcore.featurizers.descriptors import calc_descriptors
desc = calc_descriptors(smiles, backend="rdkit")
```

## When to use which backend
- `rust`: Lead optimization loops, billion-scale filtering — speed is critical
- `rdkit`: Final property report, regulatory submission, exact Lipinski check
