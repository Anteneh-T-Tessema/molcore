# Skill: property_prediction

## When to invoke
When an agent needs molecular descriptors for filtering, lead optimization, Lipinski
Ro5 assessment, or 3D shape analysis (docking prep, scaffold-hopping).

## 2D Descriptors

### Inputs
| Field | Type | Description |
|---|---|---|
| `smiles` | `list[str]` | Batch of SMILES |
| `backend` | `str` | `"rust"` (fast, approx) or `"rdkit"` (exact) |

### Outputs — `(N, 3)` float32
| col | Feature | Rust | RDKit |
|---|---|---|---|
| 0 | MW (g/mol) | exact atomic masses | `Descriptors.MolWt` |
| 1 | LogP | Crippen fragment approx | `Descriptors.MolLogP` |
| 2 | Heavy atom count | graph node count | TPSA (`Descriptors.TPSA`) |

### Entry point
```python
from molcore.featurizers.descriptors import calc_descriptors
desc = calc_descriptors(smiles, backend="rdkit")   # (N, 3) float32
```

## 3D Descriptors (requires conformer)

### Inputs
| Field | Type | Description |
|---|---|---|
| `smiles` | `str` | Single SMILES |
| `seed` | `int` | Random seed for ETKDGv3 (default: 42) |

### Outputs — `dict[str, float]`
`pmi1`, `pmi2`, `pmi3`, `asphericity`, `eccentricity`, `npr1`, `npr2`,
`radius_of_gyration`, `inertial_shape_factor`, `spherocity_index`

### Entry point
```python
from molcore.molecule import Mol
desc_3d = Mol.from_smiles("c1ccccc1C(=O)O").descriptors_3d()
```

## Lipinski Ro5 filter
```python
from molcore.featurizers.descriptors import calc_descriptors
desc = calc_descriptors(smiles, backend="rdkit")   # MW, LogP, TPSA
# Ro5: MW ≤ 500, LogP ≤ 5; also check HBA ≤ 10, HBD ≤ 5 via RDKit
passing = [smi for smi, (mw, logp, _) in zip(smiles, desc.tolist())
           if mw <= 500 and logp <= 5]
```

## Backend selection
| Use case | Backend |
|---|---|
| Lead opt loop, billion-scale filter | `rust` |
| Regulatory report, exact Lipinski | `rdkit` |
| 3D shape / docking prep | `rdkit` (conformer) |

## When NOT to use
- Do not use for exact ADMET prediction — use a trained model on these descriptors instead.
- 3D descriptors are stochastic (ETKDG seed controls conformer); always fix the seed for reproducibility.
