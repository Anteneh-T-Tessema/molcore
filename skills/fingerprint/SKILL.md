# Skill: fingerprint

## When to invoke
When an agent needs to compute molecular fingerprints for a list of SMILES strings,
for purposes of: model training input, similarity search, virtual screening, or clustering.

## Inputs
| Field | Type | Description |
|---|---|---|
| `smiles` | `list[str]` | SMILES strings to featurize |
| `radius` | `int` | Morgan radius (default: 2 = ECFP4) |
| `nbits` | `int` | Fingerprint length (default: 2048) |
| `backend` | `str` | `"rust"` (fast, new models) or `"rdkit"` (legacy model parity) |

## Outputs
`torch.Tensor` of shape `(N, nbits)` dtype `uint8`. Zero-copy from Rust.

## Entry point
```python
from molcore.pipeline import featurize_smiles
fps = featurize_smiles(smiles, backend="rust")
```

## Guardrails
- **Batch API** (`featurize_smiles`): invalid SMILES → all-zero fingerprint row (graceful
  degradation so one bad SMILES does not abort a million-compound screen).
- **Single-mol API** (`Mol.from_smiles`): invalid SMILES → `MolIngestionError` raised.
- `backend` must be explicit. Never auto-selected.
- Do not call this per-molecule in a loop — always pass the full batch.

## Performance targets
- Rust backend: ≥ 1M SMILES/sec on 8-core CPU
- RDKit backend: ~50k SMILES/sec (single-threaded, RDKit limitation)
