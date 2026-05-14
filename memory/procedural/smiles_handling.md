# Procedural Memory: SMILES Handling

## Ingestion procedure
1. Pass raw SMILES to `Mol.from_smiles()` — Rust ingestion, RDKit sanitization
2. Use `mol.smiles` (canonical) for all subsequent storage and comparison
3. Never compare raw input SMILES — always canonicalize first

## Error handling procedure
1. `MolIngestionError` from Rust → surface to user, log the offending SMILES
2. Do not attempt auto-correction of invalid SMILES
3. Log invalid SMILES to `memory/episodic/` with timestamp for debugging patterns

## Backend selection procedure
- New model training → `backend="rust"` (speed)
- Inference on legacy model → `backend="rdkit"` (parity)
- Unsure which the model was trained on → ask the user before proceeding
