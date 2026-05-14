# Procedural Memory: SMILES Handling

## Ingestion procedure

1. Pass raw SMILES to `Mol.from_smiles()` — Rust ingestion, built-in SMILES parser
2. Use `mol.smiles` (canonical) for all subsequent storage and comparison
3. Never compare raw input SMILES — always canonicalize first

## Error handling procedure

1. `MolIngestionError` from Rust → surface to user, log the offending SMILES
2. Do not attempt auto-correction of invalid SMILES
3. Log invalid SMILES to `memory/episodic/` with timestamp for debugging patterns

## Backend selection procedure

- New model training → `backend="rust"` (speed, ~40× faster than RDKit)
- Inference on legacy model trained on RDKit fingerprints → `backend="rdkit"` (bit-identical)
- Unsure which the model was trained on → ask the user before proceeding

## Chirality procedure

- Chirality is parsed from bracket atoms (`[C@H]`, `[C@@H]`) during ingestion
- `@` → node feature 7 = 1.0 (S config); `@@` → 2.0 (R config); achiral → 0.0
- When stereo matters (e.g. chiral drugs), verify SMILES contains `@`/`@@` notation
- Do not strip chirality before featurization — it is encoded in the 9-feature vector

## Scaffold split procedure

1. Call `scaffold_split(smiles, train_frac=0.8, val_frac=0.1)` from `molcore.rdkit_bridge`
2. Verify `len(train) + len(val) + len(test) == len(smiles)` — no molecules lost
3. Use scaffold split (not random split) for all drug-discovery GNN evaluations
4. Report train/val/test scaffold counts alongside R² / MAE

## Conformer generation procedure

1. Call `mol.conformers(n_confs=1, seed=42)` — returns list of (N_atoms, 3) float64 arrays
2. Always fix `seed` for reproducibility; document the seed in experiment logs
3. For 3D-dependent models (SchNet, DimeNet, GVP), generate conformers before featurization
4. For shape-based filtering, use `mol.descriptors_3d()` directly (no explicit conformer needed)

## SMARTS search procedure

1. Call `mol.matches(smarts)` for single-molecule boolean check
2. Call `filter_by_smarts(smiles_list, smarts)` for batch filtering
3. Use `invert=True` to remove reactive/unwanted groups from a library
4. Validate SMARTS with a test molecule before running on large libraries — invalid SMARTS raises `ValueError`
