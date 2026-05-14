# Template: Scaffold-Based Workflow

## Step 1 — Scaffold decomposition

```python
from molcore.rdkit_bridge import murcko_scaffold

scaffolds = [murcko_scaffold(smi) for smi in library]
unique_scaffolds = len(set(s for s in scaffolds if s))
print(f"{unique_scaffolds} unique scaffolds across {len(library)} molecules")
```

## Step 2 — Scaffold-aware train/val/test split

```python
from molcore.rdkit_bridge import scaffold_split

train, val, test = scaffold_split(
    library,
    train_frac={{train_frac}},
    val_frac={{val_frac}},
    seed=42,
)
print(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")
# Verify no scaffold overlap (guaranteed by scaffold_split)
```

## Step 3 — SMARTS-based reactive group removal (optional)

```python
from molcore import filter_by_smarts

REACTIVE = ["[CH]=O", "[CH2]=[CH]-C=O", "C(=O)Cl"]
clean = library
for smarts in REACTIVE:
    clean = filter_by_smarts(clean, smarts, invert=True)
print(f"Removed {len(library) - len(clean)} reactive molecules")
```

## Step 4 — Featurize with scaffold-split sets

```python
from molcore.pipeline import featurize_smiles
from molcore.molecule import Mol

train_fps = featurize_smiles(train, backend="rust")
val_fps   = featurize_smiles(val,   backend="rust")

# Graph tensors for GNN (9-feature nodes)
train_graphs = [Mol.from_smiles(smi).to_pyg() for smi in train]
```

## Step 5 — Report

State:
- Total molecules: {{total}}
- Unique scaffolds: {{unique_scaffolds}}
- Split sizes: {{train}} / {{val}} / {{test}}
- Reactive molecules removed: {{n_removed}}
- Backend: rust
