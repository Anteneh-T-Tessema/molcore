# molcore

**AI-native cheminformatics toolkit** — Rust performance, RDKit compatibility, native PyG integration.

[![CI](https://github.com/Anteneh-T-Tessema/molcore/actions/workflows/ci.yml/badge.svg)](https://github.com/Anteneh-T-Tessema/molcore/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/molcore-chem)](https://pypi.org/p/molcore-chem)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

```bash
pip install molcore    # Rust compiled into the wheel — no conda required
```

---

## The pitch

You already use RDKit. molcore doesn't replace it — it wraps the parts you need, accelerates the hot paths in Rust, and wires molecules directly into PyTorch Geometric without boilerplate.

**35× faster fingerprints.** Rust Rayon parallel + u64 bit-packing vs. Python per-molecule loops. At 10k molecules: 2M mol/s vs 15k mol/s.

**4.3× faster PyG conversion.** Zero-copy `IntoPyArray` → `torch.from_numpy` instead of Python atom-loop feature extraction.

**Everything else at parity.** Standardization, descriptors, scaffold split — these delegate to RDKit. Same speed, cleaner API.

---

## Five lines from SMILES to GNN prediction

```python
from molcore.molecule import Mol
from molcore.pipeline import featurize_smiles
from molcore.predictor import PropertyPredictor
from molcore.io import MolDataset
import numpy as np

# 1. Parse — immutable, Rust-backed
mol = Mol.from_smiles("CC(=O)Oc1ccccc1C(=O)O")   # aspirin
data = mol.to_pyg()   # PyG Data, zero-copy, 9 node features

# 2. Batch fingerprints — Rust Rayon parallel
fps = featurize_smiles(smiles_list, backend="rust")   # (N, 2048) uint8 Tensor, 35× faster

# 3. Full dataset pipeline
ds = MolDataset.from_smiles(smiles_list, compute_fps=True, compute_desc=True)
ds.labels = np.array(logp_values, dtype=np.float32)
train_ds, val_ds, test_ds = ds.scaffold_split()

# 4. Train GCN, get uncertainty
pred = PropertyPredictor(hidden=64, epochs=100)
pred.fit(train_ds, val_dataset=val_ds)
means, stds = pred.predict_with_uncertainty(["CCO", "c1ccccc1"], n_samples=30)
```

**[Open in Colab →](https://colab.research.google.com/github/Anteneh-T-Tessema/molcore/blob/main/examples/quickstart.ipynb)**

---

## Benchmarks

All numbers on Apple M-series (arm64), CPU-only, Python 3.12.

### ECFP4 fingerprints

| Batch size | molcore (Rust) | RDKit | Speedup |
|---|---|---|---|
| 1 000 SMILES | 1.3M mol/s | 14 800 mol/s | **88×** |
| 10 000 SMILES | 2.0M mol/s | 15 100 mol/s | **132×** |

### Tanimoto similarity matrix

| Query × Library | molcore (Rust) | RDKit BulkTanimoto | Speedup |
|---|---|---|---|
| 50 × 1 000 | 31M pairs/s | 7.3M pairs/s | **4.3×** |
| 500 × 10 000 | 224M pairs/s | 7.7M pairs/s | **29×** |

### Full pre-training pipeline (500 molecules)

| Step | molcore | RDKit | Speedup |
|---|---|---|---|
| Standardize | 242 ms | 225 ms | ~parity |
| ECFP4 fingerprints | 1.1 ms | 37.3 ms | **35×** |
| 7 Lipinski descriptors | 124 ms | 114 ms | ~parity |
| Scaffold split | 33 ms | 35 ms | ~parity |
| PyG conversion (200 mols) | 3.3 ms | 14.4 ms | **4.3×** |

### GNN property prediction — ESOL solubility (scaffold split)

ESOL dataset (Delaney 2004, 1128 molecules). Scaffold split is substantially harder
than the random split used in published MoleculeNet baselines — results are not
directly comparable to the published RMSE ≈ 0.58.

| Configuration | RMSE | R² |
|---|---|---|
| GCN, hidden=64, 3 layers, 300 epochs | 1.038 | 0.727 |
| Optuna-tuned (30 trials): hidden=128, 4 layers | 1.090 | 0.709 |

---

## What it does

### SDF and Parquet I/O

```python
from molcore.io import MolDataset

ds = MolDataset.from_sdf("library.sdf")          # or .sdf.gz
ds = MolDataset.from_sdf("library.sdf", compute_fps=True, compute_desc=True)
ds.write_sdf("output.sdf")
ds.write_parquet("library.parquet")               # Arrow columnar, snappy-compressed
ds2 = MolDataset.read_parquet("library.parquet")
```

### Pandas bridge

```python
import molcore.pandas_tools as mpt

df = mpt.load_sdf("library.sdf")                 # → DataFrame with 'Mol' + 'smiles' columns
df = mpt.add_descriptors(df, preset="lipinski")  # adds MolWt, LogP, TPSA, HBD, HBA, ...
df = mpt.add_fingerprints(df, kind="ecfp4")      # adds 'fp' column
df = mpt.filter_by_smarts(df, "c1ccncc1")        # substructure filter in-place
df = mpt.standardize_smiles(df)                  # strip salts + neutralize + canonical tautomer
```

### Full descriptor set

```python
from molcore.rdkit_bridge import calc_named_descriptors

# Any of the ~200 RDKit descriptors, batch, returns (N, D) float32
arr, names = calc_named_descriptors(smiles, preset="lipinski")    # 7 descriptors
arr, names = calc_named_descriptors(smiles, preset="druglike")    # 15 descriptors
arr, names = calc_named_descriptors(smiles, preset="all")         # all ~200
arr, names = calc_named_descriptors(smiles, names=["MolWt", "TPSA", "BertzCT"])
```

### Multiple fingerprint types

```python
fps = featurize_smiles(smiles, kind="ecfp4")               # (N, 2048) — Rust parallel
fps = featurize_smiles(smiles, kind="maccs")               # (N, 167)
fps = featurize_smiles(smiles, kind="atom_pairs")          # (N, 2048)
fps = featurize_smiles(smiles, kind="topological_torsions") # (N, 2048)
fps = featurize_smiles(smiles, kind="rdkit")               # (N, 2048) RDKit path-based
```

### 2D depiction — Jupyter-native

```python
mol = Mol.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
mol          # → renders inline in Jupyter via _repr_svg_
mol.to_png("aspirin.png")

ds = MolDataset.from_sdf("library.sdf")
ds           # → renders 8-molecule grid inline
ds.draw_grid(n=20, mols_per_row=4)
```

### Standardization

```python
from molcore.rdkit_bridge import standardize

clean = standardize("[Na+].OC(=O)c1ccccc1")   # → "OC(=O)c1ccccc1"
# strips salts → neutralizes charges → canonical tautomer → canonical SMILES

mol = Mol.from_smiles("[Na+].CC(=O)[O-]")
clean_mol = mol.standardize()   # → new Mol, original unchanged
```

### MCS and R-group decomposition

```python
from molcore.rdkit_bridge import find_mcs, rgroup_decompose

# Maximum Common Substructure of an analog series
smarts = find_mcs(["CC(=O)Oc1ccccc1", "CC(=O)Oc1ccc(F)cc1", "CC(=O)Oc1ccc(Cl)cc1"])

# R-group decomposition
rows = rgroup_decompose("c1ccc([*:1])cc1", smiles_list)
# → [{"Core": "c1ccccc1", "R1": "F"}, {"Core": "c1ccccc1", "R1": "Cl"}, ...]
```

### GCN predictor with MC Dropout uncertainty

```python
from molcore.predictor import PropertyPredictor

pred = PropertyPredictor(hidden=64, n_layers=3, epochs=100, dropout=0.1)
pred.fit(train_ds, val_dataset=val_ds, verbose=True)

# Point prediction
predictions = pred.predict(smiles_list)            # numpy array

# Uncertainty — MC Dropout, no retraining
means, stds = pred.predict_with_uncertainty(smiles_list, n_samples=30)

# Save and reload
pred.save("logp_model.pt")
pred2 = PropertyPredictor.load("logp_model.pt")
```

---

## Installation

```bash
pip install molcore
```

Requires Python 3.11+. RDKit and PyTorch are declared dependencies — no manual conda setup.

For GPU:

```bash
pip install molcore
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
```

### Build from source

```bash
git clone https://github.com/Anteneh-T-Tessema/molcore
cd molcore
./setup_dev.sh    # creates .venv, builds Rust extension, runs tests
source .venv/bin/activate
```

Requires Rust 1.70+: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

---

## Architecture

```text
SMILES strings
  │
  ▼  Rust ingest (RDKit-backed aromaticity perception)
  │  ─ sanitize, kekulize, ring perception, implicit H
  ▼
petgraph StableGraph (immutable after construction)
  │
  ├─▶ ecfp4_batch()       → (N × 2048) uint8  ─▶ torch.from_numpy()  ─▶ Tensor
  │   Rayon parallel · u64 bit-pack · hardware popcount · 35–132× faster
  │
  ├─▶ mol_to_graph_arrays() → node_feats (9-dim), edge_index, edge_attr ─▶ PyG Data
  │   Zero-copy IntoPyArray · 4.3× faster than manual Python construction
  │
  └─▶ tanimoto_matrix()   → (Q × L) float32
      Rayon parallel · u64 popcount · 29× faster at scale

Python layer (molcore/)
  molecule.py      — frozen Mol dataclass (FrozenInstanceError on mutation)
  pipeline.py      — featurize_smiles() batch-first entry point
  rdkit_bridge.py  — ALL RDKit calls isolated here (one file to update)
  io.py            — MolDataset: SDF + Parquet + DataFrame bridge
  predictor.py     — PropertyPredictor: 3-layer GCN + MC Dropout
  pandas_tools.py  — DataFrame-first API for existing RDKit workflows
  agentic_rag.py   — ChemRAG: iterative chemical literature retrieval
```

### Five invariants — never broken

1. `Mol` is always immutable — transforms return new instances
2. RDKit is never in hot paths — only `rdkit_bridge.py` imports rdkit
3. All Rust→Python array transfers use `IntoPyArray` — no Python-side loops
4. Batch API is primary — per-mol methods are convenience wrappers
5. Backend flags are explicit — `"rust"` | `"rdkit"` always caller-supplied

---

## Development

```bash
maturin develop --release --features extension-module   # build Rust extension
cargo test -p molcore-core                              # 17 Rust unit tests
pytest tests/python evals/ -v                           # 384 Python tests
python benchmarks/bench_e2e.py --n 1000                # end-to-end benchmark
python benchmarks/bench_fingerprints.py --smiles 10000
```

---

## Documentation

- **[Quickstart notebook](examples/quickstart.ipynb)** — [Open in Colab](https://colab.research.google.com/github/Anteneh-T-Tessema/molcore/blob/main/examples/quickstart.ipynb)
- **[Migrating from RDKit](docs/migrating_from_rdkit.md)** — API mapping for every common RDKit pattern
- **[End-to-end GNN example](examples/end_to_end_gnn.py)** — ESOL solubility benchmark
- **[Virtual screening pipeline](examples/virtual_screening_pipeline.py)**

---

## License

MIT — see [LICENSE](LICENSE).
