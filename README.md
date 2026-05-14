# molcore

**AI-native cheminformatics toolkit** — Rust performance core · RDKit bridge · Python AI API

[![CI](https://github.com/Anteneh-T-Tessema/molcore/actions/workflows/ci.yml/badge.svg)](https://github.com/Anteneh-T-Tessema/molcore/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What is molcore?

molcore bridges the gap between cheminformatics and modern AI/ML pipelines. It exposes molecules as **batch-first, zero-copy tensors** that flow directly into PyTorch and PyG — no Python loops, no intermediate copies.

```python
import molcore

# Batch fingerprints: Rust Rayon parallel → numpy → torch (zero copy)
fps = molcore.featurize_smiles(["CCO", "c1ccccc1", "CC(=O)O"], backend="rust")
# torch.Tensor shape (3, 2048), dtype uint8

# Frozen, immutable Mol — transforms return new instances
mol  = molcore.Mol.from_smiles("CC(=O)Oc1ccccc1C(=O)O")   # aspirin
mol2 = mol.neutralize()    # new Mol, mol is unchanged
data = mol.to_pyg()        # PyG Data object, zero-copy

# Parquet I/O for large datasets
ds = molcore.MolDataset.from_smiles(smiles_list, compute_fps=True)
ds.labels = logp_array
ds.write_parquet("screening_library.parquet")
```

---

## Architecture

```
SMILES strings
  │
  ▼  Rust ingest (built-in parser or rdkit-rs)
  │  ─ sanitize, aromaticity, implicit H, ring closure
  ▼
petgraph StableGraph (immutable after construction)
  │
  ├─▶ ecfp4_batch()      → (N × 2048) uint8  ─▶ torch.from_numpy()  ─▶ Tensor
  │   Rayon parallel · u64 bit-pack · hardware popcount
  │
  ├─▶ mol_to_graph_arrays() → node_feats, edge_index, edge_attr ─▶ PyG Data
  │   (N × 4) float32 · (2 × E) int64 · (E × 4) float32
  │
  └─▶ tanimoto_matrix()   → (Q × L) float32
      Rayon parallel · u64 popcount · 30× faster than RDKit at scale

Python layer (molcore/)
  molecule.py     — frozen Mol dataclass (FrozenInstanceError on mutation)
  pipeline.py     — featurize_smiles() batch-first entry point
  rdkit_bridge.py — ALL RDKit calls isolated here
  io.py           — MolDataset: Parquet/Arrow columnar storage
  agentic_rag.py  — ChemRAG: iterative chemical literature retrieval

AGENT_SKILLS scaffold
  skills/     — fingerprint · similarity_search · molecular_featurization
              — property_prediction · agentic_rag
  prompts/    — system prompt · query templates · safety fragments
  tools/      — local tools (featurize, similarity, screen, descriptors)
              — MCP configs (PubChem, ChEMBL)
  memory/     — semantic (chemistry knowledge) · procedural (SMILES handling)
  evals/      — per_skill · compositional · regression
  observability/ — OpenTelemetry config · domain events · skill metrics
```

---

## Benchmarks

All numbers on Apple M-series (arm64), CPU-only, Python 3.12.

### ECFP4 fingerprints — batch throughput

| Batch size | molcore (Rust) | RDKit | Speedup |
|---|---|---|---|
| 1 000 SMILES | 1.3M mol/s | 14 800 mol/s | **88×** |
| 10 000 SMILES | 2.0M mol/s | 15 100 mol/s | **132×** |

### Tanimoto similarity matrix

| Query × Library | molcore (Rust) | RDKit BulkTanimoto | Speedup |
|---|---|---|---|
| 50 × 1 000 | 31M pairs/s | 7.3M pairs/s | **4.3×** |
| 500 × 10 000 | 224M pairs/s | 7.7M pairs/s | **29×** |

> Tanimoto uses u64 bit-packing + hardware `count_ones()`. Small batch overhead dominates at 50 queries; parallelism wins at scale.

---

## Installation

### Requirements

- Python 3.12+
- Rust 1.70+ (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)

### Quick start (one command)

```bash
git clone https://github.com/Anteneh-T-Tessema/molcore
cd molcore
./setup_dev.sh          # creates .venv, installs deps, builds Rust extension, runs tests
source .venv/bin/activate
```

### Manual install

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install maturin torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy rdkit torch-geometric pytest
maturin develop --release --features extension-module
```

---

## Usage

### Fingerprints

```python
from molcore.pipeline import featurize_smiles

# Rust backend — for training new models, billion-scale screening
fps = featurize_smiles(smiles, backend="rust")   # (N, 2048) uint8

# RDKit backend — bit-identical to any model trained on RDKit fingerprints
fps = featurize_smiles(smiles, backend="rdkit")
```

### Similarity search

```python
from molcore._molcore import tanimoto_matrix
import numpy as np

q_fps = featurize_smiles(query_smiles).numpy()
l_fps = featurize_smiles(library_smiles).numpy()
sim   = tanimoto_matrix(q_fps, l_fps)   # (Q, L) float32, Rayon parallel
```

### Graph features for GNNs

```python
from molcore.molecule import Mol

mol  = Mol.from_smiles("c1ccccc1C(=O)O")
data = mol.to_pyg()   # torch_geometric.data.Data, zero-copy
# data.x          — (N, 4) float32: [atomic_num, is_aromatic, formal_charge, num_hs]
# data.edge_index — (2, E) int64 COO, bidirectional
# data.edge_attr  — (E, 4) float32 bond-type one-hot
```

### Dataset I/O

```python
from molcore.io import MolDataset
import numpy as np

ds = MolDataset.from_smiles(smiles, compute_fps=True, compute_desc=True)
ds.labels = np.array(logp_values, dtype=np.float32)
ds.write_parquet("library.parquet")

ds2 = MolDataset.read_parquet("library.parquet")
fps_tensor  = ds2.fingerprints_tensor()   # torch.Tensor uint8
desc_tensor = ds2.descriptors_tensor()    # torch.Tensor float32
```

### Agentic RAG

```python
from molcore.agentic_rag import ChemRAG

rag    = ChemRAG()
result = rag.query("What is the logP of aspirin?", verbose=True)
print(result.answer)
print(f"Sources: {result.sources}, Iterations: {result.iterations}")
```

---

## Design invariants

These are enforced across the codebase — do not break them:

1. **`Mol` is always immutable.** Transforms return new instances.
2. **RDKit is never called in hot paths.** Only `rdkit_bridge.py` imports RDKit.
3. **All Rust → Python array transfers use `IntoPyArray`.** No Python-side loops.
4. **Batch API is primary.** Per-mol methods are convenience wrappers.
5. **Backend flags are explicit.** `"rust"` | `"rdkit"` — never auto-selected.

---

## Development

```bash
# Build Rust extension
maturin develop --release --features extension-module

# Rust unit tests (ingest + fingerprints)
cargo test -p molcore-core

# Python integration + skill evals
pytest tests/python evals/ -v

# Benchmarks
python benchmarks/bench_fingerprints.py --smiles 10000
python benchmarks/bench_tanimoto.py --query 500 --library 10000

# End-to-end GCN demo
python examples/end_to_end_gnn.py
```

### Project layout

```
molcore/
├── crates/molcore-core/   # Rust: ingest, fingerprints, graph_arrays, similarity, descriptors
├── crates/molcore-io/     # Rust: Arrow/Parquet columnar I/O (stub → Python io.py)
├── molcore/               # Python package
│   ├── molecule.py        # Frozen Mol dataclass
│   ├── pipeline.py        # featurize_smiles() — primary batch entry point
│   ├── rdkit_bridge.py    # ALL RDKit calls isolated here
│   ├── io.py              # MolDataset: Parquet/Arrow I/O via pyarrow
│   ├── agentic_rag.py     # ChemRAG: iterative retrieval loop
│   ├── explainability.py  # Atom/bond attribution (gradient + integrated gradients)
│   └── featurizers/       # fingerprints, descriptors, graph, pretrained wrappers
├── skills/                # AGENT_SKILLS: SKILL.md + examples.jsonl + allowlist.yaml
├── prompts/               # system/, templates/, fragments/
├── tools/                 # registry.yaml, local/, mcp/
├── memory/                # semantic/, procedural/, episodic/, working/
├── evals/                 # per_skill/, compositional/, regression/
├── observability/         # traces/, events/, metrics/
├── benchmarks/            # bench_fingerprints.py, bench_tanimoto.py
├── examples/              # end_to_end_gnn.py
└── tests/                 # python/ (integration), crates/ (Rust unit)
```

---

## License

MIT — see [LICENSE](LICENSE).
