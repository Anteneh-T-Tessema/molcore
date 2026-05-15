# Launch announcements — v0.2.0

Ready-to-post text for each channel.

---

## RDKit mailing list (rdkit-discuss@lists.sourceforge.net)

**Subject:** molcore v0.2: Rust-accelerated cheminformatics — clustering, MMPA, diversity picking, GAT/GIN, Optuna tuning

Hi all,

I wanted to share **molcore v0.2**, an open-source cheminformatics toolkit that wraps RDKit where it matters and accelerates the hot paths in Rust.

The core idea: molecules as batch-first, zero-copy tensors that flow directly into PyTorch Geometric — no Python loops, no intermediate copies.

**Performance** (Apple M-series, CPU, 10k molecules):

- ECFP4 fingerprints via Rust Rayon: **132× faster** than RDKit
- Tanimoto matrix: **29× faster** via u64 `count_ones()`
- PyG `Data` export: **4.3× faster** than manual Python feature extraction

**What's new in v0.2:**

- **Butina clustering** (`butina_cluster`, `MolDataset.cluster`): Tanimoto-distance clustering, adds `cluster_id` metadata column
- **k-fold and scaffold k-fold CV** (`MolDataset.kfold`, `MolDataset.scaffold_kfold`): no scaffold leaks between folds
- **MMPA — single-cut fragmentation** (`mmpa`): matched molecular pairs with `core`, `smiles_a/b`, and `transform` SMARTS
- **MaxMin diversity picking** (`diversity_pick`): O(n × N) iterative Tanimoto-space selection, scales to ~500k
- **GAT and GIN architectures** (`PropertyPredictor(model_type="gat"|"gin"|"gcn")`): unified `_MolGNN` module, backward-compatible checkpoints
- **Optuna HP search** (`PropertyPredictor.tune`): searches hidden dim, layers, dropout, lr, batch size, and architecture; requires `pip install molcore[optuna]`

**ESOL benchmark** (Delaney 2004, 1128 molecules, scaffold split):

| Configuration | RMSE | R² |
| --- | --- | --- |
| GCN, hidden=64, 3 layers, 300 epochs | 1.038 | 0.727 |
| Optuna-tuned (30 trials): hidden=128, 4 layers | 1.090 | 0.709 |

Note: scaffold split is substantially harder than the random split used in published MoleculeNet baselines (RMSE ≈ 0.58) — results are not directly comparable.

**Install:**

```bash
pip install molcore          # core
pip install molcore[optuna]  # + Optuna tuning
pip install molcore[all]     # everything
```

**Quickstart notebook (Colab):**
[colab.research.google.com/github/Anteneh-T-Tessema/molcore/blob/main/examples/quickstart.ipynb](https://colab.research.google.com/github/Anteneh-T-Tessema/molcore/blob/main/examples/quickstart.ipynb)

**Migration guide** (RDKit → molcore):
[github.com/Anteneh-T-Tessema/molcore/blob/main/docs/migrating_from_rdkit.md](https://github.com/Anteneh-T-Tessema/molcore/blob/main/docs/migrating_from_rdkit.md)

The design keeps RDKit as the chemistry authority — all RDKit calls are isolated in a single `rdkit_bridge.py`. The Rust core handles batch hot paths; RDKit handles everything requiring its full chemical intelligence.

457 tests passing (17 Rust + 440 Python). Happy to answer questions.

Source: [github.com/Anteneh-T-Tessema/molcore](https://github.com/Anteneh-T-Tessema/molcore)

---

## Reddit r/cheminformatics

**Title:** molcore v0.2: RDKit-compatible cheminformatics — clustering, MMPA, diversity picking, GAT/GIN, Optuna (132× faster fingerprints)

---

v0.2 of molcore is out. If you work at the intersection of cheminformatics and ML, this eliminates the usual glue code: RDKit fingerprint loop → numpy stack → torch.from_numpy → PyG DataLoader.

**Performance** (Apple M-series, CPU, 10k molecules):

| | molcore | RDKit | Speedup |
| --- | --- | --- | --- |
| ECFP4 fingerprints | 2.0M mol/s | 15k mol/s | **132×** |
| Tanimoto 500×10k | 224M pairs/s | 7.7M pairs/s | **29×** |
| PyG conversion (200 mols) | 3.3 ms | 14.4 ms | **4.3×** |

**New in v0.2:**

- **Butina clustering** — Tanimoto-distance, adds `cluster_id` metadata column
- **k-fold / scaffold k-fold CV** — no scaffold leaks between folds
- **MMPA** — single-cut matched molecular pairs with transform SMARTS
- **MaxMin diversity picking** — scales to ~500k molecules
- **GAT and GIN architectures** — drop-in alongside GCN, backward-compatible checkpoints
- **Optuna HP search** — auto-tunes architecture + training HPs over n_trials trials

**ESOL benchmark** (scaffold split, 1128 molecules):

| | RMSE | R² |
| --- | --- | --- |
| GCN untuned | 1.038 | 0.727 |
| Optuna-tuned (30 trials) | 1.090 | 0.709 |

Scaffold split is substantially harder than the random split used in published MoleculeNet baselines (RMSE ≈ 0.58) — not directly comparable.

**Install:**

```bash
pip install molcore           # no conda needed, RDKit installs automatically
pip install molcore[optuna]   # + Optuna tuning
```

**Quickstart in Colab:**
[colab.research.google.com/github/Anteneh-T-Tessema/molcore/blob/main/examples/quickstart.ipynb](https://colab.research.google.com/github/Anteneh-T-Tessema/molcore/blob/main/examples/quickstart.ipynb)

GitHub: [github.com/Anteneh-T-Tessema/molcore](https://github.com/Anteneh-T-Tessema/molcore)

---

## PyG GitHub Discussions / torch-geometric community

**Title:** molcore v0.2 — zero-copy RDKit → PyG pipeline, GAT/GIN/GCN predictor, Optuna tuning, Butina clustering

---

v0.2 of molcore is out. If you're building GNNs on molecular graphs, it eliminates the Python-loop graph construction boilerplate with a Rust/PyO3 zero-copy bridge:

```python
from molcore.molecule import Mol
from molcore.io import MolDataset, MolTorchDataset
from torch_geometric.loader import DataLoader

mol = Mol.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
data = mol.to_pyg()
# data.x          → (N, 9) float32  — 9 node features incl. hybridization, chirality
# data.edge_index → (2, E) int64 COO bidirectional
# data.edge_attr  → (E, 4) float32 bond one-hot

ds = MolDataset.from_sdf("library.sdf")
ds.labels = logp_array
loader = DataLoader(MolTorchDataset(ds), batch_size=32, shuffle=True)
```

Graph construction is **4.3× faster** than a manual Python atom loop at 200 mol/batch.

**New in v0.2 — GNN side:**

```python
from molcore.predictor import PropertyPredictor

# GAT or GIN, drop-in alongside GCN
pred = PropertyPredictor(model_type="gat", hidden=128, epochs=300)
pred.fit(train_ds, val_dataset=val_ds)

# Optuna HP search — finds arch + training HPs automatically
best = PropertyPredictor.tune(train_ds, val_ds, n_trials=30)

# Uncertainty quantification
means, stds = best.predict_with_uncertainty(smiles, n_samples=30)
```

**New in v0.2 — cheminformatics side:**

```python
from molcore.io import MolDataset
from molcore.analysis import mmpa, diversity_pick

# Scaffold k-fold CV (no leakage)
for train, val in ds.scaffold_kfold(k=5):
    ...

# Butina clustering
ds_clustered = ds.cluster(cutoff=0.4)  # adds cluster_id metadata

# MaxMin diversity picking (scales to ~500k)
indices = diversity_pick(ds, n=500)

# Matched molecular pairs
pairs = mmpa(ds)  # [{"core": ..., "smiles_a": ..., "transform": ...}, ...]
```

**ESOL benchmark** (scaffold split, 1128 molecules):

| | RMSE | R² |
| --- | --- | --- |
| GCN, hidden=64, 3 layers | 1.038 | 0.727 |
| Optuna-tuned GAT/GIN/GCN (30 trials) | 1.090 | 0.709 |

**Install:** `pip install molcore` or `pip install molcore[optuna]`

**Colab quickstart:**
[colab.research.google.com/github/Anteneh-T-Tessema/molcore/blob/main/examples/quickstart.ipynb](https://colab.research.google.com/github/Anteneh-T-Tessema/molcore/blob/main/examples/quickstart.ipynb)

GitHub: [github.com/Anteneh-T-Tessema/molcore](https://github.com/Anteneh-T-Tessema/molcore)
