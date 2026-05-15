# Launch announcements

Ready-to-post text for each channel. Edit the Colab URL once the notebook is on main.

---

## RDKit mailing list (rdkit-discuss@lists.sourceforge.net)

**Subject:** molcore: Rust-accelerated cheminformatics with native PyG integration

Hi all,

I wanted to share a library I've been building: **molcore**, an open-source cheminformatics toolkit that wraps RDKit where it matters and accelerates the hot paths in Rust.

The core idea: molecules as batch-first, zero-copy tensors that flow directly into PyTorch Geometric — no Python loops, no intermediate copies.

**What it does:**

- ECFP4 fingerprints via Rust Rayon: **132× faster** than RDKit at 10k molecules
- Tanimoto matrix: **29× faster** at scale via u64 `count_ones()`
- PyG `Data` export: **4.3× faster** than manual Python feature extraction
- GCN property predictor: **RMSE = 0.937** on ESOL solubility (1,128 molecules, 3-layer GCN, 300 epochs)
- SDF/gzip I/O, Parquet storage, pandas DataFrame bridge
- Full descriptor set (all ~200 RDKit descriptors, with `lipinski`/`druglike`/`all` presets)
- MACCS keys, atom pairs, topological torsions, RDKit path fingerprints
- 2D depiction with Jupyter `_repr_svg_` auto-render
- Standardization, MCS, R-group decomposition
- Scaffold split, SMARTS filter, reaction SMARTS

**Install:**

```bash
pip install molcore
```

No conda. RDKit installs automatically as a declared dependency.

**Quickstart notebook (Colab):**
[colab.research.google.com/github/Anteneh-T-Tessema/molcore/blob/main/examples/quickstart.ipynb](https://colab.research.google.com/github/Anteneh-T-Tessema/molcore/blob/main/examples/quickstart.ipynb)

**Migration guide** (RDKit → molcore API mapping):
[github.com/Anteneh-T-Tessema/molcore/blob/main/docs/migrating_from_rdkit.md](https://github.com/Anteneh-T-Tessema/molcore/blob/main/docs/migrating_from_rdkit.md)

The design keeps RDKit as the chemistry authority — all RDKit calls are isolated in a single `rdkit_bridge.py` file. The Rust core handles the batch hot paths; RDKit handles everything that requires its full chemical intelligence.

Happy to answer questions. Source: [github.com/Anteneh-T-Tessema/molcore](https://github.com/Anteneh-T-Tessema/molcore)

---

## Reddit r/cheminformatics

**Title:** molcore: RDKit-compatible cheminformatics toolkit with 35–132× faster fingerprints and native PyG integration

---

If you work at the intersection of cheminformatics and ML, you've probably written the same glue code a dozen times: RDKit fingerprint loop → numpy stack → torch.from_numpy → PyG DataLoader. molcore eliminates that.

**What it is:** an open-source Python library with a Rust performance core, RDKit compatibility, and native PyTorch Geometric integration.

**The numbers** (Apple M-series, CPU, 10k molecules):

| | molcore | RDKit | Speedup |
| --- | --- | --- | --- |
| ECFP4 fingerprints | 2.0M mol/s | 15k mol/s | **132×** |
| Tanimoto 500×10k | 224M pairs/s | 7.7M pairs/s | **29×** |
| PyG conversion | 3.3 ms/200 mols | 14.4 ms | **4.3×** |
| ESOL RMSE (GCN, 300 ep) | 0.937 | — | — |

These are measured on Apple M-series. Linux x86_64 numbers are similar or faster — Rust's `u64::count_ones()` compiles to a single `POPCNT` instruction on x86_64 with SSE4.2, and Rayon's thread pool scales with core count. molcore is not Mac-only.

**Install:** `pip install molcore` — no conda, RDKit installs automatically.

**Key features:**

- SDF/gzip I/O and Parquet storage
- Full descriptor set: all ~200 RDKit descriptors with `lipinski`/`druglike`/`all` presets
- `MolDataset.from_sdf()`, `.scaffold_split()`, `.to_dataframe()`, `.draw_grid()`
- `pandas_tools` module: `load_sdf`, `add_descriptors`, `add_fingerprints`, `filter_by_smarts`
- Mol objects auto-render in Jupyter via `_repr_svg_`
- GCN predictor with MC Dropout uncertainty (`predict_with_uncertainty`)
- Standardization, MCS, R-group decomposition
- Everything else (conformers, reactions, descriptors) delegates to RDKit

**Quickstart in Colab:**
[colab.research.google.com/.../quickstart.ipynb](https://colab.research.google.com/github/Anteneh-T-Tessema/molcore/blob/main/examples/quickstart.ipynb)

**Migration guide:**
[github.com/Anteneh-T-Tessema/molcore/blob/main/docs/migrating_from_rdkit.md](https://github.com/Anteneh-T-Tessema/molcore/blob/main/docs/migrating_from_rdkit.md)

GitHub: [github.com/Anteneh-T-Tessema/molcore](https://github.com/Anteneh-T-Tessema/molcore)

---

## PyG GitHub Discussions / torch-geometric community

**Title:** molcore — zero-copy RDKit → PyG pipeline, 4.3× faster graph construction

---

If you're building GNNs on molecular graphs, molcore might save you a lot of boilerplate.

The current workflow for most people looks like this: parse SMILES with RDKit, manually loop over atoms/bonds to build feature tensors, call `torch.tensor()` per molecule, assemble a `Data` object. It works, but it's slow and fragile.

molcore does this via a Rust/PyO3 bridge with `IntoPyArray` → `torch.from_numpy()` — no Python-side loops, zero copies:

```python
from molcore.molecule import Mol
from molcore.io import MolDataset, MolTorchDataset
from torch_geometric.loader import DataLoader

mol = Mol.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
data = mol.to_pyg()
# data.x          → (N, 9) float32 — 9 node features including hybridization, chirality
# data.edge_index → (2, E) int64 COO bidirectional
# data.edge_attr  → (E, 4) float32 bond one-hot

# Full dataset → DataLoader in 3 lines
ds = MolDataset.from_sdf("library.sdf")
ds.labels = logp_array
loader = DataLoader(MolTorchDataset(ds), batch_size=32, shuffle=True)
```

The graph construction is **4.3× faster** than manual Python atom-loop construction at 200 molecules/batch.

There's also a bundled GCN predictor with MC Dropout uncertainty and a validated ESOL benchmark (RMSE = 0.937, 1,128 molecules, 3-layer GCN):

```python
from molcore.predictor import PropertyPredictor

pred = PropertyPredictor(hidden=128, epochs=300)
pred.fit(train_ds, val_dataset=val_ds)
means, stds = pred.predict_with_uncertainty(smiles, n_samples=30)
```

**Install:** `pip install molcore`

**Colab quickstart:**
[colab.research.google.com/.../quickstart.ipynb](https://colab.research.google.com/github/Anteneh-T-Tessema/molcore/blob/main/examples/quickstart.ipynb)

GitHub: [github.com/Anteneh-T-Tessema/molcore](https://github.com/Anteneh-T-Tessema/molcore)
