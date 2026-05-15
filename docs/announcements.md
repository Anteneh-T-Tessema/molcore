# Launch announcements — v0.4.0

Ready-to-post text for each channel. Copy-paste as-is.

---

## RDKit mailing list (rdkit-discuss@lists.sourceforge.net)

**Subject:** molcore v0.4: ADMET profiling, ESM-2 protein embeddings, MMPA double-cut, 407 tests

Hi all,

I wanted to share **molcore v0.4**, an open-source cheminformatics / ML toolkit that wraps RDKit where it matters and accelerates hot paths in Rust.

**Performance** (Apple M-series, CPU, 10 k molecules):

| Metric | molcore | RDKit | Speedup |
|---|---|---|---|
| ECFP4 fingerprints | 2.0 M mol/s | 15 k mol/s | **132×** |
| Tanimoto 500×10 k | 224 M pairs/s | 7.7 M pairs/s | **29×** |
| PyG `Data` export (200 mols) | 3.3 ms | 14.4 ms | **4.3×** |

**What's new across v0.3 and v0.4:**

- **ADMET profiling** (`molcore.admet`): rule-based screening — Lipinski Ro5, Veber, Egan, PAINS, Brenk — with no extra dependencies. ML predictors trained on TDC benchmarks (BBB, hERG, AMES, CYP, Caco2, solubility) via `pip install molcore-chem[bio]`.
- **Protein sequences** (`molcore.protein`): `ProteinSeq` with FASTA parsing (no BioPython), ESM-2 embeddings via HuggingFace Transformers, and residue-level PyG graphs.
- **BindingDB & TDC data loaders**: `MolDataset.from_tdc(dataset)` and `MolDataset.from_bindingdb(affinity, target)` load any TDC ADMET or DTI dataset directly.
- **MMPA double-cut** (`mmpa(smiles, max_cut_bonds=2)`): finds matched molecular pairs differing by a linker between two constant terminal groups — enables bioisostere linker replacement and scaffold-hopping SAR.
- **CLI `admet-screen`**: `molcore admet-screen mols.smi` runs ADMET profiling from the terminal and writes a TSV.
- **Input validation**: 10 000-char SMILES cap, 1 MB Mol block cap, null-byte path guard — wired into all public entry points.
- **Security pipeline**: cargo-audit, pip-audit, bandit, gitleaks, Dependabot, and SECURITY.md.

**Earlier highlights (v0.1–v0.2):**

- Butina clustering, scaffold k-fold CV, diversity picking (MaxMin), GAT/GIN/GCN architectures, Optuna HP search
- Single-cut MMPA, MCS, R-group decomposition, reaction enumeration
- Depiction (`to_svg`, `to_png`, `draw_grid`), gradient-based atom attribution (`atom_attribution`, `integrated_gradients`)

**407 tests** (17 Rust + 390 Python), zero failures.

**Install:**

```bash
pip install molcore-chem      # core (Rust + RDKit, no extras needed)
pip install molcore-chem[bio]      # + ADMET ML, ESM-2, TDC/BindingDB loaders
pip install molcore-chem[optuna]   # + Optuna HP search
pip install molcore-chem[all]      # everything
```

**Quickstart (Colab):**
https://colab.research.google.com/github/Anteneh-T-Tessema/molcore/blob/main/examples/quickstart.ipynb

**Migration guide (RDKit → molcore):**
https://github.com/Anteneh-T-Tessema/molcore/blob/main/docs/migrating_from_rdkit.md

**Source:** https://github.com/Anteneh-T-Tessema/molcore

The design keeps RDKit as the chemistry authority — all RDKit calls are isolated in `rdkit_bridge.py`. The Rust core (PyO3 + Rayon) handles batch hot paths.

Happy to answer questions.

---

## Reddit r/cheminformatics

**Title:** molcore v0.4: ADMET screening, ESM-2 protein embeddings, MMPA double-cut — 132× faster fingerprints, 407 tests

---

v0.4 of **molcore** is out. It eliminates the usual Python glue between RDKit, numpy, and PyTorch Geometric.

**Performance** (Apple M, CPU, 10 k molecules):

| | molcore | RDKit | Speedup |
|---|---|---|---|
| ECFP4 | 2.0 M mol/s | 15 k mol/s | **132×** |
| Tanimoto 500×10 k | 224 M pairs/s | 7.7 M pairs/s | **29×** |
| PyG export (200 mols) | 3.3 ms | 14.4 ms | **4.3×** |

**New in v0.3–v0.4:**

**ADMET profiling (no extra deps):**
```python
from molcore.admet import admet_screen
profiles = admet_screen(["CC(=O)Oc1ccccc1C(=O)O", "CCC1NC(=O)..."])
print(profiles[0].lipinski_pass, profiles[0].druglike, profiles[0].pains_alerts)
```

**ESM-2 protein embeddings (`pip install molcore-chem[bio]`):**
```python
from molcore.protein import ProteinSeq
p = ProteinSeq.from_sequence("MKTLLILAVLCLGFAQAS")
emb = p.embed()          # (320,) mean-pooled ESM-2 t6
graph = p.to_pyg()       # residue-level PyG graph
```

**MMPA double-cut (linker replacement):**
```python
from molcore.rdkit_bridge import mmpa
pairs = mmpa(["c1ccccc1CCc1ccccc1", "c1ccccc1CCCc1ccccc1"], max_cut_bonds=2)
# finds pairs differing by linker: -CH2CH2- vs -CH2CH2CH2-
```

```bash
pip install molcore-chem
pip install molcore-chem[bio]   # + ADMET ML, ESM-2, TDC/BindingDB
```

407 tests, zero failures. Source: https://github.com/Anteneh-T-Tessema/molcore

---

## PyG GitHub Discussions / torch-geometric community

**Title:** molcore v0.4 — zero-copy RDKit→PyG pipeline with ADMET, ESM-2 protein embeddings, MMPA double-cut

---

Hi PyG community,

Sharing **molcore v0.4**, a cheminformatics toolkit built around PyG as the primary ML target.

**The core abstraction:**

```python
from molcore.io import MolDataset
from molcore.predictor import PropertyPredictor

ds = MolDataset.from_smiles(smiles, labels=labels)
pyg_list = ds.to_pyg_list()            # list[torch_geometric.data.Data]

pred = PropertyPredictor(model_type="gin", hidden=128, n_layers=4)
pred.fit(ds, verbose=True)
mean, std = pred.predict_with_uncertainty(test_smiles)
```

Zero Python loops — the Rust core hands `int64`/`float32` arrays directly to PyG via `IntoPyArray`.

**New in v0.4:**

- `ProteinSeq.to_pyg()` — residue-level graph with 20-dim one-hot node features, ready for protein GNNs
- `MolDataset.from_tdc(dataset)` and `from_bindingdb(affinity, target)` — load TDC ADMET / BindingDB DTI datasets into `MolDataset`
- MMPA double-cut for linker-based SAR

**Architectures:** GCN, GAT, GIN (via `model_type` flag, backward-compatible checkpoints)

**Explainability:**
```python
from molcore.explainability import atom_attribution, integrated_gradients
scores = atom_attribution(model, pyg_data)   # (N,) per-atom importance
ig     = integrated_gradients(model, pyg_data, steps=50)
```

```bash
pip install molcore-chem
pip install molcore-chem[bio]   # + protein embeddings, TDC loaders
```

Source: https://github.com/Anteneh-T-Tessema/molcore
Quickstart: https://colab.research.google.com/github/Anteneh-T-Tessema/molcore/blob/main/examples/quickstart.ipynb
