# Migrating from RDKit to molcore

molcore is designed so that existing RDKit scripts can be ported incrementally.
The sections below map RDKit patterns to their molcore equivalents.
You don't need to rewrite everything at once — use molcore where it helps
(batch processing, ML pipelines, PyG integration) and keep RDKit where you need it.

---

## Installation

```bash
pip install molcore          # Rust compiled into the wheel — no conda required
# RDKit is a declared dependency and installs automatically
```

---

## Core concepts

| Concept | RDKit | molcore |
|---|---|---|
| Molecule object | `rdkit.Chem.Mol` (mutable) | `molcore.Mol` (frozen dataclass) |
| SMILES → molecule | `Chem.MolFromSmiles(s)` | `Mol.from_smiles(s)` |
| Mol block → molecule | `Chem.MolFromMolBlock(b)` | `Mol.from_molblock(b)` |
| Canonical SMILES | `Chem.MolToSmiles(mol)` | `mol.smiles` (property, set at construction) |
| Mutate a molecule | edit atoms/bonds in-place | transforms return **new** Mol |
| RDKit mol from molcore | n/a | `mol.rdmol()` — fresh on every call |

### Why is `Mol` frozen?

The frozen invariant makes molecules safe to cache, pass between threads, and
store in numpy/Arrow arrays without defensive copying. Transforms (`neutralize`,
`standardize`, `strip_salts`) always return a **new** Mol; the original is never
touched.

---

## Parsing and I/O

### SMILES

```python
# RDKit
from rdkit import Chem
mol = Chem.MolFromSmiles("CCO")
if mol is None:
    raise ValueError("bad SMILES")

# molcore
from molcore.molecule import Mol
mol = Mol.from_smiles("CCO")   # raises MolIngestionError on bad SMILES
```

### SDF files

```python
# RDKit
from rdkit.Chem import PandasTools
df = PandasTools.LoadSDF("library.sdf")

# molcore — DataFrame API
import molcore.pandas_tools as mpt
df = mpt.load_sdf("library.sdf")          # adds 'smiles' + 'Mol' columns, all SD props
df = mpt.load_sdf("library.sdf.gz")       # gzip transparent

# molcore — dataset API (faster for ML pipelines)
from molcore.io import MolDataset
ds = MolDataset.from_sdf("library.sdf", compute_fps=True, compute_desc=True)
```

### Writing SDF

```python
# RDKit
from rdkit.Chem import PandasTools
PandasTools.WriteSDF(df, "output.sdf", molColName="ROMol", properties=["logp"])

# molcore — DataFrame API
import molcore.pandas_tools as mpt
mpt.write_sdf(df, "output.sdf")           # all non-Mol columns become SD props

# molcore — dataset API
ds.write_sdf("output.sdf")
```

### Parquet (molcore-only, no RDKit equivalent)

```python
ds.write_parquet("library.parquet")       # snappy-compressed, Arrow columnar
ds2 = MolDataset.read_parquet("library.parquet")
```

---

## Standardization

```python
# RDKit (fragmented API — three separate calls)
from rdkit.Chem.MolStandardize import rdMolStandardize
mol = Chem.MolFromSmiles("[Na+].OC(=O)c1ccccc1")
mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
mol = rdMolStandardize.Uncharger().uncharge(mol)
mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
result = Chem.MolToSmiles(mol)

# molcore — one call
from molcore.rdkit_bridge import standardize
result = standardize("[Na+].OC(=O)c1ccccc1")

# or via Mol object
mol = Mol.from_smiles("[Na+].OC(=O)c1ccccc1")
clean = mol.standardize()   # → new Mol

# or via DataFrame
import molcore.pandas_tools as mpt
df = mpt.standardize_smiles(df)
```

---

## Fingerprints

### ECFP4 (Morgan)

```python
# RDKit — per-molecule loop (slow)
from rdkit.Chem import AllChem, DataStructs
import numpy as np
fps = []
for smi in smiles:
    mol = Chem.MolFromSmiles(smi)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    arr = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    fps.append(arr)
fps = np.stack(fps)   # (N, 2048) uint8

# molcore — Rust Rayon parallel, zero-copy (88–132× faster)
from molcore.pipeline import featurize_smiles
fps = featurize_smiles(smiles, backend="rust")   # torch.Tensor (N, 2048) uint8

# Exactly bit-identical to RDKit output (for legacy models)
fps = featurize_smiles(smiles, backend="rdkit")
```

### Other fingerprint types

```python
# RDKit
from rdkit.Chem import MACCSkeys
fp = MACCSkeys.GenMACCSKeys(mol)

# molcore
fps = featurize_smiles(smiles, kind="maccs")               # (N, 167)
fps = featurize_smiles(smiles, kind="atom_pairs")          # (N, 2048)
fps = featurize_smiles(smiles, kind="topological_torsions") # (N, 2048)
fps = featurize_smiles(smiles, kind="rdkit")               # (N, 2048) RDKit path-based
```

---

## Descriptors

### Basic (MW, LogP, TPSA)

```python
# RDKit
from rdkit.Chem import Descriptors
mw   = Descriptors.MolWt(mol)
logp = Descriptors.MolLogP(mol)
tpsa = Descriptors.TPSA(mol)

# molcore — batch, returns (N, 3) float32
from molcore.featurizers.descriptors import calc_descriptors
desc = calc_descriptors(smiles, backend="rdkit")   # MW, LogP, TPSA
```

### Named descriptors (any of the ~200 RDKit descriptors)

```python
# RDKit — must look up each function by name
from rdkit.Chem import Descriptors as D
vals = [D.MolWt(mol), D.TPSA(mol), D.NumHDonors(mol), D.NumHAcceptors(mol)]

# molcore — batch, by name
from molcore.rdkit_bridge import calc_named_descriptors
arr, col_names = calc_named_descriptors(smiles, names=["MolWt", "TPSA", "NumHDonors"])
# → (N, 3) float32 · invalid SMILES → NaN row

# or use a preset
arr, col_names = calc_named_descriptors(smiles, preset="lipinski")   # 7 descriptors
arr, col_names = calc_named_descriptors(smiles, preset="druglike")   # 15 descriptors
arr, col_names = calc_named_descriptors(smiles, preset="all")        # all ~200
```

### DataFrame integration

```python
# RDKit + pandas (manual)
df["MolWt"] = df["smiles"].apply(lambda s: D.MolWt(Chem.MolFromSmiles(s)))

# molcore
import molcore.pandas_tools as mpt
df = mpt.add_descriptors(df, preset="lipinski")     # adds 7 columns in one call
df = mpt.add_descriptors(df, names=["MolWt", "TPSA"])
```

---

## Substructure search

```python
# RDKit
patt = Chem.MolFromSmarts("c1ccncc1")
matches = [mol for mol in mols if mol.HasSubstructMatch(patt)]

# molcore
from molcore.rdkit_bridge import filter_by_smarts
hits = filter_by_smarts(smiles, "c1ccncc1")          # list[str] — SMILES of matches

# on a DataFrame
import molcore.pandas_tools as mpt
hits_df = mpt.filter_by_smarts(df, "c1ccncc1")

# on a Mol object
mol = Mol.from_smiles("c1cccnc1CC")
mol.matches("c1ccncc1")                              # True
mol.find_substructures("c1ccncc1")                  # [(0, 1, 2, 3, 4, 5)]
```

---

## Scaffold analysis

```python
# RDKit
from rdkit.Chem.Scaffolds import MurckoScaffold
scaffold = MurckoScaffold.GetScaffoldForMol(mol)
scaffold_smi = Chem.MolToSmiles(scaffold)

# molcore
from molcore.rdkit_bridge import murcko_scaffold
scaffold_smi = murcko_scaffold("c1ccccc1CC")         # → "c1ccccc1"

mol = Mol.from_smiles("c1ccccc1CC")
sc = mol.scaffold()                                  # → Mol

# generic framework scaffold
mol.scaffold(generic=True)
```

### Scaffold-based train/val/test split

```python
# molcore (no direct RDKit equivalent with this API)
from molcore.io import MolDataset
ds = MolDataset.from_smiles(smiles)
train_ds, val_ds, test_ds = ds.scaffold_split(train_frac=0.8, val_frac=0.1)
```

---

## Tanimoto similarity

```python
# RDKit — BulkTanimoto, sequential
from rdkit.Chem import DataStructs
sims = DataStructs.BulkTanimotoSimilarity(query_fp, library_fps)

# molcore — Rust Rayon parallel (4–30× faster at scale)
from molcore._molcore import tanimoto_matrix
from molcore.pipeline import featurize_smiles

q_fps = featurize_smiles(query_smiles).numpy()
l_fps = featurize_smiles(library_smiles).numpy()
sim   = tanimoto_matrix(q_fps, l_fps)              # (Q, L) float32
```

---

## MCS (Maximum Common Substructure)

```python
# RDKit
from rdkit.Chem import rdFMCS
result = rdFMCS.FindMCS(mols, timeout=5)
print(result.smartsString)

# molcore
from molcore.rdkit_bridge import find_mcs
smarts = find_mcs(smiles_list, timeout=5)
```

---

## R-Group decomposition

```python
# RDKit
from rdkit.Chem.rdRGroupDecomposition import RGroupDecompose
groups, unmatched = RGroupDecompose([core], mols, asSmiles=True, asRows=True)

# molcore
from molcore.rdkit_bridge import rgroup_decompose
rows = rgroup_decompose("c1ccc([*:1])cc1", smiles_list)
# → [{"Core": "...", "R1": "..."}, ...]
```

---

## 2D Depiction

```python
# RDKit
from rdkit.Chem import Draw
img = Draw.MolToImage(mol, size=(300, 200))
img.save("mol.png")

# molcore
mol = Mol.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
mol.to_png("mol.png")                              # 300×200 default
svg = mol.to_svg()                                 # SVG string

# Jupyter: just evaluate the Mol object in a cell
mol        # → renders 2D structure inline via _repr_svg_

# Grid of structures
from molcore.rdkit_bridge import mols_to_grid_svg
svg = mols_to_grid_svg(smiles_list, mols_per_row=4)

# or from a dataset
ds.draw_grid(n=20, mols_per_row=4)
ds          # → renders 8-molecule grid inline via _repr_html_
```

---

## Conformers and 3D

```python
# RDKit
from rdkit.Chem import AllChem
mol_h = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol_h)
conf = mol_h.GetConformer()
positions = conf.GetPositions()

# molcore
mol = Mol.from_smiles("CCO")
confs = mol.conformers(n_confs=1, force_field="MMFF94")
# → list of (n_atoms, 3) float64 numpy arrays (H stripped)
```

---

## Reactions

```python
# RDKit
rxn = AllChem.ReactionFromSmarts("[C:1](=O)O[CH3]>>[C:1](=O)[OH]")
products = rxn.RunReactants((mol,))

# molcore
from molcore.rdkit_bridge import react, react_bimolecular, enumerate_reactions
products = react("CC(=O)OC", "[C:1](=O)O[CH3]>>[C:1](=O)[OH]")  # list[str]

# or via Mol
mol = Mol.from_smiles("CC(=O)OC")
mol.react("[C:1](=O)O[CH3]>>[C:1](=O)[OH]")       # list[Mol]
```

---

## GNN / PyG integration

This is where molcore has no RDKit equivalent — it's the main reason to use molcore
if you're building ML models.

```python
# RDKit + PyG — manual feature extraction (~50 lines of boilerplate)
# ... atom feature loop, bond feature loop, edge index construction ...

# molcore — zero-copy, 9 node features, bidirectional edge index
from molcore.molecule import Mol
mol  = Mol.from_smiles("c1ccccc1C(=O)O")
data = mol.to_pyg()
# data.x          → (N, 9)  float32 node features
# data.edge_index → (2, E)  int64  COO bidirectional
# data.edge_attr  → (E, 4)  float32 bond one-hot

# Full dataset → DataLoader in 3 lines
from molcore.io import MolDataset, MolTorchDataset
from torch_geometric.loader import DataLoader
ds     = MolDataset.from_smiles(smiles)
ds.labels = logp_values
loader = DataLoader(MolTorchDataset(ds), batch_size=32, shuffle=True)

# Pretrained GCN predictor
from molcore.predictor import PropertyPredictor
pred = PropertyPredictor(hidden=64, epochs=50)
pred.fit(train_ds, val_dataset=val_ds)
pred.predict(["CCO", "c1ccccc1"])                  # numpy array
pred.predict_with_uncertainty(["CCO"], n_samples=30)  # (mean, std) — MC Dropout
```

---

## Checklist: porting an existing script

- [ ] Replace `Chem.MolFromSmiles` → `Mol.from_smiles` (or keep using rdmol via `mol.rdmol()`)
- [ ] Replace SDF loading with `mpt.load_sdf` or `MolDataset.from_sdf`
- [ ] Replace fingerprint loops with `featurize_smiles(smiles, backend="rdkit")` for parity, then switch to `backend="rust"` after validating
- [ ] Replace descriptor calls with `calc_named_descriptors(smiles, preset="lipinski")` or `mpt.add_descriptors(df)`
- [ ] Replace `Chem.Draw.MolToImage` with `mol.to_png()` or notebook auto-render via `mol._repr_svg_`
- [ ] Any ML pipeline using fingerprints or graphs → switch to `MolDataset` + `MolTorchDataset`
- [ ] Standardization calls → `mol.standardize()` or `mpt.standardize_smiles(df)`
