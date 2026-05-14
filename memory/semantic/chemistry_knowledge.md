# Semantic Memory: Chemistry Knowledge

## Fingerprint conventions

- ECFP4 = Morgan radius 2, 2048 bits. Standard for virtual screening.
- ECFP6 = Morgan radius 3. Better for scaffold hopping.
- RDKit and Rust ECFP4 use different hash seeds → bit vectors are NOT interchangeable across backends.
- Fix backend at dataset creation; never mix `rust` and `rdkit` vectors in the same model.

## Tanimoto thresholds (rule of thumb)

- ≥ 0.85: very similar (likely same scaffold)
- 0.65–0.85: similar (possible bioisostere)
- < 0.40: structurally diverse

## Lipinski Ro5

- MW ≤ 500, LogP ≤ 5, HBA ≤ 10, HBD ≤ 5
- Use `rdkit` backend for exact Ro5 assessment
- Rust descriptor col 2 is heavy_atom_count, NOT TPSA — use `rdkit` for TPSA

## Node feature vector (NODE_FEAT_DIM = 9)

Columns of `mol.to_pyg().x`:
0 atomic_num · 1 is_aromatic · 2 formal_charge · 3 num_hs ·
4 degree · 5 in_ring · 6 hybridization (1=sp, 2=sp2, 3=sp3) ·
7 chirality (0=none, 1=@/S, 2=@@/R) · 8 mass_norm (mass/100)

Any GNN built on molcore must use `in_features=9`. Models from before this expansion
(4-feature era) require re-training or a learned projection layer.

## SMARTS patterns — common filters

- Carboxylic acid: `C(=O)O`
- Primary amine: `[NH2]`
- Aromatic ring: `c1ccccc1`
- Michael acceptor (reactive): `[CH2]=[CH]-C=O`
- Aldehyde (reactive): `[CH]=O`
- PAINS filters: use the published Baell & Holloway 2010 SMARTS list with `filter_by_smarts`

## Scaffold concepts

- Murcko scaffold: ring systems + linker atoms, side chains removed
- Generic scaffold: all atoms → C, all bonds → single — for framework-level clustering
- Scaffold split: group molecules by scaffold, assign groups to train/val/test.
  No scaffold appears in more than one split → tests generalization to unseen scaffolds.
  This is the correct evaluation for drug discovery, not random split.

## Chirality

- `@` in SMILES bracket atom = anticlockwise / S configuration (node feature value 1)
- `@@` in SMILES bracket atom = clockwise / R configuration (node feature value 2)
- Captured in node feature col 7 — GNNs can learn stereo-dependent properties (e.g. activity cliffs)

## 3D shape descriptors

- PMI1 ≤ PMI2 ≤ PMI3 always (principal moments of inertia)
- Asphericity ∈ [0, 1]: 0 = perfect sphere, 1 = linear rod
- NPR1 + NPR2 plot classifies shape: disc (flat), rod (linear), sphere (globular)
- Benzene: asphericity ≈ 0.25 (oblate disc), NPR1 ≈ 0.46, NPR2 ≈ 0.66
- Always fix ETKDGv3 seed for reproducibility in downstream comparisons

## Reaction SMARTS conventions

- Format: `reactants>>products` using atom-map numbers `[C:1]` to track atoms
- Unimolecular: one reactant; bimolecular: `.` separates two reactants in the left side
- Empty product list = no match (not an error); ValueError = bad SMARTS syntax
- Common patterns: ester hydrolysis, amide coupling, N-Boc deprotection, reductive amination
- `enumerate_reactions` applies a transform to a library; cap with `max_products` for large libs

## GCN property prediction (PropertyPredictor)

- Architecture: 3-layer GCN → global mean pool → linear head; `in_features=9` (NODE_FEAT_DIM)
- Single-task: `n_outputs=1`, labels shape `(N,)`. Multi-task: `n_outputs=k`, labels shape `(N, k)`.
- Multi-task batch labels stored as `(1, k)` per molecule so PyG batches to `(B, k)`, not `(B*k,)`
- Scheduler: ReduceLROnPlateau(factor=0.5, patience=15); best-val checkpoint restored at end
- MC Dropout uncertainty: keep dropout active at inference; run n_samples forward passes; return mean ± std
- Epistemic std is a relative measure — use for ranking, not as calibrated confidence interval
- Always scaffold-split labelled datasets before training; random split leaks scaffold information

## Data sources

- PubChem: structure lookup, synonyms, basic properties
- ChEMBL: bioactivity (IC50, Ki, Kd) — quality-filtered experimental data
- ZINC20: purchasable compound libraries; tranches by physicochemical properties
