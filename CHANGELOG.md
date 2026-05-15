# Changelog

All notable changes to molcore are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.4.0] — 2026-05-15

- **MMPA double-cut** (`mmpa(smiles, max_cut_bonds=2)`):
  Finds Matched Molecular Pairs that differ by a linker fragment between two
  constant terminal groups. Enables linker-replacement and bioisostere SAR
  (e.g. `-CH₂OCH₂-` → `-CH₂SCH₂-`). Adjacent-bond pairs are skipped;
  linker SMILES are canonicalized symmetrically (`:1`↔`:2` normalized).
  `max_cut_bonds=3+` raises `ValueError`; single-cut behaviour unchanged.

---

## [0.3.0] — 2026-05-15

### Added

- **ADMET module** (`molcore.admet`):
  - `admet_screen(smiles)` — rule-based profiling: Lipinski Ro5, Veber,
    Egan, PAINS, and Brenk structural alerts. No extra dependencies.
  - `admet_screen_df(smiles)` — same, returns a pandas DataFrame.
  - `ADMETPredictor.from_tdc(endpoint)` — random-forest predictor trained
    on any TDC ADMET benchmark (BBB, hERG, AMES, CYP, Caco2, solubility,
    …). Requires `pip install molcore[bio]`.
  - Supports save/load for trained predictors.

- **Protein module** (`molcore.protein`):
  - `ProteinSeq` — immutable protein sequence with ESM-2 embeddings.
  - `ProteinSeq.from_sequence(seq)` — create from amino acid string (validates
    standard 20-letter IUPAC codes, rejects oversize inputs).
  - `ProteinSeq.from_fasta(path)` and `from_fasta_string(text)` — parse
    single- or multi-record FASTA without BioPython.
  - `ProteinSeq.embed(model, pooling)` — ESM-2 mean/CLS/per-residue
    embeddings via HuggingFace `transformers`. Default model:
    `facebook/esm2_t6_8M_UR50D` (8 M params, 320-dim).
  - `ProteinSeq.embed_batch(sequences)` — batch embed returning (N, hidden)
    tensor.
  - `ProteinSeq.to_pyg()` — residue-level PyG graph with 20-dim one-hot
    node features and bidirectional sequential edges.
  - Requires `pip install molcore[bio]` for embedding.

- **BindingDB & TDC data loaders**:
  - `MolDataset.from_tdc(dataset, split)` — load any TDC ADMET or DTI
    dataset (BBB, hERG, BindingDB_Kd, Davis, KIBA, …) directly into a
    `MolDataset` with labels and metadata.
  - `MolDataset.from_bindingdb(affinity, target)` — filter BindingDB by
    affinity type (Kd / IC50 / Ki / EC50) and optional target name/UniProt
    substring. Labels are log-transformed to pIC50 by default.
  - `databases.tdc_dataset(name)` — low-level TDC split loader.
  - `databases.bindingdb_search(affinity, target)` — returns
    `BindingRecord` objects with SMILES, protein sequence, and affinity.
  - Requires `pip install molcore[bio]` (PyTDC).

- **New optional dependency group** `bio`:
  `pip install molcore[bio]` installs `transformers`, `sentencepiece`,
  `PyTDC`, `scikit-learn`, and `pandas`.

### Security (also in v0.2.x patch)

- `molcore/_validation.py`: `validate_smiles` (10 000-char cap),
  `validate_molblock` (1 MB cap), `validate_path` (null-byte + extension check).
- Wired into `Mol.from_smiles`, `Mol.from_molblock`, `MolDataset.from_sdf`,
  `write_sdf`, `read_parquet`, `write_parquet`.
- `.github/workflows/security.yml`: cargo-audit, pip-audit, bandit, gitleaks.
- `.github/dependabot.yml`: weekly auto-updates for Actions, Cargo, pip.
- `ci.yml`: explicit `permissions: contents: read`.
- `SECURITY.md`: responsible disclosure policy, 14-day SLA for critical issues.

---

## [0.2.0] — 2026-05-15

### Features added

- **GAT and GIN architectures** (`PropertyPredictor(model_type="gat"|"gin"|"gcn")`):
  drop-in alternatives to GCN backed by a unified `_MolGNN` module. `model_type` is
  stored in saved checkpoints; pre-v0.2 checkpoints without the key default to `"gcn"`
  for full backward-compatibility.
- **MaxMin diversity picking** (`diversity_pick`, `MolDataset.diversity_pick`):
  O(n × N) iterative selection in Tanimoto fingerprint space — each step picks the
  molecule with maximum minimum distance to all already-selected molecules. Returns the
  N most structurally diverse indices. Scales to ~500k; see docstring for guidance
  beyond that.
- **MMPA — single-cut fragmentation** (`mmpa`): fragments acyclic bonds to produce
  matched molecular pairs; each result dict contains `core`, `smiles_a/b`, `mol_a/b`,
  and a `transform` SMARTS. Double-cut (`max_cut_bonds=2`) is planned for v0.3.
- **Butina clustering** (`butina_cluster`, `MolDataset.cluster`): Tanimoto-distance
  clustering via RDKit's Butina algorithm. `cluster(cutoff=0.4)` adds a `cluster_id`
  metadata column; cluster 0 is always the largest cluster. Invalid SMILES get ID -1.
- **k-fold cross-validation** (`MolDataset.kfold`, `MolDataset.scaffold_kfold`):
  `kfold(k=5)` produces random splits; `scaffold_kfold(k=5)` assigns whole Murcko
  scaffold groups to folds so no scaffold leaks between train and val.
- **Optuna hyperparameter search** (`PropertyPredictor.tune`): searches hidden dim,
  n_layers, dropout, lr, batch_size, and model_type over n_trials Optuna trials;
  restores the best-seen model weights. Requires `pip install molcore[optuna]`.
- `optuna` optional dependency group in `pyproject.toml`; added to `[all]`.

### Benchmarks

- ESOL scaffold-split (Delaney 2004, 1128 molecules, 80/10/10 scaffold split):
  - Untuned GCN (hidden=64, 3 layers, 300 epochs): RMSE = 1.038, R² = 0.727
  - Optuna-tuned (30 trials × 100 epochs, best: hidden=128, 4 layers, dropout=0.21,
    lr=0.00222, batch=16, 300-epoch final run): RMSE = 1.090, R² = 0.709
  - Note: scaffold split is substantially harder than the random split used in published
    MoleculeNet baselines (RMSE ≈ 0.58); these results are not directly comparable.

### Testing

- 457 tests passing (17 Rust + 440 Python/evals)
- New test modules: `test_diversity_and_gnn.py` (19 tests), `test_clustering_and_cv.py`
  (21 tests), `test_mmpa.py` (12 tests), `test_reliability.py` (13 tests)
- Frozen fixture `tests/python/fixtures/gcn_pre_v02.pt` permanently guards pre-v0.2
  checkpoint backward-compatibility

### Fixed (reliability)

- `tanimoto_matrix`: raises `ValueError` when query/library have different nbits
  instead of silently computing wrong scores via zip truncation.
- `MolDataset.scaffold_split`: duplicate SMILES no longer silently drop — uses a
  per-SMILES deque so every occurrence is assigned to exactly one split.
- Parquet multi-label roundtrip: `read_parquet` now recovers `label_0`…`label_k`
  columns written by multi-label `write_parquet`.
- `neutralize`/`strip_salts`: use the module-level `_std_objects()` cache instead of
  constructing fresh standardizer objects on every call.
- RDKit firewall: `io.py` had a bare `from rdkit import Chem` violating the invariant
  that all rdkit imports are isolated in `rdkit_bridge.py`; replaced by new
  `rdkit_bridge.mol_to_smiles(rdmol)` helper.
- `write_sdf`: validates that all `properties` value lists have the same length as
  `smiles_list` upfront, raising `ValueError` instead of crashing mid-write.

---

## [0.1.0] — 2026-05-14

Initial public release.

### Rust core (`crates/molcore-core`)

- Built-in recursive-descent SMILES parser (`ingest.rs`) — no `purr` dependency
- `ecfp4_batch()` — Rayon parallel Morgan fingerprints, 2048-bit, u64 bit-pack; **88–132× faster than RDKit**
- `tanimoto_matrix()` — u64 `count_ones()`, Rayon parallel; **29× faster than RDKit BulkTanimoto** at scale
- `mol_to_graph_arrays()` — zero-copy `IntoPyArray` node/edge arrays for PyG; **4.3× faster** than manual Python construction
- `calc_descriptors_batch()` — fast Rust MW/LogP/heavy-atom-count
- 17 Rust unit tests: ingest, fingerprints, graph arrays, implicit H

### Python package (`molcore/`)

- `Mol` — frozen dataclass; `from_smiles`, `from_molblock`, `neutralize`, `strip_salts`, `standardize`, `react`, `scaffold`, `conformers`, `to_pyg`, `to_pyg_hetero`, `to_svg`, `to_png`, `_repr_svg_` (Jupyter auto-render)
- `featurize_smiles()` — batch-first entry point; `kind`: `ecfp4` | `maccs` | `atom_pairs` | `topological_torsions` | `rdkit`; `backend`: `rust` | `rdkit`
- `MolDataset` — columnar container: `from_smiles`, `from_sdf` (incl. `.sdf.gz`), `from_chembl`, `from_zinc`, `write_sdf`, `write_parquet`, `read_parquet`, `to_dataframe`, `from_dataframe`, `scaffold_split`, `filter`, `to_pyg_list`, `draw_grid`, `_repr_html_` (Jupyter)
- `MolTorchDataset` — `torch.utils.data.Dataset` wrapper for `DataLoader`
- `PropertyPredictor` — 3-layer GCN, `fit`, `predict`, `predict_with_uncertainty` (MC Dropout), `score`, `save`, `load`
- `pandas_tools` — `load_sdf`, `write_sdf`, `add_mol_column`, `add_descriptors`, `add_fingerprints`, `filter_by_smarts`, `add_scaffold_column`, `standardize_smiles`
- `rdkit_bridge` — all RDKit calls isolated; includes `standardize`, `find_mcs`, `rgroup_decompose`, `maccs_keys`, `atom_pairs_fp`, `topological_torsions_fp`, `rdkit_path_fp`, `calc_named_descriptors` (presets: `lipinski` | `druglike` | `all`), `from_sdf_file`, `from_molblock`, `write_sdf`, `mol_to_svg`, `mol_to_png`, `mols_to_grid_svg`
- `ChemRAG` — iterative chemical literature retrieval (PubChem, ChEMBL, local tools)

### AGENT_SKILLS scaffold

- 7 skills: `fingerprint`, `similarity_search`, `molecular_featurization`, `property_prediction`, `property_prediction_ml`, `reaction`, `agentic_rag`
- 9 local tools, MCP configs for PubChem and ChEMBL
- OpenTelemetry observability wiring

### Tests

- 392 tests passing (17 Rust + 375 Python/evals)
- Regression suite guards: node feature dim, fingerprint determinism, scaffold split reproducibility, aromaticity flags (benzene, pyridine, cyclohexane, thiophene, imidazole, aspirin, indane, naphthalene)

### CI / Distribution

- Multi-platform wheel builds: Linux x86_64/aarch64, macOS arm64/x86_64, Windows x64 (`maturin-action`)
- Smoke test before publish; PyPI trusted publishing (OIDC, no stored token)
- Python 3.11 and 3.12 test matrix

### Documentation

- `docs/migrating_from_rdkit.md` — full RDKit → molcore API mapping with porting checklist
- `examples/quickstart.ipynb` — Colab-ready notebook, 9 sections, end-to-end
- `benchmarks/bench_e2e.py` — full pipeline benchmark vs vanilla RDKit baseline

### Performance (Apple M-series, CPU-only)

| Metric | molcore | RDKit | Speedup |
| --- | --- | --- | --- |
| ECFP4 @ 10k SMILES | 2.0M mol/s | 15k mol/s | **132×** |
| Tanimoto 500×10k | 224M pairs/s | 7.7M pairs/s | **29×** |
| PyG conversion (200 mols) | 3.3 ms | 14.4 ms | **4.3×** |
| End-to-end pipeline (500 mols) | 404 ms | 425 ms | 1.1× |

### Design invariants (never relaxed between releases)

1. `Mol` is always immutable — transforms return new instances
2. RDKit is never called in hot paths — only `rdkit_bridge.py` imports rdkit
3. All Rust→Python array transfers use `IntoPyArray` — no Python-side loops
4. Batch API is primary — per-mol methods are convenience wrappers
5. Backend flags are explicit — `"rust"` | `"rdkit"`, never auto-selected
