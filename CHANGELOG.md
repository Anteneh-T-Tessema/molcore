# Changelog

All notable changes to molcore are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased] — v0.2.0

### Added

- **Butina clustering** (`butina_cluster`, `MolDataset.cluster`): Tanimoto-distance
  clustering via RDKit's Butina algorithm. `cluster(cutoff=0.4)` adds a `cluster_id`
  metadata column; cluster 0 is always the largest cluster. Invalid SMILES get ID -1.
- **k-fold cross-validation** (`MolDataset.kfold`, `MolDataset.scaffold_kfold`):
  `kfold(k=5)` produces random splits; `scaffold_kfold(k=5)` assigns whole Murcko
  scaffold groups to folds so no scaffold leaks between train and val.
- **Optuna hyperparameter search** (`PropertyPredictor.tune`): searches hidden dim,
  n_layers, dropout, lr, batch_size over n_trials Optuna trials; restores the
  best-seen model weights. Requires `pip install molcore[optuna]`.
- `optuna` optional dependency group in `pyproject.toml`; added to `[all]`.

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
