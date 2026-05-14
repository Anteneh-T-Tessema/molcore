# CHANGELOG

## [Unreleased]

### Added
- Initial project scaffold: AGENT_SKILLS structure + molcore v2 Rust/Python core
- `skills/fingerprint` — ECFP4 batch fingerprints via Rust Rayon + zero-copy numpy bridge
- `skills/similarity_search` — Tanimoto pairwise matrix, Rust parallel
- `skills/molecular_featurization` — petgraph → PyG/DGL zero-copy graph arrays
- `skills/property_prediction` — MW, LogP, TPSA batch descriptors
- `skills/agentic_rag` — iterative chemical literature retrieval with query rewriting
- `tools/registry.yaml` — single source of truth for tool discovery
- `memory/` layers: working, semantic (chemistry knowledge), episodic, procedural
- `evals/` suite: per-skill, compositional, regression
- `observability/` — OpenTelemetry config, domain events schema, skill metrics
- Frozen `Mol` dataclass — immutability contract, all transforms return new instances
- `featurize_smiles()` batch-first primary entry point with `rust` / `rdkit` backend flags
- CI workflow: RDKit pre-built binary install + maturin build + cargo test + pytest
