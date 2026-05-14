# AGENTS.md — molcore Agent Operating Manual

## What this repo is

`molcore` is an AI-native cheminformatics toolkit: Rust performance core + RDKit bridge +
Python AI API. It exposes chemistry as **skills** that agents can call safely, in batch,
with zero-copy tensor handoff to PyTorch / PyG.

## Repo layout

```
molcore/
├── skills/           # self-contained cheminformatics agent skills
├── prompts/          # structured prompting system for chemistry agents
├── tools/            # MCP server config + local function tools + registry
├── memory/           # agent memory layers (working / semantic / episodic / procedural)
├── evals/            # per-skill, compositional, and regression test suites
├── observability/    # OpenTelemetry traces, domain events, skill metrics
├── crates/           # Rust workspace (molcore-core, molcore-io)
├── molcore/          # Python package (PyO3 bindings + featurizers + pipeline)
├── tests/            # Rust unit tests + Python integration tests
└── benchmarks/       # throughput comparisons (Rust vs RDKit baselines)
```

## Skill catalogue

| Skill | What it does | Primary entry point |
|---|---|---|
| `fingerprint` | ECFP4 / Morgan batch fingerprints | `featurize_smiles(smiles, backend="rust")` |
| `similarity_search` | Tanimoto pairwise matrix | `tanimoto_matrix(query, library)` |
| `molecular_featurization` | Atom/bond graph arrays → PyG/DGL | `mol.to_pyg()` |
| `property_prediction` | MW, LogP, TPSA batch | `calc_descriptors(smiles)` |
| `agentic_rag` | Iterative chemical literature retrieval | See `skills/agentic_rag/SKILL.md` |

## Design invariants — never break these

1. `Mol` is always immutable. Transformations return new `Mol` instances.
2. RDKit is never called in hot paths. Only `molcore/rdkit_bridge.py` imports RDKit.
3. All Rust → Python array transfers use `IntoPyArray`. No Python-side loops.
4. Batch API is the primary surface. Per-mol methods are convenience wrappers.
5. Fingerprint backends are explicit flags (`backend="rust"` | `"rdkit"`), never auto-selected.

## Guardrails

- SMILES must pass RDKit sanitization before entering any skill. Malformed inputs raise
  `MolIngestionError`, never silently produce garbage tensors.
- No skill modifies global state. Every call is stateless and thread-safe.
- Skills declare their `allowlist.yaml` — agents outside the allowlist cannot invoke them.

## Agentic RAG flow

```
User query (chemistry)
  → prompts/system/chemistry_agent.md   (base system prompt)
  → skills/agentic_rag                  (query rewriting + source routing)
      ├── Vector DB  (embedded literature)
      ├── tools/local (featurize, similarity, screen)
      └── Internet   (PubChem, ChEMBL APIs)
  → Relevance check → iterate or return Final Response
```

## Adding a new skill

1. Create `skills/{skill_name}/SKILL.md` — document inputs, outputs, when to invoke.
2. Add golden examples to `skills/{skill_name}/examples.jsonl`.
3. Declare access in `skills/{skill_name}/allowlist.yaml`.
4. Register in `tools/registry.yaml`.
5. Add per-skill eval in `evals/per_skill/test_{skill_name}.py`.
6. Add entry to this file's skill catalogue table.
7. Add CHANGELOG.md entry.

## Running the test suite

```bash
# Build Rust extension
maturin develop --release

# Rust unit tests
cargo test --workspace

# Python integration tests (includes zero-copy regression)
pytest tests/python -v --tb=short

# Skill evals
pytest evals/ -v

# Benchmarks (optional, slow)
python benchmarks/bench_fingerprints.py
```
