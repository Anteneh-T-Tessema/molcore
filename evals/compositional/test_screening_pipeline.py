"""
Compositional eval: fingerprint skill + similarity_search compose correctly
for a virtual screening use-case.
"""
import molcore
from molcore._molcore import tanimoto_matrix

QUERY_SMILES = ["c1ccccc1C(=O)O"]   # benzoic acid

LIBRARY = [
    "c1ccccc1C(=O)O",    # benzoic acid (identical — Tanimoto = 1.0)
    "c1ccccc1C(=O)N",    # benzamide (very similar)
    "c1ccccc1",          # benzene (similar scaffold)
    "CCO",               # ethanol (dissimilar)
    "CC(=O)O",           # acetic acid (partially similar)
]


def test_pipeline_returns_correct_shape():
    q_fps = molcore.featurize_smiles(QUERY_SMILES, backend="rust").numpy()
    l_fps = molcore.featurize_smiles(LIBRARY,      backend="rust").numpy()
    sim   = tanimoto_matrix(q_fps, l_fps)
    assert sim.shape == (1, len(LIBRARY))


def test_self_hit_is_top_ranked():
    q_fps = molcore.featurize_smiles(QUERY_SMILES, backend="rust").numpy()
    l_fps = molcore.featurize_smiles(LIBRARY,      backend="rust").numpy()
    sim   = tanimoto_matrix(q_fps, l_fps)

    scores = sim[0].tolist()
    top_idx = max(range(len(scores)), key=lambda i: scores[i])
    assert top_idx == 0, f"identical compound should be top hit, got idx={top_idx}"
    assert abs(scores[0] - 1.0) < 1e-5, "self-similarity must be 1.0"


def test_dissimilar_compound_low_score():
    q_fps = molcore.featurize_smiles(QUERY_SMILES, backend="rust").numpy()
    l_fps = molcore.featurize_smiles(LIBRARY,      backend="rust").numpy()
    sim   = tanimoto_matrix(q_fps, l_fps)

    ethanol_score = sim[0, 3]  # index 3 = "CCO"
    assert ethanol_score < 0.5, f"ethanol should be dissimilar to benzoic acid, got {ethanol_score:.3f}"


def test_rust_rdkit_backend_top_hit_agrees():
    """Both backends should rank the same compound #1."""
    q_rust  = molcore.featurize_smiles(QUERY_SMILES, backend="rust").numpy()
    l_rust  = molcore.featurize_smiles(LIBRARY,      backend="rust").numpy()
    q_rdkit = molcore.featurize_smiles(QUERY_SMILES, backend="rdkit").numpy()
    l_rdkit = molcore.featurize_smiles(LIBRARY,      backend="rdkit").numpy()

    sim_rust  = tanimoto_matrix(q_rust,  l_rust)
    sim_rdkit = tanimoto_matrix(q_rdkit, l_rdkit)

    top_rust  = int(sim_rust[0].argmax())
    top_rdkit = int(sim_rdkit[0].argmax())
    assert top_rust == top_rdkit, (
        f"Rust top hit (idx={top_rust}) disagrees with RDKit (idx={top_rdkit})"
    )
