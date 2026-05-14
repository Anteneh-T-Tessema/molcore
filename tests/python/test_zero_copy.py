import torch
import molcore


def test_fingerprint_no_copy():
    """Rust → numpy → torch must be zero-copy end to end."""
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    t = molcore.featurize_smiles(smiles, backend="rust")
    assert t.is_contiguous()
    assert not t.requires_grad
    assert t.shape == (3, 2048)
    assert t.dtype == torch.uint8


def test_pyg_no_copy():
    """Graph arrays from Rust must produce correct dtypes — no copy means no cast."""
    from molcore.molecule import Mol
    mol = Mol.from_smiles("c1ccccc1C(=O)O")
    data = mol.to_pyg()
    assert data.edge_index.dtype == torch.long,  "edge_index must be int64 for PyG"
    assert data.x.dtype == torch.float32,        "node features must be float32"
    assert data.x.shape[1] == 9,                 "9 node features"
    assert data.edge_attr.shape[1] == 4,         "4 bond features"


def test_tanimoto_output_range():
    """Tanimoto scores must be in [0, 1]."""
    from molcore._molcore import tanimoto_matrix
    fps = molcore.featurize_smiles(["CCO", "c1ccccc1"]).numpy()
    sim = tanimoto_matrix(fps, fps)
    assert sim.min() >= 0.0
    assert sim.max() <= 1.0
    # diagonal should be 1.0 (self-similarity)
    assert abs(sim[0, 0] - 1.0) < 1e-5
    assert abs(sim[1, 1] - 1.0) < 1e-5
