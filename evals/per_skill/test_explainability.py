"""
Evals for molcore.explainability — guards gradient flow for atom_attribution
and integrated_gradients.  Uses a tiny 1-layer GCN so tests stay fast.
"""
import torch
import pytest
from molcore.molecule import Mol
from molcore.explainability import atom_attribution, integrated_gradients


def _dummy_model(n_features: int = 9):
    from torch_geometric.nn import GCNConv, global_mean_pool
    import torch.nn as nn

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = GCNConv(n_features, 1)

        def forward(self, data):
            x = self.conv(data.x, data.edge_index)
            return global_mean_pool(x, torch.zeros(data.x.shape[0], dtype=torch.long))

    return _M()


# ── atom_attribution ─────────────────────────────────────────────────────────

class TestAtomAttribution:
    def test_benzene_shape(self):
        mol = Mol.from_smiles("c1ccccc1")
        scores = atom_attribution(_dummy_model(), mol.to_pyg())
        assert scores.shape == (6,), f"Expected (6,), got {scores.shape}"

    def test_scores_non_negative(self):
        mol = Mol.from_smiles("c1ccccc1")
        scores = atom_attribution(_dummy_model(), mol.to_pyg())
        assert (scores >= 0).all(), "Gradient-magnitude scores must be ≥ 0"

    def test_scores_not_all_zero(self):
        mol = Mol.from_smiles("c1ccccc1")
        scores = atom_attribution(_dummy_model(), mol.to_pyg())
        assert scores.sum() > 0, "All-zero scores indicate broken gradient flow"

    def test_different_molecules_different_scores(self):
        m1 = Mol.from_smiles("c1ccccc1")
        m2 = Mol.from_smiles("CC(=O)O")
        model = _dummy_model()
        s1 = atom_attribution(model, m1.to_pyg())
        s2 = atom_attribution(model, m2.to_pyg())
        # different atom counts → shapes must differ
        assert s1.shape != s2.shape or not torch.equal(s1, s2)

    def test_returns_tensor(self):
        mol = Mol.from_smiles("CCO")
        scores = atom_attribution(_dummy_model(), mol.to_pyg())
        assert isinstance(scores, torch.Tensor)

    def test_no_nan_in_output(self):
        mol = Mol.from_smiles("CC(=O)Nc1ccc(O)cc1")  # acetaminophen
        scores = atom_attribution(_dummy_model(), mol.to_pyg())
        assert not torch.isnan(scores).any(), "NaN in attribution scores"

    def test_shape_matches_heavy_atom_count(self):
        smiles_ha = [("CCO", 3), ("c1ccccc1", 6), ("CC(=O)O", 4)]
        model = _dummy_model()
        for smi, expected_n in smiles_ha:
            scores = atom_attribution(model, Mol.from_smiles(smi).to_pyg())
            assert scores.shape == (expected_n,), \
                f"{smi}: expected ({expected_n},), got {scores.shape}"


# ── integrated_gradients ──────────────────────────────────────────────────────

class TestIntegratedGradients:
    def test_benzene_shape(self):
        mol = Mol.from_smiles("c1ccccc1")
        scores = integrated_gradients(_dummy_model(), mol.to_pyg(), steps=5)
        assert scores.shape == (6,)

    def test_no_nan(self):
        mol = Mol.from_smiles("CC(=O)O")
        scores = integrated_gradients(_dummy_model(), mol.to_pyg(), steps=10)
        assert not torch.isnan(scores).any()

    def test_scores_non_negative(self):
        mol = Mol.from_smiles("c1ccccc1")
        scores = integrated_gradients(_dummy_model(), mol.to_pyg(), steps=5)
        assert (scores >= 0).all()

    def test_scores_not_all_zero(self):
        mol = Mol.from_smiles("c1ccccc1")
        scores = integrated_gradients(_dummy_model(), mol.to_pyg(), steps=5)
        assert scores.sum() > 0

    def test_more_steps_same_shape(self):
        mol = Mol.from_smiles("CCO")
        s5  = integrated_gradients(_dummy_model(), mol.to_pyg(), steps=5)
        s20 = integrated_gradients(_dummy_model(), mol.to_pyg(), steps=20)
        assert s5.shape == s20.shape == (3,)

    def test_acetaminophen_shape(self):
        mol = Mol.from_smiles("CC(=O)Nc1ccc(O)cc1")
        scores = integrated_gradients(_dummy_model(), mol.to_pyg(), steps=5)
        assert scores.shape == (mol.to_pyg().x.shape[0],)
