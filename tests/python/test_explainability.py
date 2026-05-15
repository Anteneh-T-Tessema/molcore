"""
Tests for molcore.explainability — gradient-based atom attribution.
Requires torch and torch_geometric (already in the dev venv).
"""
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data

from molcore.explainability import atom_attribution, integrated_gradients


# ---------------------------------------------------------------------------
# Minimal GCN that works without molcore.predictor overhead
# ---------------------------------------------------------------------------

class _TinyGCN(nn.Module):
    """Single linear layer on node features — fully differentiable, no PyG ops."""

    def __init__(self, in_dim: int = 9, out_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, data: Data) -> torch.Tensor:
        return self.fc(data.x).sum(dim=0, keepdim=True)  # (1, out_dim)


def _make_data(n_atoms: int = 5, feat_dim: int = 9) -> Data:
    x = torch.rand(n_atoms, feat_dim)
    # simple chain: 0-1-2-3-4 bidirectional
    src = list(range(n_atoms - 1)) + list(range(1, n_atoms))
    dst = list(range(1, n_atoms)) + list(range(n_atoms - 1))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)


# ---------------------------------------------------------------------------
# atom_attribution
# ---------------------------------------------------------------------------

def test_atom_attribution_shape():
    model = _TinyGCN()
    data  = _make_data(n_atoms=5)
    scores = atom_attribution(model, data, target_class=0)
    assert scores.shape == (5,)


def test_atom_attribution_non_negative():
    model = _TinyGCN()
    data  = _make_data(n_atoms=4)
    scores = atom_attribution(model, data)
    assert (scores >= 0).all()


def test_atom_attribution_returns_tensor():
    model = _TinyGCN()
    data  = _make_data()
    scores = atom_attribution(model, data)
    assert isinstance(scores, torch.Tensor)


def test_atom_attribution_single_atom():
    model = _TinyGCN()
    x = torch.rand(1, 9)
    data = Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long))
    scores = atom_attribution(model, data)
    assert scores.shape == (1,)


def test_atom_attribution_finite():
    """All scores must be finite (no NaN/Inf from gradient accumulation)."""
    torch.manual_seed(0)
    model = _TinyGCN()
    data  = _make_data(n_atoms=6)
    scores = atom_attribution(model, data)
    assert torch.isfinite(scores).all()


# ---------------------------------------------------------------------------
# integrated_gradients
# ---------------------------------------------------------------------------

def test_integrated_gradients_shape():
    model  = _TinyGCN()
    data   = _make_data(n_atoms=5)
    scores = integrated_gradients(model, data, steps=10)
    assert scores.shape == (5,)


def test_integrated_gradients_non_negative():
    model  = _TinyGCN()
    data   = _make_data(n_atoms=4)
    scores = integrated_gradients(model, data, steps=5)
    assert (scores >= 0).all()


def test_integrated_gradients_returns_tensor():
    model  = _TinyGCN()
    data   = _make_data()
    scores = integrated_gradients(model, data, steps=5)
    assert isinstance(scores, torch.Tensor)


def test_integrated_gradients_steps_affect_result():
    """More steps → smoother estimate; both should be non-zero."""
    torch.manual_seed(1)
    model  = _TinyGCN()
    data   = _make_data(n_atoms=3)
    s10  = integrated_gradients(model, data, steps=10)
    s50  = integrated_gradients(model, data, steps=50)
    # Both non-zero and finite
    assert s10.sum().item() > 0
    assert s50.sum().item() > 0
    assert torch.isfinite(s10).all()
    assert torch.isfinite(s50).all()


def test_integrated_gradients_single_atom():
    model = _TinyGCN()
    x = torch.rand(1, 9)
    data = Data(x=x, edge_index=torch.zeros(2, 0, dtype=torch.long))
    scores = integrated_gradients(model, data, steps=5)
    assert scores.shape == (1,)
