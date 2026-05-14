"""
Eval: smarts skill — substructure search and filter correctness.
"""
import pytest
from tools.local.smarts import run


# ── filter mode ──────────────────────────────────────────────────────────────

def test_filter_finds_carboxylic_acids():
    smiles = ["CCO", "CC(=O)O", "c1ccccc1C(=O)O", "CCCC"]
    result = run(smiles, "C(=O)O", mode="filter")
    assert set(result["hits"]) == {"CC(=O)O", "c1ccccc1C(=O)O"}
    assert result["n_hits"] == 2
    assert result["n_screened"] == 4


def test_filter_invert_removes_acids():
    smiles = ["CCO", "CC(=O)O", "CCCC"]
    result = run(smiles, "C(=O)O", mode="filter", invert=True)
    assert set(result["hits"]) == {"CCO", "CCCC"}
    assert result["n_hits"] == 2


def test_filter_aromatic_ring():
    smiles = ["CCO", "c1ccccc1", "Cc1ccccc1", "CCCC"]
    result = run(smiles, "c1ccccc1", mode="filter")
    assert set(result["hits"]) == {"c1ccccc1", "Cc1ccccc1"}


def test_filter_no_hits():
    smiles = ["CCO", "CCCC", "CCC"]
    result = run(smiles, "c1ccccc1", mode="filter")
    assert result["hits"] == []
    assert result["n_hits"] == 0


def test_filter_invalid_smiles_skipped():
    smiles = ["CCO", "NOT_A_SMILES", "CC(=O)O"]
    result = run(smiles, "C(=O)O", mode="filter")
    assert "CC(=O)O" in result["hits"]
    assert result["n_screened"] == 3


# ── matches mode ─────────────────────────────────────────────────────────────

def test_matches_returns_atom_indices():
    result = run(["c1ccccc1"], "c1ccccc1", mode="matches")
    matches = result["all_matches"]["c1ccccc1"]
    assert len(matches) >= 1
    assert all(len(m) == 6 for m in matches)


def test_matches_no_match():
    result = run(["CCO"], "c1ccccc1", mode="matches")
    assert result["all_matches"]["CCO"] == []


def test_matches_multiple_molecules():
    result = run(["CCO", "c1ccccc1", "Cc1ccccc1"], "c1ccccc1", mode="matches")
    assert result["all_matches"]["CCO"] == []
    assert len(result["all_matches"]["c1ccccc1"]) >= 1
    assert len(result["all_matches"]["Cc1ccccc1"]) >= 1


# ── invalid SMARTS ────────────────────────────────────────────────────────────

def test_invalid_smarts_raises():
    with pytest.raises(ValueError):
        run(["CCO"], "NOT$$VALID$$SMARTS", mode="filter")


# ── invalid mode ──────────────────────────────────────────────────────────────

def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        run(["CCO"], "C", mode="unknown_mode")
