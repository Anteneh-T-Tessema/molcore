"""Tests for to_pyg_hetero() — HeteroData output."""
import torch
import pytest
from molcore.molecule import Mol
from molcore.featurizers.graph import to_pyg_hetero, to_pyg_data


def test_hetero_benzene_only_C_nodes():
    mol  = Mol.from_smiles("c1ccccc1")
    data = to_pyg_hetero(mol._graph)
    assert "C" in data.node_types, "benzene should have C node type"
    assert data["C"].x.shape == (6, 4), "6 carbons, 4 features each"


def test_hetero_pyridine_has_C_and_N():
    mol  = Mol.from_smiles("c1ccncc1")
    data = to_pyg_hetero(mol._graph)
    assert "C" in data.node_types
    assert "N" in data.node_types
    assert data["N"].x.shape[0] == 1, "pyridine has 1 nitrogen"
    assert data["C"].x.shape[0] == 5, "pyridine has 5 carbons"


def test_hetero_ethanol_has_C_and_O():
    mol  = Mol.from_smiles("CCO")
    data = to_pyg_hetero(mol._graph)
    assert "C" in data.node_types
    assert "O" in data.node_types
    assert data["C"].x.shape[0] == 2
    assert data["O"].x.shape[0] == 1


def test_hetero_edge_types_present():
    # aspirin has C, O atoms — expect C-C and C-O bond types
    mol  = Mol.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
    data = to_pyg_hetero(mol._graph)
    edge_type_strs = {(s, r, d) for s, r, d in data.edge_types}
    # At minimum C→C and C→O bonds must exist
    assert any(s == "C" and d == "C" for s, _, d in edge_type_strs), "expect C-C bonds"
    assert any(s == "C" and d == "O" or s == "O" and d == "C"
               for s, _, d in edge_type_strs), "expect C-O bonds"


def test_hetero_node_features_dtype():
    mol  = Mol.from_smiles("CCO")
    data = to_pyg_hetero(mol._graph)
    for t in data.node_types:
        assert data[t].x.dtype == torch.float32


def test_hetero_edge_index_dtype():
    mol  = Mol.from_smiles("c1ccncc1")
    data = to_pyg_hetero(mol._graph)
    for et in data.edge_types:
        assert data[et].edge_index.dtype == torch.long


def test_hetero_total_nodes_equals_homo():
    """Sum of nodes across all types must equal homo-graph node count."""
    mol   = Mol.from_smiles("CC(=O)Nc1ccc(O)cc1")
    homo  = to_pyg_data(mol._graph)
    hetero = to_pyg_hetero(mol._graph)
    total = sum(data.x.shape[0] for data in [hetero[t] for t in hetero.node_types])
    assert total == homo.x.shape[0]
