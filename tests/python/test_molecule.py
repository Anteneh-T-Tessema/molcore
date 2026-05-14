import pytest
from molcore.molecule import Mol


def test_mol_is_frozen():
    mol = Mol.from_smiles("CCO")
    with pytest.raises(Exception):  # FrozenInstanceError
        mol.smiles = "c1ccccc1"


def test_canonical_smiles_is_set():
    mol = Mol.from_smiles("OCC")
    assert mol.smiles  # not empty
    assert isinstance(mol.smiles, str)


def test_neutralize_returns_new_mol():
    mol = Mol.from_smiles("CCO")
    mol2 = mol.neutralize()
    assert mol is not mol2
    assert isinstance(mol2, Mol)


def test_strip_salts_returns_new_mol():
    mol = Mol.from_smiles("c1ccccc1C(=O)O")
    mol2 = mol.strip_salts()
    assert mol is not mol2


def test_repr_contains_atom_count():
    mol = Mol.from_smiles("CCO")
    r = repr(mol)
    assert "atoms=" in r


def test_invalid_smiles_raises():
    with pytest.raises(ValueError):
        Mol.from_smiles("NOT_A_SMILES")
