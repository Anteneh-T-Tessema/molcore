"""Tests for reaction transforms in rdkit_bridge and Mol."""
import pytest
from molcore.molecule import Mol
from molcore import rdkit_bridge


# ── react (unimolecular) ─────────────────────────────────────────────────────

# Ester hydrolysis: ester → acid + alcohol
ESTER_HYDROLYSIS = "[C:1](=O)[O:2][C:3]>>[C:1](=O)[OH].[C:3][OH]"

def test_react_ester_hydrolysis_products():
    products = rdkit_bridge.react("CC(=O)OCC", ESTER_HYDROLYSIS)
    assert len(products) > 0


def test_react_no_match_returns_empty():
    # Benzene has no ester → no products
    products = rdkit_bridge.react("c1ccccc1", ESTER_HYDROLYSIS)
    assert products == []


def test_react_products_are_valid_smiles():
    products = rdkit_bridge.react("CC(=O)OCC", ESTER_HYDROLYSIS)
    for smi in products:
        # Each product should be parseable by RDKit
        mol = rdkit_bridge.from_smiles(smi)
        assert mol is not None


def test_react_deduplicated():
    products = rdkit_bridge.react("CC(=O)OCC", ESTER_HYDROLYSIS)
    assert len(products) == len(set(products))


def test_react_invalid_smarts_raises():
    with pytest.raises(ValueError, match="Invalid reaction SMARTS"):
        rdkit_bridge.react("CCO", ">>>")


# ── react_bimolecular ─────────────────────────────────────────────────────────

# Amide coupling: acid + amine → amide
AMIDE_COUPLING = "[C:1](=O)[OH].[N:2]>>[C:1](=O)[N:2]"

def test_react_bimolecular_amide():
    products = rdkit_bridge.react_bimolecular("CC(=O)O", "CCN", AMIDE_COUPLING)
    assert len(products) > 0


def test_react_bimolecular_no_match():
    # Neither reactant matches if SMARTS requires specific groups
    products = rdkit_bridge.react_bimolecular("CCCC", "CCCC", AMIDE_COUPLING)
    assert products == []


def test_react_bimolecular_products_are_strings():
    products = rdkit_bridge.react_bimolecular("CC(=O)O", "CCN", AMIDE_COUPLING)
    assert all(isinstance(p, str) for p in products)


# ── enumerate_reactions ───────────────────────────────────────────────────────

ESTERS = ["CC(=O)OCC", "CC(=O)OCCC", "CC(=O)OC"]

def test_enumerate_reactions_multiple():
    products = rdkit_bridge.enumerate_reactions(ESTERS, ESTER_HYDROLYSIS)
    assert len(products) > 0


def test_enumerate_reactions_max_products():
    big_lib = ESTERS * 100
    products = rdkit_bridge.enumerate_reactions(big_lib, ESTER_HYDROLYSIS, max_products=5)
    assert len(products) <= 5


def test_enumerate_reactions_skips_non_matching():
    lib = ["c1ccccc1", "CCCC", "CC(=O)OCC"]
    products = rdkit_bridge.enumerate_reactions(lib, ESTER_HYDROLYSIS)
    assert len(products) > 0   # only the ester reacts


def test_enumerate_reactions_empty_library():
    assert rdkit_bridge.enumerate_reactions([], ESTER_HYDROLYSIS) == []


# ── Mol.react ────────────────────────────────────────────────────────────────

def test_mol_react_returns_mol_list():
    mol = Mol.from_smiles("CC(=O)OCC")
    products = mol.react(ESTER_HYDROLYSIS)
    assert isinstance(products, list)
    assert all(isinstance(p, Mol) for p in products)


def test_mol_react_no_match_empty():
    mol = Mol.from_smiles("c1ccccc1")
    products = mol.react(ESTER_HYDROLYSIS)
    assert products == []


def test_mol_react_products_immutable():
    mol = Mol.from_smiles("CC(=O)OCC")
    products = mol.react(ESTER_HYDROLYSIS)
    if products:
        with pytest.raises(Exception):
            products[0].smiles = "CHANGED"
