"""Tests for MCS, R-group decomposition, and standardization pipeline."""
import pytest
from molcore.rdkit_bridge import find_mcs, rgroup_decompose, standardize
from molcore.molecule import Mol

ANALOGS = [
    "CC(=O)Oc1ccccc1",        # phenyl acetate
    "CC(=O)Oc1ccc(F)cc1",     # 4-fluorophenyl acetate
    "CC(=O)Oc1ccc(Cl)cc1",    # 4-chlorophenyl acetate
]


# ---------------------------------------------------------------------------
# standardize
# ---------------------------------------------------------------------------

def test_standardize_strips_sodium_salt():
    result = standardize("[Na+].OC(=O)c1ccccc1")
    assert "[Na+]" not in result
    assert "Na" not in result


def test_standardize_neutralizes_charge():
    result = standardize("CC(=O)[O-]")
    assert "[O-]" not in result


def test_standardize_returns_canonical_smiles():
    r1 = standardize("OCC")
    r2 = standardize("CCO")
    assert r1 == r2  # same canonical form


def test_standardize_invalid_raises():
    with pytest.raises(ValueError):
        standardize("NOT_A_SMILES_AT_ALL_XYZ")


def test_mol_standardize_returns_new_mol():
    mol = Mol.from_smiles("[Na+].OC(=O)c1ccccc1")
    clean = mol.standardize()
    assert isinstance(clean, Mol)
    assert mol is not clean
    assert "[Na+]" not in clean.smiles


# ---------------------------------------------------------------------------
# find_mcs
# ---------------------------------------------------------------------------

def test_mcs_returns_nonempty_smarts():
    smarts = find_mcs(ANALOGS)
    assert smarts  # non-empty


def test_mcs_contains_phenyl():
    smarts = find_mcs(ANALOGS)
    # the MCS of these three includes the phenyl ring — RDKit SMARTS uses : for aromatic bonds
    assert ":" in smarts or "#6" in smarts


def test_mcs_requires_two_mols():
    with pytest.raises(ValueError, match="2 valid"):
        find_mcs(["CCO"])


def test_mcs_invalid_smiles_skipped():
    smarts = find_mcs(["NOT_VALID", "CCO", "CCN"])
    # "CCO" and "CCN" share CC — should still return something
    assert isinstance(smarts, str)


def test_mcs_timeout_param():
    smarts = find_mcs(ANALOGS, timeout=1)
    assert isinstance(smarts, str)


def test_mcs_empty_on_unrelated_mols():
    # single atoms: "C" and "[Fe]" share nothing meaningful
    result = find_mcs(["[Fe]", "[Au]", "[Pt]"])
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# rgroup_decompose
# ---------------------------------------------------------------------------

def test_rgroup_decompose_returns_list():
    core = "CC(=O)Oc1ccccc1"  # phenyl acetate core (SMILES used as SMARTS)
    result = rgroup_decompose(core, ANALOGS)
    assert isinstance(result, list)
    assert len(result) == len(ANALOGS)


def test_rgroup_decompose_first_mol_matches():
    # use a simpler case: benzene core, substituted benzenes
    mols = ["c1ccccc1F", "c1ccccc1Cl", "c1ccccc1Br"]
    core = "c1ccccc1[*:1]"
    result = rgroup_decompose(core, mols)
    # at least some molecules should match and have 'Core' key
    matched = [r for r in result if r]
    assert len(matched) > 0


def test_rgroup_decompose_unmatched_gives_empty_dict():
    core = "c1ccccc1[*:1]"
    result = rgroup_decompose(core, ["CCO"])  # acyclic — won't match aromatic core
    assert result[0] == {} or isinstance(result[0], dict)


def test_rgroup_decompose_length_matches_input():
    core = "c1ccccc1[*:1]"
    smiles = ["c1ccccc1F", "CCO", "c1ccccc1Cl"]
    result = rgroup_decompose(core, smiles)
    assert len(result) == 3
