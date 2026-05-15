"""Tests for Matched Molecular Pair Analysis (MMPA)."""
import pytest
from molcore.rdkit_bridge import mmpa


# A small para-substituted benzene series — well-known SAR use case
SERIES = [
    "c1ccc(F)cc1",    # 4-fluorobenzene
    "c1ccc(Cl)cc1",   # 4-chlorobenzene
    "c1ccc(Br)cc1",   # 4-bromobenzene
    "c1ccc(N)cc1",    # 4-aminobenzene
    "c1ccccc1",       # benzene (no substituent — serves as control)
]


def test_mmpa_returns_list():
    pairs = mmpa(SERIES)
    assert isinstance(pairs, list)


def test_mmpa_each_pair_has_required_keys():
    pairs = mmpa(SERIES)
    required = {"mol_a", "mol_b", "smiles_a", "smiles_b", "core", "transform"}
    for p in pairs:
        assert required <= set(p.keys()), f"Missing keys in pair: {set(p.keys())}"


def test_mmpa_finds_pairs_in_series():
    pairs = mmpa(SERIES)
    assert len(pairs) > 0, "Expected at least one MMP in the para-substituted series"


def test_mmpa_pairs_are_distinct_molecules():
    pairs = mmpa(SERIES)
    for p in pairs:
        assert p["mol_a"] != p["mol_b"]


def test_mmpa_substituents_differ_within_pair():
    pairs = mmpa(SERIES)
    for p in pairs:
        assert p["smiles_a"] != p["smiles_b"]


def test_mmpa_transform_format():
    pairs = mmpa(SERIES)
    for p in pairs:
        assert ">>" in p["transform"]


def test_mmpa_invalid_smiles_skipped():
    smiles = ["c1ccc(F)cc1", "NOT_VALID", "c1ccc(Cl)cc1"]
    pairs = mmpa(smiles)
    # Should still find the F/Cl pair; invalid SMILES silently skipped
    mols = {p["mol_a"] for p in pairs} | {p["mol_b"] for p in pairs}
    assert not any("NOT_VALID" in m for m in mols)


def test_mmpa_empty_input():
    assert mmpa([]) == []


def test_mmpa_single_molecule():
    # Can't form a pair with one molecule
    assert mmpa(["c1ccccc1"]) == []


def test_mmpa_identical_molecules_no_pair():
    # Identical SMILES — deduplicated, no pair
    pairs = mmpa(["c1ccc(F)cc1", "c1ccc(F)cc1"])
    assert pairs == []


def test_mmpa_aliphatic_series():
    # Alkyl acid homologs
    acids = ["CC(=O)O", "CCC(=O)O", "CCCC(=O)O"]
    pairs = mmpa(acids)
    assert len(pairs) > 0


def test_mmpa_unsupported_cut_bonds_raises():
    with pytest.raises(ValueError, match="max_cut_bonds"):
        mmpa(SERIES, max_cut_bonds=2)
