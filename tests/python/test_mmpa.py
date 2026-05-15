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
        mmpa(SERIES, max_cut_bonds=3)


# ---------------------------------------------------------------------------
# Double-cut MMPA (max_cut_bonds=2)
# ---------------------------------------------------------------------------

# 1,2-/1,3-/1,4-diphenylalkanes: same two phenyl termini, linker grows by CH2
LINKER_SERIES = [
    "c1ccccc1CCc1ccccc1",    # linker = -CH2CH2-
    "c1ccccc1CCCc1ccccc1",   # linker = -CH2CH2CH2-
    "c1ccccc1CCCCc1ccccc1",  # linker = -CH2CH2CH2CH2-
]

# Heteroatom linker bioisosteres
BIOISOSTERE_SERIES = [
    "c1ccccc1COCc1ccccc1",   # linker = -CH2OCH2-  (ether)
    "c1ccccc1CSCc1ccccc1",   # linker = -CH2SCH2-  (thioether)
]


def test_mmpa_double_cut_returns_list():
    pairs = mmpa(LINKER_SERIES, max_cut_bonds=2)
    assert isinstance(pairs, list)


def test_mmpa_double_cut_finds_pairs():
    pairs = mmpa(LINKER_SERIES, max_cut_bonds=2)
    assert len(pairs) >= 1, "Expected at least one double-cut MMP in the linker series"


def test_mmpa_double_cut_required_keys():
    pairs = mmpa(LINKER_SERIES, max_cut_bonds=2)
    required = {"mol_a", "mol_b", "smiles_a", "smiles_b", "core", "transform"}
    for p in pairs:
        assert required <= set(p.keys())


def test_mmpa_double_cut_linkers_differ():
    pairs = mmpa(LINKER_SERIES, max_cut_bonds=2)
    for p in pairs:
        assert p["smiles_a"] != p["smiles_b"]


def test_mmpa_double_cut_distinct_molecules():
    pairs = mmpa(LINKER_SERIES, max_cut_bonds=2)
    for p in pairs:
        assert p["mol_a"] != p["mol_b"]


def test_mmpa_double_cut_transform_has_arrow():
    pairs = mmpa(LINKER_SERIES, max_cut_bonds=2)
    for p in pairs:
        assert ">>" in p["transform"]


def test_mmpa_double_cut_bioisostere_series():
    pairs = mmpa(BIOISOSTERE_SERIES, max_cut_bonds=2)
    assert len(pairs) >= 1, "Expected O/S bioisostere linker pair"
    # Both mols should appear in the pair
    mols = {p["mol_a"] for p in pairs} | {p["mol_b"] for p in pairs}
    assert len(mols) >= 2


def test_mmpa_double_cut_empty_input():
    assert mmpa([], max_cut_bonds=2) == []


def test_mmpa_double_cut_single_molecule():
    assert mmpa(["c1ccccc1CCc1ccccc1"], max_cut_bonds=2) == []


def test_mmpa_double_cut_invalid_smiles_skipped():
    smiles = ["c1ccccc1CCc1ccccc1", "INVALID_XYZ", "c1ccccc1CCCc1ccccc1"]
    pairs = mmpa(smiles, max_cut_bonds=2)
    mols = {p["mol_a"] for p in pairs} | {p["mol_b"] for p in pairs}
    assert not any("INVALID" in m for m in mols)


def test_mmpa_double_cut_no_pairs_identical():
    # Two identical SMILES → deduplicated → no pair possible
    pairs = mmpa(["c1ccccc1CCc1ccccc1", "c1ccccc1CCc1ccccc1"], max_cut_bonds=2)
    assert pairs == []
