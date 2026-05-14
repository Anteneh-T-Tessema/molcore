"""
Evals for the reaction skill.

Covers: react, react_bimolecular, enumerate_reactions, Mol.react,
error handling, and the reaction local tool.
"""
import pytest
from molcore.rdkit_bridge import react, react_bimolecular, enumerate_reactions
from molcore.molecule import Mol

ESTER_HYDROLYSIS  = "[C:1](=O)[O:2][C:3]>>[C:1](=O)[OH].[C:3][OH]"
AMIDE_COUPLING    = "[C:1](=O)[OH].[N:2]>>[C:1](=O)[N:2]"
NBOC_DEPROTECT    = "[N:1][C](=O)OC(C)(C)C>>[N:1]"

ETHYL_ACETATE = "CC(=O)OCC"
ACETIC_ACID   = "CC(=O)O"
ETHYLAMINE    = "CCN"
BENZENE       = "c1ccccc1"


# ── react (unimolecular) ─────────────────────────────────────────────────────

class TestReact:
    def test_ester_hydrolysis_returns_products(self):
        assert len(react(ETHYL_ACETATE, ESTER_HYDROLYSIS)) > 0

    def test_no_match_returns_empty(self):
        assert react(BENZENE, ESTER_HYDROLYSIS) == []

    def test_products_are_valid_smiles(self):
        from molcore.rdkit_bridge import from_smiles
        for smi in react(ETHYL_ACETATE, ESTER_HYDROLYSIS):
            assert from_smiles(smi) is not None

    def test_products_are_deduplicated(self):
        products = react(ETHYL_ACETATE, ESTER_HYDROLYSIS)
        assert len(products) == len(set(products))

    def test_products_are_sorted(self):
        products = react(ETHYL_ACETATE, ESTER_HYDROLYSIS)
        assert products == sorted(products)

    def test_invalid_smarts_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid reaction SMARTS"):
            react(ETHYL_ACETATE, ">>>")

    def test_invalid_smiles_raises(self):
        with pytest.raises(ValueError):
            react("NOT_A_SMILES", ESTER_HYDROLYSIS)


# ── react_bimolecular ─────────────────────────────────────────────────────────

class TestReactBimolecular:
    def test_amide_coupling_produces_products(self):
        products = react_bimolecular(ACETIC_ACID, ETHYLAMINE, AMIDE_COUPLING)
        assert len(products) > 0

    def test_no_match_returns_empty(self):
        products = react_bimolecular("CCCC", "CCCC", AMIDE_COUPLING)
        assert products == []

    def test_products_are_strings(self):
        products = react_bimolecular(ACETIC_ACID, ETHYLAMINE, AMIDE_COUPLING)
        assert all(isinstance(p, str) for p in products)

    def test_products_deduplicated(self):
        products = react_bimolecular(ACETIC_ACID, ETHYLAMINE, AMIDE_COUPLING)
        assert len(products) == len(set(products))


# ── enumerate_reactions ───────────────────────────────────────────────────────

ESTERS = ["CC(=O)OCC", "CC(=O)OCCC", "CC(=O)OC"]

class TestEnumerateReactions:
    def test_multiple_reactants_produce_products(self):
        assert len(enumerate_reactions(ESTERS, ESTER_HYDROLYSIS)) > 0

    def test_max_products_cap(self):
        big = ESTERS * 50
        products = enumerate_reactions(big, ESTER_HYDROLYSIS, max_products=3)
        assert len(products) <= 3

    def test_non_matching_skipped(self):
        lib = ["c1ccccc1", "CCCC", "CC(=O)OCC"]
        products = enumerate_reactions(lib, ESTER_HYDROLYSIS)
        assert len(products) > 0

    def test_empty_library_returns_empty(self):
        assert enumerate_reactions([], ESTER_HYDROLYSIS) == []

    def test_products_are_sorted(self):
        products = enumerate_reactions(ESTERS, ESTER_HYDROLYSIS)
        assert products == sorted(products)


# ── Mol.react ────────────────────────────────────────────────────────────────

class TestMolReact:
    def test_returns_list_of_mol(self):
        mol = Mol.from_smiles(ETHYL_ACETATE)
        products = mol.react(ESTER_HYDROLYSIS)
        assert isinstance(products, list)
        assert all(isinstance(p, Mol) for p in products)

    def test_no_match_returns_empty_list(self):
        mol = Mol.from_smiles(BENZENE)
        assert mol.react(ESTER_HYDROLYSIS) == []

    def test_products_have_smiles(self):
        mol = Mol.from_smiles(ETHYL_ACETATE)
        for p in mol.react(ESTER_HYDROLYSIS):
            assert isinstance(p.smiles, str) and len(p.smiles) > 0


# ── reaction local tool ───────────────────────────────────────────────────────

class TestReactionTool:
    def test_unimolecular_mode(self):
        from tools.local.reaction import run
        products = run(ETHYL_ACETATE, ESTER_HYDROLYSIS, mode="unimolecular")
        assert isinstance(products, list) and len(products) > 0

    def test_bimolecular_mode(self):
        from tools.local.reaction import run
        products = run(ACETIC_ACID, AMIDE_COUPLING, mode="bimolecular", smiles_b=ETHYLAMINE)
        assert isinstance(products, list) and len(products) > 0

    def test_enumerate_mode(self):
        from tools.local.reaction import run
        products = run(ESTERS, ESTER_HYDROLYSIS, mode="enumerate", max_products=10)
        assert isinstance(products, list) and 0 < len(products) <= 10

    def test_invalid_mode_raises(self):
        from tools.local.reaction import run
        with pytest.raises(ValueError, match="Unknown mode"):
            run(ETHYL_ACETATE, ESTER_HYDROLYSIS, mode="magic")
