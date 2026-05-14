"""
Regression eval: known molecules with verified properties.
These tests must never change — they guard against chemistry regressions.
"""
import pytest
import molcore
from molcore.molecule import Mol
from molcore.featurizers.descriptors import calc_descriptors


# Verified reference values (RDKit backend, locked against RDKit 2023.9+)
KNOWN = [
    # (smiles,          mw_approx, logp_approx, heavy_atoms)
    ("CCO",             46.07,     -0.14,        3),
    ("c1ccccc1",        78.11,      1.90,        6),
    ("CC(=O)O",         60.05,     -0.17,        4),
    ("c1ccccc1C(=O)O", 122.12,      1.87,        9),
]


@pytest.mark.parametrize("smiles,mw_ref,logp_ref,ha_ref", KNOWN)
def test_descriptor_heavy_atom_count_exact(smiles, mw_ref, logp_ref, ha_ref):
    """Heavy atom count from Rust must be exact — no approximation."""
    desc = calc_descriptors([smiles], backend="rust")
    heavy = int(desc[0, 2])
    assert heavy == ha_ref, f"{smiles}: expected {ha_ref} heavy atoms, got {heavy}"


@pytest.mark.parametrize("smiles,mw_ref,logp_ref,ha_ref", KNOWN)
def test_descriptor_mw_rdkit_exact(smiles, mw_ref, logp_ref, ha_ref):
    """RDKit backend MW must be within 0.1 Da of reference (includes implicit H)."""
    desc = calc_descriptors([smiles], backend="rdkit")
    mw = float(desc[0, 0])
    assert abs(mw - mw_ref) < 0.1, f"{smiles}: MW {mw:.3f} vs ref {mw_ref:.3f}"


@pytest.mark.parametrize("smiles,mw_ref,logp_ref,ha_ref", KNOWN)
def test_immutability_preserved_after_transform(smiles, mw_ref, logp_ref, ha_ref):
    """Transforms must return new Mol; original smiles field must be unchanged."""
    mol  = Mol.from_smiles(smiles)
    orig = mol.smiles
    _    = mol.neutralize()
    assert mol.smiles == orig, "neutralize() must not modify the original Mol"


def test_benzene_fingerprint_nonzero():
    fps = molcore.featurize_smiles(["c1ccccc1"], backend="rust")
    assert int(fps.sum()) > 0


def test_pyg_benzene_node_count():
    data = Mol.from_smiles("c1ccccc1").to_pyg()
    assert data.x.shape[0] == 6   # 6 carbons
    assert data.edge_index.shape[1] == 12  # 6 bonds × 2 (bidirectional)
