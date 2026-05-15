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


def test_fingerprint_shape():
    mol = Mol.from_smiles("CC(=O)Oc1ccccc1C(=O)O")
    fp = mol.fingerprint()
    import torch
    assert isinstance(fp, torch.Tensor)
    assert fp.shape == (2048,)


def test_fingerprint_nbits():
    mol = Mol.from_smiles("c1ccccc1")
    fp = mol.fingerprint(nbits=1024)
    assert fp.shape == (1024,)


def test_fingerprint_rdkit_backend():
    mol = Mol.from_smiles("CCO")
    fp = mol.fingerprint(backend="rdkit")
    import torch
    assert isinstance(fp, torch.Tensor)
    assert fp.shape[0] == 2048


def test_to_svg_returns_string():
    mol = Mol.from_smiles("c1ccccc1")
    svg = mol.to_svg()
    assert isinstance(svg, str)
    assert "<svg" in svg.lower()


def test_to_svg_custom_size():
    mol = Mol.from_smiles("CCO")
    svg = mol.to_svg(width=400, height=300)
    assert isinstance(svg, str)


def test_to_png_creates_file(tmp_path):
    mol = Mol.from_smiles("c1ccccc1")
    out = str(tmp_path / "mol.png")
    mol.to_png(out)
    import pathlib
    assert pathlib.Path(out).exists()
    assert pathlib.Path(out).stat().st_size > 0


def test_to_dgl_returns_graph():
    dgl = pytest.importorskip("dgl", reason="dgl not installed")
    mol = Mol.from_smiles("c1ccccc1")
    g = mol.to_dgl()
    assert g is not None
    assert g.num_nodes() == 6
