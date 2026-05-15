"""Tests for 2D depiction: mol_to_svg, mol_to_png, mols_to_grid_svg, Mol._repr_svg_."""
import pytest
from molcore.rdkit_bridge import mol_to_svg, mol_to_png, mols_to_grid_svg
from molcore.molecule import Mol
from molcore.io import MolDataset

SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "c1cccnc1"]


def _cairo_available() -> bool:
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        rdMolDraw2D.MolDraw2DCairo(10, 10)
        return True
    except Exception:
        return False


requires_cairo = pytest.mark.skipif(
    not _cairo_available(),
    reason="RDKit built without Cairo support — PNG rendering unavailable",
)


# ---------------------------------------------------------------------------
# mol_to_svg
# ---------------------------------------------------------------------------

def test_mol_to_svg_returns_svg_string():
    svg = mol_to_svg("CCO")
    assert svg.startswith("<svg") or "svg" in svg.lower()
    assert len(svg) > 200


def test_mol_to_svg_custom_size():
    svg = mol_to_svg("c1ccccc1", width=400, height=300)
    assert "400" in svg or "300" in svg


def test_mol_to_svg_highlight_atoms():
    svg = mol_to_svg("CCO", highlight_atoms=[0, 1])
    assert svg  # should not crash


def test_mol_to_svg_invalid_smiles_raises():
    with pytest.raises(ValueError):
        mol_to_svg("NOT_A_SMILES")


# ---------------------------------------------------------------------------
# mol_to_png
# ---------------------------------------------------------------------------

@requires_cairo
def test_mol_to_png_creates_file(tmp_path):
    p = tmp_path / "mol.png"
    mol_to_png("CCO", str(p))
    assert p.exists()
    assert p.stat().st_size > 100


@requires_cairo
def test_mol_to_png_custom_size(tmp_path):
    p = tmp_path / "mol2.png"
    mol_to_png("c1ccccc1", str(p), width=500, height=400)
    assert p.exists()


# ---------------------------------------------------------------------------
# mols_to_grid_svg
# ---------------------------------------------------------------------------

def test_grid_svg_returns_string():
    svg = mols_to_grid_svg(SMILES)
    assert "svg" in svg.lower()
    assert len(svg) > 100


def test_grid_svg_with_legends():
    svg = mols_to_grid_svg(SMILES, legends=["ethanol", "benzene", "acetic acid", "pyridine"])
    assert svg


def test_grid_svg_empty_input():
    svg = mols_to_grid_svg([])
    assert svg == "<svg/>"


# ---------------------------------------------------------------------------
# Mol Jupyter display
# ---------------------------------------------------------------------------

def test_mol_repr_svg_is_svg():
    mol = Mol.from_smiles("CCO")
    svg = mol._repr_svg_()
    assert "svg" in svg.lower()


def test_mol_repr_html_contains_smiles():
    mol = Mol.from_smiles("CCO")
    html = mol._repr_html_()
    assert "CCO" in html or mol.smiles in html
    assert "svg" in html.lower()


# ---------------------------------------------------------------------------
# MolDataset Jupyter display
# ---------------------------------------------------------------------------

def test_dataset_repr_html():
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    html = ds._repr_html_()
    assert "MolDataset" in html
    assert "svg" in html.lower()


def test_dataset_draw_grid_returns_svg():
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    svg = ds.draw_grid(n=4, mols_per_row=2)
    assert "svg" in svg.lower()
