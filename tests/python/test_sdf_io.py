"""Tests for SDF I/O: write_sdf, from_sdf, mol_to_sdf_block, from_molblock."""
import pathlib
import pytest
from molcore.rdkit_bridge import write_sdf, mol_to_sdf_block, from_molblock, from_sdf_file
from molcore.io import MolDataset
from molcore.molecule import Mol

SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "c1cccnc1"]


# ---------------------------------------------------------------------------
# write_sdf / from_sdf_file round-trip
# ---------------------------------------------------------------------------

def test_write_sdf_creates_file(tmp_path):
    p = tmp_path / "out.sdf"
    write_sdf(SMILES, str(p))
    assert p.exists()
    assert p.stat().st_size > 0


def test_write_sdf_with_properties(tmp_path):
    p = tmp_path / "props.sdf"
    props = {"logp": [1.0, 2.0, 3.0, 4.0], "name": ["ethanol", "benzene", "acetic_acid", "pyridine"]}
    write_sdf(SMILES, str(p), properties=props)
    records = from_sdf_file(str(p))
    assert len(records) == 4
    assert all("logp" in r[1] for r in records)
    assert all("name" in r[1] for r in records)


def test_from_sdf_file_returns_mols_and_props(tmp_path):
    p = tmp_path / "test.sdf"
    write_sdf(SMILES, str(p), properties={"idx": ["0", "1", "2", "3"]})
    records = from_sdf_file(str(p))
    assert len(records) == 4
    rdmols, props_list = zip(*records)
    assert all(m is not None for m in rdmols)
    assert props_list[0]["idx"] == "0"


def test_mol_to_sdf_block_nonempty():
    block = mol_to_sdf_block("CCO")
    assert "$$$$" in block
    assert len(block) > 50


# ---------------------------------------------------------------------------
# MolDataset.from_sdf / write_sdf round-trip
# ---------------------------------------------------------------------------

def test_molds_from_sdf_roundtrip(tmp_path):
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    sdf_path = tmp_path / "ds.sdf"
    ds.write_sdf(str(sdf_path))
    ds2 = MolDataset.from_sdf(str(sdf_path), compute_fps=False, compute_desc=False)
    assert len(ds2) == len(SMILES)


def test_molds_from_sdf_with_labels(tmp_path):
    import numpy as np
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    ds.labels = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    p = tmp_path / "labeled.sdf"
    ds.write_sdf(str(p))
    records = from_sdf_file(str(p))
    # label property should be written
    assert all("label" in r[1] for r in records)


def test_molds_from_sdf_metadata_preserved(tmp_path):
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    ds.metadata["source"] = ["a", "b", "c", "d"]
    p = tmp_path / "meta.sdf"
    ds.write_sdf(str(p))
    ds2 = MolDataset.from_sdf(str(p), compute_fps=False, compute_desc=False)
    assert "source" in ds2.metadata
    assert ds2.metadata["source"][0] == "a"


# ---------------------------------------------------------------------------
# from_molblock / Mol.from_molblock
# ---------------------------------------------------------------------------

def test_from_molblock_parses():
    block = mol_to_sdf_block("c1ccccc1")
    rdmol = from_molblock(block)
    assert rdmol is not None


def test_mol_from_molblock_returns_mol():
    block = mol_to_sdf_block("CCO")
    mol = Mol.from_molblock(block)
    assert isinstance(mol, Mol)
    assert mol.smiles  # non-empty


def test_from_molblock_invalid_raises():
    with pytest.raises(ValueError):
        from_molblock("NOT A MOLBLOCK")
