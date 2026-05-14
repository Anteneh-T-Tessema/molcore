import pathlib
import tempfile
import numpy as np
import torch
import pytest
from molcore.io import MolDataset

SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "c1ccccc1C(=O)O"]


def test_from_smiles_shape():
    ds = MolDataset.from_smiles(SMILES)
    assert len(ds) == 4
    assert ds.fingerprints.shape == (4, 2048)
    assert ds.descriptors.shape  == (4, 3)


def test_parquet_round_trip():
    ds = MolDataset.from_smiles(SMILES)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "mols.parquet"
        ds.write_parquet(path)
        ds2 = MolDataset.read_parquet(path)

    assert ds2.smiles == SMILES
    assert ds2.fingerprints.shape == ds.fingerprints.shape
    np.testing.assert_array_equal(ds2.fingerprints, ds.fingerprints)


def test_parquet_with_labels():
    ds = MolDataset.from_smiles(SMILES)
    ds.labels = np.array([-0.14, 1.90, -0.17, 1.87], dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "labeled.parquet"
        ds.write_parquet(path)
        ds2 = MolDataset.read_parquet(path)
    np.testing.assert_allclose(ds2.labels, ds.labels, atol=1e-5)


def test_fingerprints_tensor():
    ds = MolDataset.from_smiles(SMILES)
    t = ds.fingerprints_tensor()
    assert t.dtype == torch.uint8
    assert t.shape == (4, 2048)


def test_descriptors_tensor():
    ds = MolDataset.from_smiles(SMILES)
    t = ds.descriptors_tensor()
    assert t.dtype == torch.float32
    assert t.shape == (4, 3)


def test_repr():
    ds = MolDataset.from_smiles(SMILES)
    r = repr(ds)
    assert "n=4" in r and "fps=yes" in r


def test_no_fps_raises():
    ds = MolDataset.from_smiles(SMILES, compute_fps=False)
    with pytest.raises(ValueError, match="No fingerprints"):
        ds.fingerprints_tensor()
