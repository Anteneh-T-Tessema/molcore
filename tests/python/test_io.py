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


def test_labels_tensor_returns_tensor():
    ds = MolDataset.from_smiles(SMILES)
    ds.labels = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    t = ds.labels_tensor()
    assert t.shape == (4,)
    assert t.dtype == torch.float32


def test_labels_tensor_no_labels_raises():
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    with pytest.raises(ValueError, match="No labels"):
        ds.labels_tensor()


def test_to_arrow_table_has_smiles_column():
    import pyarrow as pa
    ds = MolDataset.from_smiles(SMILES)
    table = ds.to_arrow_table()
    assert isinstance(table, pa.Table)
    assert "smiles" in table.schema.names
    assert table.num_rows == 4


def test_to_arrow_table_has_fingerprints():
    ds = MolDataset.from_smiles(SMILES)
    table = ds.to_arrow_table()
    assert "fingerprints" in table.schema.names


def test_to_arrow_table_no_fps_still_works():
    import pyarrow as pa
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    table = ds.to_arrow_table()
    assert isinstance(table, pa.Table)
    assert table.num_rows == 4
