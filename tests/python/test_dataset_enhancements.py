"""
Tests for MolDataset enhancements:
  filter(), scaffold_split(), to_pyg_list(), add_descriptors_3d(),
  from_chembl(), from_zinc() (mocked), and __repr__ update.
"""
from unittest.mock import patch
import numpy as np
import pytest
import torch

from molcore.io import MolDataset


ACIDS    = ["CC(=O)O", "c1ccccc1C(=O)O", "CCC(=O)O"]
ALCOHOLS = ["CCO", "CCCO", "OCC(O)CO"]
MIXED    = ACIDS + ALCOHOLS + ["c1ccccc1", "Cc1ccccc1", "CCCC"]


# ── filter ───────────────────────────────────────────────────────────────────

def test_filter_keeps_acids():
    ds = MolDataset(smiles=MIXED)
    result = ds.filter("C(=O)O")
    assert set(result.smiles) == set(ACIDS)


def test_filter_invert_removes_acids():
    ds = MolDataset(smiles=MIXED)
    result = ds.filter("C(=O)O", invert=True)
    for smi in result.smiles:
        assert smi not in ACIDS


def test_filter_preserves_labels():
    labels = np.arange(len(MIXED), dtype=np.float32)
    ds = MolDataset(smiles=MIXED, labels=labels)
    result = ds.filter("C(=O)O")
    assert result.labels is not None
    assert len(result.labels) == len(result.smiles)


def test_filter_preserves_fingerprints():
    ds = MolDataset.from_smiles(MIXED)
    result = ds.filter("C(=O)O")
    assert result.fingerprints is not None
    assert result.fingerprints.shape[0] == len(result.smiles)


def test_filter_empty_result():
    ds = MolDataset(smiles=["CCO", "CCCC"])
    result = ds.filter("c1ccccc1")
    assert len(result) == 0


# ── scaffold_split ────────────────────────────────────────────────────────────

def test_scaffold_split_total():
    ds = MolDataset(smiles=MIXED)
    train, val, test = ds.scaffold_split()
    assert len(train) + len(val) + len(test) == len(MIXED)


def test_scaffold_split_no_duplicates():
    ds = MolDataset(smiles=MIXED)
    train, val, test = ds.scaffold_split()
    all_smi = train.smiles + val.smiles + test.smiles
    assert len(all_smi) == len(set(all_smi))


def test_scaffold_split_preserves_labels():
    labels = np.ones(len(MIXED), dtype=np.float32)
    ds = MolDataset(smiles=MIXED, labels=labels)
    train, val, test = ds.scaffold_split()
    for split in (train, val, test):
        if len(split) > 0:
            assert split.labels is not None
            assert len(split.labels) == len(split.smiles)


def test_scaffold_split_returns_moldatasets():
    ds = MolDataset(smiles=MIXED)
    train, val, test = ds.scaffold_split()
    for split in (train, val, test):
        assert isinstance(split, MolDataset)


# ── to_pyg_list ───────────────────────────────────────────────────────────────

def test_to_pyg_list_length():
    ds = MolDataset(smiles=["CCO", "c1ccccc1", "CC(=O)O"])
    graphs = ds.to_pyg_list()
    assert len(graphs) == 3


def test_to_pyg_list_node_feat_dim():
    ds = MolDataset(smiles=["CCO"])
    data = ds.to_pyg_list()[0]
    assert data.x.shape[1] == 9


def test_to_pyg_list_attaches_labels():
    ds = MolDataset(smiles=["CCO", "c1ccccc1"], labels=np.array([1.0, 2.0]))
    graphs = ds.to_pyg_list()
    assert graphs[0].y.item() == pytest.approx(1.0)
    assert graphs[1].y.item() == pytest.approx(2.0)


def test_to_pyg_list_skips_invalid():
    ds = MolDataset(smiles=["CCO", "INVALID$$$$", "c1ccccc1"])
    graphs = ds.to_pyg_list()
    assert len(graphs) == 2


# ── add_descriptors_3d ────────────────────────────────────────────────────────

def test_add_descriptors_3d_keys():
    ds = MolDataset(smiles=["CCO", "c1ccccc1"])
    result = ds.add_descriptors_3d()
    assert "asphericity" in result.metadata
    assert "pmi1" in result.metadata
    assert len(result.metadata["asphericity"]) == 2


def test_add_descriptors_3d_not_mutate():
    ds = MolDataset(smiles=["CCO"])
    _ = ds.add_descriptors_3d()
    assert "asphericity" not in ds.metadata


def test_add_descriptors_3d_skip_errors():
    ds = MolDataset(smiles=["CCO", "[He]"])
    result = ds.add_descriptors_3d(skip_errors=True)
    assert len(result.metadata["asphericity"]) == 2
    # noble gas may fail — value should be NaN not exception
    import math
    has_nan = any(math.isnan(v) for v in result.metadata["asphericity"]
                  if isinstance(v, float))
    assert has_nan or all(isinstance(v, float) for v in result.metadata["asphericity"])


# ── __repr__ ──────────────────────────────────────────────────────────────────

def test_repr_no_3d():
    ds = MolDataset(smiles=["CCO"])
    assert "3d=no" in repr(ds)


def test_repr_with_3d():
    ds = MolDataset(smiles=["CCO"])
    ds3 = ds.add_descriptors_3d()
    assert "3d=yes" in repr(ds3)


# ── from_chembl (mocked) ──────────────────────────────────────────────────────

def test_from_chembl_mocked(monkeypatch):
    from molcore.databases import ChEMBLCompound
    fake = [
        ChEMBLCompound(chembl_id="CHEMBL25", smiles="CC(=O)Oc1ccccc1C(=O)O", name="ASPIRIN"),
        ChEMBLCompound(chembl_id="CHEMBL16", smiles="c1ccccc1", name="BENZENE"),
    ]
    monkeypatch.setattr("molcore.databases.chembl_search", lambda *a, **kw: fake)
    ds = MolDataset.from_chembl("aspirin", compute_fps=False, compute_desc=False)
    assert len(ds) == 2
    assert ds.metadata["chembl_id"] == ["CHEMBL25", "CHEMBL16"]
    assert ds.metadata["name"] == ["ASPIRIN", "BENZENE"]


def test_from_chembl_no_results(monkeypatch):
    monkeypatch.setattr("molcore.databases.chembl_search", lambda *a, **kw: [])
    ds = MolDataset.from_chembl("notacompound", compute_fps=False, compute_desc=False)
    assert len(ds) == 0


# ── from_zinc (mocked) ────────────────────────────────────────────────────────

def test_from_zinc_mocked(monkeypatch):
    monkeypatch.setattr(
        "molcore.databases.zinc_download_tranche",
        lambda *a, **kw: ["CCO", "c1ccccc1", "CC(=O)O"],
    )
    ds = MolDataset.from_zinc(compute_fps=False, compute_desc=False)
    assert len(ds) == 3
    assert "CCO" in ds.smiles


def test_from_zinc_empty(monkeypatch):
    monkeypatch.setattr("molcore.databases.zinc_download_tranche", lambda *a, **kw: [])
    ds = MolDataset.from_zinc(compute_fps=False, compute_desc=False)
    assert len(ds) == 0
