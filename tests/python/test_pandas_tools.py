"""Tests for molcore.pandas_tools and MolDataset DataFrame bridge."""
import numpy as np
import pytest
import pandas as pd

from molcore import pandas_tools as mpt
from molcore.io import MolDataset

SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "c1cccnc1"]


# ---------------------------------------------------------------------------
# add_mol_column / add_smiles_column
# ---------------------------------------------------------------------------

def test_add_mol_column_creates_col():
    df = pd.DataFrame({"smiles": SMILES})
    out = mpt.add_mol_column(df)
    assert "Mol" in out.columns
    assert all(m is not None for m in out["Mol"])


def test_add_mol_column_invalid_coerce():
    df = pd.DataFrame({"smiles": ["CCO", "NOT_VALID", "c1ccccc1"]})
    out = mpt.add_mol_column(df, errors="coerce")
    assert out["Mol"].iloc[1] is None


def test_add_mol_column_invalid_raise():
    df = pd.DataFrame({"smiles": ["CCO", "NOT_VALID"]})
    with pytest.raises(Exception):
        mpt.add_mol_column(df, errors="raise")


def test_add_smiles_column():
    from molcore.molecule import Mol
    df = pd.DataFrame({"Mol": [Mol.from_smiles(s) for s in SMILES]})
    out = mpt.add_smiles_column(df)
    assert "smiles" in out.columns
    assert all(isinstance(s, str) for s in out["smiles"])


# ---------------------------------------------------------------------------
# add_descriptors
# ---------------------------------------------------------------------------

def test_add_descriptors_lipinski():
    df = pd.DataFrame({"smiles": SMILES})
    out = mpt.add_descriptors(df, preset="lipinski")
    assert "MolWt" in out.columns
    assert "TPSA" in out.columns
    assert out["MolWt"].iloc[0] == pytest.approx(46.07, abs=0.5)


def test_add_descriptors_named():
    df = pd.DataFrame({"smiles": SMILES})
    out = mpt.add_descriptors(df, names=["MolWt", "NumHDonors"], preset=None)
    assert "MolWt" in out.columns
    assert "NumHDonors" in out.columns


def test_add_descriptors_inplace():
    df = pd.DataFrame({"smiles": SMILES})
    original_id = id(df)
    out = mpt.add_descriptors(df, preset="lipinski", inplace=True)
    assert id(out) == original_id


def test_add_descriptors_does_not_mutate_original():
    df = pd.DataFrame({"smiles": SMILES})
    _ = mpt.add_descriptors(df, preset="lipinski", inplace=False)
    assert "MolWt" not in df.columns


# ---------------------------------------------------------------------------
# add_fingerprints
# ---------------------------------------------------------------------------

def test_add_fingerprints_ecfp4():
    df = pd.DataFrame({"smiles": SMILES})
    out = mpt.add_fingerprints(df, kind="ecfp4")
    assert "fp" in out.columns
    assert out["fp"].iloc[0].shape == (2048,)


def test_add_fingerprints_maccs():
    df = pd.DataFrame({"smiles": SMILES})
    out = mpt.add_fingerprints(df, kind="maccs")
    assert "fp" in out.columns
    assert out["fp"].iloc[0].shape == (167,)


def test_add_fingerprints_expand_cols():
    df = pd.DataFrame({"smiles": SMILES[:2]})
    out = mpt.add_fingerprints(df, kind="maccs", expand_cols=True)
    assert "fp_0" in out.columns
    assert "fp_166" in out.columns


def test_add_fingerprints_unknown_kind_raises():
    df = pd.DataFrame({"smiles": SMILES})
    with pytest.raises(ValueError):
        mpt.add_fingerprints(df, kind="bogus_fp")


# ---------------------------------------------------------------------------
# filter_by_smarts
# ---------------------------------------------------------------------------

def test_filter_by_smarts_keeps_matches():
    df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1", "CC(=O)O", "c1ccncc1"]})
    out = mpt.filter_by_smarts(df, "n1ccccc1")  # pyridine-like ring
    assert len(out) == 1
    assert out["smiles"].iloc[0] == "c1ccncc1"


def test_filter_by_smarts_invert():
    df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1", "c1ccncc1"]})
    out = mpt.filter_by_smarts(df, "n", invert=True)
    assert all("n" not in row.lower() or row == "CCO" for row in out["smiles"])


# ---------------------------------------------------------------------------
# add_scaffold_column
# ---------------------------------------------------------------------------

def test_add_scaffold_column():
    df = pd.DataFrame({"smiles": ["c1ccccc1CC", "c1ccccc1C(=O)O"]})
    out = mpt.add_scaffold_column(df)
    assert "scaffold" in out.columns
    assert all(isinstance(s, str) for s in out["scaffold"])


# ---------------------------------------------------------------------------
# standardize_smiles
# ---------------------------------------------------------------------------

def test_standardize_strips_salt():
    df = pd.DataFrame({"smiles": ["[Na+].OC(=O)c1ccccc1", "CCO"]})
    out = mpt.standardize_smiles(df)
    assert "[Na+]" not in out["smiles"].iloc[0]


def test_standardize_inplace():
    df = pd.DataFrame({"smiles": ["CCO", "c1ccccc1"]})
    orig_id = id(df)
    out = mpt.standardize_smiles(df, inplace=True)
    assert id(out) == orig_id


# ---------------------------------------------------------------------------
# write_sdf / load_sdf round-trip
# ---------------------------------------------------------------------------

def test_load_sdf_roundtrip(tmp_path):
    p = tmp_path / "test.sdf"
    df = pd.DataFrame({"smiles": SMILES, "score": [1.0, 2.0, 3.0, 4.0]})
    mpt.write_sdf(df, str(p))
    df2 = mpt.load_sdf(str(p))
    assert len(df2) == 4
    assert "smiles" in df2.columns
    assert "Mol" in df2.columns


def test_load_sdf_properties_preserved(tmp_path):
    p = tmp_path / "props.sdf"
    df = pd.DataFrame({"smiles": SMILES, "name": ["e", "b", "a", "py"]})
    mpt.write_sdf(df, str(p), extra_cols=["name"])
    df2 = mpt.load_sdf(str(p))
    assert "name" in df2.columns


# ---------------------------------------------------------------------------
# MolDataset DataFrame bridge
# ---------------------------------------------------------------------------

def test_molds_to_dataframe():
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=True)
    df = ds.to_dataframe()
    assert "smiles" in df.columns
    assert "mw" in df.columns
    assert len(df) == 4


def test_molds_to_dataframe_with_labels():
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=False)
    ds.labels = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    df = ds.to_dataframe()
    assert "label" in df.columns
    assert df["label"].iloc[0] == pytest.approx(1.0)


def test_molds_from_dataframe():
    df = pd.DataFrame({
        "smiles": SMILES,
        "y": [1.0, 2.0, 3.0, 4.0],
        "source": ["a", "b", "c", "d"],
    })
    ds = MolDataset.from_dataframe(df, label_col="y", compute_fps=False, compute_desc=False)
    assert len(ds) == 4
    assert ds.labels is not None
    assert ds.labels[0] == pytest.approx(1.0)
    assert "source" in ds.metadata


def test_molds_from_dataframe_no_labels():
    df = pd.DataFrame({"smiles": SMILES})
    ds = MolDataset.from_dataframe(df, compute_fps=False, compute_desc=False)
    assert ds.labels is None


def test_molds_dataframe_roundtrip():
    ds = MolDataset.from_smiles(SMILES, compute_fps=False, compute_desc=True)
    ds.metadata["tag"] = ["x", "y", "z", "w"]
    df = ds.to_dataframe()
    ds2 = MolDataset.from_dataframe(df, compute_fps=False, compute_desc=False)
    assert len(ds2) == 4
    assert "tag" in ds2.metadata
