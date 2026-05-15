"""Tests for named descriptors, presets, and additional fingerprint types."""
import numpy as np
import pytest
import torch
from molcore.rdkit_bridge import (
    calc_named_descriptors,
    list_descriptor_names,
    DESCRIPTOR_PRESETS,
    maccs_keys,
    atom_pairs_fp,
    topological_torsions_fp,
    rdkit_path_fp,
)
from molcore.featurizers.descriptors import (
    calc_named_descriptors as calc_named_desc_feat,
    list_descriptor_names as list_desc_feat,
    DESCRIPTOR_PRESETS as PRESETS_FEAT,
)
from molcore.pipeline import featurize_smiles

SMILES = ["CCO", "c1ccccc1", "CC(=O)O"]


# ---------------------------------------------------------------------------
# Named descriptors
# ---------------------------------------------------------------------------

def test_lipinski_preset_shape():
    arr, names = calc_named_descriptors(SMILES, preset="lipinski")
    assert arr.shape == (3, len(DESCRIPTOR_PRESETS["lipinski"]))
    assert "MolWt" in names
    assert "TPSA" in names


def test_druglike_preset_shape():
    arr, names = calc_named_descriptors(SMILES, preset="druglike")
    assert arr.shape[0] == 3
    assert "FractionCSP3" in names


def test_all_preset_returns_200_plus():
    arr, names = calc_named_descriptors(["CCO"], preset="all")
    assert arr.shape[1] > 100
    assert arr.shape[0] == 1


def test_named_descriptors_explicit():
    arr, names = calc_named_descriptors(SMILES, names=["MolWt", "TPSA"])
    assert arr.shape == (3, 2)
    assert names == ["MolWt", "TPSA"]


def test_molwt_ethanol_approx():
    arr, names = calc_named_descriptors(["CCO"], names=["MolWt"])
    assert abs(arr[0, 0] - 46.07) < 0.5


def test_tpsa_benzene_zero():
    arr, names = calc_named_descriptors(["c1ccccc1"], names=["TPSA"])
    assert arr[0, 0] == pytest.approx(0.0, abs=0.1)


def test_invalid_smiles_gives_nan():
    arr, _ = calc_named_descriptors(["NOT_SMILES", "CCO"], names=["MolWt"])
    assert np.isnan(arr[0, 0])
    assert not np.isnan(arr[1, 0])


def test_unknown_descriptor_raises():
    with pytest.raises(ValueError, match="Unknown descriptor"):
        calc_named_descriptors(SMILES, names=["NonExistentDescriptor999"])


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        calc_named_descriptors(SMILES, preset="notapreset")


def test_list_descriptor_names_nonempty():
    names = list_descriptor_names()
    assert len(names) > 100
    assert "MolWt" in names


def test_featurizer_module_exposes_same_api():
    arr, names = calc_named_desc_feat(SMILES, preset="lipinski")
    assert arr.shape[0] == 3
    assert "MolLogP" in names
    assert PRESETS_FEAT == DESCRIPTOR_PRESETS


# ---------------------------------------------------------------------------
# Additional fingerprint types
# ---------------------------------------------------------------------------

def test_maccs_shape():
    arr = maccs_keys(SMILES)
    assert arr.shape == (3, 167)
    assert arr.dtype == np.uint8


def test_maccs_nonzero_for_benzene():
    arr = maccs_keys(["c1ccccc1"])
    assert arr.sum() > 0


def test_maccs_invalid_smiles_gives_zeros():
    arr = maccs_keys(["NOT_SMILES"])
    assert arr.sum() == 0


def test_atom_pairs_shape():
    arr = atom_pairs_fp(SMILES, nbits=1024)
    assert arr.shape == (3, 1024)
    assert arr.dtype == np.uint8


def test_topological_torsions_shape():
    arr = topological_torsions_fp(SMILES, nbits=2048)
    assert arr.shape == (3, 2048)
    assert arr.dtype == np.uint8


def test_rdkit_path_fp_shape():
    arr = rdkit_path_fp(SMILES, nbits=2048)
    assert arr.shape == (3, 2048)
    assert arr.dtype == np.uint8


def test_different_fp_types_differ():
    ecfp = featurize_smiles(["c1ccccc1"], kind="ecfp4").numpy()
    maccs = maccs_keys(["c1ccccc1"])
    rdkit = rdkit_path_fp(["c1ccccc1"])
    # different shapes → clearly different; or different values
    assert not np.array_equal(ecfp, maccs[:, :ecfp.shape[1]])


# ---------------------------------------------------------------------------
# pipeline.featurize_smiles with kind=
# ---------------------------------------------------------------------------

def test_pipeline_maccs():
    t = featurize_smiles(SMILES, kind="maccs")
    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, 167)


def test_pipeline_atom_pairs():
    t = featurize_smiles(SMILES, kind="atom_pairs", nbits=512)
    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, 512)


def test_pipeline_topological_torsions():
    t = featurize_smiles(SMILES, kind="topological_torsions")
    assert isinstance(t, torch.Tensor)
    assert t.shape[0] == 3


def test_pipeline_rdkit_fp():
    t = featurize_smiles(SMILES, kind="rdkit")
    assert isinstance(t, torch.Tensor)
    assert t.shape == (3, 2048)


def test_pipeline_unknown_kind_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        featurize_smiles(SMILES, kind="nonexistent")
