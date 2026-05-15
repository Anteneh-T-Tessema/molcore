import torch
import pytest
import molcore


SMILES = ["CCO", "c1ccccc1", "CC(=O)O"]


def test_rust_backend_shape():
    t = molcore.featurize_smiles(SMILES, backend="rust")
    assert t.shape == (3, 2048)
    assert t.dtype == torch.uint8


def test_rdkit_backend_shape():
    t = molcore.featurize_smiles(SMILES, backend="rdkit")
    assert t.shape == (3, 2048)


def test_rdkit_matches_known_vector():
    """rdkit backend must be bit-identical to RDKit directly."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs
    import numpy as np

    smi = "c1ccccc1"
    t = molcore.featurize_smiles([smi], backend="rdkit")

    mol = Chem.MolFromSmiles(smi)
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    arr = np.zeros(2048, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)

    assert torch.all(t[0] == torch.from_numpy(arr))


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        molcore.featurize_smiles(SMILES, backend="magic")


# ── rdkit_bridge low-level functions ─────────────────────────────────────────

def test_canonicalize_round_trips():
    from molcore.rdkit_bridge import canonicalize
    assert canonicalize("OCC") == canonicalize("CCO")


def test_canonicalize_returns_string():
    from molcore.rdkit_bridge import canonicalize
    assert isinstance(canonicalize("c1ccccc1"), str)


def test_ecfp4_rdkit_shape():
    from molcore.rdkit_bridge import ecfp4_rdkit
    fps = ecfp4_rdkit(["CCO", "c1ccccc1"], nbits=2048)
    assert fps.shape == (2, 2048)
    assert fps.dtype == torch.uint8


def test_ecfp4_rdkit_custom_nbits():
    from molcore.rdkit_bridge import ecfp4_rdkit
    fps = ecfp4_rdkit(["CCO"], nbits=1024)
    assert fps.shape == (1, 1024)


def test_calc_descriptors_rdkit_shape():
    from molcore.rdkit_bridge import calc_descriptors_rdkit
    desc = calc_descriptors_rdkit(["CCO", "c1ccccc1"])
    assert desc.shape == (2, 3)   # MW, logP, TPSA
    assert desc.dtype == torch.float32


def test_calc_descriptors_rdkit_ethanol_mw():
    from molcore.rdkit_bridge import calc_descriptors_rdkit
    desc = calc_descriptors_rdkit(["CCO"])
    assert abs(desc[0, 0].item() - 46.07) < 0.5   # MW of ethanol
