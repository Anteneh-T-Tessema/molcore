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
