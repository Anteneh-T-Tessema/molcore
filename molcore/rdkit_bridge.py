"""
rdkit_bridge.py — ALL RDKit calls are isolated here.
One file to update when RDKit changes an API.
Never import rdkit anywhere else in the hot path.
"""
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, MolStandardize


def from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol


def neutralize(smiles: str) -> str:
    mol = from_smiles(smiles)
    uncharger = MolStandardize.rdMolStandardize.Uncharger()
    return Chem.MolToSmiles(uncharger.uncharge(mol))


def strip_salts(smiles: str) -> str:
    mol = from_smiles(smiles)
    chooser = MolStandardize.rdMolStandardize.LargestFragmentChooser()
    return Chem.MolToSmiles(chooser.choose(mol))


def canonicalize(smiles: str) -> str:
    return Chem.MolToSmiles(from_smiles(smiles))


def ecfp4_rdkit(smiles: list[str], radius: int = 2, nbits: int = 2048) -> torch.Tensor:
    """RDKit-backend fingerprints — bit-identical to legacy trained models."""
    from rdkit.Chem import DataStructs
    rows = []
    for smi in smiles:
        mol = from_smiles(smi)
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
        arr = np.zeros(nbits, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        rows.append(arr)
    return torch.from_numpy(np.stack(rows))


def calc_descriptors_rdkit(smiles: list[str]) -> torch.Tensor:
    """RDKit MW, LogP, TPSA — exact values."""
    from rdkit.Chem import Descriptors
    rows = []
    for smi in smiles:
        mol = from_smiles(smi)
        rows.append([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.TPSA(mol),
        ])
    return torch.tensor(rows, dtype=torch.float32)
