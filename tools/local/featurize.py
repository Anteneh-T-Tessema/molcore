"""
Tool: featurize
Strict function signature for agent tool-use.
"""
from __future__ import annotations
import torch
from molcore.pipeline import featurize_smiles


def run(smiles: list[str], backend: str = "rust", radius: int = 2, nbits: int = 2048) -> dict:
    """
    Returns fingerprint tensor as a list (JSON-serializable for agent tool output).
    For direct ML use, call molcore.pipeline.featurize_smiles() instead.
    """
    t: torch.Tensor = featurize_smiles(smiles, backend=backend, radius=radius, nbits=nbits)
    return {
        "shape": list(t.shape),
        "dtype": str(t.dtype),
        "data": t.tolist(),
    }
