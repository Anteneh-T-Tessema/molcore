"""
Tool: descriptors — strict function signature for agent tool-use.
"""
from __future__ import annotations
import torch
from molcore.featurizers.descriptors import calc_descriptors


def run(smiles: list[str], backend: str = "rust") -> dict:
    t: torch.Tensor = calc_descriptors(smiles, backend=backend)
    return {
        "columns": ["mw", "logp", "heavy_atoms"],
        "shape": list(t.shape),
        "data": t.tolist(),
    }
