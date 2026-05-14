"""
pipeline.py — batch-first primary entry point.
Always call this; per-mol methods are convenience wrappers around it.
"""
import torch
from observability.metrics.skill_metrics import timed as _timed


def featurize_smiles(
    smiles: list[str],
    kind: str = "ecfp4",
    backend: str = "rust",
    radius: int = 2,
    nbits: int = 2048,
) -> torch.Tensor:
    """
    Batch featurization.

    backend="rust"  → Rust Rayon parallel, zero-copy numpy → torch
                      Use for: training new models, billion-scale screening
    backend="rdkit" → RDKit via rdkit_bridge, bit-identical to legacy models
                      Use for: inference on models trained on RDKit fingerprints

    Callers decide the backend — it is never auto-selected.
    """
    if kind != "ecfp4":
        raise ValueError(f"Unsupported featurization kind: {kind!r}. Supported: 'ecfp4'")

    with _timed("fingerprint", batch_size=len(smiles)):
        if backend == "rust":
            from molcore.featurizers.fingerprints import ecfp4
            return ecfp4(smiles, radius=radius, nbits=nbits)
        elif backend == "rdkit":
            from molcore.rdkit_bridge import ecfp4_rdkit
            return ecfp4_rdkit(smiles, radius=radius, nbits=nbits)
        else:
            raise ValueError(f"Unknown backend: {backend!r}. Choose 'rust' or 'rdkit'.")
